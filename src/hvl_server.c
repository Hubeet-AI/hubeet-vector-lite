#include "hvl_server.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <libgen.h>
#include <stdatomic.h>
#include <errno.h>
#include <fnmatch.h>
#include <stdarg.h>
#include <time.h>
#include <sys/time.h>
#include "hvl_protocol.h"
#include "hvl_persistence.h"
#include "hvl_config.h"
#include <ctype.h>
#include <signal.h>

static int is_empty_or_whitespace(const char *str) {
    if (!str) return 1;
    while (*str) {
        if (!isspace((unsigned char)*str)) return 0;
        str++;
    }
    return 1;
}

#define INF_POOL_SIZE 16

typedef struct {
    hvl_server *srv;
    int client_fd;
} client_context;

// Global instance for log/signal access
hvl_server *global_srv = NULL;

static uint64_t hvl_timer_now() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

static double hvl_timer_diff(uint64_t start, uint64_t end) {
    return (double)(end - start) / 1000000.0;
}

// LOGGING
void hvl_log(hvl_server *srv, int level, const char *fmt, ...) {
    if (!srv || level > srv->settings.log_level) return;
    
    FILE *f = fopen(srv->settings.log_file, "a");
    if (!f) return;

    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm *tm_info = localtime(&tv.tv_sec);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);
    
    const char *lvl_name = (level == 0) ? "ERROR" : (level == 1 ? "INFO" : "DEBUG");
    fprintf(f, "[%s.%03d] [%s] ", time_str, (int)(tv.tv_usec / 1000), lvl_name);
    
    va_list args;
    va_start(args, fmt);
    vfprintf(f, fmt, args);
    va_end(args);
    
    fprintf(f, "\n");
    fflush(f);
    fclose(f);
}

// GLOBAL INFERENCE POOL
static hvl_inference_context *inf_pool[INF_POOL_SIZE];
static int inf_pool_busy[INF_POOL_SIZE];
static pthread_mutex_t inf_pool_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t inf_pool_cond = PTHREAD_COND_INITIALIZER;

static hvl_inference_context *acquire_inf_context() {
    pthread_mutex_lock(&inf_pool_mutex);
    while (1) {
        for (int i = 0; i < INF_POOL_SIZE; i++) {
            if (!inf_pool_busy[i]) {
                inf_pool_busy[i] = 1;
                pthread_mutex_unlock(&inf_pool_mutex);
                return inf_pool[i];
            }
        }
        pthread_cond_wait(&inf_pool_cond, &inf_pool_mutex);
    }
}

static void release_inf_context(hvl_inference_context *ctx) {
    pthread_mutex_lock(&inf_pool_mutex);
    for (int i = 0; i < INF_POOL_SIZE; i++) {
        if (inf_pool[i] == ctx) {
            inf_pool_busy[i] = 0;
            break;
        }
    }
    pthread_cond_broadcast(&inf_pool_cond);
    pthread_mutex_unlock(&inf_pool_mutex);
}

// BACKGROUND AUTO-SAVE WORKER
static void *background_save_worker(void *arg) {
    hvl_server *srv = (hvl_server *)arg;
    int interval = srv->settings.save_interval;
    if (interval <= 0) return NULL;

    while (1) {
        sleep(interval);
        if (atomic_load(&srv->dirty)) {
            pthread_mutex_lock(&srv->save_lock);
            pthread_mutex_lock(&srv->index->lock);
            if (atomic_load(&srv->dirty)) {
                hvl_log(srv, 1, "Running background auto-save to %s", srv->settings.persistence_path);
                if (hvl_persistence_save(srv->index, srv->settings.persistence_path) == 0) {
                    atomic_store(&srv->dirty, 0);
                    hvl_log(srv, 1, "Auto-save complete.");
                } else {
                    hvl_log(srv, 0, "Auto-save FAILED.");
                }
            }
            pthread_mutex_unlock(&srv->index->lock);
            pthread_mutex_unlock(&srv->save_lock);
        }
    }
    return NULL;
}

hvl_server *hvl_server_create(hvl_settings *settings) {
    hvl_server *srv = malloc(sizeof(hvl_server));
    if (!srv) return NULL;
    srv->settings = *settings;
    atomic_init(&srv->dirty, 0);
    
    pthread_mutex_init(&srv->save_lock, NULL);

    hvl_log(srv, 1, "Hubeet Server starting up...");
    
    hvl_hnsw_index *loaded = hvl_persistence_load(settings->persistence_path, hvl_dist_cosine);
    if (loaded) {
        hvl_log(srv, 1, "Index loaded from %s", settings->persistence_path);
        srv->index = loaded;
    } else {
        hvl_log(srv, 1, "Creating new index with dim %u", settings->dim);
        srv->index = hvl_hnsw_create(settings->dim, hvl_dist_cosine, settings->M, settings->ef_construction, settings->max_levels);
    }
    if (srv->index) srv->index->ef_search = settings->ef_search;

    srv->model = NULL;
    srv->tokenizer = NULL;
    if (settings->embedding_model_path[0]) {
        hvl_log(srv, 1, "Loading inference model from %s", settings->embedding_model_path);
        srv->model = hvl_model_load(settings->embedding_model_path);
        if (srv->model) {
            char vocab_path[512];
            char *dir_name = strdup(settings->embedding_model_path);
            snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.txt", dirname(dir_name));
            free(dir_name);
            srv->tokenizer = hvl_tokenizer_create(vocab_path, settings->tokenizer_normalization);
            
            for (int i = 0; i < INF_POOL_SIZE; i++) {
                inf_pool[i] = hvl_inference_context_create(srv->model);
                inf_pool_busy[i] = 0;
            }
            hvl_log(srv, 1, "Inference pool of %d contexts initialized.", INF_POOL_SIZE);
        } else {
            hvl_log(srv, 0, "FAILED to load inference model.");
        }
    }

    srv->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (srv->listen_fd < 0) return NULL;

    int opt = 1;
    setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(settings->bind_addr);
    addr.sin_port = htons(settings->port);

    if (bind(srv->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        hvl_log(srv, 0, "Bind failed on %s:%d", settings->bind_addr, settings->port);
        return NULL;
    }

    listen(srv->listen_fd, 512);
    hvl_log(srv, 1, "Listening on %s:%d", settings->bind_addr, settings->port);

    if (srv->settings.save_interval > 0) {
        pthread_t bg_thread;
        pthread_create(&bg_thread, NULL, background_save_worker, srv);
        pthread_detach(bg_thread);
    }

    global_srv = srv;
    return srv;
}

static void *handle_client(void *arg) {
    client_context *ctx_conn = (client_context *)arg;
    hvl_server *srv = ctx_conn->srv;
    int client_fd = ctx_conn->client_fd;
    free(ctx_conn);

    hvl_log(srv, 2, "NEW CLIENT [FD:%d]", client_fd);

    size_t buf_cap = 131072;
    char *buffer = malloc(buf_cap);
    size_t buf_len = 0;

    int running = 1;
    size_t batch_count = 0;
    while (running) {
        if (buf_cap - buf_len < 32768) {
            if (buf_cap < HVL_MAX_BUFFER_SIZE) {
                buf_cap *= 2;
                hvl_log(srv, 2, "[FD:%d] Resizing buffer to %zu bytes", client_fd, buf_cap);
                char *new_buf = realloc(buffer, buf_cap);
                if (!new_buf) break;
                buffer = new_buf;
            } else if (buf_cap - buf_len <= 1) break;
        }

        if (batch_count < HVL_MAX_COMMANDS_BATCH || buf_len == 0) {
            ssize_t bytes = read(client_fd, buffer + buf_len, buf_cap - buf_len - 1);
            if (bytes < 0) {
                if (errno == EINTR || errno == EAGAIN) continue;
                break;
            }
            if (bytes == 0) break;
            buf_len += (size_t)bytes;
            buffer[buf_len] = '\0';
        }

        batch_count = 0;
        while (buf_len > 0 && batch_count < HVL_MAX_COMMANDS_BATCH) {
            size_t consumed = 0;
            hvl_command *cmd = hvl_protocol_parse(buffer, buf_len, srv->index->dim, &consumed);
            if (consumed > 0) {
                batch_count++;
                if (cmd) {
                    hvl_log(srv, 2, "[FD:%d] CMD:%d (consumed:%zu)", client_fd, cmd->type, consumed);
                    if (cmd->type == CMD_UNKNOWN) {
                        write(client_fd, "-ERR unknown or unsupported command\r\n", 37);
                    } else if (cmd->type == CMD_PING) {
                        write(client_fd, "+PONG\r\n", 7);
                    } else if (cmd->type == CMD_QUIT) {
                        write(client_fd, "+OK bye\r\n", 9);
                        running = 0;
                    } else if (cmd->type == CMD_VSET || (cmd->type == CMD_TSET && srv->model)) {
                        float *data_to_insert = cmd->vec_data;
                        float *inferred_buffer = NULL;

                        if (cmd->type == CMD_TSET) {
                            if (is_empty_or_whitespace(cmd->text)) {
                                write(client_fd, "-ERR empty text rejected\r\n", 26);
                                goto next_cmd;
                            }
                            if ((int)strlen(cmd->text) > srv->settings.max_text_length) {
                                write(client_fd, "-ERR text too long\r\n", 20);
                                goto next_cmd;
                            }
                            inferred_buffer = malloc(sizeof(float) * srv->index->dim);
                            hvl_inference_context *inf_ctx = acquire_inf_context();
                            uint64_t start = hvl_timer_now();
                            int res = hvl_inference_forward(srv->model, srv->tokenizer, inf_ctx, cmd->text, inferred_buffer);
                            release_inf_context(inf_ctx);
                            
                            hvl_log(srv, 2, "[FD:%d] TSET Inference: %.4fs", client_fd, hvl_timer_diff(start, hvl_timer_now()));
                            
                            if (res == 0) {
                                data_to_insert = inferred_buffer;
                            } else {
                                free(inferred_buffer);
                                write(client_fd, "-ERR inference failed\r\n", 23);
                                goto next_cmd;
                            }
                        }

                        if (data_to_insert) {
                            hvl_vector *v = hvl_vector_create(srv->index->dim, cmd->vec_id);
                            if (v) {
                                memcpy(v->data, data_to_insert, sizeof(float) * srv->index->dim);
                                // Optimization: If vector already exists, check distance
                                uint32_t internal_id;
                                int skip_insert = 0;
                                if (hvl_dict_get(srv->index->id_map, cmd->vec_id, &internal_id)) {
                                    hvl_hnsw_node *old_n = index_get_node(srv->index, internal_id);
                                    if (old_n && !old_n->deleted) {
                                        float dist = srv->index->dist_fn(old_n->vec->data, v->data, srv->index->dim);
                                        if (dist < 0.00001f) skip_insert = 1; 
                                    }
                                }

                                if (skip_insert) {
                                    hvl_vector_free(v);
                                    // Distance is identical, graph is unchanged. No tombstone.
                                    write(client_fd, "+OK\r\n", 5);
                                } else {
                                    // UPSERT: Delete existing node if it exists
                                    hvl_hnsw_delete(srv->index, cmd->vec_id);
                                    hvl_hnsw_insert(srv->index, v);
                                    atomic_store(&srv->dirty, 1);
                                    write(client_fd, "+OK\r\n", 5);
                                }
                            } else {
                                write(client_fd, "-ERR allocation failed\r\n", 24);
                            }
                        } else {
                            write(client_fd, "-ERR missing data\r\n", 19);
                        }
                        if (inferred_buffer) free(inferred_buffer);

                    } else if (cmd->type == CMD_VSEARCH || (cmd->type == CMD_TSEARCH && srv->model)) {
                        float *search_vec = cmd->vec_data;
                        float *inferred_search = NULL;

                        if (cmd->type == CMD_TSEARCH) {
                            inferred_search = malloc(sizeof(float) * srv->index->dim);
                            hvl_inference_context *inf_ctx = acquire_inf_context();
                            int res_inf = hvl_inference_forward(srv->model, srv->tokenizer, inf_ctx, cmd->text, inferred_search);
                            release_inf_context(inf_ctx);
                            if (res_inf == 0) search_vec = inferred_search;
                            else {
                                free(inferred_search);
                                write(client_fd, "-ERR inference failed\r\n", 23);
                                goto next_cmd;
                            }
                        }

                        if (search_vec) {
                            size_t found_count = 0;
                            hvl_vector query = {.dim = srv->index->dim, .data = search_vec};
                            hvl_vector **results = hvl_hnsw_search(srv->index, &query, cmd->k, &found_count);

                            float dist_threshold = 0.85f; // Filter out completely unrelated vectors
                            size_t valid_count = 0;
                            // Calculate valid matches first
                            for (size_t i = 0; i < found_count; i++) {
                                float dist = srv->index->dist_fn(query.data, results[i]->data, srv->index->dim);
                                if (dist <= dist_threshold) valid_count++;
                            }

                            char resp[1024];
                            int hlen = snprintf(resp, sizeof(resp), "*%zu\r\n", valid_count);
                            write(client_fd, resp, (size_t)hlen);
                            
                            for (size_t i = 0; i < found_count; i++) {
                                float dist = srv->index->dist_fn(query.data, results[i]->data, srv->index->dim);
                                if (dist > dist_threshold) continue;
                                
                                char dist_str[64];
                                int dlen = snprintf(dist_str, sizeof(dist_str), "%.6f", dist);
                                
                                char item_hdr[128];
                                int ih_len = snprintf(item_hdr, sizeof(item_hdr), "*2\r\n$%zu\r\n%s\r\n$%zu\r\n%s\r\n", 
                                                     strlen(results[i]->id), results[i]->id,
                                                     (size_t)dlen, dist_str);
                                write(client_fd, item_hdr, (size_t)ih_len);
                            }
                            if (results) free(results);
                        } else {
                            write(client_fd, "-ERR missing search vector\r\n", 28);
                        }
                        if (inferred_search) free(inferred_search);

                    } else if (cmd->type == CMD_HGETALL) {
                        pthread_mutex_lock(&srv->index->lock);
                        size_t total = (size_t)atomic_load(&srv->index->count);
                        size_t count = 0, limit = cmd->limit;
                        size_t *matches = malloc(sizeof(size_t) * (total < 1024 ? 1024 : total));
                        for (size_t i = 0; i < total && count < limit; i++) {
                            hvl_hnsw_node *node = index_get_node(srv->index, (uint32_t)i);
                            if (node && node->vec &&
                                fnmatch(cmd->pattern, node->vec->id, 0) == 0) {
                                matches[count++] = i;
                            }
                        }
                        pthread_mutex_unlock(&srv->index->lock);
                        
                        char resp[1024];
                        int hlen = snprintf(resp, sizeof(resp), "*%zu\r\n", count);
                        write(client_fd, resp, (size_t)hlen);
                        for (size_t i = 0; i < count; i++) {
                            pthread_mutex_lock(&srv->index->lock);
                            hvl_hnsw_node *node = index_get_node(srv->index, (uint32_t)matches[i]);
                            pthread_mutex_unlock(&srv->index->lock);
                            if (!node) continue;
                            hvl_vector *v = node->vec;
                            write(client_fd, "*2\r\n", 4);
                            int id_len = snprintf(resp, sizeof(resp), "$%zu\r\n", strlen(v->id));
                            write(client_fd, resp, (size_t)id_len);
                            write(client_fd, v->id, strlen(v->id));
                            write(client_fd, "\r\n", 2);
                            
                            size_t vec_buf_size = srv->index->dim * 32 + 1024;
                            char *vec_buf = malloc(vec_buf_size);
                            size_t offset = 0;
                            int n = snprintf(vec_buf + offset, vec_buf_size - offset, "[");
                            if (n > 0) offset += (size_t)n;
                            for(size_t d=0; d<srv->index->dim && offset < vec_buf_size - 32; d++) {
                                n = snprintf(vec_buf + offset, vec_buf_size - offset, "%.6f%s", v->data[d], d == srv->index->dim-1 ? "" : ",");
                                if (n > 0) {
                                    size_t attempted = (size_t)n;
                                    offset += (attempted < (vec_buf_size - offset)) ? attempted : (vec_buf_size - offset - 1);
                                }
                            }
                            if (offset < vec_buf_size - 1) {
                                n = snprintf(vec_buf + offset, vec_buf_size - offset, "]");
                                if (n > 0) {
                                    size_t attempted = (size_t)n;
                                    offset += (attempted < (vec_buf_size - offset)) ? attempted : (vec_buf_size - offset - 1);
                                }
                            }
                            int head_len = snprintf(resp, sizeof(resp), "$%zu\r\n", offset);
                            write(client_fd, resp, (size_t)head_len);
                            write(client_fd, vec_buf, offset);
                            write(client_fd, "\r\n", 2);
                            free(vec_buf);
                        }
                        free(matches);
                        pthread_mutex_unlock(&srv->index_lock);
                    } else if (cmd->type == CMD_INFO) {
                        pthread_mutex_lock(&srv->index_lock);
                        char info[4096];
                        char abs_path[1024];
                        if (realpath(srv->settings.persistence_path, abs_path) == NULL) {
                            char cwd[512];
                            if (!getcwd(cwd, sizeof(cwd))) strcpy(cwd, ".");
                            snprintf(abs_path, sizeof(abs_path), "%s/%s", cwd, srv->settings.persistence_path);
                        }
                        int written = snprintf(info, sizeof(info), 
                            "# Server\r\nhvl_version:0.1.0\r\nos:macOS\r\nprocess_id:%d\r\ninf_pool:%d\r\npersistence_path:%s\r\n\r\n"
                            "# Stats\r\ntotal_keys:%zu\r\ndim:%zu\r\ndirty_flag:%d\r\n\r\n"
                            "# HNSW\r\nM:%u\r\nef_construction:%u\r\nmax_level:%u\r\n\r\n"
                            "# Inference\r\nmodel:%s\r\nnormalization:%s\r\nmax_text_length:%d\r\n",
                            (int)getpid(), (int)INF_POOL_SIZE, abs_path, (size_t)atomic_load(&srv->index->count), (size_t)srv->index->dim,
                            (int)atomic_load(&srv->dirty), srv->index->M, srv->index->ef_construction, (uint32_t)atomic_load(&srv->index->max_level),
                            srv->settings.embedding_model_path[0] ? basename(srv->settings.embedding_model_path) : "none",
                            srv->settings.tokenizer_normalization ? "on" : "off", srv->settings.max_text_length);
                        pthread_mutex_unlock(&srv->index_lock);
                        size_t actual_len = (written >= (int)sizeof(info)) ? sizeof(info)-1 : (size_t)written;
                        char head[64];
                        int hlen = snprintf(head, sizeof(head), "$%zu\r\n", actual_len);
                        write(client_fd, head, (size_t)hlen);
                        write(client_fd, info, actual_len);
                        write(client_fd, "\r\n", 2);
                    } else if (cmd->type == CMD_SAVE) {
                        pthread_mutex_lock(&srv->save_lock);
                        pthread_mutex_lock(&srv->index->lock);
                        if (hvl_persistence_save(srv->index, srv->settings.persistence_path) == 0) {
                            atomic_store(&srv->dirty, 0);
                            char abs_path[1024];
                            if (realpath(srv->settings.persistence_path, abs_path) == NULL) {
                                char cwd[512];
                                if (!getcwd(cwd, sizeof(cwd))) strcpy(cwd, ".");
                                snprintf(abs_path, sizeof(abs_path), "%s/%s", cwd, srv->settings.persistence_path);
                            }
                            char ok_msg[1100];
                            snprintf(ok_msg, sizeof(ok_msg), "+OK index saved to %s\r\n", abs_path);
                            write(client_fd, ok_msg, strlen(ok_msg));
                        } else write(client_fd, "-ERR save failed\r\n", 18);
                        pthread_mutex_unlock(&srv->index->lock);
                        pthread_mutex_unlock(&srv->save_lock);
                    } else if (cmd->type == CMD_DELETE) {
                        if (hvl_hnsw_delete(srv->index, cmd->vec_id) == 0) {
                            atomic_store(&srv->dirty, 1);
                            write(client_fd, "+OK\r\n", 5);
                        } else {
                            write(client_fd, "-ERR not found\r\n", 16);
                        }
                    } else if (cmd->type == CMD_FLUSHDB) {
                        pthread_mutex_lock(&srv->save_lock);
                        pthread_mutex_lock(&srv->index->lock);
                        hvl_hnsw_index *old_index = srv->index;
                        srv->index = hvl_hnsw_create(srv->settings.dim, 
                                                    old_index->dist_fn, 
                                                    srv->settings.M, 
                                                    srv->settings.ef_construction, 
                                                    srv->settings.max_levels);
                        hvl_hnsw_free(old_index);
                        atomic_store(&srv->dirty, 1);
                        pthread_mutex_unlock(&srv->index->lock);
                        pthread_mutex_unlock(&srv->save_lock);
                        write(client_fd, "+OK database flushed\r\n", 22);
                    }
                next_cmd:
                    hvl_command_free(cmd);
                }
                memmove(buffer, buffer + consumed, buf_len - consumed);
                buf_len -= consumed;
                buffer[buf_len] = '\0';
            } else break;
        }
    }
    hvl_log(srv, 2, "CLIENT DISCONNECTED [FD:%d]", client_fd);
    free(buffer);
    close(client_fd);
    return NULL;
}

void hvl_server_run(hvl_server *srv) {
    if (!srv) return;
    signal(SIGPIPE, SIG_IGN);
    printf("Server listening on %s:%d...\n", srv->settings.bind_addr, srv->settings.port);
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(srv->listen_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) continue;
        client_context *ctx = malloc(sizeof(client_context));
        ctx->srv = srv; ctx->client_fd = client_fd;
        // Set thread stack size to 8MB for macOS (default is too small for AI/SIMD)
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);

        pthread_t thread;
        if (pthread_create(&thread, &attr, handle_client, ctx) != 0) {
            hvl_log(srv, 0, "Failed to create client thread");
            close(client_fd);
            free(ctx);
        } else {
            pthread_detach(thread);
        }
        pthread_attr_destroy(&attr);
    }
}

void hvl_server_free(hvl_server *srv) {
    if (!srv) return;
    hvl_log(srv, 1, "Shutting down server.");
    hvl_hnsw_free(srv->index);
    if (srv->model) hvl_model_free(srv->model);
    if (srv->tokenizer) hvl_tokenizer_free(srv->tokenizer);
    pthread_mutex_destroy(&srv->index_lock);
    pthread_mutex_destroy(&srv->save_lock);
    for (int i = 0; i < INF_POOL_SIZE; i++) if (inf_pool[i]) hvl_inference_context_free(inf_pool[i]);
    close(srv->listen_fd);
    free(srv);
}
