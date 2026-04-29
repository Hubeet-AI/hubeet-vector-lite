#include "hvl_hnsw.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include "hvl_config.h"
#include "hvl_macros.h"

// --- Paged Arena (Stable Memory) ---

static hvl_arena *hvl_arena_create() {
    hvl_arena *a = calloc(1, sizeof(hvl_arena));
    pthread_mutex_init(&a->lock, NULL);
    return a;
}

static void hvl_arena_free(hvl_arena *a) {
    if (!a) return;
    for (int i = 0; i < MAX_ARENA_SEGMENTS; i++) {
        if (a->segments[i]) {
            free(a->segments[i]->ptr);
            free(a->segments[i]);
        }
    }
    pthread_mutex_destroy(&a->lock);
    free(a);
}

static void *hvl_arena_alloc(hvl_arena *a, size_t size) {
    size = (size + 31) & ~31; // 32-byte alignment for SIMD
    pthread_mutex_lock(&a->lock);
    
    if (a->current_segment >= MAX_ARENA_SEGMENTS) {
        pthread_mutex_unlock(&a->lock);
        return NULL;
    }
    
    if (!a->segments[a->current_segment] || a->segments[a->current_segment]->offset + size > ARENA_SEGMENT_SIZE) {
        if (a->segments[a->current_segment]) a->current_segment++;
        
        if (a->current_segment < MAX_ARENA_SEGMENTS && !a->segments[a->current_segment]) {
            hvl_arena_segment *seg = malloc(sizeof(hvl_arena_segment));
            if (posix_memalign(&seg->ptr, 64, ARENA_SEGMENT_SIZE) != 0) { // 64-byte for cache-line alignment
                free(seg);
                pthread_mutex_unlock(&a->lock);
                return NULL;
            }
            seg->offset = 0;
            a->segments[a->current_segment] = seg;
        }
    }
    
    if (a->current_segment >= MAX_ARENA_SEGMENTS) {
        pthread_mutex_unlock(&a->lock);
        return NULL;
    }
    
    hvl_arena_segment *s = a->segments[a->current_segment];
    void *ptr = (char*)s->ptr + s->offset;
    s->offset += size;
    pthread_mutex_unlock(&a->lock);
    return ptr;
}

// --- Search Context (TLS) ---
static pthread_key_t tls_key;
static pthread_once_t tls_key_once = PTHREAD_ONCE_INIT;

static void search_context_ensure_capacity(hvl_search_context *ctx, size_t count) {
    if (count > HVL_MAX_ALLOWED_NODES) count = HVL_MAX_ALLOWED_NODES;
    if (count < 65536) count = 65536;
    
    if (count > ctx->capacity) {
        size_t new_cap = ctx->capacity == 0 ? count : ctx->capacity * 2;
        if (new_cap > HVL_MAX_ALLOWED_NODES) new_cap = HVL_MAX_ALLOWED_NODES;
        if (new_cap < count) new_cap = count;
        
        uint32_t *new_map = NULL;
        if (posix_memalign((void**)&new_map, 64, sizeof(uint32_t) * new_cap) == 0) {
            memset(new_map, 0, sizeof(uint32_t) * new_cap);
            if (ctx->visited_map) {
                memcpy(new_map, ctx->visited_map, sizeof(uint32_t) * ctx->capacity);
                free(ctx->visited_map);
            }
            ctx->visited_map = new_map;
            ctx->capacity = (uint32_t)new_cap;
        }
    }
}

static void free_search_context(void *ptr) {
    if (ptr) {
        hvl_search_context *ctx = (hvl_search_context*)ptr;
        if (ctx->visited_map) free(ctx->visited_map);
        if (ctx->candidates) hvl_pq_free(ctx->candidates);
        if (ctx->found) hvl_pq_free(ctx->found);
        free(ctx);
    }
}

static void make_tls_key() { pthread_key_create(&tls_key, free_search_context); }

hvl_search_context *get_search_context(size_t current_count) {
    pthread_once(&tls_key_once, make_tls_key);
    hvl_search_context *ctx = pthread_getspecific(tls_key);
    if (!ctx) {
        ctx = calloc(1, sizeof(hvl_search_context));
        pthread_setspecific(tls_key, ctx);
    }
    search_context_ensure_capacity(ctx, current_count + 1024);
    size_t pq_cap = 2048; // Sufficient for most HNSW configs
    if (!ctx->candidates) ctx->candidates = hvl_pq_create(pq_cap, false); 
    if (!ctx->found) ctx->found = hvl_pq_create(pq_cap, true); 
    return ctx;
}

// Forward declarations
static hvl_pq *search_layer_ef(hvl_hnsw_index *index, hvl_hnsw_node *entry, hvl_vector *query, uint32_t level, uint32_t ef, hvl_search_context *ctx, int skip_deleted);
static hvl_pq_item search_layer_best(hvl_hnsw_index *index, hvl_hnsw_node *entry, hvl_vector *query, uint32_t level, hvl_search_context *ctx);
static hvl_pq *hvl_hnsw_search_internal(hvl_hnsw_index *index, hvl_vector *query, size_t k, size_t *found_count, hvl_search_context *ctx);

// --- HNSW Index Functions ---

hvl_hnsw_index *hvl_hnsw_create(size_t dim, hvl_dist_func dist_fn, uint32_t M, uint32_t ef_construction, uint32_t max_levels) {
    hvl_hnsw_index *index = calloc(1, sizeof(hvl_hnsw_index));
    index->dim = (uint32_t)dim;
    atomic_init(&index->max_level, 0);
    atomic_init(&index->entry_node, NULL);
    index->dist_fn = dist_fn;
    index->M = M;
    index->ef_construction = ef_construction;
    index->ef_search = ef_construction; // Default
    index->max_levels_limit = max_levels;
    index->arena = hvl_arena_create();
    index->id_map = hvl_dict_create(65536);
    memset(index->pages, 0, sizeof(index->pages));
    pthread_mutex_init(&index->lock, NULL);
    atomic_init(&index->count, 0);
    return index;
}

hvl_hnsw_node *index_get_node(hvl_hnsw_index *index, uint32_t id) {
    uint32_t page_idx = id / HVL_PAGE_SIZE;
    uint32_t offset = id % HVL_PAGE_SIZE;
    if (page_idx >= HVL_MAX_PAGES) return NULL;
    hvl_hnsw_node **page = atomic_load_explicit(&index->pages[page_idx], memory_order_acquire);
    if (!page) return NULL;
    return page[offset];
}

hvl_hnsw_node *node_create(hvl_hnsw_index *index, hvl_vector *v, uint32_t level) {
    uint32_t total_neighbors = HVL_M_MAX0 + level * HVL_M_MAX;
    
    // Proper alignment calculations for metadata and arrays
    size_t node_struct_size = (sizeof(hvl_hnsw_node) + 7) & ~7;
    size_t counts_size = sizeof(_Atomic uint32_t) * (level + 1);
    size_t counts_size_aligned = (counts_size + 7) & ~7;
    
    size_t size = node_struct_size + counts_size_aligned + sizeof(hvl_hnsw_node*) * total_neighbors;
    hvl_hnsw_node *node = hvl_arena_alloc(index->arena, size);
    
    if (!node) {
        fprintf(stderr, "[FATAL] Arena OOM for level %u\n", level);
        fflush(stderr);
        return NULL;
    }
    
    if ((uintptr_t)node % 32 != 0) {
        fprintf(stderr, "[FATAL] Unaligned Arena Memory: %p\n", node);
        fflush(stderr);
    }

    node->vec = v;
    node->level = level;
    node->internal_id = 0;
    node->deleted = 0;
    node->neighbor_counts = (_Atomic uint32_t*)((char*)node + node_struct_size);
    node->neighbors = (hvl_hnsw_node**)((char*)node->neighbor_counts + counts_size_aligned);
    
    pthread_mutex_init(&node->lock, NULL);
    for (uint32_t i = 0; i <= level; i++) atomic_init(&node->neighbor_counts[i], 0);
    
    // Memory Barrier: Ensure construction is visible before sharing
    atomic_thread_fence(memory_order_release);
    
    return node;
}

static void hvl_pq_clear(hvl_pq *pq) { pq->size = 0; }

static hvl_pq *search_layer_ef(hvl_hnsw_index *index, hvl_hnsw_node *entry, hvl_vector *query, uint32_t level, uint32_t ef, hvl_search_context *ctx, int skip_deleted) {
    hvl_pq *candidates = ctx->candidates; hvl_pq *found = ctx->found; 
    hvl_pq_clear(candidates); hvl_pq_clear(found);
    ctx->visited_token++;
    if (ctx->visited_token == 0) {
        memset(ctx->visited_map, 0, sizeof(uint32_t) * ctx->capacity);
        ctx->visited_token = 1;
    }
    
    float d = index->dist_fn(entry->vec->data, query->data, index->dim);
    hvl_pq_push(candidates, entry, d);
    if (!skip_deleted || !entry->deleted) hvl_pq_push(found, entry, d);
    ctx->visited_map[entry->internal_id] = ctx->visited_token;
    
    while (candidates->size > 0) {
        hvl_pq_item c = hvl_pq_pop(candidates);
        hvl_pq_item f_worst = hvl_pq_peek(found);
        if (c.dist > f_worst.dist && found->size >= ef) break;
        
        hvl_hnsw_node *curr = c.node;
        
        // Lock-protected reading of neighbors to prevent reading garbage during select_neighbors
        pthread_mutex_lock(&curr->lock);
        uint32_t count = atomic_load_explicit(&curr->neighbor_counts[level], memory_order_acquire);
        hvl_hnsw_node **pool = get_neighbors_at_level(curr, level);
        
        for (uint32_t i = 0; i < count; i++) {
            hvl_hnsw_node *neighbor = pool[i];
            if (!neighbor || neighbor->internal_id >= ctx->capacity) continue;
            if (ctx->visited_map[neighbor->internal_id] == ctx->visited_token) continue;
            
            ctx->visited_map[neighbor->internal_id] = ctx->visited_token;
            float neighbor_d = index->dist_fn(neighbor->vec->data, query->data, index->dim);
            
            f_worst = hvl_pq_peek(found);
            if (found->size < ef || neighbor_d < f_worst.dist) {
                hvl_pq_push(candidates, neighbor, neighbor_d);
                if (!skip_deleted || !neighbor->deleted) {
                    hvl_pq_push(found, neighbor, neighbor_d);
                    if (found->size > ef) hvl_pq_pop(found);
                }
            }
        }
        pthread_mutex_unlock(&curr->lock);
    }
    return found;
}

static hvl_pq_item search_layer_best(hvl_hnsw_index *index, hvl_hnsw_node *entry, hvl_vector *query, uint32_t level, hvl_search_context *ctx) {
    hvl_pq *res = search_layer_ef(index, entry, query, level, 1, ctx, 0); // Always include deleted for greedy search
    if (res->size == 0) return (hvl_pq_item){0};
    return res->items[0];
}

static void select_neighbors(hvl_hnsw_index *index, hvl_hnsw_node *node, hvl_pq *candidates, uint32_t M, uint32_t level) {
    hvl_pq *result = hvl_pq_create(M, false); 
    while (candidates->size > 0 && result->size < M) {
        hvl_pq_item e = hvl_pq_pop(candidates);
        int good = 1;
        for (uint32_t i = 0; i < result->size; i++) {
            float d_er = index->dist_fn(e.node->vec->data, result->items[i].node->vec->data, index->dim);
            if (d_er < e.dist) { good = 0; break; }
        }
        if (good) hvl_pq_push(result, e.node, e.dist);
    }
    hvl_hnsw_node **pool = get_neighbors_at_level(node, level);
    for (uint32_t i = 0; i < result->size; i++) pool[i] = result->items[i].node;
    atomic_store_explicit(&node->neighbor_counts[level], result->size, memory_order_release);
    hvl_pq_free(result);
}

int hvl_hnsw_insert(hvl_hnsw_index *index, hvl_vector *v) {
    uint32_t level = 0;
    while (level < index->max_levels_limit && (rand() % 100) < 50) level++;
    return hvl_hnsw_insert_at_level(index, v, level);
}

int hvl_hnsw_insert_at_level(hvl_hnsw_index *index, hvl_vector *v, uint32_t level) {
    if (!index || !v) return -1;
    
    pthread_mutex_lock(&index->lock);
    
    hvl_hnsw_node *new_node = node_create(index, v, level);
    if (!new_node) {
        pthread_mutex_unlock(&index->lock);
        return -1;
    }

    uint32_t id = atomic_fetch_add(&index->count, 1);
    new_node->internal_id = id;
    
    uint32_t page_idx = id / HVL_PAGE_SIZE;
    uint32_t offset = id % HVL_PAGE_SIZE;
    if (page_idx < HVL_MAX_PAGES) {
        hvl_hnsw_node **page = atomic_load_explicit(&index->pages[page_idx], memory_order_acquire);
        if (!page) {
            page = calloc(HVL_PAGE_SIZE, sizeof(hvl_hnsw_node*));
            atomic_store_explicit(&index->pages[page_idx], page, memory_order_release);
        }
        page[offset] = new_node;
    }

    if (v->id[0]) hvl_dict_set(index->id_map, v->id, id);

    hvl_hnsw_node *entry = atomic_load(&index->entry_node);
    if (!entry) {
        // First node: update level THEN entry point to avoid race described in review
        atomic_store(&index->max_level, level);
        hvl_hnsw_node *expected = NULL;
        atomic_compare_exchange_strong(&index->entry_node, &expected, new_node);
        pthread_mutex_unlock(&index->lock);
        return 0;
    }
    
    hvl_hnsw_node *curr = entry;
    hvl_search_context *ctx = get_search_context(id + 1024);
    uint32_t max_l = atomic_load(&index->max_level);
    
    for (int l = (int)max_l; l > (int)level; l--) {
        hvl_pq_item best = search_layer_best(index, curr, v, (uint32_t)l, ctx);
        if (best.node) curr = best.node;
    }
    
    for (int l = (int)VM_MIN((int)level, (int)max_l); l >= 0; l--) {
        hvl_pq *res_ctx = search_layer_ef(index, curr, v, (uint32_t)l, index->ef_construction, ctx, 0); // Include deleted for linking
        hvl_pq *candidates = hvl_pq_create(res_ctx->size, false);
        for(size_t i=0; i<res_ctx->size; i++) hvl_pq_push(candidates, res_ctx->items[i].node, res_ctx->items[i].dist);
        
        uint32_t max_conn = (l == 0) ? HVL_M_MAX0 : HVL_M_MAX;
        
        pthread_mutex_lock(&new_node->lock);
        select_neighbors(index, new_node, candidates, max_conn, (uint32_t)l);
        pthread_mutex_unlock(&new_node->lock);
        
        hvl_pq_free(candidates);
        
        uint32_t current_neighbor_count = atomic_load(&new_node->neighbor_counts[l]);
        hvl_hnsw_node **new_node_neighbors = get_neighbors_at_level(new_node, l);
        
        for (uint32_t i = 0; i < current_neighbor_count; i++) {
            hvl_hnsw_node *neighbor = new_node_neighbors[i];
            pthread_mutex_lock(&neighbor->lock);
            hvl_hnsw_node **neighbor_pool = get_neighbors_at_level(neighbor, l);
            uint32_t n_count = atomic_load(&neighbor->neighbor_counts[l]);
            
            if (n_count < max_conn) {
                neighbor_pool[n_count] = new_node;
                atomic_store_explicit(&neighbor->neighbor_counts[l], n_count + 1, memory_order_release);
            } else {
                hvl_pq *candidates_p = hvl_pq_create(max_conn + 1, false); 
                for (uint32_t j = 0; j < n_count; j++) {
                    hvl_pq_push(candidates_p, neighbor_pool[j], index->dist_fn(neighbor->vec->data, neighbor_pool[j]->vec->data, index->dim));
                }
                hvl_pq_push(candidates_p, new_node, index->dist_fn(neighbor->vec->data, new_node->vec->data, index->dim));
                select_neighbors(index, neighbor, candidates_p, max_conn, l);
                hvl_pq_free(candidates_p);
            }
            pthread_mutex_unlock(&neighbor->lock);
        }
        if (current_neighbor_count > 0) curr = new_node_neighbors[0];
    }
    
    if (level > atomic_load(&index->max_level)) {
        atomic_store(&index->max_level, level);
        atomic_store(&index->entry_node, new_node);
    }
    
    pthread_mutex_unlock(&index->lock);
    return 0;
}

hvl_vector **hvl_hnsw_search(hvl_hnsw_index *index, hvl_vector *query, size_t k, size_t *found_count) {
    hvl_search_context *ctx = get_search_context(atomic_load(&index->count) + 1024);
    hvl_pq *top_k_ctx = hvl_hnsw_search_internal(index, query, k, found_count, ctx);
    if (!top_k_ctx || *found_count == 0) return NULL;
    
    hvl_vector **results = malloc(sizeof(hvl_vector*) * (*found_count));
    hvl_pq *tmp = hvl_pq_create(top_k_ctx->size, false); 
    for(size_t i=0; i<top_k_ctx->size; i++) hvl_pq_push(tmp, top_k_ctx->items[i].node, top_k_ctx->items[i].dist);
    for (size_t i = 0; i < *found_count; i++) results[i] = hvl_pq_pop(tmp).node->vec;
    hvl_pq_free(tmp);
    return results;
}

static hvl_pq *hvl_hnsw_search_internal(hvl_hnsw_index *index, hvl_vector *query, size_t k, size_t *found_count, hvl_search_context *ctx) {
    hvl_hnsw_node *curr = atomic_load(&index->entry_node);
    if (!index || !query || !curr) { *found_count = 0; return NULL; }
    
    uint32_t max_l = atomic_load(&index->max_level);
    for (int l = (int)max_l; l > 0; l--) {
        hvl_pq_item best = search_layer_best(index, curr, query, (uint32_t)l, ctx);
        if (best.node) curr = best.node;
    }
    
    uint32_t ef_search = index->ef_search;
    if (k > ef_search) ef_search = (uint32_t)k;
    
    hvl_pq *top_k = search_layer_ef(index, curr, query, 0, ef_search, ctx, 1); // Filter deleted for search result
    *found_count = (top_k->size < (size_t)k) ? top_k->size : (size_t)k;
    return top_k;
}

void hvl_hnsw_free(hvl_hnsw_index *index) {
    if (!index) return;
    uint32_t count = atomic_load(&index->count);
    for (uint32_t i = 0; i < count; i++) {
        hvl_hnsw_node *node = index_get_node(index, i);
        if (node) {
            if (node->vec) hvl_vector_free(node->vec);
            pthread_mutex_destroy(&node->lock);
        }
    }
    for (int i = 0; i < HVL_MAX_PAGES; i++) {
        hvl_hnsw_node **page = atomic_load(&index->pages[i]);
        if (page) free(page);
    }
    hvl_arena_free(index->arena);
    if (index->id_map) hvl_dict_free(index->id_map);
    pthread_mutex_destroy(&index->lock);
    free(index);
}

int hvl_hnsw_delete(hvl_hnsw_index *index, const char *id) {
    if (!index || !id) return -1;
    pthread_mutex_lock(&index->lock);
    uint32_t internal_id;
    if (hvl_dict_get(index->id_map, id, &internal_id)) {
        hvl_hnsw_node *node = index_get_node(index, internal_id);
        if (node) {
            node->deleted = 1;
            hvl_dict_remove(index->id_map, id);
            pthread_mutex_unlock(&index->lock);
            return 0;
        }
    }
    pthread_mutex_unlock(&index->lock);
    return -1;
}
