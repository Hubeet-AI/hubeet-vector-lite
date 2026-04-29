#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <math.h>
#include <sys/time.h>
#include "hvl_hnsw.h"
#include "hvl_config.h"
#include "hvl_vector.h"
#include "hvl_nn.h"
#include "hvl_tokenizer.h"

#define VEC_DIM 384
#define NUM_BASE 1000
#define NUM_QUERY 100
#define K_RECALL 10

typedef struct {
    float *data;
    char id[16];
} synth_vector;

typedef struct {
    int thread_id;
    hvl_hnsw_index *index;
    synth_vector *queries;
    int queries_per_thread;
    double *latencies;
    size_t *found_counts;
} thread_arg;

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

int compare_doubles(const void *a, const void *b) {
    double arg1 = *(const double *)a;
    double arg2 = *(const double *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

// Linear scan for Ground Truth
void get_ground_truth(synth_vector *base, int n_base, synth_vector *query, int n_query, int **gt_indices) {
    printf("Generating Ground Truth (Linear Scan)... ");
    double start = get_time_ms();
    for (int i = 0; i < n_query; i++) {
        hvl_pq *pq = hvl_pq_create(K_RECALL, true);
        for (int j = 0; j < n_base; j++) {
            float d = hvl_dist_l2(query[i].data, base[j].data, VEC_DIM);
            if (pq->size < K_RECALL || d < hvl_pq_peek(pq).dist) {
                hvl_pq_push(pq, (struct hvl_hnsw_node*)(size_t)j, d);
                if (pq->size > K_RECALL) hvl_pq_pop(pq);
            }
        }
        for (int k = K_RECALL - 1; k >= 0; k--) {
            gt_indices[i][k] = (int)(size_t)hvl_pq_pop(pq).node;
        }
        hvl_pq_free(pq);
    }
    printf("Done (%.2f ms)\n", get_time_ms() - start);
}

void *search_worker(void *arg) {
    thread_arg *t = (thread_arg *)arg;
    for (int i = 0; i < t->queries_per_thread; i++) {
        hvl_vector v;
        v.dim = VEC_DIM;
        v.data = t->queries[i].data;
        double start = get_time_ms();
        size_t found = 0;
        hvl_vector **results = hvl_hnsw_search(t->index, &v, K_RECALL, &found);
        double end = get_time_ms();
        t->latencies[i] = end - start;
        t->found_counts[i] = found;
        if (results) free(results);
    }
    return NULL;
}

void run_benchmark(hvl_hnsw_index *index, synth_vector *queries, int n_threads, int n_query) {
    int q_per_thread = n_query / n_threads;
    pthread_t threads[n_threads];
    thread_arg args[n_threads];
    double *all_latencies = malloc(sizeof(double) * n_query);
    
    double start = get_time_ms();
    for (int i = 0; i < n_threads; i++) {
        args[i].thread_id = i;
        args[i].index = index;
        args[i].queries = &queries[i * q_per_thread];
        args[i].queries_per_thread = q_per_thread;
        args[i].latencies = &all_latencies[i * q_per_thread];
        args[i].found_counts = malloc(sizeof(size_t) * q_per_thread);
        pthread_create(&threads[i], NULL, search_worker, &args[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
    double total_time = get_time_ms() - start;
    
    qsort(all_latencies, n_query, sizeof(double), compare_doubles);
    double avg = 0;
    for (int i = 0; i < n_query; i++) avg += all_latencies[i];
    avg /= n_query;
    
    double p95 = all_latencies[(int)(n_query * 0.95)];
    double p99 = all_latencies[(int)(n_query * 0.99)];
    double qps = (double)n_query / (total_time / 1000.0);
    printf("| %7d | %9.2f | %12.4f | %12.4f | %12.4f |\n", n_threads, qps, avg, p95, p99);
    for(int i=0; i<n_threads; i++) free(args[i].found_counts);
    free(all_latencies);
}

int main() {
    srand(42);
    printf("Hubeet Vector Engine - Extended Benchmark Suite\n\n");

    // 1. INFERENCE BENCHMARK
    printf("--- NATIVE INFERENCE BENCHMARK ---\n");
    const char *model_path = "./models/paraphrase-multilingual-MiniLM-L12-v2.hvl_model";
    const char *vocab_path = "./models/vocab.txt";
    hvl_model *model = hvl_model_load(model_path);
    hvl_tokenizer *tokenizer = hvl_tokenizer_create(vocab_path, 1);
    
    if (!model || !tokenizer) {
        printf("Skipping Inference Benchmark: Model or Vocab not found.\n\n");
    } else {
        const char *test_text = "Modern vector engines require native transformer inference.";
        float embedding[384];
        hvl_inference_context *ctx = hvl_inference_context_create(model);
        hvl_inference_forward(model, tokenizer, ctx, test_text, embedding);
        int num_inf = 50;
        double start_inf = get_time_ms();
        for (int i = 0; i < num_inf; i++) hvl_inference_forward(model, tokenizer, ctx, test_text, embedding);
        double total_inf = get_time_ms() - start_inf;
        printf("Avg Latency per Sentence: %.2f ms (%.2f sent/sec)\n\n", total_inf / num_inf, (double)num_inf / (total_inf / 1000.0));
        hvl_inference_context_free(ctx);
    }

    // 2. HNSW SEARCH BENCHMARK
    printf("--- HNSW SEARCH BENCHMARK (Dim=%d) ---\n", VEC_DIM);
    synth_vector *base_data = malloc(sizeof(synth_vector) * NUM_BASE);
    for (int i = 0; i < NUM_BASE; i++) {
        base_data[i].data = malloc(sizeof(float) * VEC_DIM);
        for (int d = 0; d < VEC_DIM; d++) base_data[i].data[d] = (float)rand() / RAND_MAX;
        sprintf(base_data[i].id, "b_%d", i);
    }
    synth_vector *query_data = malloc(sizeof(synth_vector) * NUM_QUERY);
    for (int i = 0; i < NUM_QUERY; i++) {
        query_data[i].data = malloc(sizeof(float) * VEC_DIM);
        for (int d = 0; d < VEC_DIM; d++) query_data[i].data[d] = (float)rand() / RAND_MAX;
        sprintf(query_data[i].id, "q_%d", i);
    }
    
    int **gt_indices = malloc(sizeof(int *) * NUM_QUERY);
    for (int i = 0; i < NUM_QUERY; i++) gt_indices[i] = malloc(sizeof(int) * K_RECALL);
    get_ground_truth(base_data, NUM_BASE, query_data, NUM_QUERY, gt_indices);

    hvl_hnsw_index *index = hvl_hnsw_create(VEC_DIM, hvl_dist_l2, 16, 200, 16);
    printf("Indexing %d vectors... ", NUM_BASE);
    double start_idx = get_time_ms();
    for (int i = 0; i < NUM_BASE; i++) {
        hvl_vector *v = hvl_vector_create(VEC_DIM, base_data[i].id);
        memcpy(v->data, base_data[i].data, sizeof(float) * VEC_DIM);
        hvl_hnsw_insert(index, v);
    }
    printf("Done (%.2f ms)\n", get_time_ms() - start_idx);
    
    printf("\n+---------+-----------+--------------+--------------+--------------+\n");
    printf("| Threads |    QPS    | Avg Lat (ms) | P95 Lat (ms) | P99 Lat (ms) |\n");
    printf("+---------+-----------+--------------+--------------+--------------+\n");
    run_benchmark(index, query_data, 1, NUM_QUERY);
    run_benchmark(index, query_data, 4, NUM_QUERY);
    run_benchmark(index, query_data, 8, NUM_QUERY);
    printf("+---------+-----------+--------------+--------------+--------------+\n\n");

    printf("Calculating Recall@%d... ", K_RECALL);
    int total_hits = 0;
    for (int i = 0; i < NUM_QUERY; i++) {
        hvl_vector v = {.dim = VEC_DIM, .data = query_data[i].data};
        size_t found = 0;
        hvl_vector **results = hvl_hnsw_search(index, &v, K_RECALL, &found);
        for (size_t f = 0; f < found; f++) {
            for (int k = 0; k < K_RECALL; k++) {
                if (strcmp(results[f]->id, base_data[gt_indices[i][k]].id) == 0) {
                    total_hits++; break;
                }
            }
        }
        if (results) free(results);
    }
    printf("Recall@%d: %.4f\n", K_RECALL, (double)total_hits / (NUM_QUERY * K_RECALL));

    // Cleanup
    for(int i=0; i<NUM_BASE; i++) free(base_data[i].data);
    for(int i=0; i<NUM_QUERY; i++) { free(query_data[i].data); free(gt_indices[i]); }
    free(base_data); free(query_data); free(gt_indices);
    hvl_hnsw_free(index);
    if (model) hvl_model_free(model);
    if (tokenizer) hvl_tokenizer_free(tokenizer);
    return 0;
}
