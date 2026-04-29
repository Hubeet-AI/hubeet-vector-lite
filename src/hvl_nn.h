#ifndef HVL_NN_H
#define HVL_NN_H

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include "hvl_tokenizer.h"

// Task types for the internal inference thread pool
typedef enum {
    TASK_ATTENTION,
    TASK_FFN,
    TASK_EXIT
} hvl_nn_task_type;

typedef struct {
    hvl_nn_task_type type;
    void *data; // Parameter block for the specific task
} hvl_nn_task;

// Forward declaration
struct hvl_model;

typedef struct {
    float *hidden_states;
    float *next_hidden;
    float *qkv_buffer;
    float *att_scores;
    float *ffn_intermediate;
    
    // Internal Thread Pool (4 workers)
    pthread_t workers[4];
    pthread_mutex_t pool_mutex;
    pthread_cond_t task_cond;
    pthread_cond_t done_cond;
    hvl_nn_task current_task;
    int tasks_remaining;
    int workers_active;
} hvl_inference_context;

typedef struct hvl_model {
    int num_layers;
    int hidden_dim;
    int vocab_size;
    int max_seq_len;
    
    float *word_embeddings;
    float *pos_embeddings;
    float *type_embeddings;
    float *emb_ln_w;
    float *emb_ln_b;
    
    struct {
        float *q_w, *q_b, *k_w, *k_b, *v_w, *v_b;
        float *o_w, *o_b, *o_ln_w, *o_ln_b;
        float *ff1_w, *ff1_b, *ff2_w, *ff2_b;
        float *ff_ln_w, *ff_ln_b;
    } *layers;
    
    void *mmap_ptr;
    size_t mmap_size;
} hvl_model;

hvl_model *hvl_model_load(const char *path);
void hvl_model_free(hvl_model *m);

hvl_inference_context *hvl_inference_context_create(hvl_model *m);
void hvl_inference_context_free(hvl_inference_context *ctx);

int hvl_inference_forward(hvl_model *m, hvl_tokenizer *t, hvl_inference_context *ctx, const char *text, float *output_vector);

#endif // HVL_NN_H
