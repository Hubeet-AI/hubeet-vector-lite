#include "hvl_nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// 1. SIMD KERNEL
static inline float dot_product(const float *a, const float *b, int n) {
    float sum = 0;
#ifdef __ARM_NEON
    float32x4_t vsum = vdupq_n_f32(0);
    int i = 0;
    for (; i <= n - 16; i += 16) {
        vsum = vmlaq_f32(vsum, vld1q_f32(a + i), vld1q_f32(b + i));
        vsum = vmlaq_f32(vsum, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        vsum = vmlaq_f32(vsum, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        vsum = vmlaq_f32(vsum, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i <= n - 4; i += 4) {
        vsum = vmlaq_f32(vsum, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    sum = vaddvq_f32(vsum);
    for (; i < n; i++) sum += a[i] * b[i];
#else
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
#endif
    return sum;
}

// 2. STABILITY FIX: ROBUST MALLOC CHECKS
hvl_inference_context *hvl_inference_context_create(hvl_model *m) {
    hvl_inference_context *ctx = calloc(1, sizeof(hvl_inference_context));
    if (!ctx) return NULL;
    
    int S = 512, H = m->hidden_dim;
    size_t sh_size = sizeof(float) * S * H;
    size_t qkv_size = sizeof(float) * S * H * 3;
    size_t att_size = sizeof(float) * 12 * S * S;
    size_t ffn_size = sizeof(float) * S * H * 4;

    if (posix_memalign((void**)&ctx->hidden_states, 64, sh_size) != 0 ||
        posix_memalign((void**)&ctx->next_hidden, 64, sh_size) != 0 ||
        posix_memalign((void**)&ctx->qkv_buffer, 64, qkv_size) != 0 ||
        posix_memalign((void**)&ctx->att_scores, 64, att_size) != 0 ||
        posix_memalign((void**)&ctx->ffn_intermediate, 64, ffn_size) != 0) {
        hvl_inference_context_free(ctx);
        return NULL;
    }
    
    return ctx;
}

void hvl_inference_context_free(hvl_inference_context *ctx) {
    if (!ctx) return;
    if (ctx->hidden_states) free(ctx->hidden_states);
    if (ctx->next_hidden) free(ctx->next_hidden);
    if (ctx->qkv_buffer) free(ctx->qkv_buffer);
    if (ctx->att_scores) free(ctx->att_scores);
    if (ctx->ffn_intermediate) free(ctx->ffn_intermediate);
    free(ctx);
}

static void matvec_serial(float *y, const float *W, const float *x, const float *b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        y[i] = dot_product(W + i * cols, x, cols) + (b ? b[i] : 0);
    }
}

static void layernorm(float *x, const float *w, const float *b, int n) {
    float sum = 0, sum2 = 0;
    for (int i = 0; i < n; i++) { sum += x[i]; sum2 += x[i] * x[i]; }
    float mean = sum / n;
    float var = fabsf(sum2 / n - mean * mean);
    float inv_std = 1.0f / sqrtf(var + 1e-12f);
    for (int i = 0; i < n; i++) x[i] = (x[i] - mean) * inv_std * w[i] + b[i];
}

static void fast_gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i], v3 = v * v * v;
        x[i] = 0.5f * v * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v3)));
    }
}

static inline void* align_ptr_internal(void* ptr, size_t alignment) {
    uintptr_t p = (uintptr_t)ptr;
    return (void*)((p + alignment - 1) & ~(alignment - 1));
}

hvl_model *hvl_model_load(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st; fstat(fd, &st);
    void *ptr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return NULL;
    if (memcmp(ptr, "HVLMODEL", 8) != 0) { munmap(ptr, st.st_size); return NULL; }
    hvl_model *m = malloc(sizeof(hvl_model));
    m->mmap_ptr = ptr; m->mmap_size = st.st_size;
    int32_t *header = (int32_t *)((char *)ptr + 8);
    m->num_layers = header[0]; m->hidden_dim = header[1]; m->vocab_size = header[2]; m->max_seq_len = header[3];
    void *curr = align_ptr_internal(header + 4, 64);
    #define NEXT_T(name, count) m->name = (float*)curr; curr = align_ptr_internal((float*)curr + (count), 64);
    NEXT_T(word_embeddings, m->vocab_size * m->hidden_dim);
    NEXT_T(pos_embeddings, m->max_seq_len * m->hidden_dim);
    NEXT_T(type_embeddings, 2 * m->hidden_dim);
    NEXT_T(emb_ln_w, m->hidden_dim); NEXT_T(emb_ln_b, m->hidden_dim);
    m->layers = malloc(sizeof(*m->layers) * m->num_layers);
    for (int i = 0; i < m->num_layers; i++) {
        NEXT_T(layers[i].q_w, m->hidden_dim * m->hidden_dim); NEXT_T(layers[i].q_b, m->hidden_dim);
        NEXT_T(layers[i].k_w, m->hidden_dim * m->hidden_dim); NEXT_T(layers[i].k_b, m->hidden_dim);
        NEXT_T(layers[i].v_w, m->hidden_dim * m->hidden_dim); NEXT_T(layers[i].v_b, m->hidden_dim);
        NEXT_T(layers[i].o_w, m->hidden_dim * m->hidden_dim); NEXT_T(layers[i].o_b, m->hidden_dim);
        NEXT_T(layers[i].o_ln_w, m->hidden_dim); NEXT_T(layers[i].o_ln_b, m->hidden_dim);
        NEXT_T(layers[i].ff1_w, m->hidden_dim * (m->hidden_dim * 4)); NEXT_T(layers[i].ff1_b, m->hidden_dim * 4);
        NEXT_T(layers[i].ff2_w, (m->hidden_dim * 4) * m->hidden_dim); NEXT_T(layers[i].ff2_b, m->hidden_dim);
        NEXT_T(layers[i].ff_ln_w, m->hidden_dim); NEXT_T(layers[i].ff_ln_b, m->hidden_dim);
    }
    return m;
}

int hvl_inference_forward(hvl_model *m, hvl_tokenizer *t, hvl_inference_context *ctx, const char *text, float *output_vector) {
    if (!ctx) return -1;
    int32_t tokens[512];
    size_t seq_len = hvl_tokenizer_encode(t, text, tokens, 512);
    if (seq_len == 0) return -1;
    int H = m->hidden_dim;
    for (size_t i = 0; i < seq_len; i++) {
        float *word_emb = m->word_embeddings + (tokens[i] * H);
        float *pos_emb = m->pos_embeddings + (i * H);
        float *h = ctx->hidden_states + i * H;
        for (int j = 0; j < H; j++) h[j] = word_emb[j] + pos_emb[j] + m->type_embeddings[j];
        layernorm(h, m->emb_ln_w, m->emb_ln_b, H);
    }

    float scratch[2048]; 
    for (int l = 0; l < m->num_layers; l++) {
        for (size_t i = 0; i < seq_len; i++) {
            float *h = ctx->hidden_states + i * H;
            matvec_serial(ctx->qkv_buffer + (i * 3 + 0) * H, m->layers[l].q_w, h, m->layers[l].q_b, H, H);
            matvec_serial(ctx->qkv_buffer + (i * 3 + 1) * H, m->layers[l].k_w, h, m->layers[l].k_b, H, H);
            matvec_serial(ctx->qkv_buffer + (i * 3 + 2) * H, m->layers[l].v_w, h, m->layers[l].v_b, H, H);
        }

        // Serial Attention
        for (int head = 0; head < 12; head++) {
            int head_dim = H / 12;
            int h_off = head * head_dim;
            for (int i = 0; i < (int)seq_len; i++) {
                float *q = ctx->qkv_buffer + (i * 3 + 0) * H + h_off;
                float *scores = ctx->att_scores + (head * 512 * 512) + (i * 512);
                for (int j = 0; j < (int)seq_len; j++) {
                    float *k = ctx->qkv_buffer + (j * 3 + 1) * H + h_off;
                    scores[j] = dot_product(q, k, head_dim) / sqrtf(head_dim);
                }
                float max_v = scores[0];
                for(int j=1; j<(int)seq_len; j++) if(scores[j]>max_v) max_v = scores[j];
                float sum = 0;
                for(int j=0; j<(int)seq_len; j++) { scores[j] = expf(scores[j]-max_v); sum += scores[j]; }
                float inv_s = 1.0f/sum;
                for(int j=0; j<(int)seq_len; j++) scores[j] *= inv_s;

                float *out_h = ctx->next_hidden + i * H + h_off;
                memset(out_h, 0, sizeof(float) * head_dim);
                for (int j = 0; j < (int)seq_len; j++) {
                    float *v = ctx->qkv_buffer + (j * 3 + 2) * H + h_off;
                    float s = scores[j];
                    for (int d = 0; d < head_dim; d++) out_h[d] += s * v[d];
                }
            }
        }

        for (size_t i = 0; i < seq_len; i++) {
            float *att_out = ctx->next_hidden + i * H, *h = ctx->hidden_states + i * H;
            matvec_serial(scratch, m->layers[l].o_w, att_out, m->layers[l].o_b, H, H);
            for (int j = 0; j < H; j++) h[j] += scratch[j];
            layernorm(h, m->layers[l].o_ln_w, m->layers[l].o_ln_b, H);

            matvec_serial(ctx->ffn_intermediate + i * (H * 4), m->layers[l].ff1_w, h, m->layers[l].ff1_b, H * 4, H);
            fast_gelu(ctx->ffn_intermediate + i * (H * 4), H * 4);
            matvec_serial(scratch, m->layers[l].ff2_w, ctx->ffn_intermediate + i * (H * 4), m->layers[l].ff2_b, H, H * 4);
            for (int j = 0; j < H; j++) h[j] += scratch[j];
            layernorm(h, m->layers[l].ff_ln_w, m->layers[l].ff_ln_b, H);
        }
    }

    memset(output_vector, 0, sizeof(float) * H);
    for (size_t i = 0; i < seq_len; i++) {
        float *h = ctx->hidden_states + i * H;
        for (int j = 0; j < H; j++) output_vector[j] += h[j];
    }
    float inv_seq = 1.0f / (float)seq_len, norm_sq = 0;
    for (int j = 0; j < H; j++) { output_vector[j] *= inv_seq; norm_sq += output_vector[j] * output_vector[j]; }
    float inv_norm = 1.0f / sqrtf(norm_sq + 1e-12f);
    for (int j = 0; j < H; j++) output_vector[j] *= inv_norm;
    return 0;
}

void hvl_model_free(hvl_model *m) {
    if (!m) return;
    free(m->layers);
    munmap(m->mmap_ptr, m->mmap_size);
    free(m);
}
