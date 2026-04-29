#include "hvl_vector.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_NEON
#elif defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2
#endif

hvl_vector *hvl_vector_create(size_t dim, const char *id) {
    hvl_vector *v = NULL;
    if (posix_memalign((void**)&v, 64, sizeof(hvl_vector)) != 0) return NULL;
    
    v->dim = (uint32_t)dim;
    v->id = strdup(id);
    if (!v->id) { free(v); return NULL; }
    
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, sizeof(float) * dim) != 0) {
        free(v->id);
        free(v);
        return NULL;
    }
    v->data = (float*)ptr;
    return v;
}

void hvl_vector_free(hvl_vector *v) {
    if (!v) return;
    free(v->id);
    free(v->data);
    free(v);
}

float hvl_dist_l2(const float *a, const float *b, size_t dim) {
    float total = 0.0f;
    size_t i = 0;

#if defined(HAS_NEON)
    float32x4_t sum_v = vdupq_n_f32(0);
    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t diff = vsubq_f32(va, vb);
        sum_v = vfmaq_f32(sum_v, diff, diff);
    }
    total = vaddvq_f32(sum_v);
#elif defined(HAS_AVX2)
    __m256 sum = _mm256_setzero_ps();
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    float res[8];
    _mm256_storeu_ps(res, sum);
    total = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
#endif

    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total; // Squared L2 (No sqrtf for performance)
}

float hvl_dist_cosine(const float *a, const float *b, size_t dim) {
    float dot = 0, norm_a = 0, norm_b = 0;
    size_t i = 0;

#if defined(HAS_NEON)
    float32x4_t dot_v = vdupq_n_f32(0);
    float32x4_t na_v = vdupq_n_f32(0);
    float32x4_t nb_v = vdupq_n_f32(0);

    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        dot_v = vfmaq_f32(dot_v, va, vb);
        na_v = vfmaq_f32(na_v, va, va);
        nb_v = vfmaq_f32(nb_v, vb, vb);
    }
    dot = vaddvq_f32(dot_v);
    norm_a = vaddvq_f32(na_v);
    norm_b = vaddvq_f32(nb_v);
#elif defined(HAS_AVX2)
    __m256 dot_v = _mm256_setzero_ps();
    __m256 na_v = _mm256_setzero_ps();
    __m256 nb_v = _mm256_setzero_ps();

    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        dot_v = _mm256_fmadd_ps(va, vb, dot_v);
        na_v = _mm256_fmadd_ps(va, va, na_v);
        nb_v = _mm256_fmadd_ps(vb, vb, nb_v);
    }
    float r_dot[8], r_na[8], r_nb[8];
    _mm256_storeu_ps(r_dot, dot_v);
    _mm256_storeu_ps(r_na, na_v);
    _mm256_storeu_ps(r_nb, nb_v);
    for(int j=0; j<8; j++) {
        dot += r_dot[j]; norm_a += r_na[j]; norm_b += r_nb[j];
    }
#endif

    for (; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float den = sqrtf(norm_a) * sqrtf(norm_b);
    return (den < 1e-10) ? 1.0f : 1.0f - (dot / den);
}
