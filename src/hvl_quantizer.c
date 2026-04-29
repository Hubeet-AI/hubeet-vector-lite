#include "hvl_quantizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

hvl_quantizer *hvl_quantizer_create(size_t dim) {
    hvl_quantizer *q = malloc(sizeof(hvl_quantizer));
    q->dim = dim;
    q->min_vals = malloc(sizeof(float) * dim);
    q->max_vals = malloc(sizeof(float) * dim);
    q->scales = malloc(sizeof(float) * dim);
    return q;
}

void hvl_quantizer_free(hvl_quantizer *q) {
    if (!q) return;
    free(q->min_vals);
    free(q->max_vals);
    free(q->scales);
    free(q);
}

void hvl_quantizer_train(hvl_quantizer *q, const float **vectors, size_t count) {
    for (size_t d = 0; d < q->dim; d++) {
        q->min_vals[d] = INFINITY;
        q->max_vals[d] = -INFINITY;
    }

    for (size_t i = 0; i < count; i++) {
        for (size_t d = 0; d < q->dim; d++) {
            if (vectors[i][d] < q->min_vals[d]) q->min_vals[d] = vectors[i][d];
            if (vectors[i][d] > q->max_vals[d]) q->max_vals[d] = vectors[i][d];
        }
    }

    for (size_t d = 0; d < q->dim; d++) {
        float range = q->max_vals[d] - q->min_vals[d];
        q->scales[d] = (range < 1e-10) ? 1.0f : range / 255.0f;
    }
}

void hvl_quantize(hvl_quantizer *q, const float *src, uint8_t *dst) {
    for (size_t d = 0; d < q->dim; d++) {
        float val = (src[d] - q->min_vals[d]) / q->scales[d];
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        dst[d] = (uint8_t)val;
    }
}

void hvl_dequantize(hvl_quantizer *q, const uint8_t *src, float *dst) {
    for (size_t d = 0; d < q->dim; d++) {
        dst[d] = q->min_vals[d] + (float)src[d] * q->scales[d];
    }
}

// SIMD optimized version would be here in a real Enterprise implementation
float hvl_dist_l2_sq8(const hvl_quantizer *q, const uint8_t *a, const uint8_t *b) {
    float total = 0.0f;
    for (size_t d = 0; d < q->dim; d++) {
        float diff = (float)(a[d] - b[d]) * q->scales[d];
        total += diff * diff;
    }
    return total;
}
