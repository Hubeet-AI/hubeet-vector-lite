#ifndef HVL_QUANTIZER_H
#define HVL_QUANTIZER_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    float *min_vals;
    float *max_vals;
    float *scales;
    size_t dim;
} hvl_quantizer;

hvl_quantizer *hvl_quantizer_create(size_t dim);
void hvl_quantizer_free(hvl_quantizer *q);

// Train the quantizer on a set of vectors
void hvl_quantizer_train(hvl_quantizer *q, const float **vectors, size_t count);

// Quantize a float vector to uint8
void hvl_quantize(hvl_quantizer *q, const float *src, uint8_t *dst);

// Dequantize (reconstruct) - mostly for testing or specific metrics
void hvl_dequantize(hvl_quantizer *q, const uint8_t *src, float *dst);

// Fast L2 distance on quantized vectors (approximate)
float hvl_dist_l2_sq8(const hvl_quantizer *q, const uint8_t *a, const uint8_t *b);

#endif // HVL_QUANTIZER_H
