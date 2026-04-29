#ifndef HVL_VECTOR_H
#define HVL_VECTOR_H

#include <stddef.h>

typedef struct {
    float *data;
    size_t dim;
    char *id;
} hvl_vector;

// Distance metrics
typedef float (*hvl_dist_func)(const float *a, const float *b, size_t dim);

float hvl_dist_l2(const float *a, const float *b, size_t dim);
float hvl_dist_cosine(const float *a, const float *b, size_t dim);

// Memory management
hvl_vector *hvl_vector_create(size_t dim, const char *id);
void hvl_vector_free(hvl_vector *v);

#endif // HVL_VECTOR_H
