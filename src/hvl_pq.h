#ifndef HVL_PQ_H
#define HVL_PQ_H

#include <stddef.h>
#include <stdbool.h>

// Forward declaration to avoid circular dependency
struct hvl_hnsw_node;

typedef struct {
    struct hvl_hnsw_node *node;
    float dist;
} hvl_pq_item;

typedef struct {
    hvl_pq_item *items;
    size_t size;
    size_t capacity;
    bool is_max_heap;
} hvl_pq;

hvl_pq *hvl_pq_create(size_t capacity, bool is_max_heap);
void hvl_pq_push(hvl_pq *pq, struct hvl_hnsw_node *node, float dist);
hvl_pq_item hvl_pq_pop(hvl_pq *pq);
hvl_pq_item hvl_pq_peek(hvl_pq *pq);
void hvl_pq_free(hvl_pq *pq);

#endif // HVL_PQ_H
