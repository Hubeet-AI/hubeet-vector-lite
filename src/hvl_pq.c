#include "hvl_pq.h"
#include <stdlib.h>
#include <string.h>

hvl_pq *hvl_pq_create(size_t capacity, bool is_max_heap) {
    hvl_pq *pq = malloc(sizeof(hvl_pq));
    pq->capacity = capacity;
    pq->size = 0;
    pq->is_max_heap = is_max_heap;
    pq->items = malloc(sizeof(hvl_pq_item) * capacity);
    return pq;
}

void hvl_pq_free(hvl_pq *pq) {
    if (!pq) return;
    free(pq->items);
    free(pq);
}

static void swap(hvl_pq_item *a, hvl_pq_item *b) {
    hvl_pq_item temp = *a;
    *a = *b;
    *b = temp;
}

static bool compare(hvl_pq *pq, float d1, float d2) {
    if (pq->is_max_heap) return d1 > d2;
    return d1 < d2;
}

void hvl_pq_push(hvl_pq *pq, struct hvl_hnsw_node *node, float dist) {
    if (pq->size >= pq->capacity) {
        pq->capacity *= 2;
        pq->items = realloc(pq->items, sizeof(hvl_pq_item) * pq->capacity);
    }
    
    pq->items[pq->size] = (hvl_pq_item){node, dist};
    size_t i = pq->size++;
    
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (compare(pq, pq->items[i].dist, pq->items[p].dist)) {
            swap(&pq->items[i], &pq->items[p]);
            i = p;
        } else break;
    }
}

hvl_pq_item hvl_pq_peek(hvl_pq *pq) {
    if (pq->size == 0) return (hvl_pq_item){NULL, 0.0f};
    return pq->items[0];
}

hvl_pq_item hvl_pq_pop(hvl_pq *pq) {
    if (pq->size == 0) return (hvl_pq_item){NULL, 0.0f};
    
    hvl_pq_item root = pq->items[0];
    pq->items[0] = pq->items[--pq->size];
    
    size_t i = 0;
    while (true) {
        size_t l = 2 * i + 1;
        size_t r = 2 * i + 2;
        size_t smallest = i;
        
        if (l < pq->size && compare(pq, pq->items[l].dist, pq->items[smallest].dist)) smallest = l;
        if (r < pq->size && compare(pq, pq->items[r].dist, pq->items[smallest].dist)) smallest = r;
        
        if (smallest != i) {
            swap(&pq->items[i], &pq->items[smallest]);
            i = smallest;
        } else break;
    }
    
    return root;
}

void hvl_pq_clear(hvl_pq *pq) {
    pq->size = 0;
}
