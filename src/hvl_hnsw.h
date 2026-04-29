#ifndef HVL_HNSW_H
#define HVL_HNSW_H

#include <stddef.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>
#include "hvl_vector.h"
#include "hvl_pq.h"
#include "hvl_config.h"
#include "hvl_dict.h"

// Forward declarations
struct hvl_hnsw_node;

typedef struct hvl_hnsw_node {
    hvl_vector *vec;
    uint32_t level;
    uint32_t internal_id;
    uint8_t deleted; // 0 = active, 1 = deleted
    _Atomic uint32_t *neighbor_counts; // Array [level + 1]
    struct hvl_hnsw_node **neighbors; // Pointers to neighbors
    pthread_mutex_t lock;
} hvl_hnsw_node;

// Macro helpers
#define VM_MIN(a, b) ((a) < (b) ? (a) : (b))

static inline struct hvl_hnsw_node **get_neighbors_at_level(struct hvl_hnsw_node *node, uint32_t level) {
    if (level == 0) return node->neighbors;
    return node->neighbors + HVL_M_MAX0 + (level - 1) * HVL_M_MAX;
}

#define ARENA_SEGMENT_SIZE (64 * 1024 * 1024)
#define MAX_ARENA_SEGMENTS 256

typedef struct {
    void *ptr;
    size_t offset;
} hvl_arena_segment;

typedef struct {
    hvl_arena_segment *segments[MAX_ARENA_SEGMENTS];
    int current_segment;
    pthread_mutex_t lock;
} hvl_arena;

#define HVL_PAGE_SIZE 65536
#define HVL_MAX_PAGES 256

typedef struct hvl_search_context {
    uint32_t *visited_map;
    uint32_t visited_token;
    size_t capacity;
    hvl_pq *candidates; 
    hvl_pq *found;
} hvl_search_context;

typedef struct {
    uint32_t dim;
    _Atomic uint32_t max_level;
    _Atomic(hvl_hnsw_node*) entry_node;
    hvl_dist_func dist_fn;
    uint32_t M;
    uint32_t ef_construction;
    uint32_t ef_search; // Added: respect config
    uint32_t max_levels_limit;
    hvl_arena *arena;
    _Atomic(hvl_hnsw_node**) pages[HVL_MAX_PAGES];
    hvl_dict *id_map; // New: String ID to Internal ID mapping
    pthread_mutex_t lock;
    _Atomic uint32_t count;
} hvl_hnsw_index;

#define HVL_MAX_ALLOWED_NODES (HVL_MAX_PAGES * HVL_PAGE_SIZE)

hvl_hnsw_index *hvl_hnsw_create(size_t dim, hvl_dist_func dist_fn, uint32_t M, uint32_t ef_construction, uint32_t max_levels);
void hvl_hnsw_free(hvl_hnsw_index *index);

int hvl_hnsw_insert(hvl_hnsw_index *index, hvl_vector *v);
int hvl_hnsw_insert_at_level(hvl_hnsw_index *index, hvl_vector *v, uint32_t level);

hvl_vector **hvl_hnsw_search(hvl_hnsw_index *index, hvl_vector *query, size_t k, size_t *found_count);
int hvl_hnsw_delete(hvl_hnsw_index *index, const char *id);

// Internal helpers exposed for persistence/server
struct hvl_search_context;
struct hvl_search_context *get_search_context(size_t current_count);
hvl_hnsw_node *index_get_node(hvl_hnsw_index *index, uint32_t id);
hvl_hnsw_node *node_create(hvl_hnsw_index *index, hvl_vector *v, uint32_t level);

#endif // HVL_HNSW_H
