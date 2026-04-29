#include "hvl_persistence.h"
#include "hvl_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>

static int sanity_check_id_len(uint32_t len) {
    return (len > 0 && len < 4096);
}

int hvl_persistence_save(hvl_hnsw_index *index, const char *filename) {
    if (!index || !filename) return -1;
    
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;

    uint64_t count = (uint64_t)atomic_load(&index->count);
    uint64_t dim = (uint64_t)index->dim;
    uint32_t max_level = (uint32_t)atomic_load(&index->max_level);
    
    hvl_hnsw_node *entry = atomic_load(&index->entry_node);
    int32_t entry_id = entry ? (int32_t)entry->internal_id : -1;

    // Header
    if (fwrite("HVL2", 1, 4, f) != 4) goto err;
    if (fwrite(&dim, sizeof(uint64_t), 1, f) != 1) goto err;
    if (fwrite(&count, sizeof(uint64_t), 1, f) != 1) goto err;
    if (fwrite(&max_level, sizeof(uint32_t), 1, f) != 1) goto err;
    if (fwrite(&entry_id, sizeof(int32_t), 1, f) != 1) goto err;

    // Nodes Section
    for (uint32_t i = 0; i < (uint32_t)count; i++) {
        hvl_hnsw_node *node = index_get_node(index, i);
        if (!node) continue;
        uint32_t level = node->level;
        if (node->deleted) level |= 0x80000000;
        uint32_t id_len = (uint32_t)strlen(node->vec->id);
        fwrite(&level, sizeof(uint32_t), 1, f);
        fwrite(&id_len, sizeof(uint32_t), 1, f);
        fwrite(node->vec->id, 1, id_len, f);
        fwrite(node->vec->data, sizeof(float), (size_t)dim, f);
    }

    // Pass 2: Neighbors
    for (uint32_t i = 0; i < (uint32_t)count; i++) {
        hvl_hnsw_node *node = index_get_node(index, i);
        if (!node) continue;
        for (uint32_t l = 0; l <= node->level; l++) {
            uint32_t neighbor_count = atomic_load(&node->neighbor_counts[l]);
            fwrite(&neighbor_count, sizeof(uint32_t), 1, f);
            hvl_hnsw_node **pool = get_neighbors_at_level(node, l);
            for (uint32_t n = 0; n < neighbor_count; n++) {
                uint32_t neighbor_id = pool[n]->internal_id;
                fwrite(&neighbor_id, sizeof(uint32_t), 1, f);
            }
        }
    }

    fclose(f);
    return 0;

err:
    if (f) fclose(f);
    return -1;
}

hvl_hnsw_index *hvl_persistence_load(const char *filename, hvl_dist_func dist_fn) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4) { fclose(f); return NULL; }
    
    if (memcmp(magic, "HVL2", 4) != 0) {
        fprintf(stderr, "Unsupported version or corrupt magic\n");
        fclose(f);
        return NULL;
    }

    uint64_t dim64, count64;
    uint32_t max_level_h;
    int32_t entry_id_32;
    if (fread(&dim64, sizeof(uint64_t), 1, f) != 1) goto load_err;
    if (fread(&count64, sizeof(uint64_t), 1, f) != 1) goto load_err;
    if (fread(&max_level_h, sizeof(uint32_t), 1, f) != 1) goto load_err;
    if (fread(&entry_id_32, sizeof(int32_t), 1, f) != 1) goto load_err;

    size_t dim = (size_t)dim64;
    uint32_t count = (uint32_t)count64;

    hvl_hnsw_index *index = hvl_hnsw_create(dim, dist_fn, HVL_M, HVL_EF_CONSTRUCTION, HVL_MAX_LEVELS);
    if (!index) goto load_err;

    // Pass 1: Nodes and Paging
    for (uint32_t i = 0; i < count; i++) {
        uint32_t level, id_len;
        if (fread(&level, sizeof(uint32_t), 1, f) != 1) goto load_err;
        if (fread(&id_len, sizeof(uint32_t), 1, f) != 1) goto load_err;
        if (!sanity_check_id_len(id_len)) goto load_err;

        uint32_t is_deleted = (level & 0x80000000) ? 1 : 0;
        level &= 0x7FFFFFFF;

        char id[4096];
        if (fread(id, 1, id_len, f) != id_len) goto load_err;
        id[id_len] = '\0';

        hvl_vector *v = hvl_vector_create(dim, id);
        if (!v) goto load_err;
        if (fread(v->data, sizeof(float), dim, f) != dim) {
            hvl_vector_free(v);
            goto load_err;
        }

        hvl_hnsw_node *node = node_create(index, v, level);
        node->internal_id = i;
        node->deleted = is_deleted;
        
        if (v->id[0] && !is_deleted) hvl_dict_set(index->id_map, v->id, i);
        
        uint32_t page_idx = i / HVL_PAGE_SIZE;
        uint32_t offset = i % HVL_PAGE_SIZE;
        hvl_hnsw_node **page = atomic_load_explicit(&index->pages[page_idx], memory_order_acquire);
        if (!page) {
            page = calloc(HVL_PAGE_SIZE, sizeof(hvl_hnsw_node*));
            atomic_store_explicit(&index->pages[page_idx], page, memory_order_release);
        }
        page[offset] = node;
        atomic_store(&index->count, i + 1);
    }

    // Pass 2: Entry point
    if (entry_id_32 != -1 && (uint32_t)entry_id_32 < count) {
        atomic_store(&index->entry_node, index_get_node(index, (uint32_t)entry_id_32));
        atomic_store(&index->max_level, max_level_h);
    }

    // Pass 3: Edges
    for (uint32_t i = 0; i < count; i++) {
        hvl_hnsw_node *node = index_get_node(index, i);
        if (!node) continue;
        for (uint32_t l = 0; l <= node->level; l++) {
            uint32_t n_count;
            if (fread(&n_count, sizeof(uint32_t), 1, f) != 1) goto load_err;
            hvl_hnsw_node **pool = get_neighbors_at_level(node, l);
            for (uint32_t j = 0; j < n_count; j++) {
                uint32_t neighbor_id;
                if (fread(&neighbor_id, sizeof(uint32_t), 1, f) != 1) goto load_err;
                pool[j] = index_get_node(index, neighbor_id);
            }
            atomic_store(&node->neighbor_counts[l], n_count);
        }
    }

    fclose(f);
    return index;

load_err:
    if (f) fclose(f);
    // Note: index might be partially loaded, but persistence_load is typically called at start
    return NULL;
}
