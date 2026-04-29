#include "hvl_hnsw.h"
#include "hvl_persistence.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void print_vector(hvl_vector *v) {
    printf("%s: [", v->id);
    for (size_t i = 0; i < v->dim; i++) {
        printf("%.2f%s", v->data[i], i == v->dim - 1 ? "" : ", ");
    }
    printf("]\n");
}

int main() {
    srand(time(NULL));
    size_t dim = 4;
    hvl_hnsw_index *index = hvl_hnsw_create(dim, hvl_dist_l2, 16, 150, 16);
    
    printf("Initializing hubeet-vector-lite test...\n");

    for (int i = 0; i < 5; i++) {
        char id[16];
        sprintf(id, "vec_%d", i);
        hvl_vector *v = hvl_vector_create(dim, id);
        for (size_t j = 0; j < dim; j++) {
            v->data[j] = (float)rand() / RAND_MAX;
        }
        print_vector(v);
        hvl_hnsw_insert(index, v);
    }

    printf("Index count: %zu, Max level: %u\n", index->count, index->max_level);

    printf("Performing similarity search...\n");
    hvl_vector *query = hvl_vector_create(dim, "query_1");
    for (size_t j = 0; j < dim; j++) query->data[j] = 0.5f;

    size_t found;
    hvl_vector **res = hvl_hnsw_search(index, query, 1, &found);
    if (found > 0) {
        printf("Found nearest neighbor: %s\n", res[0]->id);
        free(res);
    }
    hvl_vector_free(query);
    
    // Persistence Test
    printf("\nTesting Fast Persistence (HVL2)...\n");
    if (hvl_persistence_save(index, "test_dump.hvl") != 0) {
        printf("FAILED to save index\n");
        return 1;
    }
    printf("Index saved to test_dump.hvl\n");

    hvl_hnsw_index *loaded = hvl_persistence_load("test_dump.hvl", hvl_dist_l2);
    if (!loaded) {
        printf("FAILED to load index\n");
        return 1;
    }
    printf("Index loaded successfully. Count: %zu, Max Level: %u\n", loaded->count, loaded->max_level);

    if (loaded->count != index->count) {
        printf("FAILED: Count mismatch! Expected %zu, got %zu\n", index->count, loaded->count);
        return 1;
    }

    printf("Verifying search consistency...\n");
    hvl_vector *query2 = hvl_vector_create(dim, "query_verify");
    for (size_t j = 0; j < dim; j++) query2->data[j] = 0.5f;

    size_t found_orig, found_loaded;
    hvl_vector **res_orig = hvl_hnsw_search(index, query2, 1, &found_orig);
    hvl_vector **res_loaded = hvl_hnsw_search(loaded, query2, 1, &found_loaded);

    if (found_orig != found_loaded || strcmp(res_orig[0]->id, res_loaded[0]->id) != 0) {
        printf("FAILED: Search result mismatch!\n");
        printf("Org: %s, Loaded: %s\n", res_orig[0]->id, res_loaded[0]->id);
        return 1;
    }
    printf("SUCCESS: Search results match: %s\n", res_orig[0]->id);

    free(res_orig);
    free(res_loaded);
    hvl_vector_free(query2);
    hvl_hnsw_free(loaded);
    
    hvl_hnsw_free(index);
    printf("Test completed successfully.\n");
    return 0;
}
