#ifndef HVL_PERSISTENCE_H
#define HVL_PERSISTENCE_H

#include "hvl_hnsw.h"

int hvl_persistence_save(hvl_hnsw_index *index, const char *filename);
hvl_hnsw_index *hvl_persistence_load(const char *filename, hvl_dist_func dist_fn);

#endif // HVL_PERSISTENCE_H
