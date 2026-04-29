#ifndef HVL_SERVER_H
#define HVL_SERVER_H

#include "hvl_hnsw.h"
#include "hvl_nn.h"
#include "hvl_tokenizer.h"
#include "hvl_settings.h"
#include <pthread.h>
#include <stdatomic.h>

#define HVL_VERSION "0.1.0"

typedef struct {
    hvl_settings settings;
    hvl_hnsw_index *index;
    hvl_model *model;
    hvl_tokenizer *tokenizer;
    int listen_fd;
    atomic_int dirty;
    pthread_mutex_t index_lock; // PROTECTS THE HNSW INDEX
    pthread_mutex_t save_lock;  // PROTECTS CONCURRENT SAVES
} hvl_server;

void hvl_log(hvl_server *srv, int level, const char *fmt, ...);

hvl_server *hvl_server_create(hvl_settings *settings);
void hvl_server_run(hvl_server *srv);
void hvl_server_free(hvl_server *srv);

#endif
