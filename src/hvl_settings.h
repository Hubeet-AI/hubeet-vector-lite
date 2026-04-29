#ifndef HVL_SETTINGS_H
#define HVL_SETTINGS_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint32_t dim;
    uint32_t M;
    uint32_t ef_construction;
    uint32_t ef_search;
    uint32_t max_levels;
    uint32_t port;
    char bind_addr[64];
    char embedding_model_path[256];
    int tokenizer_normalization;
    int save_interval; // 0 = disabled
    int max_text_length;
    char persistence_path[1024];
    int log_level;           // 0: ERR, 1: INFO, 2: DEBUG
    char log_file[1024];
} hvl_settings;

// Load defaults into settings struct
void hvl_settings_load_defaults(hvl_settings *s);

// Load from file. Returns 0 on success, -1 on error.
int hvl_settings_load_file(hvl_settings *s, const char *path);

// Display current settings
void hvl_settings_print(const hvl_settings *s);

#endif // HVL_SETTINGS_H
