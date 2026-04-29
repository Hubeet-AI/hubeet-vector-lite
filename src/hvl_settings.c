#include "hvl_settings.h"
#include "hvl_config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

void hvl_settings_load_defaults(hvl_settings *s) {
    s->dim = HVL_DEFAULT_DIM;
    s->M = HVL_M;
    s->ef_construction = HVL_EF_CONSTRUCTION;
    s->ef_search = HVL_EF_SEARCH;
    s->max_levels = HVL_MAX_LEVELS;
    s->port = 5555;
    strcpy(s->bind_addr, "0.0.0.0");
    strcpy(s->embedding_model_path, "");
    s->tokenizer_normalization = 1;
    s->save_interval = 15;
    s->max_text_length = 4096;
    strcpy(s->persistence_path, "dump.hvl");
    s->log_level = 1; // INFO
    strcpy(s->log_file, "hvl.log");
}

static char *trim(char *str) {
    char *end;
    while(isspace((unsigned char)*str)) str++;
    if(*str == 0) return str;
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

int hvl_settings_load_file(hvl_settings *s, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char *ptr = trim(line);
        if (ptr[0] == '#' || ptr[0] == '\0') continue;

        char *delim = strchr(ptr, '=');
        if (!delim) continue;

        *delim = '\0';
        char *key = trim(ptr);
        char *val = trim(delim + 1);

        if (strcmp(key, "dim") == 0) s->dim = atoi(val);
        else if (strcmp(key, "M") == 0) s->M = atoi(val);
        else if (strcmp(key, "ef_construction") == 0) s->ef_construction = atoi(val);
        else if (strcmp(key, "ef_search") == 0) s->ef_search = atoi(val);
        else if (strcmp(key, "max_levels") == 0) s->max_levels = atoi(val);
        else if (strcmp(key, "port") == 0) s->port = atoi(val);
        else if (strcmp(key, "bind_addr") == 0) strncpy(s->bind_addr, val, sizeof(s->bind_addr)-1);
        else if (strcmp(key, "embedding_model_path") == 0) strncpy(s->embedding_model_path, val, sizeof(s->embedding_model_path)-1);
        else if (strcmp(key, "tokenizer_normalization") == 0) s->tokenizer_normalization = atoi(val);
        else if (strcmp(key, "save_interval") == 0) s->save_interval = atoi(val);
        else if (strcmp(key, "max_text_length") == 0) s->max_text_length = atoi(val);
        else if (strcmp(key, "persistence_path") == 0) strncpy(s->persistence_path, val, sizeof(s->persistence_path)-1);
        else if (strcmp(key, "log_level") == 0) s->log_level = atoi(val);
        else if (strcmp(key, "log_file") == 0) strncpy(s->log_file, val, sizeof(s->log_file)-1);
    }

    fclose(f);
    return 0;
}

void hvl_settings_print(const hvl_settings *s) {
    printf("--- Hubeet Vector Engine Settings ---\n");
    printf("Dimension: %u\n", s->dim);
    printf("M: %u\n", s->M);
    printf("EF Construction: %u\n", s->ef_construction);
    printf("EF Search: %u\n", s->ef_search);
    printf("Max Levels: %u\n", s->max_levels);
    printf("Bind: %s:%u\n", s->bind_addr, s->port);
    if (s->embedding_model_path[0]) {
        printf("Inference Model: %s\n", s->embedding_model_path);
        printf("Inference Normalization: %s\n", s->tokenizer_normalization ? "Enabled" : "Disabled");
    }
    printf("Auto-Save Interval: %d seconds\n", s->save_interval);
    printf("Max Text Length: %d chars\n", s->max_text_length);
    printf("Persistence Path: %s\n", s->persistence_path);
    printf("Log Level: %d\n", s->log_level);
    printf("Log File: %s\n", s->log_file);
    printf("-------------------------------------\n");
}
