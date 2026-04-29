#include "hvl_dict.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static uint64_t hash_string(const char *str) {
    uint64_t hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;
    return hash;
}

hvl_dict *hvl_dict_create(size_t initial_size) {
    hvl_dict *dict = malloc(sizeof(hvl_dict));
    dict->size = initial_size;
    dict->count = 0;
    dict->buckets = calloc(initial_size, sizeof(hvl_dict_entry*));
    return dict;
}

void hvl_dict_free(hvl_dict *dict) {
    if (!dict) return;
    for (size_t i = 0; i < dict->size; i++) {
        hvl_dict_entry *entry = dict->buckets[i];
        while (entry) {
            hvl_dict_entry *next = entry->next;
            free(entry->key);
            free(entry);
            entry = next;
        }
    }
    free(dict->buckets);
    free(dict);
}

int hvl_dict_set(hvl_dict *dict, const char *key, uint32_t value) {
    uint64_t hash = hash_string(key) % dict->size;
    hvl_dict_entry *entry = dict->buckets[hash];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            entry->value = value;
            return 0;
        }
        entry = entry->next;
    }
    entry = malloc(sizeof(hvl_dict_entry));
    entry->key = strdup(key);
    entry->value = value;
    entry->next = dict->buckets[hash];
    dict->buckets[hash] = entry;
    dict->count++;
    return 0;
}

int hvl_dict_get(hvl_dict *dict, const char *key, uint32_t *value) {
    uint64_t hash = hash_string(key) % dict->size;
    hvl_dict_entry *entry = dict->buckets[hash];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            *value = entry->value;
            return 1;
        }
        entry = entry->next;
    }
    return 0;
}

void hvl_dict_remove(hvl_dict *dict, const char *key) {
    uint64_t hash = hash_string(key) % dict->size;
    hvl_dict_entry *entry = dict->buckets[hash];
    hvl_dict_entry *prev = NULL;
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            if (prev) prev->next = entry->next;
            else dict->buckets[hash] = entry->next;
            free(entry->key);
            free(entry);
            dict->count--;
            return;
        }
        prev = entry;
        entry = entry->next;
    }
}
