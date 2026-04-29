#ifndef HVL_DICT_H
#define HVL_DICT_H

#include <stdint.h>
#include <stddef.h>

typedef struct hvl_dict_entry {
    char *key;
    uint32_t value;
    struct hvl_dict_entry *next;
} hvl_dict_entry;

typedef struct {
    hvl_dict_entry **buckets;
    size_t size;
    size_t count;
} hvl_dict;

hvl_dict *hvl_dict_create(size_t initial_size);
void hvl_dict_free(hvl_dict *dict);
int hvl_dict_set(hvl_dict *dict, const char *key, uint32_t value);
int hvl_dict_get(hvl_dict *dict, const char *key, uint32_t *value);
void hvl_dict_remove(hvl_dict *dict, const char *key);

#endif
