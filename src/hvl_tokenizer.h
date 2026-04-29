#ifndef HVL_TOKENIZER_H
#define HVL_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    char **vocab;
    int32_t *token_ids;
    size_t size;
    // Simple hash table for lookups
    int32_t *hash_table;
    size_t hash_size;
    int normalization;
} hvl_tokenizer;

hvl_tokenizer *hvl_tokenizer_create(const char *vocab_path, int normalization);
void hvl_tokenizer_free(hvl_tokenizer *t);

// Encodes text into tokens array. Returns number of tokens.
// tokens array must be pre-allocated (usually 512).
size_t hvl_tokenizer_encode(hvl_tokenizer *t, const char *text, int32_t *tokens, size_t max_len);

#endif // HVL_TOKENIZER_H
