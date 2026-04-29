#include "hvl_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_VOCAB_SIZE 300000
#define HASH_SIZE 524288 // 2^19 for ~250k vocab

static uint32_t hash_string(const char *str) {
    uint32_t hash = 5381;
    int c;
    while ((c = *str++)) hash = ((hash << 5) + hash) + c;
    return hash;
}

hvl_tokenizer *hvl_tokenizer_create(const char *vocab_path, int normalization) {
    FILE *f = fopen(vocab_path, "r");
    if (!f) return NULL;

    hvl_tokenizer *t = malloc(sizeof(hvl_tokenizer));
    t->vocab = malloc(sizeof(char *) * MAX_VOCAB_SIZE);
    t->hash_table = malloc(sizeof(int32_t) * HASH_SIZE);
    for (int i = 0; i < HASH_SIZE; i++) t->hash_table[i] = -1;
    t->normalization = normalization;

    char line[512];
    int32_t id = 0;
    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[len-1] = '\0';
            len--;
        }
        t->vocab[id] = strdup(line);
        uint32_t h = hash_string(t->vocab[id]) & (HASH_SIZE - 1);
        while (t->hash_table[h] != -1) h = (h + 1) & (HASH_SIZE - 1);
        t->hash_table[h] = id;
        id++;
        if (id >= MAX_VOCAB_SIZE) break;
    }
    t->size = id;
    t->hash_size = HASH_SIZE;
    fclose(f);
    return t;
}

static int32_t lookup_token(hvl_tokenizer *t, const char *word) {
    uint32_t h = hash_string(word) & (HASH_SIZE - 1);
    while (t->hash_table[h] != -1) {
        int32_t id = t->hash_table[h];
        if (strcmp(t->vocab[id], word) == 0) return id;
        h = (h + 1) & (HASH_SIZE - 1);
    }
    return -1;
}

// STRING NORMALIZATION: Lowercase and strip basic punctuation (OPTIONAL)
static void normalize_text(char *str) {
    char *src = str, *dst = str;
    while (*src) {
        if (ispunct((unsigned char)*src) && *src != ' ') {
            src++; // Skip punctuation
        } else {
            *dst = (char)tolower((unsigned char)*src);
            dst++;
            src++;
        }
    }
    *dst = '\0';
}

size_t hvl_tokenizer_encode(hvl_tokenizer *t, const char *text, int32_t *tokens, size_t max_len) {
    if (max_len < 2) return 0;
    
    char *input = strdup(text);
    if (t->normalization) {
        normalize_text(input);
    }

    size_t count = 0;
    tokens[count++] = 0; // <s>

    char *saveptr;
    char *word_ptr = strtok_r(input, " \t\n\r", &saveptr);
    
    while (word_ptr && count < max_len - 1) {
        char word[1024];
        snprintf(word, sizeof(word), " %s", word_ptr);

        int word_start = 0;
        int word_len = (int)strlen(word);

        while (word_start < word_len) {
            int found_id = -1, found_len = 0;
            for (int end = word_len; end > word_start; end--) {
                char piece[512];
                snprintf(piece, sizeof(piece), "%.*s", end - word_start, word + word_start);
                int32_t id = lookup_token(t, piece);
                if (id != -1) {
                    found_id = id;
                    found_len = end - word_start;
                    break;
                }
            }
            if (found_id != -1) {
                tokens[count++] = found_id;
                word_start += found_len;
                if (count >= max_len - 1) break;
            } else {
                word_start++;
            }
        }
        word_ptr = strtok_r(NULL, " \t\n\r", &saveptr);
    }

    if (count < max_len) tokens[count++] = 2; // </s>
    free(input);
    return count;
}

void hvl_tokenizer_free(hvl_tokenizer *t) {
    if (!t) return;
    for (size_t i = 0; i < t->size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->hash_table);
    free(t);
}
