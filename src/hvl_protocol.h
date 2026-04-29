#ifndef HVL_PROTOCOL_H
#define HVL_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#define TEXT_BUFFER_SIZE 16384

typedef enum {
    CMD_PING,
    CMD_VSET,
    CMD_VSEARCH,
    CMD_QUIT,
    CMD_TSET,
    CMD_TSEARCH,
    CMD_HGETALL,
    CMD_INFO,
    CMD_SAVE,
    CMD_DELETE,
    CMD_FLUSHDB,
    CMD_UNKNOWN
} hvl_command_type;

typedef struct {
    hvl_command_type type;
    char vec_id[256];
    float *vec_data;
    char text[TEXT_BUFFER_SIZE];
    size_t k;
    char pattern[256];
    size_t limit;
} hvl_command;

hvl_command *hvl_protocol_parse(const char *buffer, size_t len, size_t dim, size_t *consumed_bytes);
void hvl_command_free(hvl_command *cmd);

#endif
