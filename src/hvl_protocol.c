#include "hvl_protocol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h> 

hvl_command *hvl_command_create() {
    hvl_command *cmd = calloc(1, sizeof(hvl_command));
    cmd->limit = 100; 
    strcpy(cmd->pattern, "*"); 
    return cmd;
}

void hvl_command_free(hvl_command *cmd) {
    if (!cmd) return;
    if (cmd->vec_data) free(cmd->vec_data);
    free(cmd);
}

// Advanced token extracter that respects double quotes
static char *extract_token(char **cursor, int rest_of_line) {
    char *start = *cursor;
    while (*start == ' ') start++; // Skip leading spaces
    if (*start == '\0') return NULL;

    if (rest_of_line) {
        *cursor = start + strlen(start);
        return start;
    }

    char *token_end;
    if (*start == '"') {
        start++; // Skip opening quote
        token_end = strchr(start, '"');
        if (token_end) {
            *token_end = '\0';
            *cursor = token_end + 1;
        } else {
            // Unclosed quote, take rest of line
            *cursor = start + strlen(start);
        }
    } else {
        token_end = strchr(start, ' ');
        if (token_end) {
            *token_end = '\0';
            *cursor = token_end + 1;
        } else {
            *cursor = start + strlen(start);
        }
    }
    return start;
}

hvl_command *hvl_protocol_parse(const char *buffer, size_t len, size_t dim, size_t *consumed_bytes) {
    (void)len; // Unused
    *consumed_bytes = 0;
    const char *line_end = strstr(buffer, "\r\n");
    if (!line_end) return NULL;

    size_t line_len = line_end - buffer;
    *consumed_bytes = line_len + 2;

    char *line = malloc(line_len + 1);
    memcpy(line, buffer, line_len);
    line[line_len] = '\0';

    char *cursor = line;
    char *token = extract_token(&cursor, 0);
    
    if (!token) { 
        free(line);
        return NULL; 
    }

    hvl_command *cmd = hvl_command_create();

    if (strcasecmp(token, "PING") == 0) {
        cmd->type = CMD_PING;
    } else if (strcasecmp(token, "QUIT") == 0) {
        cmd->type = CMD_QUIT;
    } else if (strcasecmp(token, "SAVE") == 0) {
        cmd->type = CMD_SAVE;
    } else if (strcasecmp(token, "INFO") == 0) {
        cmd->type = CMD_INFO;
    } else if (strcasecmp(token, "VSET") == 0) {
        cmd->type = CMD_VSET;
        char *id = extract_token(&cursor, 0);
        char *data = extract_token(&cursor, 1);
        if (id && data) {
            strncpy(cmd->vec_id, id, sizeof(cmd->vec_id) - 1);
            cmd->vec_data = malloc(sizeof(float) * dim);
            char *d_ptr = data;
            while (*d_ptr == ' ' || *d_ptr == '[' || *d_ptr == '"') d_ptr++;
            for (size_t i = 0; i < dim; i++) {
                cmd->vec_data[i] = strtof(d_ptr, &d_ptr);
                while (*d_ptr == ',' || *d_ptr == ' ' || *d_ptr == ']' || *d_ptr == '"') d_ptr++;
            }
        }
    } else if (strcasecmp(token, "TSET") == 0) {
        cmd->type = CMD_TSET;
        char *id = extract_token(&cursor, 0);
        char *text = extract_token(&cursor, 1);
        if (id && text) {
            strncpy(cmd->vec_id, id, sizeof(cmd->vec_id) - 1);
            // If text starts/ends with quotes, trim them
            if (text[0] == '"') {
                text++;
                size_t tlen = strlen(text);
                if (tlen > 0 && text[tlen-1] == '"') text[tlen-1] = '\0';
            }
            strncpy(cmd->text, text, sizeof(cmd->text) - 1);
        }
    } else if (strcasecmp(token, "VSEARCH") == 0) {
        cmd->type = CMD_VSEARCH;
        char *k_str = extract_token(&cursor, 0);
        char *data = extract_token(&cursor, 1);
        if (k_str && data) {
            cmd->k = atoi(k_str);
            cmd->vec_data = malloc(sizeof(float) * dim);
            char *d_ptr = data;
            while (*d_ptr == ' ' || *d_ptr == '[' || *d_ptr == '"') d_ptr++;
            for (size_t i = 0; i < dim; i++) {
                cmd->vec_data[i] = strtof(d_ptr, &d_ptr);
                while (*d_ptr == ',' || *d_ptr == ' ' || *d_ptr == ']' || *d_ptr == '"') d_ptr++;
            }
        }
    } else if (strcasecmp(token, "TSEARCH") == 0) {
        cmd->type = CMD_TSEARCH;
        char *k_str = extract_token(&cursor, 0);
        char *text = extract_token(&cursor, 1);
        if (k_str && text) {
            cmd->k = atoi(k_str);
            if (text[0] == '"') {
                text++;
                size_t tlen = strlen(text);
                if (tlen > 0 && text[tlen-1] == '"') text[tlen-1] = '\0';
            }
            strncpy(cmd->text, text, sizeof(cmd->text) - 1);
        }
    } else if (strcasecmp(token, "HGETALL") == 0) {
        cmd->type = CMD_HGETALL;
        char *pattern = extract_token(&cursor, 0);
        if (pattern) {
            strncpy(cmd->pattern, pattern, sizeof(cmd->pattern) - 1);
            char *limit_kw = extract_token(&cursor, 0);
            if (limit_kw && strcasecmp(limit_kw, "LIMIT") == 0) {
                char *limit_val = extract_token(&cursor, 0);
                if (limit_val) cmd->limit = atoi(limit_val);
            }
        }
    } else if (strcasecmp(token, "DELETE") == 0 || strcasecmp(token, "DEL") == 0 || strcasecmp(token, "TDEL") == 0) {
        cmd->type = CMD_DELETE;
        char *id = extract_token(&cursor, 0);
        if (id) strncpy(cmd->vec_id, id, sizeof(cmd->vec_id) - 1);
    } else if (strcasecmp(token, "FLUSHDB") == 0) {
        cmd->type = CMD_FLUSHDB;
    } else {
        cmd->type = CMD_UNKNOWN;
    }

    free(line);
    return cmd;
}
