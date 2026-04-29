#ifndef HVL_CONFIG_H
#define HVL_CONFIG_H

#define HVL_VERSION "0.1.0"
#define HVL_DEFAULT_DIM 128
#define HVL_M 16 // Number of neighbors #define HVL_M 16
#define HVL_M_MAX 16
#define HVL_M_MAX0 32
#define HVL_EF_CONSTRUCTION 150
#define HVL_EF_SEARCH 128
#define HVL_MAX_LEVELS 16 // Max levels for HNSW

// Networking & Pipelining
#define HVL_MAX_BUFFER_SIZE (32 * 1024 * 1024) // 32 MB
#define HVL_MAX_COMMANDS_BATCH 2000

#endif // HVL_CONFIG_H
