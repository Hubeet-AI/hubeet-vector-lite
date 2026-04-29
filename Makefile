CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c11 -march=native -ffast-math -I./src
LIBS = -lm -lpthread

SRC_DIR = src
TEST_DIR = tests
OBJ_DIR = obj

# Core engine objects
CORE_OBJS = $(OBJ_DIR)/hvl_dict.o $(OBJ_DIR)/hvl_hnsw.o $(OBJ_DIR)/hvl_persistence.o $(OBJ_DIR)/hvl_pq.o \
            $(OBJ_DIR)/hvl_protocol.o $(OBJ_DIR)/hvl_server.o $(OBJ_DIR)/hvl_vector.o \
            $(OBJ_DIR)/hvl_settings.o $(OBJ_DIR)/hvl_nn.o $(OBJ_DIR)/hvl_tokenizer.o

TARGET_TEST = test_hnsw
TARGET_SERVER = hvl-server
TARGET_BENCH = hvl-bench
TARGET_CLI = hvl-cli
TARGET_STRESS = hvl-stress

all: $(TARGET_TEST) $(TARGET_SERVER) $(TARGET_BENCH) $(TARGET_CLI) $(TARGET_STRESS)

$(TARGET_SERVER): $(CORE_OBJS) $(OBJ_DIR)/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(TARGET_TEST): $(CORE_OBJS) $(OBJ_DIR)/test_hnsw.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(TARGET_BENCH): $(CORE_OBJS) $(OBJ_DIR)/hvl_benchmark.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(TARGET_CLI): $(CORE_OBJS) $(OBJ_DIR)/hvl_client.o $(OBJ_DIR)/hvl_cli.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(TARGET_STRESS): $(OBJ_DIR)/hvl_stress.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/test_hnsw.o: $(TEST_DIR)/test_hnsw.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/hvl_benchmark.o: $(TEST_DIR)/hvl_benchmark.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/hvl_stress.o: $(TEST_DIR)/hvl_stress.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET_TEST) $(TARGET_SERVER) $(TARGET_BENCH) $(TARGET_CLI) $(TARGET_STRESS)

.PHONY: all clean
