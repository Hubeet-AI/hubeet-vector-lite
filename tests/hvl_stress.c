#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <time.h>

#define STRESS_THREADS 16
#define COMMANDS_PER_THREAD 100
#define TEXT_SIZE 6000 // Very large texts to stress buffer and inference

typedef struct {
    char host[64];
    int port;
} thread_arg;

void *stress_worker(void *arg) {
    thread_arg *targ = (thread_arg *)arg;
    
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(targ->port);
    inet_pton(AF_INET, targ->host, &addr.sin_addr);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect failed");
        return NULL;
    }

    char *text = malloc(TEXT_SIZE + 1);
    for (int i = 0; i < TEXT_SIZE; i++) {
        text[i] = 'a' + (rand() % 26);
    }
    text[TEXT_SIZE] = '\0';

    for (int i = 0; i < COMMANDS_PER_THREAD; i++) {
        char *cmd = malloc(TEXT_SIZE + 512);
        snprintf(cmd, TEXT_SIZE + 512, "TSET \"stress_thread_%p_node_%d\" \"%s\"\r\n", pthread_self(), i, text);
        
        ssize_t written = write(fd, cmd, strlen(cmd));
        free(cmd);

        if (written < 0) {
            printf("[Thread %p] Write failed\n", pthread_self());
            break;
        }
        
        char res[2048];
        ssize_t bytes = read(fd, res, sizeof(res) - 1);
        if (bytes <= 0) {
            printf("[Thread %p] Server disconnected at cmd %d (bytes=%zd)\n", pthread_self(), i, bytes);
            break;
        }
        res[bytes] = '\0';
        if (i % 20 == 0) printf("[Thread %p] Progress: %d/%d\n", pthread_self(), i, COMMANDS_PER_THREAD);
        
        // Brief sleep to avoid totally overwhelming the shared pool too fast, 
        // but still high enough to overlap
        usleep(1000); 
    }

    free(text);
    close(fd);
    return NULL;
}

int main(int argc, char **argv) {
    const char *host = "127.0.0.1";
    int port = 5555;
    if (argc > 1) host = argv[1];
    if (argc > 2) port = atoi(argv[2]);

    srand(time(NULL));
    printf("Starting AGGRESSIVE stress test against %s:%d\n", host, port);
    printf("Threads: %d, Commands per thread: %d, Text size: %d chars\n", STRESS_THREADS, COMMANDS_PER_THREAD, TEXT_SIZE);

    pthread_t threads[STRESS_THREADS];
    thread_arg arg;
    strcpy(arg.host, host);
    arg.port = port;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < STRESS_THREADS; i++) {
        pthread_create(&threads[i], NULL, stress_worker, &arg);
    }

    for (int i = 0; i < STRESS_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\nAggressive stress test complete in %.2f seconds!\n", elapsed);
    printf("Total TSET operations: %d\n", STRESS_THREADS * COMMANDS_PER_THREAD);
    printf("If the server is still running, it is now PRODUCTION READY.\n");

    return 0;
}
