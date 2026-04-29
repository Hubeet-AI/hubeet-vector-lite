#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include "hvl_client.h"

#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"

static void print_help() {
    printf("Commands:\n");
    printf("  PING              - Check server status\n");
    printf("  VSET id [v1,v2]   - Insert raw vector\n");
    printf("  TSET id text      - Insert text (auto-vectorize)\n");
    printf("  VSEARCH k [v1]    - Search raw vector\n");
    printf("  TSEARCH k text    - Search text (auto-vectorize)\n");
    printf("  HGETALL pattern   - List records matching pattern (Glob)\n");
    printf("  TDEL/DEL id       - Delete record by ID\n");
    printf("  INFO              - Get server statistics\n");
    printf("  IMPORT filename   - Bulk import text file (format: id|text)\n");
    printf("  FLUSHDB           - Clear all records from memory\n");
    printf("  SAVE              - Persist index to disk\n");
    printf("  QUIT              - Exit CLI\n");
}

static void draw_progress_bar(int current, int total, double start_time) {
    int width = 40;
    float progress = (float)current / total;
    int pos = (int)(width * progress);

    struct timeval tv;
    gettimeofday(&tv, NULL);
    double now = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
    double elapsed = now - start_time;
    double qps = current / (elapsed > 0 ? elapsed : 1);
    double eta = (total - current) / (qps > 0 ? qps : 1);

    printf("\r" ANSI_COLOR_CYAN "[" ANSI_COLOR_GREEN);
    for (int i = 0; i < width; i++) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf(ANSI_COLOR_CYAN "] " ANSI_COLOR_RESET "%d/%d (%.1f%%) | %.1f qps | ETA: %.1fs ", 
           current, total, progress * 100, qps, eta);
    fflush(stdout);
}

static void handle_import(hvl_client *client, const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf(ANSI_COLOR_RED "Error: Could not open file %s" ANSI_COLOR_RESET "\n", filename);
        return;
    }

    // Count lines
    int total_lines = 0;
    char line[4096];
    while (fgets(line, sizeof(line), f)) total_lines++;
    rewind(f);

    if (total_lines == 0) {
        fclose(f);
        return;
    }

    printf("Importing %d documents from %s...\n", total_lines, filename);
    
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double start_time = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

    int current = 0;
    char res[1024];
    while (fgets(line, sizeof(line), f)) {
        char *id = strtok(line, "|");
        char *text = strtok(NULL, "\n");
        if (id && text) {
            char cmd[5120];
            snprintf(cmd, sizeof(cmd), "TSET %s %s", id, text);
            hvl_client_send_raw(client, cmd, res, sizeof(res));
        }
        current++;
        if (current % 5 == 0 || current == total_lines) {
            draw_progress_bar(current, total_lines, start_time);
        }
    }
    printf("\n" ANSI_COLOR_GREEN "Import complete! Total time: %.2fs" ANSI_COLOR_RESET "\n", 
           ((double)clock()) / CLOCKS_PER_SEC); // Simple clock for total
    fclose(f);
}

int main(int argc, char **argv) {
    const char *host = "127.0.0.1";
    int port = 5555;

    if (argc > 1) host = argv[1];
    if (argc > 2) port = atoi(argv[2]);

    hvl_client *client = hvl_client_connect(host, port);
    if (!client) {
        fprintf(stderr, ANSI_COLOR_RED "Could not connect to Hubeet Engine at %s:%d" ANSI_COLOR_RESET "\n", host, port);
        return 1;
    }

    printf("Connected to " ANSI_COLOR_CYAN "hubeet-vector-lite" ANSI_COLOR_RESET " [%s:%d]\n", host, port);
    printf("Type 'help' for available commands.\n");

    char cmd_buffer[4096];
    char res_buffer[8192];

    while (1) {
        printf(ANSI_COLOR_CYAN "hvl> " ANSI_COLOR_RESET);
        fflush(stdout);

        if (!fgets(cmd_buffer, sizeof(cmd_buffer), stdin)) break;
        cmd_buffer[strcspn(cmd_buffer, "\r\n")] = 0;

        if (strlen(cmd_buffer) == 0) continue;

        if (strcasecmp(cmd_buffer, "help") == 0) {
            print_help();
            continue;
        }
        if (strcasecmp(cmd_buffer, "quit") == 0) break;

        if (strncasecmp(cmd_buffer, "IMPORT ", 7) == 0) {
            handle_import(client, cmd_buffer + 7);
            continue;
        }

        memset(res_buffer, 0, sizeof(res_buffer));
        if (hvl_client_send_raw(client, cmd_buffer, res_buffer, sizeof(res_buffer)) == 0) {
            if (res_buffer[0] == '-') {
                printf(ANSI_COLOR_RED "%s" ANSI_COLOR_RESET, res_buffer);
            } else if (res_buffer[0] == '+' || res_buffer[0] == '$' || res_buffer[0] == '*') {
                // If it's a bulk string, print the content after the length line for better readability
                if (res_buffer[0] == '$') {
                    char *content = strstr(res_buffer, "\r\n");
                    if (content) printf(ANSI_COLOR_GREEN "%s" ANSI_COLOR_RESET, content + 2);
                    else printf(ANSI_COLOR_GREEN "%s" ANSI_COLOR_RESET, res_buffer);
                } else {
                    printf(ANSI_COLOR_GREEN "%s" ANSI_COLOR_RESET, res_buffer);
                }
            } else {
                printf(ANSI_COLOR_YELLOW "%s" ANSI_COLOR_RESET, res_buffer);
            }
        } else {
            printf(ANSI_COLOR_RED "Error: Connection lost" ANSI_COLOR_RESET "\n");
            break;
        }
    }

    hvl_client_disconnect(client);
    return 0;
}
