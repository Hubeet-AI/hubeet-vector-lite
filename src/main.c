#include "hvl_server.h"
#include "hvl_settings.h"
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>
#include <string.h>

// Access the global server instance from hvl_server.c
extern hvl_server *global_srv;

void crash_handler(int sig) {
    void *array[10];
    size_t size;

    char msg[1024];
    snprintf(msg, sizeof(msg), "\n--- CRITICAL ERROR: Server crashed (Signal %d) ---", sig);
    fprintf(stderr, "%s\n", msg);
    
    // Attempt to log to file if server exists
    if (global_srv) {
        hvl_log(global_srv, 0, "CRITICAL CRASH: Signal %d caught.", sig);
        char cwd[512];
        if (getcwd(cwd, sizeof(cwd))) {
            hvl_log(global_srv, 0, "Current Working Directory: %s", cwd);
        }
    }

    size = backtrace(array, 10);
    fprintf(stderr, "Backtrace symbols:\n");
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    
    // Also try to write backtrace to log file
    if (global_srv) {
        FILE *f = fopen(global_srv->settings.log_file, "a");
        if (f) {
            fprintf(f, "--- STACK TRACE ---\n");
            backtrace_symbols_fd(array, size, fileno(f));
            fprintf(f, "-------------------\n");
            fclose(f);
        }
    }
    
    fprintf(stderr, "-------------------------------------------------\n");
    exit(1);
}

int main(int argc, char **argv) {
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    signal(SIGFPE, crash_handler);
    signal(SIGBUS, crash_handler);

    hvl_settings settings;
    hvl_settings_load_defaults(&settings);

    if (argc > 1) {
        if (hvl_settings_load_file(&settings, argv[1]) == 0) {
            printf("Loaded configuration from %s\n", argv[1]);
        }
    } else {
        hvl_settings_load_file(&settings, "hvl.conf");
    }

    printf("Starting hubeet-vector-lite server v%s\n", HVL_VERSION);
    char cwd[512];
    if (getcwd(cwd, sizeof(cwd))) {
        printf("Working Directory: %s\n", cwd);
    }

    hvl_settings_print(&settings);
    
    hvl_server *srv = hvl_server_create(&settings);
    if (!srv) {
        fprintf(stderr, "Failed to initialize server.\n");
        return 1;
    }
    
    hvl_server_run(srv);
    hvl_server_free(srv);
    return 0;
}
