#ifndef HVL_CLIENT_H
#define HVL_CLIENT_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int fd;
    char host[128];
    int port;
} hvl_client;

// Connect to the server
hvl_client *hvl_client_connect(const char *host, int port);

// Disconnect
void hvl_client_disconnect(hvl_client *client);

// Send a raw command and get response
// Returns 0 on success, -1 on error
int hvl_client_send_raw(hvl_client *client, const char *cmd, char *response, size_t max_len);

#endif // HVL_CLIENT_H
