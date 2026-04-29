#include "hvl_client.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>

hvl_client *hvl_client_connect(const char *host, int port) {
    hvl_client *client = malloc(sizeof(hvl_client));
    if (!client) return NULL;

    client->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client->fd < 0) {
        free(client);
        return NULL;
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    struct hostent *he = gethostbyname(host);
    if (!he) {
        close(client->fd);
        free(client);
        return NULL;
    }
    memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);

    if (connect(client->fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(client->fd);
        free(client);
        return NULL;
    }

    strncpy(client->host, host, sizeof(client->host) - 1);
    client->port = port;
    return client;
}

void hvl_client_disconnect(hvl_client *client) {
    if (!client) return;
    if (client->fd >= 0) close(client->fd);
    free(client);
}

int hvl_client_send_raw(hvl_client *client, const char *cmd, char *response, size_t max_len) {
    if (!client || client->fd < 0) return -1;

    // Send command with newline
    char full_cmd[1024];
    int len = snprintf(full_cmd, sizeof(full_cmd), "%s\r\n", cmd);
    if (write(client->fd, full_cmd, len) != len) return -1;

    // Read response
    ssize_t bytes = read(client->fd, response, max_len - 1);
    if (bytes <= 0) return -1;
    response[bytes] = '\0';
    return 0;
}
