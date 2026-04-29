#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

int main() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(5555) };
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect failed");
        return 1;
    }

    // Exact ID from the user log
    const char *id = "Second Brain/Solunika/Solúnika Ground Truth/Ataque de chons con rce de next.md#chunk3";
    
    // Exact long text from the user log (reproduced)
    const char *text = 
        "path: '/app/.next/server/app/asd67.php.html'\n"
        "}\n"
        "Failed to update prerender cache for /blog.env [Error: EACCES: permission denied, open '/app/.next/server/app/blog.env.html'] {\n"
        "  errno: -13,\n"
        "  code: 'EACCES',\n"
        "  syscall: 'open',\n"
        "  path: '/app/.next/server/app/blog.env.html'\n"
        "}\n"
        "Failed to update prerender cache for /file.php [Error: EACCES: permission denied, open '/app/.next/server/app/file.php.html'] {\n"
        "  errno: -13,\n"
        "  code: 'EACCES',\n"
        "  syscall: 'open',\n"
        "  path: '/app/.next/server/app/file.php.html'\n"
        "}\n"
        "Failed to update prerender cache for /file1.php [Error: EACCES: permission denied, open '/app/.next/server/app/file1.php.html'] {\n"
        "  errno: -13,\n"
        "  code: 'EACCES',\n"
        "  syscall: 'open',\n"
        "  path: '/app/.next/server/app/file1.php.html'\n"
        "}\n"
        "Failed to update prerender cache for /file2.php [Error: EACCES: permission denied, open '/app/.next/server/app/file2.php.html'] {\n"
        "  errno: -13,\n"
        "  code: 'EACCES',\n"
        "  syscall: 'open',\n"
        "  path: '/app/.next/server/app/file2.php.html'\n"
        "}\n";

    for (int i = 0; i < 20; i++) {
        char cmd[8192];
        snprintf(cmd, sizeof(cmd), "TSET \"%s#%d\" \"%s\"\r\n", id, i, text);
        
        printf("Sending TSET %d...\n", i);
        if (write(fd, cmd, strlen(cmd)) < 0) break;
        
        char res[1024];
        ssize_t bytes = read(fd, res, sizeof(res) - 1);
        if (bytes <= 0) {
            printf("Server disconnected at iteration %d\n", i);
            break;
        }
        res[bytes] = '\0';
        printf("Server: %s", res);
        usleep(10000); // 10ms
    }

    close(fd);
    return 0;
}
