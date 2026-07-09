#include <stdio.h>

// Include LogLog from the common library
#include <log.h>

int main(int argc, char **argv) {
    (void) argc;
    (void) argv;

    printf("Server Hello World");
    log_info("Server Program");
    return 0;
}
