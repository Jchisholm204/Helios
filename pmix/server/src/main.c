/**
 * @file main.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2026-07-09
 * @modified Last Modified: 2026-07-09
 *
 * @copyright Copyright (c) 2026
 */

// Include LogLog from the common library
#include <log.h>
#include <pmix.h>

int main(int argc, char **argv) {
    (void) argc;
    (void) argv;

    pmix_proc_t proc = {0};
    log_info("Server Program");
    PMIx_Init(&proc, NULL, 0);
    log_info("< %s > (%d)", proc.nspace, proc.rank);

    PMIx_Finalize(NULL, 0);
    log_info("Server Program Done");
    return 0;
}
