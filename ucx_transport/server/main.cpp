/**
 * @file main.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief UCX Transport Server
 * @version 0.1
 * @date Created: 2026-02-10
 * @modified Last Modified: 2026-02-10
 *
 * @copyright Copyright (c) 2026
 */

#include "main.h"

#include "transport.hpp"

#include <chrono>
#include <iostream>
#include <thread>
#include <ucp/api/ucp.h>
#include <unistd.h>

void check_ucx_features() {
    ucp_params_t ucp_params;
    ucp_config_t *config;
    ucp_context_h ucp_context;

    // Initialize UCX config
    if (ucp_config_read(NULL, NULL, &config) != UCS_OK) {
        std::cerr << "Failed to read UCX config" << std::endl;
        return;
    }

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA;

    // Initialize context
    ucs_status_t status = ucp_init(&ucp_params, config, &ucp_context);
    ucp_config_release(config);

    if (status != UCS_OK) {
        std::cerr << "UCX Init failed!" << std::endl;
    }
    else {
        std::cout << "UCX Initialized Successfully on Narval!" << std::endl;
        ucp_cleanup(ucp_context);
    }
}

int main(int argc, char **argv) {

    (void) argc;
    (void) argv;

    Transport tl = Transport();

    std::vector<char *> rbufs;

    for (;;) {
        if (tl.n_connections() > 0) {
            std::cout << "[Server] Connection Established" << std::endl;
            break;
        }
        tl.progress_loop();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    void *rbuf = NULL;
    ucp_rkey_h rkey;
    tl.register_rma_remote(&rbuf, &rkey);

    for (size_t i = 0;; i++) {
        size_t ri = 0;
        tl.read_remote(&ri, sizeof(ri), (uintptr_t) rbuf, rkey);
        std::cout << "Got: " << ri << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // size_t n_connections = tl.n_connections();

    // for (;;) {
    //     if ((tl.n_connections() - n_connections) > 0) {
    //         std::cout << "[Server] New Connection Established" << std::endl;
    //         n_connections = tl.n_connections();
    //         char *rbuf = (char *) malloc(BUF_SIZE);
    //         tl.recv(3, 0, rbuf, 1024);
    //         rbufs.push_back(rbuf);
    //         std::cout << "[Server] Recv buffer posted" << std::endl;
    //     }
    //     tl.progress_loop();
    //     for (size_t i = 1; i <= n_connections; i++) {
    //         if (tl.check_completion(i)) {
    //             std::cout << "[Server] Recv " << i
    //                       << " Completed: " << rbufs[i - 1] << std::endl;
    //             // free(rbufs[i - 1]);
    //         }
    //     }
    //     std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // }

    return 0;
}
