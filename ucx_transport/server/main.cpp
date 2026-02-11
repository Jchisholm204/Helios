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

#include <iostream>
#include <ucp/api/ucp.h>
#include <unistd.h>

void check_ucx_features() {
    ucp_params_t ucp_params;
    ucp_config_t* config;
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
    } else {
        std::cout << "UCX Initialized Successfully on Narval!" << std::endl;
        ucp_cleanup(ucp_context);
    }
}

int main(int argc, char** argv) {

    Transport tl = Transport();

    check_ucx_features();

    printf("Server Online\n");
    return 0;
}
