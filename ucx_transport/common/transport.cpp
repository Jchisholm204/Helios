/**
 * @file transport.cpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2026-02-10
 * @modified Last Modified: 2026-02-10
 *
 * @copyright Copyright (c) 2026
 */

#include "transport.hpp"

#include <arpa/inet.h>
#include <iostream>
#include <netdb.h>
#include <stdlib.h>
#include <string.h>

Transport::Transport() {

    std::cout << "[UCX] Setting up Server..." << std::endl;

    _setup();
    _setup_server();

    std::cout << "[UCX] Initialized" << std::endl;
}

Transport::Transport(const char *server_ip) {

    std::cout << "[UCX] Setting up Client..." << std::endl;

    _setup();
    _setup_client(server_ip);

    std::cout << "[UCX] Initialized" << std::endl;
}

Transport::~Transport() {
    ucp_config_release(_ucp_config);
    ucp_cleanup(_ucp_context);
    std::cout << "[UCX] Exited" << std::endl;
}

void Transport::_handle_connection(ucp_conn_request_h conn_request) {
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request = conn_request;

    ucp_ep_h client_ep;
    ucp_ep_create(_ucp_worker, &ep_params, &client_ep);
    _endpoints.push_back(client_ep);
}

void Transport::_setup() {
    ucs_status_t status;

    ucp_params_t ucp_params;
    status = ucp_config_read(NULL, NULL, &_ucp_config);
    if (status != UCS_OK) {
        std::cerr << "[UCX] Failed Configuration Read" << std::endl;
        exit(1);
    }
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_TAG;

    status = ucp_init(&ucp_params, _ucp_config, &_ucp_context);
    if (status != UCS_OK) {
        std::cerr << "[UCX] Failed Initialization" << std::endl;
        exit(1);
    }

    ucp_worker_params_t worker_params;
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    status = ucp_worker_create(_ucp_context, &worker_params, &_ucp_worker);
    if (status != UCS_OK) {
        std::cerr << "[UCX] Failed to create worker" << std::endl;
        exit(1);
    }
}
void Transport::_setup_server() {
    ucp_listener_params_t listener_params;
    listener_params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                 UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    struct sockaddr_in listener_addr;
    listener_addr.sin_family = AF_INET;
    listener_addr.sin_port = htons(TRANSPORT_CONN_PORT);
    inet_pton(AF_INET, "127.0.0.1", &listener_addr.sin_addr);
    listener_params.sockaddr.addr = (struct sockaddr *) &listener_addr;
    listener_params.sockaddr.addrlen = sizeof(listener_addr);
    listener_params.conn_handler.cb = _c_connection_handler;
    listener_params.conn_handler.arg = this;

    ucs_status_t status = UCS_OK;
    status = ucp_listener_create(_ucp_worker, &listener_params, &_listener);
    if (status != UCS_OK) {
        std::cerr << "[UCX] [Server] Failed to create listener" << std::endl;
        exit(1);
    }
    std::cout << "[UCX] [Server] Online" << std::endl;
}
void Transport::_setup_client(const char *server_ip) {
    struct sockaddr_in listener_addr;
    listener_addr.sin_family = AF_INET;
    listener_addr.sin_port = htons(TRANSPORT_CONN_PORT);
    inet_pton(AF_INET, server_ip, &listener_addr.sin_addr);

    ucp_ep_params_t ep_params;
    ep_params.field_mask =
        UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR;
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = (struct sockaddr *) &listener_addr;
    ep_params.sockaddr.addrlen = sizeof(listener_addr);

    ucs_status_t status = UCS_OK;
    ucp_ep_h server_ep;
    if ((status = ucp_ep_create(_ucp_worker, &ep_params, &server_ep))) {
        std::cerr << "[UCX] [Client] Failed to create endpoint" << std::endl;
        exit(1);
    }
    _endpoints.push_back(server_ep);
    std::cout << "[UCX] [Client] Online" << std::endl;
}
