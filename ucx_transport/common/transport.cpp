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
void Transport::progress_loop() {
    ucp_worker_progress(_ucp_worker);
}

void Transport::send_blocking(uint64_t tag, void *buf, size_t len) {
    if (_endpoints.size() == 0) {
        std::cerr << "[UCX] [send_blocking] No endpoints created" << std::endl;
    }
    ucp_ep_h ep = _endpoints[0];
    ucp_request_param_t rparam;
    rparam.op_attr_mask = 0;

    ucs_status_ptr_t pStatus = ucp_tag_send_nbx(ep, buf, len, tag, &rparam);

    if (UCS_PTR_IS_ERR(pStatus)) {
        std::cerr << "[UCX] [send_blocking] fatal error" << std::endl;
    }
    else if (UCS_PTR_IS_PTR(pStatus)) {
        while (ucp_request_check_status(pStatus) == UCS_INPROGRESS) {
            this->progress_loop();
        }
        ucp_request_free(pStatus);
    }
}

void Transport::recv_blocking(uint64_t tag, void *buf, size_t len) {
    uint64_t tag_mask = -1;
    ucp_request_param_t rparam;
    rparam.op_attr_mask = 0;
    ucs_status_ptr_t pStatus =
        ucp_tag_recv_nbx(_ucp_worker, buf, len, tag, tag_mask, &rparam);

    if (UCS_PTR_IS_ERR(pStatus)) {
        std::cerr << "[UCX] [send_blocking] fatal error" << std::endl;
    }
    else if (UCS_PTR_IS_PTR(pStatus)) {
        while (ucp_request_check_status(pStatus) == UCS_INPROGRESS) {
            this->progress_loop();
        }
        ucp_request_free(pStatus);
    }
}
size_t Transport::send(uint32_t tag, uint32_t source, void *buf, size_t len) {
    if (_endpoints.size() == 0) {
        std::cerr << "[UCX] [send] No endpoints to send to" << std::endl;
    }
    ucp_ep_h ep = _endpoints[0];
    ucp_request_param_t rparam;
    rparam.op_attr_mask =
        UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    rparam.cb.send = _ucp_send_callback;
    rparam.user_data = this;

    uint64_t full_tag = ((uint64_t) tag) << 32 | source;

    ucs_status_ptr_t pStatus =
        ucp_tag_send_nbx(ep, buf, len, full_tag, &rparam);

    if (UCS_PTR_IS_ERR(pStatus)) {
        std::cerr << "[UCX] [send] fatal error" << std::endl;
    }
    else if (UCS_PTR_IS_PTR(pStatus)) {
        _msgs_status.push_back(pStatus);
        return _msgs_status.size();
    }
    return 0;
}

size_t Transport::recv(uint32_t tag, uint32_t source, void *buf, size_t len) {
    uint64_t tag_mask = -1;
    ucp_request_param_t rparam;
    rparam.op_attr_mask =
        UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    rparam.cb.recv = _ucp_recv_callback;
    rparam.user_data = this;
    uint64_t full_tag = ((uint64_t) tag) << 32 | source;
    ucs_status_ptr_t pStatus =
        ucp_tag_recv_nbx(_ucp_worker, buf, len, full_tag, tag_mask, &rparam);

    if (UCS_PTR_IS_ERR(pStatus)) {
        std::cerr << "[UCX] [recv] fatal error" << std::endl;
    }
    else if (UCS_PTR_IS_PTR(pStatus)) {
        _msgs_status.push_back(pStatus);
        return _msgs_status.size();
    }
    return 0;
}
bool Transport::check_completion(size_t msg) {
    if (msg == 0) {
        return true;
    }
    if (msg > _msgs_status.size()) {
        return false;
    }

    ucs_status_ptr_t pStatus = _msgs_status[msg - 1];
    if (UCS_PTR_IS_PTR(pStatus) && pStatus) {
        if (ucp_request_check_status(pStatus) == UCS_INPROGRESS) {
            return false;
        }
        else {
            ucp_request_free(pStatus);
            _msgs_status[msg - 1] = NULL;
            return true;
        }
    }
    return false;
}

void Transport::_connection_callback(ucp_conn_request_h conn_request) {
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request = conn_request;

    ucp_ep_h client_ep;
    ucp_ep_create(_ucp_worker, &ep_params, &client_ep);
    _endpoints.push_back(client_ep);
    std::cout << "[UCX] [Server] New connection received" << std::endl;
}

void Transport::_send_callback(void *request, ucs_status_t status) {
    std::cout << "[UCX] Send completed" << std::endl;
}

void Transport::_recv_callback(void *request, ucs_status_t status,
                               const ucp_tag_recv_info_t *tag_info) {

    std::cout << "[UCX] Recv completed from: "
              << (tag_info->sender_tag & 0xFFFF) << std::endl;
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
    listener_params.conn_handler.cb = _ucp_connection_callback;
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
