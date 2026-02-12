/**
 * @file transport.hpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2026-02-10
 * @modified Last Modified: 2026-02-10
 *
 * @copyright Copyright (c) 2026
 */

#ifndef _TRANSPORT_HPP_
#define _TRANSPORT_HPP_

#include <ucp/api/ucp.h>
#include <vector>

#define TRANSPORT_CONN_PORT 9092

class Transport {
  public:
    // Server Init
    Transport();
    // Client Init
    Transport(const char *server_ip);
    ~Transport();

    void progress_loop();

    size_t n_connections(void) { return _endpoints.size(); }
    size_t n_requests(void) { return _msgs_status.size(); }

    void send_blocking(uint64_t tag, void *buf, size_t len);
    void recv_blocking(uint64_t tag, void *buf, size_t len);

    size_t send(uint32_t tag, uint32_t source, void *buf, size_t len);
    size_t recv(uint32_t tag, uint32_t source, void *buf, size_t len);

    bool check_completion(size_t msg);

  private:
    ucp_config_t *_ucp_config;
    ucp_context_h _ucp_context;
    ucp_worker_h _ucp_worker;
    ucp_listener_h _listener;
    std::vector<ucp_ep_h> _endpoints;
    std::vector<ucs_status_ptr_t> _msgs_status;

    void _connection_callback(ucp_conn_request_h conn_request);
    void _send_callback(void *request, ucs_status_t status);
    void _recv_callback(void *request, ucs_status_t status,
                        const ucp_tag_recv_info_t *tag_info);
    void _setup();
    void _setup_server();
    void _setup_client(const char *server_ip);

    static void _ucp_connection_callback(ucp_conn_request_h conn_request,
                                         void *arg) {
        auto *self = static_cast<Transport *>(arg);
        self->_connection_callback(conn_request);
    }

    static void _ucp_send_callback(void *request, ucs_status_t status,
                                   void *user_data) {
        auto *self = static_cast<Transport *>(user_data);
        self->_send_callback(request, status);
    }

    static void _ucp_recv_callback(void *request, ucs_status_t status,
                                   const ucp_tag_recv_info_t *tag_info,
                                   void *user_data) {
        auto *self = static_cast<Transport *>(user_data);
        self->_recv_callback(request, status, tag_info);
    }
};

#endif
