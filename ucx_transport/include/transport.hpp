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

    size_t n_connections(void){
        return _endpoints.size();
    }

    void send_blocking(uint64_t tag, void *buf, size_t len);
    void recv_blocking(uint64_t tag, void *buf, size_t len);

  private:
    ucp_config_t *_ucp_config;
    ucp_context_h _ucp_context;
    ucp_worker_h _ucp_worker;
    ucp_listener_h _listener;
    std::vector<ucp_ep_h> _endpoints;
    static void _c_connection_handler(ucp_conn_request_h conn_request,
                                      void *arg) {
        auto *self = static_cast<Transport *>(arg);
        self->_handle_connection(conn_request);
    }
    void _handle_connection(ucp_conn_request_h conn_request);
    void _setup();
    void _setup_server();
    void _setup_client(const char *server_ip);
};

#endif
