/**
 * @file main.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief UCX Transport Client
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
#include <stdio.h>
#include <string.h>
#include <thread>
#include <unistd.h>

int main(int argc, char **argv) {

    Transport tl = Transport("127.0.0.1");

    char *buf = (char *) malloc(1024);

    for (;;) {
        if (tl.n_connections() > 0) {
            std::cout << "[Client] Connection Established" << std::endl;
            break;
        }
        tl.progress_loop();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    tl.register_rma_source(buf, 1024);

    for (size_t i = 0;; i++) {
        *((size_t *) buf) = i;
        std::cout << "Updated to: " << i << std::endl;
        tl.progress_loop();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::cout << "Sent Message";

    return 0;
}
