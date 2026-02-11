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

#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {

    Transport tl = Transport("127.0.0.1");

    printf("Client Online\n");

    return 0;
}
