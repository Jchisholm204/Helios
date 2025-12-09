/**
 * @file main.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief ELEC 844 Final Project
 * @version 0.2
 * @date Created: 2025-11-15
 * @modified Last Modified: 2025-12-09
 *
 * @copyright Copyright (c) 2025
 */

#include "main.h"

#include "display/display.hpp"

#include <iostream>
#include <stdio.h>

int main(int argc, char** argv) {
    (void) argc;
    (void) argv;

    Display d(100, 100);
    while (!d.poll_quit()) {
        d.clear();
        d.label("ELEC 844 Project - FMT* - Jacob Chisholm");
        d.draw_grid(10, 10);
        d.draw_points({{44, 66}, {22, 33}}, {0, 0, 255});
        d.render();
    }
    return 0;
}
