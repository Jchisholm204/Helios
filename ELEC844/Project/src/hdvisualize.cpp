/**
 * @file hdvisualize.cpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-21
 * @modified Last Modified: 2025-11-21
 *
 * @copyright Copyright (c) 2025
 */

#include "display/display.hpp"
#include "main.h"

#include <stdio.h>

int hdvisualize(int argc, char* argv[]) {
    printf("Hello World\n");
    Display d(100, 100);

    std::vector<std::vector<std::pair<float, float>>> regions = {
        {{0, 0.2}, {0.6, 0.8}}, 
        {{0, 0.1}, {0.7, 0.8}}, 
        {{0.2, 0.3}, {0.5, 0.6}}, 
        {{0.1, 0.3}, {0.5, 0.7}}};

    while (!d.poll_quit()) {
        d.clear();
        d.label("MultiD test viewer");
        d.draw_dims(regions);
        d.render();
    }

    return 0;
}
