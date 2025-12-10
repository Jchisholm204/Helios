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
#include "statespace/statespace.h"
#include "statespace/geometric_obstacle_generator.h"

#include <iostream>
#include <stdio.h>

int main(int argc, char** argv) {
    (void) argc;
    (void) argv;

    Display d(100, 100);
    
    gog_t *gog = gog_init(122345);

    std::vector<std::pair<float, float>> points;
    for(uint8_t x = 0; x < 100; x++){
        for(uint8_t y = 0; y < 100; y++){
            state_t s = {(uint8_t)x, (uint8_t)y};
            if(gog_check(gog, &s)){
                points.push_back({x, y});
            }
        }
    }

    while (!d.poll_quit()) {
        d.clear();
        d.label("ELEC 844 Project - FMT* - Jacob Chisholm");
        d.draw_grid(10, 10);
        d.draw_points({{44, 66}, {22, 33}}, {0, 0, 255});
        d.draw_points(points, {255, 0, 0});
        d.render();
    }
    return 0;
}
