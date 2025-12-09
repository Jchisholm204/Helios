/**
 * @file noisetest.cpp
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
#include "statespace/rog.hpp"

#include <random>
#include <stdio.h>

int noisetest(int argc, char* argv[]) {
    Display d(100, 100);

    std::random_device dev;
    std::mt19937 rand(dev());
    std::uniform_real_distribution<double> dist(20, 30);

    std::vector<std::vector<std::pair<float, float>>> regions(2);

    // double per_x = M_PI/(dist(rand));
    // double per_y = M_PI/(dist(rand));
    double per_x = 0.335;
    double per_y = 0.31;
    printf("Period x=%2.5f y=%2.5f\n", per_x, per_y);

    std::vector<std::pair<float, float>> ivpts;
    for (int x = 0; x < 100; x++) {
        bool x_valid = sin(x * per_x) + sin(x*2+ M_PI / 3) > 0;
        if(!x_valid)
            regions[0].push_back({x/100.0, 12});
        for (int y = 0; y < 100; y++) {
            bool y_valid = sin(y * per_y) - sin(y*3+ M_PI / 2) > 0;
            if(!y_valid)
                regions[1].push_back({y/100.0, 12});
            if (!y_valid && !x_valid) {
                ivpts.push_back({x, y});
            }
        }
    }
    while (!d.poll_quit()) {
        d.clear();
        d.draw_points(ivpts, {255, 0, 0});
        d.draw_dims(regions);
        d.render();
    }
    return 0;
}
