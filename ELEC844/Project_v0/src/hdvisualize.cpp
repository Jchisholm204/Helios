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
#include "statespace/rog.hpp"

#include <random>
#include <stdio.h>

int hdvisualize(int argc, char* argv[]) {
    printf("Hello World\n");
    Display d(100, 100);

    std::vector<std::vector<std::pair<float, float>>> regions(N_DIMENSIONS);
    // std::vector<std::vector<std::pair<float, float>>> regions = {
    //     {{0, 0.2}, {0.6, 0.8}},
    //     {{0, 0.1}, {0.7, 0.8}},
    //     {{0.2, 0.3}, {0.5, 0.6}},
    //     {{0, 0.1}, {0.7, 0.8}},
    //     {{0.2, 0.3}, {0.5, 0.6}},
    //     {{0.1, 0.3}, {0.5, 0.7}}};

    ROG_t* rog = rog_init(100, 0.5, 0.005);
    srand(230984);
    std::random_device dev;
    std::mt19937 rand(dev());
    std::uniform_real_distribution<double> dist(0, 1.0);

    size_t n_iterations = 500;
    size_t n_invalid = 0;

    for (size_t i = 0; i < n_iterations; i++) {
        state_t s;
        for (int sd = 0; sd < N_DIMENSIONS; sd++) {
            s[sd] = dist(rand);
        }
        if (rog_check(rog, s)) {
            n_invalid++;
            for (int j = 0; j < N_DIMENSIONS; j++) {
                regions[j].push_back({s[j], 0});
            }
        }
    }
    printf("%ld Regions\n", regions.size());
    for (int i = 0; i < regions.size(); i++) {
        printf("d=%d size=%ld\n", i, regions[i].size());
    }
    printf("%ld / %ld (%3.4f %%) Coverage\n", n_invalid, n_iterations,
           (float) n_invalid / n_iterations);
    while (!d.poll_quit()) {
        d.clear();
        d.label("MultiD test viewer");
        d.draw_dims(regions);
        d.render();
    }

    return 0;
}
