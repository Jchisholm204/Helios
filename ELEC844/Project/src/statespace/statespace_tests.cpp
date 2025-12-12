/**
 * @file statespace_tests.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-12
 * @modified Last Modified: 2025-12-12
 *
 * @copyright Copyright (c) 2025
 */

#include "statespace/statespace_tests.h"

#include "display/display.hpp"
#include "statespace/geometric_obstacle_generator.h"
#include "statespace/statespace.h"

#include <chrono>
#include <ctime>
#include <stdio.h>

int statespace_test_gog(int argc, char **argv) {
    (void) argc;
    (void) argv;

    printf("Initializing GOG Tests... STATESPACE_DIMS=%d\n", STATESPACE_DIMS);

    Display d(STATESPACE_MAX, STATESPACE_MAX);

    size_t seed = 93847468;
    uint bmask = 2;
    uint pmask = 3;
    gog_t *gog = gog_init(seed, bmask, pmask);

    printf("Initialied GOG: seed=%ld bmask=%d pmask=%d\n", seed, bmask, pmask);

    auto start = std::chrono::steady_clock::now();

    printf("Expected Invalid Ratio = (1/2)^d = (1/2)^%d = %0.3f\n",
           STATESPACE_DIMS, pow(0.5, STATESPACE_DIMS));
    size_t n_invalid = 0;
    size_t n_valid = 0;
    for (size_t i = 0; i < 1000000; i++) {
        // state_t s = {(uint8_t) rand(), (uint8_t) rand()};
        state_t s = {0};
        for (int d = 0; d < STATESPACE_DIMS; d++)
            s[d] = (uint8_t) rand();
        if (gog_check(gog, &s)) {
            n_invalid++;
        } else {
            n_valid++;
        }
    }
    printf("Found %ld invalid / %ld valid points in %ld ms\n", n_invalid,
           n_valid,
           std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start)
               .count());

    printf("Plotting invalid points, z=0\n");
    std::vector<std::pair<float, float>> points;
    for (uint8_t x = 0; x < 100; x++) {
        for (uint8_t y = 0; y < 100; y++) {
            // state_t s = {(uint8_t) x, (uint8_t) y, 10};
            state_t s = {(uint8_t) x, (uint8_t) y};
            if (gog_check(gog, &s)) {
                points.push_back({x, y});
            }
        }
    }

    std::vector<std::pair<float, float>> s_t_points;
    printf("Attempting to find start point\n");
    state_t *p_start = gog_start(gog);
    if(!p_start){
        printf("Could Not find start point!!\n");
    }
    else{
        s_t_points.push_back({(*p_start)[0], (*p_start)[1]});
        printf("Found Start point: ");
        for(size_t i = 0; i < STATESPACE_DIMS; i++)
            printf("%d ", (*p_start)[i]);
        printf("\n");
    }

    printf("Attempting to find target point\n");
    state_t *p_target = gog_target(gog);
    if(!p_target){
        printf("Could Not find target point!!\n");
    }
    else{
        s_t_points.push_back({(*p_target)[0], (*p_target)[1]});
        printf("Found Target point: ");
        for(size_t i = 0; i < STATESPACE_DIMS; i++)
            printf("%d ", (*p_target)[i]);
        printf("\n");
    }
    while (!d.poll_quit()) {
        d.clear();
        d.label("ELEC 844 Project - FMT* - Jacob Chisholm");
        d.draw_grid(10, 10);
        d.draw_points({{44, 66}, {22, 33}}, {0, 0, 255});
        d.draw_points(points, {255, 0, 0});
        d.draw_points(s_t_points, {200, 0, 255});
        d.render();
    }
    return 0;
}
