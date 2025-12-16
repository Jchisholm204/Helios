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
#include "mpi/mpi_plannerbase.h"
#include "ompl/bit.hpp"
#include "ompl/fmt.hpp"
#include "ompl/ompl.hpp"
#include "statespace/geometric_obstacle_generator.h"
#include "statespace/statespace.h"
#include "statespace/statespace_tests.h"
#include "util/hashtable.h"

#include <chrono>
#include <ctime>
#include <iostream>
#include <stdio.h>

int main(int argc, char **argv) {
    (void) argc;
    (void) argv;

    hashtable_t *t = hashtable_init(1 << 8);
    vstate_t s = {{12, 2}, {88, 3}, 3};
    hashtable_insert(t, &s, state_index(s.state));

    vstate_t *k = hashtable_find(t, &s.state, state_index(s.state));
    if (k) {
        printf("Got %d\n", k->parent[0]);
    } else {
        printf("Non\n");
    }
    hashtable_free(&t);

    // mpi_planner_evaluate(argc, argv);
    // _min_heap_test();

    return 0;

    // ompl_evaluate(0.5);
    // ompl_evaluate(1);
    // ompl_evaluate(2);
    // ompl_evaluate(5);
    // ompl_evaluate(10);
    ompl_evaluate(20);
    return 0;

    // return statespace_test_gog(argc, argv);

    // struct ompl_planner *fmt = ompl_init_fmt(5000);
    struct ompl_planner *fmt = ompl_init_bit();
    ompl_solve(fmt, 10);
    struct ompl_metrics *metrics = ompl_get_path(fmt);

    printf("Took %2.2f ms to find path\n", metrics->final.time);
    printf("Path Length: %3.2f (%ld)\n", metrics->final.length,
           metrics->final.n_path);
    printf("Optimal Path Length: %3.2f\n", metrics->final.optimal_length);
    printf("Path Quality: %2.3f\n", metrics->final.quality);
    printf("Collision Checks = %ld\n", fmt->gog->getAccesses());

    std::vector<std::pair<float, float>> path;
    for (size_t i = 0; i < metrics->final.n_path; i++) {
        path.push_back({metrics->final.path[i][0], metrics->final.path[i][1]});
    }

    Display d(100, 100);

    gog_t *gog = fmt->gog->gog;

    auto start = std::chrono::steady_clock::now();

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

    while (!d.poll_quit()) {
        d.clear();
        d.label("ELEC 844 Project - FMT* - Jacob Chisholm");
        d.draw_grid(10, 10);
        d.draw_points({{44, 66}, {22, 33}}, {0, 0, 255});
        d.draw_points(points, {255, 0, 0});
        d.draw_path(path, {0, 255, 0});
        d.render();
    }
    return 0;
}
