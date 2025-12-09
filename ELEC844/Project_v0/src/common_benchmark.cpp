/**
 * @file common_benchmark.cpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-18
 * @modified Last Modified: 2025-11-18
 *
 * @copyright Copyright (c) 2025
 */

#include "statespace/hypercube.hpp"
#include "statespace/hypersphere.hpp"
#include "statespace/rog.hpp"
#include "util/benchmark.hpp"

#include <chrono>
#include <iostream>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <random>
#include <stdio.h>

int common_benchmark(int argc, char* argv[]) {

    // Setup parameters
    size_t seed = 230984;
    size_t n_obstacles = 100;
    double coverage = 0.5;
    double variation = 0.005;
    size_t dimensions = N_DIMENSIONS;
    size_t n_tests = 100;
    size_t n_samples = 1000000;

    // std::time_t now =
    //     std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now), "%H_%M_%S") << "_com.csv";
    //
    // std::ofstream of(ss.str());

    Benchmark bm("tests/rog_" + std::to_string(dimensions) + "_" +
                     std::to_string(n_obstacles),
                 {{"seed", std::to_string(seed)},
                  {"n_obstacles", std::to_string(n_obstacles)},
                  {"coverage", std::to_string(coverage)},
                  {"variation", std::to_string(variation)},
                  {"tests", std::to_string(n_tests)},
                  {"test_samples", std::to_string(n_samples)},
                  {"type", "sphere"},
                  {"dimensions", std::to_string(dimensions)}});

    std::shared_ptr<WallTimer> t_all = bm.new_timer("all");
    std::shared_ptr<WallTimer> t_setup = bm.new_timer("setup");
    std::shared_ptr<WallTimer> t_gen = bm.new_timer("generation");
    std::shared_ptr<WallTimer> t_search = bm.new_timer("search");

    std::random_device dev;
    std::mt19937 rand(dev());
    std::uniform_real_distribution<double> dist(0, 1.0);

    printf("Starting Benchmarks..\n");

    for (size_t ti = 0; ti < n_tests; ti++) {
        bm.start_benchmark();
        t_all->start();
        t_setup->start();

        size_t n_invalid = 0;

        ompl::base::StateSpacePtr space(
            new ompl::base::RealVectorStateSpace(dimensions));
        // Set the bounds of space to be in [0,1].
        space->as<ompl::base::RealVectorStateSpace>()->setBounds(0.0, 1.0);
        ompl::base::SpaceInformationPtr si(
            new ompl::base::SpaceInformation(space));

        t_setup->stop();

        printf("Starting Generation\n");

        t_gen->start();
        // ROGHypercube rog(si, seed + ti, n_obstacles, coverage);
        srand(seed);
        ROG_t* rog = rog_init(n_obstacles, coverage, variation);
        t_gen->stop();

        printf("Finished Generation\n");

        t_search->start();
        // ompl::base::State* s = si->allocState();
        state_t s;
        for (size_t i = 0; i < n_samples; i++) {
            for (size_t d = 0; d < dimensions; d++) {
                // s->as<ompl::base::RealVectorStateSpace::StateType>()
                //     ->values[d] = dist(rand);
                s[d] = dist(rand);
            }
            if (!rog_check(rog, s)) {
                n_invalid++;
            }
            // if (!rog.isValid(s)) {
            //     n_invalid++;
            // }
        }
        bm.stop_benchmark();
        printf("Benchmark %ld had %ld / %ld (%1.2f) invalid samples, took "
               "%3.4f ms\n",
               ti, n_invalid, n_samples, ((double) n_invalid / n_samples),
               t_all->get_elapsed());
    }

    // of << bm.export_csv().str() << std::endl;

    printf("Writing Output\n");

    bm.export_json();

    auto results = bm.export_mean();
    for (const auto& res : results) {
        std::cout << res.first << ": " << res.second << std::endl;
    }


    return 0;
}
