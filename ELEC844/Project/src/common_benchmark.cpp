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
#include "util/benchmark.hpp"

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
    size_t seed = 1234;
    double scale = 1.8;
    double coverage = 0.5;
    size_t dimensions = 10;
    size_t n_tests = 1;
    size_t n_samples = 1000;

    Benchmark bm("common_util", {{"seed", std::to_string(seed)},
                                 {"scale", std::to_string(scale)},
                                 {"coverage", std::to_string(coverage)},
                                 {"tests", std::to_string(n_tests)},
                                 {"test_samples", std::to_string(n_samples)},
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
        ROGHypercube rog(si, seed, scale, coverage);
        t_gen->stop();

        printf("Finished Generation\n");

        t_search->start();
        ompl::base::State* s = si->allocState();
        for (size_t i = 0; i < n_samples; i++) {
            for (size_t d = 0; d < dimensions; d++) {
                s->as<ompl::base::RealVectorStateSpace::StateType>()
                    ->values[d] = dist(rand);
            }
            if (!rog.isValid(s)) {
                n_invalid++;
            }
        }
        bm.stop_benchmark();
        printf("Benchmark %ld had %ld invalid samples, took %3.4f ms\n", ti,
               n_invalid, t_all->get_elapsed());
    }
    
    // std::cout << bm.export_csv().str() << std::endl;

    auto results = bm.export_mean();
    for (const auto& res : results) {
        std::cout << res.first << ": " << res.second << std::endl;
    }
    
    return 0;
}
