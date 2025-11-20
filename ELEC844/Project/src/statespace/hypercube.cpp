/**
 * @file hypercube.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-18
 * @modified Last Modified: 2025-11-18
 *
 * @copyright Copyright (c) 2025
 */

#include "statespace/hypercube.hpp"

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <random>

ROGHypercube::ROGHypercube(const ompl::base::SpaceInformationPtr& si, size_t seed, size_t n_obstacles,
         double coverage)
    : ompl::base::StateValidityChecker(si), mt19937(seed), rng(0, 1) {
    this->n_dims = si->getStateSpace()->getDimension();
    for (size_t i = 0; i < n_dims; i++)
        limits.push_back(si->getStateSpace()
                             ->as<ompl::base::RealVectorStateSpace>()
                             ->getBounds()
                             .high[i]);
    this->gen_obstacles(n_obstacles, coverage);
}

bool ROGHypercube::isValid(const ompl::base::State* state) const {
    const auto* s = state->as<ompl::base::RealVectorStateSpace::StateType>();

    for (const auto& ob : this->obstacles) {
        int dc = 0;
        for (size_t d = 0; d < n_dims; d++) {
            if (s->values[d] > ob.low[d] && s->values[d] < ob.high[d]) {
                dc++;
            }
        }
        if (dc == n_dims) return false;
    }

    return true;
}


void ROGHypercube::gen_obstacles(size_t n_obstacles, double coverage) {
    double ccover = 0.0;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double avg_size = coverage/n_obstacles;
    double w_avg = pow(avg_size, 1/(double)n_dims);

    while (ccover < coverage) {
        struct bounds obs;
        double volume = 1.0;
        for (size_t i = 0; i < n_dims; i++) {
            double center = dist(mt19937);
            double width = w_avg;
            obs.low.push_back(center - width/2);
            obs.high.push_back(center+ width/2);
            volume *= width;
        }

        obstacles.push_back(obs);
        ccover += volume;
        // printf("Achieved %2.4f / %2.4f Coverage\n", ccover, coverage);
    }
    printf("Generated %ld Obstacles\n", obstacles.size());
}


