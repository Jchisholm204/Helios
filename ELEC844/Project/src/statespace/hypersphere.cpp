/**
 * @file hypersphere.cpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-19
 * @modified Last Modified: 2025-11-19
 *
 * @copyright Copyright (c) 2025
 */

#include "statespace/hypersphere.hpp"

#include <float.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <random>

ROGHypersphere::ROGHypersphere(const ompl::base::SpaceInformationPtr& si,
                               size_t seed, size_t n_obstacles, double coverage)
    : ompl::base::StateValidityChecker(si), mt19937(seed), rng(0, 1) {

    this->n_dims = si->getStateSpace()->getDimension();
    this->gen_obstacles(n_obstacles, coverage);
}

bool ROGHypersphere::isValid(const ompl::base::State* state) const {
    const auto* s = state->as<ompl::base::RealVectorStateSpace::StateType>();
    for (const auto& obs : this->obstacles) {
        double d_sum = 0.0;
        for (size_t d = 0; d < n_dims; d++) {
            d_sum +=
                (s->values[d] - obs.center[d]) * (s->values[d] - obs.center[d]);
        }
        if (d_sum < obs.rad2)
            return false;
    }
    return true;
}

double ROGHypersphere::clearance(const ompl::base::State* state) const {
    const auto* s = state->as<ompl::base::RealVectorStateSpace::StateType>();
    double d_smallest = FLT_MAX;
    for (const auto& obs : this->obstacles) {
        double d_sum = 0.0;
        for (size_t d = 0; d < n_dims; d++) {
            d_sum +=
                (s->values[d] - obs.center[d]) * (s->values[d] - obs.center[d]);
        }
        if ((d_sum - obs.rad2) < d_smallest)
            d_smallest = d_sum - obs.rad2;
    }
    return d_smallest;
}

void ROGHypersphere::gen_obstacles(size_t n_obstacles, double coverage) {
    double ccover = 0.0;
    double avg_size = coverage/n_obstacles;
    double r_avg = pow(avg_size* tgamma((double) n_dims / 2 + 1) /
                           pow(M_PI, (double) n_dims / 2),
                       1 / (double) n_dims);
    // printf("Average Radius = %3.3f\n", r_avg);
    while (ccover < coverage) {
        struct sphere s;
        // s.radius = rng(mt19937) * r_avg;
        s.radius = r_avg;
        for (size_t i = 0; i < n_dims; i++) {
            s.center.push_back(rng(mt19937));
        }
        ccover += pow(M_PI, n_dims / 2.0) / tgamma(n_dims / 2.0 + 1.0) *
                  pow(s.radius, n_dims);
        s.rad2 = s.radius * s.radius;
        obstacles.push_back(s);
        // printf("Cover = %3.2f/%3.2f\n", ccover, coverage);
    }
}
