/**
 * @file hypercube.hpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date Created: 2025-11-19
 * @modified Last Modified: 2025-11-19
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _HYPERCUBE_HPP_
#define _HYPERCUBE_HPP_

#include <ompl/base/StateValidityChecker.h>
#include <random>
#include <vector>

class ROGHypercube : public ompl::base::StateValidityChecker {
  public:
    ROGHypercube(const ompl::base::SpaceInformationPtr& si, size_t seed,
        size_t n_obstacles = 100, double coverage = 0.5);

    bool isValid(const ompl::base::State* state) const;

  private:
    std::mt19937 mt19937;
    std::uniform_real_distribution<double> rng;
    void gen_obstacles(size_t n_obstacles, double coverage);
    struct bounds {
        std::vector<double> high;
        std::vector<double> low;
    };
    std::vector<struct bounds> obstacles;
    std::vector<double> limits;
    size_t n_dims;
};

#endif
