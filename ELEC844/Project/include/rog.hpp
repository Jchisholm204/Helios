/**
 * @file rog.hpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief Random Obstacle Generator
 * @version 0.1
 * @date Created: 2025-11-17
 * @modified Last Modified: 2025-11-17
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _ROG_HPP_
#define _ROG_HPP_

#include <ompl/base/StateValidityChecker.h>
#include <random>
#include <vector>

class ROG : public ompl::base::StateValidityChecker {
  public:
    ROG(const ompl::base::SpaceInformationPtr& si, size_t seed,
        double scale = 0.1, double coverage = 0.5);

    bool isValid(const ompl::base::State* state) const;

  private:
    std::mt19937 mt19937;
    std::uniform_real_distribution<double> rng;
    void gen_obstacles(double scale, double coverage);
    struct bounds {
        std::vector<double> high;
        std::vector<double> low;
    };
    std::vector<struct bounds> obstacles;
    std::vector<double> limits;
    size_t n_dims;
};

#endif
