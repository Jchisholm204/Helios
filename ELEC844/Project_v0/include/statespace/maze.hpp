/**
 * @file maze.hpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-19
 * @modified Last Modified: 2025-11-19
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _MAZE_HPP_
#define _MAZE_HPP_

#include <ompl/base/StateValidityChecker.h>
#include <random>
#include <vector>

class ROGMaze : public ompl::base::StateValidityChecker {
  public:
    ROGMaze(const ompl::base::SpaceInformationPtr& si, size_t seed,
                   size_t n_obstacles = 100, double coverage = 0.5);

    bool isValid(const ompl::base::State* state) const;

    double clearance(const ompl::base::State* state) const;

  private:
    std::mt19937 mt19937;
    std::uniform_real_distribution<double> rng;
    void gen_obstacles(size_t n_obstacles, double coverage);
    struct sphere {
        std::vector<double> center;
        double radius;
        double rad2;
    };
    std::vector<struct sphere> obstacles;
    size_t n_dims;
};

#endif

