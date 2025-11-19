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
#include "ompl/base/StateValidityChecker.h"

#include <vector>

class ROG : public ompl::base::StateValidityChecker {
  public:
    ROG(const ompl::base::SpaceInformationPtr& si);
    ROG(const ompl::base::SpaceInformationPtr& si, size_t seed,
        double scale = 0.1, double coverage = 0.5);

    bool isValid(const ompl::base::State* state) const;

  private:
    struct bounds {
        std::vector<float> high;
        std::vector<float> low;
    };
    std::vector<struct bounds> obs;
    size_t dim;
};

#endif
