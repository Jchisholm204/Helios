/**
 * @file rog.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-18
 * @modified Last Modified: 2025-11-18
 *
 * @copyright Copyright (c) 2025
 */

#include "rog.hpp"

#include "ompl/base/spaces/RealVectorStateSpace.h"
#include "util/softmax.hpp"

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <random>

ROG::ROG(const ompl::base::SpaceInformationPtr& si)
    : ompl::base::StateValidityChecker(si) {
    this->dim = si->getStateSpace()->getDimension();
    float coverage = 0;
    while (coverage < 0.5) {
        struct bounds ob;
        float size = 1;
        for (size_t d = 0; d < dim; d++) {
            ob.low.push_back((float) rand() / RAND_MAX);
            ob.high.push_back(((float) rand() / (RAND_MAX)) / 2 + ob.low[d]);
            size *= (ob.high[d] - ob.low[d]);
        }
        obs.push_back(ob);
        coverage += size;
    }
    std::cout << "Coverage: " << coverage << std::endl;
    std::cout << "Obstacles: " << obs.size() << std::endl;
}

ROG::ROG(const ompl::base::SpaceInformationPtr& si, size_t seed, double scale,
         double coverage)
    : ompl::base::StateValidityChecker(si) {
}

bool ROG::isValid(const ompl::base::State* state) const {
    const ompl::base::RealVectorStateSpace::StateType* s =
        state->as<ompl::base::RealVectorStateSpace::StateType>();
    for (struct bounds ob : obs) {
        size_t dimcol = 0;
        for (size_t d = 0; d < dim; d++) {
            if (s->values[d] < ob.high[d] && s->values[d] > ob.low[d])
                dimcol++;
        }
        if (dim == dimcol)
            return false;
    }
    return true;
}
