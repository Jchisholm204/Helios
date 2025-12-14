/**
 * @file gog_ompl_wrapper.hpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief OMPL StateValidityChecker Wrapper for the GOG
 * @version 0.1
 * @date Created: 2025-12-12
 * @modified Last Modified: 2025-12-12
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _GOG_OMPL_WRAPPER_HPP_
#define _GOG_OMPL_WRAPPER_HPP_

#include "statespace/geometric_obstacle_generator.h"

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

class GOGValidityChecker : public ompl::base::StateValidityChecker {
  public:
    GOGValidityChecker(const ompl::base::SpaceInformationPtr &si,
                       size_t seed = GOG_SEED, uint bmask = 1, uint pmask = 3)
        : ompl::base::StateValidityChecker(si) {
        this->gog = gog_init(seed, bmask, pmask);
    }

    bool isValid(const ompl::base::State *state) const {
        const auto *s =
            state->as<ompl::base::RealVectorStateSpace::StateType>();
        state_t real_state;
        for (size_t i = 0; i < STATESPACE_DIMS; i++) {
            real_state[i] = (uint8_t) (s->values[i]);
        }
        return !gog_check(this->gog, &real_state);
    }

    state_t *getStartPoint(void){
        return gog_start(this->gog);
    }

    state_t *getTargetPoint(void){
        return gog_target(this->gog);
    }

    size_t getAccesses(void) const {
        return gog_get_accesses(this->gog);
    }

    gog_t *gog = NULL;
};

#endif
