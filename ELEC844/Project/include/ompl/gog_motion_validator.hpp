/**
 * @file gog_motion_validator.hpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief MotionValidator for the GOG
 * @version 0.1
 * @date Created: 2025-12-13
 * @modified Last Modified: 2025-12-13
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _GOG_MOTION_VALIDATOR_HPP_
#define _GOG_MOTION_VALIDATOR_HPP_

#include "statespace/gog_ompl_wrapper.hpp"

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

class GOGMotionValidator : public ompl::base::MotionValidator {
  public:
    GOGMotionValidator(const ompl::base::SpaceInformationPtr &si,
                       std::shared_ptr<GOGValidityChecker> checker)
        : ompl::base::MotionValidator(si) {
        this->checker = checker;
        this->gog = checker->gog;
    }

    bool checkMotion(const ompl::base::State *s1,
                     const ompl::base::State *s2) const override {
        const auto *a = s1->as<ompl::base::RealVectorStateSpace::StateType>();
        const auto *b = s2->as<ompl::base::RealVectorStateSpace::StateType>();
        double distance2 = 0;
        double increments[STATESPACE_DIMS] = {0};
        printf("Planning:\n from: ");
        for (size_t i = 0; i < STATESPACE_DIMS; i++) {
            increments[i] = (b->values[i] - a->values[i]);
            printf("%2.2f ", a->values[i]);
            distance2 += increments[i] * increments[i];
        }
        double distance = sqrt(distance2);
        if (distance == 0) {
            return true;
        }
        state_t current = {0};
        printf("\n to: ");
        for (size_t i = 0; i < STATESPACE_DIMS; i++) {
            printf("%2.2f ", b->values[i]);
            increments[i] = increments[i] / distance;
            current[i] = a->values[i];
        }
        printf("Increments: ");
        for(int i = 0; i < STATESPACE_DIMS; i++){
            printf("%2.2f ", increments[i]);
        }
        printf("\nDistance=%3.2f\n", distance);
        printf("Running %d collision checks\n", (int)distance * STATESPACE_DIMS);

        bool invalid = 0;
        for (size_t i = 0; i < (size_t) abs(ceil(distance)); i++) {
            for (size_t d = 0; d < STATESPACE_DIMS; d++) {
                current[d] += increments[d];
                invalid |= gog_check(this->gog, &current);
            }
        }
        printf("Final: ");
        for(int i = 0; i < STATESPACE_DIMS; i++){
            printf("%d ", current[i]);
        }
        printf("\n");

        return !invalid;
    }

    bool checkMotion(const ompl::base::State *s1, const ompl::base::State *s2,
                     std::pair<ompl::base::State *, double> &) const override {
        return checkMotion(s1, s2);
    }

  private:
    std::shared_ptr<GOGValidityChecker> checker;
    gog_t *gog = NULL;
};

#endif
