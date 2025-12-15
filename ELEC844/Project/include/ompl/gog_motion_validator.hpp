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

#include <float.h>
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

        state_t voxel;    // current voxel indices
        state_t endVoxel; // target voxel indices
        state_t step;     // step direction in each dimension
        double
            tMax[STATESPACE_DIMS]; // distance along ray to next voxel boundary
        double
            tDelta[STATESPACE_DIMS]; // distance to cross one voxel in each axis

        // Initialize voxel coordinates, steps, and tMax/tDelta
        for (size_t i = 0; i < STATESPACE_DIMS; i++) {
            voxel[i] = (int) std::floor(a->values[i]);
            endVoxel[i] = (int) std::floor(b->values[i]);

            if (endVoxel[i] > voxel[i])
                step[i] = 1;
            else if (endVoxel[i] < voxel[i])
                step[i] = -1;
            else
                step[i] = 0;

            if (step[i] != 0) {
                double nextBoundary = voxel[i] + (step[i] > 0 ? 1.0 : 0.0);
                tMax[i] = (nextBoundary - a->values[i]) /
                          (b->values[i] - a->values[i]);
                tDelta[i] = 1.0 / std::abs(b->values[i] - a->values[i]);
            } else {
                tMax[i] = std::numeric_limits<double>::infinity();
                tDelta[i] = std::numeric_limits<double>::infinity();
            }
        }

        // Traverse voxels until reaching the end voxel
        while (true) {
            // Check collision at current voxel
            if (gog_check(this->gog, &voxel)) {
                return false;
            }

            // Check if we've reached the end voxel
            bool done = true;
            for (size_t i = 0; i < STATESPACE_DIMS; i++) {
                if (voxel[i] != endVoxel[i]) {
                    done = false;
                    break;
                }
            }
            if (done)
                break;

            // Find axis with smallest tMax -> next voxel to cross
            size_t minAxis = 0;
            for (size_t i = 1; i < STATESPACE_DIMS; i++) {
                if (tMax[i] < tMax[minAxis])
                    minAxis = i;
            }

            // Step in that axis
            voxel[minAxis] += step[minAxis];
            tMax[minAxis] += tDelta[minAxis];
        }

        // Final voxel check
        if (gog_check(this->gog, &endVoxel))
            return false;

        return true;
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
