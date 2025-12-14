/**
 * @file ompl.cpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-12
 * @modified Last Modified: 2025-12-12
 *
 * @copyright Copyright (c) 2025
 */

#include "ompl/ompl.hpp"

#include <float.h>
#include <stdio.h>

struct ompl_planner *_ompl_init(void) {
    struct ompl_planner *planner =
        (struct ompl_planner *) malloc(sizeof(struct ompl_planner));
    if (!planner) {
        return NULL;
    }
    memset((void *) planner, 0, sizeof(struct ompl_planner));

    // Set up the robot state space (match the definitions from statespace.h)
    planner->space = ompl::base::StateSpacePtr(
        new ompl::base::RealVectorStateSpace(STATESPACE_DIMS));

    // Set the bounds of the ompl state space to the definitions from
    // statespace.h
    planner->space->as<ompl::base::RealVectorStateSpace>()->setBounds(
        STATESPACE_MIN, STATESPACE_MAX);

    // Create the state space information object
    planner->space_information = ompl::base::SpaceInformationPtr(
        new ompl::base::SpaceInformation(planner->space));

    planner->space_information->setStateValidityCheckingResolution(0.00001);

    // double res = planner->space_information->getStateValidityCheckingResolution();
    // printf("Sampling Resolution=%f\n", res);

    // Create the GOG OMPL wrapper object
    // long seed = ompl::RNG::getSeed();
    ulong seed = GOG_SEED;
    planner->gog =
        std::make_shared<GOGValidityChecker>(planner->space_information, seed);
    // printf("Using Seed %ld\n", seed);

    // Link the GOG OMPL wrapper into the space information object
    planner->space_information->setStateValidityChecker(planner->gog);

    // Call the space information setup function before setting points
    planner->space_information->setup();

    // Create the start and target point objects
    planner->start =
        ompl::base::ScopedStatePtr(new ompl::base::ScopedState(planner->space));
    planner->target =
        ompl::base::ScopedStatePtr(new ompl::base::ScopedState(planner->space));

    // Set the start and target points to be the points returned by the GOG
    // object
    state_t *point_start = planner->gog->getStartPoint();
    state_t *point_target = planner->gog->getTargetPoint();

    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        (*planner->start)
            ->as<ompl::base::RealVectorStateSpace::StateType>()
            ->values[i] = (double) (*point_start)[i];
        (*planner->target)
            ->as<ompl::base::RealVectorStateSpace::StateType>()
            ->values[i] = (double) (*point_target)[i];
    }

    // Create the problem definitions
    planner->problem_definition = ompl::base::ProblemDefinitionPtr(
        new ompl::base::ProblemDefinition(planner->space_information));

    // Link start/target states to the problem definition
    planner->problem_definition->setStartAndGoalStates(*planner->start,
                                                       *planner->target);

    return planner;
}

int ompl_solve(struct ompl_planner *planner, double solve_time) {
    if (!planner) {
        return -1;
    }
    // Find the optimal path and path quality
    state_t *start_point = planner->gog->getStartPoint();
    state_t *target_point = planner->gog->getTargetPoint();
    double optimal_length2 = 0;

    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        optimal_length2 += ((*target_point)[i] - (*start_point)[i]) *
                           ((*target_point)[i] - (*start_point)[i]);
    }
    planner->metrics.first.optimal_length = sqrt(optimal_length2);
    planner->metrics.best.optimal_length = sqrt(optimal_length2);
    planner->metrics.final.optimal_length = sqrt(optimal_length2);

    // Setup the metrics data
    planner->metrics.first.time = -1;
    planner->metrics.first.quality = 100;
    planner->metrics.best.time = -1;
    planner->metrics.best.quality = 100;
    planner->metrics.final.time = -1;
    planner->metrics.final.quality = 100;

    // Log the start time
    auto start_time = std::chrono::steady_clock::now();

    // Set the path solution callback (called whenever the planner finds a new
    // path)
    planner->problem_definition->setIntermediateSolutionCallback(
        [&](const ompl::base::Planner *p,
            const std::vector<const ompl::base::State *> &sp,
            const ompl::base::Cost c) {
            (void) p;
            (void) sp;
            // Log the first path returned
            if (planner->metrics.first.time < 0) {
                planner->metrics.first.time =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start_time)
                        .count() /
                    1000.0;
                planner->metrics.first.length = c.value();
                planner->metrics.first.quality =
                    planner->metrics.first.length /
                    planner->metrics.first.optimal_length;
                planner->metrics.first.n_collision_checks =
                    planner->gog->getAccesses();
            }
            // Quality of the latest returned path
            double quality = c.value() / planner->metrics.best.optimal_length;
            double time = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - start_time)
                              .count() /
                          1000.0;
            // Save the first best path
            if (quality >= (((planner->metrics.first.quality - 1) / 2) + 1)) {
                planner->metrics.best.time = time;
                planner->metrics.best.length = c.value();
                planner->metrics.best.quality = quality;
                planner->metrics.best.n_collision_checks =
                    planner->gog->getAccesses();
            }
        });

    // Allow the planner to run for a maxumim of n seconds
    planner->planner->solve(solve_time);

    // Log the time of the final solution
    planner->metrics.final.time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time)
            .count();

    // Log the number of checks to the collision checker
    planner->metrics.final.n_collision_checks = planner->gog->getAccesses();
    auto *path = planner->problem_definition->getSolutionPath()
                     ->as<ompl::geometric::PathGeometric>();
    // Log the initial path length
    planner->metrics.final.length = path->length();
    planner->metrics.final.quality =
        planner->metrics.final.length / planner->metrics.final.optimal_length;

    return 0;
}

struct ompl_metrics *ompl_get_path(struct ompl_planner *planner) {
    if (!planner) {
        return NULL;
    }
    // Recover the final path
    auto *path = planner->problem_definition->getSolutionPath()
                     ->as<ompl::geometric::PathGeometric>();

    planner->metrics.final.path =
        (state_t *) malloc(sizeof(state_t) * path->getStateCount());
    if (!planner->metrics.final.path) {
        return &planner->metrics;
    }
    planner->metrics.final.n_path = path->getStateCount();

    std::vector<ompl::base::State *> states = path->getStates();

    for (size_t i = 0; i < path->getStateCount(); i++) {
        for (size_t d = 0; d < STATESPACE_DIMS; d++) {
            planner->metrics.final.path[i][d] =
                states[i]
                    ->as<ompl::base::RealVectorStateSpace::StateType>()
                    ->values[d];
        }
    }

    return &planner->metrics;
}

void ompl_free(struct ompl_planner **pPlanner) {
    if (!pPlanner) {
        return;
    }
    struct ompl_planner *planner = *pPlanner;
    if (!planner) {
        return;
    }

    planner->planner->clear();
    planner->planner.reset();
    planner->problem_definition.reset();
    planner->start.reset();
    planner->target.reset();
    planner->space_information.reset();
    planner->space.reset();

    if (planner->metrics.final.path) {
        free(planner->metrics.final.path);
    }

    free(planner);
    *pPlanner = NULL;
}
