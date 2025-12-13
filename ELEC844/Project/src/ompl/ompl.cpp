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

struct ompl_planner *_ompl_init(void) {
    struct ompl_planner *planner =
        (struct ompl_planner *) malloc(sizeof(struct ompl_planner));
    if (!planner) {
        return NULL;
    }

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

    // Create the GOG OMPL wrapper object
    planner->gog = GOGValidityChecker(planner->space_information);

    // Link the GOG OMPL wrapper into the space information object
    planner->space_information->setStateValidityChecker(
        ompl::base::StateValidityCheckerPtr(&planner->gog));

    // Call the space information setup function before setting points
    planner->space_information->setup();

    // Create the start and target point objects
    planner->start = ompl::base::ScopedState(planner->space);
    planner->target = ompl::base::ScopedState(planner->space);

    // Set the start and target points to be the points returned by the GOG
    // object
    state_t *point_start = planner->gog.getStartPoint();
    state_t *point_target = planner->gog.getTargetPoint();

    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        planner->start->as<ompl::base::RealVectorStateSpace::StateType>()
            ->values[i] = (float) (*point_start)[i];
        planner->target->as<ompl::base::RealVectorStateSpace::StateType>()
            ->values[i] = (float) (*point_target)[i];
    }

    // Create the problem definitions
    planner->problem_definition = ompl::base::ProblemDefinitionPtr(
        new ompl::base::ProblemDefinition(planner->space_information));

    // Link start/target states to the problem definition
    planner->problem_definition->setStartAndGoalStates(planner->start,
                                                       planner->target);

    return planner;
}

int ompl_solve(struct ompl_planner *planner) {
    if (!planner) {
        return -1;
    }

    ompl::base::PlannerTerminationCondition ptc =
        ompl::base::plannerOrTerminationCondition(
            ompl::base::timedPlannerTerminationCondition(5.0),
            ompl::base::PlannerTerminationCondition(
                [&]() { return planner->problem_definition->hasSolution(); }
                )
            );

    return 0;
}

struct solution_metrics *ompl_evaluate(struct ompl_planner *planner) {
    (void) planner;
    return NULL;
}
