/**
 * @file ompl.hpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-12
 * @modified Last Modified: 2025-12-12
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _OMPL_HPP_
#define _OMPL_HPP_

#include "util/solution_metrics.h"

#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>

struct ompl_planner {
    ompl::base::StateSpacePtr space;
    ompl::base::SpaceInformationPtr space_information;
    ompl::base::ScopedState<> start, target;
    ompl::base::ProblemDefinitionPtr problem_definition;
    ompl::base::PlannerPtr planner;
    ompl::base::PlannerStatus planner_status;
    ompl::base::PlannerTerminationCondition termination_condition;
};

extern struct ompl_planner *_ompl_init(void);

extern int ompl_solve(struct ompl_planner *planner);

extern struct solution_metrics *ompl_evaluate(struct ompl_planner *planner);

#endif
