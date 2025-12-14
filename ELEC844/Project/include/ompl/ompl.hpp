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

#include "statespace/gog_ompl_wrapper.hpp"
#include "statespace/statespace.h"
#include "util/solution_metrics.h"

#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/fmt/FMT.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>

struct ompl_metrics {
    struct solution_metrics first;
    struct solution_metrics best;
    struct solution_metrics final;
};

struct ompl_planner {
    ompl::base::StateSpacePtr space;
    ompl::base::SpaceInformationPtr space_information;
    ompl::base::ScopedStatePtr start, target;
    ompl::base::ProblemDefinitionPtr problem_definition;
    ompl::base::PlannerPtr planner;
    ompl::base::PlannerStatus planner_status;
    ompl::base::PlannerTerminationCondition termination_condition;
    std::shared_ptr<GOGValidityChecker> gog;
    struct ompl_metrics metrics;
};

/**
 * @brief Internal function - Should be called by ompl_init_planner
 */
extern struct ompl_planner *_ompl_init(void);

/**
 * @brief Run the planner and gather the results
 *
 * @param planner
 * @return
 */
extern int ompl_solve(struct ompl_planner *planner, double solve_time);

/**
 * @brief Get the final path returned by the planner
 *
 * @param planner
 */
extern struct ompl_metrics *ompl_get_path(struct ompl_planner *planner);

extern void ompl_free(struct ompl_planner **pPlanner);

extern void ompl_evaluate(double solve_time);

#endif
