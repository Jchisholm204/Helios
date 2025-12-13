/**
 * @file fmt.cpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-12
 * @modified Last Modified: 2025-12-12
 *
 * @copyright Copyright (c) 2025
 */

#include "ompl/fmt.hpp"

struct ompl_planner *ompl_init_fmt(const unsigned int n_samples) {
    struct ompl_planner *planner = _ompl_init();
    if (!planner) {
        return NULL;
    }

    planner->planner = ompl::base::PlannerPtr(
        new ompl::geometric::FMT(planner->space_information));
    
    planner->planner->as<ompl::geometric::FMT>()->setNumSamples(n_samples);

    planner->planner->setProblemDefinition(planner->problem_definition);
    planner->planner->setup();

    return planner;
}
