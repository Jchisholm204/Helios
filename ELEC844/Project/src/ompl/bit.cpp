/**
 * @file bit.cpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date Created: 2025-12-13
 * @modified Last Modified: 2025-12-13
 *
 * @copyright Copyright (c) 2025
 */

#include "ompl/bit.hpp"


struct ompl_planner *ompl_init_bit() {
    struct ompl_planner *planner = _ompl_init();
    if (!planner) {
        return NULL;
    }

    planner->planner = ompl::base::PlannerPtr(
        new ompl::geometric::BITstar(planner->space_information));
    planner->planner->as<ompl::geometric::BITstar>()->setSamplesPerBatch(50);

    planner->planner->setProblemDefinition(planner->problem_definition);
    planner->planner->setup();

    return planner;
}
