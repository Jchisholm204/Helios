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


struct ompl_planner *_ompl_init(void){
    return NULL;
}

int ompl_solve(struct ompl_planner *planner){
    (void)planner;
    return 0;
}

struct solution_metrics *ompl_evaluate(struct ompl_planner *planner){
    (void)planner;
    return NULL;
}

