/**
 * @file main.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief ELEC 844 Final Project
 * @version 0.1
 * @date Created: 2025-11-15
 * @modified Last Modified: 2025-11-15
 *
 * @copyright Copyright (c) 2025
 */

#include "main.h"

#include "display/display.hpp"
#include "rog.hpp"

#include <iostream>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <stdio.h>

int main(int argc, char** argv) {
    printf("ELEC 844 Final Project\nargs:\n");
    for (int i = 0; i < argc; i++) {
        printf(" %d) %s\n", i, argv[i]);
    }
    Display d(100, 100);
    ompl::base::StateSpacePtr space(new ompl::base::RealVectorStateSpace(2));
    ompl::base::SpaceInformationPtr si(new ompl::base::SpaceInformation(space));
    ROG rog(si);
    std::vector<std::pair<float, float>> points = {
        {5, 5},
        {10, 10},
        {20, 20},
        {0, 0}
    };
    std::vector<std::pair<float, float>> invalid = {};
    for(int x = 0; x < 100; x++)
        for(int y = 0; y < 100; y++){
            ompl::base::State *s = si->allocState();
            s->as<ompl::base::RealVectorStateSpace::StateType>()->values[0] = (float)x/100.0;
            s->as<ompl::base::RealVectorStateSpace::StateType>()->values[1] = (float)y/100.0;
            if(!rog.isValid(s)){
                invalid.push_back({x, y});
            }
        }
    printf("Got %ld invalid states\n", invalid.size());
    while (!d.poll_quit()) {
        d.clear();
        d.label("ELEC 844 Project - FMT* - Jacob Chisholm");
        d.draw_grid(10, 10);
        d.draw_points(points, {255, 0, 255});
        d.draw_points({{44, 66}, {22, 33}}, {0, 0, 255});
        d.draw_points(invalid, {255, 0, 0});
        d.render();
    }

    return 0;
    // std::cout << "Running OMPL Test" << std::endl;
    // return ompl_test(argc, argv);
}
