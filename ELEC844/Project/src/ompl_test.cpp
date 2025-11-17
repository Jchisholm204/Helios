/**
 * @file ompl_test.cpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-15
 * @modified Last Modified: 2025-11-15
 *
 * @copyright Copyright (c) 2025
 */

#include "main.h"

#include <iostream>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>

// Our collision checker. For this demo, our robot's state space
// lies in [0,1]x[0,1], with a circular obstacle of radius 0.25
// centered at (0.5,0.5). Any states lying in this circular region are
// considered "in collision".
class ValidityChecker : public ompl::base::StateValidityChecker {
  public:
    ValidityChecker(const ompl::base::SpaceInformationPtr& si, int seed)
        : ompl::base::StateValidityChecker(si) {
            std::cout << "Seed: " << seed << std::endl;
        }

    // Returns whether the given state's position overlaps the
    // circular obstacle
    bool isValid(const ompl::base::State* state) const {
        return this->clearance(state) > 0.0;
    }

    // Returns the distance from the given state's position to the
    // boundary of the circular obstacle.
    double clearance(const ompl::base::State* state) const {
        // We know we're working with a RealVectorStateSpace in this
        // example, so we downcast state into the specific type.
        const ompl::base::RealVectorStateSpace::StateType* state2D =
            state->as<ompl::base::RealVectorStateSpace::StateType>();

        // Extract the robot's (x,y) position from its state
        double x = state2D->values[0];
        double y = state2D->values[1];

        // Distance formula between two points, offset by the circle's
        // radius
        return sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) - 0.25;
    }
};

int ompl_test(int argc, char* argv[]) {
    std::cout << "Hello from OMPL Test" << std::endl;
    // Construct the robot state space in which we're planning. We're
    // planning in [0,1]x[0,1], a subset of R^2.
    ompl::base::StateSpacePtr space(new ompl::base::RealVectorStateSpace(2));

    // Set the bounds of space to be in [0,1].
    space->as<ompl::base::RealVectorStateSpace>()->setBounds(0.0, 1.0);

    // Construct a space information instance for this state space
    ompl::base::SpaceInformationPtr si(new ompl::base::SpaceInformation(space));

    // Set the object used to check which states in the space are valid
    si->setStateValidityChecker(
        ompl::base::StateValidityCheckerPtr(new ValidityChecker(si, 123)));

    si->setup();

    // Set our robot's starting state to be the bottom-left corner of
    // the environment, or (0,0).
    ompl::base::ScopedState<> start(space);
    start->as<ompl::base::RealVectorStateSpace::StateType>()->values[0] = 0.0;
    start->as<ompl::base::RealVectorStateSpace::StateType>()->values[1] = 0.0;

    // Set our robot's goal state to be the top-right corner of the
    // environment, or (1,1).
    ompl::base::ScopedState<> goal(space);
    goal->as<ompl::base::RealVectorStateSpace::StateType>()->values[0] = 1.0;
    goal->as<ompl::base::RealVectorStateSpace::StateType>()->values[1] = 1.0;

    // Create a problem instance
    ompl::base::ProblemDefinitionPtr pdef(
        new ompl::base::ProblemDefinition(si));

    // Set the start and goal states
    pdef->setStartAndGoalStates(start, goal);

    // Construct our optimizing planner using the RRTstar algorithm.
    ompl::base::PlannerPtr optimizingPlanner(new ompl::geometric::RRTstar(si));

    // Set the problem instance for our planner to solve
    optimizingPlanner->setProblemDefinition(pdef);
    optimizingPlanner->setup();

    // attempt to solve the planning problem within one second of
    // planning time
    ompl::base::PlannerStatus solved = optimizingPlanner->solve(1.0);

    if(solved){
        pdef->getSolutionPath()->print(std::cout);
    }

    return 0;
}
