/**
 * @file mpi_plannerbase.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-15
 * @modified Last Modified: 2025-12-15
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _MPI_PLANNERBASE_H_
#define _MPI_PLANNERBASE_H_
#ifdef __cplusplus
extern "C" {
#endif

#include "statespace/geometric_obstacle_generator.h"
#include "statespace/statespace.h"
#include "util/solution_metrics.h"

struct mpi_planner {
    gog_t *gog;
    state_t start, target;
    struct solution_metrics metrics;
};

/**
 * @brief Internal Function - Performs basic setup required for all MPI planners
 *
 */
extern struct mpi_planner *_mpi_planner_init(void);

/**
 * @brief Internal Function - Performs basic cleanup required for all MPI planners
 *
 * @param pPlanner 
 */
extern void _mpi_planner_free(struct mpi_planner **pPlanner);

/**
 * @brief Generic Function used to evaluate the MPI based planners, Call this from main
 *
 * @param argc 
 * @param argv 
 */
extern void mpi_planner_evaluate(int argc, char **argv);


#ifdef __cplusplus
}
#endif

#endif
