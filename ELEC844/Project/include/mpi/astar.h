/**
 * @file astar.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date Created: 2025-12-15
 * @modified Last Modified: 2025-12-15
 *
 * @copyright Copyright (c) 2025
 */


#ifndef _MPI_ASTAR_H_
#define _MPI_ASTAR_H_

#include "mpi_plannerbase.h"

#if STATESPACE_DIMS == 2
#define A_WEIGHT 1.0f
#elif STATESPACE_DIMS == 4
#define A_WEIGHT 1.41f
#elif STATESPACE_DIMS == 6
#define A_WEIGHT 1.75f
#elif STATESPACE_DIMS == 8
// #define A_WEIGHT 2.020f // Low bounds
#define A_WEIGHT 2.021f
// #define A_WEIGHT 2.024f // High bounds
#elif STATESPACE_DIMS == 10
#define A_WEIGHT 2.28f
#endif

extern float mpi_astar_solve(struct mpi_planner *planner);

#endif
