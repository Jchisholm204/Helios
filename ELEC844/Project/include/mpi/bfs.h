/**
 * @file bfs.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-15
 * @modified Last Modified: 2025-12-15
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _MPI_BFS_H_
#define _MPI_BFS_H_

#include "mpi_plannerbase.h"

struct mpi_bfs {
    unsigned long *visiteds;
    unsigned long *parents;
    float *distances;
    struct mpi_planner *plannerbase;
};

extern struct mpi_bfs *mpi_bfs_init(void);

extern float mpi_bfs_solve(struct mpi_bfs *planner);

extern void mpi_bfs_free(struct mpi_bfs **pPlanner);

#endif
