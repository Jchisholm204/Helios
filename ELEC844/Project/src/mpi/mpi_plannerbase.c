/**
 * @file mpi_plannerbase.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-15
 * @modified Last Modified: 2025-12-15
 *
 * @copyright Copyright (c) 2025
 */

#include "mpi/mpi_plannerbase.h"

#include "mpi/aml.h"

#include <memory.h>

struct mpi_planner *_mpi_planner_init(void) {
    MPI_Barrier(MPI_COMM_WORLD);
    struct mpi_planner *planner =
        (struct mpi_planner *) malloc(sizeof(struct mpi_planner));
    if (!planner) {
        return NULL;
    }
    planner->gog = gog_init(GOG_SEED, 2, 3);
    memcpy(planner->start, gog_start(planner->gog), sizeof(state_t));
    memcpy(planner->target, gog_target(planner->gog), sizeof(state_t));

    memset(&planner->metrics, 0, sizeof(struct solution_metrics));
    planner->metrics.length = -1;
    MPI_Barrier(MPI_COMM_WORLD);
    return planner;
}

void _mpi_planner_free(struct mpi_planner **pPlanner) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (pPlanner) {
        struct mpi_planner *planner = *pPlanner;
        if (planner) {
            gog_free(&planner->gog);
            free(planner);
        }
        *pPlanner = NULL;
    }
}

void mpi_planner_evaluate(int argc, char **argv) {
    aml_init(&argc, &argv);
    struct mpi_planner *p = _mpi_planner_init();
    printf("Hello From Process %d/%d\n", proc_id, n_procs);
    _mpi_planner_free(&p);
    MPI_Barrier(MPI_COMM_WORLD);
    p = _mpi_planner_init();
    printf("Hello From Process %d/%d\n", proc_id, n_procs);
    _mpi_planner_free(&p);
    aml_finalize();
    return;
}
