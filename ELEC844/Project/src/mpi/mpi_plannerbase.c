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

struct mpi_planner *mpi_planner_init(void) {
    // Allocate the planner object
    struct mpi_planner *planner =
        (struct mpi_planner *) malloc(sizeof(struct mpi_planner));
    if (!planner) {
        return NULL;
    }
    // Zero out planner memory
    memset(planner, 0, sizeof(struct mpi_planner));

    // Init the gog object on all processes
    gog_init_static(&planner->gog, GOG_SEED, 2, 3);
    memcpy(planner->start, gog_start(&(planner->gog)), sizeof(state_t));
    memcpy(planner->target, gog_target(&(planner->gog)), sizeof(state_t));

    // Copy the values from the gog object on rank 0 to all processes for
    // consistency
    MPI_Bcast(planner->gog.patterns, sizeof(state_t), MPI_UINT8_T, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(planner->gog.bmasks, sizeof(state_t), MPI_UINT8_T, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(planner->gog.variable, sizeof(state_t), MPI_UINT8_T, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(planner->start, sizeof(state_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(planner->target, sizeof(state_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);

    memset(&planner->metrics, 0, sizeof(struct solution_metrics));
    planner->metrics.length = -1;

    // Setup the planner data structures
    size_t default_arr_size = 2000 + 1500 * pow(STATESPACE_DIMS, 2);
    planner->table = hashtable_init(default_arr_size);
    planner->heap = mheap_init(default_arr_size);
    planner->heap2 = mheap_init(default_arr_size);

    // Ensure all processes allocate the planner before continuing
    MPI_Barrier(MPI_COMM_WORLD);

    return planner;
}

void mpi_planner_free(struct mpi_planner **pPlanner) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (pPlanner) {
        struct mpi_planner *planner = *pPlanner;
        if (planner) {
            hashtable_free(&(planner->table));
            mheap_free(&(planner->heap));
            mheap_free(&(planner->heap2));
            if (planner->metrics.path) {
                free(planner->metrics.path);
            }
        }
        free(planner);
    }
    *pPlanner = NULL;
}

void mpi_planner_evaluate(int argc, char **argv) {
    aml_init(&argc, &argv);
    struct mpi_planner *p = mpi_planner_init();
    // printf("Hello From Process %d/%d\n", proc_id, n_procs);
    printf("Proc %d: gog=0x%x\n", proc_id, p->gog.variable[0]);
    mpi_planner_free(&p);
    MPI_Barrier(MPI_COMM_WORLD);
    aml_finalize();
    return;
}
