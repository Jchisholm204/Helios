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
#include "mpi/astar.h"
#include "mpi/bfs.h"

#include <memory.h>
#include <stdio.h>
#include <time.h>

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
    planner->start = gog_start(&planner->gog);
    planner->target = gog_target(&planner->gog);

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
    size_t default_arr_size = 0x1ULL
                              << (int) ((STATESPACE_DIMS/2 + 24 - lgprocs));
    planner->table = hashtable_init(default_arr_size);
    planner->heap = mheap_init(default_arr_size);
    // planner->heap2 = mheap_init(default_arr_size);

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
            if (planner->start) {
                free(planner->start);
            }
            if (planner->target) {
                free(planner->target);
            }
            if (planner->metrics.path) {
                free(planner->metrics.path);
            }
        }
        free(planner);
    }
    *pPlanner = NULL;
}

inline double diffms(struct timespec start, struct timespec end) {
    double timems = end.tv_sec - start.tv_sec;
    timems += (end.tv_nsec - start.tv_nsec) / 1e9;
    timems *= 1000;
    return timems;
}

#define N_TESTS 20
#define STRLN 200

void mpi_planner_evaluate(int argc, char **argv) {
    aml_init(&argc, &argv);
    FILE *fp = NULL;
    if (proc_id == 0) {
        char fname[STRLN];
        snprintf(fname, STRLN, "./tests/astar_%dD_p%d.csv", STATESPACE_DIMS,
                 n_procs);
        fp = fopen(fname, "w");
        if (!fp) {
            fprintf(stderr, "Failed to open output file..\n");
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_FILE);
        }
        fprintf(fp, "init time,solve time,optimal "
                    "length,length,quality,collision checks,");
        for (int i = 0; i < n_procs; i++) {
            fprintf(fp, "process %d work,", i);
        }
        fprintf(fp, "\n");
    }

    for (size_t test_i = 0; test_i < N_TESTS; test_i++) {
        struct timespec t_start, t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        struct mpi_planner *p = mpi_planner_init();
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        // Calculate the optimal path length
        double path_optimal = 0;
        for (size_t i = 0; i < STATESPACE_DIMS; i++) {
            path_optimal += ((*p->start)[i] - (*p->target)[i]) *
                            ((*p->start)[i] - (*p->target)[i]);
        }
        path_optimal = sqrt(path_optimal);

        double t_init = diffms(t_start, t_end);

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        float path_cost = mpi_astar_solve(p);
        // Wait for all processes to exit before recording time
        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        double t_solve = diffms(t_start, t_end);

        // Calculate final metrics
        double path_quality = path_cost / path_optimal;

        // Sum the collision checks across all MPI processes
        long long n_collision_checks = 0;
        MPI_Allreduce(&p->gog.access_counter, &n_collision_checks, 1,
                      MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        // Let process 0 record metrics to the log file
        if (proc_id == 0) {
            printf("%ld) D=%d l=%3.2f c=%1.4f ti=%4.4f ms ts=%4.4f ms\n",
                   test_i, STATESPACE_DIMS, path_cost, path_quality, t_init,
                   t_solve);
            fprintf(fp, "%f,%f,%f,%f,%f,%lld,", t_init, t_solve, path_optimal,
                    path_cost, path_quality, n_collision_checks);
            fprintf(fp, "%f,",
                    (float) p->sum_local * 100.0f / (float) p->sum_global);
            for (int i = 1; i < n_procs; i++) {
                size_t sum_local = 0;
                (void) MPI_Recv(&sum_local, 1, MPI_LONG, i, 0, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE);
                fprintf(fp, "%f,",
                        (float) sum_local * 100.0f / (float) p->sum_global);
            }
            fprintf(fp, "\n");
        } else {
            MPI_Send(&p->sum_local, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
        }

        mpi_planner_free(&p);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (proc_id == 0) {
        fclose(fp);
    }

    aml_finalize();
    return;
}
