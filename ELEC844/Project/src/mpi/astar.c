/**
 * @file astar.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-15
 * @modified Last Modified: 2025-12-17
 *
 * @copyright Copyright (c) 2025
 */

#include "mpi/astar.h"

#include "mpi/aml.h"

#define AML_VISIT 1
// #define GREEDY

// Local variables for fixing aml pe/pes function call overhead
static int lgsize = 0xBEEF;
static int nproc = 0xDEAD;
static int pid = 0xBEEF;

#define OWNER(state) ((int) ((state) & ((1UL << (lgsize)) - 1)))
#define HASH(state) ((int) ((state) >> (lgsize)))

static hashtable_t *visiteds = NULL;
static min_heap_t *s1 = (void *) 0xDEADBEEF;

static uint64_t start_idx = 0x00;
static uint64_t target_idx = 0x00;

static wstate_t start_state = {{0}, 0.0, 0.0};
static wstate_t target_state = {{0}, FLT_MAX, 0.0};

static void visit_hndl(int from, void *dat, int size) {
    (void) from;
    (void) size;
    // if (size != sizeof(vstate_t) || !dat) {
    //     return;
    // }
    vstate_t *new = dat;

    // if (new->weight >= target_state.weight) {
    //     return;
    // }

    uint64_t index = state_index(new->state);

    // Check if the state is unvisited
    int r = hashtable_insert(visiteds, new, HASH(index));
    if (!r || r == 1) {
        wstate_t t;
        t.weight = new->weight;
        t.cost = 0;
        for (size_t i = 0; i < STATESPACE_DIMS; i++) {
            t.cost += (new->state[i] - target_state.state[i]) *
                      (new->state[i] - target_state.state[i]);
        }
        t.cost = sqrt(t.cost) * M_SQRT2 + t.weight;
        state_cpy(&t.state, (const state_t *) &new->state);
        mheap_push(s1, &t);
    }
}

float mpi_astar_solve(struct mpi_planner *planner) {
    // Setup local handles
    aml_register_handler(visit_hndl, AML_VISIT);
    gog_t *gog = &planner->gog;
    visiteds = planner->table;
    s1 = planner->heap;

    lgsize = lgprocs;
    nproc = n_procs;
    pid = proc_id;

    start_idx = state_index(*planner->start);
    target_idx = state_index(*planner->target);

    start_state.weight = 0.0;
    target_state.weight = FLT_MAX;
    state_cpy(&start_state.state, (const state_t *__restrict) planner->start);
    state_cpy(&target_state.state, (const state_t *__restrict) planner->target);

    if (OWNER(start_idx) == pid) {
        mheap_push(s1, &start_state);
        printf("Process %d owns the start node (%d\n", pid,
               start_state.state[0]);
        for (size_t i = 1; i < STATESPACE_DIMS; i++)
            printf(", %d", start_state.state[i]);
        printf(")\n");
    }

    if (OWNER(target_idx) == pid) {
        printf("Process %d owns the target node (%d", pid,
               target_state.state[0]);
        for (size_t i = 1; i < STATESPACE_DIMS; i++)
            printf(", %d", target_state.state[i]);
        printf(")\n");
    }

    if (pid == 0)
        printf("Starting BFS Search\n");

    unsigned long long global_work = s1->n_elements;
    unsigned long long local_work = s1->n_elements;
    aml_long_allsum(&global_work);

    if (pid == 0) {
        printf("Global Work = %lld\n", global_work);
        printf("Local Work = %lld\n", local_work);
    }

    size_t iteration = 0;
    size_t sum_global = 0;
    size_t sum_local = 0;

    while (global_work > 0 || local_work > 0) {
        iteration++;
        sum_global += global_work;
        sum_local += local_work;
        if (proc_id == 0) {
            printf("Entering Iteration %ld (global_work=%lld)\n", iteration,
                   global_work);
        }
        wstate_t node_v = {{0}, FLT_MAX, FLT_MAX};
        float local_min_f = (s1->n_elements > 0) ? s1->data[0].cost : FLT_MAX;
        float global_min_f;
        MPI_Allreduce(&local_min_f, &global_min_f, 1, MPI_FLOAT, MPI_MIN,
                      MPI_COMM_WORLD);
        if (target_state.weight <= global_min_f + 0.001f &&
            target_state.weight != FLT_MAX) {
            break;
        }
        // Pull the next state to be explored
        for (size_t batch = 0; s1->n_elements > 0 && batch < 20000; batch++) {
            mheap_pop(s1, &node_v);
#ifdef GREEDY
            if (node_v.cost > local_min_f) {
                continue;
            }
#endif
            // Clone copy for push adjustments
            vstate_t next;
            state_cpy(&next.state, (const state_t *) &node_v.state);
            state_cpy(&next.parent, (const state_t *) &node_v.state);
            next.weight = node_v.weight;

            uint64_t node_idx = state_index(node_v.state);
            if (node_idx == target_idx) {
                printf("Found Target\n");
                target_state.weight = node_v.weight;
            }

            float base_weight = next.weight;

            // state advance loop for straight xyz connections
            for (size_t i = 0; i < STATESPACE_DIMS; i++) {
                for (int d1 = -1; d1 <= 1; d1 += 2) {
                    // Check the direct connections
                    next.state[i] += d1;
                    if ((next.state[i] >= STATESPACE_MAX && (d1 > 0)) ||
                        (next.state[i] <= STATESPACE_MIN && (d1 < 0))) {
                        next.state[i] -= d1;
                        continue;
                    }
                    next.weight = base_weight + 1;
                    // Check the validity of the point
                    if (!gog_check(gog, (const state_t *) &next.state)) {
                        uint64_t next_idx = state_index(next.state);
                        // Check ownership and add to the correct queue
                        if (OWNER(next_idx) == pid) {
                            visit_hndl(pid, &next, sizeof(vstate_t));
                        } else {
                            aml_send(&next, AML_VISIT, sizeof(vstate_t),
                                     OWNER(next_idx));
                        }
                    }
                    // Setup the weight for diagonal connections
                    next.weight = base_weight + M_SQRT2;

                    // state advance loop for diagonal/jump connections
                    for (size_t j = (i + 1); j < STATESPACE_DIMS; j++) {
                        for (int d2 = -1; d2 <= 1; d2 += 2) {
                            // Advance the +1 state to allow diagonal
                            // connections
                            next.state[j] += d2;
                            if ((next.state[j] >= STATESPACE_MAX && (d2 > 0)) ||
                                (next.state[j] <= STATESPACE_MIN && (d2 < 0))) {
                                next.state[j] -= d2;
                                continue;
                            }

                            // Check the validity of the point
                            // Do not allow double length straight line
                            // connections
                            if (!gog_check(gog,
                                           (const state_t *) &next.state) &&
                                j != i) {
                                uint64_t next_idx = state_index(next.state);
                                if (OWNER(next_idx) == pid) {
                                    visit_hndl(pid, &next, sizeof(vstate_t));
                                } else {
                                    aml_send(&next, AML_VISIT, sizeof(vstate_t),
                                             OWNER(next_idx));
                                }
                            }
                            // Reset the state to the default state
                            next.state[j] -= d2;
                        } // END D2
                    } // END j
                    // Reset the state to the default state
                    next.state[i] -= d1;
                } // END D1 Shift
            } // END first step traversal
        }

        // Work Sync
        aml_barrier();

        local_work = s1->n_elements;
        global_work = local_work;
        aml_long_allsum(&global_work);

        MPI_Allreduce(&target_state.weight, &target_state.weight, 1, MPI_FLOAT,
                      MPI_MIN, MPI_COMM_WORLD);
        // if (proc_id == 0)
        //     printf("Target Distance: %3.2f\n", target_cost);
        if (target_state.weight < FLT_MAX && pid == 0) {
            printf("Target Distance: %3.2f\n", target_state.weight);
        }
    }

    if (pid == 0) {
        printf("Solved Graph in %ld iterations\n", iteration);
    }

    planner->sum_global = sum_global;
    planner->sum_local = sum_local;

    // printf("Proc %d completed %2.5f %% of work\n", pid,
    //        (double) sum_local * 100.0 / (double) sum_global);

    return target_state.weight;
}
