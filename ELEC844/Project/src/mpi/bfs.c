/**
 * @file bfs.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-15
 * @modified Last Modified: 2025-12-15
 *
 * @copyright Copyright (c) 2025
 */

#include "mpi/bfs.h"

#include "mpi/aml.h"

#define AML_VISIT 1

#define OWNER(state) ((int) ((state) >> (64 - n_procs)))
#define HASH(state) ((int) ((state) & ((1ULL << (64 - n_procs) - 1))))

hashtable_t *visiteds = NULL;
min_heap_t *s1 = (void *) 0xDEADBEEF;
min_heap_t *s2 = (void *) 0xBEEFDEAD;

uint64_t start_idx = 0x00;
uint64_t target_idx = 0x00;
float target_cost = 0;

void visit_hndl(int from, void *dat, int size) {
    (void) from;
    if (size != sizeof(vstate_t) || !dat) {
        return;
    }
    vstate_t *new = dat;

    uint64_t index = state_index(new->state);
    if (index == target_idx) {
        target_idx = 0;
        target_cost = new->weight;
    }
    // Check if the state is unvisited
    if (!hashtable_insert(visiteds, new, index)) {
        wstate_t t;
        t.weight = new->weight;
        state_cpy(&t.state, (const state_t *) &new->state);
        mheap_push(s2, &t);
    }
}

float mpi_bfs_solve(struct mpi_planner *planner) {
    // Setup local handles
    aml_register_handler(visit_hndl, AML_VISIT);
    gog_t *gog = &planner->gog;
    visiteds = planner->table;
    s1 = planner->heap;
    s2 = planner->heap2;

    start_idx = state_index(planner->start);
    target_idx = state_index(planner->target);
    target_cost = 0;

    wstate_t start_state = {{*planner->start}, 0};

    if (OWNER(start_idx) == proc_id) {
        mheap_push(s1, &start_state);
    }

    if (proc_id == 0)
        printf("Starting BFS Search\n");

    unsigned long long global_work = s1->n_elements;
    unsigned long long local_work = s1->n_elements;
    aml_long_allsum(&global_work);

    while (global_work > 0 || local_work > 0) {
        wstate_t node_v = {{0}, FLT_MAX};
        // Pull the next state to be explored
        while (!mheap_pop(s1, &node_v)) {
            // Clone copy for push adjustments
            vstate_t next;
            state_cpy(&next.state, (const state_t *) &node_v.state);
            state_cpy(&next.parent, (const state_t *) &node_v.state);
            next.weight = node_v.weight;

            // state advance loop for straight xyz connections
            for (size_t d1 = 0; d1 < (STATESPACE_DIMS << 1); d1++) {
                size_t i = d1 >> 1;
                // Check the direct connections
                next.state[i] += d1 & 0x01 ? 1 : -1;
                next.weight += 1;
                // Check the validity of the point
                if (!gog_check(gog, (const state_t *) &next.state)) {
                    uint64_t next_idx = state_index(next.state);
                    // Check ownership and add to the correct queue
                    if (OWNER(next_idx) == proc_id) {
                        visit_hndl(proc_id, &next, sizeof(wstate_t));
                    } else {
                        aml_send(&next, AML_VISIT, sizeof(wstate_t),
                                 OWNER(next_idx));
                    }
                }
                // Restore the weight value
                next.weight -= 1;
                // Setup the weight for diagonal connections
                next.weight += M_SQRT2;

                // state advance loop for diagonal/jump connections
                for (size_t d2 = 0; d2 < (STATESPACE_DIMS << 1); d2++) {
                    size_t j = d2 >> 1;
                    // Advance the +1 state to allow diagonal connections
                    next.state[j] += d1 & 0x01 ? 1 : -1;

                    // Check the validity of the point
                    // Do not allow double length straight line connections
                    if (!gog_check(gog, (const state_t *) &next.state) &&
                        j != i) {
                        uint64_t next_idx = state_index(next.state);
                        if (OWNER(next_idx) == proc_id) {
                            visit_hndl(proc_id, &next, sizeof(wstate_t));
                        } else {
                            aml_send(&next, AML_VISIT, sizeof(wstate_t),
                                     OWNER(next_idx));
                        }
                    }
                    // Reset the state to the default state
                    next.state[j] += d1 & 0x01 ? -1 : 1;
                }
                // Reset the state to the default state
                next.state[i] += d1 & 0x01 ? -1 : 1;
                next.weight -= M_SQRT2;
            }
        }
        aml_barrier();
        min_heap_t *temp = s1;
        s1 = s2;
        s2 = temp;
        MPI_Allreduce(&target_cost, &target_cost, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
        if (proc_id == 0)
            printf("Target Distance: %3.2f\n", target_cost);
        if (target_cost > 0) {
            break;
        }
    }

    return target_cost;
}
