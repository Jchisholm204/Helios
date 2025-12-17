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

#define OWNER(state) ((int) ((state) & ((1UL << (lgprocs)) - 1)))
#define HASH(state) ((int) ((state) >> (lgprocs)))

hashtable_t *visiteds = NULL;
min_heap_t *s1 = (void *) 0xDEADBEEF;
min_heap_t *s2 = (void *) 0xBEEFDEAD;

uint64_t start_idx = 0x00;
uint64_t target_idx = 0x00;

wstate_t start_state = {{0}, 0.0};
wstate_t target_state = {{0}, 0.0};

void visit_hndl(int from, void *dat, int size) {
    (void) from;
    if (size != sizeof(vstate_t) || !dat) {
        return;
    }
    vstate_t *new = dat;

    uint64_t index = state_index(new->state);
    // printf("Node %d adding (%d %d) c=%2.2f", proc_id, new->state[0],
    //        new->state[1], new->weight);
    // Check if the state is unvisited
    int r = hashtable_insert(visiteds, new, HASH(index));
    if (!r) {
        wstate_t t;
        t.weight = new->weight;
        state_cpy(&t.state, (const state_t *) &new->state);
        mheap_push(s2, &t);
        // printf("\n");
    } else {
        // printf(" -> FAIL (%d)\n", r);
    }
}

float mpi_bfs_solve(struct mpi_planner *planner) {
    // Setup local handles
    aml_register_handler(visit_hndl, AML_VISIT);
    gog_t *gog = &planner->gog;
    visiteds = planner->table;
    s1 = planner->heap;
    s2 = planner->heap2;

    start_idx = state_index(*planner->start);
    target_idx = state_index(*planner->target);

    start_state.weight = 0.0;
    target_state.weight = 0.0;
    state_cpy(&start_state.state, planner->start);
    state_cpy(&target_state.state, planner->target);

    if (OWNER(start_idx) == proc_id) {
        mheap_push(s1, &start_state);
        printf("Process %d owns the start node (%d, %d)\n", proc_id,
               start_state.state[0], start_state.state[1]);
    }

    if (OWNER(target_idx) == proc_id) {
        printf("Process %d owns the target node (%d, %d)\n", proc_id,
               target_state.state[0], target_state.state[1]);
    }

    if (proc_id == 0)
        printf("Starting BFS Search\n");

    unsigned long long global_work = s1->n_elements;
    unsigned long long local_work = s1->n_elements;
    aml_long_allsum(&global_work);

    if (proc_id == 0) {
        printf("Global Work = %lld\n", global_work);
        printf("Local Work = %lld\n", local_work);
    }

    size_t iteration = 0;

    while (global_work > 0 || local_work > 0) {
        iteration++;
        if (proc_id == 0) {
            printf("Entering Iteration %ld (global_work=%lld)\n", iteration,
                   global_work);
        }
        wstate_t node_v = {{0}, FLT_MAX};
        // Pull the next state to be explored
        while (s1->n_elements > 0) {
            mheap_pop(s1, &node_v);
            // Clone copy for push adjustments
            vstate_t next;
            state_cpy(&next.state, (const state_t *) &node_v.state);
            state_cpy(&next.parent, (const state_t *) &node_v.state);
            next.weight = node_v.weight;

            uint64_t node_idx = state_index(node_v.state);
            if(node_idx == target_idx){
                printf("Found Target\n");
                target_state.weight = node_v.weight;
            }

            // if (proc_id == 0) {
            //     printf("Exploring %d %d\n", next.state[0], next.state[1]);
            // }

            // state advance loop for straight xyz connections
            for (size_t d1 = 0; d1 < (STATESPACE_DIMS << 1); d1++) {
                size_t i = d1 >> 1;
                // Check the direct connections
                next.state[i] += d1 & 0x01 ? 1 : -1;
                if ((next.state[i] >= STATESPACE_MAX && (d1 & 0x01)) ||
                    (next.state[i] <= STATESPACE_MIN && !(d1 & 0x01))) {
                    next.state[i] += d1 & 0x01 ? -1 : 1;
                    continue;
                }
                next.weight += 1;
                // Check the validity of the point
                if (!gog_check(gog, (const state_t *) &next.state)) {
                    uint64_t next_idx = state_index(next.state);
                    // Check ownership and add to the correct queue
                    if (OWNER(next_idx) == proc_id) {
                        visit_hndl(proc_id, &next, sizeof(vstate_t));
                    } else {
                        aml_send(&next, AML_VISIT, sizeof(vstate_t),
                                 OWNER(next_idx));
                    }
                }
                // Restore the weight value
                next.weight -= 1;
                // Setup the weight for diagonal connections
                next.weight += M_SQRT2;

                // state advance loop for diagonal/jump connections
                for (size_t d2 = ((i + 1) << 1); d2 < (STATESPACE_DIMS << 1);
                     d2++) {
                    size_t j = d2 >> 1;
                    // Advance the +1 state to allow diagonal connections
                    next.state[j] += d2 & 0x01 ? 1 : -1;
                    if ((next.state[j] >= STATESPACE_MAX && (d2 & 0x01)) ||
                        (next.state[j] <= STATESPACE_MIN && !(d2 & 0x01))) {
                        next.state[j] += d2 & 0x01 ? -1 : 1;
                        continue;
                    }

                    // if (proc_id == 0) {
                    //     printf("Exploring %d %d\n", next.state[0],
                    //            next.state[1]);
                    // }

                    // Check the validity of the point
                    // Do not allow double length straight line connections
                    if (!gog_check(gog, (const state_t *) &next.state) &&
                        j != i) {
                        uint64_t next_idx = state_index(next.state);
                        if (OWNER(next_idx) == proc_id) {
                            visit_hndl(proc_id, &next, sizeof(vstate_t));
                        } else {
                            aml_send(&next, AML_VISIT, sizeof(vstate_t),
                                     OWNER(next_idx));
                        }
                    }
                    // Reset the state to the default state
                    next.state[j] += d2 & 0x01 ? -1 : 1;
                }
                // Reset the state to the default state
                next.state[i] += d1 & 0x01 ? -1 : 1;
                next.weight -= M_SQRT2;
            }
            local_work = s1->n_elements;
        }
        aml_barrier();
        min_heap_t *temp = s1;
        s1 = s2;
        s2 = temp;

        // Work Sync

        local_work = s1->n_elements;
        global_work = local_work;
        aml_long_allsum(&global_work);

        MPI_Allreduce(&target_state.weight, &target_state.weight, 1, MPI_FLOAT,
                      MPI_SUM, MPI_COMM_WORLD);
        // if (proc_id == 0)
        //     printf("Target Distance: %3.2f\n", target_cost);
        if (target_state.weight > 0) {
            printf("Target Distance: %3.2f\n", target_state.weight);
            break;
        }
    }

    if (proc_id == 0) {
        printf("Solved Graph in %ld iterations\n", iteration);
    }

    return target_state.weight;
}
