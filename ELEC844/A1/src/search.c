/**
 * @file search.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-10-08
 *
 * @copyright Copyright (c) 2025
 */

#include "search.h"

#include <math.h>

// Calculates the Euclidian Distance
float dist(struct voxel* s, struct voxel* d) {
    return sqrt(pow(s->y - d->y, 2) + pow(s->x - d->x, 2));
}

int search_stepA(struct search* pSearch) {
    if (!pSearch)
        return -1;
    // Benchmark Queue Data
    pSearch->bmd.queue_final = queue_length(pSearch->queue);
    pSearch->bmd.queue_max = pSearch->bmd.queue_max < pSearch->bmd.queue_final
                                 ? pSearch->bmd.queue_final
                                 : pSearch->bmd.queue_max;
    pSearch->bmd.queue_ttl += pSearch->bmd.queue_final;

    // Grab the lowest cost node
    struct voxel* current = queue_pop(&pSearch->queue);
    if (!current)
        return -3;
    pSearch->bmd.n_explored++;

    if(current == pSearch->target){
        pSearch->finished = 1;
        return 0;
    }

    if (current != pSearch->start && current != pSearch->target)
        current->state = eStateExplored;

    // Process all nearby voxels
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            struct voxel* nv =
                grid_index(pSearch->pGrid, current->x + x, current->y + y);

            // Null check for voxels off the edge of the map
            if (!nv)
                continue;
            // Increment number of nodes evaluated
            pSearch->bmd.n_evaluated++;

            // Found the target
            // if (nv == pSearch->target) {
            //     nv->parent = current;
            //     pSearch->finished = 1;
            //     return 0;
            // }

            // Do not explore blocked nodes
            if (nv->state == eStateBlocked)
                continue;

            float goal_dist = pSearch->heuristic(nv, pSearch->target);
            float reach_cost = dist(nv, current) + current->cost;
            if (reach_cost < nv->cost) {
                nv->cost = reach_cost;
                nv->parent = current;
                queue_push(&pSearch->queue, nv, goal_dist + reach_cost);
                if (nv != pSearch->start && nv != pSearch->target)
                    nv->state = eStateFrontier;
            }
        }
    }

    // Benchmark Path Length
    pSearch->bmd.path_length = pSearch->target->cost;

    return 1;
}

struct path* search_backtrace(struct search* pSearch) {
    if (!pSearch)
        return NULL;

    // Determine end voxel
    struct voxel* v_end = NULL;
    if (pSearch->finished) {
        v_end = pSearch->target;
    } else if (pSearch->queue && pSearch->queue->v) {
        v_end = pSearch->queue->v;
    }

    if (!v_end)
        return NULL;

    // Count path length (including start)
    size_t length = 1;
    for (struct voxel* v = v_end; v && v->parent; v = v->parent)
        length++;

    struct path* p = path_init(length);
    if (!p->voxels) {
        free(p);
        return NULL;
    }

    // Fill in reverse order (start -> end)
    struct voxel* v = v_end;
    for (size_t i = length; i-- > 0 && v; v = v->parent)
        p->voxels[i] = v;
    
    // Add the length to Benchmark struct
    pSearch->bmd.n_path = length;

    return p;
}

void search_printBM(FILE *out, struct search* pSearch) {
    fprintf(out, "Nodes Explored: %ld\n", pSearch->bmd.n_explored);
    fprintf(out, "Nodes Evaluated: %ld\n", pSearch->bmd.n_evaluated);
    fprintf(out, "Queue Max: %ld\n", pSearch->bmd.queue_max);
    fprintf(out, "Queue Total Elements: %ld\n", pSearch->bmd.queue_ttl);
    fprintf(out, "Queue Average Elements: %0.2f\n", (float)pSearch->bmd.queue_ttl/(float)pSearch->bmd.n_explored);
    fprintf(out, "Queue Final Length: %ld\n", pSearch->bmd.queue_final);
    fprintf(out, "Path Length: %ld\n", pSearch->bmd.n_path);
    fprintf(out, "Path Distance: %0.2f\n", pSearch->bmd.path_length);

}
