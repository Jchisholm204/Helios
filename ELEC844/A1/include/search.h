/**
 * @file search.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-10-08
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _SEARCH_H_
#define _SEARCH_H_
#include "linked_queue.h"
#include "types.h"

#include <math.h>
#include <stdlib.h>

// heuristic function template
typedef float (*heuristic_fn)(struct voxel* s, struct voxel* d);

static float hfn_zero(struct voxel* s, struct voxel* d) {
    return 0;
}

static float hfn_euclean(struct voxel* s, struct voxel* d) {
    return sqrt(pow(s->y - d->y, 2) + pow(s->x - d->x, 2));
}

static float hfn_manhattan(struct voxel* s, struct voxel* d) {
    return abs(s->x - d->x) + abs(s->y - d->y);
}

static float hfn_inflated(struct voxel* s, struct voxel* d) {
    return 100 * hfn_euclean(s, d);
}

struct search {
    heuristic_fn heuristic;
    struct grid* pGrid;
    struct voxel* start;
    struct voxel* target;
    struct queued_voxel* queue;
    int finished;
    // BenchMarking Data
    struct {
        size_t n_explored;
        size_t n_evaluated;
        size_t queue_max;
        size_t queue_ttl;
        size_t queue_final;
        size_t n_path;
        float path_length;
    } bmd;
};

static void search_free(struct search** ppSearch) {
    if (!ppSearch)
        return;
    if (!*ppSearch)
        return;
    free(*ppSearch);
    *ppSearch = NULL;
}

static struct search* search_init(heuristic_fn hfn, struct grid* pGrid) {
    if (!pGrid)
        return NULL;
    if (!pGrid->voxels)
        return NULL;
    // Allocate the search structure
    struct search* s = malloc(sizeof(struct search));
    if (!s)
        return NULL;
    s->pGrid = pGrid;
    s->heuristic = hfn;
    s->queue = NULL;
    s->start = NULL;
    s->target = NULL;
    s->finished = 0;
    size_t n_voxels = pGrid->size.x * pGrid->size.y;
    for (size_t i = 0; i < n_voxels; i++) {
        if (pGrid->voxels[i].state == eStateSource)
            s->start = &pGrid->voxels[i];
        if (pGrid->voxels[i].state == eStateGoal)
            s->target = &pGrid->voxels[i];
    }

    // Error condition if start or target is not found
    if (!s->start || !s->target){
        search_free(&s);
        return NULL;
    }

    s->start->cost = 0;
    // Push the start node
    queue_push(&s->queue, s->start, 0);
    
    // Zero out BenchMarking data
    s->bmd.n_explored = 0;
    s->bmd.n_evaluated = 0;
    s->bmd.queue_max = 0;
    s->bmd.queue_ttl = 0;
    s->bmd.queue_final = 0;
    s->bmd.n_path = 0;
    s->bmd.path_length = 0;

    return s;
}

static void search_swapGoal(struct search *pSearch){
    if(!pSearch) return;
    struct voxel *tv;
    tv = pSearch->start;
    pSearch->start = pSearch->target;
    pSearch->target = tv;
    pSearch->target->state = eStateGoal;
    pSearch->start->state = eStateSource;
    pSearch->start->cost = 0;
    pSearch->target->cost = FLT_MAX;
    (void)queue_pop(&pSearch->queue);
    queue_push(&pSearch->queue, pSearch->start, 0);
}

int search_stepA(struct search* pSearch);

struct path* search_backtrace(struct search* pSearch);

void search_printBM(FILE* out, struct search *pSearch);

#endif
