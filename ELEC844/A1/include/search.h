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
#include "types.h"
#include "linked_queue.h"

typedef float (*heuristic_fn)(struct voxel *s, struct voxel *d);

float dist(struct voxel *s, struct voxel *d);

struct search {
    heuristic_fn heuristic;
    struct grid *pGrid;
    struct voxel *start;
    struct voxel *target;
    struct queued_voxel **queue;
};

void search_free(struct search **ppSearch){
    if(!ppSearch) return;
    if(!*ppSearch) return;
    free(*ppSearch);
    *ppSearch = NULL;
}

struct search *search_init(heuristic_fn hfn, struct grid *pGrid){
    if(!pGrid) return NULL;
    if(!pGrid->voxels) return NULL;
    // Allocate the search structure
    struct search *s = malloc(sizeof(struct search));
    if(!s) return NULL;
    s->pGrid = pGrid;
    s->heuristic = hfn;
    s->queue = NULL;
    s->start = NULL;
    s->target = NULL;
    size_t n_voxels = pGrid->size.x*pGrid->size.y;
    for(size_t i = 0; i < n_voxels; i++){
        if(pGrid->voxels[i].state == eStateSource)
            s->start = &pGrid->voxels[i];
        if(pGrid->voxels[i].state == eStateGoal)
            s->target = &pGrid->voxels[i];
    }

    // Error condition if start or target is not found
    if(!s->start || s->target)
        search_free(&s);
    return s;
}


int search_stepA(struct search *pSearch);

#endif
