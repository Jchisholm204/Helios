/**
 * @file types.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-10-08
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _TYPES_H_
#define _TYPES_H_

#include <stddef.h>
#include <malloc.h>
#include <float.h>

struct xy{
    int x;
    int y;
};

enum eVoxelState {
    eStateEmpty,
    eStateExplored,
    eStateFrontier,
    eStateBlocked,
    eStatePath,
    eStateSource,
    eStateGoal
};

struct voxel {
    int x, y;
    enum eVoxelState state;
    float cost;
    struct voxel *parent;
    int open;
};

struct grid {
    struct voxel *voxels;
    struct xy size;
};

struct path {
    size_t n_voxels;
    struct voxel **voxels;
};

static inline struct voxel *grid_index(struct grid *pGrid, size_t col, size_t row){
    if(!pGrid) return NULL;
    if(!pGrid->voxels) return NULL;
    if(col >= pGrid->size.x) return NULL;
    if(row >= pGrid->size.y) return NULL;
    return &(pGrid->voxels[row*pGrid->size.x+col]);
}

static inline struct grid *grid_init(size_t n_cols, size_t n_rows){
    struct grid *g = malloc(sizeof(struct grid));
    if(!g) return NULL;
    g->voxels = malloc(n_cols*n_rows*sizeof(struct voxel));
    if(!g->voxels) {
        free(g);
        return NULL;
    }

    g->size.x = n_cols;
    g->size.y = n_rows;

    for(size_t x = 0; x < n_cols; x++){
        for(size_t y = 0; y < n_rows; y++){
            struct voxel *v = grid_index(g, x, y);
            v->state = eStateEmpty;
            v->x = x;
            v->y = y;
            v->cost = FLT_MAX;
            v->parent = NULL;
            v->open = 1;
        }
    }
    return g;
}

// Zero out Grid Weights
static inline void grid_zero(struct grid *pGrid){
    if(!pGrid) return;
    if(!pGrid->voxels) return;
    size_t n_cols = pGrid->size.x;
    size_t n_rows = pGrid->size.y;
    for(size_t x = 0; x < n_cols; x++){
        for(size_t y = 0; y < n_rows; y++){
            struct voxel *v = grid_index(pGrid, x, y);
            v->cost = FLT_MAX;
            v->parent = NULL;
            v->open = 1;
        }
    }

}

static inline void grid_free(struct grid **ppGrid){
    if(!ppGrid) return;
    if(!(*ppGrid)) return;
    if((*ppGrid)->voxels) free((*ppGrid)->voxels);
    free(*ppGrid);
    *ppGrid = NULL;
}

#endif
