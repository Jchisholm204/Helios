/**
 * @file spacial.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-03
 * @modified Last Modified: 2025-11-03
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _SPACIAL_H_
#define _SPACIAL_H_
#include "types.h"

#include <float.h>
#include <malloc.h>
#include <stdbool.h>
#include <stddef.h>

#define SPLIT_X 0
#define SPLIT_Y 1

struct world_info {
    xy_t start, target, dim;
};

typedef struct spacial {
    struct world_info info;
    struct voxel* voxels;
    size_t n_voxels;
} spacial_t;

struct spacial_branch {
    int kd_split;
    struct spacial_branch* pParent;
    // Left Tree
    struct spacial_branch* pLess;
    // Right Tree
    struct spacial_branch* pMore;
    // Voxel Local to the tree
    struct voxel voxel;
    // World Voxel
    struct voxel* pWorld;
};

typedef struct spacial_tree {
    // Head node of the tree
    struct spacial_branch* pHead;
    spacial_t* pSpace;
    // Voxels stored in lower levels
    size_t n_voxels;
} spacial_tree_t;

typedef struct world_info (*world_loader_fn)(struct spacial*);

extern spacial_t* spacial_init(world_loader_fn world_loader);

extern spacial_tree_t* spacial_tree_init(spacial_t* pSpacial, xy_t start);

extern void spacial_tree_free(spacial_tree_t** ppTree);

extern void spacial_free(struct spacial** ppSpacial);

extern void spacial_invalidate(struct spacial* pSpace, struct xy point);

extern int spacial_check(struct spacial* pSpace, struct xy point);

extern int spacial_checkPth(struct spacial* pSpace, struct xy p1, struct xy p2);

extern struct spacial_branch* spacial_addV(spacial_tree_t* pTree,
                                           struct xy point,
                                           struct spacial_branch* parent);

extern struct spacial_branch* spacial_nearest(spacial_tree_t* pTree,
                                              struct xy point);

extern struct spacial_branch* spacial_nearestN(spacial_tree_t* pTree,
                                               struct xy point);

extern struct voxel* spacial_getV(spacial_t* pSpace, struct xy point);

extern int spacial_pathLen(struct spacial_branch* pGoal);

#endif
