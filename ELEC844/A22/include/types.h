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

#include <float.h>
#include <malloc.h>
#include <stddef.h>

struct xy {
    int x;
    int y;
};

typedef struct xy xy_t;

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
    float x, y;
    enum eVoxelState state;
    float cost;
    float lookahead;
    struct voxel* parent;
};

struct path {
    size_t n_voxels;
    struct voxel** voxels;
};


#endif

