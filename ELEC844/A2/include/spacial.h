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

#define BLOCK_SIZE 5

struct _spacial_block {
    struct xy dim;
    struct xy loc;
    struct voxel** voxels;
    size_t n_voxels;
    size_t v_len;
    bool collision;
};

struct spacial {
    struct xy dim;
    struct _spacial_block* blocks;
    struct voxel* voxels;
    size_t n_blocks;
    size_t n_voxels;
};

extern struct spacial* spacial_init(xy_t dim);

extern void spacial_free(struct spacial** ppSpacial);

extern void spacial_invalidate(struct spacial* pSpace, struct xy point);

extern int spacial_check(struct spacial* pSpace, struct xy point);

extern int spacial_checkPth(struct spacial* pSpace, struct xy p1, struct xy p2);

extern void spacial_addV(struct spacial* pSpace, struct xy point);

extern struct voxel* spacial_nearest(struct spacial* pSpace, struct xy point);

extern struct voxel* spacial_getV(struct spacial* pSpace, struct xy point);

#endif
