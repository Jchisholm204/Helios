/**
 * @file spacial.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-03
 * @modified Last Modified: 2025-11-03
 *
 * @copyright Copyright (c) 2025
 */

#include "spacial.h"

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

struct spacial* spacial_init(xy_t dim) {
    struct spacial* s = malloc(sizeof(struct spacial));
    if (!s)
        return NULL;
    s->dim = dim;
    // Sub elements of top block
    s->n_blocks = BLOCK_SIZE * BLOCK_SIZE;
    s->n_voxels = dim.x * dim.y;
    s->n_added = 0;

    // Create the collision grid (row major)
    s->voxels = malloc(s->n_voxels * sizeof(struct voxel));
    if (!s->voxels) {
        free(s);
        return NULL;
    }
    for (int i = 0; i < s->n_voxels; i++) {
        struct voxel* v = &s->voxels[i];
        v->state = eStateEmpty;
        v->parent = NULL;
    }

    // Create the blocks
    s->blocks = malloc(s->n_blocks * sizeof(struct _spacial_block));
    if (!s->blocks) {
        free(s->voxels);
        free(s);
        return NULL;
    }
    for (int i = 0; i < s->n_blocks; i++) {
        struct _spacial_block* b = &s->blocks[i];
        b->n_voxels = 0;
        b->v_len = s->n_voxels / s->n_blocks;
        b->voxels = malloc(sizeof(struct voxel*) * b->v_len);
        b->collision = false;
        if (!b->voxels) {
            for (int j = 0; j < i; j++)
                free(s->blocks[j].voxels);
            free(s->blocks);
            free(s->voxels);
            free(s);
            return NULL;
        }
    }

    return s;
}

void spacial_free(struct spacial** ppSpacial) {
    if (!ppSpacial)
        return;
    if (!*ppSpacial)
        return;
    if ((*ppSpacial)->blocks) {
        for (int i = 0; i < (*ppSpacial)->n_blocks; i++) {
            if ((*ppSpacial)->blocks[i].voxels)
                free((*ppSpacial)->blocks[i].voxels);
        }
        free((*ppSpacial)->blocks);
    }
    if ((*ppSpacial)->voxels)
        free((*ppSpacial)->voxels);
    free(*ppSpacial);
    *ppSpacial = NULL;
}

void spacial_invalidate(struct spacial* pSpace, struct xy point) {
    if (!pSpace)
        return;
    int x = point.x;
    int y = point.y;
    // Find the block
    int block_idx = (y / BLOCK_SIZE) + (x % BLOCK_SIZE);
    if (block_idx >= pSpace->n_blocks)
        return;
    struct _spacial_block* block = &pSpace->blocks[block_idx];
    if (!block)
        return;
    // Find the point in the global grid
    int voxel_idx = (y * pSpace->dim.y) + (x);
    struct voxel* v = &pSpace->voxels[voxel_idx];
    v->y = y;
    v->x = x;
    v->state = eStateBlocked;
    block->collision = true;
    // if (block->n_voxels >= (block->v_len - 2)) {
    //     block->v_len *= 2;
    //     block->voxels = realloc(block->voxels, block->v_len);
    // }
    // block->voxels[block->n_voxels++] = v;
}

// Return 1 if in collision
int spacial_check(struct spacial* pSpace, struct xy point) {
    struct voxel* v = spacial_getV(pSpace, point);
    if (!v)
        return 1;
    // Check the voxel state
    if (v->state == eStateBlocked)
        return 1;
    return 0;
}

// Return 1 if in collision
int spacial_checkPth(struct spacial* pSpace, struct xy p1, struct xy p2) {
    if (!pSpace)
        return 1;
    float x_d = (p1.x - p2.x);
    float y_d = (p1.y - p2.y);
    float d = sqrt(pow(x_d, 2) + pow(y_d, 2));
    float x_inc = x_d / d;
    float y_inc = y_d / d;
    float x = p1.x;
    float y = p1.y;
    for (int i = 0; i < d; i++) {
        if (spacial_check(pSpace, (struct xy) {x, y}))
            return 1;
        x += x_inc;
        y += y_inc;
    }
    return 0;
}

void spacial_addV(struct spacial* pSpace, struct xy point) {
    if (!pSpace)
        return;
    // Find the block
    int block_idx = (point.y / BLOCK_SIZE) + (point.x % BLOCK_SIZE);
    if (block_idx >= pSpace->n_blocks)
        return;
    struct _spacial_block* block = &pSpace->blocks[block_idx];
    if (!block)
        return;
    if (block->n_voxels >= (block->v_len - 2)) {
        block->v_len *= 2;
        block->voxels = realloc(block->voxels, block->v_len);
    }
    struct voxel* v = spacial_getV(pSpace, point);
    v->x = point.x;
    v->y = point.y;
    if (v->state == eStateEmpty)
        v->state = eStateExplored;
    block->voxels[block->n_voxels++] = v;
    pSpace->n_added++;
}

struct voxel* spacial_getV(struct spacial* pSpace, struct xy point) {
    if (!pSpace)
        return NULL;
    int voxel_idx = (point.y * pSpace->dim.y) + point.x;
    if (voxel_idx >= pSpace->n_voxels)
        return NULL;
    return &pSpace->voxels[voxel_idx];
}

struct voxel* spacial_nearest(struct spacial* pSpace, struct xy point) {
    if (!pSpace)
        return NULL;
    struct voxel* closest = NULL;
    float closest_dist = FLT_MAX;
    for (int b_idx = 0; b_idx < pSpace->n_blocks; b_idx++) {
        struct _spacial_block* block = &pSpace->blocks[b_idx];
        for (int i = 0; i < block->n_voxels; i++) {
            struct voxel* v = block->voxels[i];
            float d_v = sqrt(pow(v->x - point.x, 2) + pow(v->y - point.y, 2));
            // Add points closer than the radius
            if (d_v < closest_dist) {
                closest = v;
                closest_dist = d_v;
            }
        }
    }
    return closest;
}

int spacial_pathLen(struct spacial* pSpace, xy_t goal) {
    struct voxel* v = spacial_getV(pSpace, goal);
    if (!v)
        return -1;
    size_t path_len = 0;
    while (v->parent) {
        if (v->state == eStateExplored)
            v->state = eStatePath;
        v = v->parent;
        path_len++;
    }
    return path_len;
}
