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

#include <stdio.h>

struct spacial* spacial_init(struct xy dim) {
    struct spacial* s = malloc(sizeof(struct spacial));
    if (!s)
        return NULL;
    s->dim = dim;
    // Sub elements of top block
    s->n_blocks = BLOCK_SIZE * BLOCK_SIZE;
    s->n_voxels = dim.x * dim.y;

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
    if(block->n_voxels >= (block->v_len-2)){
        block->v_len *= 2;
        block->voxels = realloc(block->voxels, block->v_len);
    }
    block->voxels[block->n_voxels++] = v;
}

// Return 1 if in collision
int spacial_check(struct spacial* pSpace, struct xy point) {
    int x = point.x;
    int y = point.y;
    int block_idx = (y / BLOCK_SIZE) + (x % BLOCK_SIZE);
    if (block_idx >= pSpace->n_blocks)
        return 1;
    struct _spacial_block* block = &pSpace->blocks[block_idx];
    if (!block)
        return 1;
    // if the block does not have any invalid points in it, return not in
    // collision
    if (!block->collision)
        return 0;
    // struct voxel* v = &block->voxels[block->n_elements++];
    return 0;
}

int spacial_checkp(struct spacial* pSpace, struct xy p1, struct xy p2) {
}

struct voxel** spacial_nearby(struct spacial* pSpace, struct xy point,
                              float radius) {
}
