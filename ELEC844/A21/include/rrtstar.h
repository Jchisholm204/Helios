/**
 * @file rrtstar.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.2
 * @date Created: 2025-11-04
 * @modified Last Modified: 2025-11-06
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _RRTSTAR_H_
#define _RRTSTAR_H_
#include "spacial.h"
#include "worlds.h"

typedef struct {
    struct spacial* pSpace;
    spacial_tree_t *pTree;
    struct spacial_branch *target;
    world_loader_fn world_loader;
    xy_t p_goal;
    unsigned int seed;
    float goal_prob;
    long n_iterations;
    float edge_length;
    bool found_target;
} rrtstar_t;

extern rrtstar_t *rrtstar_init(world_loader_fn world, long seed, xy_t dim, float goal_prob, float edge_length);

extern void rrtstar_free(rrtstar_t **pThis);

extern int rrtstar_main(rrtstar_t *this);

#endif

