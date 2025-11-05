/**
 * @file rrt.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-04
 * @modified Last Modified: 2025-11-04
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _RRT_H_
#define _RRT_H_
#include "spacial.h"
#include "worlds.h"

typedef struct {
    struct spacial* pSpace;
    world_loader_fn world_loader;
    xy_t p_goal;
    unsigned int seed;
    float goal_prob;
    long n_iterations;
    float edge_length;
    bool found_target;
} rrt_t;

extern rrt_t *rrt_init(world_loader_fn world, long seed, xy_t dim, float goal_prob, float edge_length);

extern void rrt_free(rrt_t **ppRRT);

extern int rrt_main(rrt_t *pRRT);

#endif
