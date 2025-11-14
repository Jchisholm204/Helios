/**
 * @file rrtc.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief RRT-Connect
 * @version 0.1
 * @date Created: 2025-11-05
 * @modified Last Modified: 2025-11-05
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _RRTC_H_
#define _RRTC_H_
#include "spacial.h"
#include "worlds.h"

typedef struct {
    spacial_t* pSpace;
    spacial_tree_t *pTree;
    spacial_tree_t *pTreeG;
    world_loader_fn world_loader;
    xy_t p_goal;
    unsigned int seed;
    float goal_prob;
    long n_iterations;
    float edge_length;
    struct spacial_branch *target;
    bool found_target;
} rrtc_t;

extern rrtc_t *rrtc_init(world_loader_fn world, long seed, xy_t dim, float goal_prob, float edge_length);

extern void rrtc_free(rrtc_t **ppRRTC);

extern int rrtc_main(rrtc_t *pRRTC);

#endif

