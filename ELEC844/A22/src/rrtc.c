/**
 * @file rrtc.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-05
 * @modified Last Modified: 2025-11-05
 *
 * @copyright Copyright (c) 2025
 */

#include "rrtc.h"

#include <math.h>
#include <stdlib.h>

rrtc_t* rrtc_init(world_loader_fn world, long seed, xy_t dim, float goal_prob,
                  float edge_length) {
    rrtc_t* this = malloc(sizeof(rrtc_t));
    if (!this)
        return NULL;
    this->pSpace = spacial_init(world);
    if (!this->pSpace) {
        free(this);
        return NULL;
    }
    this->pTree = spacial_tree_init(this->pSpace, this->pSpace->info.start);
    this->pTreeG = spacial_tree_init(this->pSpace, this->pSpace->info.target);
    // Load the world
    this->world_loader = world;
    this->p_goal = this->pSpace->info.target;

    // Seed random generator
    this->seed = seed;
    srand(this->seed);

    this->goal_prob = goal_prob;
    this->n_iterations = 0;
    this->edge_length = edge_length;
    this->found_target = false;

    return this;
}

void rrtc_free(rrtc_t** ppRRTC) {
    if (!ppRRTC)
        return;
    rrtc_t* this = *ppRRTC;
    if (!this)
        return;
    spacial_tree_free(&this->pTree);
    spacial_free(&this->pSpace);
    free(this);
    *ppRRTC = NULL;
    return;
}

int rrtc_merge_path(rrtc_t *this, struct spacial_branch *b_s, struct spacial_branch *b_t){
    if(!this || !b_s || !b_t)
        return -1;
    size_t path_len = 0;
    while (b_t) {
        b_s = spacial_addV(this->pTree, (xy_t){b_t->voxel.x, b_t->voxel.y}, b_s);
        b_t= b_t->pParent;
        path_len++;
    }
    return path_len;
}

int rrtc_connect(rrtc_t* this, float t_x, float t_y, struct spacial_branch *steer) {
    // Begin Connect Logic

    // Find nearest node in goal tree
    struct spacial_branch* nearest =
        spacial_nearest(this->pTreeG, (xy_t) {t_x, t_y});
    while (nearest && !this->found_target) {
        this->n_iterations++;

        // Normalize the vector and multiply it to get the new point
        float n_v_norm = sqrt(pow(t_x - nearest->voxel.x, 2) +
                              pow(t_y - nearest->voxel.y, 2));
        // printf("Vec Norm = %3.2f\n", n_v_norm);
        float p_x = this->edge_length * (t_x - nearest->voxel.x) / n_v_norm +
                    nearest->voxel.x;
        float p_y = this->edge_length * (t_y - nearest->voxel.y) / n_v_norm +
                    nearest->voxel.y;

        // Add the new point to the goal tree
        // Check the point exists in the space
        if (spacial_check(this->pSpace, (xy_t) {p_x, p_y})) {
            // printf("Collision\n");
            return 0;
        }

        // Ensure the path n_v->p_v is free
        if (spacial_checkPth(this->pSpace,
                             (xy_t) {nearest->voxel.x, nearest->voxel.y},
                             (xy_t) {p_x, p_y})) {
            // printf("Selected Point was in collision\n");
            return 0;
        }

        // Setup the new point
        struct spacial_branch* added_g =
            spacial_addV(this->pTreeG, (xy_t) {p_x, p_y}, nearest);

        if (!added_g) {
            return 0;
        }

        // Check if the goal is within distance to the point
        float d_goal = sqrt(pow(p_x - t_x, 2) + pow(p_y - t_y, 2));
        if (d_goal < this->edge_length) {
            // printf("Reached Goal!\n");
            this->found_target = true;
            rrtc_merge_path(this, steer, added_g);
            return 1;
        }
        // Find nearest node in goal tree
        nearest = added_g;
    }
    return 0;
}

int rrtc_main(rrtc_t* pRRTC) {
    if (!pRRTC)
        return -1;
    rrtc_t* this = pRRTC;
    if (this->found_target)
        return 1;
    this->n_iterations++;

    // Figure out if next bias point is the target
    bool target_bias = rand() < this->goal_prob * RAND_MAX;

    // Get the sample point
    float t_x = this->p_goal.x;
    float t_y = this->p_goal.y;
    if (!target_bias) {
        t_x = (float) 100.0 * rand() / RAND_MAX;
        t_y = (float) 100.0 * rand() / RAND_MAX;
    }

    // If the sampled point is blocked, rerun this function to sample a new
    // point
    if (spacial_check(this->pSpace, (xy_t) {t_x, t_y})) {
        return rrtc_main(this);
    }

    // Find the closest point to the sampled point
    struct spacial_branch* nearest =
        spacial_nearest(this->pTree, (xy_t) {t_x, t_y});
    if (!nearest) {
        // printf("Nearest NULL\n");
        rrtc_main(this);
    }

    // Normalize the vector and multiply it to get the new point
    float n_v_norm =
        sqrt(pow(t_x - nearest->voxel.x, 2) + pow(t_y - nearest->voxel.y, 2));
    // printf("Vec Norm = %3.2f\n", n_v_norm);
    float p_x = this->edge_length * (t_x - nearest->voxel.x) / n_v_norm +
                nearest->voxel.x;
    float p_y = this->edge_length * (t_y - nearest->voxel.y) / n_v_norm +
                nearest->voxel.y;

    // Check the point exists in the space
    if (spacial_check(this->pSpace, (xy_t) {p_x, p_y})) {
        return rrtc_main(this);
    }

    // Ensure the path n_v->p_v is free
    if (spacial_checkPth(this->pSpace,
                         (xy_t) {nearest->voxel.x, nearest->voxel.y},
                         (xy_t) {p_x, p_y})) {
        // printf("Selected Point was in collision\n");
        return rrtc_main(this);
    }

    // Setup the new point
    struct spacial_branch* added_v =
        spacial_addV(this->pTree, (xy_t) {p_x, p_y}, nearest);
    if(!added_v)
        return rrtc_main(this);

    return rrtc_connect(this, p_x, p_y, added_v);
}


