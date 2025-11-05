/**
 * @file rrt.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-04
 * @modified Last Modified: 2025-11-04
 *
 * @copyright Copyright (c) 2025
 */

#include "rrt.h"

#include <math.h>
#include <stdlib.h>

rrt_t* rrt_init(world_loader_fn world, long seed, xy_t dim, float goal_prob,
                float edge_length) {
    rrt_t* this = malloc(sizeof(rrt_t));
    if (!this)
        return NULL;

    this->pSpace = spacial_init(world);
    if (!this->pSpace) {
        free(this);
        return NULL;
    }
    this->pTree = spacial_tree_init(this->pSpace, this->pSpace->info.start);
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

void rrt_free(rrt_t** ppRRT) {
    if (!ppRRT)
        return;
    rrt_t* this = *ppRRT;
    if (!this)
        return;
    spacial_tree_free(&this->pTree);
    spacial_free(&this->pSpace);
    free(this);
    *ppRRT = NULL;
}

int rrt_main(rrt_t* this) {
    if (!this)
        return -1;
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

    // printf("Sampled Point (%3.1f %3.1f)\n", t_x, t_y);

    // If the sampled point is blocked, rerun this function to sample a new
    // point
    if (spacial_check(this->pSpace, (xy_t) {t_x, t_y})) {
        return rrt_main(this);
    }

    // Find the closest point to the sampled point
    struct spacial_branch* nearest =
        spacial_nearest(this->pTree, (xy_t) {t_x, t_y});
    if (!nearest) {
        // printf("Nearest NULL\n");
        rrt_main(this);
    }

    // printf("Found Nearest Point (%3.1f %3.1f)\n", nearest->voxel.x,
    // nearest->voxel.y);

    // Normalize the vector and multiply it to get the new point
    float n_v_norm =
        sqrt(pow(t_x - nearest->voxel.x, 2) + pow(t_y - nearest->voxel.y, 2));
    // printf("Vec Norm = %3.2f\n", n_v_norm);
    float p_x = this->edge_length * (t_x - nearest->voxel.x) / n_v_norm +
                nearest->voxel.x;
    float p_y = this->edge_length * (t_y - nearest->voxel.y) / n_v_norm +
                nearest->voxel.y;

    // printf("Plotting New Point (%3.1f %3.1f)\n", p_x, p_y);

    // Check the point exists in the space
    if (spacial_check(this->pSpace, (xy_t){p_x, p_y})) {
        return rrt_main(this);
    }

    // Ensure the path n_v->p_v is free
    if (spacial_checkPth(this->pSpace,
                         (xy_t) {nearest->voxel.x, nearest->voxel.y},
                         (xy_t) {p_x, p_y})) {
        // printf("Selected Point was in collision\n");
        return rrt_main(this);
    }

    // Setup the new point
    int addr = spacial_addV(this->pTree, (xy_t) {p_x, p_y}, nearest);
    if (addr != 0) {
        // printf("AddV returned %d\n", addr);
    }

    // Check if the goal is within distance to the point
    float d_goal =
        sqrt(pow(p_x - this->p_goal.x, 2) + pow(p_y - this->p_goal.y, 2));
    if (d_goal < this->edge_length) {
        // printf("Reached Goal!\n");

        spacial_addV(this->pTree, this->p_goal,
                     spacial_nearest(this->pTree, (xy_t) {p_x, p_y}));
        this->found_target = true;
        return 1;
    }

    return 0;
}
