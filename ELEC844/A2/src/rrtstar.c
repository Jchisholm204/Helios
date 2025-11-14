/**
 * @file rrtstar.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.2
 * @date Created: 2025-11-04
 * @modified Last Modified: 2025-11-06
 *
 * @copyright Copyright (c) 2025
 */

#include "rrtstar.h"

#include "linked_queue.h"

#include <math.h>
#include <stdlib.h>

rrtstar_t* rrtstar_init(world_loader_fn world, long seed, xy_t dim,
                        float goal_prob, float edge_length) {
    rrtstar_t* this = malloc(sizeof(rrtstar_t));
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
    this->target = NULL;

    // Seed random generator
    this->seed = seed;
    srand(this->seed);

    this->goal_prob = goal_prob;
    this->n_iterations = 0;
    this->edge_length = edge_length;
    this->found_target = false;
    return this;
}

void rrtstar_free(rrtstar_t** ppRRT) {
    if (!ppRRT)
        return;
    rrtstar_t* this = *ppRRT;
    if (!this)
        return;
    spacial_tree_free(&this->pTree);
    spacial_free(&this->pSpace);
    free(this);
    *ppRRT = NULL;
}

// Quick Euclidean helper
static float dist_xy(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx * dx + dy * dy);
}

void update_children_cost(struct spacial_branch* branch) {
    ll_t* children = branch->children;
    // printf("Updating Children of (%2.0f, %2.0f)\n", branch->voxel.x,
    // branch->voxel.y);
    while (children) {
        struct spacial_branch* child = children->data;
        float dd = dist_xy(branch->voxel.x, branch->voxel.y, child->voxel.x,
                           child->voxel.y);
        // printf("    Child: (%2.0f, %2.0f) %2.1f -> %2.1f\n", branch->voxel.x,
        //        branch->voxel.y, child->voxel.cost, dd + branch->voxel.cost);
        child->voxel.cost = dd + branch->voxel.cost;
        update_children_cost(child);
        children = children->next;
    }
}

int rrtstar_main(rrtstar_t* this) {
    if (!this)
        return -1;
    this->n_iterations++;

    // Figure out if next bias point is the target
    bool target_bias = rand() < this->goal_prob * RAND_MAX;

    // Get the sample point
    float s_x = this->p_goal.x;
    float s_y = this->p_goal.y;
    if (!target_bias) {
        s_x = (float) 100.0 * rand() / RAND_MAX;
        s_y = (float) 100.0 * rand() / RAND_MAX;
    }

    // If the sampled point is blocked, rerun this function to sample a new
    // point
    if (spacial_check(this->pSpace, (xy_t) {s_x, s_y})) {
        return rrtstar_main(this);
    }

    // Find the closest point to the sampled point
    struct spacial_branch* nearest =
        spacial_nearest(this->pTree, (xy_t) {s_x, s_y});
    // Nearest will be null if its in collision or does not exist
    if (!nearest) {
        return rrtstar_main(this);
    }

    // Normalize the vector and multiply it to get the new point
    float n_v_norm =
        sqrt(pow(s_x - nearest->voxel.x, 2) + pow(s_y - nearest->voxel.y, 2));
    float p_x = this->edge_length * (s_x - nearest->voxel.x) / n_v_norm +
                nearest->voxel.x;
    float p_y = this->edge_length * (s_y - nearest->voxel.y) / n_v_norm +
                nearest->voxel.y;

    // Find lowest cost parent for the new point
    ll_t* nearby =
        spacial_nearestN(this->pTree, (xy_t) {p_x, p_y}, this->edge_length);
    while (ll_length(nearby) > 0) {
        struct spacial_branch* b = ll_pop(&nearby);
        if (b->voxel.cost < nearest->voxel.cost)
            nearest = b;
    }

    // Check the point exists in the space
    if (spacial_check(this->pSpace, (xy_t) {p_x, p_y})) {
        return rrtstar_main(this);
    }

    // Ensure the path n_v->p_v is free
    if (spacial_checkPth(this->pSpace,
                         (xy_t) {nearest->voxel.x, nearest->voxel.y},
                         (xy_t) {p_x, p_y})) {
        // printf("Selected Point was in collision\n");
        return rrtstar_main(this);
    }

    // Setup the new point
    struct spacial_branch* added_v =
        spacial_addV(this->pTree, (xy_t) {p_x, p_y}, nearest);
    if (!added_v) {
        return rrtstar_main(this);
    }
    float new_cost = nearest->voxel.cost +
                     dist_xy(nearest->voxel.x, nearest->voxel.y, p_x, p_y);
    added_v->voxel.cost = new_cost;

    // printf("Added Vertex (%2.0f, %2.0f) p=(%2.0f, %2.0f) pc=%2.1f
    // nc=%2.1f\n",
    //        p_x, p_y, nearest->voxel.x, nearest->voxel.y, nearest->voxel.cost,
    //        new_cost);

    ll_t* nearestN =
        spacial_nearestN(this->pTree, (xy_t) {p_x, p_y}, this->edge_length);

    // Check if any nearby nodes can be rewired to this node
    while (ll_length(nearestN) > 0) {
        struct spacial_branch* b = ll_pop(&nearestN);
        if (b == added_v)
            continue;

        float b_cost = dist_xy(p_x, p_y, b->voxel.x, b->voxel.y);
        float candidate_cost = new_cost + b_cost;
        if (b->voxel.cost <= candidate_cost)
            continue;

        // Ensure the path n_v->p_v is free
        if (spacial_checkPth(this->pSpace, (xy_t) {b->voxel.x, b->voxel.y},
                             (xy_t) {p_x, p_y}))
            continue;

        // printf(
        //     "Found New Child: (%2.0f, %2.0f) %2.1f -> (%2.0f, %2.0f)
        //     %2.1f\n", p_x, p_y, new_cost, b->voxel.x, b->voxel.y,
        //     candidate_cost);
        // Now safe: update cost and propagate to descendants
        b->voxel.cost = candidate_cost;
        // Remove b as a child from its current parent
        ll_find_remove(&b->pParent->children, b);
        while (ll_contains(b->pParent->children, b) == 1) {
            ll_find_remove(&b->pParent->children, b);
            // printf("Excess Removals\n");
        }
        // Set its new parent to the new node
        b->pParent = added_v;
        if (ll_contains(added_v->children, b) == 0)
            ll_push(&added_v->children, b);
        update_children_cost(b);
    }

    // After the rewiring while loop ends (after line 171)

    // Check if any nearby nodes (including rewired ones) can reach the goal
    ll_t* nodes_near_goal =
        spacial_nearestN(this->pTree, this->p_goal, this->edge_length);
    while (ll_length(nodes_near_goal) > 0) {
        struct spacial_branch* candidate = ll_pop(&nodes_near_goal);

        float dist_to_goal = dist_xy(candidate->voxel.x, candidate->voxel.y,
                                     this->p_goal.x, this->p_goal.y);

        if (dist_to_goal < this->edge_length) {
            float candidate_goal_cost = candidate->voxel.cost + dist_to_goal;

            if (!this->target ||
                candidate_goal_cost < this->target->voxel.cost) {
                // Either create target or rewire it to this better parent
                if (!this->target) {
                    this->target =
                        spacial_addV(this->pTree, this->p_goal, candidate);
                } else {
                    ll_find_remove(&this->target->pParent->children,
                                   this->target);
                    this->target->pParent = candidate;
                    ll_push(&candidate->children, this->target);
                }
                this->target->voxel.cost = candidate_goal_cost;
                this->found_target = true;
                printf("TRG COST=%3.2f\n", this->target->voxel.cost);
            }
        }
    }

    // Check if the goal is within distance to the point
    // float d_goal =
    //     sqrt(pow(p_x - this->p_goal.x, 2) + pow(p_y - this->p_goal.y, 2));
    // if (d_goal < this->edge_length) {
    //     // printf("Reached Goal!\n");
    //
    //     if (this->target) {
    //         if (added_v->voxel.cost + d_goal < this->target->voxel.cost) {
    //             // Remove the targets previous parent
    //             ll_find_remove(&this->target->pParent->children,
    //             this->target); while
    //             (ll_contains(this->target->pParent->children,
    //                                this->target) == 1) {
    //                 ll_find_remove(&this->target->pParent->children,
    //                                this->target);
    //             }
    //             // Set the new parent
    //             this->target->voxel.cost = added_v->voxel.cost + d_goal;
    //             this->target->pParent = added_v;
    //             if (ll_contains(added_v->children, this->target) == 0)
    //                 ll_push(&added_v->children, this->target);
    //         }
    //     } else {
    //         this->target = spacial_addV(this->pTree, this->p_goal, added_v);
    //         this->target->voxel.cost = added_v->voxel.cost + d_goal;
    //     }
    //     this->found_target = true;
    //
    //     return 1;
    // }

    return 0;
}

int spacial_clearPath(spacial_t* pSpace) {
    if (!pSpace)
        return -1;
    for (int i = 0; i < pSpace->n_voxels; i++) {
        if (pSpace->voxels[i].state == eStatePath) {
            pSpace->voxels[i].state = eStateExplored;
        }
    }
    return 0;
}
