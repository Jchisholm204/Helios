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

bool is_descendant(struct spacial_branch* node,
                   struct spacial_branch* potential_child) {
    if (!node || !node->children)
        return false;

    struct linked_queue* child = node->children;
    while (child) {
        struct spacial_branch* b = child->data;
        if (b == potential_child)
            return true;
        if (is_descendant(b, potential_child))
            return true;
        child = child->next;
    }
    return false;
}

// Walk up parent pointers to see if 'possible_ancestor' is an ancestor of
// 'node'. This is safer than walking children. Guarded to avoid infinite loops
// on an already-cyclic tree.
static bool is_ancestor_safe(struct spacial_branch* node,
                             struct spacial_branch* possible_ancestor) {
    if (!node || !possible_ancestor)
        return false;
    struct spacial_branch* cur = node->pParent;
    int steps = 0;
    const int MAX_STEPS = 100000; // large guard
    while (cur && steps++ < MAX_STEPS) {
        if (cur == possible_ancestor)
            return true;
        cur = cur->pParent;
    }
    return false;
}

// Attempt to detach 'node' from its current parent children list and attach to
// new_parent. Returns true only on full success. Does not change voxel.cost
// (caller will update it).
static bool reparent_node_safe(struct spacial_branch* node,
                               struct spacial_branch* new_parent) {
    if (!node)
        return false;

    // Prevent trivial cycle
    if (is_ancestor_safe(new_parent, node))
        return false;

    struct spacial_branch* old_parent = node->pParent;

    // Detach from old parent first
    if (old_parent) {
        if (!queue_find_remove(&old_parent->children, node)) {
            return false; // detach failed, leave structure unchanged
        }
    }

    // Attach to new parent
    node->pParent = new_parent;
    if (new_parent) {
        // prevent duplicate children
        if (!queue_find_remove(&new_parent->children, node)) {
            // not present, safe to push
            queue_push(&new_parent->children, node, 0);
        } else {
            // we removed an accidental duplicate; re-add a single instance for
            // cleanliness
            queue_push(&new_parent->children, node, 0);
        }
    }
    return true;
}

// Quick Euclidean helper
static float dist_xy(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx * dx + dy * dy);
}

// Floyd's cycle detection on parent pointers from a starting node.
// Returns true if a cycle exists, false otherwise.
static bool detect_cycle_from(struct spacial_branch* start) {
    if (!start)
        return false;
    struct spacial_branch *t = start, *h = start;
    while (true) {
        // move tortoise by 1
        if (t->pParent)
            t = t->pParent;
        else
            return false;
        // move hare by 2
        if (h->pParent && h->pParent->pParent) {
            h = h->pParent->pParent;
        } else {
            return false;
        }
        if (t == h)
            return true;
    }
}

void update_children(rrtstar_t* this, struct linked_queue* children,
                     float cost) {
    while (children) {
        struct spacial_branch* b = children->data;
        float b_cost = sqrt(pow(b->pParent->voxel.x - b->voxel.x, 2) +
                            pow(b->pParent->voxel.y - b->voxel.y, 2));
        if (b->voxel.cost > cost + b_cost) {
            b->voxel.cost = cost + b_cost;
            update_children(this, b->children, b->voxel.cost);
        }
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

    // printf("Sampled Point (%3.1f %3.1f)\n", s_x, s_y);
    // this->pSpace->voxels[(int)s_y * this->pSpace->info.dim.y +
    // (int)s_x].state = eStatePath;

    // If the sampled point is blocked, rerun this function to sample a new
    // point
    if (spacial_check(this->pSpace, (xy_t) {s_x, s_y})) {
        return rrtstar_main(this);
    }

    // Find the closest point to the sampled point
    struct spacial_branch* nearest =
        spacial_nearest(this->pTree, (xy_t) {s_x, s_y});
    if (!nearest) {
        // printf("Nearest NULL\n");
        rrtstar_main(this);
    }

    // printf("Found Nearest Point (%3.1f %3.1f)\n", nearest->voxel.x,
    // nearest->voxel.y);

    // Normalize the vector and multiply it to get the new point
    float n_v_norm =
        sqrt(pow(s_x - nearest->voxel.x, 2) + pow(s_y - nearest->voxel.y, 2));
    // printf("Vec Norm = %3.2f\n", n_v_norm);
    float p_x = this->edge_length * (s_x - nearest->voxel.x) / n_v_norm +
                nearest->voxel.x;
    float p_y = this->edge_length * (s_y - nearest->voxel.y) / n_v_norm +
                nearest->voxel.y;

    // Find lowest cost parent for the new point
    struct linked_queue* nearby =
        spacial_nearestN(this->pTree, (xy_t) {p_x, p_y}, this->edge_length);
    while (queue_length(nearby) > 0) {
        struct spacial_branch* b = queue_pop(&nearby);
        if (b->voxel.cost < nearest->voxel.cost)
            nearest = b;
    }

    // printf("Plotting New Point (%3.1f %3.1f)\n", p_x, p_y);

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
    float new_cost = added_v->voxel.cost =
        nearest->voxel.cost +
        sqrt(pow(p_x - nearest->voxel.x, 2) + pow(p_y - nearest->voxel.y, 2));

    struct linked_queue* nearestN =
        spacial_nearestN(this->pTree, (xy_t) {p_x, p_y}, this->edge_length);

    // Check if any nearby nodes can be rewired to this node
    while (queue_length(nearestN) > 0) {
        struct spacial_branch* b = queue_pop(&nearestN);
        if (b == added_v)
            continue;

        float b_cost = dist_xy(p_x, p_y, b->voxel.x, b->voxel.y);
        float candidate_cost = new_cost + b_cost;
        if (b->voxel.cost <= candidate_cost)
            continue;

        // If b is already parented to added_v, we still may need to update
        // children costs.
        if (b->pParent == added_v) {
            b->voxel.cost = candidate_cost; // keep in sync
            update_children(this, b->children, b->voxel.cost);
            continue;
        }

        // Prevent cycles: ensure added_v is NOT inside b's ancestor chain
        if (is_ancestor_safe(added_v, b)) {
            // would create a loop, skip
            continue;
        }

        // Attempt atomic reparent: detach from old parent and attach under
        // added_v. Only if this succeeds do we update cost and propagate.
        if (!reparent_node_safe(b, added_v)) {
            // detach/attach failed -> skip
            continue;
        }

        // Now safe: update cost and propagate to descendants
        b->voxel.cost = candidate_cost;
        update_children(this, b->children, b->voxel.cost);
    }

    // Check if the goal is within distance to the point
    float d_goal =
        sqrt(pow(p_x - this->p_goal.x, 2) + pow(p_y - this->p_goal.y, 2));
    if (d_goal < this->edge_length) {
        // printf("Reached Goal!\n");

        this->target = spacial_addV(this->pTree, this->p_goal, added_v);
        this->target->voxel.cost = added_v->voxel.cost + this->edge_length;
        this->found_target = true;
        return 1;
    }

    return 0;
}
