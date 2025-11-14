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

#include "linked_queue.h"

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

struct voxel* worldV(spacial_t* pSpace, xy_t p) {
    if (!pSpace)
        return NULL;
    if (p.x >= pSpace->info.dim.x || p.y >= pSpace->info.dim.y)
        return NULL;
    if (p.x < 0 || p.y < 0)
        return NULL;
    size_t idx = p.y * pSpace->info.dim.y + p.x;
    return &pSpace->voxels[idx];
}

struct spacial* spacial_init(world_loader_fn world_loader) {
    struct spacial* s = malloc(sizeof(struct spacial));
    if (!s)
        return NULL;
    // Use null pass into the loader to get world info
    s->info = world_loader(NULL);
    // Sub elements of top block
    s->n_voxels = s->info.dim.x * s->info.dim.y;

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
        v->cost = 0;
        v->lookahead = 0;
        v->y = i / s->info.dim.y;
        v->x = i % s->info.dim.x;
    }
    // Use world loader to get the obstacles
    (void) world_loader(s);
    return s;
}

spacial_tree_t* spacial_tree_init(spacial_t* pSpace, xy_t start) {
    // Bounds check the start node (also null checks pSpace)
    struct voxel* vStart = worldV(pSpace, start);
    if (!vStart)
        return NULL;
    // Allocate the tree structure
    spacial_tree_t* t = malloc(sizeof(spacial_tree_t));
    if (!t)
        return NULL;
    t->n_voxels = 0;
    t->pSpace = pSpace;

    // Init the tree head
    t->pHead = malloc(sizeof(struct spacial_branch));
    if (!t->pHead) {
        free(t);
        return NULL;
    }
    struct spacial_branch* h = t->pHead;
    h->voxel.x = start.x;
    h->voxel.y = start.y;
    h->voxel.parent = NULL;
    h->voxel.state = eStateSource;
    h->voxel.cost = 0;
    h->voxel.lookahead = 0;
    h->pWorld = vStart;
    h->kd_split = SPLIT_X;
    h->pLess = NULL;
    h->pMore = NULL;
    h->pParent = NULL;
    h->children = NULL;
    return t;
}

void tree_free_branch(struct spacial_branch* b) {
    if (b->pLess) {
        tree_free_branch(b->pLess);
    }
    if (b->pMore) {
        tree_free_branch(b->pMore);
    }
    while(queue_length(b->children) > 0)
        (void)queue_pop(&b->children);
    free(b);
}

void spacial_tree_free(spacial_tree_t** ppTree) {
    struct spacial_branch* h = (*ppTree)->pHead;
    // Free the branches
    tree_free_branch(h);
    free(*ppTree);
    *ppTree = NULL;
}

void spacial_free(struct spacial** ppSpacial) {
    if (!ppSpacial)
        return;
    if (!*ppSpacial)
        return;
    if ((*ppSpacial)->voxels)
        free((*ppSpacial)->voxels);
    free(*ppSpacial);
    *ppSpacial = NULL;
}

void spacial_invalidate(struct spacial* pSpace, struct xy point) {
    if (!pSpace)
        return;
    struct voxel* v = worldV(pSpace, point);
    if (!v) {
        return;
    }
    v->state = eStateBlocked;
}

// Return 1 if in collision
int spacial_check(struct spacial* pSpace, struct xy point) {
    struct voxel* v = worldV(pSpace, point);
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

struct spacial_branch* spacial_addV(spacial_tree_t* pTree, struct xy point,
                                    struct spacial_branch* parent) {
    if (!pTree)
        return NULL;
    // Check/Get the world voxel
    struct voxel* wv = worldV(pTree->pSpace, point);
    if (!wv) {
        return NULL;
    }

    struct spacial_branch** current = &pTree->pHead;
    struct spacial_branch* prev = NULL;
    while (*current) {
        prev = *current;
        int cd = (*current)->kd_split;
        float p_eval = (cd == SPLIT_X) ? point.x : point.y;
        float b_eval =
            (cd == SPLIT_X) ? (*current)->voxel.x : (*current)->voxel.y;
        if (p_eval < b_eval) {
            current = &(*current)->pLess;
        } else {
            current = &(*current)->pMore;
        }
    }

    struct spacial_branch* new_branch = malloc(sizeof(struct spacial_branch));
    if (!new_branch)
        return NULL;
    new_branch->pParent = parent;
    new_branch->pMore = NULL;
    new_branch->pLess = NULL;
    new_branch->pWorld = wv;
    new_branch->kd_split = !prev->kd_split;
    new_branch->children = NULL;
    struct voxel* v = &new_branch->voxel;
    v->state = eStateExplored;
    v->parent = &parent->voxel;
    wv->parent = parent->pWorld;
    if (wv->state == eStateEmpty)
        wv->state = eStateExplored;
    v->x = point.x;
    v->y = point.y;

    *current = new_branch;
    queue_push(&parent->children, new_branch, 0);

    pTree->n_voxels++;

    return new_branch;
}

struct voxel* spacial_getV(spacial_t* pSpace, struct xy point) {
    return worldV(pSpace, point);
}

struct spacial_branch* spacial_nearest(spacial_tree_t* pTree, struct xy point) {
    if (!pTree)
        return NULL;
    if (!pTree->pSpace)
        return NULL;
    struct spacial_branch* closest = NULL;
    float closest_dist = FLT_MAX;

    // printf("Finding Nearest Voxel\n");

    struct spacial_branch* current = pTree->pHead;
    struct linked_queue* queue = NULL;
    while (current) {
        // Get distance of current to point
        float d_v = (pow((float) current->voxel.x - (float) point.x, 2) +
                         pow((float) current->voxel.y - (float) point.y, 2));
        // Update best
        if (d_v < closest_dist) {
            closest_dist = d_v;
            closest = current;
        }
        // printf("Checking Node (%3.1f %3.1f) d=%3.2f\n", current->voxel.x,
        // current->voxel.y, d_v);

        // Add more to queue
        int cd = current->kd_split;
        float point_coord = (cd == SPLIT_X) ? point.x : point.y;
        float node_coord =
            (cd == SPLIT_X) ? current->voxel.x : current->voxel.y;
        float plane_dist = (point_coord - node_coord);
        float plane_dist2 = plane_dist * plane_dist;

        // Near and far children
        struct spacial_branch* near =
            (point_coord < node_coord) ? current->pLess : current->pMore;
        struct spacial_branch* far =
            (point_coord < node_coord) ? current->pMore : current->pLess;

        if (near)
            queue_push(&queue, near, 0.0f); // explore near side first
        // only explore far side if its plane might contain a closer point
        if (far && plane_dist2 < closest_dist*closest_dist)
            queue_push(&queue, far, plane_dist2);
        current = queue_pop(&queue);
    }

    return closest;
}

struct linked_queue* spacial_nearestN(spacial_tree_t* pTree, struct xy point,
                                      float radius) {
    if (!pTree)
        return NULL;
    if (!pTree->pSpace)
        return NULL;

    struct spacial_branch* current = pTree->pHead;
    struct linked_queue* queue = NULL;
    struct linked_queue* nearby = NULL;
    while (current) {
        // Get distance of current to point
        float d_v = (pow((float) current->voxel.x - (float) point.x, 2) +
                         pow((float) current->voxel.y - (float) point.y, 2));
        // Update best
        if (d_v < radius*radius) {
            queue_push(&nearby, current, d_v);
        }
        // printf("Checking Node (%3.1f %3.1f) d=%3.2f\n", current->voxel.x,
        // current->voxel.y, d_v);

        // Add more to queue
        int cd = current->kd_split;
        float point_coord = (cd == SPLIT_X) ? point.x : point.y;
        float node_coord =
            (cd == SPLIT_X) ? current->voxel.x : current->voxel.y;
        float plane_dist = (point_coord - node_coord);
        float plane_dist2 = plane_dist * plane_dist;

        // Near and far children
        struct spacial_branch* near =
            (point_coord < node_coord) ? current->pLess : current->pMore;
        struct spacial_branch* far =
            (point_coord < node_coord) ? current->pMore : current->pLess;

        if (near)
            queue_push(&queue, near, 0.0f); // explore near side first
        // only explore far side if its plane might contain a closer point
        if (far && plane_dist2 < radius*radius)
            queue_push(&queue, far, plane_dist2);
        current = queue_pop(&queue);
    }

    return nearby;
}

int spacial_pathLen(struct spacial_branch* pGoal) {
    if (!pGoal)
        return -1;
    size_t path_len = 0;
    while (pGoal->pParent) {
        if (pGoal->pWorld->state != eStateGoal)
            pGoal->pWorld->state = eStatePath;
        // printf("Path (%3.1f, %3.1f)\n", pGoal->voxel.x, pGoal->voxel.y);
        if(pGoal == pGoal->pParent)
            return path_len;
        pGoal = pGoal->pParent;
        path_len++;
    }
    return path_len;
}
