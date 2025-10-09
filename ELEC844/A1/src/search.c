/**
 * @file search.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-10-08
 *
 * @copyright Copyright (c) 2025
 */

#include "search.h"
#include <math.h>

float dist(struct voxel *s, struct voxel *d){
    return sqrt(pow(s->y-d->y, 2) + pow(s->x - d->x, 2));
}

int search_stepA(heuristic_fn hfn, struct grid *pGrid, struct voxel *s, struct voxel *g){
    if(!hfn) return -1;
    if(!pGrid) return -1;
    if(!pGrid->voxels) return -1;
    if(!s) return -1;
    if(!g) return -1;
    
    // Setup target
    if(s->state == eStateSource)
        s->cost = 0;
    else
        return -2;
    struct voxel *vn = g;
    size_t n_vert = pGrid->size.x*pGrid->size.y;
    for(size_t i = 0; i < n_vert; i++){
        struct voxel *v = &pGrid->voxels[i];
        if(v->cost < vn->cost && v->state != eStateBlocked && v->open == 1){
            printf("Found Lowest Node (%d %d) %0.2f\n", v->x, v->y, v->cost);
            vn = v;
        }
    }
    if(!vn) return -3;
    printf("Exploring Vertex: (%d, %d)\n", vn->x, vn->y);
    vn->open = 0;
    if(vn->state != eStateSource) vn->state = eStateFrontier;

    // Process all nearby voxels
    for(int x = -1; x <= 1; x++){
        for(int y = -1; y <= 1; y++){
            struct voxel *nv = grid_index(pGrid, vn->x+x, vn->y+y);
            if(!nv) continue;
            printf("Exploring nearby node (%d, %d)\n", nv->x, nv->y);
            if(nv == g) return 1;
            float goal_dist = hfn(nv, g);
            float reach_cost = dist(nv, vn) + vn->cost;
            if(reach_cost < nv->cost){
                nv->cost = reach_cost;
                nv->parent = vn;
                if(nv->state != eStateSource && nv->state != eStateGoal) nv->state = eStateExplored;
            }
            if(goal_dist < hfn(vn, g)){
                nv->open = 1;
                if(nv->state != eStateSource && nv->state != eStateGoal) nv->state = eStateFrontier;
            }
        }
    }
        
    return 0;
}
