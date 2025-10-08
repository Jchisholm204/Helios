/**
 * @file main.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 2.1
 * @date 2024-05-22
 * 
 * @copyright Copyright (c) 2024
 * 
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <SDL.h>
#include <SDL2/SDL.h>

#include "display.h"

int main(int argc, char **argv){
    disp_t *d = disp_init(19, 19);

    grid_index(d->grid, 4, 3)->state = eStateSource;
    grid_index(d->grid, 10, 14)->state = eStateGoal;
    grid_index(d->grid, 9, 4)->state = eStateBlocked;
    grid_index(d->grid, 4, 4)->state = eStateExplored;

    disp_drawGrid(d);
    struct path path;
    path.n_voxels = 4;
    path.voxels = malloc(path.n_voxels*sizeof(struct voxel*));
    path.voxels[0] = grid_index(d->grid, 2, 3);
    path.voxels[1] = grid_index(d->grid, 3, 3);
    path.voxels[2] = grid_index(d->grid, 4, 4);
    path.voxels[3] = grid_index(d->grid, 4, 5);
    disp_drawPath(d, &path);
    disp_render(d);

    SDL_Event e;
    while(1){
        while(SDL_PollEvent(&e)){
            if(e.type == SDL_QUIT)
                goto exit;
        }
    }

exit:
    printf("Shutting Down..\n");
    disp_exit(&d);
    return 0;
}
