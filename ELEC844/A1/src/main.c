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
#include "worlds.h"
#include "search.h"


int main(int argc, char **argv){
    disp_t *d = disp_init(WORLD_X, WORLD_Y);

    world_loader(d->grid, eWorld1A);

    // grid_index(d->grid, 9, 3)->state = eStateExplored;
    // grid_index(d->grid, 9, 4)->state = eStateFrontier;


    struct path path;
    path.n_voxels = 4;
    path.voxels = malloc(path.n_voxels*sizeof(struct voxel*));
    path.voxels[0] = grid_index(d->grid, 2, 3);
    path.voxels[1] = grid_index(d->grid, 3, 3);
    path.voxels[2] = grid_index(d->grid, 4, 4);
    path.voxels[3] = grid_index(d->grid, 4, 5);
    // disp_drawPath(d, &path);
    grid_zero(d->grid);
    SDL_Event e;
    while(1){
        while(SDL_PollEvent(&e)){
            if(e.type == SDL_QUIT)
                goto exit;
        }
        int r = search_stepA(dist, d->grid, grid_index(d->grid, 4, 9), grid_index(d->grid, 14, 9));
        printf("Explored Grid with return code %d\n", r);
        disp_drawGrid(d);
        disp_render(d);
        SDL_Delay(500);
    }

exit:
    printf("Shutting Down..\n");
    disp_exit(&d);
    return 0;
}
