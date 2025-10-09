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

    // Initialize the display
    disp_t *d = disp_init(WORLD_X, WORLD_Y);

    // Load the world
    world_loader(d->grid, eWorld2B);

    // Setup the search
    grid_zero(d->grid);
    struct search *s = search_init(dist, d->grid);

    // SDL loop until finished
    SDL_Event e;
    while(1){
        while(SDL_PollEvent(&e)){
            if(e.type == SDL_QUIT)
                goto exit;
        }
        if(!s->finished){
            int r = search_stepA(s);
            // printf("Search Returned %d\n", r);
        }

        // Backtrace the path
        struct path *p = search_backtrace(s);
        disp_drawGrid(d);
        disp_drawPath(d, p);
        if(p)
            if(p->voxels) free(p->voxels);
        if(p) free(p);
        disp_render(d);
        SDL_Delay(200);
    }

exit:
    search_free(&s);
    printf("Shutting Down..\n");
    disp_exit(&d);
    return 0;
}
