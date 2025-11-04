/**
 * @file main.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 2.2
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-10-08
 *
 * @copyright Copyright (c) 2024
 *
 * ELEC 844 Assignment 1:
 *  A* and LPA*
 *
 */

#include "display.h"

#include <SDL.h>
#include <SDL2/SDL.h>
#include <stdio.h>
#include "spacial.h"

void gen_world(struct spacial *s, int l){
    for(int x = 45; x < 55; x++){
        for(int y = l; y < 100-l; y++){
            spacial_invalidate(s, (struct xy){x, y});
        }
    }
}

int main(int argc, char** argv) {


    // Initialize the display
    disp_t* d = disp_init(100);

    struct spacial *sp = spacial_init((struct xy){100, 100});
    gen_world(sp, 4);
    spacial_getV(sp, (struct xy){25, 50})->state = eStateSource;
    spacial_addV(sp, (struct xy){25, 50});
    spacial_addV(sp, (struct xy){30, 50});
    struct voxel *v = spacial_nearest(sp, (struct xy){75, 60});
    printf("Nearest: %3.2f %3.2f\n", v->x, v->y);
    


    // SDL loop until finished
    SDL_Event e;
    while (1) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                goto exit;
        }

        disp_clr(d);
        // Draw grid and path
        disp_drawGrid(d);
        disp_drawPoints(d, sp);
        // Render the display
        disp_render(d);

        // Run a delay for the animation
        SDL_Delay(50);
    }
wait_exit:
    // Print out the search results
    printf("Finished Search!\n");
    while (1) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                goto exit;
        }
    }

exit:
    printf("Shutting Down..\n");
    disp_exit(&d);
    return 0;
}
