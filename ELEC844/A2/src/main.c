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
#include "rrt.h"
#include "spacial.h"
#include "worlds.h"

#include <SDL.h>
#include <SDL2/SDL.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char** argv) {

    // Initialize the display
    disp_t* d = disp_init(100);

    rrt_t* planner =
        rrt_init(gen_world1A, time(NULL), (xy_t) {100, 100}, 0.01, 2.5);
    struct spacial* sp = planner->pSpace;

    // SDL loop until finished
    SDL_Event e;
    while (1) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                goto exit;
        }

        rrt_main(planner);
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
