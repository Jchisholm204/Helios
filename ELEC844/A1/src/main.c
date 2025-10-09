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
 * ELEC 844 Assignment 1
 *
 */

#include "display.h"
#include "search.h"
#include "worlds.h"

#include <SDL.h>
#include <SDL2/SDL.h>
#include <stdio.h>

int main(int argc, char** argv) {

    // Initialize the display
    disp_t* d = disp_init(WORLD_X, WORLD_Y);

    // Load the world
    world_loader(d->grid, eWorld2B);

    // Setup the search
    grid_zero(d->grid);

    // Select heuristic here
    struct search* s = search_init(dist, d->grid);

    // SDL loop until finished
    SDL_Event e;
    while (1) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                goto exit;
        }

        // Check to see if the search has finished, if not run next iteration
        if (!s->finished) {
            int r = search_stepA(s);
        }

        // Backtrace the path
        struct path* p = search_backtrace(s);
        // clears screen - must be run before other draw functions
        disp_clr(d);
        // Draw grid and path
        disp_drawGrid(d);
        disp_drawPath(d, p);
        // Render the display
        disp_render(d);
        // free the path
        path_free(&p);

        // Run a delay for the animation
        SDL_Delay(50);
    }

exit:
    search_free(&s);
    printf("Shutting Down..\n");
    disp_exit(&d);
    return 0;
}
