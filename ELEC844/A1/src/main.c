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
#include "search.h"
#include "worlds.h"

#include <SDL.h>
#include <SDL2/SDL.h>
#include <stdio.h>

int main(int argc, char** argv) {

    // Select the world to start with
    enum eWorlds world = eWorld2A;

    // Initialize the display
    disp_t* d = disp_init(WORLD_X, WORLD_Y);

    // Load the world
    world_loader(d->grid, world);

    // Setup the search
    grid_zero(d->grid);

    // Select heuristic here
    struct search* s = search_init(hfn_euclean, d->grid, eSearchLPA);

    // Swap the start and goal (for question 3B)
    // search_swapGoal(s);

    // SDL loop until finished
    SDL_Event e;
    while (1) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                goto exit;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_SPACE) {
                // Only change the world if the search has finished
                if (s->finished) {
                    world++;
                    if (world == eWorld_n)
                        goto wait_exit;

                    // Update the world
                    struct queued_voxel* update_list = NULL;
                    world_updater(d->grid, world, &update_list);
                    search_update(s, &update_list);
                    s->finished = 0;
                    printf("Finished World %d\n", world);
                    // Print out the search results
                    search_printBM(stdout, s);
                }
            }
        }

        // Check to see if the search has finished, if not run next iteration
        if (!s->finished) {
            search_runsearch(s);
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
wait_exit:
    printf("Finished World %d\n", world);
    // Print out the search results
    search_printBM(stdout, s);
    printf("Finished Search!\n");
    while (1) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                goto exit;
        }
    }

exit:
    search_free(&s);
    printf("Shutting Down..\n");
    disp_exit(&d);
    return 0;
}
