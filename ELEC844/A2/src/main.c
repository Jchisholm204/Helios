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
#include "kdtree.h"
#include "prng.h"
#include "rrt.h"
#include "rrtc.h"
#include "spacial.h"
#include "worlds.h"

#include <SDL.h>
#include <SDL2/SDL.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

int benchmark_rrt(int runs) {
    printf("Running %d benchmarks\n", runs);

    FILE* testf = fopen("./benchmark.csv", "w");
    if (!testf) {
        printf("Test File could not be opened\n");
        return 0;
    }
    fprintf(testf, "#Running 100 Trials#World1B#RRT#\n");
    fprintf(testf, "#run, #iterations, #verticies, #solution\n");
    float sum_iter = 0, sum_added = 0, sum_path = 0;
    pcg32_random_t rg;
    pcg32_srandom_r(&rg, time(NULL), getpid());
    // disp_t* d = disp_init(100);
    for (int run = 0; run < runs; run++) {
        // Create the planner and world
        rrt_t* planner = rrt_init(gen_world1B, pcg32_random_r(&rg),
                                  (xy_t) {100, 100}, 0.01, 2.5);

        // Run the planner to find the path
        while (!planner->found_target) {
            rrt_main(planner);
            // disp_clr(d);
            // // Draw grid and path
            // disp_drawGrid(d);
            // disp_drawPoints(d, planner->pSpace);
            // // Render the display
            // disp_render(d);
        }

        // Collect Stats
        size_t n_iterations = planner->n_iterations;
        sum_iter += n_iterations;
        size_t n_added = planner->pTree->n_voxels;
        sum_added += n_added;
        // TODO: Fix this
        size_t n_solution = spacial_pathLen(spacial_nearest(planner->pTree, planner->p_goal));
        sum_path += n_solution;
        fprintf(testf, "%d, %ld, %ld, %ld\n", run, n_iterations, n_added,
                n_solution);
        printf("%d, %3.2f, %3.2f, %3.2f\n", run, sum_iter / run,
               sum_added / run, sum_path / run);
        // Free the planner
        rrt_free(&planner);
    }
    printf("ITER | VRTX | PATH\n");
    printf("%3.2f, %3.2f, %3.2f\n", sum_iter / runs, sum_added / runs,
           sum_path / runs);
    fclose(testf);
    return 0;
}

int benchmark_rrtc(int runs) {
    printf("Running %d benchmarks\n", runs);

    FILE* testf = fopen("./benchmark_rrtc_w2c.csv", "w");
    if (!testf) {
        printf("Test File could not be opened\n");
        return 0;
    }
    fprintf(testf, "#Running 100 Trials#World2C#RRTC#\n");
    fprintf(testf, "#run, #iterations, #verticies, #solution\n");
    float sum_iter = 0, sum_added = 0, sum_path = 0;
    pcg32_random_t rg;
    pcg32_srandom_r(&rg, time(NULL), getpid());
    disp_t* d = disp_init(100);
    for (int run = 0; run < runs; run++) {
        // Create the planner and world
        rrtc_t* planner = rrtc_init(gen_world2C, pcg32_random_r(&rg),
                                  (xy_t) {100, 100}, 0.01, 2.5);

        // Run the planner to find the path
        while (!planner->found_target) {
            rrtc_main(planner);
            // disp_clr(d);
            // // Draw grid and path
            // disp_drawGrid(d);
            // disp_drawPoints(d, planner->pSpace);
            // // Render the display
            // disp_render(d);
        }

        // Collect Stats
        size_t n_iterations = planner->n_iterations;
        sum_iter += n_iterations;
        size_t n_added = planner->pTree->n_voxels;
        sum_added += n_added;
        // TODO: Fix this
        size_t n_solution = spacial_pathLen(spacial_nearest(planner->pTree, planner->p_goal));
        sum_path += n_solution;
        fprintf(testf, "%d, %ld, %ld, %ld\n", run, n_iterations, n_added,
                n_solution);
        printf("%d, %3.2f, %3.2f, %3.2f\n", run, sum_iter / run,
               sum_added / run, sum_path / run);
        // Free the planner
        rrtc_free(&planner);
    }
    printf("ITER | VRTX | PATH\n");
    printf("%3.2f, %3.2f, %3.2f\n", sum_iter / runs, sum_added / runs,
           sum_path / runs);
    fclose(testf);
    return 0;
}

int view_rrtc(void) {
    // Initialize the display
    disp_t* d = disp_init(100);

    rrtc_t* planner =
        rrtc_init(gen_world2C, time(NULL), (xy_t) {100, 100}, 0.01, 2.5);

    // SDL loop until finished
    SDL_Event e;
    while (1) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                goto exit;
        }

        if (rrtc_main(planner)) {
            goto wait_exit;
        }

        disp_clr(d);
        // Draw grid and path
        disp_drawGrid(d);
        disp_drawPoints(d, planner->pSpace);
        // Render the display
        disp_render(d);

        // Run a delay for the animation
        SDL_Delay(10);
    }
wait_exit:
    // Print out the search results
    printf("Finished Search!\n");
    printf("%ld Iterations got Path Length = %d \n", planner->n_iterations,
           spacial_pathLen(spacial_nearest(planner->pTree, planner->p_goal)));
    printf("%ld verticies were created\n", planner->pTree->n_voxels);
    while (1) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                goto exit;
        }
        disp_clr(d);
        // Draw grid and path
        disp_drawGrid(d);
        disp_drawPoints(d, planner->pSpace);
        // Render the display
        disp_render(d);
    }

exit:
    printf("Shutting Down..\n");
    disp_exit(&d);
    return 0;
}

int view_rrt(void) {
    // Initialize the display
    disp_t* d = disp_init(100);

    rrt_t* planner =
        rrt_init(gen_world1B, time(NULL), (xy_t) {100, 100}, 0.01, 2.5);

    // SDL loop until finished
    SDL_Event e;
    while (1) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                goto exit;
        }

        if (rrt_main(planner)) {
            goto wait_exit;
        }

        disp_clr(d);
        // Draw grid and path
        disp_drawGrid(d);
        disp_drawPoints(d, planner->pSpace);
        // Render the display
        disp_render(d);

        // Run a delay for the animation
        SDL_Delay(10);
    }
wait_exit:
    // Print out the search results
    printf("Finished Search!\n");
    printf("%ld Iterations got Path Length = %d \n", planner->n_iterations,
           spacial_pathLen(spacial_nearest(planner->pTree, planner->p_goal)));
    printf("%ld verticies were created\n", planner->pTree->n_voxels);
    while (1) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                goto exit;
        }
        disp_clr(d);
        // Draw grid and path
        disp_drawGrid(d);
        disp_drawPoints(d, planner->pSpace);
        // Render the display
        disp_render(d);
    }

exit:
    printf("Shutting Down..\n");
    disp_exit(&d);
    return 0;
}

int main(int argc, char** argv) {

    if (argc == 1) {
        return view_rrtc();
    } else if (argc == 2) {
        int runs = atoi(argv[1]);
        if (runs == 0) {
            printf("Invalid Usage... First argument should be the number of "
                   "tests to run\n");
            printf("Use no arguments to run the visualization\n");
            return 0;
        }
        return benchmark_rrtc(runs);
    }
}
