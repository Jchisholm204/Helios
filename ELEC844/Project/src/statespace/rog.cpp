/**
 * @file rog.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-19
 * @modified Last Modified: 2025-11-19
 *
 * @copyright Copyright (c) 2025
 */

#include "statespace/rog.hpp"

#include <malloc.h>
#include <math.h>
#include <stdlib.h>

ROG_t* rog_init(size_t n_obstacles, flt coverage, flt var) {
    ROG_t* pROG = (ROG_t*) malloc(sizeof(ROG_t));
    if (!pROG)
        return NULL;

    pROG->n_obstacles = n_obstacles;
    pROG->obstacles =
        (hypersphere_t*) malloc(sizeof(hypersphere_t) * n_obstacles);

    if (!pROG->obstacles) {
        free(pROG);
        return NULL;
    }

    // Calculate Radius based on coverage and dimensions
    flt CD = pow(M_PI, N_DIMENSIONS / 2.0) / tgamma(N_DIMENSIONS / 2.0 + 1);

    flt avg_rad = pow((coverage / (n_obstacles * CD)), 1.0 / (flt)N_DIMENSIONS);
    flt min_rad = avg_rad - var;
    flt max_rad = avg_rad + var;

    printf("ROG generating %ld obstacles\n", n_obstacles);
    printf("ROG Radius = %3.3f\n", avg_rad);
    hypersphere_t* obstacle = pROG->obstacles;
    for (size_t i = 0; i < n_obstacles; i++) {
        for (size_t d = 0; d < N_DIMENSIONS; d++) {
            obstacle->position[d] = RAND_SAMPLE(DIM_MIN, DIM_MAX);
        }
        obstacle->radius = avg_rad;
        // obstacle->radius = RAND_SAMPLE(min_rad, max_rad);
        // Store r^2 for speedy lookup
        obstacle->radius = obstacle->radius * obstacle->radius;
        obstacle++;
    }
    return pROG;
}

bool rog_check(ROG_t* pROG, state_t state) {
    if (!pROG)
        return false;
    if (!state)
        return false;
    hypersphere_t* obstacle = pROG->obstacles;
    for (size_t i = 0; i < pROG->n_obstacles; i++) {
        flt d2_sum = 0;
        for (size_t d = 0; d < N_DIMENSIONS; d++) {
            flt d2 = obstacle->position[d] - state[d];
            d2_sum += d2 * d2;
        }
        if (d2_sum < obstacle->radius)
            return false;
        obstacle++;
    }
    return true;
}

void rog_free(ROG_t** ppROG) {
    if (ppROG) {
        if (*ppROG) {
            if ((*ppROG)->obstacles)
                free((*ppROG)->obstacles);
            free(*ppROG);
            *ppROG = NULL;
        }
    }
}
