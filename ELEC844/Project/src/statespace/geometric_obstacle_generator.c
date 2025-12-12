/**
 * @file geometric_obstacle_generator.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-12
 * @modified Last Modified: 2025-12-12
 *
 * @copyright Copyright (c) 2025
 */

#include "statespace/geometric_obstacle_generator.h"

#define MAX_RETRY (STATESPACE_MAX >> 2)

state_t *gog_start(gog_t *gog) {
    if (!gog) {
        return NULL;
    }
    state_t *start = (state_t *) malloc(sizeof(state_t));
    if (!start) {
        return NULL;
    }

    // Set the initial start point to STATESPACE_MIN
    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        (*start)[i] = STATESPACE_MIN + STATESPACE_ST_OFFSET;
    }

    // Attempt to find a valid start point
    size_t tries = 0;
    for (; tries < MAX_RETRY && gog_check(gog, start); tries++) {
        // Move away from the origin on a 45 deg angle
        for (size_t i = 0; i < STATESPACE_DIMS; i++) {
            (*start)[i]++;
        }
    }
    if (tries == MAX_RETRY) {
        free(start);
        fprintf(stderr,
                "GOG: Failed to find a valid start point (retry limit hit)\n");
        return NULL;
    }

    // Reset the GOG access counter
    gog->access_counter = 0;
    return start;
}

state_t *gog_target(gog_t *gog) {
    if (!gog) {
        return NULL;
    }
    state_t *target = (state_t *) malloc(sizeof(state_t));
    if (!target) {
        return NULL;
    }

    // Set the initial start point to STATESPACE_MAX
    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        (*target)[i] = STATESPACE_MAX - 1 - STATESPACE_ST_OFFSET;
    }

    // Attempt to find a valid start point
    size_t tries = 0;
    for (; tries < MAX_RETRY && gog_check(gog, target); tries++) {
        // Move away from the end on a 45 deg angle
        for (size_t i = 0; i < STATESPACE_DIMS; i++) {
            (*target)[i]--;
        }
    }
    if (tries == MAX_RETRY) {
        free(target);
        fprintf(stderr,
                "GOG: Failed to find a valid target point (retry limit hit)\n");
        return NULL;
    }

    // Reset the GOG access counter
    gog->access_counter = 0;
    return target;
}
