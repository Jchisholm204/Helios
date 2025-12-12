/**
 * @file geometric_obstacle_generator.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-09
 * @modified Last Modified: 2025-12-11
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _GEOMETRIC_OBSTACLE_GENERATOR_H_
#define _GEOMETRIC_OBSTACLE_GENERATOR_H_

#include "statespace.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    state_t bmasks;
    state_t patterns;
    state_t variable;
    size_t access_counter;
} gog_t;

/**
 * @brief Initialize the GOG
 *
 * @param seed RNG Seed used to generate the environment
 * @param bmask Bit used to set the frequency obstacles (1 <= bmask <= 6)
 * @param pmask Bit used to introduce variability in the pattern (1 <= pmask <= 6)
 * @return Null on failure, GOG object on success
 */
static inline gog_t* gog_init(size_t seed, uint bmask, uint pmask) {
    srand(seed);
    gog_t* gog = (gog_t*) malloc(sizeof(gog_t));
    if (!gog || bmask > 6 || bmask < 1 || pmask > 6 || pmask < 1) {
        return NULL;
    }

    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        gog->variable[i] = rand() & 0xFF;
        gog->bmasks[i] = bmask + (1 - (rand() & 0x03));
        gog->patterns[i] = pmask + (1 - (rand() & 0x03));
    }

    gog->access_counter = 0;

    return gog;
}

/**
 * @brief GOG object cleanup handler
 *
 * @param gog pointer to the GOG object pointer
 */
static inline void gog_free(gog_t** gog) {
    if (gog) {
        if (*gog) {
            free(*gog);
        }
        *gog = NULL;
    }
}

/**
 * @brief Check if a point is valid/invalid
 *
 * @param gog The GOG object pointer
 * @param p point to check
 * @return 1 if invalid, 0 otherwise (including failure)
 */
static inline bool gog_check(gog_t* gog, state_t* p) {
    if (!gog || !p) {
        return false;
    }
    unsigned char invalid = 0x00;
    gog->access_counter++;
    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        unsigned char dim_val = (*p)[i] ^ gog->variable[i];
        unsigned char pattern = (((*p)[i] >> (gog->patterns[i]) & 0x07));
        unsigned char dim_inval = (pattern + dim_val);
        dim_inval = (dim_inval >> ((gog->bmasks[i]) & 0x07));
        invalid += ((dim_inval) & 0x01);
    }
    return invalid == (STATESPACE_DIMS);
}

#endif
