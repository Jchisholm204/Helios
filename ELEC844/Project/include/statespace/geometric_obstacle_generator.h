/**
 * @file geometric_obstacle_generator.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-09
 * @modified Last Modified: 2025-12-09
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
    state_t protecteds;
    size_t access_counter;
} gog_t;

static inline gog_t* gog_init(size_t seed) {
    srand(seed);
    gog_t* gog = (gog_t*) malloc(sizeof(gog_t));

    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        gog->bmasks[i] = 2;
        gog->patterns[i] = 3;
    }

    gog->access_counter = 0;

    return gog;
}

static inline void gog_free(gog_t** gog) {
    if (gog) {
        if (*gog) {
            free(*gog);
        }
        *gog = NULL;
    }
}

static inline bool gog_check(gog_t* gog, state_t* p) {
    unsigned char invalid = 0x00;
    gog->access_counter++;
    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        unsigned char dim_val = (*p)[i];
        unsigned char mbit = gog->bmasks[i];
        unsigned char pbit = gog->patterns[i];
        unsigned char pattern = (dim_val >> (pbit & 0x07)) & 0xFF;
        unsigned char dim_inval = (pattern + dim_val) & 0xFF;
        dim_inval = (dim_inval >> (mbit & 0x07));
        // dim_inval = (dim_inval ^ invalid) & 0x01;
        invalid += (dim_inval & 0x01);
    }
    return invalid == (STATESPACE_DIMS);
}

#endif
