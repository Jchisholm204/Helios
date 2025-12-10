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
#include <stdlib.h>
#include <stdio.h>

#define GOG_N_IDENTIFIERS 2

typedef struct {
    state_t patterns[GOG_N_IDENTIFIERS];
    state_t masks[GOG_N_IDENTIFIERS];
    size_t access_counter;
} gog_t;

static inline gog_t* gog_init(size_t seed) {
    srand(seed);
    gog_t* gog = (gog_t*) malloc(sizeof(gog_t));

    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        gog->masks[i][0] = 0x06;
        gog->patterns[i][0] = 0x02;
        gog->masks[i][1] = 0x018;
        gog->patterns[i][1] = 0x08;
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
    unsigned char valid = 0xFF;
    gog->access_counter++;
    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        int j = 0;
        // unsigned char dim_inval = 0x00;
        // for(size_t j = 0; j < GOG_N_IDENTIFIERS; j++){
            unsigned char temp = (*p)[i] & gog->masks[i][j];
            unsigned char temp2 = temp ^ gog->patterns[i][j];
            // printf("0x%x & 0x%x ^ 0x%x\n", (*p)[i], temp, temp2);
            valid &= temp2;
        // }
        // valid &= dim_inval;
    }
    return valid != 0x0;
}

#endif
