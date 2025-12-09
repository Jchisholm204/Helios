/**
 * @file rog.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-19
 * @modified Last Modified: 2025-11-19
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _ROG_H_
#define _ROG_H_

#include "statespace/statespace.h"
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    state_t position;
    flt radius;
} hypersphere_t;

typedef struct {
    size_t n_obstacles;
    size_t batch;
    hypersphere_t *obstacles;
} ROG_t;

extern ROG_t * rog_init(size_t n_obstacles, flt coverage, flt var);

extern bool rog_check(ROG_t *pROG, state_t state);

extern void rog_free(ROG_t **ppROG);

#endif
