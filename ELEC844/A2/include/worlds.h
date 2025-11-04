/**
 * @file worlds.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-04
 * @modified Last Modified: 2025-11-04
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _WORLDS_H_
#define _WORLDS_H_
#include "spacial.h"

typedef xy_t (*world_loader_fn)(struct spacial*, xy_t);

static xy_t gen_world1A(struct spacial* s, xy_t wd) {
    // Place main obstacle
    for (int x = 45; x < 55; x++) {
        for (int y = 25; y < wd.y - 25; y++) {
            spacial_invalidate(s, (struct xy) {x, y});
        }
    }
    // Place start point
    spacial_addV(s, (xy_t) {25, 50});
    spacial_getV(s, (xy_t) {25, 50})->state = eStateSource;
    spacial_getV(s, (xy_t) {wd.x - 25, 50})->state = eStateGoal;

    // Return goal point
    return (xy_t) {wd.x - 25, 50};
}

static xy_t gen_world1B(struct spacial* s, xy_t wd) {
    for (int x = 45; x < 55; x++) {
        for (int y = 4; y < wd.y - 4; y++) {
            spacial_invalidate(s, (struct xy) {x, y});
        }
    }
    // Place start point
    spacial_addV(s, (xy_t) {25, 50});
    spacial_getV(s, (xy_t) {25, 50})->state = eStateSource;
    spacial_getV(s, (xy_t) {wd.x - 25, 50})->state = eStateGoal;

    // Return goal point
    return (xy_t) {wd.x - 25, 50};
}

#endif
