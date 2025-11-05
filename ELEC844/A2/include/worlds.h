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

#define BLOCK(s_x, s_y, e_x, e_y) \
    for(int x = s_x; x < e_x; x++){ \
        for(int y = s_y; y < e_y; y++) \
        spacial_invalidate(s, (struct xy){x, y}); \
    } \


static xy_t gen_world1A(struct spacial* s, xy_t wd) {
    if (!s)
        // Return goal point on null
        return (xy_t) {wd.x - 25, 50};
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
    if (!s)
        // Return goal point on null
        return (xy_t) {wd.x - 25, 50};
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

static xy_t gen_world2A(struct spacial *s, xy_t wd){
    if (!s)
        // Return goal point on null
        return (xy_t) {wd.x - 25, 50};
    // Top Block
    BLOCK(10, 35, 40, 40);
    // Back
    BLOCK(35, 35, 40, 65);
    // Bottom
    BLOCK(10, 60, 40, 65);
    // Opening
    BLOCK(10, 35, 15, 49);
    BLOCK(10, 51, 15, 65);
    // Place start point
    spacial_addV(s, (xy_t) {25, 50});
    spacial_getV(s, (xy_t) {25, 50})->state = eStateSource;
    spacial_getV(s, (xy_t) {wd.x - 25, 50})->state = eStateGoal;

    // Return goal point
    return (xy_t) {wd.x - 25, 50};
}

static xy_t gen_world2B(struct spacial *s, xy_t wd){
    if (!s)
        // Return goal point on null
        return (xy_t) {wd.x - 25, 50};

    // Top Block
    BLOCK(60, 35, 90, 40);
    // Back
    BLOCK(60, 35, 65, 65);
    // Bottom
    BLOCK(60, 60, 90, 65);
    // Opening
    BLOCK(85, 35, 90, 49);
    BLOCK(85, 51, 90, 65);
    // Place start point
    spacial_addV(s, (xy_t) {25, 50});
    spacial_getV(s, (xy_t) {25, 50})->state = eStateSource;
    spacial_getV(s, (xy_t) {wd.x - 25, 50})->state = eStateGoal;

    // Return goal point
    return (xy_t) {wd.x - 25, 50};
}

static xy_t gen_world2C(struct spacial *s, xy_t wd){
    if (!s)
        // Return goal point on null
        return (xy_t) {wd.x - 25, 50};
    // Top Block
    BLOCK(10, 35, 40, 40);
    // Back
    BLOCK(35, 35, 40, 65);
    // Bottom
    BLOCK(10, 60, 40, 65);
    // Opening
    BLOCK(10, 35, 15, 49);
    BLOCK(10, 51, 15, 65);

    // Top Block
    BLOCK(60, 35, 90, 40);
    // Back
    BLOCK(60, 35, 65, 65);
    // Bottom
    BLOCK(60, 60, 90, 65);
    // Opening
    BLOCK(85, 35, 90, 49);
    BLOCK(85, 51, 90, 65);
    // Place start point
    spacial_addV(s, (xy_t) {25, 50});
    spacial_getV(s, (xy_t) {25, 50})->state = eStateSource;
    spacial_getV(s, (xy_t) {wd.x - 25, 50})->state = eStateGoal;

    // Return goal point
    return (xy_t) {wd.x - 25, 50};
}

#endif
