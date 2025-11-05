/**
 * @file worlds.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-04
 * @modified Last Modified: 2025-11-04
 *
 * @copyright Copyright (c) 2025
 * Definition for World Loader:
 *  - Takes one input (the spacial world)
 *  - Gives one output (world_info struct)
 *  - Must return the world_info on all conditions
 */

#ifndef _WORLDS_H_
#define _WORLDS_H_
#include "spacial.h"

#define BLOCK(s_x, s_y, e_x, e_y) \
    for(int x = s_x; x < e_x; x++){ \
        for(int y = s_y; y < e_y; y++) \
        spacial_invalidate(s, (struct xy){x, y}); \
    } \


static struct world_info gen_world1A(struct spacial* s) {
    struct world_info wi = {
        .start = {25, 50},
        .target = {75, 50},
        .dim = {100, 100}
    };

    if (!s)
        // Return goal point on null
        return wi;
    // Place main obstacle
    for (int x = 45; x < 55; x++) {
        for (int y = 25; y < 75; y++) {
            spacial_invalidate(s, (struct xy) {x, y});
        }
    }
    // Place start point
    spacial_getV(s, (xy_t) {25, 50})->state = eStateSource;
    spacial_getV(s, (xy_t) {75, 50})->state = eStateGoal;

    // Return goal point
    return wi;
}

static struct world_info gen_world1B(struct spacial* s) {
    struct world_info wi = {
        .start = {25, 50},
        .target = {75, 50},
        .dim = {100, 100}
    };
    if (!s)
        // Return goal point on null
        return wi;
    for (int x = 45; x < 55; x++) {
        for (int y = 4; y < 96; y++) {
            spacial_invalidate(s, (struct xy) {x, y});
        }
    }
    // Place start point
    spacial_getV(s, (xy_t) {25, 50})->state = eStateSource;
    spacial_getV(s, (xy_t) {75, 50})->state = eStateGoal;

    // Return goal point
    return wi;
}

static struct world_info gen_world2A(struct spacial *s){
    struct world_info wi = {
        .start = {25, 50},
        .target = {75, 50},
        .dim = {100, 100}
    };
    if (!s)
        // Return goal point on null
        return wi;
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
    spacial_getV(s, (xy_t) {25, 50})->state = eStateSource;
    spacial_getV(s, (xy_t) {75, 50})->state = eStateGoal;

    // Return goal point
    return wi;
}

static struct world_info gen_world2B(struct spacial *s){
    struct world_info wi = {
        .start = {25, 50},
        .target = {75, 50},
        .dim = {100, 100}
    };
    if (!s)
        // Return goal point on null
        return wi;

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
    spacial_getV(s, (xy_t) {25, 50})->state = eStateSource;
    spacial_getV(s, (xy_t) {75, 50})->state = eStateGoal;

    // Return goal point
    return wi;
}

static struct world_info gen_world2C(struct spacial *s){
    struct world_info wi = {
        .start = {25, 50},
        .target = {75, 50},
        .dim = {100, 100}
    };
    if (!s)
        // Return goal point on null
        return wi;
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
    spacial_getV(s, (xy_t) {25, 50})->state = eStateSource;
    spacial_getV(s, (xy_t) {75, 50})->state = eStateGoal;

    // Return goal point
    return wi;
}

#endif
