/**
 * @file statespace.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief Statespace Definitions
 * @version 0.1
 * @date Created: 2025-12-09
 * @modified Last Modified: 2025-12-09
 *
 *
 * NOTE:
 *  Sizes of all objects declared within this file must be known at compile
 *  time.
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _STATESPACE_H_
#define _STATESPACE_H_

#ifndef STATESPACE_DIMS
#define STATESPACE_DIMS 10
#endif
#define STATESPACE_MIN 0
#define STATESPACE_MAX 100

#define STATESPACE_MASK (0x7F)

#define STATESPACE_FLAG (0x80)
#define STATESPACE_GETFLAG(state) ((state) & STATESPACE_FLAG)
#define STATESPACE_SETFLAG(state) ((state) |= STATESPACE_FLAG)

// Offset of start and target points from the edge
#define STATESPACE_ST_OFFSET 8

// State object to hold a coordinate vector
typedef unsigned char state_t[STATESPACE_DIMS];

// Weighted State object to hold a coordinate vector and a weight
typedef struct {
    state_t state;
    float weight; // = g(n) = cost to come
    float cost; // f(n) = total cost for queue
} wstate_t;

// wstate object with a extra parameter to hold the parent state
typedef struct {
    state_t state, parent;
    float weight;
} vstate_t;

#endif
