/**
 * @file statespace_ops.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-15
 * @modified Last Modified: 2025-12-15
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _STATESPACE_OPS_H_
#define _STATESPACE_OPS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "statespace/statespace.h"

#include <memory.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


static inline bool state_eq(const state_t a, const state_t b) {
    uint8_t neq = 0x00;
    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        neq |= a[i] ^ b[i];
    }
    return neq == 0x0;
}

static inline void state_cpy(state_t *dst, const state_t *src) {
    (void) memcpy(dst, src, sizeof(state_t));
}

static inline void wstate_cpy(wstate_t *dst, const wstate_t *src) {
    (void) memcpy(dst, src, sizeof(wstate_t));
}

static inline uint64_t state_index(const state_t state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        hash |= (uint64_t) ((uint64_t) (state[i] & STATESPACE_MASK) << 7 * i);
    }
    hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ULL;
    hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebULL;
    hash = hash ^ (hash >> 31);
    return hash;
}

#ifdef __cplusplus
}
#endif
#endif
