/**
 * @file statespace_ops.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.2
 * @date Created: 2025-12-15
 * @modified Last Modified: 2025-12-17
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
    return !memcmp(a, b, STATESPACE_DIMS);
}

static inline void state_cpy(state_t *dst, const state_t *src) {
    (void) memcpy(dst, src, sizeof(state_t));
}

static inline void wstate_cpy(wstate_t *dst, const wstate_t *src) {
    (void) memcpy(dst, src, sizeof(wstate_t));
}

static inline uint64_t state_pack(const state_t state) {
    uint64_t hash = 0;
    // Use different load/compare mechanism depending on width
#if STATESPACE_DIMS == 2
    hash = *(uint16_t *) state;
#elif STATESPACE_DIMS == 4
    hash = *(uint32_t *) state;
#elif STATESPACE_DIMS == 6
    hash = *(uint32_t *) state;
    hash |= *(uint16_t *) &state[4];
#elif STATESPACE_DIMS == 8
    *(uint64_t *) (&hash) = *(uint64_t *) state;
#elif STATESPACE_DIMS == 10
    *(uint64_t *) (&hash) = *(uint64_t *) state;
    hash ^= *(uint16_t *) &state[8];
#endif
    return hash;
}

static inline uint64_t state_index(const state_t state) {
    // Use correct width load function
    uint64_t hash = state_pack(state);
    hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ULL;
    hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebULL;
    hash = hash ^ (hash >> 31);
    return hash;
}

#ifdef __cplusplus
}
#endif
#endif
