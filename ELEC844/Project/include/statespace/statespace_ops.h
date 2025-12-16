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
#include "statespace/statespace.h"

#include <memory.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

static inline bool state_eq(state_t a, state_t b) {
    uint8_t neq = 0x00;
    for (size_t i = 0; i < STATESPACE_DIMS; i++) {
        neq |= a[i] ^ b[i];
    }
    return neq == 0x0;
}

static inline void state_cpy(state_t *dst, const state_t * src) {
    (void) memcpy(dst, src, sizeof(state_t));
}

static inline void wstate_cpy(wstate_t *dst, const wstate_t * src) {
    (void) memcpy(dst, src, sizeof(wstate_t));
}

#endif
