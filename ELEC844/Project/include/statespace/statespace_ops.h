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
#include "statespace.h"

#include <stdbool.h>

static inline bool state_eq(state_t a, state_t b) {
    (void) a;
    (void) b;
    return false;
}

#endif
