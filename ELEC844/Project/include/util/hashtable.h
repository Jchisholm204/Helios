/**
 * @file hashtable.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-15
 * @modified Last Modified: 2025-12-15
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _HASHTABLE_H_
#define _HASHTABLE_H_

#include "statespace/statespace.h"

#include <stddef.h>

typedef struct {
    size_t n_elements;
    size_t n_size;
    vstate_t *data;
} hashtable_t;

static inline hashtable_t *hashtable_init(size_t n_elements) {
    (void) n_elements;
    return NULL;
}

static inline void hashtable_free(hashtable_t **pTable) {
    (void) pTable;
}

static inline int hashtable_insert(hashtable_t *table, vstate_t *ws) {
    (void) table;
    (void) ws;
    return -1;
}

static inline float hashtable_find(hashtable_t *table, state_t *s){
    (void) table;
    (void) s;
    return -1;
}

#endif
