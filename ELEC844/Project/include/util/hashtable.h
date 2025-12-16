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

#ifdef __cplusplus
extern "C" {
#endif

#include "statespace/statespace.h"

#include <stddef.h>
#include <stdint.h>

#define HASHTABLE_LOGGING

typedef struct {
    size_t n_elements;
    size_t n_size;
    vstate_t *data;
#ifdef HASHTABLE_LOGGING
    struct {
        size_t n_insertions;
        size_t n_lookups;
    } metrics;
#endif
} hashtable_t;

extern hashtable_t *hashtable_init(size_t n_elements);

extern void hashtable_free(hashtable_t **pTable);

extern int hashtable_insert(hashtable_t *table, const vstate_t *ws,
                            uint64_t index);

extern vstate_t *hashtable_find(hashtable_t *table, const state_t *s,
                                uint64_t index);

#ifdef __cplusplus
}
#endif
#endif
