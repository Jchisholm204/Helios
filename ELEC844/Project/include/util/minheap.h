/**
 * @file minheap.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-15
 * @modified Last Modified: 2025-12-15
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _MIN_HEAP_H_
#define _MIN_HEAP_H_

#include "statespace/statespace.h"

#include <stddef.h>

typedef struct {
    size_t n_elements;
    size_t n_size;
    wstate_t *data;
} min_heap_t;

static inline min_heap_t *mheap_init(size_t n_elements) {
    (void) n_elements;
    return NULL;
}

static inline void mheap_free(min_heap_t **pHeap) {
    (void) pHeap;
}

static inline void mheap_push(min_heap_t *heap, wstate_t *wstate) {
    (void) heap;
    (void) wstate;
}

static inline int mheap_pop(min_heap_t *heap, wstate_t *wstate) {
    (void) heap;
    (void) wstate;
    return -1;
}

#endif
