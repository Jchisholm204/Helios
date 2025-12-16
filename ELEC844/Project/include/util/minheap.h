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
#include "statespace/statespace_ops.h"

#include <malloc.h>
#include <stddef.h>

#define HEAP_LOGGING

typedef struct {
    size_t n_elements;
    size_t n_size;
    wstate_t *data;
    // Logger Stats
#ifdef HEAP_LOGGING
    size_t n_insertions;
    size_t n_deletions;
    size_t n_resizes;
#endif
} min_heap_t;

static inline min_heap_t *mheap_init(size_t n_size) {
    min_heap_t *heap = (min_heap_t *) malloc(sizeof(min_heap_t));
    if (!heap) {
        return NULL;
    }
    // Setup internal structure
    heap->n_size = n_size;
    heap->n_elements = 0;
    heap->data = malloc(sizeof(wstate_t) * n_size);
    if (!heap->data) {
        free(heap);
        return NULL;
    }
    // Set all unallocated nodes to have the unused signature
    for (size_t i = 0; i < n_size; i++) {
        for (size_t j = 0; j < STATESPACE_DIMS; j++)
            heap->data[i].state[j] = STATESPACE_FLAG | ~(STATESPACE_MASK);
        heap->data[i].weight = -1;
    }

    // Logging Setup
#ifdef HEAP_LOGGING
    heap->n_insertions = 0;
    heap->n_deletions = 0;
    heap->n_resizes = 0;
#endif

    return heap;
}

static inline void mheap_free(min_heap_t **pHeap) {
    if(*pHeap){
        min_heap_t *heap = *pHeap;
        if(heap){
            if(heap->data){
                free(heap->data);
            }
            free(heap);
        }
        *pHeap = NULL;
    }
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
