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
#ifdef __cplusplus
extern "C" {
#endif

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
    struct {
        size_t n_insertions;
        size_t n_deletions;
        size_t n_insert_swaps;
        size_t n_del_swaps;
        size_t n_resizes;
    } metrics;
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
    heap->data = (wstate_t *) malloc(sizeof(wstate_t) * n_size);
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
    heap->metrics.n_insertions = 0;
    heap->metrics.n_deletions = 0;
    heap->metrics.n_resizes = 0;
    heap->metrics.n_insert_swaps = 0;
    heap->metrics.n_del_swaps = 0;
#endif

    return heap;
}

static inline void mheap_free(min_heap_t **pHeap) {
    if (*pHeap) {
        min_heap_t *heap = *pHeap;
        if (heap) {
#ifdef HEAP_LOGGING
            printf("Heap Metrics:\n");
            printf("\tInsertions: %ld\n", heap->metrics.n_insertions);
            printf("\tDeletions: %ld\n", heap->metrics.n_deletions);
            printf("\tIns Swaps: %ld\n", heap->metrics.n_insert_swaps);
            printf("\tDel Swaps: %ld\n", heap->metrics.n_del_swaps);
            printf("\tResizes: %ld\n", heap->metrics.n_resizes);
#endif
            if (heap->data) {
                free(heap->data);
            }
            free(heap);
        }
        *pHeap = NULL;
    }
}

static inline void mheap_push(min_heap_t *heap, wstate_t *wstate) {
    if (!heap || !wstate) {
        return;
    }
#ifdef HEAP_LOGGING
    heap->metrics.n_insertions++;
#endif
    if (heap->n_elements == heap->n_size) {
        heap->n_size *= 2;
        heap->data =
            (wstate_t *) realloc(heap->data, sizeof(wstate_t) * heap->n_size);
#ifdef HEAP_LOGGING
        heap->metrics.n_resizes++;
#endif
    }

    size_t i = heap->n_elements++;
    wstate_cpy(&heap->data[i], wstate);

    while (i > 0) {
        size_t p = (i - 1) >> 2;
        if (heap->data[i].weight >= heap->data[p].weight) {
            break;
        }
        wstate_t tmp;
        wstate_cpy(&tmp, &heap->data[i]);
        wstate_cpy(&heap->data[i], &heap->data[p]);
        wstate_cpy(&heap->data[p], &tmp);
#ifdef HEAP_LOGGING
        heap->metrics.n_insert_swaps++;
#endif
        i = p;
    }
}

static inline int mheap_pop(min_heap_t *heap, wstate_t *wstate) {
    if (!heap || !wstate) {
        return -1;
    }
#ifdef HEAP_LOGGING
    heap->metrics.n_deletions++;
#endif
    if (heap->n_elements == 0) {
        return 1;
    }

    // Return the top level element
    wstate_cpy(wstate, &heap->data[0]);

    wstate_cpy(&heap->data[0], &heap->data[--heap->n_elements]);

    // Rebalance the tree
    size_t i = 0;
    for (;;) {
        size_t child_base = (i << 2) + 1;
        if (child_base >= heap->n_elements)
            break;
        size_t smallest = child_base;
        for (size_t j = 1; j < 4; j++) {
            if (child_base + j < heap->n_elements &&
                heap->data[child_base + j].weight <
                    heap->data[smallest].weight) {
                smallest = child_base + j;
            }
        }
        if (heap->data[smallest].weight < heap->data[i].weight) {
            wstate_t tmp;
            wstate_cpy(&tmp, &heap->data[i]);
            wstate_cpy(&heap->data[i], &heap->data[smallest]);
            wstate_cpy(&heap->data[smallest], &tmp);
            i = smallest;
#ifdef HEAP_LOGGING
            heap->metrics.n_del_swaps++;
#endif
        } else {
            break;
        }
    }
    return 0;
}

// Test function for validating the min_heap implementation
extern void _min_heap_test(void);

#ifdef __cplusplus
}
#endif
#endif
