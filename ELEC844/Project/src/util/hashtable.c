/**
 * @file hashtable.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-16
 * @modified Last Modified: 2025-12-16
 *
 * @copyright Copyright (c) 2025
 */

#include "util/hashtable.h"

#include "statespace/statespace_ops.h"

#include <assert.h>
#include <float.h>
#include <malloc.h>

inline hashtable_t *hashtable_init(size_t n_elements) {
    // n_elements must be power of two
    assert((n_elements & (n_elements - 1)) == 0);
    hashtable_t *table = (hashtable_t *) malloc(sizeof(hashtable_t));
    if (!table) {
        return NULL;
    }
    table->data = (vstate_t *) malloc(sizeof(vstate_t) * n_elements);

    for (size_t i = 0; i < n_elements; i++) {
        table->data[i] = (vstate_t) {{STATESPACE_FLAG}, {STATESPACE_FLAG}, 0};
    }

    table->n_size = n_elements;
    table->n_elements = 0;

#ifdef HASHTABLE_LOGGING
    table->metrics.n_insertions = 0;
    table->metrics.n_lookups = 0;
#endif
    return table;
}

inline void hashtable_free(hashtable_t **pTable) {
    if (pTable) {
        hashtable_t *table = *pTable;
        if (table) {
#ifdef HASHTABLE_LOGGING
#if ((int) HASHTABLE_LOGGING) == 1
            printf("HashTable Metrics:\n");
            printf("\tInsertions: %ld\n", table->metrics.n_insertions);
            printf("\tLookups: %ld\n", table->metrics.n_lookups);
#endif
#endif
            if (table->data)
                free(table->data);
            free(table);
        }
        *pTable = NULL;
    }
}

inline int hashtable_insert(hashtable_t *table, const vstate_t *ws,
                            uint64_t hash) {
    if (!table || !ws)
        return -1;

#ifdef HASHTABLE_LOGGING
    table->metrics.n_insertions++;
#endif

    size_t mask = table->n_size - 1;
    size_t idx = hash & mask;

    // Linear probing loop
    for (size_t i = 0; i < table->n_size; i++) {
        size_t current_idx = (idx + i) & mask; // This handles the wrap-around
        vstate_t *entry = &table->data[current_idx];

        // 1. Found an Empty Slot (Success)
        if (STATESPACE_GETFLAG(entry->state[0])) {
            *entry = *ws;
            return 0;
        }

        // 2. Found an Existing Match (A* Update Logic)
        // We must check if the coordinates are identical
        if (state_eq(entry->state, ws->state)) {
            if (ws->weight < (entry->weight - 0.0001f)) {
                entry->weight = ws->weight; // Found a better path!
                return 1;                   // Signal for a "re-push" to heap
            }
            return 2; // Signal "already visited, do nothing"
        }
#ifdef HASHTABLE_LOGGING
        table->metrics.n_lookups++;
#endif
    }

    return -2; // Table is 100% full (should be avoided by resizing)
}

inline vstate_t *hashtable_find(hashtable_t *table, const state_t *s,
                                uint64_t index) {
    size_t mask = table->n_size - 1;
    size_t idx = index & mask;

    for (size_t i = 0; i < table->n_size; i++) {
        size_t current_idx = (idx + i) & mask;
        vstate_t *entry = &table->data[current_idx];

        // 1. Found an empty slot: Reserve it and return it
        if (STATESPACE_GETFLAG(entry->state[0])) {
            STATESPACE_SETFLAG(entry->state[0]); // Mark as occupied
            state_cpy(&entry->state, s);
            entry->weight = FLT_MAX; // Initialize with max weight
            return entry;
        }

        // 2. Found existing match: Return it for weight comparison
        if (state_eq(entry->state, *s)) {
            return entry;
        }
    }
    return NULL;
}
