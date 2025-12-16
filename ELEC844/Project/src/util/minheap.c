/**
 * @file minheap.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-16
 * @modified Last Modified: 2025-12-16
 *
 * @copyright Copyright (c) 2025
 */

#include "util/minheap.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

void _min_heap_test(void) {
    min_heap_t *h = mheap_init(100);
    wstate_t s;
    for (int i = 0; i < 100000; i++) {
        s.weight = (float) (rand() % 5000);
        mheap_push(h, &s);
    }

    float last = 0;
    for (int i = 0; i < 100000; i++) {
        s.weight = 0;
        int r = mheap_pop(h, &s);
        if (last > s.weight) {
            fprintf(stderr, "Heap Error\n");
        }
        last = s.weight;
        printf("Heap RetVal = %2.2f (%d)\n", s.weight, r);
    }

    for (int i = 0; i < 100000; i++) {
        s.weight = (float) (rand() % 5000);
        mheap_push(h, &s);
    }
    for (int i = 0; i < 100000; i++) {
        s.weight = 0;
        int r = mheap_pop(h, &s);
        if (last > s.weight) {
            printf("Heap Error\n");
        }
        last = s.weight;
        printf("Heap RetVal = %2.2f (%d)\n", s.weight, r);
    }
    mheap_free(&h);
}
