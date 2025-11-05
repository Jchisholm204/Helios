/**
 * @file kdtree.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-05
 * @modified Last Modified: 2025-11-05
 *
 * @copyright Copyright (c) 2025
 */

#include "kdtree.h"

#include <malloc.h>
#include <memory.h>

kdtree_t* kdtree_init(void) {
    kdtree_t* t = malloc(sizeof(kdtree_t));
    if (!t)
        return NULL;
    t->head = NULL;
    return t;
}

void kdtree_add(kdtree_t* tree, float x, float y) {
    if(!tree) return;
    // If head is null
    if(!tree->head){
        tree->head = malloc(sizeof(struct kdnode));
        struct kdnode *h = tree->head;
        h->cd = 0;
        h->less = NULL;
        h->more = NULL;
        h->x = x;
        h->y = y;
        return;
    }
    struct kdnode *current = tree->head;
    struct kdnode *prev = NULL;
    while(current){
        prev = NULL;
    }
}

