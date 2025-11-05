/**
 * @file kdtree.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-05
 * @modified Last Modified: 2025-11-05
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _KDTREE_H_
#define _KDTREE_H_

struct kdnode {
    int cd;
    float x, y;
    struct kdnode *less, *more;
};

typedef struct {
    struct kdnode* head;
} kdtree_t;

kdtree_t* kdtree_init(void);

void kdtree_add(kdtree_t* tree, float x, float y);

#endif
