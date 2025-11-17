/**
 * @file dag.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-07
 * @modified Last Modified: 2025-11-07
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _DAG_H_
#define _DAG_H_
#include <stddef.h>
#include "linked_list.h"

typedef struct _dag_node {
    ll_t* parents;
    ll_t* children;
    void *pv_data;
} dag_node_t;

typedef struct _dag {
    size_t n_nodes;
} dag_t;

extern dag_t * dag_init(void);

#endif
