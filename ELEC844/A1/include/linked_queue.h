/**
 * @file linked_queue.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-10-08
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _LINKED_QUEUE_H_
#define _LINKED_QUEUE_H_
#include "types.h"

struct queued_voxel{
    struct voxel *v;
    struct queued_voxel *next;
    float cost;
};

static inline struct voxel *queue_pop(struct queued_voxel **head){
    if(!head) return NULL;
    if(!*head) return NULL;

    // Get a pointer to the current head
    struct queued_voxel *top = *head;
    // Dereference the head to get the return voxel
    struct voxel *top_voxel = top->v;
    // Change the head to be the next voxel
    *head = top->next;
    // Free the previous head
    free(top);
    // Return the voxel pointer that was in the previous head
    return top_voxel;
}

static inline int queue_push(struct queued_voxel **head, struct voxel *v, float cost){
    if(!head) return -1;
    // Setup the new node
    struct queued_voxel *new_node = malloc(sizeof(struct queued_voxel));
    if(!new_node) return -1;
    new_node->v = v;
    new_node->cost = cost;
    new_node->next = NULL;

    // Case where queue is empty
    if(!*head){
        *head = new_node;
        return 0;
    }

    // Case where insert happens at the head
    struct queued_voxel *qvc = *head;
    if(cost < qvc->cost){
        new_node->next = qvc;
        *head = new_node;
        return 0;
    }

    // Traverse the current queue to find the insert point
    while(qvc != NULL){
        // Handle reaching the end of the queue
        if(qvc->next == NULL){
            qvc->next = new_node;
            return 0;
        }
        // Insert node is less cost than the next in the chain (goes before it)
        if(cost < qvc->next->cost){
            new_node->next = qvc->next;
            qvc->next = new_node;
            return 0;
        }
        qvc = qvc->next;
    }
    // Should insert or fail before this point
    return -9;
}

#endif
