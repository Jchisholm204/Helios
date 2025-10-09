/**
 * @file linked_queue.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief Linked List Priority Queue
 * @version 0.1
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-10-08
 *
 * This is the simplest way I could think to make a priority queue without
 * memory reallocating. This queue should not hold onto any voxel memory.
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _LINKED_QUEUE_H_
#define _LINKED_QUEUE_H_
#include "types.h"

struct queued_voxel {
    struct voxel* v;
    struct queued_voxel* next;
    float cost;
    float cost_s;
};

/**
 * @brief Pop the lowest cost item from the queue
 *
 * @param head pointer to the head of the queue
 * @return popped voxel
 */
static inline struct voxel* queue_pop(struct queued_voxel** head) {
    if (!head)
        return NULL;
    if (!*head)
        return NULL;

    // Get a pointer to the current head
    struct queued_voxel* top = *head;
    // Dereference the head to get the return voxel
    struct voxel* top_voxel = top->v;
    // Change the head to be the next voxel
    *head = top->next;
    // Free the previous head
    free(top);
    // Return the voxel pointer that was in the previous head
    return top_voxel;
}

/**
 * @brief Push a voxel into the priority queue
 *
 * @param head pointer to the queue handle (head of the queue)
 * @param v pointer to the voxel to push into the queue
 * @param cost queue cost of the voxel (lower cost popped first)
 * @param cost_s secondary queue cost
 * @return 0 on success
 */
static inline int queue_push(struct queued_voxel** head, struct voxel* v,
                             float cost, float cost_s) {
    if (!head)
        return -1;
    // Setup the new node
    struct queued_voxel* new_node = malloc(sizeof(struct queued_voxel));
    if (!new_node)
        return -1;
    new_node->v = v;
    new_node->cost = cost;
    new_node->cost_s = cost_s;
    new_node->next = NULL;

    // Case where queue is empty
    if (!*head) {
        *head = new_node;
        return 0;
    }

    // Case where insert happens at the head
    struct queued_voxel* qvc = *head;
    if (cost < qvc->cost || (cost == qvc->cost && cost_s < qvc->cost_s)) {
        new_node->next = qvc;
        *head = new_node;
        return 0;
    }

    // Traverse the current queue to find the insert point
    while (qvc != NULL) {
        // Handle reaching the end of the queue
        if (qvc->next == NULL) {
            qvc->next = new_node;
            return 0;
        }
        // Insert node is less cost than the next in the chain (goes before it)
        if (cost < qvc->next->cost || (cost == qvc->cost && cost_s < qvc->next->cost_s)) {
            new_node->next = qvc->next;
            qvc->next = new_node;
            return 0;
        }
        qvc = qvc->next;
    }
    // Should insert or fail before this point
    return -9;
}

static inline size_t queue_length(struct queued_voxel* head) {
    if (!head)
        return 0;
    if (!head->next)
        return 1;
    size_t size = 0;
    for (; head; head = head->next)
        size++;
    return size;
}

/**
 * @brief Find and remove an element from the queue
 *
 * @param head 
 * @param v voxel to try and remove 
 * @return 1 on removal, 0 if voxel is not in the queue
 */
static inline int queue_find_remove(struct queued_voxel **head, struct voxel *v){
    if(!head) return -1;
    if(!*head) return -1;
    struct queued_voxel *qv = *head;
    // Check if the top node is the node to remove
    if(qv->v == v){
        *head = (*head)->next;
        free(qv);
        return 1;
    }
    
    for(; qv && qv->next; qv = qv->next){
        if(qv->next->v == v){
            struct queued_voxel *t = qv->next;
            qv->next = qv->next->next;
            free(t);
            return 1;
        }
    }

    return 0;
}

#endif
