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

struct linked_queue {
    void* data;
    float cost;
    struct linked_queue* next;
};

/**
 * @brief Pop the lowest cost item from the queue
 *
 * @param head pointer to the head of the queue
 * @return popped voxel
 */
static inline void* queue_pop(struct linked_queue** head) {
    if (!head)
        return NULL;
    if (!*head)
        return NULL;

    // Get a pointer to the current head
    struct linked_queue* top = *head;
    // Dereference the head to get the return voxel
    void* top_data = top->data;
    // Change the head to be the next voxel
    *head = top->next;
    // Free the previous head
    free(top);
    // Return the voxel pointer that was in the previous head
    return top_data;
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
static inline int queue_push(struct linked_queue** head, void* v, float cost) {
    if (!head)
        return -1;
    // Setup the new node
    struct linked_queue* new_node = malloc(sizeof(struct linked_queue));
    if (!new_node)
        return -1;
    new_node->data = v;
    new_node->cost = cost;
    new_node->next = NULL;

    // Case where queue is empty
    if (!*head) {
        *head = new_node;
        return 0;
    }

    // Case where insert happens at the head
    struct linked_queue* qvc = *head;
    if (cost < qvc->cost) {
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
        if (cost < qvc->next->cost) {
            new_node->next = qvc->next;
            qvc->next = new_node;
            return 0;
        }
        qvc = qvc->next;
    }
    // Should insert or fail before this point
    return -9;
}

static inline size_t queue_length(struct linked_queue* head) {
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
static inline int queue_find_remove(struct linked_queue** head,
                                    void* v) {
    if (!head)
        return -1;
    if (!*head)
        return -1;
    struct linked_queue* qv = *head;
    // Check if the top node is the node to remove
    if (qv->data == v) {
        *head = (*head)->next;
        free(qv);
        return 1;
    }

    for (; qv && qv->next; qv = qv->next) {
        if (qv->next->data == v) {
            struct linked_queue* t = qv->next;
            qv->next = qv->next->next;
            free(t);
            return 1;
        }
    }

    return 0;
}

#endif
