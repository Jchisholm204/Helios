/**
 * @file ll_t.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief Linked List Priority Queue
 * @version 0.1
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-10-08
 *
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _LINKED_LIST_H_
#define _LINKED_LIST_H_
#include <malloc.h>
#include <memory.h>
#include <stddef.h>

typedef struct _linked_list {
    void* data;
    size_t len;
    struct _linked_list* next;
} ll_t;

/**
 * @brief Pop the lowest cost item from the ll
 *
 * @param head pointer to the head of the ll
 * @return popped voxel
 */
static inline void* ll_pop(ll_t** head) {
    if (!head)
        return NULL;
    if (!*head)
        return NULL;

    // Get a pointer to the current head
    ll_t* top = *head;
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
 * @brief Push a voxel into the priority ll
 *
 * @param head pointer to the ll handle (head of the queue)
 * @param v address to link into the ll (queue does not handle memory)
 * @return 0 on success
 */
static inline int ll_push(ll_t** head, void* v) {
    if (!head)
        return -1;
    // Setup the new node
    ll_t* new_node = malloc(sizeof(ll_t));
    if (!new_node)
        return -1;
    new_node->data = v;
    new_node->next = NULL;

    // Case where ll is empty
    if (!*head) {
        *head = new_node;
        new_node->len = 1;
        return 0;
    }

    // Case where insert happens at the head
    ll_t* qvc = *head;
    new_node->next = qvc;
    new_node->len = qvc->len + 1;
    *head = new_node;
    return 0;
}

/**
 * @brief Return the length of the ll from the head node
 *
 * @param head 
 * @return 
 */
static inline size_t ll_len(ll_t* head) {
    if (!head)
        return 0;
    return head->len;
}

/**
 * @brief Return the length of the ll through traversal
 *
 * @param head 
 * @return 
 */
static inline size_t ll_length(ll_t* head) {
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
 * @brief Find and remove an element from the ll
 *
 * @param head
 * @param v voxel to try and remove
 * @return 1 on removal, 0 if voxel is not in the ll
 */
static inline int ll_find_remove(ll_t** head, void* v) {
    if (!head)
        return -1;
    if (!*head)
        return -1;
    ll_t* qv = *head;
    // Check if the top node is the node to remove
    if (qv->data == v) {
        *head = (*head)->next;
        free(qv);
        return 1;
    }

    for (; qv && qv->next; qv = qv->next) {
        if (qv->next->data == v) {
            ll_t* t = qv->next;
            qv->next = qv->next->next;
            free(t);
            return 1;
        }
    }

    return 0;
}

#endif
