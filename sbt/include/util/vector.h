/**
 * @file vector.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-07
 * @modified Last Modified: 2025-11-07
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _VECTOR_H_
#define _VECTOR_H_
#include <malloc.h>
#include <stddef.h>
#include <memory.h>

#define _VECTOR_DEFAULT_LEN_ 10

// Source - https://stackoverflow.com/q
// Posted by terminus, modified by community. See post 'Timeline' for change
// history Retrieved 2025-11-07, License - CC BY-SA 4.0
#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

typedef struct _vector {
    size_t size, len;
    size_t obj_len;
    void* data;
} vector_t;

static inline vector_t* vector_init(size_t obj_len) {
    vector_t* this = malloc(sizeof(vector_t));
    if (!this)
        return NULL;
    this->data = NULL;
    this->size = 0;
    this->len = 0;
    this->obj_len = obj_len;
    return this;
}

static inline int vector_resize(vector_t *pVec){
    if (!pVec)
        return -1;
    // Create vector branch
    if (unlikely(!pVec->data)) {
        pVec->data = malloc(pVec->obj_len * _VECTOR_DEFAULT_LEN_);
        pVec->size = _VECTOR_DEFAULT_LEN_;
    } 
    // Enlarge vector branch
    else if (unlikely(pVec->len + 1 >= pVec->size)) {
        pVec->size*=2;
        void *data = realloc(pVec->data, pVec->size);
        // Realloc has failed if it returns the original pointer
        if(!data){
            return -1;
        }
        pVec->data = data;
    }
    // Decrease vector size
    else if(unlikely(pVec->len < (pVec->size >> 2))){
        pVec->size = pVec->size >> 1;
        void *data = realloc(pVec->data, pVec->size);
        // If decreasing the vector size fails, we dont really care
        if(data){
            pVec->data = data;
        }
    }
    return 0;
}

static inline int vector_push(vector_t* pVec, void* data) {
    if (!pVec)
        return -1;
    if(!data)
        return -1;
    int retur = 0;
    if(unlikely(retur = vector_resize(pVec))){
        return retur;
    }
    memcpy(&pVec->data[pVec->len], data, pVec->obj_len);
    pVec->len += pVec->obj_len;
    return 0;
}

static inline int vector_pop(vector_t *pVec, void *data){
    if(!data)
        return -1;
    int retr = 0;
    if(unlikely(retr = vector_resize(pVec))){
        return retr;
    }
    pVec->len -= pVec->obj_len;
    memcpy(data, &pVec->data[pVec->len], pVec->obj_len);
    return 0;
}

#endif
