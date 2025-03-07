/**
 * @file kernel_add.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date 2025-03-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef _KERNEL_ADD_H_
#define _KERNEL_ADD_H_

#include <stdio.h>
#include "cuda_runtime.h"

extern __global__ void __noinline__ cudaAddKernel(float *A, float *B, float *C, size_t n);

extern void cudaAdd(void);

#endif
