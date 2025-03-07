/**
 * @file main.hpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date 2025-03-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <inttypes.h>

typedef float* mat_t;

#define MAT_SIZE(n) (sizeof(float)*n*n)
#define MAT(n, row, col) (n*row + col)

#define FREE(p) if(!p){ free(p); p = NULL; }

#ifndef __global__
#define __global__
#endif

typedef __global__ void (*MatMul)(mat_t, mat_t, mat_t, size_t);

/**
 * @brief Lukes Main Code
 * 
 */
extern void lukesCode(void);

extern void run_test_single(MatMul kernel, size_t mat_n);
