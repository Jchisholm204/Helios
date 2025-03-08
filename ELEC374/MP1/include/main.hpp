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
#include <inttypes.h>
#include <vector>

typedef float* mat_t;

#define MAT_SIZE(n) (sizeof(float)*n*n)
#define MAT(n, row, col) (n*row + col)

#define FREE(p) if(!p){ free(p); p = NULL; }

#ifndef __global__
#define __global__
#endif

#ifndef eKernel
enum eKernel;
#endif

typedef struct testParams {
    enum eKernel kernel;
    int dim_grid;
    int dim_block;
    size_t mat_n;
    float t_mem_alloc;
    float t_mem_host_to_device;
    float t_mem_device_to_host;
    float t_cpu_compute;
    float t_gpu_compute;
    bool test_success;
} testParams_t;

/**
 * @brief Lukes Main Code
 * 
 */
extern void lukesCode(void);

extern void run_test_single(testParams_t &params);

extern void print_test(testParams_t &result, int tn = -1);
