/**
 * @file kMatMul.hpp
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
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef mat_t
typedef float* mat_t;
#endif

#ifndef MAT_SIZE
#define MAT_SIZE(n) (sizeof(float)*n*n)
#endif
#ifndef MAT
#define MAT(n, row, col) (n*row + col)
#endif

#ifndef FREE
#define FREE(p) if(!p){ free(p); p = NULL; }
#endif

enum eKernel {
    eKernel_ssm,
    eKernel_msm,
    eKernel_tiled
};

/**
 * @brief Single Stream Matrix Multiplication
 *
 * @param P Result Matrix (M*N)
 * @param M Matrix M
 * @param N Matrix N
 * @param size Size of Square Matricies
 */
__global__ void kMatMul_ssm(mat_t P, mat_t M, mat_t N, size_t size);


/**
 * @brief Multiple Stream Matrix Multiplication Kernel
 *
 * @param P Result Matrix (M*N)
 * @param M Matrix M
 * @param N Matrix N
 * @param size Size of Square Matricies
 */
__global__ void kMatMul_msm(mat_t P, mat_t M, mat_t N, size_t size);

/**
 * @brief Single Stream Matrix Multiplication
 *
 * @param P Result Matrix (M*N)
 * @param M Matrix M
 * @param N Matrix N
 * @param size Size of Square Matricies
 */
__global__ void kMatMul_tiled(mat_t P, mat_t M, mat_t N, size_t Width);

