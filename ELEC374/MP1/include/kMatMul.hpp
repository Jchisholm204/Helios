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

#include "main.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


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

