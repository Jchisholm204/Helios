/**
 * @file cpuMatMul.hpp
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

extern int MAT_compare(mat_t A, mat_t B, size_t size, float rng);
extern void MAT_print(mat_t Mat, size_t n);
extern void MAT_fillRand(mat_t Mat, size_t N, int max_i, float div);
extern void MAT_fill(mat_t Mat, size_t N, float val);
extern void cpuMatMul(mat_t P, mat_t M, mat_t N, size_t size);