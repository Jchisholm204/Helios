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

typedef float* mat_t;

#define MAT_SIZE(n) (sizeof(float)*n*n)
#define MAT(n, row, col) (n*row + col)

#define FREE(p) if(!p){ free(p); p = NULL; }

/**
 * @brief Get the Dev Properties object
 * MP1 - Part 1
 */
extern void getDevProperties(void);

/**
 * @brief Lukes Main Code
 * 
 */
extern void lukesCode(void);
