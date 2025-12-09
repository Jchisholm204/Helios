/**
 * @file statespace.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date Created: 2025-11-19
 * @modified Last Modified: 2025-11-19
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _STATESPACE_H_
#define _STATESPACE_H_

#define N_DIMENSIONS 2
#define DIM_MIN 0.0
#define DIM_MAX 1.0


typedef double flt;

typedef flt state_t[N_DIMENSIONS];

// Select a random sample
#define RAND_SAMPLE(min, max) (((flt)rand()/(flt)RAND_MAX)*((max)-(min)) - (min))

#endif
