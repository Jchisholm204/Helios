/**
 * @file statespace.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief Statespace Definitions
 * @version 0.1
 * @date Created: 2025-12-09
 * @modified Last Modified: 2025-12-09
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _STATESPACE_H_
#define _STATESPACE_H_

#define STATESPACE_DIMS 2
#define STATESPACE_MIN 0
#define STATESPACE_MAX 100

// Offset of start and target points from the edge
#define STATESPACE_ST_OFFSET 8

typedef unsigned char state_t[STATESPACE_DIMS];

#endif

