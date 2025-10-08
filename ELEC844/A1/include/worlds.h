/**
 * @file worlds.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief World Loader Header
 * @version 0.1
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-10-08
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _WORLDS_H_
#define _WORLDS_H_

#include "types.h"

// Do not change without adjusting world arrays
#define WORLD_X 19
#define WORLD_Y 19

enum eWorlds{
    eWorld1A,
    eWorld1B,
    eWorld2A,
    eWorld2B,
    eWorld2C,
    eWorld_n
};

int world_loader(struct grid *pGrid, enum eWorlds world);

#endif
