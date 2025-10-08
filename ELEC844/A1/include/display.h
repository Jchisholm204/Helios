/**
 * @file display.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-10-08
 *
 * @copyright Copyright (c) 2025
 */


#ifndef _DISPLAY_H_
#define _DISPLAY_H_
#include <SDL.h>
#include <SDL2/SDL.h>
#include <SDL_ttf.h>

#include "types.h"

// Display Default ms per item in path
#define DISPD_MS_ITEM 100

typedef struct {
    SDL_Window *sdl_win;
    SDL_Renderer *sdl_ren;
    TTF_Font *sdl_font;
    int ms_per_item;
    struct grid *grid;
} disp_t;

disp_t *disp_init(size_t n_cols, size_t n_rows);

void disp_setSpeed(disp_t *pDisplay, int ms_per_item);

void disp_drawGrid(disp_t *pDisplay);

void disp_drawPath(disp_t* pDisplay, struct path *pPath);

void disp_render(disp_t *pDisplay);

void disp_exit(disp_t **pDisplay);

#endif
