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
#include "types.h"

#include <SDL.h>
#include <SDL2/SDL.h>
#include <SDL_ttf.h>

typedef struct {
    SDL_Window* sdl_win;
    SDL_Renderer* sdl_ren;
    TTF_Font* sdl_font;
    struct grid* grid;
} disp_t;

/**
 * @brief Initialize the Display (Voxel Grid Viewer)
 *
 * @param n_cols Number of rows in the voxel grid
 * @param n_rows Number of columns in the voxel grid
 * @returns a pointer to the display interface
 */
disp_t* disp_init(size_t n_cols, size_t n_rows);

/**
 * @brief Clear the current display rendering
 *  Must call before draw functions
 *
 * @param pDisplay 
 */
void disp_clr(disp_t* pDisplay);

/**
 * @brief Draws the voxel grid onto the display renderer
 *
 * @param pDisplay 
 */
void disp_drawGrid(disp_t* pDisplay);

/**
 * @brief Draw a path (red arrows) on top of the voxel grid
 *
 * @param pDisplay 
 * @param pPath 
 */
void disp_drawPath(disp_t* pDisplay, struct path* pPath);

/**
 * @brief Render the image to the display
 *
 * @param pDisplay 
 */
void disp_render(disp_t* pDisplay);

/**
 * @brief Cleanup the display interface.
 *  Must call on exit
 *
 * @param pDisplay 
 */
void disp_exit(disp_t** pDisplay);

#endif
