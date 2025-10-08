/**
 * @file display.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-10-08
 *
 * @copyright Copyright (c) 2025
 */

#include "display.h"
#include <SDL2/SDL_ttf.h>

#define PIXEL_PER_GRID 40
#define IN_PIXELS(x) (PIXEL_PER_GRID*x)
#define PX_BORDER 40
#define ARROW_SIZE 8

#define FONT "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"

disp_t *disp_init(size_t n_cols, size_t n_rows){
    // Create the Display Object
    disp_t *pDisp = malloc(sizeof(disp_t));
    if(!pDisp) 
        goto alloc_failure;

    pDisp->grid = grid_init(n_cols, n_rows);
    if(!pDisp->grid)
        goto sdl_failure;

    pDisp->ms_per_item = DISPD_MS_ITEM;

    if(SDL_Init(SDL_INIT_VIDEO) != 0) 
        goto sdl_failure;
    if(TTF_Init() != 0)
        goto sdl_failure;

    // Create the SDL window
    pDisp->sdl_win = SDL_CreateWindow("844A1", 100, 100, 
            IN_PIXELS(n_cols) + PX_BORDER*2, IN_PIXELS(n_rows)+PX_BORDER*2, SDL_WINDOW_SHOWN);
    if(!pDisp->sdl_win) 
        goto win_failure;

    // Create the SDL renderer
    pDisp->sdl_ren = SDL_CreateRenderer(pDisp->sdl_win, -1, SDL_RENDERER_ACCELERATED);
    if(!pDisp->sdl_ren){
        goto ren_failure;
    }

    pDisp->sdl_font = TTF_OpenFont(FONT, 16);
    if(!pDisp->sdl_font){
        printf("TTF Openfont Error: %s\n", TTF_GetError());
        goto ren_failure;
    }
    return pDisp;

ren_failure:
    SDL_DestroyWindow(pDisp->sdl_win);
win_failure:
    SDL_Quit();
sdl_failure:
    if(pDisp->grid) grid_free(&pDisp->grid);
    free(pDisp);
alloc_failure:
    return NULL;
}

void disp_setSpeed(disp_t *pDisplay, int ms_per_item){
    if(!pDisplay) return;
    pDisplay->ms_per_item = ms_per_item;
}

void draw_text(SDL_Renderer *renderer, TTF_Font *font, const char *text, int x, int y) {
    SDL_Color color = {0, 0, 0, 255};  // white text

    SDL_Surface *surface = TTF_RenderText_Solid(font, text, color);
    if (!surface) {
        return;
    }

    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_FreeSurface(surface);
    if (!texture) {
        fprintf(stderr, "CreateTexture Error: %s\n", SDL_GetError());
        return;
    }

    SDL_Rect dst = {x, y, 0, 0};
    SDL_QueryTexture(texture, NULL, NULL, &dst.w, &dst.h);
    SDL_RenderCopy(renderer, texture, NULL, &dst);
    SDL_DestroyTexture(texture);
}

void draw_arrow(SDL_Renderer *ren, int x1, int y1, int x2, int y2, int size)
{
    // Draw main line
    SDL_RenderDrawLine(ren, x1, y1, x2, y2);

    // Calculate the arrowhead angle
    double angle = atan2(y2 - y1, x2 - x1);
    double arrow_angle = M_PI / 6; // 30 degrees
    double arrow_length = size;

    // Left side of arrowhead
    int x3 = x2 - arrow_length * cos(angle - arrow_angle);
    int y3 = y2 - arrow_length * sin(angle - arrow_angle);
    SDL_RenderDrawLine(ren, x2, y2, x3, y3);

    // Right side of arrowhead
    int x4 = x2 - arrow_length * cos(angle + arrow_angle);
    int y4 = y2 - arrow_length * sin(angle + arrow_angle);
    SDL_RenderDrawLine(ren, x2, y2, x4, y4);
}

void disp_drawGrid(disp_t *pDisplay){
    if(!pDisplay) return;
    if(!pDisplay->grid) return;
    size_t max_x = pDisplay->grid->size.x;
    size_t max_y = pDisplay->grid->size.y;

    // Clear the screen
    SDL_SetRenderDrawColor(pDisplay->sdl_ren, 255, 255, 255, 255);
    SDL_RenderClear(pDisplay->sdl_ren);


    // Draw the grid on the screen (x)
    SDL_SetRenderDrawColor(pDisplay->sdl_ren, 0, 0, 0, 255);
    for(int i = 0; i < max_x; i++){
        SDL_RenderDrawLine(pDisplay->sdl_ren, 
                IN_PIXELS(i)+PX_BORDER, PX_BORDER, IN_PIXELS(i)+PX_BORDER, IN_PIXELS(max_y)+PX_BORDER);
        char text[20];
        snprintf(text, 20, "%d", i);
        draw_text(pDisplay->sdl_ren, pDisplay->sdl_font, text, IN_PIXELS(i)+PX_BORDER+PIXEL_PER_GRID/3, PX_BORDER/2);
    }
    // Draw the final line
    SDL_RenderDrawLine(pDisplay->sdl_ren, 
            IN_PIXELS(max_x)+PX_BORDER, PX_BORDER, IN_PIXELS(max_x)+PX_BORDER, IN_PIXELS(max_y)+PX_BORDER);

    // Draw the grid on the screen (y)
    for(int i = 0; i < max_y; i++){
        SDL_RenderDrawLine(pDisplay->sdl_ren, 
                PX_BORDER, IN_PIXELS(i)+PX_BORDER, IN_PIXELS(max_x)+PX_BORDER, IN_PIXELS(i)+PX_BORDER);
        char text[20];
        snprintf(text, 20, "%d", i);
        draw_text(pDisplay->sdl_ren, pDisplay->sdl_font, text,PX_BORDER/3, IN_PIXELS(i)+PX_BORDER+PIXEL_PER_GRID/3);
    }
    // Draw the final line
    SDL_RenderDrawLine(pDisplay->sdl_ren, 
            PX_BORDER, IN_PIXELS(max_y)+PX_BORDER, IN_PIXELS(max_x)+PX_BORDER, IN_PIXELS(max_y)+PX_BORDER);

    // Render all voxels on screen
    for(size_t x = 0; x < max_x; x++){
        for(size_t y = 0; y < max_y; y++){
            struct voxel *v = grid_index(pDisplay->grid, x, y);
            if(!v) continue;
            SDL_Rect r = {IN_PIXELS(x)+PX_BORDER+2, IN_PIXELS(y)+PX_BORDER+2, (PIXEL_PER_GRID-3), (PIXEL_PER_GRID-3)};
            switch(v->state){
                case eStateExplored:
                    SDL_SetRenderDrawColor(pDisplay->sdl_ren, 210, 210, 210, 255);
                    SDL_RenderFillRect(pDisplay->sdl_ren, &r);
                    break;
                case eStateFrontier:
                    SDL_SetRenderDrawColor(pDisplay->sdl_ren, 100, 100, 100, 255);
                    SDL_RenderFillRect(pDisplay->sdl_ren, &r);
                    break;
                case eStateBlocked:
                    SDL_SetRenderDrawColor(pDisplay->sdl_ren, 0, 0, 0, 255);
                    SDL_RenderFillRect(pDisplay->sdl_ren, &r);
                    break;
                case eStateSource:
                    // Render the source
                    draw_text(pDisplay->sdl_ren, pDisplay->sdl_font, "S", 
                            IN_PIXELS(x)+PX_BORDER+PIXEL_PER_GRID/3, IN_PIXELS(y)+PX_BORDER+PIXEL_PER_GRID/3);
                    break;
                case eStateGoal:
                    // Render the source
                    draw_text(pDisplay->sdl_ren, pDisplay->sdl_font, "G", 
                            IN_PIXELS(x)+PX_BORDER+PIXEL_PER_GRID/3, IN_PIXELS(y)+PX_BORDER+PIXEL_PER_GRID/3);
                    break;
                default:
                    break;
            }
        }
    }

    // Call disp_render to render the frame
}

void disp_drawPath(disp_t* pDisplay, struct path *pPath){
    if(!pDisplay) return;
    if(!pDisplay->grid) return;
    if(!pPath) return;
    if(!pPath->voxels) return;
    size_t max_x = pDisplay->grid->size.x;
    size_t max_y = pDisplay->grid->size.y;

    // Draw the Path
    SDL_SetRenderDrawColor(pDisplay->sdl_ren, 255, 0, 0, 255);
    for(size_t i = 0; i < pPath->n_voxels-1; i++){
        struct voxel *v = pPath->voxels[i];
        struct voxel *vn = pPath->voxels[i+1];
        if(!v) continue;
        if(!vn) continue;
        int sx = 0, sy = 0, gx = 0, gy = 0;
        sx = v->x;
        sy = v->y;
        gx = vn->x;
        gy = vn->y;
        printf("Drawing Path: (%d, %d) -> (%d, %d)\n", sx, sy, gx, gy);
        // SDL_RenderDrawLine(pDisplay->sdl_ren, 
        draw_arrow(pDisplay->sdl_ren, 
                IN_PIXELS(sx)+PX_BORDER+PIXEL_PER_GRID/2, IN_PIXELS(sy)+PX_BORDER+PIXEL_PER_GRID/2, 
                IN_PIXELS(gx)+PX_BORDER+PIXEL_PER_GRID/2, IN_PIXELS(gy)+PX_BORDER+PIXEL_PER_GRID/2,
                ARROW_SIZE);

    }

    // Call disp_render to render the frame
}

void disp_render(disp_t *pDisplay){
    // Render the output
    SDL_RenderPresent(pDisplay->sdl_ren);
}

void disp_exit(disp_t **ppDisplay){
    if(!ppDisplay) return;
    if(!(*ppDisplay)) return;

    // Shutdown SDL
    SDL_DestroyRenderer((*ppDisplay)->sdl_ren);
    SDL_DestroyWindow((*ppDisplay)->sdl_win);
    SDL_Quit();

    // Free Display Data Structures
    free(*ppDisplay);
    // Set userspace reference to NULL
    *ppDisplay = NULL;
}

