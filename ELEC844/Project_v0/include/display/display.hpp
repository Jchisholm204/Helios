/**
 * @file display.hpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.2
 * @date Created: 2025-10-08
 * @modified Last Modified: 2025-11-17
 *
 *  Copied from ELEC844 Assignment 2
 *  CPP adaptation of original code
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _DISPLAY_H_
#define _DISPLAY_H_

#include <SDL.h>
#include <SDL2/SDL.h>
#include <SDL_ttf.h>
#include <string>
#include <vector>

class Display {
  public:
    Display(
        int max_x, int max_y, std::string win_title = "ELEC844Project",
        std::string font = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf");
    ~Display(void);
    void clear(void);
    void render(void);
    void label(std::string label);
    bool poll_quit(void);
    void draw_grid(int rows, int cols);
    void draw_path(std::vector<std::pair<float, float>> path,
                   std::vector<int> color = {0, 0, 0});
    void draw_points(std::vector<std::pair<float, float>> points,
                     std::vector<int> color = {0, 0, 0});

    void draw_dims(std::vector<std::vector<std::pair<float, float>>> &invalds);

  private:
    SDL_Window* sdl_win;
    SDL_Renderer* sdl_ren;
    TTF_Font* sdl_font;
    int size_x, size_y;
};

#endif
