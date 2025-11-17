#include "display/display.hpp"

#include <SDL2/SDL_ttf.h>
#include <stdexcept>

#define PIXEL_PER_GRID 10
#define IN_PIXELS(x) (PIXEL_PER_GRID * x)
#define PX_BORDER 40
#define ARROW_SIZE 8
#define TEXT_PIXELS 16

#define FONT "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"

Display::Display(int max_x, int max_y, std::string win_title,
                 std::string font) {
    this->size_x = max_x;
    this->size_y = max_y;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        throw std::runtime_error("Failed to Initialize Display Interface\n");
    }
    if (TTF_Init() != 0) {
        throw std::runtime_error("Failed to Initialize Display Fonts\n");
    }

    this->sdl_win =
        SDL_CreateWindow(win_title.c_str(), 100, 100,
                         IN_PIXELS(max_x) + (PX_BORDER << 1),
                         IN_PIXELS(max_y) + (PX_BORDER << 1), SDL_WINDOW_SHOWN);
    if (!this->sdl_win) {
        SDL_Quit();
        throw std::runtime_error("Failed to Create the Display Window\n");
    }

    this->sdl_ren =
        SDL_CreateRenderer(this->sdl_win, -1, SDL_RENDERER_ACCELERATED);
    if (!this->sdl_ren) {
        SDL_DestroyWindow(this->sdl_win);
        SDL_Quit();
        throw std::runtime_error("Failed to Create Display Render");
    }

    this->sdl_font = TTF_OpenFont(font.c_str(), TEXT_PIXELS);
    if (!this->sdl_font) {
        SDL_DestroyRenderer(this->sdl_ren);
        SDL_DestroyWindow(this->sdl_win);
        SDL_Quit();
        throw std::runtime_error("Failed to Create Fonts");
    }
}

Display::~Display() {
    SDL_DestroyRenderer(this->sdl_ren);
    SDL_DestroyWindow(this->sdl_win);
    SDL_Quit();
}

void Display::clear(void) {
    SDL_SetRenderDrawColor(this->sdl_ren, 255, 255, 255, 255);
    SDL_RenderClear(this->sdl_ren);
}

void Display::render(void) {
    // Render the drawing box
    SDL_SetRenderDrawColor(this->sdl_ren, 0, 0, 0, 255);
    // Right line
    SDL_RenderDrawLine(this->sdl_ren, IN_PIXELS(this->size_x) + PX_BORDER,
                       PX_BORDER, IN_PIXELS(this->size_x) + PX_BORDER,
                       IN_PIXELS(this->size_y) + PX_BORDER);
    // Left Line
    SDL_RenderDrawLine(this->sdl_ren, PX_BORDER, PX_BORDER, PX_BORDER,
                       IN_PIXELS(this->size_y) + PX_BORDER);
    // Top Line
    SDL_RenderDrawLine(this->sdl_ren, PX_BORDER, PX_BORDER,
                       IN_PIXELS(this->size_x) + PX_BORDER, PX_BORDER);
    // // Bottom Line
    SDL_RenderDrawLine(this->sdl_ren, PX_BORDER,
                       IN_PIXELS(this->size_x) + PX_BORDER,
                       IN_PIXELS(this->size_x) + PX_BORDER,
                       IN_PIXELS(this->size_y) + PX_BORDER);
    SDL_RenderPresent(this->sdl_ren);
}

void Display::label(std::string label) {
    SDL_Color color = {0, 0, 0, 255}; // white text

    SDL_Surface* surface =
        TTF_RenderText_Solid(this->sdl_font, label.c_str(), color);
    if (!surface) {
        throw std::runtime_error("SDL Failed to create a surface");
    }

    SDL_Texture* texture = SDL_CreateTextureFromSurface(this->sdl_ren, surface);
    SDL_FreeSurface(surface);
    if (!texture) {
        fprintf(stderr, "CreateTexture Error: %s\n", SDL_GetError());
        return;
    }

    int y = IN_PIXELS(this->size_y) + PX_BORDER + (TEXT_PIXELS >> 1);
    int x = (IN_PIXELS(this->size_x) >> 1) + PX_BORDER -
            ((label.size() * TEXT_PIXELS) >> 2);

    SDL_Rect dst = {x, y, 0, 0};
    SDL_QueryTexture(texture, NULL, NULL, &dst.w, &dst.h);
    SDL_RenderCopy(this->sdl_ren, texture, NULL, &dst);
    SDL_DestroyTexture(texture);
}

bool Display::poll_quit(void) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT)
            return true;
    }
    return false;
}

void Display::draw_grid(int rows, int cols) {
}

void Display::draw_path(std::vector<std::pair<float, float>> path,
                        std::vector<int> color) {
}

void Display::draw_points(std::vector<std::pair<float, float>> points,
                          std::vector<int> color) {
}
