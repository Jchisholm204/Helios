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
    SDL_SetRenderDrawColor(this->sdl_ren, 0, 0, 0, 255);
    int col_size = IN_PIXELS(this->size_x) / cols;
    for (int i = col_size; i < cols * col_size; i += col_size) {
        SDL_RenderDrawLine(this->sdl_ren, i + PX_BORDER, PX_BORDER,
                           i + PX_BORDER, IN_PIXELS(this->size_y) + PX_BORDER);
    }
    int row_size = IN_PIXELS(this->size_x) / rows;
    for (int i = row_size; i < rows * row_size; i += row_size) {
        SDL_RenderDrawLine(this->sdl_ren, PX_BORDER, PX_BORDER + i,
                           IN_PIXELS(this->size_x) + PX_BORDER, i + PX_BORDER);
    }
}

void Display::draw_path(std::vector<std::pair<float, float>> path,
                        std::vector<int> color) {
    (void) path;
    (void) color;
}

void Display::draw_points(std::vector<std::pair<float, float>> points,
                          std::vector<int> color) {
    if (color.size() == 3) {
        SDL_SetRenderDrawColor(this->sdl_ren, color[0], color[1], color[2], 1);
    } else if (color.size() == 4) {
        SDL_SetRenderDrawColor(this->sdl_ren, color[0], color[1], color[2],
                               color[3]);
    } else {
        fprintf(stderr, "Display Color: %ld != (3, 4)\n", color.size());
        fprintf(stderr, "Display Color: (r, g, b) or (r, g, b, a)\n");
        throw std::invalid_argument("Display Color Invalid Size\n");
    }

    for (size_t i = 0; i < points.size(); i++) {
        int x = (int) IN_PIXELS(points[i].first);
        int y = (int) IN_PIXELS(points[i].second);
        SDL_Rect r = {(x) + PX_BORDER - PIXEL_PER_GRID / 2,
                      (y) + PX_BORDER - PIXEL_PER_GRID / 2, PIXEL_PER_GRID,
                      PIXEL_PER_GRID};
        SDL_RenderFillRect(this->sdl_ren, &r);
    }
}

void Display::draw_dims(
    std::vector<std::vector<std::pair<float, float>>>& invalids) {
    // Draw the boxes to contain the spaces
    size_t n = invalids.size();
    if (n == 0)
        return;

    int W = IN_PIXELS(size_x);

    int box_w = 40; // example, use your actual box width
    int box_h = IN_PIXELS(size_y) -
                2 * PX_BORDER; // example, use your actual box height

    int s = (W - n * box_w) / (n + 1); // spacing
    SDL_SetRenderDrawColor(sdl_ren, 0, 0, 0, 255);

    for (size_t i = 0; i < n; i++) {
        int x = s + i * (box_w + s) + PX_BORDER;
        int y = PX_BORDER * 2;

        SDL_Rect r;
        r.x = x;
        r.y = y;
        r.w = box_w;
        r.h = box_h;

        SDL_RenderDrawRect(sdl_ren, &r);
    }

    SDL_SetRenderDrawColor(sdl_ren, 255, 0, 0, 255);

    // Draw the invalid regions
    for (size_t i = 0; i < n; i++) {
        std::vector<std::pair<float, float>>& regions = invalids[i];
        for (size_t j = 0; j < regions.size(); j++) {
            int x = s + i * (box_w + s) + PX_BORDER;
            int y = PX_BORDER * 2;

            SDL_Rect r;
            r.x = x;
            r.y = y + box_h * regions[j].first;
            r.w = box_w;
            // r.h = box_h * (regions[j].second - regions[j].first);
            r.h = regions[j].second;
            // r.h = 1;

            SDL_RenderFillRect(sdl_ren, &r);
        }
    }
}
