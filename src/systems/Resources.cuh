//
// Created by jonah on 4/27/2023.
//

#pragma once

#include "graphics/renderers/FontRenderer.h"
#include "graphics/renderers/CircleRenderer.cuh"
#include "graphics/renderers/RectRenderer.cuh"
#include "graphics/renderers/LineRenderer.cuh"
#include "SDL.h"

// #include <thrust/random.h>
#include <random>
#include <cstdint>

class Resources {
public:
    explicit Resources(uint64_t seed);

    FontRenderer main_font{"fonts/OpenSans-Regular.arfont"};
    FontRenderer main_font_world{"fonts/OpenSans-Regular.arfont"};
    FontRenderer extra_bold_font{"fonts/OpenSans-ExtraBold.arfont"};
    FontRenderer fira_regular_font{"fonts/FiraSans-Regular.arfont"};
    CircleRenderer circle_renderer{};
    RectRenderer rect_renderer{};
    LineRenderer line_renderer{};

    std::default_random_engine generator;
    std::uniform_real_distribution<float> realDist{0.0f, 1.0f};
    std::normal_distribution<float> normDist{0.0f, 1.0f};

    SDL_Window *window{nullptr};

    float rand();

    float randn();
};
