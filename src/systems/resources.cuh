#pragma once

#include "SDL.h"
#include "graphics/renderers/circle_renderer.cuh"
#include "graphics/renderers/font_renderer.h"
#include "graphics/renderers/line_renderer.cuh"
#include "graphics/renderers/rect_renderer.cuh"

// #include <thrust/random.h>
#include <cstdint>
#include <random>

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
  std::uniform_real_distribution<float> real_dist{0.0f, 1.0f};
  std::normal_distribution<float> norm_dist{0.0f, 1.0f};

  SDL_Window *window{nullptr};

  float rand();

  float randn();
};
