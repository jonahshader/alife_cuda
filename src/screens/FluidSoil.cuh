#pragma once

#include "systems/DefaultScreen.cuh"
#include "systems/ParameterManager.h"
#include "systems/ParticleFluid2.cuh"
#include "graphics/renderers/RectTexRenderer.cuh"

#include <thrust/device_vector.h>

#include "systems/float2_ops.cuh"

#include <cmath>

class FluidSoil : public DefaultScreen {
public:
  explicit FluidSoil(Game &game);

  void render(float dt) override;
  bool handleInput(SDL_Event event) override;

private:
  const int pixels_per_meter{100};

  const float2 bounds{10.0f, 10.0f};
  const int2 tex_size{(int)std::round(bounds.x * pixels_per_meter),
                      (int)std::round(bounds.y *pixels_per_meter)};

  p2::ParticleFluid fluid{bounds.x, bounds.y, true};
  RectTexRenderer density_renderer{tex_size.x, tex_size.y, 4};
  // thrust::device_vector<float> density_data;
  thrust::device_vector<unsigned char> density_texture_data;
  bool grabbing{false};
  int2 mouse_pos{0, 0};
};
