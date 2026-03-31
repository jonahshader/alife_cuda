#pragma once

#include "graphics/fluid_render.cuh"
#include "graphics/renderers/rect_tex_renderer.cuh"
#include "systems/default_screen.cuh"
#include "systems/float2_ops.cuh"
#include "systems/particle_fluid2.cuh"

#include <cmath>

#include <thrust/device_vector.h>

class FluidTest2 : public DefaultScreen {
public:
  explicit FluidTest2(Game &game);

  void render(float dt) override;
  bool handle_input(SDL_Event event) override;

private:
  const int pixels_per_meter{100};

  const float2 bounds{10.0f, 10.0f};
  const int2 tex_size{(int)std::round(bounds.x * pixels_per_meter),
                      (int)std::round(bounds.y *pixels_per_meter)};

  p2::ParticleFluidState fluid{};
  RectTexRenderer density_renderer{tex_size.x, tex_size.y, 4};
  thrust::device_vector<unsigned char> density_texture_data;
  bool grabbing{false};
  int2 mouse_pos{0, 0};
};
