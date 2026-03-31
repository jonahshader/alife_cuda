#pragma once

#include "config/sim_params.h"
#include "graphics/fluid_render.cuh"
#include "graphics/renderers/rect_tex_renderer.cuh"
#include "graphics/renderers/simple_rect_renderer.cuh"
#include "graphics/soil_render.cuh"
#include "systems/default_screen.cuh"
#include "systems/float2_ops.cuh"
#include "systems/particle_fluid2.cuh"
#include "systems/profiler_gui.cuh"
#include "systems/soil.cuh"

#include <cmath>

#include <thrust/device_vector.h>

class FluidSoil : public DefaultScreen {
public:
  FluidSoil(Game &game, const SimParams &params);

  void update(float dt) override;
  void render() override;
  bool handle_input(SDL_Event event) override;

private:
  SimParams sim_params;
  const int pixels_per_meter{10};

  const float2 bounds;
  const int2 tex_size;

  p2::ParticleFluidState fluid{};
  SoilState soil{};
  SimpleRectRenderer soil_renderer{};
  RectTexRenderer density_renderer;
  thrust::device_vector<unsigned char> density_texture_data;
  bool grabbing{false};
  bool repelling{false};
  int2 mouse_pos{0, 0};
  float grab_radius{2.0f};
  float grab_strength{0.15f};
  ProfilerGui profiler_gui{};
  bool show_density_grid{true};
};
