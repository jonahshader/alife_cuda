#pragma once

#include "graphics/renderers/rect_tex_renderer.cuh"
#include "systems/default_screen.cuh"
#include "systems/parameter_manager.h"
#include "systems/particle_fluid2.cuh"
#include "systems/profiler_gui.cuh"
#include "systems/soil.cuh"
#include "systems/float2_ops.cuh"
// #include "graphics/renderers/circle_renderer.cuh"

#include <cmath>

#include <thrust/device_vector.h>

class FluidSoil : public DefaultScreen {
public:
  explicit FluidSoil(Game &game);

  void render(float dt) override;
  bool handle_input(SDL_Event event) override;

private:
  const float soil_size = 0.1; // TODO: this should be some ratio of the particle radius probably
  const int pixels_per_meter{10};

  const float2 bounds{32.0f, 16.0f};
  const int2 tex_size{(int)std::round(bounds.x * pixels_per_meter),
                      (int)std::round(bounds.y *pixels_per_meter)};

  p2::ParticleFluid fluid{bounds.x, bounds.y, true};
  SoilSystem soil{(unsigned int)std::round(bounds.x / soil_size),
                  (unsigned int)std::round(bounds.y / soil_size), soil_size, true};
  RectTexRenderer density_renderer{tex_size.x, tex_size.y, 4};
  // thrust::device_vector<float> density_data;
  thrust::device_vector<unsigned char> density_texture_data;
  bool grabbing{false};
  bool repelling{false};
  int2 mouse_pos{0, 0};
  float grab_radius{2.0f};
  float grab_strength{0.15f};
  ProfilerGui profiler_gui{};
  bool show_density_grid{true};

  void check_cuda(const std::string &msg);
};
