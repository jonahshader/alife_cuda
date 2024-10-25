#pragma once

#include "systems/DefaultScreen.cuh"
#include "systems/ParameterManager.h"
#include "systems/ParticleFluid2.cuh"
#include "graphics/renderers/RectTexRenderer.cuh"

#include <thrust/device_vector.h>

#include "systems/float2_ops.cuh"

#include <cmath>

class FluidTest2 : public DefaultScreen
{
public:
  explicit FluidTest2(Game &game);

  void render(float dt) override;

private:
  // ParameterManager pm{"fluid2_params.txt"}; // todo: move to p2::ParticleFluid, combine with TunableParams
  const int pixels_per_meter{100}; // 100
  const int particles_per_cell{8}; // 10

  const float2 bounds{10.0f, 10.0f}; // 10 10
  const int2 tex_size{(int)std::round(bounds.x * pixels_per_meter), (int)std::round(bounds.y *pixels_per_meter)};

  // p2::TunableParams params{};
  p2::ParticleFluid fluid{bounds.x, bounds.y, {}, true};
  RectTexRenderer density_renderer{tex_size.x, tex_size.y, 4};
  // thrust::device_vector<float> density_data;
  thrust::device_vector<unsigned char> density_texture_data;
};