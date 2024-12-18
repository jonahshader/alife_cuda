#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>

#include "SoAHelper.h"
#include "graphics/renderers/SimpleRectRenderer.cuh"

constexpr float SAND_RELATIVE_DENSITY = 0.5f;
constexpr float SILT_RELATIVE_DENSITY = 0.7f;
constexpr float CLAY_RELATIVE_DENSITY = 1.0f;

constexpr float SAND_ABSOLUTE_DENSITY = 1.0f;
constexpr float SILT_ABSOLUTE_DENSITY = 190.0f;
constexpr float CLAY_ABSOLUTE_DENSITY = 500.0f;

constexpr float SAND_PERMEABILITY = 0.5f;
constexpr float SILT_PERMEABILITY = 0.3f;
constexpr float CLAY_PERMEABILITY = 0.00f;

constexpr int PARTICLES_PER_SOIL_CELL = 1;

#define FOR_SOIL(N, D)                                                                             \
  D(float, water_density, 0)                                                                       \
  D(float, water_give_left, 0)                                                                     \
  D(float, water_give_right, 0)                                                                    \
  D(float, water_give_up, 0)                                                                       \
  D(float, water_give_down, 0)                                                                     \
  D(float, sand_density, 0)                                                                        \
  D(float, silt_density, 0)                                                                        \
  D(float, clay_density, 0)                                                                        \
  D(float, ph, 6.5)                                                                                \
  D(float, organic_matter, 0)

#define FOR_SOIL_PARTICLES(N, D)                                                                   \
  D(float, density, 0)                                                                             \
  D(float, near_density, 0)                                                                        \
  N(curandState, rand_state)                                                                       \
  D(float, x_offset, 0)                                                                            \
  D(float, y_offset, 0)

DEFINE_STRUCTS(Soil, FOR_SOIL)
DEFINE_STRUCTS(SoilParticles, FOR_SOIL_PARTICLES)

class SoilSystem {
public:
  using uint = std::uint32_t;
  SoilSystem(uint width, uint height, float cell_size, bool use_graphics);
  ~SoilSystem() = default;

  void update_cpu(float dt);
  void update_cuda(float dt);
  void render(const glm::mat4 &transform);
  void reset();
  void jitter_particles();
  uint get_width() const {
    return width;
  }
  uint get_height() const {
    return height;
  }
  float get_cell_size() const {
    return cell_size;
  }
  SoilPtrs get_read_ptrs();
  SoilParticlesPtrs get_particles_ptrs();

  void add_water(int x, int y, float amount);

private:
  uint width, height;
  float cell_size;
  SoilSoADevice read{}, write{};
  SoilParticlesSoADevice particles{};
  std::unique_ptr<SimpleRectRenderer> rect_renderer{};

  void mix_give_take_cuda(float dt);
  void mix_give_take_3_cuda(float dt);
};
