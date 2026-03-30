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
constexpr float CLAY_ABSOLUTE_DENSITY = 300.0f;

constexpr float SAND_FRICTION = 8.0f;
constexpr float SILT_FRICTION = 20.0f;
constexpr float CLAY_FRICTION = 50.0f;

constexpr int PARTICLES_PER_SOIL_CELL = 1;

#define FOR_SOIL(N, D)                                                                             \
  D(float, sand_density, 0)                                                                        \
  D(float, silt_density, 0)                                                                        \
  D(float, clay_density, 0)                                                                        \
  D(float, ph, 6.5)                                                                                \
  D(float, organic_matter, 0)

DEFINE_STRUCTS(Soil, FOR_SOIL)

class SoilSystem {
public:
  using uint = std::uint32_t;
  SoilSystem(uint width, uint height, float cell_size, bool use_graphics);
  ~SoilSystem() = default;

  void update_cpu(float dt);
  void update_cuda(float dt);
  void render(const glm::mat4 &transform);
  void reset();
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

private:
  uint width, height;
  float cell_size;
  SoilSoADevice read{}, write{};
  std::unique_ptr<SimpleRectRenderer> rect_renderer{};
};

__host__ __device__ float get_density(SoilPtrs soil, size_t i);
__host__ __device__ float get_friction(SoilPtrs soil, size_t i);
