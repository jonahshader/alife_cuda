#pragma once

#include "soa_helper.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <vector>

#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Porosity: fraction of volume that is pore space
constexpr float SAND_POROSITY = 0.38f;
constexpr float SILT_POROSITY = 0.45f;
constexpr float CLAY_POROSITY = 0.50f;

// Capillary strength: suction pressure in SPH units (clay >> silt >> sand)
constexpr float SAND_CAPILLARY = 20.0f;
constexpr float SILT_CAPILLARY = 120.0f;
constexpr float CLAY_CAPILLARY = 400.0f;

constexpr float SAND_FRICTION = 8.0f;
constexpr float SILT_FRICTION = 20.0f;
constexpr float CLAY_FRICTION = 50.0f;

constexpr int PARTICLES_PER_SOIL_CELL = 1;

#define FOR_SOIL(N, D)                                                                             \
  D(float, sand_density, 0)                                                                        \
  D(float, silt_density, 0)                                                                        \
  D(float, clay_density, 0)                                                                        \
  D(float, ph, 6.5)                                                                                \
  D(float, organic_matter, 0)                                                                      \
  D(float, saturation, 0)

DEFINE_STRUCTS(Soil, FOR_SOIL)

using uint = std::uint32_t;

// Pure data struct — no renderers, no methods
struct SoilState {
  uint width{0};
  uint height{0};
  float cell_size{0.0f};
  SoilSoA<DeviceBuffer> read{}, write{};
};

// Free functions for simulation logic
void init_soil(SoilState &state, uint width, uint height, float cell_size, uint64_t seed,
               int terrain_mode = 0);
void reset_soil(SoilState &state, uint64_t seed);
void reset_soil_capillary_test(SoilState &state);
void update_soil_cpu(SoilState &state, float dt);
void update_soil_cuda(SoilState &state, float dt);
SoilPtrs get_soil_read_ptrs(SoilState &state);

__host__ __device__ float get_solid_density(SoilPtrs soil, size_t i, float target_density);
__host__ __device__ float get_porosity(SoilPtrs soil, size_t i);
__host__ __device__ float get_pore_capacity(SoilPtrs soil, size_t i, float target_density);
__host__ __device__ float get_capillary_strength(SoilPtrs soil, size_t i);
__host__ __device__ float get_friction(SoilPtrs soil, size_t i);
