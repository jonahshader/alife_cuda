#include "FastNoiseLite.h"
#include "soil.cuh"

#include <glm/glm.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

#include <curand_kernel.h>
#include <thrust/extrema.h>

__host__ __device__ float get_density(SoilPtrs soil, size_t i) {
  return soil.sand_density[i] * SAND_ABSOLUTE_DENSITY +
         soil.silt_density[i] * SILT_ABSOLUTE_DENSITY +
         soil.clay_density[i] * CLAY_ABSOLUTE_DENSITY;
}

__host__ __device__ float get_friction(SoilPtrs soil, size_t i) {
  return soil.sand_density[i] * SAND_FRICTION + soil.silt_density[i] * SILT_FRICTION +
         soil.clay_density[i] * CLAY_FRICTION;
}

void init_soil(SoilState &state, uint width, uint height, float cell_size, uint64_t seed) {
  state.width = width;
  state.height = height;
  state.cell_size = cell_size;
  reset_soil(state, seed);
}

// __global__ void init_rng(curandState *states, unsigned long seed, size_t num_particles) {
//   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i >= num_particles)
//     return;
//   curand_init(seed, i, 0, &states[i]);
// }

void reset_soil(SoilState &state, uint64_t seed) {
  SoilSoA<HostBuffer> soil{};
  // assert(width % BLOCK_WIDTH == 0);
  // assert(height % BLOCK_WIDTH == 0);
  resize_all(soil, state.width * state.height);

  std::mt19937_64 rng(seed);

  FastNoiseLite heightmap_noise(rng());
  heightmap_noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
  heightmap_noise.SetFractalType(FastNoiseLite::FractalType_FBm);
  heightmap_noise.SetFractalOctaves(6);
  heightmap_noise.SetFrequency(1.0f / state.height);

  FastNoiseLite soil_noise(rng());
  soil_noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
  soil_noise.SetFractalType(FastNoiseLite::FractalType_FBm);
  soil_noise.SetFractalOctaves(5);
  soil_noise.SetFrequency(0.01f);

  std::vector<double> heightmap; // heightmap is 1d, but interpreted as 2d

  heightmap.reserve(state.width);
  for (uint x = 0; x < state.width; ++x) {
    heightmap.push_back(heightmap_noise.GetNoise(static_cast<float>(x), 0.0f));
  }

  const float water_height = 0.25f;
  const float min_land_height = 0.1f;
  for (uint x = 0; x < state.width; ++x) {
    float hf = heightmap[x];
    hf = tanh(hf * 2);
    float xf = x / static_cast<float>(state.width);
    xf = xf * (1 - xf) * 4;
    xf = sqrt(xf);
    hf = hf * 0.5f + 0.5f;
    hf *= xf;
    uint h = (min_land_height + hf * 0.8f) * state.height;
    for (uint y = 0; y < h; ++y) {
      const auto id = x + y * state.width;
      float sand = soil_noise.GetNoise(static_cast<float>(x), static_cast<float>(y), 0.0f);
      float silt =
          soil_noise.GetNoise(static_cast<float>(x * 0.75f), static_cast<float>(y), 300.0f);
      float clay = soil_noise.GetNoise(static_cast<float>(x * 0.5f), static_cast<float>(y), 600.0f);
      // sand = sand * 0.5f + 0.5f;
      // silt = silt * 0.5f + 0.5f;
      // clay = clay * 0.5f + 0.5f;

      // sharpen
      float sharpen = 50.0f;
      sand = sand * sharpen;
      silt = silt * sharpen;
      clay = clay * sharpen;

      // softmax
      sand = exp(sand);
      silt = exp(silt);
      clay = exp(clay);

      // int max_index = 0;
      // float max = sand;
      // if (silt > max) {
      //   max = silt;
      //   max_index = 1;
      // }
      // if (clay > max) {
      //   max = clay;
      //   max_index = 2;
      // }

      // sand = max_index == 0 ? 1 : 0;
      // silt = max_index == 1 ? 1 : 0;
      // clay = max_index == 2 ? 1 : 0;

      float density = 1 / (sand + silt + clay);
      sand *= density;
      silt *= density;
      clay *= density;
      soil.sand_density[id] = sand;
      soil.silt_density[id] = silt;
      soil.clay_density[id] = clay;
    }
  }

  for (uint y = 0; y < state.height * water_height; ++y) {
    for (uint x = 0; x < state.width; ++x) {
      const auto id = x + y * state.width;
      float sand = soil.sand_density[id];
      float silt = soil.silt_density[id];
      float clay = soil.clay_density[id];
    }
  }

  copy(state.read, soil);
  copy(state.write, soil);
}

void update_soil_cpu(SoilState &state, float dt) {
  // TODO: implement
}

void update_soil_cuda(SoilState &state, float dt) {
  // TODO: implement
}

SoilPtrs get_soil_read_ptrs(SoilState &state) {
  SoilPtrs ptrs;
  ptrs.get_ptrs(state.read);
  return ptrs;
}
