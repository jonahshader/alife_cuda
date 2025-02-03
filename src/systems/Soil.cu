#include "Soil.cuh"

#include <cmath>
#include <random>
#include <chrono>
#include <cstdint>
#include <memory>
#include <cstdint>
#include <glm/glm.hpp>
#include <thrust/extrema.h>
#include <curand_kernel.h>

#include "FastNoiseLite.h"

using uint = std::uint32_t;

__host__ __device__ float get_density(SoilPtrs soil, size_t i) {
  return soil.sand_density[i] * SAND_ABSOLUTE_DENSITY +
         soil.silt_density[i] * SILT_ABSOLUTE_DENSITY +
         soil.clay_density[i] * CLAY_ABSOLUTE_DENSITY;
}

__host__ __device__ float get_friction(SoilPtrs soil, size_t i) {
  return soil.sand_density[i] * SAND_FRICTION + soil.silt_density[i] * SILT_FRICTION +
         soil.clay_density[i] * CLAY_FRICTION;
}

__host__ __device__ inline void add_rect(float x, float y, float width, float height,
                                         glm::vec4 color, float *vbo, size_t i) {
  const auto s = i * SimpleRectRenderer::FLOATS_PER_RECT;
  vbo[s + 0] = x;
  vbo[s + 1] = y;
  vbo[s + 2] = width;
  vbo[s + 3] = height;
  vbo[s + 4] = color.r;
  vbo[s + 5] = color.g;
  vbo[s + 6] = color.b;
  vbo[s + 7] = color.a;
}

__global__ void render_kernel(float *rect_vbo, SoilPtrs read, uint width, float cell_size,
                              size_t rect_count) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rect_count) {
    return;
  }
  auto sand = read.sand_density[i];
  auto silt = read.silt_density[i];
  auto clay = read.clay_density[i];
  auto density = sand + silt + clay;
  glm::vec3 color = glm::vec3(0.0f);
  if (density > 0.001f) {
    auto inv_density = 1.0f / density;
    sand *= inv_density;
    silt *= inv_density;
    clay *= inv_density;

    const glm::vec3 sand_color(219 / 255.0f, 193 / 255.0f, 44 / 255.0f);
    const glm::vec3 silt_color(119 / 255.0f, 143 / 255.0f, 40 / 255.0f);
    const glm::vec3 clay_color(219 / 255.0f, 41 / 255.0f, 23 / 255.0f);
    color = sand_color * sand + silt_color * silt + clay_color * clay;
  }

  const auto x = i % width;
  const auto y = i / width;
  add_rect(x * cell_size, y * cell_size, cell_size, cell_size, glm::vec4(color, 1.0f), rect_vbo, i);
}

SoilSystem::SoilSystem(uint width, uint height, float cell_size, bool use_graphics)
    : width(width), height(height), cell_size(cell_size) {
  reset();
  if (use_graphics) {
    rect_renderer = std::make_unique<SimpleRectRenderer>();
  }
}

// __global__ void init_rng(curandState *states, unsigned long seed, size_t num_particles) {
//   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i >= num_particles)
//     return;
//   curand_init(seed, i, 0, &states[i]);
// }

void SoilSystem::reset() {
  SoilSoA soil{};
  // assert(width % BLOCK_WIDTH == 0);
  // assert(height % BLOCK_WIDTH == 0);
  soil.resize_all(width * height);

  // use time as seed
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937_64 rng(seed);

  FastNoiseLite heightmap_noise(rng());
  heightmap_noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
  heightmap_noise.SetFractalType(FastNoiseLite::FractalType_FBm);
  heightmap_noise.SetFractalOctaves(6);
  heightmap_noise.SetFrequency(1.0f / height);

  FastNoiseLite soil_noise(rng());
  soil_noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
  soil_noise.SetFractalType(FastNoiseLite::FractalType_FBm);
  soil_noise.SetFractalOctaves(5);
  soil_noise.SetFrequency(0.01f);

  std::vector<double> heightmap; // heightmap is 1d, but interpreted as 2d

  heightmap.reserve(width);
  for (auto x = 0; x < width; ++x) {
    heightmap.push_back(heightmap_noise.GetNoise(static_cast<float>(x), 0.0f));
  }

  const float water_height = 0.25f;
  const float min_land_height = 0.1f;
  for (auto x = 0; x < width; ++x) {
    float hf = heightmap[x];
    hf = tanh(hf * 2);
    float xf = x / static_cast<float>(width);
    xf = xf * (1 - xf) * 4;
    xf = sqrt(xf);
    hf = hf * 0.5f + 0.5f;
    hf *= xf;
    uint h = (min_land_height + hf * 0.8f) * height;
    for (auto y = 0; y < h; ++y) {
      const auto id = x + y * width;
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

  for (auto y = 0; y < height * water_height; ++y) {
    for (auto x = 0; x < width; ++x) {
      const auto id = x + y * width;
      float sand = soil.sand_density[id];
      float silt = soil.silt_density[id];
      float clay = soil.clay_density[id];
    }
  }

  read.copy_from_host(soil);
  write.copy_from_host(soil);
}

void SoilSystem::update_cpu(float dt) {
  // TODO: implement
}

void SoilSystem::update_cuda(float dt) {
  // TODO: implement
}

void SoilSystem::render(const glm::mat4 &transform) {
  // early return if we don't have a rect renderer
  if (!rect_renderer)
    return;

  rect_renderer->set_transform(transform);

  const auto rect_count = read.sand_density.size();
  rect_renderer->ensure_vbo_capacity(rect_count);
  // get a cuda compatible pointer to the vbo
  rect_renderer->cuda_register_buffer();
  auto vbo_ptr = rect_renderer->cuda_map_buffer();
  SoilPtrs ptrs;
  ptrs.get_ptrs(read);

  dim3 block(256);
  dim3 grid((rect_count + block.x - 1) / block.x);
  render_kernel<<<grid, block>>>(static_cast<float *>(vbo_ptr), ptrs, width, cell_size, rect_count);
  rect_renderer->cuda_unmap_buffer();

  rect_renderer->render(rect_count);
  rect_renderer->cuda_unregister_buffer();
}

SoilPtrs SoilSystem::get_read_ptrs() {
  SoilPtrs ptrs;
  ptrs.get_ptrs(read);
  return ptrs;
}
