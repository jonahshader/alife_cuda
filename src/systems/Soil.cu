#include "Soil.cuh"

#include <cmath>
#include <random>
#include <chrono>
#include <cstdint>
#include <memory>
#include <cstdint>
#include <glm/glm.hpp>
#include <thrust/extrema.h>

#include "FastNoiseLite.h"

constexpr auto FILTER_WIDTH = 3;
constexpr auto BLOCK_WIDTH = 16;
constexpr auto BLOCK_HEIGHT = BLOCK_WIDTH;

using uint = std::uint32_t;

__constant__ float gaussian_filter[FILTER_WIDTH * FILTER_WIDTH] = {
    0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f};

__host__ __device__ inline float get_cell_w_wrap(float *cells, int x, int y, uint width) {
  // wrap around left and right
  // TODO: try modulus solution
  if (x < 0)
    x += width;
  if (x >= width)
    x -= width;
  return cells[y * width + x];
}

__host__ __device__ inline void set_cell_w_wrap(float *cells, int x, int y, uint width,
                                                float value) {
  // wrap around left and right
  // TODO: try modulus solution
  if (x < 0)
    x += width;
  if (x >= width)
    x -= width;
  cells[y * width + x] = value;
}

__host__ __device__ inline int16_t get_cell_w_wrap(int16_t *cells, int x, int y, int width) {
  // wrap around left and right
  // TODO: try modulus solution
  if (x < 0)
    x += width;
  if (x >= width)
    x -= width;
  return cells[y * width + x];
}

__host__ __device__ inline float calc_delta(float water, float other_water) {
  return (other_water - water) / 5;
}

__host__ __device__ inline bool inbounds(int y, uint height) {
  return y >= 0 && y < height;
}

__host__ __device__ inline float get_effective_density(float sand, float silt, float clay) {
  return 1 - (sand * SAND_RELATIVE_DENSITY + silt * SILT_RELATIVE_DENSITY +
              clay * CLAY_RELATIVE_DENSITY);
}

__host__ __device__ inline float get_effective_permeability(float sand, float silt, float clay) {
  return 1 - (sand * (1 - SAND_PERMEABILITY) + silt * (1 - SILT_PERMEABILITY) +
              clay * (1 - CLAY_PERMEABILITY));
}

__host__ __device__ inline void get_effective(const SoilPtrs &ptrs, size_t id, float &density,
                                              float &permeability) {
  // does not check upper or lower bounds
  density =
      get_effective_density(ptrs.sand_density[id], ptrs.silt_density[id], ptrs.clay_density[id]);
  permeability = get_effective_permeability(ptrs.sand_density[id], ptrs.silt_density[id],
                                            ptrs.clay_density[id]);
}

__host__ __device__ inline void get_effective(const SoilPtrs &ptrs, int x, int y, uint width,
                                              float &density, float &permeability) {
  // does not check upper or lower bounds
  if (x < 0)
    x += width;
  if (x >= width)
    x -= width;
  const auto id = x + y * width;
  get_effective(ptrs, id, density, permeability);
}

__host__ __device__ inline void mix_give_take(SoilPtrs &read, SoilPtrs &write, uint width,
                                              uint height, float dt, uint x, uint y) {

  const auto id = x + y * width;

  const float give = 200.0f * dt;

  const auto water = read.water_density[id];

  float water_delta = 0;
  // for now, hard code to going down
  const float final_give_scalar = 0.5f;

  auto water_other = get_cell_w_wrap(read.water_density, x - 1, y, width);
  water_delta += calc_delta(water, water_other);
  water_other = get_cell_w_wrap(read.water_density, x + 1, y, width);
  water_delta += calc_delta(water, water_other);
  if (y < height - 1) {
    water_other = get_cell_w_wrap(read.water_density, x, y + 1, width);
    // water_delta += calc_delta(water, water_other);
    // TODO: max water should be calculated from soil composition
    if (water < 1.0f) {
      if (water_other > give) {
        water_delta += give * final_give_scalar;
      } else {
        water_delta += water_other * final_give_scalar;
      }
    }
  }
  if (y > 0) {
    water_other = get_cell_w_wrap(read.water_density, x, y - 1, width);
    // water_delta += calc_delta(water, water_other);
    // TODO: max water should be calculated from soil composition
    if (water_other < 1.0f) {
      if (water > give) {
        water_delta -= give * final_give_scalar;
      } else {
        water_delta -= water * final_give_scalar;
      }
    }
  }

  write.water_density[id] = water + water_delta;
}

__host__ __device__ inline float compute_give_normal(float current, float other, float dt,
                                                     float current_perm, float other_perm,
                                                     float current_dens, float other_dens) {
  const float perm = min(current_perm, other_perm);
  const float NORMAL_GIVE_PER_DELTA_PER_SECOND = 25.0f * dt * perm;
  const float MAX_GIVE_PER_DELTA_PER_SECOND = 33.0f * dt;
  const auto delta = current - other;
  const auto sum = current + other;
  const auto overflow = current > current_dens || other > other_dens;

  float rate = overflow ? MAX_GIVE_PER_DELTA_PER_SECOND : NORMAL_GIVE_PER_DELTA_PER_SECOND;

  // float rate;
  const auto current_overflow = current > current_dens;
  const auto other_overflow = other > other_dens;
  // if (current_overflow && other_overflow) {
  //     rate = MAX_GIVE_PER_DELTA_PER_SECOND;
  // } else if (current_overflow) {
  //     rate = MAX_GIVE_PER_DELTA_PER_SECOND;
  // } else if (other_overflow) {
  //     rate = MAX_GIVE_PER_DELTA_PER_SECOND;
  // } else {
  //     rate = NORMAL_GIVE_PER_DELTA_PER_SECOND;
  // }

  const auto epsilon = 0.15f * max(sum, 0.0f) * pow(perm, 5);
  // const auto use_epsilon = sum > epsilon / 2;
  const auto use_epsilon = false;
  if (delta > 0 && other_dens == 0) {
    // we are giving to other cell. check how much we can give
    rate = 0;
  } else if (delta < 0 && current_dens == 0) {
    // we are taking from other cell. check how much we can take
    rate = 0;
  }

  if (use_epsilon && delta > epsilon / 2) {
    return tanh(delta - epsilon) * rate;
  } else if (use_epsilon && delta < -epsilon / 2) {
    return tanh(delta + epsilon) * rate;
  } else {
    return tanh(delta) * rate;
  }
}

__host__ __device__ inline void mix_give_take_3(SoilPtrs &read, SoilPtrs &write, uint width,
                                                uint height, float dt, uint x, uint y) {
  const auto id = x + y * width;

  float current_perm = 0;
  float current_dens = 0;
  get_effective(read, x, y, width, current_dens, current_perm);

  float give_left = 0;
  float give_right = 0;
  float give_up = 0;
  float give_down = 0;

  const auto current_water = read.water_density[id];

  float other_perm = 0;
  float other_dens = 0;

  const auto left_water = get_cell_w_wrap(read.water_density, x - 1, y, width);
  get_effective(read, x - 1, y, width, other_dens, other_perm);
  give_left = compute_give_normal(current_water, left_water, dt, current_perm, other_perm,
                                  current_dens, other_dens);
  const auto right_water = get_cell_w_wrap(read.water_density, x + 1, y, width);
  get_effective(read, x + 1, y, width, other_dens, other_perm);
  give_right = compute_give_normal(current_water, right_water, dt, current_perm, other_perm,
                                   current_dens, other_dens);
  if (y > 0) {
    const auto down_water = read.water_density[x + (y - 1) * width];
    get_effective(read, x, y - 1, width, other_dens, other_perm);
    give_down = compute_give_normal(current_water, down_water, dt, current_perm, other_perm,
                                    current_dens, other_dens);
    if (down_water < other_dens) {
      give_down += pow(other_perm, 2) * current_water / 4;
    }
  }
  if (y < height - 1) {
    const auto up_water = read.water_density[x + (y + 1) * width];
    get_effective(read, x, y + 1, width, other_dens, other_perm);
    give_up = compute_give_normal(current_water, up_water, dt, current_perm, other_perm,
                                  current_dens, other_dens);
    if (current_water < current_dens) {
      give_up += pow(current_perm, 2) * -up_water / 4;
    }
  }

  write.water_density[id] = current_water - give_left - give_right - give_up - give_down;
}

__host__ __device__ inline void mix_box_blur(SoilPtrs &read, SoilPtrs &write, uint width,
                                             uint height, uint x, uint y) {
  const auto id = x + y * width;
  const auto water = read.water_density[id];
  float water_delta = 0.0f;

  for (auto xx = x - 1; xx <= x + 1; ++xx) {
    water_delta += max(0.0f, get_cell_w_wrap(read.water_density, xx, y, width) - 1);
  }
  if (y > 0) {
    for (auto xx = x - 1; xx <= x + 1; ++xx) {
      water_delta += max(0.0f, get_cell_w_wrap(read.water_density, xx, y - 1, width) - 1);
    }
  }
  if (y < height - 1) {
    for (auto xx = x - 1; xx <= x + 1; ++xx) {
      water_delta += max(0.0f, get_cell_w_wrap(read.water_density, xx, y + 1, width) - 1);
    }
  }
  // TODO: this doesn't conserve water density
  write.water_density[id] = min(1.0f, water) + water_delta / 9;
}

__host__ __device__ inline void copy_give(SoilPtrs &read, SoilPtrs &write, uint width, uint height,
                                          float dt, uint x, uint y) {
  const auto id = x + y * width;

  const auto current_water = read.water_density[id];
  const auto give_left = read.water_give_left;
  const auto give_right = read.water_give_right;
  const auto give_up = read.water_give_up;
  const auto give_down = read.water_give_down;
  const auto write_give_left = write.water_give_left;
  const auto write_give_right = write.water_give_right;
  const auto write_give_up = write.water_give_up;
  const auto write_give_down = write.water_give_down;

  // take what the right is giving to the left (me)
  float from_right = get_cell_w_wrap(give_left, x + 1, y, width);
  // take what the left is giving to the right (me)
  float from_left = get_cell_w_wrap(give_right, x - 1, y, width);
  // take what the bottom is giving to the top (me)

  float from_down = 0;
  bool down_inbounds = inbounds(y - 1, height);
  if (down_inbounds) {
    from_down = get_cell_w_wrap(give_up, x, y - 1, width);
  }
  // take what the top is giving to the bottom (me)
  float from_up = 0;
  bool up_inbounds = inbounds(y + 1, height);
  if (up_inbounds) {
    from_up = get_cell_w_wrap(give_down, x, y + 1, width);
  }

  float water_delta = from_right + from_left + from_down + from_up;
  float capacity = 1 - current_water;
  if (water_delta > capacity) {
    // const float water_delta_scaler = capacity / water_delta;
    // float new_from_right = from_right * water_delta_scaler;
    // float new_from_left = from_left * water_delta_scaler;
    // float new_from_down = from_down * water_delta_scaler;
    // float new_from_up = from_up * water_delta_scaler;
    //
    // write.water_density[id] = 1;
    // set_cell_w_wrap(write_give_left, x + 1, y, width, new_from_right);
    // set_cell_w_wrap(write_give_right, x - 1, y, width, new_from_left);
    // if (down_inbounds) {
    //     set_cell_w_wrap(write_give_up, x, y - 1, width, new_from_down);
    // }
    // if (up_inbounds) {
    //     set_cell_w_wrap(write_give_down, x, y + 1, width, new_from_up);
    // }

    write.water_density[id] = current_water;
    set_cell_w_wrap(write_give_left, x + 1, y, width, from_right);
    set_cell_w_wrap(write_give_right, x - 1, y, width, from_left);
    if (down_inbounds) {
      set_cell_w_wrap(write_give_up, x, y - 1, width, from_down);
    }
    if (up_inbounds) {
      set_cell_w_wrap(write_give_down, x, y + 1, width, from_up);
    }
  } else {
    write.water_density[id] = current_water + water_delta;
    set_cell_w_wrap(write_give_left, x + 1, y, width, 0);
    set_cell_w_wrap(write_give_right, x - 1, y, width, 0);
    if (down_inbounds) {
      set_cell_w_wrap(write_give_up, x, y - 1, width, 0);
    }
    if (up_inbounds) {
      set_cell_w_wrap(write_give_down, x, y + 1, width, 0);
    }
  }
}

__host__ __device__ inline void copy_take_back(SoilPtrs &read, SoilPtrs &write, uint width,
                                               uint height, float dt, uint x, uint y) {
  // if there are remaining water_give_{left, right, up, down}, take it back
  const auto id = x + y * width;

  const auto current_water = read.water_density[id];
  const auto give_left = read.water_give_left[id];
  const auto give_right = read.water_give_right[id];
  const auto give_up = read.water_give_up[id];
  const auto give_down = read.water_give_down[id];
  write.water_give_left[id] = 0;
  write.water_give_right[id] = 0;
  write.water_give_up[id] = 0;
  write.water_give_down[id] = 0;
  // if this goes over 1, its an error...
  write.water_density[id] = current_water + give_left + give_right + give_up + give_down;
}

__global__ void mix_give_take_kernel(SoilPtrs read, SoilPtrs write, uint width, uint height,
                                     float dt) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  mix_give_take(read, write, width, height, dt, x, y);
}

__global__ void mix_box_blur_kernel(SoilPtrs read, SoilPtrs write, uint width, uint height) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  mix_box_blur(read, write, width, height, x, y);
}

__global__ void mix_give_take_3_kernel(SoilPtrs read, SoilPtrs write, uint width, uint height,
                                       float dt) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  mix_give_take_3(read, write, width, height, dt, x, y);
}

__global__ void copy_give_kernel(SoilPtrs read, SoilPtrs write, uint width, uint height, float dt) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  copy_give(read, write, width, height, dt, x, y);
}

__global__ void copy_take_back_kernel(SoilPtrs read, SoilPtrs write, uint width, uint height,
                                      float dt) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  copy_take_back(read, write, width, height, dt, x, y);
}

__host__ __device__ inline void add_rect(float x, float y, float width, float height, float radius,
                                         glm::vec4 color, float *vbo, size_t i) {
  const auto s = i * RectRenderer::FLOATS_PER_RECT;
  vbo[s + 0] = x;
  vbo[s + 1] = y;
  vbo[s + 2] = width;
  vbo[s + 3] = height;
  vbo[s + 4] = radius;
  vbo[s + 5] = color.r;
  vbo[s + 6] = color.g;
  vbo[s + 7] = color.b;
  vbo[s + 8] = color.a;
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
  const auto water_density = read.water_density[i];
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

  const glm::vec3 water_color(0.2f, 0.3f, 0.95f);
  auto amount = min(max(water_density, 0.0f), 1.0f);
  color = glm::mix(color, water_color, amount);
  float opacity = density * (1 - amount) + amount;

  const auto x = i % width;
  const auto y = i / width;
  add_rect(x * cell_size + cell_size / 2, y * cell_size + cell_size / 2, cell_size + 1, cell_size + 1, 1,
           glm::vec4(color, opacity), rect_vbo, i);
}

SoilSystem::SoilSystem(uint width, uint height, float cell_size, bool use_graphics)
    : width(width), height(height), cell_size(cell_size) {
  reset();
  if (use_graphics) {
    rect_renderer = std::make_unique<RectRenderer>();
  }
}

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
      sand = sand * 0.5f + 0.5f;
      silt = silt * 0.5f + 0.5f;
      clay = clay * 0.5f + 0.5f;
      // sand = pow(sand, 5);
      // silt = pow(silt, 5);
      // clay = pow(clay, 5);

      int max_index = 0;
      float max = sand;
      if (silt > max) {
        max = silt;
        max_index = 1;
      }
      if (clay > max) {
        max = clay;
        max_index = 2;
      }

      sand = max_index == 0 ? 1 : 0;
      silt = max_index == 1 ? 1 : 0;
      clay = max_index == 2 ? 1 : 0;

      float density = sand + silt + clay;
      sand /= density;
      silt /= density;
      clay /= density;
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

      // TODO: pull this out into a function
      float effective_density = sand * SAND_RELATIVE_DENSITY + silt * SILT_RELATIVE_DENSITY +
                                clay * CLAY_RELATIVE_DENSITY;
      soil.water_density[id] = 1 - effective_density;
    }
  }

  read.copy_from_host(soil);
  write.copy_from_host(soil);
}

void SoilSystem::mix_give_take_cuda(float dt) {
  // grab ptrs
  SoilPtrs read_ptrs{}, write_ptrs{};
  read_ptrs.get_ptrs(read);
  write_ptrs.get_ptrs(write);

  // set up grid
  dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid(width + block.x - 1 / block.x, height + block.y - 1 / block.y);

  // launch kernel
  mix_give_take_kernel<<<grid, block>>>(read_ptrs, write_ptrs, width, height, dt);

  // kernel writes to water_density, so swap read and write
  write.water_density.swap(read.water_density);

  // re-acquire ptrs
  read_ptrs.get_ptrs(read);
  write_ptrs.get_ptrs(write);

  // launch mix box blur kernel
  mix_box_blur_kernel<<<grid, block>>>(read_ptrs, write_ptrs, width, height);

  // kernel writes to water_density, so swap read and write
  write.water_density.swap(read.water_density);
}

void SoilSystem::mix_give_take_3_cuda(float dt) {
  // grab ptrs
  SoilPtrs read_ptrs{}, write_ptrs{};
  read_ptrs.get_ptrs(read);
  write_ptrs.get_ptrs(write);

  // set up grid
  dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid(width + block.x - 1 / block.x, height + block.y - 1 / block.y);

  // launch kernel
  mix_give_take_3_kernel<<<grid, block>>>(read_ptrs, write_ptrs, width, height, dt);

  // wrote to water_density, so swap read and write
  write.water_density.swap(read.water_density);
}

void SoilSystem::update_cpu(float dt) {
  // TODO: implement
}

void SoilSystem::update_cuda(float dt) {
  // for now, just mix give take
  // mix_give_take_cuda(dt);

  // SoilSoA soil;
  // read.copy_to_host(soil);
  // find min and max water density
  auto min_iter = thrust::min_element(read.water_density.begin(), read.water_density.end());
  auto max_iter = thrust::max_element(read.water_density.begin(), read.water_density.end());
  auto min = *min_iter;
  auto max = *max_iter;

  // // get sum
  // auto sum = thrust::reduce(read.water_density.begin(), read.water_density.end(), 0.0f);

  // std::cout << "water_density statistics:\n";
  // std::cout << "min: " << min << "\n";
  // std::cout << "max: " << max << std::endl;
  // std::cout << "sum: " << sum << std::endl;

  mix_give_take_3_cuda(dt);
}

void SoilSystem::render(const glm::mat4 &transform) {
  // early return if we don't have a rect renderer
  if (!rect_renderer)
    return;

  rect_renderer->set_transform(transform);

  const auto rect_count = read.water_density.size();
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

void SoilSystem::add_water(int x, int y, float amount) {
  x /= cell_size;
  y /= cell_size;
  if (x > 0 && x < width && y > 0 && y < height) {
    read.water_density[x + y * width] += amount;
  }
}
