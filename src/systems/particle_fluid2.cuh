#pragma once

#include "config/sim_params.h"
#include "soa_helper.h"
#include "soil.cuh"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define FOR_SPH(N, D)                                                                              \
  D(float2, pos, make_float2(0, 0))                                                                \
  D(float2, ppos, make_float2(0, 0))                                                               \
  D(float2, vel, make_float2(0, 0))                                                                \
  D(float2, acc, make_float2(0, 0))                                                                \
  D(float, mass, 1)                                                                                \
  D(float, density, 0)                                                                             \
  D(float, near_density, 0)                                                                        \
  D(uint8_t, sym_break, 0)

namespace p2 {
DEFINE_STRUCTS(SPH, FOR_SPH)

template <template <typename> class Buffer>
struct ParticleGrid {
  Buffer<int> grid_indices{};
  Buffer<int> particles_per_cell{};
  int width{0};
  int height{0};
  int max_particles_per_cell{4};
};

template <template <typename> class Buffer>
void reconfigure_grid(ParticleGrid<Buffer> &grid, int new_width, int new_height,
                      int new_max_particles_per_cell) {
  grid.width = new_width;
  grid.height = new_height;
  grid.max_particles_per_cell = new_max_particles_per_cell;

  grid.grid_indices.resize(new_width * new_height * new_max_particles_per_cell);
  grid.particles_per_cell.resize(new_width * new_height);
}

template <template <typename> class Dst, template <typename> class Src>
void copy(ParticleGrid<Dst> &dst, const ParticleGrid<Src> &src) {
  dst.grid_indices = src.grid_indices;
  dst.particles_per_cell = src.particles_per_cell;
  dst.width = src.width;
  dst.height = src.height;
  dst.max_particles_per_cell = src.max_particles_per_cell;
}

struct ParticleGridPtrs {
  int *grid_indices;
  int *particles_per_cell;

  template <template <typename> class Buffer>
  void get_ptrs(ParticleGrid<Buffer> &grid) {
    grid_indices = raw_ptr(grid.grid_indices);
    particles_per_cell = raw_ptr(grid.particles_per_cell);
  }
};

// Pure data struct — no renderers, no methods beyond trivial accessors
struct ParticleFluidState {
  float2 bounds{};
  SimParams params{};
  SPHSoA<HostBuffer> particles{};
  SPHSoA<DeviceBuffer> particles_device{};
  ParticleGrid<HostBuffer> grid{};
  ParticleGrid<DeviceBuffer> grid_device{};
};

// Free functions for simulation logic
void init_fluid(ParticleFluidState &state, float width, float height, const SimParams &params);
void init_fluid(ParticleFluidState &state, float width, float height);
void init_fluid_grid(ParticleFluidState &state);
void update_fluid(ParticleFluidState &state);
void update_fluid(ParticleFluidState &state, SoilState &soil);
void attract_fluid(ParticleFluidState &state, float2 pos, float max_thrust, float radius);
void calculate_fluid_density_grid(ParticleFluidState &state,
                                  thrust::device_vector<unsigned char> &texture_data, int width,
                                  int height, float max_density);

} // namespace p2
