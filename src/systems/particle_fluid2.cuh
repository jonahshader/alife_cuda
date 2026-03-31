#pragma once

#include "parameter_manager.h"
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

struct TunableParams {
  float dt = 1 / 600.0f;
  float dt_predict = 1 / 120.0f;
  float gravity = -13.0f;
  float collision_damping = 0.5f;
  float smoothing_radius = 0.2f;
  float target_density = 234.0f;
  float pressure_mult = 225.0f;
  float near_pressure_mult = 18.0f;
  float viscosity_strength = 0.03f;

  int particles_per_cell = 4;
  int max_particles_per_cell = particles_per_cell * 32;
};

struct ParticleGrid {
  thrust::host_vector<int> grid_indices{};
  thrust::host_vector<int> particles_per_cell{};
  int width{0};
  int height{0};
  int max_particles_per_cell{4};

  void reconfigure(int new_width, int new_height, int new_max_particles_per_cell) {
    width = new_width;
    height = new_height;
    max_particles_per_cell = new_max_particles_per_cell;

    grid_indices = std::vector<int>(width * height * max_particles_per_cell);
    particles_per_cell = std::vector<int>(width * height);
  }
};

struct ParticleGridDevice {
  thrust::device_vector<int> grid_indices{};
  thrust::device_vector<int> particles_per_cell{};

  void copy_from_host(ParticleGrid &host) {
    grid_indices = host.grid_indices;
    particles_per_cell = host.particles_per_cell;
  }

  void copy_to_host(ParticleGrid &host) {
    host.grid_indices = grid_indices;
    host.particles_per_cell = particles_per_cell;
  }
};

struct ParticleGridPtrs {
  int *grid_indices;
  int *particles_per_cell;

  void get_ptrs(ParticleGridDevice &grid_device) {
    grid_indices = grid_device.grid_indices.data().get();
    particles_per_cell = grid_device.particles_per_cell.data().get();
  }

  void get_ptrs(ParticleGrid &grid) {
    grid_indices = grid.grid_indices.data();
    particles_per_cell = grid.particles_per_cell.data();
  }
};

// Pure data struct — no renderers, no methods beyond trivial accessors
struct ParticleFluidState {
  float2 bounds{};
  TunableParams params{};
  bool use_internal_params{false};
  std::unique_ptr<ParameterManager> pm{};
  SPHSoA particles{};
  SPHSoADevice particles_device{};
  ParticleGrid grid{};
  ParticleGridDevice grid_device{};
};

// Free functions for simulation logic
void init_fluid(ParticleFluidState &state, float width, float height, const TunableParams &params);
void init_fluid(ParticleFluidState &state, float width, float height);
void init_fluid_grid(ParticleFluidState &state);
void update_fluid(ParticleFluidState &state);
void update_fluid(ParticleFluidState &state, const SoilPtrs &soil_ptrs, int soil_width,
                  int soil_height, float soil_cell_size);
void attract_fluid(ParticleFluidState &state, float2 pos, float max_thrust, float radius);
void calculate_fluid_density_grid(ParticleFluidState &state,
                                  thrust::device_vector<unsigned char> &texture_data, int width,
                                  int height, float max_density);
void load_fluid_params(ParticleFluidState &state);
void save_fluid_params(ParticleFluidState &state);

} // namespace p2
