#pragma once

#include <glm/glm.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <cstdint>

#include "SoAHelper.h"
#include "graphics/renderers/CircleRenderer.cuh"

#define FOR_SPH(N, D)               \
  D(float2, pos, make_float2(0, 0)) \
  D(float2, vel, make_float2(0, 0)) \
  D(float2, acc, make_float2(0, 0)) \
  D(float, mass, 1)                 \
  D(float, density, 0)              \
  D(uint8_t, sym_break, 0)

namespace p2
{
  DEFINE_STRUCTS(SPH, FOR_SPH)

  struct TunableParams
  {
    // TODO: grab these from screenshot
    float dt = 1 / 600.0f; // 1/60.0f;
    float gravity = 0.0f; // -13.0f
    float collision_damping = 0.5f;
    float smoothing_radius = 0.2f;
    float target_density = 23.40f; // 234.0f
    float pressure_mult = 22.50f; // 225.0f;
    float near_pressure_mult = 18.0f;
    float viscosity_strength = 0.03f;

    int particles_per_cell = 64;
  };

  struct ParticleGrid
  {
    thrust::host_vector<int> grid_indices{};
    thrust::host_vector<int> particles_per_cell{};
    int width{0};
    int height{0};
    int max_particles_per_cell{4};

    void reconfigure(int new_width, int new_height, int new_max_particles_per_cell)
    {
      width = new_width;
      height = new_height;
      max_particles_per_cell = new_max_particles_per_cell;

      grid_indices = std::vector<int>(width * height * max_particles_per_cell);
      particles_per_cell = std::vector<int>(width * height);
    }
  };

  struct ParticleGridDevice
  {
    thrust::device_vector<int> grid_indices{};
    thrust::device_vector<int> particles_per_cell{};

    void copy_from_host(ParticleGrid &host)
    {
      grid_indices = host.grid_indices;
      particles_per_cell = host.particles_per_cell;
    }

    void copy_to_host(ParticleGrid &host)
    {
      host.grid_indices = grid_indices;
      host.particles_per_cell = particles_per_cell;
    }
  };

  struct ParticleGridPtrs
  {
    int *grid_indices;
    int *particles_per_cell;

    void get_ptrs(ParticleGridDevice &grid_device)
    {
      grid_indices = grid_device.grid_indices.data().get();
      particles_per_cell = grid_device.particles_per_cell.data().get();
    }

    void get_ptrs(ParticleGrid &grid)
    {
      grid_indices = grid.grid_indices.data();
      particles_per_cell = grid.particles_per_cell.data();
    }
  };

  class ParticleFluid
  {
  public:
    ParticleFluid(float width, float height, int particles_per_cell, bool use_graphics);

    void update();
    void render(const glm::mat4 &transform);

    void calculate_density_grid(thrust::device_vector<unsigned char> &texture_data, int width, int height, float max_density);

  private:
    float2 bounds;
    TunableParams params{};
    SPHSoA particles{};
    SPHSoADevice particles_device{};

    ParticleGrid grid{};
    ParticleGridDevice grid_device{};

    std::unique_ptr<CircleRenderer> circle_renderer{};
  };

} // namespace p2