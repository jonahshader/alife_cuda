#pragma once

#include "graphics/renderers/circle_renderer.cuh"
#include "soa_helper.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define FOR_PARTICLES(N, D)                                                                        \
  D(float, x_particle, 0)                                                                          \
  D(float, y_particle, 0)                                                                          \
  D(float, x_velocity, 0)                                                                          \
  D(float, y_velocity, 0)                                                                          \
  D(float, x_accel, 0)                                                                             \
  D(float, y_accel, 0)                                                                             \
  D(float, density, 0)

namespace particles {
DEFINE_STRUCTS(Particles, FOR_PARTICLES)

struct TunableParams {
  float pressure_multiplier = 225.0f;  // 4000.0f
  float viscosity_multiplier = 0.03f;  // 2.0f
  float target_density = 234.0f;       // 0.6f
  float particle_mass = 1.0f;          // 1.0f
  float gravity_acceleration = -13.0f; //-30.0f
  float drag = 0.000f;                 // 0.000f
};

// v1 ParticleGrid — not yet refactored to template-template
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

class ParticleFluid {
public:
  ParticleFluid(int width, int height, bool use_graphics);

  void update(float dt);
  void render(const glm::mat4 &transform);

private:
  TunableParams params{};
  ParticlesSoA<HostBuffer> particles{};
  ParticlesSoA<DeviceBuffer> particles_device{};

  ParticleGrid grid{};
  ParticleGridDevice grid_device{};

  std::unique_ptr<CircleRenderer> circle_renderer{};
};

} // namespace particles
