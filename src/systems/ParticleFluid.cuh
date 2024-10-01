#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "SoAHelper.h"
#include "graphics/renderers/CircleRenderer.h" // TODO: will need to be .cuh and support interop, similar to RectRenderer

#define FOR_PARTICLES(N, D) \
  D(float, x_particle, 0)   \
  D(float, y_particle, 0)   \
  D(float, x_velocity, 0)   \
  D(float, y_velocity, 0)   \
  D(float, x_accel, 0)      \
  D(float, y_accel, 0)      \
  D(float, density, 0)

namespace particles
{
  DEFINE_STRUCTS(Particles, FOR_PARTICLES)

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
    ParticleFluid(int width, int height);

    void update(float dt);

  private:
    ParticlesSoA particles{};
    ParticlesSoADevice particles_device{};

    ParticleGrid grid{};
    ParticleGridDevice grid_device{};
  };

} // namespace particles
