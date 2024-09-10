#include "ParticleFluid.cuh"



__host__ __device__
int particle_to_gid(float x, float y, int grid_width, float cell_size) {
  int grid_x = x / cell_size;
  int grid_y = y / cell_size;
  return grid_y * grid_width + grid_x;
}

__global__
void populate_grid_indices(float* x_particle, float* y_particle, int* grid_indices, int max_particles,
int* particles_per_cell, int max_particles_per_cell, int grid_width, int grid_height, float cell_size) {
  // this kernel runs once per particle

  // x_particle: N
  // y_particle: N
  // grid_indices: W * H * max_particles_per_cell
  // particles_per_cell: W * H

  int i = blockIdx.x * blockDim.x + threadIdx.x; // particle id

  if (i >= max_particles) {
    return;
  }

  float x = x_particle[i];
  float y = y_particle[i];

  // assume particle is in bounds. movement logic will wrap it around
  auto grid_index = particle_to_gid(x, y, grid_width, cell_size);

  // popluating grid indices, increment slot index
  int slot_index = atomicAdd(&particles_per_cell[grid_index], 1);
  if (slot_index < max_particles_per_cell) {
    grid_indices[grid_index * max_particles_per_cell + slot_index] = i;
  }
}


__global__
void reset_particles_per_cell(int* particles_per_cell, int particles_per_cell_length) {
  // we don't need to reset the indices, since they will be overwritten or invalid based on the number of particles in the cell
  // we do need to reset the particles per cell though:

  // this kernel runs once per cell

  int i = blockIdx.x * blockDim.x + threadIdx.x; // grid index

  if (i < particles_per_cell_length) {
    particles_per_cell[i] = 0;
  }
}

__global__
void compute_acceleration(float* x_particle, float* y_particle, float* x_velocity, float* y_velocity, 
float* x_acceleration, float* y_acceleration, int* grid_indices, int* particles_per_cell, int max_particles_per_cell,
float* x_pos_deriv_write, float* y_pos_deriv_write, float* x_vel_deriv_write, float* y_vel_deriv_write,
int max_particles, int grid_width, int grid_height, float cell_size) {
  // this kernel runs once per particle

  int i = blockIdx.x * blockDim.x + threadIdx.x; // particle id

  if (i >= max_particles) {
    return;
  }

  auto x = x_particle[i];
  auto y = y_particle[i];

  int x_cell = x / cell_size;
  int y_cell = y / cell_size;

  // assume particle is in bounds. movement logic will wrap it around
  auto grid_index = y_cell * grid_width + x_cell;

  // iterate neighbors
  int x_cell_min = max(0, x_cell - 1);
  int y_cell_min = max(0, y_cell - 1);
  int x_cell_max = min(grid_width - 1, x_cell + 1);
  int y_cell_max = min(grid_height - 1, y_cell + 1);

  // iterate through neighborhood
  for (y_cell = y_cell_min; y_cell <= y_cell_max; ++y_cell) {
    for (x_cell = x_cell_min; x_cell <= x_cell_max; ++x_cell) {
      // calculate current cell index, number of particles in cell
      int cell_index = y_cell * grid_width + x_cell;
      int particles_in_cell = particles_per_cell[cell_index];
      // iterate through particles in cell
      for (int slot_index = 0; slot_index < particles_in_cell; ++slot_index) {
        // TODO: document grid_indices size at the top of the function
        auto other_i = grid_indices[slot_index + cell_index * max_particles_per_cell];
        if (other_i == i) {
          continue; // skip ourself
        }

        auto x_other = x_particle[other_i];
        auto y_other = y_particle[other_i];

      }
    }
  }
  

}