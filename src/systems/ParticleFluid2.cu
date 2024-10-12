#pragma once

constexpr float KERNEL_RADIUS = 16.0f;
constexpr float CELL_SIZE = KERNEL_RADIUS * 1;
constexpr int PARTICLES_PER_CELL = 64;

// given a particle's position, return the cell index it belongs to
__host__ __device__ int particle_to_cid(float x, float y, int grid_width)
{
  // TODO: wrap here or after position update?
  int grid_x = x / CELL_SIZE;
  int grid_y = y / CELL_SIZE;
  return grid_y * grid_width + grid_x;
}

// reset the particles_per_cell counters to 0
__global__ void reset_particles_per_cell(int *particles_per_cell, int grid_size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // grid index
  if (i < grid_size)
  {
    particles_per_cell[i] = 0;
  }
}

// put particle IDs into the cells they belong to
__global__ void populate_grid_indices(float *x_particle, float *y_particle, int *grid_indices, int max_particles,
                                      int *particles_per_cell, int max_particles_per_cell,
                                      int grid_width, int grid_height)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // particle id
  if (i >= max_particles)
  {
    return;
  }

  float x = x_particle[i];
  float y = y_particle[i];

  int grid_index = particle_to_cid(x, y, grid_width);

  int slot_index = atomicAdd(&particles_per_cell[grid_index], 1);
  if (slot_index < max_particles_per_cell) {
    grid_indices[grid_index * max_particles_per_cell + slot_index] = i;
  }

}

__host__ __device__ 
float smoothing_kernel(float radius, float dst)
{
  // TODO: the provided dst is calculated using sqrt, but we square it here...
  float value = max(0.0f, radius * radius - dst * dst);
  return value * value * value;
}

__host__ __device__
float calculate_density_at_pos(float x_pos, float y_pos, float *x, float *y, float *mass, int *grid_indices, int *particles_per_cell, int max_particles_per_cell,
                        int grid_width, int grid_height)
{
  float density = 0.0f;
  int grid_index = particle_to_cid(x_pos, y_pos, grid_width);
  int cell_x = grid_index % grid_width;
  int cell_y = grid_index / grid_width;

  // iterate through cell neighborhood
  for (int yi = cell_y - 1; yi <= cell_y + 1; yi++)
  {
    for (int xi = cell_x - 1; xi <= cell_x + 1; xi++)
    {
      // skip if cell is out of bounds
      if (xi < 0 || xi >= grid_width || yi < 0 || yi >= grid_height)
      {
        continue;
      }

      int neighbour_index = yi * grid_width + xi;
      int num_particles = particles_per_cell[neighbour_index];
      // iterate through particles within the cell
      for (int i = 0; i < num_particles; i++)
      {
        int particle_id = grid_indices[neighbour_index * max_particles_per_cell + i];
        float x_diff = x_pos - x[particle_id];
        float y_diff = y_pos - y[particle_id];
        float distance = sqrt(x_diff * x_diff + y_diff * y_diff);
        density += mass[particle_id] * smoothing_kernel(KERNEL_RADIUS, distance);
      }
    }
  }

  return density;
}

void procedure() {
  // move
  // resolve wall collisions
}