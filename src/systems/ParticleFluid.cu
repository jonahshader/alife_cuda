#include "ParticleFluid.cuh"

#include "Kernels.cuh"

constexpr float PRESSURE_MULTIPLIER = 1200000.0f;
constexpr float VISCOSITY_MULTIPLIER = 8.0f;
constexpr float TARGET_PRESSURE = 2.0f;
constexpr float PARTICLE_MASS = 1.0f;
constexpr float GRAVITY_ACCELERATION = 108.0;
constexpr float WALL_ACCEL_PER_DIST = 6600.0f;

__host__ __device__ int particle_to_gid(float x, float y, int grid_width, float cell_size)
{
  int grid_x = x / cell_size;
  int grid_y = y / cell_size;
  return grid_y * grid_width + grid_x;
}

__global__ void reset_particles_per_cell(int *particles_per_cell, int particles_per_cell_length)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // grid index
  if (i < particles_per_cell_length)
  {
    particles_per_cell[i] = 0;
  }
}

__global__ void populate_grid_indices(float *x_particle, float *y_particle, int *grid_indices, int max_particles,
                                      int *particles_per_cell, int max_particles_per_cell, int grid_width, int grid_height, float cell_size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // particle id
  if (i >= max_particles)
  {
    return;
  }

  float x = x_particle[i];
  float y = y_particle[i];

  int grid_index = particle_to_gid(x, y, grid_width, cell_size);

  int slot_index = atomicAdd(&particles_per_cell[grid_index], 1);
  if (slot_index < max_particles_per_cell)
  {
    grid_indices[grid_index * max_particles_per_cell + slot_index] = i;
  }
}

__global__ void compute_density(float *x_particle, float *y_particle, float *density, int *grid_indices, int *particles_per_cell,
                                int max_particles_per_cell, int max_particles, int grid_width, int grid_height, float cell_size,
                                float kernel_radius, float kernel_vol_inv)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // particle id
  if (i >= max_particles)
  {
    return;
  }

  float x = x_particle[i];
  float y = y_particle[i];

  int x_cell = x / cell_size;
  int y_cell = y / cell_size;

  int x_cell_min = max(0, x_cell - 1);
  int y_cell_min = max(0, y_cell - 1);
  int x_cell_max = min(grid_width - 1, x_cell + 1);
  int y_cell_max = min(grid_height - 1, y_cell + 1);

  float density_i = 0.0f;

  // Compute density[i]
  for (int yc = y_cell_min; yc <= y_cell_max; ++yc)
  {
    for (int xc = x_cell_min; xc <= x_cell_max; ++xc)
    {
      int cell_index = yc * grid_width + xc;
      int particles_in_cell = particles_per_cell[cell_index];
      for (int slot_index = 0; slot_index < particles_in_cell; ++slot_index)
      {
        int other_i = grid_indices[slot_index + cell_index * max_particles_per_cell];
        float x_other = x_particle[other_i];
        float y_other = y_particle[other_i];

        float dx = x_other - x;
        float dy = y_other - y;
        float r = sqrtf(dx * dx + dy * dy);

        if (r >= kernel_radius)
          continue;

        float kernel_value = sharp_kernel(r, kernel_radius, kernel_vol_inv);
        density_i += PARTICLE_MASS * kernel_value;
      }
    }
  }

  // Assign computed density
  density[i] = density_i;
}

__global__ void compute_forces(float *x_particle, float *y_particle, float *x_velocity, float *y_velocity,
                               float *x_acceleration, float *y_acceleration, int *grid_indices, int *particles_per_cell, int max_particles_per_cell,
                               float *density, int max_particles, int grid_width, int grid_height, float cell_size,
                               float kernel_radius, float kernel_vol_inv)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // particle id
  if (i >= max_particles)
  {
    return;
  }

  float x = x_particle[i];
  float y = y_particle[i];

  int x_cell = x / cell_size;
  int y_cell = y / cell_size;

  int x_cell_min = max(0, x_cell - 1);
  int y_cell_min = max(0, y_cell - 1);
  int x_cell_max = min(grid_width - 1, x_cell + 1);
  int y_cell_max = min(grid_height - 1, y_cell + 1);

  float density_i = density[i];
  float pressure_i = (density_i - TARGET_PRESSURE) * PRESSURE_MULTIPLIER;

  // Compute pressure gradient and viscosity forces
  float pressure_grad_x = 0.0f;
  float pressure_grad_y = 0.0f;
  float viscosity_force_x = 0.0f;
  float viscosity_force_y = 0.0f;

  for (int yc = y_cell_min; yc <= y_cell_max; ++yc)
  {
    for (int xc = x_cell_min; xc <= x_cell_max; ++xc)
    {
      int cell_index = yc * grid_width + xc;
      int particles_in_cell = particles_per_cell[cell_index];
      for (int slot_index = 0; slot_index < particles_in_cell; ++slot_index)
      {
        int other_i = grid_indices[slot_index + cell_index * max_particles_per_cell];
        if (other_i == i)
          continue;

        float x_other = x_particle[other_i];
        float y_other = y_particle[other_i];

        float dx = x_other - x;
        float dy = y_other - y;
        float r = sqrtf(dx * dx + dy * dy);

        if (r >= kernel_radius || r < 1e-6f)
          continue;

        float dir_x = dx / r;
        float dir_y = dy / r;

        float density_j = density[other_i];
        float pressure_j = (density_j - TARGET_PRESSURE) * PRESSURE_MULTIPLIER;
        float shared_pressure = (pressure_i + pressure_j) * 0.5f;
        float kernel_derivative = sharp_kernel_derivative(r, kernel_radius, kernel_vol_inv);

        pressure_grad_x += PARTICLE_MASS * shared_pressure * kernel_derivative * dir_x / density_j;
        pressure_grad_y += PARTICLE_MASS * shared_pressure * kernel_derivative * dir_y / density_j;

        float influence = smoothstep_kernel(r, kernel_radius, kernel_vol_inv);
        float vx_i = x_velocity[i];
        float vy_i = y_velocity[i];
        float vx_j = x_velocity[other_i];
        float vy_j = y_velocity[other_i];

        viscosity_force_x += (vx_j - vx_i) * influence;
        viscosity_force_y += (vy_j - vy_i) * influence;
      }
    }
  }

  // Compute total acceleration
  float acc_x = -pressure_grad_x / density_i + VISCOSITY_MULTIPLIER * viscosity_force_x / density_i;
  float acc_y = -pressure_grad_y / density_i + VISCOSITY_MULTIPLIER * viscosity_force_y / density_i + GRAVITY_ACCELERATION;

  // Wall accelerations
  if (x < 0.0f)
  {
    acc_x += WALL_ACCEL_PER_DIST * -x;
  }
  else if (x > (grid_width * cell_size))
  {
    acc_x += WALL_ACCEL_PER_DIST * ((grid_width * cell_size) - x);
  }
  if (y < 0.0f)
  {
    acc_y += WALL_ACCEL_PER_DIST * -y;
  }
  else if (y > (grid_height * cell_size))
  {
    acc_y += WALL_ACCEL_PER_DIST * ((grid_height * cell_size) - y);
  }

  // Write accelerations
  x_acceleration[i] = acc_x;
  y_acceleration[i] = acc_y;
}

__global__ void update_positions_velocities(float *x_particle, float *y_particle, float *x_velocity, float *y_velocity,
                                            float *x_acceleration, float *y_acceleration, int max_particles,
                                            float dt, float bounds_x, float bounds_y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // particle id
  if (i >= max_particles)
  {
    return;
  }

  x_velocity[i] += dt * x_acceleration[i];
  y_velocity[i] += dt * y_acceleration[i];

  x_particle[i] += dt * x_velocity[i];
  y_particle[i] += dt * y_velocity[i];

  // Wrap positions (assuming periodic boundary conditions)
  if (x_particle[i] < 0.0f)
  {
    x_particle[i] += bounds_x;
  }
  else if (x_particle[i] >= bounds_x)
  {
    x_particle[i] -= bounds_x;
  }

  if (y_particle[i] < 0.0f)
  {
    y_particle[i] += bounds_y;
  }
  else if (y_particle[i] >= bounds_y)
  {
    y_particle[i] -= bounds_y;
  }
}

namespace particles
{

void update() {
  
}

}