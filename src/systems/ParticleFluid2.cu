#include "ParticleFluid2.cuh"

#include <random>
#include <cmath>
#include <curand_kernel.h>

#include "float2_ops.cuh"
#include "CustomMath.cuh"
#include "imgui.h"

#include "systems/TimingProfiler.cuh"

namespace p2 {

void check_cuda(const std::string &msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "ParticleFluid2: " << msg << ": " << cudaGetErrorString(err) << std::endl;
  }
}

// given a particle's position, return the cell index it belongs to
__host__ __device__ int particle_to_cid(float2 pos, int grid_width, float cell_size) {
  int grid_x = pos.x / cell_size;
  int grid_y = pos.y / cell_size;
  return grid_y * grid_width + grid_x;
}

// reset the particles_per_cell counters to 0
__global__ void reset_particles_per_cell(int *particles_per_cell, size_t grid_size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x; // grid index
  if (i < grid_size) {
    particles_per_cell[i] = 0;
  }
}

// put particle IDs into the cells they belong to
__global__ void populate_grid_indices(SPHPtrs sph, ParticleGridPtrs grid, int max_particles,
                                      int max_particles_per_cell, int2 p_grid_dims,
                                      float cell_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // particle id
  if (i >= max_particles)
    return;

  int grid_index = particle_to_cid(sph.pos[i], p_grid_dims.x, cell_size);
  int slot_index = atomicAdd(&grid.particles_per_cell[grid_index], 1);
  if (slot_index < max_particles_per_cell)
    grid.grid_indices[grid_index * max_particles_per_cell + slot_index] = i;
}

// TODO: the provided dst is calculated using sqrt, but we square it here...
__device__ float viscosity_kernel(float radius, float dst) {
  float q = dst / radius;
  if (q > 1.0f)
    return 0.0f;

  const float normalization_factor_2d = 4.0f / (M_PI_F * powf(radius, 8));

  float value = radius * radius - dst * dst;
  return normalization_factor_2d * value * value * value;
}
// gradient of the smoothing kernel
__device__ float2 viscosity_kernel_gradient(float radius, float2 diff) {
  float dst2 = length2(diff);
  float dst = sqrtf(dst2);
  float q = dst / radius;
  if (q > 1.0f)
    return make_float2(0.0f, 0.0f);

  const float normalization_factor_2d = 4.0f / (M_PI_F * powf(radius, 8));

  float value = radius * radius - dst2;
  return -6.0f * normalization_factor_2d * diff * value * value;
}

__host__ __device__ float density_kernel(float radius, float dst) {
  if (dst >= radius)
    return 0;

  float normalization_factor_2d = 6.0f / (M_PI_F * powf(radius, 4));
  float value = radius - dst;
  return normalization_factor_2d * value * value;
}

__host__ __device__ float2 density_kernel_gradient(float radius, float2 diff) {
  float dst = length(diff);
  float2 grad = make_float2(0.0f, 0.0f);
  if (dst < radius && dst > 1e-5) {
    float normalization_factor_2d = 6.0f / (M_PI_F * powf(radius, 4));
    float value = radius - dst;
    grad = -2.0f * normalization_factor_2d * diff * value / dst;
  }

  return grad;
}

__host__ __device__ float near_density_kernel(float radius, float dst) {
  if (dst >= radius)
    return 0;

  float normalization_factor_2d = 20.0f / (M_PI_F * powf(radius, 5));
  float value = radius - dst;
  return normalization_factor_2d * value * value * value;
}

__host__ __device__ float2 near_density_kernel_gradient(float radius, float2 diff) {
  float dst = length(diff);
  float2 grad = make_float2(0.0f, 0.0f);
  if (dst < radius && dst > 1e-5) {
    float normalization_factor_2d = 20.0f / (M_PI_F * powf(radius, 5));
    float value = radius - dst;
    grad = -3.0f * normalization_factor_2d * diff * value * value / dst;
  }

  return grad;
}

__host__ __device__ float2 calculate_density_at_pos(float2 pos, SPHPtrs sph, ParticleGridPtrs grid,
                                                    int max_particles_per_cell,
                                                    int2 particle_grid_dims, float cell_size,
                                                    float smoothing_radius, float2 bounds) {
  float density = 0.0f;
  float near_density = 0.0f;
  int grid_index = particle_to_cid(pos, particle_grid_dims.x, cell_size);
  int cell_x = grid_index % particle_grid_dims.x;
  int cell_y = grid_index / particle_grid_dims.x;

  int xi_neg_dist = cell_x == 0 ? 2 : 1;
  int xi_pos_dist = cell_x >= particle_grid_dims.x - 2 ? 2 : 1;

  // iterate through cell neighborhood
  for (int yi = cell_y - 1; yi <= cell_y + 1; yi++) {
    for (int xi = cell_x - xi_neg_dist; xi <= cell_x + xi_pos_dist; xi++) // TEMP
    {
      // skip if cell is out of vertical bounds
      if (yi < 0 || yi >= particle_grid_dims.y)
        continue;
      // wrap x if out of horizontal bounds
      int wrapped_x = (xi + particle_grid_dims.x) % particle_grid_dims.x;

      int neighbour_index = yi * particle_grid_dims.x + wrapped_x;
      int num_particles = min(grid.particles_per_cell[neighbour_index], max_particles_per_cell);
      // iterate through particles within the cell
      for (int i = 0; i < num_particles; i++) {
        int particle_id = grid.grid_indices[neighbour_index * max_particles_per_cell + i];
        float2 other_pos = sph.pos[particle_id];
        if (xi < 0) // wrap around
          other_pos.x -= bounds.x;
        else if (xi >= particle_grid_dims.x)
          other_pos.x += bounds.x;
        float distance = length(pos - other_pos);
        density += sph.mass[particle_id] * density_kernel(smoothing_radius, distance);
        near_density += sph.mass[particle_id] * near_density_kernel(smoothing_radius, distance);
      }
    }
  }

  return make_float2(density, near_density);
}

__host__ __device__ float2 calculate_soil_density_at_pos(float2 pos, SoilPtrs soil,
                                                         SoilParticlesPtrs soil_particles, int w,
                                                         int h, float soil_size,
                                                         float particle_radius) {
  float density = 0.0f;
  float near_density = 0.0f;

  // fake particles are at the center of each cell
  int cell_x = pos.x / soil_size;
  int cell_y = pos.y / soil_size;

  int range = ceil(particle_radius / soil_size);

  // iterate through soil cell neighborhood
  for (int yi = cell_y - range; yi <= cell_y + range; yi++) {
    for (int xi = cell_x - range; xi <= cell_x + range; xi++) {
      // skip if cell is out of vertical bounds
      if (yi < 0 || yi >= h || xi < 0 || xi >= w)
        continue;
      // wrap x if out of horizontal bounds
      int wrapped_x = (xi + w) % w; // TODO: might have issues when particle grid and soil grid
                                    // don't align

      int soil_index = yi * w + wrapped_x;
      // TODO: currently using one fake particle in the center of the cell,
      // but we could use more to get a better approximation
      // float2 other_pos = make_float2((xi + 0.5f) * soil_size, (yi + 0.5f) * soil_size);
      float other_x_offset = soil_particles.x_offset[soil_index];
      float other_y_offset = soil_particles.y_offset[soil_index];
      float2 other_pos =
          make_float2((xi + other_x_offset) * soil_size, (yi + other_y_offset) * soil_size);

      float distance = length(pos - other_pos);
      float soil_cell_mass = (soil.sand_density[soil_index] * SAND_ABSOLUTE_DENSITY +
                              soil.silt_density[soil_index] * SILT_ABSOLUTE_DENSITY +
                              soil.clay_density[soil_index] * CLAY_ABSOLUTE_DENSITY) *
                             soil_size * soil_size; // convert to mass

      density += soil_cell_mass * density_kernel(particle_radius, distance);
      near_density += soil_cell_mass * near_density_kernel(particle_radius, distance);
    }
  }

  return make_float2(density, near_density);
}

// calculate the density of each particle, with contributions from other particles and soil
// particles
__global__ void calculate_particle_density(SPHPtrs sph, ParticleGridPtrs grid,
                                           int max_particles_per_cell, int2 particle_grid_dims,
                                           float cell_size, float smoothing_radius,
                                           size_t num_particles, float2 bounds, SoilPtrs soil_read,
                                           SoilParticlesPtrs soil_ptrs, int soil_w, int soil_h,
                                           float soil_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particles)
    return;
  float2 density_from_p =
      calculate_density_at_pos(sph.pos[i], sph, grid, max_particles_per_cell, particle_grid_dims,
                               cell_size, smoothing_radius, bounds);
  float2 density_from_soil = calculate_soil_density_at_pos(sph.pos[i], soil_read, soil_ptrs, soil_w,
                                                           soil_h, soil_size, smoothing_radius);
  sph.density[i] = density_from_p.x + density_from_soil.x;
  sph.near_density[i] = density_from_p.y + density_from_soil.y;
}

// calculate the density of each particle, with contributions from other particles
__global__ void calculate_particle_density(SPHPtrs sph, ParticleGridPtrs grid,
                                           int max_particles_per_cell, int2 particle_grid_dims,
                                           float cell_size, float smoothing_radius,
                                           size_t num_particles, float2 bounds) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particles)
    return;
  float2 density_from_p =
      calculate_density_at_pos(sph.pos[i], sph, grid, max_particles_per_cell, particle_grid_dims,
                               cell_size, smoothing_radius, bounds);
  sph.density[i] = density_from_p.x;
  sph.near_density[i] = density_from_p.y;
}

// calculate the density of each soil particle, with contributions from other particles and soil
// particles
__global__ void calculate_soil_density(SPHPtrs sph, SoilParticlesPtrs soil_ptrs,
                                       ParticleGridPtrs grid, int max_particles_per_cell,
                                       int2 particle_grid_dims, float cell_size,
                                       float smoothing_radius, size_t num_particles, float2 bounds,
                                       SoilPtrs soil_read, int soil_w, int soil_h,
                                       float soil_size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particles)
    return;

  // calculate the density of each soil particle
  // pos of the soil particle is the center of the cell
  int cell_x = i % soil_w;
  int cell_y = i / soil_w;
  // TODO: eventually we might have >1 particles per cell,
  // then we need to define a function to determine where the particles are
  // since it would no longer be trivial.
  // float2 pos = make_float2((cell_x + 0.5f) * soil_size, (cell_y + 0.5f) * soil_size);
  float x_offset = soil_ptrs.x_offset[i];
  float y_offset = soil_ptrs.y_offset[i];
  float2 pos = make_float2((cell_x + x_offset) * soil_size, (cell_y + y_offset) * soil_size);
  float2 density_from_p =
      calculate_density_at_pos(pos, sph, grid, max_particles_per_cell, particle_grid_dims,
                               cell_size, smoothing_radius, bounds);
  float2 density_from_soil = calculate_soil_density_at_pos(pos, soil_read, soil_ptrs, soil_w,
                                                           soil_h, soil_size, smoothing_radius);
  soil_ptrs.density[i] = density_from_p.x + density_from_soil.x;
  soil_ptrs.near_density[i] = density_from_p.y + density_from_soil.y;
}

__host__ __device__ float calculate_pressure(float density, const TunableParams &params) {
  // TODO: this mostly describes gas, not liquid. better options might exist.
  return params.pressure_mult * (density - params.target_density);
}

__host__ __device__ float calculate_near_pressure(float near_density, const TunableParams &params) {
  return params.near_pressure_mult * near_density;
}

__host__ __device__ float2 sym_break_to_dir(uint8_t sym_break) {
  float angle = sym_break / 255.0f * 2.0f * M_PI_F;
  return make_float2(cosf(angle), sinf(angle));
}

__global__ void calculate_accel(SPHPtrs sph, ParticleGridPtrs grid, int max_particles_per_cell,
                                int2 particle_grid_dims, float cell_size, TunableParams params,
                                size_t particle_count, float2 bounds) {
  size_t pid = blockIdx.x * blockDim.x + threadIdx.x; // particle id
  if (pid >= particle_count)
    return;

  auto pos = sph.pos[pid];
  auto vel = sph.vel[pid];
  auto density = sph.density[pid];
  auto near_density = sph.near_density[pid];
  int grid_index = particle_to_cid(pos, particle_grid_dims.x, cell_size);
  int cell_x = grid_index % particle_grid_dims.x;
  int cell_y = grid_index / particle_grid_dims.x;

  float pressure = calculate_pressure(density, params);
  float near_pressure = calculate_near_pressure(near_density, params);
  float2 pressure_force = make_float2(0.0f, 0.0f);
  float2 viscosity_force = make_float2(0.0f, 0.0f);

  int xi_neg_dist = cell_x == 0 ? 2 : 1;
  int xi_pos_dist = cell_x >= particle_grid_dims.x - 2 ? 2 : 1;

  // iterate through cell neighborhood
  for (int yi = cell_y - 1; yi <= cell_y + 1; yi++) {
    for (int xi = cell_x - xi_neg_dist; xi <= cell_x + xi_pos_dist; xi++) {
      // skip if cell is out of vertical bounds
      if (yi < 0 || yi >= particle_grid_dims.y)
        continue;
      // wrap x if out of horizontal bounds
      int wrapped_x = (xi + particle_grid_dims.x) % particle_grid_dims.x;

      int neighbour_index = yi * particle_grid_dims.x + wrapped_x;
      int num_particles = min(grid.particles_per_cell[neighbour_index], max_particles_per_cell);
      // iterate through particles within the cell
      for (int i = 0; i < num_particles; i++) {
        int particle_id = grid.grid_indices[neighbour_index * max_particles_per_cell + i];
        float2 other_pos = sph.pos[particle_id];
        if (xi < 0) // wrap around
          other_pos.x -= bounds.x;
        else if (xi >= particle_grid_dims.x)
          other_pos.x += bounds.x;
        float other_density = sph.density[particle_id];
        float other_near_density = sph.near_density[particle_id];
        float other_pressure = calculate_pressure(other_density, params);
        float other_near_pressure = calculate_near_pressure(other_near_density, params);
        float other_mass = sph.mass[particle_id];
        pressure_force = pressure_force -
                         other_mass *
                             ((pressure + other_pressure + near_pressure + other_near_pressure) /
                              (4.0f * other_density)) *
                             density_kernel_gradient(params.smoothing_radius, pos - other_pos);

        // viscosity
        // TODO: distance is calculated twice. once here and once above.
        if (particle_id != pid) {
          float2 diff = sph.vel[particle_id] - vel;
          float dist = length(pos - other_pos);
          float influence = viscosity_kernel(params.smoothing_radius, dist);
          viscosity_force = viscosity_force + influence * diff; // scale with mass?
        }
      }
    }
  }

  // apply pressure force
  // float2 acc = make_float2(0.0f, params.gravity) + pressure_force / density;
  float2 acc = make_float2(0.0f, params.gravity) +
               (pressure_force + viscosity_force * params.viscosity_strength) / density;
  sph.vel[pid] = vel + acc * params.dt;
}

// calculate the acceleration of each particle, with contributions from other particles and soil
// particles
__global__ void calculate_accel(SPHPtrs sph, SoilParticlesPtrs soil_ptrs, ParticleGridPtrs grid,
                                int max_particles_per_cell, int2 particle_grid_dims,
                                float cell_size, TunableParams params, size_t particle_count,
                                float2 bounds, SoilPtrs soil_read, int soil_w, int soil_h,
                                float soil_size) {
  size_t pid = blockIdx.x * blockDim.x + threadIdx.x; // particle id
  if (pid >= particle_count)
    return;

  auto pos = sph.pos[pid];
  auto vel = sph.vel[pid];
  auto density = sph.density[pid];
  auto near_density = sph.near_density[pid];
  int grid_index = particle_to_cid(pos, particle_grid_dims.x, cell_size);
  int cell_x = grid_index % particle_grid_dims.x;
  int cell_y = grid_index / particle_grid_dims.x;

  float pressure = calculate_pressure(density, params);
  float near_pressure = calculate_near_pressure(near_density, params);
  float2 pressure_force = make_float2(0.0f, 0.0f);
  float2 viscosity_force = make_float2(0.0f, 0.0f);

  int xi_neg_dist = cell_x == 0 ? 2 : 1;
  int xi_pos_dist = cell_x >= particle_grid_dims.x - 2 ? 2 : 1;

  // iterate through cell neighborhood
  for (int yi = cell_y - 1; yi <= cell_y + 1; yi++) {
    for (int xi = cell_x - xi_neg_dist; xi <= cell_x + xi_pos_dist; xi++) {
      // skip if cell is out of vertical bounds
      if (yi < 0 || yi >= particle_grid_dims.y)
        continue;
      // wrap x if out of horizontal bounds
      int wrapped_x = (xi + particle_grid_dims.x) % particle_grid_dims.x;

      int neighbour_index = yi * particle_grid_dims.x + wrapped_x;
      int num_particles = min(grid.particles_per_cell[neighbour_index], max_particles_per_cell);
      // iterate through particles within the cell
      for (int i = 0; i < num_particles; i++) {
        int particle_id = grid.grid_indices[neighbour_index * max_particles_per_cell + i];
        float2 other_pos = sph.pos[particle_id];
        if (xi < 0) // wrap around
          other_pos.x -= bounds.x;
        else if (xi >= particle_grid_dims.x)
          other_pos.x += bounds.x;
        float other_density = sph.density[particle_id];
        float other_near_density = sph.near_density[particle_id];
        float other_pressure = calculate_pressure(other_density, params);
        float other_near_pressure = calculate_near_pressure(other_near_density, params);
        float other_mass = sph.mass[particle_id];
        pressure_force = pressure_force -
                         other_mass *
                             ((pressure + other_pressure + near_pressure + other_near_pressure) /
                              (4.0f * other_density)) *
                             density_kernel_gradient(params.smoothing_radius, pos - other_pos);

        // viscosity
        // TODO: distance is calculated twice. once here and once above.
        if (particle_id != pid) {
          float2 diff = sph.vel[particle_id] - vel;
          float dist = length(pos - other_pos);
          float influence = viscosity_kernel(params.smoothing_radius, dist);
          viscosity_force = viscosity_force + influence * diff; // scale with mass?
        }
      }
    }
  }

  // iterate through soil cell neighborhood
  int soil_range = ceil(params.smoothing_radius / soil_size);
  cell_x = pos.x / soil_size;
  cell_y = pos.y / soil_size;
  for (int yi = cell_y - soil_range; yi <= cell_y + soil_range; yi++) {
    for (int xi = cell_x - soil_range; xi <= cell_x + soil_range; xi++) {
      // skip if cell is out of vertical bounds
      if (yi < 0 || yi >= soil_h || xi < 0 || xi >= soil_w)
        continue;
      // wrap x if out of horizontal bounds
      int wrapped_x = (xi + soil_w) % soil_w;

      int soil_index = yi * soil_w + wrapped_x;
      // TODO: currently using one fake particle in the center of the cell,
      // but we could use more to get a better approximation
      // float2 other_pos = make_float2((xi + 0.5f) * soil_size, (yi + 0.5f) * soil_size);
      float other_x_offset = soil_ptrs.x_offset[soil_index];
      float other_y_offset = soil_ptrs.y_offset[soil_index];
      float2 other_pos =
          make_float2((xi + other_x_offset) * soil_size, (yi + other_y_offset) * soil_size);

      float distance = length(pos - other_pos);
      float soil_cell_mass = (soil_read.sand_density[soil_index] * SAND_ABSOLUTE_DENSITY +
                              soil_read.silt_density[soil_index] * SILT_ABSOLUTE_DENSITY +
                              soil_read.clay_density[soil_index] * CLAY_ABSOLUTE_DENSITY) *
                             soil_size * soil_size; // convert to mass
      float other_density = soil_ptrs.density[soil_index];
      float other_near_density = soil_ptrs.near_density[soil_index];
      float other_pressure = calculate_pressure(other_density, params);
      float other_near_pressure = calculate_near_pressure(other_near_density, params);
      if (other_density > 0.01) {
        pressure_force = pressure_force -
                         soil_cell_mass *
                             ((pressure + other_pressure + near_pressure + other_near_pressure) /
                              (4.0f * other_density)) *
                             density_kernel_gradient(params.smoothing_radius, pos - other_pos);

        // viscosity
        // TODO: distance is calculated twice. once here and once above.
        // TODO: define viscosity modifiers for soil types
        // for now, just disable when we have a low density
        // (pull out of if statement when we have viscosity modifiers)
        float2 diff = make_float2(0.0f, 0.0f) - vel; // soil particles are stationary
        float influence = viscosity_kernel(params.smoothing_radius, distance);
        // viscosity_force = viscosity_force + influence * diff; // scale with mass?
        viscosity_force = viscosity_force + 2 * diff;
      }
    }
  }

  // apply pressure force
  // float2 acc = make_float2(0.0f, params.gravity) + pressure_force / density;
  float2 acc = make_float2(0.0f, params.gravity) +
               (pressure_force + viscosity_force * params.viscosity_strength) / density;
  sph.vel[pid] = vel + acc * params.dt;
}

__host__ __device__ void attract_particles_at_pos(float2 pos, float max_thrust, float radius,
                                                  SPHPtrs sph, ParticleGridPtrs grid, int max_ppc,
                                                  int2 p_grid_dims, float c_size, float2 bounds) {
  int grid_index = particle_to_cid(pos, p_grid_dims.x, c_size);
  int cell_x = grid_index % p_grid_dims.x;
  int cell_y = grid_index / p_grid_dims.x;

  // compute neighborhood size from c_size and radius
  int cell_radius = ceil(radius / c_size);
  // need additional overlap to account for bounds that
  // are indivisible by c_size. TODO: do this for y too?
  int xi_neg_dist = (cell_x == 0 ? 1 : 0) + cell_radius;
  int xi_pos_dist = (cell_x >= p_grid_dims.x - 2 ? 1 : 0) + cell_radius;

  // iterate through cell neighborhood
  for (int yi = cell_y - cell_radius; yi <= cell_y + cell_radius; yi++) {
    for (int xi = cell_x - xi_neg_dist; xi <= cell_x + xi_pos_dist; xi++) {
      // skip if cell is out of vertical bounds
      if (yi < 0 || yi >= p_grid_dims.y)
        continue;
      // wrap x if out of horizontal bounds
      int wrapped_x = (xi + p_grid_dims.x) % p_grid_dims.x;

      int neighbour_index = yi * p_grid_dims.x + wrapped_x;
      int num_particles = min(grid.particles_per_cell[neighbour_index], max_ppc);
      // iterate through particles within the cell
      for (int i = 0; i < num_particles; i++) {
        int particle_id = grid.grid_indices[neighbour_index * max_ppc + i];
        float2 other_pos = sph.pos[particle_id];
        float2 other_vel = sph.vel[particle_id];
        if (xi < 0) // wrap around
          other_pos.x -= bounds.x;
        else if (xi >= p_grid_dims.x)
          other_pos.x += bounds.x;
        float distance = length(pos - other_pos);
        if (distance < radius) {
          float2 dir = normalize(pos - other_pos);
          sph.vel[particle_id] = other_vel + dir * max_thrust;
        }
      }
    }
  }
}

// attract towards a single point. run on a single thread
__global__ void attract_single_kernel(float2 pos, float max_thrust, float radius, SPHPtrs sph,
                                      ParticleGridPtrs grid, int max_ppc, int2 p_grid_dims,
                                      float c_size, float2 bounds) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > 0)
    return;
  attract_particles_at_pos(pos, max_thrust, radius, sph, grid, max_ppc, p_grid_dims, c_size,
                           bounds);
}

void ParticleFluid::attract(float2 pos, float max_thrust, float radius) {
  SPHPtrs sph;
  ParticleGridPtrs grid_ptrs;
  sph.get_ptrs(particles_device);
  grid_ptrs.get_ptrs(grid_device);

  int num_particles = particles_device.pos.size();
  int grid_size = grid.width * grid.height;
  int2 grid_dims = make_int2(grid.width, grid.height);
  float cell_size = params.smoothing_radius;

  dim3 block(1);
  dim3 grid_dim(1);

  attract_single_kernel<<<grid_dim, block>>>(pos, max_thrust, radius, sph, grid_ptrs,
                                             grid.max_particles_per_cell, grid_dims, cell_size,
                                             bounds);

  check_cuda("attract_single_kernel");
}

// signed distance function for a box
// https://iquilezles.org/articles/distfunctions/
__host__ __device__ float sd_box(float2 p, float2 b) {
  float2 d = abs(p) - b;
  return fminf(fmaxf(d.x, d.y), 0.0f) + length(max(d, make_float2(0.0f, 0.0f)));
}

void ParticleFluid::init_grid() {
  // configure grid
  int grid_width = std::ceil(bounds.x / params.smoothing_radius);
  int grid_height = std::ceil(bounds.y / params.smoothing_radius);

  grid.reconfigure(grid_width, grid_height, params.max_particles_per_cell);
}

void ParticleFluid::init() {
  init_grid();
  // init some particles
  std::default_random_engine rand;
  std::uniform_real_distribution<float> dist_x(0.0f, bounds.x);
  std::uniform_real_distribution<float> dist_y(bounds.y / 2, bounds.y);

  // velocity init with gaussian
  std::normal_distribution<float> dist_vel(0.0f, 0.001f);

  // sym_break, random uint8_t
  std::uniform_int_distribution<int> dist_sym(0, 255);

  int grid_width = std::ceil(bounds.x / params.smoothing_radius);
  int grid_height = std::ceil(bounds.y / params.smoothing_radius);
  const int NUM_PARTICLES = params.particles_per_cell * grid_width * grid_height;
  particles.resize_all(NUM_PARTICLES);

  for (int i = 0; i < NUM_PARTICLES; ++i)
    particles.vel[i] = make_float2(dist_vel(rand), dist_vel(rand));

  for (int i = 0; i < NUM_PARTICLES; ++i) {
    auto pos = make_float2(dist_x(rand), dist_y(rand));
    auto vel = particles.vel[i];
    particles.pos[i] = pos + vel * params.dt_predict;
    particles.ppos[i] = pos;
  }

  // density is already defaulted to 0, mass to 1
  for (int i = 0; i < NUM_PARTICLES; ++i)
    particles.sym_break[i] = dist_sym(rand);

  // send to device
  particles_device.copy_from_host(particles);
  grid_device.copy_from_host(grid);
}

ParticleFluid::ParticleFluid(float width, float height, const TunableParams &params,
                             bool use_graphics)
    : bounds(make_float2(width, height)), params(params) {
  if (use_graphics)
    circle_renderer = std::make_unique<CircleRenderer>();

  init();
}

// TODO: i think this constructor doesn't work as expected, because load_params is called after
// the other constructor is called
ParticleFluid::ParticleFluid(float width, float height, bool use_graphics)
    : ParticleFluid(width, height, TunableParams{}, use_graphics) {
  load_params();
}

void ParticleFluid::load_params() {
  if (pm == nullptr)
    pm = std::make_unique<ParameterManager>("fluid2_params.txt");
  pm->get<float>("dt", params.dt);
  pm->get<float>("dt_predict", params.dt_predict);
  pm->get<float>("gravity", params.gravity);
  pm->get<float>("collision_damping", params.collision_damping);
  pm->get<float>("smoothing_radius", params.smoothing_radius);
  pm->get<float>("target_density", params.target_density);
  pm->get<float>("pressure_mult", params.pressure_mult);
  pm->get<float>("near_pressure_mult", params.near_pressure_mult);
  pm->get<float>("viscosity_strength", params.viscosity_strength);
  pm->get<int>("particles_per_cell", params.particles_per_cell);
  pm->get<int>("max_particles_per_cell", params.max_particles_per_cell);
}

void ParticleFluid::save_params() {
  if (pm == nullptr)
    return;
  pm->set<float>("dt", params.dt);
  pm->set<float>("dt_predict", params.dt_predict);
  pm->set<float>("gravity", params.gravity);
  pm->set<float>("collision_damping", params.collision_damping);
  pm->set<float>("smoothing_radius", params.smoothing_radius);
  pm->set<float>("target_density", params.target_density);
  pm->set<float>("pressure_mult", params.pressure_mult);
  pm->set<float>("near_pressure_mult", params.near_pressure_mult);
  pm->set<float>("viscosity_strength", params.viscosity_strength);
  pm->set<int>("particles_per_cell", params.particles_per_cell);
  pm->set<int>("max_particles_per_cell", params.max_particles_per_cell);
  pm->save();
}

__global__ void move_particles(SPHPtrs sph, float dt, float dt_predict, size_t num_particles,
                               float2 bounds, float damping) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particles)
    return;

  auto vel = sph.vel[i];
  // use previous position to calculate new position
  float2 new_pos = sph.ppos[i] + vel * dt;
  // wrap around
  new_pos.x = fmodf(new_pos.x + bounds.x, bounds.x);

  // reflect y over bounds
  if (new_pos.y < 0.0f) {
    new_pos.y = -new_pos.y;
    sph.vel[i].y = -vel.y * damping;
  } else if (new_pos.y > bounds.y) {
    new_pos.y = (2.0f * bounds.y - new_pos.y) - 1e-4f;
    sph.vel[i].y = -vel.y * damping;
  }

  float2 new_pos2 = new_pos + vel * dt_predict;
  // wrap around
  new_pos2.x = fmodf(new_pos2.x + bounds.x, bounds.x);

  // reflect y over bounds. don't manipulate vel here
  if (new_pos2.y < 0.0f) {
    new_pos2.y = -new_pos2.y;
  } else if (new_pos2.y > bounds.y) {
    new_pos2.y = (2.0f * bounds.y - new_pos2.y) - 1e-4f;
  }

  sph.ppos[i] = new_pos;
  sph.pos[i] = new_pos2;
}

void ParticleFluid::update() {
  auto &profiler = TimingProfiler::getInstance();
  SPHPtrs sph;
  ParticleGridPtrs grid_ptrs;
  sph.get_ptrs(particles_device);
  grid_ptrs.get_ptrs(grid_device);

  size_t num_particles = particles_device.pos.size();
  size_t grid_size = grid.width * grid.height;
  int2 grid_dims = make_int2(grid.width, grid.height);
  float cell_size = params.smoothing_radius;

  dim3 sph_block(256);
  dim3 sph_grid_dim((num_particles + sph_block.x - 1) / sph_block.x);

  dim3 grid_block(256);
  dim3 grid_grid_dim((grid_size + grid_block.x - 1) / grid_block.x);

  // place in grid
  {
    auto scope = profiler.scopedMeasure("reset_particles_per_cell");
    reset_particles_per_cell<<<grid_grid_dim, grid_block>>>(grid_ptrs.particles_per_cell,
                                                            grid_size);
    check_cuda("reset_particles_per_cell");
  }

  {
    auto scope = profiler.scopedMeasure("populate_grid_indices");
    populate_grid_indices<<<sph_grid_dim, sph_block>>>(sph, grid_ptrs, num_particles,
                                                       grid.max_particles_per_cell, grid_dims,
                                                       params.smoothing_radius);
    check_cuda("populate_grid_indices");
  }

  // calculate density

  {
    auto scope = profiler.scopedMeasure("calculate_particle_density");
    calculate_particle_density<<<sph_grid_dim, sph_block>>>(
        sph, grid_ptrs, grid.max_particles_per_cell, grid_dims, cell_size, params.smoothing_radius,
        num_particles, bounds);
    check_cuda("calculate_particle_density");
  }
  // calculate acceleration
  {
    auto scope = profiler.scopedMeasure("calculate_accel");
    calculate_accel<<<sph_grid_dim, sph_block>>>(sph, grid_ptrs, grid.max_particles_per_cell,
                                                 grid_dims, cell_size, params, num_particles,
                                                 bounds);
    check_cuda("calculate_accel");
  }

  // move particles
  {
    auto scope = profiler.scopedMeasure("move_particles");
    move_particles<<<sph_grid_dim, sph_block>>>(sph, params.dt, params.dt_predict, num_particles,
                                                bounds, params.collision_damping);
    check_cuda("move_particles");
  }
}

void ParticleFluid::update(SoilSystem &soil_system) {
  auto &profiler = TimingProfiler::getInstance();
  SPHPtrs sph;
  ParticleGridPtrs grid_ptrs;
  SoilPtrs soil_ptrs;
  SoilParticlesPtrs soil_particle_ptrs;
  sph.get_ptrs(particles_device);
  grid_ptrs.get_ptrs(grid_device);
  soil_ptrs = soil_system.get_read_ptrs();
  soil_particle_ptrs = soil_system.get_particles_ptrs();
  int soil_width = soil_system.get_width();
  int soil_height = soil_system.get_height();
  float soil_size = soil_system.get_cell_size();

  size_t num_particles = particles_device.pos.size();
  size_t grid_size = grid.width * grid.height;
  int2 grid_dims = make_int2(grid.width, grid.height);
  float cell_size = params.smoothing_radius;

  dim3 sph_block(256);
  dim3 sph_grid_dim((num_particles + sph_block.x - 1) / sph_block.x);

  dim3 grid_block(256);
  dim3 grid_grid_dim((grid_size + grid_block.x - 1) / grid_block.x);

  dim3 soil_block(256);
  dim3 soil_grid_dim((soil_width * soil_height * PARTICLES_PER_SOIL_CELL + soil_block.x - 1) /
                     soil_block.x);

  // place in grid
  {
    auto scope = profiler.scopedMeasure("reset_particles_per_cell");
    reset_particles_per_cell<<<grid_grid_dim, grid_block>>>(grid_ptrs.particles_per_cell,
                                                            grid_size);
    check_cuda("reset_particles_per_cell");
  }

  {
    auto scope = profiler.scopedMeasure("populate_grid_indices");
    populate_grid_indices<<<sph_grid_dim, sph_block>>>(sph, grid_ptrs, num_particles,
                                                       grid.max_particles_per_cell, grid_dims,
                                                       params.smoothing_radius);
    check_cuda("populate_grid_indices");
  }

  // jitter soil particles
  {
    auto scope = profiler.scopedMeasure("jitter_particles");
    soil_system.jitter_particles();
  }

  // calculate density of particles, including contributions from soil
  {
    auto scope = profiler.scopedMeasure("calculate_particle_density");
    calculate_particle_density<<<sph_grid_dim, sph_block>>>(
        sph, grid_ptrs, grid.max_particles_per_cell, grid_dims, cell_size, params.smoothing_radius,
        num_particles, bounds, soil_ptrs, soil_particle_ptrs, soil_width, soil_height, soil_size);
    check_cuda("calculate_particle_density");
  }

  // calculate density of soil particles
  {
    auto scope = profiler.scopedMeasure("calculate_soil_density");
    calculate_soil_density<<<soil_grid_dim, soil_block>>>(
        sph, soil_particle_ptrs, grid_ptrs, grid.max_particles_per_cell, grid_dims, cell_size,
        params.smoothing_radius, num_particles, bounds, soil_ptrs, soil_width, soil_height,
        soil_size);
  }

  // calculate acceleration
  // calculate_accel<<<sph_grid_dim, sph_block>>>(sph, grid_ptrs, grid.max_particles_per_cell,
  //                                              grid_dims, cell_size, params, num_particles,
  //                                              bounds);
  // calculate acceleration with soil
  {
    auto scope = profiler.scopedMeasure("calculate_accel");
    calculate_accel<<<sph_grid_dim, sph_block>>>(
        sph, soil_particle_ptrs, grid_ptrs, grid.max_particles_per_cell, grid_dims, cell_size,
        params, num_particles, bounds, soil_ptrs, soil_width, soil_height, soil_size);
    check_cuda("calculate_accel");
  }

  // move particles
  {
    auto scope = profiler.scopedMeasure("move_particles");
    move_particles<<<sph_grid_dim, sph_block>>>(sph, params.dt, params.dt_predict, num_particles,
                                                bounds, params.collision_damping);
    check_cuda("move_particles");
  }
}

__global__ void render_particles(unsigned int *circle_vbo, SPHPtrs sph, TunableParams params,
                                 size_t num_particles) {
  // unsigned int color = 0xFFFFA077;
  unsigned int color = 0xFFFFFFFF; // 0xFF000000
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particles)
    return;

  auto pos = sph.pos[i];
  auto radius = params.smoothing_radius * 0.1f;
  // radius *= sph.density[i] / 400.0f;

  circle_vbo[i * 4 + 0] = reinterpret_cast<unsigned int &>(pos.x);
  circle_vbo[i * 4 + 1] = reinterpret_cast<unsigned int &>(pos.y);
  circle_vbo[i * 4 + 2] = reinterpret_cast<unsigned int &>(radius);
  circle_vbo[i * 4 + 3] = color;
}

void ParticleFluid::render(const glm::mat4 &transform) {
  // early return if we don't have a renderer
  if (!circle_renderer)
    return;

  // imgui
  if (pm) {
    // TODO: need to reconfigure when some of this changes
    ImGui::Begin("Particle Fluid");
    ImGui::SliderFloat("dt", &params.dt, 0.0f, 0.1f);
    ImGui::SliderFloat("dt_predict", &params.dt_predict, 0.0f, 0.1f);
    ImGui::SliderFloat("gravity", &params.gravity, -30.0f, 0.0f);
    ImGui::SliderFloat("collision_damping", &params.collision_damping, 0.0f, 1.0f);
    if (ImGui::SliderFloat("smoothing_radius", &params.smoothing_radius, 0.001f, 0.5f))
      init_grid();
    ImGui::SliderFloat("target_density", &params.target_density, 0.0f, 400.0f);
    ImGui::SliderFloat("pressure_mult", &params.pressure_mult, 0.0f, 1200.0f);
    ImGui::SliderFloat("near_pressure_mult", &params.near_pressure_mult, 0.0f, 100.0f);
    ImGui::SliderFloat("viscosity_strength", &params.viscosity_strength, 0.0f, 1.0f);
    if (ImGui::SliderInt("particles_per_cell", &params.particles_per_cell, 1, 32))
      init();
    if (ImGui::SliderInt("max_particles_per_cell", &params.max_particles_per_cell, 1, 1024))
      init();
    if (ImGui::Button("Save"))
      save_params();
    if (ImGui::Button("Load"))
      load_params();
    if (ImGui::Button("Reset Simulation"))
      init();
    ImGui::End();
  }

  circle_renderer->set_transform(transform);

  const auto circle_count = particles_device.pos.size();
  circle_renderer->ensure_vbo_capacity(circle_count);
  check_cuda("ensure_vbo_capacity");
  // get a cuda compatible pointer to the vbo
  auto vbo_ptr = circle_renderer->cuda_map_buffer();

  // render the particles
  dim3 block(256);
  dim3 grid_dim((circle_count + block.x - 1) / block.x);

  SPHPtrs sph;
  sph.get_ptrs(particles_device);
  render_particles<<<grid_dim, block>>>(static_cast<unsigned int *>(vbo_ptr), sph, params,
                                        circle_count);
  check_cuda("render_particles");

  // unmap the buffer
  circle_renderer->cuda_unmap_buffer();
  // render the particles
  circle_renderer->render(circle_count);
}

__global__ void calculate_density_grid_kernel(int density_grid_size, SPHPtrs sph,
                                              ParticleGridPtrs grid, int max_particles_per_cell,
                                              int2 density_grid_dims, float sample_interval,
                                              int2 particle_grid_dims, float cell_size,
                                              float smoothing_radius, float2 bounds,
                                              unsigned char *texture_data, float max_density) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= density_grid_size)
    return;

  int cell_x = i % density_grid_dims.x;
  int cell_y = i / density_grid_dims.x;
  float2 pos = make_float2(cell_x * sample_interval, cell_y * sample_interval);
  float density = calculate_density_at_pos(pos, sph, grid, max_particles_per_cell,
                                           particle_grid_dims, cell_size, smoothing_radius, bounds)
                      .x; // TODO: also report near density

  // determine particles per cell
  int grid_index = particle_to_cid(pos, particle_grid_dims.x, cell_size);
  int num_particles = grid.particles_per_cell[grid_index];

  // normalize density
  density = density / max_density;

  if (num_particles > max_particles_per_cell) {
    // red if too many particles per cell
    texture_data[i * 4] = 255;
    texture_data[i * 4 + 1] = 0;
    texture_data[i * 4 + 2] = 0;
  } else if (density > 1.0f) {
    // blue if density over max
    texture_data[i * 4] = 0;
    texture_data[i * 4 + 1] = 0;
    texture_data[i * 4 + 2] = 255;
  } else {
    // normal, grayscale rendering
    texture_data[i * 4] = 255 * density;
    texture_data[i * 4 + 1] = 255 * density;
    texture_data[i * 4 + 2] = 255 * density;
  }
  texture_data[i * 4 + 3] = 255;
}

void ParticleFluid::calculate_density_grid(thrust::device_vector<unsigned char> &texture_data,
                                           int width, int height, float max_density) {
  float cell_size = params.smoothing_radius;
  float sample_interval = bounds.x / width;
  // sample_interval *= 0.5f;

  int density_grid_size = width * height;
  int particle_grid_size = grid.width * grid.height;
  int2 density_grid_dims = make_int2(width, height);
  int2 particle_grid_dims = make_int2(grid.width, grid.height);

  SPHPtrs sph;
  sph.get_ptrs(particles_device);
  ParticleGridPtrs grid_ptrs;
  grid_ptrs.get_ptrs(grid_device);

  // calculate density grid
  calculate_density_grid_kernel<<<(density_grid_size + 255) / 256, 256>>>(
      density_grid_size, sph, grid_ptrs, grid.max_particles_per_cell, density_grid_dims,
      sample_interval, particle_grid_dims, cell_size, params.smoothing_radius, bounds,
      texture_data.data().get(), max_density);
  check_cuda("calculate_density_grid_kernel");
}

} // namespace p2
