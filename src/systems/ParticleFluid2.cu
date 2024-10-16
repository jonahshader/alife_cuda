#include "ParticleFluid2.cuh"

#include <random>

#include "float2_ops.cuh"
#include "CustomMath.cuh"

namespace p2
{
  // given a particle's position, return the cell index it belongs to
  __host__ __device__ int particle_to_cid(float2 pos, int grid_width, float cell_size)
  {
    int grid_x = pos.x / cell_size;
    int grid_y = pos.y / cell_size;
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
  __global__ void populate_grid_indices(float2 *p_pos, int *grid_indices, int max_particles,
                                        int *particles_per_cell, int max_particles_per_cell,
                                        int grid_width, int grid_height, float cell_size)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // particle id
    if (i >= max_particles)
      return;

    int grid_index = particle_to_cid(p_pos[i], grid_width, cell_size);
    int slot_index = atomicAdd(&particles_per_cell[grid_index], 1);
    if (slot_index < max_particles_per_cell)
      grid_indices[grid_index * max_particles_per_cell + slot_index] = i;
  }

    // TODO: the provided dst is calculated using sqrt, but we square it here...
  __host__ __device__ float smoothing_kernel(float radius, float dst)
  {
    float q = dst / radius;
    if (q > 1.0f) return 0.0f;
    
    const float normalization_factor_2d = 4.0f / (M_PI_F * powf(radius, 8));
    
    float value = radius * radius - dst * dst;
    return normalization_factor_2d * value * value * value;
  }

  __host__ __device__ float calculate_density_at_pos(float2 pos, float2 *p_pos, float *mass, int *grid_indices, 
                                                     int *particles_per_cell, int max_particles_per_cell,
                                                     int2 particle_grid_dims, float cell_size, float smoothing_radius)
  {
    float density = 0.0f;
    int grid_index = particle_to_cid(pos, particle_grid_dims.x, cell_size);
    int cell_x = grid_index % particle_grid_dims.x;
    int cell_y = grid_index / particle_grid_dims.x;

    // iterate through cell neighborhood
    for (int yi = cell_y - 1; yi <= cell_y + 1; yi++)
    {
      for (int xi = cell_x - 1; xi <= cell_x + 1; xi++)
      {
        // skip if cell is out of vertical bounds
        if (yi < 0 || yi >= particle_grid_dims.y)
          continue;
        // wrap x if out of horizontal bounds
        int wrapped_x = (xi + particle_grid_dims.x) % particle_grid_dims.x;

        int neighbour_index = yi * particle_grid_dims.x + wrapped_x;
        int num_particles = particles_per_cell[neighbour_index];
        // iterate through particles within the cell
        for (int i = 0; i < num_particles; i++)
        {
          int particle_id = grid_indices[neighbour_index * max_particles_per_cell + i];
          float2 other_pos = p_pos[particle_id];
          if (xi == -1) // wrap around
            other_pos.x -= particle_grid_dims.x * cell_size;
          else if (xi == particle_grid_dims.x)
            other_pos.x += particle_grid_dims.x * cell_size;
          float distance = length(pos - other_pos);
          density += mass[particle_id] * smoothing_kernel(smoothing_radius, distance);
        }
      }
    }

    return density;
  }

  ParticleFluid::ParticleFluid(float width, float height, bool use_graphics) : bounds(make_float2(width, height))
  {
    if (use_graphics)
      circle_renderer = std::make_unique<CircleRenderer>();

    // configure grid
    auto smoothing_radius = params.smoothing_radius;
    int grid_width = std::ceil(width / smoothing_radius);
    int grid_height = std::ceil(height / smoothing_radius);

    grid.reconfigure(grid_width, grid_height, 64); // TODO: want the ability to reconfigure with imgui
    // init some particles
    std::default_random_engine rand;
    std::uniform_real_distribution<float> dist_x(0.0f, width);
    std::uniform_real_distribution<float> dist_y(0.0f, height);

    // velocity init with gaussian
    std::normal_distribution<float> dist_vel(0.0f, 0.1f);

    // temp: 1000 particles
    // TODO: make particle count proportional to the grid size
    const int NUM_PARTICLES = 16 * grid_width * grid_height;
    particles.resize_all(NUM_PARTICLES);
    for (int i = 0; i < NUM_PARTICLES; ++i)
      particles.pos[i] = make_float2(dist_x(rand), dist_y(rand));
    for (int i = 0; i < NUM_PARTICLES; ++i)
      particles.vel[i] = make_float2(dist_vel(rand), dist_vel(rand));
    // density is defaulted to 0

    // send to device
    particles_device.copy_from_host(particles);
    grid_device.copy_from_host(grid);
  }


  __global__ void move_particles(float2 *pos, float2 *vel, float dt, int num_particles, float2 bounds)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles)
      return;

    float2 new_pos = pos[i] + (vel[i] * dt);
    // wrap around
    if (new_pos.x < 0.0f)
      new_pos.x += bounds.x;
    else if (new_pos.x >= bounds.x)
      new_pos.x -= bounds.x;
    if (new_pos.y < 0.0f)
      new_pos.y += bounds.y;
    else if (new_pos.y >= bounds.y)
      new_pos.y -= bounds.y;

    pos[i] = new_pos;
  }

  void ParticleFluid::update() {
    // move particles
    move_particles<<<(particles_device.pos.size() + 255) / 256, 256>>>(
      particles_device.pos.data().get(), particles_device.vel.data().get(), params.dt, particles_device.pos.size(), bounds);
    // // update device
    // particles_device.copy_to_host(particles);
  }

  void ParticleFluid::render(const glm::mat4 &transform) {}

  __global__ void calculate_density_grid_kernel(int density_grid_size, float2 *p_pos, float *mass, int *grid_indices, 
                                                int *particles_per_cell, int max_particles_per_cell,
                                                int2 density_grid_dims, float sample_interval, 
                                                int2 particle_grid_dims, float cell_size, 
                                                float smoothing_radius, float *density_grid)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= density_grid_size)
      return;

    int cell_x = i % density_grid_dims.x;
    int cell_y = i / density_grid_dims.x;
    float2 pos = make_float2(cell_x * sample_interval, cell_y * sample_interval);
    density_grid[i] = calculate_density_at_pos(
      pos, p_pos, mass, grid_indices, 
      particles_per_cell, max_particles_per_cell, 
      particle_grid_dims, cell_size, smoothing_radius);
  }

  void ParticleFluid::calculate_density_grid(thrust::device_vector<float> &density_grid, int width, int height)
  {
    float cell_size = params.smoothing_radius;
    float sample_interval = bounds.x / width;

    int density_grid_size = width * height;
    int particle_grid_size = grid.width * grid.height;
    int2 density_grid_dims = make_int2(width, height);
    int2 particle_grid_dims = make_int2(grid.width, grid.height);

    // reset particles_per_cell counters
    reset_particles_per_cell<<<(particle_grid_size + 255) / 256, 256>>>(grid_device.particles_per_cell.data().get(), particle_grid_size);

    // populate grid indices
    auto particle_count = particles_device.pos.size();
    populate_grid_indices<<<(particle_count + 255) / 256, 256>>>(particles_device.pos.data().get(), grid_device.grid_indices.data().get(), particle_count,
                                                                 grid_device.particles_per_cell.data().get(), grid.max_particles_per_cell,
                                                                 grid.width, grid.height, cell_size);
    // calculate density grid
    calculate_density_grid_kernel<<<(density_grid_size + 255) / 256, 256>>>(
      density_grid_size, particles_device.pos.data().get(), particles_device.mass.data().get(), grid_device.grid_indices.data().get(), 
      grid_device.particles_per_cell.data().get(), grid.max_particles_per_cell,
      density_grid_dims, sample_interval,
      particle_grid_dims, cell_size, 
      params.smoothing_radius, density_grid.data().get());
  }

} // namespace p2