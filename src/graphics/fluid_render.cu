#include "fluid_render.cuh"

#include <iostream>

#include <imgui.h>

namespace p2 {

static void check_cuda(const std::string &msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "fluid_render: " << msg << ": " << cudaGetErrorString(err) << std::endl;
  }
}

__global__ void render_particles_kernel(unsigned int *circle_vbo, SPHPtrs sph, TunableParams params,
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

void render_fluid(const ParticleFluidState &state, CircleRenderer &renderer,
                  const glm::mat4 &transform) {
  renderer.set_transform(transform);

  const auto circle_count = state.particles_device.pos.size();
  renderer.ensure_vbo_capacity(circle_count);
  check_cuda("ensure_vbo_capacity");

  auto vbo_ptr = renderer.cuda_map_buffer();

  dim3 block(256);
  dim3 grid_dim((circle_count + block.x - 1) / block.x);

  SPHPtrs sph;
  sph.get_ptrs(const_cast<SPHSoA<DeviceBuffer> &>(state.particles_device));
  render_particles_kernel<<<grid_dim, block>>>(static_cast<unsigned int *>(vbo_ptr), sph,
                                               state.params, circle_count);
  check_cuda("render_particles_kernel");

  renderer.cuda_unmap_buffer();
  renderer.render(circle_count);
}

void render_fluid_imgui(ParticleFluidState &state) {
  if (!state.pm)
    return;

  // TODO: need to reconfigure when some of this changes
  ImGui::Begin("Particle Fluid");
  ImGui::SliderFloat("dt", &state.params.dt, 0.0f, 0.1f);
  ImGui::SliderFloat("dt_predict", &state.params.dt_predict, 0.0f, 0.1f);
  ImGui::SliderFloat("gravity", &state.params.gravity, -30.0f, 0.0f);
  ImGui::SliderFloat("collision_damping", &state.params.collision_damping, 0.0f, 1.0f);
  if (ImGui::SliderFloat("smoothing_radius", &state.params.smoothing_radius, 0.001f, 0.5f))
    init_fluid_grid(state);
  ImGui::SliderFloat("target_density", &state.params.target_density, 0.0f, 400.0f);
  ImGui::SliderFloat("pressure_mult", &state.params.pressure_mult, 0.0f, 1200.0f);
  ImGui::SliderFloat("near_pressure_mult", &state.params.near_pressure_mult, 0.0f, 100.0f);
  ImGui::SliderFloat("viscosity_strength", &state.params.viscosity_strength, 0.0f, 10.0f);
  if (ImGui::SliderInt("particles_per_cell", &state.params.particles_per_cell, 1, 32))
    init_fluid(state, state.bounds.x, state.bounds.y, state.params);
  if (ImGui::SliderInt("max_particles_per_cell", &state.params.max_particles_per_cell, 1, 1024))
    init_fluid(state, state.bounds.x, state.bounds.y, state.params);
  if (ImGui::Button("Save"))
    save_fluid_params(state);
  if (ImGui::Button("Load"))
    load_fluid_params(state);
  if (ImGui::Button("Reset Simulation"))
    init_fluid(state, state.bounds.x, state.bounds.y, state.params);
  ImGui::End();
}

} // namespace p2
