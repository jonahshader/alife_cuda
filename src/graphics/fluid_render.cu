#include "fluid_render.cuh"
#include "systems/cuda_utils.cuh"

#include <iostream>

#include <imgui.h>

namespace p2 {

// deep blue (0) → teal (0.5) → white (1.0)
__device__ unsigned int evap_prob_to_color(float prob) {
  prob = fminf(fmaxf(prob, 0.0f), 1.0f);
  uint8_t r, g, b;
  if (prob < 0.5f) {
    float t = prob * 2.0f;
    r = (uint8_t)(t * 68);
    g = (uint8_t)(t * 200);
    b = (uint8_t)(140 + t * 115);
  } else {
    float t = (prob - 0.5f) * 2.0f;
    r = (uint8_t)(68 + t * 187);
    g = (uint8_t)(200 + t * 55);
    b = 255;
  }
  return (255u << 24) | ((uint32_t)b << 16) | ((uint32_t)g << 8) | r;
}

__global__ void render_particles_kernel(unsigned int *circle_vbo, SPHPtrs sph, SimParams params,
                                        size_t num_particles, bool debug_evap) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particles)
    return;

  auto pos = sph.pos[i];
  auto radius = params.smoothing_radius * 0.1f;

  unsigned int color;
  if (sph.state[i] == 1) {
    color = 0x40CCCCCC;
    radius *= 0.5f;
  } else if (debug_evap) {
    color = evap_prob_to_color(sph.evap_prob[i]);
  } else {
    color = 0xFFFFFFFF;
  }

  circle_vbo[i * 4 + 0] = reinterpret_cast<unsigned int &>(pos.x);
  circle_vbo[i * 4 + 1] = reinterpret_cast<unsigned int &>(pos.y);
  circle_vbo[i * 4 + 2] = reinterpret_cast<unsigned int &>(radius);
  circle_vbo[i * 4 + 3] = color;
}

void render_fluid(const ParticleFluidState &state, CircleRenderer &renderer,
                  const glm::mat4 &transform, bool debug_evap) {
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
                                               state.params, circle_count, debug_evap);
  check_cuda("render_particles_kernel");

  renderer.cuda_unmap_buffer();
  renderer.render(circle_count);
}

void render_fluid_imgui(ParticleFluidState &state) {
  // TODO: need to reconfigure when some of this changes
  ImGui::Begin("Particle Fluid Params");
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
  ImGui::Separator();
  ImGui::Text("Evaporation / Condensation");
  ImGui::SliderFloat("evap_rate", &state.params.evap_rate, 0.0f, 0.1f);
  ImGui::SliderFloat("condense_rate", &state.params.condense_rate, 0.0f, 0.1f);
  ImGui::SliderFloat("vapor_buoyancy", &state.params.vapor_buoyancy, 0.0f, 20.0f);
  ImGui::SliderFloat("vapor_drift", &state.params.vapor_drift, 0.0f, 5.0f);
  ImGui::SliderFloat("condense_alt_power", &state.params.condense_altitude_power, 0.5f, 5.0f);

  if (ImGui::Button("Reset Simulation"))
    init_fluid(state, state.bounds.x, state.bounds.y, state.params);
  ImGui::End();
}

} // namespace p2
