#include "FluidTest2.cuh"

#include "imgui.h"

FluidTest2::FluidTest2(Game &game) : DefaultScreen(game)
{
  pm.get_param("dt", dt);
  pm.get_param("particle_mass", particle_mass);
  pm.get_param("kernel_radius", kernel_radius);
}

void FluidTest2::render(float _dt)
{
  render_start();

  ImGui::Begin("Fluid Test 2");
  if (ImGui::SliderFloat("dt", &dt, 0.0001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat))
    pm.set_param("dt", dt);
  if (ImGui::SliderFloat("particle_mass", &particle_mass, 0.1f, 10.0f, "%.1f"))
    pm.set_param("particle_mass", particle_mass);
  if (ImGui::SliderFloat("kernel_radius", &kernel_radius, 0.01f, 0.5f, "%.2f"))
    pm.set_param("kernel_radius", kernel_radius);
  if (ImGui::Button("Save Parameters"))
    pm.save_params();
  ImGui::End();

  render_end();
}