#pragma once

#include "graphics/renderers/circle_renderer.cuh"
#include "systems/particle_fluid2.cuh"

#include <glm/glm.hpp>

namespace p2 {

void render_fluid(const ParticleFluidState &state, CircleRenderer &renderer,
                  const glm::mat4 &transform);

// ImGui parameter panel — mutates state.params via sliders
void render_fluid_imgui(ParticleFluidState &state);

} // namespace p2
