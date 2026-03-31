#pragma once

#include "graphics/renderers/simple_rect_renderer.cuh"
#include "systems/soil.cuh"

#include <glm/glm.hpp>

void render_soil(const SoilState &state, SimpleRectRenderer &renderer, const glm::mat4 &transform);
