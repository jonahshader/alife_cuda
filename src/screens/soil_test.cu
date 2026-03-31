#include "glad/glad.h"
#include "imgui.h"
#include "soil_test.cuh"

#include <chrono>

constexpr uint32_t SOIL_WIDTH = 1 << 10;
constexpr uint32_t SOIL_HEIGHT = 1 << 8;
constexpr uint32_t SOIL_SIZE = 8;

SoilTest::SoilTest(Game &game) : DefaultScreen(game), fluid(20, 20, true) {
  init_soil(soil, SOIL_WIDTH, SOIL_HEIGHT, SOIL_SIZE);
  vp.x_cam = SOIL_WIDTH * SOIL_SIZE / 2;
  vp.y_cam = SOIL_HEIGHT * SOIL_SIZE / 2;
}

void SoilTest::render(float dt) {
  render_start();

  static float fluid_dt = 1 / 60.0f;
  ImGui::Begin("Particle Fluid");
  ImGui::Checkbox("Running", &running);
  ImGui::SliderFloat("Fluid Frequency", &fluid_dt, 0.0001f, 0.1f, "%.4f",
                     ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
  ImGui::End();
  if (running) {
    update_soil_cuda(soil, dt);
    fluid.update(fluid_dt);
  }

  render_soil(soil, soil_renderer, vp.get_transform());
  fluid.render(vp.get_transform());

  render_end();
}

bool SoilTest::handle_input(SDL_Event event) {
  if (DefaultScreen::handle_input(event))
    return true;
  switch (event.type) {
    case SDL_KEYDOWN:
      switch (event.key.keysym.sym) {
        case SDLK_r:
          reset_soil(soil);
        default:
          break;
      }
      break;
    default:
      break;
  }

  return false;
}
