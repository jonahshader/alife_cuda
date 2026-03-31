#pragma once

#include "graphics/extend_viewport.h"
#include "graphics/soil_render.cuh"
#include "systems/default_screen.cuh"
#include "systems/game.cuh"
#include "systems/particle_fluid.cuh"
#include "systems/soil.cuh"

#include <chrono>

class SoilTest : public DefaultScreen {
public:
  explicit SoilTest(Game &game);

  void render(float dt) override;
  bool handle_input(SDL_Event event) override;

private:
  SoilState soil{};
  SimpleRectRenderer soil_renderer{};
  particles::ParticleFluid fluid;
  bool running{false};

  std::chrono::high_resolution_clock::time_point last_time;
};
