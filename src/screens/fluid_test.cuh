#pragma once

#include "graphics/extend_viewport.h"
#include "systems/fluid.cuh"
#include "systems/game.cuh"
#include "systems/screen.h"

using namespace fluid;

// TODO: should use DefaultScreen instead of Screen
class FluidTest : public Screen {
public:
  explicit FluidTest(Game &game);

  void show() override;
  void render(float dt) override;
  void resize(int width, int height) override;
  void hide() override;
  bool handle_input(SDL_Event event) override;

private:
  Game &game;
  ExtendViewport vp{720, 720};
  Cell *read_cells{nullptr};
  Cell *write_cells{nullptr};

  Fluid fluid;
};
