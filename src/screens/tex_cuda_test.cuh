#pragma once

#include "graphics/extend_viewport.h"
#include "graphics/renderers/rect_tex_renderer.cuh"
#include "systems/game.cuh"
#include "systems/screen.h"

// TODO: should use DefaultScreen instead of Screen
class TexCUDATest : public Screen {
public:
  explicit TexCUDATest(Game &game);

  void show() override;
  void render(float dt) override;
  void resize(int width, int height) override;
  void hide() override;
  bool handle_input(SDL_Event event) override;

private:
  Game &game;
  ExtendViewport vp{720, 720};
  ExtendViewport hud_vp{720, 720};

  RectTexRenderer rect{640, 480, 4};
};
