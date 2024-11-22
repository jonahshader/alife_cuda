#pragma once

#include "systems/Screen.h"
#include "systems/Game.cuh"
#include "systems/Trees.cuh"
#include "graphics/ExtendViewport.h"

class TreeTest : public Screen {
public:
  explicit TreeTest(Game &game);

  void show() override;
  void render(float dt) override;
  void resize(int width, int height) override;
  void hide() override;
  bool handleInput(SDL_Event event) override;

private:
  Game &game;
  ExtendViewport vp{720, 720};
  ExtendViewport hud_vp{720, 720};

  trees::Trees trees{true};

  bool mixing{true};
  bool mutating_len_rot{false};
  bool updating_cpu{false};
  bool updating_parallel{false};
  bool mutating_pos{false};

  bool tree_vbo_buffered{false};
};
