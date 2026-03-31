#pragma once

#include "graphics/extend_viewport.h"
#include "systems/game.cuh"
#include "systems/screen.h"
#include "systems/trees.cuh"

class TreeTest : public Screen {
public:
  explicit TreeTest(Game &game);

  void show() override;
  void render(float dt) override;
  void resize(int width, int height) override;
  void hide() override;
  bool handle_input(SDL_Event event) override;

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
