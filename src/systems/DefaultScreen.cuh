#pragma once

#include <chrono>

#include "systems/Screen.h"
#include "systems/Game.cuh"
#include "graphics/ExtendViewport.h"

class DefaultScreen : public Screen {
public:
  explicit DefaultScreen(Game &game);
  ~DefaultScreen() override = default;

  virtual void show() override;
  virtual void render(float dt) override;
  virtual void resize(int width, int height) override;
  virtual void hide() override;
  virtual bool handleInput(SDL_Event event) override;

protected:
  Game &game;
  ExtendViewport vp{720, 720};
  ExtendViewport hud_vp{720, 720};

  std::chrono::high_resolution_clock::time_point last_time;

  void render_start();
  void render_end();
};
