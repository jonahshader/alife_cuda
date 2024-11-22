#pragma once

#include "SDL.h"

class Screen {
public:
  virtual void show() = 0; // called when screen becomes the current screen
  virtual void render(float dt) = 0;
  virtual void resize(int width, int height) = 0;
  virtual void hide() = 0; // called when screen is no longer the current screen
  virtual bool handleInput(SDL_Event event) = 0;

  virtual ~Screen() = default;
};
