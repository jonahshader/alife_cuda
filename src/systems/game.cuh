#pragma once

#include "resources.cuh"
#include "screen.h"

#include <memory>
#include <random>
#include <stack>

class Game {
public:
  void update(float dt);
  void render();
  void resize(int width, int height);
  void push_screen(std::shared_ptr<Screen> screen);
  void pop_screen();
  void switch_screen(const std::shared_ptr<Screen> &screen);
  [[nodiscard]] bool is_running() const;
  void stop_game();
  void handle_input(SDL_Event event);
  [[nodiscard]] unsigned int get_seed() const;
  Resources &get_resources();

private:
  std::stack<std::shared_ptr<Screen>> screen_stack;
  unsigned int seed{std::random_device{}()};
  Resources resources{seed};
  int width{}, height{};
  bool running{true};
};
