#include "game.cuh"

void Game::update(float dt) {
  if (!screen_stack.empty())
    screen_stack.top()->update(dt);
}

void Game::render() {
  if (!screen_stack.empty())
    screen_stack.top()->render();
}

void Game::resize(int width, int height) {
  if (!screen_stack.empty())
    screen_stack.top()->resize(width, height);
  this->width = width;
  this->height = height;
}

void Game::push_screen(std::shared_ptr<Screen> screen) {
  screen_stack.push(screen);
  screen_stack.top()->resize(width, height);
}

void Game::pop_screen() {
  if (!screen_stack.empty()) {
    screen_stack.top()->hide();
    screen_stack.pop();
    screen_stack.top()->show();
    screen_stack.top()->resize(width, height);
  }
}

void Game::switch_screen(const std::shared_ptr<Screen> &screen) {
  if (!screen_stack.empty()) {
    screen_stack.top()->hide();
    screen_stack.pop();
  }
  screen_stack.push(screen);
  screen_stack.top()->show();
  screen_stack.top()->resize(width, height);
}

Resources &Game::get_resources() {
  return resources;
}

bool Game::is_running() const {
  return running;
}

void Game::stop_game() {
  running = false;
}

void Game::handle_input(SDL_Event event) {
  if (!screen_stack.empty())
    screen_stack.top()->handle_input(event);
}

unsigned int Game::get_seed() const {
  return seed;
}
