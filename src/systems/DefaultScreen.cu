#include "DefaultScreen.cuh"

#include <chrono>

#include "glad/glad.h"

DefaultScreen::DefaultScreen(Game &game) : game(game) {}

void DefaultScreen::show() {
  last_time = std::chrono::high_resolution_clock::now();
}

void DefaultScreen::render(float dt) {
  render_start();
  render_end();
}

void DefaultScreen::resize(int width, int height) {
  vp.update(width, height);
  hud_vp.update(width, height);
}

void DefaultScreen::hide() {}

bool DefaultScreen::handleInput(SDL_Event event) {
  if (event.type == SDL_KEYDOWN) {
    switch (event.key.keysym.sym) {
      case SDLK_ESCAPE:
        game.stopGame();
        return true;
      default:
        break;
    }
  } else if (event.type == SDL_MOUSEWHEEL) {
    vp.handle_scroll(event.wheel.y);
    return true;
  } else if (event.type == SDL_MOUSEMOTION) {
    if (event.motion.state & SDL_BUTTON_MMASK) {
      vp.handle_pan(event.motion.xrel, event.motion.yrel);
      return true;
    }
  }

  return false;
}

void DefaultScreen::render_start() {
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  auto &bold = game.getResources().extra_bold_font;
  auto &regular = game.getResources().main_font;
  auto &rect = game.getResources().rect_renderer;
  auto &line = game.getResources().line_renderer;
  bold.set_transform(vp.get_transform());
  regular.set_transform(hud_vp.get_transform());
  rect.set_transform(vp.get_transform());
  line.set_transform(vp.get_transform());

  bold.begin();
  regular.begin();
  rect.begin();
  line.begin();
}

void DefaultScreen::render_end() {
  auto &regular = game.getResources().main_font;

  auto now = std::chrono::high_resolution_clock::now();
  float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_time).count();
  auto left = hud_vp.get_left();
  auto bottom = hud_vp.get_bottom();
  regular.add_text(left, bottom + 30.0f, 100, "dt: " + std::to_string(dt) + "s",
                   glm::vec4(0.5f, 0.5f, 0.5f, 1.0f), FontRenderer::HAlign::RIGHT);
  regular.add_text(left, bottom + 00.0f, 150, "fps: " + std::to_string(1.0f / dt),
                   glm::vec4(0.5f, 0.5f, 0.5f, 1.0f), FontRenderer::HAlign::RIGHT);
  last_time = now;

  auto &bold = game.getResources().extra_bold_font;
  auto &rect = game.getResources().rect_renderer;
  auto &line = game.getResources().line_renderer;

  bold.end();
  regular.end();
  rect.end();
  line.end();

  bold.render();
  regular.render();
  rect.render();
  line.render();
}
