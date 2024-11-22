#pragma once

#include <SDL.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>

namespace ui {

class Container {
public:
  // finds the deepest container that contains the point, then calls handle_event on that container
  virtual bool handle_mouse_event(SDL_Event &event) {
    auto click_pos = glm::vec2(event.button.x, event.button.y);
    for (auto &child : children) {
      if (child->is_inside(click_pos)) {
        return child->handle_mouse_event(event);
      }
    }
    if (is_inside(click_pos)) {
      on_mouse_event(event);
      return true;
    }

    return false;
  }

protected:
  virtual void on_mouse_event(SDL_Event &event) = 0;
  virtual bool is_inside(glm::vec2 click_pos) = 0;

private:
  std::vector<std::shared_ptr<Container>> children;
};

} // namespace ui
