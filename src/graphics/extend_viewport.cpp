#include "extend_viewport.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

ExtendViewport::ExtendViewport(float min_width, float min_height)
    : min_width(min_width), min_height(min_height) {}

void ExtendViewport::update(int screen_width, int screen_height) {
  last_screen_width = screen_width;
  last_screen_height = screen_height;
  float screen_ratio = screen_width / (float)screen_height;
  float world_ratio = min_width / min_height;

  if (screen_ratio > world_ratio) {
    width = min_width * screen_ratio / world_ratio;
    height = min_height;
  } else {
    width = min_width;
    height = min_height * world_ratio / screen_ratio;
  }
  float w = width * zoom;
  float h = height * zoom;
  transform =
      glm::ortho(-w / 2 + x_cam, w / 2 + x_cam, -h / 2 + y_cam, h / 2 + y_cam, -100.0f, 100.0f);
}

glm::mat4 ExtendViewport::get_transform() const {
  return transform;
}

float ExtendViewport::get_width() const {
  return width;
}

float ExtendViewport::get_height() const {
  return height;
}

glm::vec2 ExtendViewport::unproject(glm::vec2 screen_coords) const {
  screen_coords /= glm::vec2(last_screen_width, last_screen_height);
  screen_coords -= glm::vec2(0.5f, 0.5f);
  screen_coords *= glm::vec2(2.0f, -2.0f);

  auto inv = get_transform();
  inv = glm::inverse(inv);

  return glm::vec2(inv * glm::vec4(screen_coords, 0.0f, 1.0f));
}

float ExtendViewport::get_left() const {
  return -width / 2 + x_cam;
}

float ExtendViewport::get_right() const {
  return width / 2 + x_cam;
}

float ExtendViewport::get_top() const {
  return height / 2 + y_cam;
}

float ExtendViewport::get_bottom() const {
  return -height / 2 + y_cam;
}

void ExtendViewport::handle_scroll(float yoffset) {
  zoom *= powf(1.5f, -yoffset);
  //    if (zoom < 0.1f) zoom = 0.1f;
  //    if (zoom > 10.0f) zoom = 10.0f;
  update(last_screen_width, last_screen_height);
}

void ExtendViewport::handle_pan(float xoffset, float yoffset) {
  //    x_cam -= xoffset * zoom;
  //    y_cam += yoffset * zoom;
  //    update(lastScreenWidth, lastScreenHeight);

  // need to account for size of viewport
  float screen_ratio = last_screen_width / (float)last_screen_height;
  float world_ratio = min_width / min_height;

  if (screen_ratio > world_ratio) {
    width = min_width * screen_ratio / world_ratio;
    height = min_height;
  } else {
    width = min_width;
    height = min_height * world_ratio / screen_ratio;
  }

  x_cam -= xoffset * zoom * width / last_screen_width;
  y_cam += yoffset * zoom * height / last_screen_height;
  update(last_screen_width, last_screen_height);
}
