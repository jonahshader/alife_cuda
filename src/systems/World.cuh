#pragma once

#include <glm/glm.hpp>
#include "systems/Trees.cuh"

class World {
public:
  explicit World(bool use_graphics);
  ~World() = default;

  void update(float dt);
  void render(const glm::mat4 &transform);

private:
  trees::Trees trees;
};
