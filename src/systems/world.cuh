#pragma once

#include "systems/trees.cuh"

#include <glm/glm.hpp>

class World {
public:
  explicit World(bool use_graphics);
  ~World() = default;

  void update(float dt);
  void render(const glm::mat4 &transform);

private:
  trees::Trees trees;
};
