#pragma once

#include "systems/trees.cuh"

// Pure data struct
struct World {
  trees::TreesState trees;
};

void update_world(World &world, float dt);
