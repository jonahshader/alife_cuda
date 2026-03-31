#include "world.cuh"

void update_world(World &world, float dt) {
  trees::update_trees(world.trees, dt);
}
