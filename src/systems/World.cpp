#include "World.h"

World::World(bool use_graphics) :
trees(use_graphics)
{}

void World::update(float dt) {
    trees.update(dt);
}

void World::render(const glm::mat4 &transform) {
    trees.render(transform);
}