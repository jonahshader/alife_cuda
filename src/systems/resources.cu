#include "resources.cuh"

Resources::Resources(uint64_t seed) : generator(seed) {}

float Resources::rand() {
  return real_dist(generator);
}

float Resources::randn() {
  return norm_dist(generator);
}
