#include "Resources.cuh"

Resources::Resources(uint64_t seed) : generator(seed) {}

float Resources::rand() {
  return realDist(generator);
}

float Resources::randn() {
  return normDist(generator);
}
