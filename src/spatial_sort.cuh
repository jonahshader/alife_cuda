#pragma once

#include <cstdint>
#include <vector>
#include <random>
#include <thrust/host_vector.h>

struct Rect {
  uint32_t id;
  float x;
  float y;
  float width;
  float height;

  Rect(uint32_t id, float x, float y, float width, float height)
      : id(id), x(x), y(y), width(width), height(height) {}
  Rect() = default;
};

struct RectComparator {
  __host__ __device__ bool operator()(const Rect &a, const Rect &b) const {
    return a.x < b.x;
  }
};

void benchmark_spatial_sort();
thrust::host_vector<Rect>
make_random_rects(uint32_t count, std::uniform_real_distribution<float> &dist, std::mt19937 &gen);
