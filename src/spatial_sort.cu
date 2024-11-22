#include "spatial_sort.cuh"

#include <iostream>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

void benchmark_spatial_sort() {
  constexpr uint32_t count = 10000000;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution dist(0.0f, 1.0f);

  // make random rects
  thrust::host_vector<Rect> rects = make_random_rects(count, dist, gen);

  // send to device
  std::cout << "Sending to device" << std::endl;
  thrust::device_vector<Rect> d_rects = rects;

  // sort by x
  std::cout << "Sorting by x" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  thrust::sort(d_rects.begin(), d_rects.end(), RectComparator());

  // copy back to host
  thrust::host_vector<Rect> sorted_rects = d_rects;

  // print the first 10 x values
  for (int i = 0; i < 10; i++) {
    std::cout << sorted_rects[i].x << std::endl;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "sort by x: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
            << std::endl;
}

thrust::host_vector<Rect>
make_random_rects(uint32_t count, std::uniform_real_distribution<float> &dist, std::mt19937 &gen) {
  thrust::host_vector<Rect> rects;
  rects.reserve(count);
  for (uint32_t i = 0; i < count; i++) {
    // make a random rect and put it in
    rects.push_back(Rect(i, dist(gen) * 1024, dist(gen) * 1024, dist(gen) * 32, dist(gen) * 32));
  }
  return rects;
}
