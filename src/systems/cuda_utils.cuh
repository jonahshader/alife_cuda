#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// Check for CUDA errors after a kernel launch or API call (non-fatal, prints warning)
inline void check_cuda(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
  }
}

// Check a CUDA API call and exit on failure
#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t error = call;                                                                      \
    if (error != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)
