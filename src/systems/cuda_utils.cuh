#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// Check for CUDA errors after a kernel launch or API call. Fatal: a failed launch leaves the
// simulation state undefined, and a run that keeps going prints plausible-looking profiler
// numbers for kernels that never executed (an sm_75 build on a Blackwell GPU exited 0 that way).
inline void check_cuda(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
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
