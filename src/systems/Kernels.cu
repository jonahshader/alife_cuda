#include "Kernels.cuh"

#include <cmath>

#include "CustomMath.cuh"

__host__ __device__
float smoothstep_kernel_volume(float kernel_radius) {
  return 3.0f * M_PI_F * kernel_radius * kernel_radius / 10.0f;
}

__host__ __device__
float smoothstep_kernel(float distance, float kernel_radius, float kernel_vol_inv) {
  if (distance >= kernel_radius) {
    return 0.0f;
  }
  float q = 1 - distance / kernel_radius;
  return q * q * (3.0f - 2.0f * q) * kernel_vol_inv;
}

__host__ __device__
float smoothstep_kernel_derivative(float distance, float kernel_radius, float kernel_vol_inv) {
  if (distance >= kernel_radius) {
    return 0.0f;
  }
  float q = 1 - distance / kernel_radius;
  return -6.0f * q * (1 - q) * kernel_vol_inv;
}

__host__ __device__
float sharp_kernel_volume(float kernel_radius) {
  return M_PI_F * std::pow(kernel_radius, 4) / 6.0f;
}

__host__ __device__
float sharp_kernel(float distance, float kernel_radius, float kernel_vol_inv) {
  if (distance >= kernel_radius) {
    return 0.0f;
  }
  float q = kernel_radius - distance;
  return q * q * kernel_vol_inv;
}

__host__ __device__
float sharp_kernel_derivative(float distance, float kernel_radius, float kernel_vol_inv) {
  if (distance >= kernel_radius) {
    return 0.0f;
  }
  return (-2.0f * kernel_radius + 2.0f * distance) * kernel_vol_inv;
}