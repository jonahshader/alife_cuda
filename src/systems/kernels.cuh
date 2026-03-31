#pragma once

__host__ __device__ float smoothstep_kernel_volume(float kernel_radius);

__host__ __device__ float smoothstep_kernel(float distance, float kernel_radius,
                                            float kernel_vol_inv);

__host__ __device__ float smoothstep_kernel_derivative(float distance, float kernel_radius,
                                                       float kernel_vol_inv);

__host__ __device__ float sharp_kernel_volume(float kernel_radius);

__host__ __device__ float sharp_kernel(float distance, float kernel_radius, float kernel_vol_inv);

__host__ __device__ float sharp_kernel_derivative(float distance, float kernel_radius,
                                                  float kernel_vol_inv);
