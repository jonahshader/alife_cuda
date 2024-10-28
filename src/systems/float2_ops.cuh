#pragma once

#include <cmath>

// Addition
__host__ __device__ inline float2 operator+(const float2 &a, const float2 &b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}

// Subtraction
__host__ __device__ inline float2 operator-(const float2 &a, const float2 &b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}

// Multiplication (element-wise)
__host__ __device__ inline float2 operator*(const float2 &a, const float2 &b)
{
  return make_float2(a.x * b.x, a.y * b.y);
}

// Division (element-wise)
__host__ __device__ inline float2 operator/(const float2 &a, const float2 &b)
{
  return make_float2(a.x / b.x, a.y / b.y);
}

// Scalar multiplication
__host__ __device__ inline float2 operator*(const float2 &a, float s)
{
  return make_float2(a.x * s, a.y * s);
}

__host__ __device__ inline float2 operator*(float s, const float2 &a)
{
  return make_float2(s * a.x, s * a.y);
}

// Scalar division
__host__ __device__ inline float2 operator/(const float2 &a, float s)
{
  float inv = 1.0f / s;
  return make_float2(a.x * inv, a.y * inv);
}

// Dot product
__host__ __device__ inline float dot(const float2 &a, const float2 &b)
{
  return a.x * b.x + a.y * b.y;
}

// Length (magnitude) of vector
__host__ __device__ inline float length(const float2 &a)
{
  return sqrtf(dot(a, a));
}

// Length squared of vector
__host__ __device__ inline float length2(const float2 &a)
{
  return dot(a, a);
}

// Normalize vector
__host__ __device__ inline float2 normalize(const float2 &a)
{
  float invLen = 1.0f / length(a);
  return a * invLen;
}