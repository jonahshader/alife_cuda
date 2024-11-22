#include "CustomMath.cuh"

int wrap(int x, int max) {
  int result = x % max;
  if (result < 0) {
    result += max;
  }
  return result;
}

float2 abs(const float2 &a) {
  return make_float2(fabsf(a.x), fabsf(a.y));
}

float2 max(const float2 &a, const float2 &b) {
  return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
