#pragma once

constexpr float M_PI_F = 3.14159265358979323846f;

__host__ __device__
int wrap(int x, int max);