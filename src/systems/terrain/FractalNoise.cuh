#pragma once

#include <cstdint>
#include <vector>

#include "external/FastNoiseLite.cuh"

class FractalNoise {
public:
    FractalNoise(int octaves, float scale, float wrap_width, float lacunarity, float persistence, int64_t seed);

    float eval(float x) const;
    float eval(float x, float y) const;

private:
    float scale, wrap_width, lacunarity, persistence;
    std::vector<FastNoiseLite> m_noises{};
};
