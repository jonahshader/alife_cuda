#pragma once

#include <cstdint>
#include <vector>

#include "external/OpenSimplexNoise.cuh"

class FractalNoise {
public:
    FractalNoise(int octaves, double scale, double wrap_width, double lacunarity, double persistence, int64_t seed);

    double eval(double x) const;
    double eval(double x, double y) const;
    double eval(double x, double y, double z) const;

private:
    double scale, wrap_width, lacunarity, persistence;
    std::vector<OpenSimplexNoise::Noise> m_noises{};
};
