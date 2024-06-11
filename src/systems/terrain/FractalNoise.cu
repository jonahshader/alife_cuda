#include "FractalNoise.cuh"

#include <random>
#include <cmath>

#define M_PI 3.14159265358979323846

FractalNoise::FractalNoise(int octaves, double scale, double wrap_width, double lacunarity, double persistence, int64_t seed) :
scale(scale),
wrap_width(wrap_width),
lacunarity(lacunarity),
persistence(persistence)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist;

    m_noises.reserve(octaves);
    for (int i = 0; i < octaves; i++) {
        m_noises.emplace_back(dist(rng));
    }
}

double FractalNoise::eval(double x) const {
    // We want the noise to be loop horizontally at wrap_width,
    // so we will project the input coordinates to a cylinder.
    double angle = (x / wrap_width) * 2.0 * M_PI;
    double d1 = std::cos(angle) * scale * wrap_width / (2.0 * M_PI);
    double d2 = std::sin(angle) * scale * wrap_width / (2.0 * M_PI);

    double sum = 0.0;
    double amp = 1.0;
    double max = 0.0;
    for (const auto& noise : m_noises) {
        sum += noise.eval(d1, d2) * amp;
        d1 *= lacunarity;
        d2 *= lacunarity;
        max += amp;
        amp *= persistence;
    }

    return sum / max;
}

double FractalNoise::eval(double x, double y) const {
    // We want the noise to be loop horizontally at wrap_width,
    // so we will project the input coordinates to a cylinder.
    double angle = (x / wrap_width) * 2.0 * M_PI;
    double d1 = std::cos(angle) * scale * wrap_width / (2.0 * M_PI);
    double d2 = std::sin(angle) * scale * wrap_width / (2.0 * M_PI);
    double d3 = y * scale;

    double sum = 0.0;
    double amp = 1.0;
    double max = 0.0;
    for (const auto& noise : m_noises) {
        sum += noise.eval(d1, d2, d3) * amp;
        d1 *= lacunarity;
        d2 *= lacunarity;
        d3 *= lacunarity;
        max += amp;
        amp *= persistence;
    }

    return sum / max;
}

double FractalNoise::eval(double x, double y, double z) const {
    // We want the noise to be loop horizontally at wrap_width,
    // so we will project the input coordinates to a cylinder.
    double angle = (x / wrap_width) * 2.0 * M_PI;
    double d1 = std::cos(angle) * scale * wrap_width / (2.0 * M_PI);
    double d2 = std::sin(angle) * scale * wrap_width / (2.0 * M_PI);
    double d3 = y * scale;
    double d4 = z * scale;

    double sum = 0.0;
    double amp = 1.0;
    double max = 0.0;
    for (const auto& noise : m_noises) {
        sum += noise.eval(d1, d2, d3, d4) * amp;
        d1 *= lacunarity;
        d2 *= lacunarity;
        d3 *= lacunarity;
        d4 *= lacunarity;
        max += amp;
        amp *= persistence;
    }

    return sum / max;
}
