#include "FractalNoise.cuh"

#include <random>
#include <cmath>

#define M_PI_f 3.14159265358979323846f

FractalNoise::FractalNoise(int octaves, float scale, float wrap_width, float lacunarity, float persistence, int64_t seed) :
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

float FractalNoise::eval(float x) const {
    // We want the noise to be loop horizontally at wrap_width,
    // so we will project the input coordinates to a cylinder.
    float angle = (x / wrap_width) * 2.0f * M_PI_f;
    float d1 = std::cos(angle) * scale * wrap_width / (2.0f * M_PI_f);
    float d2 = std::sin(angle) * scale * wrap_width / (2.0f * M_PI_f);

    float sum = 0.0;
    float amp = 1.0;
    float max = 0.0;
    for (const auto& noise : m_noises) {
        sum += noise.GetNoise(d1, d2) * amp;
        d1 *= lacunarity;
        d2 *= lacunarity;
        max += amp;
        amp *= persistence;
    }

    return sum / max;
}

float FractalNoise::eval(float x, float y) const {
    // We want the noise to be loop horizontally at wrap_width,
    // so we will project the input coordinates to a cylinder.
    float angle = 2.0f * M_PI_f * x / wrap_width;
    float d1 = std::cos(angle) * scale * wrap_width / (2.0f * M_PI_f);
    float d2 = std::sin(angle) * scale * wrap_width / (2.0f * M_PI_f);
    float d3 = y * scale;

    float sum = 0.0;
    float amp = 1.0;
    float max = 0.0;
    for (const auto& noise : m_noises) {
        sum += noise.GetNoise(d1, d2, d3) * amp;
        d1 *= lacunarity;
        d2 *= lacunarity;
        d3 *= lacunarity;
        max += amp;
        amp *= persistence;
    }

    return sum / max;
}


