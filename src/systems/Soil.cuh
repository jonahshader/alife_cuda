#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "SoAHelper.h"
#include "graphics/renderers/RectRenderer.cuh"

constexpr float SAND_RELATIVE_DENSITY = 0.5f;
constexpr float SILT_RELATIVE_DENSITY = 0.7f;
constexpr float CLAY_RELATIVE_DENSITY = 0.9f;

constexpr float SAND_PERMEABILITY = 0.5f;
constexpr float SILT_PERMEABILITY = 0.3f;
constexpr float CLAY_PERMEABILITY = 0.1f;

#define FOR_SOIL(N, D) \
D(float, water_density, 0)\
D(float, water_give_left, 0)\
D(float, water_give_right, 0)\
D(float, water_give_up, 0)\
D(float, water_give_down, 0)\
D(float, sand_density, 0)\
D(float, silt_density, 0)\
D(float, clay_density, 0)\
D(float, ph, 6.5)\
D(float, organic_matter, 0)

DEFINE_STRUCTS(Soil, FOR_SOIL)

class SoilSystem {
public:
    using uint = std::uint32_t;
    SoilSystem(uint width, uint height, uint size, bool use_graphics);
    ~SoilSystem() = default;

    void update_cpu(float dt);
    void update_cuda(float dt);
    void render(const glm::mat4 &transform);
    void reset();

    void add_water(int x, int y, float amount);

private:
    uint width, height, size;
    SoilSoADevice read{}, write{};
    std::unique_ptr<RectRenderer> rect_renderer{};

    void mix_give_take_cuda(float dt);
    void mix_give_take_2_cuda(float dt);
    void mix_give_take_3_cuda(float dt);

};
