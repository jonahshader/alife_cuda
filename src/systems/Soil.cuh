#pragma once

#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "SoAHelper.h"
#include "graphics/renderers/RectRenderer.cuh"



#define FOR_SOIL(N, D) \
D(float, water_density, 0)\
D(float, sand_density, 1)\
D(float, silt_density, 0)\
D(float, clay_density, 0)\
D(float, ph, 6.5)\
D(float, organic_matter, 0)

DEFINE_STRUCTS(Soil, FOR_SOIL)

class SoilSystem {
public:
    using uint = std::uint32_t;
    SoilSystem(uint width, uint height, bool use_graphics);
    ~SoilSystem() = default;

    void update_cpu(float dt);
    void update_cuda(float dt);
    void render(const glm::mat4 &transform);

private:
    uint width, height;
    SoilSoADevice read{}, write{};
    std::unique_ptr<RectRenderer> rect_renderer{};

    void mix_give_take_cuda(float dt);

};
