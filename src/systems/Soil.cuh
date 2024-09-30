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
constexpr float CLAY_RELATIVE_DENSITY = 1.0f;

constexpr float SAND_PERMEABILITY = 0.5f;
constexpr float SILT_PERMEABILITY = 0.3f;
constexpr float CLAY_PERMEABILITY = 0.00f;

// #define FOR_SOIL(N, D) \
// D(float, water_density, 0)\
// D(float, water_give_left, 0)\
// D(float, water_give_right, 0)\
// D(float, water_give_up, 0)\
// D(float, water_give_down, 0)\
// D(float, sand_density, 0)\
// D(float, silt_density, 0)\
// D(float, clay_density, 0)\
// D(float, ph, 6.5)\
// D(float, organic_matter, 0)

#define FOR_SOIL(N, D) \
D(float, water, 0) \
D(float, water_give_left_per_sec, 0) \
D(float, water_give_up_per_sec, 0) \
D(float, water_give_left, 0) \
D(float, water_give_up, 0) \
D(float, sand_proportion, 0) \
D(float, silt_proportion, 0) \
D(float, clay_proportion, 0)

// capacity is aggregate from sand, silt, clay.
// water_density = water / capacity
// target_density is aggregate from sand, silt, clay.
// pressure_per_density is aggregate from sand, silt, clay, or its just a constant factor. idk yet.
// water_pressure = (water_density - target_density) * pressure_per_density
// water_pressure_delta = s.water_pressure - o.water_pressure
// compute water give acceleration for all 4. this is just water_pressure_delta * some constant
// don't need constant if pressure_per_density is a constant factor as it would be redundant.
// dampening is aggregate from sand, silt, clay.
// avg_dampening = (s.dampening + o.dampening) * 0.5f
// water_give_*_per_sec -= water_give_*_per_sec^2 * avg_dampening
// water_give_*_per_sec += accel * dt
// water_give_* = water_give_*_per_sec

// apply give take logic, set water_give_*_per_sec = water_give_*
// perhaps we don't need this as a different set. unless we delete the above line.
// adjust this to work with all 8 neighbors. use weighting somehow

// gravity is just a bit of acceleration in the down direction

// TODO: do the navier stokes trick where we only store the velocity on the edges


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
    void mix_give_take_3_cuda(float dt);

};
