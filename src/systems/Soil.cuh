#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "SoAHelper.h"

namespace soil {
    // TODO: need to store width and height somewhere. probably in World

#define FOR_SOIL(N, D) \
    D(float, water_density, 0)\
    D(float, sand_density, 1)\
    D(float, silt_density, 0)\
    D(float, clay_density, 0)\
    D(float, ph, 6.5)\
    D(float, organic_matter, 0)

    DEFINE_STRUCTS(Soil, FOR_SOIL)


}