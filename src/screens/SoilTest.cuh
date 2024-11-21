#pragma once

#include <chrono>

#include "systems/DefaultScreen.cuh"
#include "systems/Game.cuh"
#include "systems/Soil.cuh"
#include "graphics/ExtendViewport.h"

#include "systems/ParticleFluid.cuh"

class SoilTest : public DefaultScreen
{
public:
    explicit SoilTest(Game &game);

    void render(float dt) override;
    bool handleInput(SDL_Event event) override;

private:
    SoilSystem soil;
    particles::ParticleFluid fluid;
    bool running{false};

    std::chrono::high_resolution_clock::time_point last_time;
};
