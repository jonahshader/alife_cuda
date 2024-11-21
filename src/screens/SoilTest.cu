#include "SoilTest.cuh"

#include <chrono>

#include "glad/glad.h"
#include "imgui.h"

constexpr uint32_t SOIL_WIDTH = 1 << 10;
constexpr uint32_t SOIL_HEIGHT = 1 << 8;
constexpr uint32_t SOIL_SIZE = 8;

SoilTest::SoilTest(Game &game) : DefaultScreen(game),
                                 soil(SOIL_WIDTH, SOIL_HEIGHT, SOIL_SIZE, true),
                                 fluid(20, 20, true)
{
    vp.x_cam = SOIL_WIDTH * SOIL_SIZE / 2;
    vp.y_cam = SOIL_HEIGHT * SOIL_SIZE / 2;
}

void SoilTest::render(float dt)
{
    render_start();

    static float fluid_dt = 1 / 60.0f;
    ImGui::Begin("Particle Fluid");
    ImGui::Checkbox("Running", &running);
    ImGui::SliderFloat("Fluid Frequency", &fluid_dt, 0.0001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
    ImGui::End();
    if (running)
    {
        soil.update_cuda(dt);
        fluid.update(fluid_dt);
    }

    soil.render(vp.get_transform());
    fluid.render(vp.get_transform());

    render_end();
}

bool SoilTest::handleInput(SDL_Event event)
{
    if (DefaultScreen::handleInput(event))
        return true;
    switch (event.type)
    {
    case SDL_MOUSEMOTION:
    {
        if (event.motion.state & SDL_BUTTON_LMASK)
        {
            // get mouse x y, project to world
            const auto world_coords = vp.unproject({event.motion.x, event.motion.y});
            soil.add_water(static_cast<int>(floor(world_coords.x)), static_cast<int>(floor(world_coords.y)), 1.0f);
        }
    }
    break;
    case SDL_KEYDOWN:
        switch (event.key.keysym.sym)
        {
        case SDLK_r:
            soil.reset();
        default:
            break;
        }
        break;
    default:
        break;
    }

    return false;
}