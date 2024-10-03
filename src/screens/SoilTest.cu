#include "SoilTest.cuh"

#include <chrono>

#include "glad/glad.h"
#include "imgui.h"

constexpr uint32_t SOIL_WIDTH = 1 << 10;
constexpr uint32_t SOIL_HEIGHT = 1 << 8;
constexpr uint32_t SOIL_SIZE = 8;

SoilTest::SoilTest(Game &game) : game(game),
                                 soil(SOIL_WIDTH, SOIL_HEIGHT, SOIL_SIZE, true),
                                 fluid(SOIL_WIDTH, SOIL_HEIGHT, true)
{
    vp.x_cam = SOIL_WIDTH * SOIL_SIZE / 2;
    vp.y_cam = SOIL_HEIGHT * SOIL_SIZE / 2;
}

void SoilTest::show()
{
}

void SoilTest::render(float dt)
{
    glClearColor(0.08f, 0.6f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    auto &bold = game.getResources().extra_bold_font;
    auto &rect = game.getResources().rect_renderer;
    auto &line = game.getResources().line_renderer;
    bold.set_transform(vp.get_transform());
    rect.set_transform(vp.get_transform());
    line.set_transform(vp.get_transform());

    bold.begin();
    rect.begin();
    line.begin();

    // TODO: time things

    static float fluid_dt = 1/ 165.0f;
    ImGui::Begin("Particle Fluid");
    ImGui::SliderFloat("Fluid Frequency", &fluid_dt, 0.0001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
    ImGui::Checkbox("Running", &running);
    ImGui::End();
    if (running)
    {
        soil.update_cuda(dt);
        fluid.update(fluid_dt);
    }

    soil.render(vp.get_transform());
    fluid.render(vp.get_transform());

    // bold.add_text(0.0f, 0.0f, 100, "hi", glm::vec4(0.9f));

    bold.end();
    rect.end();
    line.end();

    bold.render();
    rect.render();
    line.render();

    // TODO: render timings
}

void SoilTest::resize(int width, int height)
{
    vp.update(width, height);
    hud_vp.update(width, height);
}

void SoilTest::hide()
{
}

void SoilTest::handleInput(SDL_Event event)
{
    switch (event.type)
    {
    case SDL_MOUSEMOTION:
    {
        if (event.motion.state & SDL_BUTTON_LMASK)
        {
            vp.handle_pan(event.motion.xrel, event.motion.yrel);
        }
        if (event.motion.state & SDL_BUTTON_RMASK)
        {
            // get mouse x y, project to world
            const auto world_coords = vp.unproject({event.motion.x, event.motion.y});
            soil.add_water(static_cast<int>(floor(world_coords.x)), static_cast<int>(floor(world_coords.y)), 1.0f);
        }
    }
    break;
    case SDL_MOUSEWHEEL:
        vp.handle_scroll(event.wheel.y);
        break;
    case SDL_KEYDOWN:
        switch (event.key.keysym.sym)
        {
        // case SDLK_u:
        //     running = true;
        //     break;
        // case SDLK_i:
        //     soil.update_cuda(1 / 165.0f); // TODO: dt should be set in World, not passed around
        //     break;
        case SDLK_r:
            soil.reset();
        default:
            break;
        }
        break;
    // case SDL_KEYUP:
    //     if (event.key.keysym.sym == SDLK_u)
    //     {
    //         running = false;
    //     }
    //     break;
    default:
        break;
    }
}