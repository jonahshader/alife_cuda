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
    auto &regular = game.getResources().main_font;
    auto &rect = game.getResources().rect_renderer;
    auto &line = game.getResources().line_renderer;
    bold.set_transform(vp.get_transform());
    regular.set_transform(hud_vp.get_transform());
    rect.set_transform(vp.get_transform());
    line.set_transform(vp.get_transform());

    bold.begin();
    regular.begin();
    rect.begin();
    line.begin();

    // TODO: time things

    static float fluid_dt = 1/ 165.0f;
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

    // bold.add_text(0.0f, 0.0f, 100, "hi", glm::vec4(0.9f));

    // TODO: render timings
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - last_time).count();
    auto left = hud_vp.get_left();
    auto bottom = hud_vp.get_bottom();
    regular.add_text(left, bottom + 30.0f, 100, "dt: " + std::to_string(duration) + "us", glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), FontRenderer::HAlign::RIGHT);
    regular.add_text(left, bottom + 00.0f, 150, "fps: " + std::to_string(1.0f / (duration / 1e6f)), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), FontRenderer::HAlign::RIGHT);
    last_time = now;

    bold.end();
    regular.end();
    rect.end();
    line.end();

    bold.render();
    regular.render();
    rect.render();
    line.render();


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