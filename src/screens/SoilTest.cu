#include "SoilTest.cuh"

#include <chrono>

#include "glad/glad.h"

constexpr uint32_t SOIL_WIDTH = 1<<10;
constexpr uint32_t SOIL_HEIGHT = 1<<8;

SoilTest::SoilTest(Game &game) :
game(game),
soil(SOIL_WIDTH, SOIL_HEIGHT, true)
{}

void SoilTest::show() {

}

void SoilTest::render(float dt) {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    auto& bold = game.getResources().extra_bold_font;
    auto& rect = game.getResources().rect_renderer;
    auto& line = game.getResources().line_renderer;
    bold.set_transform(vp.get_transform());
    rect.set_transform(vp.get_transform());
    line.set_transform(vp.get_transform());

    bold.begin();
    rect.begin();
    line.begin();

    // TODO: time things

    if (running) {
        soil.update_cuda(dt);
    }

    soil.render(vp.get_transform());

    bold.add_text(0.0f, 0.0f, 300, "hi", glm::vec4(0.9f));

    bold.end();
    rect.end();
    line.end();


    bold.render();
    rect.render();
    line.render();

    // TODO: render timings

    SDL_GL_SwapWindow(game.getResources().window);
}

void SoilTest::resize(int width, int height) {
    vp.update(width, height);
    hud_vp.update(width, height);
}

void SoilTest::hide() {

}

void SoilTest::handleInput(SDL_Event event) {
    if (event.type == SDL_MOUSEMOTION) {
        if (event.motion.state & SDL_BUTTON_LMASK) {
            vp.handle_pan(event.motion.xrel, event.motion.yrel);
        }
    } else if (event.type == SDL_MOUSEWHEEL) {
        vp.handle_scroll(event.wheel.y);
    } else if (event.type == SDL_KEYDOWN) {
        if (event.key.keysym.sym == SDLK_u) {
            running = true;
        }
    } else if (event.type == SDL_KEYUP) {
        if (event.key.keysym.sym == SDLK_u) {
            running = false;
        }
    }
}