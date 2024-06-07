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

    soil.update_cuda(dt);
    soil.render(vp.get_transform());

    bold.end();
    rect.end();
    line.end();


    bold.render();
    rect.render();
    line.render();

    // TODO: render timings

    SDL_GL_SwapWindow(game.getResources().window);
}

