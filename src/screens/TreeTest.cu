//
// Created by jonah on 3/11/2024.
//

#include "TreeTest.cuh"
#include "glad/glad.h"

#include <iostream>

TreeTest::TreeTest(Game &game) : game(game) {}

void TreeTest::show() {

}

void TreeTest::render(float dt) {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    auto& bold = game.getResources().extra_bold_font;
    auto& rect = game.getResources().rect_renderer;
    bold.set_transform(vp.get_transform());
    rect.set_transform(vp.get_transform());
    bold.begin();
    rect.begin();

    bold.add_text(0, (vp.get_height() / 2.0f) - 60, 600, "Tree Test", glm::vec4(1), FontRenderer::HAlign::CENTER);
    rect.add_rect(0, 0, 32, 24, 8, glm::vec4(1));
    rect.add_rect(0, -100, 128, 32, 16, glm::vec4(1, 0, 1, 1));

    bold.end();
    rect.end();

    bold.render();
    rect.render();

    SDL_GL_SwapWindow(game.getResources().window);
}

void TreeTest::resize(int width, int height) {
    vp.update(width, height);
}

void TreeTest::hide() {

}

void TreeTest::handleInput(SDL_Event event) {

}