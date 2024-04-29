//
// Created by jonah on 3/11/2024.
//

#include "TreeTest.cuh"
#include "glad/glad.h"

#include <iostream>
#include <chrono>

constexpr uint32_t NUM_NODES = 2<<15;

TreeTest::TreeTest(Game &game) : game(game) {
    tree = build_tree(NUM_NODES, game.getResources().generator);
}

void TreeTest::show() {

}

void TreeTest::render(float dt) {
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

    bold.add_text(0, (vp.get_height() / 2.0f) - 120, 600, "Tree Test", glm::vec4(0.5), FontRenderer::HAlign::CENTER);
//    rect.add_rect(0, 0, 32, 24, 8, glm::vec4(1));
//    rect.add_rect(0, -100, 128, 32, 16, glm::vec4(1, 0, 1, 1));
    auto start = std::chrono::steady_clock::now();
    mutate(tree, game.getResources().generator, 0.002f);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    bold.add_text(0, (vp.get_height() / 2.0f), 500, "Mutate Time: " + std::to_string(elapsed.count()) + "us", glm::vec4(0.75), FontRenderer::HAlign::CENTER);

    start = std::chrono::steady_clock::now();
    update_tree(tree);
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    bold.add_text(0, (vp.get_height() / 2.0f) + 120, 500, "Update Time: " + std::to_string(elapsed.count()) + "us", glm::vec4(0.75), FontRenderer::HAlign::CENTER);

    render_tree(line, tree);

    bold.end();
    rect.end();
    line.end();

    bold.render();
    rect.render();
    line.render();

    SDL_GL_SwapWindow(game.getResources().window);
}

void TreeTest::resize(int width, int height) {
    vp.update(width, height);
}

void TreeTest::hide() {

}

void TreeTest::handleInput(SDL_Event event) {
    if (event.type == SDL_KEYDOWN) {
        if (event.key.keysym.sym == SDLK_ESCAPE) {
            game.stopGame();
        } else if (event.key.keysym.sym == SDLK_r) {
            tree = build_tree(NUM_NODES, game.getResources().generator);
        } else if (event.key.keysym.sym == SDLK_m) {
            mutate_and_update(tree, game.getResources().generator, 0.1f);
        } else if (event.key.keysym.sym == SDLK_s) {
            std::cout << "Sorting tree..." << std::endl;
            auto sorted_tree = sort_tree(tree);
            std::cout << tree.size() << std::endl;
            std::cout << sorted_tree.size() << std::endl;
            tree = sorted_tree;
            std::cout << "Tree sorted." << std::endl;
        } else if (event.key.keysym.sym == SDLK_p) {
            // print shit
            for (auto& node : tree) {
                std::cout << node.id << ',' << node.parent << std::endl;
                for (auto& child : node.children) {
                    std::cout << child << ',';
                }
                std::cout << std::endl << std::endl;
            }
        }
    } else if (event.type == SDL_MOUSEWHEEL) {
        vp.handle_scroll(event.wheel.y);
    } else if (event.type == SDL_MOUSEMOTION) {
        if (event.motion.state & SDL_BUTTON_LMASK) {
            vp.handle_pan(event.motion.xrel, event.motion.yrel);
        }
    }
}