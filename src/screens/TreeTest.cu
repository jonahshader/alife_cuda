//
// Created by jonah on 3/11/2024.
//

#include "TreeTest.cuh"
#include "glad/glad.h"

#include <iostream>
#include <chrono>

constexpr uint32_t NUM_NODES = 2<<7;
constexpr uint32_t NUM_TREES = 2<<8;

TreeBatch make_batch(uint32_t node_count, uint32_t tree_count, std::default_random_engine& rand) {
    std::vector<Tree> trees;
    for (int i = 0; i < NUM_TREES; ++i) {
        trees.push_back(build_tree_optimized(NUM_NODES, rand, glm::vec2(i * 64, 0)));
    }

    return concatenate_trees(trees);
}

TreeTest::TreeTest(Game &game) : game(game) {
    read_tree = make_batch(NUM_NODES, NUM_TREES, game.getResources().generator);
    write_tree = read_tree;
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

//    bold.add_text(0, (vp.get_height() / 2.0f) - 120, 600, "Tree Test", glm::vec4(0.5), FontRenderer::HAlign::CENTER);
//    rect.add_rect(0, 0, 32, 24, 8, glm::vec4(1));
//    rect.add_rect(0, -100, 128, 32, 16, glm::vec4(1, 0, 1, 1));
    auto start = std::chrono::steady_clock::now();
//    mutate(tree, game.getResources().generator, 0.002f);

// TODO: get mutation working again
//    if (!stripped_tree.empty()) {
//        mutate(stripped_tree, game.getResources().generator, 0.002f);
//    } else {
//        mutate(tree, game.getResources().generator, 0.002f);
//    }

//    if (mixing) {
//        mix_node_contents(read_tree, write_tree, 1.0f, total_energy);
//        read_tree.swap(write_tree);
//    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    bold.add_text(0, (vp.get_height() / 2.0f), 500, "Mutate Time: " + std::to_string(elapsed.count()) + "us", glm::vec4(0.75), FontRenderer::HAlign::LEFT);

    start = std::chrono::steady_clock::now();
    update_tree(read_tree);
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//    bold.add_text(0, (vp.get_height() / 2.0f) + 120, 500, "Update Time: " + std::to_string(elapsed.count()) + "us", glm::vec4(0.75), FontRenderer::HAlign::CENTER);

    render_tree(line, read_tree, game.getResources().generator);
//    std::cout << "Total energy: " << compute_total_energy(read_tree) << std::endl;
//    std::cout << "Min/max energy: " << get_min_energy(read_tree) << ',' << get_max_energy(read_tree) << std::endl;

//    bold.add_text(0, (vp.get_height() / 2.0f) + 240, 500, "Min/Max Energy: " + std::to_string(get_min_energy(read_tree)) + ',' + std::to_string(get_max_energy(read_tree)), glm::vec4(0.75), FontRenderer::HAlign::LEFT);


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
            read_tree = make_batch(NUM_NODES, NUM_TREES, game.getResources().generator);
            write_tree = read_tree;
        } else if (event.key.keysym.sym == SDLK_SPACE) {
            mixing = !mixing;
        }
    } else if (event.type == SDL_MOUSEWHEEL) {
        vp.handle_scroll(event.wheel.y);
    } else if (event.type == SDL_MOUSEMOTION) {
        if (event.motion.state & SDL_BUTTON_LMASK) {
            vp.handle_pan(event.motion.xrel, event.motion.yrel);
        }
    }
}