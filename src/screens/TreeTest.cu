//
// Created by jonah on 3/11/2024.
//

#include "TreeTest.cuh"
#include "glad/glad.h"

#include <iostream>
#include <chrono>

constexpr uint32_t NUM_NODES = 1<<6;
constexpr uint32_t NUM_TREES = 1<<12;

trees::TreeBatch make_batch_aos(uint32_t node_count, uint32_t tree_count, std::default_random_engine& rand) {
    std::vector<trees::Tree> trees;
    constexpr auto row_size = 64;
    std::normal_distribution<float> spawn_dist(0, row_size);
    std::uniform_int_distribution<int> num_nodes_dist(NUM_NODES / 2, 3 * NUM_NODES / 2);
    for (int i = 0; i < NUM_TREES; ++i) {
        int x = i % row_size;
        int y = i / row_size;
        trees.push_back(trees::build_tree_optimized((x * NUM_NODES) / (row_size/2), rand, glm::vec2(x * 128, y * 128)));
//        trees.push_back(build_tree_optimized(NUM_NODES, rand, glm::vec2(spawn_dist(rand), spawn_dist(rand))));
    }

    return trees::concatenate_trees(trees);
}

trees2::TreeBatch make_batch(uint32_t node_count, uint32_t tree_count, std::default_random_engine& rand) {
    std::vector<trees::Tree> trees;
    constexpr auto row_size = 64;
    std::normal_distribution<float> spawn_dist(0, row_size);
    std::uniform_int_distribution<int> num_nodes_dist(NUM_NODES / 2, 3 * NUM_NODES / 2);
    for (int i = 0; i < NUM_TREES; ++i) {
        int x = i % row_size;
        int y = i / row_size;
        trees.push_back(trees::build_tree_optimized((x * NUM_NODES) / (row_size/2), rand, glm::vec2(x * 128, y * 128)));
        //        trees.push_back(build_tree_optimized(NUM_NODES, rand, glm::vec2(spawn_dist(rand), spawn_dist(rand))));
    }

    auto concat = trees::concatenate_trees(trees);
    trees2::TreeBatch batch;
    batch.tree_shapes.push_back(concat.tree_shapes);
    batch.tree_data.push_back(concat.tree_data);
    batch.trees.push_back(concat.trees);

    return batch;
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

    std::string mix_time, mutate_time, update_time, mutate_pos_time, update_parallel_time;


    if (mixing) {
        auto start = std::chrono::steady_clock::now();
        trees::mix_node_contents(read_tree, write_tree, 1.0f);
        // TODO: write a swap function for TreeBatch struct
        read_tree.trees.swap_all(write_tree.trees);
        read_tree.tree_shapes.swap_all(write_tree.tree_shapes);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        mix_time = "Mix Time (CPU): " + std::to_string(elapsed.count()) + "us";
    }

    if (mutating_len_rot) {
        auto start = std::chrono::steady_clock::now();
        trees::mutate_len_rot(read_tree, game.getResources().generator, 0.0f, 0.01f);
        write_tree.trees = read_tree.trees;
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        mutate_time = "Mutate Time (CPU): " + std::to_string(elapsed.count()) + "us";
    }

    if (mutating_pos) {
        auto start = std::chrono::steady_clock::now();
        trees::mutate_pos(read_tree, game.getResources().generator, 1.0f);
        write_tree.trees = read_tree.trees;
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        mutate_pos_time = "Mutate Pos Time (CPU): " + std::to_string(elapsed.count()) + "us";
    }


    if (updating_parallel) {
        auto start = std::chrono::steady_clock::now();
        trees::update_tree_cuda(read_tree, write_tree);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        update_time = "Update Time (CUDA): " + std::to_string(elapsed.count()) + "us";
    } else if (updating_cpu) {
        auto start = std::chrono::steady_clock::now();
        trees::update_tree_parallel(read_tree, write_tree);
//        update_tree_cuda(read_tree, write_tree);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        update_time = "Update Time (CPU): " + std::to_string(elapsed.count()) + "us";
    }


    trees::render_tree(line, read_tree, game.getResources().generator);


    bold.end();
    rect.end();
    line.end();

    bold.render();
    rect.render();
    line.render();

    bold.set_transform(hud_vp.get_transform());
    rect.set_transform(hud_vp.get_transform());
    line.set_transform(hud_vp.get_transform());
    bold.begin();
    rect.begin();
    line.begin();

    constexpr int padding = 10;

    bold.add_text(hud_vp.get_left() + padding, hud_vp.get_bottom() + padding + 120, 300, mix_time, glm::vec4(0.75), FontRenderer::HAlign::RIGHT);
    bold.add_text(hud_vp.get_left() + padding, hud_vp.get_bottom() + padding + 60, 300, mutate_time, glm::vec4(0.75), FontRenderer::HAlign::RIGHT);
    bold.add_text(hud_vp.get_left() + padding, hud_vp.get_bottom() + padding, 300, update_time, glm::vec4(0.75), FontRenderer::HAlign::RIGHT);


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
    hud_vp.update(width, height);
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
        } else if (event.key.keysym.sym == SDLK_m) {
            mutating_len_rot = !mutating_len_rot;
        } else if (event.key.keysym.sym == SDLK_u) {
            updating_parallel = true;
        } else if (event.key.keysym.sym == SDLK_j) {
            mutating_pos = true;
        } else if (event.key.keysym.sym == SDLK_i) {
            updating_cpu = true;
        }
    } else if (event.type == SDL_KEYUP) {
        if (event.key.keysym.sym == SDLK_u) {
            updating_parallel = false;
        } else if (event.key.keysym.sym == SDLK_j) {
            mutating_pos = false;
        } else if (event.key.keysym.sym == SDLK_i) {
            updating_cpu = false;
        }
    } else if (event.type == SDL_MOUSEWHEEL) {
        vp.handle_scroll(event.wheel.y);
    } else if (event.type == SDL_MOUSEMOTION) {
        if (event.motion.state & SDL_BUTTON_LMASK) {
            vp.handle_pan(event.motion.xrel, event.motion.yrel);
        }
    }
}