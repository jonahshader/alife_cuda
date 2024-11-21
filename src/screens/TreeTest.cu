#include "TreeTest.cuh"

#include <chrono>

#include "glad/glad.h"

constexpr uint32_t NUM_NODES = 1<<10;
constexpr uint32_t NUM_TREES = 1<<10;



TreeTest::TreeTest(Game &game) : game(game) {
    trees.generate_random_trees(NUM_TREES, NUM_NODES, game.getResources().generator);
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

    bool update_render = mixing || mutating_len_rot || updating_parallel || mutating_pos || updating_cpu;

    bold.begin();
    rect.begin();
    line.begin();

    std::string mix_time, mutate_time, update_time, mutate_pos_time, update_parallel_time;


    // TODO: fix
//     if (mixing) {
//         read_tree_device.copy_to_host(read_tree);
//         auto start = std::chrono::steady_clock::now();
//         trees::mix_node_contents(read_tree, write_tree, 1.0f);
//         // TODO: write a swap function for TreeBatch struct
//         read_tree.trees.swap_all(write_tree.trees);
//         read_tree.tree_shapes.swap_all(write_tree.tree_shapes);
//         auto end = std::chrono::steady_clock::now();
//         auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//         mix_time = "Mix Time (CPU): " + std::to_string(elapsed.count()) + "us";
//         read_tree_device.copy_from_host(read_tree);
//     }
//
//     if (mutating_len_rot) {
//         read_tree_device.copy_to_host(read_tree);
//         auto start = std::chrono::steady_clock::now();
//         trees::mutate_len_rot(read_tree, game.getResources().generator, 0.0f, 0.01f);
//         auto end = std::chrono::steady_clock::now();
//         auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//         mutate_time = "Mutate Time (CPU): " + std::to_string(elapsed.count()) + "us";
//         read_tree_device.copy_from_host(read_tree);
//     }
//
//     if (mutating_pos) {
//         read_tree_device.copy_to_host(read_tree);
//         auto start = std::chrono::steady_clock::now();
//         trees::mutate_pos(read_tree, game.getResources().generator, 1.0f);
//         auto end = std::chrono::steady_clock::now();
//         auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//         mutate_pos_time = "Mutate Pos Time (CPU): " + std::to_string(elapsed.count()) + "us";
//         read_tree_device.copy_from_host(read_tree);
//     }
//
//
//     if (updating_parallel) {
//         // memory io is excluded from timing
//
//         auto start = std::chrono::steady_clock::now();
//         trees::update_tree_cuda(read_tree_device, write_tree_device);
//         auto end = std::chrono::steady_clock::now();
//         auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//         update_time = "Update Time (CUDA): " + std::to_string(elapsed.count()) + "us";
//
//     } else if (updating_cpu) {
//         read_tree_device.copy_to_host(read_tree);
//         auto start = std::chrono::steady_clock::now();
//         trees::update_tree_parallel(read_tree, write_tree);
// //        update_tree_cuda(read_tree, write_tree);
//         auto end =  std::chrono::steady_clock::now();
//         auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//         update_time = "Update Time (CPU): " + std::to_string(elapsed.count()) + "us";
//
//         // TODO: update_tree_parallel needs proper swaps, so this is a workaround
//         read_tree = write_tree;
//         read_tree_device.copy_from_host(read_tree);
//         write_tree_device = read_tree_device;
//     }

    if (updating_parallel) {
        auto start = std::chrono::steady_clock::now();
        trees.update(1/60.0f);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        update_time = "Update Time (CUDA): " + std::to_string(elapsed.count()) + "us";
    }

    trees.render(vp.get_transform());




    // if (update_render) {
    //     if (tree_vbo_buffered) {
    //         auto vbo_cuda_map = line.cudaMapBuffer();
    //         trees2::TreeBatchPtrs read_tree_device_ptrs;
    //         read_tree_device_ptrs.get_ptrs(read_tree_device);
    //
    //         dim3 block(256);
    //         auto node_count = read_tree.trees.core.abs_rot.size();
    //         dim3 grid((node_count + block.x - 1) / block.x);
    //
    //         trees::render_tree_kernel<<<grid, block>>>(static_cast<unsigned int *>(vbo_cuda_map), read_tree_device_ptrs, node_count);
    //         // wait for kernel to finish
    //         cudaDeviceSynchronize();
    //         line.cudaUnmapBuffer();
    //     } else {
    //         trees::render_tree(line, read_tree, game.getResources().generator, vp.get_transform());
    //     }
    //
    // }



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

    constexpr int padding = 10;

    bold.add_text(hud_vp.get_left() + padding, hud_vp.get_bottom() + padding + 120, 300, mix_time, glm::vec4(0.75), FontRenderer::HAlign::RIGHT);
    bold.add_text(hud_vp.get_left() + padding, hud_vp.get_bottom() + padding + 60, 300, mutate_time, glm::vec4(0.75), FontRenderer::HAlign::RIGHT);
    bold.add_text(hud_vp.get_left() + padding, hud_vp.get_bottom() + padding, 300, update_time, glm::vec4(0.75), FontRenderer::HAlign::RIGHT);


    bold.end();

    bold.render();
}

void TreeTest::resize(int width, int height) {
    vp.update(width, height);
    hud_vp.update(width, height);
}

void TreeTest::hide() {

}

bool TreeTest::handleInput(SDL_Event event) {
    if (event.type == SDL_KEYDOWN) {
        if (event.key.keysym.sym == SDLK_ESCAPE) {
            game.stopGame();
        } else if (event.key.keysym.sym == SDLK_r) {
            trees.generate_random_trees(NUM_TREES, NUM_NODES, game.getResources().generator);
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

    return true;
}