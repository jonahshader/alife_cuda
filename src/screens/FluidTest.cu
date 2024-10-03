//
// Created by jonah on 3/12/2024.
//

#include "FluidTest.cuh"
#include "glad/glad.h"

#include <cuda_runtime.h>

FluidTest::FluidTest(Game &game) : game(game) {
    fluid.init(256, 256, 0.5f, 1.0f, game.getResources());
    read_cells = raw_pointer_cast(fluid.a_cells.data());
    write_cells = raw_pointer_cast(fluid.b_cells.data());
}

void FluidTest::show() {

}

void FluidTest::render(float dt) {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    auto& bold = game.getResources().extra_bold_font;
    auto& rect = game.getResources().rect_renderer;
    bold.set_transform(vp.get_transform());
    rect.set_transform(vp.get_transform());
    bold.begin();
    rect.begin();

    auto w = fluid.width;
    auto h = fluid.height;

    dim3 block(16, 16);
    dim3 grid(w / block.x, h / block.y);
    // get raw pointers from device vectors
    fluid_diffuse<<<grid, block>>>(read_cells, write_cells, w, h, 1.0f);

    // synchronize before running next kernel
    // cudaDeviceSynchronize();

    // render fluid as a grid of squares
    thrust::host_vector<fluid::Cell> host_cells = fluid.a_cells;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            auto cell = host_cells[y * w + x];
            auto amount = cell.amount;
            auto x_offset = x - w / 2;
            auto y_offset = y - h / 2;
            rect.add_rect(x_offset * 4, y_offset * 4, 5, 5, 1, glm::vec4(amount));
        }
    }

    // swap read and write cells
    std::swap(read_cells, write_cells);

    bold.end();
    rect.end();

    bold.render();
    rect.render();
}

void FluidTest::resize(int width, int height) {
    vp.update(width, height);
}

void FluidTest::hide() {

}

void FluidTest::handleInput(SDL_Event event) {

}