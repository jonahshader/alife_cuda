#include "Soil.cuh"

#include <cmath>
#include <cstdint>
#include <memory>
#include <cstdint>
#include <glm/glm.hpp>


constexpr auto FILTER_WIDTH = 3;
constexpr auto BLOCK_WIDTH = 16;
constexpr auto BLOCK_HEIGHT = BLOCK_WIDTH;

using uint = std::uint32_t;

__constant__ float gaussian_filter[FILTER_WIDTH * FILTER_WIDTH] = {
    0.0625f, 0.125f, 0.0625f,
    0.125f,  0.25f,  0.125f,
    0.0625f, 0.125f, 0.0625f
  };

__host__ __device__
inline float get_cell_w_wrap(float* cells, int x, int y, int width) {
    // wrap around left and right
    // TODO: try modulus solution
    if (x < 0) x += width;
    if (x >= width) x -= width;
    return cells[y * width + x];
}

__host__ __device__
inline int16_t get_cell_w_wrap(int16_t* cells, int x, int y, int width) {
    // wrap around left and right
    // TODO: try modulus solution
    if (x < 0) x += width;
    if (x >= width) x -= width;
    return cells[y * width + x];
}

// __host__ __device__
// inline void mix_give_take(SoilPtrs &read, SoilPtrs &write, uint width, uint height, float dt, uint x, uint y) {
//
//     const auto id = x + y * width;
//
//     const float give_per_sec = 1.0f; // TODO: this should be computed from soil composition
//     const float epsilon = 0.001f;
//
//     const auto water = read.water_density[id];
//
//     float water_delta = 0;
//     // for now, hard code to going down
//
//     auto water_other = get_cell_w_wrap(read.water_density, x - 1, y, width);
//     if (water > water_other + epsilon) {
//         // give left water
//         water_delta -= give_per_sec;
//     } else if (water + epsilon < water_other) {
//         // take left water
//         // TODO: calculate give rate
//         water_delta += give_per_sec;
//     }
//     water_other = get_cell_w_wrap(read.water_density, x + 1, y, width);
//     if (water > water_other + epsilon) {
//         // give right water
//         water_delta -= give_per_sec;
//     } else if (water + epsilon < water_other) {
//         // take right water
//         // TODO: calculate give rate
//         water_delta += give_per_sec;
//     }
//     if (y < height - 1) {
//         water_other = get_cell_w_wrap(read.water_density, x, y + 1, width);
//         // only consider taking from top
//         if (water + epsilon < water_other) {
//             // take up water
//             // TODO: calculate give rate
//             water_delta += give_per_sec;
//         }
//     }
//     if (y > 0) {
//         water_other = get_cell_w_wrap(read.water_density, x, y - 1, width);
//         // only consider giving to bottom
//         if (water > water_other + epsilon) {
//             // give down water
//             // TODO: calculate give rate
//             water_delta -= give_per_sec;
//         }
//     }
//
//     write.water_density[id] = water + water_delta * dt;
// }

__host__ __device__
inline void mix_give_take(SoilPtrs &read, SoilPtrs &write, uint width, uint height, float dt, uint x, uint y) {

    const auto id = x + y * width;

    const int16_t give = max(1, static_cast<int16_t>(512 * dt)); // TODO: this should be computed from soil composition
    const int16_t epsilon = give+1;

    const auto water = read.water_density[id];

    int16_t water_delta = 0;
    // for now, hard code to going down

    auto water_other = get_cell_w_wrap(read.water_density, x - 1, y, width);
    if (water > water_other + epsilon) {
        // give left water
        water_delta -= give;
    } else if (water + epsilon < water_other) {
        // take left water
        // TODO: calculate give rate
        water_delta += give;
    }
    water_other = get_cell_w_wrap(read.water_density, x + 1, y, width);
    if (water > water_other + epsilon) {
        // give right water
        water_delta -= give;
    } else if (water + epsilon < water_other) {
        // take right water
        // TODO: calculate give rate
        water_delta += give;
    }
    if (y < height - 1) {
        water_other = get_cell_w_wrap(read.water_density, x, y + 1, width);
        // only consider taking from top
        if (water + epsilon < water_other) {
            // take up water
            // TODO: calculate give rate
            water_delta += give;
        }
    }
    if (y > 0) {
        water_other = get_cell_w_wrap(read.water_density, x, y - 1, width);
        // only consider giving to bottom
        if (water > water_other + epsilon) {
            // give down water
            // TODO: calculate give rate
            water_delta -= give;
        }
    }

    // if (y < height - 1) {
    //     water_other = get_cell_w_wrap(read.water_density, x, y + 1, width);
    //     // only consider taking from top
    //     if (water + epsilon < water_other) {
    //         // take up water
    //         // TODO: calculate give rate
    //         water_delta += give;
    //     } else if (water > water_other + epsilon) {
    //         // give down water
    //         water_delta -= give;
    //     }
    // }
    // if (y > 0) {
    //     water_other = get_cell_w_wrap(read.water_density, x, y - 1, width);
    //     // only consider giving to bottom
    //     if (water > water_other + epsilon) {
    //         // give down water
    //         // TODO: calculate give rate
    //         water_delta -= give;
    //     } else if (water + epsilon < water_other) {
    //         // take up water
    //         water_delta += give;
    //     }
    // }

    write.water_density[id] = water + water_delta;
}

__global__
void mix_give_take_kernel(SoilPtrs read, SoilPtrs write, uint width, uint height, float dt) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    mix_give_take(read, write, width, height, dt, x, y);
}

__host__ __device__
inline void add_rect(float x, float y, float width, float height, float radius, glm::vec4 color, float* vbo, size_t i) {
    const auto s = i * RectRenderer::FLOATS_PER_RECT;
    vbo[s + 0] = x;
    vbo[s + 1] = y;
    vbo[s + 2] = width;
    vbo[s + 3] = height;
    vbo[s + 4] = radius;
    vbo[s + 5] = color.r;
    vbo[s + 6] = color.g;
    vbo[s + 7] = color.b;
    vbo[s + 8] = color.a;
}

// __global__
// void render_kernel(float* rect_vbo, SoilPtrs read, uint width, size_t rect_count) {
//     const auto i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= rect_count) {
//         return;
//     }
//     const auto water_density = read.water_density[i];
//     const auto x = i % width;
//     const auto y = i / width;
//
//     const auto amount = min(max(water_density, 0.0f), 1.0f);
//
//     add_rect(x * 4, y * 4, 5, 5, 1, glm::vec4(amount), rect_vbo, i);
// }

__global__
void render_kernel(float* rect_vbo, SoilPtrs read, uint width, size_t rect_count) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rect_count) {
        return;
    }
    const auto water_density = read.water_density[i];
    const auto x = i % width;
    const auto y = i / width;

    auto amount = min(max(water_density, 0), 255) / 255.0f;
    amount = sqrt(amount);

    add_rect(x * 4, y * 4, 5, 5, 1, glm::vec4(1, 1, 1, amount), rect_vbo, i);
}

SoilSystem::SoilSystem(uint width, uint height, bool use_graphics) : width(width), height(height) {
    SoilSoA soil{};
    // assert(width % BLOCK_WIDTH == 0);
    // assert(height % BLOCK_WIDTH == 0);
    soil.resize_all(width * height);

    // for now, just put a lot in one cell in the center

    const auto x = width / 2;
    const auto y = height / 2;
    const auto id = x + y * width;

    soil.water_density[id] = 255 * 127;

    read.copy_from_host(soil);
    write.copy_from_host(soil);

    if (use_graphics) {
        rect_renderer = std::make_unique<RectRenderer>();
    }
}

void SoilSystem::mix_give_take_cuda(float dt) {
    // grab ptrs
    SoilPtrs read_ptrs{}, write_ptrs{};
    read_ptrs.get_ptrs(read);
    write_ptrs.get_ptrs(write);

    // set up grid
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(width + block.x - 1 / block.x, height + block.y - 1 / block.y);

    // launch kernel
    mix_give_take_kernel<<<grid, block>>>(read_ptrs, write_ptrs, width, height, dt);

    // kernel writes to water_density, so swap read and write
    write.water_density.swap(read.water_density);
}

void SoilSystem::update_cpu(float dt) {
    // TODO: implement
}


void SoilSystem::update_cuda(float dt) {
    // for now, just mix give take
    mix_give_take_cuda(dt);
}


void SoilSystem::render(const glm::mat4 &transform) {
    // early return if we don't have a rect renderer
    if (!rect_renderer) return;

    rect_renderer->set_transform(transform);

    const auto rect_count = read.water_density.size();
    rect_renderer->ensure_vbo_capacity(rect_count);
    // get a cuda compatible pointer to the vbo
    rect_renderer->cuda_register_buffer();
    auto vbo_ptr = rect_renderer->cuda_map_buffer();
    SoilPtrs ptrs;
    ptrs.get_ptrs(read);

    dim3 block(256);
    dim3 grid((rect_count + block.x - 1) / block.x);
    render_kernel<<<grid, block>>>(static_cast<float*>(vbo_ptr), ptrs, width, rect_count);
    rect_renderer->cuda_unmap_buffer();

    rect_renderer->render(rect_count);
    rect_renderer->cuda_unregister_buffer();
}

