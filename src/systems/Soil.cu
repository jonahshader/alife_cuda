#include "Soil.cuh"

#include <cmath>
#include <cstdint>
#include <memory>
#include <cstdint>
#include <glm/glm.hpp>

#include "systems/terrain/FractalNoise.cuh"


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

__host__ __device__
inline float calc_delta(float water, float other_water) {
    return (other_water - water) / 5;
}

__host__ __device__
inline bool inbounds(int y, uint height) {
    return y >= 0 && y < height;
}

__host__ __device__
inline void mix_give_take(SoilPtrs &read, SoilPtrs &write, uint width, uint height, float dt, uint x, uint y) {

    const auto id = x + y * width;

    const float give = 200.0f * dt;

    const auto water = read.water_density[id];

    float water_delta = 0;
    // for now, hard code to going down

    auto water_other = get_cell_w_wrap(read.water_density, x - 1, y, width);
    water_delta += calc_delta(water, water_other);
    water_other = get_cell_w_wrap(read.water_density, x + 1, y, width);
    water_delta += calc_delta(water, water_other);
    if (y < height - 1) {
        water_other = get_cell_w_wrap(read.water_density, x, y + 1, width);
        // water_delta += calc_delta(water, water_other);
        // TODO: max water should be calculated from soil composition
        if (water < 1.0f) {
            if (water_other > give) {
                water_delta += give * 0.5f;
            } else {
                water_delta += water_other * 0.5f;
            }
        }
    }
    if (y > 0) {
        water_other = get_cell_w_wrap(read.water_density, x, y - 1, width);
        // water_delta += calc_delta(water, water_other);
        // TODO: max water should be calculated from soil composition
        if (water_other < 1.0f) {
            if (water > give) {
                water_delta -= give * 0.5f;
            } else {
                water_delta -= water * 0.5f;
            }
        }
    }

    write.water_density[id] = water + water_delta;
}



__host__ __device__
inline void mix_give_take_2(SoilPtrs &read, SoilPtrs &write, uint width, uint height, float dt, uint x, uint y) {
    const auto id = x + y * width;


    float water_delta = 0.0f;

    // we need to determine if we are spreading, left is spreading, right is spreading
    const auto water = read.water_density[id];
    float give = max(min(water * 0.9f, 1.0f), 0.0f);
    auto give_left = write.water_give_left;
    auto give_right = write.water_give_right;
    auto give_up = write.water_give_up;
    auto give_down = write.water_give_down;

    float next_give_left = 0;
    float next_give_right = 0;
    float next_give_down = 0;


    if (inbounds(y - 1, height)) {
        const auto water_other = read.water_density[x + (y - 1) * width];
        const auto space = 1 - water_other;
        if (space >= give) {
            next_give_down = give;
            water_delta -= give;
            give = 0;
        } else if (space > 0) {
            next_give_down = space;
            water_delta -= space;
            give -= space;
        }
    }

    const auto space_left = 1 - get_cell_w_wrap(read.water_density, x - 1, y, width);
    const auto space_right = 1 - get_cell_w_wrap(read.water_density, x + 1, y, width);

    give *= 0.5f;

    if (space_left >= give) {
        next_give_left = give;
        water_delta -= give;
    } else if (space_left > 0) {
        next_give_left = space_left;
        water_delta -= space_left;
    }

    if (space_right >= give) {
        next_give_right = give;
        water_delta -= give;
    } else if (space_right > 0) {
        next_give_right = space_right;
        water_delta -= space_right;
    }

    if (water + water_delta < 0) {
        const float water_delta_scaler = water / water_delta;
        next_give_left *= water_delta_scaler;
        next_give_right *= water_delta_scaler;
        next_give_down *= water_delta_scaler;
        write.water_density[id] = 0;
    } else {
        write.water_density[id] = water + water_delta;
    }

    give_left[id] = next_give_left;
    give_right[id] = next_give_right;
    give_down[id] = next_give_down;
    give_up[id] = 0;
}

__host__ __device__
inline void copy_give(SoilPtrs &read, SoilPtrs &write, uint width, uint height, float dt, uint x, uint y) {
    const auto id = x + y * width;

    auto water = read.water_density[id];
    const auto give_left = read.water_give_left;
    const auto give_right = read.water_give_right;
    const auto give_up = read.water_give_up;
    const auto give_down = read.water_give_down;

    // take what the right is giving to the left (me)
    water += get_cell_w_wrap(give_left, x + 1, y, width);
    // take what the left is giving to the right (me)
    water += get_cell_w_wrap(give_right, x - 1, y, width);
    // take what the bottom is giving to the top (me)
    if (inbounds(y - 1, height)) {
        water += get_cell_w_wrap(give_up, x, y - 1, width);
    }
    // take what the top is giving to the bottom (me)
    if (inbounds(y + 1, height)) {
        water += get_cell_w_wrap(give_down, x, y + 1, width);
    }

    write.water_density[id] = water;
}

__global__
void mix_give_take_kernel(SoilPtrs read, SoilPtrs write, uint width, uint height, float dt) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    mix_give_take(read, write, width, height, dt, x, y);
}

__global__
void mix_give_take_2_kernel(SoilPtrs read, SoilPtrs write, uint width, uint height, float dt) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    mix_give_take_2(read, write, width, height, dt, x, y);
}

__global__
void copy_give_kernel(SoilPtrs read, SoilPtrs write, uint width, uint height, float dt) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    copy_give(read, write, width, height, dt, x, y);
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


__global__
void render_kernel(float* rect_vbo, SoilPtrs read, uint width, uint size, size_t rect_count) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rect_count) {
        return;
    }
    const auto water_density = read.water_density[i];
    const auto x = i % width;
    const auto y = i / width;

    auto amount = min(abs(water_density), 1.0f);
    amount = sqrt(amount);

    glm::vec4 color(1, 1, 1, amount);
    if (water_density > 1) {
        // make red
        float gb = 1 - tanh((water_density - 1) * 0.25f);
        color = glm::vec4(1, gb, gb, 1);
    } else if (water_density < 0) {
        // make blue
        float gb = 1 - tanh(-water_density);
        color = glm::vec4(gb, gb, 1, 1);
    }

    add_rect(x * size + size/2, y * size + size/2, size+1, size+1, 1, color, rect_vbo, i);
}

SoilSystem::SoilSystem(uint width, uint height, uint size, bool use_graphics) : width(width), height(height), size(size) {
    reset();
    if (use_graphics) {
        rect_renderer = std::make_unique<RectRenderer>();
    }
}

void SoilSystem::reset() {
    SoilSoA soil{};
    // assert(width % BLOCK_WIDTH == 0);
    // assert(height % BLOCK_WIDTH == 0);
    soil.resize_all(width * height);

    // for now, just put a lot in one cell in the center

    // const auto x = width / 2;
    // const auto y = height / 2;
    // const auto id = x + y * width;
    //
    // soil.water_density[id] = 127;

    FractalNoise water_noise(4, 0.01, width, 2.0, 0.5, 1234);

    for (auto y = 0; y < height; ++y) {
        for (auto x = 0; x < width; ++x) {
            const auto id = x + y * width;
            soil.water_density[id] = pow(abs(water_noise.eval(x, y)), 2);
        }
    }

    read.copy_from_host(soil);
    write.copy_from_host(soil);
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

void SoilSystem::mix_give_take_2_cuda(float dt) {
    // grab ptrs
    SoilPtrs read_ptrs{}, write_ptrs{};
    read_ptrs.get_ptrs(read);
    write_ptrs.get_ptrs(write);

    // set up grid
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(width + block.x - 1 / block.x, height + block.y - 1 / block.y);

    // launch kernel
    mix_give_take_2_kernel<<<grid, block>>>(read_ptrs, write_ptrs, width, height, dt);

    // kernel write to water_density, water_give_{left, right, up down}, so swap read and write
    write.water_density.swap(read.water_density);
    write.water_give_left.swap(read.water_give_left);
    write.water_give_right.swap(read.water_give_right);
    write.water_give_up.swap(read.water_give_up);
    write.water_give_down.swap(read.water_give_down);

    // re-acquire ptrs
    read_ptrs.get_ptrs(read);
    write_ptrs.get_ptrs(write);

    // launch copy_give kernel
    copy_give_kernel<<<grid, block>>>(read_ptrs, write_ptrs, width, height, dt);

    // kernel writes to water_density, so swap read and write
    write.water_density.swap(read.water_density);
}


void SoilSystem::update_cpu(float dt) {
    // TODO: implement
}


void SoilSystem::update_cuda(float dt) {
    // for now, just mix give take
    mix_give_take_cuda(dt);
    // mix_give_take_2_cuda(dt);
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
    render_kernel<<<grid, block>>>(static_cast<float*>(vbo_ptr), ptrs, width, size, rect_count);
    rect_renderer->cuda_unmap_buffer();

    rect_renderer->render(rect_count);
    rect_renderer->cuda_unregister_buffer();
}

void SoilSystem::add_water(int x, int y, float amount) {
    x /= size;
    y /= size;
    if (x > 0 && x < width && y > 0 && y < height) {
        read.water_density[x + y * width] += amount;
    }
}


