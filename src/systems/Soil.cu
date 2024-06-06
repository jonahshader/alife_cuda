#include "Soil.cuh"

#include <cmath>
#include <cstdint>
#include <memory>
#include <glm/glm.hpp>




namespace soil {
    constexpr auto FILTER_WIDTH = 3;
    constexpr auto BLOCK_WIDTH = 16;
    constexpr auto BLOCK_HEIGHT = BLOCK_WIDTH;

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
    inline void mix_give_take(SoilPtrs &read, SoilPtrs &write, uint width, uint height, float dt, uint x, uint y) {

        const auto id = x + y * width;

        const float give_per_sec = 1.0f; // TODO: this should be computed from soil composition

        const auto water = read.water_density[id];

        float water_delta = 0;
        // for now, hard code to going down

        auto water_other = get_cell_w_wrap(read.water_density, x - 1, y, width);
        if (water > water_other) {
            // give left water
            water_delta -= give_per_sec;
        } else if (water < water_other) {
            // take left water
            // TODO: calculate give rate
            water_delta += give_per_sec;
        }
        water_other = get_cell_w_wrap(read.water_density, x + 1, y, width);
        if (water > water_other) {
            // give right water
            water_delta -= give_per_sec;
        } else if (water < water_other) {
            // take right water
            // TODO: calculate give rate
            water_delta += give_per_sec;
        }
        if (y > 0) {
            water_other = get_cell_w_wrap(read.water_density, x, y - 1, width);
            // only consider taking from top
            if (water < water_other) {
                // take up water
                // TODO: calculate give rate
                water_delta += give_per_sec;
            }
        } else if (y < height - 1) {
            water_other = get_cell_w_wrap(read.water_density, x, y + 1, width);
            // only consider giving to bottom
            if (water > water_other) {
                // give down water
                // TODO: calculate give rate
                water_delta -= give_per_sec;
            }
        }

        write.water_density[id] = water + water_delta * dt;
    }

    __global__
    void mix_give_take_kernel(SoilPtrs read, SoilPtrs write, uint width, uint height, float dt) {
        const auto x = blockIdx.x * blockDim.x + threadIdx.x;
        const auto y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        mix_give_take(read, write, width, height, dt, x, y);
    }

    __global__
    void render_kernel(float* rect_vbo, SoilPtrs read, uint width, size_t rect_count) {
        const auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= rect_count) {
            return;
        }
        const auto water_density = read.water_density[i];
        const auto x = i % width;
        const auto y = i / width;
    }

    SoilSystem::SoilSystem(uint width, uint height, bool use_graphics) : width(width), height(height) {
        SoilSoA soil{};
        assert(width % BLOCK_WIDTH == 0, "Width not multiple of BLOCK_WIDTH");
        assert(height % BLOCK_WIDTH == 0, "Height not multiple of BLOCK_WIDTH");
        soil.resize_all(width * height);

        // for now, just put a lot in one cell in the center

        const auto x = width / 2;
        const auto y = height / 2;
        const auto id = x + y * width;

        soil.water_density[id] = 32;

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
        // TODO render_soil_kernel
        rect_renderer->cuda_unmap_buffer();

        rect_renderer->render(rect_count);
        rect_renderer->cuda_unregister_buffer();
    }

}
