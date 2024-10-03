//
// Created by jonah on 4/18/2023.
//

#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "graphics/Shader.h"

class CircleRenderer
{
public:
    CircleRenderer();
    ~CircleRenderer();
    void begin();
    void end();
    void render();
    void render(size_t circle_count);
    void add_circle(float x, float y, float radius, glm::vec4 color);
    void add_circle(float x, float y, float radius, unsigned char r, unsigned char g, unsigned char b, unsigned char a);
    void set_transform(glm::mat4 transform);

    // CUDA interop functions
    void cuda_register_buffer();
    void cuda_unregister_buffer();
    void *cuda_map_buffer();
    void cuda_unmap_buffer();

    // ensures that the vbo has at least the given size.
    // doubles the size of the vbo until it is greater than size.
    // if the vbo is bigger than 4x the size, it will be reset to the given size.
    void ensure_vbo_capacity(size_t circles);
    size_t get_circle_count() const { return buffer_size / CIRCLE_SIZE; }

    static const size_t ELEMS_PER_CIRCLE = 4;
    static const size_t CIRCLE_SIZE = ELEMS_PER_CIRCLE * sizeof(unsigned int);

private:
    Shader shader;
    std::vector<unsigned int> data;
    unsigned int vbo_base_mesh{}, vbo_data{};
    unsigned int vao{};
    unsigned int buffer_size{};
    cudaGraphicsResource_t cuda_resource{nullptr}; // CUDA graphics resource handle
};