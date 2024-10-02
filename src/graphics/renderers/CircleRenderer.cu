//
// Created by jonah on 4/18/2023.
//

#include "CircleRenderer.cuh"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <iostream>

CircleRenderer::CircleRenderer() : shader("shaders/circle.vert", "shaders/circle.frag")
{
    float baseMesh[] = {
        // t1
        -0.5f,
        -0.5f, // bottom left
        0.5f,
        -0.5f, // bottom right
        0.5f,
        0.5f, // top right
        // t2
        0.5f,
        0.5f,
        -0.5f,
        0.5f,
        -0.5f,
        -0.5f,
    };

    // create vao, buffers
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo_data);
    glGenBuffers(1, &vbo_base_mesh);

    // buffer baseMesh
    glBindBuffer(GL_ARRAY_BUFFER, vbo_base_mesh);
    glBufferData(GL_ARRAY_BUFFER, sizeof(baseMesh), baseMesh, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW); // TODO: eval GL_DYNAMIC_DRAW

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_base_mesh);

    // x y
    glVertexAttribPointer(0, 2, GL_FLOAT, false, 2 * sizeof(float), (void *)0);

    // offset
    GLsizei s = 4 * sizeof(float);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
    glVertexAttribPointer(1, 2, GL_FLOAT, false, s, (void *)0); // x y sizes r g b
    glVertexAttribDivisor(1, 1);
    // size
    glVertexAttribPointer(2, 1, GL_FLOAT, false, s, (void *)(2 * sizeof(float)));
    glVertexAttribDivisor(2, 1);
    // color (rgba)
    glVertexAttribPointer(3, 4, GL_UNSIGNED_BYTE, true, s, (void *)(3 * sizeof(float)));
    glVertexAttribDivisor(3, 1);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void CircleRenderer::begin()
{
    data.clear();
}

void CircleRenderer::end()
{
    glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
    unsigned int data_bytes = data.size() * sizeof(data[0]);
    if (data_bytes > buffer_size)
    {
        // resize buffer
        buffer_size = data_bytes * 2;
        glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);
        std::cout << "Doubled CircleRenderer buffer size from " << buffer_size / 2 << " to " << buffer_size << std::endl;
    }
    glBufferSubData(GL_ARRAY_BUFFER, 0, data_bytes, data.data());

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void CircleRenderer::add_circle(float x, float y, float radius, glm::vec4 color)
{
    add_circle(x, y, radius, color.r * 255, color.g * 255, color.b * 255, 255);
}

void CircleRenderer::render()
{
    shader.use();
    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, data.size() / ELEMS_PER_CIRCLE);
    glBindVertexArray(0);
}

CircleRenderer::~CircleRenderer()
{
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_base_mesh);
    glDeleteBuffers(1, &vbo_data);
}

void CircleRenderer::set_transform(glm::mat4 transform)
{
    shader.use();
    shader.setMatrix4("transform", transform);
}

void CircleRenderer::add_circle(float x, float y, float radius, unsigned char r, unsigned char g, unsigned char b,
                                unsigned char a)
{
    radius *= 2;
    data.emplace_back(reinterpret_cast<unsigned int &>(x));
    data.emplace_back(reinterpret_cast<unsigned int &>(y));
    data.emplace_back(reinterpret_cast<unsigned int &>(radius));
    // pack color into a single unsigned int
    unsigned int color = 0;
    color |= r;
    color |= g << 8;
    color |= b << 16;
    color |= a << 24;

    data.emplace_back(color);
}

void CircleRenderer::cuda_register_buffer()
{
    cudaGraphicsGLRegisterBuffer(&cuda_resource, vbo_data, cudaGraphicsMapFlagsWriteDiscard);
}

void CircleRenderer::cuda_unregister_buffer()
{
    cudaGraphicsUnregisterResource(cuda_resource);
}

void *CircleRenderer::cuda_map_buffer()
{
    void *device_ptr;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_resource, 0);
    cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, cuda_resource);
    return device_ptr;
}

void CircleRenderer::cuda_unmap_buffer()
{
    cudaGraphicsUnmapResources(1, &cuda_resource, 0);
}

void CircleRenderer::ensure_vbo_capacity(size_t circles)
{
    const auto size_bytes = circles * CIRCLE_SIZE;
    if (buffer_size < size_bytes) {
        if (buffer_size == 0) {
            buffer_size = size_bytes;
        } else {
            while (buffer_size < size_bytes) {
                buffer_size *= 2;
            }
        }

        std::cout << "LineRenderer buffer size changed to " << buffer_size << std::endl;

        glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
        glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    } else if (buffer_size > size_bytes * 4) {
        buffer_size = size_bytes;
        glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
        glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}