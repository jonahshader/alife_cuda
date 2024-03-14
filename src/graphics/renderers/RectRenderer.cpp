//
// Created by jonah on 4/28/2023.
//

#include "RectRenderer.h"
#include <glad/glad.h>
#include <iostream>

RectRenderer::RectRenderer() : shader("shaders/rect.vert", "shaders/rect.frag")  {
    float baseMesh[] = {
            // t1
            -0.5f, -0.5f, // bottom left
            0.5f, -0.5f,  // bottom right
            0.5f, 0.5f,   // top right
            // t2
            0.5f, 0.5f,
            -0.5f, 0.5f,
            -0.5f, -0.5f,
    };

    // create vao, buffers
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo_data);
    glGenBuffers(1, &vbo_base_mesh);

    // buffer baseMesh
    glBindBuffer(GL_ARRAY_BUFFER, vbo_base_mesh);
    glBufferData(GL_ARRAY_BUFFER, sizeof(baseMesh), baseMesh, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW); // TODO: eval GL_DYNAMIC_DRAW

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_base_mesh);

    // x y
    glVertexAttribPointer(0, 2, GL_FLOAT, false, 2 * sizeof(float), (void*)0);

    // offset
    GLsizei s = 9 * sizeof(float);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
    glVertexAttribPointer(1, 2, GL_FLOAT, false, s, (void*)0); // x y sizes r g b
    glVertexAttribDivisor(1, 1);
    // size
    glVertexAttribPointer(2, 2, GL_FLOAT, false, s, (void*)(2 * sizeof(float)));
    glVertexAttribDivisor(2, 1);
    // radius
    glVertexAttribPointer(3, 1, GL_FLOAT, false, s, (void*)(4 * sizeof(float)));
    glVertexAttribDivisor(3, 1);
    // color
    glVertexAttribPointer(4, 4, GL_FLOAT, false, s, (void*)(5 * sizeof(float)));
    glVertexAttribDivisor(4, 1);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void RectRenderer::begin() {
    data.clear();
}

void RectRenderer::end() {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
    unsigned int data_bytes = data.size() * sizeof(data[0]);
    if (data_bytes > buffer_size) {
        // full update
        // current scheme: double buffer size
        // TODO: shrink buffer when data is less than half
        buffer_size = data_bytes * 2;
        glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);
        std::cout << "Doubled RectRenderer buffer size from " << buffer_size / 2 << " to " << buffer_size << std::endl;
    }
    glBufferSubData(GL_ARRAY_BUFFER, 0, data_bytes, data.data());

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RectRenderer::add_rect(float x, float y, float width, float height, float radius, glm::vec4 color) {
    data.emplace_back(x);
    data.emplace_back(y);
    data.emplace_back(width);
    data.emplace_back(height);
    data.emplace_back(radius);
    data.emplace_back(color.r);
    data.emplace_back(color.g);
    data.emplace_back(color.b);
    data.emplace_back(color.a);
}

void RectRenderer::render() {
    shader.use();
    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, data.size() / 9);
    glBindVertexArray(0);
}

RectRenderer::~RectRenderer() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo_base_mesh);
    glDeleteBuffers(1, &vbo_data);
}

void RectRenderer::set_transform(glm::mat4 transform) {
    shader.use();
    shader.setMatrix4("transform", transform);
}