//
// Created by jonah on 5/7/2023.
//


#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "graphics/Shader.h"

class LineRenderer {
public:
    LineRenderer();
    ~LineRenderer();
    void begin();
    void end();
    void render();
    void add_line(float x1, float y1, float x2, float y2, float radius, const glm::vec4 &color);
    void add_line(float x1, float y1, float x2, float y2, float r1, float r2, const glm::vec4 &color1, const glm::vec4 &color2);
    void add_line(const glm::vec2 &v1, const glm::vec2 &v2, float r1, float r2, const glm::vec4 &color1, const glm::vec4 &color2);
    void set_transform(glm::mat4 transform);


    // CUDA interop functions
    void cudaRegisterBuffer();
    void cudaUnregisterBuffer();
    void* cudaMapBuffer();
    void cudaUnmapBuffer();

private:
    Shader shader;
    std::vector<unsigned int> data;
    unsigned int vbo{};
    unsigned int vao{};
    unsigned int buffer_size{};
    cudaGraphicsResource_t cudaResource{nullptr}; // CUDA graphics resource handle
    void add_vertex(float x, float y, float tx, float ty, float length, float radius, const glm::vec4 &color);
    void add_vertex(float x, float y, float tx, float ty, float length, float radius, unsigned char red, unsigned char green, unsigned char blue, unsigned char alpha);
};