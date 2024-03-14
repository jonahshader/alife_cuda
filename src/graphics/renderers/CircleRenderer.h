//
// Created by jonah on 4/18/2023.
//

#pragma once



#include <vector>
#include <glm/glm.hpp>
#include "graphics/Shader.h"

class CircleRenderer {
public:
    CircleRenderer();
    ~CircleRenderer();
    void begin();
    void end();
    void render();
    void add_circle(float x, float y, float radius, glm::vec4 color);
    void add_circle(float x, float y, float radius, unsigned char r, unsigned char g, unsigned char b, unsigned char a);
    void set_transform(glm::mat4 transform);

private:
    Shader shader;
    std::vector<unsigned int> data;
    unsigned int vbo_base_mesh{}, vbo_data{};
    unsigned int vao{};
    unsigned int buffer_size{};

};