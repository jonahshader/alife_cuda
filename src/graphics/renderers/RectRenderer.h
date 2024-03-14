//
// Created by jonah on 4/28/2023.
//

#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "graphics/Shader.h"

class RectRenderer {
public:
    RectRenderer();
    ~RectRenderer();
    void begin();
    void end();
    void render();
    void add_rect(float x, float y, float width, float height, float radius, glm::vec4 color);
    void set_transform(glm::mat4 transform);

private:
    Shader shader;
    std::vector<float> data;
    unsigned int vbo_base_mesh{}, vbo_data{};
    unsigned int vao{};
    unsigned int buffer_size{};

};