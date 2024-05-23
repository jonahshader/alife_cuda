#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "graphics/Shader.h"

class PolygonRenderer {
public:
    PolygonRenderer();
    ~PolygonRenderer();
    void add_polygon(const glm::vec4& color, const std::vector<glm::vec2>& points);
    void render();

private:
    Shader shader;
    unsigned int vao{};
    unsigned int vbo{};
    std::vector<glm::vec4> colors;
    std::vector<glm::vec2> polygon_points;
};