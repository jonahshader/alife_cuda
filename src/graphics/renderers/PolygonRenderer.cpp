#include "PolygonRenderer.h"
#include <glad/glad.h>

PolygonRenderer::PolygonRenderer() : shader("shaders/polygon.vert", "shaders/polygon.frag") {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
}

void PolygonRenderer::add_polygon(const glm::vec4& color, const std::vector<glm::vec2>& points) {
    colors.push_back(color);
    polygon_points.insert(polygon_points.end(), points.begin(), points.end());
}

void PolygonRenderer::render() {
    shader.use();
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, polygon_points.size() * sizeof(glm::vec2), polygon_points.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    int offset = 0;
    for (const auto& color : colors) {
        shader.setVec4("color", color);
        glDrawArrays(GL_TRIANGLE_FAN, offset, polygon_points.size() / colors.size());
        offset += polygon_points.size() / colors.size();
    }

    glBindVertexArray(0);
}

PolygonRenderer::~PolygonRenderer() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}