//
// Created by jonah on 4/25/2023.
//

#pragma once

#include "glm/glm.hpp"

class ExtendViewport {
public:
    ExtendViewport(float minWidth, float minHeight);
    void update(int screenWidth, int screenHeight);
    glm::mat4 get_transform();

    [[nodiscard]] float get_left() const;
    [[nodiscard]] float get_right() const;
    [[nodiscard]] float get_top() const;
    [[nodiscard]] float get_bottom() const;
    [[nodiscard]] float get_width() const;
    [[nodiscard]] float get_height() const;
    float x_cam{}, y_cam{};
    float zoom{1.0f};
    void handle_scroll(float yoffset);
    void handle_pan(float xoffset, float yoffset);

private:
    float min_width, min_height;
    float width{}, height{};
    int last_screen_width{}, last_screen_height{};
    glm::mat4 transform{};
};