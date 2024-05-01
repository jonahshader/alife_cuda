//
// Created by jonah on 4/22/2023.
//

#pragma once



#include <vector>
#include <string>
#include <unordered_map>

#include <std-artery-font.h>
#include <stdio-serialization.h>

#include "graphics/Shader.h"
#include "glm/glm.hpp"


class FontRenderer {
public:
    enum class HAlign {
        LEFT,
        CENTER,
        RIGHT
    };
    enum class VAlign {
        TOP,
        CENTER,
        BOTTOM,
    };
    explicit FontRenderer(std::string path);
    ~FontRenderer();
    void begin();
    void end();
    void set_transform(glm::mat4 transform);
    void add_text(float x, float y, float size, std::string text, glm::vec4 color);
    void add_text(float x, float y, float targetWidth, const std::string& text, glm::vec4 color, HAlign hAlign);
    float get_length(float size, std::string text); // get length of string in pixels/units idk
    void render();

private:
    union KernPairID {
        struct {
            unsigned char first;
            unsigned char second;
        } parts;
        unsigned short key;
    };

    Shader shader;
    artery_font::StdArteryFont<float> font;
    std::vector<float> data;
    unsigned int vbo{};
    unsigned int vao{};
    unsigned int texture{};
    std::unordered_map<unsigned char, artery_font::Glyph<float>> glyph_lookup;
    std::unordered_map<unsigned short, artery_font::KernPair<float>> kern_pair_lookup;

    void add_char(unsigned char last_c, unsigned char c, float &x, float &y, float size, glm::vec4 color);
    void add_char(unsigned char c, float &x, float &y, float size, glm::vec4 color);
    void add_char(glm::vec2 pos_begin, glm::vec2 pos_end, glm::vec2 tex_pos_begin, glm::vec2 tex_pos_end, glm::vec4 color);
    float compute_advance_amount(unsigned char last_c, unsigned char c, float size);
    float compute_advance_amount(unsigned char c, float size);

};