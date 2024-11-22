#include "FontRenderer.h"

#include "lodepng.h"
#include <glad/glad.h>
#include <iostream>

FontRenderer::FontRenderer(std::string path) : shader("shaders/text.vert", "shaders/text.frag") {
  // load font
  FILE *const font_file = fopen(path.c_str(), "rb");
  if (font_file == nullptr) {
    std::cerr << "couldn't open font file" << std::endl;
    exit(1);
  }

  if (!artery_font::read(font, font_file)) {
    std::cerr << "couldn't read artery font" << std::endl;
    exit(1);
  }
  fclose(font_file);

  // create vao, vbo, texture
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);

  // buffer nothin into vbo
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

  // buffer texture
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  auto &image = font.images[0];

  std::vector<unsigned char> png;
  for (int i = 0; i < image.data.length(); i++) {
    png.emplace_back(image.data[i]);
  }

  std::vector<unsigned char> out;
  unsigned int w, h;
  lodepng::decode(out, w, h, png);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               out.data()); // no idea if this will work

  glBindVertexArray(vao);
  // define vertex attrib pointers for vbo
  // x, y, tx, ty, r, g, b, a
  GLsizei s = 8 * sizeof(float);
  // x, y
  glVertexAttribPointer(0, 2, GL_FLOAT, false, s, (void *)(0 * sizeof(float)));
  // tx, ty
  glVertexAttribPointer(1, 2, GL_FLOAT, false, s, (void *)(2 * sizeof(float)));
  // r, g, b, a
  glVertexAttribPointer(2, 4, GL_FLOAT, false, s, (void *)(4 * sizeof(float)));

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glEnableVertexAttribArray(2);

  glBindBuffer(GL_ARRAY_BUFFER, 0); // unbind
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindVertexArray(0); // unbind

  // populate maps
  for (int i = 0; i < font.variants[0].glyphs.length(); i++) {
    auto glyph = font.variants[0].glyphs[i];
    glyph_lookup[glyph.codepoint] = glyph;
  }
  for (int i = 0; i < font.variants[0].kernPairs.length(); i++) {
    auto kern_pair = font.variants[0].kernPairs[i];
    KernPairID kpi{};
    kpi.parts.first = kern_pair.codepoint1;
    kpi.parts.second = kern_pair.codepoint2;
    kern_pair_lookup[kpi.key] = kern_pair;
  }
}

void FontRenderer::begin() {
  data.clear();
}

void FontRenderer::end() {
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(data[0]), data.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0); // unbind
  glBindVertexArray(0);
}

void FontRenderer::add_text(float x, float y, float size, std::string text, glm::vec4 color) {
  auto last_char = text[0];
  add_char(last_char, x, y, size, color);
  for (int i = 1; i < text.length(); i++) {
    auto next_char = text[i];
    add_char(last_char, next_char, x, y, size, color);
    last_char = next_char;
  }
}

void FontRenderer::render() {
  shader.use();
  shader.setInt("msdf", 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glBindVertexArray(vao);
  glDrawArrays(GL_TRIANGLES, 0, data.size());
}

FontRenderer::~FontRenderer() {
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  // TODO: delete texture
}

void FontRenderer::add_char(glm::vec2 pos_begin, glm::vec2 pos_end, glm::vec2 tex_pos_begin,
                            glm::vec2 tex_pos_end, glm::vec4 color) {
  // x y
  data.emplace_back(pos_begin.x);
  data.emplace_back(pos_begin.y);
  // tx ty
  data.emplace_back(tex_pos_begin.x);
  data.emplace_back(tex_pos_begin.y);
  // rgb
  data.emplace_back(color.r);
  data.emplace_back(color.g);
  data.emplace_back(color.b);
  data.emplace_back(color.a);

  // x y
  data.emplace_back(pos_end.x);
  data.emplace_back(pos_begin.y);
  // tx ty
  data.emplace_back(tex_pos_end.x);
  data.emplace_back(tex_pos_begin.y);
  // rgb
  data.emplace_back(color.r);
  data.emplace_back(color.g);
  data.emplace_back(color.b);
  data.emplace_back(color.a);

  // x y
  data.emplace_back(pos_end.x);
  data.emplace_back(pos_end.y);
  // tx ty
  data.emplace_back(tex_pos_end.x);
  data.emplace_back(tex_pos_end.y);
  // rgb
  data.emplace_back(color.r);
  data.emplace_back(color.g);
  data.emplace_back(color.b);
  data.emplace_back(color.a);

  // x y
  data.emplace_back(pos_end.x);
  data.emplace_back(pos_end.y);
  // tx ty
  data.emplace_back(tex_pos_end.x);
  data.emplace_back(tex_pos_end.y);
  // rgb
  data.emplace_back(color.r);
  data.emplace_back(color.g);
  data.emplace_back(color.b);
  data.emplace_back(color.a);

  // x y
  data.emplace_back(pos_begin.x);
  data.emplace_back(pos_end.y);
  // tx ty
  data.emplace_back(tex_pos_begin.x);
  data.emplace_back(tex_pos_end.y);
  // rgb
  data.emplace_back(color.r);
  data.emplace_back(color.g);
  data.emplace_back(color.b);
  data.emplace_back(color.a);

  // x y
  data.emplace_back(pos_begin.x);
  data.emplace_back(pos_begin.y);
  // tx ty
  data.emplace_back(tex_pos_begin.x);
  data.emplace_back(tex_pos_begin.y);
  // rgb
  data.emplace_back(color.r);
  data.emplace_back(color.g);
  data.emplace_back(color.b);
  data.emplace_back(color.a);
}

void FontRenderer::add_char(unsigned char last_c, unsigned char c, float &x, float &y, float size,
                            glm::vec4 color) {
  auto &glyph = glyph_lookup[c];
  glm::vec2 pos_begin, pos_end, tx_begin, tx_end;
  pos_begin.x = x + glyph.planeBounds.l * size;
  pos_begin.y = y + glyph.planeBounds.b * size;
  ;
  pos_end.x = x + glyph.planeBounds.r * size;
  pos_end.y = y + glyph.planeBounds.t * size;
  tx_begin.x = glyph.imageBounds.l / (float)font.images[0].width;
  tx_begin.y = 1 - glyph.imageBounds.b / (float)font.images[0].height;
  tx_end.x = glyph.imageBounds.r / (float)font.images[0].width;
  tx_end.y = 1 - glyph.imageBounds.t / (float)font.images[0].height;

  KernPairID id{};
  id.parts.first = last_c;
  id.parts.second = c;

  if (kern_pair_lookup.find(id.key) != kern_pair_lookup.end()) {
    auto &kern_pair = kern_pair_lookup[id.key];
    pos_begin.x += kern_pair.advance.h * size;
    pos_begin.y += kern_pair.advance.v * size;
    pos_end.x += kern_pair.advance.h * size;
    pos_end.y += kern_pair.advance.v * size;
    x += kern_pair.advance.h * size;
    y += kern_pair.advance.v * size;
  }

  x += glyph.advance.h * size;
  y += glyph.advance.v * size;

  add_char(pos_begin, pos_end, tx_begin, tx_end, color);
}

float FontRenderer::compute_advance_amount(unsigned char last_c, unsigned char c, float size) {
  auto &glyph = glyph_lookup[c];
  KernPairID id{};
  id.parts.first = last_c;
  id.parts.second = c;

  float length = 0;
  if (kern_pair_lookup.find(id.key) != kern_pair_lookup.end()) {
    auto &kern_pair = kern_pair_lookup[id.key];
    length = kern_pair.advance.h * size;
  }
  return length + glyph.advance.h * size;
}

float FontRenderer::compute_advance_amount(unsigned char c, float size) {
  auto &glyph = glyph_lookup[c];
  return glyph.advance.h * size;
}

void FontRenderer::add_char(unsigned char c, float &x, float &y, float size, glm::vec4 color) {
  auto &glyph = glyph_lookup[c];
  glm::vec2 pos_begin, pos_end, tx_begin, tx_end;
  pos_begin.x = x + glyph.planeBounds.l * size;
  pos_begin.y = y + glyph.planeBounds.b * size;
  ;
  pos_end.x = x + glyph.planeBounds.r * size;
  pos_end.y = y + glyph.planeBounds.t * size;
  tx_begin.x = glyph.imageBounds.l / (float)font.images[0].width;
  tx_begin.y = 1 - glyph.imageBounds.b / (float)font.images[0].height;
  tx_end.x = glyph.imageBounds.r / (float)font.images[0].width;
  tx_end.y = 1 - glyph.imageBounds.t / (float)font.images[0].height;

  x += glyph.advance.h * size;
  y += glyph.advance.v * size;

  add_char(pos_begin, pos_end, tx_begin, tx_end, color);
}

void FontRenderer::set_transform(glm::mat4 transform) {
  shader.use();
  shader.setMatrix4("transform", transform);
}

float FontRenderer::get_length(float size, std::string text) {
  auto last_char = text[0];
  float length = compute_advance_amount(last_char, size);
  for (int i = 1; i < text.length(); i++) {
    auto next_char = text[i];
    length += compute_advance_amount(last_char, next_char, size);
    last_char = next_char;
  }
  return length;
}

void FontRenderer::add_text(float x, float y, float targetWidth, const std::string &text,
                            glm::vec4 color, HAlign hAlign) {
  float l = get_length(1, text);
  float size = targetWidth / l;
  switch (hAlign) {
    case HAlign::LEFT:
      add_text(x - targetWidth, y, size, text, color);
      break;
    case HAlign::CENTER:
      add_text(x - targetWidth / 2, y, size, text, color);
      break;
    case HAlign::RIGHT:
      add_text(x, y, size, text, color);
      break;
  }
}
