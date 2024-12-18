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
  void render(size_t line_count);
  void add_line(float x1, float y1, float x2, float y2, float radius, const glm::vec4 &color);
  void add_line(float x1, float y1, float x2, float y2, float r1, float r2, const glm::vec4 &color1,
                const glm::vec4 &color2);
  void add_line(const glm::vec2 &v1, const glm::vec2 &v2, float r1, float r2,
                const glm::vec4 &color1, const glm::vec4 &color2);
  void set_transform(glm::mat4 transform);

  // CUDA interop functions
  void cuda_register_buffer();
  void cuda_unregister_buffer();
  void *cuda_map_buffer();
  void cuda_unmap_buffer();

  // ensures that the vbo has at least the given size.
  // doubles the size of the vbo until it is greater than size.
  // if the vbo is bigger than 4x the size, it will be reset to the given size.
  void ensure_vbo_capacity(size_t size);

  static const size_t VERTEX_ELEMS = 7;
  static const size_t VERTEX_BYTES = VERTEX_ELEMS * sizeof(float);
  static const size_t VERTICES_PER_LINE = 6;

private:
  Shader shader;
  std::vector<unsigned int> data;
  unsigned int vbo{};
  unsigned int vao{};
  unsigned int buffer_size{};
  cudaGraphicsResource_t cuda_resource{nullptr}; // CUDA graphics resource handle
  void add_vertex(float x, float y, float tx, float ty, float length, float radius,
                  const glm::vec4 &color);
  void add_vertex(float x, float y, float tx, float ty, float length, float radius,
                  unsigned char red, unsigned char green, unsigned char blue, unsigned char alpha);
};
