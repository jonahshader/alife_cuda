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
  void render(size_t rect_count);
  void add_rect(float x, float y, float width, float height, float radius, glm::vec4 color);
  void set_transform(glm::mat4 transform);

  // CUDA interop functions
  // TODO: look into making this more dry
  void cuda_register_buffer();
  void cuda_unregister_buffer();
  void *cuda_map_buffer();
  void cuda_unmap_buffer();

  // ensures that the vbo has at least the given size.
  // doubles the size of the vbo until it is greater than size.
  // if the vbo is bigger than 4x the size, it will be reset to the given size.
  void ensure_vbo_capacity(size_t size);

  static constexpr auto FLOATS_PER_RECT = 9;
  static constexpr auto BYTES_PER_RECT = FLOATS_PER_RECT * sizeof(float);

private:
  Shader shader;
  std::vector<float> data;
  unsigned int vbo_base_mesh{}, vbo_data{};
  unsigned int vao{};
  unsigned int buffer_size{};

  cudaGraphicsResource_t cuda_resource{nullptr};
};
