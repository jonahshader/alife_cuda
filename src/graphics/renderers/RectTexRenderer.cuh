#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "graphics/Shader.h"
#include <glad/glad.h>

class RectTexRenderer {
public:
  RectTexRenderer(int width, int height, int channels);
  ~RectTexRenderer();
  void begin();
  void end();
  void render(); // use when rects are defined by host
  void add_rect(float x, float y, float width, float height, glm::vec3 color); // tex streches to fit rect
  void add_rect(float x, float y, float width, float height, glm::vec3 color, float tex_x, float tex_y, float tex_width, float tex_height); // tex has custom bounds
  void set_transform(glm::mat4 transform);

  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  void set_filtering(int min, int mag);

  // CUDA interop functions
  // TODO: look into making this more dry
  // for this renderer, the buffer is referring to the texture, not the vbo
  void cuda_register_buffer();
  void cuda_unregister_buffer();
  void* cuda_map_buffer();
  void cuda_unmap_buffer();

private:
  Shader shader;
  std::vector<float> data{};
  unsigned int vbo_data{};
  unsigned int tex{};
  unsigned int vao{};
  unsigned int buffer_size{};

  cudaGraphicsResource_t cuda_resource{nullptr};

  void add_vertex(float x, float y, glm::vec3 color, float tex_x, float tex_y);
};