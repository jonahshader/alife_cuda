#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "graphics/Shader.h"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class RectTexRenderer {
public:
  RectTexRenderer(int width, int height, int channels);
  ~RectTexRenderer();
  void begin();
  void end();
  void render();
  void add_rect(float x, float y, float width, float height, glm::vec3 color);
  void add_rect(float x, float y, float width, float height, glm::vec3 color, float tex_x,
                float tex_y, float tex_width, float tex_height);
  void set_transform(glm::mat4 transform);
  void set_filtering(int min, int mag);

  int get_width() const {
    return width;
  }
  int get_height() const {
    return height;
  }
  int get_channels() const {
    return channels;
  }

  // CUDA interop functions
  void cuda_register_texture();
  void cuda_unregister_texture();
  cudaArray *cuda_map_texture();
  void cuda_unmap_texture();
  cudaTextureObject_t create_texture_object();
  void destroy_texture_object(cudaTextureObject_t texObj);
  void update_texture_from_cuda(void *device_data);

private:
  Shader shader;
  int width;
  int height;
  int channels;
  std::vector<float> data{};
  unsigned int vbo_data{};
  unsigned int tex{};
  unsigned int vao{};
  unsigned int buffer_size{};

  cudaGraphicsResource_t cuda_resource{nullptr};
  cudaArray *cuda_array{nullptr};

  void add_vertex(float x, float y, glm::vec3 color, float tex_x, float tex_y);

  void check_cuda(const std::string &msg);
};
