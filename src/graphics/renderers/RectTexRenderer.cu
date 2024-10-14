#include "RectTexRenderer.cuh"

#include <cuda_gl_interop.h>
#include <glad/glad.h>
#include <iostream>

RectTexRenderer::RectTexRenderer(int width, int height, int channels) : 
shader("shaders/rect_tex.vert", "shaders/rect_tex.frag"),
width(width), height(height), channels(channels)
{
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);

  // set the texture wrapping/filtering options (on the currently bound texture object)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


  // initial pattern will be a checkerboard
  std::vector<unsigned char> data(width * height * channels);
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int i = (y * width + x) * channels;
      for (int c = 0; c < channels; c++)
      {
        data[i + c] = (x / 16 + y / 16) % 2 == 0 ? 255 : 0;
      }
    }
  }

  // determine format from number of channels
  int format = 0;
  switch (channels)
  {
  case 1:
    format = GL_RED;
    break;
  case 2:
    format = GL_RG;
    break;
  case 3:
    format = GL_RGB;
    break;
  case 4:
    format = GL_RGBA;
    break;
  default:
    std::cerr << "RectTexRenderer: Invalid number of channels: " << channels << std::endl;
    exit(1);
  }

  // load the texture
  glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data.data());

  // create vao, buffers
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo_data);

  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_data);

  // position attribute
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  // color attribute
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);
  // texture coord attribute
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(5 * sizeof(float)));
  glEnableVertexAttribArray(2);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void RectTexRenderer::begin()
{
  data.clear();
}

void RectTexRenderer::end()
{
  glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
  unsigned int data_bytes = data.size() * sizeof(data[0]);
  if (data_bytes > buffer_size)
  {
    // full update
    // current scheme: double buffer size
    // TODO: shrink buffer when data is less than half
    buffer_size = data_bytes * 2;
    glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);
    std::cout << "Doubled RectTexRenderer buffer size from " << buffer_size / 2 << " to " << buffer_size << std::endl;
  }
  glBufferSubData(GL_ARRAY_BUFFER, 0, data_bytes, data.data());
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RectTexRenderer::add_vertex(float x, float y, glm::vec3 color, float tex_x, float tex_y)
{
  data.push_back(x);
  data.push_back(y);
  data.push_back(color.r);
  data.push_back(color.g);
  data.push_back(color.b);
  data.push_back(tex_x);
  data.push_back(tex_y);
}

void RectTexRenderer::add_rect(float x, float y, float width, float height, glm::vec3 color, float tex_x, float tex_y, float tex_width, float tex_height)
{
  // TODO: verify
  float left = x;
  float right = x + width;
  float top = y;
  float bottom = y + height;
  float left_tex = tex_x;
  float right_tex = tex_x + tex_width;
  float top_tex = tex_y;
  float bottom_tex = tex_y + tex_height;

  add_vertex(left, top, color, left_tex, top_tex);
  add_vertex(right, top, color, right_tex, top_tex);
  add_vertex(right, bottom, color, right_tex, bottom_tex);

  add_vertex(left, top, color, left_tex, top_tex);
  add_vertex(right, bottom, color, right_tex, bottom_tex);
  add_vertex(left, bottom, color, left_tex, bottom_tex);
}

void RectTexRenderer::add_rect(float x, float y, float width, float height, glm::vec3 color)
{
  add_rect(x, y, width, height, color, 0, 0, 1, 1);
}

// TODO: this is a generic shader, so we should add more than just rectangles here

void RectTexRenderer::set_transform(glm::mat4 transform)
{
  shader.use();
  shader.setMatrix4("transform", transform);
}

void RectTexRenderer::render()
{
  shader.use();
  glBindVertexArray(vao);
  glDrawArrays(GL_TRIANGLES, 0, data.size() / 7);
  glBindVertexArray(0);
}

RectTexRenderer::~RectTexRenderer()
{
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo_data);
  glDeleteTextures(1, &tex);
}

void RectTexRenderer::cuda_register_buffer() {
  cudaGraphicsGLRegisterBuffer(&cuda_resource, tex, cudaGraphicsMapFlagsWriteDiscard);
}

void RectTexRenderer::cuda_unregister_buffer() {
  cudaGraphicsUnregisterResource(cuda_resource);
}

void *RectTexRenderer::cuda_map_buffer() {
  void* device_ptr;
  size_t size;
  cudaGraphicsMapResources(1, &cuda_resource, 0);
  cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, cuda_resource);
  return device_ptr;
}

void RectTexRenderer::cuda_unmap_buffer() {
  cudaGraphicsUnmapResources(1, &cuda_resource, 0);
}

void RectTexRenderer::set_filtering(int min, int mag) {
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag);
}