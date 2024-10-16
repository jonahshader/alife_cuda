#include "RectTexRenderer.cuh"

#include <cuda_gl_interop.h>
#include <glad/glad.h>
#include <iostream>

RectTexRenderer::RectTexRenderer(int width, int height, int channels) : shader("shaders/rect_tex.vert", "shaders/rect_tex.frag"),
                                                                        width(width), height(height), channels(channels)
{
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);

  // set the texture wrapping/filtering options (on the currently bound texture object)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // determine format from number of channels
  GLenum format;
  switch (channels)
  {
  case 4:
    format = GL_RGBA;
    break;
  default:
    std::cerr << "RectTexRenderer: Invalid number of channels: " << channels << std::endl;
    exit(1);
  }

  // Ensure that the texture is 4-byte aligned
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

  // Allocate texture memory
  glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, nullptr);

  // create vao, buffers
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo_data);

  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_data);

  // position attribute
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  // color attribute
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);
  // texture coord attribute
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(5 * sizeof(float)));
  glEnableVertexAttribArray(2);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  cuda_register_texture();
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

void RectTexRenderer::set_transform(glm::mat4 transform)
{
  shader.use();
  shader.setMatrix4("transform", transform);
}

void RectTexRenderer::render()
{
  shader.use();
  glBindVertexArray(vao);
  glBindTexture(GL_TEXTURE_2D, tex);
  glDrawArrays(GL_TRIANGLES, 0, data.size() / 7);
  glBindVertexArray(0);
}

RectTexRenderer::~RectTexRenderer()
{
  cuda_unregister_texture();
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo_data);
  glDeleteTextures(1, &tex);
  if (cuda_array) {
    cudaFreeArray(cuda_array);
  }
}

void RectTexRenderer::cuda_register_texture()
{
  // Register the OpenGL texture with CUDA
  cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_resource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
  if (err != cudaSuccess)
  {
    std::cerr << "Failed to register OpenGL texture with CUDA: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

void RectTexRenderer::cuda_unregister_texture()
{
  if (cuda_resource)
  {
    cudaGraphicsUnregisterResource(cuda_resource);
    cuda_resource = nullptr;
  }
}

cudaArray *RectTexRenderer::cuda_map_texture()
{
  // Map the CUDA resource
  cudaError_t err = cudaGraphicsMapResources(1, &cuda_resource, 0);
  if (err != cudaSuccess)
  {
    std::cerr << "Failed to map CUDA resource: " << cudaGetErrorString(err) << std::endl;
    return nullptr;
  }

  // Get the mapped array
  err = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);
  if (err != cudaSuccess)
  {
    std::cerr << "Failed to get mapped array: " << cudaGetErrorString(err) << std::endl;
    cudaGraphicsUnmapResources(1, &cuda_resource, 0);
    return nullptr;
  }

  return cuda_array;
}

void RectTexRenderer::cuda_unmap_texture()
{
  cudaError_t err = cudaGraphicsUnmapResources(1, &cuda_resource, 0);
  if (err != cudaSuccess)
  {
    std::cerr << "Failed to unmap CUDA resource: " << cudaGetErrorString(err) << std::endl;
  }
}

cudaTextureObject_t RectTexRenderer::create_texture_object() {
  // Create resource description
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuda_array;

  // Create texture description
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
  if (err != cudaSuccess) {
    std::cerr << "Failed to create CUDA texture object: " << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  return texObj;
}

void RectTexRenderer::destroy_texture_object(cudaTextureObject_t texObj)
{
  cudaDestroyTextureObject(texObj);
}

void RectTexRenderer::update_texture_from_cuda(void *device_data)
{
  cudaError_t err = cudaMemcpy2DToArray(cuda_array, 0, 0, device_data, width * channels, width * channels, height, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess)
  {
    std::cerr << "Failed to copy data to CUDA array: " << cudaGetErrorString(err) << std::endl;
  }
}

void RectTexRenderer::set_filtering(int min, int mag)
{
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag);
}