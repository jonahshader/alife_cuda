#include "SimpleRectRenderer.cuh"
#include <glad/glad.h>
#include <iostream>

#include <cuda_gl_interop.h>

SimpleRectRenderer::SimpleRectRenderer()
    : shader("shaders/rect_simple.vert", "shaders/rect_simple.frag") {
  // float base_mesh[] = {
  //     // t1
  //     -0.5f,
  //     -0.5f, // bottom left
  //     0.5f,
  //     -0.5f, // bottom right
  //     0.5f,
  //     0.5f, // top right
  //     // t2
  //     0.5f,
  //     0.5f,
  //     -0.5f,
  //     0.5f,
  //     -0.5f,
  //     -0.5f,
  // };

  float base_mesh[] = {
      // t1
      0.0f,
      0.0f, // bottom left
      1.0f,
      0.0f, // bottom right
      1.0f,
      1.0f, // top right
      // t2
      1.0f,
      1.0f,
      0.0f,
      1.0f,
      0.0f,
      0.0f,
  };

  // create vao, buffers
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo_data);
  glGenBuffers(1, &vbo_base_mesh);

  // buffer baseMesh
  glBindBuffer(GL_ARRAY_BUFFER, vbo_base_mesh);
  glBufferData(GL_ARRAY_BUFFER, sizeof(base_mesh), base_mesh, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
  glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW); // TODO: eval GL_DYNAMIC_DRAW

  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_base_mesh);

  // x y
  glVertexAttribPointer(0, 2, GL_FLOAT, false, 2 * sizeof(float), (void *)0);

  // offset
  glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
  glVertexAttribPointer(1, 2, GL_FLOAT, false, BYTES_PER_RECT, (void *)0); // x y sizes r g b
  glVertexAttribDivisor(1, 1);
  // size
  glVertexAttribPointer(2, 2, GL_FLOAT, false, BYTES_PER_RECT, (void *)(2 * sizeof(float)));
  glVertexAttribDivisor(2, 1);
  // color
  glVertexAttribPointer(3, 4, GL_FLOAT, false, BYTES_PER_RECT, (void *)(4 * sizeof(float)));
  glVertexAttribDivisor(3, 1);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glEnableVertexAttribArray(2);
  glEnableVertexAttribArray(3);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void SimpleRectRenderer::begin() {
  data.clear();
}

void SimpleRectRenderer::end() {
  glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
  unsigned int data_bytes = data.size() * sizeof(data[0]);
  if (data_bytes > buffer_size) {
    // full update
    // current scheme: double buffer size
    // TODO: shrink buffer when data is less than half
    buffer_size = data_bytes * 2;
    glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);
    std::cout << "Doubled SimpleRectRenderer buffer size from " << buffer_size / 2 << " to "
              << buffer_size << std::endl;
  }
  glBufferSubData(GL_ARRAY_BUFFER, 0, data_bytes, data.data());
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SimpleRectRenderer::add_rect(float x, float y, float width, float height, glm::vec4 color) {
  data.emplace_back(x);
  data.emplace_back(y);
  data.emplace_back(width);
  data.emplace_back(height);
  data.emplace_back(color.r);
  data.emplace_back(color.g);
  data.emplace_back(color.b);
  data.emplace_back(color.a);
}

void SimpleRectRenderer::render() {
  shader.use();
  glBindVertexArray(vao);
  glDrawArraysInstanced(GL_TRIANGLES, 0, 6, data.size() / FLOATS_PER_RECT);
  glBindVertexArray(0);
}

void SimpleRectRenderer::render(size_t rect_count) {
  shader.use();
  glBindVertexArray(vao);
  glDrawArraysInstanced(GL_TRIANGLES, 0, 6, rect_count);
  glBindVertexArray(0);
}

SimpleRectRenderer::~SimpleRectRenderer() {
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo_base_mesh);
  glDeleteBuffers(1, &vbo_data);
}

void SimpleRectRenderer::set_transform(glm::mat4 transform) {
  shader.use();
  shader.setMatrix4("transform", transform);
}

void SimpleRectRenderer::cuda_register_buffer() {
  // cudaGraphicsMapFlagsWriteDiscard: cuda will only write, not read, to this buffer
  cudaGraphicsGLRegisterBuffer(&cuda_resource, vbo_data, cudaGraphicsMapFlagsWriteDiscard);
}

void SimpleRectRenderer::cuda_unregister_buffer() {
  cudaGraphicsUnregisterResource(cuda_resource);
}

void *SimpleRectRenderer::cuda_map_buffer() {
  void *device_ptr;
  size_t size;
  cudaGraphicsMapResources(1, &cuda_resource, 0);
  cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, cuda_resource);
  return device_ptr;
}

void SimpleRectRenderer::cuda_unmap_buffer() {
  cudaGraphicsUnmapResources(1, &cuda_resource, 0);
}

void SimpleRectRenderer::ensure_vbo_capacity(size_t num_rects) {
  const auto size_bytes = num_rects * BYTES_PER_RECT;
  if (buffer_size < size_bytes) {
    if (buffer_size == 0) {
      buffer_size = size_bytes;
    } else {
      while (buffer_size < size_bytes) {
        buffer_size *= 2;
      }
    }

    std::cout << "SimpleRectRenderer buffer size changed to " << buffer_size << std::endl;

    glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  } else if (buffer_size > size_bytes * 4) {
    buffer_size = size_bytes;
    glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
}
