#include "tree_render.cuh"

#include <random>

namespace trees {

void render_tree(LineRenderer &line_renderer, const trees2::TreeBatch &batch, glm::mat4 transform) {
  std::uniform_real_distribution<float> rand_color(0, 1);
  std::default_random_engine rand_const(1);
  const auto &core = batch.trees.core;
  const auto &stats = batch.trees.stats;
  for (auto shape_index = 0; shape_index < batch.tree_shapes.count.size(); ++shape_index) {
    glm::vec4 color(rand_color(rand_const), rand_color(rand_const), rand_color(rand_const), 1);
    auto start = batch.tree_shapes.start[shape_index];
    auto count = batch.tree_shapes.count[shape_index];
    for (auto node_id = start; node_id < start + count; ++node_id) {
      auto energy = stats.energy[node_id];
      color.a = std::min(1.0f, std::max(0.0f, energy));
      auto parent_id = batch.trees.core.parent[node_id];
      auto start_pos = core.pos[node_id];
      auto end_pos =
          start_pos + glm::vec2(std::cos(core.abs_rot[node_id]), std::sin(core.abs_rot[node_id])) *
                          core.length[node_id];
      auto parent_thickness = stats.thickness[parent_id];
      auto thickness = stats.thickness[node_id];
      // check if start or end pos is outside of the screen
      glm::vec4 start_pos_vec4 = transform * glm::vec4(start_pos, 0, 1);
      glm::vec4 end_pos_vec4 = transform * glm::vec4(end_pos, 0, 1);
      // if (start_pos_vec4.x < -1 || start_pos_vec4.x > 1 || start_pos_vec4.y < -1 ||
      // start_pos_vec4.y > 1 ||
      //     end_pos_vec4.x < -1 || end_pos_vec4.x > 1 || end_pos_vec4.y < -1 || end_pos_vec4.y > 1)
      //     { continue;
      // }
      line_renderer.add_line(start_pos, end_pos, parent_thickness, thickness, color, color);
      // line_renderer.add_line(start_pos, end_pos, parent_thickness, thickness, start_pos_vec4 *
      // 0.5f + 0.5f, end_pos_vec4 * 0.5f + 0.5f);
    }
  }
}

__device__ void add_vertex(unsigned int *line_vbo, unsigned int line_start_index, float x, float y,
                           float tx, float ty, float length, float radius, unsigned char red,
                           unsigned char green, unsigned char blue, unsigned char alpha) {
  auto s = line_start_index * 7;
  line_vbo[s] = reinterpret_cast<unsigned int &>(x);
  line_vbo[s + 1] = reinterpret_cast<unsigned int &>(y);
  line_vbo[s + 2] = reinterpret_cast<unsigned int &>(tx);
  line_vbo[s + 3] = reinterpret_cast<unsigned int &>(ty);
  line_vbo[s + 4] = reinterpret_cast<unsigned int &>(length);
  line_vbo[s + 5] = reinterpret_cast<unsigned int &>(radius);
  unsigned int color = 0;
  color |= red;
  color |= green << 8;
  color |= blue << 16;
  color |= alpha << 24;
  line_vbo[s + 6] = color;
}

__device__ void add_vertex(unsigned int *line_vbo, size_t line_start_index, float x, float y,
                           float tx, float ty, float length, float radius, const glm::vec4 &color) {
  add_vertex(line_vbo, line_start_index, x, y, tx, ty, length, radius, color.r * 255, color.g * 255,
             color.b * 255, color.a * 255);
}

__global__ void render_tree_kernel(unsigned int *line_vbo, const trees2::TreeBatchPtrs batch,
                                   size_t node_count) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= node_count) {
    return;
  }
  const auto parent_id = batch.trees.core.parent[i];
  const auto energy = batch.trees.stats.energy[i];
  const auto start_pos = batch.trees.core.pos[i];
  const auto abs_rot = batch.trees.core.abs_rot[i];
  const auto length = batch.trees.core.length[i];
  auto line_dir = glm::vec2(std::cos(abs_rot), std::sin(abs_rot));
  const auto end_pos = start_pos + line_dir * length;
  const auto parent_thickness = batch.trees.stats.thickness[parent_id];
  const auto thickness = batch.trees.stats.thickness[i];
  const auto max_thickness = max(parent_thickness, thickness);
  const auto energy_coerced = min(1.0f, max(0.0f, energy));
  const auto color = glm::vec4(1 - energy_coerced, 1, 1 - energy_coerced, 1);

  const auto line_start_index = i * LineRenderer::VERTICES_PER_LINE;

  line_dir *= thickness;
  const auto perp_dir = glm::vec2(-line_dir.y, line_dir.x); // counter-clockwise

  // bottom left
  glm::vec2 bl = start_pos + perp_dir - line_dir;
  // bottom right
  glm::vec2 br = start_pos - perp_dir - line_dir;
  // top left
  glm::vec2 tl = end_pos + perp_dir + line_dir;
  // top right
  glm::vec2 tr = end_pos - perp_dir + line_dir;

  // tri 1
  add_vertex(line_vbo, line_start_index + 0, bl.x, bl.y, -max_thickness, -max_thickness, length,
             parent_thickness, color);
  add_vertex(line_vbo, line_start_index + 1, br.x, br.y, max_thickness, -max_thickness, length,
             parent_thickness, color);
  add_vertex(line_vbo, line_start_index + 2, tr.x, tr.y, max_thickness, max_thickness + length,
             length, thickness, color);
  // tri 2
  add_vertex(line_vbo, line_start_index + 3, tr.x, tr.y, max_thickness, max_thickness + length,
             length, thickness, color);
  add_vertex(line_vbo, line_start_index + 4, tl.x, tl.y, -max_thickness, max_thickness + length,
             length, thickness, color);
  add_vertex(line_vbo, line_start_index + 5, bl.x, bl.y, -max_thickness, -max_thickness, length,
             parent_thickness, color);
}

void render_trees_cuda(const TreesState &state, LineRenderer &renderer,
                       const glm::mat4 &transform) {
  renderer.set_transform(transform);

  const auto node_count = state.read_device.trees.core.abs_rot.size();
  renderer.ensure_vbo_capacity(node_count);
  // get a cuda compatible pointer to the vbo
  renderer.cuda_register_buffer();
  auto vbo_ptr = renderer.cuda_map_buffer();
  trees2::TreeBatchPtrs ptrs;
  ptrs.get_ptrs(const_cast<trees2::TreeBatchDevice &>(state.read_device));

  dim3 block(256);
  dim3 grid((node_count + block.x - 1) / block.x);
  render_tree_kernel<<<grid, block>>>(static_cast<unsigned int *>(vbo_ptr), ptrs, node_count);
  cudaDeviceSynchronize();
  renderer.cuda_unmap_buffer();

  renderer.render(node_count);
  renderer.cuda_unregister_buffer();

  // read_device.copy_to_host(read_host);
  // line_renderer->begin();
  // render_tree(*line_renderer.get(), read_host, transform);
  // line_renderer->end();
  // line_renderer->render();
}

} // namespace trees
