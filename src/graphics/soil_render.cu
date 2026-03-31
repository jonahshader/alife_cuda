#include "soil_render.cuh"

__host__ __device__ inline void add_rect(float x, float y, float width, float height,
                                         glm::vec4 color, float *vbo, size_t i) {
  const auto s = i * SimpleRectRenderer::FLOATS_PER_RECT;
  vbo[s + 0] = x;
  vbo[s + 1] = y;
  vbo[s + 2] = width;
  vbo[s + 3] = height;
  vbo[s + 4] = color.r;
  vbo[s + 5] = color.g;
  vbo[s + 6] = color.b;
  vbo[s + 7] = color.a;
}

__global__ void render_soil_kernel(float *rect_vbo, SoilPtrs read, uint width, float cell_size,
                                   size_t rect_count) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rect_count) {
    return;
  }
  auto sand = read.sand_density[i];
  auto silt = read.silt_density[i];
  auto clay = read.clay_density[i];
  auto density = sand + silt + clay;
  glm::vec3 color = glm::vec3(0.0f);
  if (density > 0.001f) {
    auto inv_density = 1.0f / density;
    sand *= inv_density;
    silt *= inv_density;
    clay *= inv_density;

    const glm::vec3 sand_color(219 / 255.0f, 193 / 255.0f, 44 / 255.0f);
    const glm::vec3 silt_color(119 / 255.0f, 143 / 255.0f, 40 / 255.0f);
    const glm::vec3 clay_color(219 / 255.0f, 41 / 255.0f, 23 / 255.0f);
    color = sand_color * sand + silt_color * silt + clay_color * clay;
  }

  const auto x = i % width;
  const auto y = i / width;
  add_rect(x * cell_size, y * cell_size, cell_size, cell_size, glm::vec4(color, 1.0f), rect_vbo, i);
}

void render_soil(const SoilState &state, SimpleRectRenderer &renderer, const glm::mat4 &transform) {
  renderer.set_transform(transform);

  const auto rect_count = state.read.sand_density.size();
  renderer.ensure_vbo_capacity(rect_count);
  // get a cuda compatible pointer to the vbo
  renderer.cuda_register_buffer();
  auto vbo_ptr = renderer.cuda_map_buffer();
  SoilPtrs ptrs;
  ptrs.get_ptrs(const_cast<SoilSoADevice &>(state.read));

  dim3 block(256);
  dim3 grid((rect_count + block.x - 1) / block.x);
  render_soil_kernel<<<grid, block>>>(static_cast<float *>(vbo_ptr), ptrs, state.width,
                                      state.cell_size, rect_count);
  renderer.cuda_unmap_buffer();

  renderer.render(rect_count);
  renderer.cuda_unregister_buffer();
}
