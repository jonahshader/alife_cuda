#include "FluidTest2.cuh"

#include "systems/ParticleFluid2.cuh"

#include <imgui.h>
#include <thrust/extrema.h>

#include <iostream>

#define CUDA_CHECK(call)                                               \
  do                                                                   \
  {                                                                    \
    cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                          \
    {                                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(error));                              \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

FluidTest2::FluidTest2(Game &game) : DefaultScreen(game),
                                    //  density_data(tex_size.x * tex_size.y),
                                     density_texture_data(tex_size.x * tex_size.y * 4)
{
}

// __global__ void density_to_texture(float *density_data, unsigned char *density_texture_data, int size, float max_density)

void check_cuda(const std::string &msg)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "FluidTest2: " << msg << ": " << cudaGetErrorString(err) << std::endl;
  }
}

void FluidTest2::render(float _dt)
{
  render_start();
  fluid.update();
  // TODO: calculate expected max_density from particles per cell
  fluid.calculate_density_grid(density_texture_data, tex_size.x, tex_size.y, 300.0f);

  cudaArray *cuda_array = density_renderer.cuda_map_texture();
  if (cuda_array == nullptr)
  {
    std::cerr << "Failed to map texture to CUDA" << std::endl;
    return;
  }

  cudaDeviceSynchronize();
  density_renderer.update_texture_from_cuda(density_texture_data.data().get());
  check_cuda("update_texture_from_cuda");

  density_renderer.cuda_unmap_texture();

  density_renderer.set_transform(vp.get_transform());
  density_renderer.begin();
  for (int x_offset = -1; x_offset <= 1; ++x_offset)
  {
    for (int y_offset = -1; y_offset <= 1; ++y_offset)
    {
      float x_offset_f = x_offset * bounds.x;
      float y_offset_f = y_offset * bounds.y;
      density_renderer.add_rect(x_offset_f, y_offset_f, bounds.x, bounds.y, glm::vec3(1.0f));
    }
  }
  density_renderer.end();
  density_renderer.render();

  fluid.render(vp.get_transform());
  check_cuda("fluid.render");

  render_end();
  check_cuda("render_end");
}