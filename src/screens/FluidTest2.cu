#include "FluidTest2.cuh"

#include "systems/ParticleFluid2.cuh"

#include <imgui.h>
#include <thrust/extrema.h>

#include <iostream>

FluidTest2::FluidTest2(Game &game) : DefaultScreen(game),
                                     density_data(tex_size.x * tex_size.y),
                                     density_texture_data(tex_size.x * tex_size.y * 4)
{
}

__global__ void density_to_texture(float *density_data, unsigned char *density_texture_data, int size, float max_density)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
  {
    float d = density_data[i] / max_density;
    // if (i % 3 == 0)
    //   d = 1.0f;
    density_texture_data[i * 4] = 255 * d;
    density_texture_data[i * 4 + 1] = 255 * d;
    density_texture_data[i * 4 + 2] = 255 * d;
    density_texture_data[i * 4 + 3] = 255;
  }
}

void FluidTest2::render(float _dt)
{
  render_start();

  // ImGui::Begin("Fluid Test 2");
  // if (ImGui::SliderFloat("dt", &dt, 0.0001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat))
  //   pm.set_param("dt", dt);
  // if (ImGui::SliderFloat("particle_mass", &particle_mass, 0.1f, 10.0f, "%.1f"))
  //   pm.set_param("particle_mass", particle_mass);
  // if (ImGui::SliderFloat("kernel_radius", &kernel_radius, 0.01f, 0.5f, "%.2f"))
  //   pm.set_param("kernel_radius", kernel_radius);
  // if (ImGui::Button("Save Parameters"))
  //   pm.save_params();
  // ImGui::End();

  fluid.calculate_density_grid(density_data, tex_size.x, tex_size.y);
  for (int i = 0; i < 10; ++i) {
    std::cout << density_data[i] << ", ";
  }
  std::cout << std::endl;
  auto max_density_addr = thrust::max_element(density_data.begin(), density_data.end());
  if (max_density_addr == density_data.end())
  {
    std::cerr << "Failed to find max density" << std::endl;
    // print a few values
    for (int i = 0; i < 10; ++i)
    {
      std::cout << density_data[i] << std::endl;
    }
    return;
  }
  float max_density = *max_density_addr;
  std::cout << "max_density: " << max_density << std::endl;
  density_to_texture<<<(density_data.size() + 255) / 256, 256>>>(density_data.data().get(), density_texture_data.data().get(), density_data.size(), max_density);

  cudaArray *cuda_array = density_renderer.cuda_map_texture();
  if (cuda_array == nullptr)
  {
    std::cerr << "Failed to map texture to CUDA" << std::endl;
    return;
  }
  cudaTextureObject_t texObj = density_renderer.create_texture_object();
  if (texObj == 0)
  {
    std::cerr << "Failed to create CUDA texture object" << std::endl;
    density_renderer.cuda_unmap_texture();
    return;
  }
  // Wait for GPU to finish before accessing on host
  fluid.update();
  cudaDeviceSynchronize();
  density_renderer.update_texture_from_cuda(density_texture_data.data().get());
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
  {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    density_renderer.destroy_texture_object(texObj);
    density_renderer.cuda_unmap_texture();
    return;
  }

  density_renderer.destroy_texture_object(texObj);
  density_renderer.cuda_unmap_texture();

  density_renderer.set_transform(vp.get_transform());
  density_renderer.begin();
  for (int x_offset = -1; x_offset <= 1; ++x_offset)
  {
    for (int y_offset = -1; y_offset <= 1; ++y_offset)
    {
      float x_offset_f = x_offset * bounds.x;
      float y_offset_f = y_offset * bounds.y;
      density_renderer.add_rect(x_offset_f + -bounds.x / 2, y_offset_f + -bounds.y / 2, bounds.x, bounds.y, glm::vec3(1.0f));
    }
  }
  density_renderer.end();
  density_renderer.render();

  render_end();
}