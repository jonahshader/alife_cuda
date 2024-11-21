#include "TexCUDATest.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

TexCUDATest::TexCUDATest(Game &game) : game(game)
{
}

void TexCUDATest::show()
{
}

void TexCUDATest::hide()
{
}

__global__ void write_tex_test(int width, int height, int channels, unsigned char *output)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x_center = width / 2;
  int y_center = height / 2;
  if (x < width && y < height)
  {
    int index = (y * width + x) * channels;
    long distance = (x - (long)x_center) * (x - (long)x_center) + (y - (long)y_center) * (y - (long)y_center);
    // unsigned char brightness = x % 2 == 0 ? (y % 256) : 0;
    // unsigned char brightness = distance < height * height / 4 ? 255 : 0;
    unsigned char brightness = x % 2 != y % 2 ? 255 : 0;

    output[index] = brightness;
    output[index + 1] = brightness;
    output[index + 2] = brightness;

    if (channels == 4)
    {
      output[index + 3] = 255; // Full alpha for RGBA
    }
  }
}

void TexCUDATest::render(float dt)
{
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Map the texture to CUDA
  cudaArray *cuda_array = rect.cuda_map_texture();
  if (cuda_array == nullptr)
  {
    std::cerr << "Failed to map texture to CUDA" << std::endl;
    return;
  }

  // Create CUDA texture object
  // cudaTextureObject_t texObj = rect.create_texture_object();
  // if (texObj == 0)
  // {
  //   std::cerr << "Failed to create CUDA texture object" << std::endl;
  //   rect.cuda_unmap_texture();
  //   return;
  // }

  // Allocate device memory for output
  int width = rect.get_width();
  int height = rect.get_height();
  int channels = rect.get_channels();
  size_t size = width * height * channels * sizeof(unsigned char);
  unsigned char *d_output;
  cudaMalloc(&d_output, size);

  // Launch CUDA kernel
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  write_tex_test<<<grid, block>>>(width, height, channels, d_output);

  // Check for kernel launch errors
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
  {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    // rect.destroy_texture_object(texObj);
    rect.cuda_unmap_texture();
    cudaFree(d_output);
    return;
  }

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Update the texture with the processed data
  rect.update_texture_from_cuda(d_output);

  // Clean up
  // rect.destroy_texture_object(texObj);
  rect.cuda_unmap_texture();
  cudaFree(d_output);

  // Render the texture
  rect.set_transform(vp.get_transform());
  rect.begin();
  rect.add_rect(0.0f, 0.0f, 64.0f, 48.0f, glm::vec3(1.0f, 1.0f, 1.0f));
  rect.end();
  rect.render();
}

void TexCUDATest::resize(int width, int height)
{
  vp.update(width, height);
  hud_vp.update(width, height);
}

bool TexCUDATest::handleInput(SDL_Event event)
{
  if (event.type == SDL_KEYDOWN)
  {
    switch (event.key.keysym.sym)
    {
    case SDLK_ESCAPE:
      game.stopGame();
      break;
    default:
      break;
    }
  }
  else if (event.type == SDL_MOUSEWHEEL)
  {
    vp.handle_scroll(event.wheel.y);
  }
  else if (event.type == SDL_MOUSEMOTION)
  {
    if (event.motion.state & SDL_BUTTON_LMASK)
    {
      vp.handle_pan(event.motion.xrel, event.motion.yrel);
    }
  }

  return true;
}