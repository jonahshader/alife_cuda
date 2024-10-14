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

__global__ void write_tex_test(cudaTextureObject_t texObj, int width, int height, int channels, unsigned char *output)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height)
  {
    float4 pixel = tex2D<float4>(texObj, x, y);
    int index = (y * width + x) * channels;
    // output[index] = static_cast<unsigned char>(255 * pixel.x);
    // output[index + 1] = static_cast<unsigned char>(255 * pixel.y);
    // output[index + 2] = static_cast<unsigned char>(255 * pixel.z);
    // if (channels == 4)
    // {
    //   output[index + 3] = static_cast<unsigned char>(255 * pixel.w);
    // }

    output[index] = x % 256;
    output[index + 1] = y % 256;
    output[index + 2] = (x + y) % 256;
    if (channels == 4)
    {
      output[index + 3] = 255;
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
  cudaTextureObject_t texObj = rect.create_texture_object();
  if (texObj == 0)
  {
    std::cerr << "Failed to create CUDA texture object" << std::endl;
    rect.cuda_unmap_texture();
    return;
  }

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
  write_tex_test<<<grid, block>>>(texObj, width, height, channels, d_output);

  // Check for kernel launch errors
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
  {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    rect.destroy_texture_object(texObj);
    rect.cuda_unmap_texture();
    cudaFree(d_output);
    return;
  }

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Update the texture with the processed data
  rect.update_texture_from_cuda(d_output);

  // Clean up
  rect.destroy_texture_object(texObj);
  rect.cuda_unmap_texture();
  cudaFree(d_output);

  // Render the texture
  rect.set_transform(vp.get_transform());
  rect.begin();
  rect.add_rect(0.0f, 0.0f, 32.0f, 16.0f, glm::vec3(1.0f, 1.0f, 1.0f));
  rect.end();
  rect.render();
}

void TexCUDATest::resize(int width, int height)
{
  vp.update(width, height);
  hud_vp.update(width, height);
}

void TexCUDATest::handleInput(SDL_Event event)
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
}