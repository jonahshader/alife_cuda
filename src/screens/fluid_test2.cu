#include "fluid_test2.cuh"

#include <iostream>

#include <imgui.h>
#include <thrust/extrema.h>

#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t error = call;                                                                      \
    if (error != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

FluidTest2::FluidTest2(Game &game)
    : DefaultScreen(game), density_texture_data(tex_size.x * tex_size.y * 4) {
  p2::init_fluid(fluid, bounds.x, bounds.y);
}

bool FluidTest2::handle_input(SDL_Event event) {
  if (DefaultScreen::handle_input(event))
    return true;

  if (event.type == SDL_MOUSEBUTTONDOWN) {
    if (event.button.button == SDL_BUTTON_LEFT) {
      grabbing = true;
      return true;
    }
  } else if (event.type == SDL_MOUSEBUTTONUP) {
    if (event.button.button == SDL_BUTTON_LEFT) {
      grabbing = false;
      return true;
    }
  } else if (event.type == SDL_MOUSEMOTION) {
    mouse_pos = {event.motion.x, event.motion.y};
    return true;
  }

  return false;
}

static void check_cuda(const std::string &msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "FluidTest2: " << msg << ": " << cudaGetErrorString(err) << std::endl;
  }
}

void FluidTest2::render(float _dt) {
  render_start();
  p2::update_fluid(fluid);
  if (grabbing) {
    const auto world_coords = vp.unproject({mouse_pos.x, mouse_pos.y});
    p2::attract_fluid(fluid, {world_coords.x, world_coords.y}, 0.5f, 0.5f);
  }
  p2::calculate_fluid_density_grid(fluid, density_texture_data, tex_size.x, tex_size.y, 300.0f);

  cudaArray *cuda_array = density_renderer.cuda_map_texture();
  if (cuda_array == nullptr) {
    std::cerr << "Failed to map texture to CUDA" << std::endl;
    return;
  }

  cudaDeviceSynchronize();
  density_renderer.update_texture_from_cuda(density_texture_data.data().get());
  check_cuda("update_texture_from_cuda");

  density_renderer.cuda_unmap_texture();

  density_renderer.set_transform(vp.get_transform());
  density_renderer.begin();
  for (int x_offset = -1; x_offset <= 1; ++x_offset) {
    for (int y_offset = -1; y_offset <= 1; ++y_offset) {
      float x_offset_f = x_offset * bounds.x;
      float y_offset_f = y_offset * bounds.y;
      density_renderer.add_rect(x_offset_f, y_offset_f, bounds.x, bounds.y, glm::vec3(1.0f));
    }
  }
  density_renderer.end();
  density_renderer.render();

  auto &circle_renderer = game.get_resources().circle_renderer;
  p2::render_fluid(fluid, circle_renderer, vp.get_transform());
  check_cuda("fluid.render");

  render_end();
  check_cuda("render_end");
}
