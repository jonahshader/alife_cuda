#include "FluidSoil.cuh"

#include "systems/ParticleFluid2.cuh"
#include "systems/TimingProfiler.cuh"

#include <imgui.h>
#include <thrust/extrema.h>

#include <iostream>


#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t error = call;                                                                      \
    if (error != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

FluidSoil::FluidSoil(Game &game)
    : DefaultScreen(game),
      //  density_data(tex_size.x * tex_size.y),
      density_texture_data(tex_size.x * tex_size.y * 4) {}

bool FluidSoil::handleInput(SDL_Event event) {
  if (DefaultScreen::handleInput(event))
    return true;

  if (event.type == SDL_MOUSEBUTTONDOWN) {
    if (event.button.button == SDL_BUTTON_LEFT) {
      grabbing = true;
      return true;
    } else if (event.button.button == SDL_BUTTON_RIGHT) {
      repelling = true;
      return true;
    }
  } else if (event.type == SDL_MOUSEBUTTONUP) {
    if (event.button.button == SDL_BUTTON_LEFT) {
      grabbing = false;
      return true;
    } else if (event.button.button == SDL_BUTTON_RIGHT) {
      repelling = false;
      return true;
    }
  } else if (event.type == SDL_MOUSEMOTION) {
    mouse_pos = {event.motion.x, event.motion.y};
    return true;
  } else if (event.type == SDL_KEYDOWN) {
    switch (event.key.keysym.sym) {
      case SDLK_PLUS:
      case SDLK_EQUALS:
      case SDLK_KP_PLUS:
      case SDLK_KP_EQUALS:
        grab_radius *= 1.25f;
        return true;
      case SDLK_MINUS:
      case SDLK_KP_MINUS:
        grab_radius *= 0.8f;
        return true;
      case SDLK_LEFTBRACKET:
        grab_strength *= 0.8f;
        return true;
      case SDLK_RIGHTBRACKET:
        grab_strength *= 1.25f;
        return true;
      default:
        break;
    }
  }

  return false;
}

// __global__ void density_to_texture(float *density_data, unsigned char *density_texture_data, int
// size, float max_density)

void FluidSoil::check_cuda(const std::string &msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "FluidSoil: " << msg << ": " << cudaGetErrorString(err) << std::endl;
  }
}

void FluidSoil::render(float _dt) {
  auto &profiler = TimingProfiler::getInstance();
  render_start();
  auto &circle_renderer = game.getResources().circle_renderer;
  auto &main_font_world = game.getResources().main_font_world;
  circle_renderer.set_transform(vp.get_transform());
  main_font_world.set_transform(vp.get_transform());
  circle_renderer.begin();
  main_font_world.begin();
  soil.update_cuda(_dt); // TODO: need proper dt parameter
  fluid.update(soil);

  ImGui::Begin("Particle Fluid");
  ImGui::Checkbox("Show Density Grid", &show_density_grid);
  ImGui::End();
  if (grabbing || repelling) {
    const auto world_coords = vp.unproject({mouse_pos.x, mouse_pos.y});

    fluid.attract({world_coords.x, world_coords.y}, repelling ? -grab_strength : grab_strength,
                  grab_radius);
  }
  // TODO: calculate expected max_density from particles per cell
  {
    auto scope = profiler.scopedMeasure("calculate_density_grid");
    if (show_density_grid)
      fluid.calculate_density_grid(density_texture_data, tex_size.x, tex_size.y, 300.0f);
  }

  {
    auto scope = profiler.scopedMeasure("density_renderer.cuda_map_texture()");
    if (show_density_grid) {
      cudaArray *cuda_array = density_renderer.cuda_map_texture();
      if (cuda_array == nullptr) {
        std::cerr << "Failed to map texture to CUDA" << std::endl;
        return;
      }
    }
  }


  {
    auto scope = profiler.scopedMeasure("update_texture_from_cuda");
    if (show_density_grid) {
      density_renderer.update_texture_from_cuda(density_texture_data.data().get());
      check_cuda("update_texture_from_cuda");
    }
  }

  {
    auto scope = profiler.scopedMeasure("density_renderer.render...");
    if (show_density_grid) {
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
    }

  }

  {
    auto scope = profiler.scopedMeasure("soil.render");
    soil.render(vp.get_transform());
    check_cuda("soil.render");
  }

  {
    auto scope = profiler.scopedMeasure("fluid.render");
    fluid.render(vp.get_transform());
    check_cuda("fluid.render");
  }

  // add a circle for mouse grab tool
  float grab_color_opacity = 0.1f;
  if (grabbing || repelling) {
    grab_color_opacity = 0.25f;
  }
  const auto world_coords = vp.unproject({mouse_pos.x, mouse_pos.y});
  circle_renderer.add_circle(world_coords.x, world_coords.y, grab_radius,
                             glm::vec4(0.5f, 0.5f, 0.8f, grab_color_opacity));
  // display strength
  main_font_world.add_text(world_coords.x, world_coords.y, grab_radius,
                           "Strength: " + std::to_string(grab_strength),
                           glm::vec4(0.0f, 0.0f, 0.0f, 0.5f), FontRenderer::HAlign::CENTER);

  {
    auto scope = profiler.scopedMeasure("render_end...");

    circle_renderer.end();
    main_font_world.end();
    circle_renderer.render();
    main_font_world.render();

    render_end();
    check_cuda("render_end");
  }
  profiler_gui.render();
}
