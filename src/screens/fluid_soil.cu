#include "fluid_soil.cuh"
#include "systems/cuda_utils.cuh"
#include "systems/timing_profiler.cuh"

#include <iostream>

#include <imgui.h>
#include <thrust/extrema.h>

FluidSoil::FluidSoil(Game &game, const SimParams &params)
    : DefaultScreen(game), sim_params(params), bounds{params.world_width, params.world_height},
      tex_size{(int)std::round(params.world_width * pixels_per_meter),
               (int)std::round(params.world_height * pixels_per_meter)},
      density_renderer(tex_size.x, tex_size.y, 4),
      density_texture_data(tex_size.x * tex_size.y * 4) {
  auto resolved_seed = sim_params.resolve_seed();
  init_soil(soil, (unsigned int)std::round(bounds.x / sim_params.soil_cell_size),
            (unsigned int)std::round(bounds.y / sim_params.soil_cell_size),
            sim_params.soil_cell_size, resolved_seed, sim_params.terrain_mode);
  p2::init_fluid(fluid, bounds.x, bounds.y, sim_params);
}

bool FluidSoil::handle_input(SDL_Event event) {
  if (DefaultScreen::handle_input(event))
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

void FluidSoil::update(float dt) {
  auto &profiler = TimingProfiler::get_instance();

  update_soil_cuda(soil, dt);
  p2::update_fluid(fluid, soil);

  if (grabbing || repelling) {
    const auto world_coords = vp.unproject({mouse_pos.x, mouse_pos.y});
    p2::attract_fluid(fluid, {world_coords.x, world_coords.y},
                      repelling ? -grab_strength : grab_strength, grab_radius);
  }

  {
    auto scope = profiler.scoped_measure("calculate_density_grid");
    if (show_density_grid)
      p2::calculate_fluid_density_grid(fluid, density_texture_data, tex_size.x, tex_size.y, 300.0f);
  }
}

void FluidSoil::render() {
  auto &profiler = TimingProfiler::get_instance();
  render_start();
  auto &circle_renderer = game.get_resources().circle_renderer;
  auto &main_font_world = game.get_resources().main_font_world;
  circle_renderer.set_transform(vp.get_transform());
  main_font_world.set_transform(vp.get_transform());
  circle_renderer.begin();
  main_font_world.begin();

  ImGui::Begin("Particle Fluid");
  ImGui::Checkbox("Show Density Grid", &show_density_grid);
  ImGui::End();
  p2::render_fluid_imgui(fluid);

  {
    auto scope = profiler.scoped_measure("density_renderer.cuda_map_texture()");
    if (show_density_grid) {
      cudaArray *cuda_array = density_renderer.cuda_map_texture();
      if (cuda_array == nullptr) {
        std::cerr << "Failed to map texture to CUDA" << std::endl;
        return;
      }
    }
  }

  {
    auto scope = profiler.scoped_measure("update_texture_from_cuda");
    if (show_density_grid) {
      density_renderer.update_texture_from_cuda(density_texture_data.data().get());
      check_cuda("update_texture_from_cuda");
    }
  }

  {
    auto scope = profiler.scoped_measure("density_renderer.render...");
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
    auto scope = profiler.scoped_measure("soil.render");
    render_soil(soil, soil_renderer, vp.get_transform());
    check_cuda("soil.render");
  }

  {
    auto scope = profiler.scoped_measure("fluid.render");
    p2::render_fluid(fluid, circle_renderer, vp.get_transform());
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
    auto scope = profiler.scoped_measure("render_end...");

    circle_renderer.end();
    main_font_world.end();
    circle_renderer.render();
    main_font_world.render();

    render_end();
    check_cuda("render_end");
  }
  profiler_gui.render();
}
