// https://bcmpinc.wordpress.com/2015/08/18/creating-an-opengl-4-5-context-using-sdl2-and-glad/

#define SDL_MAIN_HANDLED
// #define GLM_FORCE_CUDA
// #define GLM_COMPILER_CUDA
#include "config/config.h"
#include "glad/glad.h"
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl2.h"
#include "screens/fluid_soil.cuh"
#include "screens/fluid_test2.cuh"
#include "screens/tree_test.cuh"
#include "systems/game.cuh"
#include "systems/timing_profiler.cuh"

#include <SDL.h>

#include <glm/glm.hpp>

#include <atomic>
#include <csignal>
#include <iostream>

#include <CLI/CLI.hpp>

static int viewport_width = 1920;
static int viewport_height = 1080;
static SDL_Window *window = nullptr;
static SDL_GLContext main_context;

static std::atomic<bool> g_stop_requested{false};

static void signal_handler(int signum) {
  g_stop_requested.store(true, std::memory_order_relaxed);
}

static void sdl_die(const char *message) {
  fprintf(stderr, "%s: %s\n", message, SDL_GetError());
  exit(2);
}

void init_screen(const char *caption) {
  // Initialize SDL
  if (SDL_Init(SDL_INIT_VIDEO) < 0)
    sdl_die("Couldn't initialize SDL");
  atexit(SDL_Quit);
  SDL_GL_LoadLibrary(nullptr); // Default OpenGL is fine.

  // Request an OpenGL 4.3 context (should be core)
  SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

  // Also request a depth buffer
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

#ifdef FULLSCREEN
  window = SDL_CreateWindow(caption, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 0, 0,
                            SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_OPENGL);
#else
  window =
      SDL_CreateWindow(caption, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, viewport_width,
                       viewport_height, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

#endif

  if (window == nullptr)
    sdl_die("Couldn't set video mode");

  main_context = SDL_GL_CreateContext(window);
  if (main_context == nullptr)
    sdl_die("Failed to create OpenGL context");

  // Check OpenGL properties
  printf("OpenGL loaded\n");
  gladLoadGLLoader(SDL_GL_GetProcAddress);
  printf("Vendor:   %s\n", glGetString(GL_VENDOR));
  printf("Renderer: %s\n", glGetString(GL_RENDERER));
  printf("Version:  %s\n", glGetString(GL_VERSION));

  // Use v-sync
  SDL_GL_SetSwapInterval(0);

  // Disable depth test and face culling.
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  int w, h;
  SDL_GetWindowSize(window, &w, &h);
  glViewport(0, 0, w, h);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void init_imgui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  // setup platform/renderer backends
  ImGui_ImplSDL2_InitForOpenGL(window, main_context);
  ImGui_ImplOpenGL3_Init("#version 430");
}

static void print_cuda_info() {
  cudaDeviceProp cuda_prop;
  cudaGetDeviceProperties(&cuda_prop, 0);
  // print the compute capability, max number of threads per block, max number of blocks, number of
  // SMs, max number of threads per SM, number of registers per block, number of registers per SM,
  // shared memory per block, shared memory per SM, warp size, number of CUDA cores
  std::cout << "Compute capability: " << cuda_prop.major << "." << cuda_prop.minor << std::endl;
  std::cout << "Max threads per block: " << cuda_prop.maxThreadsPerBlock << std::endl;
  std::cout << "Max blocks: " << cuda_prop.maxGridSize[0] << std::endl;
  std::cout << "Number of SMs: " << cuda_prop.multiProcessorCount << std::endl;
  std::cout << "Max threads per SM: " << cuda_prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Number of registers per block: " << cuda_prop.regsPerBlock << std::endl;
  std::cout << "Number of registers per SM: " << cuda_prop.regsPerMultiprocessor << std::endl;
  std::cout << "Shared memory per block: " << cuda_prop.sharedMemPerBlock << " bytes" << std::endl;
  std::cout << "Shared memory per SM: " << cuda_prop.sharedMemPerMultiprocessor << " bytes"
            << std::endl;
  std::cout << "Warp size: " << cuda_prop.warpSize << std::endl;
  std::cout << "Number of CUDA cores: " << cuda_prop.multiProcessorCount * cuda_prop.warpSize
            << std::endl;
}

static void print_profiler_stats() {
  auto &profiler = TimingProfiler::get_instance();
  auto names = profiler.get_profile_point_names();
  if (names.empty())
    return;

  std::cout << "\n=== Profiler Stats ===" << std::endl;
  for (const auto &name : names) {
    auto point = profiler.get_point(name);
    if (point && point->get_sample_count() > 0) {
      std::cout << "  " << name << ": avg=" << point->get_average_duration() * 1000.0 << "ms"
                << " min=" << point->get_min_duration() * 1000.0 << "ms"
                << " max=" << point->get_max_duration() * 1000.0 << "ms"
                << " samples=" << point->get_sample_count() << std::endl;
    }
  }
}

int main(int argc, char *argv[]) {
  // --- Pass 1: pre-parse for config file and write-config ---
  std::string config_file = "config.toml";
  bool write_config = false;
  bool headless = false;
  int iterations = 0; // 0 = run until Ctrl+C (headless) or window close (GUI)
  SimParams sim_params;

  {
    CLI::App pre_app{"ALife CUDA"};
    pre_app.allow_extras();
    pre_app.set_help_flag(); // disable --help in pre-parse; let pass 2 handle it
    pre_app.add_option("--config", config_file, "Path to TOML config file");
    pre_app.add_flag("--write-config", write_config, "Write default config file and exit");
    pre_app.parse(argc, argv);
  }

  if (write_config) {
    write_default_config(config_file);
    return 0;
  }

  // Load config from TOML if it exists
  if (auto loaded = load_sim_params(config_file)) {
    sim_params = *loaded;
    std::cout << "Loaded config from: " << config_file << std::endl;
  }

  // --- Pass 2: main parse (CLI overrides TOML overrides defaults) ---
  {
    CLI::App app{"ALife CUDA"};
    app.add_flag("--headless", headless, "Run without graphics");
    app.add_option("--iterations", iterations, "Number of simulation steps to run (0 = unlimited)");
    register_sim_params_cli(app, sim_params);
    CLI11_PARSE(app, argc, argv);
  }

  // Set up signal handler for clean shutdown
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  print_cuda_info();

  if (headless) {
    // --- Headless mode: simulation only, no graphics ---
    std::cout << "Running headless";
    if (iterations > 0)
      std::cout << " for " << iterations << " iterations";
    std::cout << " (Ctrl+C to stop)" << std::endl;

    // Initialize fluid state directly without Game/Screen
    auto resolved_seed = sim_params.resolve_seed();
    std::cout << "Using seed: " << resolved_seed << std::endl;

    p2::ParticleFluidState fluid{};
    p2::init_fluid(fluid, sim_params.world_width, sim_params.world_height, sim_params);

    SoilState soil{};
    init_soil(soil, (unsigned int)std::round(sim_params.world_width / sim_params.soil_cell_size),
              (unsigned int)std::round(sim_params.world_height / sim_params.soil_cell_size),
              sim_params.soil_cell_size, resolved_seed);

    int step = 0;
    while (!g_stop_requested.load(std::memory_order_relaxed)) {
      update_soil_cuda(soil, sim_params.dt);
      p2::update_fluid(fluid, soil);
      step++;
      if (iterations > 0 && step >= iterations)
        break;
    }

    std::cout << "Completed " << step << " steps" << std::endl;
    print_profiler_stats();
    return 0;
  }

  // --- GUI mode ---
  init_screen("ALife CUDA");
  init_imgui();

  {
    Game game;
    game.get_resources().window = window;
    game.resize(viewport_width, viewport_height);
    // game.push_screen(std::make_shared<TreeTest>(game));
    // game.push_screen(std::make_shared<FluidTest2>(game));
    game.push_screen(std::make_shared<FluidSoil>(game, sim_params));

    float time = 0;
    SDL_Event event;
    while (game.is_running() && !g_stop_requested.load(std::memory_order_relaxed)) {
      while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event);
        // skip game input handling if ImGui wants to capture the event
        bool handled = false;
        switch (event.type) {
          case SDL_KEYDOWN:
          case SDL_KEYUP:
            handled = ImGui::GetIO().WantCaptureKeyboard;
            break;
          case SDL_MOUSEMOTION:
          case SDL_MOUSEBUTTONDOWN:
          case SDL_MOUSEBUTTONUP:
          case SDL_MOUSEWHEEL:
            handled = ImGui::GetIO().WantCaptureMouse;
            break;
          default:
            break;
        }

        if (!handled) {
          game.handle_input(event);
        }

        if (event.type == SDL_QUIT) {
          game.stop_game();
        } else if (event.type == SDL_WINDOWEVENT) {
          if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
            viewport_width = event.window.data1;
            viewport_height = event.window.data2;
            game.resize(viewport_width, viewport_height);
            glViewport(0, 0, viewport_width, viewport_height);
          }
        }
      }

      // Start the Dear ImGui frame
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplSDL2_NewFrame();
      ImGui::NewFrame();

      game.update(1 / 165.0f);
      game.render();

      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      SDL_GL_SwapWindow(game.get_resources().window);
      time += 1 / 165.0f;
    }
  } // game destroyed here, while GL context is still valid

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  SDL_GL_DeleteContext(main_context);
  SDL_DestroyWindow(window);
  return 0;
}
