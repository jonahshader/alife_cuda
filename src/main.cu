// https://bcmpinc.wordpress.com/2015/08/18/creating-an-opengl-4-5-context-using-sdl2-and-glad/

#define SDL_MAIN_HANDLED
//#define GLM_FORCE_CUDA
//#define GLM_COMPILER_CUDA
#include <glm/glm.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <iostream>
#include "systems/Game.cuh"
#include "screens/TreeTest.cuh"
#include "screens/FluidTest.cuh"
#include "screens/SoilTest.cuh"
#include "spatial_sort.cuh"

#include <SDL.h>
#include "glad/glad.h"

#ifdef _WIN32
#include <Windows.h>
#endif

static int viewport_width = 1920;
static int viewport_height = 1080;
static SDL_Window* window = nullptr;
static SDL_GLContext main_context;

static void sdl_die(const char * message) {
    fprintf(stderr, "%s: %s\n", message, SDL_GetError());
    exit(2);
}

void init_screen(const char * caption) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
        sdl_die("Couldn't initialize SDL");
    atexit (SDL_Quit);
    SDL_GL_LoadLibrary(nullptr); // Default OpenGL is fine.

    // Request an OpenGL 4.3 context (should be core)
    SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

    // Also request a depth buffer
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

#ifdef FULLSCREEN
    window = SDL_CreateWindow(
            caption,
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            0, 0, SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_OPENGL
    );
#else
    window = SDL_CreateWindow(
                caption,
                SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                viewport_width, viewport_height, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
        );

#endif

    if (window == nullptr) sdl_die("Couldn't set video mode");

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

    int w,h;
    SDL_GetWindowSize(window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

int main(int argc, char* argv[]) {
    cudaDeviceProp cuda_prop;
    cudaGetDeviceProperties(&cuda_prop, 0);
    // print the compute capability, max number of threads per block, max number of blocks, number of SMs, max number of threads per SM,
    // number of registers per block, number of registers per SM, shared memory per block, shared memory per SM, warp size, number of floating point units
    std::cout << "Compute capability: " << cuda_prop.major << "." << cuda_prop.minor << std::endl;
    std::cout << "Max threads per block: " << cuda_prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max blocks: " << cuda_prop.maxGridSize[0] << std::endl;
    std::cout << "Number of SMs: " << cuda_prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per SM: " << cuda_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Number of registers per block: " << cuda_prop.regsPerBlock << std::endl;
    std::cout << "Number of registers per SM: " << cuda_prop.regsPerMultiprocessor << std::endl;
    std::cout << "Shared memory per block: " << cuda_prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Shared memory per SM: " << cuda_prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "Warp size: " << cuda_prop.warpSize << std::endl;
    std::cout << "Number of floating point units: " << cuda_prop.multiProcessorCount * cuda_prop.maxThreadsPerMultiProcessor << std::endl;
    init_screen("OpenGL 4.3");

    //    jl_init();
    //
    //    jl_eval_string("print(sqrt(2.0))");

    Game game;
    game.getResources().window = window;
    game.resize(viewport_width, viewport_height);
    //    game.pushScreen(std::make_shared<MainMenu>(game));
    // game.pushScreen(std::make_shared<FluidTest>(game));
    // game.pushScreen(std::make_shared<TreeTest>(game));
    game.pushScreen(std::make_shared<SoilTest>(game));

    float time = 0;
    SDL_Event event;
    while (game.isRunning()) {
        while (SDL_PollEvent(&event)) {
            game.handleInput(event);
            if (event.type == SDL_QUIT) {
                game.stopGame();
            } else if (event.type == SDL_WINDOWEVENT) {
                if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
                    viewport_width = event.window.data1;
                    viewport_height = event.window.data2;
                    game.resize(viewport_width, viewport_height);
                    glViewport(0, 0, viewport_width, viewport_height);
                }
            }
        }

        game.render(1/165.0f);

        //        SDL_GL_SwapWindow(window);
        time += 1/165.0f;
    }


    SDL_GL_DeleteContext(main_context);
    SDL_DestroyWindow(window);
    //    jl_atexit_hook(0);
    return 0;
}

#ifdef _WIN32
// Windows entry point
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    return main(__argc, __argv);
}
#endif