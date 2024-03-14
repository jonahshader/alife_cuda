// https://bcmpinc.wordpress.com/2015/08/18/creating-an-opengl-4-5-context-using-sdl2-and-glad/

#define SDL_MAIN_HANDLED
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include "systems/Game.cuh"
#include "screens/TreeTest.cuh"
#include "screens/FluidTest.cuh"
#include "spatial_sort.cuh"

#include <SDL.h>
#include "glad/glad.h"

static int viewport_width = 1280;
static int viewport_height = 720;
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

int main() {
    init_screen("OpenGL 4.3");

    //    jl_init();
    //
    //    jl_eval_string("print(sqrt(2.0))");

    Game game;
    game.getResources().window = window;
    game.resize(viewport_width, viewport_height);
    //    game.pushScreen(std::make_shared<MainMenu>(game));
    game.pushScreen(std::make_shared<FluidTest>(game));

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
