cmake_minimum_required(VERSION 3.28)
project(alife_cuda CUDA CXX) # removed C

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Disable CUDA response files so clangd can parse include paths
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)

# Copy compile_commands.json to project root for LSP/clangd
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  add_custom_target(
    copy-compile-commands ALL
    ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_BINARY_DIR}/compile_commands.json
      ${CMAKE_SOURCE_DIR}/compile_commands.json
    DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
  )
endif()

# Generate .clangd configuration for CUDA support
find_package(CUDAToolkit REQUIRED)
get_filename_component(CUDA_ROOT_DIR "${CUDAToolkit_BIN_DIR}" DIRECTORY)
file(TO_CMAKE_PATH "${CUDA_ROOT_DIR}" CUDA_ROOT_DIR)
if(CMAKE_CUDA_ARCHITECTURES)
  list(GET CMAKE_CUDA_ARCHITECTURES 0 CUDA_ARCH)
  set(CLANGD_CUDA_ARCH "sm_${CUDA_ARCH}")
else()
  set(CLANGD_CUDA_ARCH "sm_75")
endif()
file(WRITE "${CMAKE_SOURCE_DIR}/.clangd"
"CompileFlags:
  Remove:
    # Remove CUDA-specific flags that clangd doesn't understand
    - --extended-lambda
    - --expt-relaxed-constexpr
    - --generate-code=*
    - -forward-unknown-to-host-compiler
    - -rdc=true
    - -x
    - cu
  Add:
    # Treat .cu files as C++ with CUDA support
    - -xc++
    - --cuda-path=${CUDA_ROOT_DIR}
    - --cuda-gpu-arch=${CLANGD_CUDA_ARCH}
    - -I${CUDA_ROOT_DIR}/include
"
)

# Add CUDA extended lambda support and relaxed constexpr
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

# Add clang-format target for code formatting
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.h
    ${CMAKE_SOURCE_DIR}/src/*.cu
    ${CMAKE_SOURCE_DIR}/src/*.cuh
  )
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on all source files"
  )
endif()

# Include FetchContent for downloading dependencies
include(FetchContent)

# SDL2
FetchContent_Declare(
        SDL2
        URL https://github.com/libsdl-org/SDL/releases/download/release-2.32.10/SDL2-2.32.10.tar.gz
        URL_HASH SHA256=5f5993c530f084535c65a6879e9b26ad441169b3e25d789d83287040a9ca5165
)
FetchContent_MakeAvailable(SDL2)

# GLAD
FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/jonahshader/glad_opengl46.git
        GIT_TAG main
)
FetchContent_GetProperties(glad)
if(NOT glad_POPULATED)
    FetchContent_Populate(glad)
    add_subdirectory(${glad_SOURCE_DIR} ${glad_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# GLM
FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 0.9.9.8
)
FetchContent_MakeAvailable(glm)

# Artery Font (header-only library)
FetchContent_Declare(
        artery_font
        GIT_REPOSITORY https://github.com/Chlumsky/artery-font-format.git
        GIT_TAG master
)
FetchContent_GetProperties(artery_font)
if(NOT artery_font_POPULATED)
    FetchContent_Populate(artery_font)
    add_library(artery_font INTERFACE)
    target_include_directories(artery_font INTERFACE ${artery_font_SOURCE_DIR}/artery-font)
endif()

# lodepng
FetchContent_Declare(
  lodepng
  GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
  GIT_TAG master
)
FetchContent_GetProperties(lodepng)
if(NOT lodepng_POPULATED)
  FetchContent_Populate(lodepng)
endif()

# FastNoiseLite
FetchContent_Declare(
  fastnoiselite
  GIT_REPOSITORY https://github.com/Auburn/FastNoiseLite.git
  GIT_TAG master
)
FetchContent_GetProperties(fastnoiselite)
if(NOT fastnoiselite_POPULATED)
  FetchContent_Populate(fastnoiselite)
endif()

# Dear ImGui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.2
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
  FetchContent_Populate(imgui)
  add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  )
  target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
  )
  target_link_libraries(imgui PRIVATE SDL2::SDL2)
endif()

# CLI11 (header-only command-line parser)
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2
)
FetchContent_MakeAvailable(CLI11)

# toml++ (header-only TOML library)
FetchContent_Declare(
  tomlplusplus
  GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
  GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(alife_cuda src/main.cu
        ${lodepng_SOURCE_DIR}/lodepng.cpp
        src/graphics/extend_viewport.cpp
        src/graphics/extend_viewport.h
        src/graphics/shader.cpp
        src/graphics/shader.h
        src/graphics/renderers/line_renderer.cu
        src/graphics/renderers/line_renderer.cuh
        src/graphics/renderers/circle_renderer.cu
        src/graphics/renderers/circle_renderer.cuh
        src/graphics/renderers/font_renderer.cpp
        src/graphics/renderers/font_renderer.h
        src/graphics/renderers/rect_renderer.cu
        src/graphics/renderers/rect_renderer.cuh
        src/graphics/renderers/simple_rect_renderer.cu
        src/graphics/renderers/simple_rect_renderer.cuh
        src/graphics/renderers/rect_tex_renderer.cu
        src/graphics/renderers/rect_tex_renderer.cuh
        src/graphics/fluid_render.cu
        src/graphics/fluid_render.cuh
        src/graphics/soil_render.cu
        src/graphics/soil_render.cuh
        src/graphics/tree_render.cu
        src/graphics/tree_render.cuh
        src/systems/parameter_manager.h
        src/systems/game.cu
        src/systems/game.cuh
        src/systems/screen.h
        src/systems/default_screen.cuh
        src/systems/default_screen.cu
        src/systems/resources.cu
        src/systems/resources.cuh
        src/systems/custom_math.cu
        src/systems/custom_math.cuh
        src/screens/tree_test.cu
        src/screens/tree_test.cuh
        src/systems/trees.cu
        src/systems/trees.cuh
        src/systems/particle_fluid2.cuh
        src/systems/particle_fluid2.cu
        src/systems/profiler_gui.cuh
        src/systems/profiler_gui.cu
        src/screens/fluid_test2.cu
        src/screens/fluid_test2.cuh
        src/screens/fluid_soil.cu
        src/screens/fluid_soil.cuh
        src/systems/tree_types.cuh
        src/systems/world.cu
        src/systems/world.cuh
        src/systems/soil.cu
        src/systems/soil.cuh
        src/systems/timing_profiler.cu
        src/systems/timing_profiler.cuh
        src/systems/soa_helper.h
        src/systems/cuda_utils.cuh
        src/systems/float2_ops.cuh
        src/systems/terrain/fractal_noise.cu
        src/systems/terrain/fractal_noise.cuh
)

# Set CUDA properties
set_target_properties(alife_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(alife_cuda PRIVATE
        SDL2::SDL2
        glad
        glm::glm
        artery_font
        OpenMP::OpenMP_CXX
        imgui
        CLI11::CLI11
        tomlplusplus::tomlplusplus
)

# Include headers
target_include_directories(alife_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${glad_SOURCE_DIR}/include
        ${lodepng_SOURCE_DIR}
        ${fastnoiselite_SOURCE_DIR}/Cpp
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
)

# Platform-specific configurations
if(WIN32)
    target_link_libraries(alife_cuda PRIVATE SDL2::SDL2main)

    # Define SDL_MAIN_HANDLED
    target_compile_definitions(${PROJECT_NAME} PRIVATE SDL_MAIN_HANDLED)
    if(MSVC)
        # Use the MultiThreaded DLL runtime library for both Debug and Release
        set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    endif()
endif()

# Copy SDL2 DLL to output directory on Windows
if(WIN32)
    add_custom_command(TARGET alife_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL2::SDL2>
            $<TARGET_FILE_DIR:alife_cuda>
    )
endif()
