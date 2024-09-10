cmake_minimum_required(VERSION 3.28)
project(alife_cuda CUDA CXX) # removed C

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include FetchContent for downloading dependencies
include(FetchContent)

# SDL2
FetchContent_Declare(
        SDL2
        URL https://github.com/libsdl-org/SDL/releases/download/release-2.30.3/SDL2-2.30.3.tar.gz
        URL_HASH SHA256=820440072f8f5b50188c1dae104f2ad25984de268785be40c41a099a510f0aec
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

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(alife_cuda src/main.cu
        ${lodepng_SOURCE_DIR}/lodepng.cpp
        src/graphics/ExtendViewport.cpp
        src/graphics/ExtendViewport.h
        src/graphics/Shader.cpp
        src/graphics/Shader.h
        src/graphics/renderers/LineRenderer.cu
        src/graphics/renderers/LineRenderer.cuh
        src/graphics/renderers/CircleRenderer.cpp
        src/graphics/renderers/CircleRenderer.h
        src/graphics/renderers/FontRenderer.cpp
        src/graphics/renderers/FontRenderer.h
        src/graphics/renderers/RectRenderer.cu
        src/graphics/renderers/RectRenderer.cuh
        src/systems/Game.cu
        src/systems/Game.cuh
        src/systems/Screen.h
        src/systems/Resources.cu
        src/systems/Resources.cuh
        src/systems/CustomMath.cu
        src/systems/CustomMath.cuh
        src/systems/Kernels.cu
        src/systems/Kernels.cuh
        src/spatial_sort.cu
        src/spatial_sort.cuh
        src/screens/TreeTest.cu
        src/screens/TreeTest.cuh
        src/systems/Trees.cu
        src/systems/Trees.cuh
        src/systems/Fluid.cu
        src/systems/Fluid.cuh
        src/systems/ParticleFluid.cuh
        src/systems/ParticleFluid.cu
        src/screens/FluidTest.cu
        src/screens/FluidTest.cuh
        src/systems/TreeTypes.cuh
        src/systems/ui/Container.cu
        src/systems/ui/Container.cuh
        src/systems/ui/WorldUI.cu
        src/systems/ui/WorldUI.cuh
        src/systems/World.cu
        src/systems/World.cuh
        src/systems/Soil.cu
        src/systems/Soil.cuh
        src/systems/SoAHelper.h
        src/screens/SoilTest.cu
        src/screens/SoilTest.cuh
        src/systems/terrain/FractalNoise.cu
        src/systems/terrain/FractalNoise.cuh
        src/systems/terrain/Seamless.cu
        src/systems/terrain/Seamless.cuh
)

# Set CUDA properties
set_target_properties(alife_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Add src directory to include path
# target_include_directories(alife_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link libraries
target_link_libraries(alife_cuda PRIVATE
        SDL2::SDL2
        glad
        glm::glm
        artery_font
        OpenMP::OpenMP_CXX
)

# Include headers
target_include_directories(alife_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${glad_SOURCE_DIR}/include
        ${lodepng_SOURCE_DIR}
        ${fastnoiselite_SOURCE_DIR}/Cpp
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