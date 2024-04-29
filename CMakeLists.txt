cmake_minimum_required(VERSION 3.28)
project(alife_cuda CUDA C CXX)

set(CMAKE_CUDA_STANDARD 20)

set(LibraryRoot "F:/Programming/C++ Libraries/")

include_directories("${LibraryRoot}SDL2-devel-2.30.1-VC/SDL2-2.30.1/include")
link_directories("${LibraryRoot}SDL2-devel-2.30.1-VC/SDL2-2.30.1/lib/x64")
include_directories("${LibraryRoot}glm")
include_directories("${LibraryRoot}artery-font-format-master/artery-font")
include_directories("${LibraryRoot}glad_opengl46/include")

# Add glad sources
set(GLAD_SOURCES "${LibraryRoot}glad_opengl46/src/glad.c")
set(GLAD_HEADERS "${LibraryRoot}glad_opengl46/include/glad/glad.h" "${LibraryRoot}glad_opengl46/include/KHR/khrplatform.h")


add_executable(alife_cuda src/main.cu ${GLAD_SOURCES} ${GLAD_HEADERS}
        src/graphics/ExtendViewport.cpp
        src/graphics/ExtendViewport.h
        src/graphics/Shader.cpp
        src/graphics/Shader.h
        src/graphics/renderers/LineRenderer.cpp
        src/graphics/renderers/LineRenderer.h
        src/graphics/renderers/CircleRenderer.cpp
        src/graphics/renderers/CircleRenderer.h
        src/graphics/renderers/FontRenderer.cpp
        src/graphics/renderers/FontRenderer.h
        src/lodepng.cpp
        src/lodepng.h
        src/graphics/renderers/RectRenderer.cpp
        src/graphics/renderers/RectRenderer.h
        src/systems/Game.cu
        src/systems/Game.cuh
        src/systems/Screen.h
        src/systems/Resources.cu
        src/systems/Resources.cuh
        src/spatial_sort.cu
        src/spatial_sort.cuh
        src/screens/TreeTest.cu
        src/screens/TreeTest.cuh
        src/systems/Trees.cu
        src/systems/Trees.cuh
        src/systems/Fluid.cu
        src/systems/Fluid.cuh
        src/screens/FluidTest.cu
        src/screens/FluidTest.cuh
)
target_include_directories(alife_cuda PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} SDL2main)
target_link_libraries(${PROJECT_NAME} SDL2)

set_target_properties(alife_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)