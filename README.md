# ALife CUDA

Artificial life simulation using CUDA for compute. The guide below explains how to set up the project, the necessary dependencies, and how to create both Debug and Release builds.

## Dependencies

- **CMake**: Make sure you have CMake installed on your system. You can download it from [here](https://cmake.org/download/).
- **CUDA Toolkit**: Ensure you have a compatible version of the CUDA Toolkit installed. You can download it from [here](https://developer.nvidia.com/cuda-downloads).

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jonahshader/alife_cuda.git
   cd alife_cuda
   ```

2. Ensure CUDA and CMake are installed and properly set up. TODO: more details.

## Building the Project

### Linux

#### Debug Build

1. Create a build directory for the debug configuration:
   ```bash
   mkdir build-debug
   cd build-debug
   ```

2. Configure the project with `CMAKE_BUILD_TYPE=Debug`:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   ```

3. Build the project:
   ```bash
   cmake --build .
   ```

#### Release Build

1. Create a build directory for the release configuration:
   ```bash
   mkdir build-release
   cd build-release
   ```

2. Configure the project with `CMAKE_BUILD_TYPE=Release`:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

3. Build the project:
   ```bash
    cmake --build .
   ```

### Windows

For Windows users, it's recommended to use the **Visual Studio Developer Command Prompt** to ensure the MSVC compiler is used.

1. Open the **Visual Studio Developer Command Prompt**.

2. Create separate directories for Debug and Release builds:

   #### Debug Build

   ```bash
   mkdir build-debug
   cd build-debug
   cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
   nmake
   ```

   #### Release Build

   ```bash
   mkdir build-release
   cd build-release
   cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
   nmake
   ```

   Alternatively, you can generate Visual Studio project files using:
   TODO: verify. I thought a newer version of Visual Studio was needed.
   ```bash
   cmake -G "Visual Studio 16 2019" -A x64 ..
   ```

   Then open the generated `.sln` file in Visual Studio, select the configuration (Debug/Release), and build.

### Additional Notes

- **CMake Options**: 
  - `CMAKE_BUILD_TYPE=Debug`: Enables debugging information and disables optimizations.
  - `CMAKE_BUILD_TYPE=Release`: Optimizes the code and strips debug information.

- **CUDA Version**: Make sure that your CUDA version is compatible with your compiler. You can check the [CUDA compatibility guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for more information.

## License

Add license details here.
