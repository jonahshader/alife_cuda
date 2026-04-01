#pragma once
#include <cstdint>
#include <random>

// Single source of truth - expands to struct fields, defaults, TOML parsing, CLI options
// X(name, type, default, toml_path, cli_flag, description)
#define SIM_PARAMS_XMACRO(X)                                                                       \
  X(dt, float, 1.0f / 600.0f, "fluid.dt", "dt", "Simulation timestep")                             \
  X(dt_predict, float, 1.0f / 120.0f, "fluid.dt_predict", "dt-predict",                            \
    "Prediction timestep for position lookahead")                                                  \
  X(gravity, float, -13.0f, "fluid.gravity", "gravity", "Gravity acceleration")                    \
  X(collision_damping, float, 0.5f, "fluid.collision_damping", "collision-damping",                \
    "Velocity damping on boundary collision")                                                      \
  X(smoothing_radius, float, 0.2f, "fluid.smoothing_radius", "smoothing-radius",                   \
    "SPH smoothing radius (also grid cell size)")                                                  \
  X(target_density, float, 234.0f, "fluid.target_density", "target-density",                       \
    "Target rest density for pressure calculation")                                                \
  X(pressure_mult, float, 225.0f, "fluid.pressure_mult", "pressure-mult",                          \
    "Pressure force multiplier")                                                                   \
  X(near_pressure_mult, float, 18.0f, "fluid.near_pressure_mult", "near-pressure-mult",            \
    "Near-field pressure multiplier")                                                              \
  X(viscosity_strength, float, 0.03f, "fluid.viscosity_strength", "viscosity",                     \
    "Viscosity force strength")                                                                    \
  X(particles_per_cell, int, 4, "fluid.particles_per_cell", "particles-per-cell",                  \
    "Initial particles per grid cell")                                                             \
  X(max_particles_per_cell, int, 128, "fluid.max_particles_per_cell", "max-particles-per-cell",    \
    "Maximum particles per grid cell")                                                             \
  X(world_width, float, 32.0f, "world.width", "world-width", "World width in meters")              \
  X(world_height, float, 16.0f, "world.height", "world-height", "World height in meters")          \
  X(soil_cell_size, float, 0.1f, "world.soil_cell_size", "soil-cell-size",                         \
    "Soil grid cell size in meters")                                                               \
  X(seed, int64_t, 0, "world.seed", "seed", "RNG seed (0 = random)")

struct SimParams {
#define X_FIELD(name, type, def, toml, cli, desc) type name{def};
  SIM_PARAMS_XMACRO(X_FIELD)
#undef X_FIELD

  // Resolve seed=0 to a random value
  uint64_t resolve_seed() const {
    return seed == 0 ? std::random_device{}() : static_cast<uint64_t>(seed);
  }
};
