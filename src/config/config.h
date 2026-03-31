#pragma once

#include "config/sim_params.h"

#include <optional>
#include <string>

#include <CLI/CLI.hpp>

// Load simulation parameters from a TOML config file
// Returns nullopt if file doesn't exist or fails to parse
std::optional<SimParams> load_sim_params(const std::string &path);

// Write a default config file with all parameters documented
void write_default_config(const std::string &path);

// Register all simulation parameters as CLI options
void register_sim_params_cli(CLI::App &app, SimParams &params);
