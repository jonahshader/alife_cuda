#include "config/config.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <toml++/toml.hpp>

namespace {

// Helper to split "section.key" into components
std::pair<std::string, std::string> split_toml_path(const std::string &path) {
  auto dot = path.find('.');
  if (dot == std::string::npos) {
    return {"", path};
  }
  return {path.substr(0, dot), path.substr(dot + 1)};
}

// Helper to get value from TOML table given a dotted path
template <typename T>
std::optional<T> get_toml_value(const toml::table &tbl, const std::string &path) {
  auto [section, key] = split_toml_path(path);
  if (section.empty()) {
    if (auto val = tbl[key].value<T>()) {
      return *val;
    }
  } else {
    if (auto sec = tbl[section].as_table()) {
      if (auto val = (*sec)[key].value<T>()) {
        return *val;
      }
    }
  }
  return std::nullopt;
}

// Convert a default value to its TOML string representation
template <typename T>
std::string to_toml_string(T val) {
  return std::to_string(val);
}

template <>
std::string to_toml_string(float val) {
  // Use fixed notation, then trim trailing zeros
  std::string s = std::to_string(val); // printf %f: 6 decimal places
  size_t dot = s.find('.');
  if (dot != std::string::npos) {
    size_t last_nonzero = s.find_last_not_of('0');
    if (last_nonzero > dot) {
      s.erase(last_nonzero + 1);
    } else {
      s.erase(dot + 2); // keep at least ".0"
    }
  }
  return s;
}

struct ConfigLine {
  std::string section;
  std::string key;
  std::string value;
  std::string desc;
};

// Write config lines grouped by section, with comments aligned within each section
void write_config_lines(std::ofstream &out, const std::vector<ConfigLine> &lines) {
  size_t i = 0;
  while (i < lines.size()) {
    const std::string &section = lines[i].section;
    size_t section_start = i;
    while (i < lines.size() && lines[i].section == section)
      ++i;

    // Find max "key = value" width for alignment
    size_t max_kv = 0;
    for (size_t j = section_start; j < i; ++j) {
      max_kv = std::max(max_kv, lines[j].key.size() + 3 + lines[j].value.size());
    }

    out << "[" << section << "]\n";
    for (size_t j = section_start; j < i; ++j) {
      std::string kv = lines[j].key + " = " + lines[j].value;
      size_t pad = max_kv >= kv.size() ? max_kv + 2 - kv.size() : 1;
      out << kv << std::string(pad, ' ') << "# " << lines[j].desc << "\n";
    }
    out << "\n";
  }
}

} // namespace

std::optional<SimParams> load_sim_params(const std::string &path) {
  try {
    auto tbl = toml::parse_file(path);
    SimParams params;

// Generate TOML parsing using X macro
#define X_PARSE(name, type, def, toml_path, cli, desc)                                             \
  if (auto val = get_toml_value<type>(tbl, toml_path)) {                                           \
    params.name = *val;                                                                            \
  }
    SIM_PARAMS_XMACRO(X_PARSE)
#undef X_PARSE

    return params;
  } catch (const toml::parse_error &err) {
    // File not found or parse error — return nullopt silently for missing files,
    // print error for actual parse failures
    if (std::filesystem::exists(path)) {
      std::cerr << "TOML parse error in " << path << ": " << err << std::endl;
    }
    return std::nullopt;
  }
}

void write_default_config(const std::string &path) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Failed to create config file: " << path << std::endl;
    return;
  }

  out << "# ALife CUDA Configuration File\n";
  out << "# All values shown are defaults\n\n";

  // Collect entries from X-macro
  std::vector<ConfigLine> lines;
#define X_LINE(name, type, def, toml_path, cli, desc)                                              \
  {                                                                                                \
    auto [sec, key] = split_toml_path(toml_path);                                                  \
    lines.push_back({sec, key, to_toml_string(static_cast<type>(def)), desc});                     \
  }
  SIM_PARAMS_XMACRO(X_LINE)
#undef X_LINE

  write_config_lines(out, lines);

  std::cout << "Created default config at: " << path << std::endl;
}

void register_sim_params_cli(CLI::App &app, SimParams &params) {
#define X_CLI(name, type, def, toml_path, cli_flag, desc)                                          \
  app.add_option("--" cli_flag, params.name, desc);
  SIM_PARAMS_XMACRO(X_CLI)
#undef X_CLI
}
