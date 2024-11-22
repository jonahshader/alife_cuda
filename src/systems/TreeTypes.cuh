#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "SoAHelper.h"

namespace trees2 {
using bid_t = uint32_t;

constexpr float TORQUE_PER_RAD = 100.0f;
constexpr float MASS_PER_ENERGY = 1.0f;
constexpr float CUBIC_CM_PER_ENERGY = 1000.0f;
constexpr float MASS_PER_CUBIC_CM = MASS_PER_ENERGY / CUBIC_CM_PER_ENERGY;

#define FOR_BRANCH_CORE(N, D)                                                                      \
  D(float, length, 0)                                                                              \
  D(float, current_rel_rot, 0)                                                                     \
  D(float, target_rel_rot, 0)                                                                      \
  D(float, rot_vel, 0)                                                                             \
  D(float, rot_acc, 0)                                                                             \
  D(float, abs_rot, 0)                                                                             \
  N(glm::vec2, pos)                                                                                \
  N(glm::vec2, vel)                                                                                \
  N(glm::vec2, acc)                                                                                \
  D(bid_t, parent, 0)

#define FOR_BRANCH_SHAPE(N, D)                                                                     \
  D(bid_t, start, 0)                                                                               \
  D(bid_t, count, 0)

#define FOR_BRANCH_STATS(N, D)                                                                     \
  D(float, energy, 0.0f)                                                                           \
  D(float, energy_give_per_sec, 1.0f)                                                              \
  D(float, growth_rate, 0.5f) /* energy_per_second */                                              \
  D(float, thickness, 0.0f)                                                                        \
  D(float, target_thickness, 1.0f)                                                                 \
  D(float, target_length, 0.0f)

#define FOR_TREE_DATA(N, D) D(float, total_energy, 0.0f)

DEFINE_STRUCTS(BranchCore, FOR_BRANCH_CORE)

DEFINE_STRUCTS(BranchShape, FOR_BRANCH_SHAPE)

DEFINE_STRUCTS(BranchStats, FOR_BRANCH_STATS)

DEFINE_STRUCTS(TreeData, FOR_TREE_DATA)

// TODO: can probably use a macro to define these
struct BranchNode {
  BranchCore core{};
  BranchStats stats{};
  BranchShape ch{};
};

struct BranchNodeSoA {
  BranchCoreSoA core{};
  BranchStatsSoA stats{};
  BranchShapeSoA ch{};

  void push_back(const BranchNode &single) {
    core.push_back(single.core);
    stats.push_back(single.stats);
    ch.push_back(single.ch);
  }

  void push_back(const std::vector<BranchNode> &vec) {
    for (const auto &s : vec) {
      push_back(s);
    }
  }

  void swap_all(BranchNodeSoA &s) {
    core.swap_all(s.core);
    stats.swap_all(s.stats);
    ch.swap_all(s.ch);
  }
};

struct BranchNodeSoADevice {
  BranchCoreSoADevice core{};
  BranchStatsSoADevice stats{};
  BranchShapeSoADevice ch{};

  void copy_from_host(const BranchNodeSoA &host) {
    core.copy_from_host(host.core);
    stats.copy_from_host(host.stats);
    ch.copy_from_host(host.ch);
  }

  void copy_to_host(BranchNodeSoA &host) {
    core.copy_to_host(host.core);
    stats.copy_to_host(host.stats);
    ch.copy_to_host(host.ch);
  }
};

struct BranchNodePtrs {
  BranchCorePtrs core{};
  BranchStatsPtrs stats{};
  BranchShapePtrs ch{};

  void get_ptrs(BranchNodeSoADevice &s) {
    core.get_ptrs(s.core);
    stats.get_ptrs(s.stats);
    ch.get_ptrs(s.ch);
  }

  void get_ptrs(BranchNodeSoA &s) {
    core.get_ptrs(s.core);
    stats.get_ptrs(s.stats);
    ch.get_ptrs(s.ch);
  }
};

struct TreeBatch {
  BranchShapeSoA tree_shapes{};
  TreeDataSoA tree_data{};
  BranchNodeSoA trees{};
};

struct TreeBatchDevice {
  BranchShapeSoADevice tree_shapes{};
  TreeDataSoADevice tree_data{};
  BranchNodeSoADevice trees{};

  void copy_from_host(const TreeBatch &host) {
    tree_shapes.copy_from_host(host.tree_shapes);
    tree_data.copy_from_host(host.tree_data);
    trees.copy_from_host(host.trees);
  }

  void copy_to_host(TreeBatch &host) {
    tree_shapes.copy_to_host(host.tree_shapes);
    tree_data.copy_to_host(host.tree_data);
    trees.copy_to_host(host.trees);
  }
};

struct TreeBatchPtrs {
  BranchShapePtrs tree_shapes{};
  TreeDataPtrs tree_data{};
  BranchNodePtrs trees{};

  void get_ptrs(TreeBatchDevice &s) {
    tree_shapes.get_ptrs(s.tree_shapes);
    tree_data.get_ptrs(s.tree_data);
    trees.get_ptrs(s.trees);
  }

  void get_ptrs(TreeBatch &s) {
    tree_shapes.get_ptrs(s.tree_shapes);
    tree_data.get_ptrs(s.tree_data);
    trees.get_ptrs(s.trees);
  }
};

} // namespace trees2
