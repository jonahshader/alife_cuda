#pragma once

#include "soa_helper.h"

#include <glm/glm.hpp>

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

// Composite SoA — templated on buffer backend
template <template <typename> class Buffer>
struct BranchNodeSoA {
  BranchCoreSoA<Buffer> core{};
  BranchStatsSoA<Buffer> stats{};
  BranchShapeSoA<Buffer> ch{};
};

template <template <typename> class Dst, template <typename> class Src>
void copy(BranchNodeSoA<Dst> &dst, const BranchNodeSoA<Src> &src) {
  copy(dst.core, src.core);
  copy(dst.stats, src.stats);
  copy(dst.ch, src.ch);
}

template <template <typename> class Buffer>
void push_back(BranchNodeSoA<Buffer> &soa, const BranchNode &single) {
  push_back(soa.core, single.core);
  push_back(soa.stats, single.stats);
  push_back(soa.ch, single.ch);
}

template <template <typename> class Buffer>
void push_back(BranchNodeSoA<Buffer> &soa, const std::vector<BranchNode> &vec) {
  for (const auto &single : vec) {
    push_back(soa, single);
  }
}

template <template <typename> class Buffer>
void swap_all(BranchNodeSoA<Buffer> &a, BranchNodeSoA<Buffer> &b) {
  swap_all(a.core, b.core);
  swap_all(a.stats, b.stats);
  swap_all(a.ch, b.ch);
}

struct BranchNodePtrs {
  BranchCorePtrs core{};
  BranchStatsPtrs stats{};
  BranchShapePtrs ch{};

  template <template <typename> class Buffer>
  void get_ptrs(BranchNodeSoA<Buffer> &s) {
    core.get_ptrs(s.core);
    stats.get_ptrs(s.stats);
    ch.get_ptrs(s.ch);
  }
};

template <template <typename> class Buffer>
struct TreeBatch {
  BranchShapeSoA<Buffer> tree_shapes{};
  TreeDataSoA<Buffer> tree_data{};
  BranchNodeSoA<Buffer> trees{};
};

template <template <typename> class Dst, template <typename> class Src>
void copy(TreeBatch<Dst> &dst, const TreeBatch<Src> &src) {
  copy(dst.tree_shapes, src.tree_shapes);
  copy(dst.tree_data, src.tree_data);
  copy(dst.trees, src.trees);
}

struct TreeBatchPtrs {
  BranchShapePtrs tree_shapes{};
  TreeDataPtrs tree_data{};
  BranchNodePtrs trees{};

  template <template <typename> class Buffer>
  void get_ptrs(TreeBatch<Buffer> &s) {
    tree_shapes.get_ptrs(s.tree_shapes);
    tree_data.get_ptrs(s.tree_data);
    trees.get_ptrs(s.trees);
  }
};

} // namespace trees2
