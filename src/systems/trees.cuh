#pragma once

#include "tree_types.cuh"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <thrust/device_vector.h>

namespace trees {
struct BranchNav {
  trees2::bid_t id{};
  std::vector<trees2::bid_t> children{};
};

struct BranchNodeFull {
  trees2::BranchCore core{};
  trees2::BranchStats stats{};
  BranchNav nav{};
};

using Tree = std::vector<trees2::BranchNode>;

struct TreeBatch {
  std::vector<trees2::BranchShape> tree_shapes{};
  std::vector<trees2::TreeData> tree_data{};
  Tree trees{};
};

std::vector<BranchNodeFull> build_tree(uint32_t num_nodes, std::default_random_engine &rand,
                                       glm::vec2 start_pos = glm::vec2{});

Tree build_tree_optimized(uint32_t num_nodes, std::default_random_engine &rand,
                          glm::vec2 start_pos = glm::vec2{});

TreeBatch concatenate_trees(const std::vector<Tree> &trees);

/**
 * Propagate orientation and position changes through the tree
 * @param batch The nodes of the batch
 */
void update_tree_cuda(trees2::TreeBatch<DeviceBuffer> &read_batch_device,
                      trees2::TreeBatch<DeviceBuffer> &write_batch_device);

/**
 * Sort the tree so that the parent of each node is before the node in the vector.
 * Result will be breadth-first traversal order.
 * @param nodes The nodes of the tree
 */
std::vector<BranchNodeFull> sort_tree(const std::vector<BranchNodeFull> &nodes);

__host__ __device__ glm::vec2 get_length_vec(const trees2::BranchCore &core);

template <template <typename> class Buffer>
__host__ __device__ glm::vec2 get_length_vec(const trees2::BranchCoreSoA<Buffer> &core,
                                             trees2::bid_t i) {
  return glm::vec2(std::cos(core.abs_rot[i]), std::sin(core.abs_rot[i])) * core.length[i];
}

__host__ __device__ glm::vec2 get_length_vec(float abs_rot, float length);

// Pure data struct — no renderers, no methods
struct TreesState {
  trees2::TreeBatch<HostBuffer> read_host{}, write_host{};
  trees2::TreeBatch<DeviceBuffer> read_device{}, write_device{};
};

// Free functions for simulation logic
void init_trees(TreesState &state, uint32_t num_trees, uint32_t num_nodes,
                std::default_random_engine &rand);
void update_trees(TreesState &state, float dt);

} // namespace trees
