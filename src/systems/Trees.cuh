#pragma once


#include <cstdint>
#include <random>
#include <glm/glm.hpp>
#include <vector>
#include "systems/Game.cuh"
#include <thrust/device_vector.h>

#include "TreeTypes.cuh"

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

    Tree build_tree_optimized(uint32_t num_nodes, std::default_random_engine &rand, glm::vec2 start_pos = glm::vec2{});

    TreeBatch concatenate_trees(const std::vector<Tree> &trees);

    /**
     * Render a tree using the LineRenderer. Assumes the LineRenderer has already been set up.
     * @param line_renderer The LineRenderer to use
     * @param batch The nodes of the trees
     */
    void render_tree(LineRenderer &line_renderer, const TreeBatch &batch, std::default_random_engine &rand);
    void render_tree(LineRenderer &line_renderer, const trees2::TreeBatch &batch, std::default_random_engine &rand);


    /**
     * Mutate a tree by changing the length and rotation of the nodes scaled by the noise parameter.
     * @param batch The batch of trees
     * @param rand The random number generator to use
     * @param length_noise The amount of noise to apply to the length of the branches
     * @param rot_noise The amount of noise to apply to the rotation of the branches
     */
    void mutate_len_rot(TreeBatch &batch, std::default_random_engine &rand, float length_noise, float rot_noise);

    void mutate_pos(TreeBatch &batch, std::default_random_engine &rand, float noise);

    /**
     * Propagate orientation and position changes through the tree
     * @param batch The nodes of the batch
     */
    void update_tree(TreeBatch &batch);

    void update_tree_parallel(TreeBatch &read_batch, TreeBatch &write_batch);

    void update_tree_cuda(TreeBatch &read_batch, TreeBatch &write_batch);

    /**
     * Sort the tree so that the parent of each node is before the node in the vector.
     * Result will be breadth-first traversal order.
     * @param nodes The nodes of the tree
     */
    std::vector<BranchNodeFull> sort_tree(const std::vector<BranchNodeFull> &nodes);

    Tree strip_nav(const std::vector<BranchNodeFull> &nodes);

    std::vector<BranchNodeFull> unstrip_nav(const Tree &nodes);


    void mix_node_contents(const Tree &read_nodes, Tree &write_nodes, float interp, float total_energy);

    void mix_node_contents(const trees2::BranchNode read_nodes[], trees2::BranchNode write_nodes[], size_t start,
                           size_t node_count, float interp, float total_energy);

    void mix_node_contents(const Tree &read_nodes, Tree &write_nodes, float interp);

    void mix_node_contents(const TreeBatch &read_batch, TreeBatch &write_batch, float interp,
                           const std::vector<float> &total_energies);

    void mix_node_contents(const TreeBatch &read_batch, TreeBatch &write_batch, float interp);

    void mix_node_contents(const trees2::TreeBatch &read_batch, trees2::TreeBatch &write_batch, float inter);

    float compute_total_energy(const Tree &nodes);

    float compute_total_energy(const trees2::BranchNode nodes[], size_t node_count);

    float get_min_energy(const Tree &nodes);

    float get_max_energy(const Tree &nodes);

    __host__ __device__
    glm::vec2 get_length_vec(const trees2::BranchCore &core);
}
