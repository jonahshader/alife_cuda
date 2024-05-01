#pragma once


#include <cstdint>
#include <random>
#include <glm/glm.hpp>
#include <vector>
#include "systems/Game.cuh"

namespace trees {
    struct BranchCore {
        float length{1};
        float rel_rot{0};
        float abs_rot{0};
        glm::vec2 pos{};
        std::uint32_t parent{};
    };

    struct BranchNav {
        uint32_t id{};
        std::vector<uint32_t> children{};
    };

    struct BranchShape {
        uint32_t start{0};
        uint32_t count{0};
    };

    struct BranchStats {
        float energy{0.0f};
        float max_energy{0.0f};
        float mix_rate{0.1f};
    };

    struct BranchNodeFull {
        BranchCore core{};
        BranchStats stats{};
        BranchNav nav{};
    };

    struct BranchNode {
        BranchCore core{};
        BranchStats stats{};
        BranchShape ch{};
    };

    using Tree = std::vector<BranchNode>;

    struct TreeBatch {
        std::vector<BranchShape> tree_shapes{};
        Tree trees{};
    };


    std::vector<BranchNodeFull> build_tree(uint32_t num_nodes, std::default_random_engine& rand);
    Tree build_tree_optimized(uint32_t num_nodes, std::default_random_engine &rand);

/**
 * Render a tree using the LineRenderer. Assumes the LineRenderer has already been set up.
 * @param line_renderer The LineRenderer to use
 * @param nodes The nodes of the tree
 */
    void render_tree(LineRenderer &line_renderer, const Tree& nodes, std::default_random_engine& rand);

/**
 * Mutate a tree by changing the length and rotation of the nodes scaled by the noise parameter. Propagates the changes through the tree.
 * @param nodes The nodes of the tree
 * @param rand The random number generator to use
 * @param noise The noise to apply to the mutations
 */
    void mutate_and_update(std::vector<BranchNodeFull>& nodes, std::default_random_engine& rand, float noise);

/**
 * Mutate a tree by changing the length and rotation of the nodes scaled by the noise parameter.
 * @param nodes The nodes of the tree
 * @param rand The random number generator to use
 * @param noise The noise to apply to the mutations
 */
    void mutate(std::vector<BranchNodeFull>& nodes, std::default_random_engine& rand, float noise);
    void mutate(Tree& nodes, std::default_random_engine& rand, float noise);

/**
 * Propagate orientation and position changes through the tree
 * @param nodes The nodes of the tree
 */
    void update_tree(std::vector<BranchNodeFull>& nodes);
    void update_tree(Tree& nodes);

/**
 * Sort the tree so that the parent of each node is before the node in the vector.
 * Result will be breadth-first traversal order.
 * @param nodes The nodes of the tree
 */
    std::vector<BranchNodeFull> sort_tree(const std::vector<BranchNodeFull>& nodes);

    Tree strip_nav(const std::vector<BranchNodeFull>& nodes);
    std::vector<BranchNodeFull> unstrip_nav(const Tree& nodes);


    void mix_node_contents(const Tree& read_nodes, Tree& write_nodes, float interp, float total_energy);
    void mix_node_contents(const Tree& read_nodes, Tree& write_nodes, float interp);

    float compute_total_energy(const Tree& nodes);
    float get_min_energy(const Tree& nodes);
    float get_max_energy(const Tree& nodes);

}

