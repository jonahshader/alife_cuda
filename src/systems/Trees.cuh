#pragma once


#include <cstdint>
#include <random>
#include <glm/glm.hpp>
#include <vector>
#include "systems/Game.cuh"

struct TreeNode {
    float length;
    float relative_rotation;
    float absolute_rotation;
    glm::vec2 position;
    uint32_t id;
    std::uint32_t parent;
    std::vector<uint32_t> children{};
};

//struct TreeCore {
//    float length;
//    float rel_rot;
//    float abs_rot;
//    glm::vec2 pos;
//};


std::vector<TreeNode> build_tree(uint32_t num_nodes, std::default_random_engine& rand);

/**
 * Render a tree using the LineRenderer. Assumes the LineRenderer has already been set up.
 * @param line_renderer The LineRenderer to use
 * @param nodes The nodes of the tree
 */
void render_tree(LineRenderer &line_renderer, const std::vector<TreeNode>& nodes);

/**
 * Mutate a tree by changing the length and rotation of the nodes scaled by the noise parameter. Propagates the changes through the tree.
 * @param nodes The nodes of the tree
 * @param rand The random number generator to use
 * @param noise The noise to apply to the mutations
 */
void mutate_and_update(std::vector<TreeNode>& nodes, std::default_random_engine& rand, float noise);

/**
 * Mutate a tree by changing the length and rotation of the nodes scaled by the noise parameter.
 * @param nodes The nodes of the tree
 * @param rand The random number generator to use
 * @param noise The noise to apply to the mutations
 */
void mutate(std::vector<TreeNode>& nodes, std::default_random_engine& rand, float noise);


/**
 * Propagate orientation and position changes through the tree
 * @param nodes The nodes of the tree
 */
void update_tree(std::vector<TreeNode>& nodes);

/**
 * Sort the tree so that the parent of each node is before the node in the vector.
 * Result will be breadth-first traversal order.
 * @param nodes The nodes of the tree
 */
std::vector<TreeNode> sort_tree(const std::vector<TreeNode>& nodes);