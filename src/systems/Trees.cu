#include "Trees.cuh"
#include <cmath>
#include <iostream>
#include <stack>
#include <glm/glm.hpp>
#include "graphics/renderers/LineRenderer.h"

std::vector<TreeNode> build_tree(uint32_t num_nodes, std::default_random_engine& rand) {
//    std::cout << "Building tree with " << num_nodes << " nodes" << std::endl;
    std::uniform_real_distribution<float> length_dist(16.0f, 32.0f);
    std::normal_distribution<float> rot_dist(0.0f, 0.2f);

    std::vector<TreeNode> nodes;
    nodes.reserve(num_nodes);

    nodes.push_back(TreeNode{0.0f, 0.0f, M_PI/2, glm::vec2(0.0f), 0, 0});

    for (int i = 1; i < num_nodes; i++) {
        float length = length_dist(rand);
        float relative_rotation = rot_dist(rand);
        std::uniform_int_distribution<uint32_t> parent_dist(std::max(0, (i-1) / 2), i - 1);
        uint32_t parent = parent_dist(rand);

        auto& parent_node = nodes[parent];
        parent_node.children.push_back(i);
        float absolute_rotation = parent_node.absolute_rotation + relative_rotation;
        glm::vec2 position = parent_node.position + glm::vec2(std::cos(absolute_rotation), std::sin(absolute_rotation)) * length;

        nodes.emplace_back(TreeNode{length, relative_rotation, absolute_rotation, position, static_cast<uint32_t>(i), parent});
    }

    return nodes;
}

void render_tree(LineRenderer &line_renderer, const std::vector<TreeNode>& nodes) {
    for (const auto& node : nodes) {
        if (node.parent == 0) {
            continue;
        }
        auto& parent = nodes[node.parent];
        line_renderer.add_line(parent.position.x, parent.position.y, node.position.x, node.position.y, 2, 0, glm::vec4(1, 1, 1, 0.1), glm::vec4(1, 1, 1, 0.0));
    }
}

void mutate_and_update(std::vector<TreeNode>& nodes, std::default_random_engine& rand, float noise) {
    std::normal_distribution<float> length_dist(0.0f, noise);
    std::normal_distribution<float> rot_dist(0.0f, noise);

    for (auto& node : nodes) {
        float scale = node.parent * 0.0001f;
        node.length += length_dist(rand) * scale;
        // ensure length is positive
        node.length = std::max(0.0f, node.length);
        node.relative_rotation += rot_dist(rand) * scale;
        node.absolute_rotation = nodes[node.parent].absolute_rotation + node.relative_rotation;
        node.position = nodes[node.parent].position + glm::vec2(std::cos(node.absolute_rotation), std::sin(node.absolute_rotation)) * node.length;
    }
}

void mutate(std::vector<TreeNode>& nodes, std::default_random_engine& rand, float noise) {
    std::normal_distribution<float> length_dist(0.0f, noise);
    std::normal_distribution<float> rot_dist(0.0f, noise);

    for (auto& node : nodes) {
        float scale = node.parent * 0.0001f;
        node.length += length_dist(rand) * scale;
        // ensure length is positive
        node.length = std::max(0.0f, node.length);
        node.relative_rotation += rot_dist(rand) * scale;
    }
}

void update_tree(std::vector<TreeNode>& nodes) {
    for (auto& node : nodes) {
        node.absolute_rotation = nodes[node.parent].absolute_rotation + node.relative_rotation;
        node.position = nodes[node.parent].position + glm::vec2(std::cos(node.absolute_rotation), std::sin(node.absolute_rotation)) * node.length;
    }
}


std::vector<TreeNode> sort_tree(const std::vector<TreeNode>& nodes) {
    std::vector<TreeNode> sorted_tree;
    sorted_tree.push_back(nodes[0]);
    // add the children uhhhhhhhhh
    for (int i = 0; i < nodes.size(); ++i) {
        auto& node = sorted_tree[i];
        for (auto child_id : node.children) {
            auto new_child = nodes[child_id];
            auto new_child_id = sorted_tree.size();
            new_child.id = new_child_id;
            new_child.parent = i;
            sorted_tree.push_back(new_child);
        }
    }

    // temp hack recalc childnre
    for (auto& node : sorted_tree) {
        node.children.clear();
    }

    for (int i = 1; i < sorted_tree.size(); ++i) {
        auto& node = sorted_tree[i];
        auto& parent = sorted_tree[node.parent];
        parent.children.push_back(node.id);
    }

    return sorted_tree;
}

//void sort_tree(const std::vector<TreeNode>& unsorted, std::vector<TreeNode>& sorted) {
//
//}