#include "Trees.cuh"
#include <cmath>
#include <iostream>
#include <stack>
#include <glm/glm.hpp>
#include "graphics/renderers/LineRenderer.h"

namespace trees {
    std::vector<BranchNodeFull> build_tree(uint32_t num_nodes, std::default_random_engine& rand, glm::vec2 start_pos) {
//    std::cout << "Building tree with " << num_nodes << " nodes" << std::endl;
        std::uniform_real_distribution<float> length_dist(16.0f, 32.0f);
        std::normal_distribution<float> rot_dist(0.0f, 0.2f);
        std::uniform_real_distribution<float> energy_dist(0.0f, 1.0f);

        std::vector<BranchNodeFull> nodes;
        nodes.reserve(num_nodes);

        nodes.emplace_back(BranchNodeFull{
                .core=BranchCore{
                    // note: length of first node isn't used
                        .abs_rot=M_PI/2,
                        .pos=start_pos,
                },
                .stats=BranchStats{
//                    .energy = static_cast<float>(num_nodes)
                }
        });

        for (int i = 1; i < num_nodes; i++) {
            float length = length_dist(rand);
            float relative_rotation = rot_dist(rand);
            std::uniform_int_distribution<uint32_t> parent_dist(std::max(0, (i-1) / 2), i - 1);
            uint32_t parent = parent_dist(rand);

            auto& parent_node = nodes[parent];
            parent_node.nav.children.push_back(i);
            float absolute_rotation = parent_node.core.abs_rot + relative_rotation;
            glm::vec2 position = parent_node.core.pos + glm::vec2(std::cos(absolute_rotation), std::sin(absolute_rotation)) * length;

            nodes.emplace_back(BranchNodeFull{
                    .core=BranchCore{
                            .length=length,
                            .rel_rot=relative_rotation,
                            .abs_rot=absolute_rotation,
                            .pos=position,
                            .parent=parent
                    },
                    .stats=BranchStats{
//                        .energy = energy_dist(rand)
                    },
                    .nav=BranchNav{
                            .id=static_cast<uint32_t>(i)
                    }
            });

        }

        nodes[nodes.size() - 1].stats.energy = static_cast<float>(num_nodes);

        return nodes;
    }

    std::vector<BranchNode> build_tree_optimized(uint32_t num_nodes, std::default_random_engine &rand, glm::vec2 start_pos) {
        return strip_nav(sort_tree(build_tree(num_nodes, rand, start_pos)));
    }

    TreeBatch concatenate_trees(const std::vector<Tree> &trees) {
        TreeBatch batch{};
        // reserve the exact amount of mem needed to store the concatenated tree
        uint32_t total_size = 0;
        for (const auto& tree : trees) {
            total_size += tree.size();
        }
        batch.trees.reserve(total_size);

        // first part of the tree batch is just the first tree
        batch.tree_shapes.emplace_back(BranchShape{
            .start=0,
            .count=static_cast<uint32_t>(trees[0].size()),
        });
        batch.trees = trees[0];
        for (int i = 1; i < trees.size(); ++i) {
            auto tree = trees[i];
            const auto offset = batch.tree_shapes.back().start + batch.tree_shapes.back().count;
            // offset parent and children start fields
            for (auto& node : tree) {
                node.core.parent += offset;
                node.ch.start += offset;
            }
            // insert a shape for this new tree
            batch.tree_shapes.emplace_back(BranchShape{
                .start = offset,
                .count = static_cast<uint32_t>(tree.size()),
            });
            // append the tree to the batch
            batch.trees.insert(batch.trees.end(), tree.begin(), tree.end());
        }

        return batch;
    }


    void render_tree(LineRenderer &line_renderer, const std::vector<BranchNode>& nodes, std::default_random_engine& rand) {
        std::uniform_real_distribution<float> dither(-1/255.0f, 1/255.0f);
        for (const auto& node : nodes) {
            auto& parent = nodes[node.core.parent];
//            float energy = std::max(std::min(node.stats.energy + dither(rand), 1.0f), 0.0f);
            float energy = std::max(std::min(node.stats.energy, 1.0f), 0.0f);
            if (energy == 0) {
                line_renderer.add_line(parent.core.pos.x, parent.core.pos.y, node.core.pos.x, node.core.pos.y, 2, 0, glm::vec4(1, 0, 0, 1), glm::vec4(1, 0, 0, 1));
            } else {
                line_renderer.add_line(parent.core.pos.x, parent.core.pos.y, node.core.pos.x, node.core.pos.y, 2, 0, glm::vec4(1, 1, 1, energy * 0.5f), glm::vec4(1, 1, 1, energy * 0.5f));
            }
        }
    }

    void render_tree(LineRenderer &line_renderer, const TreeBatch& batch, std::default_random_engine& rand) {
        render_tree(line_renderer, batch.trees, rand);
    }

    void mutate_and_update(std::vector<BranchNodeFull>& nodes, std::default_random_engine& rand, float noise) {
        std::normal_distribution<float> length_dist(0.0f, noise);
        std::normal_distribution<float> rot_dist(0.0f, noise);

        for (auto& node : nodes) {
            auto& core = node.core;
            auto& nav = node.nav;
            float scale = core.parent * 0.0001f;
            core.length += length_dist(rand) * scale;
            // ensure length is positive
            core.length = std::max(0.0f, core.length);
            core.rel_rot += rot_dist(rand) * scale;
            core.abs_rot = nodes[core.parent].core.abs_rot + core.rel_rot;
            core.pos = nodes[core.parent].core.pos + glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.length;
        }
    }

    void mutate(std::vector<BranchNodeFull>& nodes, std::default_random_engine& rand, float noise) {
        std::normal_distribution<float> length_dist(0.0f, noise);
        std::normal_distribution<float> rot_dist(0.0f, noise);

        for (auto& node : nodes) {
            auto& core = node.core;
            float scale = core.parent * 0.0001f;
            core.length += length_dist(rand) * scale;
            // ensure length is positive
            core.length = std::max(0.0f, core.length);
            core.rel_rot += rot_dist(rand) * scale;
        }
    }

    void mutate(std::vector<BranchNode>& nodes, std::default_random_engine& rand, float noise) {
        std::normal_distribution<float> length_dist(0.0f, noise);
        std::normal_distribution<float> rot_dist(0.0f, noise);

        for (auto& node : nodes) {
            auto& core = node.core;
            float scale = core.parent * 0.0001f;
            core.length += length_dist(rand) * scale;
            // ensure length is positive
            core.length = std::max(0.0f, core.length);
            core.rel_rot += rot_dist(rand) * scale;
        }
    }

    void update_tree(std::vector<BranchNodeFull>& nodes) {
        for (auto i = 1; i < nodes.size(); ++i) {
            auto& core = nodes[i].core;
            core.abs_rot = nodes[core.parent].core.abs_rot + core.rel_rot;
            core.pos = nodes[core.parent].core.pos + glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.length;
        }
    }

    void update_tree(TreeBatch& batch) {
        for (int i = 0; i < batch.trees.size(); ++i) {
            auto& core = batch.trees[i].core;
            if (i != core.parent) {
                core.abs_rot = batch.trees[core.parent].core.abs_rot + core.rel_rot;
                core.pos = batch.trees[core.parent].core.pos + glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.length;
            }
        }
    }

    void update_tree(std::vector<BranchNode>& nodes) {
        for (auto i = 1; i < nodes.size(); ++i) {
            auto& core = nodes[i].core;
            core.abs_rot = nodes[core.parent].core.abs_rot + core.rel_rot;
            core.pos = nodes[core.parent].core.pos + glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.length;
        }
    }

    std::vector<BranchNodeFull> sort_tree(const std::vector<BranchNodeFull>& nodes) {
        std::vector<BranchNodeFull> sorted_tree;
        sorted_tree.push_back(nodes[0]);
        // add the children uhhhhhhhhh
        for (int i = 0; i < nodes.size(); ++i) {
            auto& node = sorted_tree[i];
            for (auto child_id : node.nav.children) {
                auto new_child = nodes[child_id];
                auto new_child_id = sorted_tree.size();
                new_child.nav.id = new_child_id;
                new_child.core.parent = i;
                sorted_tree.push_back(new_child);
            }
        }

        // recalculate children
        for (auto& node : sorted_tree) {
            node.nav.children.clear();
        }

        for (int i = 1; i < sorted_tree.size(); ++i) {
            auto& node = sorted_tree[i];
            auto& parent = sorted_tree[node.core.parent];
            parent.nav.children.push_back(node.nav.id);
        }

        return sorted_tree;
    }

    std::vector<BranchNode> strip_nav(const std::vector<BranchNodeFull>& nodes) {
        std::vector<BranchNode> stripped;
        stripped.reserve(nodes.size());
        for (const auto& node : nodes) {
            uint32_t num_children = node.nav.children.size();
            uint32_t children_start = 0;
            if (num_children > 0) {
                children_start = node.nav.children[0];
            }
            stripped.push_back(BranchNode{
                    .core=node.core,
                    .stats=node.stats,
                    .ch=BranchShape{
                        .start=children_start,
                        .count=num_children
                    }
            });
        }
        return stripped;
    }

    std::vector<BranchNodeFull> unstrip_nav(const std::vector<BranchNode>& nodes) {
        std::vector<BranchNodeFull> unstripped;
        unstripped.reserve(nodes.size());
        for (auto i = 0; i < nodes.size(); ++i) {
            auto& node = nodes[i];
            unstripped.push_back(BranchNodeFull{
                    .core=node.core,
                    .nav=BranchNav{
                            .id=static_cast<uint32_t>(i)
                    }
            });
        }

        // recalculate children
        // TODO: this can be done a little faster by using info form BranchChildren struct instead of recomputing
        for (auto& node : unstripped) {
            node.nav.children.clear();
        }

        for (int i = 1; i < unstripped.size(); ++i) {
            auto& node = unstripped[i];
            auto& parent = unstripped[node.core.parent];
            parent.nav.children.push_back(node.nav.id);
        }

        return unstripped;
    }

    void mix_node_contents(const std::vector<BranchNode>& read_nodes, std::vector<BranchNode>& write_nodes, float interp, float total_energy) {
        for (auto i = 0; i < read_nodes.size(); ++i) {
            auto& read_node = read_nodes[i];
            auto& write_node = write_nodes[i];

            float sum = read_node.stats.energy;
            float weight_sum = 1.0f;

            // if parent id equals node id, that indicates it is a root
            if (read_node.core.parent != i) {
                sum += read_nodes[read_node.core.parent].stats.energy;
                weight_sum += 1.0f;
            }

            auto child_start_index = read_node.ch.start;
            for (uint32_t j = 0; j < read_node.ch.count; ++j) {
                auto& child = read_nodes[j + child_start_index];
                sum += child.stats.energy;
                weight_sum += 1.0f;
            }

            float avg_energy = sum / weight_sum;
            write_node.stats.energy = (1 - interp) * write_nodes[i].stats.energy + interp * avg_energy;
        }

        float new_total_energy = compute_total_energy(write_nodes);
        float scale_factor = total_energy / new_total_energy;

        for (auto& node : write_nodes) {
            node.stats.energy *= scale_factor;
        }
    }

    void mix_node_contents(const std::vector<BranchNode>& read_nodes, std::vector<BranchNode>& write_nodes, float interp) {
        float total_energy = compute_total_energy(read_nodes);
        mix_node_contents(read_nodes, write_nodes, interp, total_energy);
    }

    float compute_total_energy(const std::vector<BranchNode>& nodes) {
        float sum = 0;
        for (auto& node : nodes) {
            sum += node.stats.energy;
        }
        return sum;
    }

    float get_min_energy(const std::vector<BranchNode>& nodes) {
        float min = std::numeric_limits<float>::max();
        // skip the first node
        for (auto i = 1; i < nodes.size(); ++i) {
            if (nodes[i].stats.energy < min) {
                min = nodes[i].stats.energy;
            }
        }
        return min;
    }

    float get_max_energy(const std::vector<BranchNode>& nodes) {
        float max = std::numeric_limits<float>::min();
        // skip the first node
        for (auto i = 1; i < nodes.size(); ++i) {
            if (nodes[i].stats.energy > max) {
                max = nodes[i].stats.energy;
            }
        }
        return max;
    }
}



/*
 * TODO: i think i can actually do cuda trees if children of a branch are contiguous.
 * the main issue was having a variable length array of pointers to children, but
 * instead, if the children are contiguous, i can just store pointer to first child and
 * number of children.
 */