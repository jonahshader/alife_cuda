#include "Trees.cuh"
#include <cmath>
#include <stack>
#include <glm/glm.hpp>
#include <iostream>
#include "graphics/renderers/LineRenderer.h"

namespace trees {
    std::vector<BranchNodeFull> build_tree(uint32_t num_nodes, std::default_random_engine& rand, glm::vec2 start_pos) {
        std::uniform_real_distribution<float> length_dist(16.0f, 32.0f);
        std::normal_distribution<float> rot_dist(0.0f, 0.2f);
        std::uniform_real_distribution<float> energy_dist(0.0f, 1.0f);

        std::vector<BranchNodeFull> nodes;
        nodes.reserve(num_nodes);

        nodes.emplace_back(BranchNodeFull{
                .core=BranchCore{
                        .length=length_dist(rand),
                        .abs_rot=static_cast<float>((M_PI/2) + rot_dist(rand)),
                        .pos=start_pos,
                },
                .stats=BranchStats{
                        .thickness=3.0f,
                },
        });

        for (int i = 1; i < num_nodes; i++) {
            float relative_rotation = rot_dist(rand);
            std::uniform_int_distribution<uint32_t> parent_dist(std::max(0, (i-1) / 2), i - 1);
            uint32_t parent = parent_dist(rand);

            auto& parent_node = nodes[parent];
            parent_node.nav.children.push_back(i);
            float absolute_rotation = parent_node.core.abs_rot + relative_rotation;
            glm::vec2 position = parent_node.core.pos + get_length_vec(parent_node.core);
//            float thickness = std::max(parent_node.stats.thickness * 0.9f, 1.0f);
            float thickness = parent_node.stats.thickness * 0.9f;

            nodes.emplace_back(BranchNodeFull{
                    .core=BranchCore{
                            .length=length_dist(rand),
                            .current_rel_rot=relative_rotation,
                            .target_rel_rot=relative_rotation,
                            .abs_rot=absolute_rotation,
                            .pos=position,
                            .parent=parent
                    },
                    .stats=BranchStats{
                            .energy=energy_dist(rand),
                            .thickness=thickness,
                    },
                    .nav=BranchNav{
                            .id=static_cast<uint32_t>(i)
                    }
            });

        }

        nodes[nodes.size() - 1].stats.energy = static_cast<float>(num_nodes) / 2;

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
        for (const auto& node : nodes) {
            auto& parent = nodes[node.core.parent];
            float energy = std::max(std::min(node.stats.energy, 1.0f), 0.0f);
            if (energy == 0) {
                line_renderer.add_line(parent.core.pos.x, parent.core.pos.y, node.core.pos.x, node.core.pos.y, 2, 0, glm::vec4(1, 0, 0, 1), glm::vec4(1, 0, 0, 1));
            } else {
                line_renderer.add_line(parent.core.pos.x, parent.core.pos.y, node.core.pos.x, node.core.pos.y, 2, 0, glm::vec4(1, 1, 1, energy * 0.5f), glm::vec4(1, 1, 1, energy * 0.5f));
            }
        }
    }

    void render_tree(LineRenderer &line_renderer, const TreeBatch& batch, std::default_random_engine& rand) {
        std::uniform_real_distribution<float> rand_color(0, 1);
        std::default_random_engine rand_const(1);
        for (const auto& shape : batch.tree_shapes) {
            glm::vec4 color(rand_color(rand_const), rand_color(rand_const), rand_color(rand_const), 1);
            const BranchNode* tree = &batch.trees[shape.start];
            for (auto i = shape.start; i < shape.start + shape.count; ++i) {
                const auto& node = batch.trees[i];
                color.a = std::min(1.0f, std::max(0.0f, node.stats.energy));
                const auto& parent = batch.trees[node.core.parent];
//                line_renderer.add_line(parent.core.pos.x, parent.core.pos.y, node.core.pos.x, node.core.pos.y, 2, 0, color, color);
                auto end_pos = node.core.pos + get_length_vec(node.core);
                float parent_thickness = parent.stats.thickness;
                float thickness = node.stats.thickness;
                line_renderer.add_line(node.core.pos.x, node.core.pos.y, end_pos.x, end_pos.y, parent_thickness, thickness, color, color);
            }
        }
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
            core.current_rel_rot += rot_dist(rand) * scale;
            core.abs_rot = nodes[core.parent].core.abs_rot + core.current_rel_rot;
            core.pos = nodes[core.parent].core.pos + glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.length;
        }
    }

    void mutate_len_rot(std::vector<BranchNodeFull>& nodes, std::default_random_engine& rand, float noise) {
        std::normal_distribution<float> length_dist(0.0f, noise);
        std::normal_distribution<float> rot_dist(0.0f, noise);

        for (auto& node : nodes) {
            auto& core = node.core;
            float scale = core.parent * 0.0001f;
            core.length += length_dist(rand) * scale;
            // ensure length is positive
            core.length = std::max(0.0f, core.length);
            core.current_rel_rot += rot_dist(rand) * scale;
        }
    }

    void mutate_len_rot(std::vector<BranchNode>& nodes, std::default_random_engine& rand, float noise) {
        std::normal_distribution<float> length_dist(0.0f, noise);
        std::normal_distribution<float> rot_dist(0.0f, noise);

        for (auto& node : nodes) {
            auto& core = node.core;
            float scale = core.parent * 0.0001f;
            core.length += length_dist(rand) * scale;
            // ensure length is positive
            core.length = std::max(0.0f, core.length);
            core.current_rel_rot += rot_dist(rand) * scale;
        }
    }

    void mutate_len_rot(TreeBatch &batch, std::default_random_engine &rand, float length_noise, float rot_noise) {
        std::normal_distribution<float> length_dist(0.0f, length_noise);
        std::normal_distribution<float> rot_dist(0.0f, rot_noise);

        for (auto i = 0; i < batch.trees.size(); ++i) {
            auto& core = batch.trees[i].core;
//            if (core.parent != i) {
//
//            }
            core.length += length_dist(rand);
            core.length = std::max(0.0f, core.length);
            core.current_rel_rot += rot_dist(rand);
        }
    }

    void mutate_pos(TreeBatch &batch, std::default_random_engine &rand, float noise) {
        std::normal_distribution<float> pos_dist(0.0f, noise);
        for (auto & tree : batch.trees) {
            auto& core = tree.core;
            core.pos.x += pos_dist(rand);
            core.pos.y += pos_dist(rand);
        }
    }

    void update_tree(std::vector<BranchNodeFull>& nodes) {
        for (auto i = 1; i < nodes.size(); ++i) {
            auto& core = nodes[i].core;
            core.abs_rot = nodes[core.parent].core.abs_rot + core.current_rel_rot;
            core.pos = nodes[core.parent].core.pos + glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.length;
        }
    }

    void update_tree(std::vector<BranchNode>& nodes) {
        for (auto i = 1; i < nodes.size(); ++i) {
            auto& core = nodes[i].core;
            core.abs_rot = nodes[core.parent].core.abs_rot + core.current_rel_rot;
            core.pos = nodes[core.parent].core.pos + glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.length;
        }
    }

    void update_tree(TreeBatch& batch) {
#pragma omp parallel for
        for (int j = 0; j < batch.tree_shapes.size(); ++j) {
            const auto& shape = batch.tree_shapes[j];
            for (auto i = shape.start; i < shape.start + shape.count; ++i) {
                auto& core = batch.trees[i].core;
                if (i != core.parent) {
                    core.abs_rot = batch.trees[core.parent].core.abs_rot + core.current_rel_rot;
                    core.pos = batch.trees[core.parent].core.pos + glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.length;
                }
            }
        }
    }

//    void update_tree_parallel(const TreeBatch& read_batch, TreeBatch& write_batch) {
//        for (int i = 0; i < read_batch.trees.size(); ++i) {
//            const auto& read_core = read_batch.trees[i].core;
//            const auto& read_children = read_batch.trees[i].ch;
//
//            // compute new abs_rot
//            float new_abs_rot = read_core
//
//            // compute average end pos
//            glm::vec2 avg_end = read_core.pos + get_length_vec(read_core);
//            // iterate through children, add up positions
//            for (int j = read_children.start; j < read_children.start + read_children.count; ++j) {
//                const auto& read_child_core = read_batch.trees[j].core;
//                avg_end += read_child_core.pos;
//            }
//            avg_end /= (1 + read_children.count);
//
//            // compute average start pos
//            glm::vec2 avg_start = read_core.pos;
//            if (i != read_core.parent) {
//                const auto& parent = read_batch.trees[read_core.parent];
//                avg_start += get_length_vec(parent.core) + parent.core.pos;
//                // iterate through parent's children, add their start positions
//                for (int j = parent.ch.start; j < parent.ch.start + parent.ch.count; ++j) {
//                    if (i != j) {
//                        avg_start += read_batch.trees[j].core.pos;
//                    }
//                }
//                avg_start /= (1 + parent.ch.count);
//            }
//
//            // calculate new rot
//            glm::vec2 new_vec = avg_end - avg_start;
//            float new_rot = std::atan2(new_vec.y, new_vec.x);
//
//            // we want the line to have the same length, but it has the new_rot.
//            // make the center position pass through the average between the start and end
//            // then calculate the new start position
//            glm::vec2 avg_center = (avg_start + avg_end) / 2.0f;
//            glm::vec2 new_start = avg_center - glm::vec2(std::cos(new_rot), std::sin(new_rot)) * read_core.length / 2.0f;
//
//            // update the core
//            auto& write_core = write_batch.trees[i].core;
//            write_core.abs_rot = new_rot;
//            write_core.pos = new_start;
//
//
//        }
//    }

    // PRIVATE FUNCTIONS
    void update_rot_parallel(const TreeBatch& read_batch, TreeBatch& write_batch) {
#pragma omp parallel for
        for (int i = 0; i < read_batch.trees.size(); ++i) {
            const auto& read = read_batch.trees[i];
            const auto& parent = read_batch.trees[read.core.parent];
            auto& write = write_batch.trees[i];
            if (read.core.parent != i) {
                write.core.abs_rot = read.core.current_rel_rot + parent.core.abs_rot;
            }
        }
    }

    __global__
    void update_rot_kernel(const BranchNode* read_nodes, BranchNode* write_nodes, size_t node_count) {
        auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= node_count) {
            return;
        }
        const auto& read = read_nodes[i];
        const auto& parent = read_nodes[read.core.parent];
        auto& write = write_nodes[i];
        if (read.core.parent != i) {
            write.core.abs_rot = read.core.current_rel_rot + parent.core.abs_rot;
        }
    }


    void fix_pos_parallel(const TreeBatch& read_batch, TreeBatch& write_batch) {
#pragma omp parallel for
        for (int i = 0; i < read_batch.trees.size(); ++i) {
            const auto& read = read_batch.trees[i];
            const auto& parent = read_batch.trees[read.core.parent];
            const bool has_parent = read.core.parent != i;
            auto& write = write_batch.trees[i];

            glm::vec2 avg_start_pos = read.core.pos;
            glm::vec2 avg_end_pos = read.core.pos + get_length_vec(read.core);

            // iterate through parent's children. add up their end positions then divide to get average
            if (has_parent) {
                avg_start_pos += parent.core.pos + get_length_vec(parent.core);
                for (auto j = parent.ch.start; j < parent.ch.start + parent.ch.count; ++j) {
                    const auto& parent_child = read_batch.trees[j];
                    if (j != i) {
                        avg_start_pos += parent_child.core.pos;
                    }
                }
                avg_start_pos /= (1 + parent.ch.count);
            }

            // iterate through children. add up their start positions then divide to get average
            for (auto j = read.ch.start; j < read.ch.start + read.ch.count; ++j) {
                avg_end_pos += read_batch.trees[j].core.pos;
            }
            avg_end_pos /= (1 + read.ch.count);

            // compute new angle
            const float new_angle = std::atan2(avg_end_pos.y - avg_start_pos.y, avg_end_pos.x - avg_start_pos.x);
            write.core.abs_rot = new_angle;

            // shrink line to match length
            if (has_parent) {
//                write.core.abs_rot = new_angle;
//                write.core.rel_rot = (new_angle - parent.core.abs_rot) * .5f + read.core.rel_rot * .5f;
                write.core.current_rel_rot = new_angle - parent.core.abs_rot;
                auto avg_center = (avg_start_pos + avg_end_pos) / 2.0f;
                auto new_start = avg_center - glm::vec2(std::cos(new_angle), std::sin(new_angle)) * read.core.length / 2.0f;
                write.core.pos = new_start;
            } else {
                // this is the root, so we want to keep the start position the same
                write.core.pos = read.core.pos;
            }

        }
    }

    __global__
    void fix_pos_kernel(const BranchNode* read_nodes, BranchNode* write_nodes, size_t node_count) {
        auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= node_count) {
            return;
        }
        const auto& read = read_nodes[i];
        const auto& parent = read_nodes[read.core.parent];
        const bool has_parent = read.core.parent != i;
        auto& write = write_nodes[i];

        glm::vec2 avg_start_pos = read.core.pos;
        glm::vec2 avg_end_pos = read.core.pos + get_length_vec(read.core);

        // iterate through parent's children. add up their end positions then divide to get average
        if (has_parent) {
            avg_start_pos += parent.core.pos + get_length_vec(parent.core);
            for (auto j = parent.ch.start; j < parent.ch.start + parent.ch.count; ++j) {
                const auto& parent_child = read_nodes[j];
                if (j != i) {
                    avg_start_pos += parent_child.core.pos;
                }
            }
            avg_start_pos /= (1 + parent.ch.count);
        }

        // iterate through children. add up their start positions then divide to get average
        for (auto j = read.ch.start; j < read.ch.start + read.ch.count; ++j) {
            avg_end_pos += read_nodes[j].core.pos;
        }
        avg_end_pos /= (1 + read.ch.count);

        // compute new angle
        const float new_angle = std::atan2(avg_end_pos.y - avg_start_pos.y, avg_end_pos.x - avg_start_pos.x);
        write.core.abs_rot = new_angle;

        // shrink line to match length
        if (has_parent) {
            write.core.current_rel_rot = new_angle - parent.core.abs_rot;
            auto avg_center = (avg_start_pos + avg_end_pos) / 2.0f;
            auto new_start = avg_center - glm::vec2(std::cos(new_angle), std::sin(new_angle)) * read.core.length / 2.0f;
            write.core.pos = new_start;
        } else {
            // this is the root, so we want to keep the start position the same
            write.core.pos = read.core.pos;
        }
    }

    void calc_accel_parallel(const TreeBatch& read_batch, TreeBatch& write_batch) {
        // for now, just add gravity
        for (int i = 0; i < read_batch.trees.size(); ++i) {
            auto& write = write_batch.trees[i];
            write.core.acc = glm::vec2(0, -9.8f);
        }
    }

    void integrate_accel_parallel(TreeBatch& read_batch, TreeBatch& write_batch, float dt) {
        // we want to integrate accel to vel, then vel to pos

        for (int i = 0; i < read_batch.trees.size(); ++i) {
            const auto& read = read_batch.trees[i];
            auto& write = write_batch.trees[i];
            write.core.vel = read.core.vel + read.core.acc * dt;
            write.core.pos = read.core.pos + read.core.vel * dt;
        }
    }

    void update_tree_parallel(TreeBatch& read_batch, TreeBatch& write_batch) {
        update_rot_parallel(read_batch, write_batch);
        fix_pos_parallel(write_batch, read_batch);
//        read_batch.trees = write_batch.trees;
        write_batch.trees = read_batch.trees;

    }

    void update_tree_cuda(TreeBatch& read_batch, TreeBatch& write_batch) {
        const size_t node_count = read_batch.trees.size();
        const size_t node_size = node_count * sizeof(BranchNode);
        BranchNode* d_read_nodes;
        BranchNode* d_write_nodes;
        cudaMalloc(&d_read_nodes, node_size);
        cudaMalloc(&d_write_nodes, node_size);
        cudaMemcpy(d_read_nodes, read_batch.trees.data(), node_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_write_nodes, write_batch.trees.data(), node_size, cudaMemcpyHostToDevice);

        dim3 block(256);
        dim3 grid((node_count + block.x - 1) / block.x);
        update_rot_kernel<<<grid, block>>>(d_read_nodes, d_write_nodes, node_count);
        cudaDeviceSynchronize();
        fix_pos_kernel<<<grid, block>>>(d_write_nodes, d_read_nodes, node_count);
        cudaDeviceSynchronize();

        cudaMemcpy(write_batch.trees.data(), d_read_nodes, node_size, cudaMemcpyDeviceToHost);
        cudaFree(d_read_nodes);
        cudaFree(d_write_nodes);

        read_batch.trees = write_batch.trees;


    }

//    void update_tree_parallel(TreeBatch& read_batch, TreeBatch& write_batch) {
//        for (int i = 0; i < read_batch.trees.size(); ++i) {
//            const auto& read = read_batch.trees[i];
//            bool has_parent = read.core.parent != i;
//            const auto& parent = read_batch.trees[read.core.parent];
//
//            // compute new abs_rot
//            float new_abs_rot = read.core.abs_rot;
//            if (has_parent) {
//                new_abs_rot = parent.core.abs_rot + read.core.rel_rot;
//            }
//
//            glm::vec2 new_end = read.core.pos + glm::vec2(std::cos(new_abs_rot), std::sin(new_abs_rot)) * read.core.length;
//
//            // compute average end pos
//            glm::vec2 avg_end = new_end;
//            // iterate through children, add up positions
//            for (int j = read_children.start; j < read_children.start + read_children.count; ++j) {
//                const auto& read_child_core = read_batch.trees[j].core;
//                avg_end += read_child_core.pos;
//            }
//            avg_end /= (1 + read_children.count);
//
//            // compute average start pos
//            glm::vec2 avg_start = read_core.pos;
//            if (i != read_core.parent) {
//                const auto& parent = read_batch.trees[read_core.parent];
//                avg_start += get_length_vec(parent.core) + parent.core.pos;
//                // iterate through parent's children, add their start positions
//                for (int j = parent.ch.start; j < parent.ch.start + parent.ch.count; ++j) {
//                    if (i != j) {
//                        avg_start += read_batch.trees[j].core.pos;
//                    }
//                }
//                avg_start /= (1 + parent.ch.count);
//            }
//
//            // calculate new rot
//            glm::vec2 new_vec = avg_end - avg_start;
//            float new_rot = std::atan2(new_vec.y, new_vec.x);
//
//            // we want the line to have the same length, but it has the new_rot.
//            // make the center position pass through the average between the start and end
//            // then calculate the new start position
//            glm::vec2 avg_center = (avg_start + avg_end) / 2.0f;
//            glm::vec2 new_start = avg_center - glm::vec2(std::cos(new_rot), std::sin(new_rot)) * read_core.length / 2.0f;
//
//            // update the core
//            auto& write_core = write_batch.trees[i].core;
//            write_core.abs_rot = new_rot;
//            write_core.pos = new_start;
//
//
//        }
//    }

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

    void mix_node_contents(const BranchNode read_nodes[], BranchNode write_nodes[], size_t start, size_t node_count, float interp, float total_energy) {
        for (auto i = start; i < start + node_count; ++i) {
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

        float new_total_energy = compute_total_energy(&write_nodes[start], node_count);
        float scale_factor = total_energy / new_total_energy;

        for (auto i = start; i < start + node_count; ++i) {
            auto& node = write_nodes[i];
            node.stats.energy *= scale_factor;
        }
    }

    void mix_node_contents(const std::vector<BranchNode>& read_nodes, std::vector<BranchNode>& write_nodes, float interp) {
        float total_energy = compute_total_energy(read_nodes);
        mix_node_contents(read_nodes, write_nodes, interp, total_energy);
    }

    void mix_node_contents(const TreeBatch& read_batch, TreeBatch& write_batch, float interp, const std::vector<float>& total_energies) {
        // TODO: try pragma omp parallel for
        for (int i = 0; i < read_batch.tree_shapes.size(); ++i) {
            const BranchShape& shape = read_batch.tree_shapes[i];
            const auto total_energy = total_energies[i];
            mix_node_contents(read_batch.trees.data(), write_batch.trees.data(), shape.start, shape.count, interp, total_energy);
        }
    }

    void mix_node_contents(const TreeBatch& read_batch, TreeBatch& write_batch, float interp) {
        // TODO: try pragma omp parallel for
#pragma omp parallel for
        for (int i = 0; i < read_batch.tree_shapes.size(); ++i) {
            const BranchShape& shape = read_batch.tree_shapes[i];
            const BranchNode* read_tree = &read_batch.trees[shape.start];
            const auto total_energy = compute_total_energy(read_tree, shape.count);
            mix_node_contents(read_batch.trees.data(), write_batch.trees.data(), shape.start, shape.count, interp, total_energy);
        }
    }

    float compute_total_energy(const std::vector<BranchNode>& nodes) {
        float sum = 0;
        for (auto& node : nodes) {
            sum += node.stats.energy;
        }
        return sum;
    }

    float compute_total_energy(const BranchNode nodes[], size_t node_count) {
        float sum = 0;
        for (int i = 0; i < node_count; ++i) {
            const auto& node = nodes[i];
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

    __host__ __device__
    glm::vec2 get_length_vec(const BranchCore &core) {
        return glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.length;
    }
}
