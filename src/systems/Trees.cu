#include "Trees.cuh"
#include <cmath>
#include <stack>
#include <glm/glm.hpp>
#include <iostream>
#include "graphics/renderers/LineRenderer.h"

namespace trees {
    std::vector<BranchNodeFull> build_tree(uint32_t num_nodes, std::default_random_engine &rand, glm::vec2 start_pos) {
        std::uniform_real_distribution<float> length_dist(16.0f, 32.0f);
        std::normal_distribution<float> rot_dist(0.0f, 0.2f);
        std::uniform_real_distribution<float> energy_dist(0.0f, 1.0f);

        std::vector<BranchNodeFull> nodes;
        nodes.reserve(num_nodes);

        nodes.emplace_back(BranchNodeFull{
            .core = trees2::BranchCore{
                .length = length_dist(rand),
                .abs_rot = static_cast<float>((M_PI / 2) + rot_dist(rand)),
                .pos = start_pos,
            },
            .stats = trees2::BranchStats{
                .thickness = 3.0f,
            },
        });

        for (int i = 1; i < num_nodes; i++) {
            float relative_rotation = rot_dist(rand);
            std::uniform_int_distribution<uint32_t> parent_dist(std::max(0, (i - 1) / 2), i - 1);
            uint32_t parent = parent_dist(rand);

            auto &parent_node = nodes[parent];
            parent_node.nav.children.push_back(i);
            float absolute_rotation = parent_node.core.abs_rot + relative_rotation;
            glm::vec2 position = parent_node.core.pos + get_length_vec(parent_node.core);
            //            float thickness = std::max(parent_node.stats.thickness * 0.9f, 1.0f);
            float thickness = parent_node.stats.thickness * 0.9f;

            nodes.emplace_back(BranchNodeFull{
                .core = trees2::BranchCore{
                    .length = length_dist(rand),
                    .current_rel_rot = relative_rotation,
                    .target_rel_rot = relative_rotation,
                    .abs_rot = absolute_rotation,
                    .pos = position,
                    .parent = parent
                },
                .stats = trees2::BranchStats{
                    .energy = energy_dist(rand),
                    .thickness = thickness,
                },
                .nav = BranchNav{
                    .id = static_cast<uint32_t>(i)
                }
            });
        }

        nodes[nodes.size() - 1].stats.energy = static_cast<float>(num_nodes) / 2;

        return nodes;
    }

    std::vector<trees2::BranchNode> build_tree_optimized(uint32_t num_nodes, std::default_random_engine &rand,
                                                         glm::vec2 start_pos) {
        return strip_nav(sort_tree(build_tree(num_nodes, rand, start_pos)));
    }

    TreeBatch concatenate_trees(const std::vector<Tree> &trees) {
        TreeBatch batch{};
        // reserve the exact amount of mem needed to store the concatenated tree
        uint32_t total_size = 0;
        for (const auto &tree: trees) {
            total_size += tree.size();
        }
        batch.trees.reserve(total_size);

        // first part of the tree batch is just the first tree
        batch.tree_shapes.emplace_back(trees2::BranchShape{
            .start = 0,
            .count = static_cast<uint32_t>(trees[0].size()),
        });
        batch.trees = trees[0];
        for (int i = 1; i < trees.size(); ++i) {
            auto tree = trees[i];
            const auto offset = batch.tree_shapes.back().start + batch.tree_shapes.back().count;
            // offset parent and children start fields
            for (auto &node: tree) {
                node.core.parent += offset;
                node.ch.start += offset;
            }
            // insert a shape for this new tree
            batch.tree_shapes.emplace_back(trees2::BranchShape{
                .start = offset,
                .count = static_cast<uint32_t>(tree.size()),
            });
            // append the tree to the batch
            batch.trees.insert(batch.trees.end(), tree.begin(), tree.end());
        }

        return batch;
    }

    void render_tree(LineRenderer &line_renderer, const trees2::TreeBatch &batch, std::default_random_engine &rand) {
        std::uniform_real_distribution<float> rand_color(0, 1);
        std::default_random_engine rand_const(1);
        const auto &core = batch.trees.core;
        const auto &stats = batch.trees.stats;
        for (auto shape_index = 0; shape_index < batch.tree_shapes.count.size(); ++shape_index) {
            glm::vec4 color(rand_color(rand_const), rand_color(rand_const), rand_color(rand_const), 1);
            auto start = batch.tree_shapes.start[shape_index];
            auto count = batch.tree_shapes.count[shape_index];
            for (auto node_id = start; node_id < start + count; ++node_id) {
                auto energy = stats.energy[node_id];
                color.a = std::min(1.0f, std::max(0.0f, energy));
                auto parent_id = batch.trees.core.parent[node_id];
                auto start_pos = core.pos[node_id];
                auto end_pos = start_pos + glm::vec2(std::cos(core.abs_rot[node_id]), std::sin(core.abs_rot[node_id])) *
                               core.length[node_id];
                auto parent_thickness = stats.thickness[parent_id];
                auto thickness = stats.thickness[node_id];
                line_renderer.add_line(start_pos, end_pos, parent_thickness, thickness, color, color);
            }
        }
    }

    void mutate_len_rot(trees2::TreeBatch &batch, std::default_random_engine &rand, float length_noise, float rot_noise) {
        std::normal_distribution<float> length_dist(0.0f, length_noise);
        std::normal_distribution<float> rot_dist(0.0f, rot_noise);

        for (auto i = 0; i < batch.trees.core.abs_rot.size(); ++i) {
            auto& length = batch.trees.core.length[i];
            length += length_dist(rand);
            length = std::max(0.0f, length);
            batch.trees.core.current_rel_rot[i] += rot_dist(rand);
        }
    }

    void mutate_pos(trees2::TreeBatch &batch, std::default_random_engine &rand, float noise) {
        std::normal_distribution<float> pos_dist(0.0f, noise);
        for (auto i = 0; i < batch.trees.core.pos.size(); ++i) {
            batch.trees.core.pos[i].x += pos_dist(rand);
            batch.trees.core.pos[i].y += pos_dist(rand);
        }
    }

    void update_tree(std::vector<BranchNodeFull> &nodes) {
        for (auto i = 1; i < nodes.size(); ++i) {
            auto &core = nodes[i].core;
            core.abs_rot = nodes[core.parent].core.abs_rot + core.current_rel_rot;
            core.pos = nodes[core.parent].core.pos + glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.
                       length;
        }
    }

    void update_tree(std::vector<trees2::BranchNode> &nodes) {
        for (auto i = 1; i < nodes.size(); ++i) {
            auto &core = nodes[i].core;
            core.abs_rot = nodes[core.parent].core.abs_rot + core.current_rel_rot;
            core.pos = nodes[core.parent].core.pos + glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.
                       length;
        }
    }

    void update_tree(TreeBatch &batch) {
#pragma omp parallel for
        for (int j = 0; j < batch.tree_shapes.size(); ++j) {
            const auto &shape = batch.tree_shapes[j];
            for (auto i = shape.start; i < shape.start + shape.count; ++i) {
                auto &core = batch.trees[i].core;
                if (i != core.parent) {
                    core.abs_rot = batch.trees[core.parent].core.abs_rot + core.current_rel_rot;
                    core.pos = batch.trees[core.parent].core.pos + glm::vec2(
                                   std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.length;
                }
            }
        }
    }

    // PRIVATE FUNCTIONS
    void update_rot_parallel(const TreeBatch &read_batch, TreeBatch &write_batch) {
#pragma omp parallel for
        for (int i = 0; i < read_batch.trees.size(); ++i) {
            const auto &read = read_batch.trees[i];
            const auto &parent = read_batch.trees[read.core.parent];
            auto &write = write_batch.trees[i];
            if (read.core.parent != i) {
                write.core.abs_rot = read.core.current_rel_rot + parent.core.abs_rot;
            }
        }
    }

    void update_rot_parallel(const trees2::TreeBatch &read_batch, trees2::TreeBatch &write_batch) {
        for (auto i = 0; i < read_batch.trees.core.abs_rot.size(); ++i) {
            auto parent_id = read_batch.trees.core.parent[i];
            if (parent_id != i) {
                write_batch.trees.core.abs_rot[i] =
                        read_batch.trees.core.current_rel_rot[i] + read_batch.trees.core.abs_rot[parent_id];
            }
        }
    }

    __global__
    void update_rot_kernel(const trees2::BranchNode *read_nodes, trees2::BranchNode *write_nodes, size_t node_count) {
        auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= node_count) {
            return;
        }
        const auto &read = read_nodes[i];
        const auto &parent = read_nodes[read.core.parent];
        auto &write = write_nodes[i];
        if (read.core.parent != i) {
            write.core.abs_rot = read.core.current_rel_rot + parent.core.abs_rot;
        }
    }

    __global__
    void update_rot_kernel(const trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t node_count) {
        // TODO: try passing by reference
        const auto read_parent = read.trees.core.parent;
        const auto read_current_rel_rot = read.trees.core.current_rel_rot;
        const auto read_abs_rot = read.trees.core.abs_rot;
        auto write_abs_rot = write.trees.core.abs_rot;

        auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= node_count) {
            return;
        }
        auto parent_id = read_parent[i];
        if (parent_id != i) {
            write_abs_rot[i] = read_current_rel_rot[i] + read_abs_rot[parent_id];
        }
    }


    void fix_pos_parallel(const TreeBatch &read_batch, TreeBatch &write_batch) {
#pragma omp parallel for
        for (int i = 0; i < read_batch.trees.size(); ++i) {
            const auto &read = read_batch.trees[i];
            const auto &parent = read_batch.trees[read.core.parent];
            const bool has_parent = read.core.parent != i;
            auto &write = write_batch.trees[i];

            glm::vec2 avg_start_pos = read.core.pos;
            glm::vec2 avg_end_pos = read.core.pos + get_length_vec(read.core);

            // iterate through parent's children. add up their end positions then divide to get average
            if (has_parent) {
                avg_start_pos += parent.core.pos + get_length_vec(parent.core);
                for (auto j = parent.ch.start; j < parent.ch.start + parent.ch.count; ++j) {
                    const auto &parent_child = read_batch.trees[j];
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
                write.core.current_rel_rot = new_angle - parent.core.abs_rot;
                auto avg_center = (avg_start_pos + avg_end_pos) / 2.0f;
                auto new_start = avg_center - glm::vec2(std::cos(new_angle), std::sin(new_angle)) * read.core.length /
                                 2.0f;
                write.core.pos = new_start;
            } else {
                // this is the root, so we want to keep the start position the same
                write.core.pos = read.core.pos;
            }
        }
    }

// TODO: this aos version is untested code written by claude
    void fix_pos_parallel(const trees2::TreeBatch &read_batch, trees2::TreeBatch &write_batch) {
#pragma omp parallel for
        for (int i = 0; i < read_batch.trees.core.abs_rot.size(); ++i) {
            const auto &read_core = read_batch.trees.core;
            const auto &parent_core = read_batch.trees.core;
            const auto &read_ch = read_batch.trees.ch;
            const bool has_parent = read_core.parent[i] != i;
            auto &write_core = write_batch.trees.core;

            glm::vec2 avg_start_pos = read_core.pos[i];
            glm::vec2 avg_end_pos = read_core.pos[i] + get_length_vec(read_core, i);

            // iterate through parent's children. add up their end positions then divide to get average
            if (has_parent) {
                const auto parent_index = read_core.parent[i];
                avg_start_pos += parent_core.pos[parent_index] + get_length_vec(parent_core, parent_index);
                for (auto j = read_ch.start[parent_index]; j < read_ch.start[parent_index] + read_ch.count[parent_index]; ++j) {
                    if (j != i) {
                        avg_start_pos += read_core.pos[j];
                    }
                }
                avg_start_pos /= (1 + read_ch.count[parent_index]);
            }

            // iterate through children. add up their start positions then divide to get average
            for (auto j = read_ch.start[i]; j < read_ch.start[i] + read_ch.count[i]; ++j) {
                avg_end_pos += read_core.pos[j];
            }
            avg_end_pos /= (1 + read_ch.count[i]);

            // compute new angle
            const float new_angle = std::atan2(avg_end_pos.y - avg_start_pos.y, avg_end_pos.x - avg_start_pos.x);
            write_core.abs_rot[i] = new_angle;

            // shrink line to match length
            if (has_parent) {
                const auto parent_index = read_core.parent[i];
                write_core.current_rel_rot[i] = new_angle - parent_core.abs_rot[parent_index];
                auto avg_center = (avg_start_pos + avg_end_pos) / 2.0f;
                auto new_start = avg_center - glm::vec2(std::cos(new_angle), std::sin(new_angle)) * read_core.length[i] / 2.0f;
                write_core.pos[i] = new_start;
            } else {
                // this is the root, so we want to keep the start position the same
                write_core.pos[i] = read_core.pos[i];
            }
        }
    }

    __global__
    void fix_pos_kernel(const trees2::BranchNode *read_nodes, trees2::BranchNode *write_nodes, size_t node_count) {
        auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= node_count) {
            return;
        }
        const auto &read = read_nodes[i];
        const auto &parent = read_nodes[read.core.parent];
        const bool has_parent = read.core.parent != i;
        auto &write = write_nodes[i];

        glm::vec2 avg_start_pos = read.core.pos;
        glm::vec2 avg_end_pos = read.core.pos + get_length_vec(read.core);

        // iterate through parent's children. add up their end positions then divide to get average
        if (has_parent) {
            avg_start_pos += parent.core.pos + get_length_vec(parent.core);
            for (auto j = parent.ch.start; j < parent.ch.start + parent.ch.count; ++j) {
                const auto &parent_child = read_nodes[j];
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

    __global__
    void fix_pos_kernel(const trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t node_count) {
        auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= node_count) {
            return;
        }
        // TODO: manipulate velocity based on required correction and dt
        const auto read_parent = read.trees.core.parent;
        const auto read_pos = read.trees.core.pos;
        const auto read_length = read.trees.core.length;
        const auto read_abs_rot = read.trees.core.abs_rot;
        const auto read_ch_start = read.trees.ch.start;
        const auto read_ch_count = read.trees.ch.count;
        auto write_pos = write.trees.core.pos;
        auto write_abs_rot = write.trees.core.abs_rot;
        auto write_current_rel_rot = write.trees.core.current_rel_rot;

        const bool has_parent = read_parent[i] != i;

        glm::vec2 avg_start_pos = read_pos[i];
        glm::vec2 avg_end_pos = read_pos[i] + get_length_vec(read_abs_rot[i], read_length[i]);

        // iterate through parent's children. add up their end positions then divide to get average
        if (has_parent) {
            const auto parent_index = read_parent[i];
            avg_start_pos += read_pos[parent_index] + get_length_vec(read_abs_rot[parent_index], read_length[parent_index]);
            for (auto j = read_ch_start[parent_index]; j < read_ch_start[parent_index] + read_ch_count[parent_index]; ++j) {
                if (j != i) {
                    avg_start_pos += read_pos[j];
                }
            }
            avg_start_pos /= (1 + read_ch_count[parent_index]);
        }

        // iterate through children. add up their start positions then divide to get average
        for (auto j = read_ch_start[i]; j < read_ch_start[i] + read_ch_count[i]; ++j) {
            avg_end_pos += read_pos[j];
        }
        avg_end_pos /= (1 + read_ch_count[i]);

        // compute new angle
        const float new_angle = std::atan2(avg_end_pos.y - avg_start_pos.y, avg_end_pos.x - avg_start_pos.x);
        write_abs_rot[i] = new_angle;

        // shrink line to match length
        if (has_parent) {
            const auto parent_index = read_parent[i];
            write_current_rel_rot[i] = new_angle - read_abs_rot[parent_index];
            auto avg_center = (avg_start_pos + avg_end_pos) / 2.0f;
            auto new_start = avg_center - glm::vec2(std::cos(new_angle), std::sin(new_angle)) * read_length[i] / 2.0f;
            write_pos[i] = new_start;
        } else {
            // this is the root, so we want to keep the start position the same
            write_pos[i] = read_pos[i];
        }
    }

    __global__
    void integrate_kernel(const trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t node_count, float dt) {
        auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= node_count) {
            return;
        }
        const auto parent_id = read.trees.core.parent[i];
        if (parent_id == i) {
            // there is no parent, so copy over position and set velocity to zero
            write.trees.core.pos[i] = read.trees.core.pos[i];
            write.trees.core.vel[i] = glm::vec2(0);
            // copy over angular position and velocity
            // TODO: double check which ones we need to copy. mem writes should match the angular integration below
            write.trees.core.abs_rot[i] = read.trees.core.abs_rot[i];
            // write.trees.core.current_rel_rot[i] = read.trees.core.current_rel_rot[i];
            write.trees.core.rot_vel[i] = read.trees.core.rot_vel[i];
        } else {
            // TODO: compute torque first. also, i don't think i need to store acceleration
            const auto& current_vel = read.trees.core.vel[i];
            const auto new_vel = current_vel + glm::vec2(0, -9.8f * dt);
            write.trees.core.vel[i] = new_vel;

            const auto& current_pos = read.trees.core.pos[i];
            const glm::vec2 new_pos = current_pos + current_vel * dt;
            write.trees.core.pos[i] = new_pos;

            // repeat for angular TODO
            const auto& current_rot_vel = read.trees.core.rot_vel[i];
            const auto new_rot_vel = current_rot_vel + 0.0f * dt;
            write.trees.core.rot_vel[i] = new_rot_vel;

            const auto& current_abs_rot = read.trees.core.abs_rot[i];
            const auto new_abs_rot = current_abs_rot + current_rot_vel * dt;
            write.trees.core.abs_rot[i] = new_abs_rot;
        }
    }

    void calc_accel_parallel(const TreeBatch &read_batch, TreeBatch &write_batch) {
        // for now, just add gravity
        for (int i = 0; i < read_batch.trees.size(); ++i) {
            auto &write = write_batch.trees[i];
            write.core.acc = glm::vec2(0, -9.8f);
        }
    }

    void integrate_accel_parallel(TreeBatch &read_batch, TreeBatch &write_batch, float dt) {
        // we want to integrate accel to vel, then vel to pos

        for (int i = 0; i < read_batch.trees.size(); ++i) {
            const auto &read = read_batch.trees[i];
            auto &write = write_batch.trees[i];
            write.core.vel = read.core.vel + read.core.acc * dt;
            write.core.pos = read.core.pos + read.core.vel * dt;
        }
    }

    void update_tree_parallel(trees2::TreeBatch &read_batch, trees2::TreeBatch &write_batch) {
        update_rot_parallel(read_batch, write_batch);
        fix_pos_parallel(write_batch, read_batch);
        //        read_batch.trees = write_batch.trees;
        write_batch.trees = read_batch.trees;
    }

    // write to write_batch_device, but swaps written vectors with read_batch_device vectors,
    // so the final updated version is stored in read_batch_device.
    // updates abs_rot, pos, current_rel_rot
    void update_tree_cuda(trees2::TreeBatchDevice &read_batch_device, trees2::TreeBatchDevice &write_batch_device) {
        const size_t node_count = read_batch_device.trees.core.abs_rot.size();

        dim3 block(256);
        dim3 grid((node_count + block.x - 1) / block.x);

        // const trees2::bid_t* d_read_parent;
        // const float* d_read_current_rel_rot;
        // const float* d_read_abs_rot;
        // float* d_write_abs_rot;
        //
        // d_read_parent = read_batch_device.trees.core.parent.data().get();
        // d_read_current_rel_rot = read_batch_device.trees.core.current_rel_rot.data().get();
        // d_read_abs_rot = read_batch_device.trees.core.abs_rot.data().get();
        // d_write_abs_rot = write_batch_device.trees.core.abs_rot.data().get();

        trees2::TreeBatchPtrs read_batch_ptrs, write_batch_ptrs;
        read_batch_ptrs.get_ptrs(read_batch_device);
        write_batch_ptrs.get_ptrs(write_batch_device);

        update_rot_kernel<<<grid, block>>>(read_batch_ptrs, write_batch_ptrs, node_count);
        cudaDeviceSynchronize();

        // we just wrote to write_batch_device's abs_rot, so we need to swap the pointers and re-acquire ptrs
        write_batch_device.trees.core.abs_rot.swap(read_batch_device.trees.core.abs_rot);
        read_batch_ptrs.get_ptrs(read_batch_device);
        write_batch_ptrs.get_ptrs(write_batch_device);

        integrate_kernel<<<grid, block>>>(read_batch_ptrs, write_batch_ptrs, node_count, 1/60.0f);
        cudaDeviceSynchronize();

        // we just write to pos, vel, abs_rot, and rot_vel, so we need to swap the pointers
        write_batch_device.trees.core.pos.swap(read_batch_device.trees.core.pos);
        write_batch_device.trees.core.vel.swap(read_batch_device.trees.core.vel);
        write_batch_device.trees.core.abs_rot.swap(read_batch_device.trees.core.abs_rot);
        write_batch_device.trees.core.rot_vel.swap(read_batch_device.trees.core.rot_vel);
        read_batch_ptrs.get_ptrs(read_batch_device);
        write_batch_ptrs.get_ptrs(write_batch_device);

        fix_pos_kernel<<<grid, block>>>(read_batch_ptrs, write_batch_ptrs, node_count);
        cudaDeviceSynchronize();

        // we just wrote to write_batch_device's pos, abs_rot, and current_rel_rot, so we need to swap the pointers
        write_batch_device.trees.core.pos.swap(read_batch_device.trees.core.pos);
        write_batch_device.trees.core.abs_rot.swap(read_batch_device.trees.core.abs_rot);
        write_batch_device.trees.core.current_rel_rot.swap(read_batch_device.trees.core.current_rel_rot);



    }


    std::vector<BranchNodeFull> sort_tree(const std::vector<BranchNodeFull> &nodes) {
        std::vector<BranchNodeFull> sorted_tree;
        sorted_tree.push_back(nodes[0]);
        // add the children uhhhhhhhhh
        for (int i = 0; i < nodes.size(); ++i) {
            auto &node = sorted_tree[i];
            for (auto child_id: node.nav.children) {
                auto new_child = nodes[child_id];
                auto new_child_id = sorted_tree.size();
                new_child.nav.id = new_child_id;
                new_child.core.parent = i;
                sorted_tree.push_back(new_child);
            }
        }

        // recalculate children
        for (auto &node: sorted_tree) {
            node.nav.children.clear();
        }

        for (int i = 1; i < sorted_tree.size(); ++i) {
            auto &node = sorted_tree[i];
            auto &parent = sorted_tree[node.core.parent];
            parent.nav.children.push_back(node.nav.id);
        }

        return sorted_tree;
    }

    std::vector<trees2::BranchNode> strip_nav(const std::vector<BranchNodeFull> &nodes) {
        std::vector<trees2::BranchNode> stripped;
        stripped.reserve(nodes.size());
        for (const auto &node: nodes) {
            uint32_t num_children = node.nav.children.size();
            uint32_t children_start = 0;
            if (num_children > 0) {
                children_start = node.nav.children[0];
            }
            stripped.push_back(trees2::BranchNode{
                .core = node.core,
                .stats = node.stats,
                .ch = trees2::BranchShape{
                    .start = children_start,
                    .count = num_children
                }
            });
        }
        return stripped;
    }

    void mix_node_contents(const trees2::BranchNode read_nodes[], trees2::BranchNode write_nodes[], size_t start,
                           size_t node_count, float interp, float total_energy) {
        for (auto i = start; i < start + node_count; ++i) {
            auto &read_node = read_nodes[i];
            auto &write_node = write_nodes[i];

            float sum = read_node.stats.energy;
            float weight_sum = 1.0f;

            // if parent id equals node id, that indicates it is a root
            if (read_node.core.parent != i) {
                sum += read_nodes[read_node.core.parent].stats.energy;
                weight_sum += 1.0f;
            }

            auto child_start_index = read_node.ch.start;
            for (uint32_t j = 0; j < read_node.ch.count; ++j) {
                auto &child = read_nodes[j + child_start_index];
                sum += child.stats.energy;
                weight_sum += 1.0f;
            }

            float avg_energy = sum / weight_sum;
            write_node.stats.energy = (1 - interp) * read_nodes[i].stats.energy + interp * avg_energy;
        }

        float new_total_energy = compute_total_energy(&write_nodes[start], node_count);
        float scale_factor = total_energy / new_total_energy;

        for (auto i = start; i < start + node_count; ++i) {
            auto &node = write_nodes[i];
            node.stats.energy *= scale_factor;
        }
    }

    void mix_node_contents(const trees2::TreeBatch &read_batch, trees2::TreeBatch &write_batch,
        size_t start, size_t node_count,float interp, float total_energy) {
        const auto& read_energy = read_batch.trees.stats.energy;
        auto& write_energy = write_batch.trees.stats.energy;

        for (auto i = start; i < start + node_count; ++i) {
            float sum = read_energy[i];
            float weight_sum = 1.0f;

            // if parent id equals node id, that indicates it is a root.
            // that means it doesn't have a parent so skip
            auto parent_id = read_batch.trees.core.parent[i];
            if (parent_id != i) {
                sum += read_energy[parent_id];
                weight_sum += 1.0f;
            }

            auto child_start_index = read_batch.trees.ch.start[i];
            for (auto j = 0; j < read_batch.trees.ch.count[i]; ++j) {
                sum += read_energy[j + child_start_index];
                weight_sum += 1.0f;
            }

            float avg_energy = sum / weight_sum;
            write_energy[i] = (1 - interp) * read_energy[i] + interp * avg_energy;
        }

        float new_total_energy = compute_total_energy(&write_energy[start], node_count);
        float scale_factor = total_energy / new_total_energy;

        for (auto i = start; i < start + node_count; ++i) {
            write_energy[i] *= scale_factor;
        }
    }

    void mix_node_contents(const trees2::TreeBatch &read_batch, trees2::TreeBatch &write_batch, float interp) {
#pragma omp parallel for
        for (int i = 0; i < read_batch.tree_shapes.start.size(); ++i) {
            const auto start = read_batch.tree_shapes.start[i];
            const auto count = read_batch.tree_shapes.count[i];
            const float *energies = &read_batch.trees.stats.energy[start];
            const auto total_energy = compute_total_energy(energies, count);
            mix_node_contents(read_batch, write_batch, start, count, interp, total_energy);
        }
    }



    void mix_node_contents(const TreeBatch &read_batch, TreeBatch &write_batch, float interp,
                           const std::vector<float> &total_energies) {
        // TODO: try pragma omp parallel for
        for (int i = 0; i < read_batch.tree_shapes.size(); ++i) {
            const trees2::BranchShape &shape = read_batch.tree_shapes[i];
            const auto total_energy = total_energies[i];
            mix_node_contents(read_batch.trees.data(), write_batch.trees.data(), shape.start, shape.count, interp,
                              total_energy);
        }
    }

    void mix_node_contents(const TreeBatch &read_batch, TreeBatch &write_batch, float interp) {
        // TODO: try pragma omp parallel for
#pragma omp parallel for
        for (int i = 0; i < read_batch.tree_shapes.size(); ++i) {
            const trees2::BranchShape &shape = read_batch.tree_shapes[i];
            const trees2::BranchNode *read_tree = &read_batch.trees[shape.start];
            const auto total_energy = compute_total_energy(read_tree, shape.count);
            mix_node_contents(read_batch.trees.data(), write_batch.trees.data(), shape.start, shape.count, interp,
                              total_energy);
        }
    }

    float compute_total_energy(const std::vector<trees2::BranchNode> &nodes) {
        float sum = 0;
        for (auto &node: nodes) {
            sum += node.stats.energy;
        }
        return sum;
    }

    float compute_total_energy(const trees2::BranchNode nodes[], size_t node_count) {
        float sum = 0;
        for (int i = 0; i < node_count; ++i) {
            const auto &node = nodes[i];
            sum += node.stats.energy;
        }
        return sum;
    }

    float compute_total_energy(const float energy[], size_t count) {
        float sum = 0;
        for (auto i = 0; i < count; ++i) {
            sum += energy[i];
        }
        return sum;
    }


    float get_min_energy(const std::vector<trees2::BranchNode> &nodes) {
        float min = std::numeric_limits<float>::max();
        // skip the first node
        for (auto i = 1; i < nodes.size(); ++i) {
            if (nodes[i].stats.energy < min) {
                min = nodes[i].stats.energy;
            }
        }
        return min;
    }

    float get_max_energy(const std::vector<trees2::BranchNode> &nodes) {
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
    glm::vec2 get_length_vec(const trees2::BranchCore &core) {
        return glm::vec2(std::cos(core.abs_rot), std::sin(core.abs_rot)) * core.length;
    }

   __host__ __device__
    glm::vec2 get_length_vec(const trees2::BranchCoreSoA &core, trees2::bid_t i) {
        return glm::vec2(std::cos(core.abs_rot[i]), std::sin(core.abs_rot[i])) * core.length[i];
    }

    __host__ __device__
    glm::vec2 get_length_vec(float abs_rot, float length) {
        return glm::vec2(std::cos(abs_rot), std::sin(abs_rot)) * length;
    }
}
