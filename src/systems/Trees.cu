#include "Trees.cuh"
#include <cmath>
#include <stack>
#include <glm/glm.hpp>
#include <iostream>
#include "graphics/renderers/LineRenderer.cuh"

namespace trees {

    std::vector<BranchNodeFull> build_tree(uint32_t num_nodes, std::default_random_engine &rand, glm::vec2 start_pos) {
        std::uniform_real_distribution<float> length_dist(16.0f, 32.0f);
        std::normal_distribution<float> rot_dist(0.0f, 0.2f);
        std::uniform_real_distribution<float> energy_dist(0.1f, 1.0f);

        std::vector<BranchNodeFull> nodes;
        nodes.reserve(num_nodes);

        const float root_rotation = static_cast<float>((M_PI / 2) + rot_dist(rand));
        const float root_thickness = 4.0f;
        const float root_length = length_dist(rand);
        const float root_energy = std::pow(static_cast<float>(num_nodes), 0.8f) * 2;

        nodes.emplace_back(BranchNodeFull{
            .core = trees2::BranchCore{
                .length = root_length,
                .current_rel_rot = root_rotation,
                .target_rel_rot = root_rotation,
                .abs_rot = root_rotation,
                .pos = start_pos,
            },
            .stats = trees2::BranchStats{
                .energy = root_energy,
                .thickness = root_thickness,
                .target_thickness = root_thickness,
                .target_length = root_length,
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
            float target_thickness = parent_node.stats.target_thickness * 0.95f;

            nodes.emplace_back(BranchNodeFull{
                .core = trees2::BranchCore{
                    // .length = length_dist(rand) * thickness / root_thickness,
                    .current_rel_rot = relative_rotation,
                    .target_rel_rot = relative_rotation,
                    .abs_rot = absolute_rotation,
                    .pos = position,
                    .parent = parent,
                },
                .stats = trees2::BranchStats{
                    .energy = 0,
                    .target_thickness = target_thickness,
                    .target_length = length_dist(rand) * target_thickness / root_thickness,
                },
                .nav = BranchNav{
                    .id = static_cast<uint32_t>(i)
                }
            });
        }

        // nodes[nodes.size() - 1].stats.energy = static_cast<float>(num_nodes) / 2;

        return nodes;
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

    std::vector<trees2::BranchNode> build_tree_optimized(uint32_t num_nodes, std::default_random_engine &rand,
                                                         glm::vec2 start_pos) {
        return strip_nav(sort_tree(build_tree(num_nodes, rand, start_pos)));
    }

    TreeBatch concatenate_trees(const std::vector<Tree> &trees, const std::vector<trees2::TreeData> &tree_data) {
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
        batch.tree_data.push_back(tree_data[0]);
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
            batch.tree_data.push_back(tree_data[i]);
        }

        return batch;
    }

    trees2::TreeBatch make_batch(uint32_t node_count, uint32_t tree_count, std::default_random_engine& rand) {
        std::vector<Tree> trees;
        std::vector<trees2::TreeData> tree_data;
        constexpr auto row_size = 256;
        std::normal_distribution<float> spawn_dist(0, row_size);
        std::uniform_int_distribution<int> num_nodes_dist(node_count / 2, 3 * node_count / 2);
        for (int i = 0; i < tree_count; ++i) {
            int x = i % row_size;
            int y = i / row_size;
            uint32_t nodes = std::max(2u, (x * node_count) / (row_size/2));
            trees.push_back(build_tree_optimized(nodes, rand, glm::vec2(x * 128, y * 128)));
            //        trees.push_back(build_tree_optimized(NUM_NODES, rand, glm::vec2(spawn_dist(rand), spawn_dist(rand))));
            tree_data.emplace_back();
        }

        auto concat = concatenate_trees(trees, tree_data);
        trees2::TreeBatch batch;
        batch.tree_shapes.push_back(concat.tree_shapes);
        batch.tree_data.push_back(concat.tree_data);
        batch.trees.push_back(concat.trees);

        const auto total_nodes = batch.trees.core.abs_rot.size();

        std::vector<int> branch_reads;
        // init with zeros
        branch_reads.reserve(total_nodes);
        for (int i = 0; i < total_nodes; ++i) {
            branch_reads.push_back(0);
        }

        for (auto i = 0; i < batch.tree_shapes.count.size(); ++i) {
            auto start = batch.tree_shapes.start[i];
            for (auto j = 0; j < batch.tree_shapes.count[i]; ++j) {
                auto branch_index = start + j;
                if (branch_index >= total_nodes) {
                    std::cout << "total_nodes = " << total_nodes << std::endl;
                    std::cout << "branch_index = " << branch_index << std::endl;
                    std::cout << "start = " << start << std::endl;
                    std::cout << "count = " << batch.tree_shapes.count[i] << std::endl;
                }
                branch_reads[branch_index]++;
            }
        }

        for (auto i = 0; i < total_nodes; ++i) {
            if (branch_reads[i] != 1) {
                std::cout << "branch_reads[" << i << "] = " << branch_reads[i] << std::endl;
            }
        }

        return batch;
    }

    void render_tree(LineRenderer &line_renderer, const trees2::TreeBatch &batch, glm::mat4 transform) {
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
                // check if start or end pos is outside of the screen
                glm::vec4 start_pos_vec4 = transform * glm::vec4(start_pos, 0, 1);
                glm::vec4 end_pos_vec4 = transform * glm::vec4(end_pos, 0, 1);
                // if (start_pos_vec4.x < -1 || start_pos_vec4.x > 1 || start_pos_vec4.y < -1 || start_pos_vec4.y > 1 ||
                //     end_pos_vec4.x < -1 || end_pos_vec4.x > 1 || end_pos_vec4.y < -1 || end_pos_vec4.y > 1) {
                //     continue;
                // }
                line_renderer.add_line(start_pos, end_pos, parent_thickness, thickness, color, color);
                // line_renderer.add_line(start_pos, end_pos, parent_thickness, thickness, start_pos_vec4 * 0.5f + 0.5f, end_pos_vec4 * 0.5f + 0.5f);
            }
        }
    }

    __device__
    void add_vertex(unsigned int* line_vbo, unsigned int line_start_index, float x, float y, float tx, float ty, float length, float radius, unsigned char red,
                              unsigned char green, unsigned char blue, unsigned char alpha) {
        auto s = line_start_index * 7;
        line_vbo[s] = reinterpret_cast<unsigned int &>(x);
        line_vbo[s + 1] = reinterpret_cast<unsigned int &>(y);
        line_vbo[s + 2] = reinterpret_cast<unsigned int &>(tx);
        line_vbo[s + 3] = reinterpret_cast<unsigned int &>(ty);
        line_vbo[s + 4] = reinterpret_cast<unsigned int &>(length);
        line_vbo[s + 5] = reinterpret_cast<unsigned int &>(radius);
        unsigned int color = 0;
        color |= red;
        color |= green << 8;
        color |= blue << 16;
        color |= alpha << 24;
        line_vbo[s + 6] = color;
    }

    __device__
    void add_vertex(unsigned int* line_vbo, size_t line_start_index, float x, float y, float tx, float ty, float length, float radius, const glm::vec4 &color) {
        add_vertex(line_vbo, line_start_index, x, y, tx, ty, length, radius, color.r * 255, color.g * 255, color.b * 255, color.a * 255);
    }

    __global__
    void render_tree_kernel(unsigned int* line_vbo, const trees2::TreeBatchPtrs batch, size_t node_count) {
        const auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= node_count) {
            return;
        }
        const auto parent_id = batch.trees.core.parent[i];
        const auto energy = batch.trees.stats.energy[i];
        const auto start_pos = batch.trees.core.pos[i];
        const auto abs_rot = batch.trees.core.abs_rot[i];
        const auto length = batch.trees.core.length[i];
        auto line_dir = glm::vec2(std::cos(abs_rot), std::sin(abs_rot));
        const auto end_pos = start_pos + line_dir * length;
        const auto parent_thickness = batch.trees.stats.thickness[parent_id];
        const auto thickness = batch.trees.stats.thickness[i];
        const auto max_thickness = max(parent_thickness, thickness);
        const auto energy_coerced = min(1.0f, max(0.0f, energy));
        const auto color = glm::vec4(1-energy_coerced, 1, 1-energy_coerced, 1);

        const auto line_start_index = i * LineRenderer::VERTICES_PER_LINE;

        line_dir *= thickness;
        const auto perp_dir = glm::vec2(-line_dir.y, line_dir.x); // counter-clockwise

        // bottom left
        glm::vec2 bl = start_pos + perp_dir - line_dir;
        // bottom right
        glm::vec2 br = start_pos - perp_dir - line_dir;
        // top left
        glm::vec2 tl = end_pos + perp_dir + line_dir;
        // top right
        glm::vec2 tr = end_pos - perp_dir + line_dir;

        // tri 1
        add_vertex(line_vbo, line_start_index + 0, bl.x, bl.y, -max_thickness, -max_thickness, length, parent_thickness, color);
        add_vertex(line_vbo, line_start_index + 1, br.x, br.y, max_thickness, -max_thickness, length, parent_thickness, color);
        add_vertex(line_vbo, line_start_index + 2, tr.x, tr.y, max_thickness, max_thickness + length, length, thickness, color);
        // tri 2
        add_vertex(line_vbo, line_start_index + 3, tr.x, tr.y, max_thickness, max_thickness + length, length, thickness, color);
        add_vertex(line_vbo, line_start_index + 4, tl.x, tl.y, -max_thickness, max_thickness + length, length, thickness, color);
        add_vertex(line_vbo, line_start_index + 5, bl.x, bl.y, -max_thickness, -max_thickness, length, parent_thickness, color);

    }

    __global__
    void update_rot_kernel(const trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t node_count) {
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
        } else {
            write_abs_rot[i] = read_current_rel_rot[i]; // parent's abs_rot is considered to be 0
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

    __host__ __device__
    float smallest_angle_between(float angle1, float angle2) {
        float diff = fmodf(angle2 - angle1, 2 * M_PI);
        if (diff < -M_PI) {
            diff += 2 * M_PI;
        } else if (diff > M_PI) {
            diff -= 2 * M_PI;
        }
        return diff;
    }

    __global__
    void fix_pos_kernel(const trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t node_count, float dt_inv, float vel_interp) {
        auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= node_count) {
            return;
        }
        // TODO: manipulate velocity based on required correction and dt
        const auto read_parent = read.trees.core.parent;
        const auto read_pos = read.trees.core.pos;
        const auto read_vel = read.trees.core.vel;
        const auto read_length = read.trees.core.length;
        const auto read_abs_rot = read.trees.core.abs_rot;
        const auto read_ch_start = read.trees.ch.start;
        const auto read_ch_count = read.trees.ch.count;
        const auto read_rot_vel = read.trees.core.rot_vel;
        auto write_pos = write.trees.core.pos;
        auto write_abs_rot = write.trees.core.abs_rot;
        auto write_current_rel_rot = write.trees.core.current_rel_rot;
        auto write_vel = write.trees.core.vel;
        auto write_rot_vel = write.trees.core.rot_vel;

        const auto length = read_length[i];




        const auto parent_index = read_parent[i];
        const bool has_parent = read_parent[i] != i;

        glm::vec2 new_start_pos = read_pos[i];
        glm::vec2 new_end_pos = read_pos[i] + get_length_vec(read_abs_rot[i], length);

        if (has_parent) {
            new_start_pos = read_pos[parent_index] + get_length_vec(read_abs_rot[parent_index], read_length[parent_index]);
        }

        // iterate through children. add up their start positions then divide to get average
        const auto ch_count = read_ch_count[i];
        if (ch_count > 0) {
            new_end_pos = glm::vec2(0);
            for (auto j = read_ch_start[i]; j < read_ch_start[i] + ch_count; ++j) {
                new_end_pos += read_pos[j];
            }
            // avg_end_pos /= (1 + read_ch_count[i]);
            new_end_pos /= read_ch_count[i];
        }


        // compute new angle
        const float new_angle = std::atan2(new_end_pos.y - new_start_pos.y, new_end_pos.x - new_start_pos.x);
        // write_abs_rot[i] = new_angle;

        // compute velocity of correction
        const glm::vec2 correction_vel = (new_start_pos - read_pos[i]) * dt_inv;
        // const auto correction_normal = glm::normalize(correction_vel);
        // float vel_scalar = glm::dot(correction_normal, glm::normalize(read_vel[i]));
        // vel_scalar = tanh(vel_scalar) * .5f + 0.5f;
        // vel_scalar = 0.5f;
        // vel_scalar = std::log(1.0f + exp(vel_scalar));
        // write_vel[i] = read_vel[i] * vel_scalar;
        // write_vel[i] = read_vel[i] * vel_scalar;
        // write_vel[i] = read_vel[i] * vel_scalar;
        // TODO: vel should be limited by their normalized dot product somehow
        write_vel[i] = read_vel[i] + correction_vel * 0.1f; // TODO scale with dt
        // write_vel[i] = read_vel[i];

        // const float parent_abs_rot = has_parent ? read_abs_rot[parent_index] : 0.0f;

        write_pos[i] = new_start_pos;
        // write_current_rel_rot[i] = new_angle - read_abs_rot[parent_index];
        // write_current_rel_rot[i] = new_angle - read_abs_rot[i];
        // write_current_rel_rot[i] = read.trees.core.current_rel_rot[i];

        // const auto rot_delta = new_angle - read_abs_rot[i];
        const auto rot_delta = smallest_angle_between(read_abs_rot[i], new_angle);
        const auto rot_accel = length > 0 ? rot_delta * 10.0f : 0.0f; // TODO const rot_delta scalar

        // write_current_rel_rot[i] = new_angle - parent_abs_rot;
        write_rot_vel[i] = read_rot_vel[i] + rot_accel / dt_inv; // TODO pass in dt
    }

    __device__
    float calc_torque(const float& min_thickness, const float& rot_delta) {
        // return rot_delta * min_thickness * trees2::TORQUE_PER_RAD;
        // float constrained_rot_delta = max(-static_cast<float>(M_PI) / 2, min(static_cast<float>(M_PI) / 2, rot_delta));
        return rot_delta * pow(min_thickness, 4) * trees2::TORQUE_PER_RAD;
    }

    __global__
    void integrate_kernel_condensed(const trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t node_count, float dt) {
        const float ANGULAR_DAMPENING = 0.96f;
        const float RAD_PER_SEC_CAP = 10.0f;
        auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= node_count) {
            return;
        }
        const auto parent_id = read.trees.core.parent[i];
        const auto has_parent = parent_id != i;
        // compute torque between parent and this node
        const auto rot_delta = read.trees.core.target_rel_rot[i] - read.trees.core.current_rel_rot[i];
        const auto current_thickness = read.trees.stats.thickness[i];
        const auto min_thickness = min(current_thickness, read.trees.stats.thickness[parent_id]);
        const auto torque = calc_torque(min_thickness, rot_delta);
        // const auto mass = read.trees.stats.energy[i] * trees2::MASS_PER_ENERGY;
        const auto current_length = read.trees.core.length[i];
        const auto mass = current_length * current_thickness * current_thickness * static_cast<float>(M_PI) * trees2::MASS_PER_CUBIC_CM;
        // we are rotating around the parent, so moment of inertia is (1/3) * mass * r^2
        const auto length = read.trees.core.length[i];
        const auto parent_abs_rot = has_parent ? read.trees.core.abs_rot[parent_id] : 0.0f;
        const auto is_impossible = length == 0 || mass == 0;

        const auto moment_of_inertia = (1.0f / 3.0f) * mass * length * length;
        // angular acceleration is torque / moment of inertia
        const auto rot_acc = max(-RAD_PER_SEC_CAP, min(RAD_PER_SEC_CAP, torque / moment_of_inertia));
        // integrate angular velocity
        const auto new_rot_vel = read.trees.core.rot_vel[i] * ANGULAR_DAMPENING + (is_impossible ? 0.0f : rot_acc * dt);
        // integrate angular position
        const auto new_rel_rot = read.trees.core.current_rel_rot[i] + new_rot_vel * dt;

        write.trees.core.rot_vel[i] = new_rot_vel;
        write.trees.core.current_rel_rot[i] = new_rel_rot;
        write.trees.core.abs_rot[i] = parent_abs_rot + new_rel_rot;


        const auto& current_vel = read.trees.core.vel[i];
        // TODO: here, has_parent will be replaced with is_grounded in the future
        const auto new_vel = has_parent ? current_vel + glm::vec2(0, -98.0f * dt) : glm::vec2(0);
        write.trees.core.vel[i] = new_vel;

        const auto& current_pos = read.trees.core.pos[i];
        const glm::vec2 new_pos = current_pos + current_vel * dt;
        write.trees.core.pos[i] = new_pos;
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

    __host__ __device__
    inline void compute_tree_total_energies(trees2::TreeBatchPtrs &read, trees2::TreeBatchPtrs &write, unsigned i) {
        const auto start = read.tree_shapes.start[i];
        const auto count = read.tree_shapes.count[i];
        float total_energy = 0;
        for (auto j = start; j < start + count; ++j) {
            total_energy += read.trees.stats.energy[j];
        }
        write.tree_data.total_energy[i] = total_energy;
    }

    // writes to tree_data.total_energy
    __global__
    void compute_tree_total_energies_kernel(trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t tree_count) {
        const auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= tree_count) {
            return;
        }
        compute_tree_total_energies(read, write, i);
    }

    void compute_tree_total_energies_cpu(trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t tree_count) {
#pragma omp parallel for
        for (auto i = 0; i < tree_count; ++i) {
            compute_tree_total_energies(read, write, i);
        }
    }

    __host__ __device__
    inline void set_tree_total_energies(trees2::TreeBatchPtrs &read, trees2::TreeBatchPtrs &write, unsigned i) {
        // TODO: this function can read and write from the same memory location, or
        // it can read from read.tree_data.total_energy and write to write.tree_data.total_energy.
        // need to determine which is better

        const auto start = read.tree_shapes.start[i];
        const auto count = read.tree_shapes.count[i];
        float total_energy = 0;
        for (auto j = start; j < start + count; ++j) {
            total_energy += read.trees.stats.energy[j];
        }
        float scale = total_energy > 0 ? read.tree_data.total_energy[i] / total_energy : 1.0f;
        for (auto j = start; j < start + count; ++j) {
            write.trees.stats.energy[j] = read.trees.stats.energy[j] * scale;
        }
    }

    __global__
    void set_tree_total_energies_kernel(trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t tree_count) {
        const auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= tree_count) {
            return;
        }
        set_tree_total_energies(read, write, i);
    }

    void set_tree_total_energies_cpu(trees2::TreeBatchPtrs &read, trees2::TreeBatchPtrs &write, size_t tree_count) {
#pragma omp parallel for
        for (auto i = 0; i < tree_count; ++i) {
            set_tree_total_energies(read, write, i);
        }
    }

    __host__ __device__
    inline void mix_node_contents(trees2::TreeBatchPtrs &read, trees2::TreeBatchPtrs &write, unsigned i) {
        const float interp = 1.0f;
        const auto read_energy = read.trees.stats.energy;
        const auto write_energy = write.trees.stats.energy;

        float sum = read_energy[i];
        float weight_sum = 1.0f;

        auto parent_id = read.trees.core.parent[i];
        // if parent id equals node id, that indicates it is a root.
        // that means it doesn't have a parent so skip
        if (parent_id != i) {
            sum += read_energy[parent_id];
            weight_sum += 1.0f;
        }

        auto child_start_index = read.trees.ch.start[i];
        for (auto j = 0; j < read.trees.ch.count[i]; ++j) {
            sum += read_energy[j + child_start_index];
            weight_sum += 1.0f;
        }

        float avg_energy = sum / weight_sum;
        write_energy[i] = (1 - interp) * read_energy[i] + interp * avg_energy;
    }

    __global__
    void mix_node_contents_kernel(trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t tree_count) {
        const auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= tree_count) {
            return;
        }
        mix_node_contents(read, write, i);
    }

    void mix_node_contents_cpu(trees2::TreeBatchPtrs &read, trees2::TreeBatchPtrs &write, size_t tree_count) {
#pragma omp parallel for
        for (auto i = 0; i < tree_count; ++i) {
            mix_node_contents(read, write, i);
        }
    }

    __host__ __device__
    inline void mix_node_give_take(trees2::TreeBatchPtrs &read, trees2::TreeBatchPtrs &write, float dt, unsigned i) {
        // TODO: use per node give speed
        const float give_per_sec = read.trees.stats.energy_give_per_sec[i];

        const auto parent_id = read.trees.core.parent[i];
        const auto has_parent = parent_id != i;
        const auto energy = read.trees.stats.energy[i];


        float energy_delta = 0;
        if (has_parent) {
            const auto parent_energy = read.trees.stats.energy[parent_id];
            // try giving to parent
            if (energy > parent_energy) {
                energy_delta -= give_per_sec;
            } else if (energy < parent_energy) {
                const auto parent_give_speed = read.trees.stats.energy_give_per_sec[parent_id];
                energy_delta += parent_give_speed;
            }
        }

        // iterate through children, determine give/take amounts
        const auto child_start_index = read.trees.ch.start[i];
        const auto child_end_index = child_start_index + read.trees.ch.count[i];
        for (auto j = child_start_index; j < child_end_index; ++j) {
            const auto child_energy = read.trees.stats.energy[j];
            if (energy > child_energy) {
                energy_delta -= give_per_sec;
            } else if (energy < child_energy) {
                const auto child_give_speed = read.trees.stats.energy_give_per_sec[j];
                energy_delta += child_give_speed;
            }
        }

        write.trees.stats.energy[i] = energy + energy_delta * dt;
    }

    __global__
    void mix_node_give_take_kernel(trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t tree_count, float dt) {
        const auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= tree_count) {
            return;
        }
        mix_node_give_take(read, write, dt, i);
    }

    void mix_node_give_take_cpu(trees2::TreeBatchPtrs &read, trees2::TreeBatchPtrs &write, size_t tree_count, float dt) {
#pragma omp parallel for
        for (auto i = 0; i < tree_count; ++i) {
            mix_node_give_take(read, write, dt, i);
        }
    }

    void mix_node_contents_device_full(trees2::TreeBatchDevice &read_device, trees2::TreeBatchDevice &write_device) {
        cudaDeviceSynchronize();
        const size_t tree_count = read_device.tree_shapes.count.size();
        const size_t node_count = read_device.trees.core.abs_rot.size();
        dim3 block(256);
        dim3 tree_grid((tree_count + block.x - 1) / block.x);
        dim3 node_grid((node_count + block.x - 1) / block.x);

        // std::cout << "tree_count: " << tree_count << std::endl;
        // std::cout << "node_count: " << node_count << std::endl;
        // std::cout << "last tree shape count: " << read_device.tree_shapes.count[tree_count - 1] << std::endl;
        // std::cout << "last tree shape start: " << read_device.tree_shapes.start[tree_count - 1] << std::endl;

        // first, compute total energies
        trees2::TreeBatchPtrs read_ptrs, write_ptrs;
        read_ptrs.get_ptrs(read_device);
        write_ptrs.get_ptrs(write_device);
        compute_tree_total_energies_kernel<<<tree_grid, block>>>(read_ptrs, write_ptrs, tree_count);
        cudaDeviceSynchronize();
        // compute_tree_total_energies_kernel writes to tree_data.total_energy, so we need to swap the pointers
        write_device.tree_data.total_energy.swap(read_device.tree_data.total_energy);

        // re-aquire pointers
        read_ptrs.get_ptrs(read_device);
        write_ptrs.get_ptrs(write_device);

        // next, mix node contents
        mix_node_contents_kernel<<<node_grid, block>>>(read_ptrs, write_ptrs, node_count);
        // mix_node_give_take_kernel<<<node_grid, block>>>(read_ptrs, write_ptrs, node_count, 1/60.0f);
        cudaDeviceSynchronize();
        // mix_node_contents_kernel writes to stats.energy, so we need to swap the pointers
        write_device.trees.stats.energy.swap(read_device.trees.stats.energy);

        // re-aquire pointers
        read_ptrs.get_ptrs(read_device);
        write_ptrs.get_ptrs(write_device);

        // finally, correct total energies so that they match the original total energy
        set_tree_total_energies_kernel<<<tree_grid, block>>>(read_ptrs, write_ptrs, tree_count);
        cudaDeviceSynchronize();
        // set_tree_total_energies_kernel writes to stats.energy, so we need to swap the pointers
        write_device.trees.stats.energy.swap(read_device.trees.stats.energy);

        thrust::host_vector<float> total_energy_host = read_device.tree_data.total_energy;
        for (auto i = 0; i < 4; ++i) {
            std::cout << "total_energy_host[" << i << "]: " << total_energy_host[i] << '\n';
        }
        std::cout << std::endl;
    }

    void mix_node_contents_host_full(trees2::TreeBatch &read, trees2::TreeBatch &write) {
        const size_t tree_count = read.tree_shapes.count.size();
        const size_t node_count = read.trees.core.abs_rot.size();

        // first, compute total energies
        trees2::TreeBatchPtrs read_ptrs, write_ptrs;
        read_ptrs.get_ptrs(read);
        write_ptrs.get_ptrs(write);

        compute_tree_total_energies_cpu(read_ptrs, write_ptrs, tree_count);
        // compute_tree_total_energies_cpu writes to tree_data.total_energy, so we need to swap the pointers
        write.tree_data.total_energy.swap(read.tree_data.total_energy);

        // re-aquire pointers
        read_ptrs.get_ptrs(read);
        write_ptrs.get_ptrs(write);

        // next, mix node contents
        mix_node_contents_cpu(read_ptrs, write_ptrs, node_count);
        // mix_node_contents_cpu writes to stats.energy, so we need to swap the pointers
        write.trees.stats.energy.swap(read.trees.stats.energy);

        // re-aquire pointers
        read_ptrs.get_ptrs(read);
        write_ptrs.get_ptrs(write);

        // finally, correct total energies so that they match the original total energy
        set_tree_total_energies_cpu(read_ptrs, write_ptrs, tree_count);
        // set_tree_total_energies_cpu writes to stats.energy, so we need to swap the pointers
        write.trees.stats.energy.swap(read.trees.stats.energy);
    }

    __host__ __device__
    inline void grow(trees2::TreeBatchPtrs &read, trees2::TreeBatchPtrs &write, float dt, unsigned i) {
        const auto current_thickness = read.trees.stats.thickness[i];
        const auto current_length = read.trees.core.length[i];

        const auto target_thickness = read.trees.stats.target_thickness[i];
        const auto target_length = read.trees.stats.target_length[i];

        if (current_thickness < target_thickness || current_length < target_length) {
            // TODO: should probably make current_thickness a function of
            // (target_thickness, current_length, target_length)
            // or i just hold onto it for cacheing reasons
            const auto progress = current_length / target_length;
            // const auto surface_area = current_thickness * current_thickness * static_cast<float>(M_PI);
            const auto surface_area2 = pow(sqrt(progress) * target_thickness, 2) * static_cast<float>(M_PI);

            const auto current_mass = current_length * surface_area2 * trees2::MASS_PER_CUBIC_CM;
            const auto energy = read.trees.stats.energy[i];
            const auto growth_energy = min(read.trees.stats.growth_rate[i] * dt, max(energy, 0.0f));
            const auto new_mass = current_mass + growth_energy * trees2::MASS_PER_ENERGY;

            // TODO: solve for current_length
            // const auto new_mass = current_length * pow(sqrt(current_length / target_length) * target_thickness, 2) * static_cast<float>(M_PI) * trees2::MASS_PER_CUBIC_CM + growth_energy * trees2::MASS_PER_ENERGY;
            // const auto new_length = pow((new_mass - growth_energy * trees2::MASS_PER_ENERGY) * sqrt(target_length) / (static_cast<float>(M_PI) * trees2::MASS_PER_CUBIC_CM), 2.0f/3.0f);
            const auto new_length = sqrt((new_mass) * target_length / (static_cast<float>(M_PI) * trees2::MASS_PER_CUBIC_CM * target_thickness * target_thickness));
            const auto new_progress = new_length / target_length;
            const auto new_thickness = sqrt(new_progress) * target_thickness;
            // const auto new_length =

            write.trees.stats.energy[i] = energy - growth_energy;
            write.trees.stats.thickness[i] = new_thickness;
            write.trees.core.length[i] = new_length;

            // test
            // write.trees.stats.energy[i] = energy;
            // write.trees.stats.thickness[i] = min(current_thickness + dt, target_thickness);
            // write.trees.core.length[i] = min(current_length + dt * 5, target_length);


        } else {
            // copy over energy, thickness, length
            write.trees.stats.energy[i] = read.trees.stats.energy[i];
            write.trees.stats.thickness[i] = read.trees.stats.thickness[i];
            write.trees.core.length[i] = read.trees.core.length[i];
        }
    }

    __global__
    void grow_kernel(trees2::TreeBatchPtrs read, trees2::TreeBatchPtrs write, size_t node_count, float dt) {
        const auto i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= node_count) {
            return;
        }
        grow(read, write, dt, i);
    }

    void grow_cpu(trees2::TreeBatchPtrs &read, trees2::TreeBatchPtrs &write, size_t node_count, float dt) {
#pragma omp parallel for
        for (auto i = 0; i < node_count; ++i) {
            grow(read, write, dt, i);
        }
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

    Trees::Trees(bool use_graphics) {
        if (use_graphics) {
            line_renderer = std::make_unique<LineRenderer>();
        }
    }

    void Trees::generate_random_trees(uint32_t num_trees, uint32_t num_nodes, std::default_random_engine &rand) {
        std::cout << "make_batch" << std::endl;
        read_host = make_batch(num_nodes, num_trees, rand);
        std::cout << "copy 1" << std::endl;
        write_host = read_host;

        std::cout << "copy to device 1" << std::endl;
        read_device.copy_from_host(read_host);
        std::cout << "copy to device 2" << std::endl;
        write_device.copy_from_host(write_host);
        std::cout << "sync" << std::endl;
        cudaDeviceSynchronize();
    }

    void Trees::update(float dt) {
        update_tree_cuda(read_device, write_device);
    }

    void Trees::render(const glm::mat4 &transform) {
        // early return if we don't have a line renderer
        if (!line_renderer) return;

        line_renderer->set_transform(transform);

        const auto node_count = read_device.trees.core.abs_rot.size();
        line_renderer->ensure_vbo_capacity(node_count);
        // get a cuda compatible pointer to the vbo
        line_renderer->cuda_register_buffer();
        auto vbo_ptr = line_renderer->cuda_map_buffer();
        trees2::TreeBatchPtrs ptrs;
        ptrs.get_ptrs(read_device);

        dim3 block(256);
        dim3 grid((node_count + block.x - 1) / block.x);
        render_tree_kernel<<<grid, block>>>(static_cast<unsigned int *>(vbo_ptr), ptrs, node_count);
        cudaDeviceSynchronize();
        line_renderer->cuda_unmap_buffer();

        line_renderer->render(node_count);
        line_renderer->cuda_unregister_buffer();

        // read_device.copy_to_host(read_host);
        // line_renderer->begin();
        // render_tree(*line_renderer.get(), read_host, transform);
        // line_renderer->end();
        // line_renderer->render();

    }

        // write to write_batch_device, but swaps written vectors with read_batch_device vectors,
    // so the final updated version is stored in read_batch_device.
    // updates abs_rot, pos, current_rel_rot
    void update_tree_cuda(trees2::TreeBatchDevice &read_batch_device, trees2::TreeBatchDevice &write_batch_device) {
        // trees2::TreeBatch read_batch_host, write_batch_host;
        // read_batch_device.copy_to_host(read_batch_host);
        // write_batch_device.copy_to_host(write_batch_host);
        //
        //
        // mix_node_contents_host_full(read_batch_host, write_batch_host);
        //
        // // copy back
        // read_batch_device.copy_from_host(read_batch_host);
        // write_batch_device.copy_from_host(write_batch_host);

        mix_node_contents_device_full(read_batch_device, write_batch_device);

        const size_t node_count = read_batch_device.trees.core.abs_rot.size();

        dim3 block(256);
        dim3 grid((node_count + block.x - 1) / block.x);

        trees2::TreeBatchPtrs read_batch_ptrs, write_batch_ptrs;
        read_batch_ptrs.get_ptrs(read_batch_device);
        write_batch_ptrs.get_ptrs(write_batch_device);

        update_rot_kernel<<<grid, block>>>(read_batch_ptrs, write_batch_ptrs, node_count);
        cudaDeviceSynchronize();

        // we just wrote to write_batch_device's abs_rot, so we need to swap the pointers and re-acquire ptrs
        write_batch_device.trees.core.abs_rot.swap(read_batch_device.trees.core.abs_rot);
        read_batch_ptrs.get_ptrs(read_batch_device);
        write_batch_ptrs.get_ptrs(write_batch_device);

        integrate_kernel_condensed<<<grid, block>>>(read_batch_ptrs, write_batch_ptrs, node_count, 1/60.0f);
        cudaDeviceSynchronize();

        // we just write to pos, vel, abs_rot, rot_vel, current_rel_rot, so we need to swap the pointers
        write_batch_device.trees.core.pos.swap(read_batch_device.trees.core.pos);
        write_batch_device.trees.core.vel.swap(read_batch_device.trees.core.vel);
        write_batch_device.trees.core.abs_rot.swap(read_batch_device.trees.core.abs_rot);
        write_batch_device.trees.core.rot_vel.swap(read_batch_device.trees.core.rot_vel);
        write_batch_device.trees.core.current_rel_rot.swap(read_batch_device.trees.core.current_rel_rot);
        read_batch_ptrs.get_ptrs(read_batch_device);
        write_batch_ptrs.get_ptrs(write_batch_device);

        fix_pos_kernel<<<grid, block>>>(read_batch_ptrs, write_batch_ptrs, node_count, 60.0f, 1.0f);
        cudaDeviceSynchronize();

        // we just wrote to write_batch_device's pos, abs_rot, current_rel_rot, vel, and rot_vel, so we need to swap the pointers
        write_batch_device.trees.core.pos.swap(read_batch_device.trees.core.pos);
        // write_batch_device.trees.core.abs_rot.swap(read_batch_device.trees.core.abs_rot);
        // write_batch_device.trees.core.current_rel_rot.swap(read_batch_device.trees.core.current_rel_rot);
        write_batch_device.trees.core.vel.swap(read_batch_device.trees.core.vel);
        write_batch_device.trees.core.rot_vel.swap(read_batch_device.trees.core.rot_vel);
        read_batch_ptrs.get_ptrs(read_batch_device);
        write_batch_ptrs.get_ptrs(write_batch_device);

        grow_kernel<<<grid, block>>>(read_batch_ptrs, write_batch_ptrs, node_count, 1/60.0f);
        cudaDeviceSynchronize();

        // we just wrote to write_batch_device's energy, thickness, and length, so we need to swap the pointers
        write_batch_device.trees.stats.energy.swap(read_batch_device.trees.stats.energy);
        write_batch_device.trees.stats.thickness.swap(read_batch_device.trees.stats.thickness);
        write_batch_device.trees.core.length.swap(read_batch_device.trees.core.length);
    }





}
