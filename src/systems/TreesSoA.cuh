#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "Trees.cuh"

namespace trees {
    struct BranchCoreAoS {
        std::vector<float> length{};
        std::vector<float> current_rel_rot{};
        std::vector<float> target_rel_rot{};
        std::vector<float> rot_vel{};
        std::vector<float> rot_acc{};
        std::vector<float> abs_rot{};
        std::vector<glm::vec2> pos{};
        std::vector<glm::vec2> vel{};
        std::vector<glm::vec2> acc{};
        std::vector<uint32_t> parent{};

        void push_back(const BranchCore& core) {
            length.push_back(core.length);
            current_rel_rot.push_back(core.current_rel_rot);
            target_rel_rot.push_back(core.target_rel_rot);
            rot_vel.push_back(core.rot_vel);
            rot_acc.push_back(core.rot_acc);
            abs_rot.push_back(core.abs_rot);
            pos.push_back(core.pos);
            vel.push_back(core.vel);
            acc.push_back(core.acc);
            parent.push_back(core.parent);
        }
    };

    struct BranchShapeAoS {
        std::vector<uint32_t> start{};
        std::vector<uint32_t> count{};

        void push_back(const BranchShape& shape) {
            start.push_back(shape.start);
            count.push_back(shape.count);
        }
    };

    struct BranchStatsAoS {
        std::vector<float> energy{};
        std::vector<float> max_energy{};
        std::vector<float> mix_rate{};
        std::vector<float> thickness{};

        void push_back(const BranchStats& stats) {
            energy.push_back(stats.energy);
            max_energy.push_back(stats.max_energy);
            mix_rate.push_back(stats.mix_rate);
            thickness.push_back(stats.thickness);
        }
    };

    struct TreeDataAoS {
        std::vector<float> total_energy{};

        void push_back(float total_energy) {
            this->total_energy.push_back(total_energy);
        }
    };

    struct TreeBatchAoS {
        BranchCoreAoS core{};
        BranchStatsAoS stats{};
        BranchShapeAoS shape{};
    };
}