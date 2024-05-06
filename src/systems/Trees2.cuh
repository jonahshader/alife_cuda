#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>

namespace trees2 {
    using bid_t = uint32_t;

#define FOR_BRANCH_CORE(N, D) \
    D(float, length, 0)       \
    D(float, current_rel_rot, 0) \
    D(float, target_rel_rot, 0)  \
    D(float, rot_vel, 0)      \
    D(float, rot_acc, 0)      \
    D(float, abs_rot, 0)      \
    N(glm::vec2, pos)         \
    N(glm::vec2, vel)         \
    N(glm::vec2, acc)         \
    D(bid_t, parent, 0)

#define FOR_BRANCH_SHAPE(N, D) \
    D(bid_t, start, 0) \
    D(bid_t, count, 0)

#define FOR_BRANCH_STATS(N, D) \
    D(float, energy, 0.0f)     \
    D(float, max_energy, 0.0f) \
    D(float, mix_rate, 0.1f)   \
    D(float, thickness, 1.0f)

#define FOR_TREE_DATA(N, D) \
    D(float, total_energy, 0.0f)

#define DEFINE_STRUCT(StructName, MacroName) \
    struct StructName { MacroName(DEF_SCALAR, DEF_SCALAR_WITH_INIT) };

#define DEFINE_SOA_STRUCT(StructName, MacroName) \
    struct StructName##SoA {                     \
        MacroName(DEF_VECTOR, DEF_VECTOR)        \
                                                 \
        void push_back(const StructName& single) { \
             MacroName(PUSH_BACK_SINGLE, PUSH_BACK_SINGLE)                                    \
        }                                        \
                                                 \
        void swap_all(StructName##SoA &s) {          \
             MacroName(SWAP, SWAP)                                    \
        }\
    };

#define DEFINE_DEVICE_SOA_STRUCT(StructName, MacroName) \
    struct StructName##SoADevice { MacroName(DEF_DEVICE_VECTOR, DEF_DEVICE_VECTOR) };

#define DEFINE_STRUCTS(StructName, MacroName) \
    DEFINE_STRUCT(StructName, MacroName)      \
    DEFINE_SOA_STRUCT(StructName, MacroName)  \
    DEFINE_DEVICE_SOA_STRUCT(StructName, MacroName)

#define DEF_SCALAR(type, name) type name{};
#define DEF_SCALAR_WITH_INIT(type, name, init) type name{init};
#define DEF_VECTOR(type, name, ...) std::vector<type> name{};
#define DEF_DEVICE_VECTOR(type, name, ...) thrust::device_vector<type> name{};
#define PUSH_BACK_SINGLE(type, name, ...) name.push_back(single.name);
#define SWAP(type, name, ...) name.swap(s.name);

    DEFINE_STRUCTS(BranchCore, FOR_BRANCH_CORE)
    DEFINE_STRUCTS(BranchShape, FOR_BRANCH_SHAPE)
    DEFINE_STRUCTS(BranchStats, FOR_BRANCH_STATS)
    DEFINE_STRUCTS(TreeData, FOR_TREE_DATA)

}