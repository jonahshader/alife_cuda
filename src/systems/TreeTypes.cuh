#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace trees2 {
    using bid_t = uint32_t;


#define FOR_BRANCH_CORE(N, D)       \
    D(float, length, 0)             \
    D(float, current_rel_rot, 0)    \
    D(float, target_rel_rot, 0)     \
    D(float, rot_vel, 0)            \
    D(float, rot_acc, 0)            \
    D(float, abs_rot, 0)            \
    N(glm::vec2, pos)               \
    N(glm::vec2, vel)               \
    N(glm::vec2, acc)               \
    D(bid_t, parent, 0)

#define FOR_BRANCH_SHAPE(N, D)  \
    D(bid_t, start, 0)          \
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

#define DEFINE_STRUCT_PTR(StructName, MacroName) \
    struct StructName##Ptrs { \
    MacroName(DEF_SCALAR_PTR, DEF_SCALAR_PTR) \
\
    void get_ptrs(StructName##SoADevice& s) { \
        MacroName(SET_PTR, SET_PTR) \
    } \
};



#define DEFINE_SOA_STRUCT(StructName, MacroName)            \
    struct StructName##SoA {                                \
        MacroName(DEF_VECTOR, DEF_VECTOR)                   \
                                                            \
        void push_back(const StructName& single) {          \
             MacroName(PUSH_BACK_SINGLE, PUSH_BACK_SINGLE)  \
        }                                                   \
                                                            \
        void push_back(const std::vector<StructName>& vec) {\
            for (const auto& s : vec) {                     \
                push_back(s);                               \
            }                                               \
        }                                                   \
                                                            \
        void swap_all(StructName##SoA &s) {                 \
             MacroName(SWAP, SWAP)                          \
        }                                                   \
    };

#define DEFINE_DEVICE_SOA_STRUCT(StructName, MacroName) \
    struct StructName##SoADevice {                      \
        MacroName(DEF_DEVICE_VECTOR, DEF_DEVICE_VECTOR) \
                                                        \
    void copy_from_host(const StructName##SoA& host) {  \
        MacroName(COPY_FROM_HOST, COPY_FROM_HOST)       \
    }                                                   \
                                                        \
    void copy_to_host(StructName##SoA& host) const {    \
        MacroName(COPY_TO_HOST, COPY_TO_HOST)           \
    }                                                   \
                                                        \
};

#define DEFINE_STRUCTS(StructName, MacroName) \
    DEFINE_STRUCT(StructName, MacroName)      \
    DEFINE_SOA_STRUCT(StructName, MacroName)  \
    DEFINE_DEVICE_SOA_STRUCT(StructName, MacroName) \
    DEFINE_STRUCT_PTR(StructName, MacroName)

#define DEF_SCALAR(type, name) type name{};
#define DEF_SCALAR_WITH_INIT(type, name, init) type name{init};
#define DEF_SCALAR_PTR(type, name) type* name{nullptr};
#define DEF_VECTOR(type, name, ...) thrust::host_vector<type> name{};
#define DEF_DEVICE_VECTOR(type, name, ...) thrust::device_vector<type> name{};
#define PUSH_BACK_SINGLE(type, name, ...) name.push_back(single.name);
#define SWAP(type, name, ...) name.swap(s.name);
// TODO: avoid reallocating vectors?
#define COPY_FROM_HOST(type, name, ...) name = host.name;
#define COPY_TO_HOST(type, name, ...) host.name = name;
#define SET_PTR(type, name, ...) name = s.name.data().get();

    DEFINE_STRUCTS(BranchCore, FOR_BRANCH_CORE)

    DEFINE_STRUCTS(BranchShape, FOR_BRANCH_SHAPE)

    DEFINE_STRUCTS(BranchStats, FOR_BRANCH_STATS)

    DEFINE_STRUCTS(TreeData, FOR_TREE_DATA)


#undef FOR_BRANCH_CORE
#undef FOR_BRANCH_SHAPE
#undef FOR_BRANCH_STATS
#undef FOR_TREE_DATA
#undef DEFINE_STRUCT
#undef DEFINE_STRUCT_PTR
#undef DEFINE_SOA_STRUCT
#undef DEFINE_DEVICE_SOA_STRUCT
#undef DEFINE_STRUCTS

    // TODO: can probably use a macro to define these
    struct BranchNode {
        BranchCore core{};
        BranchStats stats{};
        BranchShape ch{};
    };

    struct BranchNodeSoA {
        BranchCoreSoA core{};
        BranchStatsSoA stats{};
        BranchShapeSoA ch{};

        void push_back(const BranchNode &single) {
            core.push_back(single.core);
            stats.push_back(single.stats);
            ch.push_back(single.ch);
        }

        void push_back(const std::vector<BranchNode> &vec) {
            for (const auto& s : vec) {
                push_back(s);
            }
        }

        void swap_all(BranchNodeSoA &s) {
            core.swap_all(s.core);
            stats.swap_all(s.stats);
            ch.swap_all(s.ch);
        }
    };

    struct BranchNodeSoADevice {
        BranchCoreSoADevice core{};
        BranchStatsSoADevice stats{};
        BranchShapeSoADevice ch{};

        void copy_from_host(const BranchNodeSoA &host) {
            core.copy_from_host(host.core);
            stats.copy_from_host(host.stats);
            ch.copy_from_host(host.ch);
        }

        void copy_to_host(BranchNodeSoA &host) {
            core.copy_to_host(host.core);
            stats.copy_to_host(host.stats);
            ch.copy_to_host(host.ch);
        }
    };

    struct BranchNodePtrs {
        BranchCorePtrs core{};
        BranchStatsPtrs stats{};
        BranchShapePtrs ch{};

        void get_ptrs(BranchNodeSoADevice& s) {
            core.get_ptrs(s.core);
            stats.get_ptrs(s.stats);
            ch.get_ptrs(s.ch);
        }
    };

    struct TreeBatch {
        BranchShapeSoA tree_shapes{};
        TreeDataSoA tree_data{};
        BranchNodeSoA trees{};
    };

    struct TreeBatchDevice {
        BranchShapeSoADevice tree_shapes{};
        TreeDataSoADevice tree_data{};
        BranchNodeSoADevice trees{};

        void copy_from_host(const TreeBatch &host) {
            tree_shapes.copy_from_host(host.tree_shapes);
            tree_data.copy_from_host(host.tree_data);
            trees.copy_from_host(host.trees);
        }

        void copy_to_host(TreeBatch &host) {
            tree_shapes.copy_to_host(host.tree_shapes);
            tree_data.copy_to_host(host.tree_data);
            trees.copy_to_host(host.trees);
        }
    };

    struct TreeBatchPtrs {
        BranchShapePtrs tree_shapes{};
        TreeDataPtrs tree_data{};
        BranchNodePtrs trees{};

        void get_ptrs(TreeBatchDevice& s) {
            tree_shapes.get_ptrs(s.tree_shapes);
            tree_data.get_ptrs(s.tree_data);
            trees.get_ptrs(s.trees);
        }
    };

}
