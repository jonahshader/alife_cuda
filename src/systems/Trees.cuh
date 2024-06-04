#pragma once


#include <cstdint>
#include <random>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include "systems/Game.cuh"
#include <thrust/device_vector.h>

#include "TreeTypes.cuh"

namespace trees {
    struct BranchNav {
        trees2::bid_t id{};
        std::vector<trees2::bid_t> children{};
    };


    struct BranchNodeFull {
        trees2::BranchCore core{};
        trees2::BranchStats stats{};
        BranchNav nav{};
    };


    using Tree = std::vector<trees2::BranchNode>;


    struct TreeBatch {
        std::vector<trees2::BranchShape> tree_shapes{};
        std::vector<trees2::TreeData> tree_data{};
        Tree trees{};
    };


    std::vector<BranchNodeFull> build_tree(uint32_t num_nodes, std::default_random_engine &rand,
                                           glm::vec2 start_pos = glm::vec2{});

    Tree build_tree_optimized(uint32_t num_nodes, std::default_random_engine &rand, glm::vec2 start_pos = glm::vec2{});

    TreeBatch concatenate_trees(const std::vector<Tree> &trees);

    /**
     * Render a tree using the LineRenderer. Assumes the LineRenderer has already been set up.
     * @param line_renderer The LineRenderer to use
     * @param batch The nodes of the trees
     */
    void render_tree(LineRenderer &line_renderer, const trees2::TreeBatch &batch, glm::mat4 transform);

    /**
     * Propagate orientation and position changes through the tree
     * @param batch The nodes of the batch
     */
    void update_tree_cuda(trees2::TreeBatchDevice &read_batch_device, trees2::TreeBatchDevice &write_batch_device);

    /**
     * Sort the tree so that the parent of each node is before the node in the vector.
     * Result will be breadth-first traversal order.
     * @param nodes The nodes of the tree
     */
    std::vector<BranchNodeFull> sort_tree(const std::vector<BranchNodeFull> &nodes);

    __global__
    void render_tree_kernel(unsigned int* line_vbo, const trees2::TreeBatchPtrs batch, size_t node_count);

    __host__ __device__
    glm::vec2 get_length_vec(const trees2::BranchCore &core);

    __host__ __device__
    glm::vec2 get_length_vec(const trees2::BranchCoreSoA &core, trees2::bid_t i);

    __host__ __device__
    glm::vec2 get_length_vec(float abs_rot, float length);

    class Trees {
    public:
        explicit Trees(bool use_graphics);
        ~Trees() = default;

        void generate_random_trees(uint32_t num_trees, uint32_t num_nodes, std::default_random_engine &rand);
        void update(float dt);
        void render(const glm::mat4 &transform);

    private:
        trees2::TreeBatch read_host{}, write_host{};
        trees2::TreeBatchDevice read_device{}, write_device{};
        bool host_current{false}, device_current{false};

        std::unique_ptr<LineRenderer> line_renderer{};
    };

}
