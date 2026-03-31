#pragma once

#include "graphics/renderers/line_renderer.cuh"
#include "systems/trees.cuh"

#include <glm/glm.hpp>

namespace trees {

/**
 * Render a tree using the LineRenderer. Assumes the LineRenderer has already been set up.
 * @param line_renderer The LineRenderer to use
 * @param batch The nodes of the trees
 */
void render_tree(LineRenderer &line_renderer, const trees2::TreeBatch &batch, glm::mat4 transform);

__global__ void render_tree_kernel(unsigned int *line_vbo, const trees2::TreeBatchPtrs batch,
                                   size_t node_count);

void render_trees_cuda(const TreesState &state, LineRenderer &renderer, const glm::mat4 &transform);

} // namespace trees
