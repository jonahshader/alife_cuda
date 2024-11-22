#include "Fluid.cuh"

#include <cstdint>

namespace fluid {

__device__ Cell &get_cell(Cell *cells, int x, int y, int width, int height) {
  // wrap around edges
  if (x < 0)
    x += width;
  if (x >= width)
    x -= width;
  if (y < 0)
    y += height;
  if (y >= height)
    y -= height;
  return cells[y * width + x];
}

__global__ void fluid_diffuse(Cell *read_cells, Cell *write_cells, int width, int height,
                              float diffusion_rate) {
  int x_index = blockIdx.x * blockDim.x + threadIdx.x;
  int y_index = blockIdx.y * blockDim.y + threadIdx.y;

  // first, give some to the cell below to simulate gravity

  float new_amount = 0;
  new_amount += get_cell(read_cells, x_index - 1, y_index - 1, width, height).amount;
  new_amount += -1 * get_cell(read_cells, x_index, y_index - 1, width, height).amount; // was 2
  new_amount += get_cell(read_cells, x_index + 1, y_index - 1, width, height).amount;
  new_amount += 2 * get_cell(read_cells, x_index - 1, y_index, width, height).amount;
  new_amount += 4 * get_cell(read_cells, x_index, y_index, width, height).amount;
  new_amount += 2 * get_cell(read_cells, x_index + 1, y_index, width, height).amount;
  new_amount += get_cell(read_cells, x_index - 1, y_index + 1, width, height).amount;
  new_amount += 5 * get_cell(read_cells, x_index, y_index + 1, width, height).amount; // was 2
  new_amount += get_cell(read_cells, x_index + 1, y_index + 1, width, height).amount;

  // new cell is a lerp between old cell and new amount based on diffusion rate
  // get_cell(write_cells, x_index, y_index, width, height).amount = new_amount / 16.0f;
  Cell &write_cell = get_cell(write_cells, x_index, y_index, width, height);
  auto old_amount = write_cell.amount;
  write_cell.amount = old_amount * (1 - diffusion_rate) + new_amount / 16.0f * diffusion_rate;
}

} // namespace fluid
