//
// Created by jonah on 3/12/2024.
//

#pragma once

#include "Resources.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


namespace fluid {
    struct Cell {
        float amount;
    };

    struct Fluid {
        thrust::device_vector<Cell> a_cells, b_cells;
        int width;
        int height;

        void init(int width, int height, float amount, float variance, Resources& r) {
            this->width = width;
            this->height = height;

            // create triangular distribution
            // sample from triangle distribution, amount +- variance

            // compute on host
            thrust::host_vector<Cell> hostCells;
            for (int i = 0; i < width * height; i++) {
                float sample = (r.rand() - r.rand()) * variance + amount;
                hostCells.push_back({sample});
            }

            // copy to device
            a_cells = hostCells;
            b_cells = hostCells;
        }
    };

    __global__
    void fluid_diffuse(Cell* read_cells, Cell* write_cells, int width, int height, float diffusion_rate);


}