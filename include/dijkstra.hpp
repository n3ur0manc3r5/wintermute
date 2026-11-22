#pragma once

#include "types.hpp"

#include <array>
#include <vector>

namespace teasar {

struct NeighborOffset {
    int dx;
    int dy;
    int dz;
    float step_dist;
};

std::array<NeighborOffset, 26> build_26_neighbors(const VoxelScale& scale);

std::vector<int> compute_dfr(VolumeGrid& grid, int root_idx, bool reset_field = true);

void compute_pdrf(VolumeGrid& grid,
                  int root_idx,
                  float pdrf_scale = 5000.0f,
                  int pdrf_exponent = 4,
                  float max_dfb_hint = -1.0f,
                  bool reset_fields = true);

}
