#pragma once

#include "types.hpp"

#include <optional>
#include <vector>

namespace teasar {

enum class RootStrategy {
    MaxDFB,
    ExtremumTip
};

struct TeasarParams {
    float pdrf_scale = 5000.0f;
    int pdrf_exponent = 4;
    float invalidation_scale = 1.5f;
    float invalidation_const = 20.0f;
    bool black_border = false;
    int min_component_voxels = 1;
    int branch_swc_type = 3;
    RootStrategy root_strategy = RootStrategy::MaxDFB;
    std::optional<Point3D> explicit_root = std::nullopt;
};

struct SkeletonNode {
    int voxel_idx = -1;
    Point3D coords{0, 0, 0};
    float x_nm = 0.0f;
    float y_nm = 0.0f;
    float z_nm = 0.0f;
    float radius_nm = 0.0f;
    int parent = -1;
    int type = 3;
};

void invalidate_sphere(VolumeGrid& grid, int center_idx, float radius_nm);

std::vector<SkeletonNode> extract_skeleton(VolumeGrid& grid,
                                           const TeasarParams& params = {},
                                           bool recompute_dfb = true);

}
