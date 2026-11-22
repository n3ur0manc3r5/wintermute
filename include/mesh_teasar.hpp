#pragma once

#include "mesh.hpp"
#include "teasar.hpp"

#include <vector>

namespace teasar {

struct MeshTeasarParams {
    float stitch_max_dist_nm = 2500.0f;
    int min_component_verts = 25;
    float sdf_max_ray_nm = 16000.0f;
    float pdrf_scale = 5000.0f;
    int pdrf_exponent = 4;
    float invalidation_scale = 1.7f;
    float invalidation_const_nm = 1400.0f;
    float soma_invalidation_radius_nm = 8500.0f;
    int smooth_iterations = 3;
    float min_branch_len_nm = 2500.0f;
    int branch_swc_type = 3;
};

struct MeshSkeletonStats {
    int total_components = 0;
    int kept_components = 0;
    int stitched_bridges = 0;
    std::size_t reachable_verts = 0;
    float soma_radius_nm = 0.0f;
    float median_sdf_radius_nm = 0.0f;
};

std::vector<float> compute_mesh_sdf(const TriangleMesh& mesh,
                                    const std::vector<Vec3f>& vnorm,
                                    const std::vector<Vec3f>& fnorm,
                                    float max_ray_nm = 16000.0f);

std::vector<SkeletonNode> extract_mesh_skeleton(const TriangleMesh& mesh,
                                                const MeshTeasarParams& params = {},
                                                MeshSkeletonStats* stats = nullptr);

}
