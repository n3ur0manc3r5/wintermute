#pragma once

#include "mesh.hpp"
#include "types.hpp"
#include "volume_io.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace teasar {

struct VoxelizerParams {
    VoxelScale scale{400.0f, 400.0f, 400.0f};
    float padding_nm = 4000.0f;
    float soma_blend_nm = 1200.0f;
    float branch_blend_nm = 250.0f;
    float min_radius_scale = 0.92f;
    float soma_voxel_nm = 180.0f;
    int tube_sides = 10;
};

struct VolumeOverlapStats {
    std::size_t gt_voxels = 0;
    std::size_t recon_voxels = 0;
    std::size_t intersection_voxels = 0;
    double iou = 0.0;
    double dice = 0.0;
    double recall = 0.0;
    double precision = 0.0;
};

void rasterize_swc_into_grid(const std::vector<SWCNode>& swc,
                             const VoxelizerParams& params,
                             VolumeGrid& grid,
                             const VolumeMetadata& meta,
                             std::vector<float>& out_sdf);

bool rasterize_swc_auto_grid(const std::vector<SWCNode>& swc,
                             const VoxelizerParams& params,
                             VolumeGrid& out_grid,
                             VolumeMetadata& out_meta,
                             std::vector<float>& out_sdf);

bool rasterize_swc_roi_grid(const std::vector<SWCNode>& swc,
                            const VoxelizerParams& params,
                            const Vec3f& center_nm,
                            float half_span_nm,
                            VolumeGrid& out_grid,
                            VolumeMetadata& out_meta,
                            std::vector<float>& out_sdf);

void smooth_mesh_taubin(TriangleMesh& mesh,
                        int iterations = 4,
                        float lambda = 0.50f,
                        float mu = -0.53f);

TriangleMesh extract_isosurface_mesh(const VolumeGrid& grid,
                                     const VolumeMetadata& meta,
                                     const std::vector<float>& sdf,
                                     int taubin_iters = 4);

TriangleMesh build_hybrid_neuron_mesh(const std::vector<SWCNode>& swc,
                                      const VoxelizerParams& params = {});

bool export_mesh_obj(const std::string& filepath, const TriangleMesh& mesh);

bool export_mesh_webgl_html(const std::string& filepath,
                            const TriangleMesh& recon_mesh,
                            const std::vector<SWCNode>& swc,
                            const TriangleMesh* ref_mesh = nullptr);

VolumeOverlapStats evaluate_swc_against_volume(const VolumeGrid& gt_grid,
                                               const VolumeMetadata& gt_meta,
                                               const std::vector<SWCNode>& swc,
                                               const VoxelizerParams& params,
                                               VolumeGrid* out_recon_grid = nullptr,
                                               std::vector<float>* out_recon_sdf = nullptr);

bool render_mesh_comparison_ppm(const std::string& filepath,
                                const TriangleMesh& left_mesh,
                                const TriangleMesh& right_mesh,
                                int width = 1400,
                                int height = 760,
                                float yaw_rad = 0.35f,
                                float pitch_rad = 0.12f,
                                bool use_custom_focus = false,
                                Vec3f focus_center = {},
                                float focus_span = 0.0f);

}
