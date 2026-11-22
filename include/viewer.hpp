#pragma once

#include "mesh.hpp"
#include "types.hpp"

#include <string>
#include <vector>

namespace teasar {

bool export_html_viewer(const std::string& filepath,
                        const VolumeGrid& grid,
                        const std::vector<SWCNode>& raw_swc,
                        const std::vector<SWCNode>& rdp_swc,
                        float invalidation_scale = 1.6f,
                        float invalidation_const_nm = 24.0f,
                        int voxel_stride_xy = 2,
                        float origin_x_nm = 0.0f,
                        float origin_y_nm = 0.0f,
                        float origin_z_nm = 0.0f,
                        int max_surface_voxels = 4500);

bool export_mesh_html_viewer(const std::string& filepath,
                             const TriangleMesh& mesh,
                             const std::vector<SWCNode>& raw_swc,
                             const std::vector<SWCNode>& rdp_swc,
                             float invalidation_scale = 2.0f,
                             float invalidation_const_nm = 1800.0f,
                             int max_surface_points = 6000);

}
