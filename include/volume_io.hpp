#pragma once

#include "types.hpp"

#include <string>

namespace teasar {

struct VolumeMetadata {
    float origin_x_nm = 0.0f;
    float origin_y_nm = 0.0f;
    float origin_z_nm = 0.0f;
    Point3D root_voxel{-1, -1, -1};
};

bool load_volume_nmv(const std::string& filepath, VolumeGrid& grid, VolumeMetadata& meta);

bool save_volume_nmv(const std::string& filepath, const VolumeGrid& grid, const VolumeMetadata& meta);

}
