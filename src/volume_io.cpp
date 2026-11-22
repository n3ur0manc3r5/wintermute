#include "volume_io.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>

namespace teasar {

bool load_volume_nmv(const std::string& filepath, VolumeGrid& grid, VolumeMetadata& meta) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    char magic[4] = {};
    in.read(magic, 4);
    if (!in || std::memcmp(magic, "NMV1", 4) != 0) {
        return false;
    }

    int32_t dims[3] = {};
    float scale[3] = {};
    float origin[3] = {};
    int32_t root[3] = {};

    in.read(reinterpret_cast<char*>(dims), sizeof(dims));
    in.read(reinterpret_cast<char*>(scale), sizeof(scale));
    in.read(reinterpret_cast<char*>(origin), sizeof(origin));
    in.read(reinterpret_cast<char*>(root), sizeof(root));
    if (!in || dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
        return false;
    }

    grid = VolumeGrid(dims[0], dims[1], dims[2], VoxelScale{scale[0], scale[1], scale[2]});
    meta.origin_x_nm = origin[0];
    meta.origin_y_nm = origin[1];
    meta.origin_z_nm = origin[2];
    meta.root_voxel = {root[0], root[1], root[2]};

    const std::size_t total = grid.size();
    in.read(reinterpret_cast<char*>(grid.mask.data()), static_cast<std::streamsize>(total));
    return static_cast<std::size_t>(in.gcount()) == total;
}

bool save_volume_nmv(const std::string& filepath, const VolumeGrid& grid, const VolumeMetadata& meta) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    const char magic[4] = {'N', 'M', 'V', '1'};
    const int32_t dims[3] = {grid.dim_x, grid.dim_y, grid.dim_z};
    const float scale[3] = {grid.scale.dx, grid.scale.dy, grid.scale.dz};
    const float origin[3] = {meta.origin_x_nm, meta.origin_y_nm, meta.origin_z_nm};
    const int32_t root[3] = {meta.root_voxel.x, meta.root_voxel.y, meta.root_voxel.z};

    out.write(magic, 4);
    out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(scale), sizeof(scale));
    out.write(reinterpret_cast<const char*>(origin), sizeof(origin));
    out.write(reinterpret_cast<const char*>(root), sizeof(root));
    out.write(reinterpret_cast<const char*>(grid.mask.data()), static_cast<std::streamsize>(grid.size()));
    return out.good();
}

}
