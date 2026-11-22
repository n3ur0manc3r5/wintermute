#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>

struct VoxelScale {
    float dx = 4.0f;
    float dy = 4.0f;
    float dz = 40.0f;
};

struct Point3D {
    int x;
    int y;
    int z;

    bool operator==(const Point3D& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const Point3D& other) const {
        return !(*this == other);
    }
};

struct SWCNode {
    int id;
    int type;
    float x;
    float y;
    float z;
    float radius;
    int parent_id;
};

struct DijkstraNode {
    int index;
    float dist;

    bool operator>(const DijkstraNode& other) const {
        return dist > other.dist;
    }
};

class VolumeGrid {
public:
    int dim_x = 0;
    int dim_y = 0;
    int dim_z = 0;
    VoxelScale scale{};

    std::vector<uint8_t> mask;
    std::vector<float> dfb;
    std::vector<float> dfr;
    std::vector<float> pdrf;
    std::vector<int32_t> parent;
    std::vector<uint8_t> invalidated;

    VolumeGrid() = default;

    VolumeGrid(int dx_dim, int dy_dim, int dz_dim, VoxelScale voxel_scale = {})
        : dim_x(dx_dim),
          dim_y(dy_dim),
          dim_z(dz_dim),
          scale(voxel_scale) {
        if (dim_x <= 0 || dim_y <= 0 || dim_z <= 0) {
            throw std::invalid_argument("volumegrid dimensions must be strictly positive.");
        }
        const std::size_t total = static_cast<std::size_t>(dim_x) *
                                  static_cast<std::size_t>(dim_y) *
                                  static_cast<std::size_t>(dim_z);
        const float inf = std::numeric_limits<float>::infinity();
        mask.assign(total, 0);
        dfb.assign(total, 0.0f);
        dfr.assign(total, inf);
        pdrf.assign(total, inf);
        parent.assign(total, -1);
        invalidated.assign(total, 0);
    }

    [[nodiscard]] inline std::size_t size() const {
        return static_cast<std::size_t>(dim_x) *
               static_cast<std::size_t>(dim_y) *
               static_cast<std::size_t>(dim_z);
    }

    [[nodiscard]] inline int to_index(int x, int y, int z) const {
        return x + y * dim_x + z * dim_x * dim_y;
    }

    [[nodiscard]] inline Point3D to_coords(int idx) const {
        const int xy_stride = dim_x * dim_y;
        const int z = idx / xy_stride;
        const int rem = idx % xy_stride;
        const int y = rem / dim_x;
        const int x = rem % dim_x;
        return {x, y, z};
    }

    [[nodiscard]] inline bool is_valid(int x, int y, int z) const {
        return x >= 0 && x < dim_x &&
               y >= 0 && y < dim_y &&
               z >= 0 && z < dim_z;
    }

    [[nodiscard]] inline float physical_dist_sq(Point3D p1, Point3D p2) const {
        const float ddx = static_cast<float>(p1.x - p2.x) * scale.dx;
        const float ddy = static_cast<float>(p1.y - p2.y) * scale.dy;
        const float ddz = static_cast<float>(p1.z - p2.z) * scale.dz;
        return ddx * ddx + ddy * ddy + ddz * ddz;
    }

    [[nodiscard]] inline float physical_dist(Point3D p1, Point3D p2) const {
        return std::sqrt(physical_dist_sq(p1, p2));
    }
};
