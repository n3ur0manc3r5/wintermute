#include "teasar.hpp"

#include "dijkstra.hpp"
#include "edt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>

namespace teasar {

void invalidate_sphere(VolumeGrid& grid, int center_idx, float radius_nm) {
    grid.invalidated[static_cast<std::size_t>(center_idx)] = 1;
    if (radius_nm <= 0.0f) {
        return;
    }

    const Point3D c = grid.to_coords(center_idx);
    const float r_sq = radius_nm * radius_nm;

    const int rx = static_cast<int>(std::floor(radius_nm / grid.scale.dx));
    const int ry = static_cast<int>(std::floor(radius_nm / grid.scale.dy));
    const int rz = static_cast<int>(std::floor(radius_nm / grid.scale.dz));

    const int x0 = std::max(0, c.x - rx);
    const int x1 = std::min(grid.dim_x - 1, c.x + rx);
    const int y0 = std::max(0, c.y - ry);
    const int y1 = std::min(grid.dim_y - 1, c.y + ry);
    const int z0 = std::max(0, c.z - rz);
    const int z1 = std::min(grid.dim_z - 1, c.z + rz);

    const int xy_stride = grid.dim_x * grid.dim_y;
    for (int z = z0; z <= z1; ++z) {
        const float ddz = static_cast<float>(z - c.z) * grid.scale.dz;
        const float ddz2 = ddz * ddz;
        if (ddz2 > r_sq) {
            continue;
        }
        const int z_off = z * xy_stride;

        for (int y = y0; y <= y1; ++y) {
            const float ddy = static_cast<float>(y - c.y) * grid.scale.dy;
            const float ddyz2 = ddz2 + ddy * ddy;
            if (ddyz2 > r_sq) {
                continue;
            }
            const int row_off = z_off + y * grid.dim_x;

            for (int x = x0; x <= x1; ++x) {
                const std::size_t idx = static_cast<std::size_t>(row_off + x);
                if (grid.mask[idx] == 0 || grid.invalidated[idx] != 0) {
                    continue;
                }
                const float ddx = static_cast<float>(x - c.x) * grid.scale.dx;
                if (ddyz2 + ddx * ddx <= r_sq) {
                    grid.invalidated[idx] = 1;
                }
            }
        }
    }
}

namespace {

SkeletonNode make_node(const VolumeGrid& grid, int voxel_idx, int parent_node, int swc_type) {
    const Point3D p = grid.to_coords(voxel_idx);
    SkeletonNode node;
    node.voxel_idx = voxel_idx;
    node.coords = p;
    node.x_nm = static_cast<float>(p.x) * grid.scale.dx;
    node.y_nm = static_cast<float>(p.y) * grid.scale.dy;
    node.z_nm = static_cast<float>(p.z) * grid.scale.dz;
    node.radius_nm = grid.dfb[static_cast<std::size_t>(voxel_idx)];
    node.parent = parent_node;
    node.type = swc_type;
    return node;
}

std::vector<int> collect_component_bfs(const VolumeGrid& grid,
                                       int seed_idx,
                                       const std::array<NeighborOffset, 26>& neighbors,
                                       std::vector<uint8_t>& visited) {
    std::vector<int> comp;
    std::queue<int> q;
    visited[static_cast<std::size_t>(seed_idx)] = 1;
    q.push(seed_idx);

    while (!q.empty()) {
        const int u = q.front();
        q.pop();
        comp.push_back(u);

        const Point3D p = grid.to_coords(u);
        for (const auto& nb : neighbors) {
            const int nx = p.x + nb.dx;
            const int ny = p.y + nb.dy;
            const int nz = p.z + nb.dz;
            if (!grid.is_valid(nx, ny, nz)) {
                continue;
            }
            const int nb_idx = grid.to_index(nx, ny, nz);
            const std::size_t nb_u = static_cast<std::size_t>(nb_idx);
            if (grid.mask[nb_u] != 0 && visited[nb_u] == 0) {
                visited[nb_u] = 1;
                q.push(nb_idx);
            }
        }
    }

    return comp;
}

}

std::vector<SkeletonNode> extract_skeleton(VolumeGrid& grid,
                                           const TeasarParams& params,
                                           bool recompute_dfb) {
    const std::size_t total = grid.size();
    if (total == 0) {
        return {};
    }

    if (recompute_dfb) {
        compute_dfb(grid, params.black_border);
    }

    const float inf = std::numeric_limits<float>::infinity();
    grid.dfr.assign(total, inf);
    grid.pdrf.assign(total, inf);
    grid.parent.assign(total, -1);
    grid.invalidated.assign(total, 0);

    const auto neighbors = build_26_neighbors(grid.scale);
    std::vector<uint8_t> visited(total, 0);
    std::vector<int> voxel_to_skel_node(total, -1);
    std::vector<SkeletonNode> skeleton;

    int explicit_root_idx = -1;
    if (params.explicit_root.has_value()) {
        const Point3D r = *params.explicit_root;
        if (grid.is_valid(r.x, r.y, r.z)) {
            const int idx = grid.to_index(r.x, r.y, r.z);
            if (grid.mask[static_cast<std::size_t>(idx)] != 0) {
                explicit_root_idx = idx;
            }
        }
    }

    for (std::size_t i = 0; i < total; ++i) {
        if (grid.mask[i] == 0 || visited[i] != 0) {
            continue;
        }

        std::vector<int> comp = collect_component_bfs(grid, static_cast<int>(i), neighbors, visited);
        if (static_cast<int>(comp.size()) < params.min_component_voxels) {
            for (int v : comp) {
                grid.invalidated[static_cast<std::size_t>(v)] = 1;
            }
            continue;
        }

        float comp_max_dfb = 0.0f;
        int max_dfb_voxel = comp[0];
        bool has_explicit_root = false;
        for (int v : comp) {
            if (v == explicit_root_idx) {
                has_explicit_root = true;
            }
            const float d = grid.dfb[static_cast<std::size_t>(v)];
            if (d > comp_max_dfb) {
                comp_max_dfb = d;
                max_dfb_voxel = v;
            }
        }

        int root_idx = max_dfb_voxel;
        if (has_explicit_root) {
            root_idx = explicit_root_idx;
        } else if (params.root_strategy == RootStrategy::ExtremumTip) {
            compute_dfr(grid, comp[0], false);
            float max_d = -1.0f;
            for (int v : comp) {
                const float d = grid.dfr[static_cast<std::size_t>(v)];
                if (std::isfinite(d) && d > max_d) {
                    max_d = d;
                    root_idx = v;
                }
            }
            for (int v : comp) {
                grid.dfr[static_cast<std::size_t>(v)] = inf;
            }
        }

        std::vector<int> reachable = compute_dfr(grid, root_idx, false);
        compute_pdrf(grid,
                     root_idx,
                     params.pdrf_scale,
                     params.pdrf_exponent,
                     comp_max_dfb,
                     false);

        const int root_node_id = static_cast<int>(skeleton.size());
        skeleton.push_back(make_node(grid, root_idx, -1, 1));
        voxel_to_skel_node[static_cast<std::size_t>(root_idx)] = root_node_id;

        const float root_r = params.invalidation_scale * grid.dfb[static_cast<std::size_t>(root_idx)] +
                             params.invalidation_const;
        invalidate_sphere(grid, root_idx, root_r);

        std::sort(reachable.begin(), reachable.end(), [&grid](int a, int b) {
            const float da = grid.dfr[static_cast<std::size_t>(a)];
            const float db = grid.dfr[static_cast<std::size_t>(b)];
            if (da != db) {
                return da > db;
            }
            return a < b;
        });

        std::size_t cursor = 0;
        std::vector<int> path;
        while (cursor < reachable.size()) {
            while (cursor < reachable.size() &&
                   grid.invalidated[static_cast<std::size_t>(reachable[cursor])] != 0) {
                ++cursor;
            }
            if (cursor >= reachable.size()) {
                break;
            }

            const int target_idx = reachable[cursor];
            path.clear();

            int curr = target_idx;
            while (curr != -1 && voxel_to_skel_node[static_cast<std::size_t>(curr)] == -1) {
                path.push_back(curr);
                curr = grid.parent[static_cast<std::size_t>(curr)];
            }

            const int attach_node = (curr != -1) ? voxel_to_skel_node[static_cast<std::size_t>(curr)] : root_node_id;

            int prev_node = attach_node;
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                const int v_idx = *it;
                const int new_node_id = static_cast<int>(skeleton.size());
                skeleton.push_back(make_node(grid, v_idx, prev_node, params.branch_swc_type));
                voxel_to_skel_node[static_cast<std::size_t>(v_idx)] = new_node_id;
                prev_node = new_node_id;
            }

            for (int v_idx : path) {
                const float r = params.invalidation_scale * grid.dfb[static_cast<std::size_t>(v_idx)] +
                                params.invalidation_const;
                invalidate_sphere(grid, v_idx, r);
            }
        }
    }

    return skeleton;
}

}
