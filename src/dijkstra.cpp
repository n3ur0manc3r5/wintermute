#include "dijkstra.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>

namespace teasar {

std::array<NeighborOffset, 26> build_26_neighbors(const VoxelScale& scale) {
    std::array<NeighborOffset, 26> neighbors{};
    std::size_t idx = 0;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                const float px = static_cast<float>(dx) * scale.dx;
                const float py = static_cast<float>(dy) * scale.dy;
                const float pz = static_cast<float>(dz) * scale.dz;
                const float dist = std::sqrt(px * px + py * py + pz * pz);
                neighbors[idx++] = {dx, dy, dz, dist};
            }
        }
    }
    return neighbors;
}

std::vector<int> compute_dfr(VolumeGrid& grid, int root_idx, bool reset_field) {
    const std::size_t total = grid.size();
    if (root_idx < 0 || static_cast<std::size_t>(root_idx) >= total) {
        throw std::out_of_range("compute_dfr: root_idx out of bounds.");
    }
    if (grid.mask[static_cast<std::size_t>(root_idx)] == 0) {
        throw std::invalid_argument("compute_dfr: root voxel is not in foreground mask.");
    }

    const float inf = std::numeric_limits<float>::infinity();
    if (reset_field) {
        grid.dfr.assign(total, inf);
    }

    const auto neighbors = build_26_neighbors(grid.scale);
    std::priority_queue<DijkstraNode, std::vector<DijkstraNode>, std::greater<DijkstraNode>> pq;
    std::vector<int> reachable;

    grid.dfr[static_cast<std::size_t>(root_idx)] = 0.0f;
    pq.push({root_idx, 0.0f});

    while (!pq.empty()) {
        const DijkstraNode curr = pq.top();
        pq.pop();

        if (curr.dist > grid.dfr[static_cast<std::size_t>(curr.index)]) {
            continue;
        }

        reachable.push_back(curr.index);
        const Point3D p = grid.to_coords(curr.index);

        for (const auto& nb : neighbors) {
            const int nx = p.x + nb.dx;
            const int ny = p.y + nb.dy;
            const int nz = p.z + nb.dz;
            if (!grid.is_valid(nx, ny, nz)) {
                continue;
            }
            const int nb_idx = grid.to_index(nx, ny, nz);
            if (grid.mask[static_cast<std::size_t>(nb_idx)] == 0) {
                continue;
            }

            const float cand = curr.dist + nb.step_dist;
            if (cand < grid.dfr[static_cast<std::size_t>(nb_idx)]) {
                grid.dfr[static_cast<std::size_t>(nb_idx)] = cand;
                pq.push({nb_idx, cand});
            }
        }
    }

    return reachable;
}

void compute_pdrf(VolumeGrid& grid,
                  int root_idx,
                  float pdrf_scale,
                  int pdrf_exponent,
                  float max_dfb_hint,
                  bool reset_fields) {
    const std::size_t total = grid.size();
    if (root_idx < 0 || static_cast<std::size_t>(root_idx) >= total) {
        throw std::out_of_range("compute_pdrf: root_idx out of bounds.");
    }
    if (grid.mask[static_cast<std::size_t>(root_idx)] == 0) {
        throw std::invalid_argument("compute_pdrf: root voxel is not in foreground mask.");
    }

    const float inf = std::numeric_limits<float>::infinity();
    if (reset_fields) {
        grid.pdrf.assign(total, inf);
        grid.parent.assign(total, -1);
    }

    float max_dfb = max_dfb_hint;
    if (max_dfb <= 0.0f) {
        max_dfb = 0.0f;
        for (std::size_t i = 0; i < total; ++i) {
            if (grid.mask[i] != 0 && grid.dfb[i] > max_dfb) {
                max_dfb = grid.dfb[i];
            }
        }
    }

    const float inv_max_dfb = (max_dfb > 0.0f) ? (1.0f / max_dfb) : 0.0f;
    const auto neighbors = build_26_neighbors(grid.scale);

    std::priority_queue<DijkstraNode, std::vector<DijkstraNode>, std::greater<DijkstraNode>> pq;
    grid.pdrf[static_cast<std::size_t>(root_idx)] = 0.0f;
    grid.parent[static_cast<std::size_t>(root_idx)] = -1;
    pq.push({root_idx, 0.0f});

    while (!pq.empty()) {
        const DijkstraNode curr = pq.top();
        pq.pop();

        if (curr.dist > grid.pdrf[static_cast<std::size_t>(curr.index)]) {
            continue;
        }

        const Point3D p = grid.to_coords(curr.index);

        for (const auto& nb : neighbors) {
            const int nx = p.x + nb.dx;
            const int ny = p.y + nb.dy;
            const int nz = p.z + nb.dz;
            if (!grid.is_valid(nx, ny, nz)) {
                continue;
            }
            const int nb_idx = grid.to_index(nx, ny, nz);
            if (grid.mask[static_cast<std::size_t>(nb_idx)] == 0) {
                continue;
            }

            const float ratio = std::clamp(1.0f - grid.dfb[static_cast<std::size_t>(nb_idx)] * inv_max_dfb,
                                           0.0f, 1.0f);
            float pow_term = 1.0f;
            for (int e = 0; e < pdrf_exponent; ++e) {
                pow_term *= ratio;
            }
            const float edge_cost = nb.step_dist * (1.0f + pdrf_scale * pow_term);
            const float cand = curr.dist + edge_cost;

            if (cand < grid.pdrf[static_cast<std::size_t>(nb_idx)]) {
                grid.pdrf[static_cast<std::size_t>(nb_idx)] = cand;
                grid.parent[static_cast<std::size_t>(nb_idx)] = curr.index;
                pq.push({nb_idx, cand});
            }
        }
    }
}

}
