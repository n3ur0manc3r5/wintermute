#include "voxelizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <unordered_map>

namespace teasar {
namespace {

inline float smin_poly(float a, float b, float k) {
    if (!std::isfinite(a)) return b;
    if (k <= 1e-4f) return std::min(a, b);
    const float h = std::max(k - std::fabs(a - b), 0.0f) / k;
    return std::min(a, b) - h * h * k * 0.25f;
}

inline float sd_round_cone(const Vec3f& p, const Vec3f& a, const Vec3f& b, float r1, float r2) {
    const Vec3f ba = b - a;
    const float l2 = ba.dot(ba);
    const float rr = r1 - r2;
    const float a2 = l2 - rr * rr;
    if (l2 <= 1e-4f || a2 <= 1e-4f) {
        return std::min((p - a).norm() - r1, (p - b).norm() - r2);
    }
    const float il2 = 1.0f / l2;
    const Vec3f pa = p - a;
    const float y = pa.dot(ba);
    const float z = y - l2;
    const Vec3f perp_v = pa * l2 - ba * y;
    const float x2 = perp_v.dot(perp_v);
    const float y2 = y * y * l2;
    const float z2 = z * z * l2;
    const float sgn_rr = (rr >= 0.0f) ? 1.0f : -1.0f;
    const float k = sgn_rr * rr * rr * x2;
    if (((z >= 0.0f) ? 1.0f : -1.0f) * a2 * z2 > k) {
        return std::sqrt(x2 + z2) * il2 - r2;
    }
    if (((y >= 0.0f) ? 1.0f : -1.0f) * a2 * y2 < k) {
        return std::sqrt(x2 + y2) * il2 - r1;
    }
    return (std::sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

inline uint64_t edge_key(uint64_t a, uint64_t b) {
    if (a > b) std::swap(a, b);
    return (a * 1469598103934665603ULL) ^ (b * 1099511628211ULL);
}

struct EdgePair {
    uint64_t a;
    uint64_t b;
    bool operator==(const EdgePair& o) const { return a == o.a && b == o.b; }
};

struct EdgePairHash {
    std::size_t operator()(const EdgePair& e) const {
        return static_cast<std::size_t>(edge_key(e.a, e.b));
    }
};

void rasterize_mesh_panel(const TriangleMesh& mesh,
                          const Vec3f& center,
                          float span,
                          float yaw,
                          float pitch,
                          int x_off,
                          int pw,
                          int ph,
                          int full_w,
                          std::vector<uint8_t>& rgb,
                          const std::array<float, 3>& base_col) {
    std::vector<float> zbuf(static_cast<std::size_t>(pw) * static_cast<std::size_t>(ph),
                            -std::numeric_limits<float>::infinity());

    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float fov = static_cast<float>(std::min(pw, ph)) * 1.45f;
    const float inv_span = 1.0f / std::max(1.0f, span);

    auto project_pt = [&](const Vec3f& p, Vec3f& view_p, float& sx, float& sy_out) {
        const float px = (p.x - center.x) * inv_span;
        const float py = -(p.y - center.y) * inv_span;
        const float pz = (p.z - center.z) * inv_span;
        const float x1 = px * cy + pz * sy;
        const float z1 = -px * sy + pz * cy;
        const float y2 = py * cp - z1 * sp;
        const float z2 = py * sp + z1 * cp;
        view_p = {x1, y2, z2};
        const float sc = fov / std::max(0.25f, 2.25f - z2);
        sx = static_cast<float>(pw) * 0.5f + x1 * sc;
        sy_out = static_cast<float>(ph) * 0.5f - y2 * sc;
    };

    const Vec3f light1 = Vec3f{0.45f, 0.65f, 0.61f}.normalized();
    const Vec3f light2 = Vec3f{-0.55f, -0.35f, 0.75f}.normalized();
    const Vec3f view_dir{0.0f, 0.0f, 1.0f};

    std::vector<Vec3f> fnorm;
    const std::vector<Vec3f> vnorm = compute_vertex_normals(mesh, &fnorm);

    auto rot_vec = [&](const Vec3f& v) -> Vec3f {
        const float x1 = v.x * cy + v.z * sy;
        const float z1 = -v.x * sy + v.z * cy;
        const float y2 = -v.y * cp - z1 * sp;
        const float z2 = -v.y * sp + z1 * cp;
        return Vec3f{x1, y2, z2}.normalized();
    };

    const std::size_t nv = mesh.vertices.size();
    std::vector<Vec3f> proj_v(nv);
    std::vector<float> scr_x(nv);
    std::vector<float> scr_y(nv);
    std::vector<Vec3f> rot_n(nv);

    for (std::size_t i = 0; i < nv; ++i) {
        project_pt(mesh.vertices[i], proj_v[i], scr_x[i], scr_y[i]);
        rot_n[i] = rot_vec(vnorm[i]);
    }

    for (const auto& tri : mesh.faces) {
        if (tri[0] >= nv || tri[1] >= nv || tri[2] >= nv) continue;
        const float x0 = scr_x[tri[0]], y0 = scr_y[tri[0]], z0 = proj_v[tri[0]].z;
        const float x1 = scr_x[tri[1]], y1 = scr_y[tri[1]], z1 = proj_v[tri[1]].z;
        const float x2 = scr_x[tri[2]], y2 = scr_y[tri[2]], z2 = proj_v[tri[2]].z;
        if (z0 > 1.85f || z1 > 1.85f || z2 > 1.85f) continue;

        const float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (std::fabs(area) < 1e-4f) continue;
        const float inv_area = 1.0f / area;

        const int min_x = std::max(0, static_cast<int>(std::floor(std::min({x0, x1, x2}))));
        const int max_x = std::min(pw - 1, static_cast<int>(std::ceil(std::max({x0, x1, x2}))));
        const int min_y = std::max(0, static_cast<int>(std::floor(std::min({y0, y1, y2}))));
        const int max_y = std::min(ph - 1, static_cast<int>(std::ceil(std::max({y0, y1, y2}))));

        const Vec3f& n0 = rot_n[tri[0]];
        const Vec3f& n1 = rot_n[tri[1]];
        const Vec3f& n2 = rot_n[tri[2]];

        for (int py_i = min_y; py_i <= max_y; ++py_i) {
            const float py_f = static_cast<float>(py_i) + 0.5f;
            for (int px_i = min_x; px_i <= max_x; ++px_i) {
                const float px_f = static_cast<float>(px_i) + 0.5f;
                const float w0 = ((x1 - px_f) * (y2 - py_f) - (x2 - px_f) * (y1 - py_f)) * inv_area;
                const float w1 = ((x2 - px_f) * (y0 - py_f) - (x0 - px_f) * (y2 - py_f)) * inv_area;
                const float w2 = 1.0f - w0 - w1;
                if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) continue;

                const float z = w0 * z0 + w1 * z1 + w2 * z2;
                const std::size_t zi = static_cast<std::size_t>(py_i) * static_cast<std::size_t>(pw) + static_cast<std::size_t>(px_i);
                if (z <= zbuf[zi]) continue;
                zbuf[zi] = z;

                Vec3f n = (n0 * w0 + n1 * w1 + n2 * w2).normalized();
                if (n.dot(view_dir) < 0.0f) n *= -1.0f;

                const float diff1 = std::max(0.0f, n.dot(light1));
                const float diff2 = std::max(0.0f, n.dot(light2));
                const Vec3f hvec = (light1 + view_dir).normalized();
                const float spec = std::pow(std::max(0.0f, n.dot(hvec)), 24.0f);
                const float shade = 0.18f + 0.72f * diff1 + 0.22f * diff2;

                const float r = std::clamp(base_col[0] * shade + 0.25f * spec, 0.0f, 1.0f);
                const float g = std::clamp(base_col[1] * shade + 0.25f * spec, 0.0f, 1.0f);
                const float b = std::clamp(base_col[2] * shade + 0.25f * spec, 0.0f, 1.0f);

                const std::size_t out_i = (static_cast<std::size_t>(py_i) * static_cast<std::size_t>(full_w) +
                                           static_cast<std::size_t>(x_off + px_i)) * 3;
                rgb[out_i + 0] = static_cast<uint8_t>(r * 255.0f);
                rgb[out_i + 1] = static_cast<uint8_t>(g * 255.0f);
                rgb[out_i + 2] = static_cast<uint8_t>(b * 255.0f);
            }
        }
    }
}

}

void rasterize_swc_into_grid(const std::vector<SWCNode>& swc,
                             const VoxelizerParams& params,
                             VolumeGrid& grid,
                             const VolumeMetadata& meta,
                             std::vector<float>& out_sdf) {
    const std::size_t total = grid.size();
    const float inf = std::numeric_limits<float>::infinity();
    out_sdf.assign(total, inf);
    grid.mask.assign(total, 0);
    if (swc.empty() || total == 0) {
        return;
    }

    std::unordered_map<int, std::size_t> id_to_idx;
    id_to_idx.reserve(swc.size() * 2);
    for (std::size_t i = 0; i < swc.size(); ++i) {
        id_to_idx[swc[i].id] = i;
    }

    const float vox_floor_r = params.min_radius_scale * std::max({grid.scale.dx, grid.scale.dy, grid.scale.dz});
    const std::size_t xy_stride = static_cast<std::size_t>(grid.dim_x) * static_cast<std::size_t>(grid.dim_y);

    std::vector<Vec3f> soma_centers;
    std::vector<float> soma_radii;

    for (const SWCNode& n : swc) {
        const Vec3f p{n.x * 1000.0f, n.y * 1000.0f, n.z * 1000.0f};
        const float rn = std::max(n.radius * 1000.0f, vox_floor_r);
        if (n.type == 1 || n.parent_id == -1 || id_to_idx.find(n.parent_id) == id_to_idx.end()) {
            soma_centers.push_back(p);
            soma_radii.push_back(rn);
            continue;
        }

        const SWCNode& par = swc[id_to_idx[n.parent_id]];
        const Vec3f a{par.x * 1000.0f, par.y * 1000.0f, par.z * 1000.0f};
        float ra = std::max(par.radius * 1000.0f, vox_floor_r);
        if (par.type == 1 || par.parent_id == -1) {
            ra = std::min(ra, std::max(rn * 1.40f, vox_floor_r));
        }

        const float max_r = std::max(ra, rn);
        const float margin = max_r + params.soma_blend_nm + std::max({grid.scale.dx, grid.scale.dy, grid.scale.dz});

        const int x0 = std::max(0, static_cast<int>(std::floor((std::min(a.x, p.x) - margin - meta.origin_x_nm) / grid.scale.dx)));
        const int x1 = std::min(grid.dim_x - 1, static_cast<int>(std::ceil((std::max(a.x, p.x) + margin - meta.origin_x_nm) / grid.scale.dx)));
        const int y0 = std::max(0, static_cast<int>(std::floor((std::min(a.y, p.y) - margin - meta.origin_y_nm) / grid.scale.dy)));
        const int y1 = std::min(grid.dim_y - 1, static_cast<int>(std::ceil((std::max(a.y, p.y) + margin - meta.origin_y_nm) / grid.scale.dy)));
        const int z0 = std::max(0, static_cast<int>(std::floor((std::min(a.z, p.z) - margin - meta.origin_z_nm) / grid.scale.dz)));
        const int z1 = std::min(grid.dim_z - 1, static_cast<int>(std::ceil((std::max(a.z, p.z) + margin - meta.origin_z_nm) / grid.scale.dz)));

        for (int z = z0; z <= z1; ++z) {
            const float pz = meta.origin_z_nm + static_cast<float>(z) * grid.scale.dz;
            const std::size_t z_off = static_cast<std::size_t>(z) * xy_stride;
            for (int y = y0; y <= y1; ++y) {
                const float py = meta.origin_y_nm + static_cast<float>(y) * grid.scale.dy;
                const std::size_t row_off = z_off + static_cast<std::size_t>(y) * static_cast<std::size_t>(grid.dim_x);
                for (int x = x0; x <= x1; ++x) {
                    const float px = meta.origin_x_nm + static_cast<float>(x) * grid.scale.dx;
                    const float d = sd_round_cone({px, py, pz}, a, p, ra, rn);
                    const std::size_t idx = row_off + static_cast<std::size_t>(x);
                    if (d < out_sdf[idx]) {
                        out_sdf[idx] = d;
                    }
                }
            }
        }
    }

    if (!soma_centers.empty()) {
        const float max_vox = std::max({grid.scale.dx, grid.scale.dy, grid.scale.dz});
        const float pad = params.soma_blend_nm + 3.0f * max_vox;
        float smin_x = inf, smin_y = inf, smin_z = inf;
        float smax_x = -inf, smax_y = -inf, smax_z = -inf;
        for (std::size_t s = 0; s < soma_centers.size(); ++s) {
            const Vec3f& c = soma_centers[s];
            const float m = soma_radii[s] + pad;
            smin_x = std::min(smin_x, c.x - m); smax_x = std::max(smax_x, c.x + m);
            smin_y = std::min(smin_y, c.y - m); smax_y = std::max(smax_y, c.y + m);
            smin_z = std::min(smin_z, c.z - m); smax_z = std::max(smax_z, c.z + m);
        }

        const int sx0 = std::max(0, static_cast<int>(std::floor((smin_x - meta.origin_x_nm) / grid.scale.dx)));
        const int sx1 = std::min(grid.dim_x - 1, static_cast<int>(std::ceil((smax_x - meta.origin_x_nm) / grid.scale.dx)));
        const int sy0 = std::max(0, static_cast<int>(std::floor((smin_y - meta.origin_y_nm) / grid.scale.dy)));
        const int sy1 = std::min(grid.dim_y - 1, static_cast<int>(std::ceil((smax_y - meta.origin_y_nm) / grid.scale.dy)));
        const int sz0 = std::max(0, static_cast<int>(std::floor((smin_z - meta.origin_z_nm) / grid.scale.dz)));
        const int sz1 = std::min(grid.dim_z - 1, static_cast<int>(std::ceil((smax_z - meta.origin_z_nm) / grid.scale.dz)));

        if (sx0 <= sx1 && sy0 <= sy1 && sz0 <= sz1) {
            const int bnx = sx1 - sx0 + 1;
            const int bny = sy1 - sy0 + 1;
            const int bnz = sz1 - sz0 + 1;
            const std::size_t bxy = static_cast<std::size_t>(bnx) * static_cast<std::size_t>(bny);
            const std::size_t btot = bxy * static_cast<std::size_t>(bnz);
            std::vector<float> soma_field(btot, inf);

            for (int lz = 0; lz < bnz; ++lz) {
                const float pz = meta.origin_z_nm + static_cast<float>(sz0 + lz) * grid.scale.dz;
                const std::size_t bz = static_cast<std::size_t>(lz) * bxy;
                for (int ly = 0; ly < bny; ++ly) {
                    const float py = meta.origin_y_nm + static_cast<float>(sy0 + ly) * grid.scale.dy;
                    const std::size_t by = bz + static_cast<std::size_t>(ly) * static_cast<std::size_t>(bnx);
                    for (int lx = 0; lx < bnx; ++lx) {
                        const float px = meta.origin_x_nm + static_cast<float>(sx0 + lx) * grid.scale.dx;
                        const Vec3f p{px, py, pz};
                        float best = inf;
                        for (std::size_t s = 0; s < soma_centers.size(); ++s) {
                            const float d = (p - soma_centers[s]).norm() - soma_radii[s];
                            if (d < best) best = d;
                        }
                        soma_field[by + static_cast<std::size_t>(lx)] = best;
                    }
                }
            }

            if (soma_centers.size() > 1) {
                std::vector<float> tmp(btot);
                for (int pass = 0; pass < 8; ++pass) {
                    for (int lz = 0; lz < bnz; ++lz) {
                        const std::size_t bz = static_cast<std::size_t>(lz) * bxy;
                        for (int ly = 0; ly < bny; ++ly) {
                            const std::size_t by = bz + static_cast<std::size_t>(ly) * static_cast<std::size_t>(bnx);
                            for (int lx = 0; lx < bnx; ++lx) {
                                const int xm = std::max(0, lx - 1);
                                const int xp = std::min(bnx - 1, lx + 1);
                                tmp[by + static_cast<std::size_t>(lx)] =
                                    0.25f * soma_field[by + static_cast<std::size_t>(xm)] +
                                    0.50f * soma_field[by + static_cast<std::size_t>(lx)] +
                                    0.25f * soma_field[by + static_cast<std::size_t>(xp)];
                            }
                        }
                    }
                    for (int lz = 0; lz < bnz; ++lz) {
                        const std::size_t bz = static_cast<std::size_t>(lz) * bxy;
                        for (int ly = 0; ly < bny; ++ly) {
                            const int ym = std::max(0, ly - 1);
                            const int yp = std::min(bny - 1, ly + 1);
                            const std::size_t by = bz + static_cast<std::size_t>(ly) * static_cast<std::size_t>(bnx);
                            const std::size_t bym = bz + static_cast<std::size_t>(ym) * static_cast<std::size_t>(bnx);
                            const std::size_t byp = bz + static_cast<std::size_t>(yp) * static_cast<std::size_t>(bnx);
                            for (int lx = 0; lx < bnx; ++lx) {
                                soma_field[by + static_cast<std::size_t>(lx)] =
                                    0.25f * tmp[bym + static_cast<std::size_t>(lx)] +
                                    0.50f * tmp[by + static_cast<std::size_t>(lx)] +
                                    0.25f * tmp[byp + static_cast<std::size_t>(lx)];
                            }
                        }
                    }
                    for (int lz = 0; lz < bnz; ++lz) {
                        const int zm = std::max(0, lz - 1);
                        const int zp = std::min(bnz - 1, lz + 1);
                        const std::size_t bz = static_cast<std::size_t>(lz) * bxy;
                        const std::size_t bzm = static_cast<std::size_t>(zm) * bxy;
                        const std::size_t bzp = static_cast<std::size_t>(zp) * bxy;
                        for (std::size_t p_i = 0; p_i < bxy; ++p_i) {
                            tmp[bz + p_i] =
                                0.25f * soma_field[bzm + p_i] +
                                0.50f * soma_field[bz + p_i] +
                                0.25f * soma_field[bzp + p_i];
                        }
                    }
                    soma_field.swap(tmp);
                }
            }

            for (int lz = 0; lz < bnz; ++lz) {
                const std::size_t gz = static_cast<std::size_t>(sz0 + lz) * xy_stride;
                const std::size_t bz = static_cast<std::size_t>(lz) * bxy;
                for (int ly = 0; ly < bny; ++ly) {
                    const std::size_t gy = gz + static_cast<std::size_t>(sy0 + ly) * static_cast<std::size_t>(grid.dim_x);
                    const std::size_t by = bz + static_cast<std::size_t>(ly) * static_cast<std::size_t>(bnx);
                    for (int lx = 0; lx < bnx; ++lx) {
                        const std::size_t idx = gy + static_cast<std::size_t>(sx0 + lx);
                        out_sdf[idx] = smin_poly(out_sdf[idx], soma_field[by + static_cast<std::size_t>(lx)], params.soma_blend_nm);
                    }
                }
            }
        }
    }

    for (std::size_t i = 0; i < total; ++i) {
        grid.mask[i] = (out_sdf[i] <= 0.0f) ? 1 : 0;
    }
}

bool rasterize_swc_auto_grid(const std::vector<SWCNode>& swc,
                             const VoxelizerParams& params,
                             VolumeGrid& out_grid,
                             VolumeMetadata& out_meta,
                             std::vector<float>& out_sdf) {
    if (swc.empty()) {
        return false;
    }

    float min_x = std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float min_z = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    float max_z = -std::numeric_limits<float>::infinity();

    for (const SWCNode& n : swc) {
        const float x = n.x * 1000.0f;
        const float y = n.y * 1000.0f;
        const float z = n.z * 1000.0f;
        const float r = n.radius * 1000.0f + params.padding_nm;
        min_x = std::min(min_x, x - r);
        min_y = std::min(min_y, y - r);
        min_z = std::min(min_z, z - r);
        max_x = std::max(max_x, x + r);
        max_y = std::max(max_y, y + r);
        max_z = std::max(max_z, z + r);
    }

    const int dim_x = std::max(4, static_cast<int>(std::ceil((max_x - min_x) / params.scale.dx)) + 2);
    const int dim_y = std::max(4, static_cast<int>(std::ceil((max_y - min_y) / params.scale.dy)) + 2);
    const int dim_z = std::max(4, static_cast<int>(std::ceil((max_z - min_z) / params.scale.dz)) + 2);

    out_grid = VolumeGrid(dim_x, dim_y, dim_z, params.scale);
    out_meta.origin_x_nm = min_x;
    out_meta.origin_y_nm = min_y;
    out_meta.origin_z_nm = min_z;
    out_meta.root_voxel = {
        static_cast<int>(std::round((swc[0].x * 1000.0f - min_x) / params.scale.dx)),
        static_cast<int>(std::round((swc[0].y * 1000.0f - min_y) / params.scale.dy)),
        static_cast<int>(std::round((swc[0].z * 1000.0f - min_z) / params.scale.dz))
    };

    rasterize_swc_into_grid(swc, params, out_grid, out_meta, out_sdf);
    return true;
}

bool rasterize_swc_roi_grid(const std::vector<SWCNode>& swc,
                            const VoxelizerParams& params,
                            const Vec3f& center_nm,
                            float half_span_nm,
                            VolumeGrid& out_grid,
                            VolumeMetadata& out_meta,
                            std::vector<float>& out_sdf) {
    if (swc.empty() || half_span_nm <= 0.0f) {
        return false;
    }

    const float min_x = center_nm.x - half_span_nm;
    const float min_y = center_nm.y - half_span_nm;
    const float min_z = center_nm.z - half_span_nm;
    const float span = 2.0f * half_span_nm;

    const int dim_x = std::max(4, static_cast<int>(std::ceil(span / params.scale.dx)) + 2);
    const int dim_y = std::max(4, static_cast<int>(std::ceil(span / params.scale.dy)) + 2);
    const int dim_z = std::max(4, static_cast<int>(std::ceil(span / params.scale.dz)) + 2);

    out_grid = VolumeGrid(dim_x, dim_y, dim_z, params.scale);
    out_meta.origin_x_nm = min_x;
    out_meta.origin_y_nm = min_y;
    out_meta.origin_z_nm = min_z;
    out_meta.root_voxel = {
        static_cast<int>(std::round((swc[0].x * 1000.0f - min_x) / params.scale.dx)),
        static_cast<int>(std::round((swc[0].y * 1000.0f - min_y) / params.scale.dy)),
        static_cast<int>(std::round((swc[0].z * 1000.0f - min_z) / params.scale.dz))
    };

    rasterize_swc_into_grid(swc, params, out_grid, out_meta, out_sdf);
    return true;
}

void smooth_mesh_taubin(TriangleMesh& mesh,
                        int iterations,
                        float lambda,
                        float mu) {
    const std::size_t nv = mesh.vertices.size();
    if (nv == 0 || iterations <= 0) {
        return;
    }

    std::vector<std::vector<uint32_t>> nb(nv);
    for (const auto& f : mesh.faces) {
        if (f[0] >= nv || f[1] >= nv || f[2] >= nv) continue;
        nb[f[0]].push_back(f[1]); nb[f[0]].push_back(f[2]);
        nb[f[1]].push_back(f[0]); nb[f[1]].push_back(f[2]);
        nb[f[2]].push_back(f[0]); nb[f[2]].push_back(f[1]);
    }
    for (std::size_t i = 0; i < nv; ++i) {
        auto& v = nb[i];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    std::vector<Vec3f> next(nv);
    auto step = [&](float factor) {
        for (std::size_t i = 0; i < nv; ++i) {
            const auto& v = nb[i];
            if (v.empty()) {
                next[i] = mesh.vertices[i];
                continue;
            }
            Vec3f avg{0.0f, 0.0f, 0.0f};
            for (uint32_t j : v) {
                avg += mesh.vertices[j];
            }
            avg *= (1.0f / static_cast<float>(v.size()));
            next[i] = mesh.vertices[i] + (avg - mesh.vertices[i]) * factor;
        }
        mesh.vertices.swap(next);
    };

    for (int it = 0; it < iterations; ++it) {
        step(lambda);
        step(mu);
    }
}

TriangleMesh extract_isosurface_mesh(const VolumeGrid& grid,
                                     const VolumeMetadata& meta,
                                     const std::vector<float>& sdf,
                                     int taubin_iters) {
    TriangleMesh mesh;
    if (grid.dim_x < 2 || grid.dim_y < 2 || grid.dim_z < 2) {
        return mesh;
    }

    const std::size_t nx = static_cast<std::size_t>(grid.dim_x);
    const std::size_t xy = nx * static_cast<std::size_t>(grid.dim_y);

    static constexpr int kTets[6][4] = {
        {0, 1, 3, 7},
        {0, 5, 1, 7},
        {0, 3, 2, 7},
        {0, 2, 6, 7},
        {0, 4, 5, 7},
        {0, 6, 4, 7}
    };
    static constexpr int kCornerOff[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };

    std::unordered_map<EdgePair, uint32_t, EdgePairHash> edge_to_vert;

    auto get_edge_vert = [&](uint64_t id_a, uint64_t id_b, const Vec3f& pa, const Vec3f& pb, float sa, float sb) -> uint32_t {
        EdgePair ep = (id_a < id_b) ? EdgePair{id_a, id_b} : EdgePair{id_b, id_a};
        const auto it = edge_to_vert.find(ep);
        if (it != edge_to_vert.end()) {
            return it->second;
        }
        float t = 0.5f;
        if (std::isfinite(sa) && std::isfinite(sb) && std::fabs(sa - sb) > 1e-6f) {
            t = std::clamp(sa / (sa - sb), 0.0f, 1.0f);
        } else if (!std::isfinite(sa) && std::isfinite(sb)) {
            t = 0.75f;
        } else if (std::isfinite(sa) && !std::isfinite(sb)) {
            t = 0.25f;
        }
        const Vec3f pos = pa + (pb - pa) * t;
        const uint32_t vid = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(pos);
        edge_to_vert.emplace(ep, vid);
        return vid;
    };

    auto emit_tri = [&](uint32_t v0, uint32_t v1, uint32_t v2, const Vec3f& outward_hint) {
        const Vec3f& p0 = mesh.vertices[v0];
        const Vec3f& p1 = mesh.vertices[v1];
        const Vec3f& p2 = mesh.vertices[v2];
        const Vec3f n = (p1 - p0).cross(p2 - p0);
        if (n.dot(outward_hint) < 0.0f) {
            mesh.faces.push_back({v0, v2, v1});
        } else {
            mesh.faces.push_back({v0, v1, v2});
        }
    };

    for (int z = 0; z < grid.dim_z - 1; ++z) {
        for (int y = 0; y < grid.dim_y - 1; ++y) {
            const std::size_t row = static_cast<std::size_t>(z) * xy + static_cast<std::size_t>(y) * nx;
            for (int x = 0; x < grid.dim_x - 1; ++x) {
                uint64_t cid[8];
                float val[8];
                int inside_cnt = 0;
                for (int c = 0; c < 8; ++c) {
                    const std::size_t idx = row + static_cast<std::size_t>(x + kCornerOff[c][0])
                                          + static_cast<std::size_t>(kCornerOff[c][1]) * nx
                                          + static_cast<std::size_t>(kCornerOff[c][2]) * xy;
                    cid[c] = static_cast<uint64_t>(idx);
                    val[c] = sdf[idx];
                    if (val[c] <= 0.0f) ++inside_cnt;
                }
                if (inside_cnt == 0 || inside_cnt == 8) {
                    continue;
                }

                Vec3f cpos[8];
                for (int c = 0; c < 8; ++c) {
                    cpos[c] = {
                        meta.origin_x_nm + static_cast<float>(x + kCornerOff[c][0]) * grid.scale.dx,
                        meta.origin_y_nm + static_cast<float>(y + kCornerOff[c][1]) * grid.scale.dy,
                        meta.origin_z_nm + static_cast<float>(z + kCornerOff[c][2]) * grid.scale.dz
                    };
                }

                for (int t = 0; t < 6; ++t) {
                    int in_v[4], out_v[4];
                    int ni = 0, no = 0;
                    for (int k = 0; k < 4; ++k) {
                        const int c = kTets[t][k];
                        if (val[c] <= 0.0f) in_v[ni++] = c;
                        else out_v[no++] = c;
                    }
                    if (ni == 0 || no == 0) continue;

                    if (ni == 1) {
                        const int i0 = in_v[0];
                        const uint32_t e0 = get_edge_vert(cid[i0], cid[out_v[0]], cpos[i0], cpos[out_v[0]], val[i0], val[out_v[0]]);
                        const uint32_t e1 = get_edge_vert(cid[i0], cid[out_v[1]], cpos[i0], cpos[out_v[1]], val[i0], val[out_v[1]]);
                        const uint32_t e2 = get_edge_vert(cid[i0], cid[out_v[2]], cpos[i0], cpos[out_v[2]], val[i0], val[out_v[2]]);
                        const Vec3f out_c = (cpos[out_v[0]] + cpos[out_v[1]] + cpos[out_v[2]]) * (1.0f / 3.0f);
                        emit_tri(e0, e1, e2, out_c - cpos[i0]);
                    } else if (ni == 3) {
                        const int o0 = out_v[0];
                        const uint32_t e0 = get_edge_vert(cid[in_v[0]], cid[o0], cpos[in_v[0]], cpos[o0], val[in_v[0]], val[o0]);
                        const uint32_t e1 = get_edge_vert(cid[in_v[1]], cid[o0], cpos[in_v[1]], cpos[o0], val[in_v[1]], val[o0]);
                        const uint32_t e2 = get_edge_vert(cid[in_v[2]], cid[o0], cpos[in_v[2]], cpos[o0], val[in_v[2]], val[o0]);
                        const Vec3f in_c = (cpos[in_v[0]] + cpos[in_v[1]] + cpos[in_v[2]]) * (1.0f / 3.0f);
                        emit_tri(e0, e1, e2, cpos[o0] - in_c);
                    } else {
                        const int i0 = in_v[0], i1 = in_v[1];
                        const int o0 = out_v[0], o1 = out_v[1];
                        const uint32_t e00 = get_edge_vert(cid[i0], cid[o0], cpos[i0], cpos[o0], val[i0], val[o0]);
                        const uint32_t e01 = get_edge_vert(cid[i0], cid[o1], cpos[i0], cpos[o1], val[i0], val[o1]);
                        const uint32_t e10 = get_edge_vert(cid[i1], cid[o0], cpos[i1], cpos[o0], val[i1], val[o0]);
                        const uint32_t e11 = get_edge_vert(cid[i1], cid[o1], cpos[i1], cpos[o1], val[i1], val[o1]);
                        const Vec3f hint = (cpos[o0] + cpos[o1]) * 0.5f - (cpos[i0] + cpos[i1]) * 0.5f;
                        emit_tri(e00, e01, e11, hint);
                        emit_tri(e00, e11, e10, hint);
                    }
                }
            }
        }
    }

    if (taubin_iters > 0) {
        smooth_mesh_taubin(mesh, taubin_iters, 0.50f, -0.53f);
    }
    return mesh;
}

namespace {

Vec3f ortho_basis(const Vec3f& t) {
    const Vec3f ref = (std::fabs(t.x) < 0.8f) ? Vec3f{1.0f, 0.0f, 0.0f} : Vec3f{0.0f, 1.0f, 0.0f};
    return t.cross(ref).normalized();
}

Vec3f transport_frame(const Vec3f& u, const Vec3f& t0, const Vec3f& t1) {
    const Vec3f axis = t0.cross(t1);
    const float sin_a = axis.norm();
    const float cos_a = std::clamp(t0.dot(t1), -1.0f, 1.0f);
    if (sin_a < 1e-5f) {
        Vec3f p = u - t1 * u.dot(t1);
        const float l = p.norm();
        return (l > 1e-5f) ? (p * (1.0f / l)) : ortho_basis(t1);
    }
    const Vec3f k = axis * (1.0f / sin_a);
    const Vec3f rot = u * cos_a + k.cross(u) * sin_a + k * (k.dot(u) * (1.0f - cos_a));
    Vec3f p = rot - t1 * rot.dot(t1);
    const float l = p.norm();
    return (l > 1e-5f) ? (p * (1.0f / l)) : ortho_basis(t1);
}

void append_sphere(TriangleMesh& mesh, const Vec3f& center, float radius, int segs, int rings) {
    if (radius <= 1.0f) return;
    static constexpr float kPi = 3.14159265358979323846f;
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());

    mesh.vertices.push_back({center.x, center.y + radius, center.z});
    for (int r = 1; r < rings; ++r) {
        const float phi = kPi * static_cast<float>(r) / static_cast<float>(rings);
        const float sp = std::sin(phi);
        const float cp = std::cos(phi);
        for (int s = 0; s < segs; ++s) {
            const float th = 2.0f * kPi * static_cast<float>(s) / static_cast<float>(segs);
            mesh.vertices.push_back({
                center.x + radius * sp * std::cos(th),
                center.y + radius * cp,
                center.z + radius * sp * std::sin(th)
            });
        }
    }
    const uint32_t south = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({center.x, center.y - radius, center.z});

    for (int s = 0; s < segs; ++s) {
        const uint32_t a = base + 1 + static_cast<uint32_t>(s);
        const uint32_t b = base + 1 + static_cast<uint32_t>((s + 1) % segs);
        mesh.faces.push_back({base, b, a});
    }
    for (int r = 0; r < rings - 2; ++r) {
        const uint32_t row0 = base + 1 + static_cast<uint32_t>(r * segs);
        const uint32_t row1 = base + 1 + static_cast<uint32_t>((r + 1) * segs);
        for (int s = 0; s < segs; ++s) {
            const uint32_t sn = static_cast<uint32_t>((s + 1) % segs);
            mesh.faces.push_back({row0 + static_cast<uint32_t>(s), row0 + sn, row1 + sn});
            mesh.faces.push_back({row0 + static_cast<uint32_t>(s), row1 + sn, row1 + static_cast<uint32_t>(s)});
        }
    }
    const uint32_t last_row = base + 1 + static_cast<uint32_t>((rings - 2) * segs);
    for (int s = 0; s < segs; ++s) {
        const uint32_t a = last_row + static_cast<uint32_t>(s);
        const uint32_t b = last_row + static_cast<uint32_t>((s + 1) % segs);
        mesh.faces.push_back({south, a, b});
    }
}

void append_mesh(TriangleMesh& dst, const TriangleMesh& src) {
    const uint32_t off = static_cast<uint32_t>(dst.vertices.size());
    dst.vertices.insert(dst.vertices.end(), src.vertices.begin(), src.vertices.end());
    dst.faces.reserve(dst.faces.size() + src.faces.size());
    for (const auto& f : src.faces) {
        dst.faces.push_back({f[0] + off, f[1] + off, f[2] + off});
    }
}

}

TriangleMesh build_hybrid_neuron_mesh(const std::vector<SWCNode>& swc,
                                      const VoxelizerParams& params) {
    TriangleMesh out_mesh;
    if (swc.empty()) {
        return out_mesh;
    }

    const std::size_t n = swc.size();
    std::unordered_map<int, std::size_t> id_to_idx;
    id_to_idx.reserve(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
        id_to_idx[swc[i].id] = i;
    }

    std::vector<int> par_idx(n, -1);
    std::vector<std::vector<std::size_t>> children(n);
    std::vector<SWCNode> soma_swc;
    Vec3f soma_mn{std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::infinity()};
    Vec3f soma_mx{-std::numeric_limits<float>::infinity(),
                  -std::numeric_limits<float>::infinity(),
                  -std::numeric_limits<float>::infinity()};

    for (std::size_t i = 0; i < n; ++i) {
        if (swc[i].parent_id != -1) {
            const auto it = id_to_idx.find(swc[i].parent_id);
            if (it != id_to_idx.end()) {
                par_idx[i] = static_cast<int>(it->second);
                children[it->second].push_back(i);
            }
        }
        if (swc[i].type == 1 || par_idx[i] == -1) {
            soma_swc.push_back(swc[i]);
            const float x = swc[i].x * 1000.0f;
            const float y = swc[i].y * 1000.0f;
            const float z = swc[i].z * 1000.0f;
            const float r = swc[i].radius * 1000.0f;
            soma_mn.x = std::min(soma_mn.x, x - r); soma_mx.x = std::max(soma_mx.x, x + r);
            soma_mn.y = std::min(soma_mn.y, y - r); soma_mx.y = std::max(soma_mx.y, y + r);
            soma_mn.z = std::min(soma_mn.z, z - r); soma_mx.z = std::max(soma_mx.z, z + r);
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (swc[i].type != 1 && par_idx[i] >= 0) {
            const std::size_t p = static_cast<std::size_t>(par_idx[i]);
            if (swc[p].type == 1 || par_idx[p] == -1) {
                soma_swc.push_back(swc[i]);
                const float x = swc[i].x * 1000.0f;
                const float y = swc[i].y * 1000.0f;
                const float z = swc[i].z * 1000.0f;
                const float r = swc[i].radius * 1000.0f;
                soma_mn.x = std::min(soma_mn.x, x - r); soma_mx.x = std::max(soma_mx.x, x + r);
                soma_mn.y = std::min(soma_mn.y, y - r); soma_mx.y = std::max(soma_mx.y, y + r);
                soma_mn.z = std::min(soma_mn.z, z - r); soma_mx.z = std::max(soma_mx.z, z + r);
            }
        }
    }

    if (!soma_swc.empty()) {
        VoxelizerParams sp = params;
        const Vec3f sc = (soma_mn + soma_mx) * 0.5f;
        const float raw_hs = std::max({soma_mx.x - soma_mn.x, soma_mx.y - soma_mn.y, soma_mx.z - soma_mn.z}) * 0.5f;
        const float sv = std::max({120.0f, params.soma_voxel_nm, raw_hs / 80.0f});
        sp.scale = {sv, sv, sv};
        sp.padding_nm = params.soma_blend_nm + 4.0f * sv;
        sp.min_radius_scale = 0.0f;

        const float hs = raw_hs + sp.padding_nm;
        VolumeGrid sgrid;
        VolumeMetadata smeta;
        std::vector<float> ssdf;
        if (rasterize_swc_roi_grid(soma_swc, sp, sc, hs, sgrid, smeta, ssdf)) {
            const TriangleMesh soma_mesh = extract_isosurface_mesh(sgrid, smeta, ssdf, 4);
            append_mesh(out_mesh, soma_mesh);
        }
    }

    const int segs = std::max(6, params.tube_sides);
    static constexpr float kPi = 3.14159265358979323846f;
    std::vector<float> cos_t(static_cast<std::size_t>(segs));
    std::vector<float> sin_t(static_cast<std::size_t>(segs));
    for (int s = 0; s < segs; ++s) {
        const float a = 2.0f * kPi * static_cast<float>(s) / static_cast<float>(segs);
        cos_t[static_cast<std::size_t>(s)] = std::cos(a);
        sin_t[static_cast<std::size_t>(s)] = std::sin(a);
    }

    auto node_pos = [&swc](std::size_t idx) -> Vec3f {
        return {swc[idx].x * 1000.0f, swc[idx].y * 1000.0f, swc[idx].z * 1000.0f};
    };

    auto is_soma_node = [&swc, &par_idx](std::size_t idx) -> bool {
        return swc[idx].type == 1 || par_idx[idx] == -1;
    };

    auto emit_chain = [&](const std::vector<std::size_t>& chain) {
        if (chain.size() < 2) return;
        std::vector<Vec3f> pts;
        std::vector<float> rads;
        pts.reserve(chain.size());
        rads.reserve(chain.size());

        for (std::size_t k = 0; k < chain.size(); ++k) {
            const Vec3f p = node_pos(chain[k]);
            float r = std::max(90.0f, swc[chain[k]].radius * 1000.0f);
            if (k == 0 && is_soma_node(chain[0]) && chain.size() >= 2) {
                const float r1 = std::max(90.0f, swc[chain[1]].radius * 1000.0f);
                r = std::min(r, r1 * 1.35f);
            }
            if (!pts.empty() && (p - pts.back()).norm_sq() < 1.0f) {
                rads.back() = std::max(rads.back(), r);
                continue;
            }
            pts.push_back(p);
            rads.push_back(r);
        }
        if (pts.size() < 2) return;

        const std::size_t m = pts.size();
        std::vector<Vec3f> seg_dir(m - 1);
        for (std::size_t k = 0; k + 1 < m; ++k) {
            seg_dir[k] = (pts[k + 1] - pts[k]).normalized();
        }

        Vec3f u = ortho_basis(seg_dir[0]);
        Vec3f prev_t = seg_dir[0];
        const uint32_t base_v = static_cast<uint32_t>(out_mesh.vertices.size());

        for (std::size_t k = 0; k < m; ++k) {
            Vec3f t;
            if (k == 0) {
                t = seg_dir[0];
            } else if (k + 1 == m) {
                t = seg_dir[m - 2];
            } else {
                const Vec3f sum = seg_dir[k - 1] + seg_dir[k];
                t = (sum.norm_sq() > 1e-5f) ? sum.normalized() : seg_dir[k];
            }
            if (k > 0) {
                u = transport_frame(u, prev_t, t);
            }
            prev_t = t;
            const Vec3f v = t.cross(u).normalized();
            const float r = rads[k];
            const Vec3f& c = pts[k];
            for (int s = 0; s < segs; ++s) {
                const Vec3f off = u * (cos_t[static_cast<std::size_t>(s)] * r) +
                                  v * (sin_t[static_cast<std::size_t>(s)] * r);
                out_mesh.vertices.push_back(c + off);
            }
        }

        for (std::size_t k = 0; k + 1 < m; ++k) {
            const uint32_t r0 = base_v + static_cast<uint32_t>(k * static_cast<std::size_t>(segs));
            const uint32_t r1 = base_v + static_cast<uint32_t>((k + 1) * static_cast<std::size_t>(segs));
            for (int s = 0; s < segs; ++s) {
                const uint32_t sn = static_cast<uint32_t>((s + 1) % segs);
                out_mesh.faces.push_back({r0 + static_cast<uint32_t>(s), r1 + sn, r0 + sn});
                out_mesh.faces.push_back({r0 + static_cast<uint32_t>(s), r1 + static_cast<uint32_t>(s), r1 + sn});
            }
        }
    };

    for (std::size_t i = 0; i < n; ++i) {
        const bool start_node = is_soma_node(i) || children[i].size() >= 2;
        if (!start_node) continue;

        for (std::size_t ch : children[i]) {
            if (is_soma_node(ch)) continue;
            std::vector<std::size_t> chain;
            chain.push_back(i);
            std::size_t curr = ch;
            chain.push_back(curr);
            while (children[curr].size() == 1 && !is_soma_node(children[curr][0])) {
                curr = children[curr][0];
                chain.push_back(curr);
            }
            emit_chain(chain);
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (is_soma_node(i)) continue;
        if (children[i].empty() || children[i].size() >= 2) {
            const float r = std::max(90.0f, swc[i].radius * 1000.0f);
            append_sphere(out_mesh, node_pos(i), r, 8, 4);
        }
    }

    return out_mesh;
}

bool export_mesh_obj(const std::string& filepath, const TriangleMesh& mesh) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        return false;
    }

    out << std::fixed << std::setprecision(4);
    for (const Vec3f& v : mesh.vertices) {
        out << "v " << (v.x * 1e-3f) << ' ' << (v.y * 1e-3f) << ' ' << (v.z * 1e-3f) << '\n';
    }
    for (const auto& f : mesh.faces) {
        out << "f " << (f[0] + 1) << ' ' << (f[1] + 1) << ' ' << (f[2] + 1) << '\n';
    }
    return out.good();
}

namespace {

std::string base64_encode(const uint8_t* data, std::size_t len) {
    static constexpr char kTbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (std::size_t i = 0; i < len; i += 3) {
        const uint32_t b0 = data[i];
        const uint32_t b1 = (i + 1 < len) ? data[i + 1] : 0;
        const uint32_t b2 = (i + 2 < len) ? data[i + 2] : 0;
        const uint32_t tri = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kTbl[(tri >> 18) & 63]);
        out.push_back(kTbl[(tri >> 12) & 63]);
        out.push_back((i + 1 < len) ? kTbl[(tri >> 6) & 63] : '=');
        out.push_back((i + 2 < len) ? kTbl[tri & 63] : '=');
    }
    return out;
}

void write_packed_mesh_js(std::ofstream& out,
                          const char* var_name,
                          const TriangleMesh& mesh,
                          const Vec3f& center,
                          float inv_half_span) {
    const std::size_t nv = mesh.vertices.size();
    const std::size_t nf = mesh.faces.size();
    std::vector<Vec3f> fnorm;
    const std::vector<Vec3f> vnorm = compute_vertex_normals(mesh, &fnorm);

    std::vector<int16_t> pos_q(nv * 3);
    std::vector<int8_t> norm_q(nv * 3);
    for (std::size_t i = 0; i < nv; ++i) {
        const float nx = std::clamp((mesh.vertices[i].x - center.x) * inv_half_span, -1.0f, 1.0f);
        const float ny = std::clamp((mesh.vertices[i].y - center.y) * inv_half_span, -1.0f, 1.0f);
        const float nz = std::clamp((mesh.vertices[i].z - center.z) * inv_half_span, -1.0f, 1.0f);
        pos_q[i * 3 + 0] = static_cast<int16_t>(std::round(nx * 32767.0f));
        pos_q[i * 3 + 1] = static_cast<int16_t>(std::round(ny * 32767.0f));
        pos_q[i * 3 + 2] = static_cast<int16_t>(std::round(nz * 32767.0f));
        norm_q[i * 3 + 0] = static_cast<int8_t>(std::round(std::clamp(vnorm[i].x, -1.0f, 1.0f) * 127.0f));
        norm_q[i * 3 + 1] = static_cast<int8_t>(std::round(std::clamp(vnorm[i].y, -1.0f, 1.0f) * 127.0f));
        norm_q[i * 3 + 2] = static_cast<int8_t>(std::round(std::clamp(vnorm[i].z, -1.0f, 1.0f) * 127.0f));
    }

    std::vector<uint32_t> idx(nf * 3);
    for (std::size_t i = 0; i < nf; ++i) {
        idx[i * 3 + 0] = mesh.faces[i][0];
        idx[i * 3 + 1] = mesh.faces[i][1];
        idx[i * 3 + 2] = mesh.faces[i][2];
    }

    out << "const " << var_name << "={nv:" << nv << ",nf:" << nf << ",pos:\"";
    out << base64_encode(reinterpret_cast<const uint8_t*>(pos_q.data()), pos_q.size() * sizeof(int16_t));
    out << "\",nrm:\"";
    out << base64_encode(reinterpret_cast<const uint8_t*>(norm_q.data()), norm_q.size() * sizeof(int8_t));
    out << "\",idx:\"";
    out << base64_encode(reinterpret_cast<const uint8_t*>(idx.data()), idx.size() * sizeof(uint32_t));
    out << "\"};\n";
}

}

bool export_mesh_webgl_html(const std::string& filepath,
                            const TriangleMesh& recon_mesh,
                            const std::vector<SWCNode>& swc,
                            const TriangleMesh* ref_mesh) {
    if (recon_mesh.vertices.empty() || recon_mesh.faces.empty()) {
        return false;
    }

    std::ofstream out(filepath);
    if (!out.is_open()) {
        return false;
    }

    Vec3f mn{std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::infinity()};
    Vec3f mx{-std::numeric_limits<float>::infinity(),
             -std::numeric_limits<float>::infinity(),
             -std::numeric_limits<float>::infinity()};

    for (const Vec3f& v : recon_mesh.vertices) {
        mn.x = std::min(mn.x, v.x); mn.y = std::min(mn.y, v.y); mn.z = std::min(mn.z, v.z);
        mx.x = std::max(mx.x, v.x); mx.y = std::max(mx.y, v.y); mx.z = std::max(mx.z, v.z);
    }
    if (ref_mesh && !ref_mesh->vertices.empty()) {
        for (const Vec3f& v : ref_mesh->vertices) {
            mn.x = std::min(mn.x, v.x); mn.y = std::min(mn.y, v.y); mn.z = std::min(mn.z, v.z);
            mx.x = std::max(mx.x, v.x); mx.y = std::max(mx.y, v.y); mx.z = std::max(mx.z, v.z);
        }
    }

    const Vec3f center = (mn + mx) * 0.5f;
    const float half_span = std::max({1.0f, (mx.x - mn.x) * 0.5f, (mx.y - mn.y) * 0.5f, (mx.z - mn.z) * 0.5f});
    const float inv_half = 1.0f / half_span;

    Vec3f soma_norm{0.0f, 0.0f, 0.0f};
    if (!swc.empty()) {
        soma_norm = {
            (swc[0].x * 1000.0f - center.x) * inv_half,
            -(swc[0].y * 1000.0f - center.y) * inv_half,
            (swc[0].z * 1000.0f - center.z) * inv_half
        };
    }

    std::unordered_map<int, std::size_t> id_map;
    id_map.reserve(swc.size() * 2);
    for (std::size_t i = 0; i < swc.size(); ++i) {
        id_map[swc[i].id] = i;
    }
    std::vector<int16_t> skel_q;
    skel_q.reserve(swc.size() * 6);
    for (const SWCNode& n : swc) {
        if (n.parent_id == -1) continue;
        const auto it = id_map.find(n.parent_id);
        if (it == id_map.end()) continue;
        const SWCNode& p = swc[it->second];
        const float ax = std::clamp((p.x * 1000.0f - center.x) * inv_half, -1.0f, 1.0f);
        const float ay = std::clamp((p.y * 1000.0f - center.y) * inv_half, -1.0f, 1.0f);
        const float az = std::clamp((p.z * 1000.0f - center.z) * inv_half, -1.0f, 1.0f);
        const float bx = std::clamp((n.x * 1000.0f - center.x) * inv_half, -1.0f, 1.0f);
        const float by = std::clamp((n.y * 1000.0f - center.y) * inv_half, -1.0f, 1.0f);
        const float bz = std::clamp((n.z * 1000.0f - center.z) * inv_half, -1.0f, 1.0f);
        skel_q.push_back(static_cast<int16_t>(std::round(ax * 32767.0f)));
        skel_q.push_back(static_cast<int16_t>(std::round(ay * 32767.0f)));
        skel_q.push_back(static_cast<int16_t>(std::round(az * 32767.0f)));
        skel_q.push_back(static_cast<int16_t>(std::round(bx * 32767.0f)));
        skel_q.push_back(static_cast<int16_t>(std::round(by * 32767.0f)));
        skel_q.push_back(static_cast<int16_t>(std::round(bz * 32767.0f)));
    }

    out << "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        << "<title>wintermute 3d mesh viewer</title>"
        << "<style>"
        << "html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#070a0f;color:#d8e6f2;font-family:monospace;}"
        << "canvas{display:block;width:100%;height:100%;cursor:grab;}canvas:active{cursor:grabbing;}"
        << "#ui{position:fixed;top:12px;left:12px;background:rgba(10,15,24,0.88);border:1px solid #24354c;border-radius:8px;padding:10px 12px;font-size:12px;display:flex;flex-direction:column;gap:7px;z-index:10;max-width:300px;}"
        << ".row{display:flex;gap:6px;flex-wrap:wrap;}"
        << "button,label.btn{background:#152233;color:#cfe7ff;border:1px solid #2d4666;border-radius:5px;padding:4px 8px;font-family:monospace;font-size:11px;cursor:pointer;}"
        << "button:hover,label.btn:hover{background:#1f334d;}button.on{background:#1b5a7a;border-color:#4ec5ff;color:#fff;}"
        << "input[type=file]{display:none;}#info{color:#8aa4bd;font-size:11px;}"
        << "</style></head><body><canvas id=\"c\"></canvas>"
        << "<div id=\"ui\">"
        << "<div style=\"color:#55d0ff;font-weight:bold;\">wintermute 3d obj viewer</div>"
        << "<div id=\"info\"></div>"
        << "<div class=\"row\">"
        << "<button id=\"bRecon\" class=\"on\">wintermute mesh</button>"
        << "<button id=\"bRef\">em mesh</button>"
        << "<button id=\"bSkel\">swc skeleton</button>"
        << "<button id=\"bXray\">x-ray</button>"
        << "</div><div class=\"row\">"
        << "<button id=\"bSoma\">focus soma</button>"
        << "<button id=\"bReset\">reset view</button>"
        << "<label class=\"btn\">load .obj<input id=\"fObj\" type=\"file\" accept=\".obj\"></label>"
        << "</div><div style=\"color:#6c849c;font-size:10px;\">drag: orbit | shift+drag/right: pan | scroll: zoom | drop .obj</div></div>"
        << "<script>\n";

    out << std::fixed << std::setprecision(5);
    out << "const SOMA=[" << soma_norm.x << "," << soma_norm.y << "," << soma_norm.z << "];\n";
    out << "const SPAN_UM=" << (half_span * 2e-3f) << ";\n";
    write_packed_mesh_js(out, "RECON", recon_mesh, center, inv_half);
    if (ref_mesh && !ref_mesh->vertices.empty() && !ref_mesh->faces.empty()) {
        write_packed_mesh_js(out, "REFM", *ref_mesh, center, inv_half);
    } else {
        out << "const REFM=null;\n";
    }
    out << "const SKEL_N=" << (skel_q.size() / 3) << ",SKEL_B64=\""
        << base64_encode(reinterpret_cast<const uint8_t*>(skel_q.data()), skel_q.size() * sizeof(int16_t))
        << "\";\n";

    out << R"JS(
const canvas=document.getElementById('c');
const gl=canvas.getContext('webgl2',{antialias:true,alpha:false});
if(!gl){document.body.innerHTML='<p style="padding:20px">webgl2 required</p>';throw new Error('no webgl2');}
function b64ToBuf(s){const bin=atob(s);const n=bin.length;const u=new Uint8Array(n);for(let i=0;i<n;++i)u[i]=bin.charCodeAt(i);return u.buffer;}
function mkShader(type,src){const sh=gl.createShader(type);gl.shaderSource(sh,src);gl.compileShader(sh);return sh;}
function mkProg(vs,fs){const p=gl.createProgram();gl.attachShader(p,mkShader(gl.VERTEX_SHADER,vs));gl.attachShader(p,mkShader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);return p;}
const meshProg=mkProg(`#version 300 es
in vec3 aPos;in vec3 aNrm;
uniform mat4 uMVP;uniform mat3 uNrmMat;
out vec3 vNrm;
void main(){vec3 p=vec3(aPos.x,-aPos.y,aPos.z);vec3 n=vec3(aNrm.x,-aNrm.y,aNrm.z);vNrm=normalize(uNrmMat*n);gl_Position=uMVP*vec4(p,1.0);}`,
`#version 300 es
precision highp float;
in vec3 vNrm;uniform vec3 uColor;uniform float uAlpha;
out vec4 oCol;
void main(){vec3 n=normalize(vNrm);if(!gl_FrontFacing)n=-n;
vec3 l1=normalize(vec3(0.45,0.65,0.61));vec3 l2=normalize(vec3(-0.55,-0.35,0.75));vec3 v=vec3(0.0,0.0,1.0);
float d1=max(dot(n,l1),0.0);float d2=max(dot(n,l2),0.0);
float sp=pow(max(dot(n,normalize(l1+v)),0.0),28.0);
vec3 col=uColor*(0.20+0.72*d1+0.22*d2)+vec3(0.24*sp);
oCol=vec4(clamp(col,0.0,1.0),uAlpha);}`);
const lineProg=mkProg(`#version 300 es
in vec3 aPos;uniform mat4 uMVP;
void main(){gl_Position=uMVP*vec4(aPos.x,-aPos.y,aPos.z,1.0);}`,
`#version 300 es
precision highp float;uniform vec3 uColor;out vec4 oCol;
void main(){oCol=vec4(uColor,1.0);}`);
function buildMeshGPU(m){
if(!m)return null;
const pos16=new Int16Array(b64ToBuf(m.pos));const nrm8=new Int8Array(b64ToBuf(m.nrm));const idx32=new Uint32Array(b64ToBuf(m.idx));
const vao=gl.createVertexArray();gl.bindVertexArray(vao);
const pb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.bufferData(gl.ARRAY_BUFFER,pos16,gl.STATIC_DRAW);
const locP=gl.getAttribLocation(meshProg,'aPos');gl.enableVertexAttribArray(locP);gl.vertexAttribPointer(locP,3,gl.SHORT,true,0,0);
const nb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,nb);gl.bufferData(gl.ARRAY_BUFFER,nrm8,gl.STATIC_DRAW);
const locN=gl.getAttribLocation(meshProg,'aNrm');gl.enableVertexAttribArray(locN);gl.vertexAttribPointer(locN,3,gl.BYTE,true,0,0);
const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,idx32,gl.STATIC_DRAW);
gl.bindVertexArray(null);
return {vao,count:idx32.length,nv:m.nv,nf:m.nf};}
function buildFloatMeshGPU(posF,nrmF,idx32){
const vao=gl.createVertexArray();gl.bindVertexArray(vao);
const pb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.bufferData(gl.ARRAY_BUFFER,posF,gl.STATIC_DRAW);
const locP=gl.getAttribLocation(meshProg,'aPos');gl.enableVertexAttribArray(locP);gl.vertexAttribPointer(locP,3,gl.FLOAT,false,0,0);
const nb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,nb);gl.bufferData(gl.ARRAY_BUFFER,nrmF,gl.STATIC_DRAW);
const locN=gl.getAttribLocation(meshProg,'aNrm');gl.enableVertexAttribArray(locN);gl.vertexAttribPointer(locN,3,gl.FLOAT,false,0,0);
const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,idx32,gl.STATIC_DRAW);
gl.bindVertexArray(null);
return {vao,count:idx32.length,nv:posF.length/3,nf:idx32.length/3};}
let gpuRecon=buildMeshGPU(RECON);
let gpuRef=buildMeshGPU(REFM);
let gpuSkel=null;
if(SKEL_N>0){
const s16=new Int16Array(b64ToBuf(SKEL_B64));
const vao=gl.createVertexArray();gl.bindVertexArray(vao);
const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,s16,gl.STATIC_DRAW);
const loc=gl.getAttribLocation(lineProg,'aPos');gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.SHORT,true,0,0);
gl.bindVertexArray(null);gpuSkel={vao,count:SKEL_N};}
let showRecon=true,showRef=false,showSkel=false,xray=false;
let yaw=0.35,pitch=0.12,dist=2.3,tx=SOMA[0],ty=SOMA[1],tz=SOMA[2];
const bRecon=document.getElementById('bRecon'),bRef=document.getElementById('bRef'),bSkel=document.getElementById('bSkel'),bXray=document.getElementById('bXray');
if(!gpuRef){bRef.style.display='none';}
if(!gpuSkel){bSkel.style.display='none';}
function updInfo(){document.getElementById('info').textContent=`verts=${gpuRecon.nv.toLocaleString()} faces=${gpuRecon.nf.toLocaleString()} span=${SPAN_UM.toFixed(1)} um`;}
updInfo();
bRecon.onclick=()=>{showRecon=!showRecon;bRecon.classList.toggle('on',showRecon);draw();};
bRef.onclick=()=>{showRef=!showRef;bRef.classList.toggle('on',showRef);draw();};
bSkel.onclick=()=>{showSkel=!showSkel;bSkel.classList.toggle('on',showSkel);draw();};
bXray.onclick=()=>{xray=!xray;bXray.classList.toggle('on',xray);draw();};
document.getElementById('bSoma').onclick=()=>{tx=SOMA[0];ty=SOMA[1];tz=SOMA[2];dist=0.55;draw();};
document.getElementById('bReset').onclick=()=>{tx=0;ty=0;tz=0;yaw=0.35;pitch=0.12;dist=2.3;draw();};
function parseOBJ(txt){
const vs=[],fs=[];const lines=txt.split('\n');
for(let i=0;i<lines.length;++i){
const l=lines[i].trim();
if(l.startsWith('v ')){const p=l.split(/\s+/);vs.push(parseFloat(p[1]),parseFloat(p[2]),parseFloat(p[3]));}
else if(l.startsWith('f ')){const p=l.split(/\s+/);const a=parseInt(p[1])-1;for(let k=2;k+1<p.length;++k)fs.push(a,parseInt(p[k])-1,parseInt(p[k+1])-1);}}
const nv=vs.length/3;if(nv===0)return;
let mnx=1e30,mny=1e30,mnz=1e30,mxx=-1e30,mxy=-1e30,mxz=-1e30;
for(let i=0;i<vs.length;i+=3){mnx=Math.min(mnx,vs[i]);mxx=Math.max(mxx,vs[i]);mny=Math.min(mny,vs[i+1]);mxy=Math.max(mxy,vs[i+1]);mnz=Math.min(mnz,vs[i+2]);mxz=Math.max(mxz,vs[i+2]);}
const cx=(mnx+mxx)*0.5,cy=(mny+mxy)*0.5,cz=(mnz+mxz)*0.5;
const hs=Math.max(1e-4,(mxx-mnx)*0.5,(mxy-mny)*0.5,(mxz-mnz)*0.5);
const pos=new Float32Array(vs.length),nrm=new Float32Array(vs.length);
for(let i=0;i<vs.length;i+=3){pos[i]=(vs[i]-cx)/hs;pos[i+1]=(vs[i+1]-cy)/hs;pos[i+2]=(vs[i+2]-cz)/hs;}
const idx=new Uint32Array(fs);
for(let i=0;i<idx.length;i+=3){
const i0=idx[i]*3,i1=idx[i+1]*3,i2=idx[i+2]*3;
const e1x=pos[i1]-pos[i0],e1y=pos[i1+1]-pos[i0+1],e1z=pos[i1+2]-pos[i0+2];
const e2x=pos[i2]-pos[i0],e2y=pos[i2+1]-pos[i0+1],e2z=pos[i2+2]-pos[i0+2];
const nx=e1y*e2z-e1z*e2y,ny=e1z*e2x-e1x*e2z,nz=e1x*e2y-e1y*e2x;
nrm[i0]+=nx;nrm[i0+1]+=ny;nrm[i0+2]+=nz;nrm[i1]+=nx;nrm[i1+1]+=ny;nrm[i1+2]+=nz;nrm[i2]+=nx;nrm[i2+1]+=ny;nrm[i2+2]+=nz;}
for(let i=0;i<nrm.length;i+=3){const l=Math.hypot(nrm[i],nrm[i+1],nrm[i+2])||1;nrm[i]/=l;nrm[i+1]/=l;nrm[i+2]/=l;}
gpuRecon=buildFloatMeshGPU(pos,nrm,idx);showRecon=true;bRecon.classList.add('on');tx=0;ty=0;tz=0;dist=2.3;updInfo();draw();}
document.getElementById('fObj').onchange=e=>{const f=e.target.files[0];if(f)f.text().then(parseOBJ);};
window.ondragover=e=>e.preventDefault();
window.ondrop=e=>{e.preventDefault();const f=e.dataTransfer.files[0];if(f)f.text().then(parseOBJ);};
function mul4(a,b){const r=new Float32Array(16);for(let c=0;c<4;++c)for(let ro=0;ro<4;++ro){r[c*4+ro]=a[ro]*b[c*4]+a[4+ro]*b[c*4+1]+a[8+ro]*b[c*4+2]+a[12+ro]*b[c*4+3];}return r;}
function persp(fov,asp,n,f){const t=1/Math.tan(fov*0.5);const m=new Float32Array(16);m[0]=t/asp;m[5]=t;m[10]=(f+n)/(n-f);m[11]=-1;m[14]=(2*f*n)/(n-f);return m;}
function viewMat(){
const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
const r00=cy,r01=0,r02=sy,r10=sy*sp,r11=cp,r12=-cy*sp,r20=-sy*cp,r21=sp,r22=cy*cp;
const px=-(r00*tx+r01*ty+r02*tz),py=-(r10*tx+r11*ty+r12*tz),pz=-(r20*tx+r21*ty+r22*tz)-dist;
const m=new Float32Array([r00,r10,r20,0, r01,r11,r21,0, r02,r12,r22,0, px,py,pz,1]);
const nm=new Float32Array([r00,r10,r20, r01,r11,r21, r02,r12,r22]);
return {m,nm};}
function draw(){
const dpr=Math.min(window.devicePixelRatio||1,2);
const w=Math.floor(canvas.clientWidth*dpr),h=Math.floor(canvas.clientHeight*dpr);
if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
gl.viewport(0,0,w,h);gl.clearColor(0.025,0.035,0.055,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
const proj=persp(0.75,w/Math.max(1,h),0.01,100.0);
const {m:vm,nm}=viewMat();const mvp=mul4(proj,vm);
gl.enable(gl.DEPTH_TEST);
if(xray){gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);}
else{gl.disable(gl.BLEND);gl.depthMask(true);}
gl.useProgram(meshProg);
gl.uniformMatrix4fv(gl.getUniformLocation(meshProg,'uMVP'),false,mvp);
gl.uniformMatrix3fv(gl.getUniformLocation(meshProg,'uNrmMat'),false,nm);
gl.uniform1f(gl.getUniformLocation(meshProg,'uAlpha'),xray?0.35:1.0);
if(showRef&&gpuRef){gl.uniform3f(gl.getUniformLocation(meshProg,'uColor'),0.22,0.90,0.42);gl.bindVertexArray(gpuRef.vao);gl.drawElements(gl.TRIANGLES,gpuRef.count,gl.UNSIGNED_INT,0);}
if(showRecon&&gpuRecon){gl.uniform3f(gl.getUniformLocation(meshProg,'uColor'),0.24,0.78,0.98);gl.bindVertexArray(gpuRecon.vao);gl.drawElements(gl.TRIANGLES,gpuRecon.count,gl.UNSIGNED_INT,0);}
gl.depthMask(true);gl.disable(gl.BLEND);
if(showSkel&&gpuSkel){gl.useProgram(lineProg);gl.uniformMatrix4fv(gl.getUniformLocation(lineProg,'uMVP'),false,mvp);gl.uniform3f(gl.getUniformLocation(lineProg,'uColor'),1.0,0.75,0.22);gl.bindVertexArray(gpuSkel.vao);gl.drawArrays(gl.LINES,0,gpuSkel.count);}
}
let drag=0,lx=0,ly=0;
canvas.oncontextmenu=e=>e.preventDefault();
canvas.onmousedown=e=>{drag=(e.button===2||e.shiftKey)?2:1;lx=e.clientX;ly=e.clientY;};
window.onmouseup=()=>drag=0;
window.onmousemove=e=>{if(!drag)return;const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
if(drag===1){yaw+=dx*0.007;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.007));}
else{const s=dist*0.0018;const cy=Math.cos(yaw),sy=Math.sin(yaw);tx-=dx*s*cy;tz-=dx*s*sy;ty+=dy*s;}draw();};
canvas.onwheel=e=>{e.preventDefault();dist=Math.max(0.08,Math.min(12.0,dist*Math.exp(e.deltaY*0.001)));draw();};
window.onresize=draw;draw();
</script></body></html>
)JS";

    return out.good();
}

VolumeOverlapStats evaluate_swc_against_volume(const VolumeGrid& gt_grid,
                                               const VolumeMetadata& gt_meta,
                                               const std::vector<SWCNode>& swc,
                                               const VoxelizerParams& params,
                                               VolumeGrid* out_recon_grid,
                                               std::vector<float>* out_recon_sdf) {
    VolumeGrid recon(gt_grid.dim_x, gt_grid.dim_y, gt_grid.dim_z, gt_grid.scale);
    std::vector<float> sdf;
    rasterize_swc_into_grid(swc, params, recon, gt_meta, sdf);

    VolumeOverlapStats s{};
    const std::size_t total = gt_grid.size();
    for (std::size_t i = 0; i < total; ++i) {
        const bool g = gt_grid.mask[i] != 0;
        const bool r = recon.mask[i] != 0;
        if (g) ++s.gt_voxels;
        if (r) ++s.recon_voxels;
        if (g && r) ++s.intersection_voxels;
    }

    const std::size_t uni = s.gt_voxels + s.recon_voxels - s.intersection_voxels;
    s.iou = (uni > 0) ? static_cast<double>(s.intersection_voxels) / static_cast<double>(uni) : 0.0;
    s.dice = (s.gt_voxels + s.recon_voxels > 0)
        ? (2.0 * static_cast<double>(s.intersection_voxels)) / static_cast<double>(s.gt_voxels + s.recon_voxels)
        : 0.0;
    s.recall = (s.gt_voxels > 0) ? static_cast<double>(s.intersection_voxels) / static_cast<double>(s.gt_voxels) : 0.0;
    s.precision = (s.recon_voxels > 0) ? static_cast<double>(s.intersection_voxels) / static_cast<double>(s.recon_voxels) : 0.0;

    if (out_recon_grid) *out_recon_grid = std::move(recon);
    if (out_recon_sdf) *out_recon_sdf = std::move(sdf);
    return s;
}

bool render_mesh_comparison_ppm(const std::string& filepath,
                                const TriangleMesh& left_mesh,
                                const TriangleMesh& right_mesh,
                                int width,
                                int height,
                                float yaw_rad,
                                float pitch_rad,
                                bool use_custom_focus,
                                Vec3f focus_center,
                                float focus_span) {
    const int pw = width / 2;
    std::vector<uint8_t> rgb(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3, 6);

    Vec3f center = focus_center;
    float span = focus_span;
    if (!use_custom_focus || span <= 1.0f) {
        Vec3f mn{std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::infinity()};
        Vec3f mx{-std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<float>::infinity()};

        for (const Vec3f& v : left_mesh.vertices) {
            mn.x = std::min(mn.x, v.x); mn.y = std::min(mn.y, v.y); mn.z = std::min(mn.z, v.z);
            mx.x = std::max(mx.x, v.x); mx.y = std::max(mx.y, v.y); mx.z = std::max(mx.z, v.z);
        }
        for (const Vec3f& v : right_mesh.vertices) {
            mn.x = std::min(mn.x, v.x); mn.y = std::min(mn.y, v.y); mn.z = std::min(mn.z, v.z);
            mx.x = std::max(mx.x, v.x); mx.y = std::max(mx.y, v.y); mx.z = std::max(mx.z, v.z);
        }

        center = (mn + mx) * 0.5f;
        span = std::max({1.0f, mx.x - mn.x, mx.y - mn.y, mx.z - mn.z});
    }

    rasterize_mesh_panel(left_mesh, center, span, yaw_rad, pitch_rad, 0, pw, height, width, rgb, {0.22f, 0.90f, 0.42f});
    rasterize_mesh_panel(right_mesh, center, span, yaw_rad, pitch_rad, pw, width - pw, height, width, rgb, {0.24f, 0.78f, 0.98f});

    for (int y = 0; y < height; ++y) {
        const std::size_t idx = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(pw)) * 3;
        rgb[idx + 0] = 70;
        rgb[idx + 1] = 80;
        rgb[idx + 2] = 95;
    }

    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }
    out << "P6\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return out.good();
}

}
