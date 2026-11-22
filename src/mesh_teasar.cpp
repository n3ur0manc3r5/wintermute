#include "mesh_teasar.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace teasar {
namespace {

inline uint64_t pack_cell(int32_t x, int32_t y, int32_t z) {
    const uint64_t ux = static_cast<uint64_t>(static_cast<int64_t>(x) + (1LL << 20)) & 0x1FFFFFULL;
    const uint64_t uy = static_cast<uint64_t>(static_cast<int64_t>(y) + (1LL << 20)) & 0x1FFFFFULL;
    const uint64_t uz = static_cast<uint64_t>(static_cast<int64_t>(z) + (1LL << 20)) & 0x1FFFFFULL;
    return (ux << 42) | (uy << 21) | uz;
}

inline int32_t cell_coord(float x, float inv_h) {
    return static_cast<int32_t>(std::floor(x * inv_h));
}

struct CellSpan {
    uint32_t offset;
    uint32_t count;
};

struct TriGrid {
    float cell_size = 2000.0f;
    float inv_cell = 1.0f / 2000.0f;
    std::vector<uint32_t> tri_indices;
    std::unordered_map<uint64_t, CellSpan> cells;

    void build(const TriangleMesh& mesh, float h) {
        cell_size = h;
        inv_cell = 1.0f / h;
        const std::size_t nf = mesh.faces.size();
        const std::size_t nv = mesh.vertices.size();

        std::vector<std::pair<uint64_t, uint32_t>> entries;
        entries.reserve(nf * 2);

        for (std::size_t f = 0; f < nf; ++f) {
            const auto& t = mesh.faces[f];
            if (t[0] >= nv || t[1] >= nv || t[2] >= nv) {
                continue;
            }
            const Vec3f& a = mesh.vertices[t[0]];
            const Vec3f& b = mesh.vertices[t[1]];
            const Vec3f& c = mesh.vertices[t[2]];

            const int32_t x0 = cell_coord(std::min({a.x, b.x, c.x}), inv_cell);
            const int32_t x1 = cell_coord(std::max({a.x, b.x, c.x}), inv_cell);
            const int32_t y0 = cell_coord(std::min({a.y, b.y, c.y}), inv_cell);
            const int32_t y1 = cell_coord(std::max({a.y, b.y, c.y}), inv_cell);
            const int32_t z0 = cell_coord(std::min({a.z, b.z, c.z}), inv_cell);
            const int32_t z1 = cell_coord(std::max({a.z, b.z, c.z}), inv_cell);

            for (int32_t cz = z0; cz <= z1; ++cz) {
                for (int32_t cy = y0; cy <= y1; ++cy) {
                    for (int32_t cx = x0; cx <= x1; ++cx) {
                        entries.emplace_back(pack_cell(cx, cy, cz), static_cast<uint32_t>(f));
                    }
                }
            }
        }

        std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });

        tri_indices.resize(entries.size());
        cells.clear();
        cells.reserve(entries.size() / 4 + 16);

        std::size_t i = 0;
        while (i < entries.size()) {
            const uint64_t key = entries[i].first;
            const std::size_t start = i;
            while (i < entries.size() && entries[i].first == key) {
                tri_indices[i] = entries[i].second;
                ++i;
            }
            cells.emplace(key, CellSpan{static_cast<uint32_t>(start), static_cast<uint32_t>(i - start)});
        }
    }
};

inline bool ray_triangle_hit(const Vec3f& orig,
                             const Vec3f& dir,
                             const Vec3f& v0,
                             const Vec3f& v1,
                             const Vec3f& v2,
                             float t_min,
                             float t_max,
                             float& out_t) {
    const Vec3f e1 = v1 - v0;
    const Vec3f e2 = v2 - v0;
    const Vec3f pvec = dir.cross(e2);
    const float det = e1.dot(pvec);
    if (std::fabs(det) < 1e-8f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    const Vec3f tvec = orig - v0;
    const float u = tvec.dot(pvec) * inv_det;
    if (u < -1e-4f || u > 1.0001f) {
        return false;
    }
    const Vec3f qvec = tvec.cross(e1);
    const float v = dir.dot(qvec) * inv_det;
    if (v < -1e-4f || u + v > 1.0001f) {
        return false;
    }
    const float t = e2.dot(qvec) * inv_det;
    if (t > t_min && t < t_max) {
        out_t = t;
        return true;
    }
    return false;
}

float cast_inward_ray(const TriangleMesh& mesh,
                      const std::vector<Vec3f>& fnorm,
                      const TriGrid& grid,
                      uint32_t self_v,
                      const Vec3f& orig,
                      const Vec3f& dir,
                      float max_t) {
    const float h = grid.cell_size;
    const float inv_h = grid.inv_cell;

    int32_t cx = cell_coord(orig.x, inv_h);
    int32_t cy = cell_coord(orig.y, inv_h);
    int32_t cz = cell_coord(orig.z, inv_h);

    const float inf = std::numeric_limits<float>::infinity();
    auto init_axis = [h, inf](float o, float d, int32_t c, int32_t& step, float& t_max, float& t_delta) {
        if (d > 1e-8f) {
            step = 1;
            t_max = (static_cast<float>(c + 1) * h - o) / d;
            t_delta = h / d;
        } else if (d < -1e-8f) {
            step = -1;
            t_max = (static_cast<float>(c) * h - o) / d;
            t_delta = -h / d;
        } else {
            step = 0;
            t_max = inf;
            t_delta = inf;
        }
    };

    int32_t sx = 0, sy = 0, sz = 0;
    float tmx = inf, tmy = inf, tmz = inf;
    float tdx = inf, tdy = inf, tdz = inf;
    init_axis(orig.x, dir.x, cx, sx, tmx, tdx);
    init_axis(orig.y, dir.y, cy, sy, tmy, tdy);
    init_axis(orig.z, dir.z, cz, sz, tmz, tdz);

    float best_t = max_t;
    bool hit_any = false;
    float t_cur = 0.0f;

    while (t_cur <= best_t) {
        const auto it = grid.cells.find(pack_cell(cx, cy, cz));
        if (it != grid.cells.end()) {
            const CellSpan span = it->second;
            for (uint32_t k = 0; k < span.count; ++k) {
                const uint32_t f = grid.tri_indices[span.offset + k];
                const auto& tri = mesh.faces[f];
                if (tri[0] == self_v || tri[1] == self_v || tri[2] == self_v) {
                    continue;
                }
                if (dir.dot(fnorm[f]) <= 0.2f) {
                    continue;
                }
                float t_hit = 0.0f;
                if (ray_triangle_hit(orig,
                                     dir,
                                     mesh.vertices[tri[0]],
                                     mesh.vertices[tri[1]],
                                     mesh.vertices[tri[2]],
                                     15.0f,
                                     best_t,
                                     t_hit)) {
                    best_t = t_hit;
                    hit_any = true;
                }
            }
        }

        if (tmx <= tmy && tmx <= tmz) {
            t_cur = tmx;
            cx += sx;
            tmx += tdx;
        } else if (tmy <= tmz) {
            t_cur = tmy;
            cy += sy;
            tmy += tdy;
        } else {
            t_cur = tmz;
            cz += sz;
            tmz += tdz;
        }
    }

    return hit_any ? best_t : -1.0f;
}

struct GraphEdge {
    int to;
    float len;
};

struct DSU {
    std::vector<int> p;
    explicit DSU(int n) : p(static_cast<std::size_t>(n)) {
        for (int i = 0; i < n; ++i) p[static_cast<std::size_t>(i)] = i;
    }
    int find(int x) {
        while (p[static_cast<std::size_t>(x)] != x) {
            p[static_cast<std::size_t>(x)] = p[static_cast<std::size_t>(p[static_cast<std::size_t>(x)])];
            x = p[static_cast<std::size_t>(x)];
        }
        return x;
    }
    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        p[static_cast<std::size_t>(a)] = b;
        return true;
    }
};

float median_of(std::vector<float>& vals) {
    if (vals.empty()) {
        return 0.0f;
    }
    const std::size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid), vals.end());
    return vals[mid];
}

void build_mesh_adjacency(const TriangleMesh& mesh, std::vector<std::vector<GraphEdge>>& adj) {
    const std::size_t nv = mesh.vertices.size();
    adj.assign(nv, {});
    std::vector<std::vector<int>> raw_nb(nv);

    for (const auto& tri : mesh.faces) {
        if (tri[0] >= nv || tri[1] >= nv || tri[2] >= nv) {
            continue;
        }
        const int a = static_cast<int>(tri[0]);
        const int b = static_cast<int>(tri[1]);
        const int c = static_cast<int>(tri[2]);
        raw_nb[tri[0]].push_back(b);
        raw_nb[tri[0]].push_back(c);
        raw_nb[tri[1]].push_back(a);
        raw_nb[tri[1]].push_back(c);
        raw_nb[tri[2]].push_back(a);
        raw_nb[tri[2]].push_back(b);
    }

    for (std::size_t i = 0; i < nv; ++i) {
        auto& nb = raw_nb[i];
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        adj[i].reserve(nb.size());
        const Vec3f& pi = mesh.vertices[i];
        for (int j : nb) {
            if (j == static_cast<int>(i)) continue;
            const float d = (mesh.vertices[static_cast<std::size_t>(j)] - pi).norm();
            adj[i].push_back({j, std::max(1.0f, d)});
        }
    }
}

void prune_short_stubs(std::vector<SkeletonNode>& skel, float min_branch_len_nm) {
    if (skel.size() <= 2 || min_branch_len_nm <= 0.0f) {
        return;
    }

    bool changed = true;
    std::vector<uint8_t> alive(skel.size(), 1);

    while (changed) {
        changed = false;
        std::vector<int> child_cnt(skel.size(), 0);
        for (std::size_t i = 0; i < skel.size(); ++i) {
            if (!alive[i]) continue;
            const int p = skel[i].parent;
            if (p >= 0 && alive[static_cast<std::size_t>(p)]) {
                ++child_cnt[static_cast<std::size_t>(p)];
            }
        }

        for (std::size_t i = 0; i < skel.size(); ++i) {
            if (!alive[i] || skel[i].parent == -1 || skel[i].type == 1 || child_cnt[i] != 0) {
                continue;
            }
            std::vector<int> twig;
            float len = 0.0f;
            int curr = static_cast<int>(i);
            while (curr != -1) {
                const int p = skel[static_cast<std::size_t>(curr)].parent;
                if (p == -1) {
                    break;
                }
                twig.push_back(curr);
                const auto& a = skel[static_cast<std::size_t>(curr)];
                const auto& b = skel[static_cast<std::size_t>(p)];
                const float dx = a.x_nm - b.x_nm;
                const float dy = a.y_nm - b.y_nm;
                const float dz = a.z_nm - b.z_nm;
                len += std::sqrt(dx * dx + dy * dy + dz * dz);
                if (child_cnt[static_cast<std::size_t>(p)] >= 2 ||
                    skel[static_cast<std::size_t>(p)].parent == -1 ||
                    skel[static_cast<std::size_t>(p)].type == 1) {
                    break;
                }
                curr = p;
            }
            if (len < min_branch_len_nm && !twig.empty()) {
                for (int idx : twig) {
                    alive[static_cast<std::size_t>(idx)] = 0;
                }
                changed = true;
            }
        }
    }

    std::vector<int> remap(skel.size(), -1);
    std::vector<SkeletonNode> compact;
    compact.reserve(skel.size());
    for (std::size_t i = 0; i < skel.size(); ++i) {
        if (!alive[i]) continue;
        remap[i] = static_cast<int>(compact.size());
        compact.push_back(skel[i]);
    }
    for (auto& n : compact) {
        if (n.parent >= 0) {
            n.parent = remap[static_cast<std::size_t>(n.parent)];
        }
    }
    skel = std::move(compact);
}

struct SomaBall {
    Vec3f center;
    float radius;
};

std::vector<SomaBall> fit_soma_medial_balls(const TriangleMesh& mesh,
                                            const std::vector<Vec3f>&,
                                            const std::vector<float>&,
                                            const std::vector<int>& reachable,
                                            const Vec3f& ref,
                                            float approx_r) {
    std::vector<Vec3f> local_pts;
    for (int v : reachable) {
        const Vec3f& p = mesh.vertices[static_cast<std::size_t>(v)];
        if ((p - ref).norm_sq() <= 8500.0f * 8500.0f) {
            local_pts.push_back(p);
        }
    }

    if (local_pts.size() < 20) {
        return {{ref, std::clamp(approx_r, 3000.0f, 6500.0f)}};
    }

    Vec3f centroid{0.0f, 0.0f, 0.0f};
    for (const Vec3f& p : local_pts) centroid += p;
    centroid *= (1.0f / static_cast<float>(local_pts.size()));

    constexpr int kTheta = 12;
    constexpr int kPhi = 24;
    constexpr float kPi = 3.14159265358979323846f;
    std::vector<std::vector<float>> bin_r(kTheta * kPhi);
    for (const Vec3f& p : local_pts) {
        const Vec3f rel = p - centroid;
        const float r = rel.norm();
        if (r < 2200.0f || r > 8500.0f) continue;
        const float th = std::acos(std::clamp(rel.y / r, -1.0f, 1.0f));
        float ph = std::atan2(rel.z, rel.x);
        if (ph < 0.0f) ph += 2.0f * kPi;
        const int tb = std::clamp(static_cast<int>(th / kPi * kTheta), 0, kTheta - 1);
        const int pb = std::clamp(static_cast<int>(ph / (2.0f * kPi) * kPhi), 0, kPhi - 1);
        bin_r[static_cast<std::size_t>(tb * kPhi + pb)].push_back(r);
    }

    std::vector<float> all_med;
    std::vector<float> r_map(kTheta * kPhi, -1.0f);
    for (int i = 0; i < kTheta * kPhi; ++i) {
        auto& br = bin_r[static_cast<std::size_t>(i)];
        if (br.size() >= 2) {
            std::sort(br.begin(), br.end());
            const float r_val = br[br.size() / 2];
            r_map[static_cast<std::size_t>(i)] = r_val;
            all_med.push_back(r_val);
        }
    }

    const float global_r = all_med.empty() ? std::clamp(approx_r, 3500.0f, 6500.0f) : median_of(all_med);
    for (float& rv : r_map) {
        if (rv < 0.0f) rv = global_r;
        else rv = std::clamp(rv, global_r * 0.65f, global_r * 1.45f);
    }

    for (int sp = 0; sp < 2; ++sp) {
        std::vector<float> nxt(kTheta * kPhi, 0.0f);
        for (int tb = 0; tb < kTheta; ++tb) {
            for (int pb = 0; pb < kPhi; ++pb) {
                float sum = 0.0f;
                float wsum = 0.0f;
                for (int dt = -1; dt <= 1; ++dt) {
                    const int nt = std::clamp(tb + dt, 0, kTheta - 1);
                    for (int dp = -1; dp <= 1; ++dp) {
                        const int np = (pb + dp + kPhi) % kPhi;
                        const float w = (dt == 0 && dp == 0) ? 4.0f : 1.0f;
                        sum += w * r_map[static_cast<std::size_t>(nt * kPhi + np)];
                        wsum += w;
                    }
                }
                nxt[static_cast<std::size_t>(tb * kPhi + pb)] = sum / wsum;
            }
        }
        r_map.swap(nxt);
    }

    std::vector<Vec3f> sv;
    std::vector<Vec3f> sn;
    sv.reserve(kTheta * kPhi);
    sn.reserve(kTheta * kPhi);
    Vec3f bmin{std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity()};
    Vec3f bmax{-std::numeric_limits<float>::infinity(),
               -std::numeric_limits<float>::infinity(),
               -std::numeric_limits<float>::infinity()};

    for (int tb = 0; tb < kTheta; ++tb) {
        const float th = (static_cast<float>(tb) + 0.5f) * kPi / static_cast<float>(kTheta);
        const float st = std::sin(th);
        const float ct = std::cos(th);
        for (int pb = 0; pb < kPhi; ++pb) {
            const float ph = (static_cast<float>(pb) + 0.5f) * 2.0f * kPi / static_cast<float>(kPhi);
            const Vec3f dir{st * std::cos(ph), ct, st * std::sin(ph)};
            const float rv = r_map[static_cast<std::size_t>(tb * kPhi + pb)];
            const Vec3f p = centroid + dir * rv;
            sv.push_back(p);
            sn.push_back(dir);
            bmin.x = std::min(bmin.x, p.x); bmin.y = std::min(bmin.y, p.y); bmin.z = std::min(bmin.z, p.z);
            bmax.x = std::max(bmax.x, p.x); bmax.y = std::max(bmax.y, p.y); bmax.z = std::max(bmax.z, p.z);
        }
    }

    const float shell_r = global_r * 1.6f;
    const float shell_r2 = shell_r * shell_r;
    const float step = 550.0f;
    const int nx = std::max(1, static_cast<int>(std::floor((bmax.x - bmin.x) / step)) + 1);
    const int ny = std::max(1, static_cast<int>(std::floor((bmax.y - bmin.y) / step)) + 1);
    const int nz = std::max(1, static_cast<int>(std::floor((bmax.z - bmin.z) / step)) + 1);

    std::vector<Vec3f> grid_pts;
    grid_pts.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz));
    for (int iz = 0; iz < nz; ++iz) {
        const float pz = bmin.z + static_cast<float>(iz) * step;
        for (int iy = 0; iy < ny; ++iy) {
            const float py = bmin.y + static_cast<float>(iy) * step;
            for (int ix = 0; ix < nx; ++ix) {
                const float px = bmin.x + static_cast<float>(ix) * step;
                const Vec3f q{px, py, pz};
                if ((q - centroid).norm_sq() <= shell_r2) {
                    grid_pts.push_back(q);
                }
            }
        }
    }

    const int nq = static_cast<int>(grid_pts.size());
    std::vector<float> cand_r(static_cast<std::size_t>(nq), -1.0f);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 128)
#endif
    for (int qi = 0; qi < nq; ++qi) {
        const Vec3f& q = grid_pts[static_cast<std::size_t>(qi)];
        const Vec3f rel = q - centroid;
        const float rq = rel.norm();
        if (rq > 1.0f) {
            const float th = std::acos(std::clamp(rel.y / rq, -1.0f, 1.0f));
            float ph = std::atan2(rel.z, rel.x);
            if (ph < 0.0f) ph += 2.0f * kPi;
            const int tb = std::clamp(static_cast<int>(th / kPi * kTheta), 0, kTheta - 1);
            const int pb = std::clamp(static_cast<int>(ph / (2.0f * kPi) * kPhi), 0, kPhi - 1);
            if (rq >= r_map[static_cast<std::size_t>(tb * kPhi + pb)] - 400.0f) {
                continue;
            }
        }
        float min_d2 = std::numeric_limits<float>::infinity();
        for (const Vec3f& p : sv) {
            const float d2 = (q - p).norm_sq();
            if (d2 < min_d2) min_d2 = d2;
        }
        const float r = std::sqrt(min_d2);
        if (r >= 2000.0f && r <= 7500.0f) {
            cand_r[static_cast<std::size_t>(qi)] = r;
        }
    }

    std::vector<SomaBall> cands;
    for (int qi = 0; qi < nq; ++qi) {
        if (cand_r[static_cast<std::size_t>(qi)] > 0.0f) {
            cands.push_back({grid_pts[static_cast<std::size_t>(qi)], cand_r[static_cast<std::size_t>(qi)]});
        }
    }

    if (cands.empty()) {
        return {{centroid, std::clamp(global_r, 3000.0f, 6500.0f)}};
    }

    std::size_t root_c = 0;
    for (std::size_t i = 1; i < cands.size(); ++i) {
        if (cands[i].radius > cands[root_c].radius) {
            root_c = i;
        }
    }

    std::vector<SomaBall> balls;
    balls.reserve(8);
    balls.push_back(cands[root_c]);

    std::vector<float> cov_err(sv.size(), std::numeric_limits<float>::infinity());
    for (std::size_t j = 0; j < sv.size(); ++j) {
        cov_err[j] = (sv[j] - balls[0].center).norm() - balls[0].radius;
    }

    for (int step_k = 1; step_k < 8; ++step_k) {
        std::size_t worst_j = 0;
        float worst_e = cov_err[0];
        for (std::size_t j = 1; j < sv.size(); ++j) {
            if (cov_err[j] > worst_e) {
                worst_e = cov_err[j];
                worst_j = j;
            }
        }
        if (worst_e <= 500.0f) {
            break;
        }

        const Vec3f& target_p = sv[worst_j];
        std::vector<std::pair<float, std::size_t>> near_c;
        near_c.reserve(cands.size());
        for (std::size_t ci = 0; ci < cands.size(); ++ci) {
            const float gap = (target_p - cands[ci].center).norm() - cands[ci].radius;
            if (gap < worst_e - 120.0f) {
                near_c.emplace_back(gap, ci);
            }
        }
        if (near_c.empty()) break;
        std::sort(near_c.begin(), near_c.end());
        const std::size_t eval_limit = std::min<std::size_t>(near_c.size(), 96);

        double best_gain = 0.0;
        std::size_t best_ci = static_cast<std::size_t>(-1);
        for (std::size_t ni = 0; ni < eval_limit; ++ni) {
            const std::size_t ci = near_c[ni].second;
            const SomaBall& cb = cands[ci];
            bool too_close = false;
            for (const SomaBall& eb : balls) {
                if ((cb.center - eb.center).norm() < 0.35f * cb.radius) {
                    too_close = true;
                    break;
                }
            }
            if (too_close) continue;

            double gain = 0.0;
            for (std::size_t j = 0; j < sv.size(); ++j) {
                if (cov_err[j] <= 0.0f) continue;
                const float nd = (sv[j] - cb.center).norm() - cb.radius;
                if (nd < cov_err[j]) {
                    const double old_e = static_cast<double>(cov_err[j]);
                    const double new_e = static_cast<double>(std::max(0.0f, nd));
                    gain += old_e * old_e - new_e * new_e;
                }
            }
            if (gain > best_gain) {
                best_gain = gain;
                best_ci = ci;
            }
        }

        if (best_ci == static_cast<std::size_t>(-1) || best_gain <= 5e4) break;
        const SomaBall chosen = cands[best_ci];
        balls.push_back(chosen);
        for (std::size_t j = 0; j < sv.size(); ++j) {
            const float nd = (sv[j] - chosen.center).norm() - chosen.radius;
            if (nd < cov_err[j]) cov_err[j] = nd;
        }
    }

    return balls;
}

}

std::vector<float> compute_mesh_sdf(const TriangleMesh& mesh,
                                    const std::vector<Vec3f>& vnorm,
                                    const std::vector<Vec3f>& fnorm,
                                    float max_ray_nm) {
    const std::size_t nv = mesh.vertices.size();
    std::vector<float> radius(nv, 0.0f);
    if (nv == 0 || mesh.faces.empty()) {
        return radius;
    }

    TriGrid grid;
    grid.build(mesh, 2000.0f);

    #pragma omp parallel for schedule(dynamic, 256)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(nv); ++i) {
        const std::size_t ui = static_cast<std::size_t>(i);
        const Vec3f& n = vnorm[ui];
        if (n.norm_sq() < 1e-6f) {
            continue;
        }
        const Vec3f inward = n * -1.0f;
        const Vec3f up = (std::fabs(inward.x) < 0.9f) ? Vec3f{1.0f, 0.0f, 0.0f} : Vec3f{0.0f, 1.0f, 0.0f};
        const Vec3f u = inward.cross(up).normalized();
        const Vec3f w = inward.cross(u).normalized();

        const Vec3f dirs[5] = {
            inward,
            (inward + u * 0.30f).normalized(),
            (inward - u * 0.30f).normalized(),
            (inward + w * 0.30f).normalized(),
            (inward - w * 0.30f).normalized()
        };

        float hits[5];
        int nhits = 0;
        const Vec3f orig = mesh.vertices[ui] + inward * 5.0f;
        for (int r = 0; r < 5; ++r) {
            const float d = cast_inward_ray(mesh, fnorm, grid, static_cast<uint32_t>(ui), orig, dirs[r], max_ray_nm);
            if (d > 0.0f) {
                hits[nhits++] = d;
            }
        }
        if (nhits > 0) {
            for (int a = 1; a < nhits; ++a) {
                const float key = hits[a];
                int b = a - 1;
                while (b >= 0 && hits[b] > key) {
                    hits[b + 1] = hits[b];
                    --b;
                }
                hits[b + 1] = key;
            }
            radius[ui] = 0.5f * hits[nhits / 2];
        }
    }

    return radius;
}

std::vector<SkeletonNode> extract_mesh_skeleton(const TriangleMesh& mesh,
                                                const MeshTeasarParams& params,
                                                MeshSkeletonStats* stats) {
    const std::size_t nv = mesh.vertices.size();
    if (nv == 0) {
        return {};
    }

    std::vector<Vec3f> fnorm;
    const std::vector<Vec3f> vnorm = compute_vertex_normals(mesh, &fnorm);
    std::vector<float> radius = compute_mesh_sdf(mesh, vnorm, fnorm, params.sdf_max_ray_nm);

    std::vector<std::vector<GraphEdge>> adj;
    build_mesh_adjacency(mesh, adj);

    for (int iter = 0; iter < 3; ++iter) {
        std::vector<float> next_r = radius;
        std::vector<float> tmp;
        for (std::size_t i = 0; i < nv; ++i) {
            tmp.clear();
            if (radius[i] > 0.0f) tmp.push_back(radius[i]);
            for (const auto& e : adj[i]) {
                const float nr = radius[static_cast<std::size_t>(e.to)];
                if (nr > 0.0f) tmp.push_back(nr);
            }
            if (!tmp.empty()) {
                next_r[i] = median_of(tmp);
            }
        }
        radius = std::move(next_r);
    }

    std::vector<float> valid_r;
    valid_r.reserve(nv);
    for (float r : radius) {
        if (r > 0.0f) valid_r.push_back(r);
    }
    const float fallback_r = valid_r.empty() ? 250.0f : median_of(valid_r);
    for (float& r : radius) {
        if (r <= 0.0f) r = fallback_r;
    }

    std::vector<int> comp_id(nv, -1);
    std::vector<std::vector<int>> comps;
    std::queue<int> bfs_q;
    for (std::size_t i = 0; i < nv; ++i) {
        if (comp_id[i] != -1) continue;
        const int cid = static_cast<int>(comps.size());
        comps.emplace_back();
        comp_id[i] = cid;
        bfs_q.push(static_cast<int>(i));
        while (!bfs_q.empty()) {
            const int u = bfs_q.front();
            bfs_q.pop();
            comps.back().push_back(u);
            for (const auto& e : adj[static_cast<std::size_t>(u)]) {
                if (comp_id[static_cast<std::size_t>(e.to)] == -1) {
                    comp_id[static_cast<std::size_t>(e.to)] = cid;
                    bfs_q.push(e.to);
                }
            }
        }
    }

    std::size_t largest_comp_sz = 0;
    int largest_comp = 0;
    for (std::size_t c = 0; c < comps.size(); ++c) {
        if (comps[c].size() > largest_comp_sz) {
            largest_comp_sz = comps[c].size();
            largest_comp = static_cast<int>(c);
        }
    }

    int soma_nearest_v = comps[static_cast<std::size_t>(largest_comp)][0];
    if (mesh.has_soma) {
        float best_d2 = std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < nv; ++i) {
            if (static_cast<int>(comps[static_cast<std::size_t>(comp_id[i])].size()) < params.min_component_verts) {
                continue;
            }
            const float d2 = (mesh.vertices[i] - mesh.soma_nm).norm_sq();
            if (d2 < best_d2) {
                best_d2 = d2;
                soma_nearest_v = static_cast<int>(i);
            }
        }
    }
    const int main_comp = comp_id[static_cast<std::size_t>(soma_nearest_v)];

    std::vector<uint8_t> keep_comp(comps.size(), 0);
    int kept_count = 0;
    for (std::size_t c = 0; c < comps.size(); ++c) {
        if (static_cast<int>(c) == main_comp || static_cast<int>(comps[c].size()) >= params.min_component_verts) {
            keep_comp[c] = 1;
            ++kept_count;
        }
    }

    int bridges_added = 0;
    if (kept_count > 1 && params.stitch_max_dist_nm > 0.0f) {
        const float h = params.stitch_max_dist_nm;
        const float inv_h = 1.0f / h;
        const float max_d2 = h * h;

        std::unordered_map<uint64_t, std::vector<int>> vgrid;
        vgrid.reserve(nv / 4 + 16);
        for (std::size_t i = 0; i < nv; ++i) {
            if (!keep_comp[static_cast<std::size_t>(comp_id[i])]) continue;
            const Vec3f& p = mesh.vertices[i];
            vgrid[pack_cell(cell_coord(p.x, inv_h), cell_coord(p.y, inv_h), cell_coord(p.z, inv_h))].push_back(static_cast<int>(i));
        }

        struct Bridge {
            float dist;
            int c1;
            int c2;
            int u;
            int v;
        };
        std::vector<Bridge> candidates;

        for (std::size_t i = 0; i < nv; ++i) {
            const int c1 = comp_id[i];
            if (!keep_comp[static_cast<std::size_t>(c1)]) continue;
            const Vec3f& p = mesh.vertices[i];
            const int32_t cx = cell_coord(p.x, inv_h);
            const int32_t cy = cell_coord(p.y, inv_h);
            const int32_t cz = cell_coord(p.z, inv_h);

            for (int32_t dz = -1; dz <= 1; ++dz) {
                for (int32_t dy = -1; dy <= 1; ++dy) {
                    for (int32_t dx = -1; dx <= 1; ++dx) {
                        const auto it = vgrid.find(pack_cell(cx + dx, cy + dy, cz + dz));
                        if (it == vgrid.end()) continue;
                        for (int j : it->second) {
                            const int c2 = comp_id[static_cast<std::size_t>(j)];
                            if (c2 <= c1) continue;
                            const float d2 = (mesh.vertices[static_cast<std::size_t>(j)] - p).norm_sq();
                            if (d2 <= max_d2) {
                                candidates.push_back({std::sqrt(d2), c1, c2, static_cast<int>(i), j});
                            }
                        }
                    }
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const Bridge& a, const Bridge& b) {
            return a.dist < b.dist;
        });

        DSU dsu(static_cast<int>(comps.size()));
        for (const auto& br : candidates) {
            if (dsu.unite(br.c1, br.c2)) {
                const float w = std::max(1.0f, br.dist);
                adj[static_cast<std::size_t>(br.u)].push_back({br.v, w});
                adj[static_cast<std::size_t>(br.v)].push_back({br.u, w});
                ++bridges_added;
            }
        }
    }

    int root_v = soma_nearest_v;
    if (mesh.has_soma) {
        float best_r = -1.0f;
        const float search_r2 = 8000.0f * 8000.0f;
        for (int v : comps[static_cast<std::size_t>(main_comp)]) {
            const float d2 = (mesh.vertices[static_cast<std::size_t>(v)] - mesh.soma_nm).norm_sq();
            if (d2 <= search_r2 && radius[static_cast<std::size_t>(v)] > best_r) {
                best_r = radius[static_cast<std::size_t>(v)];
                root_v = v;
            }
        }
    } else {
        float best_r = -1.0f;
        for (int v : comps[static_cast<std::size_t>(main_comp)]) {
            if (radius[static_cast<std::size_t>(v)] > best_r) {
                best_r = radius[static_cast<std::size_t>(v)];
                root_v = v;
            }
        }
    }

    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> dfr(nv, inf);
    std::vector<int> reachable;
    reachable.reserve(nv);
    {
        std::priority_queue<DijkstraNode, std::vector<DijkstraNode>, std::greater<DijkstraNode>> pq;
        dfr[static_cast<std::size_t>(root_v)] = 0.0f;
        pq.push({root_v, 0.0f});
        while (!pq.empty()) {
            const DijkstraNode curr = pq.top();
            pq.pop();
            if (curr.dist > dfr[static_cast<std::size_t>(curr.index)]) continue;
            reachable.push_back(curr.index);
            for (const auto& e : adj[static_cast<std::size_t>(curr.index)]) {
                const float cand = curr.dist + e.len;
                if (cand < dfr[static_cast<std::size_t>(e.to)]) {
                    dfr[static_cast<std::size_t>(e.to)] = cand;
                    pq.push({e.to, cand});
                }
            }
        }
    }

    float max_r = 0.0f;
    for (int v : reachable) {
        if (radius[static_cast<std::size_t>(v)] > max_r) {
            max_r = radius[static_cast<std::size_t>(v)];
        }
    }
    const float inv_max_r = (max_r > 0.0f) ? (1.0f / max_r) : 0.0f;

    std::vector<float> pdrf(nv, inf);
    std::vector<int> parent_v(nv, -1);
    {
        std::priority_queue<DijkstraNode, std::vector<DijkstraNode>, std::greater<DijkstraNode>> pq;
        pdrf[static_cast<std::size_t>(root_v)] = 0.0f;
        pq.push({root_v, 0.0f});
        while (!pq.empty()) {
            const DijkstraNode curr = pq.top();
            pq.pop();
            if (curr.dist > pdrf[static_cast<std::size_t>(curr.index)]) continue;
            for (const auto& e : adj[static_cast<std::size_t>(curr.index)]) {
                const float ratio = std::clamp(1.0f - radius[static_cast<std::size_t>(e.to)] * inv_max_r, 0.0f, 1.0f);
                float pterm = 1.0f;
                for (int k = 0; k < params.pdrf_exponent; ++k) pterm *= ratio;
                const float cand = curr.dist + e.len * (1.0f + params.pdrf_scale * pterm);
                if (cand < pdrf[static_cast<std::size_t>(e.to)]) {
                    pdrf[static_cast<std::size_t>(e.to)] = cand;
                    parent_v[static_cast<std::size_t>(e.to)] = curr.index;
                    pq.push({e.to, cand});
                }
            }
        }
    }

    std::vector<uint8_t> invalidated(nv, 0);
    std::vector<int> vert_to_skel(nv, -1);
    std::vector<int> owner(nv, -1);
    std::vector<float> owner_dist(nv, inf);
    std::vector<SkeletonNode> skeleton;

    std::vector<int> local_stamp(nv, 0);
    std::vector<float> local_geo(nv, inf);
    int stamp_id = 0;

    auto invalidate_geodesic_ball = [&](int center_v, int skel_id, float ball_r) {
        invalidated[static_cast<std::size_t>(center_v)] = 1;
        if (0.0f < owner_dist[static_cast<std::size_t>(center_v)]) {
            owner_dist[static_cast<std::size_t>(center_v)] = 0.0f;
            owner[static_cast<std::size_t>(center_v)] = skel_id;
        }
        if (ball_r <= 0.0f) {
            return;
        }

        const Vec3f& cp = mesh.vertices[static_cast<std::size_t>(center_v)];
        const float r_sq = ball_r * ball_r;
        const float max_geo = ball_r * 2.2f;

        ++stamp_id;
        std::priority_queue<DijkstraNode, std::vector<DijkstraNode>, std::greater<DijkstraNode>> pq;
        local_stamp[static_cast<std::size_t>(center_v)] = stamp_id;
        local_geo[static_cast<std::size_t>(center_v)] = 0.0f;
        pq.push({center_v, 0.0f});

        while (!pq.empty()) {
            const DijkstraNode cur = pq.top();
            pq.pop();
            if (cur.dist > local_geo[static_cast<std::size_t>(cur.index)]) continue;

            const float euclid_sq = (mesh.vertices[static_cast<std::size_t>(cur.index)] - cp).norm_sq();
            if (euclid_sq <= r_sq) {
                invalidated[static_cast<std::size_t>(cur.index)] = 1;
                const float ed = std::sqrt(euclid_sq);
                if (ed < owner_dist[static_cast<std::size_t>(cur.index)]) {
                    owner_dist[static_cast<std::size_t>(cur.index)] = ed;
                    owner[static_cast<std::size_t>(cur.index)] = skel_id;
                }
            }

            for (const auto& e : adj[static_cast<std::size_t>(cur.index)]) {
                const float ng = cur.dist + e.len;
                if (ng > max_geo) continue;
                const std::size_t nu = static_cast<std::size_t>(e.to);
                const float nd2 = (mesh.vertices[nu] - cp).norm_sq();
                if (nd2 > r_sq * 2.25f) continue;
                if (local_stamp[nu] != stamp_id || ng < local_geo[nu]) {
                    local_stamp[nu] = stamp_id;
                    local_geo[nu] = ng;
                    pq.push({e.to, ng});
                }
            }
        }
    };

    auto make_skel_node = [&](int v_idx, int parent_skel, int swc_type) {
        const Vec3f& p = mesh.vertices[static_cast<std::size_t>(v_idx)];
        SkeletonNode n;
        n.voxel_idx = v_idx;
        n.x_nm = p.x;
        n.y_nm = p.y;
        n.z_nm = p.z;
        n.radius_nm = radius[static_cast<std::size_t>(v_idx)];
        n.parent = parent_skel;
        n.type = swc_type;
        return n;
    };

    skeleton.push_back(make_skel_node(root_v, -1, 1));
    vert_to_skel[static_cast<std::size_t>(root_v)] = 0;

    const float root_ball = std::max(params.soma_invalidation_radius_nm,
                                     params.invalidation_scale * radius[static_cast<std::size_t>(root_v)] + params.invalidation_const_nm);
    invalidate_geodesic_ball(root_v, 0, root_ball);

    std::sort(reachable.begin(), reachable.end(), [&dfr](int a, int b) {
        const float da = dfr[static_cast<std::size_t>(a)];
        const float db = dfr[static_cast<std::size_t>(b)];
        if (da != db) return da > db;
        return a < b;
    });

    std::size_t cursor = 0;
    std::vector<int> path;
    while (cursor < reachable.size()) {
        while (cursor < reachable.size() && invalidated[static_cast<std::size_t>(reachable[cursor])] != 0) {
            ++cursor;
        }
        if (cursor >= reachable.size()) {
            break;
        }

        const int target_v = reachable[cursor];
        path.clear();
        int curr = target_v;
        while (curr != -1 && vert_to_skel[static_cast<std::size_t>(curr)] == -1) {
            path.push_back(curr);
            curr = parent_v[static_cast<std::size_t>(curr)];
        }

        const int attach_node = (curr != -1) ? vert_to_skel[static_cast<std::size_t>(curr)] : 0;
        int prev_node = attach_node;
        std::vector<std::pair<int, int>> added;
        added.reserve(path.size());

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            const int v = *it;
            const int new_id = static_cast<int>(skeleton.size());
            skeleton.push_back(make_skel_node(v, prev_node, params.branch_swc_type));
            vert_to_skel[static_cast<std::size_t>(v)] = new_id;
            added.emplace_back(v, new_id);
            prev_node = new_id;
        }

        for (const auto& [v, sid] : added) {
            const float r = params.invalidation_scale * radius[static_cast<std::size_t>(v)] + params.invalidation_const_nm;
            invalidate_geodesic_ball(v, sid, r);
        }
    }

    const std::size_t ns = skeleton.size();
    std::vector<std::vector<int>> owned_verts(ns);
    for (int v : reachable) {
        const int o = owner[static_cast<std::size_t>(v)];
        if (o >= 0 && static_cast<std::size_t>(o) < ns) {
            owned_verts[static_cast<std::size_t>(o)].push_back(v);
        }
    }

    std::vector<std::vector<int>> skel_children(ns);
    for (std::size_t i = 0; i < ns; ++i) {
        const int p = skeleton[i].parent;
        if (p >= 0 && static_cast<std::size_t>(p) < ns) {
            skel_children[static_cast<std::size_t>(p)].push_back(static_cast<int>(i));
        }
    }

    std::vector<float> rad_buf;
    for (std::size_t i = 0; i < ns; ++i) {
        const auto& ov = owned_verts[i];
        if (i == 0) {
            Vec3f sum{0.0f, 0.0f, 0.0f};
            int cnt = 0;
            const Vec3f ref = mesh.has_soma ? mesh.soma_nm : Vec3f{skeleton[0].x_nm, skeleton[0].y_nm, skeleton[0].z_nm};
            for (int v : reachable) {
                const Vec3f& p = mesh.vertices[static_cast<std::size_t>(v)];
                if ((p - ref).norm_sq() <= 8500.0f * 8500.0f) {
                    sum += p;
                    ++cnt;
                }
            }
            if (cnt >= 10) {
                const Vec3f c = sum * (1.0f / static_cast<float>(cnt));
                skeleton[0].x_nm = c.x;
                skeleton[0].y_nm = c.y;
                skeleton[0].z_nm = c.z;
                rad_buf.clear();
                for (int v : reachable) {
                    const Vec3f& p = mesh.vertices[static_cast<std::size_t>(v)];
                    if ((p - ref).norm_sq() <= 8500.0f * 8500.0f) {
                        rad_buf.push_back((p - c).norm());
                    }
                }
                if (!rad_buf.empty()) {
                    skeleton[0].radius_nm = median_of(rad_buf);
                }
            }
            continue;
        }

        if (ov.size() < 3) {
            continue;
        }

        const bool is_tip = skel_children[i].empty();
        Vec3f mean{0.0f, 0.0f, 0.0f};
        for (int v : ov) {
            mean += mesh.vertices[static_cast<std::size_t>(v)];
        }
        mean *= (1.0f / static_cast<float>(ov.size()));

        const SkeletonNode& par = skeleton[static_cast<std::size_t>(skeleton[i].parent)];
        Vec3f tangent = Vec3f{skeleton[i].x_nm - par.x_nm, skeleton[i].y_nm - par.y_nm, skeleton[i].z_nm - par.z_nm};
        if (skel_children[i].size() == 1) {
            const SkeletonNode& ch = skeleton[static_cast<std::size_t>(skel_children[i][0])];
            tangent = Vec3f{ch.x_nm - par.x_nm, ch.y_nm - par.y_nm, ch.z_nm - par.z_nm};
        }
        tangent = tangent.normalized();

        if (!is_tip) {
            skeleton[i].x_nm = mean.x;
            skeleton[i].y_nm = mean.y;
            skeleton[i].z_nm = mean.z;
        } else if (tangent.norm_sq() > 0.5f) {
            const Vec3f cur{skeleton[i].x_nm, skeleton[i].y_nm, skeleton[i].z_nm};
            const Vec3f delta = mean - cur;
            const Vec3f perp = delta - tangent * delta.dot(tangent);
            skeleton[i].x_nm = cur.x + perp.x;
            skeleton[i].y_nm = cur.y + perp.y;
            skeleton[i].z_nm = cur.z + perp.z;
        }

        const Vec3f center{skeleton[i].x_nm, skeleton[i].y_nm, skeleton[i].z_nm};
        rad_buf.clear();
        for (int v : ov) {
            const Vec3f d = mesh.vertices[static_cast<std::size_t>(v)] - center;
            if (tangent.norm_sq() > 0.5f) {
                const float proj = d.dot(tangent);
                const Vec3f perp = d - tangent * proj;
                rad_buf.push_back(perp.norm());
            } else {
                rad_buf.push_back(d.norm());
            }
        }
        if (rad_buf.size() >= 6) {
            const float cross_r = median_of(rad_buf);
            if (cross_r > 40.0f && cross_r < 10000.0f) {
                const float sdf_r = skeleton[i].radius_nm;
                if (cross_r <= 2.0f * sdf_r && sdf_r <= 2.0f * cross_r) {
                    skeleton[i].radius_nm = 0.5f * (sdf_r + cross_r);
                } else {
                    skeleton[i].radius_nm = std::min(sdf_r, cross_r);
                }
            }
        }
    }

    const Vec3f soma_ref = mesh.has_soma ? mesh.soma_nm : Vec3f{skeleton[0].x_nm, skeleton[0].y_nm, skeleton[0].z_nm};
    const std::vector<SomaBall> soma_balls = fit_soma_medial_balls(
        mesh, vnorm, radius, reachable, soma_ref, skeleton[0].radius_nm);

    auto eval_soma_sdf = [&soma_balls](const Vec3f& p) -> float {
        float best = std::numeric_limits<float>::infinity();
        for (const SomaBall& b : soma_balls) {
            const float d = (p - b.center).norm() - b.radius;
            if (d < best) best = d;
        }
        return best;
    };

    std::vector<uint8_t> in_soma(ns, 0);
    in_soma[0] = 1;
    {
        std::queue<int> sq;
        sq.push(0);
        while (!sq.empty()) {
            const int u = sq.front();
            sq.pop();
            for (int c : skel_children[static_cast<std::size_t>(u)]) {
                const Vec3f pc{skeleton[static_cast<std::size_t>(c)].x_nm,
                               skeleton[static_cast<std::size_t>(c)].y_nm,
                               skeleton[static_cast<std::size_t>(c)].z_nm};
                const float sd = eval_soma_sdf(pc);
                if (sd <= 1200.0f || (sd <= 3500.0f && skeleton[static_cast<std::size_t>(c)].radius_nm >= 2000.0f)) {
                    in_soma[static_cast<std::size_t>(c)] = 1;
                    sq.push(c);
                }
            }
        }
    }

    std::vector<int> old_to_new(ns, -1);
    std::vector<SkeletonNode> collapsed;
    collapsed.reserve(ns + soma_balls.size());
    for (std::size_t b = 0; b < soma_balls.size(); ++b) {
        SkeletonNode sn_node;
        sn_node.voxel_idx = (b == 0) ? skeleton[0].voxel_idx : -1;
        sn_node.x_nm = soma_balls[b].center.x;
        sn_node.y_nm = soma_balls[b].center.y;
        sn_node.z_nm = soma_balls[b].center.z;
        sn_node.radius_nm = soma_balls[b].radius;
        sn_node.parent = (b == 0) ? -1 : 0;
        sn_node.type = 1;
        collapsed.push_back(sn_node);
    }
    old_to_new[0] = 0;
    for (std::size_t i = 1; i < ns; ++i) {
        if (in_soma[i]) continue;
        old_to_new[i] = static_cast<int>(collapsed.size());
        collapsed.push_back(skeleton[i]);
    }
    for (std::size_t i = 1; i < ns; ++i) {
        if (in_soma[i]) continue;
        const int ni = old_to_new[i];
        const int old_p = skeleton[i].parent;
        if (old_p < 0 || in_soma[static_cast<std::size_t>(old_p)]) {
            const Vec3f pi{skeleton[i].x_nm, skeleton[i].y_nm, skeleton[i].z_nm};
            int best_b = 0;
            float best_d = std::numeric_limits<float>::infinity();
            for (std::size_t b = 0; b < soma_balls.size(); ++b) {
                const float d = (pi - soma_balls[b].center).norm() - soma_balls[b].radius;
                if (d < best_d) {
                    best_d = d;
                    best_b = static_cast<int>(b);
                }
            }
            collapsed[static_cast<std::size_t>(ni)].parent = best_b;
        } else {
            collapsed[static_cast<std::size_t>(ni)].parent = old_to_new[static_cast<std::size_t>(old_p)];
        }
    }
    skeleton = std::move(collapsed);
    prune_short_stubs(skeleton, params.min_branch_len_nm);
    const std::size_t ns2 = skeleton.size();

    skel_children.assign(ns2, {});
    for (std::size_t i = 0; i < ns2; ++i) {
        const int p = skeleton[i].parent;
        if (p >= 0 && static_cast<std::size_t>(p) < ns2) {
            skel_children[static_cast<std::size_t>(p)].push_back(static_cast<int>(i));
        }
    }

    auto seg_len = [&skeleton](int a, int b) -> float {
        const float dx = skeleton[static_cast<std::size_t>(a)].x_nm - skeleton[static_cast<std::size_t>(b)].x_nm;
        const float dy = skeleton[static_cast<std::size_t>(a)].y_nm - skeleton[static_cast<std::size_t>(b)].y_nm;
        const float dz = skeleton[static_cast<std::size_t>(a)].z_nm - skeleton[static_cast<std::size_t>(b)].z_nm;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> next_r(ns2);
        std::vector<float> win;
        for (std::size_t i = 0; i < ns2; ++i) {
            next_r[i] = skeleton[i].radius_nm;
            if (skeleton[i].type == 1) continue;
            win.clear();
            win.push_back(skeleton[i].radius_nm);

            int cur = static_cast<int>(i);
            float dist_up = 0.0f;
            for (int steps = 0; steps < 24 && dist_up < 4500.0f; ++steps) {
                const int p = skeleton[static_cast<std::size_t>(cur)].parent;
                if (p < 0 || skeleton[static_cast<std::size_t>(p)].type == 1) break;
                dist_up += seg_len(cur, p);
                win.push_back(skeleton[static_cast<std::size_t>(p)].radius_nm);
                cur = p;
            }

            for (int ch0 : skel_children[i]) {
                if (skeleton[static_cast<std::size_t>(ch0)].type == 1) continue;
                int c = ch0;
                float dist_dn = seg_len(static_cast<int>(i), c);
                win.push_back(skeleton[static_cast<std::size_t>(c)].radius_nm);
                for (int steps = 0; steps < 24 && dist_dn < 4500.0f; ++steps) {
                    if (skel_children[static_cast<std::size_t>(c)].size() != 1) break;
                    const int nc = skel_children[static_cast<std::size_t>(c)][0];
                    if (skeleton[static_cast<std::size_t>(nc)].type == 1) break;
                    dist_dn += seg_len(c, nc);
                    win.push_back(skeleton[static_cast<std::size_t>(nc)].radius_nm);
                    c = nc;
                }
            }

            next_r[i] = median_of(win);
        }
        for (std::size_t i = 0; i < ns2; ++i) {
            if (skeleton[i].type != 1) {
                skeleton[i].radius_nm = next_r[i];
            }
        }
    }

    {
        const Vec3f soma_c = soma_balls.empty() ? soma_ref : soma_balls[0].center;
        std::queue<int> rq;
        for (std::size_t i = 0; i < ns2; ++i) {
            if (skeleton[i].type == 1) {
                rq.push(static_cast<int>(i));
            }
        }
        std::vector<float> shaft;
        while (!rq.empty()) {
            const int u = rq.front();
            rq.pop();
            const bool u_is_soma = (skeleton[static_cast<std::size_t>(u)].type == 1);
            for (int c : skel_children[static_cast<std::size_t>(u)]) {
                auto& cn = skeleton[static_cast<std::size_t>(c)];
                if (cn.type == 1) continue;
                const float d_soma = (Vec3f{cn.x_nm, cn.y_nm, cn.z_nm} - soma_c).norm();
                const float dist_cap = (d_soma < 18000.0f) ? 1150.0f : 850.0f;
                if (u_is_soma) {
                    shaft.clear();
                    int cur = c;
                    float dist_dn = 0.0f;
                    for (int steps = 0; steps < 30 && dist_dn < 8000.0f; ++steps) {
                        if (dist_dn >= 2000.0f) {
                            shaft.push_back(skeleton[static_cast<std::size_t>(cur)].radius_nm);
                        }
                        if (skel_children[static_cast<std::size_t>(cur)].size() != 1) break;
                        const int nc = skel_children[static_cast<std::size_t>(cur)][0];
                        dist_dn += seg_len(cur, nc);
                        cur = nc;
                    }
                    float root_cap = dist_cap;
                    if (!shaft.empty()) {
                        root_cap = std::min(root_cap, std::max(median_of(shaft) * 1.30f, 350.0f));
                    }
                    cn.radius_nm = std::clamp(cn.radius_nm, 140.0f, root_cap);
                } else {
                    const float pr = skeleton[static_cast<std::size_t>(u)].radius_nm;
                    const float step_cap = skel_children[static_cast<std::size_t>(c)].empty()
                        ? std::max(pr * 1.02f, 140.0f)
                        : std::max(pr * 1.06f, 220.0f);
                    cn.radius_nm = std::clamp(cn.radius_nm, 110.0f, std::min(step_cap, dist_cap));
                }
                rq.push(c);
            }
        }
    }

    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> sm_r(ns2);
        for (std::size_t i = 0; i < ns2; ++i) {
            sm_r[i] = skeleton[i].radius_nm;
            if (skeleton[i].type != 1 && skeleton[i].parent >= 0 &&
                skeleton[static_cast<std::size_t>(skeleton[i].parent)].type != 1 &&
                skel_children[i].size() == 1) {
                const float rp = skeleton[static_cast<std::size_t>(skeleton[i].parent)].radius_nm;
                const float rc = skeleton[static_cast<std::size_t>(skel_children[i][0])].radius_nm;
                sm_r[i] = 0.5f * skeleton[i].radius_nm + 0.25f * (rp + rc);
            }
        }
        for (std::size_t i = 0; i < ns2; ++i) {
            if (skeleton[i].type != 1) {
                skeleton[i].radius_nm = sm_r[i];
            }
        }
    }

    for (int it = 0; it < params.smooth_iterations; ++it) {
        std::vector<Vec3f> next_pos(ns2);
        for (std::size_t i = 0; i < ns2; ++i) {
            next_pos[i] = {skeleton[i].x_nm, skeleton[i].y_nm, skeleton[i].z_nm};
            if (skeleton[i].type != 1 && skeleton[i].parent >= 0 &&
                skeleton[static_cast<std::size_t>(skeleton[i].parent)].type != 1 &&
                skel_children[i].size() == 1) {
                const SkeletonNode& p = skeleton[static_cast<std::size_t>(skeleton[i].parent)];
                const SkeletonNode& c = skeleton[static_cast<std::size_t>(skel_children[i][0])];
                next_pos[i].x = 0.5f * skeleton[i].x_nm + 0.25f * (p.x_nm + c.x_nm);
                next_pos[i].y = 0.5f * skeleton[i].y_nm + 0.25f * (p.y_nm + c.y_nm);
                next_pos[i].z = 0.5f * skeleton[i].z_nm + 0.25f * (p.z_nm + c.z_nm);
            }
        }
        for (std::size_t i = 0; i < ns2; ++i) {
            skeleton[i].x_nm = next_pos[i].x;
            skeleton[i].y_nm = next_pos[i].y;
            skeleton[i].z_nm = next_pos[i].z;
        }
    }

    if (stats) {
        stats->total_components = static_cast<int>(comps.size());
        stats->kept_components = kept_count;
        stats->stitched_bridges = bridges_added;
        stats->reachable_verts = reachable.size();
        stats->soma_radius_nm = skeleton.empty() ? 0.0f : skeleton[0].radius_nm;
        stats->median_sdf_radius_nm = fallback_r;
    }

    return skeleton;
}

}
