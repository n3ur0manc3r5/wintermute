#include "edt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace teasar {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void edt_1d(const double* f,
            double* out,
            int n,
            double w,
            std::vector<int>& v,
            std::vector<double>& z) {
    const double w2 = w * w;
    int k = -1;

    for (int q = 0; q < n; ++q) {
        if (!std::isfinite(f[q])) {
            continue;
        }
        if (k == -1) {
            k = 0;
            v[0] = q;
            z[0] = -kInf;
            z[1] = kInf;
            continue;
        }

        double s = 0.0;
        while (k > 0) {
            const int vk = v[k];
            const double qd = static_cast<double>(q);
            const double vkd = static_cast<double>(vk);
            s = ((f[q] + w2 * qd * qd) - (f[vk] + w2 * vkd * vkd)) /
                (2.0 * w2 * (qd - vkd));
            if (s <= z[k]) {
                --k;
            } else {
                break;
            }
        }

        {
            const int vk = v[k];
            const double qd = static_cast<double>(q);
            const double vkd = static_cast<double>(vk);
            s = ((f[q] + w2 * qd * qd) - (f[vk] + w2 * vkd * vkd)) /
                (2.0 * w2 * (qd - vkd));
        }

        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = kInf;
    }

    if (k == -1) {
        std::fill(out, out + n, kInf);
        return;
    }

    int idx = 0;
    for (int p = 0; p < n; ++p) {
        const double pd = static_cast<double>(p);
        while (z[idx + 1] < pd) {
            ++idx;
        }
        const double diff = pd - static_cast<double>(v[idx]);
        out[p] = w2 * diff * diff + f[v[idx]];
    }
}

void edt_1d_with_border(const double* in,
                        double* out,
                        int n,
                        double w,
                        bool black_border,
                        std::vector<double>& padded_in,
                        std::vector<double>& padded_out,
                        std::vector<int>& v,
                        std::vector<double>& z) {
    if (!black_border) {
        edt_1d(in, out, n, w, v, z);
        return;
    }
    padded_in[0] = 0.0;
    std::copy(in, in + n, padded_in.data() + 1);
    padded_in[n + 1] = 0.0;
    edt_1d(padded_in.data(), padded_out.data(), n + 2, w, v, z);
    std::copy(padded_out.data() + 1, padded_out.data() + 1 + n, out);
}

}

void compute_dfb(VolumeGrid& grid, bool black_border) {
    const int nx = grid.dim_x;
    const int ny = grid.dim_y;
    const int nz = grid.dim_z;
    const std::size_t total = grid.size();

    if (total == 0) {
        grid.dfb.clear();
        return;
    }

    const float finf = std::numeric_limits<float>::infinity();
    grid.dfb.assign(total, finf);

    bool has_background = false;
    for (std::size_t i = 0; i < total; ++i) {
        if (grid.mask[i] == 0) {
            grid.dfb[i] = 0.0f;
            has_background = true;
        }
    }

    const bool use_border = black_border || !has_background;

    const int max_dim = std::max({nx, ny, nz});
    std::vector<double> line_in(static_cast<std::size_t>(max_dim));
    std::vector<double> line_out(static_cast<std::size_t>(max_dim));
    std::vector<double> padded_in(static_cast<std::size_t>(max_dim + 2));
    std::vector<double> padded_out(static_cast<std::size_t>(max_dim + 2));
    std::vector<int> v(static_cast<std::size_t>(max_dim + 2));
    std::vector<double> z(static_cast<std::size_t>(max_dim + 3));

    const std::size_t xy_stride = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);

    for (int z_idx = 0; z_idx < nz; ++z_idx) {
        const std::size_t z_off = static_cast<std::size_t>(z_idx) * xy_stride;
        for (int y_idx = 0; y_idx < ny; ++y_idx) {
            const std::size_t row_off = z_off + static_cast<std::size_t>(y_idx) * static_cast<std::size_t>(nx);
            for (int x_idx = 0; x_idx < nx; ++x_idx) {
                line_in[static_cast<std::size_t>(x_idx)] = static_cast<double>(grid.dfb[row_off + static_cast<std::size_t>(x_idx)]);
            }
            edt_1d_with_border(line_in.data(),
                               line_out.data(),
                               nx,
                               static_cast<double>(grid.scale.dx),
                               use_border,
                               padded_in,
                               padded_out,
                               v,
                               z);
            for (int x_idx = 0; x_idx < nx; ++x_idx) {
                grid.dfb[row_off + static_cast<std::size_t>(x_idx)] = static_cast<float>(line_out[static_cast<std::size_t>(x_idx)]);
            }
        }
    }

    for (int z_idx = 0; z_idx < nz; ++z_idx) {
        const std::size_t z_off = static_cast<std::size_t>(z_idx) * xy_stride;
        for (int x_idx = 0; x_idx < nx; ++x_idx) {
            for (int y_idx = 0; y_idx < ny; ++y_idx) {
                line_in[static_cast<std::size_t>(y_idx)] = static_cast<double>(grid.dfb[z_off + static_cast<std::size_t>(y_idx) * static_cast<std::size_t>(nx) + static_cast<std::size_t>(x_idx)]);
            }
            edt_1d_with_border(line_in.data(),
                               line_out.data(),
                               ny,
                               static_cast<double>(grid.scale.dy),
                               use_border,
                               padded_in,
                               padded_out,
                               v,
                               z);
            for (int y_idx = 0; y_idx < ny; ++y_idx) {
                grid.dfb[z_off + static_cast<std::size_t>(y_idx) * static_cast<std::size_t>(nx) + static_cast<std::size_t>(x_idx)] = static_cast<float>(line_out[static_cast<std::size_t>(y_idx)]);
            }
        }
    }

    for (int y_idx = 0; y_idx < ny; ++y_idx) {
        const std::size_t y_off = static_cast<std::size_t>(y_idx) * static_cast<std::size_t>(nx);
        for (int x_idx = 0; x_idx < nx; ++x_idx) {
            const std::size_t base = y_off + static_cast<std::size_t>(x_idx);
            for (int z_idx = 0; z_idx < nz; ++z_idx) {
                line_in[static_cast<std::size_t>(z_idx)] = static_cast<double>(grid.dfb[static_cast<std::size_t>(z_idx) * xy_stride + base]);
            }
            edt_1d_with_border(line_in.data(),
                               line_out.data(),
                               nz,
                               static_cast<double>(grid.scale.dz),
                               use_border,
                               padded_in,
                               padded_out,
                               v,
                               z);
            for (int z_idx = 0; z_idx < nz; ++z_idx) {
                grid.dfb[static_cast<std::size_t>(z_idx) * xy_stride + base] = static_cast<float>(line_out[static_cast<std::size_t>(z_idx)]);
            }
        }
    }

    for (std::size_t i = 0; i < total; ++i) {
        if (grid.mask[i] != 0 && std::isfinite(grid.dfb[i])) {
            grid.dfb[i] = std::sqrt(std::max(0.0f, grid.dfb[i]));
        } else {
            grid.dfb[i] = 0.0f;
        }
    }
}

}
