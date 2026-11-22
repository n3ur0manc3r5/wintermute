#include "mesh.hpp"

#include <cmath>
#include <cstring>
#include <fstream>

namespace teasar {

bool load_mesh_nmm(const std::string& filepath, TriangleMesh& mesh) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    char magic[4] = {};
    in.read(magic, 4);
    if (!in || std::memcmp(magic, "NMM1", 4) != 0) {
        return false;
    }

    uint32_t counts[2] = {};
    float soma[3] = {};
    in.read(reinterpret_cast<char*>(counts), sizeof(counts));
    in.read(reinterpret_cast<char*>(soma), sizeof(soma));
    if (!in || counts[0] == 0) {
        return false;
    }

    const std::size_t nv = static_cast<std::size_t>(counts[0]);
    const std::size_t nf = static_cast<std::size_t>(counts[1]);

    mesh.vertices.resize(nv);
    mesh.faces.resize(nf);
    mesh.soma_nm = {soma[0], soma[1], soma[2]};
    mesh.has_soma = std::isfinite(soma[0]) && std::isfinite(soma[1]) && std::isfinite(soma[2]);

    in.read(reinterpret_cast<char*>(mesh.vertices.data()),
            static_cast<std::streamsize>(nv * sizeof(Vec3f)));
    if (!in) {
        return false;
    }

    if (nf > 0) {
        in.read(reinterpret_cast<char*>(mesh.faces.data()),
                static_cast<std::streamsize>(nf * sizeof(std::array<uint32_t, 3>)));
        if (!in) {
            return false;
        }
    }

    return true;
}

std::vector<Vec3f> compute_vertex_normals(const TriangleMesh& mesh,
                                          std::vector<Vec3f>* out_face_normals) {
    const std::size_t nv = mesh.vertices.size();
    const std::size_t nf = mesh.faces.size();
    std::vector<Vec3f> vnorm(nv, {0.0f, 0.0f, 0.0f});
    std::vector<Vec3f> fnorm(nf, {0.0f, 0.0f, 0.0f});

    Vec3f ref{0.0f, 0.0f, 0.0f};
    if (nv > 0) {
        ref = mesh.vertices[0];
    }

    double signed_vol6 = 0.0;
    for (std::size_t f = 0; f < nf; ++f) {
        const auto& tri = mesh.faces[f];
        if (tri[0] >= nv || tri[1] >= nv || tri[2] >= nv) {
            continue;
        }
        const Vec3f p0 = mesh.vertices[tri[0]] - ref;
        const Vec3f p1 = mesh.vertices[tri[1]] - ref;
        const Vec3f p2 = mesh.vertices[tri[2]] - ref;
        const Vec3f cr = (p1 - p0).cross(p2 - p0);

        signed_vol6 += static_cast<double>(p0.x) * (static_cast<double>(p1.y) * p2.z - static_cast<double>(p1.z) * p2.y)
                     + static_cast<double>(p0.y) * (static_cast<double>(p1.z) * p2.x - static_cast<double>(p1.x) * p2.z)
                     + static_cast<double>(p0.z) * (static_cast<double>(p1.x) * p2.y - static_cast<double>(p1.y) * p2.x);

        vnorm[tri[0]] += cr;
        vnorm[tri[1]] += cr;
        vnorm[tri[2]] += cr;
        fnorm[f] = cr.normalized();
    }

    const float sign = (signed_vol6 < 0.0) ? -1.0f : 1.0f;
    for (std::size_t i = 0; i < nv; ++i) {
        vnorm[i] = (vnorm[i] * sign).normalized();
    }
    if (out_face_normals) {
        for (std::size_t f = 0; f < nf; ++f) {
            fnorm[f] *= sign;
        }
        *out_face_normals = std::move(fnorm);
    }

    return vnorm;
}

}
