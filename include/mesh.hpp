#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace teasar {

struct Vec3f {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    inline Vec3f operator+(const Vec3f& o) const { return {x + o.x, y + o.y, z + o.z}; }
    inline Vec3f operator-(const Vec3f& o) const { return {x - o.x, y - o.y, z - o.z}; }
    inline Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
    inline Vec3f& operator+=(const Vec3f& o) { x += o.x; y += o.y; z += o.z; return *this; }
    inline Vec3f& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    [[nodiscard]] inline float dot(const Vec3f& o) const {
        return x * o.x + y * o.y + z * o.z;
    }

    [[nodiscard]] inline Vec3f cross(const Vec3f& o) const {
        return {
            y * o.z - z * o.y,
            z * o.x - x * o.z,
            x * o.y - y * o.x
        };
    }

    [[nodiscard]] inline float norm_sq() const {
        return x * x + y * y + z * z;
    }

    [[nodiscard]] inline float norm() const {
        return std::sqrt(norm_sq());
    }

    [[nodiscard]] inline Vec3f normalized() const {
        const float n = norm();
        if (n <= 1e-12f) {
            return {0.0f, 0.0f, 0.0f};
        }
        const float inv = 1.0f / n;
        return {x * inv, y * inv, z * inv};
    }
};

struct TriangleMesh {
    std::vector<Vec3f> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    Vec3f soma_nm{};
    bool has_soma = false;
};

bool load_mesh_nmm(const std::string& filepath, TriangleMesh& mesh);

std::vector<Vec3f> compute_vertex_normals(const TriangleMesh& mesh,
                                          std::vector<Vec3f>* out_face_normals = nullptr);

}
