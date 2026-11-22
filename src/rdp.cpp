#include "rdp.hpp"

#include <algorithm>
#include <cmath>
#include <stack>
#include <vector>

namespace teasar {

float point_to_segment_distance_nm(const SkeletonNode& p,
                                   const SkeletonNode& a,
                                   const SkeletonNode& b) {
    const float abx = b.x_nm - a.x_nm;
    const float aby = b.y_nm - a.y_nm;
    const float abz = b.z_nm - a.z_nm;

    const float apx = p.x_nm - a.x_nm;
    const float apy = p.y_nm - a.y_nm;
    const float apz = p.z_nm - a.z_nm;

    const float ab_len_sq = abx * abx + aby * aby + abz * abz;
    if (ab_len_sq <= 1e-12f) {
        return std::sqrt(apx * apx + apy * apy + apz * apz);
    }

    const float t = std::clamp((apx * abx + apy * aby + apz * abz) / ab_len_sq, 0.0f, 1.0f);
    const float cx = a.x_nm + t * abx;
    const float cy = a.y_nm + t * aby;
    const float cz = a.z_nm + t * abz;

    const float dx = p.x_nm - cx;
    const float dy = p.y_nm - cy;
    const float dz = p.z_nm - cz;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

namespace {

void rdp_recursive(const std::vector<SkeletonNode>& nodes,
                   const std::vector<int>& chain,
                   std::size_t start_pos,
                   std::size_t end_pos,
                   float epsilon_nm,
                   std::vector<uint8_t>& keep_flags) {
    if (end_pos <= start_pos + 1) {
        return;
    }

    const SkeletonNode& a = nodes[static_cast<std::size_t>(chain[start_pos])];
    const SkeletonNode& b = nodes[static_cast<std::size_t>(chain[end_pos])];

    float max_dist = -1.0f;
    std::size_t max_pos = start_pos;

    for (std::size_t i = start_pos + 1; i < end_pos; ++i) {
        const float d = point_to_segment_distance_nm(nodes[static_cast<std::size_t>(chain[i])], a, b);
        if (d > max_dist) {
            max_dist = d;
            max_pos = i;
        }
    }

    if (max_dist > epsilon_nm) {
        keep_flags[max_pos] = 1;
        rdp_recursive(nodes, chain, start_pos, max_pos, epsilon_nm, keep_flags);
        rdp_recursive(nodes, chain, max_pos, end_pos, epsilon_nm, keep_flags);
    }
}

std::vector<int> simplify_chain(const std::vector<SkeletonNode>& nodes,
                                const std::vector<int>& chain,
                                float epsilon_nm) {
    if (chain.size() <= 2) {
        return chain;
    }

    std::vector<uint8_t> keep(chain.size(), 0);
    keep.front() = 1;
    keep.back() = 1;

    rdp_recursive(nodes, chain, 0, chain.size() - 1, epsilon_nm, keep);

    std::vector<int> result;
    result.reserve(chain.size());
    for (std::size_t i = 0; i < chain.size(); ++i) {
        if (keep[i] != 0) {
            result.push_back(chain[i]);
        }
    }
    return result;
}

}

std::vector<SkeletonNode> simplify_skeleton_rdp(const std::vector<SkeletonNode>& nodes,
                                                float epsilon_nm) {
    if (nodes.empty() || epsilon_nm <= 0.0f) {
        return nodes;
    }

    const std::size_t n = nodes.size();
    std::vector<std::vector<int>> children(n);
    std::vector<int> roots;

    for (std::size_t i = 0; i < n; ++i) {
        const int p = nodes[i].parent;
        if (p < 0 || static_cast<std::size_t>(p) >= n) {
            roots.push_back(static_cast<int>(i));
        } else {
            children[static_cast<std::size_t>(p)].push_back(static_cast<int>(i));
        }
    }

    std::vector<SkeletonNode> simplified;
    simplified.reserve(n);
    std::vector<int> old_to_new(n, -1);

    for (int root : roots) {
        SkeletonNode root_copy = nodes[static_cast<std::size_t>(root)];
        root_copy.parent = -1;
        const int new_root_idx = static_cast<int>(simplified.size());
        simplified.push_back(root_copy);
        old_to_new[static_cast<std::size_t>(root)] = new_root_idx;

        std::stack<int> keypoints;
        keypoints.push(root);

        while (!keypoints.empty()) {
            const int u = keypoints.top();
            keypoints.pop();

            for (int child : children[static_cast<std::size_t>(u)]) {
                std::vector<int> chain;
                chain.push_back(u);
                int curr = child;
                chain.push_back(curr);

                while (children[static_cast<std::size_t>(curr)].size() == 1 &&
                       nodes[static_cast<std::size_t>(curr)].type != 1) {
                    curr = children[static_cast<std::size_t>(curr)][0];
                    chain.push_back(curr);
                }

                const std::vector<int> kept = simplify_chain(nodes, chain, epsilon_nm);

                int prev_new = old_to_new[static_cast<std::size_t>(u)];
                for (std::size_t k = 1; k < kept.size(); ++k) {
                    const int old_idx = kept[k];
                    SkeletonNode copy = nodes[static_cast<std::size_t>(old_idx)];
                    copy.parent = prev_new;
                    const int new_idx = static_cast<int>(simplified.size());
                    simplified.push_back(copy);
                    old_to_new[static_cast<std::size_t>(old_idx)] = new_idx;
                    prev_new = new_idx;
                }

                const int end_node = chain.back();
                if (children[static_cast<std::size_t>(end_node)].size() >= 2 ||
                    (nodes[static_cast<std::size_t>(end_node)].type == 1 &&
                     !children[static_cast<std::size_t>(end_node)].empty())) {
                    keypoints.push(end_node);
                }
            }
        }
    }

    return simplified;
}

}
