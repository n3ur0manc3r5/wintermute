#include "swc.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stack>
#include <utility>

namespace teasar {

std::vector<SWCNode> build_swc_nodes(const std::vector<SkeletonNode>& skeleton) {
    if (skeleton.empty()) {
        return {};
    }

    const std::size_t n = skeleton.size();
    std::vector<std::vector<int>> children(n);
    std::vector<int> roots;

    for (std::size_t i = 0; i < n; ++i) {
        const int p = skeleton[i].parent;
        if (p < 0 || static_cast<std::size_t>(p) >= n) {
            roots.push_back(static_cast<int>(i));
        } else {
            children[static_cast<std::size_t>(p)].push_back(static_cast<int>(i));
        }
    }

    constexpr float kNmToUm = 1e-3f;
    std::vector<SWCNode> swc_nodes;
    swc_nodes.reserve(n);

    for (int root : roots) {
        std::stack<std::pair<int, int>> st;
        st.push({root, -1});

        while (!st.empty()) {
            const auto [old_idx, parent_swc_id] = st.top();
            st.pop();

            const int swc_id = static_cast<int>(swc_nodes.size()) + 1;
            const SkeletonNode& src = skeleton[static_cast<std::size_t>(old_idx)];

            SWCNode node{};
            node.id = swc_id;
            node.type = src.type;
            node.x = src.x_nm * kNmToUm;
            node.y = src.y_nm * kNmToUm;
            node.z = src.z_nm * kNmToUm;
            node.radius = src.radius_nm * kNmToUm;
            node.parent_id = parent_swc_id;
            swc_nodes.push_back(node);

            const auto& ch = children[static_cast<std::size_t>(old_idx)];
            for (auto it = ch.rbegin(); it != ch.rend(); ++it) {
                st.push({*it, swc_id});
            }
        }
    }

    return swc_nodes;
}

bool validate_swc(const std::vector<SWCNode>& nodes, std::string* error) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const SWCNode& n = nodes[i];
        const int expected_id = static_cast<int>(i) + 1;
        if (n.id != expected_id) {
            if (error) {
                *error = "non-contiguous swc id at index " + std::to_string(i) +
                         ": expected " + std::to_string(expected_id) +
                         ", got " + std::to_string(n.id);
            }
            return false;
        }
        if (n.parent_id != -1 && (n.parent_id < 1 || n.parent_id >= n.id)) {
            if (error) {
                *error = "invalid parent_id " + std::to_string(n.parent_id) +
                         " for node " + std::to_string(n.id) +
                         " (must be -1 or strictly less than id).";
            }
            return false;
        }
        if (!std::isfinite(n.x) || !std::isfinite(n.y) || !std::isfinite(n.z) ||
            !std::isfinite(n.radius) || n.radius < 0.0f) {
            if (error) {
                *error = "invalid coordinates or radius for node " + std::to_string(n.id);
            }
            return false;
        }
    }
    return true;
}

bool export_swc(const std::string& filepath,
                const std::vector<SWCNode>& nodes,
                const std::vector<std::string>& header) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        return false;
    }

    out << "# source wintermute\n";
    out << "# scale 1.0 1.0 1.0\n";
    out << "# units micrometers\n";
    out << "# columns id type x y z radius parent_id\n";
    for (const std::string& line : header) {
        out << "# " << line << "\n";
    }

    out << std::fixed << std::setprecision(6);
    for (const SWCNode& n : nodes) {
        out << n.id << ' '
            << n.type << ' '
            << n.x << ' '
            << n.y << ' '
            << n.z << ' '
            << n.radius << ' '
            << n.parent_id << '\n';
    }

    return out.good();
}

bool load_swc(const std::string& filepath, std::vector<SWCNode>& out_nodes) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        return false;
    }

    out_nodes.clear();
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        std::istringstream iss(line);
        SWCNode n{};
        if (iss >> n.id >> n.type >> n.x >> n.y >> n.z >> n.radius >> n.parent_id) {
            out_nodes.push_back(n);
        }
    }
    return !out_nodes.empty();
}

}
