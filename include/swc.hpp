#pragma once

#include "teasar.hpp"
#include "types.hpp"

#include <string>
#include <vector>

namespace teasar {

std::vector<SWCNode> build_swc_nodes(const std::vector<SkeletonNode>& skeleton);

bool validate_swc(const std::vector<SWCNode>& nodes, std::string* error = nullptr);

bool export_swc(const std::string& filepath,
                const std::vector<SWCNode>& nodes,
                const std::vector<std::string>& header = {});

bool load_swc(const std::string& filepath, std::vector<SWCNode>& out_nodes);

}
