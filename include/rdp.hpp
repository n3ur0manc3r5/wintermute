#pragma once

#include "teasar.hpp"

#include <vector>

namespace teasar {

float point_to_segment_distance_nm(const SkeletonNode& p,
                                   const SkeletonNode& a,
                                   const SkeletonNode& b);

std::vector<SkeletonNode> simplify_skeleton_rdp(const std::vector<SkeletonNode>& nodes,
                                                float epsilon_nm);

}
