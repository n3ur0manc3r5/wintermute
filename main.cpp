#include "mesh.hpp"
#include "mesh_teasar.hpp"
#include "rdp.hpp"
#include "swc.hpp"
#include "teasar.hpp"
#include "types.hpp"
#include "viewer.hpp"
#include "volume_io.hpp"
#include "voxelizer.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace {

std::tuple<int, int, int> count_topology(const std::vector<teasar::SkeletonNode>& skel) {
    std::vector<int> child_counts(skel.size(), 0);
    int roots = 0;
    for (const auto& n : skel) {
        if (n.parent == -1) {
            ++roots;
        } else {
            ++child_counts[static_cast<std::size_t>(n.parent)];
        }
    }
    int branches = 0;
    int leaves = 0;
    for (std::size_t i = 0; i < skel.size(); ++i) {
        if (child_counts[i] >= 2) ++branches;
        if (child_counts[i] == 0) ++leaves;
    }
    return {roots, branches, leaves};
}

double compute_cable_length_um(const std::vector<teasar::SkeletonNode>& skel) {
    double total_nm = 0.0;
    for (const auto& n : skel) {
        if (n.parent >= 0 && static_cast<std::size_t>(n.parent) < skel.size()) {
            const auto& p = skel[static_cast<std::size_t>(n.parent)];
            const double dx = static_cast<double>(n.x_nm - p.x_nm);
            const double dy = static_cast<double>(n.y_nm - p.y_nm);
            const double dz = static_cast<double>(n.z_nm - p.z_nm);
            total_nm += std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }
    return total_nm * 1e-3;
}

int run_voxel_pipeline(const std::string& input_path,
                       const std::string& swc_path,
                       const std::string& html_path,
                       teasar::TeasarParams params,
                       float rdp_epsilon_nm) {
    const auto t0 = std::chrono::steady_clock::now();

    VolumeGrid grid;
    teasar::VolumeMetadata meta;
    if (!teasar::load_volume_nmv(input_path, grid, meta)) {
        std::cerr << "error: failed to load volume " << input_path << '\n';
        return EXIT_FAILURE;
    }

    std::size_t fg_count = 0;
    for (uint8_t m : grid.mask) {
        if (m != 0) ++fg_count;
    }
    std::cout << "loaded " << input_path << ": "
              << grid.dim_x << "x" << grid.dim_y << "x" << grid.dim_z
              << " voxels (" << grid.scale.dx << "x" << grid.scale.dy << "x" << grid.scale.dz
              << " nm), fg=" << fg_count << '\n';

    if (meta.root_voxel.x >= 0 && grid.is_valid(meta.root_voxel.x, meta.root_voxel.y, meta.root_voxel.z) &&
        grid.mask[static_cast<std::size_t>(grid.to_index(meta.root_voxel.x, meta.root_voxel.y, meta.root_voxel.z))] != 0) {
        params.explicit_root = meta.root_voxel;
    }

    std::vector<teasar::SkeletonNode> raw_skel = teasar::extract_skeleton(grid, params);
    const auto t1 = std::chrono::steady_clock::now();

    for (auto& n : raw_skel) {
        n.x_nm += meta.origin_x_nm;
        n.y_nm += meta.origin_y_nm;
        n.z_nm += meta.origin_z_nm;
    }

    const std::vector<teasar::SkeletonNode> simplified = teasar::simplify_skeleton_rdp(raw_skel, rdp_epsilon_nm);
    const auto t2 = std::chrono::steady_clock::now();

    const auto [raw_r, raw_b, raw_l] = count_topology(raw_skel);
    const auto [sim_r, sim_b, sim_l] = count_topology(simplified);
    const double cable_um = compute_cable_length_um(simplified);

    const std::vector<SWCNode> raw_swc = teasar::build_swc_nodes(raw_skel);
    const std::vector<SWCNode> swc_nodes = teasar::build_swc_nodes(simplified);
    std::string err;
    if (!teasar::validate_swc(swc_nodes, &err)) {
        std::cerr << "error: swc validation failed: " << err << '\n';
        return EXIT_FAILURE;
    }

    if (!teasar::export_swc(swc_path, swc_nodes, {"source " + input_path})) {
        std::cerr << "error: failed to write swc " << swc_path << '\n';
        return EXIT_FAILURE;
    }

    if (!html_path.empty()) {
        const bool ok = teasar::export_html_viewer(
            html_path,
            grid,
            raw_swc,
            swc_nodes,
            params.invalidation_scale,
            params.invalidation_const,
            1,
            meta.origin_x_nm,
            meta.origin_y_nm,
            meta.origin_z_nm,
            5000);
        if (!ok) {
            std::cerr << "error: failed to write html viewer " << html_path << '\n';
            return EXIT_FAILURE;
        }
    }

    const double sec_teasar = std::chrono::duration<double>(t1 - t0).count();
    const double sec_total = std::chrono::duration<double>(t2 - t0).count();
    std::cout << "raw skeleton: " << raw_skel.size() << " nodes ("
              << raw_r << " root, " << raw_b << " branch pts, " << raw_l << " tips)\n";
    std::cout << "simplified swc (eps=" << rdp_epsilon_nm << " nm): " << swc_nodes.size() << " nodes ("
              << sim_r << " root, " << sim_b << " branch pts, " << sim_l << " tips), cable="
              << cable_um << " um\n";
    std::cout << "timing: teasar=" << sec_teasar << "s, total=" << sec_total << "s\n";
    std::cout << "exported swc: " << std::filesystem::absolute(swc_path).string() << '\n';
    if (!html_path.empty()) {
        std::cout << "exported html: " << std::filesystem::absolute(html_path).string() << '\n';
    }
    return EXIT_SUCCESS;
}

int run_mesh_pipeline(const std::string& mesh_path,
                      const std::string& swc_path,
                      const std::string& html_path,
                      const teasar::MeshTeasarParams& params,
                      float rdp_epsilon_nm) {
    const auto t0 = std::chrono::steady_clock::now();

    teasar::TriangleMesh mesh;
    if (!teasar::load_mesh_nmm(mesh_path, mesh)) {
        std::cerr << "error: failed to load mesh " << mesh_path << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "loaded mesh " << mesh_path << ": verts=" << mesh.vertices.size()
              << ", faces=" << mesh.faces.size() << '\n';

    teasar::MeshSkeletonStats stats;
    const std::vector<teasar::SkeletonNode> raw_skel = teasar::extract_mesh_skeleton(mesh, params, &stats);
    const auto t1 = std::chrono::steady_clock::now();

    const std::vector<teasar::SkeletonNode> simplified = teasar::simplify_skeleton_rdp(raw_skel, rdp_epsilon_nm);
    const auto t2 = std::chrono::steady_clock::now();

    const auto [raw_r, raw_b, raw_l] = count_topology(raw_skel);
    const auto [sim_r, sim_b, sim_l] = count_topology(simplified);
    const double cable_um = compute_cable_length_um(simplified);

    const std::vector<SWCNode> raw_swc = teasar::build_swc_nodes(raw_skel);
    const std::vector<SWCNode> swc_nodes = teasar::build_swc_nodes(simplified);
    std::string err;
    if (!teasar::validate_swc(swc_nodes, &err)) {
        std::cerr << "error: swc validation failed: " << err << '\n';
        return EXIT_FAILURE;
    }

    if (!teasar::export_swc(swc_path, swc_nodes, {"source " + mesh_path})) {
        std::cerr << "error: failed to write swc " << swc_path << '\n';
        return EXIT_FAILURE;
    }

    if (!html_path.empty()) {
        const bool ok = teasar::export_mesh_html_viewer(
            html_path,
            mesh,
            raw_swc,
            swc_nodes,
            params.invalidation_scale,
            params.invalidation_const_nm,
            6000);
        if (!ok) {
            std::cerr << "error: failed to write html viewer " << html_path << '\n';
            return EXIT_FAILURE;
        }
    }

    const double sec_teasar = std::chrono::duration<double>(t1 - t0).count();
    const double sec_total = std::chrono::duration<double>(t2 - t0).count();
    std::cout << "mesh graph: components=" << stats.total_components
              << " (kept=" << stats.kept_components << ", stitched=" << stats.stitched_bridges
              << "), reachable=" << stats.reachable_verts
              << ", soma_r=" << (stats.soma_radius_nm * 1e-3f) << " um\n";
    std::cout << "raw skeleton: " << raw_skel.size() << " nodes ("
              << raw_r << " root, " << raw_b << " branch pts, " << raw_l << " tips)\n";
    std::cout << "simplified swc (eps=" << rdp_epsilon_nm << " nm): " << swc_nodes.size() << " nodes ("
              << sim_r << " root, " << sim_b << " branch pts, " << sim_l << " tips), cable="
              << cable_um << " um\n";
    std::cout << "timing: mesh_teasar=" << sec_teasar << "s, total=" << sec_total << "s\n";
    std::cout << "exported swc: " << std::filesystem::absolute(swc_path).string() << '\n';
    if (!html_path.empty()) {
        std::cout << "exported html: " << std::filesystem::absolute(html_path).string() << '\n';
    }
    return EXIT_SUCCESS;
}

int run_voxelize_pipeline(const std::string& swc_in_path,
                          const std::string& out_vol_path,
                          const std::string& out_obj_path,
                          const std::string& html_path,
                          const std::string& compare_gt_path,
                          const std::string& compare_mesh_path,
                          const std::string& render_ppm_path,
                          float focus_soma_um,
                          const teasar::VoxelizerParams& vparams) {
    const auto t0 = std::chrono::steady_clock::now();

    std::vector<SWCNode> swc;
    if (!teasar::load_swc(swc_in_path, swc)) {
        std::cerr << "error: failed to load swc " << swc_in_path << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "loaded swc " << swc_in_path << ": nodes=" << swc.size() << '\n';

    const bool need_volume = !compare_gt_path.empty() || !out_vol_path.empty();
    const bool use_hybrid_mesh = !need_volume;

    VolumeGrid recon_grid;
    teasar::VolumeMetadata recon_meta;
    std::vector<float> recon_sdf;

    if (!compare_gt_path.empty()) {
        VolumeGrid gt_grid;
        teasar::VolumeMetadata gt_meta;
        if (!teasar::load_volume_nmv(compare_gt_path, gt_grid, gt_meta)) {
            std::cerr << "error: failed to load ground truth volume " << compare_gt_path << '\n';
            return EXIT_FAILURE;
        }
        const teasar::VolumeOverlapStats ov = teasar::evaluate_swc_against_volume(
            gt_grid, gt_meta, swc, vparams, &recon_grid, &recon_sdf);
        recon_meta = gt_meta;
        std::cout << "ground truth overlap (" << gt_grid.dim_x << "x" << gt_grid.dim_y << "x" << gt_grid.dim_z
                  << "): gt=" << ov.gt_voxels << ", recon=" << ov.recon_voxels
                  << ", hit=" << ov.intersection_voxels
                  << ", dice=" << ov.dice << ", iou=" << ov.iou
                  << ", recall=" << ov.recall << ", precision=" << ov.precision << '\n';
    } else if (need_volume) {
        if (focus_soma_um > 0.0f && out_vol_path.empty() && out_obj_path.empty() && !swc.empty()) {
            const teasar::Vec3f sc{swc[0].x * 1000.0f, swc[0].y * 1000.0f, swc[0].z * 1000.0f};
            if (!teasar::rasterize_swc_roi_grid(swc, vparams, sc, focus_soma_um * 850.0f, recon_grid, recon_meta, recon_sdf)) {
                std::cerr << "error: failed to rasterize swc roi\n";
                return EXIT_FAILURE;
            }
        } else {
            if (!teasar::rasterize_swc_auto_grid(swc, vparams, recon_grid, recon_meta, recon_sdf)) {
                std::cerr << "error: failed to rasterize swc\n";
                return EXIT_FAILURE;
            }
        }
        std::size_t fg = 0;
        for (uint8_t m : recon_grid.mask) {
            if (m != 0) ++fg;
        }
        std::cout << "voxelized volume: " << recon_grid.dim_x << "x" << recon_grid.dim_y << "x" << recon_grid.dim_z
                  << " (" << recon_grid.scale.dx << "x" << recon_grid.scale.dy << "x" << recon_grid.scale.dz
                  << " nm), fg=" << fg << '\n';
    }

    if (!out_vol_path.empty()) {
        if (!teasar::save_volume_nmv(out_vol_path, recon_grid, recon_meta)) {
            std::cerr << "error: failed to write volume " << out_vol_path << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "exported volume: " << std::filesystem::absolute(out_vol_path).string() << '\n';
    }

    teasar::TriangleMesh iso_mesh;
    if (!out_obj_path.empty() || !render_ppm_path.empty() || !html_path.empty()) {
        if (use_hybrid_mesh) {
            iso_mesh = teasar::build_hybrid_neuron_mesh(swc, vparams);
            std::cout << "hybrid surface mesh: verts=" << iso_mesh.vertices.size()
                      << ", faces=" << iso_mesh.faces.size() << '\n';
        } else {
            iso_mesh = teasar::extract_isosurface_mesh(recon_grid, recon_meta, recon_sdf);
            std::cout << "isosurface mesh: verts=" << iso_mesh.vertices.size()
                      << ", faces=" << iso_mesh.faces.size() << '\n';
        }
    }

    if (!out_obj_path.empty()) {
        if (!teasar::export_mesh_obj(out_obj_path, iso_mesh)) {
            std::cerr << "error: failed to write obj " << out_obj_path << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "exported obj: " << std::filesystem::absolute(out_obj_path).string() << '\n';
    }

    teasar::TriangleMesh ref_mesh;
    bool has_ref_mesh = false;
    if (!compare_mesh_path.empty() && (!render_ppm_path.empty() || !html_path.empty())) {
        if (!teasar::load_mesh_nmm(compare_mesh_path, ref_mesh)) {
            std::cerr << "error: failed to load comparison mesh " << compare_mesh_path << '\n';
            return EXIT_FAILURE;
        }
        has_ref_mesh = true;
    }

    if (!html_path.empty()) {
        if (!teasar::export_mesh_webgl_html(html_path, iso_mesh, swc, has_ref_mesh ? &ref_mesh : nullptr)) {
            std::cerr << "error: failed to write html " << html_path << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "exported html: " << std::filesystem::absolute(html_path).string() << '\n';
    }

    if (!render_ppm_path.empty()) {
        const teasar::TriangleMesh& left_mesh = has_ref_mesh ? ref_mesh : iso_mesh;
        bool use_focus = false;
        teasar::Vec3f focus_c{};
        float focus_span = 0.0f;
        if (focus_soma_um > 0.0f && !swc.empty()) {
            use_focus = true;
            focus_c = {swc[0].x * 1000.0f, swc[0].y * 1000.0f, swc[0].z * 1000.0f};
            focus_span = focus_soma_um * 1000.0f;
        }
        if (!teasar::render_mesh_comparison_ppm(render_ppm_path, left_mesh, iso_mesh, 1400, 760, 0.35f, 0.12f,
                                                use_focus, focus_c, focus_span)) {
            std::cerr << "error: failed to render ppm " << render_ppm_path << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "exported render: " << std::filesystem::absolute(render_ppm_path).string() << '\n';
    }

    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "timing: voxelize=" << std::chrono::duration<double>(t1 - t0).count() << "s\n";
    return EXIT_SUCCESS;
}

}

int main(int argc, char** argv) {
    std::string input_path;
    std::string mesh_path;
    std::string voxelize_swc_path;
    std::string swc_path = "neuron_skeleton.swc";
    std::string html_path;
    std::string out_vol_path;
    std::string out_obj_path;
    std::string compare_gt_path;
    std::string compare_mesh_path;
    std::string render_ppm_path;

    teasar::TeasarParams vparams;
    vparams.pdrf_scale = 5000.0f;
    vparams.pdrf_exponent = 4;
    vparams.invalidation_scale = 1.8f;
    vparams.invalidation_const = 2200.0f;
    vparams.min_component_voxels = 50;

    teasar::MeshTeasarParams mparams;
    teasar::VoxelizerParams voxparams;
    float rdp_epsilon_nm = 200.0f;
    float focus_soma_um = 0.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (arg == "--mesh" && i + 1 < argc) mesh_path = argv[++i];
        else if (arg == "--voxelize" && i + 1 < argc) voxelize_swc_path = argv[++i];
        else if (arg == "--swc" && i + 1 < argc) swc_path = argv[++i];
        else if (arg == "--html" && i + 1 < argc) html_path = argv[++i];
        else if (arg == "--out-vol" && i + 1 < argc) out_vol_path = argv[++i];
        else if (arg == "--out-obj" && i + 1 < argc) out_obj_path = argv[++i];
        else if (arg == "--compare-gt" && i + 1 < argc) compare_gt_path = argv[++i];
        else if (arg == "--compare-mesh" && i + 1 < argc) compare_mesh_path = argv[++i];
        else if (arg == "--render-ppm" && i + 1 < argc) render_ppm_path = argv[++i];
        else if (arg == "--vox-res" && i + 1 < argc) {
            const float r = std::stof(argv[++i]);
            voxparams.scale = {r, r, r};
        } else if (arg == "--soma-blend" && i + 1 < argc) voxparams.soma_blend_nm = std::stof(argv[++i]);
        else if (arg == "--branch-blend" && i + 1 < argc) voxparams.branch_blend_nm = std::stof(argv[++i]);
        else if (arg == "--min-r-scale" && i + 1 < argc) voxparams.min_radius_scale = std::stof(argv[++i]);
        else if (arg == "--tube-sides" && i + 1 < argc) voxparams.tube_sides = std::stoi(argv[++i]);
        else if (arg == "--soma-vox" && i + 1 < argc) voxparams.soma_voxel_nm = std::stof(argv[++i]);
        else if (arg == "--focus-soma" && i + 1 < argc) focus_soma_um = std::stof(argv[++i]);
        else if (arg == "--inv-scale" && i + 1 < argc) {
            const float v = std::stof(argv[++i]);
            vparams.invalidation_scale = v;
            mparams.invalidation_scale = v;
        } else if (arg == "--inv-const" && i + 1 < argc) {
            const float v = std::stof(argv[++i]);
            vparams.invalidation_const = v;
            mparams.invalidation_const_nm = v;
        } else if (arg == "--rdp" && i + 1 < argc) rdp_epsilon_nm = std::stof(argv[++i]);
        else if (arg == "--min-comp" && i + 1 < argc) {
            const int v = std::stoi(argv[++i]);
            vparams.min_component_voxels = v;
            mparams.min_component_verts = v;
        } else if (arg == "--stitch" && i + 1 < argc) mparams.stitch_max_dist_nm = std::stof(argv[++i]);
        else if (arg == "--min-branch" && i + 1 < argc) mparams.min_branch_len_nm = std::stof(argv[++i]);
        else if (arg == "--soma-r" && i + 1 < argc) mparams.soma_invalidation_radius_nm = std::stof(argv[++i]);
        else if (arg == "--smooth" && i + 1 < argc) mparams.smooth_iterations = std::stoi(argv[++i]);
        else if (arg == "--pdrf-scale" && i + 1 < argc) {
            const float v = std::stof(argv[++i]);
            vparams.pdrf_scale = v;
            mparams.pdrf_scale = v;
        } else if (arg == "--pdrf-exp" && i + 1 < argc) {
            const int v = std::stoi(argv[++i]);
            vparams.pdrf_exponent = v;
            mparams.pdrf_exponent = v;
        }
    }

    if (!voxelize_swc_path.empty()) {
        return run_voxelize_pipeline(voxelize_swc_path,
                                     out_vol_path,
                                     out_obj_path,
                                     html_path,
                                     compare_gt_path,
                                     compare_mesh_path,
                                     render_ppm_path,
                                     focus_soma_um,
                                     voxparams);
    }
    if (!mesh_path.empty()) {
        return run_mesh_pipeline(mesh_path, swc_path, html_path, mparams, rdp_epsilon_nm);
    }
    if (!input_path.empty()) {
        return run_voxel_pipeline(input_path, swc_path, html_path, vparams, rdp_epsilon_nm);
    }

    std::cerr << "usage:\n"
              << "  neuromancers --mesh <file.nmm> --swc <out.swc> [--inv-scale 2.0] [--inv-const 1800] [--rdp 500]\n"
              << "  neuromancers --input <file.nmv> --swc <out.swc> [--inv-scale 1.8] [--inv-const 2200] [--rdp 500]\n"
              << "  neuromancers --voxelize <in.swc> [--vox-res 500] [--out-vol out.nmv] [--out-obj out.obj] "
              << "[--compare-gt gt.nmv] [--compare-mesh gt.nmm] [--render-ppm out.ppm]\n";
    return EXIT_FAILURE;
}
