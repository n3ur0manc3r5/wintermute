import argparse
import csv
import os
import struct
import numpy as np
from scipy import ndimage
import caveclient
from cloudvolume import CloudVolume


PUBLIC_SEG_URL = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300"


def resolve_seg_source(datastack: str) -> str:
    try:
        client = caveclient.CAVEclient(datastack)
        src = client.info.segmentation_source()
        if src:
            print(f"caveclient connected to {datastack} (v{client.materialize.version}): {src}")
            return src
    except Exception as exc:
        print(f"caveclient no active token ({type(exc).__name__.lower()}); using public microns v1300 bucket.")
    return PUBLIC_SEG_URL


def load_node_row(csv_path: str, root_id: int):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["pt_root_id"]) == root_id:
                return row
    raise ValueError(f"pt_root_id {root_id} not found in {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-csv", default="nodes.csv")
    parser.add_argument("--root-id", type=int, required=True)
    parser.add_argument("--datastack", default="minnie65_public")
    parser.add_argument("--mip", type=int, default=5)
    parser.add_argument("--pad-x-um", type=float, default=95.0)
    parser.add_argument("--pad-y-up-um", type=float, default=180.0)
    parser.add_argument("--pad-y-down-um", type=float, default=90.0)
    parser.add_argument("--pad-z-um", type=float, default=95.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    row = load_node_row(args.nodes_csv, args.root_id)
    cell_type = row["cell_type"].lower()
    soma_um = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
    print(f"target: {args.root_id} ({cell_type}, node_id={row['node_id']}) soma={soma_um.round(2).tolist()} um")

    seg_url = resolve_seg_source(args.datastack)
    cv = CloudVolume(seg_url, mip=args.mip, progress=False, fill_missing=True)
    res_nm = np.array(cv.resolution, dtype=np.float64)
    vol_shape = np.array(cv.shape[:3], dtype=np.int64)

    lo_um = np.array([
        soma_um[0] - args.pad_x_um,
        soma_um[1] - args.pad_y_up_um,
        soma_um[2] - args.pad_z_um,
    ])
    hi_um = np.array([
        soma_um[0] + args.pad_x_um,
        soma_um[1] + args.pad_y_down_um,
        soma_um[2] + args.pad_z_um,
    ])

    lo_vox = np.maximum(0, np.floor(lo_um * 1000.0 / res_nm).astype(np.int64))
    hi_vox = np.minimum(vol_shape, np.ceil(hi_um * 1000.0 / res_nm).astype(np.int64))
    print(f"downloading mip={args.mip} ({res_nm.tolist()} nm) box={lo_vox.tolist()}..{hi_vox.tolist()} shape={(hi_vox - lo_vox).tolist()}...")

    cut = np.asarray(cv[lo_vox[0]:hi_vox[0], lo_vox[1]:hi_vox[1], lo_vox[2]:hi_vox[2]])[..., 0]
    mask = (cut == args.root_id)
    total_fg = int(mask.sum())
    if total_fg == 0:
        raise RuntimeError(f"root id {args.root_id} has 0 voxels in this cutout.")

    soma_local = np.clip(np.round(soma_um * 1000.0 / res_nm).astype(np.int64) - lo_vox, 0, np.array(mask.shape) - 1)
    labeled, ncomp = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    soma_lbl = int(labeled[soma_local[0], soma_local[1], soma_local[2]])
    if soma_lbl == 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        soma_lbl = int(np.argmax(counts))

    soma_mask = (labeled == soma_lbl)
    soma_fg = int(soma_mask.sum())
    print(f"foreground voxels: total={total_fg}, connected_components={ncomp}, soma_component={soma_fg} ({100.0 * soma_fg / total_fg:.1f}%)")

    coords = np.argwhere(soma_mask)
    cmin = np.maximum(0, coords.min(axis=0) - 2)
    cmax = np.minimum(np.array(soma_mask.shape), coords.max(axis=0) + 3)
    cropped = soma_mask[cmin[0]:cmax[0], cmin[1]:cmax[1], cmin[2]:cmax[2]].astype(np.uint8)

    origin_vox = lo_vox + cmin
    origin_nm = origin_vox.astype(np.float64) * res_nm
    root_cropped = soma_local - cmin
    if not (0 <= root_cropped[0] < cropped.shape[0] and
            0 <= root_cropped[1] < cropped.shape[1] and
            0 <= root_cropped[2] < cropped.shape[2] and
            cropped[root_cropped[0], root_cropped[1], root_cropped[2]] == 1):
        root_cropped = np.array([-1, -1, -1], dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"NMV1")
        f.write(struct.pack("<3i", int(cropped.shape[0]), int(cropped.shape[1]), int(cropped.shape[2])))
        f.write(struct.pack("<3f", float(res_nm[0]), float(res_nm[1]), float(res_nm[2])))
        f.write(struct.pack("<3f", float(origin_nm[0]), float(origin_nm[1]), float(origin_nm[2])))
        f.write(struct.pack("<3i", int(root_cropped[0]), int(root_cropped[1]), int(root_cropped[2])))
        f.write(cropped.ravel(order="F").tobytes())

    size_mb = os.path.getsize(args.out) / (1024.0 * 1024.0)
    print(f"saved {args.out} ({size_mb:.1f} mb): cropped_shape={cropped.shape}, origin_um={(origin_nm * 1e-3).round(2).tolist()}, root_vox={root_cropped.tolist()}")


if __name__ == "__main__":
    main()
