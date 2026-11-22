import argparse
import csv
import os
import struct
import numpy as np
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


def load_soma_nm(csv_path: str, root_id: int):
    if not csv_path or not os.path.exists(csv_path):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32), "unknown"
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["pt_root_id"]) == root_id:
                soma = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float32) * 1000.0
                return soma, row["cell_type"].lower()
    return np.array([np.nan, np.nan, np.nan], dtype=np.float32), "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-csv", default="")
    parser.add_argument("--root-id", type=int, required=True)
    parser.add_argument("--datastack", default="minnie65_public")
    parser.add_argument("--lod", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    soma_nm, cell_type = load_soma_nm(args.nodes_csv, args.root_id)
    print(f"target: {args.root_id} ({cell_type.lower()}) lod={args.lod} soma_um={(soma_nm * 1e-3).round(2).tolist()}")

    seg_url = resolve_seg_source(args.datastack)
    cv = CloudVolume(seg_url, progress=False)
    mesh = cv.mesh.get(args.root_id, lod=args.lod)[args.root_id]

    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"NMM1")
        f.write(struct.pack("<2I", int(len(verts)), int(len(faces))))
        f.write(struct.pack("<3f", float(soma_nm[0]), float(soma_nm[1]), float(soma_nm[2])))
        f.write(verts.tobytes())
        f.write(faces.tobytes())

    mn = (verts.min(axis=0) * 1e-3).round(1).tolist()
    mx = (verts.max(axis=0) * 1e-3).round(1).tolist()
    size_mb = os.path.getsize(args.out) / (1024.0 * 1024.0)
    print(f"saved {args.out} ({size_mb:.2f} mb): verts={len(verts)}, faces={len(faces)}, bbox_um={mn}..{mx}")


if __name__ == "__main__":
    main()
