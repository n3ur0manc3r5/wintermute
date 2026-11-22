# wintermute

## setup

### 1. build (c++20 / cmake / openmp)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 2. python environment

```bash
uv venv .venv
uv pip install --python .venv/bin/python caveclient cloud-volume numpy scipy
```

### 3. fetch a mesh & run mesh-teasar (recommended)

```bash
.venv/bin/python scripts/fetch_microns_mesh.py \
  --nodes-csv /path/to/selected_20_nodes_ei.csv \
  --root-id 864691135866586245 \
  --lod 2 \
  --out data/23P_864691135866586245.nmm

./build/neuromancers \
  --mesh data/23P_864691135866586245.nmm \
  --swc data/23P_864691135866586245_mesh.swc \
  --inv-scale 2.0 --inv-const 1800 --rdp 500 --min-branch 2500
```

### 4. reconstruct smooth 3d voxel volume & surface mesh from swc

```bash
./build/neuromancers \
  --voxelize data/23P_864691135866586245_mesh.swc \
  --vox-res 450 \
  --out-vol data/23P_864691135866586245_voxel.nmv \
  --out-obj data/23P_864691135866586245_voxel.obj \
  --compare-mesh data/23P_864691135866586245.nmm \
  --html data/23P_3d_viewer.html
```
