#include "viewer.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>

namespace teasar {

bool export_html_viewer(const std::string& filepath,
                        const VolumeGrid& grid,
                        const std::vector<SWCNode>& raw_swc,
                        const std::vector<SWCNode>& rdp_swc,
                        float invalidation_scale,
                        float invalidation_const_nm,
                        int voxel_stride_xy,
                        float origin_x_nm,
                        float origin_y_nm,
                        float origin_z_nm,
                        int max_surface_voxels) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        return false;
    }

    const int stride = std::max(1, voxel_stride_xy);
    std::vector<std::array<float, 3>> surface_pts;
    const int dirs[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for (int z = 0; z < grid.dim_z; ++z) {
        for (int y = 0; y < grid.dim_y; y += stride) {
            for (int x = 0; x < grid.dim_x; x += stride) {
                const int idx = grid.to_index(x, y, z);
                if (grid.mask[static_cast<std::size_t>(idx)] == 0) {
                    continue;
                }
                bool boundary = false;
                for (const auto& d : dirs) {
                    const int nx = x + d[0];
                    const int ny = y + d[1];
                    const int nz = z + d[2];
                    if (!grid.is_valid(nx, ny, nz) ||
                        grid.mask[static_cast<std::size_t>(grid.to_index(nx, ny, nz))] == 0) {
                        boundary = true;
                        break;
                    }
                }
                if (!boundary) {
                    continue;
                }
                surface_pts.push_back({
                    (origin_x_nm + static_cast<float>(x) * grid.scale.dx) * 1e-3f,
                    (origin_y_nm + static_cast<float>(y) * grid.scale.dy) * 1e-3f,
                    (origin_z_nm + static_cast<float>(z) * grid.scale.dz) * 1e-3f
                });
            }
        }
    }

    const std::size_t max_pts = static_cast<std::size_t>(std::max(100, max_surface_voxels));
    const std::size_t step = (surface_pts.size() > max_pts)
        ? (surface_pts.size() + max_pts - 1) / max_pts
        : 1;

    out << R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neuromancers TEASAR 3D Morphology</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #020617; color: #e2e8f0; font-family: system-ui, -apple-system, sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
.bar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 16px; background: #0f172a; border-bottom: 1px solid #1e293b; }
.title { font-size: 15px; font-weight: 600; }
.sub { font-size: 12px; color: #94a3b8; }
.group { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 12px; }
button { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 5px 10px; font-size: 12px; cursor: pointer; }
button:hover { background: #334155; }
button.active { background: #0284c7; border-color: #38bdf8; color: #ffffff; }
label { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; user-select: none; }
.stage { position: relative; flex: 1; width: 100%; }
canvas { display: block; width: 100%; height: 100%; cursor: grab; }
canvas:active { cursor: grabbing; }
.overlay { position: absolute; left: 12px; background: rgba(15,23,42,0.88); border: 1px solid #334155; border-radius: 8px; padding: 6px 10px; font-size: 12px; pointer-events: none; }
.legend { top: 12px; display: flex; gap: 12px; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
.inspector { bottom: 12px; font-family: ui-monospace, monospace; display: none; }
</style>
</head>
<body>
<div class="bar">
  <div>
    <div class="title">Neuromancers TEASAR 3D Morphology <span id="badge" class="sub"></span></div>
    <div class="sub">Drag to orbit &middot; Scroll to zoom &middot; Hover any SWC node to inspect</div>
  </div>
  <div class="group">
    <button data-v="iso">3D</button>
    <button data-v="xy">XY</button>
    <button data-v="xz">XZ</button>
    <button data-v="yz">YZ</button>
    <button id="spin" class="active">Spin</button>
  </div>
</div>
<div class="stage">
  <canvas id="c"></canvas>
  <div class="overlay legend">
    <span><span class="dot" style="background:#f59e0b"></span>Soma</span>
    <span><span class="dot" style="background:#ec4899"></span>Bifurcation</span>
    <span><span class="dot" style="background:#38bdf8"></span>Node</span>
    <span><span class="dot" style="background:#10b981"></span>Tip</span>
  </div>
  <div id="info" class="overlay inspector"></div>
</div>
<div class="bar" style="border-top:1px solid #1e293b;border-bottom:none;">
  <div class="group">
    <label><input id="chkV" type="checkbox" checked> Voxel Membrane</label>
    <label><input id="chkT" type="checkbox" checked> SWC Tubes (DFB r)</label>
    <label><input id="chkS" type="checkbox"> Rolling Spheres</label>
  </div>
  <div class="group">
    <button id="mRdp" class="active">RDP SWC</button>
    <button id="mRaw">Raw TEASAR</button>
  </div>
</div>
<script>
const DATA = {
)HTML";

    out << std::fixed << std::setprecision(3);
    out << "  scale: [" << grid.scale.dx << "," << grid.scale.dy << "," << grid.scale.dz << "],\n";
    out << "  inv: [" << invalidation_scale << "," << (invalidation_const_nm * 1e-3f) << "],\n";
    out << "  voxels: [";
    bool first = true;
    for (std::size_t i = 0; i < surface_pts.size(); i += step) {
        if (!first) {
            out << ',';
        }
        first = false;
        out << '[' << surface_pts[i][0] << ',' << surface_pts[i][1] << ',' << surface_pts[i][2] << ']';
    }
    out << "],\n";

    auto write_swc = [&out](const char* key, const std::vector<SWCNode>& nodes) {
        out << "  " << key << ": [";
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (i > 0) {
                out << ',';
            }
            const SWCNode& n = nodes[i];
            out << '[' << n.id << ',' << n.type << ',' << n.x << ',' << n.y << ','
                << n.z << ',' << n.radius << ',' << n.parent_id << ']';
        }
        out << ']';
    };

    write_swc("raw", raw_swc);
    out << ",\n";
    write_swc("rdp", rdp_swc);
    out << "\n};\n";

    out << R"HTML(
function parseSwc(arr) {
  const nodes = arr.map(([id,type,x,y,z,r,parent]) => ({id,type,x,y,z,r,parent,children:0}));
  const byId = new Map(nodes.map(n => [n.id, n]));
  for (const n of nodes) if (n.parent !== -1 && byId.has(n.parent)) byId.get(n.parent).children++;
  for (const n of nodes) {
    if (n.parent === -1) n.role = 'Soma';
    else if (n.children >= 2) n.role = 'Bifurcation';
    else if (n.children === 0) n.role = 'Tip';
    else n.role = 'Branch';
  }
  return { nodes, byId };
}
const ds = { rdp: parseSwc(DATA.rdp), raw: parseSwc(DATA.raw) };
const pts = DATA.rdp.length ? DATA.rdp.map(n => [n[2],n[3],n[4]]) : DATA.voxels;
let mn = [Infinity,Infinity,Infinity], mx = [-Infinity,-Infinity,-Infinity];
for (const p of pts) {
  for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
}
const center = [(mn[0]+mx[0])*0.5, (mn[1]+mx[1])*0.5, (mn[2]+mx[2])*0.5];
const span = Math.max(0.05, mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]);
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const info = document.getElementById('info');
const badge = document.getElementById('badge');
const st = { yaw: 0.45, pitch: 0.25, zoom: 1.0, spin: true, mode: 'rdp', showV: true, showT: true, showS: false, drag: false, lx: 0, ly: 0, hover: null, pNodes: [] };
function roleColor(r) { return r==='Soma' ? '#f59e0b' : r==='Bifurcation' ? '#ec4899' : r==='Tip' ? '#10b981' : '#38bdf8'; }
function project(x,y,z,w,h) {
  const px = (x - center[0]) / span;
  const py = -(y - center[1]) / span;
  const pz = (z - center[2]) / span;
  const cy = Math.cos(st.yaw), sy = Math.sin(st.yaw);
  const x1 = px*cy + pz*sy, z1 = -px*sy + pz*cy;
  const cp = Math.cos(st.pitch), sp = Math.sin(st.pitch);
  const y2 = py*cp - z1*sp, z2 = py*sp + z1*cp;
  const fov = Math.min(w,h) * 1.45 * st.zoom;
  const sc = fov / Math.max(0.25, 2.2 - z2);
  return { sx: w*0.5 + x1*sc, sy: h*0.52 - y2*sc, depth: z2, sc: sc / span };
}
function drawSeg(p1,p2,r1,r2) {
  const dx = p2.sx - p1.sx, dy = p2.sy - p1.sy, len = Math.hypot(dx,dy);
  if (len < 0.4) return;
  const nx = -dy/len, ny = dx/len, mr = Math.max(r1,r2);
  const g = ctx.createLinearGradient((p1.sx+p2.sx)*0.5-nx*mr,(p1.sy+p2.sy)*0.5-ny*mr,(p1.sx+p2.sx)*0.5+nx*mr,(p1.sy+p2.sy)*0.5+ny*mr);
  g.addColorStop(0,'rgba(14,165,233,0.85)'); g.addColorStop(0.35,'rgba(125,211,252,0.95)'); g.addColorStop(1,'rgba(3,105,161,0.85)');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.moveTo(p1.sx+nx*r1,p1.sy+ny*r1); ctx.lineTo(p2.sx+nx*r2,p2.sy+ny*r2);
  ctx.lineTo(p2.sx-nx*r2,p2.sy-ny*r2); ctx.lineTo(p1.sx-nx*r1,p1.sy-ny*r1);
  ctx.closePath(); ctx.fill();
}
function drawBall(x,y,r,col,hi) {
  const rad = Math.max(2.0, r);
  const g = ctx.createRadialGradient(x-rad*0.3,y-rad*0.3,rad*0.08,x,y,rad);
  g.addColorStop(0,'#ffffff'); g.addColorStop(0.35,col); g.addColorStop(1,'#0f172a');
  ctx.beginPath(); ctx.arc(x,y,rad,0,Math.PI*2); ctx.fillStyle = g; ctx.fill();
  if (hi) { ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke(); }
}
function render() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(100, rect.width), h = Math.max(100, rect.height);
  if (canvas.width !== Math.round(w*dpr) || canvas.height !== Math.round(h*dpr)) {
    canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr);
  }
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const bg = ctx.createRadialGradient(w*0.5,h*0.45,20,w*0.5,h*0.5,Math.max(w,h)*0.75);
  bg.addColorStop(0,'#0f172a'); bg.addColorStop(1,'#020617');
  ctx.fillStyle = bg; ctx.fillRect(0,0,w,h);
  const pad = span * 0.05;
  const b = [mn[0]-pad,mx[0]+pad,mn[1]-pad,mx[1]+pad,mn[2]-pad,mx[2]+pad];
  const cs = [[b[0],b[2],b[4]],[b[1],b[2],b[4]],[b[1],b[3],b[4]],[b[0],b[3],b[4]],[b[0],b[2],b[5]],[b[1],b[2],b[5]],[b[1],b[3],b[5]],[b[0],b[3],b[5]]].map(p => project(p[0],p[1],p[2],w,h));
  ctx.strokeStyle = 'rgba(148,163,184,0.16)'; ctx.lineWidth = 1; ctx.beginPath();
  for (const [i,j] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]) { ctx.moveTo(cs[i].sx,cs[i].sy); ctx.lineTo(cs[j].sx,cs[j].sy); }
  ctx.stroke();
  ctx.font = '12px ui-monospace, monospace';
  for (const [to,col,lbl] of [[cs[1],'#f87171','X'],[cs[3],'#4ade80','Y (depth)'],[cs[4],'#60a5fa','Z']]) {
    ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(cs[0].sx,cs[0].sy); ctx.lineTo(to.sx,to.sy); ctx.stroke();
    ctx.fillStyle = col; ctx.fillText(lbl, to.sx+4, to.sy+4);
  }
  const cur = ds[st.mode], list = [], pMap = new Map();
  st.pNodes = [];
  for (const n of cur.nodes) {
    const p = project(n.x,n.y,n.z,w,h);
    pMap.set(n.id, p);
    const isKey = n.role !== 'Branch';
    const rPx = n.role==='Soma' ? Math.max(6, n.r*p.sc) : isKey ? (n.role==='Bifurcation' ? 4.2 : 3.6) : (st.mode==='raw' ? 1.5 : 2.6);
    st.pNodes.push({ n, sx: p.sx, sy: p.sy, hit: Math.max(8, rPx+3) });
    if (isKey || st.mode === 'rdp' || st.hover === n.id) {
      list.push({ k: 'n', d: p.depth+0.002, n, p, r: rPx });
    }
    if (st.showS && (st.mode === 'rdp' || isKey)) list.push({ k: 's', d: p.depth-0.003, p, r: (DATA.inv[0]*n.r + DATA.inv[1])*p.sc });
  }
  for (const n of cur.nodes) {
    if (n.parent === -1 || !pMap.has(n.parent)) continue;
    const p1 = pMap.get(n.parent), p2 = pMap.get(n.id), par = cur.byId.get(n.parent);
    list.push({ k: 'e', d: (p1.depth+p2.depth)*0.5, p1, p2, r1: Math.max(1.2, par.r*p1.sc*0.85), r2: Math.max(1.2, n.r*p2.sc*0.85) });
  }
  if (st.showV) {
    for (const v of DATA.voxels) {
      const p = project(v[0],v[1],v[2],w,h);
      list.push({ k: 'v', d: p.depth, p });
    }
  }
  list.sort((a,b) => a.d - b.d);
  for (const it of list) {
    if (it.k === 'v') {
      const a = 0.08 + 0.16 * Math.max(0, Math.min(1, (it.d + 0.8) / 1.6));
      ctx.fillStyle = `rgba(56,189,248,${a.toFixed(3)})`;
      ctx.fillRect(it.p.sx - 1.1, it.p.sy - 1.1, 2.2, 2.2);
    } else if (it.k === 's') {
      ctx.beginPath(); ctx.arc(it.p.sx, it.p.sy, it.r, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(251,191,36,0.05)'; ctx.strokeStyle = 'rgba(251,191,36,0.22)'; ctx.lineWidth = 1;
      ctx.fill(); ctx.stroke();
    } else if (it.k === 'e') {
      if (st.showT) drawSeg(it.p1, it.p2, it.r1, it.r2);
      else { ctx.strokeStyle = '#7dd3fc'; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(it.p1.sx,it.p1.sy); ctx.lineTo(it.p2.sx,it.p2.sy); ctx.stroke(); }
    } else {
      const hi = st.hover === it.n.id;
      drawBall(it.p.sx, it.p.sy, hi ? it.r+2.5 : it.r, roleColor(it.n.role), hi);
    }
  }
}
function updateInfo() {
  if (st.hover == null) { info.style.display = 'none'; return; }
  const n = ds[st.mode].byId.get(st.hover);
  if (!n) { info.style.display = 'none'; return; }
  info.style.display = 'block';
  info.textContent = `SWC #${n.id} [${n.role}]  xyz=(${n.x.toFixed(2)}, ${n.y.toFixed(2)}, ${n.z.toFixed(2)}) um  r=${n.r.toFixed(3)} um  parent=${n.parent}`;
}
canvas.addEventListener('pointerdown', e => { st.drag = true; st.spin = false; document.getElementById('spin').classList.remove('active'); st.lx = e.clientX; st.ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e => {
  if (st.drag) { st.yaw += (e.clientX - st.lx)*0.01; st.pitch = Math.max(-1.45, Math.min(1.45, st.pitch + (e.clientY - st.ly)*0.01)); st.lx = e.clientX; st.ly = e.clientY; return; }
  const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  let best = null, bd = Infinity;
  for (const pn of st.pNodes) { const d = Math.hypot(mx - pn.sx, my - pn.sy); if (d <= pn.hit && d < bd) { bd = d; best = pn.n.id; } }
  if (best !== st.hover) { st.hover = best; updateInfo(); }
});
canvas.addEventListener('pointerup', () => st.drag = false);
canvas.addEventListener('pointerleave', () => { st.drag = false; st.hover = null; updateInfo(); });
canvas.addEventListener('wheel', e => { e.preventDefault(); st.zoom = Math.max(0.4, Math.min(5.0, st.zoom * (e.deltaY < 0 ? 1.12 : 0.89))); }, { passive: false });
document.getElementById('spin').onclick = e => { st.spin = !st.spin; e.target.classList.toggle('active', st.spin); };
document.querySelectorAll('[data-v]').forEach(b => b.onclick = () => {
  st.spin = false; document.getElementById('spin').classList.remove('active'); st.zoom = 1.0;
  if (b.dataset.v==='iso') { st.yaw=0.45; st.pitch=0.25; }
  if (b.dataset.v==='xy')  { st.yaw=0; st.pitch=0; }
  if (b.dataset.v==='xz')  { st.yaw=0; st.pitch=Math.PI/2-0.001; }
  if (b.dataset.v==='yz')  { st.yaw=Math.PI/2; st.pitch=0; }
});
document.getElementById('chkV').onchange = e => st.showV = e.target.checked;
document.getElementById('chkT').onchange = e => st.showT = e.target.checked;
document.getElementById('chkS').onchange = e => st.showS = e.target.checked;
function setMode(m) {
  st.mode = m; st.hover = null; updateInfo();
  document.getElementById('mRdp').classList.toggle('active', m==='rdp');
  document.getElementById('mRaw').classList.toggle('active', m==='raw');
  badge.textContent = '(' + ds[m].nodes.length + ' nodes)';
}
document.getElementById('mRdp').onclick = () => setMode('rdp');
document.getElementById('mRaw').onclick = () => setMode('raw');
setMode('rdp');
(function loop(){ if (st.spin && !st.drag) st.yaw += 0.0045; render(); requestAnimationFrame(loop); })();
</script>
</body>
</html>
)HTML";

    return out.good();
}

bool export_mesh_html_viewer(const std::string& filepath,
                             const TriangleMesh& mesh,
                             const std::vector<SWCNode>& raw_swc,
                             const std::vector<SWCNode>& rdp_swc,
                             float invalidation_scale,
                             float invalidation_const_nm,
                             int max_surface_points) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        return false;
    }

    const std::size_t nv = mesh.vertices.size();
    const std::size_t max_pts = static_cast<std::size_t>(std::max(200, max_surface_points));
    const std::size_t step = (nv > max_pts) ? (nv + max_pts - 1) / max_pts : 1;

    out << R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>wintermute 3d mesh morphology</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #020617; color: #e2e8f0; font-family: system-ui, -apple-system, sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
.bar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 16px; background: #0f172a; border-bottom: 1px solid #1e293b; }
.title { font-size: 15px; font-weight: 600; }
.sub { font-size: 12px; color: #94a3b8; }
.group { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 12px; }
button { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 5px 10px; font-size: 12px; cursor: pointer; }
button:hover { background: #334155; }
button.active { background: #0284c7; border-color: #38bdf8; color: #ffffff; }
label { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; user-select: none; }
.stage { position: relative; flex: 1; width: 100%; }
canvas { display: block; width: 100%; height: 100%; cursor: grab; }
canvas:active { cursor: grabbing; }
.overlay { position: absolute; left: 12px; background: rgba(15,23,42,0.88); border: 1px solid #334155; border-radius: 8px; padding: 6px 10px; font-size: 12px; pointer-events: none; }
.legend { top: 12px; display: flex; gap: 12px; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
.inspector { bottom: 12px; font-family: ui-monospace, monospace; display: none; }
</style>
</head>
<body>
<div class="bar">
  <div>
    <div class="title">wintermute mesh-teasar 3d morphology <span id="badge" class="sub"></span></div>
    <div class="sub">drag to orbit &middot; scroll to zoom &middot; hover any swc node</div>
  </div>
  <div class="group">
    <button data-v="iso">3d</button>
    <button data-v="xy">xy</button>
    <button data-v="xz">xz</button>
    <button data-v="yz">yz</button>
    <button id="spin" class="active">spin</button>
  </div>
</div>
<div class="stage">
  <canvas id="c"></canvas>
  <div class="overlay legend">
    <span><span class="dot" style="background:#f59e0b"></span>soma</span>
    <span><span class="dot" style="background:#ec4899"></span>bifurcation</span>
    <span><span class="dot" style="background:#38bdf8"></span>branch</span>
    <span><span class="dot" style="background:#10b981"></span>tip</span>
  </div>
  <div id="info" class="overlay inspector"></div>
</div>
<div class="bar" style="border-top:1px solid #1e293b;border-bottom:none;">
  <div class="group">
    <label><input id="chkV" type="checkbox" checked> mesh membrane</label>
    <label><input id="chkT" type="checkbox" checked> swc tubes (sdf r)</label>
    <label><input id="chkS" type="checkbox"> rolling spheres</label>
  </div>
  <div class="group">
    <button id="mRdp" class="active">rdp swc</button>
    <button id="mRaw">raw mesh-teasar</button>
  </div>
</div>
<script>
const DATA = {
)HTML";

    out << std::fixed << std::setprecision(2);
    out << "  inv: [" << invalidation_scale << "," << (invalidation_const_nm * 1e-3f) << "],\n";
    out << "  voxels: [";
    bool first = true;
    for (std::size_t i = 0; i < nv; i += step) {
        if (!first) out << ',';
        first = false;
        const Vec3f& v = mesh.vertices[i];
        out << '[' << (v.x * 1e-3f) << ',' << (v.y * 1e-3f) << ',' << (v.z * 1e-3f) << ']';
    }
    out << "],\n";

    auto write_swc = [&out](const char* key, const std::vector<SWCNode>& nodes) {
        out << "  " << key << ": [";
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (i > 0) out << ',';
            const SWCNode& n = nodes[i];
            out << '[' << n.id << ',' << n.type << ',' << n.x << ',' << n.y << ','
                << n.z << ',' << n.radius << ',' << n.parent_id << ']';
        }
        out << ']';
    };

    write_swc("raw", raw_swc);
    out << ",\n";
    write_swc("rdp", rdp_swc);
    out << "\n};\n";

    out << R"HTML(
function parseSwc(arr) {
  const nodes = arr.map(([id,type,x,y,z,r,parent]) => ({id,type,x,y,z,r,parent,children:0}));
  const byId = new Map(nodes.map(n => [n.id, n]));
  for (const n of nodes) if (n.parent !== -1 && byId.has(n.parent)) byId.get(n.parent).children++;
  for (const n of nodes) {
    if (n.parent === -1) n.role = 'soma';
    else if (n.children >= 2) n.role = 'bifurcation';
    else if (n.children === 0) n.role = 'tip';
    else n.role = 'branch';
  }
  return { nodes, byId };
}
const ds = { rdp: parseSwc(DATA.rdp), raw: parseSwc(DATA.raw) };
const pts = DATA.rdp.length ? DATA.rdp.map(n => [n[2],n[3],n[4]]) : DATA.voxels;
let mn = [Infinity,Infinity,Infinity], mx = [-Infinity,-Infinity,-Infinity];
for (const p of pts) {
  for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
}
const center = [(mn[0]+mx[0])*0.5, (mn[1]+mx[1])*0.5, (mn[2]+mx[2])*0.5];
const span = Math.max(1, mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]);
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const info = document.getElementById('info');
const badge = document.getElementById('badge');
const st = { yaw: 0.45, pitch: 0.16, zoom: 1.05, spin: true, mode: 'rdp', showV: true, showT: true, showS: false, drag: false, lx: 0, ly: 0, hover: null, pNodes: [] };
function roleColor(r) { return r==='soma' ? '#f59e0b' : r==='bifurcation' ? '#ec4899' : r==='tip' ? '#10b981' : '#38bdf8'; }
function project(x,y,z,w,h) {
  const px = (x - center[0]) / span;
  const py = -(y - center[1]) / span;
  const pz = (z - center[2]) / span;
  const cy = Math.cos(st.yaw), sy = Math.sin(st.yaw);
  const x1 = px*cy + pz*sy, z1 = -px*sy + pz*cy;
  const cp = Math.cos(st.pitch), sp = Math.sin(st.pitch);
  const y2 = py*cp - z1*sp, z2 = py*sp + z1*cp;
  const fov = Math.min(w,h) * 1.45 * st.zoom;
  const sc = fov / Math.max(0.25, 2.25 - z2);
  return { sx: w*0.5 + x1*sc, sy: h*0.52 - y2*sc, depth: z2, sc: sc / span };
}
function drawBall(x,y,r,col,hi) {
  const rad = Math.max(2.0, r);
  const g = ctx.createRadialGradient(x-rad*0.3,y-rad*0.3,rad*0.08,x,y,rad);
  g.addColorStop(0,'#ffffff'); g.addColorStop(0.35,col); g.addColorStop(1,'#0f172a');
  ctx.beginPath(); ctx.arc(x,y,rad,0,Math.PI*2); ctx.fillStyle = g; ctx.fill();
  if (hi) { ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke(); }
}
function render() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(100, rect.width), h = Math.max(100, rect.height);
  if (canvas.width !== Math.round(w*dpr) || canvas.height !== Math.round(h*dpr)) {
    canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr);
  }
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const bg = ctx.createRadialGradient(w*0.5,h*0.45,20,w*0.5,h*0.5,Math.max(w,h)*0.75);
  bg.addColorStop(0,'#0f172a'); bg.addColorStop(1,'#020617');
  ctx.fillStyle = bg; ctx.fillRect(0,0,w,h);
  const cur = ds[st.mode], list = [], pMap = new Map();
  st.pNodes = [];
  for (const n of cur.nodes) {
    const p = project(n.x,n.y,n.z,w,h);
    pMap.set(n.id, p);
    const isKey = n.role !== 'branch';
    const rPx = n.role==='soma' ? Math.max(7, n.r*p.sc) : isKey ? (n.role==='bifurcation' ? 3.5 : 3.2) : 1.8;
    st.pNodes.push({ n, sx: p.sx, sy: p.sy, hit: Math.max(7, rPx+3) });
    if (isKey || st.hover === n.id) list.push({ k: 'n', d: p.depth+0.002, n, p, r: rPx });
    if (st.showS && isKey) list.push({ k: 's', d: p.depth-0.003, p, r: (DATA.inv[0]*n.r + DATA.inv[1])*p.sc });
  }
  for (const n of cur.nodes) {
    if (n.parent === -1 || !pMap.has(n.parent)) continue;
    const p1 = pMap.get(n.parent), p2 = pMap.get(n.id), par = cur.byId.get(n.parent);
    const avgR = (par.r + n.r) * 0.5;
    list.push({ k: 'e', d: (p1.depth+p2.depth)*0.5, p1, p2, lw: st.showT ? Math.max(1.0, Math.min(8, avgR*p2.sc*1.7)) : 1.2 });
  }
  if (st.showV) {
    for (const v of DATA.voxels) {
      const p = project(v[0],v[1],v[2],w,h);
      list.push({ k: 'v', d: p.depth, p });
    }
  }
  list.sort((a,b) => a.d - b.d);
  ctx.lineCap = 'round';
  for (const it of list) {
    if (it.k === 'v') {
      const a = 0.08 + 0.14 * Math.max(0, Math.min(1, (it.d + 0.7) / 1.4));
      ctx.fillStyle = `rgba(148,163,184,${a.toFixed(3)})`;
      ctx.fillRect(it.p.sx - 1.0, it.p.sy - 1.0, 2.0, 2.0);
    } else if (it.k === 's') {
      ctx.beginPath(); ctx.arc(it.p.sx, it.p.sy, it.r, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(251,191,36,0.04)'; ctx.strokeStyle = 'rgba(251,191,36,0.20)'; ctx.lineWidth = 1;
      ctx.fill(); ctx.stroke();
    } else if (it.k === 'e') {
      const a = 0.45 + 0.55 * Math.max(0, Math.min(1, (it.d + 0.65) / 1.3));
      ctx.strokeStyle = `rgba(56,189,248,${a.toFixed(3)})`;
      ctx.lineWidth = it.lw;
      ctx.beginPath(); ctx.moveTo(it.p1.sx,it.p1.sy); ctx.lineTo(it.p2.sx,it.p2.sy); ctx.stroke();
    } else {
      const hi = st.hover === it.n.id;
      drawBall(it.p.sx, it.p.sy, hi ? it.r+2.5 : it.r, roleColor(it.n.role), hi);
    }
  }
}
function updateInfo() {
  if (st.hover == null) { info.style.display = 'none'; return; }
  const n = ds[st.mode].byId.get(st.hover);
  if (!n) { info.style.display = 'none'; return; }
  info.style.display = 'block';
  info.textContent = `swc #${n.id} [${n.role}]  xyz=(${n.x.toFixed(2)}, ${n.y.toFixed(2)}, ${n.z.toFixed(2)}) um  r=${n.r.toFixed(3)} um  parent=${n.parent}`;
}
canvas.addEventListener('pointerdown', e => { st.drag = true; st.spin = false; document.getElementById('spin').classList.remove('active'); st.lx = e.clientX; st.ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e => {
  if (st.drag) { st.yaw += (e.clientX - st.lx)*0.01; st.pitch = Math.max(-1.45, Math.min(1.45, st.pitch + (e.clientY - st.ly)*0.01)); st.lx = e.clientX; st.ly = e.clientY; return; }
  const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  let best = null, bd = Infinity;
  for (const pn of st.pNodes) { const d = Math.hypot(mx - pn.sx, my - pn.sy); if (d <= pn.hit && d < bd) { bd = d; best = pn.n.id; } }
  if (best !== st.hover) { st.hover = best; updateInfo(); }
});
canvas.addEventListener('pointerup', () => st.drag = false);
canvas.addEventListener('pointerleave', () => { st.drag = false; st.hover = null; updateInfo(); });
canvas.addEventListener('wheel', e => { e.preventDefault(); st.zoom = Math.max(0.35, Math.min(6.0, st.zoom * (e.deltaY < 0 ? 1.12 : 0.89))); }, { passive: false });
document.getElementById('spin').onclick = e => { st.spin = !st.spin; e.target.classList.toggle('active', st.spin); };
document.querySelectorAll('[data-v]').forEach(b => b.onclick = () => {
  st.spin = false; document.getElementById('spin').classList.remove('active'); st.zoom = 1.05;
  if (b.dataset.v==='iso') { st.yaw=0.45; st.pitch=0.16; }
  if (b.dataset.v==='xy')  { st.yaw=0; st.pitch=0; }
  if (b.dataset.v==='xz')  { st.yaw=0; st.pitch=Math.PI/2-0.001; }
  if (b.dataset.v==='yz')  { st.yaw=Math.PI/2; st.pitch=0; }
});
document.getElementById('chkV').onchange = e => st.showV = e.target.checked;
document.getElementById('chkT').onchange = e => st.showT = e.target.checked;
document.getElementById('chkS').onchange = e => st.showS = e.target.checked;
function setMode(m) {
  st.mode = m; st.hover = null; updateInfo();
  document.getElementById('mRdp').classList.toggle('active', m==='rdp');
  document.getElementById('mRaw').classList.toggle('active', m==='raw');
  badge.textContent = '(' + ds[m].nodes.length + ' nodes)';
}
document.getElementById('mRdp').onclick = () => setMode('rdp');
document.getElementById('mRaw').onclick = () => setMode('raw');
setMode('rdp');
(function loop(){ if (st.spin && !st.drag) st.yaw += 0.004; render(); requestAnimationFrame(loop); })();
</script>
</body>
</html>
)HTML";

    return out.good();
}

}
