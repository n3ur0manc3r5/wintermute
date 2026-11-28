import argparse
import base64
import csv
import json
import os
import struct
import subprocess
import time
import numpy as np
import caveclient
from cloudvolume import CloudVolume


PUBLIC_M1300 = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300"

EXCITATORY_TYPES = {"23P", "4P", "5P-IT", "5P-ET", "5P-NP", "6P-IT", "6P-CT"}
INHIBITORY_TYPES = {"BC", "MC", "BPC", "NGC"}

# colour for each inhibitory cell type; BC and MC keep their old colours
CELL_TYPE_COLORS = {
    "BC": [1.00, 0.76, 0.18],   # yellow
    "MC": [0.98, 0.30, 0.58],   # pink
    "BPC": [0.62, 0.40, 1.00],  # purple
    "NGC": [1.00, 0.45, 0.10],  # orange
}

# any cell type not listed above is drawn grey, so it stands out
UNKNOWN_COLOR = [0.55, 0.55, 0.55]


# colour of each synapse dot, by direction (sending cell to receiving cell)
SYNAPSE_COLORS = {
    "E→E": [0.35, 0.98, 0.45],  # green
    "E→I": [1.00, 0.90, 0.25],  # yellow
    "I→E": [1.00, 0.28, 0.32],  # red
    "I→I": [0.78, 0.42, 1.00],  # purple
}

def is_inhibitory(cell_type):
    return cell_type in INHIBITORY_TYPES


def legend_html():
    def swatch(rgb, label):
        r, g, b = (round(c * 255) for c in rgb)
        return (f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:8px">'
                f'<span class="sw" style="background:rgb({r},{g},{b})"></span>{label}</span>')

    cells = [swatch([0.16, 0.78, 0.96], "excitatory")]
    cells += [swatch(rgb, name) for name, rgb in CELL_TYPE_COLORS.items()]
    synapses = [swatch(rgb, name) for name, rgb in SYNAPSE_COLORS.items()]
    return ('<div style="color:#8aa4bd;font-size:10px;">cells</div>'
            '<div class="row">' + "".join(cells) + '</div>'
            '<div style="color:#8aa4bd;font-size:10px;">synapses (sender→receiver)</div>'
            '<div class="row">' + "".join(synapses) + '</div>')

def decimate_mesh_grid(verts: np.ndarray, faces: np.ndarray, cell_nm: float = 240.0):
    q = np.floor(verts / cell_nm).astype(np.int64)
    q_min = q.min(axis=0)
    q_rel = q - q_min
    dims = q_rel.max(axis=0) + 1
    keys = (q_rel[:, 0] * dims[1] + q_rel[:, 1]) * dims[2] + q_rel[:, 2]
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_new = len(uniq)
    new_v = np.zeros((n_new, 3), dtype=np.float64)
    np.add.at(new_v, inv, verts.astype(np.float64))
    new_v /= counts[:, None]
    new_f = inv[faces].astype(np.uint32)
    valid = (new_f[:, 0] != new_f[:, 1]) & (new_f[:, 1] != new_f[:, 2]) & (new_f[:, 2] != new_f[:, 0])
    new_f = new_f[valid]
    return np.ascontiguousarray(new_v, dtype=np.float32), np.ascontiguousarray(new_f, dtype=np.uint32)


def write_nmm(path: str, verts: np.ndarray, faces: np.ndarray, soma_nm: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"NMM1")
        f.write(struct.pack("<2I", int(len(verts)), int(len(faces))))
        f.write(struct.pack("<3f", float(soma_nm[0]), float(soma_nm[1]), float(soma_nm[2])))
        f.write(verts.tobytes())
        f.write(faces.tobytes())


def load_obj_mesh(path: str):
    verts = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                p = line.split()
                faces.append((int(p[1]) - 1, int(p[2]) - 1, int(p[3]) - 1))
    v = np.array(verts, dtype=np.float32)
    fl = np.array(faces, dtype=np.uint32)
    return v, fl


def compute_vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    fn = np.cross(e1, e2)
    vn = np.zeros_like(v, dtype=np.float32)
    np.add.at(vn, f[:, 0], fn)
    np.add.at(vn, f[:, 1], fn)
    np.add.at(vn, f[:, 2], fn)
    lens = np.linalg.norm(vn, axis=1, keepdims=True)
    lens = np.where(lens > 1e-8, lens, 1.0)
    return (vn / lens).astype(np.float32)


def build_circuit_html(out_html: str, neuron_records: list, edges_csv: str):
    all_mn = np.array([1e30, 1e30, 1e30], dtype=np.float32)
    all_mx = np.array([-1e30, -1e30, -1e30], dtype=np.float32)
    loaded = []

    for rec in neuron_records:
        v, f = load_obj_mesh(rec["circuit_obj_path"])
        vn = compute_vertex_normals(v, f)
        all_mn = np.minimum(all_mn, v.min(axis=0))
        all_mx = np.maximum(all_mx, v.max(axis=0))
        loaded.append((rec, v, vn, f))

    center = (all_mn + all_mx) * 0.5
    half_span = float(max(1.0, np.max((all_mx - all_mn) * 0.5)))
    inv_half = 1.0 / half_span

    e_palette = [
        [0.16, 0.78, 0.96], [0.22, 0.86, 0.84], [0.28, 0.68, 0.98],
        [0.18, 0.88, 0.72], [0.35, 0.74, 0.96], [0.20, 0.82, 0.90],
        [0.42, 0.80, 0.98], [0.24, 0.90, 0.78], [0.30, 0.62, 0.95],
    ]

    packed_neurons = []
    root_to_type = {}
    root_to_idx = {}
    for idx, (rec, v, vn, f) in enumerate(loaded):
        root_to_type[str(rec["root_id"])] = rec["cell_type"]
        root_to_idx[str(rec["root_id"])] = idx
        ct = rec["cell_type"]

        if ct in CELL_TYPE_COLORS:
            col = CELL_TYPE_COLORS[ct]
        elif ct in EXCITATORY_TYPES:
            col = e_palette[idx % len(e_palette)]
        else:
            print(f"warning: unknown cell type {ct!r} for {rec['root_id']}, drawn grey")
            col = UNKNOWN_COLOR

        pos_q = np.round(np.clip((v - center) * inv_half, -1.0, 1.0) * 32767.0).astype("<i2")
        nrm_q = np.round(np.clip(vn, -1.0, 1.0) * 127.0).astype("<i1")
        idx_u = f.astype("<u4")
        soma = np.array(rec["soma_um"], dtype=np.float32)
        sn = np.clip((soma - center) * inv_half, -1.0, 1.0)
        packed_neurons.append({
            "node_id": rec["node_id"],
            "root_id": str(rec["root_id"]),
            "cell_type": ct,
            "exc": ct in EXCITATORY_TYPES,
            "inh": is_inhibitory(ct),
            "html": rec.get("html_path", ""),
            "color": col,
            "soma": [float(sn[0]), float(-sn[1]), float(sn[2])],
            "cable_um": round(rec.get("cable_um", 0.0), 1),
            "swc_nodes": rec.get("swc_nodes", 0),
            "pref_ori": rec.get("pref_ori", ""),
            "osi": rec.get("osi", ""),
            "nv": int(len(v)),
            "nf": int(len(f)),
            "pos": base64.b64encode(pos_q.tobytes()).decode("ascii"),
            "nrm": base64.b64encode(nrm_q.tobytes()).decode("ascii"),
            "idx": base64.b64encode(idx_u.tobytes()).decode("ascii"),
        })

    syn_pts = []
    syn_cols = []
    syn_pre = []
    syn_post = []
    if edges_csv and os.path.exists(edges_csv):
        with open(edges_csv, newline="") as ef:
            for row in csv.DictReader(ef):
                p = np.array([float(row["syn_x"]), float(row["syn_y"]), float(row["syn_z"])], dtype=np.float32)
                pn = np.clip((p - center) * inv_half, -1.0, 1.0)
                pre_t = root_to_type.get(row["pre_pt_root_id"], "23P")
                post_t = root_to_type.get(row["post_pt_root_id"], "23P")
                pre_i = is_inhibitory(pre_t)
                post_i = is_inhibitory(post_t)
                direction = ("I" if pre_i else "E") + "→" + ("I" if post_i else "E")
                sc = SYNAPSE_COLORS[direction]
                syn_pts.extend([float(pn[0]), float(-pn[1]), float(pn[2])])
                syn_cols.extend(sc)
                syn_pre.append(root_to_idx.get(row["pre_pt_root_id"], -1))
                syn_post.append(root_to_idx.get(row["post_pt_root_id"], -1))

    n_total = len(packed_neurons)
    n_exc = sum(1 for p in packed_neurons if p["exc"])
    n_inh = sum(1 for p in packed_neurons if p["inh"])

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>wintermute circuit viewer</title>
<style>
html,body{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#06090f;color:#d8e6f2;font-family:monospace;}}
canvas{{display:block;width:100%;height:100%;cursor:grab;}}canvas:active{{cursor:grabbing;}}
#panel{{position:fixed;top:10px;left:10px;bottom:10px;width:270px;background:rgba(9,14,22,0.90);border:1px solid #22344b;border-radius:8px;padding:10px;display:flex;flex-direction:column;gap:8px;z-index:10;font-size:11px;}}
.row{{display:flex;gap:5px;flex-wrap:wrap;}}
button{{background:#142233;color:#cfe7ff;border:1px solid #2c4666;border-radius:5px;padding:4px 7px;font-family:monospace;font-size:10px;cursor:pointer;}}
button:hover{{background:#1f334d;}}button.on{{background:#1b5a7a;border-color:#4ec5ff;color:#fff;}}
#list{{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:3px;padding-right:3px;}}
.item{{display:flex;align-items:center;gap:6px;padding:3px 5px;border-radius:4px;background:rgba(20,32,48,0.55);cursor:pointer;}}
.item:hover{{background:rgba(32,54,82,0.8);}}
.sw{{width:9px;height:9px;border-radius:50%;flex-shrink:0;}}
.meta{{color:#7f9bb5;font-size:10px;margin-left:auto;}}
</style></head><body><canvas id="c"></canvas>
<div id="panel">
<div style="color:#55d0ff;font-weight:bold;font-size:12px;">wintermute circuit viewer</div>
<div id="stats" style="color:#8aa4bd;font-size:10px;"></div>
<div class="row">
<button id="bAll" class="on">all ({n_total})</button>
<button id="bExc">exc ({n_exc})</button>
<button id="bInh">inh ({n_inh})</button>
<button id="bSyn" class="on">synapses ({len(syn_pts)//3})</button>
</div>
<div class="row">
<button id="bXray">x-ray</button>
<button id="bReset">reset view</button>
<button id="bNone">hide all</button>
</div>
{legend_html()}
<div id="list"></div>
<div style="color:#6c849c;font-size:10px;">drag: orbit | shift/right: pan | scroll: zoom | click row: focus soma</div>
</div>
<script>
const SPAN_UM={half_span*2.0:.2f};
const NEURONS={json.dumps(packed_neurons, separators=(",", ":"))};
const SYN_PTS=new Float32Array({json.dumps(syn_pts, separators=(",", ":"))});
const SYN_COLS=new Float32Array({json.dumps(syn_cols, separators=(",", ":"))});
const SYN_PRE={json.dumps(syn_pre)};
const SYN_POST={json.dumps(syn_post)};
const canvas=document.getElementById('c');
const gl=canvas.getContext('webgl2',{{antialias:true,alpha:false}});
if(!gl){{document.body.innerHTML='<p style="padding:20px">webgl2 required</p>';throw new Error('no webgl2');}}
function b64ToBuf(s){{const bin=atob(s);const n=bin.length;const u=new Uint8Array(n);for(let i=0;i<n;++i)u[i]=bin.charCodeAt(i);return u.buffer;}}
function mkShader(type,src){{const sh=gl.createShader(type);gl.shaderSource(sh,src);gl.compileShader(sh);return sh;}}
function mkProg(vs,fs){{const p=gl.createProgram();gl.attachShader(p,mkShader(gl.VERTEX_SHADER,vs));gl.attachShader(p,mkShader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);return p;}}
const meshProg=mkProg(`#version 300 es
in vec3 aPos;in vec3 aNrm;uniform mat4 uMVP;uniform mat3 uNrmMat;out vec3 vNrm;
void main(){{vec3 p=vec3(aPos.x,-aPos.y,aPos.z);vec3 n=vec3(aNrm.x,-aNrm.y,aNrm.z);vNrm=normalize(uNrmMat*n);gl_Position=uMVP*vec4(p,1.0);}}`,
`#version 300 es
precision highp float;in vec3 vNrm;uniform vec3 uColor;uniform float uAlpha;out vec4 oCol;
void main(){{vec3 n=normalize(vNrm);if(!gl_FrontFacing)n=-n;
vec3 l1=normalize(vec3(0.45,0.65,0.61));vec3 l2=normalize(vec3(-0.55,-0.35,0.75));vec3 v=vec3(0.0,0.0,1.0);
float d1=max(dot(n,l1),0.0);float d2=max(dot(n,l2),0.0);float sp=pow(max(dot(n,normalize(l1+v)),0.0),26.0);
oCol=vec4(clamp(uColor*(0.20+0.72*d1+0.22*d2)+vec3(0.22*sp),0.0,1.0),uAlpha);}}`);
const ptProg=mkProg(`#version 300 es
in vec3 aPos;in vec3 aCol;uniform mat4 uMVP;uniform float uSize;out vec3 vCol;
void main(){{vCol=aCol;gl_Position=uMVP*vec4(aPos,1.0);gl_PointSize=uSize;}}`,
`#version 300 es
precision highp float;in vec3 vCol;out vec4 oCol;
void main(){{vec2 d=gl_PointCoord-vec2(0.5);if(dot(d,d)>0.25)discard;oCol=vec4(vCol,0.95);}}`);
let totalFaces=0;
const gpuNeurons=NEURONS.map(m=>{{
const pos16=new Int16Array(b64ToBuf(m.pos));const nrm8=new Int8Array(b64ToBuf(m.nrm));const idx32=new Uint32Array(b64ToBuf(m.idx));
const vao=gl.createVertexArray();gl.bindVertexArray(vao);
const pb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.bufferData(gl.ARRAY_BUFFER,pos16,gl.STATIC_DRAW);
const lp=gl.getAttribLocation(meshProg,'aPos');gl.enableVertexAttribArray(lp);gl.vertexAttribPointer(lp,3,gl.SHORT,true,0,0);
const nb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,nb);gl.bufferData(gl.ARRAY_BUFFER,nrm8,gl.STATIC_DRAW);
const ln=gl.getAttribLocation(meshProg,'aNrm');gl.enableVertexAttribArray(ln);gl.vertexAttribPointer(ln,3,gl.BYTE,true,0,0);
const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,idx32,gl.STATIC_DRAW);
gl.bindVertexArray(null);totalFaces+=m.nf;
return {{meta:m,vao,count:idx32.length,visible:true}};
}});
let gpuSyn=null;
if(SYN_PTS.length>0){{
const vao=gl.createVertexArray();gl.bindVertexArray(vao);
const pb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.bufferData(gl.ARRAY_BUFFER,SYN_PTS,gl.STATIC_DRAW);
const lp=gl.getAttribLocation(ptProg,'aPos');gl.enableVertexAttribArray(lp);gl.vertexAttribPointer(lp,3,gl.FLOAT,false,0,0);
const cb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,cb);gl.bufferData(gl.ARRAY_BUFFER,SYN_COLS,gl.STATIC_DRAW);
const lc=gl.getAttribLocation(ptProg,'aCol');gl.enableVertexAttribArray(lc);gl.vertexAttribPointer(lc,3,gl.FLOAT,false,0,0);
gl.bindVertexArray(null);gpuSyn={{vao,pb,cb,count:SYN_PTS.length/3}};
}}
document.getElementById('stats').textContent=`${{gpuNeurons.length}} neurons | ${{totalFaces.toLocaleString()}} faces | ${{SYN_PTS.length/3}} synapses | ${{SPAN_UM.toFixed(0)}} um`;
const listEl=document.getElementById('list');
const checks=[];
gpuNeurons.forEach((gn,idx)=>{{
const m=gn.meta;
const row=document.createElement('div');row.className='item';
const cb=document.createElement('input');cb.type='checkbox';cb.checked=true;
cb.onclick=e=>{{e.stopPropagation();gn.visible=cb.checked;draw();}};
checks.push(cb);
const sw=document.createElement('span');sw.className='sw';
sw.style.background=`rgb(${{Math.round(m.color[0]*255)}},${{Math.round(m.color[1]*255)}},${{Math.round(m.color[2]*255)}})`;
const lbl=document.createElement('span');lbl.textContent=`#${{String(m.node_id).padStart(2,'0')}} ${{m.cell_type}}`;
const mt=document.createElement('span');mt.className='meta';mt.textContent=`${{(m.cable_um/1000).toFixed(1)}}mm`;
const op=document.createElement('a');op.textContent='open';op.href=m.html;op.target='_blank';op.rel='noopener';op.style.cssText='color:#55d0ff;margin-left:6px;font-size:10px;';op.onclick=e=>e.stopPropagation();
row.append(cb,sw,lbl,mt,op);
row.onclick=()=>{{tx=m.soma[0];ty=m.soma[1];tz=m.soma[2];dist=0.55;draw();}};
listEl.appendChild(row);
}});
let showSyn=true,xray=false,yaw=0.35,pitch=0.14,dist=2.3,tx=0,ty=0,tz=0;
function setMask(fn){{gpuNeurons.forEach((g,i)=>{{g.visible=fn(g.meta);checks[i].checked=g.visible;}});draw();}}
document.getElementById('bAll').onclick=()=>setMask(()=>true);
document.getElementById('bExc').onclick=()=>setMask(m=>m.exc);
document.getElementById('bInh').onclick=()=>setMask(m=>m.inh);
document.getElementById('bNone').onclick=()=>setMask(()=>false);
document.getElementById('bSyn').onclick=e=>{{showSyn=!showSyn;e.target.classList.toggle('on',showSyn);draw();}};
document.getElementById('bXray').onclick=e=>{{xray=!xray;e.target.classList.toggle('on',xray);draw();}};
document.getElementById('bReset').onclick=()=>{{tx=0;ty=0;tz=0;yaw=0.35;pitch=0.14;dist=2.3;draw();}};
function mul4(a,b){{const r=new Float32Array(16);for(let c=0;c<4;++c)for(let ro=0;ro<4;++ro)r[c*4+ro]=a[ro]*b[c*4]+a[4+ro]*b[c*4+1]+a[8+ro]*b[c*4+2]+a[12+ro]*b[c*4+3];return r;}}
function persp(fov,asp,n,f){{const t=1/Math.tan(fov*0.5);const m=new Float32Array(16);m[0]=t/asp;m[5]=t;m[10]=(f+n)/(n-f);m[11]=-1;m[14]=(2*f*n)/(n-f);return m;}}
function viewMat(){{
const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
const r00=cy,r01=0,r02=sy,r10=sy*sp,r11=cp,r12=-cy*sp,r20=-sy*cp,r21=sp,r22=cy*cp;
const px=-(r00*tx+r01*ty+r02*tz),py=-(r10*tx+r11*ty+r12*tz),pz=-(r20*tx+r21*ty+r22*tz)-dist;
return {{m:new Float32Array([r00,r10,r20,0,r01,r11,r21,0,r02,r12,r22,0,px,py,pz,1]),nm:new Float32Array([r00,r10,r20,r01,r11,r21,r02,r12,r22])}};
}}
function draw(){{
const dpr=Math.min(window.devicePixelRatio||1,2);
const w=Math.floor(canvas.clientWidth*dpr),h=Math.floor(canvas.clientHeight*dpr);
if(canvas.width!==w||canvas.height!==h){{canvas.width=w;canvas.height=h;}}
gl.viewport(0,0,w,h);gl.clearColor(0.022,0.032,0.052,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
const proj=persp(0.75,w/Math.max(1,h),0.01,100.0);
const {{m:vm,nm}}=viewMat();const mvp=mul4(proj,vm);
gl.enable(gl.DEPTH_TEST);
if(xray){{gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);}}
else{{gl.disable(gl.BLEND);gl.depthMask(true);}}
gl.useProgram(meshProg);
gl.uniformMatrix4fv(gl.getUniformLocation(meshProg,'uMVP'),false,mvp);
gl.uniformMatrix3fv(gl.getUniformLocation(meshProg,'uNrmMat'),false,nm);
gl.uniform1f(gl.getUniformLocation(meshProg,'uAlpha'),xray?0.32:1.0);
const uCol=gl.getUniformLocation(meshProg,'uColor');
for(const gn of gpuNeurons){{
if(!gn.visible)continue;
const c=gn.meta.color;gl.uniform3f(uCol,c[0],c[1],c[2]);
gl.bindVertexArray(gn.vao);gl.drawElements(gl.TRIANGLES,gn.count,gl.UNSIGNED_INT,0);
}}
gl.depthMask(true);gl.disable(gl.BLEND);
if(showSyn&&gpuSyn){{
gl.useProgram(ptProg);
gl.uniformMatrix4fv(gl.getUniformLocation(ptProg,'uMVP'),false,mvp);
gl.uniform1f(gl.getUniformLocation(ptProg,'uSize'),Math.max(4.0,Math.min(10.0,6.5/Math.sqrt(dist))));
const vis=gpuNeurons.map(g=>g.visible);const fp=[],fc=[];
for(let k=0;k<SYN_PRE.length;k++){{if((SYN_PRE[k]>=0&&vis[SYN_PRE[k]])||(SYN_POST[k]>=0&&vis[SYN_POST[k]])){{fp.push(SYN_PTS[3*k],SYN_PTS[3*k+1],SYN_PTS[3*k+2]);fc.push(SYN_COLS[3*k],SYN_COLS[3*k+1],SYN_COLS[3*k+2]);}}}}
gl.bindBuffer(gl.ARRAY_BUFFER,gpuSyn.pb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(fp),gl.DYNAMIC_DRAW);
gl.bindBuffer(gl.ARRAY_BUFFER,gpuSyn.cb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(fc),gl.DYNAMIC_DRAW);gpuSyn.count=fp.length/3;
gl.bindVertexArray(gpuSyn.vao);gl.drawArrays(gl.POINTS,0,gpuSyn.count);
}}
}}
let drag=0,lx=0,ly=0;
canvas.oncontextmenu=e=>e.preventDefault();
canvas.onmousedown=e=>{{drag=(e.button===2||e.shiftKey)?2:1;lx=e.clientX;ly=e.clientY;}};
window.onmouseup=()=>drag=0;
window.onmousemove=e=>{{if(!drag)return;const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
if(drag===1){{yaw+=dx*0.007;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.007));}}
else{{const s=dist*0.0018;const cy=Math.cos(yaw),sy=Math.sin(yaw);tx-=dx*s*cy;tz-=dx*s*sy;ty+=dy*s;}}draw();}};
canvas.onwheel=e=>{{e.preventDefault();dist=Math.max(0.08,Math.min(12.0,dist*Math.exp(e.deltaY*0.001)));draw();}};
window.onresize=draw;draw();
</script></body></html>"""

    with open(out_html, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(out_html) / (1024.0 * 1024.0)
    print(f"exported circuit viewer: {out_html.lower()} ({size_mb:.1f} mb)")


def parse_teasar_output(stdout_text: str):
    cable_um = 0.0
    swc_nodes = 0
    for line in stdout_text.splitlines():
        if "simplified swc" in line:
            parts = line.split()
            for idx, tok in enumerate(parts):
                if tok == "nodes" and idx > 0:
                    try:
                        swc_nodes = int(parts[idx - 1])
                    except ValueError:
                        pass
                if tok.startswith("cable="):
                    try:
                        cable_um = float(tok.split("=")[1])
                    except ValueError:
                        pass
    return swc_nodes, cable_um


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-csv", default="nodes.csv")
    parser.add_argument("--edges-csv", default="edges.csv")
    parser.add_argument("--bin", default="./build/neuromancers")
    parser.add_argument("--out-dir", default="./data/circuit")
    parser.add_argument("--datastack", default="minnie65_public")
    parser.add_argument("--circuit-tube-sides", type=int, default=6,
                        help="sides of each tube in the circuit viewer (more = smoother, bigger file)")
    parser.add_argument("--circuit-soma-vox", type=float, default=450,
                        help="soma voxel size in nm for the circuit viewer (smaller = more detail, bigger file)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(args.nodes_csv, newline="")))
    total_n = len(rows)
    print(f"processing {total_n} neurons into {args.out_dir.lower()}")

    cv_pub = CloudVolume(PUBLIC_M1300, use_https=True, progress=False)
    cv_cave = None
    try:
        client = caveclient.CAVEclient(args.datastack)
        cv_cave = CloudVolume(client.info.segmentation_source(), use_https=True, progress=False)
        print(f"caveclient authenticated: {args.datastack.lower()} v{client.materialize.version}")
    except Exception as exc:
        print(f"caveclient unavailable ({type(exc).__name__.lower()}); public m1300 only")

    records = []
    t_start = time.time()

    for idx_row, row in enumerate(rows):
        nid = int(row.get("node_id", idx_row))
        rid = int(row["pt_root_id"])
        ctype = row.get("cell_type", "neuron").strip()
        soma_um = [float(row["x"]), float(row["y"]), float(row["z"])]
        soma_nm = np.array(soma_um, dtype=np.float32) * 1000.0
        stem = f"n{nid:02d}_{ctype}_{rid}"

        nmm_path = os.path.join(args.out_dir, f"{stem}.nmm")
        swc_path = os.path.join(args.out_dir, f"{stem}.swc")
        obj_path = os.path.join(args.out_dir, f"{stem}.obj")
        cobj_path = os.path.join(args.out_dir, f"{stem}_circuit6.obj")
        html_path = os.path.join(args.out_dir, f"{stem}_viewer.html")

        t0 = time.time()
        source_used = "cached"
        if not os.path.exists(nmm_path):
            mesh = None
            try:
                res = cv_pub.mesh.get(rid, lod=2)
                if rid in res and len(res[rid].vertices) > 0:
                    mesh = res[rid]
                    source_used = "m1300_lod2"
            except Exception:
                mesh = None

            if mesh is None and cv_cave is not None:
                res = cv_cave.mesh.get(rid)
                mesh = res[rid]
                source_used = "cave_graphene"

            if mesh is None:
                print(f"[{idx_row + 1:02d}/{total_n:02d}] failed to fetch {ctype.lower()} {rid}")
                continue

            verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
            faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
            if len(verts) > 450000:
                verts, faces = decimate_mesh_grid(verts, faces, cell_nm=240.0)
            write_nmm(nmm_path, verts, faces, soma_nm)

        p1 = subprocess.run([args.bin, "--mesh", nmm_path, "--swc", swc_path], capture_output=True, text=True, check=True)
        swc_nodes, cable_um = parse_teasar_output(p1.stdout)

        subprocess.run(
            [args.bin, "--voxelize", swc_path, "--out-obj", obj_path, "--compare-mesh", nmm_path, "--html", html_path],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [args.bin, "--voxelize", swc_path, "--tube-sides", str(args.circuit_tube_sides), "--soma-vox", str(args.circuit_soma_vox), "--out-obj", cobj_path],
            capture_output=True,
            text=True,
            check=True,
        )

        dt = time.time() - t0
        print(f"[{idx_row + 1:02d}/{total_n:02d}] {stem.lower()} ({source_used}): swc={swc_nodes} nodes, cable={cable_um:.1f} um, dt={dt:.2f}s", flush=True)
        records.append({
            "node_id": nid,
            "root_id": rid,
            "cell_type": ctype,
            "soma_um": soma_um,
            "pref_ori": row.get("pref_ori", ""),
            "osi": row.get("OSI", ""),
            "dsi": row.get("DSI", ""),
            "swc_nodes": swc_nodes,
            "cable_um": cable_um,
            "nmm_path": os.path.basename(nmm_path),
            "swc_path": os.path.basename(swc_path),
            "obj_path": os.path.basename(obj_path),
            "circuit_obj_path": cobj_path,
            "html_path": os.path.basename(html_path),
        })

    circuit_html = os.path.join(args.out_dir, "circuit_viewer.html")
    build_circuit_html(circuit_html, records, args.edges_csv)
    for rec in records:
        p = rec.pop("circuit_obj_path", "")
        if p and os.path.exists(p):
            os.remove(p)

    summary_path = os.path.join(args.out_dir, "circuit_summary.json")
    with open(summary_path, "w") as sf:
        json.dump(records, sf, indent=2)

    print(f"completed {len(records)}/{total_n} neurons in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
