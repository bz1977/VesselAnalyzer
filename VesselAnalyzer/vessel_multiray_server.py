"""vessel_multiray_server.py
Pure-Python multi-ray radius estimation server for VesselAnalyzer.

Drop-in replacement for the former Blender-based server.
Requires only Python stdlib + numpy (already present in Slicer's Python).
trimesh is used when available for faster BVH; falls back to a manual
Möller–Trumbore implementation otherwise.

Start once per session (Slicer "Start Server" button, or manually):
    python vessel_multiray_server.py [port]          # default port 6789

HTTP endpoints
--------------
POST /multiray   – main radius-refinement call (see JSON contract below)
POST /shutdown   – graceful shutdown
GET  /ping       – health-check → 200 {"ok": true}

JSON contract (identical to the former Blender server)
-------------------------------------------------------
Request:
{
  "mesh":               "/abs/path/to/vessel.stl",   # required on first call
  "n_rays_normal":      12,       # optional
  "n_rays_bifurcation": 32,       # optional
  "angle_thresh":       0.5,      # radians
  "tangent_window":     2,
  "max_dist":           30.0,
  "min_hits":           4,
  "branches": [
    {"id": 0, "points": [[x,y,z], ...]},
    ...
  ]
}

Response:
{
  "radius_refined": {"0": [r0, r1, ...], "1": [...]},
  "confidence":     {"0": [c0, c1, ...], "1": [...]},
  "is_bifurcation": {"0": [bool, ...],   "1": [...]}
}
"""

from __future__ import annotations

import http.server
import json
import math
import os
import struct
import sys
import threading
import time

# ── numpy ─────────────────────────────────────────────────────────────────────
try:
    import numpy as np
    _NP = True
except ImportError:
    np = None
    _NP = False

# ── trimesh (optional — much faster BVH) ──────────────────────────────────────
try:
    import trimesh
    _TRIMESH = True
except ImportError:
    trimesh = None
    _TRIMESH = False

_LOG = "[MultiRayServer]"

# ── Stdout pipe-deadlock prevention ───────────────────────────────────────────
# When launched via subprocess.Popen(stdout=PIPE) without a drain thread, every
# print() blocks once the OS pipe buffer fills (~64 KB on Windows), causing the
# handler thread to stall before writing the HTTP response.  The client then
# sees "Remote end closed connection without response".
#
# Fix: redirect all print() to a log file at startup so the pipe stays empty
# regardless of how the process is launched.  This is a belt-and-suspenders
# guard; the drain-thread fix in the Popen caller is still the primary fix.
def _redirect_print_to_logfile(port):
    """Swap sys.stdout/stderr to a line-buffered file; return log path."""
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"vessel_multiray_server_{port}.log",
    )
    try:
        fh = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = fh
        sys.stderr = fh
        print(f"{_LOG} stdout/stderr redirected to {log_path}  "
              f"(pipe-deadlock prevention active)")
        return log_path
    except Exception:
        # Can't open log file — redirect to devnull so pipe still never fills
        try:
            null = open(os.devnull, "w")
            sys.stdout = null
            sys.stderr = null
        except Exception:
            pass
        return None

# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers (pure Python — mirrors vessel_blender_multiray_mixin.py)
# ══════════════════════════════════════════════════════════════════════════════

def _sub(a, b):   return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def _add(a, b):   return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def _scale(a, s): return (a[0]*s, a[1]*s, a[2]*s)
def _dot(a, b):   return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def _len(a):      return math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
def _norm(a):
    l = _len(a)
    return (0.,0.,0.) if l < 1e-12 else (a[0]/l, a[1]/l, a[2]/l)
def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def smooth_tangent(pts, i, window=2):
    lo = max(0, i - window)
    hi = min(len(pts) - 1, i + window)
    dirs = []
    for j in range(lo, hi):
        d = _sub(tuple(pts[j+1]), tuple(pts[j]))
        l = _len(d)
        if l > 1e-12:
            dirs.append(_scale(d, 1.0/l))
    if not dirs:
        if i == 0:             return _norm(_sub(tuple(pts[1]),    tuple(pts[0])))
        if i == len(pts) - 1: return _norm(_sub(tuple(pts[-1]),   tuple(pts[-2])))
        return _norm(_sub(tuple(pts[i+1]), tuple(pts[i-1])))
    ax = sum(d[0] for d in dirs)
    ay = sum(d[1] for d in dirs)
    az = sum(d[2] for d in dirs)
    return _norm((ax, ay, az))


def orthonormal_basis(tangent):
    ref = (0.,0.,1.) if abs(tangent[2]) < 0.9 else (1.,0.,0.)
    v1 = _norm(_cross(tangent, ref))
    v2 = _norm(_cross(tangent, v1))
    return v1, v2


def sample_directions(v1, v2, n):
    dirs = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        c, s = math.cos(a), math.sin(a)
        dirs.append(_norm((c*v1[0]+s*v2[0], c*v1[1]+s*v2[1], c*v1[2]+s*v2[2])))
    return dirs


def is_bifurcation_zone(pts, i, angle_thresh=0.5):
    if i == 0 or i >= len(pts) - 1:
        return False
    v1 = _norm(_sub(tuple(pts[i]),   tuple(pts[i-1])))
    v2 = _norm(_sub(tuple(pts[i+1]), tuple(pts[i])))
    dot = max(-1.0, min(1.0, _dot(v1, v2)))
    return math.acos(dot) > angle_thresh


def filter_outliers(distances):
    n = len(distances)
    if n < 4: return distances[:]
    s = sorted(distances)
    return s[n//4: 3*n//4]


def compute_confidence(distances):
    n = len(distances)
    if n < 2: return 0.1
    mean = sum(distances) / n
    var  = sum((d-mean)**2 for d in distances) / n
    return 1.0 / (1.0 + var)


def robust_radius(distances):
    if not distances: return 0.0
    s = sorted(distances)
    m = len(s)//2
    return (s[m-1]+s[m])/2.0 if len(s)%2==0 else s[m]


# ══════════════════════════════════════════════════════════════════════════════
# STL loader (binary STL — pure Python, no extra deps)
# ══════════════════════════════════════════════════════════════════════════════

def _is_ascii_stl(path):
    """Heuristic: ASCII STL files start with 'solid'."""
    try:
        with open(path, "rb") as f:
            header = f.read(80)
            # Binary STLs sometimes also start with 'solid'; cross-check
            # triangle count against file size to be sure.
            if not header.lower().startswith(b"solid"):
                return False
            n_tri = struct.unpack("<I", f.read(4))[0]
            import os as _os2
            expected = 84 + n_tri * 50
            return _os2.path.getsize(path) != expected
    except Exception:
        return False


def _load_stl_numpy(path):
    """Read a binary STL directly into a numpy (F,3,3) float32 array.

    This is ~100× faster than the Python-loop loader for large meshes
    (87k tris in ~0.05 s vs several seconds) because it uses a single
    np.frombuffer call and avoids all Python-level object allocation.

    Binary STL record layout (50 bytes per triangle):
        12 bytes — normal (3 × float32, ignored)
        36 bytes — 3 vertices × 3 × float32
         2 bytes — attribute byte count (ignored)
    """
    import numpy as _np
    with open(path, "rb") as f:
        f.read(80)              # 80-byte header
        n_tri = struct.unpack("<I", f.read(4))[0]
        raw   = f.read()        # read the rest in one shot

    # Each record is 50 bytes. Layout as float32 (12.5 floats per record):
    # [nx,ny,nz, v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z, attr(2 bytes)]
    # We reshape to (n_tri, 25) uint16 — but it's easier to read as uint8
    # and then reinterpret the float fields we need.
    dt = _np.dtype([
        ('normal', _np.float32, (3,)),
        ('v0',     _np.float32, (3,)),
        ('v1',     _np.float32, (3,)),
        ('v2',     _np.float32, (3,)),
        ('attr',   _np.uint16),
    ])
    recs = _np.frombuffer(raw, dtype=dt, count=n_tri)
    # Stack into (F, 3, 3): axis 1 = vertex index, axis 2 = xyz
    tris = _np.stack([recs['v0'], recs['v1'], recs['v2']], axis=1)
    return tris.astype(_np.float32)   # ensure float32


def _load_stl_ascii(path):
    """Minimal ASCII STL parser — pure Python, no deps."""
    verts, faces = [], []
    tri = []
    vi  = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            tok = line.strip().split()
            if not tok:
                continue
            if tok[0] == "vertex" and len(tok) >= 4:
                verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
                tri.append(vi)
                vi += 1
                if len(tri) == 3:
                    faces.append(tri)
                    tri = []
    return verts, faces


def _load_stl_binary(path):
    """Return (vertices Nx3, faces Mx3) as plain Python lists.

    Used only when numpy is unavailable (last-resort pure-Python path).
    """
    with open(path, "rb") as f:
        f.read(80)  # header
        n_tri = struct.unpack("<I", f.read(4))[0]
        verts = []
        faces = []
        vi = 0
        for _ in range(n_tri):
            f.read(12)  # normal (ignored)
            tri = []
            for _ in range(3):
                x, y, z = struct.unpack("<fff", f.read(12))
                verts.append((x, y, z))
                tri.append(vi)
                vi += 1
            f.read(2)  # attribute byte count
            faces.append(tri)
    return verts, faces


def load_mesh(path):
    """Return a BVH object appropriate to the backend in use.

    Fast path (numpy available, binary STL): reads with np.frombuffer —
    no Python-level loops, completes in < 1 s for 87k-triangle meshes.
    """
    if _TRIMESH:
        mesh = trimesh.load(path, force="mesh", process=False)
        print(f"{_LOG} trimesh loaded {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        return ("trimesh", mesh)

    ext = os.path.splitext(path)[1].lower()

    if _NP and ext == ".stl" and not _is_ascii_stl(path):
        # Fast numpy path — builds (F,3,3) array directly from binary STL
        import time as _t
        t0   = _t.time()
        tris = _load_stl_numpy(path)
        bvh  = _NumpyAABBGrid(tris)
        print(f"{_LOG} numpy fast-load + grid: {len(tris)} tris in {_t.time()-t0:.2f}s")
        return ("numpy_aabb", bvh)

    # Slow path — Python-loop loader (ASCII STL, OBJ, or no numpy)
    if ext == ".obj":
        verts, faces = _load_obj(path)
    elif _is_ascii_stl(path):
        verts, faces = _load_stl_ascii(path)
    else:
        verts, faces = _load_stl_binary(path)

    if _NP:
        bvh = _NumpyAABBGrid(verts, faces)
        print(f"{_LOG} numpy-AABB grid: {len(faces)} tris (no trimesh)")
        return ("numpy_aabb", bvh)

    # Last-resort pure-Python brute force
    triangles = [
        (verts[fi[0]], verts[fi[1]], verts[fi[2]])
        for fi in faces
    ]
    print(f"{_LOG} Python loader: {len(faces)} triangles from {ext} "
          "(no trimesh, no numpy — expect slow performance)")
    return ("python", triangles)


def _load_obj(path):
    """Minimal OBJ parser — pure Python, handles v/f lines only."""
    verts, faces = [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            tok = line.strip().split()
            if not tok:
                continue
            if tok[0] == "v" and len(tok) >= 4:
                verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
            elif tok[0] == "f" and len(tok) >= 4:
                # face indices are 1-based; strip texture/normal suffixes
                idx = [int(t.split("/")[0]) - 1 for t in tok[1:4]]
                faces.append(idx)
    return verts, faces




# ══════════════════════════════════════════════════════════════════════════════
# numpy AABB-grid BVH (used when trimesh absent but numpy present)
# Partitions triangles into a regular voxel grid so each ray only tests the
# small subset of triangles that overlap the cells it traverses — typically
# reducing the per-ray work from O(N_tris) to O(N_tris / cells^2).
# ══════════════════════════════════════════════════════════════════════════════

class _NumpyAABBGrid:
    """Axis-aligned grid accelerator for Möller–Trumbore ray casting.

    Resolution choice
    -----------------
    res=16 gives 16³ = 4096 cells.  For a vessel mesh (~88k tris) that spans
    a long tubular volume, res=32 causes many triangles to touch O(32²) cells
    in the elongated dimension, making the O(F × span_cells³) build loop take
    minutes and crash the server under its HTTP timeout.  res=16 keeps build
    time under 10 s on the IVC dataset while still cutting ray-test cost by
    ~16× vs brute force.

    Per-triangle cell cap
    ---------------------
    A single degenerate large triangle must not be allowed to register in
    thousands of cells.  MAX_TRI_CELLS caps the number of cells any one
    triangle can touch; oversized triangles fall back to brute-force testing
    at query time (flagged in self.large_tris).
    """

    MAX_TRI_CELLS = 64   # triangles touching more cells go into large_tris

    def __init__(self, verts_or_tris, faces=None, res=16):
        """Build the AABB grid.

        Two calling conventions:
          _NumpyAABBGrid(tris_array)          — fast path: tris is (F,3,3) float32
          _NumpyAABBGrid(verts, faces, res)   — slow path: Python lists
        """
        import numpy as _np_local
        import time as _t
        n   = _np_local
        t0  = _t.time()

        if faces is None:
            # Fast path — caller already built the (F,3,3) array
            tris = verts_or_tris
        else:
            # Slow path — build from Python vertex/face lists
            tris = n.array(
                [[verts_or_tris[fi[0]], verts_or_tris[fi[1]], verts_or_tris[fi[2]]]
                 for fi in faces],
                dtype=n.float32
            )  # shape (F, 3, 3)

        self.tris = tris
        self.res  = res

        # Compute per-triangle AABBs
        mn = tris.min(axis=1)   # (F, 3)
        mx = tris.max(axis=1)   # (F, 3)

        # World AABB with a small margin
        self.world_min = mn.min(axis=0) - 0.5
        self.world_max = mx.max(axis=0) + 0.5
        span = self.world_max - self.world_min
        span[span < 1e-6] = 1e-6
        self.cell_size = span / res

        # Assign each triangle to every cell its AABB overlaps.
        # Triangles whose AABB spans more than MAX_TRI_CELLS cells are stored
        # separately in self.large_tris and tested against every ray (brute
        # force) — they are rare but must not be silently dropped.
        cells_min = n.floor((mn - self.world_min) / self.cell_size).astype(int).clip(0, res-1)
        cells_max = n.floor((mx - self.world_min) / self.cell_size).astype(int).clip(0, res-1)

        grid       = {}
        large_tris = []
        n_tris     = len(tris)
        for fi in range(n_tris):
            x0, y0, z0 = int(cells_min[fi, 0]), int(cells_min[fi, 1]), int(cells_min[fi, 2])
            x1, y1, z1 = int(cells_max[fi, 0]), int(cells_max[fi, 1]), int(cells_max[fi, 2])
            n_cells = (x1-x0+1) * (y1-y0+1) * (z1-z0+1)
            if n_cells > self.MAX_TRI_CELLS:
                large_tris.append(fi)
                continue
            for xi in range(x0, x1+1):
                for yi in range(y0, y1+1):
                    for zi in range(z0, z1+1):
                        key = (xi, yi, zi)
                        if key not in grid:
                            grid[key] = []
                        grid[key].append(fi)

        self.grid       = grid
        self.large_tris = large_tris
        print(f"{_LOG} numpy-AABB grid built in {_t.time()-t0:.1f}s: "
              f"res={res} cells={len(grid)} large_tris={len(large_tris)}")

    def candidate_tris(self, origin, direction, max_dist):
        """Return a deduplicated list of triangle indices near the ray.

        Combines grid-accelerated DDA traversal with brute-force inclusion of
        self.large_tris (triangles too big to register in the grid).
        """
        import numpy as _np_local
        n = _np_local
        o = n.array(origin,    dtype=n.float32)
        d = n.array(direction, dtype=n.float32)

        # DDA traversal through the grid
        res  = self.res
        wmin = self.world_min
        cs   = self.cell_size

        cell = n.floor((o - wmin) / cs).astype(int)
        step = n.sign(d).astype(int)

        # Avoid div-by-zero
        d_safe = n.where(n.abs(d) < 1e-12, 1e-12, d)

        t_delta = n.abs(cs / d_safe)
        t_max   = n.where(
            step > 0,
            (wmin + (cell + 1) * cs - o) / d_safe,
            (wmin + cell       * cs - o) / d_safe,
        )

        seen   = set()
        result = []
        t_cur  = 0.0

        for _ in range(res * 3 + 3):
            if t_cur > max_dist:
                break
            key = (int(cell[0]), int(cell[1]), int(cell[2]))
            if not (0 <= key[0] < res and 0 <= key[1] < res and 0 <= key[2] < res):
                break
            for fi in self.grid.get(key, []):
                if fi not in seen:
                    seen.add(fi)
                    result.append(fi)
            # Advance to next cell
            ax = int(n.argmin(t_max))
            t_cur    = float(t_max[ax])
            t_max[ax] += t_delta[ax]
            cell[ax]  += step[ax]

        # Always include oversized triangles that weren't registered in grid
        for fi in self.large_tris:
            if fi not in seen:
                result.append(fi)

        return result

# ══════════════════════════════════════════════════════════════════════════════
# Ray-casting backends
# ══════════════════════════════════════════════════════════════════════════════

_EPS = 1e-9

def _moller_trumbore(orig, direction, v0, v1, v2):
    """Return hit distance along ray, or None."""
    e1 = _sub(v1, v0)
    e2 = _sub(v2, v0)
    h  = _cross(direction, e2)
    a  = _dot(e1, h)
    if -_EPS < a < _EPS:
        return None
    f = 1.0 / a
    s = _sub(orig, v0)
    u = f * _dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = _cross(s, e1)
    v = f * _dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * _dot(e2, q)
    return t if t > _EPS else None


def raycast_distances(bvh_obj, origin, directions, max_dist):
    """Return list of hit distances (one per direction, or absent if no hit)."""
    backend, data = bvh_obj

    if backend == "trimesh":
        mesh = data
        if _NP:
            origs = np.tile(np.array(origin, dtype=float), (len(directions), 1))
            dirs  = np.array(directions, dtype=float)
            locs, index_ray, _ = mesh.ray.intersects_location(
                ray_origins=origs, ray_directions=dirs
            )
            hits = [None] * len(directions)
            for loc, ri in zip(locs, index_ray):
                d = np.linalg.norm(np.array(loc) - np.array(origin))
                if d <= max_dist:
                    if hits[ri] is None or d < hits[ri]:
                        hits[ri] = float(d)
            return [h for h in hits if h is not None]
        else:
            return []

    elif backend == "numpy_aabb":
        # AABB-grid accelerated Möller–Trumbore — fast without trimesh
        grid   = data        # _NumpyAABBGrid instance
        tris   = grid.tris   # numpy (F,3,3)
        hits   = []
        for direction in directions:
            cand_idx = grid.candidate_tris(origin, direction, max_dist)
            best     = None
            for fi in cand_idx:
                tri  = tris[fi]
                v0, v1, v2 = tuple(tri[0]), tuple(tri[1]), tuple(tri[2])
                t = _moller_trumbore(origin, direction, v0, v1, v2)
                if t is not None and t <= max_dist:
                    if best is None or t < best:
                        best = float(t)   # coerce np.float32 → Python float
            if best is not None:
                hits.append(best)
        return hits

    else:  # pure Python Möller–Trumbore brute-force (last resort)
        triangles = data
        hits = []
        for d in directions:
            best = None
            for v0, v1, v2 in triangles:
                t = _moller_trumbore(origin, d, v0, v1, v2)
                if t is not None and t <= max_dist:
                    if best is None or t < best:
                        best = t
            if best is not None:
                hits.append(best)
        return hits


def nearest_distance(bvh_obj, origin):
    """Fallback: nearest-point distance to mesh surface."""
    backend, data = bvh_obj
    if backend == "trimesh":
        if _NP:
            _, dist, _ = trimesh.proximity.closest_point(data, [origin])
            return float(dist[0])
        return 0.0
    elif backend == "numpy_aabb":
        if _NP:
            tris = data.tris          # (F,3,3)
            o    = np.array(origin, dtype=np.float32)
            cen  = tris.mean(axis=1)  # (F,3) centroid per tri
            diff = cen - o
            d    = np.sqrt((diff*diff).sum(axis=1))
            return float(d.min())
        return 0.0
    else:
        # Python brute-force closest triangle centroid
        best = None
        ox, oy, oz = origin
        for v0, v1, v2 in data:
            cx = (v0[0]+v1[0]+v2[0])/3
            cy = (v0[1]+v1[1]+v2[1])/3
            cz = (v0[2]+v1[2]+v2[2])/3
            d  = math.sqrt((cx-ox)**2+(cy-oy)**2+(cz-oz)**2)
            if best is None or d < best:
                best = d
        return best or 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Per-branch computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_branch_multiray(bvh_obj, points, cfg):
    n_normal = cfg.get("n_rays_normal",      12)
    n_bif    = cfg.get("n_rays_bifurcation", 32)
    a_thresh = cfg.get("angle_thresh",       0.5)
    t_win    = cfg.get("tangent_window",     2)
    max_d    = cfg.get("max_dist",           30.0)
    min_hits = cfg.get("min_hits",           4)

    radii, confidences, bif_flags = [], [], []
    n_miss = 0   # count of points with too few ray hits

    for i, p in enumerate(points):
        origin  = tuple(p)
        in_bif  = is_bifurcation_zone(points, i, a_thresh)
        bif_flags.append(in_bif)

        tangent   = smooth_tangent(points, i, t_win)
        v1, v2    = orthonormal_basis(tangent)
        n_rays    = n_bif if in_bif else n_normal
        dirs      = sample_directions(v1, v2, n_rays)

        raw_hits  = raycast_distances(bvh_obj, origin, dirs, max_d)
        filtered  = filter_outliers(raw_hits)

        if len(filtered) < min_hits:
            # Too few ray hits — geometry is ambiguous (bifurcation zone,
            # coordinate mismatch, or thin wall).  nearest_distance() via the
            # numpy-AABB backend approximates using triangle centroids, which
            # is unreliable and can return wildly wrong values when the origin
            # is nowhere near the mesh.  Return 0.0 / conf=0.05 so downstream
            # code can detect and skip these points rather than silently
            # ingesting garbage radii.
            radius = 0.0
            conf   = 0.05
            n_miss += 1
            if n_miss == 1:
                print(f"{_LOG} WARNING: point {i} got <{min_hits} ray hits "
                      f"(raw={len(raw_hits)} filtered={len(filtered)}). "
                      f"If ALL points miss, check for RAS/LPS coordinate mismatch "
                      f"between centerline points and the exported STL.")
        else:
            radius = robust_radius(filtered)
            conf   = compute_confidence(filtered)

        radii.append(radius)
        confidences.append(conf)

    if n_miss == len(points):
        print(f"{_LOG} ERROR: 0/{len(points)} points had valid ray hits — "
              f"origin space almost certainly does not match mesh space. "
              f"Verify centerline points are in the same coordinate system as "
              f"the STL (Slicer exports LPS; centerlines are RAS — "
              f"negate X and Y before sending).")
    elif n_miss > 0:
        print(f"{_LOG}   {n_miss}/{len(points)} points fell back to zero-radius "
              f"(conf=0.05) — likely bifurcation zone or mesh boundary.")

    return radii, confidences, bif_flags


# ══════════════════════════════════════════════════════════════════════════════
# HTTP server
# ══════════════════════════════════════════════════════════════════════════════

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that coerces numpy scalar and array types to plain Python.

    numpy scalars (float32, float64, int32, …) are not JSON-serialisable by
    the default encoder.  compute_branch_multiray builds its hit-distance lists
    from triangle vertices that live in float32 arrays, so the accumulated
    radii / confidences end up as numpy float32 scalars and trigger:

        TypeError: Object of type float32 is not JSON serializable

    That exception propagates out of _send_json *before* any response bytes
    are written, causing the client to see RemoteDisconnected.  This encoder
    stops that at source.
    """
    def default(self, obj):
        if _NP:
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        # Fallback: coerce any other numeric scalar (e.g. np.float32 in an
        # unexpected wrapper, or a future numpy type) rather than crashing.
        try:
            return float(obj)
        except (TypeError, ValueError):
            pass
        return super().default(obj)


class _Handler(http.server.BaseHTTPRequestHandler):

    # Suppress per-request console noise
    def log_message(self, fmt, *args):
        pass

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def _send_json(self, code, obj):
        try:
            body = json.dumps(obj, cls=_NumpyEncoder).encode()
        except Exception as exc:
            # Serialisation failed — return a real 500 so the client never
            # sees RemoteDisconnected (which happens when we raise here and
            # the response is never written).
            err_body = json.dumps({"error": f"JSON serialisation failed: {exc}"}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
            return
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/ping":
            self._send_json(200, {
                "ok":      True,
                "trimesh": _TRIMESH,
                "numpy":   _NP,
                "pid":     os.getpid(),
                "log_file": getattr(self.server, "_log_file", None),
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/shutdown":
            self._send_json(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        if self.path == "/abort":
            self.server._abort_flag = True
            self._send_json(200, {"ok": True, "msg": "abort flag set"})
            print(f"{_LOG} /abort received — will stop after current branch")
            return

        if self.path != "/multiray":
            self._send_json(404, {"error": "not found"})
            return

        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_json(400, {"error": f"bad JSON: {exc}"})
            return

        # ── Mesh (re-)load ─────────────────────────────────────────────────
        mesh_path = payload.get("mesh")
        srv = self.server

        if mesh_path and mesh_path != getattr(srv, "_mesh_path", None):
            if not os.path.isfile(mesh_path):
                self._send_json(400, {"error": f"mesh not found: {mesh_path}"})
                return
            with srv._bvh_lock:
                # Re-check under the lock — another thread may have loaded it
                if mesh_path != getattr(srv, "_mesh_path", None):
                    print(f"{_LOG} Loading mesh: {mesh_path}")
                    t0 = time.time()
                    srv._bvh       = load_mesh(mesh_path)
                    srv._mesh_path = mesh_path
                    print(f"{_LOG} Mesh ready in {time.time()-t0:.2f}s")

        if not hasattr(srv, "_bvh") or srv._bvh is None:
            self._send_json(400, {"error": "no mesh loaded — pass 'mesh' key"})
            return

        cfg      = {k: payload[k] for k in
                    ("n_rays_normal","n_rays_bifurcation","angle_thresh",
                     "tangent_window","max_dist","min_hits")
                    if k in payload}
        branches = payload.get("branches", [])

        out_radii = {}
        out_conf  = {}
        out_bif   = {}

        # Reset abort flag at the start of each new /multiray request
        srv._abort_flag = False

        for branch in branches:
            # Check abort flag before starting each branch — allows clean
            # mid-run cancellation without corrupting a partial result set
            if srv._abort_flag:
                print(f"{_LOG} Abort flag set — stopping after "
                      f"{len(out_radii)} of {len(branches)} branches")
                self._send_json(200, {
                    "radius_refined": out_radii,
                    "confidence":     out_conf,
                    "is_bifurcation": out_bif,
                    "aborted":        True,
                })
                return

            bid    = str(branch["id"])
            points = branch["points"]
            print(f"{_LOG} Branch {bid}: {len(points)} pts")
            t0     = time.time()
            r, c, b = compute_branch_multiray(srv._bvh, points, cfg)
            dt = time.time() - t0
            print(f"{_LOG} Branch {bid}: done in {dt:.2f}s")
            out_radii[bid] = r
            out_conf[bid]  = c
            out_bif[bid]   = b

        self._send_json(200, {
            "radius_refined": out_radii,
            "confidence":     out_conf,
            "is_bifurcation": out_bif,
        })


class _Server(http.server.ThreadingHTTPServer):
    """Threading HTTP server — each request handled in its own daemon thread.

    This is critical for the numpy-AABB path: mesh load + BVH build + ray
    casting can take 30-120 s.  With a plain HTTPServer the main select() loop
    is blocked for the entire duration, the TCP socket's send buffer empties,
    and the client (urlopen) sees the connection go dead and raises
    "Remote end closed connection without response" — even though the server
    eventually computes the right answer.

    ThreadingHTTPServer keeps the main loop alive during long computes, so
    /ping and /abort remain responsive and the client's socket stays open.
    """
    _bvh        = None
    _mesh_path  = None
    _abort_flag = False   # set by POST /abort; checked between branches

    # Ensure the OS doesn't reclaim the port immediately after restart
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bvh_lock = threading.Lock()   # guards _bvh / _mesh_path writes

    def server_bind(self):
        # SO_KEEPALIVE keeps the TCP connection alive during long compute
        self.socket.setsockopt(
            __import__("socket").SOL_SOCKET,
            __import__("socket").SO_KEEPALIVE,
            1,
        )
        super().server_bind()


def run(port=6789):
    # Redirect stdout/stderr to a log file BEFORE anything else.
    # This is the definitive fix for the pipe-deadlock: once the OS pipe buffer
    # fills, every print() in the handler thread blocks — the HTTP response is
    # never written and the client sees RemoteDisconnected.  Redirecting to a
    # file means the pipe stays empty regardless of whether the caller started
    # a drain thread.
    log_path = _redirect_print_to_logfile(port)
    print(f"{_LOG} Log file: {log_path}")

    srv = _Server(("127.0.0.1", port), _Handler)
    srv._log_file = log_path   # exposed via /ping so Slicer can show the path
    print(f"{_LOG} Listening on port {port}  "
          f"(threading=yes  trimesh={'yes' if _TRIMESH else 'no'}  "
          f"numpy={'yes' if _NP else 'no'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"{_LOG} Stopped.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 6789
    run(port)


# ── Slicer module-scanner guard ───────────────────────────────────────────────
# Slicer auto-scans every .py file in the module folder and looks for a class
# whose name matches the filename.  This file is a standalone HTTP server
# script — NOT a loadable module — but the stub below satisfies the scanner
# and suppresses the RuntimeError / Qt instantiation warnings in the console.
class vessel_multiray_server:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title  = "vessel_multiray_server"
            parent.hidden = True
