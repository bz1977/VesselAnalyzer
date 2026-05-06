# vessel_blender_multiray_mixin.py
# Smart adaptive multi-ray radius estimation for VesselAnalyzer
# Build tag: v1.1 (2026-04-28)
#
# OVERVIEW
# ────────
# Replaces nearest-distance radius with a geometry-aware multi-ray approach.
# At each centerline point the algorithm:
#
#   1. Computes a smoothed tangent (window-averaged over neighbours)
#   2. Detects whether the point is in a bifurcation / sharp-bend zone
#   3. Chooses an adaptive ray count  (32 in bif zones, 12 elsewhere)
#   4. Builds an orthonormal plane basis perpendicular to the tangent
#   5. Casts rays radially outward, intersects the BVH vessel mesh
#   6. Rejects outliers (keeps the middle 50 % of hit distances)
#   7. Returns the median of the filtered set as the radius estimate
#   8. Attaches a confidence score  1/(1+variance) so downstream code
#      can weight or skip low-quality points
#
# The pipeline communicates with an external HTTP server
# (vessel_blender_multiray_server.py, started externally) via a simple
# JSON POST to /multiray.  Start the server once per session; the port
# is configurable in the VesselAnalyzer UI (default 6789).
#
# PUBLIC API (Logic side)
# ───────────────────────
#   MultiRayBlenderMixin (mix into VesselAnalyzerLogic)
#       runMultiRayBlender(meshPath=None, port=None)  →  True | False
#       _multiray_apply_to_branch_meta(result_dict)
#
# PUBLIC API (Widget side)
# ────────────────────────
#   MultiRayBlenderWidgetMixin (mix into VesselAnalyzerWidget)
#       onMultiRayBlenderMap()
#       onMultiRayBlenderClear()
#
# INTEGRATION (VesselAnalyzer.py)
# ────────────────────────────────
#   # widget class:
#   from vessel_blender_multiray_mixin import MultiRayBlenderWidgetMixin as _MRW
#   onMultiRayBlenderMap   = _MRW.onMultiRayBlenderMap
#   onMultiRayBlenderClear = _MRW.onMultiRayBlenderClear
#
#   # logic imports:
#   from vessel_blender_multiray_mixin import MultiRayBlenderMixin
#
#   # VesselAnalyzerLogic MRO — insert before ScriptedLoadableModuleLogic:
#   class VesselAnalyzerLogic(..., MultiRayBlenderMixin, ...): ...
#
# JSON CONTRACT (POST /multiray)
# ──────────────────────────────
# Request body:
# {
#   "mesh": "/abs/path/to/vessel.stl",
#   "n_rays_normal": 12,       // optional, default 12
#   "n_rays_bifurcation": 32,  // optional, default 32
#   "angle_thresh": 0.5,       // radians — curvature threshold for bif detection
#   "tangent_window": 2,       // points either side for tangent smoothing
#   "max_dist": 30.0,          // BVH ray max distance (must exceed vessel radius)
#   "branches": [
#     { "id": 0, "points": [[x,y,z], ...] },
#     ...
#   ]
# }
#
# Response body:
# {
#   "radius_refined": { "0": [r0, r1, ...], "1": [...] },
#   "confidence":     { "0": [c0, c1, ...], "1": [...] },
#   "is_bifurcation": { "0": [bool, ...],   "1": [...] }
# }
#
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import math
import os
import threading
import time as _time
import traceback

# Optional — only needed for the Slicer-side mixin, not the Blender script
try:
    import slicer
    import qt
    _SLICER_OK = True
except ImportError:
    slicer = qt = None
    _SLICER_OK = False

_LOG = "[MultiRayBlender]"

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULTS = dict(
    n_rays_normal     = 12,
    n_rays_bifurcation= 32,
    angle_thresh      = 0.5,    # radians — ~28.6°, plenty sensitive for vessels
    tangent_window    = 2,
    max_dist          = 30.0,
    min_hits          = 4,      # fewer hits → fallback to nearest-distance
)

# Confidence threshold below which a radius value is considered unreliable.
# Downstream code (renal gating, stent sizing, etc.) should skip these points.
CONFIDENCE_THRESHOLD = 0.3


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers (pure Python — no bpy, usable inside and outside Blender)
# ══════════════════════════════════════════════════════════════════════════════

def _vec3_sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def _vec3_add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def _vec3_scale(a, s):
    return (a[0]*s, a[1]*s, a[2]*s)

def _vec3_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _vec3_length(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def _vec3_normalized(a):
    l = _vec3_length(a)
    if l < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0]/l, a[1]/l, a[2]/l)

def _vec3_cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def smooth_tangent(points, i, window=2):
    """Tangent at point i, averaged over 'window' neighbours each side.

    Averaging over a window greatly reduces direction noise on densely-sampled
    centerlines, which in turn stabilises the orthonormal plane used for rays.
    """
    lo = max(0, i - window)
    hi = min(len(points) - 1, i + window)
    dirs = []
    for j in range(lo, hi):
        d = _vec3_sub(tuple(points[j+1]), tuple(points[j]))
        l = _vec3_length(d)
        if l > 1e-12:
            dirs.append(_vec3_scale(d, 1.0/l))
    if not dirs:
        # degenerate — try simple finite difference
        if i == 0:
            return _vec3_normalized(_vec3_sub(tuple(points[1]), tuple(points[0])))
        if i == len(points) - 1:
            return _vec3_normalized(_vec3_sub(tuple(points[-1]), tuple(points[-2])))
        return _vec3_normalized(_vec3_sub(tuple(points[i+1]), tuple(points[i-1])))
    ax = sum(d[0] for d in dirs)
    ay = sum(d[1] for d in dirs)
    az = sum(d[2] for d in dirs)
    return _vec3_normalized((ax, ay, az))


def orthonormal_basis(tangent):
    """Return two unit vectors (v1, v2) perpendicular to tangent.

    Chooses a reference axis that avoids near-parallel singularities.
    """
    if abs(tangent[2]) < 0.9:
        ref = (0.0, 0.0, 1.0)
    else:
        ref = (1.0, 0.0, 0.0)
    v1 = _vec3_normalized(_vec3_cross(tangent, ref))
    v2 = _vec3_normalized(_vec3_cross(tangent, v1))
    return v1, v2


def sample_directions(v1, v2, n):
    """Return n unit vectors uniformly distributed in the v1-v2 plane."""
    dirs = []
    for k in range(n):
        angle = 2.0 * math.pi * k / n
        c, s = math.cos(angle), math.sin(angle)
        d = (c*v1[0]+s*v2[0], c*v1[1]+s*v2[1], c*v1[2]+s*v2[2])
        dirs.append(_vec3_normalized(d))
    return dirs


def is_bifurcation_zone(points, i, angle_thresh=0.5):
    """True when the local curvature exceeds angle_thresh (radians).

    Uses a simple 3-point angle: angle between the incoming and outgoing
    direction vectors.  Endpoints are never considered bifurcation zones.
    """
    if i == 0 or i >= len(points) - 1:
        return False
    v1 = _vec3_normalized(_vec3_sub(tuple(points[i]),   tuple(points[i-1])))
    v2 = _vec3_normalized(_vec3_sub(tuple(points[i+1]), tuple(points[i])))
    dot = max(-1.0, min(1.0, _vec3_dot(v1, v2)))
    angle = math.acos(dot)
    return angle > angle_thresh


def filter_outliers(distances):
    """Keep the middle 50 % of hit distances (IQR filter).

    Outliers (rays that escape the vessel or hit a nearby branch) tend to
    cluster at one end of the sorted list.  Discarding the outer quartiles
    leaves only the rays that genuinely hit the local wall.
    Returns the filtered list; may be empty.
    """
    n = len(distances)
    if n < 4:
        return distances[:]
    s = sorted(distances)
    lo = n // 4
    hi = 3 * n // 4
    return s[lo:hi]


def compute_confidence(distances):
    """Return confidence ∈ (0, 1].

    confidence = 1 / (1 + variance)

    High variance → rays hit very different distances → geometry is complex
    or the mesh is noisy → low confidence.  Low variance → all rays measure
    similar radii → the local cross-section is well-defined → high confidence.
    """
    n = len(distances)
    if n < 2:
        return 0.1
    mean = sum(distances) / n
    var  = sum((d - mean)**2 for d in distances) / n
    return 1.0 / (1.0 + var)


def robust_radius(distances):
    """Median of the provided distance list."""
    if not distances:
        return 0.0
    s = sorted(distances)
    m = len(s) // 2
    if len(s) % 2 == 0:
        return (s[m-1] + s[m]) / 2.0
    return s[m]


# ══════════════════════════════════════════════════════════════════════════════
# Core per-branch computation (pure Python — runs inside Blender via bpy BVH)
# ══════════════════════════════════════════════════════════════════════════════

def compute_branch_multiray(bvh, points, cfg):
    """Process one branch.

    Parameters
    ----------
    bvh    : mathutils.bvhtree.BVHTree  (Blender object)
    points : list of [x, y, z]
    cfg    : dict with keys from _DEFAULTS

    Returns
    -------
    radii       : list[float]   one per point
    confidences : list[float]   one per point
    bif_flags   : list[bool]    one per point
    """
    # Late import — only available inside Blender
    from mathutils import Vector as _V

    n_normal = cfg.get("n_rays_normal",      _DEFAULTS["n_rays_normal"])
    n_bif    = cfg.get("n_rays_bifurcation", _DEFAULTS["n_rays_bifurcation"])
    a_thresh = cfg.get("angle_thresh",       _DEFAULTS["angle_thresh"])
    t_win    = cfg.get("tangent_window",     _DEFAULTS["tangent_window"])
    max_d    = cfg.get("max_dist",           _DEFAULTS["max_dist"])
    min_hits = cfg.get("min_hits",           _DEFAULTS["min_hits"])

    radii       = []
    confidences = []
    bif_flags   = []

    for i, p in enumerate(points):
        p_vec = _V(p)

        # ── 1. Bifurcation detection ──────────────────────────────────────
        in_bif = is_bifurcation_zone(points, i, a_thresh)
        bif_flags.append(in_bif)

        # ── 2. Smoothed tangent + orthonormal plane ───────────────────────
        tangent = smooth_tangent(points, i, t_win)
        # Convert to mathutils.Vector for orthonormal_basis (pure-Python tuples)
        v1, v2 = orthonormal_basis(tangent)

        # ── 3. Adaptive ray count ─────────────────────────────────────────
        n_rays = n_bif if in_bif else n_normal

        # ── 4. Cast rays ──────────────────────────────────────────────────
        dirs = sample_directions(v1, v2, n_rays)
        raw_hits = []

        for d in dirs:
            # Blender BVH expects mathutils.Vector
            d_vec = _V(d)
            hit, _, _, _ = bvh.ray_cast(p_vec, d_vec, max_d)
            if hit is not None:
                raw_hits.append((hit - p_vec).length)

        # ── 5. Outlier rejection ──────────────────────────────────────────
        filtered = filter_outliers(raw_hits)

        # ── 6. Fallback if too few valid hits ─────────────────────────────
        if len(filtered) < min_hits:
            # Nearest-distance fallback with low confidence
            _, _, _, nd = bvh.find_nearest(p_vec)
            radius = float(nd) if nd is not None else 0.0
            conf   = 0.15   # explicit low-confidence flag for downstream
        else:
            radius = robust_radius(filtered)
            conf   = compute_confidence(filtered)

        radii.append(radius)
        confidences.append(conf)

    return radii, confidences, bif_flags


# ══════════════════════════════════════════════════════════════════════════════
# Slicer logic mixin
# ══════════════════════════════════════════════════════════════════════════════

class MultiRayBlenderMixin:
    """Mix into VesselAnalyzerLogic.

    Requires self.branchMeta (populated by CenterlineMixin) and the user to
    have set self._blender_exe_path (or passed meshPath explicitly).
    """

    # ── Public entry point ────────────────────────────────────────────────────

    def runMultiRayBlender(self, meshPath=None, port=None, cfg_overrides=None):
        """Run multi-ray radius refinement via the external HTTP API server.

        Parameters
        ----------
        meshPath     : str | None  — STL path; auto-exported from modelNode if omitted
        port         : int | None  — server port; falls back to self._blender_server_port
                                     (default 6789)
        cfg_overrides: dict | None — override any _DEFAULTS keys

        Returns True on success, False on failure.
        """
        print(f"{_LOG} ══ Multi-Ray Radius Refinement ══════════════════════")

        port = port or getattr(self, "_blender_server_port", 6789)

        # ── Debug: report server process ownership ────────────────────────
        _srv_proc = getattr(self, "_mr_server_proc", None)
        if _srv_proc is not None:
            _rc = _srv_proc.poll()
            print(f"{_LOG} [DBG] owned server: pid={_srv_proc.pid}  "
                  f"alive={_rc is None}  port={port}")
            if _rc is not None:
                print(f"{_LOG} [DBG] WARNING: owned process is dead "
                      f"(returncode={_rc}) — pipe-deadlock diagnosis inconclusive")
        else:
            print(f"{_LOG} [DBG] no owned server process (external or pre-fix session) "
                  f"port={port}")

        self._mr_progress(5)  # 5 % — checking server
        self._mr_flush()

        if not self._mr_ping(port):
            self._mr_status(
                f"No multi-ray server found on port {port}. "
                "Start server via \u25b6 Start Server and confirm the port matches."
            )
            self._mr_progress(0, visible=False)
            return False

        # ── Verify the owned process is the one serving requests ─────────
        # onBlenderServerStart now always kills stale servers before launching,
        # so by the time we reach here the owned process should match the ping
        # PID.  Log a warning if they differ but do not attempt auto-restart —
        # a second Popen from here would itself become the deadlocking child.
        _srv_proc = getattr(self, "_mr_server_proc", None)
        if _srv_proc is not None and _srv_proc.poll() is None:
            try:
                import urllib.request as _ur3, json as _json3
                _ping2 = _json3.loads(
                    _ur3.urlopen(f"http://localhost:{port}/ping", timeout=2).read()
                )
                _serving_pid = _ping2.get("pid")
                if _serving_pid and _serving_pid != _srv_proc.pid:
                    print(f"{_LOG} WARNING: owned pid={_srv_proc.pid} but "
                          f"serving pid={_serving_pid} — click Stop Server "
                          f"then Start Server to resync.")
            except Exception:
                pass

        # ── Resolve mesh path ─────────────────────────────────────────────
        # Use cached path if available; otherwise auto-export from modelNode.
        mesh = meshPath or getattr(self, "_blender_mesh_path", None)
        if not mesh:
            mesh = self._mr_export_mesh_stl()
            if mesh:
                self._blender_mesh_path = mesh  # cache for repeat calls

        return self._runMultiRayViaServer(
            meshPath=mesh, port=port, cfg_overrides=cfg_overrides
        )

    def cancelMultiRay(self):
        """Signal a running runMultiRayBlender call to abort.

        Safe to call from any thread.  The polling loop in
        _runMultiRayViaServer checks this flag every 200 ms and will
        POST /abort to the server before returning False.
        """
        self._mr_cancel_flag = True
        self._mr_status("Cancelling\u2026 (finishing current server request)")
        print(f"{_LOG} cancelMultiRay: abort flag set")

    def _mr_export_mesh_stl(self):
        """Export self.modelNode (or the first visible vessel model) to a temp STL.

        Returns the path string on success, or None on failure.
        """
        if not _SLICER_OK:
            return None
        try:
            import tempfile, os as _os
            node = getattr(self, "modelNode", None)
            if node is None:
                # Try to find it by name
                scene = slicer.mrmlScene
                for i in range(scene.GetNumberOfNodesByClass("vtkMRMLModelNode")):
                    n = scene.GetNthNodeByClass(i, "vtkMRMLModelNode")
                    if n and n.GetPolyData() and n.GetPolyData().GetNumberOfPoints() > 0:
                        node = n
                        break
            if node is None:
                print(f"{_LOG} _mr_export_mesh_stl: no model node found")
                return None

            tmp = tempfile.mktemp(suffix=".stl", prefix="vessel_mr_")
            ok  = slicer.util.saveNode(node, tmp)
            if ok and _os.path.isfile(tmp):
                print(f"{_LOG} Exported mesh → {tmp} "
                      f"({_os.path.getsize(tmp)//1024} KB)")
                return tmp
            print(f"{_LOG} _mr_export_mesh_stl: export failed (ok={ok})")
            return None
        except Exception as exc:
            print(f"{_LOG} _mr_export_mesh_stl error: {exc}")
            return None

    # ── Server API path ───────────────────────────────────────────────────────────────

    def _runMultiRayViaServer(self, meshPath=None, port=6789, cfg_overrides=None):
        """POST branch data to the running multi-ray HTTP server and apply results.

        The BVH is cached inside the server, so only JSON round-trip overhead is
        incurred on repeat calls.  The mesh path is included in the payload so the
        server reloads the STL automatically if it has changed since the last call.

        Returns True on success, False on failure.
        """
        import urllib.request
        import urllib.error

        mesh = meshPath or getattr(self, "_blender_mesh_path", None)

        bm = getattr(self, "branchMeta", None)
        if not bm:
            self._mr_status("No centerline data. Run Extract Centerline first.")
            return False

        # ── Build per-branch point lists ──────────────────────────────────
        # _rawBranches (or self.branches) is a list of (startGi, endGi, ...)
        # tuples — NOT dicts. The coordinates live in self.points[gi].
        # branchMeta["pts"] is not reliably populated at runtime.
        raw_branches = getattr(self, "_rawBranches", None) or getattr(self, "branches", [])
        all_points   = getattr(self, "points", [])

        branches_payload = []

        if raw_branches and all_points:
            for bi, br in enumerate(raw_branches):
                try:
                    bs = int(br[0])   # startGi (inclusive)
                    be = int(br[1])   # endGi   (inclusive)
                    pts_ras = [list(all_points[gi]) for gi in range(bs, be + 1)
                               if gi < len(all_points)]
                    # Slicer centerline points are in RAS space; the STL is
                    # exported by saveNode() in LPS space (X and Y negated).
                    # Convert RAS → LPS so ray origins match the mesh geometry:
                    #   LPS = (-R, -A, S)
                    pts = [[-p[0], -p[1], p[2]] for p in pts_ras]
                    if pts:
                        branches_payload.append({"id": bi, "points": pts})
                except Exception as exc:
                    print(f"{_LOG} Branch {bi} skip: {exc}")
            total_pts = sum(len(b["points"]) for b in branches_payload)
            print(f"{_LOG} Point source: _rawBranches+points "
                  f"({len(branches_payload)} branches, {total_pts} total pts)")
        else:
            # Final fallback: branchMeta["pts"] (may be empty in some mixin versions)
            items = bm.items() if isinstance(bm, dict) else enumerate(bm)
            for bi, meta in items:
                pts = meta.get("pts", []) if isinstance(meta, dict) else []
                if pts:
                    branches_payload.append({
                        "id":     int(bi),
                        "points": [list(p) for p in pts],
                    })
            print(f"{_LOG} Point source: branchMeta[pts] (fallback)")

        if not branches_payload:
            has_raw = bool(raw_branches)
            has_pts = bool(all_points)
            self._mr_status(
                "No point data found. Run Extract Centerline + Analyse first."
            )
            self._mr_progress(0, visible=False)
            print(f"{_LOG} raw_branches={has_raw} ({len(raw_branches) if has_raw else 0} entries)  "
                  f"points={has_pts} ({len(all_points) if has_pts else 0} pts)")
            return False

        payload = dict(_DEFAULTS)
        if cfg_overrides:
            payload.update(cfg_overrides)
        payload["branches"] = branches_payload
        if mesh:
            payload["mesh"] = mesh   # server auto-reloads if path changed

        self._mr_status("Sending branch data to multi-ray server\u2026")
        self._mr_progress(10)      # 10 % — payload ready, about to send
        self._mr_flush()

        # ── Pre-flight: probe /load to verify the server is truly alive ──────
        # vessel_blender_multiray_server.py (Blender-based) supports /load.
        # vessel_multiray_server.py (pure-Python) does NOT — it returns 404.
        # Strategy:
        #   • 404  → pure-Python server; skip /load, mesh is sent inline with /multiray
        #   • 200  → Blender server; /load succeeded, continue
        #   • other error (connection refused, timeout) → server is not running; abort
        if mesh:
            try:
                load_body = json.dumps({"mesh": mesh}).encode()
                load_req  = urllib.request.Request(
                    f"http://localhost:{port}/load",
                    data    = load_body,
                    headers = {"Content-Type": "application/json"},
                    method  = "POST",
                )
                self._mr_status("Pre-flight: contacting server\u2026")
                self._mr_progress(15)
                self._mr_flush()
                load_resp = json.loads(
                    urllib.request.urlopen(load_req, timeout=10).read()
                )
                if not load_resp.get("ok"):
                    err = load_resp.get("error", "unknown error")
                    self._mr_status(f"Server rejected mesh: {err}")
                    self._mr_progress(0, visible=False)
                    print(f"{_LOG} /load rejected: {err}")
                    return False
                print(f"{_LOG} /load ok (Blender server) \u2014 "
                      f"verts={load_resp.get('verts')}  tris={load_resp.get('tris')}")
            except urllib.error.HTTPError as http_exc:
                if http_exc.code == 404:
                    # Pure-Python server — /load not implemented; that is fine.
                    # The mesh path is already embedded in the /multiray payload.
                    # Check /ping to see if trimesh is available so we can set
                    # an appropriate timeout for /multiray.
                    print(f"{_LOG} /load → 404 (pure-Python server); "
                          "mesh will be sent inline with /multiray")
                    try:
                        ping_resp = json.loads(
                            urllib.request.urlopen(
                                f"http://localhost:{port}/ping", timeout=3
                            ).read()
                        )
                        has_trimesh = ping_resp.get("trimesh", False)
                        has_numpy   = ping_resp.get("numpy",   False)
                        if has_trimesh:
                            _mr_timeout = 60
                            self._mr_status(
                                "Pre-flight: pure-Python server with trimesh ✓ "
                                "(timeout 60 s)"
                            )
                        elif has_numpy:
                            _mr_timeout = 300
                            self._mr_status(
                                "Pre-flight: pure-Python server with numpy AABB "
                                "(no trimesh — timeout 300 s, may be slow)"
                            )
                        else:
                            _mr_timeout = 600
                            self._mr_status(
                                "⚠ Pre-flight: pure-Python server — no trimesh, "
                                "no numpy. Performance will be very slow (timeout 600 s). "
                                "Install trimesh for best results: "
                                "pip install trimesh"
                            )
                        print(f"{_LOG} server caps: "
                              f"trimesh={has_trimesh} numpy={has_numpy} "
                              f"→ timeout={_mr_timeout}s")
                    except Exception:
                        _mr_timeout = 120  # safe default if ping fails
                else:
                    self._mr_status(
                        f"Server error on /load (HTTP {http_exc.code}): {http_exc}\n"
                        "Is the server running? Check the port and restart if needed."
                    )
                    self._mr_progress(0, visible=False)
                    print(f"{_LOG} /load pre-flight HTTP error: {http_exc}")
                    return False
            except Exception as exc:
                self._mr_status(
                    f"Server not responding: {exc}\n"
                    "Is the server running? Check the port and restart if needed."
                )
                self._mr_progress(0, visible=False)
                print(f"{_LOG} /load pre-flight error: {exc}")
                return False

        # ── Non-blocking /multiray dispatch ──────────────────────────────
        # urlopen blocks for up to _mr_timeout seconds (up to 600 s on slow
        # servers).  We run it in a daemon thread and poll every 200 ms so
        # the Slicer UI stays responsive and the user can cancel at any time.

        body = json.dumps(payload).encode()
        req  = urllib.request.Request(
            f"http://localhost:{port}/multiray",
            data    = body,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        self._mr_progress(25)  # 25 % — request dispatched, waiting for server
        self._mr_flush()

        # Timeout adapts to server capability (set during pre-flight):
        # trimesh=60s  numpy-AABB=300s  pure-Python=600s  fallback=120s
        _mr_timeout = locals().get("_mr_timeout", 120)

        # ── Debug: log the full dispatch context ──────────────────────────
        _body_kb = len(body) / 1024
        _n_branches = len(branches_payload)
        _n_pts_total = sum(len(b["points"]) for b in branches_payload)
        print(f"{_LOG} [DBG] dispatch /multiray  port={port}  "
              f"body={_body_kb:.1f}KB  timeout={_mr_timeout}s  "
              f"branches={_n_branches}  total_pts={_n_pts_total}")

        # ── Debug: check if the server process (if we own it) is alive ────
        _srv_proc = getattr(self, "_mr_server_proc", None)
        if _srv_proc is not None:
            _rc = _srv_proc.poll()
            if _rc is not None:
                print(f"{_LOG} [DBG] WARNING: owned server process has already "
                      f"exited (returncode={_rc}) — connection will fail")
            else:
                print(f"{_LOG} [DBG] owned server process is alive (pid={_srv_proc.pid})")
        else:
            print(f"{_LOG} [DBG] no owned server process — server started externally")

        # Reset cancel flag for this run
        self._mr_cancel_flag = False
        _result_box = [None]   # receives raw bytes or an Exception
        _t_start    = _time.time()

        def _do_request():
            try:
                print(f"{_LOG} [DBG] _do_request: calling urlopen …")
                resp = urllib.request.urlopen(req, timeout=_mr_timeout)
                _elapsed = _time.time() - _t_start
                print(f"{_LOG} [DBG] _do_request: response received in "
                      f"{_elapsed:.1f}s  status={resp.status}")
                _result_box[0] = resp.read()
                print(f"{_LOG} [DBG] _do_request: body read "
                      f"({len(_result_box[0])} bytes)")
            except Exception as exc:
                _elapsed = _time.time() - _t_start
                import traceback as _tb
                print(f"{_LOG} [DBG] _do_request: EXCEPTION after {_elapsed:.1f}s  "
                      f"type={type(exc).__name__}  msg={exc}")
                print(_tb.format_exc())
                _result_box[0] = exc

        t = threading.Thread(target=_do_request, daemon=True)
        t.start()

        _POLL_S   = 0.2
        _last_log = _time.time()
        _tick     = 0
        while t.is_alive():
            self._mr_flush()   # keep Qt event loop alive
            if getattr(self, "_mr_cancel_flag", False):
                # Tell server to stop between branches, then bail out
                try:
                    abort_req = urllib.request.Request(
                        f"http://localhost:{port}/abort",
                        data    = b"{}",
                        headers = {"Content-Type": "application/json"},
                        method  = "POST",
                    )
                    urllib.request.urlopen(abort_req, timeout=2)
                    print(f"{_LOG} /abort sent to server")
                except Exception:
                    pass  # server may not support /abort — that is fine
                self._mr_status("Cancelled.")
                self._mr_progress(0, visible=False)
                print(f"{_LOG} Run cancelled by user.")
                return False
            _time.sleep(_POLL_S)
            _tick += 1
            # Heartbeat every 10 s so we can see the run is alive in the log
            if _time.time() - _last_log >= 10.0:
                _elapsed = _time.time() - _t_start
                print(f"{_LOG} [DBG] still waiting … {_elapsed:.0f}s elapsed")
                _last_log = _time.time()

        raw = _result_box[0]
        if isinstance(raw, Exception):
            import traceback as _tb2
            _elapsed = _time.time() - _t_start
            print(f"{_LOG} [DBG] request thread finished with exception "
                  f"after {_elapsed:.1f}s  type={type(raw).__name__}")
            # Re-emit the ping to see if server is still alive after the failure
            try:
                _ping_r = urllib.request.urlopen(
                    f"http://localhost:{port}/ping", timeout=3
                )
                _ping_body = json.loads(_ping_r.read())
                print(f"{_LOG} [DBG] post-failure /ping: server STILL UP → "
                      f"{_ping_body}  (pipe-deadlock confirmed if no [ServerLog] lines above)")
            except Exception as _pe:
                print(f"{_LOG} [DBG] post-failure /ping: FAILED → {_pe}  "
                      f"(server died or port changed)")
            self._mr_status(f"Server request failed: {raw}")
            self._mr_progress(0, visible=False)
            print(f"{_LOG} _runMultiRayViaServer error: {raw}")
            return False

        self._mr_progress(80)  # 80 % — response received, parsing
        self._mr_flush()

        try:
            output = json.loads(raw)
        except Exception as exc:
            self._mr_status(f"Bad response from server: {exc}")
            self._mr_progress(0, visible=False)
            print(f"{_LOG} JSON parse error: {exc}")
            return False

        # Partial result from a server-side abort
        if output.get("aborted"):
            n_applied = self._multiray_apply_to_branch_meta(output)
            self._mr_status(
                f"\u26a0 Run aborted \u2014 partial results applied "
                f"to {n_applied} branch(es)."
            )
            self._mr_progress(0, visible=False)
            print(f"{_LOG} Server returned partial (aborted) result \u2014 "
                  f"{n_applied} branches updated.")
            return False

        self._mr_progress(90)      # 90 % — applying results to branchMeta
        self._mr_flush()
        n_applied = self._multiray_apply_to_branch_meta(output)
        self._mr_progress(100)     # 100 % — done
        self._mr_flush()
        self._mr_status(
            f"\u2713 Multi-ray radii applied to {n_applied} branch(es). "
            f"Confidence threshold = {CONFIDENCE_THRESHOLD:.2f}."
        )
        # Hide bar after a brief moment (next processEvents will repaint at 100 %)
        self._mr_progress(100, visible=False)
        print(f"{_LOG} \u2713 Done \u2014 {n_applied} branches updated.")
        return True

    def _mr_ping(self, port=6789):
        """Return True if the server is reachable at GET /ping.

        Side-effects: stores server pid in self._mr_server_pid and log file
        path in self._mr_server_log so Stop Server can kill by pid even when
        there is no subprocess handle.
        """
        try:
            import urllib.request
            r = urllib.request.urlopen(
                f"http://localhost:{port}/ping", timeout=2
            )
            ok = r.status == 200
            try:
                body = json.loads(r.read())
                print(f"{_LOG} [DBG] /ping → {body}")
                # Cache pid so we can kill an externally-started server
                if "pid" in body:
                    self._mr_server_pid = body["pid"]
                if "log_file" in body:
                    self._mr_server_log = body["log_file"]
            except Exception:
                pass
            return ok
        except Exception as exc:
            print(f"{_LOG} [DBG] /ping failed: {exc}")
            return False

    def _mr_kill_by_port(self, port):
        """Kill the server process listening on *port*, with fallbacks.

        Priority:
          1. Graceful POST /shutdown
          2. Kill by PID obtained from the last /ping (self._mr_server_pid)
          3. Kill by PID obtained from the subprocess handle (_mr_server_proc)
          4. psutil port scan (if psutil available)
        Returns True if a process was killed, False otherwise.
        """
        import urllib.request

        # 1. Graceful shutdown
        try:
            req = urllib.request.Request(
                f"http://localhost:{port}/shutdown",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=3)
            print(f"{_LOG} /shutdown accepted — server stopping gracefully")
            return True
        except Exception:
            pass  # server may already be unresponsive (pipe-deadlocked)

        # 2. Kill by cached PID from /ping
        pid = getattr(self, "_mr_server_pid", None)
        if pid:
            try:
                import signal as _sig, os as _os
                if sys.platform == "win32":
                    import ctypes
                    handle = ctypes.windll.kernel32.OpenProcess(1, False, pid)
                    if handle:
                        ctypes.windll.kernel32.TerminateProcess(handle, 1)
                        ctypes.windll.kernel32.CloseHandle(handle)
                        print(f"{_LOG} Killed server pid={pid} via TerminateProcess")
                        self._mr_server_pid = None
                        return True
                else:
                    _os.kill(pid, _sig.SIGTERM)
                    print(f"{_LOG} Killed server pid={pid} via SIGTERM")
                    self._mr_server_pid = None
                    return True
            except Exception as exc:
                print(f"{_LOG} Kill pid={pid} failed: {exc}")

        # 3. Kill via subprocess handle
        proc = getattr(self, "_mr_server_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
                print(f"{_LOG} Terminated owned server process pid={proc.pid}")
                self.logic._mr_server_proc = None
                return True
            except Exception as exc:
                print(f"{_LOG} terminate() failed: {exc}")

        # 4. psutil port scan (last resort)
        try:
            import psutil
            for conn in psutil.net_connections(kind="tcp"):
                if conn.laddr.port == port and conn.status == "LISTEN":
                    p = psutil.Process(conn.pid)
                    p.terminate()
                    print(f"{_LOG} psutil: killed pid={conn.pid} on port {port}")
                    return True
        except Exception:
            pass

        print(f"{_LOG} _mr_kill_by_port: could not kill server on port {port}")
        return False

    # ── Apply results to branchMeta ───────────────────────────────────────────

    def _multiray_apply_to_branch_meta(self, output):
        """Merge Blender output into branchMeta in-place.

        Adds / overwrites:
            meta["radius"]       — list[float]  multi-ray refined radii
            meta["confidence"]   — list[float]  per-point confidence scores
            meta["is_bifurcation"] — list[bool]   bifurcation zone flags

        The old "diam" scalar is updated to 2 × median(radius) so existing
        code that reads meta["diam"] continues to work correctly.

        Returns the number of branches updated.
        """
        bm = getattr(self, "branchMeta", None)
        if not bm:
            return 0

        radii_map  = output.get("radius_refined", {})
        conf_map   = output.get("confidence",     {})
        bif_map    = output.get("is_bifurcation", {})

        items = bm.items() if isinstance(bm, dict) else enumerate(bm)
        n_updated = 0
        for bi, meta in items:
            key = str(bi)
            if key not in radii_map:
                continue

            radii  = radii_map[key]
            confs  = conf_map.get(key, [1.0] * len(radii))
            bifs   = bif_map.get(key, [False] * len(radii))

            meta["radius"]        = radii
            meta["confidence"]    = confs
            meta["is_bifurcation"]= bifs

            # Derive a scalar diameter from reliable points only
            reliable = [r for r, c in zip(radii, confs) if c >= CONFIDENCE_THRESHOLD]
            if reliable:
                med_r = robust_radius(reliable)
                meta["diam"] = 2.0 * med_r   # keep existing API surface intact
            n_updated += 1

            # Log a brief per-branch summary
            n_pts    = len(radii)
            n_lowc   = sum(1 for c in confs if c < CONFIDENCE_THRESHOLD)
            n_bif    = sum(1 for b in bifs if b)
            role     = meta.get("role", "?")
            mean_r   = sum(radii) / len(radii) if radii else 0.0
            print(
                f"{_LOG}   bi={bi} [{role:12s}]  pts={n_pts}"
                f"  mean_r={mean_r:.2f}mm  diam={meta.get('diam',0):.2f}mm"
                f"  low_conf={n_lowc}  bif_zone_pts={n_bif}"
            )

        return n_updated

    # ── Remove multi-ray data from branchMeta ─────────────────────────────────

    def clearMultiRayData(self):
        """Strip multi-ray fields from branchMeta (revert to original radii)."""
        bm = getattr(self, "branchMeta", None)
        if not bm:
            return
        items = bm.items() if isinstance(bm, dict) else enumerate(bm)
        n = 0
        for _, meta in items:
            for key in ("radius", "confidence", "is_bifurcation"):
                meta.pop(key, None)
            n += 1
        self._mr_status(f"Multi-ray data cleared from {n} branch(es).")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _mr_status(self, msg):
        print(f"{_LOG} {msg}")
        try:
            w = getattr(slicer.modules, "VesselAnalyzerWidget", None) if _SLICER_OK else None
            if w and hasattr(w, "multirayStatusLabel"):
                w.multirayStatusLabel.setText(msg)
        except Exception:
            pass

    def _mr_progress(self, value, maximum=100, visible=True):
        """Update the multi-ray progress bar.

        Parameters
        ----------
        value   : int  — current progress value (0 .. maximum).
                         Pass value == maximum to show 100 % then hide after flush.
        maximum : int  — total steps (default 100 for percentage mode).
        visible : bool — show or hide the bar.
        """
        try:
            w = getattr(slicer.modules, "VesselAnalyzerWidget", None) if _SLICER_OK else None
            if w and hasattr(w, "multirayProgressBar"):
                bar = w.multirayProgressBar
                bar.setMaximum(maximum)
                bar.setValue(value)
                # Show completion percentage as "N / total" when using step counts
                if maximum != 100:
                    bar.setFormat(f"%v / {maximum}  (%p%)")
                else:
                    bar.setFormat("%p%")
                bar.setVisible(visible)
        except Exception:
            pass

    def _mr_flush(self):
        try:
            if _SLICER_OK:
                qt.QApplication.processEvents()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Slicer widget mixin
# ══════════════════════════════════════════════════════════════════════════════

# ── Module-level helpers (no inheritance dependency) ──────────────────────────

def _mr_server_script_path():
    """Absolute path to vessel_multiray_server.py (same dir as this file)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "vessel_multiray_server.py")


def _mr_set_dot(widget_self, running: bool):
    """Flip the status dot colour on *widget_self*: green = running, red = stopped."""
    dot = getattr(widget_self, "multirayStateDot", None)
    if dot:
        dot.setStyleSheet(
            "color: #27ae60; font-size: 14px;"
            if running else
            "color: #c0392b; font-size: 14px;"
        )


def _refine_set_dot(widget_self, running: bool):
    """Flip the refinement-server state dot: green = running, red = stopped."""
    dot = getattr(widget_self, "refineStateDot", None)
    if dot:
        dot.setStyleSheet(
            "color: #27ae60; font-size: 14px;"
            if running else
            "color: #c0392b; font-size: 14px;"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Slicer widget mixin
# ══════════════════════════════════════════════════════════════════════════════

class MultiRayBlenderWidgetMixin:
    """Mix into VesselAnalyzerWidget.

    Note: helpers are module-level functions (_mr_server_script_path,
    _mr_set_dot) so they work correctly whether this class is used via
    inheritance OR via the class-attribute binding pattern used in
    VesselAnalyzer.py (onBlenderServerStart = _MRW.onBlenderServerStart).
    """

    def onSendToBlender(self, _checked=False):
        """Export vessel mesh to the refinement server, load result, run pipeline.

        Flow
        ----
        1. Export current modelNode → temp STL
        2. POST raw STL bytes to ``<refineUrl>/refine``
           vessel_refinement_server.py returns refined STL bytes with
           Content-Type: application/octet-stream and X-Mesh-Format: stl.
        3. Write response bytes to a temp STL file.
        4. Load the file as a new vtkMRMLModelNode named "RefinedVesselModel".
        5. Set self.modelNode / self.logic.modelNode to the refined node.
        6. Call onExtractCenterline() to run the full downstream pipeline.

        The network wait runs on a daemon thread; a QTimer poll loop keeps the
        Slicer UI fully responsive while waiting for the server.
        """
        import tempfile, os as _os, threading as _thr

        lbl = getattr(self, "refineStatusLabel", None)
        bar = getattr(self, "refineProgressBar", None)

        def _set_status(msg, colour="#6c3483"):
            if lbl:
                lbl.setText(msg)
                lbl.setStyleSheet(
                    f"font-size: 11px; color: {colour}; font-style: italic;"
                )

        def _set_busy(busy):
            if bar:
                bar.setMaximum(0 if busy else 100)  # 0 = indeterminate spinner
                bar.setVisible(busy)
            btn = getattr(self, "sendToRefineButton", None)
            if btn:
                btn.setEnabled(not busy)

        # ── 1. Export mesh ────────────────────────────────────────────────────
        _set_status("Exporting mesh…")
        _set_busy(True)
        qt.QApplication.processEvents()

        # Resolve the model node from the Step 1 selector first.
        # self.logic._mr_export_mesh_stl() reads self.logic.modelNode, which is
        # never set by the selector (that only sets the widget's modelNode).
        # Read it from the selector directly and sync it to logic before export.
        selector = getattr(self, "modelSelector", None)
        selected_node = selector.currentNode() if selector else None
        if selected_node and selected_node.GetPolyData() and \
                selected_node.GetPolyData().GetNumberOfPoints() > 0:
            self.logic.modelNode = selected_node
        elif getattr(self, "modelNode", None):
            self.logic.modelNode = self.modelNode  # fallback: widget attribute

        stl_path = self.logic._mr_export_mesh_stl()
        if not stl_path:
            _set_status("❌ Export failed — no model node found.", "#c0392b")
            _set_busy(False)
            return

        import os as _os2
        stl_size = _os2.path.getsize(stl_path)
        if stl_size < 200:
            # A valid STL has at least an 80-byte header + 4-byte tri count.
            # 0 KB means the model node had no surface data at export time.
            # Common causes:
            #   • The active node is a Segmentation, not a Model — right-click
            #     the segment in Data module → Export to Model first.
            #   • The model node's PolyData is still being generated.
            node = getattr(self, "modelNode", None) or getattr(self.logic, "modelNode", None)
            node_name = node.GetName() if node else "unknown"
            node_pts  = (node.GetPolyData().GetNumberOfPoints()
                         if node and node.GetPolyData() else 0)
            msg = (f"❌ Export produced an empty STL ({stl_size} B). "
                   f"Node: '{node_name}' has {node_pts} pts. "
                   f"If this is a Segmentation, export it to a Model first "
                   f"(right-click segment → Export as Model).")
            _set_status(msg, "#c0392b")
            print(f"{_LOG} onSendToBlender: {msg}")
            _set_busy(False)
            return

        blender_url = "http://localhost:6790"
        url_widget = getattr(self, "refineApiUrlEdit", None)
        if url_widget:
            blender_url = url_widget.text.strip().rstrip("/")

        endpoint = f"{blender_url}/refine"

        # ── 2 & 3. POST to Blender on a background thread ────────────────────
        _result = {}   # shared dict: populated by worker, read by completion cb

        def _worker():
            try:
                import urllib.request as _ur
                with open(stl_path, "rb") as fh:
                    stl_bytes = fh.read()

                req = _ur.Request(
                    endpoint,
                    data=stl_bytes,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-Mesh-Format": "stl",
                    },
                    method="POST",
                )
                # Refinement can take a while — 5 min ceiling
                with _ur.urlopen(req, timeout=300) as resp:
                    fmt = resp.headers.get("X-Mesh-Format", "stl").lower()
                    steps = resp.headers.get("X-Refinement-Steps", "")
                    _result["bytes"] = resp.read()
                    _result["fmt"]   = fmt
                    _result["steps"] = steps
            except Exception as exc:
                _result["error"] = str(exc)

        def _on_done():
            """Called on the main thread once the worker finishes."""
            _set_busy(False)

            if "error" in _result:
                _set_status(f"❌ Server error: {_result['error']}", "#c0392b")
                print(f"{_LOG} onSendToBlender: {_result['error']}")
                return

            mesh_bytes = _result.get("bytes", b"")
            fmt        = _result.get("fmt", "stl")
            steps      = _result.get("steps", "")

            if not mesh_bytes:
                _set_status("❌ Server returned empty response.", "#c0392b")
                return

            # ── 3. Write refined mesh to temp file ────────────────────────
            suffix   = f".{fmt}"
            tmp_path = tempfile.mktemp(suffix=suffix, prefix="vessel_refined_")
            with open(tmp_path, "wb") as fh:
                fh.write(mesh_bytes)
            print(f"{_LOG} Refined mesh saved → {tmp_path} "
                  f"({len(mesh_bytes)//1024} KB, steps: {steps})")

            # ── 4. Load into Slicer ───────────────────────────────────────
            _set_status("Loading refined mesh…")
            qt.QApplication.processEvents()
            try:
                loaded = slicer.util.loadModel(tmp_path)
            except Exception as exc:
                _set_status(f"❌ Load failed: {exc}", "#c0392b")
                print(f"{_LOG} onSendToBlender load error: {exc}")
                return

            if not loaded:
                _set_status("❌ slicer.util.loadModel returned None.", "#c0392b")
                return

            loaded.SetName("RefinedVesselModel")

            # ── 5. Swap modelNode ─────────────────────────────────────────
            self.modelNode       = loaded
            self.logic.modelNode = loaded
            print(f"{_LOG} modelNode → RefinedVesselModel")

            # ── 6. Auto-detect endpoints on refined mesh ──────────────────
            # Try automatic endpoint detection first so the user gets an
            # immediate preview without having to place points manually.
            # If auto-detect fails they can still place endpoints manually
            # and click Extract Centerline themselves.
            _set_status("Detecting endpoints…")
            qt.QApplication.processEvents()
            try:
                self.onAutoDetect()
            except Exception as exc:
                _set_status(
                    "⚠ Auto-detect failed — place endpoints manually, "
                    "then click Extract Centerline.", "#e67e22"
                )
                print(f"{_LOG} onSendToBlender: auto-detect error: {exc}")
                print(f"{_LOG} onSendToBlender: pipeline complete (manual endpoints needed).")
                return

            # ── 7. Show preview centerline ────────────────────────────────
            # Extract centerline immediately so the user sees the result and
            # can decide to accept it or adjust endpoints before continuing.
            _set_status("Computing centerline preview…")
            qt.QApplication.processEvents()
            try:
                self.onExtractCenterline()
            except Exception as exc:
                _set_status(
                    "⚠ Centerline preview failed — adjust endpoints and "
                    "click Extract Centerline.", "#e67e22"
                )
                print(f"{_LOG} onSendToBlender centerline preview error: {exc}")
                import traceback as _tb; print(_tb.format_exc())
                return

            _set_status(
                "✓ Refined mesh loaded — review centerline preview, "
                "adjust endpoints if needed, then continue.", "#27ae60"
            )
            print(f"{_LOG} onSendToBlender: pipeline complete.")

        # Fire worker thread, schedule completion check via QTimer
        _thr.Thread(target=_worker, daemon=True).start()

        def _poll_worker():
            if _thr.active_count() > 0 and "bytes" not in _result and "error" not in _result:
                # Still running — update spinner label and re-arm
                dots = "." * (getattr(self, "_refine_dots", 0) % 4 + 1)
                self._refine_dots = getattr(self, "_refine_dots", 0) + 1
                _set_status(f"Waiting for Blender{dots}")
                qt.QTimer.singleShot(600, _poll_worker)
            else:
                _on_done()

        _set_status("Sending mesh to Blender…")
        qt.QTimer.singleShot(600, _poll_worker)
        # Keep strong reference so GC doesn't collect it
        self._refine_poll_cb = _poll_worker

    def onBlenderServerStart(self, _checked=False):
        """Launch vessel_multiray_server.py as a background Python process.

        Uses Slicer's own Python executable so all dependencies (numpy, trimesh
        if installed) are already on the path — no Blender required.

        Startup polling is done via QTimer so the UI stays fully responsive —
        no time.sleep() blocking the main thread.
        """
        import subprocess, sys as _sys, threading as _threading
        try:
            port = int(self.multirayPortSpin.value)
            lbl  = getattr(self, "multirayStatusLabel", None)

            # ── Kill any stale server on this port before launching ───────
            # A server from a previous session may still be listening.
            # If we skip killing it, the new process loses the port race and
            # sits idle while the stale process (with no drain thread) handles
            # requests — its stdout pipe fills, it deadlocks mid-computation,
            # and the client sees "Remote end closed connection without response".
            # Always kill first so the process we Popen is always the one that
            # serves requests, and the drain thread is therefore always effective.
            if self.logic._mr_ping(port):
                ping_resp = {}
                try:
                    import urllib.request as _ur2, json as _json2
                    ping_resp = _json2.loads(
                        _ur2.urlopen(f"http://localhost:{port}/ping", timeout=2).read()
                    )
                except Exception:
                    pass
                stale_pid = ping_resp.get("pid")
                owned_pid = getattr(self.logic, "_mr_server_proc", None)
                owned_pid = owned_pid.pid if owned_pid else None
                if stale_pid and stale_pid == owned_pid:
                    # We own this process and its drain thread is live — reuse it.
                    if lbl:
                        lbl.setText(f"Server already running on port {port}.")
                    _mr_set_dot(self, True)
                    return
                # Stale or unowned process — kill it so our new launch wins the port.
                print(f"{_LOG} Killing stale server pid={stale_pid} on port {port} "
                      f"(owned pid={owned_pid}) — restarting with fresh drain thread.")
                if lbl:
                    lbl.setText(f"Restarting server on port {port}\u2026")
                self.logic._mr_kill_by_port(port)
                import time as _tkill
                _tkill.sleep(1.0)   # wait for OS to release the port

            # ── Locate server script (same directory as this mixin) ───────
            script = _mr_server_script_path()
            if not os.path.isfile(script):
                msg = f"Server script not found:\n{script}"
                if lbl:
                    lbl.setText(msg)
                print(f"{_LOG} {msg}")
                return

            # ── Resolve Python executable ─────────────────────────────────
            python_exe = _sys.executable or "python"

            if lbl:
                lbl.setText(f"Starting server on port {port}\u2026")
            qt.QApplication.processEvents()

            proc = subprocess.Popen(
                [python_exe, script, str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.logic._mr_server_proc = proc

            # ── Drain server stdout in a background thread ────────────────
            # CRITICAL: without this the OS pipe buffer fills up (~64 KB on
            # Windows) once the server starts printing per-branch log lines.
            # When the buffer is full, every server-side print() call blocks,
            # the HTTP response is never written, and the client sees
            # "Remote end closed connection without response".
            # Draining in a daemon thread keeps the pipe empty and forwards
            # server logs to the Slicer Python console for diagnostics.
            _drain_active = [True]   # flag so _poll_tick knows not to read stdout

            def _drain_stdout(p):
                try:
                    for line in p.stdout:
                        print(f"[ServerLog] {line}", end="")
                except Exception:
                    pass   # process has exited — normal shutdown
                finally:
                    _drain_active[0] = False

            _threading.Thread(
                target=_drain_stdout, args=(proc,), daemon=True
            ).start()
            print(f"{_LOG} [DBG] stdout drain thread started (pid={proc.pid})")

            # ── Non-blocking poll via QTimer ──────────────────────────────
            # Fires every 500 ms — UI stays live between ticks.
            # Max 30 ticks = 15 s before giving up.
            _state = {"ticks": 0, "max": 30}

            def _poll_tick():
                _state["ticks"] += 1

                # Did the process die before becoming ready?
                if proc.poll() is not None:
                    # Do NOT read proc.stdout here — the drain thread owns it.
                    # Reading from two threads simultaneously corrupts the stream.
                    # The drain thread will have already printed the output.
                    out = ""
                    msg = (f"Server process exited (code {proc.returncode}). "
                           f"Check [ServerLog] lines above for server output.")
                    if lbl:
                        lbl.setText(msg)
                    _mr_set_dot(self, False)
                    _timer.stop()
                    print(f"{_LOG} {msg}")
                    return

                # Quick ping — 0.4 s timeout so the tick never blocks long
                try:
                    import urllib.request as _ur
                    r = _ur.urlopen(
                        f"http://localhost:{port}/ping", timeout=0.4
                    )
                    if r.status == 200:
                        self.logic._blender_server_port = port
                        dots = "\u2022" * _state["ticks"]
                        if lbl:
                            lbl.setText(
                                f"\u2713 Server ready on port {port}."
                            )
                        _mr_set_dot(self, True)
                        _timer.stop()
                        print(f"{_LOG} Server ready on port {port}.")
                        return
                except Exception:
                    # Not ready yet — show a progress dot in the status label
                    dots = "." * (_state["ticks"] % 4 + 1)
                    if lbl:
                        lbl.setText(f"Starting server on port {port}{dots}")

                # Timeout guard
                if _state["ticks"] >= _state["max"]:
                    proc.terminate()
                    if lbl:
                        lbl.setText(
                            f"Server did not respond within 15 s on port {port}. "
                            "Check the port is free and the script exists."
                        )
                    _mr_set_dot(self, False)
                    _timer.stop()
                    print(f"{_LOG} Server startup timed out after 15 s.")

            _timer = qt.QTimer()
            _timer.setInterval(500)
            _timer.setSingleShot(False)
            _timer.connect("timeout()", _poll_tick)
            _timer.start()
            # Hold strong references so Python GC doesn't collect either
            # the timer or the poll callback (PythonQt only weak-refs callables
            # passed to old-style connect, so _poll_tick must be kept alive).
            self._mr_start_timer    = _timer
            self._mr_poll_tick_cb   = _poll_tick

        except Exception as exc:
            print(f"{_LOG} onBlenderServerStart error: {exc}")
            print(traceback.format_exc())

    def onBlenderServerStop(self, _checked=False):
        """Stop the server — graceful /shutdown first, then force-kill by PID.

        Works even when the server was started externally (no subprocess handle)
        because _mr_ping() caches the server's PID from the /ping response.
        A pipe-deadlocked server won't respond to /shutdown, so the PID path
        is the critical fallback.
        """
        try:
            port = int(self.multirayPortSpin.value)
            lbl  = getattr(self, "multirayStatusLabel", None)

            killed = self.logic._mr_kill_by_port(port)

            self.logic._blender_server_port = None
            self.logic._mr_server_proc      = None
            if lbl:
                lbl.setText("Server stopped." if killed else
                            "Stop attempted — server may already be gone.")
            _mr_set_dot(self, False)
            print(f"{_LOG} Server stopped (killed={killed}).")

        except Exception as exc:
            print(f"{_LOG} onBlenderServerStop error: {exc}")
            print(traceback.format_exc())

    def onRefineServerStart(self, _checked=False):
        """Launch vessel_refinement_server.py as a background Python process.

        Mirrors onBlenderServerStart but targets vessel_refinement_server.py
        on its own port (default 6790) with its own status label / dot widget.
        On first run the server auto-installs open3d and pymeshfix (~1 min).
        Startup polling is done via QTimer so the UI stays fully responsive.
        """
        import subprocess, sys as _sys, threading as _threading
        try:
            port = int(self.refinePortSpin.value)
            lbl  = getattr(self, "refineStatusLabel", None)

            # ── Kill any stale server on this port before launching ───────
            if self.logic._mr_ping(port):
                ping_resp = {}
                try:
                    import urllib.request as _ur2, json as _json2
                    ping_resp = _json2.loads(
                        _ur2.urlopen(f"http://localhost:{port}/ping", timeout=2).read()
                    )
                except Exception:
                    pass
                stale_pid = ping_resp.get("pid")
                owned_pid = getattr(self.logic, "_refine_server_proc", None)
                owned_pid = owned_pid.pid if owned_pid else None
                if stale_pid and stale_pid == owned_pid:
                    if lbl:
                        lbl.setText(f"Server already running on port {port}.")
                    _refine_set_dot(self, True)
                    return
                print(f"{_LOG} Killing stale refinement server pid={stale_pid} "
                      f"on port {port} — restarting.")
                if lbl:
                    lbl.setText(f"Restarting server on port {port}\u2026")
                self.logic._mr_kill_by_port(port)
                import time as _tkill
                _tkill.sleep(1.0)

            # ── Locate server script ──────────────────────────────────────
            script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "vessel_refinement_server.py"
            )
            if not os.path.isfile(script):
                msg = f"Server script not found:\n{script}"
                if lbl:
                    lbl.setText(msg)
                print(f"{_LOG} {msg}")
                return

            python_exe = _sys.executable or "python"

            if lbl:
                lbl.setText(f"Starting server on port {port}\u2026")
            qt.QApplication.processEvents()

            proc = subprocess.Popen(
                [python_exe, script, str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.logic._refine_server_proc = proc

            # ── Drain stdout so the pipe never fills ──────────────────────
            def _drain_refine(p):
                try:
                    for line in p.stdout:
                        print(f"[RefineServerLog] {line}", end="")
                except Exception:
                    pass

            _threading.Thread(
                target=_drain_refine, args=(proc,), daemon=True
            ).start()
            print(f"{_LOG} Refinement server drain thread started (pid={proc.pid})")

            # ── Non-blocking QTimer poll ──────────────────────────────────
            # 60 ticks × 500 ms = 30 s — extra time for first-run dep install
            _state = {"ticks": 0, "max": 60}

            def _poll_refine():
                _state["ticks"] += 1
                if proc.poll() is not None:
                    msg = (f"Refinement server exited (code {proc.returncode}). "
                           "Check [RefineServerLog] lines above.")
                    if lbl:
                        lbl.setText(msg)
                    _refine_set_dot(self, False)
                    _timer.stop()
                    print(f"{_LOG} {msg}")
                    return
                try:
                    import urllib.request as _ur
                    r = _ur.urlopen(f"http://localhost:{port}/ping", timeout=0.4)
                    if r.status == 200:
                        if lbl:
                            lbl.setText(f"\u2713 Server ready on port {port}.")
                        _refine_set_dot(self, True)
                        _timer.stop()
                        print(f"{_LOG} Refinement server ready on port {port}.")
                        return
                except Exception:
                    dots = "." * (_state["ticks"] % 4 + 1)
                    if lbl:
                        lbl.setText(
                            f"Starting server on port {port}{dots}\n"
                            "(First run installs dependencies — may take ~1 min)"
                        )
                if _state["ticks"] >= _state["max"]:
                    proc.terminate()
                    if lbl:
                        lbl.setText(
                            f"Server did not respond within 30 s on port {port}. "
                            "Check the port is free and the script exists."
                        )
                    _refine_set_dot(self, False)
                    _timer.stop()
                    print(f"{_LOG} Refinement server startup timed out.")

            _timer = qt.QTimer()
            _timer.setInterval(500)
            _timer.setSingleShot(False)
            _timer.connect("timeout()", _poll_refine)
            _timer.start()
            self._refine_start_timer  = _timer
            self._refine_poll_tick_cb = _poll_refine

        except Exception as exc:
            print(f"{_LOG} onRefineServerStart error: {exc}")
            print(traceback.format_exc())

    def onRefineServerStop(self, _checked=False):
        """Stop the refinement server — graceful /shutdown then force-kill."""
        try:
            port = int(self.refinePortSpin.value)
            lbl  = getattr(self, "refineStatusLabel", None)

            killed = self.logic._mr_kill_by_port(port)

            self.logic._refine_server_proc = None
            if lbl:
                lbl.setText("Server stopped." if killed else
                            "Stop attempted — server may already be gone.")
            _refine_set_dot(self, False)
            print(f"{_LOG} Refinement server stopped (killed={killed}).")

        except Exception as exc:
            print(f"{_LOG} onRefineServerStop error: {exc}")
            print(traceback.format_exc())

    def onMultiRayBlenderMap(self, _checked=False):
        # Re-entrancy guard: processEvents() inside the poll loop can deliver
        # another button click while a run is already in-flight.
        if getattr(self.logic, "_mr_cancel_flag", None) is False:
            print(f"{_LOG} onMultiRayBlenderMap: re-entrant call ignored (run in-flight)")
            return
        try:
            port = int(self.multirayPortSpin.value)
            self.logic._blender_server_port = port

            lbl = getattr(self, "multirayStatusLabel", None)
            if lbl:
                lbl.setText(f"Connecting to server on port {port}\u2026")

            # Reset + show progress bar before the pipeline starts
            bar = getattr(self, "multirayProgressBar", None)
            if bar:
                bar.setValue(0)
                bar.setMaximum(100)
                bar.setFormat("%p%")
                bar.setVisible(True)

            qt.QApplication.processEvents()

            self.logic.runMultiRayBlender(port=port)

            # Refresh dot — server should still be up after a successful run
            _mr_set_dot(self, self.logic._mr_ping(port))

        except Exception as exc:
            print(f"{_LOG} onMultiRayBlenderMap error: {exc}")
            print(traceback.format_exc())

    def onMultiRayBlenderClear(self, _checked=False):
        try:
            # If a run is currently in-flight (_mr_cancel_flag is False, meaning
            # it was explicitly reset to False at dispatch time), cancel it.
            # Any other state (True = already cancelling, None/absent = idle)
            # falls through to a normal data-clear.
            if getattr(self.logic, "_mr_cancel_flag", None) is False:
                self.logic.cancelMultiRay()
                return
            self.logic.clearMultiRayData()
        except Exception as exc:
            print(f"{_LOG} onMultiRayBlenderClear error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: weighted radius smoothing (call from Slicer after apply)
# ══════════════════════════════════════════════════════════════════════════════

def confidence_weighted_smooth(radii, confidences):
    """Return a smoothed radius list using per-point confidence as weights.

    For each position i the smoothed value is the confidence-weighted mean
    of the immediate neighbourhood [i-1, i, i+1].  Points with
    confidence < CONFIDENCE_THRESHOLD are skipped entirely (radius = NaN in
    the output, caller should handle by interpolation or masking).

    This is the recommended post-processing step before feeding radii into
    the feature extraction / renal gating logic.

    Parameters
    ----------
    radii       : list[float]
    confidences : list[float]

    Returns
    -------
    smoothed : list[float | None]
        None marks a point where no reliable neighbours exist.
    """
    n = len(radii)
    smoothed = []
    for i in range(n):
        indices = [j for j in (i-1, i, i+1) if 0 <= j < n]
        w_sum = 0.0
        r_sum = 0.0
        for j in indices:
            c = confidences[j]
            if c >= CONFIDENCE_THRESHOLD:
                w_sum += c
                r_sum += c * radii[j]
        if w_sum < 1e-9:
            smoothed.append(None)    # no reliable data — caller interpolates
        else:
            smoothed.append(r_sum / w_sum)
    return smoothed


# ══════════════════════════════════════════════════════════════════════════════
# Slicer module stub (satisfies Slicer auto-scanner)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule as _SLM

    class vessel_blender_multiray_mixin(_SLM):
        """Slicer stub — this file is a mixin, not a standalone module."""
        def __init__(self, parent):
            super().__init__(parent)
            parent.title  = "vessel_blender_multiray_mixin"
            parent.hidden = True
except ImportError:
    pass
