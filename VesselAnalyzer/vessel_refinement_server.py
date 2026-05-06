"""vessel_refinement_server.py
Local mesh refinement server for VesselAnalyzer.

Receives a raw STL from Slicer, cleans and smooths it for better centerline
extraction and radius computation, returns a refined STL.

Design goals
------------
• Truth-preserving: Taubin smoothing (not Laplacian) to avoid volume shrinkage
  and stenosis erasure.
• Repair first, smooth second: pymeshfix fills holes and removes self-
  intersections before any smoothing touches the surface.
• Conservative defaults: enough to remove segmentation staircase artefacts
  without destroying real anatomy.

Start once per session (Slicer "Start Refinement Server" button, or manually):
    python vessel_refinement_server.py [port]      # default 6790

HTTP endpoints
--------------
POST /refine    – main call; raw STL in, refined STL out
POST /shutdown  – graceful shutdown
POST /abort     – cancel any in-progress refinement
GET  /ping      – health-check → 200 {"ok": true, "pid": <pid>}

Request  : raw STL bytes, Content-Type: application/octet-stream
Response : raw STL bytes, Content-Type: application/octet-stream
           X-Mesh-Format: stl
           X-Refinement-Steps: <comma-separated list of steps applied>

Dependencies (pip install …)
----------------------------
Required:
    numpy          – always available in Slicer's Python
    open3d         – mesh I/O, Taubin smoothing, normal repair, resampling
Optional (strongly recommended):
    pymeshfix      – watertight repair; skipped with a warning if absent

Both packages must be installed into the same Python that runs this script
(i.e. Slicer's Python, NOT a system Python):
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "open3d", "pymeshfix"])
"""

from __future__ import annotations

import http.server
import io
import json
import os
import struct
import sys
import tempfile
import threading
import time

_LOG = "[RefinementServer]"

# ── Stdout pipe-deadlock prevention (same pattern as multiray server) ──────────
def _redirect_print_to_logfile(port: int) -> str:
    """Redirect stdout+stderr to a per-port log file and return the path."""
    log_dir  = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, f"vessel_refinement_server_{port}.log")
    _fh = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _fh
    sys.stderr = _fh
    print(f"{_LOG} stdout/stderr redirected to {log_path}  "
          f"(pipe-deadlock prevention active)")
    return log_path


# ── Dependency bootstrap ───────────────────────────────────────────────────────
# Auto-installs missing packages into the running Python (Slicer's Python or
# a plain venv) so the user never has to run pip manually.

def _ensure_deps():
    """Install open3d and pymeshfix if not already importable.

    Uses --break-system-packages so it works inside Slicer's managed Python
    on Windows without needing a venv or admin rights.
    Uses --quiet to suppress pip progress bars that would flood the log.
    """
    needed = []
    try:
        import open3d  # noqa: F401
    except ImportError:
        needed.append("open3d")
    try:
        import pymeshfix  # noqa: F401
    except ImportError:
        needed.append("pymeshfix")

    if not needed:
        return   # nothing to do

    print(f"{_LOG} Auto-installing missing packages: {needed}")
    import subprocess
    for pkg in needed:
        print(f"{_LOG} pip install {pkg} …")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg,
             "--quiet", "--break-system-packages"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"{_LOG} {pkg} installed OK.")
        else:
            # pip may print the real error on stderr
            print(f"{_LOG} WARNING: pip install {pkg} failed "
                  f"(code {result.returncode}):\n{result.stderr.strip()}")


# ── Dependencies ───────────────────────────────────────────────────────────────
# Imports happen AFTER _ensure_deps() is called inside run() so that a freshly
# installed package is visible in the same process without a restart.
np        = None;  _NP  = False
o3d       = None;  _O3D = False
pymeshfix = None;  _PMF = False


def _import_deps():
    """Import dependencies after ensuring they are installed."""
    global np, _NP, o3d, _O3D, pymeshfix, _PMF
    try:
        import numpy as _np
        np  = _np;  _NP = True
    except ImportError:
        print(f"{_LOG} ERROR: numpy still not importable after install attempt.")

    try:
        import open3d as _o3d
        o3d  = _o3d;  _O3D = True
    except ImportError:
        print(f"{_LOG} ERROR: open3d still not importable after install attempt.")

    try:
        import pymeshfix as _pmf
        pymeshfix  = _pmf;  _PMF = True
    except ImportError:
        print(f"{_LOG} INFO: pymeshfix not importable — hole-filling will be skipped.")


# ── JSON encoder (matches multiray server) ────────────────────────────────────
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if _NP:
            if isinstance(obj, np.floating):  return float(obj)
            if isinstance(obj, np.integer):   return int(obj)
            if isinstance(obj, np.bool_):     return bool(obj)
            if isinstance(obj, np.ndarray):   return obj.tolist()
        try:
            return float(obj)
        except (TypeError, ValueError):
            pass
        return super().default(obj)


# ── STL I/O (stdlib only — no open3d needed for raw byte read/write) ──────────

def _stl_bytes_to_o3d(stl_bytes: bytes) -> "o3d.geometry.TriangleMesh":
    """Parse raw STL bytes → open3d TriangleMesh without touching disk."""
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        f.write(stl_bytes)
        tmp = f.name
    try:
        mesh = o3d.io.read_triangle_mesh(tmp)
    finally:
        os.unlink(tmp)
    return mesh


def _o3d_to_stl_bytes(mesh: "o3d.geometry.TriangleMesh") -> bytes:
    """Serialize open3d TriangleMesh → raw binary STL bytes."""
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    try:
        o3d.io.write_triangle_mesh(tmp, mesh, write_ascii=False,
                                   write_vertex_normals=False)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


# ── Refinement pipeline ────────────────────────────────────────────────────────

# Tuning knobs — conservative defaults that preserve lesion geometry.
# Slicer can override these by sending JSON config via X-Refine-Config header
# (see _parse_config below).
_DEFAULTS = {
    # Taubin smoothing — λ/μ scheme, volume-preserving
    "taubin_iters":     5,       # number of λ+μ iteration pairs
    "taubin_lambda":    0.5,     # positive pass weight  (0 < λ < 1)
    "taubin_mu":       -0.53,    # negative pass weight  (μ < -λ)
    # Pymeshfix
    "repair":           True,    # run pymeshfix watertight repair
    # Normals
    "recalc_normals":   True,
    # Resampling — set to None to skip
    "target_triangles": None,    # e.g. 80000 to decimate; None = keep original
}


def _parse_config(headers) -> dict:
    """Merge per-request config from X-Refine-Config JSON header."""
    cfg = dict(_DEFAULTS)
    raw = headers.get("X-Refine-Config", "")
    if raw:
        try:
            overrides = json.loads(raw)
            cfg.update({k: v for k, v in overrides.items() if k in _DEFAULTS})
        except Exception:
            pass
    return cfg


def refine(stl_bytes: bytes, cfg: dict) -> tuple[bytes, list[str]]:
    """
    Run the refinement pipeline.

    Returns
    -------
    (refined_stl_bytes, steps_applied)
        steps_applied is a list of strings describing what was done,
        forwarded to Slicer as the X-Refinement-Steps response header.
    """
    if not _O3D:
        raise RuntimeError("open3d is not installed — cannot refine mesh.")

    steps: list[str] = []
    t0 = time.time()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    mesh = _stl_bytes_to_o3d(stl_bytes)
    n_verts_in = len(mesh.vertices)
    n_tris_in  = len(mesh.triangles)
    print(f"{_LOG} Loaded: {n_verts_in} verts, {n_tris_in} tris")
    steps.append(f"load({n_verts_in}v,{n_tris_in}t)")

    # ── 2. Remove degenerate triangles ───────────────────────────────────────
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    after_clean = len(mesh.triangles)
    removed = n_tris_in - after_clean
    if removed:
        print(f"{_LOG} Removed {removed} degenerate/duplicate tris")
    steps.append(f"clean(removed={removed})")

    # ── 3. Pymeshfix watertight repair ────────────────────────────────────────
    if cfg["repair"] and _PMF:
        try:
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            tin   = pymeshfix.PyTMesh()
            tin.load_array(verts, faces)
            tin.fill_small_boundaries()
            tin.clean(max_iters=10, inner_loops=3)
            v_out, f_out = tin.return_arrays()
            mesh.vertices  = o3d.utility.Vector3dVector(v_out)
            mesh.triangles = o3d.utility.Vector3iVector(f_out)
            print(f"{_LOG} pymeshfix: {len(v_out)} verts, {len(f_out)} tris")
            steps.append(f"repair(pymeshfix,{len(f_out)}t)")
        except Exception as exc:
            print(f"{_LOG} pymeshfix failed (skipping): {exc}")
            steps.append("repair(skipped)")
    elif cfg["repair"] and not _PMF:
        print(f"{_LOG} pymeshfix not installed — skipping repair")
        steps.append("repair(skipped,not_installed)")

    # ── 4. Recalculate normals ────────────────────────────────────────────────
    if cfg["recalc_normals"]:
        mesh.compute_vertex_normals()
        steps.append("normals")

    # ── 5. Taubin smoothing (volume-preserving) ───────────────────────────────
    # Taubin alternates a positive (shrink) pass and a negative (inflate) pass.
    # Net volume change ≈ 0, so stenoses and lesion geometry are preserved.
    # Laplacian-only would shrink the mesh and blur small features.
    n_iters = int(cfg["taubin_iters"])
    if n_iters > 0:
        lam = float(cfg["taubin_lambda"])
        mu  = float(cfg["taubin_mu"])
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=n_iters,
            lambda_filter=lam,
            mu=mu,
        )
        mesh.compute_vertex_normals()
        print(f"{_LOG} Taubin smooth: {n_iters} iters, λ={lam}, μ={mu}")
        steps.append(f"taubin(iters={n_iters},lambda={lam},mu={mu})")

    # ── 6. Optional decimation ────────────────────────────────────────────────
    target = cfg.get("target_triangles")
    if target and len(mesh.triangles) > target:
        ratio = target / len(mesh.triangles)
        mesh  = mesh.simplify_quadric_decimation(int(target))
        mesh.compute_vertex_normals()
        print(f"{_LOG} Decimated to {len(mesh.triangles)} tris "
              f"(ratio={ratio:.2f})")
        steps.append(f"decimate(target={target})")

    # ── 7. Final normal recalc + orient ──────────────────────────────────────
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    steps.append("orient")

    n_verts_out = len(mesh.vertices)
    n_tris_out  = len(mesh.triangles)
    elapsed     = time.time() - t0
    print(f"{_LOG} Done in {elapsed:.2f}s: "
          f"{n_verts_out} verts, {n_tris_out} tris")
    steps.append(f"output({n_verts_out}v,{n_tris_out}t,{elapsed:.1f}s)")

    return _o3d_to_stl_bytes(mesh), steps


# ── HTTP handler ───────────────────────────────────────────────────────────────

class _Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass   # suppress per-request noise; we print explicitly

    def _send_bytes(self, code: int, body: bytes,
                    content_type="application/octet-stream",
                    extra_headers: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj, cls=_NumpyEncoder).encode()
        self._send_bytes(code, body, "application/json")

    # ── GET /ping ─────────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == "/ping":
            srv = self.server
            self._send_json(200, {
                "ok":      True,
                "pid":     os.getpid(),
                "open3d":  _O3D,
                "pymeshfix": _PMF,
                "busy":    getattr(srv, "_busy", False),
            })
        else:
            self._send_json(404, {"error": "not found"})

    # ── POST ──────────────────────────────────────────────────────────────────
    def do_POST(self):
        srv = self.server

        # /abort — signal any in-progress refinement to stop early
        if self.path == "/abort":
            srv._abort_flag = True
            self._send_json(200, {"ok": True, "message": "abort signalled"})
            print(f"{_LOG} /abort received")
            return

        # /shutdown — graceful stop
        if self.path == "/shutdown":
            self._send_json(200, {"ok": True})
            print(f"{_LOG} /shutdown received — stopping.")
            threading.Thread(
                target=srv.shutdown, daemon=True
            ).start()
            return

        # /refine — main endpoint
        if self.path != "/refine":
            self._send_json(404, {"error": "not found"})
            return

        # Re-entrancy guard: only one refinement at a time
        if getattr(srv, "_busy", False):
            self._send_json(503, {
                "error": "server busy — another refinement is in progress"
            })
            return

        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json(400, {"error": "empty body — send raw STL bytes"})
            return

        stl_bytes = self.rfile.read(length)
        print(f"{_LOG} /refine: received {len(stl_bytes)//1024} KB STL")

        # Parse per-request config from optional header
        cfg = _parse_config(self.headers)

        srv._busy       = True
        srv._abort_flag = False
        try:
            refined_bytes, steps = refine(stl_bytes, cfg)
            if srv._abort_flag:
                self._send_json(200, {
                    "aborted": True,
                    "message": "refinement aborted before completion",
                })
                print(f"{_LOG} Aborted.")
                return
            self._send_bytes(
                200, refined_bytes,
                extra_headers={
                    "X-Mesh-Format":       "stl",
                    "X-Refinement-Steps":  ",".join(steps),
                },
            )
            print(f"{_LOG} Response sent: {len(refined_bytes)//1024} KB")
        except Exception as exc:
            import traceback
            msg = f"{exc}\n{traceback.format_exc()}"
            print(f"{_LOG} ERROR: {msg}")
            self._send_json(500, {"error": str(exc), "detail": msg})
        finally:
            srv._busy       = False
            srv._abort_flag = False


# ── Server class ───────────────────────────────────────────────────────────────

class _Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    _busy       = False
    _abort_flag = False

    def server_bind(self):
        import socket
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        super().server_bind()


# ── Entry point ────────────────────────────────────────────────────────────────

def run(port=6790):
    log_path = _redirect_print_to_logfile(port)
    print(f"{_LOG} Log file: {log_path}")

    # Auto-install then import — must happen before anything uses o3d/pymeshfix
    _ensure_deps()
    _import_deps()

    o3d_ver  = getattr(o3d,  "__version__", "n/a") if _O3D else "NOT INSTALLED"
    pmf_ver  = getattr(pymeshfix, "__version__", "installed") if _PMF else "NOT INSTALLED"
    print(f"{_LOG} open3d:    {o3d_ver}")
    print(f"{_LOG} pymeshfix: {pmf_ver}")

    if not _O3D:
        print(f"{_LOG} FATAL: open3d could not be installed or imported. "
              f"Check network access and pip output above.")
        sys.exit(1)

    srv = _Server(("127.0.0.1", port), _Handler)
    srv._log_path = log_path
    print(f"{_LOG} Listening on port {port}  "
          f"(open3d={'yes' if _O3D else 'no'}  "
          f"pymeshfix={'yes' if _PMF else 'no'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"{_LOG} Stopped.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 6790
    run(port)


# ── Slicer module-scanner stub ────────────────────────────────────────────────
class vessel_refinement_server:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title  = "vessel_refinement_server"
            parent.hidden = True
