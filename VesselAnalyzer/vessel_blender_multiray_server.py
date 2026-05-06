"""vessel_blender_multiray_server.py
Persistent Blender HTTP API server — multi-ray adaptive radius sampling.
Build tag: v1.0 (2026-04-27)

Run once from Blender:
    blender --background --python vessel_blender_multiray_server.py -- 6789

Then Slicer POSTs JSON to http://localhost:6789 for each analysis.
The BVH is rebuilt only when the mesh path changes — not on every request.

ENDPOINTS
─────────
POST /load        {"mesh": "/abs/path/vessel.stl"}
                  → {"ok": true, "verts": N, "tris": M}

POST /multiray    {same branches payload as subprocess contract}
                  → {"radius_refined":{...}, "confidence":{...},
                     "is_bifurcation":{...}}

POST /shutdown    {}
                  → {"ok": true}  then server exits

GET  /ping        → {"ok": true, "mesh_loaded": true/false}
"""

import json
import sys
import os

try:
    import bpy
    import bmesh
    from mathutils.bvhtree import BVHTree as _BVHTree
    _INSIDE_BLENDER = True
except ModuleNotFoundError:
    _INSIDE_BLENDER = False

if not _INSIDE_BLENDER:
    pass  # Slicer scanner — load cleanly, do nothing
else:
    import http.server
    import threading

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

    from vessel_blender_multiray_mixin import (
        _DEFAULTS,
        compute_branch_multiray,
        robust_radius,
        CONFIDENCE_THRESHOLD,
    )

    # ── Global BVH state ─────────────────────────────────────────────────────
    _state = {
        "bvh":        None,
        "mesh_path":  None,
        "verts":      0,
        "tris":       0,
    }

    # ── Mesh loader ──────────────────────────────────────────────────────────

    def _load_mesh(mesh_path):
        """Load STL into Blender and rebuild BVH.  No-op if path unchanged."""
        if _state["mesh_path"] == mesh_path and _state["bvh"] is not None:
            print(f"[BlenderAPI] Mesh already loaded: {mesh_path}")
            return True

        if not os.path.isfile(mesh_path):
            print(f"[BlenderAPI] ERROR: mesh not found: {mesh_path}")
            return False

        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
        bpy.ops.import_mesh.stl(filepath=mesh_path)
        obj = bpy.context.object

        if obj is None:
            print("[BlenderAPI] ERROR: STL import returned no active object")
            return False

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)
        _state["bvh"]       = _BVHTree.FromBMesh(bm)
        _state["mesh_path"] = mesh_path
        _state["verts"]     = len(obj.data.vertices)
        _state["tris"]      = len(obj.data.polygons)
        bm.free()

        print(f"[BlenderAPI] Mesh loaded: '{obj.name}'"
              f"  verts={_state['verts']}  tris={_state['tris']}")
        return True

    # ── Request handler ──────────────────────────────────────────────────────

    class _Handler(http.server.BaseHTTPRequestHandler):

        def log_message(self, fmt, *args):
            # Route to Python print so it appears in Blender's console
            print(f"[BlenderAPI] {self.address_string()} {fmt % args}")

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0))
            raw    = self.rfile.read(length) if length else b"{}"
            return json.loads(raw)

        def _send_json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type",   "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, msg, code=400):
            self._send_json(code, {"ok": False, "error": msg})

        # ── GET /ping ────────────────────────────────────────────────────────
        def do_GET(self):
            if self.path == "/ping":
                self._send_json(200, {
                    "ok":          True,
                    "mesh_loaded": _state["bvh"] is not None,
                    "mesh_path":   _state["mesh_path"],
                })
            else:
                self._send_error("Unknown endpoint", 404)

        # ── POST ─────────────────────────────────────────────────────────────
        def do_POST(self):
            try:
                payload = self._read_json()
            except Exception as exc:
                self._send_error(f"Bad JSON: {exc}")
                return

            if self.path == "/load":
                self._handle_load(payload)
            elif self.path == "/multiray":
                self._handle_multiray(payload)
            elif self.path == "/shutdown":
                self._send_json(200, {"ok": True})
                # Shut down server from a thread so the response is sent first
                threading.Thread(target=_server.shutdown, daemon=True).start()
            else:
                self._send_error("Unknown endpoint", 404)

        def _handle_load(self, payload):
            mesh = payload.get("mesh", "")
            if not mesh:
                self._send_error("'mesh' field required")
                return
            ok = _load_mesh(mesh)
            if ok:
                self._send_json(200, {
                    "ok":    True,
                    "verts": _state["verts"],
                    "tris":  _state["tris"],
                })
            else:
                self._send_error(f"Failed to load mesh: {mesh}", 500)

        def _handle_multiray(self, payload):
            # Auto-load mesh if provided and not yet loaded
            mesh = payload.get("mesh")
            if mesh and mesh != _state["mesh_path"]:
                ok = _load_mesh(mesh)
                if not ok:
                    self._send_error(f"Failed to load mesh: {mesh}", 500)
                    return

            if _state["bvh"] is None:
                self._send_error(
                    "No mesh loaded. POST /load first or include 'mesh' in payload."
                )
                return

            branches = payload.get("branches", [])
            if not branches:
                self._send_error("'branches' field required")
                return

            cfg = dict(_DEFAULTS)
            for k in ("n_rays_normal", "n_rays_bifurcation", "angle_thresh",
                      "tangent_window", "max_dist", "min_hits"):
                if k in payload:
                    cfg[k] = payload[k]

            radii_out = {}
            conf_out  = {}
            bif_out   = {}

            for branch in branches:
                bid    = branch["id"]
                points = branch["points"]

                if not points:
                    radii_out[str(bid)] = []
                    conf_out [str(bid)] = []
                    bif_out  [str(bid)] = []
                    continue

                radii, confs, bifs = compute_branch_multiray(
                    _state["bvh"], points, cfg
                )
                radii_out[str(bid)] = radii
                conf_out [str(bid)] = confs
                bif_out  [str(bid)] = bifs

                n_low  = sum(1 for c in confs if c < CONFIDENCE_THRESHOLD)
                n_bif  = sum(1 for b in bifs  if b)
                mean_r = sum(radii) / len(radii) if radii else 0.0
                print(
                    f"[BlenderAPI]   bid={bid}  pts={len(points)}"
                    f"  mean_r={mean_r:.2f}mm"
                    f"  low_conf={n_low}  bif_pts={n_bif}"
                )

            self._send_json(200, {
                "radius_refined": radii_out,
                "confidence":     conf_out,
                "is_bifurcation": bif_out,
            })

    # ── Start server ─────────────────────────────────────────────────────────

    port = 6789
    if len(sys.argv) >= 2:
        try:
            port = int(sys.argv[-1])
        except ValueError:
            pass

    _server = http.server.HTTPServer(("localhost", port), _Handler)
    print(f"[BlenderAPI] Listening on http://localhost:{port}")
    print("[BlenderAPI] Endpoints: GET /ping  POST /load /multiray /shutdown")
    _server.serve_forever()


# ── Slicer module-scanner stub ────────────────────────────────────────────────
try:
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule as _SLM

    class vessel_blender_multiray_server(_SLM):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title  = "vessel_blender_multiray_server"
            parent.hidden = True
except ImportError:
    class vessel_blender_multiray_server:
        def __init__(self, parent=None):
            pass
