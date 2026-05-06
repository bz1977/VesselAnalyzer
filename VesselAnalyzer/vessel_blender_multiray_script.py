"""vessel_blender_multiray_script.py
Headless Blender script — multi-ray adaptive radius sampling.
Build tag: v1.1 (2026-04-27)

Usage (called by VesselAnalyzerLogic.runMultiRayBlender):
    blender --background --python vessel_blender_multiray_script.py \\
            -- /path/to/input.json /path/to/output.json

Input / output contracts are documented in vessel_blender_multiray_mixin.py.
"""

import json
import sys
import os

# ── Blender-only imports — guarded so Slicer's module scanner doesn't crash ──
# Slicer auto-scans every .py in the module directory.  Without this guard,
# `import bpy` raises ModuleNotFoundError and Slicer refuses to load the module.
# When launched by Blender the imports succeed and _INSIDE_BLENDER = True.
try:
    import bpy
    import bmesh
    from mathutils.bvhtree import BVHTree as _BVHTree
    _INSIDE_BLENDER = True
except ModuleNotFoundError:
    _INSIDE_BLENDER = False

if not _INSIDE_BLENDER:
    # Slicer scanned this file — nothing to do, load cleanly and exit.
    pass
else:
    # ── Everything below only runs when launched by Blender ──────────────────

    # Add the directory containing this script to sys.path so we can import
    # the pure-Python geometry helpers from vessel_blender_multiray_mixin.py.
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

    from vessel_blender_multiray_mixin import (
        _DEFAULTS,
        compute_branch_multiray,
        robust_radius,
        CONFIDENCE_THRESHOLD,
    )

    # ── Parse CLI arguments ──────────────────────────────────────────────────
    # Blender forwards everything after "--" as sys.argv.
    # Convention: sys.argv[-2] = input.json   sys.argv[-1] = output.json
    if len(sys.argv) < 2:
        print("[MultiRayScript] ERROR: expected <input.json> <output.json> after --")
        sys.exit(1)

    input_path  = sys.argv[-2]
    output_path = sys.argv[-1]

    # ── Load config ──────────────────────────────────────────────────────────
    with open(input_path) as _f:
        cfg = json.load(_f)

    mesh_path = cfg.get("mesh", "")
    branches  = cfg.get("branches", [])

    if not os.path.isfile(mesh_path):
        print(f"[MultiRayScript] ERROR: mesh file not found: {mesh_path}")
        sys.exit(2)

    if not branches:
        print("[MultiRayScript] ERROR: no branches in input JSON")
        sys.exit(3)

    print(f"[MultiRayScript] Mesh:     {mesh_path}")
    print(f"[MultiRayScript] Branches: {len(branches)}")
    print(
        f"[MultiRayScript] Config:   "
        f"n_normal={cfg.get('n_rays_normal',      _DEFAULTS['n_rays_normal'])}"
        f"  n_bif={cfg.get('n_rays_bifurcation',  _DEFAULTS['n_rays_bifurcation'])}"
        f"  max_dist={cfg.get('max_dist',          _DEFAULTS['max_dist'])}"
    )

    # ── Load mesh and build BVH ──────────────────────────────────────────────
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    bpy.ops.import_mesh.stl(filepath=mesh_path)
    obj = bpy.context.object

    if obj is None:
        print("[MultiRayScript] ERROR: STL import returned no active object")
        sys.exit(4)

    print(
        f"[MultiRayScript] Loaded: '{obj.name}'"
        f"  verts={len(obj.data.vertices)}"
        f"  tris={len(obj.data.polygons)}"
    )

    bm_mesh = bmesh.new()
    bm_mesh.from_mesh(obj.data)
    bm_mesh.transform(obj.matrix_world)
    bvh = _BVHTree.FromBMesh(bm_mesh)
    bm_mesh.free()

    print("[MultiRayScript] BVH built — starting ray sampling…")

    # ── Process branches ─────────────────────────────────────────────────────
    radii_out = {}
    conf_out  = {}
    bif_out   = {}

    for branch in branches:
        bid    = branch["id"]
        points = branch["points"]
        n_pts  = len(points)

        if n_pts == 0:
            radii_out[str(bid)] = []
            conf_out [str(bid)] = []
            bif_out  [str(bid)] = []
            continue

        radii, confs, bifs = compute_branch_multiray(bvh, points, cfg)

        radii_out[str(bid)] = radii
        conf_out [str(bid)] = confs
        bif_out  [str(bid)] = bifs

        n_low_conf  = sum(1 for c in confs if c < CONFIDENCE_THRESHOLD)
        n_bif_zones = sum(1 for b in bifs if b)
        mean_r   = sum(radii) / len(radii) if radii else 0.0
        reliable = [r for r, c in zip(radii, confs) if c >= CONFIDENCE_THRESHOLD]
        med_r    = robust_radius(reliable) if reliable else 0.0

        print(
            f"[MultiRayScript]   bid={bid:3d}  pts={n_pts:4d}"
            f"  mean_r={mean_r:5.2f}mm  med_r={med_r:5.2f}mm"
            f"  low_conf={n_low_conf}  bif_pts={n_bif_zones}"
        )

    # ── Write output ─────────────────────────────────────────────────────────
    output = {
        "radius_refined": radii_out,
        "confidence":     conf_out,
        "is_bifurcation": bif_out,
    }

    with open(output_path, "w") as _f:
        json.dump(output, _f)

    print(f"[MultiRayScript] Output written: {output_path}")
    print("[MultiRayScript] ✓ Done.")


# ── Slicer module-scanner stub ───────────────────────────────────────────────
# Slicer requires a class matching the filename in every .py it scans.
# This stub satisfies that requirement without registering as a real module.
try:
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule as _SLM

    class vessel_blender_multiray_script(_SLM):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title  = "vessel_blender_multiray_script"
            parent.hidden = True
except ImportError:
    class vessel_blender_multiray_script:
        def __init__(self, parent=None):
            pass
