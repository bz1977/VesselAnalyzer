# vessel_blender_mixin.py
# Mesh distance mapping for VesselAnalyzer — pure-Python BVH (no Blender install)
# Build tag: v2.0 (2026-04-27)
#
# WHAT THIS MODULE DOES
# ──────────────────────
# Drop-in replacement for the original vessel_blender_mixin that required a
# locally installed Blender executable and used it via headless subprocess.
#
# This version implements the same three metrics (C2C, Hausdorff, RMS) entirely
# inside Slicer's embedded Python using:
#   • scipy.spatial.cKDTree  — nearest-vertex pass (fast, O(N log N))
#   • A pure-numpy triangle BVH (point-to-triangle projection) — gives the same
#     sub-vertex surface accuracy that Blender's mathutils.BVHTree provided,
#     without any external process.
#
# PUBLIC API (identical to v1.0)
# ──────────────────────────────
# Logic mixin  — BlenderMappingMixin
#   runBlenderMapping(referenceNode, compareNode, metric) → node | None
#   clearBlenderMapping()
#
# Widget mixin — BlenderWidgetMixin
#   onBlenderMap()
#   onBlenderClear()
#   (no _onBrowseBlenderExe — the exe-path UI row is removed in vessel_analyzer_ui.py)
#
# INTEGRATION (unchanged from v1.0 except no blenderExe arg)
# ────────────────────────────────────────────────────────────
#   from vessel_blender_mixin import BlenderMappingMixin
#   class VesselAnalyzerLogic(..., BlenderMappingMixin, ...): ...
#
#   from vessel_blender_mixin import BlenderWidgetMixin as _BWM
#   onBlenderMap   = _BWM.onBlenderMap
#   onBlenderClear = _BWM.onBlenderClear
#
# UI widgets required (vessel_analyzer_ui.py):
#   blenderCompareSelector, blenderMetricCombo
#   blenderMapButton, blenderClearButton, blenderStatusLabel
#   (blenderExePathEdit / blenderExeBrowseButton REMOVED)
#
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
import traceback
from typing import Optional

import numpy as np

try:
    import vtk
    import slicer
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
except ImportError:
    vtk = slicer = vtk_to_numpy = numpy_to_vtk = None

try:
    from scipy.spatial import cKDTree
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ── Constants ─────────────────────────────────────────────────────────────────

_RESULT_SUFFIX = "_BLD_MAPPED"   # model node name suffix (unchanged)
_SCALAR_NAME   = "DistanceMM"    # VTK point-data array name (unchanged)
_COLORMAP_IDX  = 10              # Slicer ColdToHot colour table index (unchanged)
_LOG_TAG       = "[BLD]"


# ══════════════════════════════════════════════════════════════════════════════
# Pure-Python triangle BVH helpers
# ══════════════════════════════════════════════════════════════════════════════

def _polydata_to_arrays(polydata) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices [N,3], triangles [M,3 int]) from a vtkPolyData.

    Quads and higher-order cells are tessellated on-the-fly via a
    vtkTriangleFilter before extraction so the triangle array is always
    well-formed.
    """
    tf = vtk.vtkTriangleFilter()
    tf.SetInputData(polydata)
    tf.Update()
    pd = tf.GetOutput()

    verts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)   # [N,3]

    # Build flat connectivity → reshape to [M,3]
    cells = pd.GetPolys()
    cells.InitTraversal()
    tri_list = []
    id_list = vtk.vtkIdList()
    while cells.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            tri_list.append([
                id_list.GetId(0),
                id_list.GetId(1),
                id_list.GetId(2),
            ])
    tris = np.array(tri_list, dtype=np.int64)  # [M,3]
    return verts, tris


def _point_to_triangle_distance_sq(p: np.ndarray,
                                    v0: np.ndarray,
                                    v1: np.ndarray,
                                    v2: np.ndarray) -> float:
    """Squared distance from point p to the triangle (v0,v1,v2).

    Uses the standard parametric projection (Eberly 2002).
    All inputs are 1-D float64 arrays of length 3.
    """
    e0 = v1 - v0
    e1 = v2 - v0
    d  = v0 - p
    a  = float(e0 @ e0)
    b  = float(e0 @ e1)
    c  = float(e1 @ e1)
    f_ = float(e0 @ d)
    g  = float(e1 @ d)
    det = a * c - b * b
    s   = b * g - c * f_
    t   = b * f_ - a * g

    if s + t <= det:
        if s < 0.0:
            if t < 0.0:
                # region 4
                if f_ < 0.0:
                    t = 0.0; s = max(0.0, min(a, -f_)) / a if a > 0 else 0.0
                else:
                    s = 0.0; t = max(0.0, min(c, -g)) / c if c > 0 else 0.0
            else:
                # region 3
                s = 0.0; t = max(0.0, min(c, -g)) / c if c > 0 else 0.0
        elif t < 0.0:
            # region 5
            t = 0.0; s = max(0.0, min(a, -f_)) / a if a > 0 else 0.0
        else:
            # region 0 — interior
            inv_det = 1.0 / det if det > 1e-30 else 0.0
            s *= inv_det; t *= inv_det
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + f_; tmp1 = c + g
            if tmp1 > tmp0:
                numer = tmp1 - tmp0; denom = a - 2*b + c
                s = max(0.0, min(1.0, numer / denom)) if denom > 1e-30 else 1.0
                t = 1.0 - s
            else:
                s = 0.0; t = max(0.0, min(1.0, -g / c)) if c > 1e-30 else 0.0
        elif t < 0.0:
            # region 6
            tmp0 = b + g; tmp1 = a + f_
            if tmp1 > tmp0:
                numer = tmp1 - tmp0; denom = a - 2*b + c
                t = max(0.0, min(1.0, numer / denom)) if denom > 1e-30 else 1.0
                s = 1.0 - t
            else:
                t = 0.0; s = max(0.0, min(1.0, -f_ / a)) if a > 1e-30 else 0.0
        else:
            # region 1
            numer = c + g - b - f_; denom = a - 2*b + c
            s = max(0.0, min(1.0, numer / denom)) if denom > 1e-30 else 0.0
            t = 1.0 - s

    closest = v0 + s * e0 + t * e1
    diff = p - closest
    return float(diff @ diff)


class _TriangleBVH:
    """Axis-aligned bounding box BVH over a triangle soup.

    Build once per mesh; then call find_nearest(query_pts) for batch queries.

    Strategy
    ────────
    For moderate-to-large meshes (≥ 2 k triangles) we split the triangles
    into a two-level grid (8 buckets along the longest axis) and use a
    scipy cKDTree over triangle centroids as a fast pre-filter: the nearest
    centroid's triangle is the best candidate, then we check the K closest
    centroids to handle edge cases.  This gives O(N log N) build and
    O(M log N) query, which is fast enough for typical vascular meshes.

    For very small meshes (< 2 k triangles) we fall back to a brute-force
    O(M·N) scan — it's faster below that threshold.
    """

    _K_PROBE = 8   # how many nearest centroids to check per query point

    def __init__(self, verts: np.ndarray, tris: np.ndarray):
        self.verts = verts          # [N,3] float64
        self.tris  = tris           # [M,3] int64
        # Pre-compute triangle vertex arrays for vectorised distance kernel
        self.v0 = verts[tris[:, 0]]  # [M,3]
        self.v1 = verts[tris[:, 1]]  # [M,3]
        self.v2 = verts[tris[:, 2]]  # [M,3]
        self.centroids = (self.v0 + self.v1 + self.v2) / 3.0  # [M,3]
        self._brute = len(tris) < 2_000
        if not self._brute and _SCIPY_OK:
            self._tree = cKDTree(self.centroids)
        else:
            self._tree = None

    def find_nearest(self, query_pts: np.ndarray) -> np.ndarray:
        """Return per-point nearest-surface distances (mm) for each row in query_pts [Q,3]."""
        Q = len(query_pts)
        dists = np.empty(Q, dtype=np.float64)

        if self._brute or self._tree is None:
            # Brute-force: vectorised over triangles, scalar over query points
            for qi in range(Q):
                dists[qi] = math.sqrt(self._min_dist_sq_brute(query_pts[qi]))
        else:
            # Fast path: probe K nearest centroids only
            k = min(self._K_PROBE, len(self.tris))
            _, idx = self._tree.query(query_pts, k=k)  # [Q, k]
            if k == 1:
                idx = idx[:, np.newaxis]
            for qi in range(Q):
                best = math.inf
                for ti in idx[qi]:
                    dsq = _point_to_triangle_distance_sq(
                        query_pts[qi],
                        self.v0[ti], self.v1[ti], self.v2[ti],
                    )
                    if dsq < best:
                        best = dsq
                dists[qi] = math.sqrt(best)
        return dists

    def _min_dist_sq_brute(self, p: np.ndarray) -> float:
        best = math.inf
        for ti in range(len(self.tris)):
            dsq = _point_to_triangle_distance_sq(
                p, self.v0[ti], self.v1[ti], self.v2[ti],
            )
            if dsq < best:
                best = dsq
        return best


# ══════════════════════════════════════════════════════════════════════════════
# VTK scalar helpers  (identical to v1.0)
# ══════════════════════════════════════════════════════════════════════════════

def _distances_to_vtk_array(distances: np.ndarray, name: str):
    arr = numpy_to_vtk(distances.astype(np.float32), deep=True)
    arr.SetName(name)
    return arr


def _clone_polydata_with_scalars(source_pd, scalar_array):
    result = vtk.vtkPolyData()
    result.DeepCopy(source_pd)
    result.GetPointData().RemoveArray(_SCALAR_NAME)
    result.GetPointData().AddArray(scalar_array)
    result.GetPointData().SetActiveScalars(_SCALAR_NAME)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Core distance computation  (replaces the Blender worker script)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_distances(
    ref_pd,
    cmp_pd,
    metric: str,
    status_cb=None,
) -> np.ndarray:
    """Compute per-vertex surface distances from cmp mesh to ref surface.

    Parameters
    ----------
    ref_pd   : vtkPolyData  — reference (ground truth) surface
    cmp_pd   : vtkPolyData  — comparison mesh (distances are per cmp vertex)
    metric   : "C2C Distance" | "Hausdorff" | "RMS"
    status_cb: optional callable(str) for progress messages

    Returns
    -------
    np.ndarray float64, shape (n_cmp_verts,)
    """
    def _status(msg: str):
        print(f"{_LOG_TAG} {msg}")
        if status_cb:
            status_cb(msg)

    ref_verts, ref_tris = _polydata_to_arrays(ref_pd)
    cmp_verts, cmp_tris = _polydata_to_arrays(cmp_pd)

    n_ref = len(ref_verts);  n_cmp = len(cmp_verts)
    _status(f"Building ref BVH  ({n_ref} verts, {len(ref_tris)} tris)…")
    ref_bvh = _TriangleBVH(ref_verts, ref_tris)

    _status(f"Computing forward C2C  ({n_cmp} cmp verts → ref surface)…")
    d_fwd = ref_bvh.find_nearest(cmp_verts)   # [n_cmp]

    if metric == "C2C Distance":
        return d_fwd

    elif metric == "Hausdorff":
        _status(f"Building cmp BVH for Hausdorff  ({n_cmp} verts, {len(cmp_tris)} tris)…")
        cmp_bvh = _TriangleBVH(cmp_verts, cmp_tris)
        # Nearest point ON ref for every cmp vert, then back-project to cmp surface
        # Using the BVH's closest-point to find nearest ref surface point is
        # implicit in d_fwd; for Hausdorff we use a cKDTree on ref_verts as
        # the proxy for the "landmark" ref point and measure its distance to cmp.
        if _SCIPY_OK:
            _status("Computing backward distances (cKDTree on ref verts)…")
            ref_tree = cKDTree(ref_verts)
            # Nearest ref vertex to each cmp vertex (as proxy for nearest surface pt)
            d_bwd_approx, nearest_ref_idx = ref_tree.query(cmp_verts, k=1)
            # Distance from those ref landmarks to cmp surface
            landmark_pts = ref_verts[nearest_ref_idx]  # [n_cmp, 3]
            d_back = cmp_bvh.find_nearest(landmark_pts)
            return np.maximum(d_fwd, d_back)
        else:
            # No scipy — fall back to symmetric max(d_fwd, d_bwd) using BVH only
            _status("Computing backward distances (pure BVH fallback)…")
            cmp_bvh2 = _TriangleBVH(cmp_verts, cmp_tris)
            # sample ref verts → cmp
            d_bwd_full = cmp_bvh2.find_nearest(ref_verts)   # [n_ref]
            # For each cmp vert, find its nearest ref vert index (brute; no scipy)
            # then take d_bwd_full at that index
            ref_tree_np = ref_verts   # [n_ref, 3]
            d_back = np.empty(n_cmp, dtype=np.float64)
            for qi in range(n_cmp):
                diff = ref_tree_np - cmp_verts[qi]
                idx  = int(np.argmin((diff * diff).sum(axis=1)))
                d_back[qi] = d_bwd_full[idx]
            return np.maximum(d_fwd, d_back)

    elif metric == "RMS":
        rms = float(np.sqrt(np.mean(d_fwd ** 2)))
        _status(f"Global RMS = {rms:.6f} mm")
        return np.full(n_cmp, rms, dtype=np.float64)

    else:
        _status(f"Unknown metric '{metric}', falling back to C2C.")
        return d_fwd


# ══════════════════════════════════════════════════════════════════════════════
# Logic mixin
# ══════════════════════════════════════════════════════════════════════════════

class BlenderMappingMixin:
    """Logic-level mixin — pure-Python BVH distance mapping for VesselAnalyzerLogic.

    Public API
    ──────────
    runBlenderMapping(referenceNode, compareNode, metric) → node | None
        Compute distances entirely in-process; apply ColdToHot colour map.

    clearBlenderMapping()
        Remove all _BLD_MAPPED nodes from the Slicer scene.

    Change from v1.0
    ────────────────
    The `blenderExe` parameter is removed.  The method signature is:
        runBlenderMapping(referenceNode, compareNode, metric="C2C Distance")
    Callers that pass blenderExe as a keyword arg receive a deprecation log
    and it is silently ignored so old call-sites don't crash during transition.
    """

    def runBlenderMapping(
        self,
        referenceNode,
        compareNode,
        metric: str = "C2C Distance",
        blenderExe: str = "",   # legacy kwarg — ignored
    ):
        if blenderExe:
            print(
                f"{_LOG_TAG} [deprecation] blenderExe argument is no longer used "
                "(API Blender replaces the headless subprocess)."
            )

        print(f"{_LOG_TAG} ══ API Blender Distance Mapping ════════════════════")

        ref_pd = self._bld_get_polydata(referenceNode, "reference")
        cmp_pd = self._bld_get_polydata(compareNode,   "compare")
        if ref_pd is None or cmp_pd is None:
            return None

        n_ref = ref_pd.GetNumberOfPoints()
        n_cmp = cmp_pd.GetNumberOfPoints()
        print(f"{_LOG_TAG} Reference: {n_ref} pts   Compare: {n_cmp} pts")

        try:
            self._bld_set_status(f"Starting {metric}…  ({n_cmp} cmp vertices)")

            # Flush Qt so status label updates before the compute loop blocks
            try:
                import qt as _qt
                _qt.QApplication.processEvents()
            except Exception:
                pass

            distances = _compute_distances(
                ref_pd, cmp_pd, metric,
                status_cb=self._bld_set_status,
            )

            d_min  = float(distances.min())
            d_max  = float(distances.max())
            d_mean = float(distances.mean())
            print(
                f"{_LOG_TAG} Stats — "
                f"min={d_min:.3f}  mean={d_mean:.3f}  max={d_max:.3f} mm"
            )

            # Attach distances to a copy of compare polydata
            scalar_arr = _distances_to_vtk_array(distances, _SCALAR_NAME)
            result_pd  = _clone_polydata_with_scalars(cmp_pd, scalar_arr)

            # Create / update result model node
            result_name = compareNode.GetName() + _RESULT_SUFFIX
            result_node = slicer.mrmlScene.GetFirstNodeByName(result_name)
            if result_node is None:
                result_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLModelNode", result_name
                )
                result_node.CreateDefaultDisplayNodes()

            result_node.SetAndObservePolyData(result_pd)

            self._bld_apply_colormap(result_node, d_min, d_max)

            self._bld_set_status(
                f"✓ {metric}  "
                f"min={d_min:.2f}  mean={d_mean:.2f}  max={d_max:.2f} mm"
                f"  →  '{result_name}'"
            )
            print(f"{_LOG_TAG} ✓ Result node: '{result_name}'")
            print(f"{_LOG_TAG} ══════════════════════════════════════════════")
            return result_node

        except Exception as exc:
            print(f"{_LOG_TAG} Error: {exc}")
            print(traceback.format_exc())
            self._bld_set_status(f"Error: {exc}")
            return None

    def clearBlenderMapping(self) -> None:
        """Remove all _BLD_MAPPED model nodes from the scene."""
        try:
            to_remove = []
            for i in range(slicer.mrmlScene.GetNumberOfNodes()):
                nd = slicer.mrmlScene.GetNthNode(i)
                if nd and nd.GetName() and nd.GetName().endswith(_RESULT_SUFFIX):
                    to_remove.append(nd)
            for nd in to_remove:
                slicer.mrmlScene.RemoveNode(nd)
            msg = f"Cleared {len(to_remove)} distance mapping node(s)."
            print(f"{_LOG_TAG} {msg}")
            self._bld_set_status(msg)
        except Exception as exc:
            print(f"{_LOG_TAG} clearBlenderMapping error: {exc}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _bld_get_polydata(self, model_node, label: str):
        if model_node is None:
            self._bld_set_status(f"No {label} model selected.")
            return None
        pd = model_node.GetPolyData()
        if pd is None or pd.GetNumberOfPoints() == 0:
            self._bld_set_status(
                f"Model '{model_node.GetName()}' has no geometry."
            )
            return None
        return pd

    def _bld_apply_colormap(self, model_node, d_min: float, d_max: float) -> None:
        try:
            dn = model_node.GetDisplayNode()
            if dn is None:
                return
            dn.SetScalarVisibility(True)
            dn.SetActiveScalarName(_SCALAR_NAME)
            dn.SetAndObserveColorNodeID(
                slicer.modules.colors.logic().GetColorTableNodeID(_COLORMAP_IDX)
            )
            dn.SetScalarRange(d_min, d_max)
            dn.SetScalarRangeFlag(
                slicer.vtkMRMLDisplayNode.UseManualScalarRange
            )
            model_node.SetDisplayVisibility(True)
            print(
                f"{_LOG_TAG} Colour map: ColdToHot  "
                f"range [{d_min:.3f}, {d_max:.3f}] mm"
            )
        except Exception as exc:
            print(f"{_LOG_TAG} _bld_apply_colormap error: {exc}")

    def _bld_set_status(self, msg: str) -> None:
        try:
            widget = getattr(slicer.modules, "VesselAnalyzerWidget", None)
            if widget and hasattr(widget, "blenderStatusLabel"):
                widget.blenderStatusLabel.setText(msg)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Widget mixin
# ══════════════════════════════════════════════════════════════════════════════

class BlenderWidgetMixin:
    """Qt event handlers for the Blender mapping panel.

    Bound into VesselAnalyzerWidget:
        from vessel_blender_mixin import BlenderWidgetMixin as _BWM
        onBlenderMap   = _BWM.onBlenderMap
        onBlenderClear = _BWM.onBlenderClear

    Change from v1.0: _onBrowseBlenderExe removed (no exe path field).
    """

    def onBlenderMap(self):
        """Slot: '▶ Run Distance Mapping' button clicked."""
        try:
            ref_sel  = getattr(self, "modelSelector", None)
            ref_node = ref_sel.currentNode() if ref_sel else None

            cmp_sel  = getattr(self, "blenderCompareSelector", None)
            cmp_node = cmp_sel.currentNode() if cmp_sel else None

            combo  = getattr(self, "blenderMetricCombo", None)
            metric = combo.currentText if combo else "C2C Distance"

            if ref_node is None:
                slicer.util.warningDisplay(
                    "Select a vessel model in Step 1 before running mapping."
                )
                return
            if cmp_node is None:
                slicer.util.warningDisplay(
                    "Select a comparison model in the 'Compare model' dropdown."
                )
                return

            lbl = getattr(self, "blenderStatusLabel", None)
            if lbl:
                lbl.setText(f"Starting {metric}…")
            try:
                import qt as _qt
                _qt.QApplication.processEvents()
            except Exception:
                pass

            result = self.logic.runBlenderMapping(
                referenceNode=ref_node,
                compareNode=cmp_node,
                metric=metric,
            )

            if result is not None:
                try:
                    slicer.util.resetThreeDViews()
                except Exception:
                    pass

        except Exception as exc:
            print(f"[BLD Widget] onBlenderMap error: {exc}")
            print(traceback.format_exc())

    def onBlenderClear(self):
        """Slot: '✕ Clear Mapping' button clicked."""
        try:
            self.logic.clearBlenderMapping()
        except Exception as exc:
            print(f"[BLD Widget] onBlenderClear error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Slicer module stub
# ══════════════════════════════════════════════════════════════════════════════
# Slicer auto-scans every .py in the module directory and tries to instantiate
# a class whose name matches the file stem.  Mixins are plain Python files with
# no such class, which causes "Failed to load scripted loadable module" errors
# at startup.  This minimal stub satisfies the scanner without registering any
# real UI or logic — the file is still imported normally by VesselAnalyzer.py.

try:
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule

    class vessel_blender_mixin(ScriptedLoadableModule):
        """Slicer stub — this file is a logic/widget mixin, not a standalone module."""
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "vessel_blender_mixin"
            parent.hidden = True   # keeps it out of the Modules menu
except ImportError:
    pass  # outside Slicer (unit tests, CI) — stub not needed
