# vessel_surface_map_mixin.py
# Single-model surface scalar mapping for VesselAnalyzer
# Build tag: v1.0 (2026-04-27)
#
# FEATURES
# ────────
# 1. Radius Map
#    Assigns each vertex of the vessel surface a scalar equal to its distance
#    to the nearest centerline point.  This is the local vessel radius at that
#    vertex.  Requires: vessel surface model (Step 1) + completed centerline
#    pipeline (branchMeta populated).
#
# 2. Wall Thickness Map
#    Assigns each vertex of the outer-wall surface a scalar equal to its
#    distance to the nearest point on the inner lumen surface.
#    Requires: two model nodes — outer wall + inner lumen.
#
# Both features use the same _TriangleBVH / cKDTree stack already present in
# vessel_blender_mixin.py, avoiding any new dependencies.
#
# PUBLIC API
# ──────────
# Logic mixin  — SurfaceMapMixin
#   runRadiusMap(modelNode)              → result node | None
#   runWallThicknessMap(wallNode, lumenNode) → result node | None
#   clearSurfaceMaps()
#
# Widget mixin — SurfaceMapWidgetMixin
#   onRadiusMap()
#   onWallThicknessMap()
#   onSurfaceMapClear()
#
# UI widgets required (added by build_surface_map_panel in vessel_analyzer_ui.py):
#   smapWallSelector, smapLumenSelector
#   smapRadiusButton, smapWallButton, smapClearButton
#   smapStatusLabel
#
# INTEGRATION
# ───────────
# In VesselAnalyzer.py widget class:
#   from vessel_surface_map_mixin import SurfaceMapWidgetMixin as _SMW
#   onRadiusMap       = _SMW.onRadiusMap
#   onWallThicknessMap= _SMW.onWallThicknessMap
#   onSurfaceMapClear = _SMW.onSurfaceMapClear
#
# In VesselAnalyzer.py logic imports + MRO:
#   from vessel_surface_map_mixin import SurfaceMapMixin
#   class VesselAnalyzerLogic(..., SurfaceMapMixin, ...): ...
#
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
import traceback

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

_RADIUS_SCALAR    = "RadiusMM"
_THICKNESS_SCALAR = "WallThicknessMM"
_RADIUS_SUFFIX    = "_RADIUS_MAP"
_THICKNESS_SUFFIX = "_WALL_THICK"
_COLORMAP_IDX     = 10              # ColdToHot
_LOG_TAG          = "[SMAP]"


# ══════════════════════════════════════════════════════════════════════════════
# VTK / numpy helpers
# ══════════════════════════════════════════════════════════════════════════════

def _surface_verts(polydata) -> np.ndarray:
    """Return [N,3] float64 array of surface vertex positions."""
    return vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)


def _attach_scalar(polydata, values: np.ndarray, name: str):
    """Deep-copy polydata, attach scalar array, return new vtkPolyData."""
    result = vtk.vtkPolyData()
    result.DeepCopy(polydata)
    arr = numpy_to_vtk(values.astype(np.float32), deep=True)
    arr.SetName(name)
    result.GetPointData().RemoveArray(name)
    result.GetPointData().AddArray(arr)
    result.GetPointData().SetActiveScalars(name)
    return result


def _apply_colormap(model_node, scalar_name: str, d_min: float, d_max: float):
    dn = model_node.GetDisplayNode()
    if dn is None:
        return
    dn.SetScalarVisibility(True)
    dn.SetActiveScalarName(scalar_name)
    dn.SetColorModeToScalarData()
    dn.SetAndObserveColorNodeID(
        slicer.modules.colors.logic().GetColorTableNodeID(_COLORMAP_IDX)
    )
    dn.SetScalarRange(d_min, d_max)
    dn.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
    model_node.SetDisplayVisibility(True)


def _get_or_create_node(name: str):
    node = slicer.mrmlScene.GetFirstNodeByName(name)
    if node is None:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.CreateDefaultDisplayNodes()
    return node


# ══════════════════════════════════════════════════════════════════════════════
# Centerline point cloud from branchMeta
# ══════════════════════════════════════════════════════════════════════════════

def _centerline_points(logic) -> np.ndarray | None:
    """Collect all centerline pts from branchMeta into a single [K,3] array."""
    bm = getattr(logic, "branchMeta", None)
    if not bm:
        return None

    all_pts = []
    items = bm.items() if isinstance(bm, dict) else enumerate(bm)
    for _, meta in items:
        pts = meta.get("pts", [])
        for p in pts:
            all_pts.append(p)

    if not all_pts:
        return None
    return np.array(all_pts, dtype=np.float64)  # [K, 3]


# ══════════════════════════════════════════════════════════════════════════════
# Core computations
# ══════════════════════════════════════════════════════════════════════════════

def _compute_radius_map(surface_verts: np.ndarray,
                        centerline_pts: np.ndarray) -> np.ndarray:
    """Per-surface-vertex distance to nearest centerline point."""
    if _SCIPY_OK:
        tree = cKDTree(centerline_pts)
        dists, _ = tree.query(surface_verts, k=1)
        return dists.astype(np.float64)
    else:
        # Brute-force fallback (no scipy)
        dists = np.empty(len(surface_verts), dtype=np.float64)
        for i, v in enumerate(surface_verts):
            diff = centerline_pts - v
            dists[i] = math.sqrt(float(np.min((diff * diff).sum(axis=1))))
        return dists


def _compute_wall_thickness(wall_verts: np.ndarray,
                            lumen_verts: np.ndarray) -> np.ndarray:
    """Per-wall-vertex distance to nearest lumen surface vertex.

    Uses the lumen vertex cloud as the reference — equivalent to the
    nearest-point C2C distance from the BVH mixin but wall → lumen.
    For typical lumen meshes the vertex spacing is fine enough that
    vertex-to-vertex distance is a good proxy for wall thickness.
    For higher accuracy the full triangle BVH from vessel_blender_mixin
    can be substituted here without changing the API.
    """
    if _SCIPY_OK:
        tree = cKDTree(lumen_verts)
        dists, _ = tree.query(wall_verts, k=1)
        return dists.astype(np.float64)
    else:
        dists = np.empty(len(wall_verts), dtype=np.float64)
        for i, v in enumerate(wall_verts):
            diff = lumen_verts - v
            dists[i] = math.sqrt(float(np.min((diff * diff).sum(axis=1))))
        return dists


# ══════════════════════════════════════════════════════════════════════════════
# Logic mixin
# ══════════════════════════════════════════════════════════════════════════════

class SurfaceMapMixin:

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _smap_get_pd(self, node, label: str):
        if node is None:
            self._smap_status(f"No {label} model selected.")
            return None
        pd = node.GetPolyData()
        if pd is None or pd.GetNumberOfPoints() == 0:
            self._smap_status(f"Model '{node.GetName()}' has no geometry.")
            return None
        return pd

    def _smap_status(self, msg: str):
        print(f"{_LOG_TAG} {msg}")
        try:
            w = getattr(slicer.modules, "VesselAnalyzerWidget", None)
            if w and hasattr(w, "smapStatusLabel"):
                w.smapStatusLabel.setText(msg)
        except Exception:
            pass

    def _smap_flush(self):
        try:
            import qt as _qt
            _qt.QApplication.processEvents()
        except Exception:
            pass

    # ── Radius Map ────────────────────────────────────────────────────────────

    def runRadiusMap(self, modelNode):
        """Colour vessel surface by local radius (distance to nearest centerline pt).

        Requires branchMeta to be populated (run the centerline pipeline first).
        """
        print(f"{_LOG_TAG} ══ Radius Map ══════════════════════════════════════")

        surface_pd = self._smap_get_pd(modelNode, "vessel surface")
        if surface_pd is None:
            return None

        cl_pts = _centerline_points(self)
        if cl_pts is None or len(cl_pts) == 0:
            msg = (
                "No centerline data found.\n"
                "Run the centerline pipeline (Step 2 → Extract Centerline) first."
            )
            self._smap_status(msg)
            try:
                slicer.util.warningDisplay(msg)
            except Exception:
                pass
            return None

        self._smap_status(
            f"Computing radius map…  "
            f"({surface_pd.GetNumberOfPoints()} verts, {len(cl_pts)} CL pts)"
        )
        self._smap_flush()

        surface_verts = _surface_verts(surface_pd)
        radii = _compute_radius_map(surface_verts, cl_pts)

        r_min  = float(radii.min())
        r_max  = float(radii.max())
        r_mean = float(radii.mean())
        print(
            f"{_LOG_TAG} Radius — "
            f"min={r_min:.2f}  mean={r_mean:.2f}  max={r_max:.2f} mm"
        )

        result_pd   = _attach_scalar(surface_pd, radii, _RADIUS_SCALAR)
        result_name = modelNode.GetName() + _RADIUS_SUFFIX
        result_node = _get_or_create_node(result_name)
        result_node.SetAndObservePolyData(result_pd)
        _apply_colormap(result_node, _RADIUS_SCALAR, r_min, r_max)

        self._smap_status(
            f"✓ Radius map  "
            f"min={r_min:.2f}  mean={r_mean:.2f}  max={r_max:.2f} mm"
            f"  →  '{result_name}'"
        )
        print(f"{_LOG_TAG} ✓ '{result_name}'")
        return result_node

    # ── Wall Thickness Map ────────────────────────────────────────────────────

    def runWallThicknessMap(self, wallNode, lumenNode):
        """Colour outer-wall surface by distance to nearest lumen surface vertex."""
        print(f"{_LOG_TAG} ══ Wall Thickness Map ══════════════════════════════")

        wall_pd  = self._smap_get_pd(wallNode,  "outer wall")
        lumen_pd = self._smap_get_pd(lumenNode, "inner lumen")
        if wall_pd is None or lumen_pd is None:
            return None

        n_wall  = wall_pd.GetNumberOfPoints()
        n_lumen = lumen_pd.GetNumberOfPoints()
        self._smap_status(
            f"Computing wall thickness…  "
            f"({n_wall} wall verts → {n_lumen} lumen verts)"
        )
        self._smap_flush()

        wall_verts  = _surface_verts(wall_pd)
        lumen_verts = _surface_verts(lumen_pd)
        thickness   = _compute_wall_thickness(wall_verts, lumen_verts)

        t_min  = float(thickness.min())
        t_max  = float(thickness.max())
        t_mean = float(thickness.mean())
        print(
            f"{_LOG_TAG} Thickness — "
            f"min={t_min:.2f}  mean={t_mean:.2f}  max={t_max:.2f} mm"
        )

        result_pd   = _attach_scalar(wall_pd, thickness, _THICKNESS_SCALAR)
        result_name = wallNode.GetName() + _THICKNESS_SUFFIX
        result_node = _get_or_create_node(result_name)
        result_node.SetAndObservePolyData(result_pd)
        _apply_colormap(result_node, _THICKNESS_SCALAR, t_min, t_max)

        self._smap_status(
            f"✓ Wall thickness  "
            f"min={t_min:.2f}  mean={t_mean:.2f}  max={t_max:.2f} mm"
            f"  →  '{result_name}'"
        )
        print(f"{_LOG_TAG} ✓ '{result_name}'")
        return result_node

    # ── Clear ─────────────────────────────────────────────────────────────────

    def clearSurfaceMaps(self):
        suffixes = (_RADIUS_SUFFIX, _THICKNESS_SUFFIX)
        to_remove = []
        for i in range(slicer.mrmlScene.GetNumberOfNodes()):
            nd = slicer.mrmlScene.GetNthNode(i)
            if nd and nd.GetName() and any(nd.GetName().endswith(s) for s in suffixes):
                to_remove.append(nd)
        for nd in to_remove:
            slicer.mrmlScene.RemoveNode(nd)
        msg = f"Cleared {len(to_remove)} surface map node(s)."
        self._smap_status(msg)


# ══════════════════════════════════════════════════════════════════════════════
# Widget mixin
# ══════════════════════════════════════════════════════════════════════════════

class SurfaceMapWidgetMixin:

    def onRadiusMap(self):
        try:
            ref_sel  = getattr(self, "modelSelector", None)
            ref_node = ref_sel.currentNode() if ref_sel else None
            if ref_node is None:
                slicer.util.warningDisplay(
                    "Select a vessel model in Step 1 first."
                )
                return
            lbl = getattr(self, "smapStatusLabel", None)
            if lbl:
                lbl.setText("Computing radius map…")
            result = self.logic.runRadiusMap(ref_node)
            if result is not None:
                try:
                    slicer.util.resetThreeDViews()
                except Exception:
                    pass
        except Exception as exc:
            print(f"[SMAP Widget] onRadiusMap error: {exc}")
            print(traceback.format_exc())

    def onWallThicknessMap(self):
        try:
            wall_sel  = getattr(self, "smapWallSelector",  None)
            lumen_sel = getattr(self, "smapLumenSelector", None)
            wall_node  = wall_sel.currentNode()  if wall_sel  else None
            lumen_node = lumen_sel.currentNode() if lumen_sel else None
            if wall_node is None or lumen_node is None:
                slicer.util.warningDisplay(
                    "Select both an outer-wall model and an inner-lumen model."
                )
                return
            lbl = getattr(self, "smapStatusLabel", None)
            if lbl:
                lbl.setText("Computing wall thickness…")
            result = self.logic.runWallThicknessMap(wall_node, lumen_node)
            if result is not None:
                try:
                    slicer.util.resetThreeDViews()
                except Exception:
                    pass
        except Exception as exc:
            print(f"[SMAP Widget] onWallThicknessMap error: {exc}")
            print(traceback.format_exc())

    def onSurfaceMapClear(self):
        try:
            self.logic.clearSurfaceMaps()
        except Exception as exc:
            print(f"[SMAP Widget] onSurfaceMapClear error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Slicer module stub
# ══════════════════════════════════════════════════════════════════════════════

try:
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule

    class vessel_surface_map_mixin(ScriptedLoadableModule):
        """Slicer stub — this file is a mixin, not a standalone module."""
        def __init__(self, parent):
            super().__init__(parent)
            parent.title  = "vessel_surface_map_mixin"
            parent.hidden = True
except ImportError:
    pass
