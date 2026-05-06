# vessel_cloudcompare_mixin.py
# Mesh distance mapping for VesselAnalyzer — powered by Open3D
# Build tag: v2.0 (2026-04-26)
#
# Replaces the original CloudCompare CLI approach with direct in-process
# computation using Open3D.  No subprocess, no file I/O, no CloudCompare
# installation required.
#
# What it does
# ─────────────
# Given two vtkMRMLModelNodes (reference vessel + comparison mesh), it:
#   1. Extracts numpy point arrays directly from VTK polydata — no disk writes
#   2. Runs the chosen distance metric via Open3D
#   3. Writes the resulting per-point distances back as a VTK scalar array
#   4. Creates a new colour-mapped model node in the Slicer scene
#
# Supported metrics
# ─────────────────
#   "C2C Distance"   — cloud-to-cloud nearest-point distance (Open3D KD-tree)
#   "Hausdorff"      — symmetric max(C2C_ref→cmp, C2C_cmp→ref) per point
#   "RMS"            — root-mean-square of C2C distances (single scalar → uniform map)
#
# Integration (unchanged from v1 — no edits needed in VesselAnalyzer.py)
# ────────────────────────────────────────────────────────────────────────
# VesselAnalyzerLogic base classes:
#   from vessel_cloudcompare_mixin import CloudCompareMixin
#   class VesselAnalyzerLogic(..., CloudCompareMixin, ScriptedLoadableModuleLogic):
#
# VesselAnalyzerWidget method bindings:
#   from vessel_cloudcompare_mixin import CloudCompareWidgetMixin as _CCWM
#   onCloudCompareMap   = _CCWM.onCloudCompareMap
#   onCloudCompareClear = _CCWM.onCloudCompareClear
#
# vessel_analyzer_ui.py widgets (already added — no change needed):
#   ccCompareSelector, ccMetricCombo, ccMapButton, ccClearButton, ccStatusLabel
#   (cloudComparePathEdit is no longer used — left in UI as a no-op)
#
# Open3D installation (run once inside Slicer's Python console)
# ─────────────────────────────────────────────────────────────
#   slicer.util.pip_install("open3d")
#
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import traceback
from typing import Optional

# VTK / Slicer — available at runtime inside 3D Slicer
try:
    import vtk
    import slicer
    import numpy as np
except ImportError:
    vtk = None
    slicer = None
    np = None

# ── Output node naming ────────────────────────────────────────────────────────
_RESULT_SUFFIX = "_O3D_MAPPED"   # appended to comparison node name
_SCALAR_NAME   = "DistanceMM"   # VTK point-data array name on result node
_COLORMAP_IDX  = 10             # Slicer built-in colour table index (ColdToHot)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers — VTK ↔ numpy / Open3D conversions (no disk I/O)
# ══════════════════════════════════════════════════════════════════════════════

def _vtk_polydata_to_o3d(polydata):
    """Convert a vtkPolyData to an open3d.geometry.PointCloud in memory.

    Parameters
    ----------
    polydata : vtk.vtkPolyData

    Returns
    -------
    open3d.geometry.PointCloud
    """
    import open3d as o3d
    from vtk.util.numpy_support import vtk_to_numpy

    pts_vtk = polydata.GetPoints()
    if pts_vtk is None or pts_vtk.GetNumberOfPoints() == 0:
        raise ValueError("vtkPolyData has no points")

    pts_np = vtk_to_numpy(pts_vtk.GetData()).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)

    # Copy normals if available (improves some downstream metrics)
    normals_vtk = polydata.GetPointData().GetNormals()
    if normals_vtk is not None:
        normals_np = vtk_to_numpy(normals_vtk).astype(np.float64)
        pcd.normals = o3d.utility.Vector3dVector(normals_np)

    return pcd


def _distances_to_vtk_array(distances: np.ndarray, name: str):
    """Wrap a 1-D numpy float array as a named vtkFloatArray."""
    from vtk.util.numpy_support import numpy_to_vtk

    arr = numpy_to_vtk(distances.astype(np.float32), deep=True)
    arr.SetName(name)
    return arr


def _clone_polydata_with_scalars(source_pd, scalar_array):
    """Deep-copy source_pd and attach scalar_array as active point scalars."""
    result = vtk.vtkPolyData()
    result.DeepCopy(source_pd)
    result.GetPointData().RemoveArray(_SCALAR_NAME)
    result.GetPointData().AddArray(scalar_array)
    result.GetPointData().SetActiveScalars(_SCALAR_NAME)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Distance metrics
# ══════════════════════════════════════════════════════════════════════════════

def _compute_c2c(ref_pcd, cmp_pcd) -> np.ndarray:
    """Cloud-to-cloud nearest-point distance for every point in cmp_pcd.

    Returns a (N,) float64 array in mm (Slicer RAS units).
    """
    return np.asarray(cmp_pcd.compute_point_cloud_distance(ref_pcd))


def _compute_hausdorff(ref_pcd, cmp_pcd) -> np.ndarray:
    """Per-point symmetric Hausdorff distance.

    For each point in cmp_pcd: max(d_forward, d_backward) where
    d_forward = nearest distance from that point to ref,
    d_backward = nearest distance from the corresponding ref point back to cmp.
    """
    import open3d as o3d

    d_fwd    = np.asarray(cmp_pcd.compute_point_cloud_distance(ref_pcd))
    ref_tree = o3d.geometry.KDTreeFlann(ref_pcd)
    cmp_tree = o3d.geometry.KDTreeFlann(cmp_pcd)
    ref_pts  = np.asarray(ref_pcd.points)
    cmp_pts  = np.asarray(cmp_pcd.points)

    d_bwd = np.empty(len(cmp_pts), dtype=np.float64)
    for i, pt in enumerate(cmp_pts):
        _, idx_ref, _ = ref_tree.search_knn_vector_3d(pt, 1)
        nearest_ref   = ref_pts[idx_ref[0]]
        _, idx_cmp, _ = cmp_tree.search_knn_vector_3d(nearest_ref, 1)
        d_bwd[i]      = np.linalg.norm(nearest_ref - cmp_pts[idx_cmp[0]])

    return np.maximum(d_fwd, d_bwd)


def _compute_rms(ref_pcd, cmp_pcd) -> np.ndarray:
    """Global RMS C2C distance broadcast to every point.

    Produces a uniform colour map; the status label reports the scalar value.
    """
    d   = _compute_c2c(ref_pcd, cmp_pcd)
    rms = float(np.sqrt(np.mean(d ** 2)))
    print(f"[O3D] Global RMS distance: {rms:.4f} mm")
    return np.full(len(d), rms, dtype=np.float64)


# Metric registry — extend here to add new metrics without touching the mixin
_METRICS = {
    "C2C Distance": _compute_c2c,
    "Hausdorff":    _compute_hausdorff,
    "RMS":          _compute_rms,
}


# ══════════════════════════════════════════════════════════════════════════════
# Logic mixin
# ══════════════════════════════════════════════════════════════════════════════

class CloudCompareMixin:
    """Logic-level mixin — adds Open3D distance mapping to VesselAnalyzerLogic.

    Public API
    ──────────
    runCloudCompareMapping(referenceNode, compareNode, metric) → node | None
        Compute per-point distances in-process; return a colour-mapped node.

    clearCloudCompareMapping()
        Remove all O3D result nodes from the scene.
    """

    def runCloudCompareMapping(
        self,
        referenceNode,
        compareNode,
        metric: str = "C2C Distance",
        cc_exe: str = "",          # kept for API compat with v1 callers — ignored
    ):
        """Compute distance mapping between two mesh nodes using Open3D.

        Parameters
        ----------
        referenceNode : vtkMRMLModelNode
            Ground-truth vessel surface (from VesselAnalyzer pipeline).
        compareNode   : vtkMRMLModelNode
            Mesh to measure — follow-up scan, stent model, external import.
        metric        : str
            One of "C2C Distance", "Hausdorff", or "RMS".
        cc_exe        : str
            Ignored (kept for backward-compatibility with v1 callers).

        Returns
        -------
        vtkMRMLModelNode or None
        """
        print("[O3D] ══ Open3D Distance Mapping ══════════════════════════════")
        try:
            # ── 0. Ensure Open3D is available ──────────────────────────────
            if self._o3d_import() is None:
                return None

            # ── 1. Extract polydata from Slicer nodes ─────────────────────
            ref_pd = self._o3d_get_polydata(referenceNode, "reference")
            cmp_pd = self._o3d_get_polydata(compareNode,   "compare")
            if ref_pd is None or cmp_pd is None:
                return None

            n_ref = ref_pd.GetNumberOfPoints()
            n_cmp = cmp_pd.GetNumberOfPoints()
            print(f"[O3D] Reference: {n_ref} pts   Compare: {n_cmp} pts")

            # ── 2. Convert to Open3D point clouds (entirely in memory) ─────
            self._cc_set_status(f"Converting meshes…  ({n_ref} + {n_cmp} pts)")
            ref_pcd = _vtk_polydata_to_o3d(ref_pd)
            cmp_pcd = _vtk_polydata_to_o3d(cmp_pd)

            # ── 3. Compute distances ───────────────────────────────────────
            metric_fn = _METRICS.get(metric, _compute_c2c)
            self._cc_set_status(f"Computing {metric}…")
            print(f"[O3D] Metric: {metric}")

            distances = metric_fn(ref_pcd, cmp_pcd)   # shape (N,) float64

            d_min  = float(distances.min())
            d_max  = float(distances.max())
            d_mean = float(distances.mean())
            print(
                f"[O3D] Stats — "
                f"min={d_min:.3f} mm  mean={d_mean:.3f} mm  max={d_max:.3f} mm"
            )

            # ── 4. Attach distances to a copy of the compare polydata ──────
            scalar_arr = _distances_to_vtk_array(distances, _SCALAR_NAME)
            result_pd  = _clone_polydata_with_scalars(cmp_pd, scalar_arr)

            # ── 5. Create result model node in the scene ───────────────────
            result_name = compareNode.GetName() + _RESULT_SUFFIX
            result_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", result_name
            )
            result_node.SetAndObservePolyData(result_pd)
            result_node.CreateDefaultDisplayNodes()

            # ── 6. Apply colour map ────────────────────────────────────────
            self._o3d_apply_colormap(result_node, d_min, d_max)

            self._cc_set_status(
                f"✓ {metric}  "
                f"min={d_min:.2f}  mean={d_mean:.2f}  max={d_max:.2f} mm"
                f"  →  '{result_name}'"
            )
            print(f"[O3D] ✓ Result node: '{result_name}'")
            print("[O3D] ══════════════════════════════════════════════════════")
            return result_node

        except Exception as exc:
            print(f"[O3D] Error: {exc}")
            print(traceback.format_exc())
            self._cc_set_status(f"Error: {exc}")
            return None

    def clearCloudCompareMapping(self):
        """Remove all model nodes whose names end with _O3D_MAPPED."""
        try:
            to_remove = []
            for i in range(slicer.mrmlScene.GetNumberOfNodes()):
                nd = slicer.mrmlScene.GetNthNode(i)
                if nd and nd.GetName() and nd.GetName().endswith(_RESULT_SUFFIX):
                    to_remove.append(nd)
            for nd in to_remove:
                slicer.mrmlScene.RemoveNode(nd)
            msg = f"Cleared {len(to_remove)} mapping node(s)."
            print(f"[O3D] {msg}")
            self._cc_set_status(msg)
        except Exception as exc:
            print(f"[O3D] clearCloudCompareMapping error: {exc}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _o3d_import(self):
        """Import open3d and return the module, or None with a helpful message."""
        try:
            import open3d as o3d
            return o3d
        except ImportError:
            msg = (
                "Open3D is not installed.\n"
                "Run this once in Slicer's Python console:\n\n"
                "    slicer.util.pip_install('open3d')\n\n"
                "Then restart Slicer and try again."
            )
            print(f"[O3D] {msg}")
            self._cc_set_status(
                "Open3D not installed — run: slicer.util.pip_install('open3d')"
            )
            try:
                slicer.util.warningDisplay(msg)
            except Exception:
                pass
            return None

    def _o3d_get_polydata(self, model_node, label: str):
        """Validate and return polydata from a model node, or None on failure."""
        if model_node is None:
            print(f"[O3D] {label} node is None")
            self._cc_set_status(f"No {label} model selected.")
            return None
        pd = model_node.GetPolyData()
        if pd is None or pd.GetNumberOfPoints() == 0:
            print(f"[O3D] {label} '{model_node.GetName()}' has no geometry")
            self._cc_set_status(
                f"Model '{model_node.GetName()}' has no geometry."
            )
            return None
        return pd

    def _o3d_apply_colormap(
        self, model_node, d_min: float, d_max: float
    ) -> None:
        """Set ColdToHot colour mapping on the distance scalar field."""
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
                f"[O3D] Colour map: ColdToHot  "
                f"range [{d_min:.3f}, {d_max:.3f}] mm"
            )
        except Exception as exc:
            print(f"[O3D] _o3d_apply_colormap error: {exc}")

    def _cc_set_status(self, msg: str) -> None:
        """Push a status string to the ccStatusLabel widget."""
        try:
            widget = getattr(slicer.modules, "VesselAnalyzerWidget", None)
            if widget and hasattr(widget, "ccStatusLabel"):
                widget.ccStatusLabel.setText(msg)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Widget mixin — Qt slot methods bound into VesselAnalyzerWidget
# ══════════════════════════════════════════════════════════════════════════════

class CloudCompareWidgetMixin:
    """Qt event handlers for the Open3D mapping panel.

    Bound into VesselAnalyzerWidget:
        from vessel_cloudcompare_mixin import CloudCompareWidgetMixin as _CCWM
        onCloudCompareMap   = _CCWM.onCloudCompareMap
        onCloudCompareClear = _CCWM.onCloudCompareClear
    """

    def onCloudCompareMap(self):
        """Slot: '▶ Map with Open3D' button clicked."""
        try:
            # ── Reference = Step 1 model selector ─────────────────────────
            ref_node = getattr(self, "modelSelector", None)
            ref_node = ref_node.currentNode() if ref_node else None

            # ── Compare = ccCompareSelector, else most recent other model ──
            cmp_node = None
            cmp_sel  = getattr(self, "ccCompareSelector", None)
            if cmp_sel is not None:
                cmp_node = cmp_sel.currentNode()

            if cmp_node is None:
                for i in range(slicer.mrmlScene.GetNumberOfNodes() - 1, -1, -1):
                    nd = slicer.mrmlScene.GetNthNode(i)
                    if (
                        nd
                        and nd.IsA("vtkMRMLModelNode")
                        and nd != ref_node
                        and not (nd.GetName() or "").endswith(_RESULT_SUFFIX)
                        and nd.GetPolyData()
                        and nd.GetPolyData().GetNumberOfPoints() > 0
                    ):
                        cmp_node = nd
                        break

            if ref_node is None:
                slicer.util.warningDisplay(
                    "Select a vessel model in Step 1 before running mapping."
                )
                return
            if cmp_node is None:
                slicer.util.warningDisplay(
                    "No comparison model found.\n"
                    "Load a second model and select it in the 'Compare model' dropdown."
                )
                return

            metric = "C2C Distance"
            combo  = getattr(self, "ccMetricCombo", None)
            if combo:
                metric = combo.currentText

            lbl = getattr(self, "ccStatusLabel", None)
            if lbl:
                lbl.setText(f"Running {metric}…")
            import qt
            qt.QApplication.processEvents()

            result = self.logic.runCloudCompareMapping(
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
            print(f"[O3D Widget] onCloudCompareMap error: {exc}")
            print(traceback.format_exc())

    def onCloudCompareClear(self):
        """Slot: '✕ Clear Mapping' button clicked."""
        try:
            self.logic.clearCloudCompareMapping()
        except Exception as exc:
            print(f"[O3D Widget] onCloudCompareClear error: {exc}")


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

    class vessel_cloudcompare_mixin(ScriptedLoadableModule):
        """Slicer stub — this file is a mixin/patch, not a standalone module."""
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "vessel_cloudcompare_mixin"
            parent.hidden = True   # keeps it out of the Modules menu
except ImportError:
    pass  # outside Slicer (unit tests, CI) — stub not needed
