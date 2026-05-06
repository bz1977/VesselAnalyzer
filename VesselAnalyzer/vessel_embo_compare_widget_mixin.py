"""
vessel_embo_compare_widget_mixin.py — VesselAnalyzerWidget embolization, comparison and lesion-table handlers

Extracted from VesselAnalyzer.py (VesselAnalyzerWidget methods).

Usage in VesselAnalyzer.py
--------------------------
Add to the VesselAnalyzerWidget class definition::

    from vessel_embo_compare_widget_mixin import EmboCompareWidgetMixin

    class VesselAnalyzerWidget(EmboCompareWidgetMixin, ScriptedLoadableModuleWidget, VTKObservationMixin):
        ...

All method bodies are unchanged — only their location has moved.
"""

import os
import math
import vtk
import slicer
import ctk
import qt
import numpy as _np
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


class EmboCompareWidgetMixin:
    """Mixin: VesselAnalyzerWidget embolization, comparison and lesion-table handlers"""
    def onAddRuler(self):
        """Place a ruler parallel to the lesion path, offset 15mm to the side."""
        import math, vtk

        # Get path from preDilationMap (balloon) or stent node
        pts = self.logic.points
        diams = self.logic.diameters

        # Collect path points from preDilationMap if available
        pdmap = getattr(self.logic, "preDilationMap", {})
        if pdmap:
            path_gis = sorted(set(gi for gi in pdmap if 0 <= gi < len(pts)))
        else:
            # Fall back to active stent branch
            bi = self.logic.activeBranch
            if bi < 0 or bi >= len(self.logic.branches):
                slicer.util.warningDisplay("Place a balloon or stent first.")
                return
            bs, be = self.logic.branches[bi]
            path_gis = list(range(bs, be))

        if len(path_gis) < 2:
            slicer.util.warningDisplay("No lesion path found.")
            return

        # Remove old ruler
        self.onRemoveRuler()

        # Start and end points of path
        p_start = pts[path_gis[0]]
        p_end = pts[path_gis[-1]]

        # Compute path length
        length_mm = sum(
            math.sqrt(
                sum(
                    (pts[path_gis[i + 1]][k] - pts[path_gis[i]][k]) ** 2
                    for k in range(3)
                )
            )
            for i in range(len(path_gis) - 1)
        )

        # Offset vector: perpendicular to path direction in the RAS XY plane
        path_dir = [p_end[k] - p_start[k] for k in range(3)]
        path_len = math.sqrt(sum(x * x for x in path_dir)) or 1.0
        path_dir = [x / path_len for x in path_dir]
        # Perpendicular in XY plane (rotate 90° around Z)
        perp = [-path_dir[1], path_dir[0], 0.0]
        perp_len = math.sqrt(sum(x * x for x in perp)) or 1.0
        perp = [x / perp_len for x in perp]
        OFFSET = 15.0  # mm offset from vessel

        r_start = [p_start[k] + perp[k] * OFFSET for k in range(3)]
        r_end = [p_end[k] + perp[k] * OFFSET for k in range(3)]

        # Create ruler (vtkMRMLMarkupsLineNode)
        rulerNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsLineNode", "Ruler"
        )
        rulerNode.AddControlPoint(vtk.vtkVector3d(*r_start))
        rulerNode.AddControlPoint(vtk.vtkVector3d(*r_end))
        rulerNode.SetAttribute("VesselAnalyzerRuler", "1")

        dn = rulerNode.GetDisplayNode()
        if dn:
            dn.SetSelectedColor(1.0, 0.9, 0.0)
            dn.SetColor(1.0, 0.9, 0.0)
            dn.SetLineThickness(0.5)
            dn.SetTextScale(4.0)
            dn.SetVisibility(1)

        self.rulerStatusLabel.setText(
            f"📏 Ruler: {length_mm:.1f}mm  (offset {OFFSET}mm)"
        )
        print(
            f"[Ruler] Placed: {length_mm:.1f}mm, "
            f"start={[round(x,1) for x in r_start]}, "
            f"end={[round(x,1) for x in r_end]}"
        )

    def onRemoveRuler(self):
        """Remove all ruler nodes from scene."""
        for node in list(slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode")):
            if node.GetAttribute("VesselAnalyzerRuler") == "1":
                slicer.mrmlScene.RemoveNode(node)
        self.rulerStatusLabel.setText("")

    def onModelOpacityChanged(self, value):
        """Set vessel model opacity from slider.
        Applies to both the raw vessel model and the '_Surface' node so the
        slider works regardless of which the Model/Surface toggle is showing.
        """
        # Determine the primary vessel model node (prefer selector, fall back to logic)
        modelNode = self.modelSelector.currentNode()
        if not modelNode and self.logic:
            modelNode = self.logic.modelNode

        # Also find the surface node derived from the model (created by onCreateSurface)
        surfNode = None
        if modelNode:
            surfName = modelNode.GetName() + "_Surface"
            surfNode = slicer.mrmlScene.GetFirstNodeByName(surfName)

        # Apply opacity to the vessel model
        if modelNode:
            dn = modelNode.GetDisplayNode()
            if dn:
                dn.SetOpacity(value)

        # Apply the same opacity to the surface node so switching between
        # Model/Surface views always honours the user's chosen opacity level
        if surfNode:
            dn = surfNode.GetDisplayNode()
            if dn:
                dn.SetOpacity(value)

    def onModelVisToggle(self, checked):
        """Toggle between showing the vessel model or the surface (mutually exclusive)."""
        if checked:
            self.modelVisButton.setText("👁 Showing: Model")
            self.modelVisButton.setStyleSheet(
                "background-color: #27ae60; color: white; font-weight: bold; padding: 4px;"
            )
        else:
            self.modelVisButton.setText("👁 Showing: Surface")
            self.modelVisButton.setStyleSheet(
                "background-color: #2980b9; color: white; font-weight: bold; padding: 4px;"
            )

        modelNode = self.modelSelector.currentNode()
        # Derive the surface node name (created by onCreateSurface as "<model>_Surface")
        surfNode = None
        if modelNode:
            surfName = modelNode.GetName() + "_Surface"
            surfNode = slicer.mrmlScene.GetFirstNodeByName(surfName)

        # Preserve the user's chosen opacity when switching views
        currentOpacity = self.modelOpacitySlider.value

        if checked:
            # Show model, hide surface
            if modelNode and modelNode.GetDisplayNode():
                modelNode.GetDisplayNode().SetOpacity(currentOpacity)
                modelNode.GetDisplayNode().SetVisibility(1)
            if surfNode and surfNode.GetDisplayNode():
                surfNode.GetDisplayNode().SetVisibility(0)
        else:
            # Show surface, hide model
            if modelNode and modelNode.GetDisplayNode():
                modelNode.GetDisplayNode().SetVisibility(0)
            if surfNode and surfNode.GetDisplayNode():
                surfNode.GetDisplayNode().SetOpacity(currentOpacity)
                surfNode.GetDisplayNode().SetVisibility(1)

    # ── Embolization Planner ─────────────────────────────────────────────

    def onEmboDeviceChanged(self, index):
        """Update default size and labels when device type changes."""
        defaults = {
            0: (8.0, "Coil packing diameter"),
            1: (10.0, "Plug diameter"),
            2: (0.0, "Volume (mL) — set manually"),
            3: (5.0, "Flow diverter diameter"),
        }
        size, tip = defaults.get(index, (8.0, "Device size"))
        if size > 0:
            self.emboSizeSpin.setValue(size)

    def onEmboPickZone(self):
        """Enter 2-point pick mode for embolization zone — same as balloon picker."""
        try:
            if not self.logic or not self.logic.points:
                slicer.util.errorDisplay("Load a centerline first.")
                return
            self.onEmboCancelPick()
            self._emboPickNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "EmboPickPoints"
            )
            dn = self._emboPickNode.GetDisplayNode()
            if dn:
                dn.SetSelectedColor(0.2, 0.5, 1.0)
                dn.SetColor(0.2, 0.5, 1.0)
                dn.SetTextScale(3.0)
                dn.SetGlyphScale(3.0)
            slicer.modules.markups.logic().StartPlaceMode(1)
            self.emboPickZoneButton.setEnabled(False)
            self.emboCancelPickButton.setEnabled(True)
            self.emboPlaceButton.setEnabled(False)
            self.emboStatusLabel.setText("📍 Click point 1 (proximal end)...")
            self._emboPickLastCount = 0
            self._emboPickSavedPositions = []
            self._emboPickFinished = False
            self._emboPickTimer = qt.QTimer()
            self._emboPickTimer.setInterval(250)
            self._emboPickTimer.connect("timeout()", self._pollEmboPickPoints)
            self._emboPickTimer.start()
        except Exception as _e:
            import traceback

            traceback.print_exc()
            self.emboStatusLabel.setText(f"⚠️ Start failed: {_e}")

    def _pollEmboPickPoints(self):
        """Poll every 250ms until 2 points placed."""
        try:
            if getattr(self, "_emboPickFinished", False):
                return
            node = getattr(self, "_emboPickNode", None)
            if not node or not slicer.mrmlScene.IsNodePresent(node):
                self._emboPickTimer.stop()
                return
            n_total = node.GetNumberOfControlPoints()
            n_placed = max(0, n_total - 1)
            positions = []
            for i in range(n_total):
                p = [0.0, 0.0, 0.0]
                try:
                    node.GetNthControlPointPosition(i, p)
                    positions.append(list(p))
                except:
                    pass
            if len(positions) >= 2:
                self._emboPickSavedPositions = positions[:]
            if n_placed == self._emboPickLastCount:
                return
            self._emboPickLastCount = n_placed
            if n_placed == 1:
                self.emboStatusLabel.setText("📍 Click point 2 (distal end)...")
            elif n_placed >= 2:
                self._emboPickFinished = True
                self._emboPickTimer.stop()
                qt.QTimer.singleShot(200, self._finishEmboPick)
        except Exception as _e:
            import traceback

            traceback.print_exc()
            try:
                self._emboPickTimer.stop()
            except:
                pass

    def _finishEmboPick(self):
        """Snap picked points to centerline and set zone."""
        try:
            inode = slicer.app.applicationLogic().GetInteractionNode()
            inode.SetCurrentInteractionMode(inode.ViewTransform)
        except:
            pass
        self.emboPickZoneButton.setEnabled(True)
        self.emboCancelPickButton.setEnabled(False)
        if getattr(self, "_emboPickNode", None) and slicer.mrmlScene.IsNodePresent(
            self._emboPickNode
        ):
            slicer.mrmlScene.RemoveNode(self._emboPickNode)
        self._emboPickNode = None
        saved = getattr(self, "_emboPickSavedPositions", [])
        if len(saved) < 2:
            self.emboStatusLabel.setText("⚠️ Need 2 points — try again")
            return
        p1, p2 = saved[0], saved[1]
        gi1 = self._snapToCenterline(p1)
        gi2 = self._snapToCenterline(p2)
        if gi1 == gi2:
            self.emboStatusLabel.setText("⚠️ Points too close — try again")
            return
        if gi1 > gi2:
            gi1, gi2 = gi2, gi1
        self._emboProxGi = gi1
        self._emboDistGi = gi2
        pt1 = self.logic.points[gi1]
        d1 = self.logic.diameters[gi1] if self.logic.diameters else 0
        pt2 = self.logic.points[gi2]
        d2 = self.logic.diameters[gi2] if self.logic.diameters else 0
        length = self.logic.distances[gi2] - self.logic.distances[gi1]
        # Auto-size from mean diameter
        mean_d = (d1 + d2) / 2 if d1 > 0 and d2 > 0 else max(d1, d2)
        if mean_d > 0:
            self.emboSizeSpin.setValue(round(mean_d * 1.15 / 0.5) * 0.5)
        self.emboZoneLabel.setText(
            f"Proximal: pt{gi1} Ø{d1:.1f}mm  →  Distal: pt{gi2} Ø{d2:.1f}mm  ({length:.0f}mm)"
        )
        self.emboPlaceButton.setEnabled(True)
        self.emboRemoveButton.setEnabled(True)
        self.emboStatusLabel.setText("✓ Zone set — adjust size and click Place Device")
        # Show markers
        import vtk

        for gi, name, color in [
            (gi1, "EmboProxMarker", (0.2, 0.5, 1.0)),
            (gi2, "EmboDistMarker", (0.9, 0.5, 0.1)),
        ]:
            pt = self.logic.points[gi]
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", name
            )
            node.AddControlPoint(vtk.vtkVector3d(pt[0], pt[1], pt[2]))
            node.SetNthControlPointLabel(0, "Prox" if gi == gi1 else "Dist")
            dn = node.GetDisplayNode()
            dn.SetSelectedColor(*color)
            dn.SetColor(*color)
            dn.SetGlyphScale(5.0)
            dn.SetTextScale(3.5)
            dn.SetVisibility(1)
            node.SetAttribute("VesselAnalyzerEmbo", "1")

    def onEmboCancelPick(self):
        """Cancel embolization zone picking."""
        try:
            inode = slicer.app.applicationLogic().GetInteractionNode()
            inode.SetCurrentInteractionMode(inode.ViewTransform)
        except:
            pass
        if getattr(self, "_emboPickTimer", None):
            try:
                self._emboPickTimer.stop()
            except:
                pass
        if getattr(self, "_emboPickNode", None) and slicer.mrmlScene.IsNodePresent(
            self._emboPickNode
        ):
            slicer.mrmlScene.RemoveNode(self._emboPickNode)
        self._emboPickNode = None
        self.emboPickZoneButton.setEnabled(True)
        self.emboCancelPickButton.setEnabled(False)
        self.emboStatusLabel.setText("")

    def onEmboPlace(self):
        """Place embolization device between the two picked points."""
        import vtk, math

        prox_gi = getattr(self, "_emboProxGi", -1)
        dist_gi = getattr(self, "_emboDistGi", -1)
        if prox_gi < 0 or dist_gi < 0:
            self.emboStatusLabel.setText("Pick both proximal and distal points first.")
            return
        if prox_gi > dist_gi:
            prox_gi, dist_gi = dist_gi, prox_gi
        pts = self.logic.points
        device_size = self.emboSizeSpin.value
        device_idx = self.emboDeviceCombo.currentIndex
        total_len = self.logic.distances[dist_gi] - self.logic.distances[prox_gi]

        self.onEmboRemove()

        device_names = ["CoilEmbo", "PlugEmbo", "LiquidEmbo", "FlowDiverter"]
        node_name = device_names[device_idx]
        colors = [(0.9, 0.7, 0.1), (0.2, 0.5, 1.0), (0.6, 0.1, 0.8), (0.1, 0.8, 0.5)]
        color = colors[device_idx]

        try:
            if device_idx == 0:
                self._placeCoilEmbo(
                    pts, prox_gi, dist_gi, device_size, node_name, color
                )
            elif device_idx == 1:
                self._placePlugEmbo(
                    pts, prox_gi, dist_gi, device_size, node_name, color
                )
            elif device_idx == 2:
                self._placeLiquidEmbo(
                    pts, prox_gi, dist_gi, device_size, node_name, color
                )
            elif device_idx == 3:
                self._placeFlowDiverter(
                    pts, prox_gi, dist_gi, device_size, node_name, color
                )
            # Re-place the landing markers (removed by onEmboRemove)
            self._emboPlaceMarker(prox_gi, "EmboProxMarker", (0.2, 0.5, 1.0))
            self._emboPlaceMarker(dist_gi, "EmboDistMarker", (0.9, 0.5, 0.1))
            dev_name = self.emboDeviceCombo.currentText.split(" ", 1)[-1]
            self.emboStatusLabel.setText(
                f"{dev_name} placed  Ø{device_size:.1f}mm  length={total_len:.0f}mm"
            )
        except Exception as _e:
            import traceback

            self.emboStatusLabel.setText(f"Error: {_e}")
            traceback.print_exc()

    def _emboPlaceMarker(self, gi, name, color):
        """Place a fiducial marker at centerline point gi."""
        import vtk

        for n in list(slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")):
            if n.GetName() == name:
                slicer.mrmlScene.RemoveNode(n)
        pt = self.logic.points[gi]
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        node.AddControlPoint(vtk.vtkVector3d(pt[0], pt[1], pt[2]))
        node.SetNthControlPointLabel(0, "Prox" if "Prox" in name else "Dist")
        dn = node.GetDisplayNode()
        dn.SetSelectedColor(*color)
        dn.SetColor(*color)
        dn.SetGlyphScale(5.0)
        dn.SetTextScale(3.5)
        dn.SetVisibility(1)
        node.SetAttribute("VesselAnalyzerEmbo", "1")

    def _emboMakeTube(self, pts, gi_start, gi_end, radii, n_sides, capping):
        """Build a variable-radius tube along a centerline path.
        radii: list of per-point radii (same length as path).
        Returns vtkPolyData of the tube."""
        import vtk

        path = pts[gi_start : gi_end + 1]
        cl_pts = vtk.vtkPoints()
        cl_lines = vtk.vtkCellArray()
        rad_arr = vtk.vtkFloatArray()
        rad_arr.SetName("TubeRadius")
        cl_lines.InsertNextCell(len(path))
        for k, p in enumerate(path):
            cl_lines.InsertCellPoint(cl_pts.InsertNextPoint(p[0], p[1], p[2]))
            rad_arr.InsertNextValue(float(radii[k]))
        pd = vtk.vtkPolyData()
        pd.SetPoints(cl_pts)
        pd.SetLines(cl_lines)
        pd.GetPointData().AddArray(rad_arr)
        pd.GetPointData().SetActiveScalars("TubeRadius")
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(pd)
        tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        tube.SetRadius(1.0)  # base radius (overridden by scalar)
        tube.SetNumberOfSides(n_sides)
        if capping:
            tube.CappingOn()
        else:
            tube.CappingOff()
        tube.Update()
        return tube.GetOutput()

    def _emboMakeNode(self, pd, name, color, opacity, wireframe=False):
        """Create a model node from polydata with display settings."""
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.SetAndObservePolyData(pd)
        node.CreateDefaultDisplayNodes()
        dn = node.GetDisplayNode()
        dn.SetColor(*color)
        dn.SetOpacity(opacity)
        dn.SetBackfaceCulling(0)
        dn.SetVisibility(1)
        if wireframe:
            dn.SetRepresentation(1)
            dn.SetLineWidth(2)
        node.SetAttribute("VesselAnalyzerEmbo", "1")
        return node

    def _emboRadii(self, gi_start, gi_end, scale=1.0):
        """Return per-point radii along gi_start..gi_end using actual vessel diameters."""
        diams = self.logic.diameters
        radii = []
        for gi in range(gi_start, gi_end + 1):
            d = diams[gi] if diams and gi < len(diams) and diams[gi] > 0 else 8.0
            radii.append(d / 2.0 * scale)
        return radii

    def _placeCoilEmbo(self, pts, gi_start, gi_end, diameter, name, color):
        """Helical coil that follows vessel wall diameter at each point."""
        import vtk, math

        path = pts[gi_start : gi_end + 1]
        if len(path) < 2:
            return
        coil_pts = vtk.vtkPoints()
        coil_lines = vtk.vtkCellArray()
        TURNS = 8
        n = len(path)
        coil_lines.InsertNextCell(n * TURNS)
        diams = self.logic.diameters
        for i in range(n):
            gi = gi_start + i
            d = diams[gi] if diams and gi < len(diams) and diams[gi] > 0 else diameter
            r = d / 2.0 * 0.92  # reach vessel wall
            base = path[i]
            ni = min(i + 1, n - 1)
            pi = max(i - 1, 0)
            tx = path[ni][0] - path[pi][0]
            ty = path[ni][1] - path[pi][1]
            tz = path[ni][2] - path[pi][2]
            tl = math.sqrt(tx * tx + ty * ty + tz * tz) or 1
            tx /= tl
            ty /= tl
            tz /= tl
            px = -ty
            py = tx
            pz = 0.0
            pl = math.sqrt(px * px + py * py) or 1
            px /= pl
            py /= pl
            cx = ty * pz - tz * py
            cy = tz * px - tx * pz
            cz = tx * py - ty * px
            t = i / max(n - 1, 1)
            for j in range(TURNS):
                angle = t * TURNS * 2 * math.pi + j * 2 * math.pi / TURNS
                x = base[0] + r * (math.cos(angle) * px + math.sin(angle) * cx)
                y = base[1] + r * (math.cos(angle) * py + math.sin(angle) * cy)
                z = base[2] + r * (math.cos(angle) * pz + math.sin(angle) * cz)
                coil_lines.InsertCellPoint(coil_pts.InsertNextPoint(x, y, z))
        pd = vtk.vtkPolyData()
        pd.SetPoints(coil_pts)
        pd.SetLines(coil_lines)
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(pd)
        tube.SetRadius(0.5)
        tube.SetNumberOfSides(6)
        tube.Update()
        self._emboMakeNode(tube.GetOutput(), name, color, 0.95)

    def _placePlugEmbo(self, pts, gi_start, gi_end, diameter, name, color):
        """Solid plug sized exactly to vessel diameter at each point."""
        radii = self._emboRadii(gi_start, gi_end, scale=1.0)
        pd = self._emboMakeTube(pts, gi_start, gi_end, radii, 20, True)
        self._emboMakeNode(pd, name, color, 0.85)

    def _placeLiquidEmbo(self, pts, gi_start, gi_end, diameter, name, color):
        """Semi-transparent zone sized to vessel diameter at each point."""
        radii = self._emboRadii(gi_start, gi_end, scale=1.0)
        pd = self._emboMakeTube(pts, gi_start, gi_end, radii, 24, True)
        self._emboMakeNode(pd, name, color, 0.45)

    def _placeFlowDiverter(self, pts, gi_start, gi_end, diameter, name, color):
        """Wireframe mesh sized to vessel diameter at each point."""
        radii = self._emboRadii(gi_start, gi_end, scale=1.0)
        pd = self._emboMakeTube(pts, gi_start, gi_end, radii, 16, False)
        self._emboMakeNode(pd, name, color, 0.75, wireframe=True)

    def onEmboRemove(self):
        """Remove all embolization device nodes from scene."""
        for node in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
            if node.GetAttribute("VesselAnalyzerEmbo") == "1":
                slicer.mrmlScene.RemoveNode(node)
        for node in list(slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")):
            if node.GetAttribute("VesselAnalyzerEmbo") == "1":
                slicer.mrmlScene.RemoveNode(node)
        self._emboProxGi = -1
        self._emboDistGi = -1
        self.emboZoneLabel.setText("No zone set")
        self.emboPlaceButton.setEnabled(False)
        self.emboStatusLabel.setText("")

    def onExitCompare(self):
        """Restore single 3D view layout and maximize it."""
        # Remove view-node restrictions so model shows in all views again
        if self.logic.modelNode:
            _mdn = self.logic.modelNode.GetDisplayNode()
            if _mdn:
                _mdn.RemoveAllViewNodeIDs()
                _mdn.SetVisibility(1)
                _mdn.SetOpacity(1.0)
        # Remove the before-snapshot node
        for node in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
            if node.GetName() in (
                "VesselBefore_Snapshot",
            ) or node.GetName().startswith("BeforeStenosis_"):
                slicer.mrmlScene.RemoveNode(node)
        # Remove view restrictions from all remaining model nodes
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            dn = node.GetDisplayNode()
            if dn:
                dn.RemoveAllViewNodeIDs()
        # Restore slice visibility — keep them hidden (we always want 3D-only)
        self._sliceVis3DBackup = {}
        # Switch to 3D-only layout and maximize
        self._TWO3D_LAYOUT_ID = None  # clear whitelist so observer enforces single-3D
        self._force3DLayout()
        qt.QTimer.singleShot(100, self._force3DLayout)
        qt.QTimer.singleShot(400, self._force3DLayout)
        qt.QTimer.singleShot(800, self._force3DLayout)

    def onCompareBeforeAfter(self):
        """Split into two 3D viewports using Slicer built-in layout 15 (Two3D).
        LEFT  = vessel with compressed diameters shown as RED stenosis markers
        RIGHT = vessel with balloon overlays (after pre-dilation state)
        """
        import vtk

        try:
            pdmap = getattr(self.logic, "preDilationMap", {})
            if not pdmap:
                slicer.util.warningDisplay(
                    "Run Pre-Dilate first to generate the expanded state."
                )
                return

            # Do NOT re-apply color overlay here — ballooned lesion markers were
            # already removed in onPreDilate. Re-applying would bring them back.
            # Only apply if there are non-ballooned findings remaining.
            try:
                pdmap = getattr(self.logic, "preDilationMap", {})
                remaining = [
                    f
                    for f in getattr(self.logic, "findings", [])
                    if f.get("pointIdx", -1) not in pdmap
                ]
                if remaining:
                    self.logic.applyColorOverlay()
            except Exception as _oe:
                print(f"[Compare] overlay failed (non-fatal): {_oe}")

            # ── Switch to Two-3D layout ───────────────────────────────────
            layoutMgr = slicer.app.layoutManager()
            # Use Slicer's built-in side-by-side dual-3D layout (ID 15).
            # Avoids leaving a custom XML description that confuses exit.
            TWO3D_LAYOUT_ID = slicer.vtkMRMLLayoutNode.SlicerLayoutDual3DView
            self._TWO3D_LAYOUT_ID = TWO3D_LAYOUT_ID  # whitelist in _onLayoutChanged
            layoutMgr.setLayout(TWO3D_LAYOUT_ID)

            # Hide all slice planes / G R Y layouts immediately after switching
            self._hideAllSlicePlanes()
            qt.QTimer.singleShot(300, self._hideAllSlicePlanes)

            # ── Get the two 3D view nodes ─────────────────────────────────
            # After layout switch, threedWidget(0) = left, threedWidget(1) = right
            import qt

            slicer.app.processEvents()
            try:
                slicer.app.layoutManager().resetThreeDViews()
            except Exception:
                pass
            slicer.app.processEvents()
            qt.QTimer.singleShot(600, self._compareAfterLayoutSwitch)
            self.preDilateStatusLabel.setText(
                "LEFT: before (compressed)  |  RIGHT: after (live)"
            )

        except Exception as e:
            import traceback

            print("[Compare] Error: " + str(e) + traceback.format_exc())
            slicer.util.errorDisplay("Compare view failed: " + str(e))

    def _compareAfterLayoutSwitch(self):
        """Called 300ms after layout switch — views are now initialised."""
        import vtk

        try:
            # Ensure G/R/Y slice planes are hidden in both 3D views
            self._hideAllSlicePlanes()
            pdmap = getattr(self.logic, "preDilationMap", {})
            ballooned_pts = set(pdmap.keys())

            # Remove overlay markers for ballooned points — they should not
            # appear in the After (Live) view
            if ballooned_pts:
                to_remove = []
                for i in range(slicer.mrmlScene.GetNumberOfNodes()):
                    n = slicer.mrmlScene.GetNthNode(i)
                    if n and n.GetAttribute("VesselAnalyzerOverlay") == "1":
                        try:
                            pt_idx = int(n.GetName().rsplit("_", 1)[-1])
                            if pt_idx in ballooned_pts:
                                to_remove.append(n)
                        except (ValueError, IndexError):
                            pass
                for n in to_remove:
                    slicer.mrmlScene.RemoveNode(n)

            layoutMgr = slicer.app.layoutManager()
            n3d = layoutMgr.threeDViewCount
            if n3d < 2:
                slicer.util.warningDisplay("Could not create split view.")
                return

            viewNode1 = layoutMgr.threeDWidget(0).mrmlViewNode()  # TOP    = Before
            viewNode2 = layoutMgr.threeDWidget(
                1
            ).mrmlViewNode()  # BOTTOM = After (Live)

            # ── Build "Before" snapshot (same mesh geometry, different color) ──
            origPD = self.logic.modelNode.GetPolyData()
            if not origPD:
                return

            for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
                if node.GetName() == "VesselBefore_Snapshot":
                    slicer.mrmlScene.RemoveNode(node)

            snapPD = vtk.vtkPolyData()
            snapPD.DeepCopy(origPD)

            # Draw compressed diameter spheres on the before-snapshot
            # by adding red stenosis cylinders at lesion points
            snapNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", "VesselBefore_Snapshot"
            )
            snapNode.SetAndObservePolyData(snapPD)
            snapNode.CreateDefaultDisplayNodes()
            snapDN = snapNode.GetDisplayNode()
            snapDN.SetColor(0.85, 0.3, 0.15)  # orange-red = compressed
            snapDN.SetOpacity(0.60)
            snapDN.SetRepresentation(2)

            # No stenosis cylinders — color overlay shows compression clearly

            # ── Hide CT from both 3D views ────────────────────────────────
            self._view3DBackgroundIDs = {}
            self._sliceVis3DBackup = {}
            layoutMgr2 = slicer.app.layoutManager()
            # 1. Clear background/foreground volumes from 3D view nodes
            for i in range(layoutMgr2.threeDViewCount):
                vn = layoutMgr2.threeDWidget(i).mrmlViewNode()
                self._view3DBackgroundIDs[vn.GetID()] = None
            # 2. Hide slice planes in 3D
            for sliceNode in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
                self._sliceVis3DBackup[sliceNode.GetID()] = sliceNode.GetSliceVisible()
                sliceNode.SetSliceVisible(0)
            # 3. Hide all volume nodes from 3D
            self._hiddenVolDisplayNodes = []
            for volNode in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
                dn = volNode.GetDisplayNode()
                if dn and dn.GetVisibility():
                    dn.SetVisibility(0)
                    self._hiddenVolDisplayNodes.append(dn)
            # 4. Disable volume rendering
            try:
                vrLogic = slicer.modules.volumerendering.logic()
                self._hiddenVRNodes = []
                for volNode in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
                    vrDN = vrLogic.GetFirstVolumeRenderingDisplayNode(volNode)
                    if vrDN and vrDN.GetVisibility():
                        vrDN.SetVisibility(0)
                        self._hiddenVRNodes.append(vrDN)
            except Exception:
                self._hiddenVRNodes = []

            # ── Assign nodes to views ─────────────────────────────────────
            mainDN = self.logic.modelNode.GetDisplayNode()
            mainDN.RemoveAllViewNodeIDs()
            snapDN.RemoveAllViewNodeIDs()

            # TOP (Before): snapshot vessel + stenosis markers only
            snapDN.AddViewNodeID(viewNode1.GetID())

            # BOTTOM (After Live): main vessel + balloons + stent
            mainDN.AddViewNodeID(viewNode2.GetID())

            for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
                nm = node.GetName()
                dn = node.GetDisplayNode()
                if not dn:
                    continue
                if nm.startswith("BeforeStenosis_"):
                    dn.RemoveAllViewNodeIDs()
                    dn.AddViewNodeID(viewNode1.GetID())
                elif (
                    nm.startswith("BalloonDilate")
                    or nm.endswith("_Ring")
                    or nm.endswith("_wire")
                    or nm == "POT_Balloon"
                    or nm == "CarinaSupport"
                ):
                    dn.RemoveAllViewNodeIDs()
                    dn.AddViewNodeID(viewNode2.GetID())
                elif nm in ("StentModel", "StentModel_Frame"):
                    dn.RemoveAllViewNodeIDs()
                    dn.AddViewNodeID(viewNode2.GetID())

            # ── Labels and orient views ───────────────────────────────────
            viewNode1.SetName("Before (compressed)")
            viewNode1.SetLayoutLabel("Before")
            viewNode2.SetName("After (Live)")
            viewNode2.SetLayoutLabel("After (Live)")
            for vn in [viewNode1, viewNode2]:
                vn.SetBoxVisible(0)
                vn.SetAxisLabelsVisible(1)  # show L R S I labels

            # Get vessel centre for camera focal point
            pts = self.logic.points
            if pts:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                cz = sum(p[2] for p in pts) / len(pts)
                z_range = max(p[2] for p in pts) - min(p[2] for p in pts)
                x_range = max(p[0] for p in pts) - min(p[0] for p in pts)
                cam_dist = max(z_range, x_range) * 2.0
                cam_dist = max(cam_dist, 400)
            else:
                cx, cy, cz = 0, 0, 0
                cam_dist = 600

            for i in range(min(n3d, 2)):
                try:
                    w = layoutMgr.threeDWidget(i)
                    vn = w.mrmlViewNode()
                    vn.SetAxisLabelsVisible(1)
                    renderer = (
                        w.threeDView().renderWindow().GetRenderers().GetFirstRenderer()
                    )
                    cam = renderer.GetActiveCamera()
                    # Anterior view: camera at +Y looking toward vessel centre
                    # Superior (Z) is up — gives standard L/R/S/I orientation
                    cam.SetFocalPoint(cx, cy, cz)
                    cam.SetPosition(cx, cy + cam_dist, cz)
                    cam.SetViewUp(0, 0, 1)
                    renderer.ResetCameraClippingRange()
                    w.threeDView().resetCamera()
                    w.threeDView().renderWindow().Render()
                except Exception:
                    pass

            # Force equal size on both 3D views — delayed to ensure layout is ready
            def _equalSplit():
                try:
                    lm = slicer.app.layoutManager()
                    w0 = lm.threeDWidget(0)
                    w1 = lm.threeDWidget(1)
                    splitter = w0.parent()
                    if hasattr(splitter, "setSizes"):
                        total = splitter.width if splitter.width > 10 else 1000
                        splitter.setSizes([total // 2, total // 2])
                except Exception:
                    pass

            qt.QTimer.singleShot(100, _equalSplit)
            qt.QTimer.singleShot(800, _equalSplit)  # second pass after render
            slicer.app.processEvents()

        except Exception as e:
            import traceback

            print(
                "[Compare] _compareAfterLayoutSwitch error: "
                + str(e)
                + traceback.format_exc()
            )

    def _addStenosisMarker(self, centre, radius, name):
        """Add a dark red narrow cylinder at a stenosis site for the Before view."""
        import vtk, math

        try:
            cyl = vtk.vtkCylinderSource()
            cyl.SetRadius(max(0.5, radius))
            cyl.SetHeight(8.0)
            cyl.SetResolution(20)
            cyl.Update()
            tf = vtk.vtkTransform()
            tf.Translate(centre[0], centre[1], centre[2])
            tf.RotateX(90)
            tpf = vtk.vtkTransformPolyDataFilter()
            tpf.SetTransform(tf)
            tpf.SetInputConnection(cyl.GetOutputPort())
            tpf.Update()
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
            node.SetAndObservePolyData(tpf.GetOutput())
            node.CreateDefaultDisplayNodes()
            dn = node.GetDisplayNode()
            dn.SetColor(0.7, 0.0, 0.0)  # dark red = stenosis
            dn.SetOpacity(0.9)
        except Exception:
            pass

    def _buildLesionTable(self):
        """Populate lesion status table from current findings."""
        findings = getattr(self.logic, "findings", [])
        stenoses = [
            f
            for f in findings
            if "Pancak" in f.get("type", "") or "Compress" in f.get("type", "")
        ]
        if not stenoses:
            self.lesionTableWidget.setVisible(False)
            return
        self.lesionTableWidget.setVisible(True)
        self.lesionTableWidget.setRowCount(len(stenoses))
        pdmap = getattr(self.logic, "preDilationMap", {})
        stented_pts = getattr(self, "_stentedPoints", set())
        for row, f in enumerate(stenoses):
            gi = f.get("pointIdx", -1)
            bi = f.get("branchIdx", 0)
            typ = f.get("type", "")
            minD = f.get("value", 0.0)
            # Max = branch healthy (75th pct)
            if bi < len(self.logic.branches):
                bs, be = self.logic.branches[bi]
                diams = [
                    (
                        self.logic._origDiameters.get(j, self.logic.diameters[j])
                        if hasattr(self.logic, "_origDiameters")
                        else self.logic.diameters[j]
                    )
                    for j in range(bs, be)
                    if j < len(self.logic.diameters) and self.logic.diameters[j] > 1.0
                ]
                orig_diams = (
                    [
                        self.logic._origDiameters.get(j, self.logic.diameters[j])
                        for j in range(bs, be)
                        if j < len(self.logic.diameters)
                        and self.logic._origDiameters.get(j, self.logic.diameters[j])
                        > 1.0
                    ]
                    if hasattr(self.logic, "_origDiameters")
                    else diams
                )
                orig_diams_s = sorted(orig_diams)
                maxD = (
                    orig_diams_s[int(len(orig_diams_s) * 0.75)]
                    if len(orig_diams_s) >= 4
                    else (max(orig_diams_s) if orig_diams_s else 0)
                )
            else:
                maxD = 0.0
            # Status
            if gi in stented_pts:
                status = "✅ Stented"
                color = qt.QColor(39, 174, 96)
            elif gi in pdmap:
                status = "🟡 Pre-dilated"
                color = qt.QColor(142, 68, 173)
            else:
                status = "🔴 Detected"
                color = qt.QColor(192, 57, 43)
            items = [
                qt.QTableWidgetItem(str(row + 1)),
                qt.QTableWidgetItem(f"Br{bi+1} pt{gi}"),
                qt.QTableWidgetItem(typ),
                qt.QTableWidgetItem(f"{minD:.1f}→{maxD:.1f}mm"),
                qt.QTableWidgetItem(status),
            ]
            for col, item in enumerate(items):
                item.setForeground(color)
                item.setData(qt.Qt.UserRole, gi)  # store gi for navigation
                self.lesionTableWidget.setItem(row, col, item)

    def onLesionTableClicked(self, row, col):
        """Navigate IVUS to clicked lesion."""
        item = self.lesionTableWidget.item(row, 0)
        if not item:
            return
        gi = item.data(qt.Qt.UserRole)
        if gi is None or gi < 0:
            return
        try:
            # Switch to All branches
            self.stentBranchCombo.blockSignals(True)
            self.stentBranchCombo.setCurrentIndex(0)
            self.stentBranchCombo.blockSignals(False)
            self.logic.activeBranch = -1
            # getNumPoints() returns traversal length when activeBranch == -1
            self.pointSlider.maximum = self.logic.getNumPoints() - 1
            trav_idx = self.logic.globalToTraversal(gi)
            self.pointSlider.blockSignals(True)
            self.pointSlider.setValue(trav_idx)
            self.pointSlider.blockSignals(False)
            self.updateMeasurementsAtIndex(trav_idx, globalIdx=gi)
        except Exception as e:
            print(f"[LesionTable] Nav error: {e}")

    def onCopyPreDilateStatus(self):
        """Copy balloon pre-dilation summary to clipboard."""
        msg = getattr(self, "_preDilateStatusMsg", self.preDilateStatusLabel.text)
        if msg:
            qt.QApplication.clipboard().setText(msg)
            self.preDilateCopyBtn.setText("✅ Copied!")
            qt.QTimer.singleShot(1200, lambda: self.preDilateCopyBtn.setText("📋 Copy"))

    def onCopyCoordinates(self):
        """Copy current IVUS coordinates + diameter to clipboard in paste-friendly format."""
        try:
            coord_txt = self.coordLabel.text  # e.g. "R -12.1, A 140.1, S 1766.9"
            diam_txt = self.diameterLabel.text  # e.g. "4.64 mm"
            if coord_txt == "--":
                slicer.util.warningDisplay("Navigate to a point first.")
                return
            clip_txt = coord_txt + "  |  Ø " + diam_txt.strip()
            qt.QApplication.clipboard().setText(clip_txt)
            # Flash button green to confirm
            self.copyCoordButton.setStyleSheet(
                "padding: 1px; font-size: 13px; background: #27ae60; color: white; "
                "border-radius: 3px;"
            )
            qt.QTimer.singleShot(
                800,
                lambda: self.copyCoordButton.setStyleSheet(
                    "padding: 1px; font-size: 13px; background: #2c3e50; color: white; "
                    "border-radius: 3px;"
                ),
            )
        except Exception as e:
            import traceback

            traceback.print_exc()

    # ── Manual branch range handlers ──────────────────────────────────────────

    def onApplyManualRanges(self):
        """Parse the manual range fields and call logic.setManualBranchRanges()."""
        import re as _re

        if not hasattr(self, "logic") or not self.logic:
            self._mrStatusLabel.setText("⚠ Run analysis first.")
            self._mrStatusLabel.setStyleSheet("color: #e74c3c; font-size: 11px;")
            return
        if not getattr(self.logic, "points", None):
            self._mrStatusLabel.setText("⚠ No centerline loaded yet.")
            self._mrStatusLabel.setStyleSheet("color: #e74c3c; font-size: 11px;")
            return

        def _parse_coord(txt):
            """Parse 'R x.x, A y.y, S z.z  |  Ø d.d mm' → (x, y, z) or None."""
            txt = txt.strip()
            if not txt:
                return None
            # Extract R, A, S values — tolerant of extra spaces and the Ø suffix
            m = _re.search(
                r"R\s*([-\d.]+)[,\s]+A\s*([-\d.]+)[,\s]+S\s*([-\d.]+)",
                txt,
                _re.IGNORECASE,
            )
            if m:
                return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            return None

        # Map role_key → bi using branchMeta
        # role_key "trunk"       → bi with role in ("trunk", "main") AND bi==0
        # role_key "iliac_left"  → bi with role "iliac_left"
        # role_key "iliac_right" → bi with role "iliac_right"
        # role_key "branch_4"    → 4th branch by bi index (bi=3, 0-based)
        # role_key "branch_5"    → 5th branch by bi index (bi=4, 0-based)
        bm = getattr(self.logic, "branchMeta", {})

        def _bi_for_role(rkey):
            if rkey == "trunk":
                return 0
            if rkey == "iliac_left":
                for bi, m in bm.items():
                    if m.get("role") in ("iliac_left",):
                        return bi
                # fallback: smallest-X tip among main branches
            if rkey == "iliac_right":
                for bi, m in bm.items():
                    if m.get("role") in ("iliac_right",):
                        return bi
            if rkey == "branch_4":
                return 3  # 0-based index
            if rkey == "branch_5":
                return 4
            return None

        _mr_rows = [
            ("Trunk", "trunk"),
            ("Left Iliac", "iliac_left"),
            ("Right Iliac", "iliac_right"),
            ("Branch 4", "branch_4"),
            ("Branch 5", "branch_5"),
        ]

        ranges = {}
        messages = []
        for lbl, rkey in _mr_rows:
            bi = _bi_for_role(rkey)
            if bi is None:
                continue
            start_txt = self._mrStartEdits[rkey].text
            end_txt = self._mrEndEdits[rkey].text
            start_coord = _parse_coord(start_txt)
            end_coord = _parse_coord(end_txt)
            if start_coord is None and end_coord is None:
                continue
            entry = {}
            if start_coord:
                entry["start"] = start_coord
            if end_coord:
                entry["end"] = end_coord
            ranges[bi] = entry
            messages.append(
                f"{lbl} (bi={bi}): "
                + (f"start={start_coord}" if start_coord else "start=auto")
                + "  "
                + (f"end={end_coord}" if end_coord else "end=auto")
            )

        if not ranges:
            self._mrStatusLabel.setText(
                "⚠ No valid coordinates found — paste R x, A y, S z format."
            )
            self._mrStatusLabel.setStyleSheet("color: #e74c3c; font-size: 11px;")
            return

        self.logic.setManualBranchRanges(ranges)

        # Refresh navigator point count so slider reflects new range
        if hasattr(self, "slider"):
            bi = self.logic.activeBranch
            if bi >= 0:
                n = self.logic.getActiveBranchPointCount()
                self.slider.maximum = max(0, n - 1)
                self.slider.value = min(self.slider.value, self.slider.maximum)

        status = "✅ Applied:\n" + "\n".join(messages)
        self._mrStatusLabel.setText(status)
        self._mrStatusLabel.setStyleSheet("color: #27ae60; font-size: 11px;")
        print("[ManualRange] UI applied ranges:", ranges)

    def onClearManualRanges(self):
        """Clear all manual range overrides and reset the text fields."""
        for rkey in self._mrStartEdits:
            self._mrStartEdits[rkey].setText("")
            self._mrEndEdits[rkey].setText("")
        if hasattr(self, "logic") and self.logic:
            self.logic.manualBranchRanges = {}
            # Also wipe any stored gi overrides from branchMeta
            bm = getattr(self.logic, "branchMeta", {})
            for meta in bm.values():
                meta.pop("manualStartGi", None)
                meta.pop("manualEndGi", None)
            # Restore slider range to full branch
            if hasattr(self, "slider") and self.logic.activeBranch >= 0:
                n = self.logic.getActiveBranchPointCount()
                self.slider.maximum = max(0, n - 1)
        self._mrStatusLabel.setText("🗑 Manual ranges cleared.")
        self._mrStatusLabel.setStyleSheet("color: #aaa; font-size: 11px;")



# ── Slicer module-scanner guard ───────────────────────────────────────────────
class vessel_embo_compare_widget_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_embo_compare_widget_mixin"
            parent.hidden = True

