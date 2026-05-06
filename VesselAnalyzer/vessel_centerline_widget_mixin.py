"""
vessel_centerline_widget_mixin.py — VesselAnalyzerWidget centerline-extraction and refinement handlers

Extracted from VesselAnalyzer.py (VesselAnalyzerWidget methods).

Usage in VesselAnalyzer.py
--------------------------
Add to the VesselAnalyzerWidget class definition::

    from vessel_centerline_widget_mixin import CenterlineWidgetMixin

    class VesselAnalyzerWidget(CenterlineWidgetMixin, ScriptedLoadableModuleWidget, VTKObservationMixin):
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


class CenterlineWidgetMixin:
    """Mixin: VesselAnalyzerWidget centerline-extraction and refinement handlers"""
    def onModelSelected(self, node):
        if node:
            polydata = node.GetPolyData()
            if polydata:
                nCells = polydata.GetNumberOfCells()
                nPts = polydata.GetNumberOfPoints()
                self.modelInfoLabel.setText(f"{nCells} polygons, {nPts} points")
                self.modelInfoLabel.setStyleSheet("color: #2ecc71;")
            self.extractButton.enabled = True
        else:
            self.modelInfoLabel.setText("No model selected")
            self.modelInfoLabel.setStyleSheet("color: gray;")
            self.extractButton.enabled = False

    def onAutoDetect(self):
        modelNode = self.modelSelector.currentNode()
        if not modelNode:
            slicer.util.errorDisplay("Please select a Vessel Model in Step 1 first.")
            return

        self.extractStatusLabel.setText("Auto-detecting endpoints...")
        self.extractStatusLabel.setStyleSheet("color: #e67e22;")
        slicer.app.processEvents()
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)

        try:
            # Create or reuse endpoints node
            endpointsNode = self.endpointsSelector.currentNode()
            if not endpointsNode:
                endpointsNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsFiducialNode", "Endpoints"
                )
                endpointsNode.CreateDefaultDisplayNodes()
                self.endpointsSelector.setCurrentNode(endpointsNode)
            else:
                endpointsNode.RemoveAllControlPoints()

            # Use VMTK to find open boundaries
            try:
                endpoints = self.logic.autoDetectEndpoints(modelNode)
            except Exception as e:
                print(f"VMTK auto-detect failed: {e}, trying fallback...")
                endpoints = self.logic._autoDetectEndpointsFallback(modelNode)

            if not endpoints:
                slicer.util.errorDisplay(
                    "Auto-detect failed. Please place endpoints manually."
                )
                self.extractStatusLabel.setText("Auto-detect failed — place manually")
                self.extractStatusLabel.setStyleSheet("color: #e74c3c;")
                return

            # Add detected endpoints to markup node
            for pt in endpoints:
                endpointsNode.AddControlPoint(vtk.vtkVector3d(pt))

            n = len(endpoints)
            self.endpointCountLabel.setText(f"{n} endpoints detected")
            self.endpointCountLabel.setStyleSheet("color: #8e44ad; font-weight: bold;")
            self.extractStatusLabel.setText(f"✔ {n} endpoints auto-detected!")
            self.extractStatusLabel.setStyleSheet("color: #2ecc71;")

        except Exception as e:
            slicer.util.errorDisplay(f"Auto-detect error: {str(e)}")
            self.extractStatusLabel.setText(f"Error: {str(e)}")
            self.extractStatusLabel.setStyleSheet("color: #e74c3c;")
            import traceback

            print(traceback.format_exc())
        finally:
            slicer.app.restoreOverrideCursor()

    def onPlaceEndpoints(self):
        # Create or get endpoints node
        endpointsNode = self.endpointsSelector.currentNode()
        if not endpointsNode:
            endpointsNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "Endpoints"
            )
            endpointsNode.CreateDefaultDisplayNodes()
            self.endpointsSelector.setCurrentNode(endpointsNode)

        # Use persistent=False so each click places ONE point then auto-releases
        # This matches Extract Centerline behavior exactly
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(endpointsNode.GetID())

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        interactionNode.SetPlaceModePersistence(0)  # 0 = place one point then release

        self.extractStatusLabel.setText(
            "Click on a vessel opening to place one endpoint..."
        )
        self.extractStatusLabel.setStyleSheet("color: #e67e22;")
        self.cancelPlaceButton.setVisible(True)

        # Observer to auto-update count and re-enable button after each placement
        if not hasattr(self, "_endpointObserver") or self._endpointObserver is None:
            self._endpointObserver = endpointsNode.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onEndpointAdded
            )

        # Also observe interaction mode change to hide cancel button when done
        if (
            not hasattr(self, "_interactionObserver")
            or self._interactionObserver is None
        ):
            self._interactionObserver = interactionNode.AddObserver(
                vtk.vtkCommand.ModifiedEvent, self.onInteractionModeChanged
            )

    def onCancelPlace(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
        self.cancelPlaceButton.setVisible(False)
        self._cleanupObservers()
        n = 0
        endpointsNode = self.endpointsSelector.currentNode()
        if endpointsNode:
            n = endpointsNode.GetNumberOfControlPoints()
        self.extractStatusLabel.setText(f"Done. {n} endpoint(s) placed.")
        self.extractStatusLabel.setStyleSheet(
            "color: #2ecc71;" if n >= 2 else "color: #e74c3c;"
        )

    def onInteractionModeChanged(self, caller, event):
        # Auto-hide cancel button when placement mode exits automatically
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if interactionNode.GetCurrentInteractionMode() != interactionNode.Place:
            self.cancelPlaceButton.setVisible(False)
            self._cleanupObservers()
            n = 0
            endpointsNode = self.endpointsSelector.currentNode()
            if endpointsNode:
                n = endpointsNode.GetNumberOfControlPoints()
            self.extractStatusLabel.setText(
                f"{n} endpoint(s) placed. Click again to add more."
            )
            self.extractStatusLabel.setStyleSheet(
                "color: #2ecc71;" if n >= 2 else "color: #e67e22;"
            )

    def _cleanupObservers(self):
        if hasattr(self, "_interactionObserver") and self._interactionObserver:
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.RemoveObserver(self._interactionObserver)
            self._interactionObserver = None

    def onEndpointAdded(self, caller, event):
        n = caller.GetNumberOfControlPoints()
        self.endpointCountLabel.setText(f"{n} endpoint{'s' if n != 1 else ''} placed")
        self.endpointCountLabel.setStyleSheet(
            "color: #2ecc71;" if n >= 2 else "color: #e67e22;"
        )

    def onClearEndpoints(self):
        node = self.endpointsSelector.currentNode()
        if node:
            node.RemoveAllControlPoints()
            self.endpointCountLabel.setText("0 endpoints placed")
            self.endpointCountLabel.setStyleSheet("color: gray;")

    def onExtractCenterline(self):
        modelNode = self.modelSelector.currentNode()
        endpointsNode = self.endpointsSelector.currentNode()

        print(f"[ExtractCenterline] modelNode={modelNode}")
        print(f"[ExtractCenterline] endpointsNode={endpointsNode}")
        if endpointsNode:
            print(
                f"[ExtractCenterline] n_endpoints={endpointsNode.GetNumberOfControlPoints()}"
            )

        if not modelNode:
            slicer.util.errorDisplay("Please select a vessel model first.")
            return
        if not endpointsNode or endpointsNode.GetNumberOfControlPoints() < 2:
            slicer.util.errorDisplay(
                "Please place at least 2 endpoints on the vessel openings."
            )
            return

        # Prevent double-clicks while running
        self.extractButton.setEnabled(False)
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)

        def _status(msg, color="#e67e22"):
            self.extractStatusLabel.setText(msg)
            self.extractStatusLabel.setStyleSheet(f"color: {color};")
            slicer.app.processEvents()  # flush repaint so label updates visibly

        _status("Starting centerline extraction...")

        # ── Resolve per_branch_centerline_pipeline.py location ────────────
        import os as _pbos, sys as _pbsys

        def _find_pb_dir():
            try:
                d = _pbos.path.dirname(_pbos.path.abspath(__file__))
                if _pbos.path.isfile(
                    _pbos.path.join(d, "per_branch_centerline_pipeline.py")
                ):
                    return d
            except Exception:
                pass
            for _p in _pbsys.path:
                if _pbos.path.isfile(
                    _pbos.path.join(_p, "per_branch_centerline_pipeline.py")
                ):
                    return _p
            try:
                import slicer as _sl

                factory = _sl.app.moduleManager().factoryManager()
                path = factory.moduleDescription("VesselAnalyzer").get("path", "")
                if path:
                    d = _pbos.path.dirname(path)
                    if _pbos.path.isfile(
                        _pbos.path.join(d, "per_branch_centerline_pipeline.py")
                    ):
                        return d
            except Exception:
                pass
            return None

        _pb_dir = _find_pb_dir()
        if _pb_dir is None:
            self.extractButton.setEnabled(True)
            slicer.app.restoreOverrideCursor()
            _status("per_branch_centerline_pipeline.py not found", "#e74c3c")
            slicer.util.errorDisplay(
                "per_branch_centerline_pipeline.py not found.\n"
                f"Place it next to VesselAnalyzer.py"
            )
            return

        if _pb_dir not in _pbsys.path:
            _pbsys.path.insert(0, _pb_dir)
        if "per_branch_centerline_pipeline" in _pbsys.modules:
            del _pbsys.modules["per_branch_centerline_pipeline"]

        from per_branch_centerline_pipeline import PerBranchCenterlinePipeline

        # ── Run pipeline on the main thread via run_with_progress() ─────────
        #
        # VTK/VMTK/MRML are NOT thread-safe in Slicer.  run_with_progress()
        # accepts status_cb and process_events_cb so Qt can repaint between
        # each heavy phase without any background threading.
        #
        # v288 pipeline strategy:
        #   Phase 1: fast 3-endpoint VMTK solve (inlet + 2 iliac tips only)
        #            to locate the true anatomical bifurcation point.
        #            Falls back to geometric estimate if VMTK fails.
        #   Phase 2: per-branch solves — iliac branches use tip-inward
        #            Dijkstra (tip → validated wall seed, then reversed);
        #            renal/side branches use trunk-inlet → tip solve.
        #            Wall seeds validated by multi-Z ray scan with
        #            wall_dist gating; lumen-fraction fallback if all fail.
        #   Phase 3: branch trim at first geometrically diverging point
        #            (cKDTree shared-point walk) rather than nearest-to-bif_pt.
        #   Phase 4: tangent-preserving junction snap (smoothstep blend).
        #   Phase 5: optional light refinement (disabled by default).

        try:
            pipeline = PerBranchCenterlinePipeline(
                logic=self.logic,
                widget=self,
                model_node=modelNode,
                refine=False,
            )
            centerlineNode = pipeline.run_with_progress(
                endpoints_node=endpointsNode,
                status_cb=_status,
                process_events_cb=slicer.app.processEvents,
            )

            if centerlineNode:
                self.centerlineSelector.setCurrentNode(centerlineNode)
                _status("Centerline extracted successfully!", "#2ecc71")
                slicer.util.infoDisplay(
                    "Centerline extracted! Now click 'Load for IVUS Navigation'."
                )
            else:
                _status("Extraction failed — no centerline returned", "#e74c3c")

        except Exception as e:
            import traceback

            _status(f"Error: {str(e)}", "#e74c3c")
            print(traceback.format_exc())
        finally:
            slicer.app.restoreOverrideCursor()
            self.extractButton.setEnabled(True)

    def onDrawCenterlineManually(self):
        """
        Enter persistent snap-branching draw mode.

        Single button press enters the mode.  Every subsequent click in
        any 2-D or 3-D view places a point.  The algorithm:

          • First click ever  → start branch-0 (trunk).
          • Click within _snapRadiusMm of any existing node → end current
            branch at that shared node and start a new branch from it
            (true topological fork, no backtracking needed).
          • Any other click   → extend the active branch normally.
          • Stop Drawing      → exit placement, keep all geometry.

        All state lives in:
            self._manualCurves  – one vtkMRMLMarkupsCurveNode per branch
            self._allNodes      – flat list of {pos, curve, idx} records
            self._activeCurve   – the branch currently being extended
            self._drawProxy     – invisible proxy curve that receives clicks
        """
        import vtk

        # ── helpers ────────────────────────────────────────────────────────

        COLOURS = [
            (1.0, 0.6, 0.0),  # orange  – trunk
            (0.2, 0.8, 1.0),  # cyan
            (0.4, 1.0, 0.4),  # green
            (1.0, 0.4, 0.8),  # pink
            (1.0, 1.0, 0.3),  # yellow
            (0.8, 0.4, 1.0),  # violet
        ]

        def _dist(a, b):
            return vtk.vtkMath.Distance2BetweenPoints(a, b) ** 0.5

        def _nearest_node(pos):
            best, best_d = None, self._snapRadiusMm
            for n in self._allNodes:
                d = _dist(pos, n["pos"])
                if d < best_d:
                    best, best_d = n, d
            return best

        def _make_branch(start_pos=None):
            """Create a new curve node and register it.  Returns the node."""
            idx = len(self._manualCurves)
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode", f"ManualCenterline_{idx + 1}"
            )
            node.CreateDefaultDisplayNodes()
            node.SetCurveTypeToLinear()
            dn = node.GetDisplayNode()
            if dn:
                c = COLOURS[idx % len(COLOURS)]
                dn.SetColor(*c)
                dn.SetSelectedColor(*c)
                dn.SetLineThickness(0.5)
                dn.SetGlyphScale(2.5)
                dn.SetVisibility(1)
            self._manualCurves.append(node)
            self._activeCurve = node

            if start_pos is not None:
                pt_idx = node.AddControlPointWorld(vtk.vtkVector3d(*start_pos))
                self._allNodes.append(
                    {"pos": tuple(start_pos), "curve": node, "idx": pt_idx}
                )

            return node

        def _center_in_lumen(pos):
            """
            Project pos toward the lumen center using the vessel surface model.

            Strategy: find nearest surface point + its inward normal, then walk
            inward in 0.5 mm steps and keep the step that is farthest from the
            surface (i.e. deepest inside the lumen).  Falls back to the original
            pos if no model is loaded or the model has no poly data.
            """
            modelNode = self.modelSelector.currentNode()
            if modelNode is None:
                return pos
            polyData = modelNode.GetPolyData()
            if polyData is None or polyData.GetNumberOfPoints() == 0:
                return pos

            locator = vtk.vtkPointLocator()
            locator.SetDataSet(polyData)
            locator.BuildLocator()

            # nearest surface point
            pid = locator.FindClosestPoint(pos)
            surf_pt = [0.0, 0.0, 0.0]
            polyData.GetPoint(pid, surf_pt)

            # inward normal (pointing away from surface, i.e. into lumen)
            normals = polyData.GetPointData().GetNormals()
            if normals:
                normal = list(normals.GetTuple(pid))
            else:
                bounds = polyData.GetBounds()
                centroid = [
                    0.5 * (bounds[0] + bounds[1]),
                    0.5 * (bounds[2] + bounds[3]),
                    0.5 * (bounds[4] + bounds[5]),
                ]
                normal = [centroid[i] - surf_pt[i] for i in range(3)]
            vtk.vtkMath.Normalize(normal)

            # walk inward, keep point farthest from surface
            step_mm = 0.5
            max_steps = 30
            best_pt = list(pos)
            best_dist_sq = 0.0

            for i in range(1, max_steps + 1):
                test = [surf_pt[j] + normal[j] * step_mm * i for j in range(3)]
                pid2 = locator.FindClosestPoint(test)
                p2 = [0.0, 0.0, 0.0]
                polyData.GetPoint(pid2, p2)
                d2 = vtk.vtkMath.Distance2BetweenPoints(test, p2)
                if d2 > best_dist_sq:
                    best_dist_sq = d2
                    best_pt = test

            return tuple(best_pt)

        def _add_point_to_active(pos):
            centered = _center_in_lumen(pos)
            pt_idx = self._activeCurve.AddControlPointWorld(vtk.vtkVector3d(*centered))
            self._allNodes.append(
                {"pos": centered, "curve": self._activeCurve, "idx": pt_idx}
            )

        # ── proxy-point callback ────────────────────────────────────────────

        def _on_proxy_point(caller, event):
            """Fired each time the user clicks and places a proxy point."""
            n_proxy = caller.GetNumberOfControlPoints()
            if n_proxy == 0:
                return

            # Read position of the just-placed proxy point
            raw = [0.0, 0.0, 0.0]
            caller.GetNthControlPointPositionWorld(n_proxy - 1, raw)
            pos = tuple(raw)

            # CRITICAL: wipe proxy immediately so next click fires a fresh event
            caller.RemoveAllControlPoints()

            # Re-assert placement mode — Slicer may drop it after each point
            inode = slicer.app.applicationLogic().GetInteractionNode()
            inode.SetPlaceModePersistence(1)
            inode.SetCurrentInteractionMode(inode.Place)

            snap = _nearest_node(pos)

            if self._activeCurve is None:
                # ── very first click (or after a fork left activeCurve=None) ─
                start = snap["pos"] if snap else _center_in_lumen(pos)
                _make_branch(start_pos=start)
                self._update_manual_cl_ui()
                return

            # ── subsequent clicks ─────────────────────────────────────────
            last_idx = self._activeCurve.GetNumberOfControlPoints() - 1
            last_pos = [0.0, 0.0, 0.0]
            self._activeCurve.GetNthControlPointPositionWorld(last_idx, last_pos)

            if snap and _dist(last_pos, snap["pos"]) > 1e-3:
                # Clicked near an existing node that is NOT the tip of this
                # branch → end current branch, fork from the snapped node.
                self._activeCurve = None
                _make_branch(start_pos=snap["pos"])
            else:
                # Normal extension — project to lumen center
                _add_point_to_active(pos)

            self._update_manual_cl_ui()

        # ── set up the proxy node ───────────────────────────────────────────
        # IMPORTANT: re-register the observer every time Draw is pressed so the
        # closure always refers to the current call's helpers.  Remove any
        # stale observer first.

        if self._drawProxy is not None and slicer.mrmlScene.IsNodePresent(
            self._drawProxy
        ):
            if self._drawProxyObserver is not None:
                try:
                    self._drawProxy.RemoveObserver(self._drawProxyObserver)
                except Exception:
                    pass
            self._drawProxy.RemoveAllControlPoints()
        else:
            proxy = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode", "DrawProxy_DO_NOT_USE"
            )
            proxy.CreateDefaultDisplayNodes()
            dn = proxy.GetDisplayNode()
            if dn:
                dn.SetVisibility(0)  # invisible — purely a click receiver
            self._drawProxy = proxy

        # Fresh observer using this call's closure
        self._drawProxyObserver = self._drawProxy.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, _on_proxy_point
        )

        # ── activate persistent placement on the proxy ──────────────────────

        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsCurveNode")
        selectionNode.SetActivePlaceNodeID(self._drawProxy.GetID())

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        # ── UI ─────────────────────────────────────────────────────────────

        self.drawCenterlineButton.setEnabled(False)
        self.stopDrawCenterlineButton.setVisible(True)
        self.useManualCenterlineButton.setEnabled(False)
        self.clearManualCenterlineButton.setEnabled(bool(self._allNodes))
        self.refineCenterlineButton.setEnabled(False)
        self.manualCLStatusLabel.setText(
            "Click trunk first, then branches — click near any existing point to fork"
        )
        self.manualCLStatusLabel.setStyleSheet("color: #e67e22; font-size: 11px;")

    def _update_manual_cl_ui(self):
        """Refresh status label and button states after each placed point."""
        total = sum(c.GetNumberOfControlPoints() for c in self._manualCurves)
        n_branches = len(self._manualCurves)
        active_pts = (
            self._activeCurve.GetNumberOfControlPoints() if self._activeCurve else 0
        )
        self.manualCLStatusLabel.setText(
            f"Branch {n_branches}: {active_pts} pts | total {total} pts "
            f"— click near existing point to fork a new branch"
        )
        if total >= 2:
            self.useManualCenterlineButton.setEnabled(True)
            self.clearManualCenterlineButton.setEnabled(True)
            self.refineCenterlineButton.setEnabled(True)
            self.autoTuneButton.setEnabled(True)

    def onStopDrawCenterline(self):
        """Exit placement mode; keep all branch geometry intact."""
        # Exit placement mode
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

        # Remember last placed point for informational purposes
        if self._activeCurve and self._activeCurve.GetNumberOfControlPoints() > 0:
            p = [0.0, 0.0, 0.0]
            self._activeCurve.GetNthControlPointPositionWorld(
                self._activeCurve.GetNumberOfControlPoints() - 1, p
            )
            self._lastPlacedPoint = tuple(p)

        # Detach proxy observer (keep proxy node alive for re-entry)
        if self._drawProxyObserver is not None and self._drawProxy is not None:
            try:
                self._drawProxy.RemoveObserver(self._drawProxyObserver)
            except Exception:
                pass
        self._drawProxyObserver = None

        # Remove legacy per-curve observer if any
        if self._activeCurveObserver is not None and self._activeCurve is not None:
            try:
                self._activeCurve.RemoveObserver(self._activeCurveObserver)
            except Exception:
                pass
        self._activeCurveObserver = None
        self._activeCurve = None

        self.stopDrawCenterlineButton.setVisible(False)
        self.drawCenterlineButton.setEnabled(True)

        total = sum(c.GetNumberOfControlPoints() for c in self._manualCurves)
        n_branches = len(self._manualCurves)
        if total >= 2:
            self.useManualCenterlineButton.setEnabled(True)
            self.clearManualCenterlineButton.setEnabled(True)
            self.refineCenterlineButton.setEnabled(True)
            self.autoTuneButton.setEnabled(True)
            self.manualCLStatusLabel.setText(
                f"\u2713 {n_branches} branch(es), {total} pts — "
                f"click 'Draw Centerline' to add more branches, or 'Use This Centerline'"
            )
            self.manualCLStatusLabel.setStyleSheet("color: #2ecc71; font-size: 11px;")
        else:
            self.useManualCenterlineButton.setEnabled(False)
            self.refineCenterlineButton.setEnabled(False)
            self.manualCLStatusLabel.setText("Need at least 2 points total")
            self.manualCLStatusLabel.setStyleSheet("color: #e74c3c; font-size: 11px;")
        self._updateSeedQualityPanel()

    def onUseManualCenterline(self):
        """Merge all strokes into a single resampled curve and set as active centerline."""
        import vtk
        import numpy as np

        total = sum(c.GetNumberOfControlPoints() for c in self._manualCurves)
        if total < 2:
            slicer.util.errorDisplay("Draw at least 2 points first.")
            return

        # Collect all points from all strokes in order
        all_pts = []
        for curve in self._manualCurves:
            for i in range(curve.GetNumberOfControlPoints()):
                p = [0.0, 0.0, 0.0]
                curve.GetNthControlPointPositionWorld(i, p)
                all_pts.append(p)

        # Build or reuse the combined output node
        combined = slicer.mrmlScene.GetFirstNodeByName("ManualCenterline_Combined")
        if not combined:
            combined = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode", "ManualCenterline_Combined"
            )
            combined.CreateDefaultDisplayNodes()
        combined.SetCurveTypeToLinear()
        dn = combined.GetDisplayNode()
        if dn:
            dn.SetColor(1.0, 0.6, 0.0)
            dn.SetSelectedColor(1.0, 0.6, 0.0)
            dn.SetLineThickness(0.5)
            dn.SetGlyphScale(2.0)
            dn.SetVisibility(1)

        combined.RemoveAllControlPoints()
        for p in all_pts:
            combined.AddControlPointWorld(vtk.vtkVector3d(p[0], p[1], p[2]))

        # Resample to 1 mm spacing
        try:
            combined.ResampleCurveWorld(1.0)
            print(
                f"[ManualCenterline] Combined+resampled: {combined.GetNumberOfControlPoints()} pts"
            )
        except Exception as e:
            print(f"[ManualCenterline] Resample warning: {e}")

        self.centerlineSelector.setCurrentNode(combined)
        self.manualCLStatusLabel.setText(
            f"\u2713 {len(self._manualCurves)} stroke(s) merged "
            f"({combined.GetNumberOfControlPoints()} pts) — click 'Load for IVUS Navigation'"
        )
        self.manualCLStatusLabel.setStyleSheet("color: #2ecc71; font-size: 11px;")
        print(f"[ManualCenterline] Set as active: ManualCenterline_Combined")

    def onClearManualCenterline(self):
        """Remove all branches, proxy node, and reset to blank state."""
        # Exit placement mode if active
        try:
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
        except Exception:
            pass

        # Detach proxy observer and remove proxy node
        if self._drawProxyObserver is not None and self._drawProxy is not None:
            try:
                self._drawProxy.RemoveObserver(self._drawProxyObserver)
            except Exception:
                pass
        self._drawProxyObserver = None
        if self._drawProxy is not None and slicer.mrmlScene.IsNodePresent(
            self._drawProxy
        ):
            try:
                slicer.mrmlScene.RemoveNode(self._drawProxy)
            except Exception:
                pass
        self._drawProxy = None

        # Remove legacy per-curve observer if any
        if self._activeCurveObserver is not None and self._activeCurve is not None:
            try:
                self._activeCurve.RemoveObserver(self._activeCurveObserver)
            except Exception:
                pass
        self._activeCurveObserver = None
        self._activeCurve = None
        self._lastPlacedPoint = None
        self._allNodes = []

        # Remove all branch nodes from scene
        for curve in self._manualCurves:
            try:
                slicer.mrmlScene.RemoveNode(curve)
            except Exception:
                pass
        self._manualCurves = []

        # Remove combined node if it exists
        combined = slicer.mrmlScene.GetFirstNodeByName("ManualCenterline_Combined")
        if combined:
            if self.centerlineSelector.currentNode() == combined:
                self.centerlineSelector.setCurrentNode(None)
            slicer.mrmlScene.RemoveNode(combined)

        # Reset UI
        self.drawCenterlineButton.setEnabled(True)
        self.stopDrawCenterlineButton.setVisible(False)
        self.useManualCenterlineButton.setEnabled(False)
        self.clearManualCenterlineButton.setEnabled(False)
        self.refineCenterlineButton.setEnabled(False)
        self.autoTuneButton.setEnabled(False)
        self.autoTuneStatusLabel.setText("")
        self.manualCLStatusLabel.setText(
            "Cleared — draw trunk, then fork branches by clicking near bifurcation points"
        )
        self.manualCLStatusLabel.setStyleSheet("color: gray; font-size: 11px;")
        self.refineMetricsLabel.setText("")
        self._updateSeedQualityPanel()

    # ------------------------------------------------------------------
    #  Seed quality helpers  (v269 SeedQualityPreflight)
    # ------------------------------------------------------------------
    _SQ_MIN_ARC_LEN = 20.0  # mm
    _SQ_MIN_LATERAL = 8.0  # mm — must exit junction zone
    _SQ_MIN_FREE_PTS = 2  # movable (non-endpoint) points

    def _computeSeedQuality(self, curve_node):
        """Classify one ManualCenterline curve as 'good', 'weak', or 'invalid'.

        Returns a dict with keys: quality, arc_len_mm, tip_lateral_mm,
        n_free, reason.
        """
        import numpy as np

        n = curve_node.GetNumberOfControlPoints()
        pts = []
        for i in range(n):
            p = [0.0, 0.0, 0.0]
            curve_node.GetNthControlPointPosition(i, p)
            pts.append(np.array(p))

        arc = sum(
            float(np.linalg.norm(pts[i] - pts[i - 1])) for i in range(1, len(pts))
        )

        if len(pts) >= 2:
            dep = pts[1] - pts[0]
            dep_norm = np.linalg.norm(dep)
            dep_dir = dep / dep_norm if dep_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
            tip_lateral = float(np.dot(pts[-1] - pts[0], dep_dir))
        else:
            tip_lateral = 0.0

        n_free = max(0, n - 2)
        reasons = []
        if arc < self._SQ_MIN_ARC_LEN:
            reasons.append(f"arc {arc:.1f} mm < {self._SQ_MIN_ARC_LEN} mm")
        if tip_lateral < self._SQ_MIN_LATERAL:
            reasons.append(f"lateral {tip_lateral:.1f} mm < {self._SQ_MIN_LATERAL} mm")

        if reasons:
            quality = "invalid"
        elif n_free < self._SQ_MIN_FREE_PTS:
            quality = "weak"
            reasons.append(f"only {n_free} free pt(s)")
        else:
            quality = "good"

        return {
            "quality": quality,
            "arc_len_mm": arc,
            "tip_lateral_mm": tip_lateral,
            "n_free": n_free,
            "reason": ", ".join(reasons),
        }

    def _updateSeedQualityPanel(self):
        """Rebuild the per-curve seed-quality status label."""
        if not hasattr(self, "seedQualityLabel"):
            return
        if not self._manualCurves:
            self.seedQualityLabel.setText("")
            return
        ICON = {"good": "\u2705", "weak": "\u26a0\ufe0f", "invalid": "\u274c"}
        COLOR = {"good": "#27ae60", "weak": "#e67e22", "invalid": "#c0392b"}
        lines = []
        for curve in self._manualCurves:
            sq = self._computeSeedQuality(curve)
            q = sq["quality"]
            name = curve.GetName()
            arc = sq["arc_len_mm"]
            lat = sq["tip_lateral_mm"]
            free = sq["n_free"]
            if q == "good":
                detail = f"arc {arc:.1f} mm  lat {lat:.1f} mm  free {free}"
            else:
                detail = sq["reason"]
                if q == "invalid":
                    detail += " \u2014 extend into lumen"
            lines.append(
                f'<span style="color:{COLOR[q]}">{ICON[q]} <b>{name}</b>: {detail}</span>'
            )
        self.seedQualityLabel.setText("<br>".join(lines))

    def _densifyCurve(self, curve_node, spacing_mm=5.0, max_pts=15):
        """Resample a ManualCenterline stroke to evenly-spaced points.

        spacing_mm : target point spacing (default 5mm — balances coverage
                     vs optimizer cost; 15-pt cap prevents runaway on long strokes)
        max_pts    : hard cap on total points including endpoints

        Endpoints are preserved exactly.  Interior points become free DOFs
        for _refineSingleCurve, activating the Laplacian smoothness term
        (requires n>=7) across the full stroke.
        """
        import numpy as np
        import vtk

        n = curve_node.GetNumberOfControlPoints()
        if n < 2:
            return

        raw = []
        for i in range(n):
            p = [0.0, 0.0, 0.0]
            curve_node.GetNthControlPointPositionWorld(i, p)
            raw.append(np.array(p))

        dists = [0.0]
        for i in range(1, len(raw)):
            dists.append(dists[-1] + float(np.linalg.norm(raw[i] - raw[i - 1])))
        total_arc = dists[-1]

        if total_arc < spacing_mm:
            return  # too short — leave as-is

        # number of segments: respect both spacing and max_pts cap
        n_segs = min(max_pts - 1, max(1, int(round(total_arc / spacing_mm))))
        targets = [total_arc * k / n_segs for k in range(n_segs + 1)]

        new_pts = []
        seg = 0
        for t in targets:
            while seg < len(dists) - 2 and dists[seg + 1] < t:
                seg += 1
            d0, d1 = dists[seg], dists[seg + 1]
            span = d1 - d0
            alpha = (t - d0) / span if span > 1e-9 else 0.0
            new_pts.append(raw[seg] * (1.0 - alpha) + raw[seg + 1] * alpha)

        new_pts[0] = raw[0].copy()
        new_pts[-1] = raw[-1].copy()

        curve_node.RemoveAllControlPoints()
        for pt in new_pts:
            curve_node.AddControlPointWorld(vtk.vtkVector3d(pt[0], pt[1], pt[2]))

        print(
            f"[Densify] '{curve_node.GetName()}': {n} pts → {len(new_pts)} pts "
            f"(spacing≈{total_arc/n_segs:.1f} mm, arc={total_arc:.1f} mm)"
        )

    def onRefineCenterline(self):
        """Preflight seed quality, then refine each branch independently."""
        import numpy as np

        if not self._manualCurves:
            slicer.util.errorDisplay("No manual centerline to refine.")
            return
        total = sum(c.GetNumberOfControlPoints() for c in self._manualCurves)
        if total < 2:
            slicer.util.errorDisplay("Need at least 2 points to refine.")
            return

        # ── preflight ─────────────────────────────────────────────────
        qualities = {c: self._computeSeedQuality(c) for c in self._manualCurves}
        valid_curves = [c for c, sq in qualities.items() if sq["quality"] != "invalid"]

        if not valid_curves:
            msg = "All curves are too short or confined to the junction zone.\n\n"
            for c, sq in qualities.items():
                msg += f"  {c.GetName()}: {sq['reason']}\n"
            msg += "\nExtend each curve at least 10\u201320 mm into the branch lumen."
            slicer.util.errorDisplay(msg)
            return

        # ── snapshot BEFORE ───────────────────────────────────────────
        before_metrics = [self._computeCurveMetrics(c) for c in valid_curves]

        # ── densify strokes before refining ──────────────────────────
        # Resamples to max 15 pts at ~5mm spacing so the optimizer has
        # enough free DOFs to centre the full stroke without ray-cast overload.
        for curve in self._manualCurves:
            if qualities[curve]["quality"] != "invalid":
                self._densifyCurve(curve, spacing_mm=5.0, max_pts=15)
        # recompute seed quality after densification (n_free has changed)
        qualities = {c: self._computeSeedQuality(c) for c in self._manualCurves}

        # ── refine (quality-aware params) ─────────────────────────────
        status_lines = []
        for curve in self._manualCurves:
            sq = qualities[curve]
            q = sq["quality"]
            name = curve.GetName()
            if q == "invalid":
                print(f"[SeedQuality] SKIP {name}: {sq['reason']}")
                status_lines.append(f"\u274c {name}: skipped \u2014 {sq['reason']}")
                status_lines.append(
                    "   \u2192 Extend this curve further into the branch lumen "
                    "(add a point ~10\u201320 mm laterally)."
                )
                continue
            if q == "weak":
                print(f"[SeedQuality] WEAK {name}: {sq['reason']} — reduced w_j & lr")
                self._refineSingleCurve(
                    curve, params={"w_j": 2.5, "lr": 0.056, "n_iters": 15}
                )
                status_lines.append(
                    f"\u26a0\ufe0f {name}: refined with reduced params ({sq['reason']})"
                )
            else:
                print(
                    f"[SeedQuality] GOOD {name}: arc={sq['arc_len_mm']:.1f} mm "
                    f"lat={sq['tip_lateral_mm']:.1f} mm free={sq['n_free']}"
                )
                self._refineSingleCurve(curve, params={})
                status_lines.append(f"\u2705 {name}: refined normally")

        # ── snapshot AFTER ────────────────────────────────────────────
        after_metrics = [self._computeCurveMetrics(c) for c in valid_curves]

        # ── aggregate across branches ─────────────────────────────────────
        def _agg(metrics_list, key):
            vals = [m[key] for m in metrics_list if m.get(key) is not None]
            return np.mean(vals) if vals else None

        def _fmt(v, decimals=2):
            return f"{v:.{decimals}f}" if v is not None else "n/a"

        def _arrow(before, after, lower_is_better=True):
            if before is None or after is None:
                return "→"
            improved = (after < before) if lower_is_better else (after > before)
            return "✓" if improved else "△"

        keys = [
            ("mean_curv", "Mean κ", True),
            ("std_curv", "Std κ", True),
            ("tortuosity", "Tortuosity", True),
            ("length_mm", "Length (mm)", False),
            ("wall_dist_mean", "Wall dist mean", False),
            ("wall_dist_std", "Wall dist std", True),
        ]

        lines = ["── Refinement metrics ──────────────────"]
        lines.append(f"{'Metric':<18} {'Before':>7} {'After':>7}  ")
        lines.append("─" * 38)
        for key, label, lib in keys:
            b = _agg(before_metrics, key)
            a = _agg(after_metrics, key)
            arr = _arrow(b, a, lib)
            lines.append(f"{label:<18} {_fmt(b):>7} {_fmt(a):>7}  {arr}")

        # branch angle (only meaningful if >1 branch)
        ba_before = [
            m["branch_angle_deg"]
            for m in before_metrics
            if m.get("branch_angle_deg") is not None
        ]
        ba_after = [
            m["branch_angle_deg"]
            for m in after_metrics
            if m.get("branch_angle_deg") is not None
        ]
        if ba_before and ba_after:
            ba_b = np.mean(ba_before)
            ba_a = np.mean(ba_after)
            arr = _arrow(abs(ba_b - 90), abs(ba_a - 90), lower_is_better=True)
            lines.append(
                f"{'Branch angle°':<18} {_fmt(ba_b):>7} {_fmt(ba_a):>7}  {arr}"
            )

        report = "\n".join(lines)
        print(f"[ManualCenterline] Refinement report:\n{report}")
        self.refineMetricsLabel.setText(report)

        n_skipped = len(self._manualCurves) - len(valid_curves)
        skip_note = f"  ({n_skipped} skipped \u2014 too short)" if n_skipped else ""
        total_out = sum(c.GetNumberOfControlPoints() for c in valid_curves)
        self.manualCLStatusLabel.setText(
            f"\u2728 Refined {len(valid_curves)} branch(es) ({total_out} pts){skip_note}"
            f" \u2014 click 'Use This Centerline' to load\n" + "\n".join(status_lines)
        )
        self.manualCLStatusLabel.setStyleSheet("color: #8e44ad; font-size: 11px;")

    def onAutoTune(self):
        """Preflight seed quality, then run _refineAutoTune on good/weak curves."""
        if not self._manualCurves:
            slicer.util.errorDisplay("No manual centerline to auto-tune.")
            return

        # ── preflight ─────────────────────────────────────────────────
        qualities = {c: self._computeSeedQuality(c) for c in self._manualCurves}
        valid_curves = [c for c, sq in qualities.items() if sq["quality"] != "invalid"]

        if not valid_curves:
            msg = "All curves are too short or confined to the junction zone.\n\n"
            for c, sq in qualities.items():
                msg += f"  {c.GetName()}: {sq['reason']}\n"
            msg += "\nExtend each curve at least 10\u201320 mm into the branch lumen."
            slicer.util.errorDisplay(msg)
            return

        n_skipped = len(self._manualCurves) - len(valid_curves)
        skip_note = f" ({n_skipped} invalid curve(s) skipped)" if n_skipped else ""

        self.autoTuneButton.setEnabled(False)
        self.autoTuneStatusLabel.setText(
            f"Running auto-tuner (20 trials){skip_note}\u2026 this may take ~30-60 s"
        )
        slicer.app.processEvents()

        try:
            best = self._refineAutoTune(n_trials=20)
        except Exception as e:
            self.autoTuneButton.setEnabled(True)
            self.autoTuneStatusLabel.setText(f"Auto-tune error: {e}")
            print(f"[AutoTune] Exception: {e}")
            return

        if best is None:
            self.autoTuneButton.setEnabled(True)
            self.autoTuneStatusLabel.setText("Auto-tune found no valid result.")
            return

        # ── apply best params (quality-aware) ─────────────────────────
        status_lines = []
        for curve in self._manualCurves:
            sq = qualities[curve]
            q = sq["quality"]
            name = curve.GetName()
            if q == "invalid":
                print(f"[SeedQuality] SKIP {name} (AutoTune apply): {sq['reason']}")
                status_lines.append(f"\u274c {name}: skipped")
                continue
            if q == "weak":
                weak_best = dict(best)
                weak_best["w_j"] = best.get("w_j", 5.0) * 0.5
                weak_best["lr"] = best.get("lr", 0.08) * 0.7
                print(
                    f"[SeedQuality] WEAK {name} (AutoTune apply): "
                    f"w_j={weak_best['w_j']:.2f} lr={weak_best['lr']:.3f}"
                )
                self._refineSingleCurve(curve, params=weak_best)
                status_lines.append(f"\u26a0\ufe0f {name}: applied with reduced w_j/lr")
            else:
                self._refineSingleCurve(curve, params=best)
                status_lines.append(f"\u2705 {name}: applied normally")

        self.autoTuneButton.setEnabled(True)
        self.autoTuneStatusLabel.setText(
            f"\u2705 Best params applied{skip_note} \u2014 "
            f"w_s={best['w_s']:.2f} w_c={best['w_c']:.2f} "
            f"w_a={best['w_a']:.2f} w_j={best['w_j']:.2f} lr={best['lr']:.3f}"
        )
        total_out = sum(c.GetNumberOfControlPoints() for c in valid_curves)
        self.manualCLStatusLabel.setText(
            f"\u2728 Auto-tuned {len(valid_curves)} branch(es) ({total_out} pts)"
            f" \u2014 click 'Use This Centerline' to load\n" + "\n".join(status_lines)
        )
        self.manualCLStatusLabel.setStyleSheet("color: #8e44ad; font-size: 11px;")

    def _computeCurveMetrics(self, curve_node):
        """
        Return a dict of quality metrics for one vtkMRMLMarkupsCurveNode.

        Keys (all may be None if not computable):
            mean_curv       – mean discrete curvature (rad / mm)
            std_curv        – std of curvature
            tortuosity      – path length / chord length
            length_mm       – total arc length
            wall_dist_mean  – mean distance to vessel wall (requires model)
            wall_dist_std   – std of wall distance
            branch_angle_deg– angle between first and last segment (degrees)
        """
        import numpy as np
        import vtk

        n = curve_node.GetNumberOfControlPoints()
        if n < 2:
            return {}

        pts = []
        for i in range(n):
            p = [0.0, 0.0, 0.0]
            curve_node.GetNthControlPointPositionWorld(i, p)
            pts.append(np.array(p))
        pts = np.array(pts)

        # segment vectors and lengths
        segs = np.diff(pts, axis=0)  # (n-1, 3)
        seg_lens = np.linalg.norm(segs, axis=1)  # (n-1,)
        seg_lens_safe = np.where(seg_lens < 1e-9, 1e-9, seg_lens)
        dirs = segs / seg_lens_safe[:, None]  # unit tangents

        # ── length & tortuosity ───────────────────────────────────────────
        length_mm = float(np.sum(seg_lens))
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        tortuosity = (length_mm / chord) if chord > 1e-6 else None

        # ── curvature: angle change between consecutive segments / avg len ─
        if n >= 3:
            dots = np.clip(np.einsum("ij,ij->i", dirs[:-1], dirs[1:]), -1.0, 1.0)
            angles = np.arccos(dots)  # rad, (n-2,)
            avg_seg = 0.5 * (seg_lens[:-1] + seg_lens[1:])
            avg_seg = np.where(avg_seg < 1e-9, 1e-9, avg_seg)
            kappas = angles / avg_seg
            mean_curv = float(np.mean(kappas))
            std_curv = float(np.std(kappas))
        else:
            mean_curv = std_curv = None

        # ── branch angle: angle between first and last direction ───────────
        if n >= 3:
            d_first = dirs[0]
            d_last = dirs[-1]
            dot_ba = float(np.clip(np.dot(d_first, d_last), -1.0, 1.0))
            branch_angle_deg = float(np.degrees(np.arccos(dot_ba)))
        else:
            branch_angle_deg = None

        # ── wall distance (requires vessel model) ─────────────────────────
        wall_dist_mean = wall_dist_std = None
        modelNode = self.modelSelector.currentNode()
        if modelNode is not None:
            pd = modelNode.GetPolyData()
            if pd is not None and pd.GetNumberOfPoints() > 0:
                loc = vtk.vtkPointLocator()
                loc.SetDataSet(pd)
                loc.BuildLocator()
                dists = []
                for p in pts:
                    pid = loc.FindClosestPoint(p.tolist())
                    wp = [0.0, 0.0, 0.0]
                    pd.GetPoint(pid, wp)
                    dists.append(np.linalg.norm(p - np.array(wp)))
                if dists:
                    wall_dist_mean = float(np.mean(dists))
                    wall_dist_std = float(np.std(dists))

        return {
            "mean_curv": mean_curv,
            "std_curv": std_curv,
            "tortuosity": tortuosity,
            "length_mm": length_mm,
            "wall_dist_mean": wall_dist_mean,
            "wall_dist_std": wall_dist_std,
            "branch_angle_deg": branch_angle_deg,
        }

    def _refineSingleCurve(self, curve_node, params=None):
        """
        Soft-constraint optimizer for one curve node.

        Replaces hard freezing with a continuous penalty system so every
        interior point can move.  Junction nodes are anchored with a strong
        position penalty (not frozen), which allows the angle-preservation
        energy to find the natural equilibrium rather than locking the point.

        Energy terms
        ────────────
        E_smooth    bending energy (Laplacian)          ← backbone stiffness
        E_center    wall-centering (relative, /radius)  ← lumen alignment
        E_junction  soft position anchor for junctions  ← topology fidelity
        E_angle     pairwise angle-preservation energy  ← bifurcation shape
        E_length    total + per-segment length guard    ← no runaway growth

        All forces are flow-aware: wall force acts perpendicular to the
        local tangent so the point slides across the lumen, not along it.

        params dict (optional, used by auto-tuner)
        ──────────────────────────────────────────
          w_s        smoothness weight        default 1.0
          w_c        wall-centering weight    default 1.0
          w_a        junction angle weight    default 3.0
          w_j        junction anchor weight   default 5.0
          w_length   total-length guard       default 0.30
          w_edge     per-segment guard        default 0.15
          lr         learning rate            default 0.08
          proj_blend partial projection α     default 0.20
          n_iters    iterations               default 20
        """
        import numpy as np
        import vtk

        # ── defaults / param override ─────────────────────────────────────
        p = {
            "w_s": 1.00,
            "w_c": 1.00,
            "w_a": 3.00,
            "w_j": 5.00,  # soft junction anchor (replaces hard freeze)
            "w_length": 0.30,
            "w_edge": 0.15,
            "lr": 0.08,
            "proj_blend": 0.20,
            "n_iters": 30,
        }
        if params:
            p.update(params)

        n = curve_node.GetNumberOfControlPoints()
        if n < 3:
            print(f"[RefineSkip] '{curve_node.GetName()}' has {n} pts — skipping")
            return

        # ── 1. Collect positions + originals ──────────────────────────────
        pts = np.zeros((n, 3), dtype=float)
        for i in range(n):
            q = [0.0, 0.0, 0.0]
            curve_node.GetNthControlPointPositionWorld(i, q)
            pts[i] = q

        x = pts.copy()
        x0 = pts.copy()  # original positions for anchor penalty

        # For short strokes (≤6 pts) the only reliable force is wall-centering
        # constrained by a strong position anchor.  Boost w_j to prevent the
        # free point from drifting outside the vessel between iterations.
        if n <= 6 and "w_j" not in (params or {}):
            p["w_j"] = max(p["w_j"], 8.0)

        # ── 2. Classify points ────────────────────────────────────────────
        # Endpoints are hard-frozen (only these).
        # Junction-region points get a strong soft anchor instead of a freeze.
        hard_frozen = np.zeros(n, dtype=bool)
        hard_frozen[0] = True
        hard_frozen[-1] = True

        segs0 = np.diff(x0, axis=0)
        seg_l0 = np.linalg.norm(segs0, axis=1)
        safe_l0 = np.where(seg_l0 < 1e-9, 1e-9, seg_l0)
        dirs0 = segs0 / safe_l0[:, None]

        # Junction detection: only fire on SHARP turns (dot < 0 = >90°).
        # The old 0.70 threshold fired on every mild kink in a manually
        # drawn curve, generating junctions=3 on a 5-pt stroke and making
        # all interior points soft-anchored with overlapping angle targets.
        BIF_DOT_THRESH = 0.0  # dot < 0 → angle > 90° → genuine bifurcation
        BIF_ZONE_MM = 4.0  # soft-anchor radius around junction

        junction_nodes = []  # indices of detected junction points
        junction_anchors = []  # their original positions

        for i in range(1, n - 1):
            dot = float(np.dot(dirs0[i - 1], dirs0[i]))
            if dot < BIF_DOT_THRESH:
                junction_nodes.append(i)
                junction_anchors.append(x0[i].copy())

        # Nodes near a junction get the soft anchor; collect their set
        near_junction = np.zeros(n, dtype=bool)
        for jp in junction_anchors:
            near_junction |= np.linalg.norm(x0 - jp, axis=1) < BIF_ZONE_MM

        n_free = int(np.sum(~hard_frozen))
        print(
            f"[RefineStart] '{curve_node.GetName()}': {n} pts, "
            f"{n_free} free ({int(np.sum(near_junction & ~hard_frozen))} soft-anchored), "
            f"junctions={len(junction_nodes)}, has_model=",
            end="",
        )

        # ── 3. Build VTK locators ─────────────────────────────────────────
        RAY_LEN = 35.0
        N_RAYS = 6  # 6 rays sufficient for lumen-centre estimate; 12 is overkill for seed refine

        modelNode = self.modelSelector.currentNode()
        has_model = (
            modelNode is not None
            and modelNode.GetPolyData() is not None
            and modelNode.GetPolyData().GetNumberOfPoints() > 0
        )
        print(has_model)  # finish the print above

        cell_loc = pt_loc = pd = None
        if has_model:
            pd = modelNode.GetPolyData()
            cell_loc = vtk.vtkCellLocator()
            cell_loc.SetDataSet(pd)
            cell_loc.BuildLocator()
            pt_loc = vtk.vtkPointLocator()
            pt_loc.SetDataSet(pd)
            pt_loc.BuildLocator()

        # ── helpers ───────────────────────────────────────────────────────

        def _perp_frame(t):
            arb = (
                np.array([1.0, 0.0, 0.0])
                if abs(t[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            u = np.cross(t, arb)
            u /= np.linalg.norm(u)
            v = np.cross(t, u)
            v /= np.linalg.norm(v)
            return u, v

        def _ray_centre_radius(pos, tangent):
            """Ray-cast cross-section; return (lumen_centre, radius)."""
            u, v = _perp_frame(tangent)
            hits = []
            for k in range(N_RAYS):
                angle = 2.0 * np.pi * k / N_RAYS
                d = np.cos(angle) * u + np.sin(angle) * v
                p0 = (pos - d * RAY_LEN).tolist()
                p1 = (pos + d * RAY_LEN).tolist()
                hp = [0.0, 0.0, 0.0]
                hc = vtk.reference(0)
                ht = vtk.reference(0.0)
                if cell_loc.IntersectWithLine(
                    p0, p1, 1e-3, ht, hp, [0.0, 0.0, 0.0], hc
                ):
                    hits.append(np.array(hp))
            if len(hits) >= 4:
                ctr = np.mean(hits, axis=0)
                r = float(np.mean([np.linalg.norm(h - ctr) for h in hits]))
                return ctr, r
            return pos.copy(), 0.0

        def _wall_dist(pos):
            if pt_loc is None:
                return 0.0
            pid = pt_loc.FindClosestPoint(pos.tolist())
            wp = [0.0, 0.0, 0.0]
            pd.GetPoint(pid, wp)
            return float(np.linalg.norm(pos - np.array(wp)))

        def _project_inside(pos, tangent):
            """Partial projection (soft nudge, not hard reset)."""
            if cell_loc is None:
                return pos
            ctr, r = _ray_centre_radius(pos, tangent)
            if r < 1e-6:
                return pos
            delta = ctr - pos
            delta_perp = delta - np.dot(delta, tangent) * tangent
            return pos + p["proj_blend"] * delta_perp

        # ── 4. Pre-compute reference geometry ─────────────────────────────
        segs0_ref = np.diff(x0, axis=0)
        d0 = np.linalg.norm(segs0_ref, axis=1)  # target seg lengths
        L0 = float(np.sum(d0))

        # Junction angle targets: for each junction i, record directions to
        # the two flanking points and their dot product.
        angle_targets = []  # list of (i_prev, i_junc, i_next, dot_orig)
        for ji in junction_nodes:
            if 0 < ji < n - 1:
                uk = x0[ji - 1] - x0[ji]
                ul = x0[ji + 1] - x0[ji]
                nk = np.linalg.norm(uk)
                nl = np.linalg.norm(ul)
                if nk > 1e-6 and nl > 1e-6:
                    vk = uk / nk
                    vl = ul / nl
                    angle_targets.append((ji - 1, ji, ji + 1, float(np.dot(vk, vl))))

        # ── 5. Optimization loop ──────────────────────────────────────────
        _converge_streak = 0  # consecutive iters below movement threshold
        _mv_history = []  # full history for plateau detection
        _plateau_count = 0  # 2-stage plateau: 1=lr↓, 2=stop
        LR_DECAY = p.get(
            "lr_decay", 0.94
        )  # 1.0 = no decay (AutoTune trials); 0.94 = full refine
        _lr = p["lr"]  # local decaying lr — p["lr"] unchanged for AutoTune
        for _iter in range(p["n_iters"]):
            grad = np.zeros_like(x)

            # ── compute tangents (flow direction) ─────────────────────────
            tangents = np.zeros_like(x)
            tangents[0] = x[1] - x[0]
            tangents[-1] = x[-1] - x[-2]
            tangents[1:-1] = x[2:] - x[:-2]
            t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            t_norms = np.where(t_norms < 1e-9, 1e-9, t_norms)
            tangents /= t_norms

            # ── Term 1: smoothness (Laplacian) ────────────────────────────
            # Skip for short curves (≤6 pts): insufficient interior DOF and
            # the term conflicts badly with junction anchors on dense curves.
            if n >= 7:
                d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
                g_sm = np.zeros_like(x)
                g_sm[1:-1] += 4.0 * d2
                g_sm[:-2] -= 2.0 * d2
                g_sm[2:] -= 2.0 * d2
                grad += p["w_s"] * g_sm

            # ── Term 2: wall-centering (flow-aware, relative form) ────────
            if has_model:
                for i in range(n):
                    if hard_frozen[i]:
                        continue
                    ctr, ri = _ray_centre_radius(x[i], tangents[i])
                    if ri < 1e-6:
                        continue
                    dw = _wall_dist(x[i])
                    # relative centering: normalised by radius → consistent
                    # across IVC (r≈10mm) and renals (r≈4mm)
                    err = (dw / ri) - 1.0

                    if pt_loc is not None:
                        pid = pt_loc.FindClosestPoint(x[i].tolist())
                        wp = [0.0, 0.0, 0.0]
                        pd.GetPoint(pid, wp)
                        to_wall = np.array(wp) - x[i]
                        tw_n = np.linalg.norm(to_wall)
                        if tw_n > 1e-6:
                            g_dw = -to_wall / tw_n
                            # flow-aware: only perpendicular component
                            # (don't slide the point along the vessel)
                            t = tangents[i]
                            g_dw -= np.dot(g_dw, t) * t
                            # err = (dw/ri - 1) already normalised; do NOT divide
                            # by ri again or the gradient scales as 1/ri² and
                            # explodes for thin vessels / bad ray casts.
                            grad[i] += p["w_c"] * 2.0 * err * g_dw

            # ── Term 3: soft junction anchor ─────────────────────────────
            # Instead of hard-freezing junction-zone nodes we penalise
            # displacement from the original position.  This allows the
            # angle-preservation energy to shift the junction slightly
            # while preventing large drift.
            for i in range(n):
                if hard_frozen[i]:
                    continue
                if near_junction[i]:
                    diff = x[i] - x0[i]
                    grad[i] += p["w_j"] * 2.0 * diff

            # ── Term 4: junction angle preservation ───────────────────────
            # Only active for longer curves (>6 pts) where junction geometry
            # is stable enough for angle targets to be meaningful.
            # On short curves the overlapping triplets fight the anchor term.
            for ik, ij, il, dot_orig in angle_targets if n >= 7 else []:
                uk = x[ik] - x[ij]
                nk = max(np.linalg.norm(uk), 1e-6)
                ul = x[il] - x[ij]
                nl = max(np.linalg.norm(ul), 1e-6)
                vk = uk / nk
                vl = ul / nl
                d = float(np.clip(np.dot(vk, vl), -1.0, 1.0))
                diff = d - dot_orig

                Pk = np.eye(3) - np.outer(vk, vk)
                Pl = np.eye(3) - np.outer(vl, vl)

                gk = 2.0 * diff * (Pk @ vl) / nk
                gl = 2.0 * diff * (Pl @ vk) / nl
                gj = -(gk + gl)

                if not hard_frozen[ik]:
                    grad[ik] += p["w_a"] * gk
                if not hard_frozen[il]:
                    grad[il] += p["w_a"] * gl
                if not hard_frozen[ij]:
                    grad[ij] += p["w_a"] * gj

            # ── Term 5: total length guard ────────────────────────────────
            segs_c = np.diff(x, axis=0)
            seg_lc = np.linalg.norm(segs_c, axis=1)
            L_cur = float(np.sum(seg_lc))
            dL = 2.0 * (L_cur - L0)
            safe_lc = np.where(seg_lc < 1e-9, 1e-9, seg_lc)
            sd = segs_c / safe_lc[:, None]
            g_len = np.zeros_like(x)
            g_len[1:] += dL * sd
            g_len[:-1] -= dL * sd
            grad += p["w_length"] * g_len

            # ── Term 6: per-segment edge guard ────────────────────────────
            g_edge = np.zeros_like(x)
            for j in range(n - 1):
                seg = x[j + 1] - x[j]
                sl = float(np.linalg.norm(seg))
                if sl < 1e-9:
                    continue
                sd_j = seg / sl
                err_j = 2.0 * (sl - d0[j])
                g_edge[j + 1] += err_j * sd_j
                g_edge[j] -= err_j * sd_j
            grad += p["w_edge"] * g_edge

            # ── gradient step (hard endpoints frozen) ─────────────────────
            grad[hard_frozen] = 0.0

            # Per-point gradient clamp: cap each point's gradient magnitude
            # so no point moves more than MAX_STEP_MM per iteration.
            # This prevents the angle/centering terms from launching points
            # into outer space when the geometry is far from equilibrium.
            MAX_STEP_MM = 1.5  # tighter cap: max 1.5mm per point per iteration
            g_norms = np.linalg.norm(grad, axis=1, keepdims=True)
            scale = np.where(
                g_norms * _lr > MAX_STEP_MM, MAX_STEP_MM / (g_norms * _lr + 1e-9), 1.0
            )
            grad *= scale

            x_old = x.copy()
            x -= _lr * grad

            # ── partial projection ────────────────────────────────────────
            if has_model:
                for i in range(n):
                    if hard_frozen[i]:
                        continue
                    x[i] = _project_inside(x[i], tangents[i])

            # ── movement diagnostic ───────────────────────────────────────
            avg_mv = float(np.linalg.norm(x - x_old, axis=1).mean())
            # Single-free-point WEAK curves have no smoothing/angle terms active;
            # sub-voxel oscillation around the lumen centre IS convergence.
            # Scale: 0.010mm for n_free=1, 0.005mm for n_free>=2.
            CONVERGE_THRESH = 0.005 * max(1, 3 - n_free)
            # Projection-snap guard: if movement jumped >3× vs previous iter,
            # the point briefly exited the wall and was snapped back hard.
            # Reset convergence streak — don't count a corrupted step.
            if _mv_history and avg_mv > 3.0 * _mv_history[-1]:
                _converge_streak = 0
            if avg_mv < CONVERGE_THRESH:
                _converge_streak += 1
            else:
                _converge_streak = 0
            if avg_mv > 0.5:
                tag = "unstable ✗"
            elif avg_mv >= CONVERGE_THRESH:
                tag = "good ✓"
            elif _converge_streak < 2:
                tag = "settling"
            else:
                tag = "converged ✓"
            # note: avg_mv > 0.5 = "unstable" label, but values up to MAX_STEP_MM
            # are expected during initial settling for very displaced points
            # ── plateau detection (optimizer-style: gate + rel-improve + variance) ──
            # Replaces old band check. Three signals must all pass:
            #   1. Absolute gate  — must already be near convergence
            #   2. Relative improvement — window start→end improvement < 1%
            #   3. Stability — low variance across the window
            # AutoTune trials (LR_DECAY==1.0) bypass this entirely (short, near-converged).
            _mv_history.append(avg_mv)
            _plateau_fire = False
            if LR_DECAY < 1.0 and _iter >= 5:
                _PW = 8  # plateau window length
                _GATE = max(2.5 * CONVERGE_THRESH, 0.01)  # ~0.0125 mm
                _REL = 0.01  # 1% relative improvement threshold
                _VAR = 1e-6  # variance tolerance (mm²)
                if len(_mv_history) >= _PW:
                    _win = _mv_history[-_PW:]
                    _cur = _win[-1]
                    _prev = _win[0]
                    _gate_ok = _cur <= _GATE
                    _rel_ok = (_prev - _cur) / max(_prev, 1e-8) <= _REL
                    _mean = sum(_win) / _PW
                    _var = sum((v - _mean) ** 2 for v in _win) / _PW
                    _var_ok = _var <= _VAR
                    _plateau_fire = _gate_ok and _rel_ok and _var_ok
            if _plateau_fire:
                _plateau_count += 1
                if _plateau_count == 1:
                    tag += " (plateau→lr↓)"
                else:
                    tag += " (plateau ✓)"
            else:
                _plateau_count = 0
            print(
                f"[RefineIter] iter={_iter:02d}  avg_movement={avg_mv:.4f} mm  ({tag})  lr={_lr:.4f}"
            )
            if _converge_streak >= 2:
                break
            if _plateau_fire:
                if _plateau_count >= 2:
                    break
                _lr *= 0.5  # stage-1: halve lr, continue
                continue  # skip normal lr decay this iter
            _lr *= LR_DECAY

        # ── 6. Replace control points in-place ───────────────────────────
        curve_node.RemoveAllControlPoints()
        for q in x:
            curve_node.AddControlPointWorld(vtk.vtkVector3d(q[0], q[1], q[2]))

    def _refineAutoTune(self, n_trials=20):
        """
        Auto-tuner: runs _refineSingleCurve with random parameter samples,
        scores each trial against the before/after metrics, returns the best
        param dict.  Called from onAutoTune().

        Search space (log-uniform random):
          w_s        [0.5, 2.0]
          w_c        [0.3, 1.5]
          w_a        [1.0, 5.0]
          w_j        [2.0, 8.0]
          w_length   [0.1, 0.6]
          lr         [0.05, 0.15]
        """
        import numpy as np

        def _sample():
            def _logu(lo, hi):
                return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))

            return {
                "w_s": _logu(0.5, 2.0),
                "w_c": _logu(0.3, 1.5),
                "w_a": _logu(1.0, 5.0),
                "w_j": _logu(2.0, 8.0),
                "w_length": _logu(0.1, 0.6),
                "w_edge": 0.15,
                "lr": _logu(0.05, 0.15),
                "lr_decay": 1.0,  # no decay during trials — starts near-converged
                "proj_blend": 0.20,
                "n_iters": 8,  # fewer iters per trial for speed
            }

        def _score(before, after):
            s = 0.0
            # centering (most important)
            if before.get("wall_dist_std") and after.get("wall_dist_std"):
                s += 5.0 * (before["wall_dist_std"] - after["wall_dist_std"])
            # smoothness
            if before.get("std_curv") and after.get("std_curv"):
                s += 3.0 * (before["std_curv"] - after["std_curv"])
            # length guard: penalise >5% growth
            if before.get("length_mm") and after.get("length_mm"):
                ratio = after["length_mm"] / (before["length_mm"] + 1e-6)
                if ratio > 1.05:
                    s -= 4.0 * (ratio - 1.05)
            # tortuosity: penalise >8% growth
            if before.get("tortuosity") and after.get("tortuosity"):
                ratio = after["tortuosity"] / (before["tortuosity"] + 1e-6)
                if ratio > 1.08:
                    s -= 2.0 * (ratio - 1.08)
            # branch-angle stability
            if before.get("branch_angle_deg") and after.get("branch_angle_deg"):
                s -= 0.5 * abs(after["branch_angle_deg"] - before["branch_angle_deg"])
            return s

        if not self._manualCurves:
            return None

        # snapshot original geometry
        import vtk

        originals = []
        for curve in self._manualCurves:
            pts = []
            for i in range(curve.GetNumberOfControlPoints()):
                q = [0.0, 0.0, 0.0]
                curve.GetNthControlPointPositionWorld(i, q)
                pts.append(list(q))
            originals.append(pts)

        def _restore():
            for curve, pts in zip(self._manualCurves, originals):
                curve.RemoveAllControlPoints()
                for q in pts:
                    curve.AddControlPointWorld(vtk.vtkVector3d(*q))

        before_metrics = [self._computeCurveMetrics(c) for c in self._manualCurves]

        best_score = -1e9
        best_params = None

        for trial in range(n_trials):
            _restore()
            params = _sample()

            for curve in self._manualCurves:
                self._refineSingleCurve(curve, params=params)

            after_metrics = [self._computeCurveMetrics(c) for c in self._manualCurves]

            # aggregate across curves
            def _agg(key):
                vals = [m[key] for m in after_metrics if m.get(key) is not None]
                return float(np.mean(vals)) if vals else None

            def _agg_b(key):
                vals = [m[key] for m in before_metrics if m.get(key) is not None]
                return float(np.mean(vals)) if vals else None

            after_agg = {
                k: _agg(k)
                for k in [
                    "wall_dist_std",
                    "std_curv",
                    "length_mm",
                    "tortuosity",
                    "branch_angle_deg",
                ]
            }
            before_agg = {k: _agg_b(k) for k in after_agg}

            s = _score(before_agg, after_agg)
            print(
                f"[AutoTune] trial {trial+1:02d}/{n_trials}  score={s:+.3f}  "
                f"ws={params['w_s']:.2f} wc={params['w_c']:.2f} "
                f"wa={params['w_a']:.2f} wj={params['w_j']:.2f} lr={params['lr']:.3f}"
            )

            if s > best_score:
                best_score = s
                best_params = dict(params)
                best_params["n_iters"] = (
                    30  # restore full iters for apply (extra room for WEAK branches)
                )
                best_params["lr_decay"] = 0.94  # re-enable decay for final apply pass

        _restore()
        print(f"[AutoTune] Best score={best_score:+.3f}  params={best_params}")
        return best_params

    def onCenterlineNodeChanged(self, node):
        """Sync the visibility toggle button state when the selected centerline changes."""
        if node and node.GetDisplayNode():
            visible = bool(node.GetDisplayNode().GetVisibility())
            self.centerlineVisButton.blockSignals(True)
            self.centerlineVisButton.setChecked(visible)
            self.centerlineVisButton.setText(
                "👁 Hide Centerline" if visible else "👁 Show Centerline"
            )
            self.centerlineVisButton.blockSignals(False)

    def onCenterlineVisToggle(self, checked):
        """Show or hide the centerline model in the 3D view."""
        node = self.centerlineSelector.currentNode()
        if not node:
            self.centerlineVisButton.blockSignals(True)
            self.centerlineVisButton.setChecked(False)
            self.centerlineVisButton.blockSignals(False)
            return
        dn = node.GetDisplayNode()
        if dn:
            dn.SetVisibility(1 if checked else 0)
        self.centerlineVisButton.setText(
            "👁 Hide Centerline" if checked else "👁 Show Centerline"
        )



# ── Slicer module-scanner guard ───────────────────────────────────────────────
class vessel_centerline_widget_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_centerline_widget_mixin"
            parent.hidden = True

