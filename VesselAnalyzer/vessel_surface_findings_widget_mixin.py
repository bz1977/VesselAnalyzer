"""
vessel_surface_findings_widget_mixin.py — VesselAnalyzerWidget surface classification and findings handlers

Extracted from VesselAnalyzer.py (VesselAnalyzerWidget methods).

Usage in VesselAnalyzer.py
--------------------------
Add to the VesselAnalyzerWidget class definition::

    from vessel_surface_findings_widget_mixin import SurfaceFindingsWidgetMixin

    class VesselAnalyzerWidget(SurfaceFindingsWidgetMixin, ScriptedLoadableModuleWidget, VTKObservationMixin):
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


class SurfaceFindingsWidgetMixin:
    """Mixin: VesselAnalyzerWidget surface classification and findings handlers"""
    def onFindAndPlaceStents(self):
        """Single-click: detect findings then auto-place stents.
        Pre-dilation balloon is only shown in the manual workflow
        when user explicitly clicks Pre-Dilate Lesion button.
        """
        self.onDetectFindings()
        self.onAutoPlaceAllStents()

    def onVesselTypeChanged(self, index):
        """Sync UI vessel-type combo to logic.vesselType.
        Index 0 = unselected placeholder, 1 = Arterial, 2 = Venous."""
        if index == 0:
            self.vesselTypeHintLabel.setText(
                "⚠  Set vessel type before running analysis"
            )
            self.vesselTypeHintLabel.setStyleSheet("color: #e67e22; font-size: 11px;")
            return
        vtype = "venous" if index == 2 else "arterial"
        # Store on widget so it survives logic re-instantiation (onLoadIVUS creates
        # a fresh VesselAnalyzerLogic and must pick this up).
        self._pendingVesselType = vtype
        if self.logic is not None:
            self.logic.vesselType = vtype
        label = "Venous" if vtype == "venous" else "Arterial"
        self.vesselTypeHintLabel.setText(f"✓  {label} thresholds active")
        self.vesselTypeHintLabel.setStyleSheet("color: #27ae60; font-size: 11px;")
        print(f"[VesselAnalyzer] Vessel type set to: {vtype}")

    def onDetectFindings(self):
        if not self.logic.points:
            slicer.util.errorDisplay("Please load a centerline first.")
            return
        if not self.logic.modelNode:
            slicer.util.errorDisplay("No vessel model loaded.")
            return
        if self.vesselTypeCombo.currentIndex == 0:
            if not slicer.util.confirmYesNoDisplay(
                "Vessel Type has not been selected.\n\n"
                "Detection thresholds differ significantly between Arterial and Venous.\n\n"
                "Proceed with default Arterial thresholds, or cancel and set the type in Step 1?",
                windowTitle="Vessel Type Not Set",
            ):
                return
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            findings = self.logic._detectFindings()
            self.logic._findingsFrozen = True  # freeze — don't re-detect after balloon
            self.logic.applyColorOverlay()
            n_aneurysm = sum(1 for f in findings if f["type"] == "Aneurysm")
            n_ectasia = sum(1 for f in findings if f["type"] == "Ectasia")
            n_pancake = sum(1 for f in findings if "Pancak" in f["type"])
            n_mild = sum(1 for f in findings if "Compress" in f["type"])
            msg = (
                "Detection complete:\n"
                + f"  Aneurysm:         {n_aneurysm} locations\n"
                + f"  Ectasia:          {n_ectasia} locations\n"
                + f"  Pancaking:        {n_pancake} locations\n"
                + f"  Mild compression: {n_mild} locations\n\n"
                + "Vessel mesh: Green=Normal, Yellow=Mild, Orange=Ectasia, Red=Aneurysm"
            )
            # Auto-select the branch with the worst finding in stent planner
            if findings:
                # Auto-select: prefer compression findings in long downstream branches
                # Aneurysm/ectasia in short proximal side branches should not be selected
                severity = {
                    "Pancaking": 4,
                    "Eccentric Compression": 3,
                    "Mild Compression": 2,
                    "Aneurysm": 2,
                    "Ectasia": 1,
                }
                trunk_end_z = (
                    self.logic.points[self.logic.branches[0][1] - 1][2]
                    if self.logic.branches
                    else 0.0
                )
                downstream_findings = [
                    f
                    for f in findings
                    if self.logic.getBranchStats(f["branchIdx"])["length"] > 100.0
                    and self.logic.points[self.logic.branches[f["branchIdx"]][0]][2]
                    <= trunk_end_z + 10.0
                ]
                pool = (
                    downstream_findings
                    if downstream_findings
                    else [
                        f
                        for f in findings
                        if self.logic.getBranchStats(f["branchIdx"])["length"] > 50.0
                    ]
                )
                if not pool:
                    pool = findings
                worst = max(
                    pool, key=lambda f: (severity.get(f["type"], 0), -f["value"])
                )
                worst_branch = worst["branchIdx"]
                # Find combo position — sub-branches are excluded so index != raw+1
                _combo_pos = -1
                for _row in range(self.stentBranchCombo.count):
                    if self.stentBranchCombo.itemData(_row) == worst_branch:
                        _combo_pos = _row
                        break
                self.stentBranchCombo.blockSignals(True)
                if _combo_pos >= 0:
                    self.stentBranchCombo.setCurrentIndex(_combo_pos)
                self.stentBranchCombo.blockSignals(False)
                print(
                    "[VesselAnalyzer] Stent branch auto-selected: "
                    + self.logic.getBranchDisplayName(worst_branch)
                    + " ("
                    + worst["type"]
                    + ")"
                    + " length="
                    + str(round(self.logic.getBranchStats(worst_branch)["length"], 1))
                    + "mm"
                )
            # ── Enable Pre-Dilate button if any stenosis found ──────────
            stenoses = [
                f
                for f in findings
                if "Pancak" in f.get("type", "") or "Compress" in f.get("type", "")
            ]
            self.preDilateButton.setEnabled(len(stenoses) > 0)
            self.manualBalloonButton.setEnabled(True)
            if stenoses:
                self.preDilateStatusLabel.setText(
                    f"{len(stenoses)} stenosis found — click Pre-Dilate before placing stent"
                )
            self._buildLesionTable()
            # Also run collateral detection silently
            try:
                collaterals = self.logic._detectCollaterals()
                self.logic.applyCollateralOverlay()
                if collaterals:
                    msg += f"\n\nCollaterals detected: {len(collaterals)} (shown in cyan on mesh)"
            except Exception:
                pass
            slicer.util.infoDisplay(msg)
            # Re-apply overlay using ORIGINAL findings (before ballooning)
            # Do NOT re-run _detectFindings — expanded mesh would show false ectasia
            self.logic.applyColorOverlay()
        except Exception as e:
            import traceback

            slicer.util.errorDisplay(
                f"Detection failed:\n{e}\n{traceback.format_exc()}"
            )
        finally:
            slicer.app.restoreOverrideCursor()

    def onClearOverlay(self):
        if self.logic.modelNode:
            self.logic.clearColorOverlay()
            self.findingWarningLabel.setText("")

    def onCreateSurface(self):
        """Clean + decimate model in 4 steps, save as new node."""
        import vtk, math

        node = self.surfModelSelector.currentNode()
        if not node or not node.GetPolyData():
            self.surfResultLabel.setText("⚠️ Select a model first.")
            return
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.surfResultLabel.setStyleSheet(
                "font-size: 12px; padding: 6px; border-radius: 4px; "
                "background: #fff9c4; color: #5d4037;"
            )
            self.surfResultLabel.setText("⏳ Processing…")
            slicer.app.processEvents()

            mesh = node.GetPolyData()
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(mesh)
            cleaner.Update()
            current = cleaner.GetOutput()

            for i, step in enumerate([0.25, 0.25, 0.25, 0.25]):
                dec = vtk.vtkDecimatePro()
                dec.SetInputData(current)
                dec.SetTargetReduction(step)
                dec.PreserveTopologyOn()
                dec.SplittingOff()
                dec.BoundaryVertexDeletionOff()
                dec.Update()
                current = dec.GetOutput()

            out_name = node.GetName() + "_Surface"
            existing = slicer.mrmlScene.GetFirstNodeByName(out_name)
            if existing:
                slicer.mrmlScene.RemoveNode(existing)
            newNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", out_name)
            newNode.SetAndObservePolyData(current)
            newNode.CreateDefaultDisplayNodes()
            newNode.GetDisplayNode().SetColor(0.85, 0.85, 0.85)

            pct = (
                1.0 - current.GetNumberOfCells() / max(mesh.GetNumberOfCells(), 1)
            ) * 100
            self.surfResultLabel.setStyleSheet(
                "font-size: 12px; padding: 6px; border-radius: 4px; "
                "background: #e8f5e9; color: #1b5e20;"
            )
            self.surfResultLabel.setText(
                f"✅ Surface created: '{out_name}'\n"
                f"Points: {current.GetNumberOfPoints():,}   "
                f"Cells: {current.GetNumberOfCells():,}   "
                f"({pct:.0f}% reduced)\n"
                f"→ Now click 🔍 Check Surface to verify"
            )
            self.surfModelSelector.setCurrentNode(newNode)
        except Exception as e:
            self.surfResultLabel.setText(f"❌ Error: {e}")
        finally:
            slicer.app.restoreOverrideCursor()

    def onCheckSurface(self):
        """Check mesh for boundary/non-manifold edges and display results."""
        import vtk

        node = self.surfModelSelector.currentNode()
        if not node or not node.GetPolyData():
            self.surfResultLabel.setText("⚠️ Select a model first.")
            return
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            mesh = node.GetPolyData()
            feature = vtk.vtkFeatureEdges()
            feature.SetInputData(mesh)
            feature.BoundaryEdgesOn()
            feature.NonManifoldEdgesOn()
            feature.FeatureEdgesOff()
            feature.ManifoldEdgesOff()
            feature.Update()
            boundary = feature.GetOutput().GetNumberOfLines()

            pts = mesh.GetNumberOfPoints()
            cells = mesh.GetNumberOfCells()

            if boundary == 0:
                self.surfResultLabel.setStyleSheet(
                    "font-size: 12px; padding: 6px; border-radius: 4px; "
                    "background: #e8f5e9; color: #1b5e20;"
                )
                self.surfResultLabel.setText(
                    f"✅  Clean watertight surface\n"
                    f"Points: {pts:,}   Cells: {cells:,}\n"
                    f"Open / non-manifold edges: 0"
                )
            else:
                self.surfResultLabel.setStyleSheet(
                    "font-size: 12px; padding: 6px; border-radius: 4px; "
                    "background: #ffebee; color: #7f0000;"
                )
                self.surfResultLabel.setText(
                    f"⚠️  Open / non-manifold edges detected\n"
                    f"Points: {pts:,}   Cells: {cells:,}\n"
                    f"Problem edges: {boundary:,}\n"
                    f"Tip: run Create Surface again, or use Surface Toolbox → Fill holes"
                )
        except Exception as e:
            self.surfResultLabel.setText(f"❌ Error: {e}")
        finally:
            slicer.app.restoreOverrideCursor()

    def onClassifySurface(self):
        """Detect branch emergence borders on the 3D vessel surface and
        colorize the mesh: trunk = blue, each branch = distinct color."""
        import vtk

        node = self.branchSurfModelSelector.currentNode()
        if not node or not node.GetPolyData():
            self.branchSurfResultLabel.setText("⚠️ Select a vessel model first.")
            return
        if not getattr(self.logic, "branches", []):
            self.branchSurfResultLabel.setText(
                "⚠️ No centerline loaded.\n" "Please Load for IVUS Navigation first."
            )
            return

        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.branchSurfResultLabel.setStyleSheet(
                "font-size: 12px; padding: 6px; border-radius: 4px; "
                "background: #fff9c4; color: #5d4037;"
            )
            self.branchSurfResultLabel.setText("⏳ Classifying surface…")
            slicer.app.processEvents()

            thresh = self.curvatureThreshSlider.value
            main_len_gate = self.mainLenGateSlider.value
            result = self.logic.classifySurfaceBranches(
                node.GetPolyData(), thresh, main_len_gate
            )

            node.SetAndObservePolyData(result["poly"])
            dn = node.GetDisplayNode()

            # ── Step 1: set active scalar on the poly ─────────────────────────
            result["poly"].GetPointData().SetActiveScalars("BranchClassification")

            # ── Step 2: build an identity LUT ─────────────────────────────────
            # Instead of fighting vtkMRMLModelDisplayableManager (which resets
            # SetColorModeToDirectScalars every time it rebuilds the actor),
            # use a 256-entry LUT in MapScalars mode that Slicer respects.
            # The LUT uses the R channel of each group's RGB as the index key.
            try:
                _lut = vtk.vtkLookupTable()
                _lut.SetNumberOfTableValues(256)
                _lut.SetRange(0, 255)
                for _i in range(256):
                    _lut.SetTableValue(_i, _i / 255.0, _i / 255.0, _i / 255.0, 1.0)
                _TRUNK_COLOR = result.get("TRUNK_COLOR", (100, 149, 237))
                _RENAL_COLOR = result.get("RENAL_COLOR", (80, 200, 220))
                _MAIN_COLORS = result.get("MAIN_COLORS", [(220, 80, 60), (50, 180, 100)])
                _SIDE_COLOR  = result.get("SIDE_COLOR",  (180, 120, 60))
                for _rgb in [_TRUNK_COLOR, _RENAL_COLOR, _SIDE_COLOR] + list(_MAIN_COLORS):
                    _r, _g, _b = _rgb
                    _lut.SetTableValue(_r, _r / 255.0, _g / 255.0, _b / 255.0, 1.0)
                _lut.Build()
                print(f"[ClassifySurface] Identity LUT built: "
                      f"trunk@{_TRUNK_COLOR[0]} renal@{_RENAL_COLOR[0]} "
                      f"main@{[c[0] for c in _MAIN_COLORS]} side@{_SIDE_COLOR[0]}")
            except Exception as _lut_e:
                print(f"[ClassifySurface] LUT build error: {_lut_e}")
                _lut = None

            # ── Step 3: configure MRML display node ───────────────────────────
            dn.SetActiveScalarName("BranchClassification")
            dn.SetScalarVisibility(True)
            dn.SetScalarRangeFlag(0)
            dn.SetScalarRange(0, 255)

            # ── Step 4: apply LUT to the VTK mapper ───────────────────────────
            _mapped = False
            try:
                import slicer as _slicer
                _slicer.app.processEvents()
                view = _slicer.app.layoutManager().threeDWidget(0).threeDView()
                renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
                n_target = result["poly"].GetNumberOfPoints()
                for ac in renderer.GetActors():
                    mp = ac.GetMapper()
                    if mp is None:
                        continue
                    try:
                        inp = mp.GetInputDataObject(0, 0)
                    except Exception:
                        continue
                    if inp is None:
                        continue
                    if getattr(inp, "GetNumberOfPoints", lambda: -1)() != n_target:
                        continue
                    if inp.GetPointData().GetArray("BranchClassification") is None:
                        continue
                    inp.GetPointData().SetActiveScalars("BranchClassification")
                    mp.SetColorModeToMapScalars()
                    mp.SetScalarModeToUsePointData()
                    mp.ScalarVisibilityOn()
                    mp.SetScalarRange(0, 255)
                    if _lut is not None:
                        mp.SetLookupTable(_lut)
                        mp.UseLookupTableScalarRangeOn()
                    _mapped = True
                    print("[ClassifySurface] Mapper: identity LUT applied ✓")
                    break
            except Exception as _me:
                print(f"[ClassifySurface] Mapper setup error: {_me}")

            # ── Step 5: render ────────────────────────────────────────────────
            try:
                view.renderWindow().Render()
            except Exception:
                pass

            if not _mapped:
                print(
                    "[ClassifySurface] Mapper not found — colors may not display. "
                    "Open Models module → Display → Scalars → BranchClassification."
                )

            # Store LUT on self so _reapplyDirectRGBMapper can reuse it
            self._branchClassLUT = _lut

            # Stash original color so Clear can restore it
            orig = dn.GetColor()
            node.SetAttribute(
                "_origColor", f"{orig[0]:.4f},{orig[1]:.4f},{orig[2]:.4f}"
            )

            lines = ["✅ Surface classified (4 display groups):"]
            trunk_names, main_names, renal_names, side_names = [], [], [], []

            # Read the color map and sentinel values directly from the logic result
            # so this display code never diverges from the actual painted colors.
            bi_to_color  = result.get("bi_to_color", {})
            TRUNK_COLOR  = result.get("TRUNK_COLOR",  (100, 149, 237))
            RENAL_COLOR  = result.get("RENAL_COLOR",  (80,  200, 220))
            MAIN_COLORS  = result.get("MAIN_COLORS",  [(220, 80, 60), (50, 180, 100)])
            SIDE_COLOR   = result.get("SIDE_COLOR",   (180, 120, 60))

            for bi, info in sorted(result["classifications"].items()):
                if getattr(self.logic, "branchMeta", {}).get(bi, {}).get("role") == "iliac_fragment":
                    continue
                name   = self.logic.getBranchDisplayName(bi)
                r_mm   = info.get("mean_radius_mm", 0.0)
                entry  = f"{name} ({info['length_mm']:.0f}mm, r\u2248{r_mm:.1f}mm)"
                color  = bi_to_color.get(bi, SIDE_COLOR)

                if color == TRUNK_COLOR:
                    trunk_names.append(entry)
                elif color == RENAL_COLOR:
                    renal_names.append(entry)
                elif color in MAIN_COLORS:
                    main_names.append(entry)
                else:
                    side_names.append(entry)

            if trunk_names:
                lines.append(f"  \U0001f535 Trunk: {', '.join(trunk_names)}")
            for i, n in enumerate(main_names):
                color_icon = "\U0001f534" if i == 0 else "\U0001f7e2"
                lines.append(f"  {color_icon} Main limb: {n}")
            if renal_names:
                lines.append(f"  \U0001f7e1 Renal: {', '.join(renal_names)}")
            if side_names:
                lines.append(f"  \U0001f7e0 Side branches: {', '.join(side_names)}")
            if result.get("ostium_count", 0) > 0:
                lines.append(f"\n  Ostium ring pts: {result['ostium_count']}")

            self.branchSurfResultLabel.setStyleSheet(
                "font-size: 12px; padding: 6px; border-radius: 4px; "
                "background: #ede7f6; color: #1a0030;"
            )
            self.branchSurfResultLabel.setText("\n".join(lines))

        except Exception as e:
            import traceback

            self.branchSurfResultLabel.setStyleSheet(
                "font-size: 12px; padding: 6px; border-radius: 4px; "
                "background: #ffebee; color: #7f0000;"
            )
            self.branchSurfResultLabel.setText(f"❌ Error: {e}")
            print(traceback.format_exc())
        finally:
            slicer.app.restoreOverrideCursor()

    def onClearSurfaceClassification(self):
        """Remove branch coloring scalar and restore original model color."""
        import vtk

        node = self.branchSurfModelSelector.currentNode()
        if not node or not node.GetPolyData():
            self.branchSurfResultLabel.setText("⚠️ Select a model first.")
            return
        poly = node.GetPolyData()
        pd = poly.GetPointData()
        if pd.GetArray("BranchClassification"):
            pd.RemoveArray("BranchClassification")
            poly.Modified()
        dn = node.GetDisplayNode()
        if dn:
            dn.SetScalarVisibility(False)
            # Restore mapper to LUT-based mode
            try:
                import slicer as _slicer

                view = _slicer.app.layoutManager().threeDWidget(0).threeDView()
                renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
                poly = node.GetPolyData()
                for ac in renderer.GetActors():
                    mp = ac.GetMapper()
                    if mp and mp.GetInput() is not None and mp.GetInput() == poly:
                        mp.SetColorModeToDefault()
                        mp.Update()
            except Exception:
                pass
            orig = node.GetAttribute("_origColor")
            if orig:
                r, g, b = [float(x) for x in orig.split(",")]
                dn.SetColor(r, g, b)
            else:
                dn.SetColor(0.85, 0.85, 0.85)
        self.branchSurfResultLabel.setStyleSheet(
            "font-size: 12px; padding: 6px; border-radius: 4px; "
            "background: #e8f5e9; color: #1b5e20;"
        )
        self.branchSurfResultLabel.setText("✅ Coloring cleared.")

    def onDebugOstium(self):
        """Place three coloured fiducial sets for visual ostium validation.

        Red    — snapped bifurcation graph node (raw topology start)
        Yellow — Pass-2 ostium (sibling-separation + radius stabilisation)
        Green  — (not currently used)

        Call after running analysis.  Re-running clears previous markers.
        """
        import slicer, math

        logic = self.logic
        if not getattr(logic, "branchMeta", {}):
            slicer.util.errorDisplay(
                "No branch metadata found.\n"
                "Run the vessel analysis first, then click this button."
            )
            return

        scene = slicer.mrmlScene

        # ── Remove any existing debug ostium nodes ──────────────────────────
        for tag in (
            "OstiumDebug_Bif",
            "OstiumDebug_P2",
            "OstiumDebug_P3",
            "OstiumDebug_Flare",
            "OstiumDebug_Refined",
            "OstiumDebug_LowConf",
            "OstiumDebug_SphereRing_Target",
            "OstiumDebug_SphereRing_Actual",
        ):
            existing = scene.GetFirstNodeByName(tag)
            while existing:
                scene.RemoveNode(existing)
                existing = scene.GetFirstNodeByName(tag)

        def _make_node(name, rgb):
            node = scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
            node.CreateDefaultDisplayNodes()
            dn = node.GetDisplayNode()
            dn.SetSelectedColor(*rgb)
            dn.SetColor(*rgb)
            dn.SetPointLabelsVisibility(True)
            dn.SetTextScale(3.0)
            return node

        bif_node = _make_node("OstiumDebug_Bif", (1.00, 0.15, 0.15))
        p2_node = _make_node("OstiumDebug_P2", (1.00, 0.85, 0.00))
        refined_node = _make_node("OstiumDebug_Refined", (1.00, 0.50, 0.00))
        p3_node = _make_node("OstiumDebug_P3", (0.18, 0.80, 0.44))
        flare_node = _make_node("OstiumDebug_Flare", (0.20, 0.60, 1.00))
        lowconf_node = _make_node(
            "OstiumDebug_LowConf", (0.80, 0.80, 0.00)
        )  # yellow-grey = REJECT/low-conf

        n_bif = n_p2 = n_p3 = n_flare = n_refined = n_lowconf = 0
        pts = logic.points
        npts = len(pts)

        for bi, meta in logic.branchMeta.items():
            if bi == 0:
                continue

            bif_pt = meta.get("bif_pt") or meta.get("ostium_p2")
            if bif_pt is not None:
                bif_node.AddControlPoint(
                    float(bif_pt[0]), float(bif_pt[1]), float(bif_pt[2]), f"Bif{bi}"
                )
                n_bif += 1

            p2_pt = meta.get("ostium_p2")
            if p2_pt is not None:
                p2_node.AddControlPoint(
                    float(p2_pt[0]), float(p2_pt[1]), float(p2_pt[2]), f"Ost{bi}"
                )
                n_p2 += 1

            # Orange marker: refined ostium (result of _refineOstia composite
            # score).  Only shown when refinement advanced beyond original ostium.
            rgi = meta.get("ostiumGiRefined")
            if rgi is not None and 0 <= rgi < npts:
                rp = pts[rgi]
                refined_node.AddControlPoint(
                    float(rp[0]), float(rp[1]), float(rp[2]), f"Ref{bi}"
                )
                n_refined += 1

            p3_pt = meta.get("ostium_p3")
            if p3_pt is not None:
                p3_node.AddControlPoint(
                    float(p3_pt[0]), float(p3_pt[1]), float(p3_pt[2]), f"P3_{bi}"
                )
                n_p3 += 1

            # Low-confidence marker: REJECT-grade suppressed branches show their
            # best-guess snap coordinate in a distinct dim-yellow colour so the
            # user can review the position even though it is not clinically trusted.
            lc_pt = meta.get("ostium_p3_lowconf")
            if lc_pt is not None:
                lowconf_node.AddControlPoint(
                    float(lc_pt[0]), float(lc_pt[1]), float(lc_pt[2]), f"LC{bi}"
                )
                n_lowconf += 1

            sgi = meta.get("stableStartGi")
            if sgi is not None and 0 <= sgi < npts:
                sp = pts[sgi]
                flare_node.AddControlPoint(
                    float(sp[0]), float(sp[1]), float(sp[2]), f"Stb{bi}"
                )
                n_flare += 1

        msg = (
            f"Ostium debug markers:\n"
            f"  Red    (Bif/Ost)    : {n_bif} — snapped bif / ostium coord\n"
            f"  Yellow (ostiumGi)   : {n_p2} — index-based ostium (pre-refinement)\n"
            f"  Orange (refined)    : {n_refined} — composite-score refined ostium\n"
            f"  Blue   (stableStart): {n_flare} — flare-zone end (sampling start)\n"
            f"  Green  (P3)         : {n_p3} — wall-snapped ostium (high/medium conf)\n"
            f"  Dim-yellow (LowConf): {n_lowconf} — REJECT-grade best-guess (review required)\n\n"
            f"Expected order: Yellow ≤ Orange ≤ Blue (distal direction).\n"
            f"Diameter sampling and finding detection start at Blue.\n"
        )

        # Append per-branch confidence summary if available
        conf_lines = []
        for bi, meta in logic.branchMeta.items():
            if bi == 0:
                continue
            _oc = meta.get("ostium_confidence")
            if _oc:
                label = (
                    logic.getBranchDisplayName(bi)
                    if hasattr(logic, "getBranchDisplayName")
                    else f"Branch {bi}"
                )
                flag_str = f"  ⚠{','.join(_oc['flags'])}" if _oc.get("flags") else ""
                _eff = _oc.get("effective_score", _oc["score"])
                _pen = _oc.get("flag_penalty", 0.0)
                conf_lines.append(
                    f"  {label}: {_oc['grade']} eff={_eff:.3f} "
                    f"(raw={_oc['score']:.3f} pen={_pen:.2f} ±{_oc.get('uncertainty', 0):.3f}){flag_str}"
                )
        if conf_lines:
            msg += "\nOstium Confidence:\n" + "\n".join(conf_lines)

        # ── Sphere-ring markers for right iliac ostium diagnosis ─────────────
        # Two hollow sphere rings placed at:
        #   CYAN  — known-correct target  (R 20.9, A 149.8, S 1780.2, Ø 11.19mm)
        #   MAGENTA — currently committed ostium for the right-iliac branch
        # Both use vtkSphereSource + vtkPolyDataMapper + vtkActor so they
        # appear as wireframe rings on top of the vessel surface without
        # blocking the view.  Radius = half the expected vessel diameter at
        # that point so the ring sits exactly at the lumen wall.
        try:
            import vtk

            def _add_sphere_ring(name, centre, radius_mm, rgb, label_text=""):
                """Add a wireframe sphere at *centre* (R,A,S) with given radius."""
                src = vtk.vtkSphereSource()
                src.SetCenter(float(centre[0]), float(centre[1]), float(centre[2]))
                src.SetRadius(float(radius_mm))
                src.SetThetaResolution(24)
                src.SetPhiResolution(24)
                src.Update()

                model_node = scene.AddNewNodeByClass("vtkMRMLModelNode", name)
                model_node.SetAndObservePolyData(src.GetOutput())
                model_node.CreateDefaultDisplayNodes()
                dn = model_node.GetDisplayNode()
                dn.SetColor(*rgb)
                dn.SetOpacity(0.55)
                dn.SetRepresentation(1)  # 1 = Wireframe
                dn.SetLineWidth(3)
                dn.SetVisibility(True)

                # Add a tiny fiducial at centre so it shows up in the list
                fid = scene.AddNewNodeByClass(
                    "vtkMRMLMarkupsFiducialNode", name + "_Label"
                )
                fid.CreateDefaultDisplayNodes()
                fdn = fid.GetDisplayNode()
                fdn.SetSelectedColor(*rgb)
                fdn.SetColor(*rgb)
                fdn.SetTextScale(3.5)
                fdn.SetGlyphScale(0.5)
                fid.AddControlPoint(
                    float(centre[0]), float(centre[1]), float(centre[2]), label_text
                )
                return model_node

            # Target ostium (known-correct from user measurement)
            _TARGET_PT = (20.9, 149.8, 1780.2)
            _TARGET_R = 11.19 / 2.0  # Ø 11.19 mm → radius 5.595 mm
            _add_sphere_ring(
                "OstiumDebug_SphereRing_Target",
                _TARGET_PT,
                _TARGET_R,
                rgb=(0.0, 0.9, 0.9),  # cyan
                label_text=f"TARGET R20.9 A149.8 S1780.2 Ø11.19",
            )

            # Actual committed ostium for the right iliac branch
            _right_bi = getattr(logic, "_iliacRightBi", None)
            if _right_bi is None:
                # Fallback: find the iliac_right role branch
                for _xbi, _xm in logic.branchMeta.items():
                    if (
                        _xm.get("role") in ("iliac_right", "main")
                        and _xm.get("lateral_label") == "Right"
                    ):
                        _right_bi = _xbi
                        break
            if _right_bi is None:
                # Last resort: the main branch with the largest X tip
                _best_x = None
                for _xbi, _xm in logic.branchMeta.items():
                    if _xm.get("role") in ("iliac_right", "main"):
                        _ogi_x = _xm.get("ostiumGi")
                        if _ogi_x is not None and _ogi_x < len(logic.points):
                            _px = logic.points[_ogi_x][0]
                            if _best_x is None or _px > _best_x:
                                _best_x = _px
                                _right_bi = _xbi

            if _right_bi is not None:
                _rm = logic.branchMeta.get(_right_bi, {})
                _r_ogi = _rm.get("ostiumGi")
                if _r_ogi is not None and _r_ogi < len(logic.points):
                    _r_pt = logic.points[_r_ogi]
                    _r_diam = (
                        logic.diameters[_r_ogi]
                        if logic.diameters and _r_ogi < len(logic.diameters)
                        else 12.0
                    )
                    _add_sphere_ring(
                        "OstiumDebug_SphereRing_Actual",
                        _r_pt,
                        _r_diam / 2.0,
                        rgb=(1.0, 0.2, 0.8),  # magenta
                        label_text=(
                            f"ACTUAL R{_r_pt[0]:.1f} A{_r_pt[1]:.1f} "
                            f"S{_r_pt[2]:.1f} Ø{_r_diam:.2f}"
                        ),
                    )
                    print(
                        f"[IliacOstium] SphereRing ACTUAL bi={_right_bi} "
                        f"R{_r_pt[0]:.1f} A{_r_pt[1]:.1f} S{_r_pt[2]:.1f} "
                        f"Ø{_r_diam:.2f}mm"
                    )
                else:
                    print(
                        "[IliacOstium] SphereRing ACTUAL: right iliac ostiumGi missing"
                    )
            else:
                print(
                    "[IliacOstium] SphereRing ACTUAL: could not identify right iliac branch"
                )

            msg += (
                "\n\nSphere rings:\n"
                "  CYAN    — target   R20.9 A149.8 S1780.2  Ø11.19mm\n"
                "  MAGENTA — actual committed right-iliac ostium\n"
                "  Axial offset between rings = positioning error to fix."
            )
            print(
                "[IliacOstium] SphereRing TARGET placed at (20.9, 149.8, 1780.2) r=5.60mm"
            )

        except Exception as _sph_err:
            import traceback

            print(
                f"[IliacOstium] SphereRing creation failed: {_sph_err}\n"
                f"{traceback.format_exc()}"
            )

        print(f"[VesselAnalyzer] {msg}")
        self.branchSurfResultLabel.setStyleSheet(
            "font-size: 12px; padding: 6px; border-radius: 4px; "
            "background: #e3f2fd; color: #0d47a1;"
        )
        self.branchSurfResultLabel.setText(
            f"Debug: {n_bif} bif / {n_p2} ostium / {n_refined} refined / "
            f"{n_flare} stable / {n_p3} P3 / {n_lowconf} low-conf | "
            f"SphereRings: CYAN=target MAGENTA=actual"
        )

    def onDetectCollaterals(self):
        if not self.logic.points:
            slicer.util.errorDisplay("Please load a centerline first.")
            return
        if not getattr(self.logic, "branchMeta", {}):
            slicer.util.errorDisplay(
                "Branch metadata not available.\n"
                "Please reload the centerline using 'Load for IVUS Navigation' first."
            )
            return
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            collaterals = self.logic._detectCollaterals()
            self.logic.applyCollateralOverlay()
            if not collaterals:
                msg = (
                    "No collaterals detected.\n\n"
                    "All branches match expected anatomy "
                    "(normal bifurcation angles and diameters)."
                )
            else:
                lines = [f"Detected {len(collaterals)} potential collateral(s):\n"]
                for c in collaterals:
                    lines.append(
                        f"  {c['label']}\n"
                        f"    Confidence: {c['confidence']}  (score {c['score']})\n"
                        f"    Length: {c['length_mm']:.0f}mm  |  "
                        f"Max diameter: {c['maxD']:.1f}mm\n"
                        f"    Branch angle: {c['angle_deg']:.0f}°  |  "
                        f"Origin Z: {c['originZ']:.0f}\n"
                        f"    Criteria: {c['reasons']}\n"
                    )
                lines.append("\nCyan = collateral on vessel mesh.")
                msg = "\n".join(lines)
            slicer.util.infoDisplay(msg)
        except Exception as e:
            import traceback

            slicer.util.errorDisplay(
                f"Collateral detection failed:\n{e}\n{traceback.format_exc()}"
            )
        finally:
            slicer.app.restoreOverrideCursor()



# ── Slicer module-scanner guard ───────────────────────────────────────────────
class vessel_surface_findings_widget_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_surface_findings_widget_mixin"
            parent.hidden = True
