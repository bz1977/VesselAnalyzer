"""
vessel_intervention_widget_mixin.py — VesselAnalyzerWidget pre-dilate, POT, carina, balloon, stent-pick and report handlers

Extracted from VesselAnalyzer.py (VesselAnalyzerWidget methods).

Usage in VesselAnalyzer.py
--------------------------
Add to the VesselAnalyzerWidget class definition::

    from vessel_intervention_widget_mixin import InterventionWidgetMixin

    class VesselAnalyzerWidget(InterventionWidgetMixin, ScriptedLoadableModuleWidget, VTKObservationMixin):
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


class InterventionWidgetMixin:
    """Mixin: VesselAnalyzerWidget pre-dilate, POT, carina, balloon, stent-pick and report handlers"""
    def onPreDilate(self):
        """Visualize balloon pre-dilation at all detected stenoses."""
        # Disable immediately to prevent double-click re-triggers
        self.preDilateButton.setEnabled(False)
        self.preDilateButton.setText("⏳ Pre-dilating...")
        slicer.app.processEvents()

        findings = getattr(self.logic, "findings", [])
        stenoses = [
            f
            for f in findings
            if "Pancak" in f.get("type", "") or "Compress" in f.get("type", "")
        ]

        if not stenoses:
            self.preDilateStatusLabel.setText(
                "⚠️ No stenoses found — run Find Lesions first."
            )
            self.preDilateButton.setEnabled(True)
            self.preDilateButton.setText("🎈 Pre-Dilate Lesion  (balloon angioplasty)")
            return

        # Freeze findings — balloon inflates diameters, which would cause
        # false ectasia/aneurysm detections if _detectFindings ran again
        self.logic._findingsFrozen = True

        # Remove any existing balloons
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if node.GetName().startswith("BalloonDilate"):
                slicer.mrmlScene.RemoveNode(node)

        self.preDilateStatusLabel.setText("Step 2: old balloons removed...")
        slicer.app.processEvents()

        placed = 0
        balloon_info = []
        diams = self.logic.diameters

        # ── Detect kissing configuration ─────────────────────────────────────
        # If two stenoses are in sibling downstream branches (both branch off
        # the same trunk end), size both balloons to the smaller reference
        # diameter for symmetric simultaneous expansion.
        hub_gi = self.logic.branches[0][1] - 1 if self.logic.branches else -1

        def _branch_stable_median(bs, be):
            st_s = bs + max(5, (be - bs) // 5)
            st_e = be - max(3, (be - bs) // 10)
            vals = sorted(
                [
                    diams[j]
                    for j in range(st_s, st_e)
                    if j < len(diams) and diams[j] > 1.0
                ]
            )
            return vals[len(vals) // 2] if vals else 0.0

        def _compute_healthy_d(gi, bi):
            bs, be = self.logic.branches[bi]
            branch_median = _branch_stable_median(bs, be)
            lo_near = max(bs, gi - 30)
            hi_near = min(be - 1, gi + 30)
            lo_excl = max(bs, gi - 5)
            hi_excl = min(be - 1, gi + 5)
            near_pts = sorted(
                [
                    diams[j]
                    for j in range(lo_near, hi_near + 1)
                    if j < len(diams)
                    and diams[j] > 1.0
                    and not (lo_excl <= j <= hi_excl)
                ]
            )
            if len(near_pts) < 4:
                lo2 = max(bs, gi - 60)
                hi2 = min(be - 1, gi + 60)
                near_pts = sorted(
                    [
                        diams[j]
                        for j in range(lo2, hi2 + 1)
                        if j < len(diams)
                        and diams[j] > 1.0
                        and not (lo_excl <= j <= hi_excl)
                    ]
                )
            hd = (
                near_pts[int(len(near_pts) * 0.75)]
                if len(near_pts) >= 4
                else max(diams[gi] * 2.0, 6.0)
            )
            if branch_median > 0:
                hd = min(hd, branch_median * 1.1)
            if len(near_pts) >= 4:
                hd = min(hd, near_pts[len(near_pts) // 2] * 1.2)
            return hd, branch_median, len(near_pts)

        # Check if any two stenoses are in different downstream branches
        # (sibling branches = both connected to the same hub)
        kissing_override = None  # if set, (diam_A, diam_B) both use matched size
        if len(stenoses) >= 2:
            # Group stenoses by branch
            by_branch = {}
            for f in stenoses:
                bi = f.get("branchIdx", 0)
                if bi > 0:  # downstream branch only
                    by_branch.setdefault(bi, []).append(f)
            sibling_branches = [bi for bi in by_branch if bi != 0]
            if len(sibling_branches) >= 2:
                # Two or more sibling stenoses — kissing configuration
                # Compute healthy_d for each, use the smaller for both
                healthy_ds = {}
                for bi in sibling_branches:
                    f0 = by_branch[bi][0]
                    hd, _, _ = _compute_healthy_d(f0["pointIdx"], bi)
                    healthy_ds[bi] = hd
                matched_d = min(healthy_ds.values())
                kissing_override = {bi: matched_d for bi in sibling_branches}
                print(
                    f"[Balloon] Kissing config detected — matching balloon diameter to {matched_d:.1f}mm "
                    f"(branches: {list(sibling_branches)})"
                )

        for f in stenoses:
            gi = f.get("pointIdx", -1)
            bi = f.get("branchIdx", 0)
            if gi < 0 or gi >= len(self.logic.points):
                continue
            bs, be = self.logic.branches[bi]

            # Use kissing-matched diameter if applicable, else compute independently
            if kissing_override and bi in kissing_override:
                healthy_d = kissing_override[bi]
                branch_median = _branch_stable_median(bs, be)
                near_count = 0  # already computed above
            else:
                healthy_d, branch_median, near_count = _compute_healthy_d(gi, bi)
            # Compressed diameter at lesion
            compressed_d = diams[gi] if gi < len(diams) else 4.0

            # Balloon sized to healthy vessel (simulates full expansion)
            # 5% oversize is standard practice to ensure full apposition
            balloon_diam = round(healthy_d * 1.05 / 0.5) * 0.5

            # ── Balloon span: scan for compressed zone then cap by physical length ──
            recovery_thresh = healthy_d * 0.75  # 75% of healthy = recovered
            CONSEC_NEEDED = 3  # consecutive recovered pts to stop
            BUFFER = 5  # safety pts beyond compressed zone
            MAX_BALLOON_MM = 60.0  # hard cap — balloon never > 60mm long

            def median3(idx):
                """3-point median diameter — noise-resistant."""
                vals = sorted(
                    self.logic.diameters[j]
                    for j in range(
                        max(0, idx - 1), min(len(self.logic.diameters), idx + 2)
                    )
                    if self.logic.diameters[j] > 0.5
                )
                return vals[len(vals) // 2] if vals else 0.0

            # Walk DISTALLY from lesion centre
            zone_end = gi
            consec = 0
            for gj in range(gi, min(be, gi + 80)):
                if median3(gj) < recovery_thresh:
                    zone_end = gj
                    consec = 0
                else:
                    consec += 1
                    if consec >= CONSEC_NEEDED:
                        break

            # Walk PROXIMALLY from lesion centre
            zone_start = gi
            consec = 0
            for gj in range(gi, max(bs - 1, gi - 80), -1):
                if median3(gj) < recovery_thresh:
                    zone_start = gj
                    consec = 0
                else:
                    consec += 1
                    if consec >= CONSEC_NEEDED:
                        break

            # Apply buffer and clamp to branch
            b_end = min(be - 1, zone_end + BUFFER)
            b_start = max(bs, zone_start - BUFFER)

            # Hard cap: limit balloon to MAX_BALLOON_MM physical length
            if (
                len(self.logic.distances) > b_end
                and len(self.logic.distances) > b_start
            ):
                span_mm = self.logic.distances[b_end] - self.logic.distances[b_start]
                if span_mm > MAX_BALLOON_MM:
                    # Trim distally to stay within cap
                    for _ti in range(b_end, b_start, -1):
                        if _ti < len(self.logic.distances):
                            if (
                                self.logic.distances[_ti]
                                - self.logic.distances[b_start]
                                <= MAX_BALLOON_MM
                            ):
                                b_end = _ti
                                break

            # For bifurcation lesions: extend proximally into parent branch (capped)
            if zone_start - bs <= 10 and len(self.logic.branches) > 1:
                parent_bs, parent_be = self.logic.branches[0]
                b_start = max(parent_bs, bs - 10)

            balloon_pts = list(range(b_start, b_end + 1))
            if len(balloon_pts) < 2:
                continue
            self.logic._placeBalloon3D(
                balloon_pts, balloon_diam / 2.0, f"BalloonDilate_{placed}"
            )

            # ── FUNCTIONAL EFFECT 1: Override diameters ────────────────
            # Replace compressed diameter values with healthy_d over the
            # balloon span so IVUS, measurements and stent sizing see expanded lumen
            if not hasattr(self.logic, "preDilationMap"):
                self.logic.preDilationMap = {}
            if not hasattr(self.logic, "_origDiameters"):
                self.logic._origDiameters = {}  # backup for restore
            import math as _mth

            _n_bp = max(len(balloon_pts) - 1, 1)
            for _k, gj in enumerate(balloon_pts):
                if gj < len(self.logic.diameters):
                    if gj not in self.logic._origDiameters:
                        self.logic._origDiameters[gj] = self.logic.diameters[gj]
                    # Taper: plateau at healthy_d in middle 60%, taper at edges
                    _rel = _k / _n_bp  # 0..1
                    if _rel < 0.15:
                        t = _rel / 0.15  # ramp up
                    elif _rel > 0.85:
                        t = (1.0 - _rel) / 0.15  # ramp down
                    else:
                        t = 1.0  # full healthy in middle
                    self.logic.diameters[gj] = max(
                        self.logic._origDiameters[gj],  # never shrink a point
                        compressed_d + (healthy_d - compressed_d) * t,
                    )

            # Key every balloon point so placeStent3D floors radius across the full zone
            _pdentry = {
                "balloon_diam": balloon_diam,
                "healthy_diam": healthy_d,
                "compressed_diam": compressed_d,
                "branch": bi,
                "balloon_pts": balloon_pts,
            }
            for _gj in balloon_pts:
                self.logic.preDilationMap[_gj] = _pdentry
            # Also key by lesion center for lookup convenience
            self.logic.preDilationMap[gi] = _pdentry

            # Clear finding_type for all ballooned points so the
            # COMPRESSION / PANCAKING label no longer appears in
            # the IVUS panel after balloon expansion
            if hasattr(self.logic, "finding_type"):
                for _gj in balloon_pts:
                    if _gj < len(self.logic.finding_type):
                        self.logic.finding_type[_gj] = 0
            # Expand vessel mesh to match balloon diameter
            try:
                self.logic._expandMeshAtLesion(
                    balloon_pts, compressed_d / 2.0, balloon_diam / 2.0
                )
            except Exception as _me:
                print(f"[PreDilate] mesh expand failed (non-fatal): {_me}")

            # Store for summary display
            expansion_pct = (
                int((balloon_diam / compressed_d - 1) * 100) if compressed_d > 0 else 0
            )
            balloon_info.append(
                {
                    "type": f.get("type", ""),
                    "compressed_d": compressed_d,
                    "balloon_d": balloon_diam,
                    "healthy_d": healthy_d,
                    "expansion_pct": expansion_pct,
                }
            )
            placed += 1

        # Refresh IVUS display so updated diameters show immediately
        if placed:
            try:
                self.updateMeasurementsAtIndex(int(self.pointSlider.value))
            except Exception:
                pass

        if placed:
            lines = [f"🎈 {placed} balloon(s) pre-dilation:"]
            for i, bi in enumerate(balloon_info):
                lines.append(
                    f"  #{i+1} {bi['type']}: "
                    f"Ø{bi['compressed_d']:.1f}mm → Ø{bi['balloon_d']:.1f}mm "
                    f"(+{bi['expansion_pct']}% expansion)"
                )
            lines.append("➡️ Now place stent over the dilated segment.")
            msg = "\n".join(lines)
            self.preDilateStatusLabel.setStyleSheet(
                "font-size: 11px; color: #6c3483; font-style: italic; font-weight: bold;"
            )
            self.preDilateStatusLabel.setText(msg)
            self._preDilateStatusMsg = msg  # stored for copy button
            self.preDilateCopyBtn.setVisible(True)
            # Update button to show balloons are active
            self.preDilateButton.setText("🎈 Balloons inflated — ready to place stent")
            self.compareBeforeAfterButton.setEnabled(True)
            self.exitCompareButton.setEnabled(True)
            self.addRulerButton.setEnabled(True)
            self.removeRulerButton.setEnabled(True)
            self._buildLesionTable()

            # ── Remove 3D overlay markers for ballooned stenoses ──────────
            # The finding markers (vtkMRMLMarkupsFiducialNode) placed by
            # applyColorOverlay() persist in the scene after ballooning.
            # Remove markers whose pointIdx is now covered by a balloon,
            # then redraw the overlay so only un-ballooned findings remain.
            pdmap = getattr(self.logic, "preDilationMap", {})
            ballooned_pts = set(pdmap.keys())
            to_remove = []
            for i in range(slicer.mrmlScene.GetNumberOfNodes()):
                n = slicer.mrmlScene.GetNthNode(i)
                if n and n.GetAttribute("VesselAnalyzerOverlay") == "1":
                    name = n.GetName()  # e.g. "Finding_Pancaking_142"
                    try:
                        pt_idx = int(name.rsplit("_", 1)[-1])
                        if pt_idx in ballooned_pts:
                            to_remove.append(n)
                    except (ValueError, IndexError):
                        pass
            for n in to_remove:
                slicer.mrmlScene.RemoveNode(n)

            # Navigate IVUS to first lesion
            gi0 = stenoses[0].get("pointIdx", -1)
            bi0 = stenoses[0].get("branchIdx", 0)
            if gi0 >= 0:
                # Switch to "All branches" view (index 0 in combo = All)
                # so gi0 maps directly as local index
                try:
                    self.branchSelector.blockSignals(True)
                    self.branchSelector.setCurrentIndex(0)  # "All branches"
                    self.branchSelector.blockSignals(False)
                    self.logic.activeBranch = -1
                    self.pointSlider.maximum = self.logic.getNumPoints() - 1
                except Exception:
                    pass
                self.pointSlider.blockSignals(True)
                self.pointSlider.setValue(self.logic.globalToTraversal(gi0))
                self.pointSlider.blockSignals(False)
                self.updateMeasurementsAtIndex(
                    self.logic.globalToTraversal(gi0), globalIdx=gi0
                )
                self.logic.moveCrosshairToPoint(gi0)
                self.logic.updateVisualizations(
                    gi0,
                    showSphere=self.showSphereCheck.isChecked(),
                    showRing=self.showRingCheck.isChecked(),
                    showLine=self.showLineCheck.isChecked(),
                    sphereColor=self._getSphereColor(gi0),
                )
                # Force refresh of branch min/max stats to reflect dilated diameters
                try:
                    bi0 = self.logic.getBranchForPoint(gi0)
                    if bi0 >= 0:
                        bStats = self.logic.getBranchStats(bi0)
                        self.branchMinLabel.setText(f"{bStats['min']:.2f} mm")
                        self.branchMaxLabel.setText(f"{bStats['max']:.2f} mm")
                        self.branchAvgLabel.setText(f"{bStats['avg']:.2f} mm")
                except Exception:
                    pass

            # ── Set opacity LAST — deferred so MRML scene events can't override it ──
            _model_dn = None
            if self.logic.modelNode:
                _model_dn = self.logic.modelNode.GetDisplayNode()
            if _model_dn is None:
                for _mn in slicer.util.getNodesByClass("vtkMRMLModelNode"):
                    _nm = _mn.GetName()
                    if not any(
                        x in _nm
                        for x in ("Stent", "Balloon", "Expanded", "Ring", "Embo")
                    ):
                        _model_dn = _mn.GetDisplayNode()
                        if _model_dn:
                            break

            def _applyZeroOpacity():
                for _mn in slicer.util.getNodesByClass("vtkMRMLModelNode"):
                    _nm = _mn.GetName()
                    # Keep only balloon and its ring visible — hide everything else
                    if _nm.startswith("BalloonDilate"):
                        continue
                    _dn2 = _mn.GetDisplayNode()
                    if _dn2:
                        _dn2.SetVisibility(0)
                        _dn2.SetOpacity(0.0)
                self.modelOpacitySlider.blockSignals(True)
                self.modelOpacitySlider.setValue(0.0)
                self.modelOpacitySlider.setEnabled(False)
                self.modelOpacitySlider.blockSignals(False)

            qt.QTimer.singleShot(100, _applyZeroOpacity)
            qt.QTimer.singleShot(600, _applyZeroOpacity)
        else:
            self.preDilateStatusLabel.setText(
                "⚠️ Could not place balloon — check centerline."
            )
            self.preDilateButton.setEnabled(True)
            self.preDilateButton.setText("🎈 Pre-Dilate Lesion  (balloon angioplasty)")

    # ── POT (Proximal Optimization Technique) ────────────────────────────

    def onApplyPOT(self):
        """Apply Proximal Optimization Technique: inflate a single larger balloon
        in the trunk segment to restore circular lumen after kissing/Y-stent deployment.
        """
        import math

        try:
            plan = getattr(self.logic, "stentPlan", None)
            if not plan:
                self.potStatusLabel.setText("⚠️ Place a kissing or Y-stent first.")
                return

            plan_type = plan.get("type", "")
            if "Kissing" not in plan_type and "Bifurcated" not in plan_type:
                self.potStatusLabel.setText(
                    "⚠️ POT applies after Kissing or Y-stent only."
                )
                return

            branches = self.logic.branches
            pts = self.logic.points
            hub_gi = branches[0][1] - 1  # trunk end = bifurcation hub

            # POT spans from gi_prox to hub
            y_trunk = plan.get("y_trunk", [])
            if y_trunk and len(y_trunk) >= 2:
                # y_trunk is ascending [prox, ..., hub] — first point is most proximal
                prox_gi = y_trunk[0]
            else:
                prox_gi = plan.get("proxPt", max(0, hub_gi - 20))
                trunk_bs, trunk_be = branches[0]
                prox_gi = max(trunk_bs, min(prox_gi, trunk_be - 1))

            # pot_path: proximal → distal (ascending)
            pot_start = min(prox_gi, hub_gi)
            pot_end = max(prox_gi, hub_gi)

            # Enforce minimum POT length of 15mm
            MIN_POT_MM = 15.0
            import math as _pm

            pot_path = list(range(pot_start, pot_end + 1))
            pot_span = sum(
                _pm.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
                for i in range(pot_start, min(pot_end, len(pts) - 1))
            )
            if pot_span < MIN_POT_MM and len(branches) > 0:
                # Walk further back into trunk until we have ≥ 15mm
                trunk_bs2 = branches[0][0]
                acc = 0.0
                new_start = pot_start
                for ti in range(pot_start - 1, trunk_bs2 - 1, -1):
                    if ti + 1 < len(pts):
                        p0, p1 = pts[ti], pts[ti + 1]
                        acc += _pm.sqrt(sum((p1[k] - p0[k]) ** 2 for k in range(3)))
                    new_start = ti
                    if acc >= MIN_POT_MM:
                        break
                pot_start = new_start
                pot_path = list(range(pot_start, pot_end + 1))
                print(
                    f"[POT] Extended prox from pt{prox_gi} to pt{pot_start} for min {MIN_POT_MM}mm"
                )

            if len(pot_path) < 2:
                self.potStatusLabel.setText("⚠️ Trunk segment too short for POT.")
                return

            # POT diameter: sample ONLY from the distal trunk zone (near bifurcation)
            # This avoids the wide proximal aortic section inflating the reference
            diams = self.logic.diameters
            distal_zone = pot_path[max(0, len(pot_path) - len(pot_path) // 3) :]
            trunk_vals = sorted(
                [diams[i] for i in distal_zone if i < len(diams) and diams[i] > 1.0]
            )
            # Use median (not 75th pct) to get true pre-bifurcation diameter
            trunk_vd = trunk_vals[len(trunk_vals) // 2] if trunk_vals else 12.0
            pot_diam = round(trunk_vd * 1.05 / 0.5) * 0.5

            # Remove existing POT balloon if present
            for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
                if node.GetName() == "POT_Balloon":
                    slicer.mrmlScene.RemoveNode(node)

            # Place POT balloon
            self.logic._placeBalloon3D(pot_path, pot_diam / 2.0, "POT_Balloon")

            pot_mm = sum(
                math.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
                for i in range(pot_start, min(pot_end, len(pts) - 1))
            )

            self.potStatusLabel.setText(
                f"✅ POT applied: Ø{pot_diam:.1f}mm × {pot_mm:.0f}mm in trunk  "
                f"(vessel Ø{trunk_vd:.1f}mm)"
            )
            self.potButton.setEnabled(False)
            self.potRemoveButton.setEnabled(True)
            print(
                f"[POT] Ø{pot_diam:.1f}mm × {pot_mm:.0f}mm "
                f"trunk pt{pot_start}→{pot_end}  vessel_vd={trunk_vd:.1f}mm"
            )

        except Exception as e:
            import traceback

            self.potStatusLabel.setText(f"⚠️ POT error: {e}")
            print(f"[POT] Error: {e}\n{traceback.format_exc()}")

    def onRemovePOT(self):
        """Remove the POT balloon."""
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if node.GetName() == "POT_Balloon":
                slicer.mrmlScene.RemoveNode(node)
        self.potButton.setEnabled(True)
        self.potRemoveButton.setEnabled(False)
        self.potStatusLabel.setText("POT balloon removed.")

    def onApplyCarinaSupportSIM(self):
        """Place a short flared tube at the bifurcation confluence to simulate
        carina support. Spans ~10-15pts into each downstream branch from hub,
        sized larger than the kissing stents to force circular lumen at junction.
        This represents the ideal geometry correction — not an additional stent,
        but a visualization of the target lumen shape at the confluence."""
        import math, vtk

        try:
            plan = getattr(self.logic, "stentPlan", None)
            if not plan:
                self.carinaStatusLabel.setText(
                    "⚠️ Place a Y-stent or kissing stent first."
                )
                return

            branches = self.logic.branches
            pts = self.logic.points
            diams = self.logic.diameters
            hub_gi = branches[0][1] - 1

            # Remove existing carina node
            for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
                if node.GetName() == "CarinaSupport":
                    slicer.mrmlScene.RemoveNode(node)

            # Target confluence diameter: force circular lumen
            # Use the geometric mean of the two branch diameters
            # This is the diameter that would give equal flow area in both branches
            y_left = plan.get("y_left", [])
            y_right = plan.get("y_right", [])

            def _branch_stable_diam(limb_path):
                """Get median diameter from stable mid-branch zone.
                Uses the branch's own stable range (20%-80% of branch length),
                completely avoiding the proximal junction artifact zone."""
                # Find which branch this limb belongs to
                real_pts = [gi for gi in limb_path if gi >= 0 and gi < len(diams)]
                if not real_pts:
                    return 10.0
                # Find branch for the middle point of the limb
                mid_gi = real_pts[len(real_pts) // 2]
                bi_found = None
                for bi, (bs, be) in enumerate(branches):
                    if bs <= mid_gi < be:
                        bi_found = bi
                        break
                if bi_found is None or bi_found == 0:
                    bi_found = 1  # default to branch 1 if not found

                bs_b, be_b = branches[bi_found]
                # Sample from 25%-75% of branch length — fully avoids junction zone
                stable_s = bs_b + (be_b - bs_b) // 4
                stable_e = bs_b + (be_b - bs_b) * 3 // 4
                stable_pts = list(range(stable_s, stable_e))
                vals = sorted(
                    [
                        diams[gi]
                        for gi in stable_pts
                        if gi < len(diams) and diams[gi] > 1.0
                    ]
                )
                if not vals:
                    return 10.0
                return vals[len(vals) // 2]  # median

            d_left = _branch_stable_diam(y_left)
            d_right = _branch_stable_diam(y_right)

            # Target confluence diameter: use UI override if set, else auto
            override_val = getattr(self, "carinaDiamSpin", None)
            asym_spin = getattr(self, "carinaAsymSpin", None)
            override_d = override_val.value if override_val else 0.0
            asym_val = asym_spin.value if asym_spin else 0.0

            if override_d > 0.0:
                target_d = round(override_d / 0.5) * 0.5
            else:
                # Auto: average of healthy branch zones × 0.85
                # Carina balloon targets 10-11mm — not the full branch diameter.
                # The confluence needs less force than the branches to open;
                # × 0.85 of branch average gives the correct 10-11mm range
                # for typical iliac vein bifurcations (branches ~12-13mm)
                target_d = round(((d_left + d_right) / 2.0) * 0.85 / 0.5) * 0.5
                target_d = max(10.0, target_d)  # floor at 10mm
            target_r = target_d / 2.0

            # Asymmetry: asym_val>0 → left gets larger, right smaller
            # Breaking symmetric collision can open central lumen
            r_branch_left = max(3.0, (d_left + asym_val) / 2.0)
            r_branch_right = max(3.0, (d_right - asym_val) / 2.0)

            # Build confluence geometry: hub point + short entry into each branch
            SPAN_PTS = 15  # points into each branch from hub

            # Trunk approach: last 10pts of y_trunk before hub (proximal → hub direction)
            y_trunk = plan.get("y_trunk", [])
            # y_trunk is ascending [prox, ..., hub] — take last 10 points (near hub)
            trunk_near_hub = [gi for gi in y_trunk[-min(10, len(y_trunk)) :] if gi >= 0]
            # trunk_near_hub is already [hub-9, ..., hub] — correct direction for tube
            trunk_approach = trunk_near_hub  # proximal end → hub

            # Branch A approach: first SPAN_PTS of left limb (after hub point)
            left_approach = [gi for gi in y_left[1 : SPAN_PTS + 1] if gi >= 0]
            # Branch B approach: first SPAN_PTS of right limb (after hub point)
            right_approach = [gi for gi in y_right[1 : SPAN_PTS + 1] if gi >= 0]

            if not left_approach or not right_approach:
                self.carinaStatusLabel.setText("⚠️ Could not build confluence geometry.")
                return

            # Build three short tube segments meeting at hub with tapered radius:
            # trunk_approach: tapers FROM smaller (branch size) TO target_r at hub
            # left/right:     tapers FROM target_r at hub TO smaller at branch end
            appender = vtk.vtkAppendPolyData()

            def _make_flared_tube(path_gis, r_start, r_end, name):
                """Build a tube that flares from r_start to r_end along path."""
                if len(path_gis) < 2:
                    return None
                cp = vtk.vtkPoints()
                cl = vtk.vtkCellArray()
                cr = vtk.vtkFloatArray()
                cr.SetName("TubeRadius")
                cl.InsertNextCell(len(path_gis))
                for ri, gi in enumerate(path_gis):
                    if gi < 0 or gi >= len(pts):
                        continue
                    t = ri / max(len(path_gis) - 1, 1)
                    r = r_start + (r_end - r_start) * t
                    cp.InsertNextPoint(pts[gi][0], pts[gi][1], pts[gi][2])
                    cl.InsertCellPoint(ri)
                    cr.InsertNextValue(r)
                cpd = vtk.vtkPolyData()
                cpd.SetPoints(cp)
                cpd.SetLines(cl)
                cpd.GetPointData().AddArray(cr)
                cpd.GetPointData().SetActiveScalars("TubeRadius")
                tf = vtk.vtkTubeFilter()
                tf.SetInputData(cpd)
                tf.SetNumberOfSides(32)
                tf.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
                tf.CappingOn()
                tf.Update()
                return tf.GetOutput()

            # r_branch_left/right already computed above with asymmetry offset

            # Trunk approach: small → large (flares outward toward hub)
            t1 = _make_flared_tube(
                trunk_approach + [hub_gi], r_branch_left, target_r, "trunk"
            )
            # Left limb: large → small (narrows from hub into branch)
            t2 = _make_flared_tube(
                [hub_gi] + left_approach, target_r, r_branch_left, "left"
            )
            # Right limb: large → small
            t3 = _make_flared_tube(
                [hub_gi] + right_approach, target_r, r_branch_right, "right"
            )

            for t in [t1, t2, t3]:
                if t:
                    appender.AddInputData(t)
            appender.Update()

            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", "CarinaSupport"
            )
            node.SetAndObservePolyData(appender.GetOutput())
            node.CreateDefaultDisplayNodes()
            dn = node.GetDisplayNode()
            dn.SetColor(0.0, 0.6, 1.0)  # bright blue
            dn.SetOpacity(0.75)
            dn.SetRepresentation(2)
            dn.SetAmbient(0.4)
            dn.SetVisibility(1)

            conf_mm_left = sum(
                math.sqrt(
                    sum(
                        (pts[left_approach[i + 1]][k] - pts[left_approach[i]][k]) ** 2
                        for k in range(3)
                    )
                )
                for i in range(len(left_approach) - 1)
            )
            conf_mm_right = sum(
                math.sqrt(
                    sum(
                        (pts[right_approach[i + 1]][k] - pts[right_approach[i]][k]) ** 2
                        for k in range(3)
                    )
                )
                for i in range(len(right_approach) - 1)
            )

            self.carinaStatusLabel.setText(
                f"✅ Carina Ø{target_d:.1f}mm at confluence  "
                f"L={r_branch_left*2:.1f}mm / R={r_branch_right*2:.1f}mm"
                + (f"  asym={asym_val:+.1f}mm" if asym_val != 0 else "")
            )
            self.carinaButton.setEnabled(False)
            self.carinaRemoveButton.setEnabled(True)
            print(
                f"[Carina] Ø{target_d:.1f}mm at hub pt{hub_gi}  "
                f"left={d_left:.1f}mm right={d_right:.1f}mm"
            )

        except Exception as e:
            import traceback

            self.carinaStatusLabel.setText(f"⚠️ Carina error: {e}")
            print(f"[Carina] Error: {e}\n{traceback.format_exc()}")

    def onYStentReplace(self):
        """Re-place Y-stent using saved limb paths with updated branch diameters.
        Uses yLeftDiamSpin/yRightDiamSpin for branch asymmetry, and
        stentProxDiamSpin for trunk diameter override."""
        import math as _ksm

        try:
            plan = getattr(self.logic, "stentPlan", None)
            if not plan or "y_left" not in plan:
                self.stentPickStatusLabel.setText("⚠️ Place a Y-stent first.")
                return

            y_left = plan.get("y_left", [])
            y_right = plan.get("y_right", [])
            diams = self.logic.diameters

            def _limb_avg(limb, n=20):
                vals = [
                    diams[gi]
                    for gi in limb[:n]
                    if gi >= 0 and gi < len(diams) and diams[gi] > 1.0
                ]
                return sum(vals) / len(vals) if vals else 10.0

            # Branch diameters: use branch-specific spinboxes if set, else auto
            left_override = (
                self.yLeftDiamSpin.value if self.yLeftDiamSpin.value >= 4.0 else 0.0
            )
            right_override = (
                self.yRightDiamSpin.value if self.yRightDiamSpin.value >= 4.0 else 0.0
            )
            # Also check main distDiam spinbox as a global override
            main_dist = (
                self.stentDistDiamSpin.value
                if self.stentDistDiamSpin.value >= 4.0
                else 0.0
            )
            if main_dist >= 4.0 and left_override == 0.0:
                left_override = main_dist
            if main_dist >= 4.0 and right_override == 0.0:
                right_override = main_dist

            vd_left = left_override if left_override > 0 else _limb_avg(y_left) * 1.15
            vd_right = (
                right_override if right_override > 0 else _limb_avg(y_right) * 1.15
            )

            r_left = round(vd_left / 0.5) * 0.5 / 2.0
            r_right = round(vd_right / 0.5) * 0.5 / 2.0

            # Trunk diameter: use main proxDiam spinbox if set, else auto from vessel
            main_prox = (
                self.stentProxDiamSpin.value
                if self.stentProxDiamSpin.value >= 4.0
                else 0.0
            )
            if main_prox >= 4.0:
                r_trunk = main_prox / 2.0
                print(
                    f"[YReplace] trunk override: proxDiam={main_prox:.1f}mm → r_trunk={r_trunk:.2f}mm"
                )
            else:
                bif_pt = self.logic.branches[0][1] - 1
                trunk_bs_val = self.logic.branches[0][0]
                distal_trunk_start = max(
                    trunk_bs_val, bif_pt - max(10, (bif_pt - trunk_bs_val) // 3)
                )
                trunk_real_pts = list(range(distal_trunk_start, bif_pt + 1))
                trunk_vals = sorted(
                    [
                        diams[i]
                        for i in trunk_real_pts
                        if i < len(diams) and diams[i] > 1.0
                    ]
                )
                vd_trunk = trunk_vals[len(trunk_vals) // 2] if trunk_vals else 12.0
                r_trunk_geom = _ksm.sqrt(r_left**2 + r_right**2)
                r_trunk_max = round(vd_trunk * 1.15 / 0.5) * 0.5 / 2.0
                r_trunk = min(r_trunk_geom, r_trunk_max)

            # Update plan with new radii
            plan["r_trunk"] = r_trunk
            plan["r_left"] = r_left
            plan["r_right"] = r_right
            plan["proxDiam"] = r_trunk * 2.0
            plan["distDiam"] = r_left * 2.0

            self.logic.placeStent3D(plan)
            self._refreshStentSummary()

            self.stentPickStatusLabel.setText(
                f"↺ Re-placed: Left Ø{r_left*2:.1f}mm  Right Ø{r_right*2:.1f}mm  "
                f"Trunk Ø{r_trunk*2:.1f}mm"
            )
            print(
                f"[YReplace] L={r_left*2:.1f}mm R={r_right*2:.1f}mm "
                f"T={r_trunk*2:.1f}mm"
            )

        except Exception as e:
            import traceback

            self.stentPickStatusLabel.setText(f"⚠️ Re-place error: {e}")
            print(f"[YReplace] Error: {e}\n{traceback.format_exc()}")

    def onRemoveCarinaSupport(self):
        """Remove carina support node."""
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if node.GetName() == "CarinaSupport":
                slicer.mrmlScene.RemoveNode(node)
        self.carinaButton.setEnabled(True)
        self.carinaRemoveButton.setEnabled(False)
        self.carinaStatusLabel.setText("Carina support removed.")

    # ── Manual balloon placement ─────────────────────────────────────────

    def onManualBalloonStart(self):
        """Enter 2-point pick mode for manual balloon placement."""
        try:
            if not self.logic.points:
                slicer.util.errorDisplay("Load a centerline first.")
                return
            self.onManualBalloonCancel()  # clean up previous session
            self._balloonPickNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "BalloonPickPoints"
            )
            dn = self._balloonPickNode.GetDisplayNode()
            if dn:
                dn.SetSelectedColor(0.9, 0.3, 0.0)
                dn.SetColor(0.9, 0.3, 0.0)
                dn.SetTextScale(3.0)
                dn.SetGlyphScale(3.0)
            slicer.modules.markups.logic().StartPlaceMode(1)  # persistent
            self.manualBalloonButton.setEnabled(False)
            self.manualBalloonCancelBtn.setEnabled(True)
            self.manualBalloonStatusLabel.setText(
                "📍 Click point 1 (proximal end of balloon zone)..."
            )
            self._balloonPickLastCount = 0
            self._balloonPickSavedPositions = []
            self._balloonPickFinished = False
            self._balloonPickTimer = qt.QTimer()
            self._balloonPickTimer.setInterval(250)
            self._balloonPickTimer.connect("timeout()", self._pollBalloonPickPoints)
            self._balloonPickTimer.start()
            print("[ManualBalloon] pick mode started")
        except Exception as _e:
            import traceback

            traceback.print_exc()
            self.manualBalloonStatusLabel.setText(f"⚠️ Start failed: {_e}")

    def _pollBalloonPickPoints(self):
        """Poll every 250ms until 2 points placed."""
        try:
            if getattr(self, "_balloonPickFinished", False):
                return
            node = getattr(self, "_balloonPickNode", None)
            if not node or not slicer.mrmlScene.IsNodePresent(node):
                self._balloonPickTimer.stop()
                return
            n_total = node.GetNumberOfControlPoints()
            # Slicer Markups: while in place mode, there's always 1 "preview" point
            # n_total-1 = actual placed points
            n_placed = max(0, n_total - 1)

            # Save all confirmed point positions
            positions = []
            for i in range(n_total):
                p = [0.0, 0.0, 0.0]
                try:
                    node.GetNthControlPointPosition(i, p)
                    positions.append(list(p))
                except Exception:
                    pass
            if len(positions) >= 2:
                self._balloonPickSavedPositions = positions[:]

            if n_placed == self._balloonPickLastCount:
                return
            self._balloonPickLastCount = n_placed
            print(f"[ManualBalloon] poll: n_placed={n_placed}")

            if n_placed == 1:
                self.manualBalloonStatusLabel.setText(
                    "📍 Click point 2 (distal end of balloon zone)..."
                )
            elif n_placed >= 2:
                self._balloonPickFinished = True
                self._balloonPickTimer.stop()
                # Use singleShot to finish outside the timer callback
                qt.QTimer.singleShot(200, self._finishBalloonPick)
        except Exception as _e:
            import traceback

            traceback.print_exc()
            print(f"[ManualBalloon] poll error: {_e}")
            try:
                self._balloonPickTimer.stop()
            except:
                pass

    def _finishBalloonPick(self):
        """Snap picked points to centerline and place manual balloon."""
        import traceback

        print("[ManualBalloon] _finishBalloonPick START")
        try:
            self._finishBalloonPickImpl()
        except Exception as _outer:
            traceback.print_exc()
            print(f"[ManualBalloon] OUTER CRASH: {_outer}")
            try:
                self.manualBalloonStatusLabel.setText(f"⚠️ Crashed: {_outer}")
                self.manualBalloonButton.setEnabled(True)
                self.manualBalloonCancelBtn.setEnabled(False)
            except:
                pass

    def _finishBalloonPickImpl(self):
        """Inner implementation — called inside try/except by _finishBalloonPick."""
        import traceback

        try:
            inode = slicer.app.applicationLogic().GetInteractionNode()
            inode.SetCurrentInteractionMode(inode.ViewTransform)
        except Exception as _e:
            print(f"[ManualBalloon] interaction reset error: {_e}")
        self.manualBalloonButton.setEnabled(True)
        self.manualBalloonCancelBtn.setEnabled(False)
        if getattr(self, "_balloonPickNode", None) and slicer.mrmlScene.IsNodePresent(
            self._balloonPickNode
        ):
            slicer.mrmlScene.RemoveNode(self._balloonPickNode)
        self._balloonPickNode = None

        saved = getattr(self, "_balloonPickSavedPositions", [])
        print(f"[ManualBalloon] saved positions: {len(saved)}")
        if len(saved) < 2:
            self.manualBalloonStatusLabel.setText("⚠️ Need 2 points — try again")
            return

        p1, p2 = saved[0], saved[1]
        print(
            f"[ManualBalloon] p1={[round(x,1) for x in p1]}  p2={[round(x,1) for x in p2]}"
        )

        try:
            gi1 = self._snapToCenterline(p1)
            gi2 = self._snapToCenterline(p2)
        except Exception as _e:
            print(f"[ManualBalloon] snap error: {_e}")
            traceback.print_exc()
            self.manualBalloonStatusLabel.setText(f"⚠️ Snap failed: {_e}")
            return
        print(f"[ManualBalloon] snapped gi1={gi1}  gi2={gi2}")

        if gi1 == gi2:
            self.manualBalloonStatusLabel.setText("⚠️ Points too close — try again")
            return

        pts = self.logic.points
        gi_prox, gi_dist = (gi1, gi2) if pts[gi1][2] >= pts[gi2][2] else (gi2, gi1)
        print(f"[ManualBalloon] gi_prox={gi_prox} gi_dist={gi_dist}")

        try:
            path = self._traceCenterlinePath(gi_prox, gi_dist)
            print(f"[ManualBalloon] trace A: {len(path)} pts")
            if not path or len(path) < 2:
                path = self._traceCenterlinePath(gi_dist, gi_prox)
                print(f"[ManualBalloon] trace B: {len(path)} pts")
            if not path or len(path) < 2:
                path = list(range(min(gi_prox, gi_dist), max(gi_prox, gi_dist) + 1))
                print(f"[ManualBalloon] fallback range: {len(path)} pts")
        except Exception as _e:
            print(f"[ManualBalloon] trace error: {_e}")
            traceback.print_exc()
            self.manualBalloonStatusLabel.setText(f"⚠️ Trace failed: {_e}")
            return

        # ── Debug: print full path with coords, gap sizes, and logical branch ──
        # The "logical branch" label follows the proximal snap point's branch
        # until a bifurcation crossing occurs (real-index branch change), then
        # labels the distal segment as "B<prox>→B<dist>" to show it is the
        # anatomical continuation of the proximal branch into the distal one.
        import math as _dbm

        _interp_pts = getattr(self.logic, "_interpPoints", {})

        def _resolve_pt(gi):
            if gi < 0:
                return _interp_pts.get(gi, (0, 0, 0))
            return self.logic.points[gi]

        def _real_branch_of(gi):
            """Return label string for a branch index containing gi, or None."""
            if gi < 0:
                return None
            for _bi, (_bs, _be) in enumerate(self.logic.branches):
                if _bs <= gi < _be:
                    return str(_bi + 1)
            return None

        # Determine the proximal branch (branch of gi_prox)
        _prox_branch = _real_branch_of(gi_prox)
        _crossed_bif = False  # have we crossed the bifurcation yet?
        _distal_branch = None  # real branch of the distal segment

        print(f"[ManualBalloon] PATH DUMP ({len(path)} pts):")
        prev_coord = None
        big_gaps = []
        for _di, _gi in enumerate(path):
            _c = _resolve_pt(_gi)
            _real_br = _real_branch_of(_gi)

            # Detect bifurcation crossing: first time we see a real point
            # whose branch differs from the proximal branch
            if _real_br is not None and _real_br != _prox_branch and not _crossed_bif:
                _crossed_bif = True
                _distal_branch = _real_br

            # Build label
            if _gi < 0:
                # Synthetic Hermite bridge point — label as B<prox>→B<dist>
                if _distal_branch:
                    _label = f"{_prox_branch}→{_distal_branch}(bridge)"
                else:
                    _label = f"{_prox_branch}(bridge)"
            elif not _crossed_bif:
                # Still on the proximal branch
                _label = str(_prox_branch)
            else:
                # On the distal branch — label as B<prox>→B<dist> so it is
                # clear this segment belongs to the B1 lesion zone
                _label = f"{_prox_branch}→{_real_br}"

            gap_str = ""
            if prev_coord is not None:
                _gap = _dbm.sqrt(sum((_c[k] - prev_coord[k]) ** 2 for k in range(3)))
                gap_str = f"  gap={_gap:.1f}mm"
                if _gap > 3.0:
                    big_gaps.append((_di, _gi, _gap))
            print(
                f"  [{_di:3d}] gi={_gi:4d} B{_label}  "
                f"({_c[0]:6.1f},{_c[1]:6.1f},{_c[2]:7.1f}){gap_str}"
            )
            prev_coord = _c
        print(f"[ManualBalloon] Big gaps (>3mm): {len(big_gaps)}")
        for _di, _gi, _gap in big_gaps:
            _c = _resolve_pt(_gi)
            print(
                f"  idx={_di} gi={_gi} gap={_gap:.1f}mm coord=({_c[0]:.1f},{_c[1]:.1f},{_c[2]:.1f})"
            )

        balloon_diam = self.manualBalloonDiamSpin.value
        balloon_r = balloon_diam / 2.0
        name = f"BalloonDilate_manual_{gi_prox}"
        print(
            f"[ManualBalloon] balloon Ø{balloon_diam}mm  path={len(path)}pts  name={name}"
        )

        for nd in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if nd.GetName() == name:
                slicer.mrmlScene.RemoveNode(nd)

        try:
            self.logic._placeBalloon3D(path, balloon_r, name)
        except Exception as _e:
            print(f"[ManualBalloon] _placeBalloon3D error: {_e}")
            traceback.print_exc()
            self.manualBalloonStatusLabel.setText(f"⚠️ Balloon geometry failed: {_e}")
            return

        compressed_d = balloon_diam * 0.6  # safe default if stats block fails
        try:
            import statistics

            # Only real (non-synthetic) indices have diameter entries
            path_diams = [
                self.logic.diameters[gi]
                for gi in path
                if gi >= 0
                and gi < len(self.logic.diameters)
                and self.logic.diameters[gi] > 0
            ]
            print(
                f"[ManualBalloon] path_diams count={len(path_diams)} sample={path_diams[:5]}"
            )
            compressed_d = (
                statistics.median(path_diams) if path_diams else balloon_diam * 0.6
            )
            print(f"[ManualBalloon] compressed_d={compressed_d:.2f}mm")
            self.logic._expandMeshAtLesion(path, compressed_d / 2.0, balloon_r)
        except Exception as _e:
            print(f"[ManualBalloon] mesh expand error (non-fatal): {_e}")
            traceback.print_exc()

        try:
            if not hasattr(self.logic, "preDilationMap"):
                self.logic.preDilationMap = {}
            _entry = {
                "balloon_diam": balloon_diam,
                "healthy_diam": balloon_diam,
                "compressed_diam": compressed_d,
                "balloon_pts": path,
            }
            for gj in path:
                self.logic.preDilationMap[gj] = _entry
            print(f"[ManualBalloon] preDilationMap updated: {len(path)} keys")
        except Exception as _e:
            print(f"[ManualBalloon] preDilationMap error: {_e}")

        try:
            n_pts = len(path)
            _ipts = getattr(self.logic, "_interpPoints", {})

            def _rp(gi):
                return _ipts[gi] if gi < 0 else pts[gi]

            length_mm = sum(
                (
                    (_rp(path[i + 1])[0] - _rp(path[i])[0]) ** 2
                    + (_rp(path[i + 1])[1] - _rp(path[i])[1]) ** 2
                    + (_rp(path[i + 1])[2] - _rp(path[i])[2]) ** 2
                )
                ** 0.5
                for i in range(n_pts - 1)
            )
            self.manualBalloonStatusLabel.setText(
                f"✅ Manual balloon: Ø{balloon_diam:.1f}mm × {length_mm:.0f}mm "
                f"({n_pts} pts, gi {gi_prox}→{gi_dist})"
            )
            print(f"[ManualBalloon] DONE — Ø{balloon_diam}mm × {length_mm:.0f}mm")
        except Exception as _e:
            print(f"[ManualBalloon] summary error: {_e}")

    def onManualBalloonCancel(self):
        """Cancel manual balloon picking."""
        try:
            timer = getattr(self, "_balloonPickTimer", None)
            if timer:
                timer.stop()
        except:
            pass
        try:
            inode = slicer.app.applicationLogic().GetInteractionNode()
            inode.SetCurrentInteractionMode(inode.ViewTransform)
        except:
            pass
        node = getattr(self, "_balloonPickNode", None)
        if node and slicer.mrmlScene.IsNodePresent(node):
            slicer.mrmlScene.RemoveNode(node)
        self._balloonPickNode = None
        self.manualBalloonButton.setEnabled(True)
        self.manualBalloonCancelBtn.setEnabled(False)
        self.manualBalloonStatusLabel.setText("")

    def onStentPickPoints(self):
        """Enter markup placement mode — timer polls for 2 placed points."""
        if not self.logic.points:
            slicer.util.errorDisplay("Load a centerline first.")
            return
        self.onStentPickCancel()  # clean up any previous session
        self._stentPickNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "StentPickPoints"
        )
        dn = self._stentPickNode.GetDisplayNode()
        dn.SetSelectedColor(1, 0.3, 0)
        dn.SetTextScale(3.0)
        dn.SetGlyphScale(3.0)
        slicer.modules.markups.logic().StartPlaceMode(1)  # persistent
        self.stentPickButton.setEnabled(False)
        self.stentPickCancelButton.setEnabled(True)
        self.stentPickStatusLabel.setText(
            "📍 Click point 1 (proximal) on the vessel..."
        )
        self._stentPickMode = ""  # regular 2-point mode
        self._stentPickLastCount = 0
        self._stentPickTimer = qt.QTimer()
        self._stentPickTimer.setInterval(200)
        self._stentPickTimer.connect("timeout()", self._pollStentPickPoints)
        self._stentPickTimer.start()

    def _pollStentPickPoints(self):
        """Poll every 200ms — handles both 2-point and 3-point (Y-stent) modes."""
        is_y = getattr(self, "_stentPickMode", "") == "Y"
        is_kissing_mode = (
            getattr(self, "_stentPickMode", "") == "Y"
            and "Kissing" in self.stentTypeCombo.currentText
        )
        required = 2 if is_kissing_mode else (3 if is_y else 2)
        node = getattr(self, "_stentPickNode", None)
        if not node or not slicer.mrmlScene.IsNodePresent(node):
            self._stentPickTimer.stop()
            saved = getattr(self, "_stentPickSavedPositions", [])
            if len(saved) >= required + 1:
                finish_fn = self._finishStentPickY if is_y else self._finishStentPick
                qt.QTimer.singleShot(50, finish_fn)
            return
        n_total = node.GetNumberOfControlPoints()
        positions = []
        for i in range(n_total):
            p = [0, 0, 0]
            try:
                node.GetNthControlPointPosition(i, p)
                positions.append(list(p))
            except:
                pass
        if positions:
            self._stentPickSavedPositions = positions[:]
        n_clicked = max(0, n_total - 1)
        # print(f"[StentPick] Poll: n_total={n_total} n_clicked={n_clicked} last={self._stentPickLastCount} required={required} is_y={is_y}")
        if n_clicked == self._stentPickLastCount:
            return
        self._stentPickLastCount = n_clicked
        if is_kissing_mode:
            msgs = {1: "✅ LEFT Iliac set.  📍 Click point 2: DEEP in RIGHT Iliac"}
        elif is_y:
            msgs = {
                1: "✅ Point 1 set.  📍 Click point 2: DEEP in LEFT Iliac — at or past the lesion",
                2: "✅ Points 1+2 set.  📍 Click point 3: DEEP in RIGHT Iliac — matching depth",
            }
        else:
            msgs = {1: "📍 Click point 2 (distal) on the vessel..."}
        if n_clicked in msgs:
            self.stentPickStatusLabel.setText(msgs[n_clicked])
        if n_clicked >= required:
            self._stentPickTimer.stop()
            finish_fn = self._finishStentPickY if is_y else self._finishStentPick
            qt.QTimer.singleShot(150, finish_fn)

    def _finishStentPick(self):
        """Trace path and place stent using saved pick positions."""
        try:
            inode = slicer.app.applicationLogic().GetInteractionNode()
            inode.SetCurrentInteractionMode(inode.ViewTransform)
        except Exception as e:
            pass
        self.stentPickButton.setEnabled(True)
        self.stentPickCancelButton.setEnabled(False)
        if getattr(self, "_stentPickNode", None) and slicer.mrmlScene.IsNodePresent(
            self._stentPickNode
        ):
            slicer.mrmlScene.RemoveNode(self._stentPickNode)
        self._stentPickNode = None
        saved = getattr(self, "_stentPickSavedPositions", [])
        if len(saved) < 2:
            self.stentPickStatusLabel.setText("⚠️ Could not read positions — try again")
            return
        # The saved list has n_total points including the preview — take first 2
        # but we need the 2 REAL clicked points, not the preview
        # With n_total=3: positions[0]=pt1, positions[1]=pt2, positions[2]=preview
        p1, p2 = saved[0], saved[1]
        try:
            gi1 = self._snapToCenterline(p1)
            gi2 = self._snapToCenterline(p2)
            if gi1 == gi2:
                self.stentPickStatusLabel.setText("⚠️ Points too close — try again")
                return
            # Sort so gi_prox has higher Z (superior/proximal) → gi_dist lower Z
            pts = self.logic.points
            z1 = pts[gi1][2] if gi1 < len(pts) else 0
            z2 = pts[gi2][2] if gi2 < len(pts) else 0
            gi_prox, gi_dist = (gi1, gi2) if z1 >= z2 else (gi2, gi1)
            path = self._traceCenterlinePath(gi_prox, gi_dist)
            if not path or len(path) < 2:
                path = self._traceCenterlinePath(gi_dist, gi_prox)
            gi1, gi2 = gi_prox, gi_dist
            if not path or len(path) < 2:
                self.stentPickStatusLabel.setText("⚠️ Could not trace path — try again")
                return
            length = sum(
                (
                    (self.logic.points[path[i + 1]][0] - self.logic.points[path[i]][0])
                    ** 2
                    + (
                        self.logic.points[path[i + 1]][1]
                        - self.logic.points[path[i]][1]
                    )
                    ** 2
                    + (
                        self.logic.points[path[i + 1]][2]
                        - self.logic.points[path[i]][2]
                    )
                    ** 2
                )
                ** 0.5
                for i in range(len(path) - 1)
            )
            diams = self.logic.diameters
            vd1 = diams[gi1] if gi1 < len(diams) else 10.0
            vd2 = diams[gi2] if gi2 < len(diams) else 10.0
            pd = round(max(vd1, vd2) * 1.15 / 0.5) * 0.5
            branches_crossed = set()
            for gi in path:
                for bi, (bs, be) in enumerate(self.logic.branches):
                    if bs <= gi < be:
                        branches_crossed.add(bi + 1)
                        break
            branch_str = "+".join(f"Br{b}" for b in sorted(branches_crossed))
            self.stentPickStatusLabel.setText(
                f"✅ {branch_str} | {length:.1f}mm | Ø{vd1:.1f}→{vd2:.1f}mm"
            )
            # Count existing stent nodes to generate a unique name
            existing_stents = [
                n
                for n in slicer.util.getNodesByClass("vtkMRMLModelNode")
                if n.GetName().startswith("StentModel")
            ]
            stent_idx = len(existing_stents)
            plan = {
                "type": self.stentTypeCombo.currentText,
                "proxPt": path[0],
                "distPt": path[-1],
                "proxDiam": pd,
                "distDiam": pd,
                "length": round(length, 1),
                "vd_prox": round(vd1, 2),
                "vd_dist": round(vd2, 2),
                "warnings": [],
                "branchIdx": -1,
                "trunk_mm": 0.0,
                "custom_path": path,
                "_additional": stent_idx > 0,  # don't remove previous stents
                "_stent_idx": stent_idx,
            }
            self.logic.stentPlan = plan
            self.logic.placeStent3D(plan)
            self._removeBalloons()
        except Exception as e:
            import traceback

            print(f"[StentPick] ERROR: {e}")
            traceback.print_exc()
            self.stentPickStatusLabel.setText(f"⚠️ Error: {e}")

    def onStentPickCancel(self):
        """Cancel active pick session."""
        if getattr(self, "_stentPickTimer", None):
            try:
                self._stentPickTimer.stop()
            except:
                pass
            self._stentPickTimer = None
        if getattr(self, "_stentPickNode", None) and slicer.mrmlScene.IsNodePresent(
            self._stentPickNode
        ):
            slicer.mrmlScene.RemoveNode(self._stentPickNode)
        self._stentPickNode = None
        try:
            inode = slicer.app.applicationLogic().GetInteractionNode()
            inode.SetCurrentInteractionMode(inode.ViewTransform)
        except:
            pass
        self.stentPickButton.setEnabled(True)
        self.stentPickYButton.setEnabled(True)
        self.manualBalloonButton.setEnabled(True)
        self.stentPickCancelButton.setEnabled(False)
        self.stentPickStatusLabel.setText("Cancelled — click to start again")
        self.manualBalloonButton.setEnabled(True)

    def onStentPickY(self):
        # Always default to Kissing (Parallel) — user can change after placement
        current_type = self.stentTypeCombo.currentText.strip()
        is_separator = current_type.startswith("──") or not current_type
        if (
            is_separator
            or "Bifurcated" in current_type
            or "Straight" in current_type
            or "Kissing" not in current_type
        ):
            for ci in range(self.stentTypeCombo.count):
                if "Kissing" in self.stentTypeCombo.itemText(ci):
                    self.stentTypeCombo.blockSignals(True)
                    self.stentTypeCombo.setCurrentIndex(ci)
                    self.stentTypeCombo.blockSignals(False)
                    break
        """Pick 3 points for Y/trouser bifurcation stent — reuses same poll as 2-pt pick."""
        if not self.logic.points:
            slicer.util.errorDisplay("Load a centerline first.")
            return
        # Warn if stent type is incompatible with bifurcated placement
        stype = self.stentTypeCombo.currentText.strip()
        single_vessel_types = [
            "Straight",
            "Tapered",
            "Covered · Straight (PTFE/ePTFE)",
            "Covered · Tapered (PTFE/ePTFE)",
            "DES · Sirolimus-Eluting",
            "DES · Paclitaxel-Eluting",
            "DES · Zotarolimus-Eluting",
        ]
        if stype in single_vessel_types:
            self.stentPickStatusLabel.setText(
                f"⚠️ Note: '{stype}' is a single-vessel type. "
                f"Placing Y-stent regardless — consider switching to "
                f"'Bifurcated (Y / Trouser)' or 'Covered · Bifurcated (EVAR Graft)'."
            )
            self.stentPickStatusLabel.setStyleSheet(
                "font-size: 11px; color: #c0392b; font-style: italic;"
            )
        self.onStentPickCancel()
        self._stentPickNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "StentPickPoints"
        )
        dn = self._stentPickNode.GetDisplayNode()
        dn.SetSelectedColor(0.1, 0.6, 0.5)
        dn.SetTextScale(3.0)
        dn.SetGlyphScale(3.0)
        slicer.modules.markups.logic().StartPlaceMode(1)
        self.stentPickButton.setEnabled(False)
        self.stentPickYButton.setEnabled(False)
        self.stentPickCancelButton.setEnabled(True)
        stype_start = self.stentTypeCombo.currentText.strip()
        if "Kissing" in stype_start:
            self.stentPickStatusLabel.setText(
                "💋 KISSING STENT — Click 2 points only:\n"
                "  📍 1: DEEP in LEFT Iliac (at or past the lesion)\n"
                "  📍 2: DEEP in RIGHT Iliac\n"
                "  (Proximal IVC landing is set automatically from Trunk coverage)"
            )
        else:
            self.stentPickStatusLabel.setText(
                "🔱 Y-STENT — Click 3 points:\n"
                "  📍 1: IVC trunk above bifurcation\n"
                "  📍 2: Deep in LEFT Iliac\n"
                "  📍 3: Deep in RIGHT Iliac"
            )
        self._stentPickMode = "Y"  # flag checked in shared poll
        self._stentPickLastCount = 0
        self._stentPickSavedPositions = []
        self._stentPickTimer = qt.QTimer()
        self._stentPickTimer.setInterval(200)
        self._stentPickTimer.connect("timeout()", self._pollStentPickPoints)
        self._stentPickTimer.start()

    def _finishStentPickY(self):
        """Build and place Y/trouser bifurcation stent from 3 picked points."""
        # Stop timer and reset mode immediately
        if getattr(self, "_stentPickTimer", None):
            try:
                self._stentPickTimer.stop()
            except:
                pass
            self._stentPickTimer = None
        self._stentPickMode = ""
        try:
            slicer.app.applicationLogic().GetInteractionNode().SetCurrentInteractionMode(
                slicer.app.applicationLogic().GetInteractionNode().ViewTransform
            )
        except:
            pass
        self.stentPickButton.setEnabled(True)
        self.stentPickYButton.setEnabled(True)
        self.manualBalloonButton.setEnabled(True)
        self.stentPickCancelButton.setEnabled(False)
        if getattr(self, "_stentPickNode", None) and slicer.mrmlScene.IsNodePresent(
            self._stentPickNode
        ):
            slicer.mrmlScene.RemoveNode(self._stentPickNode)
        self._stentPickNode = None
        saved = getattr(self, "_stentPickSavedPositions", [])
        stype_finish = self.stentTypeCombo.currentText.strip()
        is_kissing_finish = "Kissing" in stype_finish

        # Kissing stent: only 2 clicks needed (LEFT Iliac + RIGHT Iliac)
        # Proximal is set automatically from trunk
        if is_kissing_finish:
            if len(saved) < 2:
                self.stentPickStatusLabel.setText("⚠️ Need 2 points — try again")
                return
            p_left, p_right = saved[0], saved[1]
            gi_left = self._snapToCenterline(p_left)
            gi_right = self._snapToCenterline(p_right)
            branches = self.logic.branches

            def get_branch(gi):
                for bi, (bs, be) in enumerate(branches):
                    if bs <= gi < be:
                        return bi
                return 0

            bi_left = get_branch(gi_left)
            bi_right = get_branch(gi_right)
            # Ensure left/right are in different downstream branches
            if bi_left == bi_right:
                # Both in same branch — swap second to sibling
                for _bj, (_bjs, _bje) in enumerate(branches):
                    if _bj > 0 and _bj != bi_left:
                        bi_right = _bj
                        gi_right = _bjs + 20
                        break
            # Proximal = hub gi for role assignment (actual landing set by kissing code)
            gi_prox = branches[0][1] - 1
            bi_prox = 0
            print(
                f"[StentPickY] Kissing 2-click: left=gi{gi_left}(bi{bi_left}) "
                f"right=gi{gi_right}(bi{bi_right}) prox=hub(gi{gi_prox})"
            )
        else:
            # Y-stent: 3 clicks needed
            if len(saved) < 3:
                self.stentPickStatusLabel.setText("⚠️ Need 3 points — try again")
                return
            p_prox, p_left, p_right = saved[0], saved[1], saved[2]
            gi_prox = self._snapToCenterline(p_prox)
            gi_left = self._snapToCenterline(p_left)
            gi_right = self._snapToCenterline(p_right)
            branches = self.logic.branches

            def get_branch(gi):
                for bi, (bs, be) in enumerate(branches):
                    if bs <= gi < be:
                        return bi
                return 0

            bi_prox = get_branch(gi_prox)
            bi_left = get_branch(gi_left)
            bi_right = get_branch(gi_right)

        # Auto-reassign roles so that:
        picks = [(gi_prox, bi_prox), (gi_left, bi_left), (gi_right, bi_right)]
        trunk_picks = [(gi, bi) for gi, bi in picks if bi == 0]
        downstream_picks = [(gi, bi) for gi, bi in picks if bi != 0]
        print(
            f"[StentPickY] Role reassign: trunk_picks={len(trunk_picks)} downstream={len(downstream_picks)}"
        )
        print(f"[StentPickY] bi_prox={bi_prox} bi_left={bi_left} bi_right={bi_right}")
        if len(trunk_picks) == 1 and len(downstream_picks) == 2:
            gi_prox = trunk_picks[0][0]
            bi_prox = 0
            gi_left = downstream_picks[0][0]
            bi_left = downstream_picks[0][1]
            gi_right = downstream_picks[1][0]
            bi_right = downstream_picks[1][1]
        elif len(trunk_picks) == 0:
            # No trunk pick — all 3 clicks are in downstream branches.
            # Strategy: find the two DISTINCT branch indices among the picks.
            # The pick in the branch that appears TWICE (or nearest the hub) becomes
            # prox (it is in the target branch, upstream within that branch).
            # The two distinct downstream picks become left/right.
            hub_gi_tmp = branches[0][1] - 1
            # Collect unique downstream branch indices
            branch_ids = [bi for _, bi in picks]
            unique_bis = list(dict.fromkeys(branch_ids))  # preserves order, deduped
            if len(unique_bis) >= 2:
                # We have picks in at least 2 different branches.
                branch_near = {}
                branch_far = {}
                for gi, bi in picks:
                    if bi not in branch_near or gi < branch_near[bi][0]:
                        branch_near[bi] = (gi, bi)
                    if bi not in branch_far or gi > branch_far[bi][0]:
                        branch_far[bi] = (gi, bi)
                bi_A, bi_B = unique_bis[0], unique_bis[1]
                gi_left = branch_far[bi_A][0]
                bi_left = bi_A
                gi_right = branch_far[bi_B][0]
                bi_right = bi_B
                # For kissing: snap gi_prox to trunk hub automatically
                # Use the point nearest to hub across both branches as depth hint,
                # but override gi_prox to be in the trunk (hub point).
                near_A = branch_near[bi_A][0]
                near_B = branch_near[bi_B][0]
                if near_A <= near_B:
                    gi_prox = near_A
                    bi_prox = bi_A
                else:
                    gi_prox = near_B
                    bi_prox = bi_B
                print(
                    f"[StentPickY] 0-trunk: bi_A={bi_A} far=gi{gi_left}  "
                    f"bi_B={bi_B} far=gi{gi_right}  prox_hint=gi{gi_prox}"
                )
            else:
                # All 3 clicks in the same branch — take the nearest to hub as prox
                closest = min(picks, key=lambda x: abs(x[0] - hub_gi_tmp))
                others = [p for p in picks if p[0] != closest[0]]
                gi_prox = closest[0]
                bi_prox = closest[1]
                gi_left = others[0][0]
                bi_left = others[0][1]
                gi_right = others[1][0] if len(others) > 1 else others[0][0]
                bi_right = others[1][1] if len(others) > 1 else others[0][1]
        elif len(trunk_picks) == 2 and len(downstream_picks) == 1:
            # Two trunk clicks + one downstream — reassign the downstream pick as the
            # true prox (use the one nearest hub), and the other trunk pick also stays.
            hub_gi_tmp = branches[0][1] - 1
            closest_trunk = min(trunk_picks, key=lambda x: abs(x[0] - hub_gi_tmp))
            other_trunk = [p for p in trunk_picks if p[0] != closest_trunk[0]]
            gi_prox = closest_trunk[0]
            bi_prox = 0
            gi_left = downstream_picks[0][0]
            bi_left = downstream_picks[0][1]
            gi_right = other_trunk[0][0] if other_trunk else downstream_picks[0][0]
            bi_right = other_trunk[0][1] if other_trunk else downstream_picks[0][1]
        # SAFETY: bi_left and bi_right must never be 0 (trunk) — kissing renderer
        # requires a valid downstream branch index for sibling detection.
        # If still 0 after reassignment, fall back to the first non-trunk branch.
        if bi_left == 0:
            for _bj, (_bjs, _bje) in enumerate(branches):
                if _bj > 0:
                    bi_left = _bj
                    gi_left = _bjs + 20
                    break
            print(f"[StentPickY] WARNING: bi_left was 0, forced to bi={bi_left}")
        if bi_right == 0 or bi_right == bi_left:
            for _bj, (_bjs, _bje) in enumerate(branches):
                if _bj > 0 and _bj != bi_left:
                    bi_right = _bj
                    gi_right = _bjs + 20
                    break
            print(f"[StentPickY] WARNING: bi_right was 0/same, forced to bi={bi_right}")
        # SAFETY: gi_left must not equal gi_prox (would make left limb zero-length).
        # If they match, advance gi_left at least 20 pts into its branch.
        if gi_left == gi_prox and bi_left > 0:
            bs_left_safe, be_left_safe = branches[bi_left]
            gi_left = min(bs_left_safe + 20, be_left_safe - 1)
            print(
                f"[StentPickY] WARNING: gi_left==gi_prox, advanced gi_left to {gi_left}"
            )
        # SAFETY: gi_right must not equal gi_prox either.
        if gi_right == gi_prox and bi_right > 0:
            bs_right_safe, be_right_safe = branches[bi_right]
            gi_right = min(bs_right_safe + 20, be_right_safe - 1)
            print(
                f"[StentPickY] WARNING: gi_right==gi_prox, advanced gi_right to {gi_right}"
            )
        print(
            f"[StentPickY] Final roles: prox=gi{gi_prox}(bi{bi_prox}) "
            f"left=gi{gi_left}(bi{bi_left}) right=gi{gi_right}(bi{bi_right})"
        )

        # Find hub = branch junction point where the 3 paths meet.
        path_AB = self._traceCenterlinePath(gi_prox, gi_left)
        path_AC = self._traceCenterlinePath(gi_prox, gi_right)
        path_BC = self._traceCenterlinePath(gi_left, gi_right)
        if not path_AB or not path_AC or not path_BC:
            self.stentPickStatusLabel.setText("⚠️ Could not trace paths — try again")
            return

        pts = self.logic.points

        def dist3d(a, b):
            return sum((pts[a][k] - pts[b][k]) ** 2 for k in range(3)) ** 0.5

        # Build list of branch junction points (the seams between branches)
        junction_pts = set()
        for bi, (bs, be) in enumerate(branches):
            if bs > 0:
                junction_pts.add(bs)
                junction_pts.add(bs - 1)

        set_AB = set(path_AB)
        set_AC = set(path_AC)
        set_BC = set(path_BC)
        excluded = {gi_prox, gi_left, gi_right}

        # Find hub: the trunk end point where all downstream branches originate.
        # Always use trunk end (branches[0][1]-1) as the hub — it is the true
        # bifurcation point regardless of which branches the user clicked in.
        bif_pt = branches[0][1] - 1

        # Build 3 limbs from hub outward to each picked point
        # Helper: cap a limb at the bifurcation boundary if it passes through Branch 0 (trunk)
        # Trunk = branch 0; bif_boundary = last trunk point before bifurcation
        trunk_bs, trunk_be = branches[0]
        bif_boundary = trunk_be - 1

        def cap_at_trunk(path):
            """Cap path at bif_boundary if it travels backward through trunk.
            Paths that START at the trunk end (bif_boundary) pass through unchanged."""
            if not path:
                return path
            for i, pt in enumerate(path):
                if pt < 0:
                    continue  # synthetic bridge point — skip
                if trunk_bs <= pt < trunk_be:
                    # If this is the first real point and it's the hub, let it pass
                    if i == 0 and pt == bif_boundary:
                        continue
                    # Entered trunk mid-path — cap here
                    cap_idx = min(i + 1, len(path))
                    for j in range(i, len(path)):
                        if path[j] == bif_boundary:
                            return path[: j + 1]
                    return path[:cap_idx]
            return path

        def branch_of(gi):
            for bi, (bs, be) in enumerate(branches):
                if bs <= gi < be:
                    return bi
            return 0

        # Build limbs.
        # trunk_path: gi_prox → bif_pt (shared stem approaching bifurcation)
        # left_limb:  bif_pt → gi_left  (branch A outward)
        # right_limb: bif_pt → gi_right (branch B outward)
        trunk_bs_val, trunk_be_val = branches[0]

        def _seg_len_simple(seg, all_pts):
            import math as _m

            total = 0.0
            for i in range(len(seg) - 1):
                a, b = seg[i], seg[i + 1]
                if 0 <= a < len(all_pts) and 0 <= b < len(all_pts):
                    total += _m.sqrt(
                        sum((all_pts[b][k] - all_pts[a][k]) ** 2 for k in range(3))
                    )
            return total

        # Build trunk_path: always walk back from hub INTO Branch 0 (trunk).
        # gi_prox may be in a downstream branch (user clicked past bifurcation) —
        # in that case ignore it entirely and build trunk from the hub backward.
        # Also handles when gi_prox > bif_pt (downstream branch has higher indices).
        _renal_z2 = min(
            pts[branches[bi_rv][0]][2] for bi_rv in [3, 4] if bi_rv < len(branches)
        )
        if trunk_bs_val <= gi_prox < trunk_be_val:
            # gi_prox is genuinely in the trunk — walk forward from it to hub
            trunk_path = list(range(gi_prox, bif_pt + 1))
            print(
                f"[StentPickY] gi_prox={gi_prox} is in trunk → trunk_path range({gi_prox},{bif_pt+1})"
            )
        else:
            # gi_prox is outside trunk (downstream branch or before trunk start) —
            # build trunk by walking BACKWARD from hub up to MIN_TRUNK_MM
            trunk_path = [bif_pt]
            acc_tp = 0.0
            for _ti in range(bif_pt - 1, trunk_bs_val - 1, -1):
                if pts[_ti][2] >= _renal_z2:
                    break  # stop before renal vein level
                if _ti + 1 < len(pts):
                    _p0, _p1 = pts[_ti], pts[_ti + 1]
                    acc_tp += (
                        (_p1[0] - _p0[0]) ** 2
                        + (_p1[1] - _p0[1]) ** 2
                        + (_p1[2] - _p0[2]) ** 2
                    ) ** 0.5
                trunk_path.insert(0, _ti)
                if acc_tp >= 20.0:
                    break
            print(
                f"[StentPickY] gi_prox={gi_prox} outside trunk → walked back from hub={bif_pt}, "
                f"trunk={trunk_path[0]}..{trunk_path[-1]} ({acc_tp:.1f}mm)"
            )

        # ── Enforce minimum trunk length (25mm) ──────────────────────────
        MIN_TRUNK_MM = 20.0
        MAX_TRUNK_MM = 20.0  # clinical cap: infrarenal IVC only, avoid renal veins
        trunk_mm_now = _seg_len_simple(trunk_path, pts)
        if trunk_mm_now < MIN_TRUNK_MM:
            import math as _m2

            # Hard proximal limit: never go further from hub than gi_prox
            # (user picked that point intentionally) and never past trunk_bs_val.
            # Also cap total trunk length at MAX_TRUNK_MM so we don't reach
            # the renal vessels / suprarenal aorta.
            prox_limit = max(
                trunk_bs_val,
                gi_prox if trunk_bs_val <= gi_prox <= bif_pt else trunk_bs_val,
            )
            _renal_z3 = min(
                pts[branches[bi_rv][0]][2] for bi_rv in [3, 4] if bi_rv < len(branches)
            )
            extended_prox = bif_pt  # start at hub, walk backward
            acc = 0.0
            for ti in range(bif_pt - 1, prox_limit - 1, -1):
                if pts[ti][2] >= _renal_z3:
                    break  # stop before renal vein level
                if ti + 1 < len(pts):
                    p0, p1 = pts[ti], pts[ti + 1]
                    acc += _m2.sqrt(sum((p1[k] - p0[k]) ** 2 for k in range(3)))
                extended_prox = ti
                if acc >= MIN_TRUNK_MM or acc >= MAX_TRUNK_MM:
                    break
            trunk_path = list(range(extended_prox, bif_pt + 1))
            print(
                f"[StentPickY] Trunk extended to {_seg_len_simple(trunk_path, pts):.0f}mm "
                f"(was {trunk_mm_now:.0f}mm, min={MIN_TRUNK_MM}mm)"
            )

        def make_limb(hub, target):
            """Build limb from hub directly into target's branch — no path tracing needed.
            Hub is bif_pt (trunk end), branch starts right after it sequentially."""
            # Find which branch target belongs to
            bi_tgt = get_branch(target)
            if bi_tgt <= 0:
                # target is in trunk — fallback to trace
                return cap_at_trunk(self._traceCenterlinePath(hub, target))
            bs_tgt, be_tgt = branches[bi_tgt]
            # Build: hub → branch_start → ... → target (all real sequential points)
            limb = [hub] + list(range(bs_tgt, min(target + 1, be_tgt)))
            return limb

        left_limb = make_limb(bif_pt, gi_left)
        right_limb = make_limb(bif_pt, gi_right)

        print(
            f"[StentPickY] trunk: {len(trunk_path)}pts gi_prox={gi_prox}→hub={bif_pt}  "
            f"first={trunk_path[:3]}  last={trunk_path[-3:]}"
        )
        if not left_limb or not right_limb:
            self.stentPickStatusLabel.setText(
                "⚠️ Could not build limbs from hub — try again"
            )
            return
        if len(left_limb) < 2 or len(right_limb) < 2:
            self.stentPickStatusLabel.setText(
                "⚠️ Hub too close to a picked point — pick points further apart"
            )
            return

        pts = self.logic.points
        _interp = getattr(self.logic, "_interpPoints", {})

        def _resolve_pt(gi):
            if gi < 0:
                return _interp.get(gi, (0, 0, 0))
            return pts[gi] if gi < len(pts) else (0, 0, 0)

        def fmt_pt(gi):
            p = _resolve_pt(gi)
            return f"pt{gi}=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})"

        diams = self.logic.diameters

        def avg_diam(path):
            vals = [
                diams[gi]
                for gi in path
                if gi >= 0 and gi < len(diams) and diams[gi] > 0
            ]
            return sum(vals) / len(vals) if vals else 10.0

        # Trunk diameter: use ONLY last 10pts of Branch 1 before the hub
        # (the true pre-bifurcation zone — not the wide proximal aortic section)
        last10_trunk = list(range(max(trunk_bs_val, bif_pt - 10), bif_pt + 1))
        vd_trunk = (
            avg_diam(last10_trunk)
            if last10_trunk
            else (diams[bif_pt] if bif_pt < len(diams) else 10.0)
        )
        vd_left = avg_diam(left_limb)
        vd_right = avg_diam(right_limb)

        import math as _ksm

        r_left = round(vd_left * 1.15 / 0.5) * 0.5
        r_right = round(vd_right * 1.15 / 0.5) * 0.5
        # Trunk stent: match vessel size (vessel × 1.10), NOT geometric kissing rule
        # Geometric rule oversizes trunk; clinical practice sizes to the actual vessel
        r_trunk = round(vd_trunk * 1.10 / 0.5) * 0.5
        print(
            f"[StentPickY] diams: left={vd_left:.1f}mm right={vd_right:.1f}mm trunk={vd_trunk:.1f}mm"
        )
        print(
            f"[StentPickY] sizing: left={r_left:.1f}mm  right={r_right:.1f}mm  trunk={r_trunk:.1f}mm (vessel×1.10)"
        )

        def _seg_len(seg):
            total = 0.0
            for i in range(len(seg) - 1):
                p0 = _resolve_pt(seg[i])
                p1 = _resolve_pt(seg[i + 1])
                if p0 and p1:
                    total += _ksm.sqrt(sum((p1[k] - p0[k]) ** 2 for k in range(3)))
            return total

        total_len = _seg_len(trunk_path) + max(
            _seg_len(left_limb), _seg_len(right_limb)
        )
        plan = {
            "type": "Bifurcated",
            "proxPt": gi_prox,
            "distPt": gi_left,
            "proxDiam": r_trunk,
            "distDiam": r_left,
            "length": round(total_len, 1),
            "vd_prox": round(vd_trunk, 2),
            "vd_dist": round(vd_right, 2),
            "warnings": [],
            "branchIdx": -1,
            "trunk_mm": 0.0,
            "custom_path": None,
            # Suppress single-point stubs only (need ≥2 pts for a tube segment)
            "y_trunk": trunk_path if len(trunk_path) >= 2 else [],
            "y_left": left_limb if len(left_limb) >= 2 else [],
            "y_right": right_limb if len(right_limb) >= 2 else [],
            "r_trunk": r_trunk / 2.0,
            "r_left": r_left / 2.0,
            "r_right": r_right / 2.0,
        }
        self.logic.stentPlan = plan
        # Route to kissing renderer ONLY if user explicitly selected kissing type
        stype_now = self.stentTypeCombo.currentText.strip()
        is_kissing = "Kissing" in stype_now

        if is_kissing:
            import math as _ikm

            pts_ik = self.logic.points
            branches_ik = self.logic.branches
            hub_gi_ik = branches_ik[0][1] - 1
            trunk_bs_ik = branches_ik[0][0]

            VBX_MAX_MM = 79.0
            DISTAL_MARGIN = 10.0  # mm past lesion centre for distal landing zone
            TRUNK_MM = 20.0  # mm of IVC coverage above hub

            def _walk_mm_forward(bs, be, target_mm):
                """Walk forward from branch start by target_mm, return gi."""
                acc = 0.0
                gi = bs
                for i in range(bs, be - 1):
                    if i + 1 < len(pts_ik):
                        acc += _ikm.sqrt(
                            sum(
                                (pts_ik[i + 1][k] - pts_ik[i][k]) ** 2 for k in range(3)
                            )
                        )
                    gi = i + 1
                    if acc >= target_mm:
                        break
                return min(gi, be - 1)

            def _mm_from_bs(bs, gi_end):
                """Distance from branch start to gi_end."""
                return sum(
                    _ikm.sqrt(
                        sum((pts_ik[i + 1][k] - pts_ik[i][k]) ** 2 for k in range(3))
                    )
                    for i in range(bs, min(gi_end, len(pts_ik) - 1))
                    if i + 1 < len(pts_ik)
                )

            # ── Find lesion on Branch A ───────────────────────────────────────
            findings = getattr(self.logic, "findings", [])
            lesion_gi_A = None
            for _f in findings:
                if _f.get("branchIdx", -1) == bi_left and _f.get("type", "") in (
                    "Mild Compression",
                    "Pancaking",
                ):
                    lesion_gi_A = _f.get("pointIdx", None)
                    break

            bs_A_ik = branches_ik[bi_left][0]
            be_A_ik = branches_ik[bi_left][1]
            bs_B_ik = branches_ik[bi_right][0]
            be_B_ik = branches_ik[bi_right][1]

            # ── Proximal landing ──────────────────────────────────────────────
            # In 2-click mode: gi_prox was auto-set to hub — always walk back by
            # trunk coverage spinbox value (or TRUNK_MM default).
            # In 3-click mode: honour the user's trunk click if it's in the trunk.
            stype_kiss = self.stentTypeCombo.currentText.strip()
            _two_click = "Kissing" in stype_kiss
            _trunk_target = (
                self.kissProxExtSpin.value
                if hasattr(self, "kissProxExtSpin")
                else TRUNK_MM
            )

            if not _two_click and trunk_bs_ik <= gi_prox < hub_gi_ik:
                # 3-click: user clicked in trunk — use that point directly
                gi_prox_kiss = gi_prox
                acc_prox = sum(
                    _ikm.sqrt(
                        sum((pts_ik[i + 1][k] - pts_ik[i][k]) ** 2 for k in range(3))
                    )
                    for i in range(gi_prox_kiss, min(hub_gi_ik, len(pts_ik) - 1))
                    if i + 1 < len(pts_ik)
                )
                print(
                    f"[StentPickY] gi_prox={gi_prox} in trunk → using user click directly"
                )
            else:
                # 2-click or click missed trunk: walk back trunk_target mm from hub
                acc_prox = 0.0
                gi_prox_kiss = hub_gi_ik
                for _ti in range(hub_gi_ik - 1, trunk_bs_ik - 1, -1):
                    if _ti + 1 < len(pts_ik):
                        _p0, _p1 = pts_ik[_ti], pts_ik[_ti + 1]
                        acc_prox += _ikm.sqrt(
                            sum((_p1[k] - _p0[k]) ** 2 for k in range(3))
                        )
                    gi_prox_kiss = _ti
                    if acc_prox >= _trunk_target:
                        break

            # ── Distal landing: use user's actual clicked points ──────────────
            # The operator clicks define where coverage is needed.
            # We honour their picks exactly and warn if beyond VBX 79mm limit.
            kiss_dist = min(gi_left, branches_ik[bi_left][1] - 1)
            kiss_dist_B = min(gi_right, branches_ik[bi_right][1] - 1)

            # Measure actual limb lengths for warning only
            _lmm = _mm_from_bs(bs_A_ik, kiss_dist)
            _rmm = _mm_from_bs(bs_B_ik, kiss_dist_B)
            print(
                f"[KissLesion] Branch A: {_lmm:.0f}mm from hub "
                f"({'within' if _lmm <= VBX_MAX_MM else 'EXCEEDS'} 79mm VBX)"
            )
            print(
                f"[KissLesion] Branch B: {_rmm:.0f}mm from hub "
                f"({'within' if _rmm <= VBX_MAX_MM else 'EXCEEDS'} 79mm VBX)"
            )
            print(
                f"[StentPickY] proximal landing: gi{gi_prox_kiss} "
                f"({acc_prox:.1f}mm above hub gi{hub_gi_ik})"
            )

            print(
                f"[StentPickY] kiss_plan: prox=gi{gi_prox_kiss} "
                f"distA=gi{kiss_dist}(bi{bi_left}) distB=gi{kiss_dist_B}(bi{bi_right})"
            )
            print(
                f"[KissDebug] user clicks: gi_prox={gi_prox}(bi{bi_prox}) "
                f"gi_left={gi_left}(bi{bi_left}) gi_right={gi_right}(bi{bi_right})"
            )
            print(
                f"[KissDebug] trunk: hub=gi{hub_gi_ik} bs=gi{trunk_bs_ik} "
                f"prox_final=gi{gi_prox_kiss} ({acc_prox:.1f}mm above hub)"
            )
            print(
                f"[KissDebug] distal: A=gi{kiss_dist}(bi{bi_left}) "
                f"B=gi{kiss_dist_B}(bi{bi_right})"
            )
            kiss_plan = {
                "type": "Kissing",
                "branchIdx": bi_left,
                "siblingIdx": bi_right,
                "proxPt": gi_prox_kiss,
                "distPt": kiss_dist,
                "distPt_B": kiss_dist_B,
                # Use VBX nominal from spinbox (clamped 5–11), NOT raw vessel diameter
                "proxDiam": float(min(11, max(5, round(self.stentProxDiamSpin.value)))),
                "distDiam": float(min(11, max(5, round(self.stentDistDiamSpin.value)))),
                "warnings": [],
                "trunk_mm": 0.0,
            }
            self.logic.stentPlan = kiss_plan
            placed = self.logic._placeKissingStent3D(kiss_plan)
            if placed:
                self.stentPickStatusLabel.setText(
                    f"💋 Kissing stent | Br{bi_left}+sibling | "
                    f"Ø{r_trunk:.1f}/Ø{r_left:.1f}mm hub=pt{bif_pt}"
                )

                # ── Populate kissing spinboxes from actual placed geometry ──
                import math as _math

                pts_k = self.logic.points
                hub_gi_k = self.logic.branches[0][1] - 1
                trunk_mm_k = sum(
                    _math.sqrt(
                        sum((pts_k[i + 1][k] - pts_k[i][k]) ** 2 for k in range(3))
                    )
                    for i in range(gi_prox_kiss, min(hub_gi_k, len(pts_k) - 1))
                    if i + 1 < len(pts_k)
                )
                # Branch depth: measure to user's actual clicked point (gi_left/gi_right)
                # not to kiss_dist which may have been capped earlier
                branch_A_mm_k = sum(
                    _math.sqrt(
                        sum((pts_k[i + 1][k] - pts_k[i][k]) ** 2 for k in range(3))
                    )
                    for i in range(
                        self.logic.branches[bi_left][0], min(gi_left, len(pts_k) - 1)
                    )
                    if i + 1 < len(pts_k)
                )
                branch_B_mm_k = sum(
                    _math.sqrt(
                        sum((pts_k[i + 1][k] - pts_k[i][k]) ** 2 for k in range(3))
                    )
                    for i in range(
                        self.logic.branches[bi_right][0], min(gi_right, len(pts_k) - 1)
                    )
                    if i + 1 < len(pts_k)
                )

                # VBX nominal: honour spinbox if already set by user, else auto from vessel
                _vbx_noms = [5, 6, 7, 8, 9, 10, 11]
                _cur_spin = self.stentProxDiamSpin.value
                if 5.0 <= _cur_spin <= 11.0:
                    _nom_k = int(round(_cur_spin))
                else:
                    _nom_k = min(_vbx_noms, key=lambda s: abs(s - r_left))

                # ── Block ALL signals for entire UI population ─────────────
                self._autoPlacing = True
                for _w in [
                    self.kissProxExtSpin,
                    self.kissDistASpin,
                    self.kissDistBSpin,
                    self.stentProxDiamSpin,
                    self.stentDistDiamSpin,
                    self.stentProxSlider,
                    self.stentDistSlider,
                ]:
                    _w.blockSignals(True)

                # Spinboxes
                self.kissProxExtSpin.setValue(round(trunk_mm_k))
                self.kissDistASpin.setValue(round(branch_A_mm_k))
                self.kissDistBSpin.setValue(round(branch_B_mm_k))

                # Diameter: locked to VBX nominals 5-11mm
                for _sp in [self.stentProxDiamSpin, self.stentDistDiamSpin]:
                    _sp.setRange(5.0, 11.0)
                    _sp.setSingleStep(1.0)
                    _sp.setDecimals(0)
                    _sp.setVisible(True)
                self.stentProxDiamSpin.setValue(float(_nom_k))
                self.stentDistDiamSpin.setValue(float(_nom_k))

                # VBX combo
                _combo_idx = max(0, min(6, int(_nom_k) - 5))
                if hasattr(self, "kissNomCombo"):
                    self.kissNomCombo.blockSignals(True)
                    self.kissNomCombo.setCurrentIndex(_combo_idx)
                    self.kissNomCombo.blockSignals(False)

                # Sliders
                self.stentProxSlider.setValue(
                    max(0, min(100, int((trunk_mm_k - 5) / 0.55)))
                )
                self.stentDistSlider.setValue(
                    max(0, min(100, int((branch_A_mm_k - 20) / 1.8)))
                )
                self.stentProxLabel.setText(f"Trunk {trunk_mm_k:.0f}mm")
                self.stentDistLabel.setText(
                    f"A:{branch_A_mm_k:.0f}mm B:{branch_B_mm_k:.0f}mm"
                )

                # Unblock ALL at once
                for _w in [
                    self.kissProxExtSpin,
                    self.kissDistASpin,
                    self.kissDistBSpin,
                    self.stentProxDiamSpin,
                    self.stentDistDiamSpin,
                    self.stentProxSlider,
                    self.stentDistSlider,
                ]:
                    _w.blockSignals(False)

                # Show controls AFTER unblock so Qt doesn't re-fire valueChanged
                if hasattr(self, "_vbxDiamWidget"):
                    self._vbxDiamWidget.setVisible(True)
                if hasattr(self, "_kissingControlsWidget"):
                    self._kissingControlsWidget.setVisible(True)

                self._autoPlacing = False

                total_k = round(trunk_mm_k + max(branch_A_mm_k, branch_B_mm_k), 1)
                summary_txt = (
                    "Type: GORE VBX Kissing (Covered Stent Graft)  |  Length: "
                    + str(total_k)
                    + "mm\n"
                    + "Proximal: "
                    + str(round(r_trunk, 1))
                    + "mm stent / "
                    + str(round(vd_trunk, 1))
                    + "mm vessel"
                    + "  [Trunk pt"
                    + str(gi_prox_kiss)
                    + " to hub pt"
                    + str(bif_pt)
                    + "]\n"
                    + "Distal:   "
                    + str(round(r_left, 1))
                    + "mm stent / "
                    + str(round(vd_left, 1))
                    + "mm vessel  (ePTFE covered, nom 7–11mm)"
                )
                self.stentSummaryLabel.setText(summary_txt)
                # Coverage warning
                _warns_init = []
                if branch_A_mm_k > 79.0:
                    _warns_init.append(
                        f"⚠️ Br{bi_left+1}: {branch_A_mm_k:.0f}mm > VBX max 79mm "
                        f"→ need 2nd stent (+{branch_A_mm_k-79:.0f}mm)"
                    )
                if branch_B_mm_k > 79.0:
                    _warns_init.append(
                        f"⚠️ Br{bi_right+1}: {branch_B_mm_k:.0f}mm > VBX max 79mm "
                        f"→ need 2nd stent (+{branch_B_mm_k-79:.0f}mm)"
                    )
                if _warns_init:
                    self.stentWarningLabel.setText("  |  ".join(_warns_init))
                    self.stentWarningLabel.setStyleSheet(
                        "color:#c0392b; font-weight:bold; font-size:11px; "
                        "padding:4px; border-radius:4px; background:#FDEDEC;"
                    )
                else:
                    self.stentWarningLabel.setStyleSheet(
                        "color: #27ae60; font-weight: bold;"
                    )
                self._removeBalloons()
                # Enable POT after kissing stent placement
                self.potButton.setEnabled(True)
                self.potRemoveButton.setEnabled(False)
                self.potStatusLabel.setText(
                    "💡 Apply POT to restore circular trunk lumen"
                )
                self.carinaButton.setEnabled(True)
                self.carinaRemoveButton.setEnabled(False)
                self.carinaStatusLabel.setText(
                    "💡 Apply Carina Support at the confluence"
                )
                # Do NOT clear color overlay — lesion markers stay visible after stenting
                self.findingWarningLabel.setText("")
                # Mark stenoses as stented ONLY if within actual stent coverage
                if not hasattr(self, "_stentedPoints"):
                    self._stentedPoints = set()
                _plan_k = self.logic.stentPlan
                _dist_A = _plan_k.get("distPt", 9999)
                _dist_B = _plan_k.get("distPt_B", 9999)
                _prox_k = _plan_k.get("proxPt", 0)
                for f in getattr(self.logic, "findings", []):
                    if "Pancak" in f.get("type", "") or "Compress" in f.get("type", ""):
                        _fgi = f.get("pointIdx", -1)
                        # Check if lesion point is within either limb's coverage
                        _bi_f = f.get("branchIdx", -1)
                        _covered = False
                        if _bi_f == _plan_k.get("branchIdx", -1):
                            # Branch A: from branch start to distPt
                            _bs_A = (
                                self.logic.branches[_bi_f][0]
                                if _bi_f < len(self.logic.branches)
                                else 0
                            )
                            _covered = _bs_A <= _fgi <= _dist_A
                        elif _bi_f == _plan_k.get("siblingIdx", -1):
                            # Branch B
                            _bs_B = (
                                self.logic.branches[_bi_f][0]
                                if _bi_f < len(self.logic.branches)
                                else 0
                            )
                            _covered = _bs_B <= _fgi <= _dist_B
                        # Also check trunk coverage
                        if not _covered and _prox_k <= _fgi <= 95:
                            _covered = True
                        if _covered:
                            self._stentedPoints.add(_fgi)
                            print(
                                f"[Stented] Finding gi{_fgi} covered by kissing stent ✓"
                            )
                        else:
                            print(
                                f"[NotStented] Finding gi{_fgi} NOT covered "
                                f"(distA=gi{_dist_A}, distB=gi{_dist_B}) — lesion remains"
                            )
                self._removeStentedMarkers(self._stentedPoints)
                self._buildLesionTable()
                return
        # Default: Y/Trouser bifurcation stent
        self.logic.placeStent3D(plan)
        self._removeBalloons()
        n_limbs = sum(
            1 for l in [plan["y_trunk"], plan["y_left"], plan["y_right"]] if l
        )
        self.stentPickStatusLabel.setText(
            f"🔱 Y-stent ({n_limbs} limbs) | Ø{r_trunk:.1f}→Ø{r_left:.1f}/Ø{r_right:.1f}mm hub=pt{bif_pt}"
        )
        # Enable POT and Carina Support after Y-stent placement
        self.potButton.setEnabled(True)
        self.potRemoveButton.setEnabled(False)
        self.potStatusLabel.setText("💡 Apply POT to restore circular trunk lumen")
        self.carinaButton.setEnabled(True)
        self.carinaRemoveButton.setEnabled(False)
        self.carinaStatusLabel.setText("💡 Apply Carina Support at the confluence")
        self.yReplacStentButton.setEnabled(True)
        # Keep lesion overlay visible after stenting
        self.findingWarningLabel.setText("")
        # Mark all stenoses as stented and refresh table
        if not hasattr(self, "_stentedPoints"):
            self._stentedPoints = set()
        for f in getattr(self.logic, "findings", []):
            if "Pancak" in f.get("type", "") or "Compress" in f.get("type", ""):
                self._stentedPoints.add(f.get("pointIdx", -1))
        self._removeStentedMarkers(self._stentedPoints)
        self._buildLesionTable()

    def _snapToCenterline(self, ras):
        """Return global point index of centerline point nearest to RAS coord."""
        best_i, best_d2 = 0, float("inf")
        for i, pt in enumerate(self.logic.points):
            d2 = (pt[0] - ras[0]) ** 2 + (pt[1] - ras[1]) ** 2 + (pt[2] - ras[2]) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def _traceCenterlinePath(self, gi1, gi2):
        """BFS through branch graph to find centerline path from gi1 to gi2."""
        import math

        pts = self.logic.points
        branches = self.logic.branches
        N = len(pts)
        # Build neighbor map: consecutive pairs within branches + junction edges
        neighbors = [[] for _ in range(N)]
        import math

        # Find trunk end (last point of branch 0 = bifurcation hub)
        trunk_end_gi = branches[0][1] - 1 if branches else -1

        for bs, be in branches:
            for i in range(bs, be - 1):
                if i + 1 not in neighbors[i]:
                    neighbors[i].append(i + 1)
                if i not in neighbors[i + 1]:
                    neighbors[i + 1].append(i)
            # Rule A: connect every downstream branch start to the trunk end.
            # Use a generous distance threshold (200mm) to handle vessels where
            # branches are trimmed far from the trunk end.
            if bs > 0 and trunk_end_gi >= 0:
                if trunk_end_gi not in neighbors[bs]:
                    neighbors[bs].append(trunk_end_gi)
                if bs not in neighbors[trunk_end_gi]:
                    neighbors[trunk_end_gi].append(bs)

        # Connect bifurcation junctions.
        # Rule A: branch-start ↔ its predecessor (bs↔bs-1).
        #         Already done above in the branch loop.
        # Rule B: sibling branch-starts ↔ each other.
        #         Allows Dijkstra to cross between sibling branches.
        # Rule C: trunk-end ↔ ALL sibling branch-starts directly.
        #         This allows Dijkstra to route between sibling branches directly.
        import math

        branch_starts = [(bi, branches[bi][0]) for bi in range(1, len(branches))]

        def _link(g1, g2):
            if g2 not in neighbors[g1]:
                neighbors[g1].append(g2)
            if g1 not in neighbors[g2]:
                neighbors[g2].append(g1)

        # Rule B: sibling start ↔ sibling start (within 35 mm)
        for i in range(len(branch_starts)):
            for j in range(i + 1, len(branch_starts)):
                bi_i, bsi = branch_starts[i]
                bi_j, bsj = branch_starts[j]
                d = math.sqrt(sum((pts[bsi][k] - pts[bsj][k]) ** 2 for k in range(3)))
                if d < 35.0:
                    _link(bsi, bsj)

        # Rule C: for every branch-start, also link its direct predecessor (bs-1)
        # to ALL other nearby branch-starts.
        # This means branch ends connect to sibling branch starts directly.
        for bi_i, bsi in branch_starts:
            trunk_end = bsi - 1
            if trunk_end < 0:
                continue
            for bi_j, bsj in branch_starts:
                if bsj == bsi:
                    continue
                d = math.sqrt(
                    sum((pts[trunk_end][k] - pts[bsj][k]) ** 2 for k in range(3))
                )
                if d < 35.0:
                    _link(trunk_end, bsj)
        # Rule D: sub-branch end → downstream branch start (1a→B2, 1b→B3)
        #          and branch end → sub-branch start (B2→1b)
        for _sr in getattr(self.logic, "branchSubRanges", []):
            _bi_sub = _sr.get("branch_idx")
            _bi_down = _sr.get("leads_to_branch")
            if _bi_sub is None or _bi_down is None:
                continue
            if _bi_sub >= len(branches) or _bi_down >= len(branches):
                continue
            _sub_end = branches[_bi_sub][1] - 1
            _down_start = branches[_bi_down][0]
            _link(_sub_end, _down_start)
            print(
                f"[Trace] Rule D: Branch {_sr['name']} end gi={_sub_end} "
                f"→ Branch {_bi_down+1} start gi={_down_start}"
            )
        for _bi_from, _bi_to in getattr(self.logic, "branchConnections", {}).items():
            if _bi_from >= len(branches) or _bi_to >= len(branches):
                continue
            _from_end = branches[_bi_from][1] - 1
            _to_start = branches[_bi_to][0]
            _link(_from_end, _to_start)
            print(
                f"[Trace] Rule D: Branch {_bi_from+1} end gi={_from_end} "
                f"→ sub-branch start gi={_to_start}"
            )

        # Dijkstra weighted by spatial distance — picks path through anatomical hub
        # (BFS would skip the hub and jump directly between branch starts)
        import heapq

        dist_map = {gi1: 0.0}
        prev_map = {gi1: None}
        heap = [(0.0, gi1)]
        while heap:
            cost, cur = heapq.heappop(heap)
            if cur == gi2:
                break
            if cost > dist_map.get(cur, float("inf")):
                continue
            for nb in neighbors[cur]:
                p0 = pts[cur]
                p1 = pts[nb]
                edge_w = math.sqrt(sum((p1[k] - p0[k]) ** 2 for k in range(3)))
                new_cost = cost + edge_w
                if new_cost < dist_map.get(nb, float("inf")):
                    dist_map[nb] = new_cost
                    prev_map[nb] = cur
                    heapq.heappush(heap, (new_cost, nb))
        if gi2 not in prev_map:
            return []
        # Reconstruct path
        path = []
        cur = gi2
        while cur is not None:
            path.append(cur)
            cur = prev_map[cur]
        path.reverse()

        # ── Gap bridging ─────────────────────────────────────────────────────
        # Dijkstra hops between sibling branch-starts across the bifurcation
        # hub (gap between branch junction points, no centerline data in between).
        # We fill that space with a Hermite cubic spline using:
        #   t0 = outgoing tangent of the SOURCE branch (last few pts before hop)
        #   t1 = INCOMING tangent of the TARGET branch (first few pts, reversed)
        # This produces a smooth curve that leaves p0 in the source direction
        # and arrives at p1 in the target branch direction — no U-turns,
        # no duplicates.
        #
        # Non-bifurcation large gaps (sparse intra-branch sampling) get simple
        # linear interpolation which is sufficient there.
        #
        # All synthetic bridge points get negative indices stored in
        # self.logic._interpPoints so downstream code can resolve them.
        GAP_INTERP_THRESH = 5.0  # mm — gaps larger than this get bridged
        INTERP_STEP = 2.0  # mm — spacing for linear interp fallback
        TANGENT_LOOK = 4  # points to use for one-sided tangent estimate

        if not hasattr(self.logic, "_interpPoints"):
            self.logic._interpPoints = {}
        self.logic._interpPoints.clear()
        _next_synth = -1

        branch_start_set = {bs for bs, _be in branches}

        def _branch_of(gi):
            for bi, (bs, be) in enumerate(branches):
                if bs <= gi < be:
                    return bi, bs, be
            return None

        def _unit(v):
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            return tuple(x / n for x in v)

        def _outgoing_tangent(gi):
            """Tangent leaving gi — look forward along its branch."""
            info = _branch_of(gi)
            if not info:
                return (0, 0, -1)
            _, bs, be = info
            # walk forward up to TANGENT_LOOK steps
            i0 = gi
            i1 = min(gi + TANGENT_LOOK, be - 1)
            if i1 == i0:
                i0 = max(bs, i0 - 1)
            v = tuple(pts[i1][k] - pts[i0][k] for k in range(3))
            return _unit(v)

        def _incoming_tangent(gi):
            """Tangent arriving INTO gi — look forward from branch start."""
            info = _branch_of(gi)
            if not info:
                return (0, 0, -1)
            _, bs, be = info
            # tangent of the branch at its start = forward direction
            i0 = bs
            i1 = min(bs + TANGENT_LOOK, be - 1)
            v = tuple(pts[i1][k] - pts[i0][k] for k in range(3))
            # arriving INTO the start means coming from the opposite direction
            return _unit(tuple(-x for x in v))

        def _hermite_bridge(p0, t0, p1, t1, gap):
            """Hermite cubic: leaves p0 along t0, arrives at p1 along t1."""
            n_steps = max(4, int(gap / INTERP_STEP))
            chord = gap  # scale tangents by gap length
            result = []
            for s in range(1, n_steps):  # exclude endpoints (already in path)
                t = s / n_steps
                t2, t3 = t * t, t * t * t
                h00 = 2 * t3 - 3 * t2 + 1
                h10 = t3 - 2 * t2 + t
                h01 = -2 * t3 + 3 * t2
                h11 = t3 - t2
                p = tuple(
                    h00 * p0[k]
                    + h10 * chord * t0[k]
                    + h01 * p1[k]
                    + h11 * chord * t1[k]
                    for k in range(3)
                )
                result.append(p)
            return result

        smoothed = [path[0]]
        for k in range(1, len(path)):
            prev_gi = path[k - 1]
            curr_gi = path[k]
            p0 = (
                pts[prev_gi]
                if prev_gi >= 0
                else self.logic._interpPoints.get(prev_gi, (0, 0, 0))
            )
            p1 = (
                pts[curr_gi]
                if curr_gi >= 0
                else self.logic._interpPoints.get(curr_gi, (0, 0, 0))
            )
            gap = math.sqrt(sum((p1[j] - p0[j]) ** 2 for j in range(3)))

            if gap > GAP_INTERP_THRESH:
                # A bifurcation hop is any large gap where we cross between
                # different branches — either start→start (Rule B) or
                # trunk-end→sibling-start (Rule C, the new direct edge).
                prev_branch = _branch_of(prev_gi)
                curr_branch = _branch_of(curr_gi)
                is_bifurcation_hop = (
                    curr_gi >= 0
                    and prev_gi >= 0
                    and prev_branch is not None
                    and curr_branch is not None
                    and prev_branch[0] != curr_branch[0]  # different branches
                )

                if is_bifurcation_hop:
                    # Hermite spline: source outgoing → target incoming
                    t0 = _outgoing_tangent(prev_gi)
                    t1 = _incoming_tangent(curr_gi)
                    bridge_pts = _hermite_bridge(p0, t0, p1, t1, gap)
                    for bp in bridge_pts:
                        self.logic._interpPoints[_next_synth] = bp
                        smoothed.append(_next_synth)
                        _next_synth -= 1
                    info = _branch_of(curr_gi)
                    _bi = info[0] if info else "?"
                    _disp = (
                        self.logic.getBranchDisplayName(_bi)
                        if isinstance(_bi, int)
                        else str(_bi)
                    )
                    print(
                        f"[Trace] Hermite bridge: {len(bridge_pts)} pts "
                        f"gi={prev_gi}→{curr_gi} ({_disp})"
                    )
                else:
                    # Linear for non-bifurcation sparse gaps
                    n_steps = max(1, int(gap / INTERP_STEP))
                    for s in range(1, n_steps):
                        t = s / n_steps
                        interp = (
                            p0[0] + t * (p1[0] - p0[0]),
                            p0[1] + t * (p1[1] - p0[1]),
                            p0[2] + t * (p1[2] - p0[2]),
                        )
                        self.logic._interpPoints[_next_synth] = interp
                        smoothed.append(_next_synth)
                        _next_synth -= 1

            smoothed.append(curr_gi)
        path = smoothed
        # ─────────────────────────────────────────────────────────────────────
        return path

    def _removeBalloons(self):
        """Remove all balloon nodes from scene after stenting."""
        import traceback as _tb

        _caller = "".join(_tb.format_stack()[-3:-1])[-80:]
        all_models = slicer.util.getNodesByClass("vtkMRMLModelNode")
        balloon_nodes = [
            n for n in all_models if n.GetName().startswith("BalloonDilate")
        ]
        for node in balloon_nodes:
            slicer.mrmlScene.RemoveNode(node)
        # Verify removal
        remaining = [
            n
            for n in slicer.util.getNodesByClass("vtkMRMLModelNode")
            if n.GetName().startswith("BalloonDilate")
        ]

    def _removeStentedMarkers(self, stented_pts_set):
        """Hide lesion markers whose pointIdx is in the stented set."""
        if not stented_pts_set:
            return
        for i in range(slicer.mrmlScene.GetNumberOfNodes()):
            n = slicer.mrmlScene.GetNthNode(i)
            if n and n.GetAttribute("VesselAnalyzerOverlay") == "1":
                name = n.GetName()
                # Finding nodes are named Finding_<Type>_<pointIdx>
                if name.startswith("Finding_"):
                    parts = name.rsplit("_", 1)
                    if len(parts) == 2:
                        try:
                            pt_idx = int(parts[1])
                            if pt_idx in stented_pts_set:
                                dn = n.GetDisplayNode()
                                if dn:
                                    dn.SetVisibility(0)
                        except ValueError:
                            pass

    def onStentPlace(self):
        if not self.logic.points:
            slicer.util.errorDisplay("Load a centerline first.")
            return
        # Always recompute plan fresh at click time with current branch
        self._refreshStentSummary()
        plan = self.logic.stentPlan
        stent_bi = self.stentBranchCombo.currentIndex - 1
        branch_start = self.logic.branches[stent_bi][0] if stent_bi >= 0 else 0
        print(
            "[VesselAnalyzer] Placing stent: stentBranch="
            + str(stent_bi + 1)
            + " branch_start="
            + str(branch_start)
            + " proxPt="
            + str(plan["proxPt"])
            + " distPt="
            + str(plan["distPt"])
            + " length="
            + str(plan["length"])
            + "mm"
        )
        self.logic.placeStent3D(plan)
        # Remove balloon, restore mesh and diameters — stent now holds expansion
        self._removeBalloons()
        # (mesh deformation disabled — nothing to restore geometrically)
        # Restore original diameters (stent holds lumen — diameters stay healthy)
        if hasattr(self.logic, "_origDiameters"):
            self.logic._origDiameters = {}
        # Keep lesion overlay visible after stenting
        self.findingWarningLabel.setText("")
        self.preDilateStatusLabel.setText("✅ Stent placed over pre-dilated segment")
        # Mark ALL stenosis lesions as stented
        if not hasattr(self, "_stentedPoints"):
            self._stentedPoints = set()
        findings = getattr(self.logic, "findings", [])
        for f in findings:
            if "Pancak" in f.get("type", "") or "Compress" in f.get("type", ""):
                self._stentedPoints.add(f.get("pointIdx", -1))
        self._removeStentedMarkers(self._stentedPoints)
        self._buildLesionTable()
        self.preDilateButton.setText("🎈 Pre-Dilate Lesion  (balloon angioplasty)")
        self.compareBeforeAfterButton.setEnabled(False)
        self.addRulerButton.setEnabled(False)
        self.removeRulerButton.setEnabled(False)
        self._restoreDefaultLayout()
        # Restore vessel visibility and re-enable slider
        for _mn in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            _nm = _mn.GetName()
            if _nm.startswith("BalloonDilate"):
                continue
            _dn2 = _mn.GetDisplayNode()
            if _dn2:
                _dn2.SetVisibility(1)
                _dn2.SetOpacity(1.0)
        self.modelOpacitySlider.blockSignals(True)
        self.modelOpacitySlider.setValue(1.0)
        self.modelOpacitySlider.setEnabled(True)
        self.modelOpacitySlider.blockSignals(False)

    def onStentRemove(self):
        try:
            self.logic.removeStent3D()
            # Remove ALL intervention nodes: balloons, POT, carina, expanded vessel
            nodes_to_remove = []
            for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
                nm = node.GetName()
                if (
                    nm.startswith("BalloonDilate")
                    or nm.endswith("_Ring")
                    or nm.endswith("_wire")
                    or nm == "POT_Balloon"
                    or nm == "CarinaSupport"
                    or node.GetAttribute("VesselAnalyzerExpanded") == "1"
                    or nm == "ExpandedVessel"
                ):
                    nodes_to_remove.append(node)
            for node in nodes_to_remove:
                slicer.mrmlScene.RemoveNode(node)

            self.stentWarningLabel.setText("")
            # Restore diameters to original compressed state
            if hasattr(self.logic, "_origDiameters"):
                for gj, orig_d in self.logic._origDiameters.items():
                    if gj < len(self.logic.diameters):
                        self.logic.diameters[gj] = orig_d
                self.logic._origDiameters = {}
            if hasattr(self.logic, "preDilationMap"):
                self.logic.preDilationMap = {}
            # Unfreeze findings so they can be re-detected after stent removal
            self.logic._findingsFrozen = False

            # Re-enable pre-dilation if stenoses still exist
            stenoses = [
                f
                for f in getattr(self.logic, "findings", [])
                if "Pancak" in f.get("type", "") or "Compress" in f.get("type", "")
            ]
            self.preDilateButton.setEnabled(len(stenoses) > 0)
            self.manualBalloonButton.setEnabled(True)
            self.preDilateButton.setText("🎈 Pre-Dilate Lesion  (balloon angioplasty)")
            self.preDilateButton.setToolTip(
                "Visualize balloon pre-dilation at the detected stenosis before stent placement."
            )
            self.preDilateStatusLabel.setText(
                str(len(stenoses))
                + " stenosis found — click Pre-Dilate before placing stent"
                if stenoses
                else ""
            )
            self._stentedPoints = set()
            self._buildLesionTable()
            self.preDilateCopyBtn.setVisible(False)

            # Reset POT and Carina buttons
            self.potButton.setEnabled(False)
            self.potRemoveButton.setEnabled(False)
            self.potStatusLabel.setText("")
            self.carinaButton.setEnabled(False)
            self.carinaRemoveButton.setEnabled(False)
            self.carinaStatusLabel.setText("")
            self.yReplacStentButton.setEnabled(False)

            # Restore vessel visibility — but NEVER change the vessel model's opacity
            # The user controls vessel opacity; stent removal must not override it.
            _vessel_node_id = (
                self.logic.modelNode.GetID() if self.logic.modelNode else None
            )
            for _mn in slicer.util.getNodesByClass("vtkMRMLModelNode"):
                _nm = _mn.GetName()
                if (
                    _nm.startswith("BalloonDilate")
                    or _nm.endswith("_Ring")
                    or _nm.endswith("_wire")
                    or _nm == "POT_Balloon"
                    or _nm == "CarinaSupport"
                ):
                    continue
                # Skip the main vessel model — never touch its opacity
                if _vessel_node_id and _mn.GetID() == _vessel_node_id:
                    _dn2 = _mn.GetDisplayNode()
                    if _dn2:
                        _dn2.SetVisibility(1)  # ensure visible but keep opacity
                    continue
                _dn2 = _mn.GetDisplayNode()
                if _dn2:
                    _dn2.SetVisibility(1)
                    _dn2.SetOpacity(1.0)
            # Hide centerlines
            for _node in (
                list(slicer.util.getNodesByClass("vtkMRMLModelNode"))
                + list(slicer.util.getNodesByClass("vtkMRMLMarkupsCurveNode"))
                + list(slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"))
            ):
                if any(
                    x in _node.GetName().lower()
                    for x in ("centerline", "endpoint", "network")
                ):
                    _dn2 = _node.GetDisplayNode()
                    if _dn2:
                        _dn2.SetVisibility(0)
            # Restore original model color
            if self.logic.modelNode:
                _mdn = self.logic.modelNode.GetDisplayNode()
                if _mdn:
                    _oc = getattr(self, "_origModelColor", (0.9, 0.9, 0.9))
                    _mdn.SetColor(_oc[0], _oc[1], _oc[2])
            self.modelOpacitySlider.blockSignals(True)
            self.modelOpacitySlider.setValue(1.0)
            self.modelOpacitySlider.setEnabled(True)
            self.modelOpacitySlider.blockSignals(False)

            # Hide IVUS visualizations (ring, sphere) — stent removal resets the scene
            try:
                self.logic.updateVisualizations(
                    int(self.pointSlider.value),
                    showSphere=False,
                    showRing=False,
                    showLine=False,
                )
            except Exception:
                pass

            # Always return to single 3D view after stent removal.
            # Multiple deferred calls needed: Slicer fires NodeRemovedEvent after
            # RemoveNode returns, which can re-trigger layout switches.
            self._TWO3D_LAYOUT_ID = (
                None  # clear whitelist so observer enforces single-3D
            )
            self._force3DLayout()
            qt.QTimer.singleShot(100, self._force3DLayout)
            qt.QTimer.singleShot(400, self._force3DLayout)
            qt.QTimer.singleShot(800, self._force3DLayout)
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print("[StentRemove] Error: " + str(e) + "\n" + tb)
            slicer.util.errorDisplay("Remove stent failed:\n" + str(e))

    def onGenerateReport(self):
        if not self.logic.points:
            slicer.util.errorDisplay("Please load a centerline first.")
            return
        if self.vesselTypeCombo.currentIndex == 0:
            if not slicer.util.confirmYesNoDisplay(
                "Vessel Type has not been selected.\n\n"
                "The report will be generated with default Arterial thresholds.\n\n"
                "Proceed, or cancel and set the type in Step 1?",
                windowTitle="Vessel Type Not Set",
            ):
                return
        import datetime as _dt

        default_name = (
            "VesselReport_" + _dt.datetime.now().strftime("%Y%m%d_%H%M") + ".docx"
        )
        path = qt.QFileDialog.getSaveFileName(
            None, "Save Report", default_name, "Word Documents (*.docx)"
        )
        if not path:
            return
        if not path.endswith(".docx"):
            path += ".docx"
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.logic.generateReport(path)
            slicer.util.infoDisplay(f"Report saved to:\n{path}")
        except Exception as e:
            import traceback

            slicer.util.errorDisplay(
                f"Report generation failed:\n{e}\n{traceback.format_exc()}"
            )
        finally:
            slicer.app.restoreOverrideCursor()


# =============================================================================
# Branch Detection Utilities
# =============================================================================
# These module-level classes and functions implement the standalone branch
# detection algorithm.  They operate purely on numpy arrays (no VTK / Slicer
# dependencies) so they can be unit-tested independently of the Slicer runtime.
#
# Integration points inside VesselAnalyzerLogic:
#   • loadCenterline  – score_bifurcation() augments step-3 validation; the
#                       score is stored in branchMeta['bifScore'] for each node.
#   • _detectFindings – detect_compression() cross-validates the existing
#                       run-detection result with a fast dmin/dmax ratio screen.



# ── Logic mixins ─────────────────────────────────────────────────────────────
from vessel_centerline_mixin import CenterlineMixin
from vessel_ostium_mixin import OstiumMixin
from vessel_findings_mixin import FindingsMixin
from vessel_visualization_mixin import VisualizationMixin
from vessel_stent_mixin import StentMixin
from vessel_branch_accessor_mixin import BranchAccessorMixin
from vessel_report_mixin import ReportMixin




# ── Slicer module-scanner guard ───────────────────────────────────────────────
class vessel_intervention_widget_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_intervention_widget_mixin"
            parent.hidden = True

