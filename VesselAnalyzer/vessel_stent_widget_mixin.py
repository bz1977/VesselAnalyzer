"""
vessel_stent_widget_mixin.py — VesselAnalyzerWidget stent-planning handlers

Extracted from VesselAnalyzer.py (VesselAnalyzerWidget methods).

Usage in VesselAnalyzer.py
--------------------------
Add to the VesselAnalyzerWidget class definition::

    from vessel_stent_widget_mixin import StentWidgetMixin

    class VesselAnalyzerWidget(StentWidgetMixin, ScriptedLoadableModuleWidget, VTKObservationMixin):
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


class StentWidgetMixin:
    """Mixin: VesselAnalyzerWidget stent-planning handlers"""
    def _currentStentBranchIndex(self):
        """Return the real branch id selected in the stent planner."""
        try:
            index = self.stentBranchCombo.currentIndex
            if index < 1:
                return -1
            bi = self.stentBranchCombo.itemData(index)
            if bi is None:
                bi = index - 1
            bi = int(bi)
            return bi if 0 <= bi < len(self.logic.branches) else -1
        except Exception:
            return -1

    def _selectStentBranchByIndex(self, branchIdx):
        """Select the hidden stent combo row whose userData is branchIdx."""
        try:
            for row in range(self.stentBranchCombo.count):
                if self.stentBranchCombo.itemData(row) == branchIdx:
                    self.stentBranchCombo.setCurrentIndex(row)
                    return True
        except Exception:
            pass
        return False

    def _updateStentSliderRanges(self):
        """Populate branch list + hidden combo. Called after centerline loads."""
        # Block ALL stent UI signals during repopulation to prevent
        # _liveUpdateStent from firing (causes 10x placeStent3D on load)
        _block = [
            self.stentProxSlider,
            self.stentDistSlider,
            self.stentTypeCombo,
            self.stentBranchCombo,
            self.stentProxDiamSpin,
            self.stentDistDiamSpin,
        ]
        for _w in _block:
            try:
                _w.blockSignals(True)
            except Exception:
                pass
        # Populate visual list widget
        self.stentBranchList.blockSignals(True)
        self.stentBranchList.clear()
        # Keep hidden combo in sync for backward-compat code
        self.stentBranchCombo.clear()
        self.stentBranchCombo.addItem("-- Select branch --", -1)
        for bi in range(len(self.logic.branches)):
            role = getattr(self.logic, "branchMeta", {}).get(bi, {}).get("role", "")
            if bi == 0 or role in ("trunk", "noise", "renal_fragment", "iliac_fragment"):
                continue
            stats = self.logic.getBranchStats(bi)
            label = (
                f"{self.logic.getBranchDisplayName(bi)}  "
                f"({stats['length']:.0f}mm, min={stats['min']:.1f}mm, max={stats['max']:.1f}mm)"
            )
            item = qt.QListWidgetItem(label)
            item.setData(qt.Qt.UserRole, bi)
            self.stentBranchList.addItem(item)
            self.stentBranchCombo.addItem(label, bi)
        self.stentBranchList.blockSignals(False)
        self.logic.stentPlan = None
        for _w in _block:
            try:
                _w.blockSignals(False)
            except Exception:
                pass

    def _resetStentSlidersForBranch(self, branchIdx):
        """Set slider range and defaults for a specific branch."""
        s, e = self.logic.branches[branchIdx]
        total = e - s
        if total < 2:
            return
        self.stentProxSlider.blockSignals(True)
        self.stentDistSlider.blockSignals(True)
        self.stentProxSlider.setMinimum(0)
        self.stentProxSlider.setMaximum(total - 1)
        self.stentDistSlider.setMinimum(0)
        self.stentDistSlider.setMaximum(total - 1)
        self.stentProxSlider.setValue(int(total * 0.15))
        self.stentDistSlider.setValue(int(total * 0.85))
        self.stentProxLabel.setText("Pt " + str(int(total * 0.15)))
        self.stentDistLabel.setText("Pt " + str(int(total * 0.85)))
        self.stentProxSlider.blockSignals(False)
        self.stentDistSlider.blockSignals(False)

    def onStentBranchListChanged(self):
        """User selected branch(es) in the list widget."""
        selected = self.stentBranchList.selectedItems()
        if not selected:
            self.stentSummaryLabel.setText("Select a target branch above.")
            return
        # Sync hidden combo to first selected branch for slider/summary compat
        branchIdx = selected[0].data(qt.Qt.UserRole)
        if branchIdx is None:
            first_row = self.stentBranchList.row(selected[0])
            branchIdx = first_row
        branchIdx = int(branchIdx)
        self.stentBranchCombo.blockSignals(True)
        self._selectStentBranchByIndex(branchIdx)
        self.stentBranchCombo.blockSignals(False)
        self._resetStentSlidersForBranch(branchIdx)
        # Show summary with all selected branches
        branch_names = [item.text().split("  ")[0] for item in selected]
        if len(selected) > 1:
            self.stentSummaryLabel.setText(
                "Selected: "
                + ", ".join(branch_names)
                + "  — Place stent per branch or use Y-stent button"
            )
        self._refreshStentSummary()

    def onStentBranchChanged(self, index):
        """User selected a branch in the stent planner (hidden combo sync)."""
        if index < 1:
            self.stentSummaryLabel.setText("Select a target branch above.")
            self.stentWarningLabel.setText("")
            return
        branchIdx = self._currentStentBranchIndex()
        if branchIdx < 0:
            return
        self._resetStentSlidersForBranch(branchIdx)
        self._refreshStentSummary()

    def onStentSliderChanged(self, value=0):
        # For kissing stents: prox slider = trunk coverage, dist slider = branch depth
        stype = self.stentTypeCombo.currentText.strip()
        is_kissing = "Kissing" in stype or "VBX" in stype
        if is_kissing and getattr(self.logic, "stentPlan", None):
            # Map prox slider 0-100 → 5-60mm trunk coverage
            prox_mm = 5.0 + self.stentProxSlider.value * 0.55  # 0→5mm, 100→60mm
            # Map dist slider 0-100 → 10-79mm branch depth (VBX max)
            dist_mm = 10.0 + self.stentDistSlider.value * 0.69  # 0→10mm, 100→79mm
            self.stentProxLabel.setText(f"Trunk {prox_mm:.0f}mm")
            self.stentDistLabel.setText(f"Branch {dist_mm:.0f}mm")
            # Push values into the kissing spinboxes and re-apply
            self.kissProxExtSpin.blockSignals(True)
            self.kissDistASpin.blockSignals(True)
            self.kissDistBSpin.blockSignals(True)
            self.kissProxExtSpin.setValue(prox_mm)
            self.kissDistASpin.setValue(dist_mm)
            self.kissDistBSpin.setValue(dist_mm)
            self.kissProxExtSpin.blockSignals(False)
            self.kissDistASpin.blockSignals(False)
            self.kissDistBSpin.blockSignals(False)
            self.onKissingApply()
            return

        p = self.stentProxSlider.value
        di = self.stentDistSlider.value
        # Enforce prox < dist — clamp whichever moved last
        if p >= di - 4:
            if value == p and p >= di - 4:
                self.stentProxSlider.blockSignals(True)
                self.stentProxSlider.setValue(max(0, di - 5))
                self.stentProxSlider.blockSignals(False)
            else:
                self.stentDistSlider.blockSignals(True)
                self.stentDistSlider.setValue(min(self.stentDistSlider.maximum, p + 5))
                self.stentDistSlider.blockSignals(False)
        p = self.stentProxSlider.value
        di = self.stentDistSlider.value
        self.stentProxLabel.setText("Pt " + str(p))
        self.stentDistLabel.setText("Pt " + str(di))
        self._refreshStentSummary()
        self._liveUpdateStent()

    def onStentParamChanged(self, value=0):
        stype = self.stentTypeCombo.currentText.strip()
        is_kissing = "Kissing" in stype or "VBX" in stype
        if hasattr(self, "_kissingControlsWidget"):
            self._kissingControlsWidget.setVisible(is_kissing)
        if hasattr(self, "_vbxDiamWidget"):
            self._vbxDiamWidget.setVisible(is_kissing)
        # For kissing: restrict diameter spinboxes to VBX nominal range 5–11mm
        if is_kissing:
            for _sp in [self.stentProxDiamSpin, self.stentDistDiamSpin]:
                if _sp.maximum > 11.0:
                    _sp.blockSignals(True)
                    cur = _sp.value
                    _sp.setRange(5.0, 11.0)
                    _sp.setSingleStep(1.0)
                    _sp.setDecimals(0)
                    _sp.setValue(min(11.0, max(5.0, round(cur))))
                    _sp.blockSignals(False)
        else:
            for _sp in [self.stentProxDiamSpin, self.stentDistDiamSpin]:
                if _sp.maximum < 12.0:
                    _sp.blockSignals(True)
                    _sp.setRange(4.0, 40.0)
                    _sp.setSingleStep(0.5)
                    _sp.setDecimals(1)
                    _sp.blockSignals(False)
        self.stentProxDiamSpin.setVisible(True)
        self.stentDistDiamSpin.setVisible(True)
        self._refreshStentSummary()
        self._liveUpdateStent()

    def onStentLengthApply(self, value=0):
        """For kissing stents: length spinbox = Branch A depth in mm (lesion side).
        Branch B is set independently via its own spinbox.
        Snaps to nearest VBX available length."""
        if not self.logic.points:
            return
        stype = self.stentTypeCombo.currentText.strip()
        is_kissing = "Kissing" in stype or "VBX" in stype

        if is_kissing and getattr(self.logic, "stentPlan", None):
            # Snap to nearest VBX length (15/19/29/39/59/79mm)
            VBX_LENGTHS = [15, 19, 29, 39, 59, 79]
            raw = self.stentLengthSpin.value
            snapped = min(VBX_LENGTHS, key=lambda l: abs(l - raw))
            if abs(snapped - raw) > 0.5:
                self.stentLengthSpin.blockSignals(True)
                self.stentLengthSpin.setValue(snapped)
                self.stentLengthSpin.blockSignals(False)
            # Length spinbox = Branch A depth directly (NOT total - trunk)
            # This way 79mm → 79mm into the iliac, matching GORE device length
            self.kissDistASpin.blockSignals(True)
            self.kissDistASpin.setValue(float(snapped))
            self.kissDistASpin.blockSignals(False)
            print(
                f"[LengthApply] Kissing: branchA={snapped}mm "
                f"(branchB stays at {self.kissDistBSpin.value:.0f}mm)"
            )
            self.onKissingApply()
            return

        # Snap to nearest real GORE length
        raw_len = self.stentLengthSpin.value
        snapped = self._snapStentLength(raw_len)
        if abs(snapped - raw_len) > 0.5:
            self.stentLengthSpin.blockSignals(True)
            self.stentLengthSpin.setValue(snapped)
            self.stentLengthSpin.blockSignals(False)
        target_mm = snapped
        stent_bi = self._currentStentBranchIndex()
        if stent_bi < 0:
            return
        bs, be = self.logic.branches[stent_bi]
        branch_len = be - bs
        # Compute avg spacing for this branch
        sp = [
            (
                (self.logic.points[j + 1][0] - self.logic.points[j][0]) ** 2
                + (self.logic.points[j + 1][1] - self.logic.points[j][1]) ** 2
                + (self.logic.points[j + 1][2] - self.logic.points[j][2]) ** 2
            )
            ** 0.5
            for j in range(bs, min(bs + 20, be - 1))
        ]
        avg_sp = sum(sp) / len(sp) if sp else 1.3
        n_pts = max(5, int(round(target_mm / avg_sp)))
        prox = self.stentProxSlider.value
        dist = min(branch_len - 1, prox + n_pts)
        self.stentDistSlider.blockSignals(True)
        self.stentDistSlider.setValue(dist)
        self.stentDistSlider.blockSignals(False)
        self.stentDistLabel.setText("Pt " + str(dist))
        self._refreshStentSummary()
        self._liveUpdateStent()

    def onKissNomChanged(self, index=0):
        """VBX nominal combo changed — sync to spinbox and redraw."""
        nom = index + 5  # combo index 0=5mm, 1=6mm, ... 6=11mm
        # Push selected nominal into the spinbox so both controls are in sync
        self.stentProxDiamSpin.blockSignals(True)
        self.stentDistDiamSpin.blockSignals(True)
        self.stentProxDiamSpin.setValue(float(nom))
        self.stentDistDiamSpin.setValue(float(nom))
        self.stentProxDiamSpin.blockSignals(False)
        self.stentDistDiamSpin.blockSignals(False)
        plan = getattr(self.logic, "stentPlan", None)
        if plan:
            plan["proxDiam"] = float(nom)
            plan["distDiam"] = float(nom)
        print(f"[KissNom] selected {nom}mm → spinbox updated")
        self._liveUpdateStent()

    def onTrunkSpinChanged(self, value=0.0):
        """Trunk overlap spinbox changed — update plan and redraw stent."""
        self._refreshStentSummary()
        self._liveUpdateStent()

    def onKissingApply(self, value=0):
        """Re-place kissing stent with trunk coverage and branch depth from spinboxes."""
        if getattr(self, "_autoPlacing", False):
            return
        plan = getattr(self.logic, "stentPlan", None)
        if not plan:
            return
        stype = self.stentTypeCombo.currentText.strip()
        if "Kissing" not in stype and "VBX" not in stype:
            return
        if not (
            hasattr(self.logic, "_stentNode")
            and self.logic._stentNode
            and slicer.mrmlScene.IsNodePresent(self.logic._stentNode)
        ):
            return

        import math as _km

        pts = self.logic.points
        branches = self.logic.branches
        bi_A = plan.get("branchIdx", 1)
        bi_B = plan.get("siblingIdx", 2)
        hub_gi = branches[0][1] - 1
        trunk_bs = branches[0][0]

        # ── Proximal: walk back from hub by kissProxExtSpin mm ───────────
        prox_target_mm = max(5.0, self.kissProxExtSpin.value)
        acc = 0.0
        new_prox = hub_gi
        for ti in range(hub_gi - 1, trunk_bs - 1, -1):
            if ti + 1 < len(pts):
                p0, p1 = pts[ti], pts[ti + 1]
                acc += _km.sqrt(sum((p1[k] - p0[k]) ** 2 for k in range(3)))
            new_prox = ti
            if acc >= prox_target_mm:
                break

        # ── Compute actual trunk length from new_prox to hub ────────────────
        import math as _cap_m

        trunk_mm_actual = sum(
            _cap_m.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
            for i in range(new_prox, min(hub_gi, len(pts) - 1))
            if i + 1 < len(pts)
        )
        branch_budget_mm = max(10.0, 79.0 - trunk_mm_actual)

        # ── Distal A/B: walk spinbox mm, capped to VBX device budget ─────────
        def walk_mm_from_start(bi, target_mm):
            bs, be = branches[bi]
            acc = 0.0
            gi_end = bs
            for i in range(bs, be - 1):
                if i + 1 < len(pts):
                    acc += _km.sqrt(
                        sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3))
                    )
                gi_end = i + 1
                if acc >= target_mm:
                    break
            return min(gi_end, be - 1)

        capped_A = self.kissDistASpin.value
        capped_B = self.kissDistBSpin.value
        new_dist_A = walk_mm_from_start(bi_A, capped_A)
        new_dist_B = walk_mm_from_start(bi_B, capped_B)

        # Update plan geometry
        plan["proxPt"] = new_prox
        plan["distPt"] = new_dist_A
        plan["distPt_B"] = new_dist_B
        # Read diameter from spinbox (VBX nominal snap happens inside _placeKissingStent3D)
        plan["proxDiam"] = self.stentProxDiamSpin.value
        plan["distDiam"] = self.stentDistDiamSpin.value

        print(
            f"[KissingApply] prox=gi{new_prox} ({acc:.0f}mm from hub) "
            f"trunk={trunk_mm_actual:.0f}mm budget={branch_budget_mm:.0f}mm  "
            f"distA=gi{new_dist_A}(bi{bi_A},{capped_A:.0f}mm)  "
            f"distB=gi{new_dist_B}(bi{bi_B},{capped_B:.0f}mm)  "
            f"nom={self.stentProxDiamSpin.value:.0f}mm"
        )

        self.logic.placeStent3D(plan)
        self._removeBalloons()

        # ── Coverage warning ─────────────────────────────────────────────────
        _dA = self.kissDistASpin.value
        _dB = self.kissDistBSpin.value
        _warns = []
        if _dA > 79.0:
            _overlap = _dA - 79.0
            _warns.append(
                f"⚠️ Br A: {_dA:.0f}mm > VBX max 79mm  "
                f"(need 2nd stent, {_overlap:.0f}mm extension)"
            )
        if _dB > 79.0:
            _overlap_b = _dB - 79.0
            _warns.append(
                f"⚠️ Br B: {_dB:.0f}mm > VBX max 79mm  "
                f"(need 2nd stent, {_overlap_b:.0f}mm extension)"
            )
        if _warns:
            self.stentWarningLabel.setText("\n".join(_warns))
            self.stentWarningLabel.setStyleSheet(
                "color:#c0392b; font-weight:bold; font-size:11px; "
                "padding:4px; border-radius:4px; background:#FDEDEC;"
            )
        else:
            self.stentWarningLabel.setText("✓ Within VBX device limits")
            self.stentWarningLabel.setStyleSheet(
                "color:#1B5E20; font-weight:bold; font-size:11px; "
                "padding:4px; border-radius:4px; background:#E8F5E9;"
            )

        self._refreshStentSummary()

    def _liveUpdateStent(self):
        """Re-place stent silently if one already exists (live drag repositioning).
        Handles straight, kissing, and Y-stent modes correctly."""
        if getattr(self, "_autoPlacing", False):
            return
        if getattr(self, "_liveUpdateActive", False):
            return  # re-entrancy guard
        if not (
            hasattr(self.logic, "_stentNode")
            and self.logic._stentNode
            and slicer.mrmlScene.IsNodePresent(self.logic._stentNode)
        ):
            return
        plan = self.logic.stentPlan
        if not plan:
            return
        self._liveUpdateActive = True
        try:
            combo_type = self.stentTypeCombo.currentText.strip()
            plan["type"] = combo_type

            is_kissing = "Kissing" in combo_type or "VBX" in combo_type
            is_y = "Bifurcated" in combo_type and plan.get("y_trunk") is not None
            is_bifurcated = is_kissing or is_y

            # ── Kissing stent: read diameter from spinbox ─────────────────────
            if is_kissing:
                nom_val = min(11, max(5, round(self.stentProxDiamSpin.value)))
                plan["proxDiam"] = float(nom_val)
                plan["distDiam"] = float(nom_val)
                # Keep VBX combo in sync with spinbox
                if hasattr(self, "kissNomCombo"):
                    _ci = nom_val - 5  # 5→0, 6→1, ... 11→6
                    if self.kissNomCombo.currentIndex != _ci:
                        self.kissNomCombo.blockSignals(True)
                        self.kissNomCombo.setCurrentIndex(_ci)
                        self.kissNomCombo.blockSignals(False)
                print(f"[LiveUpdate] Kissing: proxDiam={nom_val}mm")
                self.logic.placeStent3D(plan)
                self._removeBalloons()
                return

            # ── Y-stent: diameter spinboxes update r_left/r_right/r_trunk ────
            if is_y:
                pd = self.stentProxDiamSpin.value
                dd = self.stentDistDiamSpin.value
                import math as _ym

                r_left = dd / 2.0
                r_right = dd / 2.0
                r_trunk_geom = _ym.sqrt(r_left**2 + r_right**2)
                r_trunk = min(r_trunk_geom, pd / 2.0)
                plan["r_trunk"] = r_trunk
                plan["r_left"] = r_left
                plan["r_right"] = r_right
                plan["proxDiam"] = pd
                plan["distDiam"] = dd
                print(f"[LiveUpdate] Y-stent: trunk={pd:.1f}mm branch={dd:.1f}mm")
                self.logic.placeStent3D(plan)
                self._removeBalloons()
                return

            # ── Straight/Tapered/Covered etc: full slider + spinbox update ────
            plan["proxDiam"] = self.stentProxDiamSpin.value
            plan["distDiam"] = self.stentDistDiamSpin.value
            if hasattr(self, "stentTrunkSpin"):
                plan["trunk_mm"] = self.stentTrunkSpin.value

            has_y_paths = plan.get("y_trunk") is not None
            if has_y_paths and not is_bifurcated:
                print(
                    f"[LiveUpdate] Type changed to '{combo_type}' — "
                    f"clearing Y paths, rebuilding as single tube"
                )
                plan.pop("y_trunk", None)
                plan.pop("y_left", None)
                plan.pop("y_right", None)
                plan.pop("r_trunk", None)
                plan.pop("r_left", None)
                plan.pop("r_right", None)

            self.logic.placeStent3D(plan)
            self._removeBalloons()
        finally:
            self._liveUpdateActive = False

    def _refreshStentSummary(self):
        if not self.logic.points:
            return
        pd = self.stentProxDiamSpin.value
        dd = self.stentDistDiamSpin.value
        stype = self.stentTypeCombo.currentText.strip()

        is_kissing = "Kissing" in stype or "VBX" in stype
        is_y = "Bifurcated" in stype

        # ── Kissing / Y-stent: show plan-based summary, skip slider math ─
        existing = getattr(self.logic, "stentPlan", None)
        if (is_kissing or is_y) and existing:
            # Always sync diameters from spinboxes into plan
            existing["type"] = stype
            existing["proxDiam"] = pd
            existing["distDiam"] = dd

            if is_kissing:
                nom = min([5, 6, 7, 8, 9, 10, 11], key=lambda s: abs(s - pd))
                VBX_MAX = {5: 8, 6: 9, 7: 11, 8: 13, 9: 13, 10: 13, 11: 16}
                max_post = VBX_MAX.get(nom, 13)
                bi_A = existing.get("branchIdx", 1)
                bi_B = existing.get("siblingIdx", 2)
                _dA = self.kissDistASpin.value if hasattr(self, "kissDistASpin") else 0
                _dB = self.kissDistBSpin.value if hasattr(self, "kissDistBSpin") else 0
                _trk = (
                    self.kissProxExtSpin.value
                    if hasattr(self, "kissProxExtSpin")
                    else 0
                )
                _tot = round(_trk + max(_dA, _dB), 1)
                _nameA = self.logic.getBranchDisplayName(bi_A)
                _nameB = self.logic.getBranchDisplayName(bi_B)
                _warn = (
                    f"⚠️ {_tot}mm > 79mm VBX max" if _tot > 79 else f"✓ {_tot}mm ≤ 79mm"
                )
                self.stentSummaryLabel.setText(
                    f"GORE VBX Kissing  nom={nom}mm  max-post={max_post}mm\n"
                    f"Trunk: {_trk:.0f}mm  |  {_nameA}: {_dA:.0f}mm  |  {_nameB}: {_dB:.0f}mm\n"
                    f"Total: {_tot}mm / 79mm max  —  {_warn}"
                )
                self.stentWarningLabel.setText(_warn)
                self.stentWarningLabel.setStyleSheet(
                    "font-weight:bold;font-size:11px;padding:4px;border-radius:4px;"
                    + (
                        "background:#FDEDEC;color:#c0392b;"
                        if _tot > 79
                        else "background:#E8F5E9;color:#1B5E20;"
                    )
                )
            else:  # Y-stent
                r_l = existing.get("r_left", dd / 2.0)
                r_r = existing.get("r_right", dd / 2.0)
                r_t = existing.get("r_trunk", pd / 2.0)
                self.stentSummaryLabel.setText(
                    f"Y / Trouser stent\n"
                    f"Trunk: {pd:.1f}mm  |  Left: {r_l*2:.1f}mm  Right: {r_r*2:.1f}mm\n"
                    f"Use Left/Right Ø spinners + ↺ Re-place to resize branches"
                )
                self.stentWarningLabel.setText(
                    "✓ Y-stent mode — use branch Ø overrides"
                )
                self.stentWarningLabel.setStyleSheet(
                    "font-weight:bold;font-size:11px;padding:4px;border-radius:4px;"
                    "background:#E8F5E9;color:#1B5E20;"
                )
            return

        # ── Standard single-vessel stent ─────────────────────────────────
        p = self.stentProxSlider.value
        di = self.stentDistSlider.value
        stent_bi = self._currentStentBranchIndex()
        if stent_bi < 0 or stent_bi >= len(self.logic.branches):
            self.stentSummaryLabel.setText("Select a branch above first.")
            self.stentWarningLabel.setText("")
            return
        branch_start = self.logic.branches[stent_bi][0]
        gp = min(p + branch_start, len(self.logic.distances) - 1)
        gd = min(di + branch_start, len(self.logic.distances) - 1)
        length = self.logic.distances[gd] - self.logic.distances[gp]
        bs, be = self.logic.branches[stent_bi]
        meta_bi = self.logic.branchMeta.get(stent_bi, {})
        # Use anatomical ostium as stable start instead of heuristic bif_skip
        stable_s = self.logic.getBranchStartGi(stent_bi)

        def _sample_vd_healthy(center, n=8):
            if not self.logic.diameters:
                return 0
            lo = (
                max(stable_s, center - n)
                if center < stable_s + n
                else max(bs, center - n)
            )
            hi = min(be - 1, lo + 2 * n)
            raw = sorted(
                [
                    self.logic.diameters[j]
                    for j in range(lo, hi + 1)
                    if self.logic.diameters[j] > 1.0
                ]
            )
            return raw[int(len(raw) * 0.75)] if raw else 0

        vd_prox = _sample_vd_healthy(gp)
        vd_dist = _sample_vd_healthy(gd)
        warnings = []
        for lbl, sd, vd in [("Proximal", pd, vd_prox), ("Distal", dd, vd_dist)]:
            if vd > 0:
                ratio = sd / vd
                if ratio < 1.0:
                    warnings.append(f"UNDERSIZED: {lbl} {sd:.1f}mm < vessel {vd:.1f}mm")
                elif ratio > 1.35:
                    warnings.append(
                        f"OVERSIZED: {lbl} {sd:.1f}mm > vessel {vd:.1f}mm ×1.35"
                    )
        self.stentSummaryLabel.setText(
            f"Type: {stype}  |  Length: {length:.1f}mm\n"
            f"Proximal: {pd:.1f}mm stent / {vd_prox:.1f}mm vessel\n"
            f"Distal:   {dd:.1f}mm stent / {vd_dist:.1f}mm vessel"
        )
        if hasattr(self, "stentLengthSpin") and self.logic.distances:
            cur_len = abs(
                self.logic.distances[
                    min(di + branch_start, len(self.logic.distances) - 1)
                ]
                - self.logic.distances[
                    min(p + branch_start, len(self.logic.distances) - 1)
                ]
            )
            self.stentLengthSpin.blockSignals(True)
            self.stentLengthSpin.setValue(round(cur_len, 1))
            self.stentLengthSpin.blockSignals(False)
        if warnings:
            self.stentWarningLabel.setText("\n".join(warnings))
            self.stentWarningLabel.setStyleSheet(
                "font-weight:bold;font-size:12px;padding:4px;border-radius:4px;"
                "background:#FFF3CD;color:#856404;"
            )
        else:
            self.stentWarningLabel.setText("Sizing OK")
            self.stentWarningLabel.setStyleSheet(
                "font-weight:bold;font-size:12px;padding:4px;border-radius:4px;"
                "background:#D4EDDA;color:#155724;"
            )
        has_pick = existing and existing.get("custom_path") is not None
        if has_pick:
            existing["type"] = stype
            existing["proxDiam"] = pd
            existing["distDiam"] = dd
            existing["trunk_mm"] = (
                self.stentTrunkSpin.value if hasattr(self, "stentTrunkSpin") else 0.0
            )
        else:
            self.logic.stentPlan = {
                "type": stype,
                "proxPt": gp,
                "distPt": gd,
                "proxDiam": pd,
                "distDiam": dd,
                "length": round(length, 1),
                "vd_prox": round(vd_prox, 2),
                "vd_dist": round(vd_dist, 2),
                "warnings": warnings,
                "branchIdx": stent_bi,
                "trunk_mm": (
                    self.stentTrunkSpin.value
                    if hasattr(self, "stentTrunkSpin")
                    else 0.0
                ),
            }

    def onStentAutoSize(self):
        """Auto-size stent. If pancaking findings exist in this branch,
        center the landing zone around them automatically."""
        if not self.logic.points or not self.logic.diameters:
            slicer.util.errorDisplay("Load a centerline first.")
            return
        stent_bi = self._currentStentBranchIndex()
        if stent_bi < 0 or stent_bi >= len(self.logic.branches):
            slicer.util.errorDisplay("Select a target branch in Stent Planner first.")
            return
        # Suppress _liveUpdateStent while we change sliders/spinboxes
        self._autoPlacing = True
        branch_start, branch_end = self.logic.branches[stent_bi]
        branch_len = branch_end - branch_start

        # ── Try to find pancaking/compression zone in this branch ────────
        findings = getattr(self.logic, "findings", [])
        branch_findings = [
            f
            for f in findings
            if f["branchIdx"] == stent_bi
            and f["type"] in ("Pancaking", "Mild Compression", "Eccentric Compression")
        ]

        if branch_findings:
            # Sort: Pancaking > Eccentric Compression > Mild Compression, then worst diameter
            _ftype_prio = {
                "Pancaking": 3,
                "Eccentric Compression": 2,
                "Mild Compression": 1,
            }
            branch_findings.sort(
                key=lambda f: (-_ftype_prio.get(f["type"], 0), f["value"])
            )
            worst = branch_findings[0]
            worst_global = worst["pointIdx"]
            worst_local = worst_global - branch_start

            # Landing: 20mm each side of lesion, converted to point count
            import math

            PAD_MM = 20.0
            # Sample spacing from the healthy mid-section of the branch, not near lesion
            mid_s = branch_start + branch_len // 3
            mid_e = min(
                branch_start + 2 * branch_len // 3, branch_start + branch_len - 2
            )
            spacings = []
            for pi in range(mid_s, min(mid_e, len(self.logic.points) - 1)):
                p1 = self.logic.points[pi]
                p2 = self.logic.points[pi + 1]
                sp = math.sqrt(sum((p2[k] - p1[k]) ** 2 for k in range(3)))
                if sp > 0:
                    spacings.append(sp)
            avg_sp = sum(spacings) / len(spacings) if spacings else 1.3
            PAD = max(15, int(PAD_MM / avg_sp))  # minimum 15 pts each side

            prox_local = max(0, worst_local - PAD)
            dist_local = min(branch_len - 1, worst_local + PAD)
            # Only enforce proximal margin if lesion is NOT near branch start
            # (if lesion is at local_pt < 10, the stent must reach it — don't push away)
            if prox_local < 5 and worst_local >= 8:
                extra_dist = 5 - prox_local
                prox_local = 5
                dist_local = min(branch_len - 1, dist_local + extra_dist)

            # If lesion is near branch start, extend distally to reach minimum stent length
            # If lesion is near branch end, extend proximally
            min_pts = max(PAD * 2, int(40.0 / avg_sp))  # at least 40mm
            actual_span = dist_local - prox_local
            if actual_span < min_pts:
                extra = min_pts - actual_span
                # Try to extend distally first
                dist_local = min(branch_len - 1, dist_local + extra)
                # If still short, extend proximally too
                actual_span = dist_local - prox_local
                if actual_span < min_pts:
                    prox_local = max(0, prox_local - (min_pts - actual_span))

            stent_mm = round((dist_local - prox_local) * avg_sp, 1)

            self.stentProxSlider.setValue(prox_local)
            self.stentDistSlider.setValue(dist_local)
            self.stentProxLabel.setText("Pt " + str(prox_local))
            self.stentDistLabel.setText("Pt " + str(dist_local))
            print(
                "[VesselAnalyzer] Stent on "
                + worst["type"]
                + " local="
                + str(worst_local)
                + " global="
                + str(worst_global)
                + " PAD="
                + str(PAD)
                + "pts  ~"
                + str(stent_mm)
                + "mm total"
            )
        else:
            print(
                "[VesselAnalyzer] No findings in branch "
                + str(stent_bi + 1)
                + " — using current slider positions"
            )

        # ── Size diameters from vessel at landing points ──────────────────
        # Use same healthy-window sampling as _refreshStentSummary
        meta_bi = self.logic.branchMeta.get(stent_bi, {})
        # Use anatomical ostium as stable start instead of heuristic bif_skip
        stable_s = self.logic.getBranchStartGi(stent_bi)

        def _auto_vd(center, n=8):
            if not self.logic.diameters:
                return 0
            lo = (
                max(stable_s, center - n)
                if center < stable_s + n
                else max(branch_start, center - n)
            )
            hi = min(branch_end - 1, lo + 2 * n)
            raw = sorted(
                [
                    self.logic.diameters[j]
                    for j in range(lo, hi + 1)
                    if self.logic.diameters[j] > 1.0
                ]
            )
            return raw[int(len(raw) * 0.75)] if raw else 0

        gp = min(
            self.stentProxSlider.value + branch_start, len(self.logic.diameters) - 1
        )
        gd = min(
            self.stentDistSlider.value + branch_start, len(self.logic.diameters) - 1
        )
        vd_p = _auto_vd(gp)
        vd_d = _auto_vd(gd)

        # ── Pre-dilation override ─────────────────────────────────────────
        pdmap = getattr(self.logic, "preDilationMap", {})
        if pdmap and branch_findings:
            worst_gi = branch_findings[0]["pointIdx"]
            # Find closest pre-dilation entry
            closest = min(pdmap.keys(), key=lambda g: abs(g - worst_gi))
            if abs(closest - worst_gi) < 30:
                pd_entry = pdmap[closest]
                expanded_d = pd_entry["healthy_diam"]
                # Stent sized to healthy diameter (NOT compressed)
                # 10% oversize standard for bare-metal stent
                stent_d = round(expanded_d * 1.10 * 2) / 2
                self.stentProxDiamSpin.blockSignals(True)
                self.stentDistDiamSpin.blockSignals(True)
                self.stentProxDiamSpin.setValue(stent_d)
                self.stentDistDiamSpin.setValue(stent_d)
                self.stentProxDiamSpin.blockSignals(False)
                self.stentDistDiamSpin.blockSignals(False)
                self._refreshStentSummary()
                self._autoPlacing = False
                return  # skip normal sizing below
        # ── Normal sizing (no pre-dilation) ──────────────────────────────
        stype_now = self.stentTypeCombo.currentText.strip()

        def _snap_venous(vd):
            """Snap to nearest GORE Viabahn Venous size."""
            sizes = [10, 12, 14, 16, 18, 20, 24, 28]
            target = vd * 1.1
            return min(sizes, key=lambda s: abs(s - target))

        def _snap_vbx(vd):
            """Snap to largest VBX nominal that fits vessel."""
            sizes = [5, 6, 7, 8, 9, 10, 11]
            best = sizes[0]
            for s in sizes:
                if s <= vd:
                    best = s
            return float(best)

        if "VBX" in stype_now:
            if vd_p > 0:
                self.stentProxDiamSpin.setValue(_snap_vbx(vd_p))
            if vd_d > 0:
                self.stentDistDiamSpin.setValue(_snap_vbx(vd_d))
        elif "Venous" in stype_now:
            if vd_p > 0:
                self.stentProxDiamSpin.setValue(float(_snap_venous(vd_p)))
            if vd_d > 0:
                self.stentDistDiamSpin.setValue(float(_snap_venous(vd_d)))
        else:
            if vd_p > 0:
                self.stentProxDiamSpin.setValue(round(vd_p * 1.15 * 2) / 2)
            if vd_d > 0:
                self.stentDistDiamSpin.setValue(round(vd_d * 1.15 * 2) / 2)
        self._refreshStentSummary()
        self._autoPlacing = False

    def onAutoPlaceAllStents(self):
        """Find all lesions by clinical priority, choose optimal stent for each,
        navigate to each finding, and place stents sequentially — fully automatic."""
        import math

        if not self.logic.points or not self.logic.branches:
            self.detectFindingsButton.setText("⚠️ Load centerline first")
            return
        findings = getattr(self.logic, "findings", [])
        if not findings:

            return

        branches = self.logic.branches
        diams = self.logic.diameters
        pts = self.logic.points

        def healthy_diam(gi, branch_start, branch_end, window=15):
            """Reference diameter sampled outside the lesion, capped at branch stable median."""
            # Branch stable median (middle 20-80% of branch)
            st_s = branch_start + max(5, (branch_end - branch_start) // 5)
            st_e = branch_end - max(3, (branch_end - branch_start) // 10)
            stable = sorted(
                [
                    diams[i]
                    for i in range(st_s, st_e)
                    if i < len(diams) and diams[i] > 1.0
                ]
            )
            branch_median = stable[len(stable) // 2] if stable else 0.0

            samples = []
            for delta in list(range(-window * 3, -window)) + list(
                range(window, window * 3)
            ):
                idx = gi + delta
                if (
                    branch_start <= idx < branch_end
                    and idx < len(diams)
                    and diams[idx] > 1.0
                ):
                    samples.append(diams[idx])
            if samples:
                samples.sort()
                ref = samples[int(len(samples) * 0.75)]
            else:
                vals = sorted(
                    [
                        diams[i]
                        for i in range(branch_start, branch_end)
                        if i < len(diams) and diams[i] > 1.0
                    ]
                )
                ref = vals[int(len(vals) * 0.75)] if vals else 8.0
            # Cap at branch stable median × 1.1
            if branch_median > 0:
                ref = min(ref, branch_median * 1.1)
            return ref

        def seg_length(gi_start, gi_end):
            return sum(
                math.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
                for i in range(gi_start, min(gi_end, len(pts) - 1))
            )

        # Junction set: bifurcation proximity ±20pts from every branch start
        junction_set = set()
        for bsj, _ in branches:
            if bsj > 0:
                for delta in range(-20, 21):
                    if bsj + delta >= 0:
                        junction_set.add(bsj + delta)

        # Sort by clinical priority then severity; deduplicate within 30pts
        priority_map = {
            "Pancaking": 4,
            "Eccentric Compression": 4,
            "Aneurysm": 3,
            "Ectasia": 2,
            "Mild Compression": 1,
        }
        sorted_f = sorted(
            findings,
            key=lambda f: (priority_map.get(f["type"], 0), f.get("ratio", 1.0)),
            reverse=True,
        )
        processed_gis, unique_findings = [], []
        for f in sorted_f:
            gi = f.get("pointIdx", -1)
            if gi < 0:
                continue
            if all(abs(gi - pg) > 30 for pg in processed_gis):
                unique_findings.append(f)
                processed_gis.append(gi)

        if not unique_findings:

            return

        # Block ALL live-update signals for entire batch to prevent cascading re-renders
        self._autoPlacing = True
        widgets_to_block = [
            self.stentProxSlider,
            self.stentDistSlider,
            self.stentTypeCombo,
            self.stentBranchCombo,
            self.stentProxDiamSpin,
            self.stentDistDiamSpin,
        ]
        for w in widgets_to_block:
            try:
                w.blockSignals(True)
            except Exception:
                pass

        stent_count, summary_lines = 0, []

        try:
            for f_idx, f in enumerate(unique_findings):
                f_type = f.get("type", "")
                f_bi = f.get("branchIdx", 0)
                f_gi = f.get("pointIdx", 0)
                f_diam = f.get("value", 0.0)
                f_ratio = f.get("ratio", 1.0)
                f_at_bif = f_gi in junction_set
                bs, be = branches[f_bi]

                # Navigate IVUS to finding (suppress ring — model is opaque during planning)
                try:
                    self.pointSlider.setValue(self.logic.globalToTraversal(f_gi))
                    self.logic.updateVisualizations(
                        f_gi,
                        showSphere=True,
                        showRing=False,
                        showLine=False,
                        sphereColor=self._getSphereColor(f_gi),
                    )
                except Exception:
                    pass

                # Select branch (signals already blocked)
                try:
                    self._selectStentBranchByIndex(f_bi)
                except Exception:
                    pass

                # Determine stent type + landing zone
                n_downstream = len(branches) - 1
                if f_type == "Pancaking" and f_at_bif:
                    stent_type, stent_config = "Kissing", "Kissing"
                    # Proximal landing: walk back from hub max 30mm — never past trunk_bs0
                    trunk_bs0, trunk_be0 = branches[0]
                    hub0 = trunk_be0 - 1
                    # Renal vein floor: lowest start-Z of branches 3 and 4 (renal veins)
                    _renal_z = min(
                        pts[branches[bi_rv][0]][2]
                        for bi_rv in [3, 4]
                        if bi_rv < len(branches)
                    )
                    acc0 = 0.0
                    prox_gi = hub0
                    for _ti in range(hub0 - 1, trunk_bs0 - 1, -1):
                        if _ti + 1 < len(pts):
                            _p0, _p1 = pts[_ti], pts[_ti + 1]
                            acc0 += (
                                (_p1[0] - _p0[0]) ** 2
                                + (_p1[1] - _p0[1]) ** 2
                                + (_p1[2] - _p0[2]) ** 2
                            ) ** 0.5
                        if pts[_ti][2] >= _renal_z:
                            break  # stop before renal vein level
                        prox_gi = _ti
                        if acc0 >= 20.0:
                            break
                    dist_gi = min(be - 1, f_gi + 40)
                elif f_type == "Pancaking" and f_ratio > 2.5:
                    stent_type, stent_config = "Covered Straight", "Covered"
                    prox_gi = max(bs, f_gi - 20)
                    dist_gi = min(be - 1, f_gi + 20)
                elif f_type == "Pancaking":
                    stent_type = "Kissing" if n_downstream >= 2 else "Covered Straight"
                    stent_config = "Kissing" if n_downstream >= 2 else "Covered"
                    prox_gi = max(bs, f_gi - 15)
                    dist_gi = min(be - 1, f_gi + 30)
                elif f_type == "Ectasia":
                    # Ectasia = dilation — kissing is contraindicated.
                    # Use large self-expanding Venous stent (Viabahn Venous) or skip.
                    stent_type, stent_config = "Venous Straight", "Straight"
                    prox_gi = max(bs, f_gi - 30)
                    dist_gi = min(be - 1, f_gi + 30)
                elif f_type == "Aneurysm":
                    # Aneurysm → covered stent graft to exclude sac
                    stent_type, stent_config = "Covered Straight", "Covered"
                    prox_gi = max(bs, f_gi - 25)
                    dist_gi = min(be - 1, f_gi + 25)
                else:  # Mild Compression
                    # Skip if within 15pts of branch tip — stenting near tip
                    # is clinically unreliable and visually confusing
                    if (be - 1 - f_gi) < 15:
                        summary_lines.append(
                            f"⚠️ Skipped Mild Compression pt{f_gi} — near branch tip"
                        )
                        continue
                    stent_type = "Straight" if f_diam >= 6.0 else "DES Sirolimus"
                    stent_config = "Straight"
                    prox_gi = max(bs, f_gi - 20)
                    dist_gi = min(be - 1, f_gi + 20)

                # Set stent type combo (signals already blocked)
                type_to_idx = {
                    "Straight": 1,
                    "Tapered": 2,
                    "Bifurcated": 3,
                    "VBX Single": 5,
                    "VBX Kissing": 6,
                    "Kissing": 6,
                    "Venous Straight": 8,
                    "Venous Tapered": 9,
                    "Covered Straight": 11,
                    "Covered Bifurcated (EVAR)": 12,
                    "DES Sirolimus": 14,
                    "DES Paclitaxel": 15,
                    "DES Zotarolimus": 16,
                }
                try:
                    self.stentTypeCombo.setCurrentIndex(type_to_idx.get(stent_type, 0))
                except Exception:
                    pass

                # Healthy reference diameters — NOT from the lesion point itself
                prox_d = healthy_diam(prox_gi, bs, be)
                dist_d = healthy_diam(dist_gi, bs, be)
                prox_r = round(prox_d * 1.15 / 0.5) * 0.5
                dist_r = round(dist_d * 1.15 / 0.5) * 0.5
                seg_len = seg_length(prox_gi, dist_gi)

                # Flash marker in 3D
                try:
                    fname = f"AutoFind_{f_idx+1}_{f_type.replace(' ', '_')}"
                    flash = slicer.mrmlScene.AddNewNodeByClass(
                        "vtkMRMLMarkupsFiducialNode", fname
                    )
                    flash.AddControlPoint(pts[f_gi][0], pts[f_gi][1], pts[f_gi][2])
                    flash.SetNthControlPointLabel(0, f"⚠ {f_type}")
                    fdn = flash.GetDisplayNode()
                    fdn.SetSelectedColor(1.0, 0.2, 0.0)
                    fdn.SetGlyphScale(4.5)
                    fdn.SetTextScale(4.0)
                except Exception as e:
                    pass

                # Place stent directly via logic — no UI slider involvement
                try:
                    plan = {
                        "type": stent_config,
                        "branchIdx": f_bi,
                        "proxPt": prox_gi,
                        "distPt": dist_gi,
                        "proxDiam": prox_r,
                        "distDiam": dist_r,
                        "length": round(seg_len, 1),
                        "vd_prox": round(prox_d, 2),
                        "vd_dist": round(dist_d, 2),
                        "warnings": [],
                        "trunk_mm": 0.0,
                        "custom_path": list(range(prox_gi, dist_gi + 1)),
                    }
                    self.logic.placeStent3D(plan)
                    self._removeBalloons()
                    stent_count += 1
                    line = f"✅ {stent_count}: {stent_type} | Br{f_bi} pt{prox_gi}→{dist_gi} | Ø{prox_r:.1f}→Ø{dist_r:.1f}mm | {seg_len:.0f}mm"
                    summary_lines.append(line)
                except Exception as e:
                    summary_lines.append(f"❌ Stent {f_idx+1}: {e}")
        finally:
            # Always restore signals and clear flag even on exception
            self._autoPlacing = False
            for w in widgets_to_block:
                try:
                    w.blockSignals(False)
                except Exception:
                    pass

        summary = f"🎯 Auto-placed {stent_count}/{len(unique_findings)} stents:\n"
        summary += "\n".join(summary_lines)
        self.stentPickStatusLabel.setText(summary)
        self.stentPickStatusLabel.setStyleSheet(
            "font-size: 11px; color: #1a5276; padding: 4px; "
            "background: #eaf4fb; border-radius: 4px;"
        )
        # Disable Pre-Dilate after auto-place — stents already deployed,
        # pre-dilation no longer relevant
        self.preDilateButton.setEnabled(False)
        self.preDilateButton.setToolTip(
            "Stents already placed. Remove stents first to re-use pre-dilation."
        )
        self.preDilateStatusLabel.setText(
            "✅ " + str(stent_count) + " stent(s) placed. Pre-dilation complete."
        )

    def onAutoPlaceKissing(self):
        """Auto-place kissing stent.
        Proximal : IN Branch 0 (IVC trunk), between trunk_bs and hub_gi.
                   Walks back from hub by kissProxExtSpin mm (default 20mm).
        Distal   : IN Branch 1 / Branch 2 (iliacs), BELOW the hub.
                   Lands past the balloon distal edge (preDilationMap),
                   or past the deepest lesion + 15mm, whichever is available.
                   Both limbs equalised to the longer one so lengths match.
        Hard cap : trunk_mm + branch_mm <= 79mm per VBX device.
        Spinboxes and stent planner are set from the same numbers used to
        render the 3D stent."""
        import math as _m

        if not self.logic.points or not self.logic.branches:
            self.stentPickStatusLabel.setText("\u26a0\ufe0f Load centerline first")
            return

        pts = self.logic.points
        branches = self.logic.branches
        findings = getattr(self.logic, "findings", [])
        pdmap = getattr(self.logic, "preDilationMap", {})

        hub_gi = branches[0][1] - 1
        trunk_bs = branches[0][0]
        hub_pt = pts[hub_gi]

        # Pick the two downstream branches whose start point is spatially
        # closest to the hub — these are the iliacs. Avoids picking renal
        # veins or short collaterals that happen to be long by index count.
        # Also require minimum length of 80mm to exclude short side branches.
        candidates = []
        for bi, (bs, be) in enumerate(branches):
            if bi == 0:
                continue
            branch_mm = sum(
                math.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
                for i in range(bs, min(be - 1, len(pts) - 1))
                if i + 1 < len(pts)
            )
            if branch_mm < 80.0:
                continue
            dist_to_hub = math.sqrt(
                sum((pts[bs][k] - hub_pt[k]) ** 2 for k in range(3))
            )
            candidates.append((dist_to_hub, bi, branch_mm))
        candidates.sort()  # closest to hub first
        if len(candidates) < 2:
            self.stentPickStatusLabel.setText(
                "\u26a0\ufe0f Need at least 2 iliac branches (>80mm)"
            )
            return
        bi_A = candidates[0][1]
        bi_B = candidates[1][1]

        def walk_mm(bi, target_mm):
            bs, be = branches[bi]
            acc = 0.0
            gi = bs
            for i in range(bs, min(be - 1, len(pts) - 1)):
                if i + 1 < len(pts):
                    step = math.sqrt(
                        sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3))
                    )
                    if acc + step > target_mm:
                        break
                    acc += step
                gi = i + 1
            return min(gi, be - 1), acc

        def measure_mm(bi, gi_end):
            bs = self.logic.getBranchStartGi(bi)  # anatomical start (ostium)
            return sum(
                math.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
                for i in range(bs, min(gi_end, len(pts) - 1))
                if i + 1 < len(pts)
            )

        # PROXIMAL: walk back from hub into Branch 0 (IVC trunk)
        # PROXIMAL: land AT the bifurcation hub (gi105), giving 0mm trunk.
        # The two iliac limbs start right at the carina — standard kissing technique.
        # kissProxExtSpin adds optional IVC coverage above the hub (default 10mm).
        trunk_target = (
            self.kissProxExtSpin.value if hasattr(self, "kissProxExtSpin") else 10.0
        )
        trunk_target = max(0.0, trunk_target)
        prox_gi = hub_gi
        acc_prox = 0.0
        if trunk_target > 0.0:
            for ti in range(hub_gi - 1, trunk_bs - 1, -1):
                if ti + 1 < len(pts):
                    p0, p1 = pts[ti], pts[ti + 1]
                    step = math.sqrt(sum((p1[k] - p0[k]) ** 2 for k in range(3)))
                    if acc_prox + step > trunk_target:
                        break
                    acc_prox += step
                prox_gi = ti
        prox_gi = max(trunk_bs, min(prox_gi, hub_gi))
        acc_prox = measure_mm(0, hub_gi) - measure_mm(0, prox_gi)
        print(f"[AutoKiss] PROXIMAL: gi{prox_gi}->gi{hub_gi}  trunk={acc_prox:.1f}mm")

        # DISTAL budget — full 79mm minus actual trunk used
        VBX_MAX = 79.0
        budget = max(10.0, VBX_MAX - acc_prox)

        def distal_target_mm(bi):
            bs, be = branches[bi]
            balloon_gis = [
                gj
                for gj, e in pdmap.items()
                if isinstance(e, dict) and e.get("branch") == bi and bs <= gj < be
            ]
            if balloon_gis:
                d_mm = measure_mm(bi, max(balloon_gis)) + 5.0
                print(f"[AutoKiss] bi{bi}: balloon->{d_mm:.0f}mm")
                return d_mm
            # Pancaking / Eccentric Compression first (most severe), then Mild Compression
            for ftype in ("Pancaking", "Eccentric Compression", "Mild Compression"):
                f_list = [
                    f
                    for f in findings
                    if f.get("branchIdx") == bi and f.get("type") == ftype
                ]
                if f_list:
                    worst_gi = max(f_list, key=lambda f: f.get("ratio", 1.0))[
                        "pointIdx"
                    ]
                    d_mm = measure_mm(bi, worst_gi) + 15.0
                    print(f"[AutoKiss] bi{bi}: {ftype} gi{worst_gi}->{d_mm:.0f}mm")
                    return d_mm
            spin = getattr(
                self, "kissDistASpin" if bi == bi_A else "kissDistBSpin", None
            )
            if spin is not None:
                return float(spin.value)
            return 40.0

        want_A = distal_target_mm(bi_A)
        want_B = distal_target_mm(bi_B)

        # Extension needed when the lesion target exceeds the single VBX budget
        needs_extension_A = want_A > budget
        needs_extension_B = want_B > budget
        needs_extension = needs_extension_A or needs_extension_B
        extension_gap = round(
            max(
                want_A - budget if needs_extension_A else 0.0,
                want_B - budget if needs_extension_B else 0.0,
            ),
            1,
        )

        # Walk each limb to its OWN lesion-driven target — uncapped.
        # _placeKissingStent3D enforces the 79mm hard limit per limb internally.
        # Passing the true lesion gi lets the placer reach the pancaking point
        # and correctly report NOT COVERED vs COVERED.
        dist_gi_A, mm_A = walk_mm(bi_A, want_A)
        dist_gi_B, mm_B = walk_mm(bi_B, want_B)
        total_mm = round(acc_prox + max(mm_A, mm_B), 1)

        print(
            f"[AutoKiss] DISTAL: budget={budget:.0f}mm "
            f"wantA={want_A:.0f}mm wantB={want_B:.0f}mm "
            f"A=gi{dist_gi_A}({mm_A:.0f}mm) B=gi{dist_gi_B}({mm_B:.0f}mm) total={total_mm}mm"
            + (
                f"  *** LESION BEYOND VBX REACH — add distal stent: {extension_gap:.0f}mm gap ***"
                if needs_extension
                else ""
            )
        )

        # Set stent type combo to VBX Kissing index 6
        self._autoPlacing = True
        self.stentTypeCombo.blockSignals(True)
        self.stentTypeCombo.setCurrentIndex(6)
        self.stentTypeCombo.blockSignals(False)
        self._autoPlacing = False

        nom = float(min(11, max(5, round(self.stentProxDiamSpin.value))))
        kiss_plan = {
            "type": "  VBX \u00b7 Kissing (Parallel) \u2014 7\u201310mm + POT",
            "branchIdx": bi_A,
            "siblingIdx": bi_B,
            "proxPt": prox_gi,
            "distPt": dist_gi_A,
            "distPt_B": dist_gi_B,
            "proxDiam": nom,
            "distDiam": nom,
            "warnings": [],
            "trunk_mm": 0.0,
        }
        self.logic.stentPlan = kiss_plan
        placed = self.logic._placeKissingStent3D(kiss_plan)

        if placed:
            # FIX: write the snapped VBX parameters back into stentPlan so the
            # planner display is always consistent with the 3D placed stent.
            VBX_MAX_POST = {5: 8, 6: 9, 7: 11, 8: 13, 9: 13, 10: 13, 11: 16}
            VBX_LENGTHS_OUTER = [15, 19, 29, 39, 59, 79]
            nom_i = int(round(nom))
            snapped_len = min(
                VBX_LENGTHS_OUTER, key=lambda l: abs(l - (acc_prox + max(mm_A, mm_B)))
            )
            max_post_val = VBX_MAX_POST.get(nom_i, 13)
            kiss_plan.update(
                {
                    "proxDiam": float(nom_i),
                    "distDiam": float(nom_i),
                    "vbx_nom": nom_i,
                    "vbx_len": snapped_len,
                    "max_post": max_post_val,
                    "trunk_mm": round(acc_prox, 1),
                    "branchA_mm": round(mm_A, 1),
                    "branchB_mm": round(mm_B, 1),
                    "total_mm": round(total_mm, 1),
                }
            )
            self.logic.stentPlan = kiss_plan

        if not placed:
            self.stentPickStatusLabel.setText(
                "\u26a0\ufe0f Auto kissing placement failed"
            )
            self.stentPickStatusLabel.setStyleSheet(
                "font-size:11px;color:#c0392b;padding:4px;background:#fdedec;border-radius:4px;"
            )
            return

        # Sync spinboxes from the same numbers used to render
        self._autoPlacing = True
        for w in [
            w
            for w in [
                getattr(self, "kissProxExtSpin", None),
                getattr(self, "kissDistASpin", None),
                getattr(self, "kissDistBSpin", None),
                self.stentProxDiamSpin,
                self.stentDistDiamSpin,
            ]
            if w is not None
        ]:
            w.blockSignals(True)
        if hasattr(self, "kissProxExtSpin"):
            self.kissProxExtSpin.setValue(round(acc_prox))
        if hasattr(self, "kissDistASpin"):
            self.kissDistASpin.setValue(round(mm_A))
        if hasattr(self, "kissDistBSpin"):
            self.kissDistBSpin.setValue(round(mm_B))
        self.stentProxDiamSpin.setValue(nom)
        self.stentDistDiamSpin.setValue(nom)
        for w in [
            w
            for w in [
                getattr(self, "kissProxExtSpin", None),
                getattr(self, "kissDistASpin", None),
                getattr(self, "kissDistBSpin", None),
                self.stentProxDiamSpin,
                self.stentDistDiamSpin,
            ]
            if w is not None
        ]:
            w.blockSignals(False)
        self._autoPlacing = False

        nom_i = int(nom)
        max_post = {5: 8, 6: 9, 7: 11, 8: 13, 9: 13, 10: 13, 11: 16}.get(nom_i, 13)

        if needs_extension:
            _ext_branch = "B" if (want_B - budget) >= (want_A - budget) else "A"
            _ext_want = want_B if _ext_branch == "B" else want_A
            warn_txt = (
                f"\u26a0\ufe0f Lesion at {_ext_want:.0f}mm in Iliac {_ext_branch} "
                f"\u2014 beyond single VBX reach (budget={budget:.0f}mm)\n"
                f"Kissing stent placed to budget — {extension_gap:.0f}mm short of pancaking lesion\n"
                f"\u2192 Add straight stent on Iliac {_ext_branch} distally to cover {extension_gap:.0f}mm gap"
            )
            self.stentWarningLabel.setStyleSheet(
                "font-weight:bold;font-size:11px;padding:4px;border-radius:4px;"
                "background:#FDEDEC;color:#c0392b;"
            )
        elif total_mm > VBX_MAX:
            warn_txt = f"\u26a0\ufe0f {total_mm}mm exceeds 79mm VBX max"
            self.stentWarningLabel.setStyleSheet(
                "font-weight:bold;font-size:11px;padding:4px;border-radius:4px;"
                "background:#FFF3CD;color:#856404;"
            )
        else:
            warn_txt = f"\u2713 {total_mm}mm \u2264 79mm  |  pancaking lesion covered"
            self.stentWarningLabel.setStyleSheet(
                "font-weight:bold;font-size:11px;padding:4px;border-radius:4px;"
                "background:#E8F5E9;color:#1B5E20;"
            )
        self.stentWarningLabel.setText(warn_txt)

        coverage_note = (
            f"\u26a0\ufe0f {extension_gap:.0f}mm short \u2192 add distal straight stent"
            if needs_extension
            else "balloon covered \u2713"
        )
        # FIX: read snapped values from stentPlan to guarantee UI == placed stent
        _sp = self.logic.stentPlan
        _nom_disp = int(_sp.get("vbx_nom", nom_i))
        _mp_disp = _sp.get("max_post", max_post)
        _len_disp = _sp.get("vbx_len", snapped_len)
        _trunk_disp = _sp.get("trunk_mm", acc_prox)
        _brA_disp = _sp.get("branchA_mm", mm_A)
        _brB_disp = _sp.get("branchB_mm", mm_B)
        _total_disp = _sp.get("total_mm", total_mm)
        self.stentSummaryLabel.setText(
            f"GORE VBX Kissing  nom={_nom_disp}mm×{_len_disp}mm  max-post={_mp_disp}mm\n"
            f"Trunk (IVC): {_trunk_disp:.0f}mm  |  "
            f"Iliac A: {_brA_disp:.0f}mm  |  Iliac B: {_brB_disp:.0f}mm\n"
            f"Total: {_total_disp}mm / 79mm max  \u2014  {coverage_note}"
        )
        self.stentPickStatusLabel.setText(
            f"{'\u26a0\ufe0f' if needs_extension else '\u2705'} Kissing placed \u2014 "
            f"trunk {_trunk_disp:.0f}mm + A {_brA_disp:.0f}mm / B {_brB_disp:.0f}mm = {_total_disp}mm"
            + (
                f"  |  add distal stent: {extension_gap:.0f}mm gap"
                if needs_extension
                else ""
            )
        )
        self.stentPickStatusLabel.setStyleSheet(
            "font-size:11px;padding:4px;border-radius:4px;"
            + (
                "color:#c0392b;background:#fdedec;"
                if needs_extension
                else "color:#1a5276;background:#eaf4fb;"
            )
        )

    def onAutoDetectStentType(self):
        """Analyze vessel geometry and auto-detect + configure the optimal stent type."""
        if not self.logic.points or not self.logic.branches:
            self.stentAutoTypeLabel.setText("⚠️ Load a centerline first.")
            return

        # Auto-run _detectFindings if not done yet so decision tree has data
        findings = getattr(self.logic, "findings", [])
        if not findings and self.logic.diameters:
            try:
                # Never re-run while balloon is active — inflated diameters cause false ectasia
                balloon_active = bool(getattr(self.logic, "preDilationMap", {}))
                if (
                    not getattr(self.logic, "_findingsFrozen", False)
                    and not balloon_active
                ):
                    self.logic._detectFindings()
                    self.logic.applyColorOverlay()
                findings = getattr(self.logic, "findings", [])
            except Exception as e:
                print(f"[AutoType] detectFindings pre-run failed: {e}")

        branches = self.logic.branches
        diams = self.logic.diameters
        findings = getattr(self.logic, "findings", [])
        bi = self._currentStentBranchIndex()  # -1 = no branch selected

        # ── per-branch metrics ──────────────────────────────────────────────
        def branch_diams(b):
            bs, be = branches[b]
            return [diams[i] for i in range(bs, be) if i < len(diams) and diams[i] > 0]

        def branch_avg(b):
            d = branch_diams(b)
            return sum(d) / len(d) if d else 0.0

        # Count total branches
        n_branches = len(branches)
        n_downstream = n_branches - 1  # branches beyond trunk

        # Trunk = branch 0 (longest segment)
        trunk_avg = branch_avg(0)
        trunk_max = max(branch_diams(0)) if branch_diams(0) else 0.0

        # Aneurysm threshold: relative to trunk median (not a fixed aortic value)
        trunk_diams_sorted = sorted(branch_diams(0))
        trunk_median = (
            trunk_diams_sorted[len(trunk_diams_sorted) // 2]
            if trunk_diams_sorted
            else 0.0
        )

        # Selected branch metrics (or branch 1 default)
        sel = bi if bi >= 0 else 1
        sel = min(sel, n_branches - 1)
        sel_diams = branch_diams(sel)
        sel_avg = branch_avg(sel)
        sel_prox_d = sel_diams[0] if sel_diams else 0.0
        sel_dist_d = sel_diams[-1] if sel_diams else 0.0
        taper_ratio = sel_prox_d / sel_dist_d if sel_dist_d > 0 else 1.0

        # ── findings analysis ───────────────────────────────────────────────
        pancaking_findings = [
            f
            for f in findings
            if f.get("type") in ("Pancaking", "Eccentric Compression")
        ]
        compression_findings = [
            f for f in findings if f.get("type") == "Mild Compression"
        ]
        has_pancaking = len(pancaking_findings) > 0
        has_compression = len(compression_findings) > 0
        has_aneurysm = trunk_median > 0 and trunk_max > trunk_median * 1.5

        worst_finding = None
        all_findings_sorted = sorted(
            pancaking_findings + compression_findings,
            key=lambda f: f.get("ratio", 1.0),
            reverse=True,
        )
        if all_findings_sorted:
            worst_finding = all_findings_sorted[0]

        finding_branch = worst_finding.get("branchIdx", -1) if worst_finding else -1
        finding_gi = worst_finding.get("pointIdx", -1) if worst_finding else -1
        finding_diam = worst_finding.get("value", 0.0) if worst_finding else 0.0
        finding_ratio = worst_finding.get("ratio", 1.0) if worst_finding else 1.0
        finding_type = worst_finding.get("type", "") if worst_finding else ""

        # ── Bifurcation proximity test ───────────────────────────────────────
        # A finding is AT the bifurcation only if it is within KISS_ZONE_MM of
        # the branch start (hub). Lesions deeper in the iliac are NOT bifurcation
        # lesions and should be treated with a straight stent, not kissing.
        KISS_ZONE_MM = 60.0  # mm from hub within which compression → kissing

        def mm_from_branch_start(bi, gi):
            """Measure arc-length from anatomical branch start (ostium) to gi."""
            if bi < 0 or bi >= len(branches):
                return float("inf")
            bs = self.logic.getBranchStartGi(bi)
            return sum(
                math.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
                for i in range(bs, min(gi, len(pts) - 1))
                if i + 1 < len(pts)
            )

        # Kissing criteria (all three must hold):
        # 1. Finding is within KISS_ZONE_MM of its branch start (near bifurcation)
        # 2. Both iliac branches are functionally important (length > 80mm each)
        # 3. No ectasia / aneurysm (dilation contraindications)
        iliac_branches = [
            bi
            for bi, (bs, be) in enumerate(branches)
            if bi > 0
            and sum(
                math.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
                for i in range(bs, min(be - 1, len(pts) - 1))
                if i + 1 < len(pts)
            )
            >= 80.0
        ]
        has_two_iliacs = len(iliac_branches) >= 2
        has_dilation = any(f.get("type") in ("Ectasia", "Aneurysm") for f in findings)

        # Check how many iliac branches have compression near the bifurcation
        near_bif_findings = [
            f
            for f in (pancaking_findings + compression_findings)
            if f.get("branchIdx", 0) > 0
            and mm_from_branch_start(f["branchIdx"], f["pointIdx"]) <= KISS_ZONE_MM
        ]
        n_affected_iliacs_near_bif = len(set(f["branchIdx"] for f in near_bif_findings))

        # Is the worst finding near the bifurcation?
        finding_mm_from_bs = (
            mm_from_branch_start(finding_branch, finding_gi)
            if finding_gi >= 0
            else float("inf")
        )
        finding_near_bif = finding_mm_from_bs <= KISS_ZONE_MM

        # Use kissing if:
        #   - Finding is within 60mm of bifurcation AND both branches important, OR
        #   - Multiple branches affected near the bifurcation
        use_kissing = (
            has_two_iliacs
            and not has_dilation
            and (finding_near_bif or n_affected_iliacs_near_bif >= 2)
        )

        # ── decision tree ───────────────────────────────────────────────────
        stent_type = "Straight"
        reasons = []
        config_tips = []
        nav_to_gi = -1

        if worst_finding is not None:
            nav_to_gi = finding_gi

            if has_dilation:
                stent_type = "Venous Straight"
                reasons.append(f"dilation detected — kissing contraindicated")
                reasons.append("use large self-expanding Venous stent")
                config_tips.append("GORE Viabahn Venous (10–28mm), single vessel")

            elif (
                finding_type
                in ("Pancaking", "Eccentric Compression", "Mild Compression")
                and use_kissing
            ):
                stent_type = "Kissing"
                reasons.append(
                    f"compression {finding_mm_from_bs:.0f}mm from bifurcation "
                    f"(≤{KISS_ZONE_MM:.0f}mm zone) on Br{finding_branch}"
                )
                reasons.append("both iliacs important → kissing VBX stents")
                if n_affected_iliacs_near_bif >= 2:
                    reasons.append(
                        f"{n_affected_iliacs_near_bif} branches compressed near hub"
                    )
                config_tips.append(
                    "Deploy two parallel VBX kissing stents from IVC bifurcation"
                )

            elif (
                finding_type
                in ("Pancaking", "Eccentric Compression", "Mild Compression")
                and not finding_near_bif
            ):
                # Lesion is deep in one iliac — straight stent on that branch
                stent_type = "Straight"
                reasons.append(
                    f"compression {finding_mm_from_bs:.0f}mm from bifurcation "
                    f"(>{KISS_ZONE_MM:.0f}mm) on Br{finding_branch} — distal iliac lesion"
                )
                reasons.append(
                    "lesion too distal for kissing → straight stent on affected branch"
                )
                config_tips.append(
                    f"Straight stent on Branch {finding_branch+1} covering lesion at "
                    f"{finding_mm_from_bs:.0f}mm — no kissing needed"
                )

            elif finding_type == "Pancaking" and finding_ratio > 2.5:
                stent_type = "Covered (PTFE)"
                reasons.append(
                    f"severe pancaking Ø{finding_diam:.1f}mm ratio={finding_ratio:.1f}"
                )
                reasons.append("high compression ratio → covered stent")
                config_tips.append("PTFE-covered stent to resist external compression")

            elif finding_type == "Mild Compression":
                if finding_diam < 6.0:
                    stent_type = "Drug-Eluting (DES)"
                    reasons.append(
                        f"small vessel Ø{finding_diam:.1f}mm — DES preferred"
                    )
                else:
                    stent_type = "Straight"
                    reasons.append(
                        f"mild compression Ø{finding_diam:.1f}mm ratio={finding_ratio:.1f} "
                        f"— {finding_mm_from_bs:.0f}mm from bifurcation"
                    )
                config_tips.append("Straight stent with 10mm landing zone each end")

        elif has_aneurysm:
            stent_type = "Covered (PTFE)"
            reasons.append(
                f"aneurysmal dilation trunk max Ø{trunk_max:.1f}mm > {trunk_median:.1f}mm × 1.5"
            )
            reasons.append("kissing contraindicated — covered stent graft")
            config_tips.append("Covered stent graft to exclude aneurysm sac")

        elif has_two_iliacs and not has_dilation:
            stent_type = "Kissing"
            reasons.append(
                f"{len(iliac_branches)} downstream iliacs, no focal findings near hub"
            )
            config_tips.append("Prophylactic kissing VBX if compression suspected")

        elif sel_avg < 6.0:
            stent_type = "Drug-Eluting (DES)"
            reasons.append(f"small vessel Ø{sel_avg:.1f}mm — DES preferred for <6mm")
            config_tips.append("Sirolimus or paclitaxel-eluting stent")

        elif taper_ratio > 1.15:
            stent_type = "Tapered"
            reasons.append(
                f"taper {taper_ratio:.2f} (Ø{sel_prox_d:.1f}→Ø{sel_dist_d:.1f}mm)"
            )
            config_tips.append(
                f"Proximal Ø{round(sel_prox_d*1.1/0.5)*0.5:.1f}mm, "
                f"distal Ø{round(sel_dist_d*1.1/0.5)*0.5:.1f}mm"
            )

        else:
            stent_type = "Straight"
            reasons.append(f"uniform vessel Ø{sel_avg:.1f}mm, taper={taper_ratio:.2f}")

        # ── apply to UI ──────────────────────────────────────────────────────
        type_map = {
            # Bare metal
            "Straight": 1,
            "Tapered": 2,
            "Bifurcated": 3,
            # GORE VBX
            "VBX Single": 5,
            "VBX Kissing": 6,
            "Kissing": 6,  # maps to VBX Kissing
            # GORE Viabahn Venous — for ectasia/dilation (NOT kissing)
            "Venous Straight": 8,
            "Venous Tapered": 9,
            # Covered
            "Covered (PTFE)": 11,
            "Covered Straight": 11,
            "Covered Bifurcated (EVAR)": 12,
            # Drug-Eluting
            "Drug-Eluting (DES)": 14,
            "DES Sirolimus": 14,
            "DES Paclitaxel": 15,
            "DES Zotarolimus": 16,
        }
        # Refine covered sub-type based on geometry
        if stent_type == "Covered (PTFE)":
            if n_downstream >= 2 and trunk_avg > trunk_median * 1.2:
                stent_type = "Covered Bifurcated (EVAR)"
            elif taper_ratio > 1.15:
                stent_type = "Covered Tapered"
            else:
                stent_type = "Covered Straight"
            # Note: "Covered Kissing" removed — kissing is VBX (balloon-expandable) only
        # Refine DES sub-type: paclitaxel for peripheral, sirolimus for small vessels
        if stent_type == "Drug-Eluting (DES)":
            if sel_avg > 4.0:
                stent_type = "DES Paclitaxel"  # peripheral vessels
            else:
                stent_type = "DES Sirolimus"  # very small vessels
        idx = type_map.get(stent_type, 0)
        self.stentTypeCombo.blockSignals(True)
        self.stentTypeCombo.setCurrentIndex(idx)
        self.stentTypeCombo.blockSignals(False)

        # Auto-configure diameters — block signals to prevent accidental stent placement
        prox_r = round(sel_prox_d * 1.15 / 0.5) * 0.5
        dist_r = round(sel_dist_d * 1.15 / 0.5) * 0.5
        self.stentProxDiamSpin.blockSignals(True)
        self.stentDistDiamSpin.blockSignals(True)
        self.stentTypeCombo.blockSignals(True)
        self.stentProxDiamSpin.setValue(prox_r)
        self.stentDistDiamSpin.setValue(dist_r)
        self.stentProxDiamSpin.blockSignals(False)
        self.stentDistDiamSpin.blockSignals(False)
        self.stentTypeCombo.blockSignals(False)

        # Build display message
        icon_map = {
            "Straight": "➡️",
            "Tapered": "📐",
            "Bifurcated": "🔱",
            "Kissing": "💋",
            "Covered (PTFE)": "🛡️",
            "Drug-Eluting (DES)": "💊",
        }
        icon = icon_map.get(stent_type, "🧠")
        msg = f"{icon} Recommended: {stent_type}\n"
        msg += "Reasons: " + "; ".join(reasons) + "\n"
        if config_tips:
            msg += "Tip: " + config_tips[0]
        self.stentAutoTypeLabel.setText(msg)
        self.stentAutoTypeLabel.setStyleSheet(
            "font-size: 11px; color: #1a6b3c; padding: 4px; "
            "background: #eafaf1; border-radius: 4px; font-style: italic;"
        )
        print(f"[AutoType] → {stent_type}: {'; '.join(reasons)}")

        # Navigate IVUS to the finding location
        if nav_to_gi >= 0 and nav_to_gi < len(self.logic.points):
            try:
                _trav_idx = self.logic.globalToTraversal(nav_to_gi)
                self.pointSlider.setValue(_trav_idx)
                self.onSliderChanged(_trav_idx)
                self.stentAutoTypeLabel.setText(
                    self.stentAutoTypeLabel.text
                    + f"\n📍 Navigator jumped to finding at pt{nav_to_gi} (Br{finding_branch})"
                )
            except Exception as e:
                pass

        # Refresh summary label only — do NOT place stent
        try:
            self._updateStentSummaryOnly()
        except Exception:
            pass

    def _updateStentSummaryOnly(self):
        """Refresh the stent summary/warning labels from current UI state without placing a stent."""
        if not self.logic.points or not self.logic.branches:
            return
        stype = self.stentTypeCombo.currentText.strip()
        pd = self.stentProxDiamSpin.value
        dd = self.stentDistDiamSpin.value
        # Estimate length from slider positions
        p = self.stentProxSlider.value
        di = self.stentDistSlider.value
        stent_bi = self._currentStentBranchIndex()
        if stent_bi < 0 or stent_bi >= len(self.logic.branches):
            return
        bs, be = self.logic.branches[stent_bi]
        gp = min(p + bs, be - 1)
        gd = min(di + bs, be - 1)
        if gp >= gd:
            return
        length = sum(
            (
                (self.logic.points[i + 1][0] - self.logic.points[i][0]) ** 2
                + (self.logic.points[i + 1][1] - self.logic.points[i][1]) ** 2
                + (self.logic.points[i + 1][2] - self.logic.points[i][2]) ** 2
            )
            ** 0.5
            for i in range(gp, min(gd, len(self.logic.points) - 2))
        )
        vd_prox = self.logic.diameters[gp] if gp < len(self.logic.diameters) else 0
        vd_dist = self.logic.diameters[gd] if gd < len(self.logic.diameters) else 0
        self.stentSummaryLabel.setText(
            "Type: "
            + stype
            + "  |  Length: "
            + str(round(length, 1))
            + "mm\n"
            + "Proximal: "
            + str(round(pd, 1))
            + "mm stent / "
            + str(round(vd_prox, 1))
            + "mm vessel\n"
            + "Distal:   "
            + str(round(dd, 1))
            + "mm stent / "
            + str(round(vd_dist, 1))
            + "mm vessel"
        )
        # Clear warnings for kissing stents
        if "Kissing" in stype:
            self.stentWarningLabel.setText("Sizing OK")
            self.stentWarningLabel.setStyleSheet("color: #27ae60; font-weight: bold;")
        else:
            self.stentWarningLabel.setText("")

    def _restoreDefaultLayout(self):
        """Restore Slicer to single 3D view."""
        try:
            self._TWO3D_LAYOUT_ID = None
            self._force3DLayout()

            # Remove snapshot model if present
            for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
                if node.GetName() in (
                    "VesselBefore_Snapshot",
                ) or node.GetName().startswith("BeforeStenosis_"):
                    slicer.mrmlScene.RemoveNode(node)
            # Restore all model display nodes to show in all views
            if self.logic.modelNode:
                dn = self.logic.modelNode.GetDisplayNode()
                if dn:
                    dn.RemoveAllViewNodeIDs()
            # Restore CT slice visibility in 3D
            for sliceNode in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
                wasVisible = getattr(self, "_sliceVis3DBackup", {}).get(
                    sliceNode.GetID(), 0
                )
                sliceNode.SetSliceVisible(wasVisible)
            self._sliceVis3DBackup = {}
            # Restore volume display nodes
            for dn in getattr(self, "_hiddenVolDisplayNodes", []):
                try:
                    dn.SetVisibility(1)
                except Exception:
                    pass
            self._hiddenVolDisplayNodes = []
            # Restore VR nodes
            for vrDN in getattr(self, "_hiddenVRNodes", []):
                try:
                    vrDN.SetVisibility(1)
                except Exception:
                    pass
            self._hiddenVRNodes = []
        except Exception as e:
            print("[Layout] Restore failed: " + str(e))

    def _snapStentLength(self, length_mm):
        """Snap stent length to nearest real GORE available length.
        VBX (balloon-expandable): 15, 19, 29, 39, 59, 79 mm
        Viabahn Venous (self-expanding): 50, 75, 100, 150 mm
        Other: round to nearest 5mm."""
        stype = self.stentTypeCombo.currentText.strip()
        if "VBX" in stype:
            sizes = [15, 19, 29, 39, 59, 79]
        elif "Venous" in stype:
            sizes = [50, 75, 100, 150]
        else:
            # Generic: round to nearest 5mm
            return round(length_mm / 5.0) * 5.0
        return float(min(sizes, key=lambda s: abs(s - length_mm)))



# ── Slicer module-scanner guard ───────────────────────────────────────────────
class vessel_stent_widget_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_stent_widget_mixin"
            parent.hidden = True
