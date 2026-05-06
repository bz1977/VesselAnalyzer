"""
vessel_navigation_widget_mixin.py — VesselAnalyzerWidget IVUS navigation and measurement handlers

Extracted from VesselAnalyzer.py (VesselAnalyzerWidget methods).

Usage in VesselAnalyzer.py
--------------------------
Add to the VesselAnalyzerWidget class definition::

    from vessel_navigation_widget_mixin import NavigationWidgetMixin

    class VesselAnalyzerWidget(NavigationWidgetMixin, ScriptedLoadableModuleWidget, VTKObservationMixin):
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


class NavigationWidgetMixin:
    """Mixin: VesselAnalyzerWidget IVUS navigation and measurement handlers"""
    def onLoadIVUS(self):
        centerlineNode = self.centerlineSelector.currentNode()
        modelNode = self.modelSelector.currentNode()

        if not centerlineNode:
            slicer.util.errorDisplay("Please select or extract a centerline first.")
            return

        # Clean up all VesselAnalyzer scene nodes from previous run
        # Save current layout now — this is when user is in the 3D view
        try:
            _cur = slicer.app.layoutManager().layout
            self._preBalloonLayout = _cur if _cur > 0 else 3
        except Exception:
            self._preBalloonLayout = 3

        # Restore only nodes that VesselAnalyzer explicitly dimmed (intervention nodes).
        # Never touch the main vessel model opacity — preserve whatever the user set.
        for _cn in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
            _nm = _cn.GetName()
            if (
                _cn.GetName() in ("ExpandedVessel", "StentModel", "BalloonDilate_0")
                or _cn.GetAttribute("VesselAnalyzerExpanded") == "1"
                or _cn.GetAttribute("VesselAnalyzerEmbo") == "1"
            ):
                slicer.mrmlScene.RemoveNode(_cn)
        # Restore intervention nodes that were dimmed — but NOT the vessel model itself
        _vessel_node = self.modelSelector.currentNode()
        for _mn in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if _vessel_node and _mn.GetID() == _vessel_node.GetID():
                continue  # never touch the main vessel model opacity on load
            _dn = _mn.GetDisplayNode()
            if _dn and _dn.GetOpacity() < 0.5:
                _dn.SetOpacity(1.0)
                _dn.SetVisibility(1)

        # Always use a fresh logic instance to ensure latest code is used
        from VesselAnalyzer import VesselAnalyzerLogic
        self.logic = VesselAnalyzerLogic()
        self.logic._widget = self  # back-reference so logic can call widget methods
        # Carry over vessel type chosen in Step 1 (stored on widget to survive re-instantiation)
        if getattr(self, "_pendingVesselType", None):
            self.logic.vesselType = self._pendingVesselType
        # Always confirm vessel type in console so it's never ambiguous
        _vt = getattr(self.logic, "vesselType", None) or "NOT SET"
        print(f"[VesselAnalyzer] *** VESSEL TYPE: {_vt.upper()} ***")
        # Save original model color so we can restore it after stent removal
        if modelNode and modelNode.GetDisplayNode():
            _c = modelNode.GetDisplayNode().GetColor()
            self._origModelColor = (_c[0], _c[1], _c[2])
        else:
            self._origModelColor = (0.9, 0.9, 0.9)

        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            result = self.logic.loadCenterline(
                centerlineNode, modelNode, segModelNode=self.modelSelector.currentNode()
            )
            if result:
                numPoints = self.logic.getNumPoints()
                self.pointSlider.maximum = numPoints - 1
                self.pointSlider.enabled = True
                self.pointSlider.value = 0
                # Populate branch selector
                self.branchSelector.blockSignals(True)
                self.branchSelector.clear()
                self.branchSelector.addItem("All branches")
                numBranches = self.logic.getNumBranches()
                # Label branches by their matched endpoint
                epNode = slicer.mrmlScene.GetFirstNodeByName("Endpoints")
                epLabels = []
                if epNode:
                    for _ei in range(epNode.GetNumberOfControlPoints()):
                        epLabels.append(f"Endpoint {_ei}")

                # Branch 0 = trunk (longest seg, maps to some endpoint)
                # Branch 1,2,3 = unique parts sorted by endpoint index
                branchEpMap = getattr(self.logic, "branchEndpointMap", {})

                # Populate selector using bi as item userData so display order
                # never affects which branch data is fetched.
                branchNames = getattr(
                    self.logic,
                    "branchNames",
                    [f"Branch {i+1}" for i in range(numBranches)],
                )

                # ── [SelectorDebug] dump everything the selector will display ──
                print(f"[SelectorDebug] ══ BRANCH SELECTOR POPULATION ══════════")
                print(f"[SelectorDebug] numBranches={numBranches}  branchNames={branchNames}")
                _bm_d = getattr(self.logic, "branchMeta", {})
                _ep_d = getattr(self.logic, "branchEndpointMap", {})
                _br_d = getattr(self.logic, "branches", [])
                _pts_d = getattr(self.logic, "points", [])
                for _dbi in range(len(branchNames)):
                    _dm = _bm_d.get(_dbi, {})
                    _d_role   = _dm.get("role", "MISSING")
                    _d_lat    = _dm.get("lateral_label", "")
                    _d_len    = _dm.get("length_mm", 0.0)
                    _d_ep_idx = _ep_d.get(_dbi, "?")
                    _d_name   = branchNames[_dbi]
                    try:
                        _d_disp = self.logic.getBranchDisplayName(_dbi)
                    except Exception as _e:
                        _d_disp = f"ERROR:{_e}"
                    # Tip and root X from committed points
                    _d_tip_x = _d_root_x = "?"
                    if _dbi < len(_br_d):
                        _bs, _be = _br_d[_dbi]
                        if _bs < len(_pts_d):
                            _d_root_x = f"{_pts_d[_bs][0]:+.1f}"
                        if _be > _bs and _be - 1 < len(_pts_d):
                            _d_tip_x = f"{_pts_d[_be-1][0]:+.1f}"
                    _d_will_show = _d_disp  # getBranchDisplayName is now the single source
                    print(
                        f"[SelectorDebug]  bi={_dbi}"
                        f"  branchName={_d_name!r}"
                        f"  role={_d_role!r}"
                        f"  lateral={_d_lat!r}"
                        f"  ep_idx={_d_ep_idx}"
                        f"  root_X={_d_root_x}  tip_X={_d_tip_x}"
                        f"  len={_d_len:.0f}mm"
                        f"  getBranchDisplayName→{_d_disp!r}"
                        f"  WILL_SHOW→{_d_will_show!r}"
                    )
                # Dump Endpoints node control-point descriptions (ground truth for L/R)
                _ep_nd = slicer.mrmlScene.GetFirstNodeByName("Endpoints")
                if _ep_nd:
                    print(f"[SelectorDebug] Endpoints node ({_ep_nd.GetNumberOfControlPoints()} pts):")
                    for _ci in range(_ep_nd.GetNumberOfControlPoints()):
                        _cp = [0.0, 0.0, 0.0]
                        _ep_nd.GetNthControlPointPositionWorld(_ci, _cp)
                        try:
                            _cp_desc = _ep_nd.GetNthControlPointDescription(_ci)
                        except Exception:
                            _cp_desc = ""
                        _cp_lbl = _ep_nd.GetNthControlPointLabel(_ci)
                        print(
                            f"[SelectorDebug]   cp{_ci}"
                            f"  label={_cp_lbl!r}"
                            f"  desc={_cp_desc!r}"
                            f"  X={_cp[0]:+.1f} Y={_cp[1]:+.1f} Z={_cp[2]:+.1f}"
                        )
                else:
                    print("[SelectorDebug] No Endpoints node in scene")
                print(f"[SelectorDebug] ════════════════════════════════════════════")
                # ── end SelectorDebug ─────────────────────────────────────────

                for bi in range(len(branchNames)):
                    # Exclude trunk (bi=0) and non-navigable roles from selector.
                    # Trunk is traversed via the main slider; noise/fragments have
                    # no meaningful diameter data to display.
                    _bi_role = getattr(self.logic, "branchMeta", {}).get(bi, {}).get("role", "")
                    if _bi_role in ("trunk", "noise", "renal_fragment", "iliac_fragment"):
                        print(f"[SelectorPopulate] bi={bi} role={_bi_role!r} — excluded from selector")
                        continue
                    bStats = self.logic.getBranchStats(bi)
                    label = self.logic.getBranchDisplayName(bi)
                    self.branchSelector.addItem(
                        f"{label}  ({bStats['length']:.0f}mm, min={bStats['min']:.1f}mm, max={bStats['max']:.1f}mm)",
                        bi,  # stored as Qt item userData — retrieved via itemData()
                    )
                print("[NavigatorCombo DEBUG] initial selector rows:")
                _bm_dbg = getattr(self.logic, "branchMeta", {})
                _br_dbg = getattr(self.logic, "branches", [])
                for _row in range(self.branchSelector.count):
                    _data = self.branchSelector.itemData(_row)
                    _txt = self.branchSelector.itemText(_row)
                    if _data is None or _data == -1:
                        print(f"[NavigatorCombo DEBUG]   row={_row} text={_txt!r} data={_data!r}")
                        continue
                    _meta = _bm_dbg.get(_data, {})
                    _rng = _br_dbg[_data] if 0 <= _data < len(_br_dbg) else ("?", "?")
                    print(
                        f"[NavigatorCombo DEBUG]   row={_row} data_bi={_data} "
                        f"text={_txt!r} role={_meta.get('role','?')!r} "
                        f"label={_meta.get('label','?')!r} "
                        f"display={self.logic.getBranchDisplayName(_data)!r} "
                        f"range={_rng}"
                    )
                self.branchSelector.setEnabled(True)
                self.branchSelector.blockSignals(False)
                # Enable embolization controls after load
                self.emboPickZoneButton.setEnabled(True)
                self.emboSizeSpin.setEnabled(True)
                self.updateStats()
                self.onSliderChanged(0)
                self.updateBranchStatsBox(0)
                self.logic.debugBranches()
                self._updateStentSliderRanges()
                # Hide endpoint markers — keep scene clean
                try:
                    epNode2 = slicer.mrmlScene.GetFirstNodeByName("Endpoints")
                    if epNode2:
                        dn2 = epNode2.GetDisplayNode()
                        if dn2:
                            dn2.SetVisibility(0)
                except Exception:
                    pass
                # Auto-populate Branch Surface Classification selector and
                # run classify so the button works without manual node selection.
                if modelNode and modelNode.GetPolyData():
                    self.branchSurfModelSelector.setCurrentNode(modelNode)
                    try:
                        self.onClassifySurface()
                    except Exception as _ce:
                        import traceback

                        print(f"[ClassifySurface] Auto-classify failed: {_ce}")
                        traceback.print_exc()
                slicer.util.infoDisplay(
                    f"Loaded {numPoints} points across {numBranches} branches!"
                )
            else:
                slicer.util.errorDisplay("Failed to load centerline points.")
        except Exception as e:
            slicer.util.errorDisplay(f"Error: {str(e)}")
            import traceback

            print(traceback.format_exc())
        finally:
            slicer.app.restoreOverrideCursor()

    def onBranchChanged(self, index):
        """Switch navigation to a specific branch or all branches"""
        def _nav_label_probe(_tag):
            try:
                _idx = self.branchSelector.currentIndex
                _data = self.branchSelector.itemData(_idx)
                _text = self.branchSelector.itemText(_idx)
            except Exception as _e:
                _idx, _data, _text = "ERR", "ERR", f"ERR:{_e}"
            _active = getattr(self.logic, "activeBranch", "?")
            _role = getattr(self.logic, "branchMeta", {}).get(_active, {}).get("role", "?") if isinstance(_active, int) and _active >= 0 else "ALL"
            try:
                _disp = self.logic.getBranchDisplayName(_active) if isinstance(_active, int) and _active >= 0 else "All branches"
            except Exception as _e:
                _disp = f"ERR:{_e}"
            try:
                _label_txt = self.branchNameLabel.text
            except Exception as _e:
                _label_txt = f"ERR:{_e}"
            print(
                f"[NavigatorLabel DEBUG] {_tag}: comboIndex={_idx} "
                f"comboData={_data!r} comboText={_text!r} "
                f"activeBranch={_active!r} activeRole={_role!r} "
                f"activeDisplay={_disp!r} branchNameLabel={_label_txt!r}"
            )

        _nav_label_probe("onBranchChanged:entry")
        # ── [DEBUG] UI → bi mapping boundary ─────────────────────────────
        print(f"\n[DEBUG onBranchChanged]")
        print(f"  selected index = {index}")
        print(f"  text           = {self.branchSelector.itemText(index)!r}")
        _raw_data = self.branchSelector.itemData(index)
        print(f"  itemData(bi)   = {_raw_data!r}  (type={type(_raw_data).__name__})")
        print("[NavigatorCombo DEBUG] current selector rows at selection:")
        _bm_dbg = getattr(self.logic, "branchMeta", {})
        _br_dbg = getattr(self.logic, "branches", [])
        for _row in range(self.branchSelector.count):
            _data = self.branchSelector.itemData(_row)
            _txt = self.branchSelector.itemText(_row)
            if _data is None or _data == -1:
                print(f"[NavigatorCombo DEBUG]   row={_row} text={_txt!r} data={_data!r}")
                continue
            _meta = _bm_dbg.get(_data, {})
            _rng = _br_dbg[_data] if 0 <= _data < len(_br_dbg) else ("?", "?")
            _mark = "  <-- selected" if _row == index else ""
            print(
                f"[NavigatorCombo DEBUG]   row={_row} data_bi={_data} "
                f"text={_txt!r} role={_meta.get('role','?')!r} "
                f"label={_meta.get('label','?')!r} "
                f"display={self.logic.getBranchDisplayName(_data)!r} "
                f"range={_rng}{_mark}"
            )
        if index != 0:
            _role = getattr(self.logic, "branchMeta", {}).get(_raw_data, {}).get("role", "MISSING")
            print(f"  role           = {_role!r}")
        # ── end DEBUG ─────────────────────────────────────────────────────
        if index == 0:
            self.logic.setActiveBranch(-1)
            numPoints = self.logic.getNumPoints()
            self.branchNameLabel.setText("All branches")
            _nav_label_probe("onBranchChanged:set-all")
        else:
            # bi is stored as Qt item userData — never depends on display order
            bi = self.branchSelector.itemData(index)
            if bi is None:
                bi = index - 1  # safe fallback for items added without userData
            self.logic.setActiveBranch(bi)
            numPoints = self.logic.getActiveBranchPointCount()
            self.branchNameLabel.setText(
                self.branchSelector.itemText(index).split("  (")[0]
            )
            _nav_label_probe("onBranchChanged:set-branch")
            print(
                f"[Nav] index={index} bi={bi} "
                f"branches[bi]={self.logic.branches[bi]} "
                f"pointCount={numPoints}"
            )

        self.pointSlider.maximum = max(0, numPoints - 1)
        self.pointSlider.value = 0
        self.updateStats()
        _nav_label_probe("onBranchChanged:after-updateStats")
        self.onSliderChanged(0)
        _nav_label_probe("onBranchChanged:after-onSliderChanged")
        self.updateBranchStatsBox(index)
        _nav_label_probe("onBranchChanged:after-updateBranchStatsBox")
        self._updateStentSliderRanges()
        self._refreshStentSummary()
        _nav_label_probe("onBranchChanged:exit")

    def updateBranchStatsBox(self, index):
        """Update the branch statistics box for selected branch."""
        # ── [DEBUG] stats box bi ──────────────────────────────────────────
        print(f"\n[DEBUG updateBranchStatsBox]")
        print(f"  index          = {index}")
        print(f"  activeBranch   = {getattr(self.logic, 'activeBranch', '?')}")
        # ── end DEBUG ─────────────────────────────────────────────────────
        if index == 0:
            stats = self.logic.getStats()
            if stats:
                self.branchLengthLabel.setText(f"{stats['total_length']:.1f} mm")
                self.branchMinLabel.setText(
                    f"{stats['min']:.2f} mm" if stats["min"] > 0 else "N/A"
                )
                self.branchMaxLabel.setText(
                    f"{stats['max']:.2f} mm" if stats["max"] > 0 else "N/A"
                )
                self.branchAvgLabel.setText(
                    f"{stats['avg']:.2f} mm" if stats["avg"] > 0 else "N/A"
                )
        else:
            bi = self.branchSelector.itemData(index)
            if bi is None:
                bi = index - 1
            bStats = self.logic.getBranchStats(bi)
            self.branchLengthLabel.setText(
                f"{bStats['length']:.1f} mm  ({bStats['points']} points)"
            )
            self.branchMinLabel.setText(
                f"{bStats['min']:.2f} mm" if bStats["min"] > 0 else "N/A"
            )
            self.branchMaxLabel.setText(
                f"{bStats['max']:.2f} mm" if bStats["max"] > 0 else "N/A"
            )
            self.branchAvgLabel.setText(
                f"{bStats['avg']:.2f} mm" if bStats["avg"] > 0 else "N/A"
            )

    def onSliderChanged(self, value):
        print(
            f"[NavTrace] onSliderChanged entry: value={value}  _updatingGUI={self._updatingGUI}  _fromButton={getattr(self,'_fromButton',False)}"
        )
        if self._updatingGUI:
            print(f"[NavTrace] onSliderChanged EXIT: _updatingGUI=True")
            return
        idx = int(value)
        try:
            globalIdx = self.logic.localToGlobal(idx)
        except Exception as _e:
            import traceback

            print(f"[NavTrace] onSliderChanged EXIT: localToGlobal({idx}) raised {_e}")
            print(traceback.format_exc())
            return
        print(
            f"[NavTrace] onSliderChanged: idx={idx}  globalIdx={globalIdx}  activeBranch={getattr(self.logic,'activeBranch','?')}  _fromButton={getattr(self,'_fromButton',False)}"
        )
        # Jump guard: suppress large spatial gaps, but ALLOW inter-branch
        # transitions in traversal mode.  In _allBranchTraversal, crossing
        # from one branch end to the next branch start is a known discontinuity
        # — detect it by checking if prevGi and globalIdx are in DIFFERENT branches.
        # BYPASSED when called from Prev/Next buttons (always step by 1).
        # NavRobust: entire guard wrapped in try/except — a crash inside the guard
        # (e.g. from a future traversal change) must never block slider navigation.
        # If the guard throws, we log and fall through to updateMeasurementsAtIndex.
        try:
            if (
                not getattr(self, "_fromButton", False)
                and globalIdx > 0
                and self.logic.activeBranch < 0
            ):
                import math

                prevIdx = self.logic.localToGlobal(max(0, idx - 1))
                p0 = self.logic.points[prevIdx]
                p1 = self.logic.points[globalIdx]
                dist = math.sqrt(
                    (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2
                )
                if dist > 40.0:
                    # Allow if prev and current are in different branches
                    # (inter-branch transition in traversal — expected gap)
                    _prev_branch = self.logic.getBranchForPoint(prevIdx)
                    _curr_branch = self.logic.getBranchForPoint(globalIdx)
                    _is_inter_branch = (
                        _prev_branch != _curr_branch
                        and _prev_branch >= 0
                        and _curr_branch >= 0
                    )
                    # Also allow known sub-branch connections
                    for _bi_from, _bi_to in getattr(
                        self.logic, "branchConnections", {}
                    ).items():
                        if _bi_from < len(self.logic.branches) and _bi_to < len(
                            self.logic.branches
                        ):
                            _from_last = self.logic.branches[_bi_from][1] - 1
                            _to_first = self.logic.branches[_bi_to][0]
                            if globalIdx in (_from_last, _to_first):
                                _is_inter_branch = True
                                break
                    if not _is_inter_branch:
                        print(
                            f"[NavTrace] onSliderChanged EXIT: jump guard dist={dist:.1f}mm prev_branch={_prev_branch} curr_branch={_curr_branch} _is_inter_branch={_is_inter_branch}"
                        )
                        return
        except Exception as _guard_exc:
            import traceback

            print(
                f"[NavTrace] onSliderChanged: jump guard EXCEPTION (ignored, navigation continues): {_guard_exc}"
            )
            print(traceback.format_exc())
        # ── Live branch tracking in all-branches traversal mode ──────────────
        # When the user is in "All branches" mode (activeBranch == -1) the
        # logic object never knows which anatomical branch the cursor is on.
        # Resolve it now from the globalIdx so every downstream consumer
        # (findings panel, stent panel, surface coloring) sees the correct
        # branch index — then restore -1 after the update so the slider range
        # and traversal logic remain in all-branches mode.
        if self.logic.activeBranch < 0:
            _resolved = self.logic.getBranchForPoint(globalIdx)
            if _resolved >= 0:
                print(
                    f"[NavTrace] onSliderChanged: all-branches point resolves to "
                    f"{_resolved} (globalIdx={globalIdx}); activeBranch remains -1"
                )
            else:
                print(
                    f"[NavTrace] onSliderChanged: all-branches point resolves to "
                    f"-1 (globalIdx={globalIdx}); activeBranch remains -1"
                )
        else:
            _restore_all_branches = False

        # One-shot geometry dump: show tip X of each branch so we can verify
        # whether bi=3/bi=5 point ranges match their iliac_left/iliac_right roles.
        _pts  = getattr(self.logic, "points", [])
        _brs  = getattr(self.logic, "branches", [])
        if not getattr(self, "_branchGeoDumped", False) and len(_brs) > 1 and len(_pts) > 0:
            self._branchGeoDumped = True
            _bmeta = getattr(self.logic, "branchMeta", {})
            print(f"[BranchGeo DEBUG] branch point-range X coordinates: {len(_brs)} branches, {len(_pts)} pts")
            for _gi, (_bs, _be) in enumerate(_brs):
                _role = _bmeta.get(_gi, {}).get("role", "?")
                _root_x = f"{_pts[_bs][0]:+.1f}" if _bs < len(_pts) else "?"
                _tip_x  = f"{_pts[_be-1][0]:+.1f}" if _be > _bs and _be-1 < len(_pts) else "?"
                print(f"  bi={_gi} role={_role!r:15s} gi={_bs}..{_be-1}  root_X={_root_x}  tip_X={_tip_x}")
        print(f"[NavTrace] onSliderChanged: calling updateMeasurementsAtIndex({idx}) globalIdx={globalIdx} slider.value={int(self.pointSlider.value)}")
        self.updateMeasurementsAtIndex(idx, globalIdx=globalIdx)
        self.logic.moveCrosshairToPoint(globalIdx)

        endoluminal = self.endoluminalButton.isChecked()
        self.logic.updateVisualizations(
            globalIdx,
            # In endoluminal mode: show sphere/ring/line in normal view (widget 0)
            # but hide ring and line to keep endoluminal view clean
            showSphere=self.showSphereCheck.isChecked(),  # sphere visible in normal view
            showRing=False if endoluminal else self.showRingCheck.isChecked(),
            showLine=False if endoluminal else self.showLineCheck.isChecked(),
            sphereColor=self._getSphereColor(globalIdx),
        )
        if endoluminal:
            self.logic.setEndoluminalCamera(globalIdx)

    def _getSphereColor(self, globalIdx=None):
        """Return sphere color matching current branch surface classification.
        Falls back to self._sphereColor if no surface classification is active."""
        _surf_colors = getattr(self.logic, "_branch_surface_colors", None)
        if _surf_colors is not None and globalIdx is not None:
            try:
                _cur_bi = self.logic.getBranchForPoint(globalIdx)
                if _cur_bi >= 0 and _cur_bi in _surf_colors:
                    _rgb255 = _surf_colors[_cur_bi]
                    return (_rgb255[0] / 255.0, _rgb255[1] / 255.0, _rgb255[2] / 255.0)
            except Exception:
                pass
        return self._sphereColor

    def onEndoluminalToggle(self, checked):
        layoutNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLayoutNode")
        ENDO_LAYOUT_ID = 502
        if checked:
            self.endoluminalButton.setText("🔭 Endoluminal View  ON")
            self.endoluminalButton.setStyleSheet(
                "background-color: #27ae60; color: white; font-weight: bold; padding: 5px;"
            )
            # Register and switch to side-by-side 3D layout:
            # Left = normal external view, Right = endoluminal camera
            endoLayout = """<layout type="horizontal" split="true">
  <item><view class="vtkMRMLViewNode" singletontag="1">
    <property name="viewlabel" action="default">3D</property>
  </view></item>
  <item><view class="vtkMRMLViewNode" singletontag="2">
    <property name="viewlabel" action="default">Endoluminal</property>
  </view></item>
</layout>"""
            if layoutNode:
                self._preEndoluminalLayout = layoutNode.GetViewArrangement()
                if not layoutNode.IsLayoutDescription(ENDO_LAYOUT_ID):
                    layoutNode.AddLayoutDescription(ENDO_LAYOUT_ID, endoLayout)
                layoutNode.SetViewArrangement(ENDO_LAYOUT_ID)
            if self.logic.modelNode:
                self.logic.modelNode.GetDisplayNode().SetOpacity(0.15)
            self.logic.updateVisualizations(
                int(self.pointSlider.value),
                showSphere=False,
                showRing=False,
                showLine=False,
            )
            # Set endoluminal camera on view 1 (right panel), keep view 0 as normal
            self.logic.setEndoluminalCamera(int(self.pointSlider.value))
        else:
            self.endoluminalButton.setText("🔭 Endoluminal View  OFF")
            self.endoluminalButton.setStyleSheet(
                "background-color: #2c3e50; color: white; font-weight: bold; padding: 5px;"
            )
            if layoutNode:
                prev = getattr(self, "_preEndoluminalLayout", 2)
                layoutNode.SetViewArrangement(prev)
            if self.logic.modelNode:
                self.logic.modelNode.GetDisplayNode().SetOpacity(1.0)
            # Sphere already visible in all views — no restriction to clear
            self.logic.resetCamera()
            _endo_gi = self.logic.localToGlobal(int(self.pointSlider.value))
            self.logic.updateVisualizations(
                int(self.pointSlider.value),
                showSphere=self.showSphereCheck.isChecked(),
                showRing=self.showRingCheck.isChecked(),
                showLine=self.showLineCheck.isChecked(),
                sphereColor=self._getSphereColor(_endo_gi),
            )

    def onPrevPoint(self):
        new_val = max(0, int(self.pointSlider.value) - 1)
        print(
            f"[NavTrace] PREV btn: slider {int(self.pointSlider.value)} → {new_val}  max={int(self.pointSlider.maximum)}"
        )
        self.pointSlider.blockSignals(True)
        self.pointSlider.setValue(new_val)
        self.pointSlider.blockSignals(False)
        self._fromButton = True
        try:
            self.onSliderChanged(new_val)
        except Exception as _e:
            import traceback

            print(f"[NavTrace] PREV onSliderChanged EXCEPTION: {_e}")
            print(traceback.format_exc())
        finally:
            self._fromButton = False

    def onNextPoint(self):
        new_val = min(int(self.pointSlider.maximum), int(self.pointSlider.value) + 1)
        print(
            f"[NavTrace] NEXT btn: slider {int(self.pointSlider.value)} → {new_val}  max={int(self.pointSlider.maximum)}"
        )
        self.pointSlider.blockSignals(True)
        self.pointSlider.setValue(new_val)
        self.pointSlider.blockSignals(False)
        self._fromButton = True
        try:
            self.onSliderChanged(new_val)
        except Exception as _e:
            import traceback

            print(f"[NavTrace] NEXT onSliderChanged EXCEPTION: {_e}")
            print(traceback.format_exc())
        finally:
            self._fromButton = False

    def _branchMinMaxGi(self, bi, want_min):
        """Return the global point index of the min (want_min=True) or max
        (want_min=False) diameter within branch bi.

        Strategy:
        1. Try getBranchStats — fast path when the logic populates min_idx/max_idx.
        2. If min_idx/max_idx is absent or -1, scan the branch's own point range
           directly using logic.diameters.  This handles logic implementations
           that compute the stat value but omit the index.
        3. Fall back to the global getter only if the branch range is unavailable.
        """
        key = "min_idx" if want_min else "max_idx"
        stat_key = "min" if want_min else "max"

        # ── Step 1: getBranchStats fast path ──────────────────────────────
        try:
            bStats = self.logic.getBranchStats(bi)
            gi = bStats.get(key, -1)
            if gi is not None and gi >= 0:
                print(f"[DEBUG _branchMinMaxGi] bi={bi} {'min' if want_min else 'max'}"
                      f" via getBranchStats: gi={gi}  diam={bStats.get(stat_key,'?'):.2f}mm")
                return gi
        except Exception as _e:
            print(f"[DEBUG _branchMinMaxGi] getBranchStats error: {_e}")

        # ── Step 2: scan the branch point range directly ──────────────────
        try:
            branches  = getattr(self.logic, "branches", [])
            diameters = getattr(self.logic, "diameters", [])
            if bi < len(branches) and diameters:
                bs, be = branches[bi]          # [bs, be) — be is exclusive
                best_gi  = -1
                best_val = None
                for gi in range(bs, be):
                    if gi >= len(diameters):
                        break
                    d = diameters[gi]
                    if d <= 0:
                        continue
                    if best_val is None:
                        best_val, best_gi = d, gi
                    elif want_min and d < best_val:
                        best_val, best_gi = d, gi
                    elif (not want_min) and d > best_val:
                        best_val, best_gi = d, gi
                if best_gi >= 0:
                    print(f"[DEBUG _branchMinMaxGi] bi={bi} {'min' if want_min else 'max'}"
                          f" via range scan: gi={best_gi}  diam={best_val:.2f}mm"
                          f"  range=[{bs},{be})")
                    return best_gi
        except Exception as _e:
            print(f"[DEBUG _branchMinMaxGi] range scan error: {_e}")

        # ── Step 3: global fallback ───────────────────────────────────────
        gi = (self.logic.getMinDiameterIndex() if want_min
              else self.logic.getMaxDiameterIndex())
        print(f"[DEBUG _branchMinMaxGi] bi={bi} {'min' if want_min else 'max'}"
              f" via global fallback: gi={gi}")
        return gi

    def onGoToMin(self):
        bi = getattr(self.logic, "activeBranch", -1)
        print(f"\n[DEBUG onGoToMin]  activeBranch={bi}")
        gi = self._branchMinMaxGi(bi, want_min=True) if bi >= 0 else self.logic.getMinDiameterIndex()
        print(f"  → jumping to gi={gi}  traversal={self.logic.globalToTraversal(gi) if gi>=0 else 'n/a'}")
        if gi >= 0:
            self.pointSlider.value = self.logic.globalToTraversal(gi)

    def onGoToMax(self):
        bi = getattr(self.logic, "activeBranch", -1)
        print(f"\n[DEBUG onGoToMax]  activeBranch={bi}")
        gi = self._branchMinMaxGi(bi, want_min=False) if bi >= 0 else self.logic.getMaxDiameterIndex()
        print(f"  → jumping to gi={gi}  traversal={self.logic.globalToTraversal(gi) if gi>=0 else 'n/a'}")
        if gi >= 0:
            self.pointSlider.value = self.logic.globalToTraversal(gi)

    def onPickColor(self):
        color = qt.QColorDialog.getColor()
        if color.isValid():
            self._sphereColor = (color.redF(), color.greenF(), color.blueF())
            self.sphereColorButton.setStyleSheet(
                f"background-color: rgb({color.red()},{color.green()},{color.blue()}); color: white;"
            )

    def updateMeasurementsAtIndex(self, idx, globalIdx=None):
        def _nav_measure_probe(_tag, _branch_idx=None, _global_idx=None):
            try:
                _combo_idx = self.branchSelector.currentIndex
                _combo_data = self.branchSelector.itemData(_combo_idx)
                _combo_text = self.branchSelector.itemText(_combo_idx)
            except Exception as _e:
                _combo_idx, _combo_data, _combo_text = "ERR", "ERR", f"ERR:{_e}"
            _active = getattr(self.logic, "activeBranch", "?")
            try:
                _point_branch = self.logic.getBranchForPoint(_global_idx) if _global_idx is not None else None
            except Exception as _e:
                _point_branch = f"ERR:{_e}"
            try:
                _label_txt = self.branchNameLabel.text
            except Exception as _e:
                _label_txt = f"ERR:{_e}"
            _bm = getattr(self.logic, "branchMeta", {})
            _branch_role = _bm.get(_branch_idx, {}).get("role", "?") if isinstance(_branch_idx, int) and _branch_idx >= 0 else "n/a"
            _point_role = _bm.get(_point_branch, {}).get("role", "?") if isinstance(_point_branch, int) and _point_branch >= 0 else "n/a"
            try:
                _branch_disp = self.logic.getBranchDisplayName(_branch_idx) if isinstance(_branch_idx, int) and _branch_idx >= 0 else "n/a"
            except Exception as _e:
                _branch_disp = f"ERR:{_e}"
            try:
                _point_disp = self.logic.getBranchDisplayName(_point_branch) if isinstance(_point_branch, int) and _point_branch >= 0 else "n/a"
            except Exception as _e:
                _point_disp = f"ERR:{_e}"
            print(
                f"[NavigatorLabel DEBUG] {_tag}: idx={idx} gi={_global_idx} "
                f"comboIndex={_combo_idx} comboData={_combo_data!r} comboText={_combo_text!r} "
                f"activeBranch={_active!r} selectedBranch={_branch_idx!r} "
                f"selectedRole={_branch_role!r} selectedDisplay={_branch_disp!r} "
                f"pointBranch={_point_branch!r} pointRole={_point_role!r} "
                f"pointDisplay={_point_disp!r} branchNameLabel={_label_txt!r}"
            )

        data = self.logic.getMeasurementAtIndex(idx, globalIdx=globalIdx)
        if not data:
            return

        # Get global index (idx + branch offset)
        # Use pre-computed globalIdx from localToGlobal (traversal-aware);
        # fall back to localToGlobal for single-branch / direct calls
        if globalIdx is None:
            globalIdx = self.logic.localToGlobal(idx)
        _nav_measure_probe("updateMeasurements:entry", None, globalIdx)

        total = self.logic.getNumPoints()
        self.pointIndexLabel.setText(f"{idx} / {total - 1}")
        self.distanceLabel.setText(f"{data['distance']:.2f} mm")
        self.coordLabel.setText(
            f"R {data['x']:.1f}, A {data['y']:.1f}, S {data['z']:.1f}"
        )

        curDiam = data["diameter"]
        if curDiam > 0:
            self.diameterLabel.setText(f"{curDiam:.2f} mm")
            self.radiusLabel.setText(
                f"{curDiam:.2f} mm"
            )  # FIX: was curDiam/2 (radius), now full diameter
        else:
            self.diameterLabel.setText("N/A (no model)")
            self.radiusLabel.setText("N/A")

        # Find which branch this point belongs to and show its min/max live.
        # In single-branch mode the combo selection is authoritative.  Many
        # anatomical starts are shared ostium/bifurcation points, so resolving
        # the current gi by raw point ownership can incorrectly report trunk or
        # a hidden fragment at slider position 0.
        _active_bi = getattr(self.logic, "activeBranch", -1)
        if _active_bi is not None and _active_bi >= 0:
            branchIdx = _active_bi
        else:
            _trav_branch_ids = getattr(self.logic, "_allBranchTraversalBranchIds", None)
            if (
                _trav_branch_ids
                and 0 <= idx < len(_trav_branch_ids)
                and isinstance(_trav_branch_ids[idx], int)
            ):
                branchIdx = _trav_branch_ids[idx]
            else:
                branchIdx = self.logic.getBranchForPoint(globalIdx)
            _bm_for_label = getattr(self.logic, "branchMeta", {})
            _role_for_label = _bm_for_label.get(branchIdx, {}).get("role", "")
            if _role_for_label == "iliac_fragment":
                _parent_bi = _bm_for_label.get(branchIdx, {}).get("fragment_of")
                if isinstance(_parent_bi, int) and _parent_bi >= 0:
                    print(
                        f"[NavigatorLabel DEBUG] resolve hidden iliac_fragment: "
                        f"pointBranch={branchIdx} -> parentBranch={_parent_bi}"
                    )
                    branchIdx = _parent_bi
            elif _role_for_label == "renal_fragment":
                _parent_bi = _bm_for_label.get(branchIdx, {}).get("fragment_of")
                if isinstance(_parent_bi, int) and _parent_bi >= 0:
                    print(
                        f"[NavigatorLabel DEBUG] resolve hidden renal_fragment: "
                        f"pointBranch={branchIdx} -> parentBranch={_parent_bi}"
                    )
                    branchIdx = _parent_bi
        _nav_measure_probe("updateMeasurements:resolved-branch", branchIdx, globalIdx)
        # ── [DEBUG] final truth — which branch does gi land in? ───────────
        print(f"\n[DEBUG updateMeasurementsAtIndex]")
        print(f"  idx={idx}  gi={globalIdx}  activeBranch={getattr(self.logic,'activeBranch','?')}")
        for _dbi, (_bs, _be) in enumerate(getattr(self.logic, "branches", [])):
            if _bs <= globalIdx < _be:
                _drole = getattr(self.logic, "branchMeta", {}).get(_dbi, {}).get("role", "?")
                print(f"  → gi belongs to bi={_dbi}  role={_drole!r}  range=[{_bs},{_be})")
                break
        else:
            print(f"  → gi={globalIdx} not found in any branch range")
        # ── end DEBUG ─────────────────────────────────────────────────────
        if branchIdx >= 0:
            bStats = self.logic.getBranchStats(branchIdx)
            # Get display name for current branch
            branchName = self.logic.getBranchDisplayName(branchIdx)
            _bm = getattr(self.logic, "branchMeta", {}).get(branchIdx, {})
            print(
                f"[BranchStats DEBUG] idx={idx} globalIdx={globalIdx}"
                f" → branchIdx={branchIdx}"
                f" role={_bm.get('role','?')!r}"
                f" lateral_label={_bm.get('lateral_label','?')!r}"
                f" → name={branchName!r}"
            )
            # In all-branches traversal mode, show the real anatomical branch
            # under the cursor instead of the traversal mode name.
            self.branchNameLabel.setText(branchName)
            _nav_measure_probe("updateMeasurements:after-set-branchNameLabel", branchIdx, globalIdx)
            self.branchLengthLabel.setText(
                f"{bStats['length']:.1f} mm  ({bStats['points']} pts)"
            )

            minD = bStats["min"]
            maxD = bStats["max"]
            avgD = bStats["avg"]

            # Highlight current diameter vs min/max
            minTxt = f"{minD:.2f} mm"
            maxTxt = f"{maxD:.2f} mm"
            if curDiam > 0 and abs(curDiam - minD) < 0.1:
                minTxt += "  ◀ HERE"
            if curDiam > 0 and abs(curDiam - maxD) < 0.1:
                maxTxt += "  ◀ HERE"

            self.branchMinLabel.setText(minTxt)
            self.branchMaxLabel.setText(maxTxt)
            self.branchAvgLabel.setText(f"{avgD:.2f} mm" if avgD > 0 else "N/A")

            # Also update global min/max labels
            self.minDiamLabel.setText(f"{minD:.2f} mm  ({branchName})")
            self.maxDiamLabel.setText(f"{maxD:.2f} mm  ({branchName})")

        # ── Finding warning at current point ──
        # If this point has been balloon-dilated, suppress the lesion label
        _pdmap = getattr(self.logic, "preDilationMap", {})
        if globalIdx in _pdmap:
            _pde = _pdmap[globalIdx]
            self.findingWarningLabel.setText(
                f"✅ BALLOONED  —  Ø{_pde['compressed_diam']:.1f}mm → Ø{_pde['balloon_diam']:.1f}mm"
            )
            self.findingWarningLabel.setStyleSheet(
                "font-weight:bold;font-size:13px;padding:4px;border-radius:4px;"
                "background:#e8f5e9;color:#1b5e20;"
            )
        elif hasattr(self.logic, "finding_type") and globalIdx < len(
            self.logic.finding_type
        ):
            ftype = self.logic.finding_type[globalIdx]
            ratio = (
                self.logic.pancake_ratios[globalIdx]
                if hasattr(self.logic, "pancake_ratios")
                else 1.0
            )
            minor = (
                self.logic.minor_axes[globalIdx]
                if hasattr(self.logic, "minor_axes")
                else 0.0
            )
            major = (
                self.logic.major_axes[globalIdx]
                if hasattr(self.logic, "major_axes")
                else 0.0
            )
            if ftype == 3:
                self.findingWarningLabel.setText(
                    f"🔴 ANEURYSM  —  {minor:.1f}mm diameter"
                )
                self.findingWarningLabel.setStyleSheet(
                    "font-weight:bold;font-size:13px;padding:4px;border-radius:4px;"
                    "background:#ffcccc;color:#8b0000;"
                )
            elif ftype == 2:
                self.findingWarningLabel.setText(
                    f"🟠 ECTASIA  —  {minor:.1f}mm ({ratio:.1f}× median)"
                )
                self.findingWarningLabel.setStyleSheet(
                    "font-weight:bold;font-size:13px;padding:4px;border-radius:4px;"
                    "background:#ffe0b2;color:#7d4e00;"
                )
            elif ftype == 1:
                self.findingWarningLabel.setText(
                    f"🟡 COMPRESSION  —  ratio {ratio:.1f}  "
                    f"({minor:.1f}mm × {major:.1f}mm)"
                )
                self.findingWarningLabel.setStyleSheet(
                    "font-weight:bold;font-size:13px;padding:4px;border-radius:4px;"
                    "background:#fff9c4;color:#5d4037;"
                )
            else:
                # Check if current point is in a collateral branch
                if hasattr(self.logic, "collaterals") and self.logic.collaterals:
                    cur_branch = self.logic.getBranchForPoint(globalIdx)
                    coll_match = next(
                        (
                            c
                            for c in self.logic.collaterals
                            if c["branchIdx"] == cur_branch
                        ),
                        None,
                    )
                    if coll_match:
                        self.findingWarningLabel.setText(
                            f"🩸 COLLATERAL  —  {coll_match['confidence']} confidence  "
                            f"({coll_match['angle_deg']}°, {coll_match['length_mm']:.0f}mm, "
                            f"max {coll_match['maxD']:.1f}mm)"
                        )
                        self.findingWarningLabel.setStyleSheet(
                            "font-weight:bold;font-size:13px;padding:4px;border-radius:4px;"
                            "background:#e0f7fa;color:#006064;"
                        )
                    else:
                        self.findingWarningLabel.setText("")
                        self.findingWarningLabel.setStyleSheet(
                            "font-weight:bold;font-size:13px;padding:4px;border-radius:4px;"
                        )
                else:
                    self.findingWarningLabel.setText("")
                    self.findingWarningLabel.setStyleSheet(
                        "font-weight:bold;font-size:13px;padding:4px;border-radius:4px;"
                    )
        else:
            self.findingWarningLabel.setText("")

        # ── Stent zone indicator ──────────────────────────────────────────
        plan = getattr(self.logic, "stentPlan", None)
        if plan:
            gp = plan["proxPt"]
            gd = plan["distPt"]
            if gp <= globalIdx <= gd:
                t = (globalIdx - gp) / max(gd - gp, 1)
                cur_r = (
                    plan["proxDiam"] / 2
                    + (plan["distDiam"] / 2 - plan["proxDiam"] / 2) * t
                )
                pos_pct = int(t * 100)
                self.findingWarningLabel.setText(
                    "STENT ZONE  "
                    + str(pos_pct)
                    + "% along  |  "
                    + "stent r="
                    + str(round(cur_r, 1))
                    + "mm"
                )
                self.findingWarningLabel.setStyleSheet(
                    "font-weight:bold;font-size:13px;padding:4px;border-radius:4px;"
                    "background:#FFF8E1;color:#E65100;"
                )

    def updateStats(self):
        stats = self.logic.getStats()
        if not stats:
            return
        if stats["min"] > 0:
            self.minDiamLabel.setText(
                f"{stats['min']:.2f} mm  (point {stats['min_idx']})"
            )
            self.maxDiamLabel.setText(
                f"{stats['max']:.2f} mm  (point {stats['max_idx']})"
            )
            self.avgDiamLabel.setText(f"{stats['avg']:.2f} mm")
        else:
            self.minDiamLabel.setText("N/A (no model)")
            self.maxDiamLabel.setText("N/A (no model)")
            self.avgDiamLabel.setText("N/A (no model)")
        self.totalLengthLabel.setText(
            f"{stats['total_length']:.2f} mm  ({stats['total_length']/10:.2f} cm)"
        )

    def onExport(self):
        path = qt.QFileDialog.getSaveFileName(None, "Save CSV", "", "CSV Files (*.csv)")
        if path:
            self.logic.exportToCSV(path)
            slicer.util.infoDisplay(f"Exported to:\n{path}")



# ── Slicer module-scanner guard ───────────────────────────────────────────────
class vessel_navigation_widget_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_navigation_widget_mixin"
            parent.hidden = True
