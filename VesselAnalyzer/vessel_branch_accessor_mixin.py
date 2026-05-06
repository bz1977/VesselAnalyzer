"""
vessel_branch_accessor_mixin.py — Read-only branch accessors, traversal helpers, and measurement queries.

Part of the VesselAnalyzer mixin decomposition.
These methods are mixed into VesselAnalyzerLogic via multiple inheritance:

    class VesselAnalyzerLogic(
        BranchAccessorMixin,
        ...
        ScriptedLoadableModuleLogic,
    ): ...

All methods use ``self`` normally — no changes to call sites required.
"""
# ruff: noqa  (this file is auto-extracted; formatting is inherited from VesselAnalyzer.py)


class BranchAccessorMixin:
    """Mixin: Read-only branch accessors, traversal helpers, and measurement queries."""

    def getBranchLabels(self):
        """Return branch labels for UI combos."""
        labels = []
        for bi, (s, e) in enumerate(self.branches):
            pts = e - s
            stats = self.getBranchStats(bi)
            lbl = (
                self.getBranchDisplayName(bi)
                + "  ("
                + str(pts)
                + " pts, "
                + str(round(stats["length"], 1))
                + "mm, "
                + "mean "
                + str(round(stats["avg"], 1))
                + "mm)"
            )
            labels.append(lbl)
        return labels


    def getNumPoints(self):
        """Return number of navigable points in the active branch or traversal."""
        if self.activeBranch >= 0:
            return self.getActiveBranchPointCount()
        traversal = getattr(self, "_allBranchTraversal", None)
        if traversal:
            return len(traversal)
        return len(self.points)


    def getBranchDisplayName(self, raw_idx):
        """Return display name for a branch.

        Trunk (index 0, or role=='trunk')  → 'Trunk'
        Iliac (role iliac_right/iliac_left, lateral_label set) → 'Left iliac' / 'Right iliac'
        Renal (role=='renal_vein', lateral_label set)          → 'Left Renal vein' / 'Right Renal vein'
        Other branches use their endpoint index so Branch N matches Endpoint N.
        """
        if raw_idx is None or raw_idx < 0:
            return "All branches"

        meta = getattr(self, "branchMeta", {}).get(raw_idx, {})
        role = meta.get("role", "")
        lat  = meta.get("lateral_label", "")  # 'Left' or 'Right'

        if raw_idx == 0 or role == "trunk":
            return "Trunk"

        if role in ("iliac_right", "iliac_left", "main") and lat:
            return f"{lat} Iliac"

        if role == "renal_vein" and lat:
            return f"{lat} Renal Vein"

        if role == "iliac_fragment":
            parent = meta.get("fragment_of")
            side = meta.get("lateral_label") or meta.get("iliac_role", "").replace("iliac_", "").title()
            return f"{side} Iliac Fragment" if side else f"Iliac Fragment {raw_idx}"

        ep_map = getattr(self, "branchEndpointMap", {})
        ep_idx = ep_map.get(raw_idx, raw_idx)
        return f"Branch {ep_idx + 1}"


    def getNumBranches(self):
        """Return number of branches."""
        return len(self.branches)


    def getNumBranchesTotal(self):
        """Return total branch count including hidden sub-branches."""
        return len(self.branches)


    def setActiveBranch(self, branchIdx):
        self.activeBranch = branchIdx


    def getActiveBranchPointCount(self):
        if self.activeBranch < 0 or self.activeBranch >= len(self.branches):
            return len(self.points)
        start = self.getBranchStartGi(self.activeBranch)
        # Respect user-specified end override (manualEndGi), else raw branch end
        meta = getattr(self, "branchMeta", {}).get(self.activeBranch, {})
        mend = meta.get("manualEndGi")
        end = (mend + 1) if mend is not None else self.branches[self.activeBranch][1]
        return max(1, end - start)


    def getBranchStartGi(self, branchIdx):
        """Return the anatomy-grounded start global index for a branch.

        Priority order:
          1. manualStartGi — user-specified coordinate snapped to nearest gi
             (set by setManualBranchRanges / the UI panel).  Highest authority.
          2. ostiumGi — anatomical ostium / bifurcation point (the correct
             navigator "branch start" the user sees and copies as coordinates).
          3. branches[bi][0] — raw graph topology start (fallback only).

        NOTE: stableStartGi is the post-flare diameter-measurement anchor and
        is intentionally NOT used here — it sits several mm distal to the
        anatomical ostium and would make the branch appear to start too far in.
        stableStartGi is used in _annotateTreeDiameters / getBranchStats only.

        Use this everywhere instead of branches[bi][0] when you need the
        clinical/anatomical branch start, not the graph topology start.
        """
        if branchIdx < 0 or branchIdx >= len(self.branches):
            return 0
        meta = getattr(self, "branchMeta", {}).get(branchIdx, {})
        # 1. User-specified manual override (highest authority)
        mgi = meta.get("manualStartGi")
        if mgi is not None:
            return mgi
        # 2. Anatomical ostium (bifurcation point / ostium detection result)
        ogi = meta.get("ostiumGi")
        if ogi is not None:
            return ogi
        return self.branches[branchIdx][0]


    def getActiveBranchOffset(self):
        if self.activeBranch < 0 or self.activeBranch >= len(self.branches):
            return 0
        # For the active branch, return the anatomical start (ostiumGi when available)
        return self.getBranchStartGi(self.activeBranch)


    def localToGlobal(self, localIdx):
        """Convert slider local index to global point index.
        For a single branch: add branch start offset.
        For All branches (-1): look up _allBranchTraversal so navigation
        follows branch order."""
        if self.activeBranch < 0:
            traversal = getattr(self, "_allBranchTraversal", None)
            # NavRobust: guard None traversal and out-of-range index — always
            # returns a valid integer so callers never receive None or raise.
            if traversal and 0 <= localIdx < len(traversal):
                return traversal[localIdx]
            # Fallback: clamp to valid points range rather than returning a raw
            # localIdx that could exceed len(self.points).
            n = len(self.points) if self.points else 0
            return max(0, min(localIdx, n - 1)) if n > 0 else 0
        # Single-branch mode: offset by anatomy-grounded start (stableStartGi
        # → ostiumGi → raw start), matching getActiveBranchOffset exactly so
        # the sphere lands on the correct point for every slider position.
        if 0 <= self.activeBranch < len(self.branches):
            return localIdx + self.getBranchStartGi(self.activeBranch)
        return localIdx


    def globalToTraversal(self, gi):
        """Convert a global point index to the slider (traversal) index.

        In 'All branches' mode the slider runs over _allBranchTraversal, so
        setting slider.value = gi is wrong — it must be set to the position
        of gi inside the traversal list.  Returns the first occurrence so the
        navigator always lands on the outbound pass (not the return leg).
        Falls back to nearest traversal entry if gi is absent (noise-gated
        branch points that were trimmed from the traversal).

        In single-branch mode the slider local index is simply gi minus the
        branch start offset.
        """
        if self.activeBranch < 0:
            traversal = getattr(self, "_allBranchTraversal", None)
            if traversal:
                try:
                    # Return the FIRST occurrence whose immediate successor in the
                    # traversal is numerically further from the trunk root (i.e. on
                    # the outbound pass of that branch), not the first absolute
                    # occurrence.  This matters when bif_gi appears multiple times
                    # (once at trunk_rev[0] and once as a ret_to_bif bridge after
                    # each iliac return leg): callers like onGoToMin / finding-jump
                    # want to land on the outbound pass, not trunk start.
                    first_idx = None
                    for _ti, _tgi in enumerate(traversal):
                        if _tgi == gi:
                            if first_idx is None:
                                first_idx = _ti  # record absolute first as fallback
                            # Accept this occurrence if the next step moves away
                            # from trunk (gi increases — outbound direction).
                            if _ti + 1 < len(traversal) and traversal[_ti + 1] > gi:
                                return _ti
                    # All occurrences were followed by a lower gi (return legs or
                    # end of traversal) — fall back to absolute first.
                    if first_idx is not None:
                        return first_idx
                    raise ValueError
                except ValueError:
                    # gi not in traversal — snap to nearest entry
                    return min(
                        range(len(traversal)), key=lambda i: abs(traversal[i] - gi)
                    )
        if 0 <= self.activeBranch < len(self.branches):
            return max(0, gi - self.getBranchStartGi(self.activeBranch))
        return gi


    def getBranchForPoint(self, globalIdx):
        """Return branch index that contains this global point index.

        Uses _rawBranches (the frozen pre-trim snapshot used by the traversal
        builder) when available, so that ostium-trimmed points (e.g. gi=457
        trimmed off branch 4's start when ostiumGi advanced to 458) are still
        found in their original branch.  Without this, getBranchForPoint returns
        -1 for trimmed-off gi values, causing the jump guard to incorrectly
        classify inter-branch traversal transitions as large spatial jumps and
        block slider navigation.
        """
        raw = getattr(self, "_rawBranches", None)
        branch_ranges = raw if raw else self.branches
        for i, (start, end) in enumerate(branch_ranges):
            if start <= globalIdx < end:
                # DEBUG: verify _rawBranches index matches self.branches index
                _self_bi = -1
                for _j, (_s, _e) in enumerate(self.branches):
                    if _s <= globalIdx < _e:
                        _self_bi = _j
                        break
                if raw is not None and _self_bi != i:
                    print(
                        f"[getBranchForPoint DEBUG] globalIdx={globalIdx}"
                        f" _rawBranches says bi={i} ({start}..{end})"
                        f" but self.branches says bi={_self_bi}"
                        f" → name from raw={self.getBranchDisplayName(i)!r}"
                        f"  name from branches={self.getBranchDisplayName(_self_bi)!r}"
                    )
                return i
        return -1


    def debugBranches(self):
        print(f"[VesselAnalyzer] Total branches: {len(self.branches)}")
        for i, (start, end) in enumerate(self.branches):
            bStats = self.getBranchStats(i)
            # Anatomical start = ostium (where branch genuinely separates from siblings)
            ost_gi = self.getBranchStartGi(i)
            p0 = self.points[ost_gi]  # anatomical start point
            p1 = self.points[end - 1]
            label = self.getBranchDisplayName(i)
            n = end - start
            n_skip = ost_gi - start  # junction points skipped before ostium
            print(
                f"  {label}: local pt 0-{n-1} ({n} pts, {bStats['length']:.1f}mm, "
                f"min={bStats['min']:.2f}mm, max={bStats['max']:.2f}mm, avg={bStats['avg']:.2f}mm)"
            )
            skip_note = f" [junction: skip {n_skip} pts]" if n_skip > 0 else ""
            print(
                f"    START local pt {ost_gi - start} (global gi{ost_gi}){skip_note}: "
                f"D={abs(p0[0])*2:.1f} A={p0[1]:.1f} S={p0[2]:.1f}"
            )
            print(
                f"    END   local pt {n-1} (global gi{end-1}): "
                f"D={abs(p1[0])*2:.1f} A={p1[1]:.1f} S={p1[2]:.1f}"
            )

        if self.diameters:
            print(f"[VesselAnalyzer] Mean diameter per branch:")
            for _mi, (_mbs, _mbe) in enumerate(self.branches):
                _ms = self.getBranchStats(_mi)
                if _ms["avg"] > 0:
                    print(
                        f"  {self.getBranchDisplayName(_mi)}: "
                        f"mean={_ms['avg']:.2f}mm  min={_ms['min']:.2f}mm  "
                        f"max={_ms['max']:.2f}mm  length={_ms['length']:.1f}mm"
                    )


    def getBranchStats(self, branchIdx):
        if branchIdx < 0 or branchIdx >= len(self.branches):
            return {"min": 0, "max": 0, "avg": 0, "length": 0, "points": 0}
        start, end = self.branches[branchIdx]
        bDists = self.distances[start:end]
        length = bDists[-1] - bDists[0] if len(bDists) > 1 else 0

        # ── Proximal exclusion zone ───────────────────────────────────────────
        # Raw `start` sits at the graph-topology boundary (trunk wall / carina).
        # At that location the rays hit *both* the branch wall and the adjacent
        # aortic wall, producing inflated "bleed" diameters.  We skip forward to
        # stableStartGi (flare-free) when available, then enforce a hard minimum
        # arc-distance exclusion zone regardless.
        #
        # For the trunk (bi=0) this only skips the tiny junction overlap (1–2 pts)
        # since stableStartGi is already correct there.  For side branches (renals)
        # it removes the 10–20 mm that are dominated by trunk-wall geometry.
        PROX_EXCL_MAX = 10.0  # mm: nominal proximal exclusion from ostium
        MAX_DIAM_RATIO = 0.70  # branch mean must be ≤ 70 % of trunk mean (safety cap)

        # Anatomy-grounded start (stableStartGi > ostiumGi > raw)
        anat_start = self.getBranchStartGi(branchIdx)

        # Branch arc length — needed to scale PROX_EXCL_MM so it cannot consume
        # more than 30% of a short branch (renals can be as short as 30 mm).
        _br_arc = 0.0
        if branchIdx > 0 and end > start + 1 and end - 1 < len(self.distances):
            _br_arc = self.distances[end - 1] - self.distances[start]
        PROX_EXCL_MM = (
            min(PROX_EXCL_MAX, max(0.0, _br_arc * 0.30))
            if _br_arc > 0
            else PROX_EXCL_MAX
        )

        # Walk forward from anat_start until arc distance ≥ PROX_EXCL_MM
        excl_start = anat_start
        if branchIdx > 0:  # trunk does not need extra exclusion
            arc_from_ost = 0.0
            for gi in range(anat_start, end - 1):
                if gi + 1 >= len(self.distances):
                    break
                arc_from_ost += self.distances[gi + 1] - self.distances[gi]
                if arc_from_ost >= PROX_EXCL_MM:
                    excl_start = gi + 1
                    break

        bDiams = [d for d in self.diameters[excl_start:end] if d > 0]

        # ── Outlier rejection ────────────────────────────────────────────────
        # Isolated spikes (e.g. max=28 mm in a 10 mm renal) come from rays
        # leaking through thin walls into adjacent vessels at one specific point.
        #
        # Two-pass rejection for side branches (renals / collaterals):
        #
        #   Pass A — distal-anchored hard cap:
        #     The OUTLIER_K=2.0 filter in _computeDiameters uses per-point r_min
        #     as its reference.  At the proximal end of a shallow-angle renal
        #     (Branch 4, 31.5°) the centerline still overlaps the IVC lumen, so
        #     r_min itself is already ~11 mm (IVC radius) — the cap becomes 22 mm
        #     and trunk-wall hits at 18–22 mm pass unchallenged.
        #
        #     Fix: compute a distal reference here (distal 55–85 % of bDiams,
        #     mirroring _mean_ref_d in _detectFindings) and reject anything above
        #     DISTAL_CAP_K × distal_ref.  DISTAL_CAP_K = 1.6 allows substantial
        #     eccentricity / tapering while rejecting the 2–3× trunk-wall spike.
        #
        #   Pass B — IQR spike filter (unchanged, runs after Pass A):
        #     Catches any remaining isolated noise spikes.
        #
        if branchIdx > 0 and len(bDiams) >= 8:
            # Distal reference: trimmed mean of middle-to-distal segment.
            _n = len(bDiams)
            _lo = int(_n * 0.55)
            _hi = int(_n * 0.85)
            _ref_vals = sorted(bDiams[_lo:_hi]) if _hi > _lo else []
            if len(_ref_vals) >= 3:
                _trim = max(1, len(_ref_vals) // 10)
                _distal_ref = sum(_ref_vals[_trim : -_trim or None]) / max(
                    1, len(_ref_vals) - 2 * _trim
                )
                DISTAL_CAP_K = 1.6
                _distal_cap = _distal_ref * DISTAL_CAP_K
                _filtered = [d for d in bDiams if d <= _distal_cap]
                if len(_filtered) >= max(4, int(_n * 0.50)):
                    # Only apply if we keep at least half the pts (safety net).
                    bDiams = _filtered

        if len(bDiams) >= 6:
            _sd = sorted(bDiams)
            _q1 = _sd[len(_sd) // 4]
            _q3 = _sd[(len(_sd) * 3) // 4]
            _iqr = _q3 - _q1
            _spike_ceil = _q3 + 1.5 * _iqr
            bDiams = [d for d in bDiams if d <= _spike_ceil]

        # ── Trunk mean (used only for the diameter cap on side branches) ─────
        trunk_mean = None
        if branchIdx > 0 and len(self.branches) > 0:
            _t_start, _t_end = self.branches[0]
            _t_anat = self.getBranchStartGi(0)
            _td = [d for d in self.diameters[_t_anat:_t_end] if d > 0]
            if _td:
                trunk_mean = sum(_td) / len(_td)

        # ── Mean of healthy zone: exclude compressed points ──────────────────
        # Healthy = diameter ≥ 75 % of the 75th-percentile value.
        mean_healthy = 0.0
        if bDiams:
            sorted_d = sorted(bDiams)
            p75 = sorted_d[int(len(sorted_d) * 0.75)]
            healthy_threshold = p75 * 0.75
            healthy_diams = [d for d in bDiams if d >= healthy_threshold]
            mean_healthy = (
                sum(healthy_diams) / len(healthy_diams)
                if healthy_diams
                else sum(bDiams) / len(bDiams)
            )

            # Safety cap: side-branch mean must not exceed MAX_DIAM_RATIO × trunk mean.
            # Prevents residual aortic-wall ray leakage from inflating renal stats.
            if branchIdx > 0 and trunk_mean and trunk_mean > 0:
                cap = trunk_mean * MAX_DIAM_RATIO
                if mean_healthy > cap:
                    mean_healthy = cap

        return {
            "min": min(bDiams) if bDiams else 0,
            "max": max(bDiams) if bDiams else 0,
            "avg": mean_healthy,  # mean of healthy zone only (proximal-excluded + capped)
            "length": length,
            "points": end - start,
            "confidence": self.branchMeta.get(branchIdx, {}).get("ostium_confidence"),
        }


    def getMeasurementAtIndex(self, idx, globalIdx=None):
        # Offset by branch start if a branch is active
        # Use pre-computed globalIdx when available (traversal-aware),
        # otherwise fall back to offset-based for single-branch mode
        realIdx = (
            globalIdx if globalIdx is not None else (idx + self.getActiveBranchOffset())
        )
        if realIdx < 0 or realIdx >= len(self.points):
            return None
        p = self.points[realIdx]
        # ── NavRobust: guard distances array bounds ───────────────────────────
        # distances is built alongside points but a partial loadCenterline failure
        # can leave it shorter than points.  All index accesses below are guarded
        # so an array-bounds error never blocks the navigator.
        _n_dist = len(self.distances) if self.distances else 0
        if realIdx >= _n_dist:
            # distances not yet populated for this index — return a safe stub
            return {
                "x": p[0],
                "y": p[1],
                "z": p[2],
                "distance": 0.0,
                "diameter": (
                    self.diameters[realIdx]
                    if self.diameters and realIdx < len(self.diameters)
                    else 0.0
                ),
            }
        if self.activeBranch < 0:
            # All-branches traversal mode: show cumulative arc from trav[0].
            # Cannot use getBranchForPoint to anchor distance — adjacent traversal
            # positions can belong to different branches (renal walk_down reuses
            # trunk gi values) causing the label to flip 700+mm at the same spot.
            traversal = getattr(self, "_allBranchTraversal", None)
            if traversal:
                _trav0 = traversal[0]
                # NavRobust: guard traversal[0] index against distances array length
                if 0 <= _trav0 < _n_dist:
                    dist_val = abs(self.distances[realIdx] - self.distances[_trav0])
                else:
                    dist_val = self.distances[realIdx]
            else:
                dist_val = self.distances[realIdx]
        else:
            branchStart = self.getActiveBranchOffset()
            # NavRobust: guard branchStart index against distances array length
            if 0 <= branchStart < _n_dist:
                dist_val = self.distances[realIdx] - self.distances[branchStart]
            else:
                dist_val = self.distances[realIdx]
        return {
            "x": p[0],
            "y": p[1],
            "z": p[2],
            "distance": abs(dist_val),
            "diameter": (
                self.diameters[realIdx]
                if self.diameters and realIdx < len(self.diameters)
                else 0.0
            ),
        }


    def getMinDiameterIndex(self):
        offset = self.getActiveBranchOffset()
        count = self.getActiveBranchPointCount()
        valid = [
            (d, i)
            for i, d in enumerate(self.diameters[offset : offset + count])
            if d > 0
        ]
        return min(valid, key=lambda x: x[0])[1] if valid else -1


    def getMaxDiameterIndex(self):
        offset = self.getActiveBranchOffset()
        count = self.getActiveBranchPointCount()
        valid = [
            (d, i)
            for i, d in enumerate(self.diameters[offset : offset + count])
            if d > 0
        ]
        return max(valid, key=lambda x: x[0])[1] if valid else -1


    def getStats(self):
        if not self.points:
            return None
        offset = self.getActiveBranchOffset()
        count = self.getActiveBranchPointCount()
        activeDiams = self.diameters[offset : offset + count]
        activeDists = self.distances[offset : offset + count]
        valid = [d for d in activeDiams if d > 0]
        totalLength = (activeDists[-1] - activeDists[0]) if len(activeDists) > 1 else 0
        if valid:
            return {
                "min": min(valid),
                "max": max(valid),
                "avg": sum(valid) / len(valid),
                "min_idx": self.getMinDiameterIndex(),
                "max_idx": self.getMaxDiameterIndex(),
                "total_length": totalLength,
            }
        return {
            "min": 0,
            "max": 0,
            "avg": 0,
            "min_idx": -1,
            "max_idx": -1,
            "total_length": totalLength,
        }


    def moveCrosshairToPoint(self, idx):
        if 0 <= idx < len(self.points):
            p = self.points[idx]
            # Move the crosshair position without triggering slice-panel jumps.
            # JumpSlicesToLocation causes Slicer to make slice widgets visible even
            # when the layout is single-3D, surfacing DICOM panels unexpectedly.
            try:
                crosshairNode = slicer.mrmlScene.GetFirstNodeByClass(
                    "vtkMRMLCrosshairNode"
                )
                if crosshairNode:
                    crosshairNode.SetCrosshairRAS(p[0], p[1], p[2])
            except Exception:
                pass



# ── Slicer module-scanner guard ───────────────────────────────────────────────
# Slicer auto-scans all .py files in the module folder and expects a class
# matching the filename.  This stub satisfies that requirement without
# registering as a real loadable module (no ScriptedLoadableModule base).
class vessel_branch_accessor_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_branch_accessor_mixin"
            parent.hidden = True  # hide from Slicer module list
