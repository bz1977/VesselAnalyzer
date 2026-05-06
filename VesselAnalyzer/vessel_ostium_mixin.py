"""
vessel_ostium_mixin.py — Ostium detection, refinement, and logistic-regression confidence scoring.

Part of the VesselAnalyzer mixin decomposition.
These methods are mixed into VesselAnalyzerLogic via multiple inheritance:

    class VesselAnalyzerLogic(
        OstiumMixin,
        ...
        ScriptedLoadableModuleLogic,
    ): ...

All methods use ``self`` normally — no changes to call sites required.
"""
# ruff: noqa  (this file is auto-extracted; formatting is inherited from VesselAnalyzer.py)

from vessel_geometry import geo_clamp

class OstiumMixin:
    """Mixin: Ostium detection, refinement, and logistic-regression confidence scoring."""

    def _applyArtifactCorrection(self):
        """Replace artifact-like diameter drops with local median estimates.

        For each branch with stenosisCandidate phenotype='artifact-like':
          - Walk the contiguous below-threshold run (gi_run_start..gi_run_end)
          - Replace each point's diameter with the median of clean flanking
            windows (FLANK_PTS on each side of the run, outside the run)
          - Log the correction so it is auditable

        This corrects self.diameters in-place so all downstream consumers
        (getBranchStats, _annotateTreeDiameters re-run if called, stent
        sizing, report generation) see physiologic values rather than the
        artifact minimum.

        Only fires for 'artifact-like' phenotype — 'compression-like' and
        'focal' findings are preserved as clinically meaningful.
        """
        if not getattr(self, "branchMeta", None) or not self.diameters:
            return

        FLANK_PTS = 8  # pts on each side to sample for the replacement median
        MIN_FLANK = 3  # minimum pts needed for a reliable flank estimate

        for bi, meta in self.branchMeta.items():
            sc = meta.get("stenosisCandidate")
            if not sc or sc.get("phenotype") != "artifact-like":
                continue

            gi_start = sc.get("gi_run_start")
            gi_end = sc.get("gi_run_end")
            if gi_start is None or gi_end is None or gi_end < gi_start:
                continue

            # Branch bounds
            if bi >= len(self.branches):
                continue
            bs, be = self.branches[bi]

            # Proximal flank: FLANK_PTS pts immediately before the run
            _prox_lo = max(bs, gi_start - FLANK_PTS)
            _prox_hi = gi_start
            prox_vals = [
                self.diameters[i]
                for i in range(_prox_lo, _prox_hi)
                if i < len(self.diameters) and self.diameters[i] > 1.0
            ]

            # Distal flank: FLANK_PTS pts immediately after the run
            _dist_lo = gi_end + 1
            _dist_hi = min(be, gi_end + 1 + FLANK_PTS)
            dist_vals = [
                self.diameters[i]
                for i in range(_dist_lo, _dist_hi)
                if i < len(self.diameters) and self.diameters[i] > 1.0
            ]

            flank_vals = prox_vals + dist_vals
            if len(flank_vals) < MIN_FLANK:
                print(
                    f"[ArtifactCorr] Branch {bi}: insufficient flank data "
                    f"({len(flank_vals)} pts) — skipping correction"
                )
                continue

            flank_vals.sort()
            replacement = flank_vals[len(flank_vals) // 2]  # median

            n_corrected = 0
            for gi in range(gi_start, gi_end + 1):
                if gi < len(self.diameters) and self.diameters[gi] > 0.5:
                    self.diameters[gi] = replacement
                    n_corrected += 1

            run_mm = sc.get("run_mm", 0)
            print(
                f"[ArtifactCorr] Branch {bi}: corrected {n_corrected} pts "
                f"(gi {gi_start}–{gi_end}, {run_mm:.1f}mm) "
                f"→ {sc['min_diam']:.1f}mm replaced with {replacement:.1f}mm "
                f"(flank median, {len(flank_vals)} pts)"
            )

            # Update the candidate to record correction was applied
            sc["corrected_to"] = round(replacement, 2)
            sc["n_corrected"] = n_corrected

        return any(
            meta.get("stenosisCandidate", {}).get("n_corrected", 0) > 0
            for meta in self.branchMeta.values()
        )

    # ── Manual branch range override ─────────────────────────────────────────


    def setManualBranchRanges(self, ranges_dict):
        """Store manual start/end coordinate overrides for branches.

        Args:
            ranges_dict: dict of branchIdx (int) →
                         {"start": (x, y, z), "end": (x, y, z)}
                         Coordinates in RAS mm (same as self.points).
                         Either "start" or "end" may be omitted to leave that
                         end at its auto-detected value.

        Example:
            logic.setManualBranchRanges({
                0: {"start": (34.0, 167.0, 1969.9), "end": (17.9, 151.0, 1780.1)},
                1: {"start": (22.3, 149.0, 1779.5), "end": (-48.3, 172.3, 1672.2)},
            })

        Call _applyManualRanges() afterwards (or re-run loadCenterline) to
        resolve the coordinates into gi indices.
        """
        self.manualBranchRanges = dict(ranges_dict)
        # If points are already loaded, apply immediately
        if self.points and self.branches:
            self._applyManualRanges()


    def _applyManualRanges(self):
        """Resolve self.manualBranchRanges coordinates → gi indices in branchMeta.

        For each branch with a manual override, finds the gi within that
        branch's raw point range whose 3D coordinate is nearest to the
        specified start/end coordinate, then stores:
          branchMeta[bi]["manualStartGi"] = nearest gi to start coord
          branchMeta[bi]["manualEndGi"]   = nearest gi to end coord

        getBranchStartGi() checks manualStartGi first (highest priority).
        getActiveBranchPointCount() checks manualEndGi first.

        Logs: [ManualRange] bi=N role=X  start gi=Y (dist=Z.Zmm)  end gi=W (dist=V.Vmm)
        """
        if not self.manualBranchRanges:
            return
        if not self.points or not self.branches:
            print("[ManualRange] No points/branches yet — deferring to next load")
            return

        import math as _math

        def _nearest_gi_in_range(coord, bs, be):
            """Return gi in [bs, be) nearest to coord (x,y,z)."""
            cx, cy, cz = coord
            best_gi = bs
            best_d2 = float("inf")
            for gi in range(bs, min(be, len(self.points))):
                px, py, pz = self.points[gi]
                d2 = (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_gi = gi
            return best_gi, _math.sqrt(best_d2)

        for bi, spec in self.manualBranchRanges.items():
            if bi < 0 or bi >= len(self.branches):
                print(
                    f"[ManualRange] bi={bi} out of range ({len(self.branches)} branches) — skipped"
                )
                continue
            bs, be = self.branches[bi]
            meta = self.branchMeta.setdefault(bi, {})
            role = meta.get("role", "?")
            log_parts = [f"[ManualRange] bi={bi} role={role}"]

            if "start" in spec and spec["start"] is not None:
                gi_s, dist_s = _nearest_gi_in_range(spec["start"], bs, be)
                meta["manualStartGi"] = gi_s
                log_parts.append(f"start gi={gi_s} (dist={dist_s:.1f}mm)")

            if "end" in spec and spec["end"] is not None:
                gi_e, dist_e = _nearest_gi_in_range(spec["end"], bs, be)
                meta["manualEndGi"] = gi_e
                log_parts.append(f"end gi={gi_e} (dist={dist_e:.1f}mm)")

            print("  ".join(log_parts))


    def _suppressRejectedOstia(self):
        """Post-confidence pass: suppress ostiumGi for REJECT-grade branches.

        _refineOstia cannot do this because _computeOstiumConfidence runs after
        it.  This method runs after both and applies the suppression rule:

            if confidence.grade == 'REJECT':
                ostiumGi = None   (no ostium reported)

        Main branches and already-suppressed branches are skipped.
        Branches whose ostium_method is 'attachment_short' (short non-renal,
        kept as-is by Gate 2) are also exempt — their ogi is the flow-norm
        attachment point, which is reliable regardless of confidence grade.

        Logs: [OstiumSuppress] Branch N: REJECT (eff=X) — ostium suppressed
        """
        if not getattr(self, "branchMeta", None):
            return
        EXEMPT_METHODS = {"attachment_short", "suppressed_reject"}
        for bi, meta in self.branchMeta.items():
            if bi == 0:
                continue
            role = meta.get("role", "side_branch")
            if role in ("main", "iliac_right", "iliac_left"):
                continue
            if meta.get("ostium_suppressed"):
                continue
            method = meta.get("ostium_method", "")
            if method in EXEMPT_METHODS:
                continue
            oc = meta.get("ostium_confidence", {})
            if not isinstance(oc, dict):
                continue
            grade = oc.get("grade", "")
            if grade == "REJECT":
                # v229: Do NOT suppress the ostium marker.  A visible low-
                # confidence marker is more useful than silence, especially for
                # short or horizontal branches (e.g. Branch5) where the snap
                # geometry is imperfect but the general location is still
                # anatomically meaningful.
                #
                # Previous behaviour: ostiumGi=None, ostium_p3=None → no marker
                # at all, so the branch appeared correctly placed at ogi which
                # is inside the IVC flare zone (too proximal).
                #
                # New behaviour: ostiumGi and ostium_p3 are left intact;
                # ostium_p3_lowconf is populated so the debug renderer can show
                # a distinct dim-yellow marker instead of the normal green one.
                # The diameter pipeline and self.branches[bi] are unaffected.
                eff = oc.get("effective_score", oc.get("score", 0.0))
                raw = oc.get("score", eff)
                pen = oc.get("flag_penalty", 0.0)
                spread = oc.get("uncertainty", 0.0)
                flags = oc.get("flags", [])
                # Dominant flag: pick the flag with the highest individual
                # penalty weight, or fall back to the first flag listed.
                _FLAG_WEIGHTS = {
                    "gradient_weak": 0.04,
                    "topology_uncertain": 0.08,
                    "curvature_flat": 0.04,
                    "geometry_weak": 0.06,
                    "area_low": 0.05,
                    "lumen_unstable": 0.07,
                    "zone_distant": 0.06,
                    "divergence_low": 0.05,
                }
                _dominant_flag = (
                    max(flags, key=lambda f: _FLAG_WEIGHTS.get(f, 0.0))
                    if flags
                    else "none"
                )
                # Thresholds (venous mode — from _computeOstiumConfidence)
                _REJECT_THRESH = 0.35
                _LOW_THRESH = 0.25
                _state = "LOW_CONF" if eff >= _LOW_THRESH else "VERY_LOW"
                print(
                    f"[OstiumQuality] branch={bi} "
                    f"eff={eff:.3f} "
                    f"raw={raw:.3f} "
                    f"pen={pen:.3f} "
                    f"spread={spread:.3f} "
                    f"reject_thresh={_REJECT_THRESH} "
                    f"state={_state} "
                    f"dominant_flag={_dominant_flag} "
                    f"all_flags={','.join(flags) if flags else 'none'}"
                )
                print(
                    f"[OstiumSuppress] Branch {bi}: REJECT (eff={eff:.3f}) "
                    f"— keeping marker as low-confidence (dim-yellow)"
                )
                # Copy ostium_p3 → ostium_p3_lowconf so renderer uses dim colour.
                _p3 = meta.get("ostium_p3")
                if _p3 is None:
                    _ogi_lc = meta.get("ostiumGi")
                    if _ogi_lc is not None and 0 <= _ogi_lc < len(self.points):
                        _p3 = tuple(float(x) for x in self.points[_ogi_lc])
                if _p3 is not None:
                    meta["ostium_p3_lowconf"] = _p3
                # Leave ostiumGi / ostium_p3 / self.branches[bi] intact.


    def _computeStableStarts(self):
        """Compute and store stableStartGi for every non-trunk branch.

        This is the flare-zone walk that previously only ran inside
        _detectFindings (triggered by "Find Lesions" button click).
        Moving it here ensures stableStartGi is always set before
        _annotateTreeDiameters uses it for the proximal diameter window.

        For each branch: walk forward from ostiumGi until the diameter
        profile stabilises (flare-walk pass 1), then enforce a minimum
        arc-distance skip (hard floor pass 2).  If neither advances past
        ostiumGi by ≥ FLARE_COMMIT_MIN_PTS, use a 10mm fallback skip so
        prox sampling is never contaminated by the junction zone.

        Already-set stableStartGi values (e.g. from a prior _detectFindings
        run that advanced further) are not overwritten — this function only
        writes if the value is currently missing or equals ostiumGi.
        """
        if not getattr(self, "branches", None) or not getattr(self, "diameters", None):
            return

        import math

        FLARE_THR = 0.08
        FLARE_PERSIST = 3
        DIAM_WINDOW_PTS = 3
        FLARE_COMMIT_MIN_PTS = 2
        FLARE_FALLBACK_MM = 10.0

        # Trunk root Z for the skip-above-root gate
        trunk_root_z = (
            self.points[self.branches[0][0]][2]
            if self.branches and self.branches[0][0] < len(self.points)
            else 1e9
        )

        for bi, (s, e) in enumerate(self.branches):
            if bi == 0:
                continue
            # IliacZGateFix: iliac branches are exempt — their trimmed `s` == ostiumGi
            # == cranial apex (max-Z) which can exceed trunk_root_z and silently skip
            # the branch.  Use _rawBranches start Z for non-iliac branches.
            _bm_role_sgi = self.branchMeta.get(bi, {}).get("role", "")
            if _bm_role_sgi not in ("main", "iliac_left", "iliac_right"):
                _raw_branches_sgi = getattr(self, "_rawBranches", self.branches)
                _raw_s_sgi = (
                    _raw_branches_sgi[bi][0] if bi < len(_raw_branches_sgi) else s
                )
                branch_start_z = (
                    self.points[_raw_s_sgi][2] if _raw_s_sgi < len(self.points) else 0.0
                )
                if branch_start_z > trunk_root_z + 10.0:
                    continue

            meta = self.branchMeta.get(bi, {})
            ogi = meta.get("ostiumGi", None)
            scan_from = ogi if (ogi is not None and s <= ogi < e) else s

            # Skip if stableStartGi already advanced beyond ostium (prior run)
            existing_sgi = meta.get("stableStartGi")
            if (
                existing_sgi is not None
                and existing_sgi > scan_from + FLARE_COMMIT_MIN_PTS
            ):
                continue

            _branch_len = e - scan_from
            MAX_FLARE_PTS = min(20, max(5, int((e - s) * 0.25)))

            # Reference diameter: distal 55–85% of branch (fractional, no hard floors)
            ref_lo = scan_from + int(_branch_len * 0.55)
            ref_hi = scan_from + int(_branch_len * 0.85)
            ref_lo = min(ref_lo, e - 5)
            ref_hi = min(ref_hi, e)
            if ref_hi - ref_lo < 5:
                ref_lo = max(scan_from, e - max(10, _branch_len // 3))
                ref_hi = e
            _ref_vals = [
                self.diameters[i]
                for i in range(ref_lo, ref_hi)
                if i < len(self.diameters) and self.diameters[i] > 1.0
            ]
            if _ref_vals:
                _ref_vals.sort()
                _trim = max(1, len(_ref_vals) // 10)
                _mean_ref_d = sum(_ref_vals[_trim : -_trim or None]) / max(
                    1, len(_ref_vals) - 2 * _trim
                )
            else:
                _mean_ref_d = max(
                    (
                        self.diameters[i]
                        for i in range(scan_from, e)
                        if i < len(self.diameters) and self.diameters[i] > 1.0
                    ),
                    default=10.0,
                )

            # Local smoothed diameter
            def _smooth(gi):
                lo = max(s, gi - DIAM_WINDOW_PTS)
                hi = min(e - 1, gi + DIAM_WINDOW_PTS)
                vals = [
                    self.diameters[i]
                    for i in range(lo, hi + 1)
                    if i < len(self.diameters) and self.diameters[i] > 1.0
                ]
                if not vals:
                    return 0.0
                vals.sort()
                trim = max(1, len(vals) // 4)
                return sum(vals[trim : -trim or None]) / max(1, len(vals) - 2 * trim)

            # Pass 1: diameter-walk
            stable_start = scan_from
            if _mean_ref_d > 1.0:
                flare_ceil = _mean_ref_d * (1.0 + FLARE_THR)
                consec_stable = 0
                for _fi in range(scan_from, min(scan_from + MAX_FLARE_PTS, e - 1)):
                    d_curr = _smooth(_fi)
                    if d_curr <= 0:
                        continue
                    if d_curr <= flare_ceil:
                        consec_stable += 1
                        if consec_stable >= FLARE_PERSIST:
                            stable_start = max(scan_from, _fi - FLARE_PERSIST + 1)
                            break
                    else:
                        consec_stable = 0

            # Pass 2: hard arc floor
            _branch_arc_mm = (
                self.distances[e - 1] - self.distances[s]
                if e - 1 < len(self.distances) and s < len(self.distances)
                else (e - s) * 1.35
            )
            FLARE_HARD_MM = min(45.0, max(8.0, _branch_arc_mm * 0.30))
            _arc_acc = 0.0
            _hard_floor = scan_from
            for _hfi in range(scan_from, min(e - 1, scan_from + MAX_FLARE_PTS + 30)):
                if _hfi >= len(self.points) - 1:
                    break
                _arc_acc += (
                    sum(
                        (self.points[_hfi + 1][k] - self.points[_hfi][k]) ** 2
                        for k in range(3)
                    )
                    ** 0.5
                )
                if _arc_acc >= FLARE_HARD_MM:
                    _hard_floor = _hfi + 1
                    break
            stable_start = max(stable_start, _hard_floor)

            # Commit
            _advanced = stable_start >= scan_from + FLARE_COMMIT_MIN_PTS
            if _advanced:
                self.branchMeta.setdefault(bi, {})["stableStartGi"] = stable_start
                print(
                    f"[StableStart] Branch {bi}: gi={stable_start} "
                    f"(+{stable_start - scan_from} pts from ostium, "
                    f"ref_diam={_mean_ref_d:.1f}mm)"
                )
            else:
                # Fallback: hard 10mm arc skip
                _fb_arc = 0.0
                _fb_gi = scan_from
                for _fbi in range(
                    scan_from, min(e - 1, scan_from + MAX_FLARE_PTS + 30)
                ):
                    if _fbi >= len(self.points) - 1:
                        break
                    _fb_arc += (
                        sum(
                            (self.points[_fbi + 1][k] - self.points[_fbi][k]) ** 2
                            for k in range(3)
                        )
                        ** 0.5
                    )
                    if _fb_arc >= FLARE_FALLBACK_MM:
                        _fb_gi = _fbi + 1
                        break
                if _fb_gi > scan_from:
                    self.branchMeta.setdefault(bi, {})["stableStartGi"] = _fb_gi
                    print(
                        f"[StableStart] Branch {bi}: fallback gi={_fb_gi} "
                        f"(+{_fb_gi - scan_from} pts, {_fb_arc:.1f}mm arc skip, "
                        f"ref_diam={_mean_ref_d:.1f}mm)"
                    )


    def _stabilizeRenalOstia(self):
        """Anchor renal-vein ostia at the trunk/branch contact boundary.

        Problem: renal branches tagged 'renal_vein' have their ostium
        placed by the generic snappedBifPt logic, which works well for
        iliacs at the primary bifurcation but leaves renal ostia at the
        topological graph node rather than the true trunk wall exit.
        The result is ostiumGi landing 1–5mm too proximal (still inside
        the IVC lumen), the proximity-dominant RefineOstium scorer then
        drags it even further toward the bifurcation (low prox penalty),
        and OstiumConfidence gives MEDIUM with 'topology_uncertain'.

        Fix: for each renal branch, walk forward from the current ostiumGi
        along the branch centerline and find the last point that is still
        within CONTACT_THRESH_MM of the trunk centerline KD-tree.  That
        contact boundary is the anatomical exit point — where the renal
        lumen separates from the IVC wall.  Set ostiumGi to this point
        and flag the branch as 'renal_stabilized' so _refineOstia skips
        it (the proximity scorer would otherwise fight the new placement).

        Safety guards:
          - Z sanity: stabilized ostium must be within 120mm of the
            primary bifurcation Z (prevents runaway on pathological graphs)
          - No contact found → no change (branch already clear of trunk)
          - Short contact runs (<2 pts) with no further advance → no change
          - Never moves ostiumGi beyond stableStartGi
        """
        if not getattr(self, "branches", None) or not getattr(self, "points", None):
            return

        CONTACT_THRESH_MM = 2.5  # mm — point is "touching" trunk if closer than this
        Z_SANITY_MM = 120.0  # mm — stabilized ostium must be near bif Z

        # Build trunk KD-tree once from trunk centerline points
        try:
            from scipy.spatial import cKDTree as _cKDTree
        except ImportError:
            return  # scipy unavailable — skip silently

        t_s, t_e = self.branches[0]
        trunk_pts = [self.points[gi] for gi in range(t_s, t_e) if gi < len(self.points)]
        if len(trunk_pts) < 2:
            return
        import numpy as _np

        trunk_arr = _np.array(trunk_pts)
        trunk_kdtree = _cKDTree(trunk_arr)

        # Reference Z: primary bifurcation (distal end of trunk)
        bif_z = (getattr(self, "bifurcationPoint", None) or [None, None, None])[2]
        if bif_z is None and trunk_pts:
            bif_z = trunk_pts[-1][2]  # trunk distal end as fallback

        for bi, (s, e) in enumerate(self.branches):
            if bi == 0:
                continue
            meta = self.branchMeta.get(bi, {})
            if meta.get("role") not in ("renal_vein", "renal_fragment"):
                continue

            ogi = meta.get("ostiumGi")
            sgi = meta.get("stableStartGi")
            if ogi is None:
                continue
            # Don't move past stableStartGi (that's already post-flare territory)
            limit = sgi if (sgi is not None and sgi > ogi) else min(ogi + 20, e - 1)

            # Walk forward from ostiumGi; collect indices that are in contact with trunk
            contact_indices = []
            for gi in range(ogi, limit):
                if gi >= len(self.points):
                    break
                pt = self.points[gi]
                d, _ = trunk_kdtree.query(pt)
                if d < CONTACT_THRESH_MM:
                    contact_indices.append(gi)
                else:
                    break  # first point clear of trunk = boundary

            if not contact_indices:
                print(
                    f"[RenalStabilizer] Branch {bi}: no trunk contact at ostium — kept gi={ogi}"
                )
                continue

            new_gi = contact_indices[-1]  # last touching point = exit boundary

            # Z sanity check
            if bif_z is not None and abs(self.points[new_gi][2] - bif_z) > Z_SANITY_MM:
                print(
                    f"[RenalStabilizer] Branch {bi}: Z sanity failed "
                    f"(|{self.points[new_gi][2]:.1f} - {bif_z:.1f}| > {Z_SANITY_MM}mm) — kept gi={ogi}"
                )
                continue

            # Require at least a 1-pt advance to bother updating
            if new_gi <= ogi:
                print(
                    f"[RenalStabilizer] Branch {bi}: contact_len={len(contact_indices)} pts, "
                    f"no advance — kept gi={ogi}"
                )
                continue

            # Commit
            self.branchMeta.setdefault(bi, {})["ostiumGi"] = new_gi
            self.branchMeta[bi]["ostiumGiRefined"] = new_gi
            self.branchMeta[bi].setdefault("flags", set()).add("renal_stabilized")
            # Update self.branches so branch[bi][0] == new_gi
            self.branches[bi] = (new_gi, e)
            print(
                f"[RenalStabilizer] Branch {bi}: contact_len={len(contact_indices)} pts "
                f"→ ostium_gi={new_gi} (was {ogi}, +{new_gi - ogi} pts, stabilized)"
            )

    # ── Phase 1 ostium helpers ────────────────────────────────────────────────


    def _computeBranchTangents(self, pts, smooth_half=2):
        """Central-difference tangents with ±smooth_half smoothing window.

        pts         : list of (x,y,z) tuples (branch point sequence)
        smooth_half : half-window in points (2 pts ≈ 2-3mm at typical spacing)

        Returns list of (tx,ty,tz) unit vectors, same length as pts.
        """
        import math

        n = len(pts)
        tangents = []
        for i in range(n):
            lo = max(0, i - smooth_half)
            hi = min(n - 1, i + smooth_half)
            dx = pts[hi][0] - pts[lo][0]
            dy = pts[hi][1] - pts[lo][1]
            dz = pts[hi][2] - pts[lo][2]
            mag = math.sqrt(dx * dx + dy * dy + dz * dz)
            if mag > 1e-9:
                tangents.append((dx / mag, dy / mag, dz / mag))
            else:
                tangents.append((0.0, 0.0, 1.0))
        return tangents


    def _computeCrossSectionArea(
        self, surface_polydata, center_pt, tangent, clip_radius_mm=12.0
    ):
        """Plane-cut cross-sectional area via shoelace on ordered 2-D ring.

        1. Clips surface to a sphere of clip_radius_mm around center_pt to
           prevent adjacent branches from contaminating the cut.
        2. Cuts the clipped mesh with a plane ⟂ tangent at center_pt.
        3. Orders the resulting polyline with vtkStripper.
        4. Picks the largest strip by perimeter (not point count — noise
           fragments can have many points but tiny extent).
        5. Projects the strip to the local 2-D plane and runs the shoelace
           formula.

        Returns area in mm², or None if the cut is degenerate (fewer than
        6 ordered points, open contour, or area < 0.5 mm²).
        """
        try:
            import vtk
            import math

            cx, cy, cz = float(center_pt[0]), float(center_pt[1]), float(center_pt[2])
            tx, ty, tz = float(tangent[0]), float(tangent[1]), float(tangent[2])
            mag = math.sqrt(tx * tx + ty * ty + tz * tz)
            if mag < 1e-9:
                return None
            tx, ty, tz = tx / mag, ty / mag, tz / mag

            # ── 1. Local clip (sphere) ──────────────────────────────────────
            sphere = vtk.vtkSphere()
            sphere.SetCenter(cx, cy, cz)
            sphere.SetRadius(clip_radius_mm)

            clipper = vtk.vtkClipPolyData()
            clipper.SetClipFunction(sphere)
            clipper.SetInputData(surface_polydata)
            clipper.InsideOutOn()
            clipper.Update()
            clipped = clipper.GetOutput()
            if clipped.GetNumberOfPoints() < 6:
                return None

            # ── 2. Plane cut ────────────────────────────────────────────────
            plane = vtk.vtkPlane()
            plane.SetOrigin(cx, cy, cz)
            plane.SetNormal(tx, ty, tz)

            cutter = vtk.vtkCutter()
            cutter.SetCutFunction(plane)
            cutter.SetInputData(clipped)
            cutter.Update()

            # ── 3. Order into polylines ─────────────────────────────────────
            stripper = vtk.vtkStripper()
            stripper.SetInputData(cutter.GetOutput())
            stripper.Update()
            contour = stripper.GetOutput()
            if contour.GetNumberOfPoints() < 6:
                return None

            # ── 4. Pick largest strip by perimeter ─────────────────────────
            best_ids = None
            best_peri = 0.0
            pts3d = contour.GetPoints()
            lines = contour.GetLines()
            id_list = vtk.vtkIdList()
            lines.InitTraversal()
            while lines.GetNextCell(id_list):
                n_ids = id_list.GetNumberOfIds()
                if n_ids < 6:
                    continue
                peri = 0.0
                for j in range(n_ids - 1):
                    p0 = pts3d.GetPoint(id_list.GetId(j))
                    p1 = pts3d.GetPoint(id_list.GetId(j + 1))
                    dx2 = p1[0] - p0[0]
                    dy2 = p1[1] - p0[1]
                    dz2 = p1[2] - p0[2]
                    peri += math.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2)
                if peri > best_peri:
                    best_peri = peri
                    best_ids = [id_list.GetId(j) for j in range(n_ids)]

            if best_ids is None:
                return None

            # ── 5. Orthonormal basis in the cut plane ───────────────────────
            # Pick ref vector not parallel to tangent
            if abs(tx) < 0.9:
                rx, ry, rz = 1.0, 0.0, 0.0
            else:
                rx, ry, rz = 0.0, 1.0, 0.0
            # x_ax = ref × tangent
            xax_x = ry * tz - rz * ty
            xax_y = rz * tx - rx * tz
            xax_z = rx * ty - ry * tx
            xm = math.sqrt(xax_x * xax_x + xax_y * xax_y + xax_z * xax_z)
            xax_x /= xm
            xax_y /= xm
            xax_z /= xm
            # y_ax = tangent × x_ax
            yax_x = ty * xax_z - tz * xax_y
            yax_y = tz * xax_x - tx * xax_z
            yax_z = tx * xax_y - ty * xax_x

            # ── 6. Project to 2-D and run shoelace ─────────────────────────
            pts2d = []
            for idx in best_ids:
                p = pts3d.GetPoint(idx)
                vx = p[0] - cx
                vy = p[1] - cy
                vz = p[2] - cz
                pts2d.append(
                    (
                        vx * xax_x + vy * xax_y + vz * xax_z,
                        vx * yax_x + vy * yax_y + vz * yax_z,
                    )
                )

            area = 0.0
            n2 = len(pts2d)
            for i in range(n2):
                x1, y1 = pts2d[i]
                x2, y2 = pts2d[(i + 1) % n2]
                area += x1 * y2 - x2 * y1
            area = abs(area) * 0.5
            return area if area > 0.5 else None

        except Exception:
            return None  # vtk unavailable or cut degenerate


    def _computeRingNormal(
        self, surface_polydata, center_pt, tangent, clip_radius_mm=10.0
    ):
        """Fit a plane normal to the vtkCutter cross-section ring at center_pt.

        This is the ring-based perpendicularity signal for ostium detection:
        the sphere navigator (cutting plane) is perpendicular to the vessel
        wall when the ring's own best-fit plane normal aligns with the branch
        tangent.  This is more robust than per-vertex surface normals because:
          - It averages over the entire contour (not a single nearest vertex)
          - Mesh normal artifacts at the IVC/renal junction don't affect it
          - It directly reflects what the navigator cross-section shows

        Algorithm:
          1. Sphere-clip + vtkCutter (same as _computeCrossSectionArea)
          2. Pick largest strip by perimeter
          3. PCA on ring point cloud → smallest singular vector = plane normal
          4. Flip normal to agree with tangent sign convention
          5. Return (ring_normal, ellipticity, n_ring_pts) or (None, None, 0)

        ellipticity = major_axis / minor_axis of the ring bounding ellipse.
          < 1.3 → near-circular (perpendicular cut) ✓
          > 2.0 → strongly elliptical (oblique cut through IVC junction) ✗
        """
        try:
            import vtk
            import math

            try:
                import numpy as np
            except ImportError:
                return None, None, 0

            cx, cy, cz = float(center_pt[0]), float(center_pt[1]), float(center_pt[2])
            tx, ty, tz = float(tangent[0]), float(tangent[1]), float(tangent[2])
            mag = math.sqrt(tx * tx + ty * ty + tz * tz)
            if mag < 1e-9:
                return None, None, 0
            tx, ty, tz = tx / mag, ty / mag, tz / mag

            # ── 1. Sphere clip ──────────────────────────────────────────────
            sphere = vtk.vtkSphere()
            sphere.SetCenter(cx, cy, cz)
            sphere.SetRadius(clip_radius_mm)
            clipper = vtk.vtkClipPolyData()
            clipper.SetClipFunction(sphere)
            clipper.SetInputData(surface_polydata)
            clipper.InsideOutOn()
            clipper.Update()
            clipped = clipper.GetOutput()
            if clipped.GetNumberOfPoints() < 6:
                return None, None, 0

            # ── 2. Plane cut ────────────────────────────────────────────────
            plane = vtk.vtkPlane()
            plane.SetOrigin(cx, cy, cz)
            plane.SetNormal(tx, ty, tz)
            cutter = vtk.vtkCutter()
            cutter.SetCutFunction(plane)
            cutter.SetInputData(clipped)
            cutter.Update()

            stripper = vtk.vtkStripper()
            stripper.SetInputData(cutter.GetOutput())
            stripper.Update()
            contour = stripper.GetOutput()
            if contour.GetNumberOfPoints() < 6:
                return None, None, 0

            # ── 3. Pick largest strip by perimeter ──────────────────────────
            best_ids = None
            best_peri = 0.0
            pts3d = contour.GetPoints()
            lines = contour.GetLines()
            id_list = vtk.vtkIdList()
            lines.InitTraversal()
            while lines.GetNextCell(id_list):
                n_ids = id_list.GetNumberOfIds()
                if n_ids < 6:
                    continue
                peri = 0.0
                for j in range(n_ids - 1):
                    p0 = pts3d.GetPoint(id_list.GetId(j))
                    p1 = pts3d.GetPoint(id_list.GetId(j + 1))
                    peri += math.sqrt(
                        (p1[0] - p0[0]) ** 2
                        + (p1[1] - p0[1]) ** 2
                        + (p1[2] - p0[2]) ** 2
                    )
                if peri > best_peri:
                    best_peri = peri
                    best_ids = [id_list.GetId(j) for j in range(n_ids)]

            if best_ids is None or len(best_ids) < 6:
                return None, None, 0

            # ── 4. PCA on ring point cloud ──────────────────────────────────
            ring_pts = np.array([pts3d.GetPoint(idx) for idx in best_ids], dtype=float)
            n_ring = len(ring_pts)
            centre = ring_pts.mean(axis=0)
            _, S, Vt = np.linalg.svd(ring_pts - centre, full_matrices=False)
            # Smallest singular vector = best-fit plane normal
            ring_normal = Vt[-1]
            # Flip to agree with tangent direction
            t_vec = np.array([tx, ty, tz])
            if np.dot(ring_normal, t_vec) < 0:
                ring_normal = -ring_normal

            # ── 5. Ellipticity from singular values ─────────────────────────
            # S[0] ≥ S[1] ≥ S[2].  In-plane spread → S[0], S[1].
            # S[2] ≈ 0 (all points in a plane).
            # Ellipticity ≈ S[0] / max(S[1], 1e-3).
            ellipticity = float(S[0]) / max(float(S[1]), 1e-3)

            return ring_normal.tolist(), ellipticity, n_ring

        except Exception:
            return None, None, 0


    def _refineOstia(self):
        """Refine ostiumGi using a multi-point window with cross-sectional area.

        Phase 1 upgrade over the previous single-point proximity-dominated
        scorer.  Key changes:

        1. Window scanning  — scans every candidate gi in a clamped window
           [ostiumGi − BACK_MM, stableStartGi] instead of picking one point.

        2. Cross-sectional area gradient (weight 0.40, PRIMARY signal)
           vtkCutter + vtkStripper + shoelace on the 2-D projected ring.
           True lumen opening outperforms diameter alone in venous anatomy
           where walls are thin and diameter gradients are shallow.

        3. Diameter gradient (weight 0.25) — retained from previous version
           as secondary corroboration.  Uses self.diameters[gi] directly
           (global gi space, already smoothed).

        4. Topology prior (weight 0.20) — exponential decay from ogi (the
           topological / stabilizer anchor).  Prevents area signal from
           drifting far from the expected anatomical location.

        5. Stability score (weight 0.15) — diameter CV over 6 pts distal of
           gi.  High stability = lumen is cylindrical = past the flare zone.

        Renal branches
        ─────────────
        Previously skipped after _stabilizeRenalOstia.  Now refined inside a
        tight ±RENAL_BACK_MM window centred on the stabilizer result.  The
        stabilizer is the prior; area + stability find the exact exit point.

        Confidence 2.0
        ──────────────
        confidence = mean(window_scores) − std(window_scores)
        A high stable peak = real ostium.  A noisy plateau = uncertainty.
        Stored in branchMeta[bi]['ostium_confidence_v2'] alongside the
        per-component breakdown for the winning gi.

        Guard conditions (unchanged)
        ─────────────────────────────
        - Trunk (bi==0) always skipped.
        - Result always clamped to [ogi, stableStartGi].
        - stableStartGi never modified.
        - Commits branchMeta['ostiumGiRefined'], ['ostiumGi'],
          ['ostium_method']='scored_v2' and self.branches[bi][0].
        """
        import math

        if not getattr(self, "branches", None) or not getattr(self, "diameters", None):
            return

        # ── Surface polydata (full mesh, used for area cutting) ───────────────
        # segModelNode = open-boundary segmentation surface (correct mesh for
        # ring-normal detection — has real vessel wall geometry at branch takeoffs).
        # modelNode    = VMTK centerline tube (wrong — ring cuts on a tube always
        #                produce near-circular rings regardless of anatomy).
        # Prefer segModelNode; fall back to modelNode only if absent.
        _seg_node = getattr(self, "segModelNode", None)
        _model_node = _seg_node if _seg_node else getattr(self, "modelNode", None)
        _surface_pd = _model_node.GetPolyData() if _model_node else None

        # ── Surface vertex arrays for per-branch ostium snapping ─────────────
        # Strategy: snap ostium marker to the nearest surface vertex that is
        # classified as belonging to the BRANCH (not the trunk).
        #
        # Root cause of previous cranial placement: snapping to nearest trunk-
        # surface vertex from detect_gi found the IVC wall face at that Z level,
        # which is the cranial/proximal face of the opening — not the ostial rim.
        #
        # Fix: use a Voronoi-like assignment (same logic as classifySurfaceBranches)
        # to label every surface vertex by its nearest branch CL.  For branch bi,
        # collect only vertices assigned to bi.  The nearest such vertex to
        # detect_gi is on the branch side of the color boundary — exactly the
        # point where the surface transitions from IVC (green) to branch (cyan/orange).
        #
        # _surf_all_v  : np.array of all surface vertex coords  (N×3)
        # _surf_branch_assignment : np.array of branch index per vertex (N,)
        # _surf_kd_per_branch[bi] : cKDTree of vertices assigned to branch bi
        _surf_all_v = None
        _surf_branch_assignment = None
        _surf_kd_per_branch = {}  # bi → (cKDTree, pts_array)
        # Keep trunk KDTree too as fallback
        _trunk_surf_kd = None
        _trunk_surf_pts = None
        try:
            import numpy as np
            from scipy.spatial import cKDTree as _CKD

            if _surface_pd is not None and _surface_pd.GetNumberOfPoints() > 0:
                _surf_all_v = np.array(
                    [
                        _surface_pd.GetPoint(i)
                        for i in range(_surface_pd.GetNumberOfPoints())
                    ],
                    dtype=float,
                )

                if len(_surf_all_v) > 0:
                    # Build one flat CL array tagging each point with its branch index
                    _cl_pts = []
                    _cl_tags = []
                    for _bi, (_bs, _be) in enumerate(self.branches):
                        for _g in range(_bs, _be):
                            if _g < len(self.points):
                                _cl_pts.append(self.points[_g])
                                _cl_tags.append(_bi)
                    _cl_pts_arr = np.array(_cl_pts, dtype=float)
                    _cl_tags_arr = np.array(_cl_tags, dtype=int)

                    if len(_cl_pts_arr) >= 2:
                        _cl_kd = _CKD(_cl_pts_arr)
                        _, _nearest_cl = _cl_kd.query(_surf_all_v)
                        _surf_branch_assignment = _cl_tags_arr[_nearest_cl]

                        # Build per-branch KDTrees
                        for _bi in range(len(self.branches)):
                            _mask = _surf_branch_assignment == _bi
                            _bi_pts = _surf_all_v[_mask]
                            if len(_bi_pts) >= 4:
                                _surf_kd_per_branch[_bi] = (_CKD(_bi_pts), _bi_pts)

                        # Trunk fallback KDTree (bi==0 vertices)
                        if 0 in _surf_kd_per_branch:
                            _trunk_surf_kd, _trunk_surf_pts = _surf_kd_per_branch[0]

        except Exception:
            _surf_all_v = None
            _surf_branch_assignment = None
            _surf_kd_per_branch = {}
            _trunk_surf_kd = None
            _trunk_surf_pts = None

        # ── Scoring window constants ──────────────────────────────────────────
        BACK_MM = 8.0  # mm to scan proximal of ogi for main branches
        RENAL_BACK_MM = 5.0  # tighter window for renal (stabilizer is reliable)
        # LOOK is set adaptively per-branch below (min 8, ~8% of branch pts)
        LOOK_MIN = 8  # absolute floor for finite-difference window
        STAB_WIN = 6  # pts for stability CV check (distal of gi)
        CURV_MM = 18.0  # mm forward for curvature angle

        # ── Unified Ostium Score: 4-signal adaptive weights ───────────────────
        # Weights are set per-branch below based on branch length.
        # Short branches (<60mm): lateral separation dominates (angle unreliable
        #   in the short proximal zone), stability gates chaotic junctions.
        # Long branches (≥60mm): angle becomes reliable and gets more weight.
        #
        # Signal definitions:
        #   W_LAT   – normalised lateral separation rate (d_lat = lat[gi] - lat[gi-1])
        #             PRIMARY for short branches: "branch is peeling away from trunk"
        #   W_ANG   – angle between branch tangent (delayed by ANGLE_OFFSET_MM) and
        #             trunk direction; DELAYED to skip bifurcation contamination
        #   W_STAB  – direction stability = dot(dir[gi], dir[gi+delta])
        #             ~1 = past the chaotic junction zone; low = still at bifurcation
        #   W_PROX  – anti-proximal penalty = exp(-arc_from_ogi / PROX_SCALE_MM)
        #             prevents picking too early (fixes the proximal-bias bug)
        #
        # Retained legacy signals (area, diam, topology) contribute secondary
        # corroboration but are de-weighted relative to the 4 primary signals.
        W_AREA = 0.08  # cross-sectional area gradient (secondary corroboration)
        W_DIAM = 0.05  # diameter gradient (secondary corroboration)
        W_TOPO = 0.10  # topology prior (exp decay from ogi — soft anchor only)
        W_CURV = 0.05  # centerline curvature (kept for backward compat, minor role)
        # v2 retained signals
        W_DROP = 0.10  # diameter drop vs trunk (trunk-influence indicator)
        W_CONS = 0.07  # distal consistency (lumen settled to stable calibre)

        # Unified 4 primary signals (weights set per-branch below)
        # Placeholders — overwritten in the per-branch block.
        W_LAT = 0.0
        W_ANG = 0.0
        W_STAB_DIR = 0.0  # direction stability weight (distinct from old W_STAB)
        W_PROX = 0.0

        # Parameters for the 4 primary signals
        ANGLE_OFFSET_MM = 12.0  # skip this far before measuring angle (clears dome)
        STAB_DELTA_MM = 8.0  # forward distance for stability dot product
        PROX_SCALE_MM = 10.0  # exp decay scale for anti-proximal penalty
        LAT_NORM_MM = 5.0  # d_lat this large → lat_sc = 1.0
        ANG_NORM_DEG = 45.0  # angle this large → ang_sc = 1.0

        # Local-peak scan parameters
        PEAK_THRESHOLD = 0.35  # minimum score to qualify as a peak
        PEAK_LOOKBACK = 3  # must beat all candidates in the previous N pts

        # EXT_MM: how far beyond stableStartGi to keep searching (v2 key insight)
        EXT_MM = 20.0  # default; widened to 30mm for renal below

        # Renal-specific bonus added ON TOP of composite (outside weight budget)
        RENAL_LAT_W = 0.15  # lateral-offset bonus for renal branches
        RENAL_ALIGN_W = 0.10  # trunk-orthogonality bonus for renal branches

        # Normalisers (legacy signals)
        AREA_NORM = 0.30  # 30% symmetric area increase = full score
        DIAM_NORM = 0.15  # 15% relative diameter increase = full score
        CURV_NORM = 45.0  # degrees; 45° bend = full curvature score
        TOPO_TAU = 15.0  # half-weight at ~15 pts from ogi (wider = less proximity bias)

        def _arc_mm(gi_a, gi_b):
            """Arc distance between two global gi values via self.distances."""
            if gi_a < len(self.distances) and gi_b < len(self.distances):
                return abs(self.distances[gi_b] - self.distances[gi_a])
            return abs(gi_b - gi_a) * 1.4  # rough fallback ~1.4mm/pt

        def _mm_to_pts(mm, gi_ref, e):
            """Convert mm to approximate point count using local arc spacing."""
            if gi_ref + 1 < min(e, len(self.distances)):
                spacing = (
                    self.distances[min(gi_ref + 5, e - 1)] - self.distances[gi_ref]
                ) / max(1, min(5, e - gi_ref - 1))
                spacing = max(spacing, 0.3)
            else:
                spacing = 1.4
            return max(1, int(round(mm / spacing)))

        def _get_d(gi):
            """Safe diameter lookup."""
            if 0 <= gi < len(self.diameters) and self.diameters[gi] > 0:
                return self.diameters[gi]
            return 0.0

        for bi, (s, e) in enumerate(self.branches):
            if bi == 0:
                continue

            meta = self.branchMeta.get(bi, {})
            role = meta.get("role", "side_branch")
            is_renal = "renal_stabilized" in meta.get("flags", set()) or role in (
                "renal_vein",
                "renal_fragment",
            )
            is_main = role in ("main", "iliac_right", "iliac_left")
            ogi = meta.get("ostiumGi")
            sgi = meta.get("stableStartGi")

            if ogi is None or sgi is None:
                print(f"[RefineOstium] Branch {bi}: ogi or sgi missing — skipped")
                continue

            # ── Gate 1: Main branches (iliacs) — snap to wall, do NOT refine ─
            # Main branches have a clear bifurcation node as their anchor;
            # refinement over-shifts them.  But ostiumPt is still the CL medial-
            # axis point from the dome walk — ~5-7mm medial to the wall in R.
            # IliacSurfaceSnap runs here (inside _refineOstia) where the Voronoi
            # surface KDTrees are already built and use the correct segmentation
            # surface.  Candidate pool: iliac-classified verts (NOT trunk —
            # trunk surface at ostium Z is the shared carina face).
            # ostiumGi / stableStartGi / self.branches are NOT changed.
            if is_main:
                print(
                    f"[RefineOstium] Branch {bi}: main branch ({role}) — skipping refinement, running IliacSurfaceSnap"
                )
                try:
                    import numpy as _np_iss

                    # _ogi_pt: always use the branch CL point at ostiumGi.
                    # For CaudalDepartureSnap, meta['ostium'] is a trunk CL
                    # point — using it as the radial origin collapses the
                    # direction (trunk→trunk = zero vector).  _ogi_pt is
                    # always on the iliac CL, giving a valid lateral offset
                    # from the IVC axis regardless of which snap path ran.
                    _ogi_pt = np.array(self.points[ogi], dtype=float)
                    _ostium_p = meta.get("ostium")
                    _iss_pt = None
                    _iss_d = 0.0
                    _iss_lbl = "CL fallback"
                    # Trunk CL array for radial direction
                    _ts0, _te0 = self.branches[0]
                    _itr_pts = np.array(
                        [
                            self.points[g]
                            for g in range(_ts0, _te0)
                            if g < len(self.points)
                        ],
                        dtype=float,
                    )
                    _itr_zs = _itr_pts[:, 2]
                    # Radial direction: trunk CL at ogi Z → ogi, XY plane only.
                    # If _ogi_pt coincides with the bif node (Right Iliac
                    # CaudalDepartureSnap: ogi=bif node), the XY vector
                    # collapses to zero.  Walk forward up to 8 pts into the
                    # branch until XY separation from trunk > 2mm.
                    _isnap_z = float(_ogi_pt[2])
                    _itr_idx = int(np.argmin(np.abs(_itr_zs - _isnap_z)))
                    _itr_pt = _itr_pts[_itr_idx]
                    _irad = _ogi_pt - _itr_pt
                    _irad[2] = 0.0
                    _irad_len = float(np.linalg.norm(_irad))
                    if _irad_len < 2.0:
                        _bs_gi, _be_gi = self.branches[bi]
                        for _fwd in range(1, min(9, _be_gi - ogi)):
                            _fwd_pt = np.array(self.points[ogi + _fwd], dtype=float)
                            _fwd_tri = int(np.argmin(np.abs(_itr_zs - _fwd_pt[2])))
                            _fwd_rad = _fwd_pt - _itr_pts[_fwd_tri]
                            _fwd_rad[2] = 0.0
                            if float(np.linalg.norm(_fwd_rad)) > 2.0:
                                _irad = _fwd_rad
                                _irad_len = float(np.linalg.norm(_irad))
                                _isnap_z = float(_fwd_pt[2])
                                print(
                                    f"[RefineOstium] bi={bi} radial degenerate at ogi "
                                    f"— walked +{_fwd} pts, len={_irad_len:.1f}mm"
                                )
                                break
                    # bifurcationPoint for proximity selection
                    _ibif = np.array(
                        getattr(self, "bifurcationPoint", self.points[ogi]), dtype=float
                    )
                    # Primary: iliac-classified Voronoi verts
                    if bi in _surf_kd_per_branch:
                        _ikd, _ipts = _surf_kd_per_branch[bi]
                        # Z-band around ogi Z (not ostiumPt Z for CaudalDepartureSnap)
                        _IZ_BAND = 6.0  # slightly wider: iliac dome is shallow
                        _iz_mask = np.abs(_ipts[:, 2] - _isnap_z) <= _IZ_BAND
                        _icand = _ipts[_iz_mask] if np.any(_iz_mask) else _ipts
                        if _irad_len > 1e-3:
                            _irad_dir = _irad / _irad_len
                            _ivecs = _icand - _ogi_pt
                            _idists = np.linalg.norm(_ivecs, axis=1)
                            _ilens = np.maximum(_idists, 1e-6)
                            _idots = (_ivecs / _ilens[:, None]) @ _irad_dir
                            # Select: positive lateral dot (correct hemisphere)
                            # AND nearest to _ogi_pt — the rim vertex closest
                            # to the CL ostium point (not nearest-to-bif).
                            _pos_mask = _idots > 0.0
                            if np.any(_pos_mask):
                                _pcand_pts = _icand[_pos_mask]
                                _pcand_d = _idists[_pos_mask]
                                _iidx = int(np.argmin(_pcand_d))
                                _iss_pt = _pcand_pts[_iidx]
                                _iss_d = float(_pcand_d[_iidx])
                                _iss_lbl = (
                                    f"iliac_surf_proxrim bi={bi} "
                                    f"dot={float(_idots[_pos_mask][_iidx]):.2f} "
                                    f"n_pos={int(np.sum(_pos_mask))} "
                                    f"n_cand={len(_icand)} "
                                    f"Z_band={_IZ_BAND}mm"
                                )
                            else:
                                # All candidates on wrong side — nearest fallback
                                _iidx = int(np.argmin(_idists))
                                _iss_pt = _icand[_iidx]
                                _iss_d = float(_idists[_iidx])
                                _iss_lbl = (
                                    f"iliac_surf_nearest bi={bi} (no pos-dot cands)"
                                )
                        else:
                            _id2, _ii2 = _ikd.query(_ogi_pt)
                            _iss_pt = _ipts[_ii2]
                            _iss_d = float(_id2)
                            _iss_lbl = f"iliac_surf_nearest bi={bi} (degenerate radial)"
                    # Cap at 20mm
                    if _iss_pt is not None and _iss_d > 20.0:
                        print(
                            f"[RefineOstium] bi={bi} IliacSurfaceSnap: "
                            f"snap_d={_iss_d:.1f}mm > 20mm — rejected"
                        )
                        _iss_pt = None
                    if _iss_pt is not None:
                        _wall = tuple(float(x) for x in _iss_pt)
                        meta["ostium"] = _wall
                        meta["ostium_p3"] = _wall
                        _cl_log = [round(float(x), 1) for x in _ogi_pt]
                        _wp_log = [round(float(x), 1) for x in _wall]
                        print(
                            f"[RefineOstium] bi={bi} IliacSurfaceSnap: "
                            f"surface={_iss_lbl} "
                            f"CL_pt={_cl_log} wall_pt={_wp_log} "
                            f"snap_d={_iss_d:.1f}mm"
                        )
                        # SurfClassAlignFix: update surf_classify_ogi_local so
                        # classifySurfaceBranches starts the branch Voronoi
                        # competition at the CL point nearest to the wall vertex
                        # Z, not at first_k which may be caudal to the marker.
                        # This ensures the surface color boundary overlaps the
                        # green ostium marker rather than starting distal to it.
                        try:
                            _wall_z = float(_iss_pt[2])
                            _bs_gi2, _be_gi2 = self.branches[bi]
                            _br_pts2 = [
                                self.points[g]
                                for g in range(_bs_gi2, _be_gi2)
                                if g < len(self.points)
                            ]
                            if _br_pts2:
                                _br_zs2 = _np_iss.array([p[2] for p in _br_pts2])
                                _surf_ogi_new = int(
                                    _np_iss.argmin(_np_iss.abs(_br_zs2 - _wall_z))
                                )
                                # Never push it past existing ogi_local — only adjust if
                                # the new value is MORE proximal (smaller index) so we
                                # include more of the branch in the competition.
                                _old_surf_ogi = meta.get("surf_classify_ogi_local")
                                if (
                                    _old_surf_ogi is None
                                    or _surf_ogi_new < _old_surf_ogi
                                ):
                                    meta["surf_classify_ogi_local"] = _surf_ogi_new
                                    print(
                                        f"[RefineOstium] bi={bi} SurfClassAlignFix: "
                                        f"surf_classify_ogi_local {_old_surf_ogi} → {_surf_ogi_new} "
                                        f"(wall_Z={_wall_z:.1f}mm)"
                                    )
                        except Exception as _sca_err:
                            print(
                                f"[RefineOstium] bi={bi} SurfClassAlignFix failed: {_sca_err}"
                            )
                    else:
                        print(
                            f"[RefineOstium] bi={bi} IliacSurfaceSnap: "
                            f"no iliac verts in KDTree — CL fallback"
                        )
                except Exception as _iss_err:
                    import traceback

                    print(f"[RefineOstium] bi={bi} IliacSurfaceSnap failed: {_iss_err}")
                    traceback.print_exc()
                continue

            # ── (Gate 3 — REJECT suppression — runs in _suppressRejectedOstia) ─

            # ── Geometric attachment detector (replaces length-based Gate 2) ──
            #
            # Ostium = FIRST point where BOTH conditions hold simultaneously:
            #
            #   (A) Perpendicularity: surface normal at i ⟂ branch tangent
            #       perp = 1 - |dot(n_surface, v_branch)| > PERP_THRESH
            #       Physical meaning: surface wraps around the branch opening
            #       → the wall is no longer running parallel to the branch axis
            #       → we are past the shared trunk/branch wall region.
            #
            #   (B) Separation gradient: branch is moving away from trunk
            #       dd = d_trunk(i+1) - d_trunk(i) > SEP_THRESH
            #       Physical meaning: the branch centerline is detaching from
            #       the trunk wall → we have crossed into independent lumen.
            #
            # Stability: require (A) to hold for STABLE_N consecutive points
            # before committing, to reject noise spikes.
            #
            # This is purely local geometry — no branch-length term anywhere.
            # Works identically on 20mm collaterals and 200mm renals.
            #
            # Fallback: if no qualifying point found within the search window,
            # the original ogi is preserved and the composite scorer below runs.

            PERP_THRESH = 0.80  # |dot(n,v)| < 0.20 = surface ⟂ branch (20° tolerance)
            SEP_THRESH = 0.3  # mm per step minimum separation increase
            STABLE_N = 2  # consecutive perp-passing points before commit
            GEO_MAX_MM = 35.0  # max arc to search from ogi (keeps it local)
            GEO_RENAL_EXT_MM = (
                15.0  # extra arc allowed past sgi for near-orthogonal branches
            )
            # Near-orthogonal branches (renal veins, ~90° takeoff): the stabilizer
            # advances ogi to the wall-separation point, so ogi ≈ sgi and the
            # normal scan window [ogi, sgi] is ~12 pts of still-ambiguous junction
            # geometry.  Two adjustments:
            #   1. Extend _geo_end past sgi by GEO_RENAL_EXT_MM so the detector
            #      can scan into clean branch lumen where the ring is truly round.
            #   2. Use a tighter sphere clip radius (6 mm vs 10 mm) to exclude
            #      adjacent IVC wall that bleeds into the 10 mm sphere when the
            #      renal centerline runs inside or adjacent to the IVC.
            _clip_r = 6.0 if is_renal else 10.0

            # Build trunk KD-tree (subsampled ≤60 pts) for separation distance
            _trunk_kd_geo = None
            try:
                import numpy as np
                from scipy.spatial import cKDTree as _CKD

                _ts, _te = self.branches[0]
                _step_g = max(1, (_te - _ts) // 60)
                _tpts = np.array(
                    [
                        self.points[_g]
                        for _g in range(_ts, _te, _step_g)
                        if _g < len(self.points)
                    ],
                    dtype=float,
                )
                if len(_tpts) >= 2:
                    _trunk_kd_geo = _CKD(_tpts)
            except Exception:
                _trunk_kd_geo = None

            def _d_trunk_geo(gi):
                if _trunk_kd_geo is None or gi >= len(self.points):
                    return 0.0
                try:
                    import numpy as np

                    d, _ = _trunk_kd_geo.query(np.array(self.points[gi], dtype=float))
                    return float(d)
                except Exception:
                    return 0.0

            # Precompute tangents for this branch
            branch_pts = self.points[s:e]
            tangents = self._computeBranchTangents(branch_pts, smooth_half=2)

            # Compute GEO_MAX_MM in points from ogi.
            # For renal (near-orthogonal) branches extend past sgi by GEO_RENAL_EXT_MM
            # so the detector reaches clean branch lumen where the ring is circular.
            # For all other branches hard-cap at sgi as before.
            _arc_ref = 0.0
            _geo_end = ogi
            for _gi in range(ogi, min(e - 1, ogi + 200)):
                if _gi + 1 < len(self.distances):
                    _arc_ref += abs(self.distances[_gi + 1] - self.distances[_gi])
                if _arc_ref >= GEO_MAX_MM:
                    _geo_end = _gi
                    break
            else:
                _geo_end = min(e - 2, ogi + 200)
            if is_renal:
                # Extend up to GEO_RENAL_EXT_MM past sgi (still bounded by branch end)
                _renal_ext_arc = 0.0
                _geo_end_ext = sgi
                for _gi in range(sgi, min(e - 1, sgi + 200)):
                    if _gi + 1 < len(self.distances):
                        _renal_ext_arc += abs(
                            self.distances[_gi + 1] - self.distances[_gi]
                        )
                    if _renal_ext_arc >= GEO_RENAL_EXT_MM:
                        _geo_end_ext = _gi
                        break
                else:
                    _geo_end_ext = min(e - 2, sgi + 50)
                _geo_end = min(max(_geo_end, _geo_end_ext), e - 2)
            else:
                _geo_end = min(_geo_end, sgi)  # never scan past stableStart

            # ── Ring-normal perpendicularity detector ─────────────────────────
            # Primary signal: fit a plane to the vtkCutter cross-section ring.
            # When the ring plane normal aligns with the branch tangent, the
            # navigator sphere is perpendicular to the vessel wall — this is
            # the ostium definition.
            #
            # Secondary signal: ring ellipticity.  A perpendicular cut through
            # a cylinder produces a circle (ellipticity ≈ 1.0).  An oblique
            # cut at the IVC/renal junction produces an ellipse (> 1.5).
            #
            # Thresholds:
            #   RING_DOT_THRESH   : dot(ring_normal, tangent) > threshold
            #                       (ring plane ⟂ branch axis)
            #   RING_ELIP_THRESH  : ellipticity < threshold (near-circular)
            #   SEP_THRESH        : d_trunk increasing (branch separating)
            #   STABLE_N          : consecutive passes before commit

            RING_DOT_THRESH = 0.85  # ring_normal · tangent > 0.85 (≈32° tolerance)
            RING_ELIP_THRESH = 1.6  # major/minor < 1.6 = acceptably round
            # SEP_THRESH and STABLE_N already defined above
            # (PERP_THRESH from v192 vertex-normal approach is superseded by RING_DOT_THRESH)

            # Backward scan from sgi — find the proximal boundary of the
            # perpendicular/separating zone.
            #
            # Root cause of the forward-scan failure (B4, B5):
            #   Forward scan starts at ogi which is inside the IVC flare zone.
            #   At oblique IVC/branch junctions the vtkCutter ring is still
            #   near-circular (thin IVC wall → small sphere clip picks up the
            #   branch wall cleanly even at the junction), so RING_DOT_THRESH
            #   fires immediately at ogi → marker placed inside the shared lumen.
            #
            # Fix: scan BACKWARD from sgi (the flare-walk's clean-lumen anchor).
            #   sgi is guaranteed to be in stable branch tissue where the ring IS
            #   genuinely perpendicular.  Walking proximally, the FIRST point where
            #   EITHER signal fails (ring becomes oblique OR branch stops separating)
            #   marks the proximal boundary of the junction zone.  The gi one step
            #   DISTAL of that failure is the true ostium.
            #
            # Selection logic:
            #   1. Scan backward from sgi to ogi.
            #   2. For each gi compute ring_dot, ellipticity, and separation dd.
            #   3. Stop at the first gi where _perp_ok is False OR _sep_ok is False.
            #   4. Commit gi+1 (one step distal = still inside the good zone).
            #   5. If no failure found (whole window is good), commit ogi
            #      (the proximal anchor is already in clean tissue).
            #   6. Clamp result to [ogi, sgi].
            _geo_ostium = None
            _best_dot = 0.0
            _best_elip = 999.0
            _best_n_ring = 0

            # Collect ring results for the whole window first, backward
            _ring_results = []  # list of (gi, perp_ok, sep_ok, ring_dot, elip, n_ring)
            for _gi in range(ogi, _geo_end + 1):
                _li = _gi - s
                if _li < 0 or _li >= len(tangents):
                    continue
                _v = tangents[_li]

                _perp_ok = False
                _ring_dot = 0.0
                _elip = 999.0
                _n_ring = 0

                if _surface_pd is not None and _gi < len(self.points):
                    _rn, _elip_raw, _n_ring = self._computeRingNormal(
                        _surface_pd, self.points[_gi], _v, clip_radius_mm=_clip_r
                    )
                    if _rn is not None:
                        _ring_dot = abs(
                            _rn[0] * _v[0] + _rn[1] * _v[1] + _rn[2] * _v[2]
                        )
                        _elip = _elip_raw if _elip_raw is not None else 999.0
                        _perp_ok = (
                            _ring_dot > RING_DOT_THRESH and _elip < RING_ELIP_THRESH
                        )

                _d_here = _d_trunk_geo(_gi)
                _d_next = _d_trunk_geo(min(_gi + 1, e - 1))
                _sep_ok = (_d_next - _d_here) > SEP_THRESH

                _ring_results.append(
                    (_gi, _perp_ok, _sep_ok, _ring_dot, _elip, _n_ring)
                )

            # Backward walk: find the proximal boundary of the good zone
            if _ring_results:
                _commit_idx = None
                for _idx in range(len(_ring_results) - 1, -1, -1):
                    _gi, _perp_ok, _sep_ok, _ring_dot, _elip, _n_ring = _ring_results[
                        _idx
                    ]
                    if not _perp_ok or not _sep_ok:
                        # This point fails — the good zone starts one step distal
                        if _idx + 1 < len(_ring_results):
                            _commit_idx = _idx + 1
                        # else: failure at the very last point (most distal) —
                        # no good zone found, leave _geo_ostium = None (fallback)
                        break
                else:
                    # No failure found — entire window passed.  Commit ogi
                    # (proximal anchor is already clean, don't push distal).
                    _commit_idx = 0

                if _commit_idx is not None:
                    _geo_ostium = _ring_results[_commit_idx][0]
                    # Clamp to [ogi, _geo_end] for renal branches so the
                    # GEO_RENAL_EXT_MM extension past sgi is not immediately
                    # undone.  Non-renal branches still clamp to sgi.
                    if is_renal:
                        _geo_ostium = max(ogi, min(_geo_ostium, _geo_end))
                    else:
                        _geo_ostium = max(ogi, min(_geo_ostium, sgi))
                    # Track best signal values for logging
                    for _, _, _, _rd, _el, _nr in _ring_results:
                        if _rd > _best_dot:
                            _best_dot = _rd
                            _best_elip = _el
                            _best_n_ring = _nr

            if _geo_ostium is not None:
                _arc_shift = (
                    abs(self.distances[_geo_ostium] - self.distances[ogi])
                    if (_geo_ostium < len(self.distances) and ogi < len(self.distances))
                    else 0.0
                )
                print(
                    f"[RefineOstium] Branch {bi}: ring-normal(bwd) ogi={ogi}→{_geo_ostium} "
                    f"(+{_geo_ostium - ogi}pts, +{_arc_shift:.1f}mm) sgi={sgi} "
                    f"[dot={_best_dot:.3f}>{RING_DOT_THRESH} "
                    f"elip={_best_elip:.2f}<{RING_ELIP_THRESH} "
                    f"ring_pts={_best_n_ring} sep>{SEP_THRESH}mm/step]"
                )
                self.branchMeta.setdefault(bi, {})["ostiumGiRefined"] = _geo_ostium
                self.branchMeta[bi]["ostiumGi"] = _geo_ostium
                self.branchMeta[bi]["ostium_method"] = "ring_normal"
                self.branches[bi] = (_geo_ostium, e)
                continue  # skip composite scorer for this branch

            # ── No geometric hit — fall through to composite window scorer ────
            # (runs for renals and branches where surface normals are weak/absent)

            # ── Renal-specific bonus signals (lateral offset + orthogonality) ─
            _renal_lat_sc = 0.0
            _renal_align_sc = 0.0
            if is_renal:
                _ortho = meta.get("ortho_score")
                _lat_mm = meta.get("tip_lateral_mm")
                if _lat_mm is not None:
                    _renal_lat_sc = max(0.0, min(1.0, (_lat_mm - 15.0) / 35.0))
                if _ortho is not None:
                    _renal_align_sc = max(0.0, min(1.0, _ortho))
            back_mm = RENAL_BACK_MM if is_renal else BACK_MM
            back_pts = _mm_to_pts(back_mm, ogi, e)

            win_start = max(s, ogi - back_pts)
            # ── Adaptive LOOK: ~8% of branch length, minimum LOOK_MIN ──────────
            branch_len_pts = e - s
            LOOK = max(LOOK_MIN, int(0.08 * branch_len_pts))

            # ── v2: extend search window BEYOND stableStartGi ────────────────
            # Critical fix: the true physiologically independent ostium (stable
            # branch calibre, post-flare) is typically 10–25mm past stableStartGi.
            # Old code capped at sgi → always landed in the flare / trunk zone.
            ext_mm = 30.0 if is_renal else EXT_MM
            ext_pts = _mm_to_pts(ext_mm, sgi, e)
            win_end = min(e - LOOK - 1, sgi + ext_pts)  # extend beyond stable zone

            if win_end <= win_start:
                print(
                    f"[RefineOstium] Branch {bi}: "
                    f"{'renal ' if is_renal else ''}window collapsed "
                    f"[{win_start},{win_end}] — kept ogi={ogi}"
                )
                continue

            # tangents already computed in the geometric detector block above.
            # ── Per-candidate diameter cache (avoids redundant lookups) ───────
            diam_cache = {}

            def get_d_cached(gi):
                if gi not in diam_cache:
                    diam_cache[gi] = _get_d(gi)
                return diam_cache[gi]

            # ── v2: trunk diameter reference and branch stable mean ──────────
            # trunk_diam: median of non-zero trunk diameters (branch 0 gi range)
            _t_s, _t_e = self.branches[0]
            _t_diams = [
                self.diameters[_g]
                for _g in range(_t_s, _t_e)
                if 0 <= _g < len(self.diameters) and self.diameters[_g] > 0
            ]
            if _t_diams:
                _t_diams.sort()
                trunk_diam = _t_diams[len(_t_diams) // 2]
            else:
                trunk_diam = 20.0  # safe fallback

            # branch_mean: mean of first 10 post-stableStart diameters (clean tissue)
            _bm_gis = [
                sgi + k
                for k in range(10)
                if sgi + k < e
                and 0 <= sgi + k < len(self.diameters)
                and self.diameters[sgi + k] > 0
            ]
            branch_mean = (
                sum(self.diameters[_g] for _g in _bm_gis) / len(_bm_gis)
                if _bm_gis
                else trunk_diam * 0.55
            )

            # ── Trunk KD-tree for lateral separation signal ───────────────────
            # Lateral separation = perpendicular distance from branch CL point
            # to the nearest point on the trunk centerline.
            # We subsample the trunk to ≤60 pts for speed.
            _trunk_pts_arr = None
            _trunk_kd = None
            try:
                import numpy as np

                _t_s0, _t_e0 = self.branches[0]
                _step = max(1, (_t_e0 - _t_s0) // 60)
                _trunk_pts_arr = np.array(
                    [
                        self.points[_g]
                        for _g in range(_t_s0, _t_e0, _step)
                        if _g < len(self.points)
                    ],
                    dtype=float,
                )
                if len(_trunk_pts_arr) >= 2:
                    from scipy.spatial import cKDTree

                    _trunk_kd = cKDTree(_trunk_pts_arr)
            except Exception:
                _trunk_kd = None

            def _lateral_dist(gi):
                """Perpendicular distance from gi to trunk CL (mm)."""
                if _trunk_kd is None or gi >= len(self.points):
                    return 0.0
                try:
                    pt = np.array(self.points[gi], dtype=float)
                    d, _ = _trunk_kd.query(pt)
                    return float(d)
                except Exception:
                    return 0.0

            # ── Trunk direction (for angle signal) ───────────────────────────
            _t_s0, _t_e0 = self.branches[0]
            _trunk_dir = [0.0, 0.0, 1.0]  # fallback
            if _t_e0 - _t_s0 >= 4:
                _tp0 = self.points[_t_s0]
                _tp1 = self.points[_t_e0 - 1]
                _tdx = _tp1[0] - _tp0[0]
                _tdy = _tp1[1] - _tp0[1]
                _tdz = _tp1[2] - _tp0[2]
                _tdn = math.sqrt(_tdx**2 + _tdy**2 + _tdz**2) or 1.0
                _trunk_dir = [_tdx / _tdn, _tdy / _tdn, _tdz / _tdn]

            # ── Adaptive weights: short vs long branch ────────────────────────
            # Short branches (<60mm): angle measurement is unreliable because
            # the proximal zone is dominated by bifurcation geometry; lateral
            # separation is the strongest signal.
            # Long branches (≥60mm): angle has settled and is the best signal.
            branch_arc_mm = _arc_mm(s, e - 1)
            if branch_arc_mm < 60.0:
                # Short branch regime
                W_LAT = 0.50
                W_ANG = 0.20
                W_STAB_DIR = 0.30
                W_PROX = 0.25  # subtracted; scaled to match sum
            else:
                # Long branch regime
                W_LAT = 0.30
                W_ANG = 0.40
                W_STAB_DIR = 0.30
                W_PROX = 0.20

            # ── Scan window: PASS A — collect raw cross-section areas ─────────
            # We pre-compute all vtkCutter areas first so we can median-smooth
            # the series before using it in gradient scoring.  Unstable cuts
            # (wrong contour, plane-near-bifurcation) produce isolated spikes;
            # a 5-point median kills the spike without blurring the real trend.
            _DEBUG_BI = 1  # branch to instrument (change to 3 for renal)
            _area_debug = []  # [(gi, A_raw, area_sc_post)] — debug only

            scan_gis = [
                gi
                for gi in range(win_start, win_end + 1)
                if gi < len(self.points) and 0 <= (gi - s) < len(tangents)
            ]

            # Raw area at every scan point (None = cut failed)
            _raw_area = {}  # gi → mm² or None
            if _surface_pd is not None:
                # Collect areas for every unique gi we will sample (prox + dist)
                _area_gis_needed = set()
                for gi in scan_gis:
                    _area_gis_needed.add(max(s, gi - LOOK))
                    _area_gis_needed.add(min(e - 1, gi + LOOK))
                for _agi in _area_gis_needed:
                    if _agi >= len(self.points):
                        continue
                    _local = _agi - s
                    if _local < 0 or _local >= len(tangents):
                        continue
                    _raw_area[_agi] = self._computeCrossSectionArea(
                        _surface_pd, self.points[_agi], tangents[_local]
                    )

                # ── Median-smooth the raw area series (window=5) ────────────
                # Sort gi keys, fill None with neighbours, then slide median.
                _area_sorted_gis = sorted(_raw_area.keys())
                _area_vals = [_raw_area[_g] for _g in _area_sorted_gis]

                # Forward-fill / back-fill None gaps so median works everywhere
                _filled = list(_area_vals)
                for _k in range(1, len(_filled)):
                    if _filled[_k] is None:
                        _filled[_k] = _filled[_k - 1]
                for _k in range(len(_filled) - 2, -1, -1):
                    if _filled[_k] is None:
                        _filled[_k] = _filled[_k + 1]

                # 5-point sliding median (reflect at boundaries)
                _MED_HALF = 2
                _smoothed = {}
                for _idx, _g in enumerate(_area_sorted_gis):
                    _lo = max(0, _idx - _MED_HALF)
                    _hi = min(len(_filled) - 1, _idx + _MED_HALF)
                    _win = [v for v in _filled[_lo : _hi + 1] if v is not None]
                    if _win:
                        _win_s = sorted(_win)
                        _smoothed[_g] = _win_s[len(_win_s) // 2]
                    else:
                        _smoothed[_g] = None

                # Replace raw dict with smoothed values
                _raw_area = _smoothed

                # ── Local-median outlier rejection ──────────────────────────
                # Any individual area that is < 30% of the local median is
                # almost certainly a bad cutter contour (plane near bifurcation
                # wall, wrong loop chosen).  Replace with the smoothed value
                # computed above — which is already the 5-pt median — so the
                # caller naturally gets the fallback without special-casing.
                # (Smoothing already handled this; this guard is a belt-and-
                # suspenders check for extreme outliers that survived smoothing.)
                _sm_vals = [v for v in _smoothed.values() if v is not None]
                if _sm_vals:
                    _sm_vals_s = sorted(_sm_vals)
                    _global_med = _sm_vals_s[len(_sm_vals_s) // 2]
                    _low_thresh = 0.30 * _global_med
                    for _g in list(_raw_area.keys()):
                        if (
                            _raw_area[_g] is not None
                            and _raw_area[_g] < _low_thresh
                            and _global_med > 1.0
                        ):
                            # Replace with smoothed median (already in _smoothed)
                            _raw_area[_g] = _global_med

            def _get_area(gi):
                return _raw_area.get(gi, None)

            # ── Precompute lateral distances for all scan gis ────────────────
            # We need lat[gi] and lat[gi-1] for the separation rate signal.
            _lat_cache = {}
            for _lgi in set(scan_gis) | {max(s, _g - 1) for _g in scan_gis}:
                if 0 <= _lgi < len(self.points):
                    _lat_cache[_lgi] = _lateral_dist(_lgi)

            def _get_lat(gi):
                if gi in _lat_cache:
                    return _lat_cache[gi]
                return _lateral_dist(gi)

            # ── Scan window: PASS B — score every candidate gi ───────────────
            # Unified Ostium Score with 4 primary signals + legacy corroboration.
            candidates = []

            for gi in scan_gis:
                local_i = gi - s

                # ── PRIMARY SIGNAL 1: Lateral separation rate ─────────────────
                # d_lat = distance-to-trunk[gi] - distance-to-trunk[gi-1]
                # Positive = branch is peeling away from trunk = strong evidence.
                lat_here = _get_lat(gi)
                lat_prev = _get_lat(max(s, gi - 1))
                d_lat = lat_here - lat_prev  # mm gained per step
                lat_sc = max(0.0, min(1.0, d_lat / LAT_NORM_MM))

                # ── PRIMARY SIGNAL 2: Angle to trunk (delayed) ────────────────
                # Skips ANGLE_OFFSET_MM forward to clear bifurcation contamination.
                # For short branches this region barely exists, so the delayed
                # measurement still lands inside the branch body.
                ang_sc = 0.0
                if local_i + 1 < len(tangents):
                    _spc_ang = max(
                        0.3,
                        (_arc_mm(gi, min(e - 1, gi + 5)) / max(1, min(5, e - gi - 1))),
                    )
                else:
                    _spc_ang = 1.4
                _off_pts = max(1, int(round(ANGLE_OFFSET_MM / _spc_ang)))
                _ang_i = min(local_i + _off_pts, len(tangents) - 1)
                if _ang_i != local_i:
                    _t_ang = tangents[_ang_i]
                    _dot = max(
                        -1.0,
                        min(
                            1.0,
                            _t_ang[0] * _trunk_dir[0]
                            + _t_ang[1] * _trunk_dir[1]
                            + _t_ang[2] * _trunk_dir[2],
                        ),
                    )
                    _ang_deg = math.degrees(math.acos(abs(_dot)))
                    ang_sc = max(0.0, min(1.0, _ang_deg / ANG_NORM_DEG))

                # ── PRIMARY SIGNAL 3: Direction stability ─────────────────────
                # dot(dir[gi], dir[gi + delta_pts]) ≈ 1 → past chaotic junction.
                # Low dot → still in the bifurcation dome → penalise.
                stab_dir_sc = 0.5  # neutral default
                if local_i + 1 < len(tangents):
                    _spc_s = max(
                        0.3,
                        (_arc_mm(gi, min(e - 1, gi + 5)) / max(1, min(5, e - gi - 1))),
                    )
                else:
                    _spc_s = 1.4
                _delta_pts = max(2, int(round(STAB_DELTA_MM / _spc_s)))
                _fwd_s = min(local_i + _delta_pts, len(tangents) - 1)
                if _fwd_s != local_i:
                    _t0 = tangents[local_i]
                    _t1 = tangents[_fwd_s]
                    _dot_s = max(
                        -1.0,
                        min(1.0, _t0[0] * _t1[0] + _t0[1] * _t1[1] + _t0[2] * _t1[2]),
                    )
                    stab_dir_sc = max(0.0, min(1.0, (_dot_s + 1.0) / 2.0))
                    # dot=1 → same direction → stable → sc=1.0
                    # dot=0 → 90° change → unstable → sc=0.5
                    # dot=-1 → U-turn → sc=0.0

                # ── PRIMARY SIGNAL 4: Anti-proximal penalty ───────────────────
                # Exponential decay from ogi: discourages picking too early.
                # exp(-0) = 1 at ogi (full penalty); decays as gi moves distal.
                arc_from_ogi = _arc_mm(ogi, gi)
                prox_pen = math.exp(-arc_from_ogi / PROX_SCALE_MM)
                # prox_pen approaches 0 as gi moves 20-30mm past ogi.

                # ── Legacy corroboration signals ──────────────────────────────
                # Cross-sectional area gradient (smoothed)
                area_sc = 0.0
                _A_prox_raw = None
                _A_dist_raw = None
                if _surface_pd is not None:
                    lo_gi = max(s, gi - LOOK)
                    hi_gi = min(e - 1, gi + LOOK)
                    _A_prox_raw = _get_area(lo_gi)
                    _A_dist_raw = _get_area(hi_gi)
                    if (
                        _A_prox_raw is not None
                        and _A_dist_raw is not None
                        and (_A_prox_raw + _A_dist_raw) > 1.0
                    ):
                        denom = max((_A_dist_raw + _A_prox_raw) * 0.5, 1e-3)
                        ag = (_A_dist_raw - _A_prox_raw) / denom
                        area_sc = ag  # raw; percentile-normalised below
                if bi == _DEBUG_BI:
                    _area_debug.append((gi, _A_prox_raw, _A_dist_raw, area_sc))

                # Diameter gradient
                d_lo = get_d_cached(max(s, gi - LOOK))
                d_hi = get_d_cached(min(e - 1, gi + LOOK))
                diam_sc = 0.0
                if d_lo > 0:
                    dg = (d_hi - d_lo) / d_lo
                    diam_sc = max(0.0, min(1.0, dg / DIAM_NORM))

                # Centerline curvature
                curv_sc = 0.0
                if local_i + 1 < len(tangents):
                    _spc_c = max(
                        0.3,
                        (_arc_mm(gi, min(e - 1, gi + 5)) / max(1, min(5, e - gi - 1))),
                    )
                else:
                    _spc_c = 1.4
                _curv_pts = max(2, int(round(CURV_MM / _spc_c)))
                fwd_c = min(local_i + _curv_pts, len(tangents) - 1)
                if local_i != fwd_c:
                    t0 = tangents[local_i]
                    t1 = tangents[fwd_c]
                    dot_c = max(
                        -1.0, min(1.0, t0[0] * t1[0] + t0[1] * t1[1] + t0[2] * t1[2])
                    )
                    bend_deg = math.degrees(math.acos(dot_c))
                    curv_sc = max(0.0, min(1.0, bend_deg / CURV_NORM))

                # Topology prior (soft anchor to ogi)
                topo_sc = math.exp(-abs(gi - ogi) / TOPO_TAU)

                # Diameter drop vs trunk
                _d_here = get_d_cached(gi)
                drop_sc = 0.0
                if _d_here > 0 and trunk_diam > 0:
                    drop_sc = max(0.0, min(1.0, 1.0 - _d_here / trunk_diam))

                # Distal consistency
                cons_sc = 0.5
                _d_lo5 = gi + 5
                _d_hi5 = gi + 15
                if _d_lo5 < e and _d_hi5 <= e:
                    _dp = [
                        get_d_cached(_g)
                        for _g in range(_d_lo5, _d_hi5)
                        if get_d_cached(_g) > 0
                    ]
                    if len(_dp) >= 5 and _d_here > 0:
                        _dm = sum(_dp) / len(_dp)
                        if _dm > 0:
                            cons_sc = max(
                                0.0, min(1.0, 1.0 - abs(_d_here - _dm) / (_dm + 1e-6))
                            )

                # Trunk proximity penalty (diameter-based bleed check)
                trunk_pen = 0.0
                if _d_here > 0 and branch_mean > 0:
                    if _d_here > 1.3 * branch_mean:
                        trunk_pen = min(1.0, (_d_here / branch_mean) * 0.40)

                # Store; area percentile-norm applied in post-scan step.
                # Tuple layout: (gi, lat_sc, area_sc_raw, diam_sc, topo_sc,
                #                stab_dir_sc, curv_sc, drop_sc, cons_sc,
                #                trunk_pen, ang_sc, prox_pen)
                candidates.append(
                    (
                        gi,
                        lat_sc,
                        area_sc,
                        diam_sc,
                        topo_sc,
                        stab_dir_sc,
                        curv_sc,
                        drop_sc,
                        cons_sc,
                        trunk_pen,
                        ang_sc,
                        prox_pen,
                    )
                )

            # ── Percentile-normalise area_sc within window ───────────────────
            raw_ags = [c[2] for c in candidates]
            if len(raw_ags) >= 4:
                _sorted = sorted(raw_ags)
                _n = len(_sorted)
                p_low = _sorted[max(0, int(0.10 * _n))]
                p_high = _sorted[min(_n - 1, int(0.90 * _n))]
            elif raw_ags:
                p_low, p_high = min(raw_ags), max(raw_ags)
            else:
                p_low, p_high = 0.0, 1.0
            _prange = max(p_high - p_low, 1e-6)
            if bi == _DEBUG_BI:
                print(
                    f"[AreaDebug] Branch {bi}: percentile norm "
                    f"p10={p_low:.3f} p90={p_high:.3f} range={_prange:.3f}"
                )

            # ── Compute unified composite score for each candidate ────────────
            # Unified Ostium Score = primary 4-signal model + legacy corroboration.
            #
            # composite = w_lat   * lat_sc           (lateral separation rate)
            #           + w_ang   * ang_sc            (delayed angle to trunk)
            #           + w_stab  * stab_dir_sc       (direction stability gate)
            #           - w_prox  * prox_pen          (anti-proximal penalty)
            #           + legacy corroboration terms  (area, diam, topo, drop, cons)
            #           - trunk_pen                   (hard bleed-through guard)
            #           + renal bonus (if renal)
            ranked_candidates = []
            for (
                gi,
                lat_sc,
                raw_ag,
                diam_sc,
                topo_sc,
                stab_dir_sc,
                curv_sc,
                drop_sc,
                cons_sc,
                trunk_pen,
                ang_sc,
                prox_pen,
            ) in candidates:

                area_sc_norm = max(0.0, min(1.0, (raw_ag - p_low) / _prange))

                # Primary 4-signal unified score
                unified = (
                    W_LAT * lat_sc
                    + W_ANG * ang_sc
                    + W_STAB_DIR * stab_dir_sc
                    - W_PROX * prox_pen
                )

                # Legacy corroboration (secondary, lower weights)
                legacy = (
                    W_AREA * area_sc_norm
                    + W_DIAM * diam_sc
                    + W_TOPO * topo_sc
                    + W_CURV * curv_sc
                    + W_DROP * drop_sc
                    + W_CONS * cons_sc
                    - 0.35 * trunk_pen
                )

                composite = unified + legacy

                # Renal-specific bonus (branch-level constant, outside weight budget)
                if is_renal:
                    composite += (
                        RENAL_LAT_W * _renal_lat_sc + RENAL_ALIGN_W * _renal_align_sc
                    )

                ranked_candidates.append(
                    (
                        gi,
                        composite,
                        area_sc_norm,
                        diam_sc,
                        topo_sc,
                        stab_dir_sc,
                        curv_sc,
                        drop_sc,
                        cons_sc,
                        trunk_pen,
                        lat_sc,
                        ang_sc,
                        prox_pen,
                    )
                )

            candidates = ranked_candidates

            # ── Area debug summary (Branch _DEBUG_BI only) ────────────────────
            # _area_debug holds post-smoothing areas and raw ag (pre-percentile-norm).
            if bi == _DEBUG_BI and _area_debug:
                sm_prox = [r[1] for r in _area_debug if r[1] is not None]
                sm_dist = [r[2] for r in _area_debug if r[2] is not None]
                all_sm = sm_prox + sm_dist
                if all_sm:
                    import math as _m

                    _mean = sum(all_sm) / len(all_sm)
                    _std = _m.sqrt(sum((x - _mean) ** 2 for x in all_sm) / len(all_sm))
                    print(
                        f"[AreaDebug] Branch {bi} (smoothed): "
                        f"LOOK={LOOK}pts  window={len(_area_debug)}pts  "
                        f"area_min={min(all_sm):.1f}  area_max={max(all_sm):.1f}  "
                        f"area_std={_std:.2f}mm²  "
                        f"prox_mean={sum(sm_prox)/max(len(sm_prox),1):.1f}  "
                        f"dist_mean={sum(sm_dist)/max(len(sm_dist),1):.1f}"
                    )
                    step = max(1, len(_area_debug) // 10)
                    for _gi, _ap, _ad, _asc in _area_debug[::step]:
                        _ap_s = f"{_ap:.1f}" if _ap is not None else "None"
                        _ad_s = f"{_ad:.1f}" if _ad is not None else "None"
                        print(
                            f"[AreaDebug]   gi={_gi} "
                            f"A_prox={_ap_s} A_dist={_ad_s} ag_raw={_asc:.3f}"
                        )
                else:
                    print(
                        f"[AreaDebug] Branch {bi}: all cuts returned None "
                        f"(surface missing or degenerate)"
                    )

            if not candidates:
                print(
                    f"[RefineOstium] Branch {bi}: no valid candidates — kept ogi={ogi}"
                )
                continue

            # ── Select best: forward local-peak scan ──────────────────────────
            # Instead of global max (which is unstable), scan forward and pick
            # the FIRST point that:
            #   (a) exceeds PEAK_THRESHOLD, AND
            #   (b) is higher than all N preceding candidates (local peak).
            # Safety guards:
            #   - gi must be >= ogi + 5 pts (minimum distance from start)
            #   - gi must be <= sgi (must be before or at stableStart)
            # If no qualifying peak found, fall back to global argmax (clamped).
            best = None
            scores_in_order = [(c[0], c[1]) for c in candidates]  # (gi, score)
            scores_in_order.sort(key=lambda x: x[0])  # sort by gi

            min_dist_pts = 5  # hard minimum from ogi
            for _idx, (_gi, _sc) in enumerate(scores_in_order):
                # Hard guard: must not be within min_dist_pts of start
                if _gi < ogi + min_dist_pts:
                    continue
                # Hard guard: must be at or before sgi
                if _gi > sgi:
                    break
                # Must exceed threshold
                if _sc < PEAK_THRESHOLD:
                    continue
                # Must be a local peak: beat all preceding PEAK_LOOKBACK scores
                _lookback_start = max(0, _idx - PEAK_LOOKBACK)
                _is_peak = all(
                    _sc >= scores_in_order[_j][1] for _j in range(_lookback_start, _idx)
                )
                if _is_peak:
                    best = next(c for c in candidates if c[0] == _gi)
                    break

            # Fallback: global argmax clamped to [ogi, sgi]
            if best is None:
                _valid = [c for c in candidates if ogi <= c[0] <= sgi]
                if _valid:
                    best = max(_valid, key=lambda c: c[1])
                else:
                    best = max(candidates, key=lambda c: c[1])

            best_gi = best[0]
            best_score = best[1]
            best_area = best[2]
            best_diam = best[3]
            best_topo = best[4]
            best_stab = best[5]  # stab_dir_sc
            best_curv = best[6]
            best_drop = best[7]
            best_cons = best[8]
            best_tp = best[9]
            best_lat = best[10]
            best_ang = best[11]
            best_prox = best[12]

            # Clamp to [ogi, sgi]
            best_gi = max(ogi, min(best_gi, sgi))

            # ── Confidence 2.0: mean − std of window composite scores ─────────
            all_scores = [c[1] for c in candidates]
            mean_sc = sum(all_scores) / len(all_scores)
            std_sc = math.sqrt(
                sum((x - mean_sc) ** 2 for x in all_scores) / len(all_scores)
            )
            conf_v2 = max(0.0, min(1.0, mean_sc - std_sc))

            # ── Log ───────────────────────────────────────────────────────────
            arc_mm = _arc_mm(ogi, best_gi)
            tag = "renal " if is_renal else ""
            _renal_bonus_str = ""
            if is_renal:
                _renal_bonus_str = (
                    f" renal_bonus=[lat={_renal_lat_sc:.2f}"
                    f" align={_renal_align_sc:.2f}]"
                )
            print(
                f"[RefineOstium] Branch {bi}: {tag}ogi={ogi}→{best_gi} "
                f"(+{best_gi-ogi}pts, +{arc_mm:.1f}mm) sgi={sgi} "
                f"score={best_score:.3f} conf={conf_v2:.3f} "
                f"window=[{win_start},{win_end}]({len(candidates)}pts) "
                f"[lat={best_lat:.2f} ang={best_ang:.2f} "
                f"stab={best_stab:.2f} prox_pen={best_prox:.2f} "
                f"area={best_area:.2f} diam={best_diam:.2f} "
                f"curv={best_curv:.2f} topo={best_topo:.2f} "
                f"drop={best_drop:.2f} cons={best_cons:.2f} "
                f"trunk_pen={best_tp:.2f}]{_renal_bonus_str} "
                f"weights=[short={branch_arc_mm<60:.0f} "
                f"w_lat={W_LAT} w_ang={W_ANG} w_stab={W_STAB_DIR}]"
            )

            # ── Commit ────────────────────────────────────────────────────────
            # Also store area signal quality stats so _computeOstiumConfidence
            # can use distribution-based gradient_weak instead of value threshold.
            _area_norm_vals = [
                c[2] for c in candidates
            ]  # percentile-normalised area_sc
            _peak_strength = (
                (
                    max(_area_norm_vals)
                    - sorted(_area_norm_vals)[len(_area_norm_vals) // 2]
                )
                if _area_norm_vals
                else 0.0
            )

            self.branchMeta.setdefault(bi, {})["ostiumGiRefined"] = best_gi
            self.branchMeta[bi]["ostiumGi"] = best_gi
            self.branchMeta[bi]["ostium_method"] = "scored_v2"
            self.branchMeta[bi]["ostium_confidence_v2"] = conf_v2
            self.branchMeta[bi]["ostium_scores_v2"] = {
                # Primary unified signals
                "lat": best_lat,  # lateral separation rate
                "ang": best_ang,  # delayed angle to trunk
                "stab_dir": best_stab,  # direction stability
                "prox_pen": best_prox,  # anti-proximal penalty (info only)
                # Legacy corroboration
                "area": best_area,
                "diam": best_diam,
                "curv": best_curv,
                "topo": best_topo,
                "drop": best_drop,
                "cons": best_cons,
                # Adaptive weight regime
                "short_branch": branch_arc_mm < 60.0,
                "w_lat": W_LAT,
                "w_ang": W_ANG,
                "w_stab_dir": W_STAB_DIR,
                # Area signal quality — used by _computeOstiumConfidence
                "area_p10": p_low,
                "area_p90": p_high,
                "area_range": p_high - p_low,  # gradient_range
                "area_peak_str": _peak_strength,  # max - median of normalised scores
            }
            self.branches[bi] = (best_gi, e)
            ogi = best_gi  # update local ogi so OstiumSnap scans from refined position

            # ── Renal: find true ostium by walking branch CL from ogi ────────
            # Algorithm (per user specification):
            #   1. Walk forward from ogi along branch CL.
            #   2. Find FIRST point where BOTH:
            #      (A) branch diameter < 70% of trunk diameter at same Z
            #      (B) branch direction diverges >30° from trunk direction
            #   3. Snap that CL point to the nearest trunk surface vertex.
            #   4. Use the snapped coordinate as the actual ostiumGi (not just display).
            if not is_main and _trunk_surf_kd is not None:
                try:
                    import numpy as np
                    import math as _math

                    RADIUS_RATIO_THRESH = 0.70  # branch diam < 70% of trunk diam
                    DIVERGE_DEG_THRESH = 30.0  # branch dir > 30° from trunk dir
                    DIVERGE_DOT_THRESH = _math.cos(_math.radians(DIVERGE_DEG_THRESH))

                    # Trunk direction: root→bif unit vector from trunk CL
                    _ts0, _te0 = self.branches[0]
                    _tp0 = np.array(self.points[_ts0], dtype=float)
                    _tp1 = np.array(self.points[_te0 - 1], dtype=float)
                    _tv = _tp1 - _tp0
                    _tvl = float(np.linalg.norm(_tv))
                    _trunk_unit = (
                        (_tv / _tvl) if _tvl > 1e-6 else np.array([0.0, 0.0, 1.0])
                    )

                    # Trunk diameter at a given gi: find nearest trunk gi by Z
                    _trunk_gis = list(range(_ts0, _te0))
                    _trunk_zs = np.array(
                        [self.points[g][2] for g in _trunk_gis], dtype=float
                    )

                    def _trunk_diam_at_z(z):
                        idx = int(np.argmin(np.abs(_trunk_zs - z)))
                        tgi = _trunk_gis[idx]
                        d = self.diameters[tgi] if tgi < len(self.diameters) else 0.0
                        return d if d > 0 else 20.0  # fallback to typical IVC diam

                    # ── Backward scan from sgi — find proximal edge of true lumen ─
                    #
                    # Root cause of the previous forward-scan failure:
                    #   The scan started at ogi where the branch diameter is
                    #   trunk-contaminated (15–16 mm vs true 10–12 mm).  The
                    #   diameter signal (diam_sc) was therefore ≈0 for the entire
                    #   shared-lumen zone, so the composite score peaked right at
                    #   ogi driven by divergence alone — leaving the marker inside
                    #   the IVC flare zone.
                    #
                    # Fix: scan BACKWARD from sgi (the flare-walk's stable-lumen
                    #   anchor) toward ogi.  sgi is guaranteed to be in clean
                    #   branch tissue.  Walking proximally, the FIRST point where
                    #   ratio = branch_diam / trunk_diam_at_z rises ABOVE
                    #   DIAM_ENTER_THRESH (0.80) marks the proximal boundary of
                    #   the true branch lumen — i.e. the anatomical ostium.
                    #
                    # Selection logic:
                    #   1. Collect ratios for every gi in [ogi, sgi].
                    #   2. Walk backward from sgi; stop at the first gi where
                    #      ratio > DIAM_ENTER_THRESH (entering trunk influence).
                    #   3. The gi JUST DISTAL of that crossing is the ostium.
                    #   4. If no crossing found (branch always separated), keep sgi.
                    #   5. If branch always trunk-inflated (always above thresh),
                    #      use the global minimum-ratio point as fallback.

                    DIAM_ENTER_THRESH = 0.75  # ratio above this = still in trunk zone
                    # Lowered 0.80→0.75 (v229): at 0.80 the Branch4 ratio was
                    # exactly on the boundary (ratio=0.80) so < 0.80 never fired
                    # and the fallback min-ratio gi was used instead, landing the
                    # marker ~4mm too proximal (S=1889.5 vs target S=1893.9).
                    # 0.75 requires the branch diameter to genuinely separate to
                    # 75% of trunk before committing, pushing detect_gi distally
                    # into clean independent-lumen tissue.

                    # Trunk CL KD-tree for lateral offset (used in log only)
                    _snap_trunk_kd = None
                    try:
                        from scipy.spatial import cKDTree as _cKDTree2

                        _snap_trunk_pts = np.array(
                            [
                                self.points[_g]
                                for _g in range(_ts0, _te0)
                                if _g < len(self.points)
                            ],
                            dtype=float,
                        )
                        if len(_snap_trunk_pts) >= 2:
                            _snap_trunk_kd = _cKDTree2(_snap_trunk_pts)
                    except Exception:
                        pass

                    def _lat_offset(gi):
                        """Perpendicular distance from branch CL point to trunk CL."""
                        if _snap_trunk_kd is None or gi >= len(self.points):
                            return 0.0
                        _pt = np.array(self.points[gi], dtype=float)
                        _d, _ = _snap_trunk_kd.query(_pt)
                        return float(_d)

                    # Scan window: [ogi, e-2] — full branch extent.
                    # Critical: for short branches (B4=31mm, B5=31mm) sgi is only
                    # ~10pts past ogi and still inside the IVC junction zone.
                    # The true ostium (where diameter drops to branch scale) can be
                    # past sgi — we must scan the full branch to find it reliably.
                    _scan_start = max(ogi, 0)
                    _scan_end = max(e - 2, _scan_start)

                    # Build ratio array for the window
                    _snap_scores = []  # (gi, ratio)
                    for _wgi in range(_scan_start, _scan_end + 1):
                        if _wgi >= len(self.points) or _wgi >= len(self.diameters):
                            break
                        _b_diam = self.diameters[_wgi]
                        _t_diam = _trunk_diam_at_z(self.points[_wgi][2])
                        _ratio = (_b_diam / _t_diam) if _t_diam > 0 else 1.0
                        _snap_scores.append((_wgi, _ratio))

                    # Forward walk from ogi: find first crossing where ratio DROPS
                    # below DIAM_ENTER_THRESH — this is where the branch lumen leaves
                    # trunk influence and enters its own clean lumen.
                    #
                    # Root cause of the backward-scan failure (B4, B5):
                    #   The IVC flare zone inflates diameters above 0.80 across most
                    #   of the branch (not just near ogi).  Walking backward from e-2,
                    #   ratio > 0.80 fires almost immediately, committing one step
                    #   distal — which is still well inside the contaminated zone.
                    #
                    # Forward scan logic:
                    #   1. Walk forward from ogi.
                    #   2. Find the FIRST gi where ratio < DIAM_ENTER_THRESH
                    #      AND the ratio stays below threshold for CONFIRM_PTS
                    #      consecutive points (guards against single-point dips
                    #      in the shared-lumen / flare zone that fired too early
                    #      for B4/B5 — detect_gi was landing ~1mm past ogi).
                    #   3. Floor result to sgi (stableStartGi) — sgi is guaranteed
                    #      to be in post-flare clean tissue by the flare-walk;
                    #      no anatomically valid ostium can be proximal to it.
                    #   4. Fallback: if ratio never drops, use minimum-ratio gi.
                    CONFIRM_PTS = 3  # must stay below threshold for this many pts
                    _detect_gi = None
                    for _idx, (_wgi, _ratio) in enumerate(_snap_scores):
                        if _ratio < DIAM_ENTER_THRESH:
                            # Check that the next CONFIRM_PTS-1 points also stay below
                            _confirm_ok = True
                            for _ci in range(1, CONFIRM_PTS):
                                if _idx + _ci >= len(_snap_scores):
                                    break  # reached end — accept anyway
                                if _snap_scores[_idx + _ci][1] >= DIAM_ENTER_THRESH:
                                    _confirm_ok = False
                                    break
                            if _confirm_ok:
                                _detect_gi = _wgi
                                break
                            # else: single-point dip — keep scanning

                    # ── DetectGI mode classification ──────────────────────────────
                    # Track whether we got a clean confirmed crossing, a weak
                    # (unconfirmed single-dip) crossing, or had to fall back to
                    # the minimum-ratio point.  Stored for the [DetectGI] log.
                    _detect_mode = "ENTER"  # default: confirmed crossing
                    _min_ratio_gi = (
                        min(_snap_scores, key=lambda x: x[1])[0]
                        if _snap_scores
                        else _scan_start
                    )
                    _min_ratio_val = (
                        min(_snap_scores, key=lambda x: x[1])[1]
                        if _snap_scores
                        else 1.0
                    )

                    if _detect_gi is None:
                        # Ratio never sustained a confirmed drop — use minimum-ratio
                        # point as best guess (least trunk-contaminated)
                        _detect_gi = _min_ratio_gi
                        _detect_mode = "FALLBACK"
                        print(
                            f"[OstiumSnap] Branch {bi}: ratio never dropped below "
                            f"{DIAM_ENTER_THRESH} (confirmed) — using min-ratio gi={_detect_gi}"
                        )
                    elif _detect_gi == _min_ratio_gi:
                        # Confirmed crossing happened to land at the global minimum
                        # (consistent scan — clean entry)
                        _detect_mode = "ENTER"
                    else:
                        # Confirmed crossing landed before the global minimum —
                        # ratio crossed threshold then rose again (noisy / shared
                        # lumen zone); still valid but worth flagging.
                        _detect_mode = (
                            "WEAK_ENTER"
                            if _min_ratio_val < DIAM_ENTER_THRESH
                            else "ENTER"
                        )

                    # ── [DetectGI] structured decision trace ──────────────────────
                    _det_enter_ratio = next(
                        (r for gi, r in _snap_scores if gi == _detect_gi), 0.0
                    )
                    print(
                        f"[DetectGI] branch={bi} "
                        f"mode={_detect_mode} "
                        f"enter_gi={_detect_gi} "
                        f"enter_ratio={_det_enter_ratio:.3f} "
                        f"min_ratio_gi={_min_ratio_gi} "
                        f"min_ratio={_min_ratio_val:.3f} "
                        f"thresh={DIAM_ENTER_THRESH} "
                        f"scan=[{_scan_start},{_scan_end}] "
                        f"confirm_pts={CONFIRM_PTS}"
                    )

                    # NOTE: WEAK_ENTER drift reverted (v232).
                    # Shifting detect_gi deeper moved the Z-matched trunk CL
                    # reference past the lateral takeoff zone, flipping the radial
                    # direction inward (all top-5 dots negative for Branch3 after
                    # shift 423→434, eff collapsed 0.428→0.072).  detect_gi must
                    # stay at the first confirmed threshold crossing; the snap query
                    # origin is now decoupled from detect_gi — see _snap_gi below.

                    # Hard clamp: [ogi, e-2]
                    _detect_gi = max(_scan_start, min(_detect_gi, e - 2))

                    # ── Radial direction: prefer tip-based when detect_gi is proximal ──
                    # Root cause of wrong radial for B3/B4: when detect_gi lands
                    # near ogi the branch CL is still running axially alongside or
                    # medial to the IVC, so the Z-matched trunk→branch radial points
                    # INWARD (toward IVC interior) rather than outward to the wall.
                    # Fix: if detect_gi is within PROX_THRESH_MM of ogi in arc distance,
                    # use the branch-tip XY displacement from the trunk CL as the
                    # radial direction — the tip is always well separated laterally
                    # and gives a reliable outward direction even for short branches.
                    _PROX_THRESH_MM = 15.0
                    _t_fwd = np.array(
                        [0.0, 0.0, 1.0]
                    )  # safe default — overwritten below
                    _arc_from_ogi = 0.0
                    for _ag in range(ogi, min(_detect_gi, e - 1)):
                        if _ag + 1 < len(self.points):
                            _arc_from_ogi += float(
                                np.linalg.norm(
                                    np.array(self.points[_ag + 1])
                                    - np.array(self.points[_ag])
                                )
                            )
                    if _arc_from_ogi < _PROX_THRESH_MM and e - 1 < len(self.points):
                        # Compute tip XY displacement from trunk CL
                        _tip_pt = np.array(self.points[e - 1], dtype=float)
                        _tip_z = float(_tip_pt[2])
                        _tz_idx = int(np.argmin(np.abs(_trunk_zs - _tip_z)))
                        _ttrunk_pt = np.array(
                            self.points[_trunk_gis[_tz_idx]], dtype=float
                        )
                        _tip_rad = _tip_pt - _ttrunk_pt
                        _tip_rad[2] = 0.0  # project to XY plane
                        _tip_rad_len = float(np.linalg.norm(_tip_rad))
                        if _tip_rad_len > 1e-3:
                            _t_fwd = _tip_rad / _tip_rad_len
                            print(
                                f"[SnapDiag] Branch {bi}: proximal detect_gi "
                                f"({_arc_from_ogi:.1f}mm from ogi) — using tip-based "
                                f"radial dir={[round(float(x), 2) for x in _t_fwd]}"
                            )

                    # ── OstiumSnap: snap detect_gi CL point to nearest trunk wall ──
                    #
                    # Strategy (v222):
                    #   Primary pool = trunk-classified surface verts (the lateral IVC
                    #   wall at the takeoff Z).  The VMTK CL runs through the vessel
                    #   lumen centre; at detect_gi the nearest trunk-classified vertex
                    #   IS the lateral wall face that forms the anatomical ostial rim.
                    #
                    #   SEARCH_R_MM = 15mm: the IVC has a mean radius of ~10mm; the
                    #   trunk wall at the renal takeoff sits ~8–10mm from the CL.
                    #   8mm (previous cap) was too tight and caused snap_capped fallback
                    #   (observed: snap_d=8.7mm and 8.1mm for B3/B4).  15mm still
                    #   excludes the contralateral IVC wall (~20+ mm away).
                    #
                    #   No tangent offset: the offset was designed for branch-surf
                    #   queries and pushed the sphere in the wrong direction for trunk.
                    #   Querying directly from _snap_src finds the nearest trunk wall
                    #   in all directions — for a lateral renal that IS the lateral rim.
                    #
                    #   Fallback chain:
                    #     1. trunk KDTree  (primary)  — lateral IVC wall at takeoff
                    #     2. branch KDTree (fallback) — renal/branch surface verts
                    #     3. raw CL[detect_gi] coordinate
                    #
                    #   ostiumGi / self.branches[bi] commit to detect_gi so the
                    #   diameter pipeline and finding detection are unaffected.

                    SEARCH_R_MM = 15.0  # IVC radius ~10mm; 15mm safely reaches wall

                    # ── Decouple snap query origin from diameter anchor ────────────
                    #
                    # Root cause of v231 failure: using detect_gi as both the
                    # diameter anchor AND the snap query origin creates a conflict.
                    # When detect_gi is deep inside the branch (past the flare rim),
                    # the Z-matched trunk CL point at that Z is no longer laterally
                    # aligned with the branch takeoff — the radial vector flips
                    # inward, and argmax(dot) selects the wrong wall face.
                    #
                    # Fix (v232 decoupling):
                    #   _snap_gi  = ogi  (geometric wall-contact anchor — always at
                    #                     the proximal rim of the ostium where the
                    #                     trunk surface is correctly lateral to the CL)
                    #   _detect_gi = unchanged (diameter pipeline / stableStart anchor)
                    #
                    # ogi is set by RenalStabilizer (last point within 2.5mm of
                    # trunk KDTree) for renal branches, or by _refineOstia composite
                    # scorer for side branches.  In both cases it is geometrically
                    # inside the IVC wall zone — exactly where the trunk surface
                    # candidates are laterally outward from the CL.
                    #
                    # The [OstiumSnap] log now shows snap_gi separately from gi
                    # (detect_gi) so both are auditable.
                    # ── Snap source selection ────────────────────────────────────
                    # ogi = RenalStabilizer wall-contact anchor (last CL point
                    #   within 2.5mm of trunk KDTree).  For renal branches ogi is
                    #   geometrically inside the IVC wall zone — the radial direction
                    #   from ogi to the trunk surface is nearly tangential (dot≈0.1),
                    #   so argmax(dot) picks a barely-lateral wall face several mm
                    #   proximal to the true ostial rim.
                    #
                    # detect_gi = first confirmed lumen-separation point (ratio <
                    #   DIAM_ENTER_THRESH for CONFIRM_PTS consecutive pts).  When
                    #   detect_gi is meaningfully distal to ogi the branch CL has
                    #   emerged from the IVC wall and the Z-matched radial correctly
                    #   points outward toward the lateral takeoff rim.
                    #
                    # Rule: use detect_gi as snap source when it is ≥ SNAP_USE_DETECT_MM
                    # ahead of ogi in arc distance; otherwise keep ogi (detect_gi is
                    # still inside the wall zone and the tip-based radial is already
                    # being used — ogi is the better anchor in that case).
                    SNAP_USE_DETECT_MM = 8.0
                    _snap_arc_ogi_to_detect = 0.0
                    for _sg in range(ogi, min(_detect_gi, e - 1)):
                        if _sg + 1 < len(self.points):
                            _snap_arc_ogi_to_detect += float(
                                np.linalg.norm(
                                    np.array(self.points[_sg + 1])
                                    - np.array(self.points[_sg])
                                )
                            )
                    if _snap_arc_ogi_to_detect >= SNAP_USE_DETECT_MM:
                        _snap_gi = max(min(_detect_gi, e - 1), 0)
                        _snap_reason = (
                            f"detect_gi (arc={_snap_arc_ogi_to_detect:.1f}mm from ogi)"
                        )
                    else:
                        _snap_gi = max(min(ogi, e - 1), 0)
                        _snap_reason = f"ogi (detect_gi only {_snap_arc_ogi_to_detect:.1f}mm ahead)"
                    _snap_src = np.array(self.points[_snap_gi], dtype=float)
                    print(
                        f"[OstiumSnap] Branch {bi}: snap_gi={_snap_gi} ({_snap_reason}) "
                        f"detect_gi={_detect_gi} — decoupled"
                    )

                    # ── Radial direction: Z-matched trunk CL → detect_gi ─────
                    # Root cause of previous wrong radial: nearest-3D trunk CL
                    #   point can be axially offset from detect_gi Z, so the
                    #   radial vector has a large Z component and the argmax(dot)
                    #   still picks cranial trunk verts.
                    #
                    # Fix: find the trunk CL point at the SAME Z as detect_gi
                    #   (using the existing _trunk_zs / _trunk_gis arrays built
                    #   above for _trunk_diam_at_z).  The vector from that
                    #   Z-matched trunk CL point to detect_gi lies in the XY plane
                    #   and points purely radially outward from the IVC axis toward
                    #   the lateral wall — the correct direction for argmax(dot).
                    # Only compute Z-matched radial when tip-based was NOT set
                    if _arc_from_ogi >= _PROX_THRESH_MM:
                        _t_fwd = np.array([0.0, 0.0, 1.0])  # fallback only
                        try:
                            _snap_z = float(_snap_src[2])
                            _z_idx = int(np.argmin(np.abs(_trunk_zs - _snap_z)))
                            _tr_gi = _trunk_gis[_z_idx]
                            _tr_pt = np.array(self.points[_tr_gi], dtype=float)
                            # Zero out the Z component so direction is purely lateral
                            _radial = _snap_src - _tr_pt
                            _radial[2] = 0.0  # project onto XY plane
                            _radial_len = float(np.linalg.norm(_radial))
                            if _radial_len > 1e-3:
                                _t_fwd = _radial / _radial_len
                                print(
                                    f"[SnapDiag] Branch {bi}: Z-matched radial "
                                    f"dir={[round(float(x),2) for x in _t_fwd]} "
                                    f"trunk_cl={[round(float(x),1) for x in _tr_pt]} "
                                    f"detect_Z={_snap_z:.1f}"
                                )
                            else:
                                print(
                                    f"[SnapDiag] Branch {bi}: radial too small "
                                    f"({_radial_len:.2f}mm) — using fallback"
                                )
                        except Exception as _rad_err:
                            print(f"[SnapDiag] Branch {bi}: radial error {_rad_err}")

                    # No offset — query directly from the CL point.
                    _snap_query_pt = _snap_src

                    _snap_pt = None
                    _snap_d = 0.0
                    _snap_label = "CL fallback"

                    # ── OstiumSnap surface pool selection ─────────────────────────
                    #
                    # Root cause of wrong snap for renals/side branches:
                    #   The Voronoi assignment gives _surf_kd_per_branch[bi] a mix of
                    #   true renal-wall verts AND the near-side IVC wall faces that are
                    #   geometrically closer to the renal CL than to the trunk CL.
                    #   Querying branch verts therefore still returns an IVC wall vertex,
                    #   not the lateral takeoff rim — identical to the old trunk snap.
                    #
                    # Correct candidate pool for renal/side branches:
                    #   TRUNK-classified verts within SEARCH_R_MM of detect_gi.
                    #   At the takeoff Z the trunk surface IS the lateral IVC wall that
                    #   forms the ostial rim — that is exactly where the user expects
                    #   the marker to land.
                    #
                    # Fallback chain (renal/side):
                    #   1. trunk KDTree  → nearest trunk-classified vert near takeoff
                    #   2. branch KDTree → nearest branch-classified vert (previous primary)
                    #   3. raw CL[detect_gi] coordinate
                    #
                    # For completeness the branch KDTree is still attempted as fallback
                    # in case the trunk KDTree has no candidates in the search radius
                    # (e.g. very distal side branch that has already fully separated).

                    # ── Primary snap: trunk-classified verts at the takeoff zone ──
                    if _trunk_surf_kd is not None:
                        _tk_cand_idxs = _trunk_surf_kd.query_ball_point(
                            _snap_query_pt, SEARCH_R_MM
                        )
                        if _tk_cand_idxs:
                            _tk_cand_pts = _trunk_surf_pts[_tk_cand_idxs]
                            _tk_vecs = _tk_cand_pts - _snap_src
                            _tk_dists = np.linalg.norm(_tk_vecs, axis=1)
                            # ── Z-band filter: restrict to verts at same axial level ──
                            # Root cause of cranial snap (B3 Z=1883→wall Z=1896):
                            #   argmax(dot) on the full 15mm sphere selects trunk verts
                            #   that are lateral-ish in XY but axially displaced, because
                            #   the IVC wall curves and the 15mm sphere captures verts
                            #   13mm cranial that happen to have a favourable XY dot.
                            # Fix: mask candidates to ±Z_BAND_MM of detect_gi Z before
                            #   lateral selection.  The ostial rim is at the SAME axial
                            #   level as the branch takeoff — not cranial or caudal.
                            Z_BAND_MM = 4.0
                            _snap_z_val = float(_snap_src[2])
                            _z_mask = (
                                np.abs(_tk_cand_pts[:, 2] - _snap_z_val) <= Z_BAND_MM
                            )
                            if np.any(_z_mask):
                                _tk_cand_pts = _tk_cand_pts[_z_mask]
                                _tk_vecs = _tk_vecs[_z_mask]
                                _tk_dists = _tk_dists[_z_mask]
                            # else: no verts in Z band — keep full set (fallback)
                            # ── Select the trunk vert most lateral to the branch ──
                            # Root cause of anterior snap: argmin(dist) picks the
                            # nearest trunk vert which is the ANTERIOR IVC wall
                            # (closest face to the CL regardless of direction).
                            # Fix: project candidates onto _t_fwd (branch tangent at
                            # detect_gi, pointing laterally away from the IVC).
                            # argmax(dot) picks the vert furthest in the branch
                            # departure direction = the lateral IVC wall face where
                            # the renal vein actually enters.
                            # Guard: normalise vectors, zero-length vecs get dot=0.
                            _tk_lens = np.maximum(_tk_dists, 1e-6)
                            _tk_dots = (_tk_vecs / _tk_lens[:, None]) @ _t_fwd
                            _tk_lat_idx = int(np.argmax(_tk_dots))
                            _snap_pt = _tk_cand_pts[_tk_lat_idx]
                            _snap_d = float(_tk_dists[_tk_lat_idx])
                            _snap_label = (
                                f"trunk_surf_lateral detect_gi={_detect_gi} "
                                f"dot={float(_tk_dots[_tk_lat_idx]):.2f} "
                                f"n_cand={len(_tk_cand_idxs)}"
                            )
                            # Diagnostics: top-5 by lateral dot score
                            _tk_sort = np.argsort(-_tk_dots)[:5]
                            _tk_diag = [
                                f"  [{float(p[0]):.1f},{float(p[1]):.1f},{float(p[2]):.1f}] d={float(_tk_dists[i]):.1f}mm dot={float(_tk_dots[i]):.2f}"
                                for i, p in zip(_tk_sort, _tk_cand_pts[_tk_sort])
                            ]
                            print(
                                f"[SnapDiag] Branch {bi}: top-5 TRUNK verts by lateral dot "
                                f"from CL[{float(_snap_src[0]):.1f},{float(_snap_src[1]):.1f},{float(_snap_src[2]):.1f}]:\n"
                                + "\n".join(_tk_diag)
                            )
                        else:
                            # No trunk verts in radius — try unshifted origin
                            _tk_cand_idxs2 = _trunk_surf_kd.query_ball_point(
                                _snap_src, SEARCH_R_MM
                            )
                            if _tk_cand_idxs2:
                                _tk_cand_pts2 = _trunk_surf_pts[_tk_cand_idxs2]
                                _tk_vecs2 = _tk_cand_pts2 - _snap_src
                                _tk_dists2 = np.linalg.norm(_tk_vecs2, axis=1)
                                _ni2 = int(np.argmin(_tk_dists2))
                                _snap_pt = _tk_cand_pts2[_ni2]
                                _snap_d = float(_tk_dists2[_ni2])
                                _snap_label = f"trunk_surf_unshifted detect_gi={_detect_gi} (no offset cands)"
                            else:
                                _tk_d, _tk_idx = _trunk_surf_kd.query(_snap_src)
                                _snap_pt = _trunk_surf_pts[_tk_idx]
                                _snap_d = float(_tk_d)
                                _snap_label = f"trunk_surf_nearest detect_gi={_detect_gi} (no radius cands)"

                    # ── Fallback: branch-classified verts (previous primary) ───────
                    if _snap_pt is None and bi in _surf_kd_per_branch:
                        _bkd, _bpts = _surf_kd_per_branch[bi]
                        _cand_idxs = _bkd.query_ball_point(_snap_query_pt, SEARCH_R_MM)
                        if _cand_idxs:
                            _cand_pts = _bpts[_cand_idxs]
                            _vecs = _cand_pts - _snap_src
                            _dists = np.linalg.norm(_vecs, axis=1)
                            _near_idx = int(np.argmin(_dists))
                            _snap_pt = _cand_pts[_near_idx]
                            _snap_d = float(_dists[_near_idx])
                            _snap_label = (
                                f"branch_surf_fallback bi={bi} "
                                f"detect_gi={_detect_gi} "
                                f"n_cand={len(_cand_idxs)}"
                            )
                            _sort_idx = np.argsort(_dists)[:5]
                            _diag_pts = [
                                f"  [{float(p[0]):.1f},{float(p[1]):.1f},{float(p[2]):.1f}] d={float(_dists[i]):.1f}mm"
                                for i, p in zip(_sort_idx, _cand_pts[_sort_idx])
                            ]
                            print(
                                f"[SnapDiag] Branch {bi}: 5 nearest branch-classified verts "
                                f"(fallback) from CL[{float(_snap_src[0]):.1f},{float(_snap_src[1]):.1f},{float(_snap_src[2]):.1f}]:\n"
                                + "\n".join(_diag_pts)
                            )
                        else:
                            _bd, _bidx = _bkd.query(_snap_src)
                            _snap_pt = _bpts[_bidx]
                            _snap_d = float(_bd)
                            _snap_label = (
                                f"branch_surf_fallback bi={bi} (no radius cands)"
                            )

                    # ── Raw CL coordinate fallback ────────────────────────────────
                    if _snap_pt is None:
                        _snap_pt = _snap_src
                        _snap_d = 0.0
                        _snap_label = f"CL[detect_gi={_detect_gi}] fallback"

                    # ── Hard distance cap (belt-and-suspenders) ──────────────────
                    # argmax(dot) selects the most lateral trunk vert, which can be
                    # up to SEARCH_R_MM=15mm from the CL (IVC radius ~10mm).
                    # If snap_d exceeds the search radius fall back to raw CL.
                    MAX_SNAP_MM = SEARCH_R_MM
                    if _snap_d > MAX_SNAP_MM:
                        print(
                            f"[OstiumSnap] Branch {bi}: snap_d={_snap_d:.1f}mm "
                            f"> MAX_SNAP_MM={MAX_SNAP_MM:.1f}mm — "
                            f"snap rejected, using CL[detect_gi={_detect_gi}]"
                        )
                        _snap_pt = _snap_src
                        _snap_d = 0.0
                        _snap_label = f"CL[detect_gi={_detect_gi}] snap_capped"

                    _snap_coord = tuple(float(x) for x in _snap_pt)

                    # Commit ostiumGi → detect_gi (diameter pipeline anchor)
                    # Commit display coords → directionally-filtered wall snap
                    self.branchMeta[bi]["ostiumGi"] = _detect_gi
                    self.branchMeta[bi]["ostiumGiRefined"] = _detect_gi
                    self.branchMeta[bi]["ostium"] = _snap_coord
                    self.branchMeta[bi]["ostium_p3"] = _snap_coord
                    self.branches[bi] = (_detect_gi, e)

                    _det_ratio = next(
                        (r for gi, r in _snap_scores if gi == _detect_gi), 0.0
                    )
                    _det_lat = _lat_offset(_detect_gi)
                    _cl_pt = [round(float(x), 1) for x in _snap_src]
                    _wall_pt = [round(float(x), 1) for x in _snap_pt]
                    _delta_mm = float(
                        np.linalg.norm(
                            np.array(_snap_pt, dtype=float)
                            - np.array(_snap_src, dtype=float)
                        )
                    )
                    _snap_stability = (
                        "stable"
                        if _delta_mm < 2.0
                        else "drifting" if _delta_mm < 8.0 else "far"
                    )
                    print(
                        f"[OstiumSnap] branch={bi} "
                        f"gi={_detect_gi} "
                        f"ratio={_det_ratio:.3f} "
                        f"thresh={DIAM_ENTER_THRESH} "
                        f"lat={_det_lat:.1f}mm "
                        f"surface={_snap_label} "
                        f"CL_pt={_cl_pt} "
                        f"wall_pt={_wall_pt} "
                        f"snap_d={_snap_d:.1f}mm "
                        f"delta_mm={_delta_mm:.1f} "
                        f"stability={_snap_stability}"
                    )
                except Exception as _snap_err:
                    import traceback

                    print(f"[OstiumSnap] Branch {bi}: failed — {_snap_err}")
                    traceback.print_exc()


    def _logOstiumTrainingRecord(self):
        """Log per-branch features + outcome; auto-generate negative samples.

        Called automatically after _computeOstiumConfidence().

        Strategy
        ────────
        For each true (positive) ostium gi we generate multiple negative
        training samples by shifting the observation window along the branch
        centerline.  This gives the model realistic "almost right but wrong"
        examples using existing geometry — no manual annotation needed.

        Positive record  → y = 1.0   (true ostium, grade HIGH/MEDIUM)
        Near-miss neg    → y = 0.35  (shifted ≤ 15 pts; similar signals, wrong pos)
        Hard neg         → y = 0.0   (shifted > 15 pts; clearly wrong signals)
        Cross-branch neg → y = 0.0   (another branch's ostium features, wrong branch)

        Record schema (all records)
        ────────────────────────────
        {
            "type":     str    main / renal / side_short / side
            "mode":     str    arterial | venous
            "features": dict   14 FEATURE_KEYS (built via build_features)
            "y":        float  target label 0.0 / 0.35 / 1.0
            "grade":    str    original pipeline grade (positives only)
            "eff":      float  original effective_score (positives only)
            "neg_type": str    absent on positives; "near" / "hard" / "cross"
        }
        """
        import json, os, math

        if not getattr(self, "branches", None):
            return
        if not getattr(self, "_logreg", None):
            return

        _venous = getattr(self, "vesselType", "arterial") == "venous"

        # ── Collect per-branch data ───────────────────────────────────────
        branch_data = []  # list of (bi, btype, ogi, s, e, oc, meta)
        for bi, (s, e) in enumerate(self.branches):
            if bi == 0:
                continue
            meta = self.branchMeta.get(bi, {})
            oc = meta.get("ostium_confidence")
            if oc is None:
                continue
            _arc = 0.0
            try:
                _bs, _be = self.branches[bi]
                if _be - 1 < len(self.distances) and _bs < len(self.distances):
                    _arc = abs(self.distances[_be - 1] - self.distances[_bs])
            except Exception:
                pass
            role = meta.get("role", "")
            btype = self._logreg.classify_branch(role, _arc)
            ogi = meta.get("ostiumGi", s)
            branch_data.append((bi, btype, ogi, s, e, oc, meta))

        if not branch_data:
            return

        new_records = []

        # ── Per-branch: positive + shifted negatives ──────────────────────
        DIAM_WIN = 8

        def _safe_mean(vals):
            v = [x for x in vals if x and x > 0.5]
            return sum(v) / len(v) if v else 0.0

        def _extract_signals_at(gi_target, s, e, meta, oc_ref):
            """Extract a compact signal approximation at a shifted gi.

            We can't rerun the full scoring pipeline at arbitrary gi values,
            so we approximate the 9 core signals using diameter data and the
            reference geometry already stored in ostium_confidence.
            The key discriminating signals are V (diameter gradient), S
            (stability), and Z (trunk proximity), all of which change
            meaningfully when gi shifts away from the true ostium.
            """
            comps_ref = dict(oc_ref.get("components", {}))
            flags_ref = list(oc_ref.get("flags", []))

            gi = max(s, min(e - 1, gi_target))

            # V: diameter gradient — proxy via diameter drop at shifted gi
            prox_lo = max(s, gi - DIAM_WIN)
            dist_hi = min(e, gi + DIAM_WIN)
            prox_v = [
                self.diameters[g]
                for g in range(prox_lo, gi)
                if g < len(self.diameters) and self.diameters[g] > 0.5
            ]
            dist_v = [
                self.diameters[g]
                for g in range(gi, dist_hi)
                if g < len(self.diameters) and self.diameters[g] > 0.5
            ]
            pm, dm = _safe_mean(prox_v), _safe_mean(dist_v)
            if pm > 0 and dm > 0:
                grad = (pm - dm) / pm
                v_sc = max(0.0, min(1.0, (grad - 0.08) / 0.17))
            else:
                v_sc = comps_ref.get("V", 0.45)

            # S: stability — CV of post-shift window
            stab_v2 = [
                self.diameters[g]
                for g in range(gi, dist_hi)
                if g < len(self.diameters) and self.diameters[g] > 0.5
            ]
            sm2 = _safe_mean(stab_v2)
            if sm2 > 0 and len(stab_v2) >= 2:
                std2 = math.sqrt(sum((x - sm2) ** 2 for x in stab_v2) / len(stab_v2))
                cv2 = std2 / sm2
                s_sc = max(0.0, min(1.0, 1.0 - (cv2 - 0.08) / 0.17))
            else:
                s_sc = comps_ref.get("S", 0.5)

            # Z: trunk proximity — degrades if shifted away from true ogi
            # Use reference Z score degraded by distance
            true_ogi = meta.get("ostiumGi", s)
            dist_pts = abs(gi - true_ogi)
            z_sc = comps_ref.get("Z", 0.5) * math.exp(-dist_pts / 12.0)

            # A / L / T / G / C / D: keep from reference (geometry doesn't change
            # much over ±20 pts; the key signal shifts are V, S, Z)
            comps_shifted = {
                "V": v_sc,
                "A": comps_ref.get("A", 0.5),
                "L": comps_ref.get("L", 0.5),
                "T": comps_ref.get("T", 0.5) * math.exp(-dist_pts / 20.0),
                "G": comps_ref.get("G", 0.5),
                "S": s_sc,
                "C": comps_ref.get("C", 0.35),
                "Z": z_sc,
                "D": comps_ref.get("D", 0.0),
            }
            # Flags: add area_low if V degraded, zone_distant if Z degraded
            flags_shifted = list(flags_ref)
            if v_sc < 0.20 and "area_low" not in flags_shifted:
                flags_shifted.append("area_low")
            if z_sc < 0.30 and "zone_distant" not in flags_shifted:
                flags_shifted.append("zone_distant")

            return comps_shifted, flags_shifted

        for bi, btype, ogi, s, e, oc, meta in branch_data:
            comps = oc.get("components", {})
            flags = oc.get("flags", [])
            grade = oc.get("grade", "LOW")
            eff = float(oc.get("effective_score", 0.0))

            feats_pos = self._logreg.build_features(comps, flags)

            # ── Positive record ───────────────────────────────────────────
            new_records.append(
                {
                    "type": btype,
                    "mode": "venous" if _venous else "arterial",
                    "features": feats_pos,
                    "y": 1.0,
                    "grade": grade,
                    "eff": eff,
                }
            )

            # ── Shifted negatives ─────────────────────────────────────────
            # Near-miss: ±8–15 pts from true ostium (hard case, y=0.35)
            # Hard neg:  ±20–40 pts from true ostium (easy case, y=0.0)
            # Limit negative window to branch boundaries.
            branch_len = e - s
            near_offsets = []
            hard_offsets = []

            for sign in (-1, +1):
                for d in (8, 12):
                    gi2 = ogi + sign * d
                    if s <= gi2 < e:
                        near_offsets.append(gi2)
                for d in (20, 35):
                    gi2 = ogi + sign * d
                    if s <= gi2 < e:
                        hard_offsets.append(gi2)

            for gi2 in near_offsets:
                c2, fl2 = _extract_signals_at(gi2, s, e, meta, oc)
                new_records.append(
                    {
                        "type": btype,
                        "mode": "venous" if _venous else "arterial",
                        "features": self._logreg.build_features(c2, fl2),
                        "y": 0.35,
                        "neg_type": "near",
                    }
                )

            for gi2 in hard_offsets:
                c2, fl2 = _extract_signals_at(gi2, s, e, meta, oc)
                new_records.append(
                    {
                        "type": btype,
                        "mode": "venous" if _venous else "arterial",
                        "features": self._logreg.build_features(c2, fl2),
                        "y": 0.0,
                        "neg_type": "hard",
                    }
                )

        # ── Cross-branch negatives ────────────────────────────────────────
        # Each branch's ostium features are a negative for every OTHER branch.
        # y = 0.0 — a renal ostium signature on a main branch is clearly wrong.
        for i, (bi_a, btype_a, _, _, _, oc_a, _) in enumerate(branch_data):
            for j, (bi_b, btype_b, _, _, _, oc_b, _) in enumerate(branch_data):
                if i == j:
                    continue
                if btype_a == btype_b:
                    continue  # skip same type — too ambiguous
                comps_b = oc_b.get("components", {})
                flags_b = oc_b.get("flags", [])
                new_records.append(
                    {
                        "type": btype_a,  # scored as if it were branch A
                        "mode": "venous" if _venous else "arterial",
                        "features": self._logreg.build_features(comps_b, flags_b),
                        "y": 0.0,
                        "neg_type": "cross",
                    }
                )

        # ── Load existing data, append, save ─────────────────────────────
        all_records = []
        try:
            if os.path.exists(self._logreg_data_path):
                with open(self._logreg_data_path, "r") as fh:
                    all_records = json.load(fh)
        except Exception:
            all_records = []

        all_records.extend(new_records)

        # Cap total at 2000 most recent records to prevent unbounded growth
        if len(all_records) > 2000:
            all_records = all_records[-2000:]

        try:
            with open(self._logreg_data_path, "w") as fh:
                json.dump(all_records, fh, indent=2)
        except Exception as exc:
            print(f"[LogReg] data save failed: {exc}")
            return

        # ── Train if enough data per type ─────────────────────────────────
        trained_any = False
        for btype in self._logreg.BRANCH_TYPES:
            type_recs = [r for r in all_records if r.get("type") == btype]
            if len(type_recs) >= self._logreg.MIN_TRAIN_N:
                self._logreg.train_step(type_recs, btype)
                trained_any = True

        if trained_any:
            self._logreg.save(self._logreg_weights_path)

        total = len(all_records)
        n_pos = sum(1 for r in all_records if r.get("y", 1.0) >= 1.0)
        n_neg = total - n_pos
        per = {
            bt: sum(1 for r in all_records if r.get("type") == bt)
            for bt in self._logreg.BRANCH_TYPES
        }
        print(
            f"[LogReg] dataset: total={total} pos={n_pos} neg={n_neg} "
            + " ".join(f"{bt}={n}" for bt, n in per.items())
        )


    def _computeOstiumConfidence(self):
        """Compute a per-ostium confidence score for every non-trunk branch.

        Six independent signals + renal bonus, weighted additive fusion.  No single zero can
        collapse the overall score.  A strong-signal boost layer can promote a
        branch to HIGH when multiple signals agree, even if one is weak.

        Signals (weighted sum → base confidence in [0, 1])
        ────────────────────────────────────────────────────
        V  gradient_score   (weight 0.12)
            Diameter gradient across the ostium zone.

        A  angle_score      (weight 0.17)
            Sigmoid of the refined departure angle centred at 30°, width 15°.

        L  length_score     (weight 0.17)
            branch_length / (0.4 × trunk_length), capped at 1.

        T  topology_score   (weight 0.12)
            Graph-topology quality of the branch classification.
            Now includes a floor of 0.60 when ostium is within 25mm of bif
            (dense bif region penalised topology even for genuine branches).

        G  geometry_score   (weight 0.20)
            Spatial sanity: anchor attachment, length, diameter ratio.

        C  curvature_score  (weight 0.09)
            Centerline bend at winning ostium, measured over ~18mm arc
            (was 4pts ≈ 5mm — too local for renal bends).

        Z  zone_score       (weight 0.13)  ← NEW (Fix 1)
            exp(-dist_to_trunk / 5mm) at the refined ostium point.
            Converts "correct but weak local signals" → confident.
            Branch 4 is ON the trunk wall → zone_score ≈ 0.8–1.0.

        D  div_score        (weight 0.08, side branches only)  ← NEW (Fix 4)
            Perpendicular offset from trunk axis at ~20mm arc downstream.
            Renal veins diverge 20–60mm off-axis even with shallow proximal
            angle.  IVC duplicates stay within 5–10mm.

        S  stability_score  (weight 0.00, boost signal only)
            Diameter CV in the post-ostium stable window.

        R  renal_bonus (additive, renal branches only)
            +0.10 × ortho_score + 0.07 × lat_sc.

        Grade bands (lower-bound grading)
        ───────────────────────────────────
        effective_score = (confidence − flag_penalty) − component_spread

        Arterial mode:  HIGH ≥ 0.70 / MEDIUM ≥ 0.50 / LOW ≥ 0.35 / REJECT
        Venous mode:    HIGH ≥ 0.60 / MEDIUM ≥ 0.40 / LOW ≥ 0.25 / REJECT
        (venous thresholds −0.10 across the board: flat gradients and low
        curvature are physiological in IVC/iliac confluences, not defects)

        Weight profiles
        ───────────────
        Arterial: wA=0.17 wL=0.17 wT=0.12 wG=0.20 wC=0.09 wZ=0.13 wD=0.08 wV≈0.12+0.12
        Venous:   wA=0.15 wL=0.18 wT=0.22 wG=0.22 wC=0.04 wZ=0.14 wD=0.05 wV≈0.06+0.06
        (venous: topology+geometry dominant; gradient/curvature de-weighted)

        Flag enforcement (penalties subtracted from confidence, capped at 0.30)
        ─────────────────────────────────────────────────────────────────────────
        gradient_weak / topology_uncertain: −0.08 each
        curvature_flat / geometry_weak:     −0.05 each
        area_low / lumen_unstable / zone_distant / divergence_low: −0.04 each

        Strong-signal count (diagnostic only, no longer promotes grade)
        ──────────────────────────────────────────────────────────────────
        Signals: V>0.50, A>0.55, L>0.65, T>0.55, G>0.65, S>0.65, C>0.50,
                 Z>0.55, D>0.45.
        Retained in result dict as 'strong_count' for log inspection.

        Results stored in branchMeta[bi]['ostium_confidence']:
            score            float   raw weighted confidence
            penalised_score  float   after flag penalties
            effective_score  float   penalised − component_spread (drives grade)
            grade            str     HIGH / MEDIUM / LOW / REJECT
            uncertainty      float   std of 9 component values (spread measure)
            flag_penalty     float   total deduction from flags (capped 0.30)
            strong_count     int     number of strong signals (0–9, diagnostic)
            components       dict    V/A/L/T/G/S/C/Z/D breakdown
            flags            list    enforced penalty keys
        """
        import math

        if not getattr(self, "branches", None):
            return

        bif_pt = getattr(self, "bifurcationPoint", None)
        has_radius = getattr(self, "_hasVmtkRadius", False)
        # ── Vessel-type mode ─────────────────────────────────────────────────
        # In venous mode (IVC/iliac veins) the geometry is a confluence, not a
        # bifurcation.  Area gradients are shallow, curvature is low, and taper
        # is minimal — all by normal physiology.  We swap to a topology-dominant
        # weight profile and redefine ostium scoring accordingly.
        _venous = getattr(self, "vesselType", "arterial") == "venous"

        trunk_len = (
            self.branchMeta.get(0, {}).get("length_mm")
            or getattr(self, "_trunkLengthMm", 0.0)
            or 200.0
        )

        # ── Trunk pts for T / G anchor signals ───────────────────────────────
        t_s, t_e = self.branches[0]
        trunk_pts = self.points[t_s:t_e]
        _step = max(1, len(trunk_pts) // 60)
        trunk_sample = trunk_pts[::_step]

        # Trunk mean diameter (for diam_ratio in G)
        trunk_diams = [
            self.diameters[gi]
            for gi in range(t_s, t_e)
            if gi < len(self.diameters) and self.diameters[gi] > 0
        ]
        trunk_mean_d = (sum(trunk_diams) / len(trunk_diams)) if trunk_diams else 20.0

        def _nearest_trunk_dist(pt):
            best = 1e9
            best_tp = None
            px, py, pz = pt[0], pt[1], pt[2]
            for tp in trunk_sample:
                d = math.sqrt((px - tp[0]) ** 2 + (py - tp[1]) ** 2 + (pz - tp[2]) ** 2)
                if d < best:
                    best = d
                    best_tp = tp
            return best, best_tp

        _clamp = lambda v, lo=0.0, hi=1.0: geo_clamp(v, lo, hi)

        def _sigmoid(x, centre, width):
            return 1.0 / (1.0 + math.exp(-(x - centre) / max(width, 1e-6)))

        def _safe_mean(vals):
            v = [x for x in vals if x and x > 0.5]
            return sum(v) / len(v) if v else 0.0

        def _safe_stdev(vals, mean):
            v = [x for x in vals if x and x > 0.5]
            if len(v) < 2:
                return 0.0
            return math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))

        DIAM_WIN = 10  # pts for proximal/distal windows in gradient + stability

        print(
            f"[OstiumConfidence] Per-branch confidence (mode={'venous' if _venous else 'arterial'}):"
        )

        qa_high, qa_review, qa_reject = [], [], []

        for bi, (s, e) in enumerate(self.branches):
            if bi == 0:
                continue  # trunk has no ostium

            meta = self.branchMeta.get(bi, {})
            role = meta.get("role", "")
            angle = meta.get("angle_deg", 0.0) or 0.0
            length = meta.get("length_mm", 0.0) or 0.0
            ogi = meta.get("ostiumGi")  # refined ostium (== s after trim+refine)
            sgi = meta.get("stableStartGi")
            is_main = role in ("main", "iliac_left", "iliac_right")
            is_renal_fragment = (
                role == "renal_fragment"
            )  # short but anatomically real renal branch

            # s = self.branches[bi][0] after branch-trim AND _refineOstia, so s == ogi.
            # The pre-refinement topological start was the PREVIOUS self.branches[bi][0]
            # before _refineOstia updated it.  We recover it as the point closest to
            # snappedBifPt on the branch for main branches; for side branches, s is
            # already very close to the trunk (within 1–2 pts of the topological node).
            # For topology and geometry anchor checks we use `s` (current start), which
            # for side branches is reliable; for main branches we additionally check
            # against snappedBifPt using the first point of the branch (s) which was the
            # topological ogi BEFORE refinement marched it into individual vessel lumen.
            #
            # NOTE: after _refineOstia the refined ogi can be 20–35mm distal from the
            # topological bif node.  Using the refined point for T would give T=0.00
            # for main branches.  We therefore use self.points[s] for topology/anchor
            # checks — s is the current branch start (after trim but this is typically
            # within 1pt of the original graph ogi).  For the geometry anchor we also
            # fall back to the raw s-point distance to snappedBifPt.

            n_pts = len(self.points)
            topo_pt = self.points[s] if s < n_pts else None  # topological anchor
            ostium_pt = (
                self.points[ogi] if (ogi is not None and ogi < n_pts) else topo_pt
            )

            # ── V: diameter gradient across ostium ────────────────────────────
            # Prox window: up to DIAM_WIN pts from trunk ending just before s
            # (uses trunk diameters at the junction zone).
            # Dist window: DIAM_WIN pts starting from sgi (post-flare stable zone).
            prox_gi_lo = max(t_s, s - DIAM_WIN)
            prox_gi_hi = s
            prox_vals = [
                self.diameters[gi]
                for gi in range(prox_gi_lo, prox_gi_hi)
                if gi < len(self.diameters) and self.diameters[gi] > 0.5
            ]

            dist_gi_lo = sgi if (sgi is not None and sgi > s) else s
            dist_gi_hi = min(e, dist_gi_lo + DIAM_WIN)
            dist_vals = [
                self.diameters[gi]
                for gi in range(dist_gi_lo, dist_gi_hi)
                if gi < len(self.diameters) and self.diameters[gi] > 0.5
            ]

            prox_mean = _safe_mean(prox_vals)
            dist_mean = _safe_mean(dist_vals)

            if prox_mean > 0 and dist_mean > 0:
                # Positive gradient = narrowing from junction → branch lumen (expected)
                gradient = (prox_mean - dist_mean) / prox_mean
                # Score: 8% drop → 0.5; 25% drop → 1.0; negative (flare) → 0.0
                _raw_v = _clamp((gradient - 0.08) / (0.25 - 0.08))
            else:
                _raw_v = 0.45  # no data → slightly below neutral

            if has_radius:
                gradient_score = _raw_v
            else:
                # Surface-distance mode: blend toward 0.55 prior (partial trust)
                gradient_score = 0.55 * _raw_v + 0.45 * 0.55

            # ── A: angle score ────────────────────────────────────────────────
            angle_score = _clamp(_sigmoid(angle, centre=30.0, width=15.0))

            # ── L: length score ───────────────────────────────────────────────
            # renal_fragment branches are short by construction (RenalConsolidate
            # marks the shorter of a co-ostial pair as fragment).  Use the primary
            # branch length if available so the fragment isn't penalised for a
            # structural decision made earlier in the pipeline.
            _length_for_score = length
            if is_renal_fragment:
                _primary_bi = meta.get("fragment_of")
                if _primary_bi is not None:
                    _plen = (
                        self.branchMeta.get(_primary_bi, {}).get("length_mm", 0.0)
                        or 0.0
                    )
                    if _plen > length:
                        _length_for_score = _plen
            length_score = _clamp(_length_for_score / max(0.4 * trunk_len, 1.0))

            # ── T: topology score ─────────────────────────────────────────────
            # For MAIN branches: use graph role certainty.
            # Role iliac_left/iliac_right was assigned by IliacLabel after the full
            # bifurcation analysis — it is high-quality anatomical knowledge.
            # We do NOT use dist(refined_ostium, bif) here because _refineOstia
            # intentionally marches the ostium 20–35mm into the branch body; that
            # distance says nothing about topology quality.
            #
            # For SIDE branches: use dist(topo_pt, trunk) normalised over 20mm.
            # The topological ogi (s) is 1–2 pts from the trunk wall, so dist ≈ 1–5mm.
            _dist_trunk_topo = None  # shared between T and new Z/D signals
            _nearest_trunk_pt = None
            if is_main:
                # Role is confirmed — start from a high base
                topology_score = 0.90
                # Micro-segment upstream (e.g. 3.1mm node 318) → soft degradation
                _seg_arc = meta.get("_entrySegArc", 999.0) or 999.0
                if _seg_arc < 10.0:
                    topology_score *= 0.85
            else:
                # Side branch: how far has the topo_pt already separated from trunk?
                if topo_pt is not None:
                    _dist_trunk_topo, _nearest_trunk_pt = _nearest_trunk_dist(topo_pt)
                    topology_score = _clamp(_dist_trunk_topo / 20.0)

                    # Fix 3: topo floor near dense bif region
                    # When the ostium is within 25mm of the primary bif the graph is
                    # crowded and topo_sc reads low (0.1–0.2) even for genuine branches.
                    # This is attachment uncertainty, not anatomical ambiguity — floor it.
                    if bif_pt is not None:
                        _dist_from_bif = math.sqrt(
                            sum((topo_pt[k] - bif_pt[k]) ** 2 for k in range(3))
                        )
                        if _dist_from_bif < 25.0:
                            topology_score = max(topology_score, 0.60)
                else:
                    topology_score = 0.55
                _seg_arc = meta.get("_entrySegArc", 999.0) or 999.0
                if _seg_arc < 10.0:
                    topology_score *= 0.85
                topology_score = _clamp(topology_score)

            # ── G: geometry sanity ────────────────────────────────────────────
            G = 0.0

            # +0.40 — topological attachment to correct anatomical anchor.
            # Use topo_pt (branch start s) distance to snappedBifPt / trunk.
            if is_main and bif_pt is not None and topo_pt is not None:
                dist_bif_topo = math.sqrt(
                    sum((topo_pt[k] - bif_pt[k]) ** 2 for k in range(3))
                )
                # Topological ogi should be within ~15mm of bif; refined can be 30mm
                if dist_bif_topo <= 15.0:
                    G += 0.40
                else:
                    G += 0.40 * _clamp(1.0 - (dist_bif_topo - 15.0) / 30.0)
            elif not is_main and topo_pt is not None:
                if _dist_trunk_topo is None:
                    _dist_trunk_topo, _nearest_trunk_pt = _nearest_trunk_dist(topo_pt)
                if _dist_trunk_topo <= 8.0:
                    G += 0.40
                else:
                    G += 0.40 * _clamp(1.0 - (_dist_trunk_topo - 8.0) / 15.0)
            else:
                G += 0.20  # no anchor info → partial credit

            # +0.30 — branch long enough to be anatomically real
            # renal_fragment: _length_for_score already set to primary length above
            if _length_for_score > 60.0:
                G += 0.30
            elif _length_for_score > 30.0:
                G += 0.30 * ((_length_for_score - 30.0) / 30.0)

            # +0.30 — branch diameter meaningful fraction of trunk
            branch_diams = [
                self.diameters[gi]
                for gi in range(s, e)
                if gi < len(self.diameters) and self.diameters[gi] > 0
            ]
            branch_mean_d = _safe_mean(branch_diams)
            diam_ratio = branch_mean_d / max(trunk_mean_d, 1.0)
            if diam_ratio >= 0.40:
                G += 0.30
            elif diam_ratio > 0.20:
                G += 0.30 * ((diam_ratio - 0.20) / 0.20)

            geometry_score = _clamp(G)

            # ── S: stability (boost signal only, not in weighted sum) ─────────
            # Coefficient of variation of diameters in the first DIAM_WIN pts
            # of the stable zone.  Low CV = reproducible lumen = confident ostium.
            stab_vals = [
                self.diameters[gi]
                for gi in range(dist_gi_lo, min(e, dist_gi_lo + DIAM_WIN))
                if gi < len(self.diameters) and self.diameters[gi] > 0.5
            ]
            stab_mean = _safe_mean(stab_vals)
            stab_cv = (
                _safe_stdev(stab_vals, stab_mean) / stab_mean if stab_mean > 0 else 1.0
            )
            # CV < 0.08 → 1.0; CV > 0.25 → 0.0
            stability_score = _clamp(1.0 - (stab_cv - 0.08) / (0.25 - 0.08))

            # ── C: curvature at best ostium gi (from _refineOstia scores) ────
            # ostium_scores_v2['curv'] is the percentile-normalised centerline
            # bend angle at the winning candidate, measured over ~18mm arc.
            # 0 = straight, 1 = 45°+ bend.  Orthogonal to V — works even when
            # lumen gradient is flat.
            _scores_v2 = meta.get("ostium_scores_v2", {})
            curvature_score = _scores_v2.get("curv", 0.35)  # 0.35 neutral if absent

            # ── Area signal quality (from _refineOstia percentile stats) ─────
            # gradient_range = p90 - p10 of the raw ag distribution in the scan
            # window.  A flat signal (all ag ≈ 0) has range ≈ 0.  A meaningful
            # gradient has range ≥ 0.25 regardless of where the peak sits.
            # peak_strength = max(area_sc_norm) - median(area_sc_norm).
            # Together these answer "is the area signal informative?" rather than
            # "is the area value high?", which is invalid after percentile-norm.
            _area_range = _scores_v2.get("area_range", None)
            _area_peak_str = _scores_v2.get("area_peak_str", None)
            _area_sc_best = _scores_v2.get("area", 0.0)  # percentile-norm best area_sc

            # Gradient informativeness — distribution-based (Fix 1)
            # Main branches (iliacs) have a gradual area transition at the aortic Y
            # — the thresholds are relaxed so a smooth but real gradient is not
            # penalised as "weak".  Side branches keep the stricter thresholds.
            if _area_range is not None and _area_peak_str is not None:
                if is_main:
                    _area_signal_weak = (
                        _area_range < 0.15  # main: accept gentler gradient
                        or _area_peak_str < 0.10
                    )
                else:
                    _area_signal_weak = (
                        _area_range < 0.25  # flat distribution → no real gradient
                        or _area_peak_str < 0.15  # no dominant peak → noise
                    )
            else:
                # _refineOstia was not run (no surface) — fall back to V score
                _area_signal_weak = gradient_score < 0.40

            # ── Z: ostium zone prior (Fix 1 — biggest impact for smooth renals) ─
            # exp(-dist_to_trunk / 5mm) at the refined ostium point.
            # A branch with weak local signals (gradient_flat, curvature_flat) but
            # whose ostium is demonstrably ON the trunk wall gets promoted here.
            # "Correct but weak" → confident.
            zone_score = 0.35  # neutral default
            if not is_main and ostium_pt is not None:
                _z_dist, _ = _nearest_trunk_dist(ostium_pt)
                zone_score = _clamp(math.exp(-_z_dist / 5.0))
            elif is_main:
                zone_score = 0.70  # main branches always close to bif → high prior

            # ── D: lateral divergence from trunk (Fix 4 — shallow-angle renals) ─
            # Sample the branch ~20mm downstream of the ostium and measure
            # perpendicular offset from the trunk axis.  Renal veins diverge
            # 20–60mm off-axis even when their proximal angle is shallow (<20°).
            # IVC duplicates/collaterals stay within 5–10mm.
            div_score = 0.0
            if not is_main and ostium_pt is not None:
                # Find the branch point ~20mm arc past the refined ostium
                _ogi = ogi if (ogi is not None and ogi < len(self.points)) else s
                _div_mm = 20.0
                _div_gi = _ogi
                _acc = 0.0
                while _div_gi + 1 < e and _acc < _div_mm:
                    _p0 = self.points[_div_gi]
                    _p1 = self.points[min(_div_gi + 1, e - 1)]
                    _acc += math.sqrt(sum((_p1[k] - _p0[k]) ** 2 for k in range(3)))
                    _div_gi += 1
                _div_pt = self.points[min(_div_gi, e - 1)]

                # Trunk axis direction (root → bif)
                if len(trunk_pts) >= 2:
                    _tr0, _tr1 = trunk_pts[0], trunk_pts[-1]
                    _tax = [_tr1[k] - _tr0[k] for k in range(3)]
                    _tl = math.sqrt(sum(v**2 for v in _tax)) + 1e-9
                    _tax = [v / _tl for v in _tax]
                    # Project div_pt onto trunk axis relative to ostium_pt
                    _v = [_div_pt[k] - ostium_pt[k] for k in range(3)]
                    _prj = sum(_v[k] * _tax[k] for k in range(3))
                    _perp = math.sqrt(
                        max(0.0, sum(_v[k] ** 2 for k in range(3)) - _prj**2)
                    )
                    # 20mm perp offset = full score; <5mm ≈ 0
                    div_score = _clamp((_perp - 5.0) / 15.0)

            # ── Branch arc (needed by LogReg branch classifier) ─────────────
            _branch_arc_lr = 0.0
            try:
                _bs_lr, _be_lr = self.branches[bi]
                if _be_lr - 1 < len(self.distances) and _bs_lr < len(self.distances):
                    _branch_arc_lr = abs(
                        self.distances[_be_lr - 1] - self.distances[_bs_lr]
                    )
            except Exception:
                pass

            # ── Confidence fusion — dual path (venous vs arterial) ───────────

            if _venous:
                # ═══════════════════════════════════════════════════════════════
                # VENOUS confidence formula
                # Primary signals: lateral offset + Z-position + diam ratio.
                # Area gradient and curvature are flat by physiology → minimised.
                # ═══════════════════════════════════════════════════════════════

                # (A) Lateral score — PRIMARY for renal veins / iliac tributaries
                # tip_lateral_mm = perpendicular offset of branch tip from trunk axis.
                # Renal veins: 30–80mm.  IVC duplicates: <15mm.
                _lat_mm = meta.get("tip_lateral_mm") or 0.0
                lateral_score = _clamp(_lat_mm / 40.0)  # 40mm → 1.0; floor 0.0

                # (B) Z-position score — renal veins sit mid-IVC, above bifurcation
                # Use the ostium Z relative to the primary bifurcation Z.
                # bif_z already computed as _bif_z_ref via trunkPtsOrdered[-1][2].
                _bif_z = getattr(self, "_bif_z_ref", None)
                if (
                    _bif_z is None
                    and hasattr(self, "trunkPtsOrdered")
                    and self.trunkPtsOrdered
                ):
                    _bif_z = self.trunkPtsOrdered[-1][2]
                _ost_z = self.points[s][2] if s < len(self.points) else None
                if _bif_z is not None and _ost_z is not None:
                    _dz = _ost_z - _bif_z  # positive = superior to bifurcation
                    # Renal band: 20–120mm above bif (Gaussian centred at 70mm σ=40mm)
                    _z_centre = 70.0
                    _z_sigma = 40.0
                    z_pos_score = math.exp(-0.5 * ((_dz - _z_centre) / _z_sigma) ** 2)
                    # Side branches near bif (<10mm) get a gentle floor
                    if _dz < 10.0:
                        z_pos_score = max(z_pos_score, 0.30)
                else:
                    z_pos_score = 0.50  # no anchor info → neutral

                # (C) Diameter ratio score
                # Optimal venous range: 0.45–0.90 × trunk.  Outside → soft ramp.
                if diam_ratio >= 0.45 and diam_ratio <= 0.90:
                    diam_ratio_score = 1.0
                elif diam_ratio < 0.45:
                    diam_ratio_score = _clamp(diam_ratio / 0.45)
                else:  # > 0.90 — unusually large, slight penalty
                    diam_ratio_score = _clamp(1.0 - (diam_ratio - 0.90) / 0.40)

                # (D) Topology score — softened default for venous
                # Venous confluences have diffuse graph attachments → floor at 0.60
                # so "slightly diffuse" doesn't collapse the score.
                topo_v = max(topology_score, 0.60) if topo_pt is not None else 0.60

                # (E) Stability — unchanged (still useful to reject noise)
                stab_v = stability_score

                # (F) Curvature — neutral 0.50 default; only penalise erratic values
                # (very high curvature = noisy mesh, not a real bend)
                curv_v = 0.50 if curvature_score < 0.70 else curvature_score

                # (G) Area gradient — optional boost; flat = 0.50 neutral
                area_v = (
                    0.50 if _area_signal_weak else min(1.0, 0.50 + _area_sc_best * 0.50)
                )

                # ── LogReg: replace venous manual weighted sum ───────────────
                # Map venous sub-signals onto the canonical V/A/L/T/G/S/C/Z/D vector:
                #   V = area_v (gradient quality, neutral in venous)
                #   A = lateral_score (PRIMARY — replaces angle in venous)
                #   L = length_score (already computed above)
                #   T = topo_v (topology, softened floor)
                #   G = diam_ratio_score (diameter-ratio sanity)
                #   S = stab_v (lumen stability)
                #   C = curv_v (curvature, neutral in venous)
                #   Z = z_pos_score (Z-position in IVC)
                #   D = div_score (divergence — already 0 for main in venous path)
                _lr_btype_v = self._logreg.classify_branch(role, _branch_arc_lr)
                # Build 9-signal component dict; flags computed here (before result dict)
                _lr_comps_v = {
                    "V": area_v,
                    "A": lateral_score,
                    "L": length_score,
                    "T": topo_v,
                    "G": diam_ratio_score,
                    "S": stab_v,
                    "C": curv_v,
                    "Z": z_pos_score,
                    "D": div_score,
                }
                # Pre-compute venous flags for feature builder
                # (full flag list built later; use key signals here)
                _lr_flags_v_pre = []
                if _area_signal_weak or _area_sc_best < 0.20:
                    _lr_flags_v_pre.append("area_low")
                if not is_main and zone_score < 0.30:
                    _lr_flags_v_pre.append("zone_distant")
                _lr_feats_v = self._logreg.build_features(_lr_comps_v, _lr_flags_v_pre)
                confidence = self._logreg.score(_lr_btype_v, _lr_feats_v)
                print(
                    f"  [LogReg-V] Branch {bi} type={_lr_btype_v} "
                    f"conf={confidence:.3f}"
                )

                # ── Venous override rule ──────────────────────────────────────
                # If ALL three anatomical gates pass, the branch is demonstrably
                # a real vessel — floor confidence at 0.75 (HIGH territory).
                # Gates: lat > 30mm AND diam_ratio > 0.45 AND length > 50mm.
                # renal_fragment: use primary length (_length_for_score) so a short
                # fragment of a confirmed real renal vein can still qualify.
                _venous_override = (
                    _lat_mm > 30.0 and diam_ratio > 0.45 and _length_for_score > 50.0
                )
                if _venous_override and confidence < 0.75:
                    confidence = 0.75

                # ── ShortBranchBoost (venous) ───────────────────────────────
                # Short branches (<30mm) in venous mode are often over-segmented
                # stubs. Their area/diam signals are weak by construction.
                # Boost topo by +0.15, neutralise area+curv, apply safety floor.
                if not is_main and not is_renal_fragment:
                    _branch_arc_sbb = 0.0
                    try:
                        _bs_sbb, _be_sbb = self.branches[bi]
                        if _be_sbb - 1 < len(self.distances) and _bs_sbb < len(
                            self.distances
                        ):
                            _branch_arc_sbb = abs(
                                self.distances[_be_sbb - 1] - self.distances[_bs_sbb]
                            )
                    except Exception:
                        pass
                    if _branch_arc_sbb > 0 and _branch_arc_sbb < 30.0:
                        _topo_v_sbb = min(1.0, topo_v + 0.15)
                        # Recompute via LogReg with boosted topo and neutral area/curv
                        _lr_comps_sbb = dict(_lr_comps_v)
                        _lr_comps_sbb["T"] = _topo_v_sbb  # boosted topology
                        _lr_comps_sbb["V"] = 0.50  # neutralise area
                        _lr_comps_sbb["C"] = 0.50  # neutralise curvature
                        _lr_feats_sbb = self._logreg.build_features(
                            _lr_comps_sbb, _lr_flags_v_pre
                        )
                        _conf2 = max(
                            0.0, self._logreg.score(_lr_btype_v, _lr_feats_sbb)
                        )
                        if _conf2 > confidence:
                            print(
                                f"[ShortBranchBoost] Branch {bi}: arc={_branch_arc_sbb:.1f}mm "
                                f"topo {topo_v:.2f}→{_topo_v_sbb:.2f} "
                                f"conf {confidence:.3f}→{_conf2:.3f}"
                            )
                            confidence = _conf2
                            topo_v = _topo_v_sbb
                        # Safety floor: strong topology + stable lumen → MEDIUM
                        if topo_v >= 0.60 and stab_v >= 0.80 and confidence < 0.45:
                            print(
                                f"[ShortBranchBoost] Branch {bi}: topo+stab floor 0.45"
                            )
                            confidence = 0.45

                # Expose venous components for result dict / log
                area_conf = area_v
                div_score = lateral_score  # reuse div_score slot for log compat

                # strong_count: count venous signals above their own thresholds
                strong = sum(
                    [
                        lateral_score > 0.60,
                        z_pos_score > 0.60,
                        diam_ratio_score > 0.80,
                        topo_v > 0.65,
                        stab_v > 0.65,
                    ]
                )

            else:
                # ═══════════════════════════════════════════════════════════════
                # ARTERIAL confidence formula (unchanged from v186)
                # ═══════════════════════════════════════════════════════════════

                # Renal orthogonality bonus (arterial mode — lateral is a bonus not primary)
                renal_bonus = 0.0
                if not is_main:
                    _ortho = meta.get("ortho_score")
                    _lat_mm_a = meta.get("tip_lateral_mm")
                    if _ortho is not None:
                        renal_bonus += 0.10 * max(0.0, min(1.0, _ortho))
                    if _lat_mm_a is not None:
                        renal_bonus += 0.07 * max(
                            0.0, min(1.0, (_lat_mm_a - 15.0) / 35.0)
                        )

                if _area_signal_weak:
                    area_conf = gradient_score
                else:
                    area_conf = 0.5 * _area_sc_best + 0.5 * gradient_score

                # ── LogReg: replace arterial manual weighted sum ─────────
                _lr_btype = self._logreg.classify_branch(role, _branch_arc_lr)
                _lr_comps = {
                    "V": area_conf,
                    "A": angle_score,
                    "L": length_score,
                    "T": topology_score,
                    "G": geometry_score,
                    "S": stability_score,
                    "C": curvature_score,
                    "Z": zone_score,
                    "D": div_score,
                }
                # Pre-compute arterial flags for feature builder
                _lr_flags_a_pre = []
                if _area_signal_weak:
                    _lr_flags_a_pre.append("gradient_weak")
                elif _area_sc_best < 0.20:
                    _lr_flags_a_pre.append("area_low")
                if zone_score < 0.30:
                    _lr_flags_a_pre.append("zone_distant")
                if not is_main and div_score < 0.20:
                    _lr_flags_a_pre.append("divergence_low")
                _lr_feats = self._logreg.build_features(_lr_comps, _lr_flags_a_pre)
                _lr_raw = self._logreg.score(_lr_btype, _lr_feats)
                confidence = min(1.0, _lr_raw + renal_bonus)
                print(
                    f"  [LogReg-A] Branch {bi} type={_lr_btype} "
                    f"lr={_lr_raw:.3f} renal_bonus={renal_bonus:.3f}"
                )

                _venous_override = False

                strong = sum(
                    [
                        not _area_signal_weak,
                        angle_score > 0.55,
                        length_score > 0.65,
                        topology_score > 0.55,
                        geometry_score > 0.65,
                        stability_score > 0.65,
                        curvature_score > 0.50,
                        zone_score > 0.55,
                        div_score > 0.45,
                    ]
                )

            # ── Flags → enforced penalties (not just logged) ──────────────────
            # Each flag represents a genuinely weak signal.  Penalties reduce
            # the raw confidence before grading so that weak signals actually
            # affect the grade.  Total cap: 0.30 so one bad signal cannot alone
            # collapse an otherwise strong result.
            #
            # Main branches (confirmed iliac role) receive halved penalties for
            # gradient/curvature/geometry flags: smooth anatomy at the aortic Y is
            # expected physiology, not evidence of misdetection.
            #
            # In venous mode, gradient_weak and curvature_flat carry zero penalty
            # (weight already de-rated in the weight profile) — flat gradients and
            # low curvature are expected in IVC/iliac vein confluences.
            flags = []
            _flag_penalty = 0.0
            _pen_scale = (
                0.5 if is_main else 1.0
            )  # halve structural penalties for confirmed iliacs
            # gradient_weak is distribution-based (range + peak_strength),
            # NOT a value threshold — prevents "A=0.91 but gradient_weak" contradiction.
            if _area_signal_weak:
                flags.append("gradient_weak")
                _flag_penalty += 0.0 if _venous else (0.08 * _pen_scale)
            elif _area_sc_best < 0.20:
                flags.append("area_low")
                _flag_penalty += 0.0 if _venous else (0.04 * _pen_scale)
            if curvature_score < 0.30:
                flags.append("curvature_flat")
                # renal_fragment: raw marching cubes surface gives noisy curvature
                # on short branches — suppress penalty (signal is unreliable, not wrong)
                _flag_penalty += (
                    0.0 if (_venous or is_renal_fragment) else (0.05 * _pen_scale)
                )
            if topology_score < 0.45 and not _venous:
                flags.append("topology_uncertain")
                _flag_penalty += 0.08  # full weight — topology is role-agnostic
            if geometry_score < 0.40:
                flags.append("geometry_weak")
                _flag_penalty += 0.05 * _pen_scale
            if stability_score < 0.40:
                flags.append("lumen_unstable")
                _flag_penalty += 0.04
            if zone_score < 0.30:
                flags.append("zone_distant")
                _flag_penalty += 0.04
            if (
                not is_main
                and div_score < 0.20
                and not _venous
                and not is_renal_fragment
            ):
                flags.append("divergence_low")
                _flag_penalty += 0.04
            _flag_penalty = min(_flag_penalty, 0.30)
            penalised_confidence = max(0.0, confidence - _flag_penalty)

            # ── Uncertainty (component spread) ────────────────────────────────
            # std of the 9-component distribution; wide spread = unreliable mean.
            _comp_vals = [
                area_conf,
                angle_score,
                length_score,
                topology_score,
                geometry_score,
                stability_score,
                curvature_score,
                zone_score,
                div_score,
            ]
            _mean_c = sum(_comp_vals) / len(_comp_vals)
            import math as _math

            _std_c = _math.sqrt(
                sum((v - _mean_c) ** 2 for v in _comp_vals) / len(_comp_vals)
            )
            uncertainty = round(_std_c, 3)

            # ── Grade (lower-bound) ───────────────────────────────────────────
            # effective_score = penalised_confidence − component_spread
            # Collapses "HIGH but shaky" outcomes: wide component spread and
            # flag penalties both pull effective down before grading.
            # strong_count is retained for diagnostics only (no longer promotes grade).
            effective = penalised_confidence - _std_c

            # Main-branch floor: a confirmed iliac with topology=0.90, zone=0.70,
            # and correct geometry cannot anatomically be a "REJECT" — the pipeline
            # has already proven it is a real branch via graph topology.
            # Arterial floor: 0.50 (MEDIUM) — smooth Y may have weak area signal.
            # Venous floor:   0.60 — flat gradients are physiological, not weak;
            #                 0.60 maps to HIGH under venous thresholds (see below).
            _main_floor = 0.60 if _venous else 0.50
            _main_floor_applied = False
            if is_main and effective < _main_floor:
                effective = _main_floor
                _main_floor_applied = True

            # ── Grading thresholds ────────────────────────────────────────────
            # Venous thresholds are raised by +0.10 across the board because
            # physiologically correct venous anatomy produces lower raw scores
            # (flat gradients, low curvature) — effective=0.50 in a vein is
            # equivalent confidence to effective=0.60 in an artery.
            if _venous:
                if effective >= 0.60:
                    grade = "HIGH"
                elif effective >= 0.40:
                    grade = "MEDIUM"
                elif effective >= 0.25:
                    grade = "LOW"
                else:
                    grade = "REJECT"
            else:
                if effective >= 0.70:
                    grade = "HIGH"
                elif effective >= 0.50:
                    grade = "MEDIUM"
                elif effective >= 0.35:
                    grade = "LOW"
                else:
                    grade = "REJECT"

            # area_quality_str for log legibility
            if _area_range is not None:
                _aq = (
                    f"range={_area_range:.2f} peak={_area_peak_str:.2f} "
                    f"{'[weak]' if _area_signal_weak else '[ok]'}"
                )
            else:
                _aq = "no_surface"

            result = {
                "score": round(confidence, 3),  # raw weighted score
                "penalised_score": round(penalised_confidence, 3),
                "effective_score": round(effective, 3),  # = penalised - spread
                "grade": grade,
                "uncertainty": uncertainty,  # component spread (std)
                "flag_penalty": round(_flag_penalty, 3),
                "strong_count": strong,
                "components": {
                    "V": round(area_conf, 2),  # area/gradient (venous: area_v)
                    "A": round(angle_score, 2),
                    "L": round(length_score, 2),
                    "T": round(topology_score, 2),
                    "G": round(geometry_score, 2),
                    "S": round(stability_score, 2),
                    "C": round(curvature_score, 2),
                    "Z": round(zone_score, 2),
                    "D": round(div_score, 2),  # venous: lateral_score
                },
                "flags": flags,
                "venous_override": _venous_override if _venous else False,
            }

            self.branchMeta.setdefault(bi, {})["ostium_confidence"] = result

            # ── Log ──────────────────────────────────────────────────────────
            label = (
                self.getBranchDisplayName(bi)
                if hasattr(self, "getBranchDisplayName")
                else f"Branch {bi}"
            )
            _pen_str = f" pen={_flag_penalty:.2f}[{','.join(flags)}]" if flags else ""
            _floor_str = " [main-floor]" if _main_floor_applied else ""
            _override_str = " [OVERRIDE]" if (_venous and _venous_override) else ""
            _mode_str = " [venous]" if _venous else ""
            print(
                f"  {label}: raw={confidence:.3f}{_pen_str} "
                f"→ eff={effective:.3f} ±{uncertainty:.3f} grade={grade}"
                f"{_floor_str}{_override_str}{_mode_str}"
            )
            comp = result["components"]
            if _venous:
                print(
                    f"    venous: lat={div_score:.2f} z={z_pos_score:.2f} "
                    f"diam={diam_ratio_score:.2f} topo={topo_v:.2f} "
                    f"stab={stab_v:.2f} curv={curv_v:.2f} area={area_v:.2f}"
                )
            else:
                print(
                    f"    components: V={comp['V']:.2f} A={comp['A']:.2f} "
                    f"L={comp['L']:.2f} T={comp['T']:.2f} G={comp['G']:.2f} "
                    f"S={comp['S']:.2f} C={comp['C']:.2f} "
                    f"Z={comp['Z']:.2f} D={comp['D']:.2f}"
                )
            print(f"    area_signal: {_aq}")
            if flags:
                print(f"    flags: {', '.join(flags)}")

            # ── QA bucket ─────────────────────────────────────────────────────
            if grade == "HIGH":
                qa_high.append(bi)
            elif grade in ("MEDIUM", "LOW"):
                qa_review.append(bi)
            else:  # REJECT
                qa_reject.append(bi)

        # ── QA filter summary ─────────────────────────────────────────────────
        print(
            f"[OstiumConfidence] QA summary: "
            f"HIGH={len(qa_high)} REVIEW={len(qa_review)} REJECT={len(qa_reject)}"
        )
        if qa_reject:
            names = [
                (
                    self.getBranchDisplayName(bi)
                    if hasattr(self, "getBranchDisplayName")
                    else f"Branch {bi}"
                )
                for bi in qa_reject
            ]
            print(
                f"[OstiumConfidence] REJECT branches (effective<0.35): {', '.join(names)}"
            )
        if qa_review:
            names = [
                (
                    self.getBranchDisplayName(bi)
                    if hasattr(self, "getBranchDisplayName")
                    else f"Branch {bi}"
                )
                for bi in qa_review
            ]
            print(
                f"[OstiumConfidence] REVIEW branches (manual check advised): {', '.join(names)}"
            )


# ── Slicer module-scanner guard ───────────────────────────────────────────────
# Slicer auto-scans all .py files in the module folder and expects a class
# matching the filename.  This stub satisfies that requirement without
# registering as a real loadable module (no ScriptedLoadableModule base).
class vessel_ostium_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_ostium_mixin"
            parent.hidden = True  # hide from Slicer module list
