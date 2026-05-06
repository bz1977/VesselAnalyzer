"""
vessel_findings_mixin.py — Diameter computation, ellipse fitting, tree annotation, and finding detection.

Part of the VesselAnalyzer mixin decomposition.
These methods are mixed into VesselAnalyzerLogic via multiple inheritance:

    class VesselAnalyzerLogic(
        FindingsMixin,
        ...
        ScriptedLoadableModuleLogic,
    ): ...

All methods use ``self`` normally — no changes to call sites required.
"""
# ruff: noqa  (this file is auto-extracted; formatting is inherited from VesselAnalyzer.py)

from centerline_analysis import detect_compression
import vtk
import math

class FindingsMixin:
    """Mixin: Diameter computation, ellipse fitting, tree annotation, and finding detection."""

    def _annotateTreeDiameters(self):
        """Annotate every segment in self.branchTree with diameter statistics.

        Strategy:
          Each tree segment carries a list of sparse cEdge pts (graph centroids).
          We map those pts to the nearest indices in self.points (dense centerline)
          via closest-point lookup, collect all diameters in that gi range, and
          compute mean / min / max / proximal / distal / taper.

        Segment dict gains these keys:
          diam_mean   float   trimmed mean over segment (mm)
          diam_min    float   minimum (mm) — lesion sentinel
          diam_max    float   maximum (mm)
          diam_prox   float   mean of first 10% of pts (proximal reference)
          diam_dist   float   mean of last 10% of pts (distal reference)
          taper       float   diam_prox - diam_dist  (+ = narrowing distally)
          gi_start    int     first global index in self.points for this segment
          gi_end      int     last  global index in self.points for this segment
          n_pts       int     number of self.points indices used
        """
        import math as _math

        pts_arr = self.points  # list of (x,y,z)
        diams = self.diameters  # list of float, same length

        if not pts_arr or not diams:
            return

        n_pts_total = len(pts_arr)

        # Build KD-tree once for fast nearest-point lookup
        try:
            import numpy as _np
            from scipy.spatial import cKDTree as _cKDTree

            _pts_np = _np.array(pts_arr, dtype=float)
            _ptree = _cKDTree(_pts_np)

            def _nearest_gi(xyz):
                _, idx = _ptree.query(xyz)
                return int(idx)

        except Exception:
            # Fallback: linear scan
            def _nearest_gi(xyz):
                best_d, best_i = 1e18, 0
                for _i, _p in enumerate(pts_arr):
                    _d = (
                        (_p[0] - xyz[0]) ** 2
                        + (_p[1] - xyz[1]) ** 2
                        + (_p[2] - xyz[2]) ** 2
                    )
                    if _d < best_d:
                        best_d = _d
                        best_i = _i
                return best_i

        def _trimmed_mean(vals, frac=0.1):
            if not vals:
                return 0.0
            s = sorted(v for v in vals if v > 0.5)
            if not s:
                return 0.0
            trim = max(1, int(len(s) * frac))
            mid = s[trim:-trim] if len(s) > 2 * trim else s
            return sum(mid) / len(mid) if mid else s[len(s) // 2]

        # Pre-build per-branch KD-trees so each segment searches only within
        # its correct branch gi range, preventing cross-branch contamination.
        # self.branches[bi] = (start_gi, end_gi exclusive)
        _branch_trees = []  # list of (cKDTree | None, gi_start, gi_end_excl)
        try:
            import numpy as _np2
            from scipy.spatial import cKDTree as _cKDTree2

            for _bi, (_bs, _be) in enumerate(self.branches):
                _bpts = [pts_arr[_gi] for _gi in range(_bs, _be) if _gi < n_pts_total]
                if _bpts:
                    _bt = _cKDTree2(_np2.array(_bpts, dtype=float))
                    _branch_trees.append((_bt, _bs, _be))
                else:
                    _branch_trees.append((None, _bs, _be))
        except Exception:
            _branch_trees = []

        def _nearest_gi_in_branch(xyz, bi):
            """Nearest gi within branch bi's range."""
            if _branch_trees and bi < len(_branch_trees):
                _bt, _bs, _be = _branch_trees[bi]
                if _bt is not None:
                    _, _local = _bt.query(xyz)
                    return _bs + int(_local)
            # fallback: global search
            return _nearest_gi(xyz)

        def _best_branch_for_seg(seg_pts):
            """Find which branch index best contains this segment's midpoint."""
            if not seg_pts:
                return 0
            mid = seg_pts[len(seg_pts) // 2]
            best_bi, best_d = 0, 1e18
            for _bi, (_bs, _be) in enumerate(self.branches):
                # Check a sample of pts in this branch
                _step = max(1, (_be - _bs) // 10)
                for _gi in range(_bs, _be, _step):
                    if _gi >= n_pts_total:
                        break
                    _p = pts_arr[_gi]
                    _d = (
                        (_p[0] - mid[0]) ** 2
                        + (_p[1] - mid[1]) ** 2
                        + (_p[2] - mid[2]) ** 2
                    )
                    if _d < best_d:
                        best_d = _d
                        best_bi = _bi
            return best_bi

        def _annotate_seg(node):
            seg = node.get("segment")
            if seg is not None and seg.get("pts"):
                sparse_pts = seg["pts"]

                # Identify which branch this segment belongs to
                best_bi = _best_branch_for_seg(sparse_pts)
                _bs, _be = self.branches[best_bi]
                _be = min(_be, n_pts_total)

                # Map segment endpoints to gi within the correct branch range
                gi_s = _nearest_gi_in_branch(sparse_pts[0], best_bi)
                gi_e = _nearest_gi_in_branch(sparse_pts[-1], best_bi)
                if gi_s > gi_e:
                    gi_s, gi_e = gi_e, gi_s
                # Clamp to branch bounds
                gi_s = max(gi_s, _bs)
                gi_e = min(gi_e, _be - 1)

                seg_diams = [diams[i] for i in range(gi_s, gi_e + 1) if diams[i] > 0.5]

                if seg_diams:
                    n = len(seg_diams)
                    window = max(1, n // 10)

                    # ── Proximal window — clamp to stableStartGi ─────────────
                    # gi_s may sit inside the IVC/trunk junction zone (the first
                    # 10–20 pts of a renal branch are still geometry-contaminated
                    # by the parent vessel wall).  stableStartGi (computed by
                    # _detectFindings flare-walk) is the first post-flare point.
                    # Using raw gi_s for prox gives prox=18.7mm on Branch 3
                    # even after smoothing, because all pts in the first 10%
                    # are inside the contamination zone.
                    #
                    # Fix: advance gi_s_prox to stableStartGi for the prox
                    # window only; the full seg_diams list is unchanged (mean /
                    # min / max still cover the whole segment).
                    #
                    # Additionally apply a post-stable offset (PROX_OFFSET_PTS)
                    # so the window starts past any residual non-cylindrical
                    # geometry at the junction — even after the flare zone the
                    # first few pts can still be elliptical / oblique, biasing
                    # the mean high (~15mm instead of ~10mm on 74mm renal).
                    # Use trimmed-minimum (lower 50% median) instead of mean so
                    # the estimator locks onto the true lumen wall rather than
                    # being pulled up by inflated outliers at the window edge.
                    # ─────────────────────────────────────────────────────────
                    gi_s_prox = gi_s
                    _bm = getattr(self, "branchMeta", {})
                    # Find which branch bi matches this segment by gi range
                    for _tbi, (_tbs, _tbe) in enumerate(self.branches):
                        if _tbs <= gi_s <= _tbe:
                            _sgi = _bm.get(_tbi, {}).get("stableStartGi")
                            if _sgi is not None and _sgi > gi_s and _sgi <= gi_e:
                                gi_s_prox = _sgi
                            break
                    # Post-stable offset: skip a further ~5mm of residual
                    # junction geometry past stableStartGi.  Cap at 20% of
                    # the remaining segment so short branches are not consumed.
                    PROX_POST_OFFSET_MM = 5.0
                    if gi_s_prox > gi_s and gi_s_prox < gi_e:
                        _seg_arc = (
                            self.distances[gi_e] - self.distances[gi_s_prox]
                            if gi_e < len(self.distances)
                            and gi_s_prox < len(self.distances)
                            else 0.0
                        )
                        _max_skip = max(0, int((gi_e - gi_s_prox) * 0.20))
                        _off_arc = 0.0
                        _off_gi = gi_s_prox
                        for _oi in range(
                            gi_s_prox, min(gi_e - 1, gi_s_prox + _max_skip + 10)
                        ):
                            if _oi >= len(self.distances) - 1:
                                break
                            _off_arc += self.distances[_oi + 1] - self.distances[_oi]
                            if _off_arc >= PROX_POST_OFFSET_MM:
                                _off_gi = _oi + 1
                                break
                        if _off_gi > gi_s_prox:
                            gi_s_prox = _off_gi
                    # Rebuild prox_diams from the advanced start
                    prox_diams = [
                        diams[i] for i in range(gi_s_prox, gi_e + 1) if diams[i] > 0.5
                    ]
                    prox_window = max(1, len(prox_diams) // 10)
                    # Trimmed-minimum: median of lower 50% of window samples.
                    # More robust than mean at the proximal end where residual
                    # wall-blending still inflates occasional samples.
                    _pw = sorted(prox_diams[:prox_window])
                    _pw_half = max(1, len(_pw) // 2)
                    _pw_lo = _pw[:_pw_half]
                    prox = _pw_lo[len(_pw_lo) // 2] if _pw_lo else 0.0
                    dist = _trimmed_mean(seg_diams[n - window :])

                    seg["diam_mean"] = round(_trimmed_mean(seg_diams), 2)
                    seg["diam_min"] = round(min(seg_diams), 2)
                    seg["diam_max"] = round(max(seg_diams), 2)
                    seg["diam_prox"] = round(prox, 2)
                    seg["diam_dist"] = round(dist, 2)
                    seg["taper"] = round(prox - dist, 2)
                    seg["gi_start"] = gi_s
                    seg["gi_end"] = gi_e
                    seg["n_pts"] = gi_e - gi_s + 1

                    # ── Sanity guard: prox inflation check ───────────────
                    # Surface-distance method can still over-estimate prox
                    # by 15–30% even after flare correction.  Cap prox at
                    # PROX_SANITY_K × dist (clinical bound: no vessel widens
                    # >40% proximally in a single short segment).  Flag if
                    # clamped so the log stays auditable.
                    PROX_SANITY_K = 1.4
                    if dist > 0 and prox > dist * PROX_SANITY_K:
                        prox_clamped = round(dist * PROX_SANITY_K, 2)
                        print(
                            f"[TreeDiam] Branch {best_bi} prox inflation: "
                            f"{seg['diam_prox']:.1f}mm → clamped to "
                            f"{prox_clamped:.1f}mm ({PROX_SANITY_K}× dist={dist:.1f}mm)"
                        )
                        seg["diam_prox"] = prox_clamped
                        seg["taper"] = round(prox_clamped - dist, 2)

                    # ── Write sizing fields back to branchMeta ────────────
                    # Allows generateReport / bdata to read proxD/distD
                    # directly from branchMeta without traversing branchTree.
                    _bm_write = getattr(self, "branchMeta", {})
                    if best_bi in _bm_write or True:
                        _bm_write.setdefault(best_bi, {})
                        _bm_write[best_bi]["proxD"] = seg["diam_prox"]
                        _bm_write[best_bi]["distD"] = seg["diam_dist"]
                else:
                    seg["diam_mean"] = seg["diam_min"] = seg["diam_max"] = 0.0
                    seg["diam_prox"] = seg["diam_dist"] = seg["taper"] = 0.0
                    seg["gi_start"] = seg["gi_end"] = 0
                    seg["n_pts"] = 0

            for ch in node.get("children", []):
                _annotate_seg(ch)

        _annotate_seg(self.branchTree)

        # Log segment diameter summary
        def _log_seg_diams(node, prefix=""):
            seg = node.get("segment")
            if seg and seg.get("diam_mean", 0) > 0:
                role = seg.get("role", "?")
                taper = seg.get("taper", 0.0)
                t_tag = f" taper={taper:+.1f}mm" if abs(taper) >= 0.5 else ""
                min_tag = ""
                if seg["diam_min"] < seg["diam_mean"] * 0.75:
                    # Locate the minimum within seg_diams to distinguish
                    # distal-tip mesh artifact from mid-branch stenosis.
                    _gi_s = seg.get("gi_start", 0)
                    _gi_e = seg.get("gi_end", _gi_s)
                    _seg_d = [
                        self.diameters[i]
                        for i in range(_gi_s, _gi_e + 1)
                        if i < len(self.diameters) and self.diameters[i] > 0.5
                    ]
                    if _seg_d:
                        _min_val = min(_seg_d)
                        _min_idx = _seg_d.index(_min_val)
                        _pct = round(100 * _min_idx / max(1, len(_seg_d) - 1))
                        _loc = (
                            "distal-tip"
                            if _pct >= 80
                            else "proximal" if _pct <= 20 else f"{_pct}% along"
                        )

                        # Stenosis confidence metrics
                        # % drop vs proximal reference (prox = post-flare anchor)
                        _ref = (
                            seg["diam_prox"]
                            if seg["diam_prox"] > 0
                            else seg["diam_mean"]
                        )
                        _drop_pct = (
                            round(100 * (1.0 - _min_val / _ref)) if _ref > 0 else 0
                        )

                        # Longest contiguous run below 80% of reference.
                        # Previously used total-count (non-contiguous) which inflated
                        # run_mm — a 21mm "run" was actually scattered pts, not a
                        # solid 21mm narrowing.  Now find the longest unbroken streak.
                        _thresh = _ref * 0.80
                        _mm_per_pt = seg["arc_mm"] / max(1, len(_seg_d) - 1)
                        _best_run_len = 0
                        _best_run_start_idx = _min_idx  # default: worst-pt only
                        _best_run_end_idx = _min_idx
                        _cur_run_len = 0
                        _cur_run_start = 0
                        for _ri, _rd in enumerate(_seg_d):
                            if _rd < _thresh:
                                if _cur_run_len == 0:
                                    _cur_run_start = _ri
                                _cur_run_len += 1
                                if _cur_run_len > _best_run_len:
                                    _best_run_len = _cur_run_len
                                    _best_run_start_idx = _cur_run_start
                                    _best_run_end_idx = _ri
                            else:
                                _cur_run_len = 0
                        _run_mm = round(max(1, _best_run_len) * _mm_per_pt, 1)

                        _conf = (
                            "HIGH"
                            if _drop_pct >= 30 and _run_mm >= 5.0 and _pct < 80
                            else "MED" if _drop_pct >= 20 and _pct < 80 else "LOW"
                        )

                        # Narrowing phenotype: helps distinguish segmentation
                        # artifact from genuine compression.
                        # Artifact-like: sharp drop (>35%) but short run (<25mm)
                        #   → single bad ray cluster or mesh pinch
                        # Compression-like: large drop + long run (≥25mm)
                        #   → diffuse extrinsic narrowing (e.g. May-Thurner)
                        # Focal: moderate drop, short-medium run
                        _branch_arc = seg.get("arc_mm", 1.0)
                        _run_fraction = _run_mm / max(_branch_arc, 1.0)
                        if _drop_pct >= 35 and _run_mm < 25.0:
                            _phenotype = "artifact-like"
                        elif _drop_pct >= 30 and _run_fraction >= 0.25:
                            _phenotype = "compression-like"
                        elif _drop_pct >= 20:
                            _phenotype = "focal"
                        else:
                            _phenotype = "mild"

                        min_tag = (
                            f" ⚠MIN={_min_val:.1f}mm@{_loc}"
                            f" drop={_drop_pct}% run={_run_mm}mm [{_conf}/{_phenotype}]"
                        )

                        # Write stenosis candidate to branchMeta for _detectFindings
                        # Only for non-distal-tip findings with ≥MED confidence
                        if _conf in ("HIGH", "MED") and _pct < 80:
                            _bm_node = getattr(self, "branchMeta", {})
                            for _sbi, (_sbs, _sbe) in enumerate(self.branches):
                                if _sbs <= _gi_s <= _sbe:
                                    _bm_node.setdefault(_sbi, {})
                                    _bm_node[_sbi]["stenosisCandidate"] = {
                                        "min_diam": round(_min_val, 2),
                                        "ref_diam": round(_ref, 2),
                                        "drop_pct": _drop_pct,
                                        "run_mm": _run_mm,
                                        "loc_pct": _pct,
                                        "confidence": _conf,
                                        "phenotype": _phenotype,
                                        "gi_min": _gi_s + _min_idx,
                                        "gi_run_start": _gi_s + _best_run_start_idx,
                                        "gi_run_end": _gi_s + _best_run_end_idx,
                                    }
                                    break
                    else:
                        _loc = "unknown"
                        min_tag = f" ⚠MIN={seg['diam_min']:.1f}mm@{_loc}"
                print(
                    f"[TreeDiam]   {prefix}node {node['node_id']} "
                    f"[{role}] {seg['arc_mm']:.1f}mm: "
                    f"mean={seg['diam_mean']:.1f} "
                    f"prox={seg['diam_prox']:.1f} "
                    f"dist={seg['diam_dist']:.1f}"
                    f"{t_tag}{min_tag}"
                )
            for ch in node.get("children", []):
                _log_seg_diams(ch, prefix + "  ")

        print("[TreeDiam] Segment diameter annotation complete:")
        _log_seg_diams(self.branchTree)

    # """
    # VesselAnalyzer — Pancaking / Ellipse-Axis Diameter Fix
    # =======================================================
    # Build tag: DiameterEllipseFix

    # ROOT CAUSE
    # ----------
    # _computeDiameters averaged all 8 radial ray hits into a single scalar per
    # centerline point.  For a pancaked (elliptical) cross-section the mean of the
    # long-axis and short-axis rays produces a value close to the geometric mean of
    # the two axes — appearing "normal" in the IVUS slider even when the vessel is
    # severely compressed.

    # Additionally, the OUTLIER_K=2.0 gate (designed to reject trunk-wall bleed for
    # renal branches) could incorrectly reject the long-axis rays of a flattened IVC
    # lumen (e.g. at a May-Thurner compression site), further masking the pathology.

    # FIX
    # ---
    # After outlier filtering, store THREE values per point instead of one:
    # self.diameters[i]      — mean diameter (unchanged, existing pipeline unaffected)
    # self._diam_minor[i]    — 2 × r_min from clean rays  (short / compressed axis)
    # self._diam_major[i]    — 2 × r_max from clean rays  (long / patent axis)

    # _computeEllipseAxes is rewritten to simply return these parallel arrays instead
    # of the approximate sliding-window proxy.  The pancake_ratio at each point is
    # now diam_major / diam_minor, computed directly from the ray geometry.

    # INTEGRATION NOTES
    # -----------------
    # • self.diameters is UNCHANGED in length and values — all downstream code
    # (navigator, report, stent sizing, _annotateTreeDiameters) is unaffected.
    # • self._diam_minor / self._diam_major are initialised to [] in __init__ and
    # populated by _computeDiameters.  They are guaranteed to be the same length
    # as self.diameters after _computeDiameters returns.
    # • _computeEllipseAxes now just returns these arrays.  Its call site in
    # _detectFindings is unchanged.
    # • The OUTLIER_K gate is KEPT for trunk-wall bleed rejection but is now applied
    # BEFORE computing minor/major, so it only affects contaminated rays.

    # SEARCH & REPLACE INSTRUCTIONS
    # ------------------------------
    # In VesselAnalyzer.py:

    # 1. In __init__ (wherever self.diameters = [] is initialised), add two lines:
    # self._diam_minor = []
    # self._diam_major = []

    # 2. Replace the entire _computeDiameters method body (lines 16867–16997)
    # with the code in REPLACEMENT_computeDiameters below.

    # 3. Replace the entire _computeEllipseAxes method body (lines 16999–17058)
    # with the code in REPLACEMENT_computeEllipseAxes below.

    # No other changes required.
    # """

    # ═══════════════════════════════════════════════════════════════════════════
    # REPLACEMENT: _computeDiameters   (replaces lines 16867–16997)
    # ═══════════════════════════════════════════════════════════════════════════


    def _computeDiameters(self):
        """Compute vessel diameter at each centerline point via multi-ray casting.

        For each point we cast N_RAYS rays radially outward in the plane
        perpendicular to the local centerline tangent, intersect them with the
        vessel surface mesh, and compute:

          self.diameters[i]   — mean diameter (mm)  — used by all existing pipeline
          self._diam_minor[i] — minor axis (mm)     — 2 × shortest clean ray hit
          self._diam_major[i] — major axis (mm)     — 2 × longest  clean ray hit

        Storing min/max axes separately lets _computeEllipseAxes report the TRUE
        compression ratio for pancaked lumens instead of the approximate
        sliding-window proxy, which could not distinguish a uniformly narrow vessel
        from a focally compressed one.

        DiameterEllipseFix: parallel _diam_minor / _diam_major arrays added.
        """
        polydata = self.modelNode.GetPolyData()
        if not polydata:
            self._diam_minor = []
            self._diam_major = []
            return [0.0] * len(self.points)

        # Use OBBTree for fast ray-mesh intersection
        obbTree = vtk.vtkOBBTree()
        obbTree.SetDataSet(polydata)
        obbTree.BuildLocator()
        obbTree.SetTolerance(0.001)

        N_RAYS = 8  # rays per point (evenly spread 360°)
        RAY_LEN = 60.0  # mm — long enough to exit any iliac vessel
        N_PTS = len(self.points)
        diameters = []
        diam_minor = []  # NEW: short axis per point
        diam_major = []  # NEW: long  axis per point

        # ── Local tangent (finite differences) ───────────────────────────────
        def _tangent(i):
            if i == 0:
                p0, p1 = self.points[0], self.points[1]
            elif i == N_PTS - 1:
                p0, p1 = self.points[-2], self.points[-1]
            else:
                p0, p1 = self.points[i - 1], self.points[i + 1]
            d = [p1[k] - p0[k] for k in range(3)]
            ln = math.sqrt(sum(x * x for x in d))
            return [x / ln for x in d] if ln > 1e-6 else [0.0, 0.0, 1.0]

        # ── Stable perpendicular basis ────────────────────────────────────────
        def _perp_basis(t):
            helper = [0.0, 0.0, 1.0] if abs(t[2]) < 0.9 else [1.0, 0.0, 0.0]
            e1 = [
                t[1] * helper[2] - t[2] * helper[1],
                t[2] * helper[0] - t[0] * helper[2],
                t[0] * helper[1] - t[1] * helper[0],
            ]
            l1 = math.sqrt(sum(x * x for x in e1))
            if l1 < 1e-6:
                return [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
            e1 = [x / l1 for x in e1]
            e2 = [
                t[1] * e1[2] - t[2] * e1[1],
                t[2] * e1[0] - t[0] * e1[2],
                t[0] * e1[1] - t[1] * e1[0],
            ]
            l2 = math.sqrt(sum(x * x for x in e2))
            e2 = [x / l2 for x in e2] if l2 > 1e-6 else [0.0, 1.0, 0.0]
            return e1, e2

        # ── Fallback: closest-face distance ──────────────────────────────────
        _fb_loc = vtk.vtkCellLocator()
        _fb_loc.SetDataSet(polydata)
        _fb_loc.BuildLocator()

        def _fallback_d(pt):
            cp = [0.0, 0.0, 0.0]
            ci = vtk.reference(0)
            si = vtk.reference(0)
            d2 = vtk.reference(0.0)
            _fb_loc.FindClosestPoint(list(pt), cp, ci, si, d2)
            return math.sqrt(float(d2)) * 2.0

        pts_intersect = vtk.vtkPoints()
        cell_ids = vtk.vtkIdList()

        for i, pt in enumerate(self.points):
            tang = _tangent(i)
            e1, e2 = _perp_basis(tang)
            radii_hits = []

            for r in range(N_RAYS):
                angle = 2.0 * math.pi * r / N_RAYS
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                ray_dir = [e1[k] * cos_a + e2[k] * sin_a for k in range(3)]
                p_start = [pt[k] + ray_dir[k] * 0.1 for k in range(3)]
                p_end = [pt[k] + ray_dir[k] * RAY_LEN for k in range(3)]

                pts_intersect.Reset()
                ret = obbTree.IntersectWithLine(p_start, p_end, pts_intersect, cell_ids)
                if ret != 0 and pts_intersect.GetNumberOfPoints() > 0:
                    best_d = 1e18
                    for j in range(pts_intersect.GetNumberOfPoints()):
                        ip = pts_intersect.GetPoint(j)
                        d = math.sqrt(sum((ip[k] - pt[k]) ** 2 for k in range(3)))
                        if d < best_d:
                            best_d = d
                    if best_d < RAY_LEN:
                        radii_hits.append(best_d)

            if len(radii_hits) >= 2:
                # ── Outlier-ray rejection (trunk-wall bleed at branch junctions) ──
                # r_min = closest wall = always the true branch wall.
                # Reject rays > OUTLIER_K × r_min (trunk-wall hits are 3-5× r_min).
                # OUTLIER_K=2.0 tolerates ±100% eccentricity — a 2:1 minor:major
                # ratio covers severe May-Thurner compression without rejecting the
                # long-axis rays.  Only rays ≥3× the minimum are discarded.
                #
                # DiameterEllipseFix: after outlier filtering, record min and max
                # of the CLEAN ray set separately so _computeEllipseAxes can return
                # the true compression ratio rather than the sliding-window proxy.
                # ─────────────────────────────────────────────────────────────────
                OUTLIER_K = 2.0
                r_min = min(radii_hits)
                clean = [r for r in radii_hits if r <= OUTLIER_K * r_min]
                if len(clean) < 2:
                    # Very few hits survived — use full set (better than 1 point)
                    clean = radii_hits

                # Mean diameter (unchanged semantics — all downstream consumers OK)
                r_mean = sum(clean) / len(clean)
                # Ellipse axes — directly from ray geometry
                r_minor = min(clean)  # short axis (compressed direction)
                r_major = max(clean)  # long  axis (patent  direction)

                diameters.append(round(2.0 * r_mean, 3))
                diam_minor.append(round(2.0 * r_minor, 3))
                diam_major.append(round(2.0 * r_major, 3))

            elif radii_hits:
                d_val = 2.0 * radii_hits[0]
                diameters.append(round(d_val, 3))
                diam_minor.append(round(d_val, 3))
                diam_major.append(round(d_val, 3))

            else:
                # No ray hits — closest-face fallback
                d_val = _fallback_d(pt)
                diameters.append(round(d_val, 3))
                diam_minor.append(round(d_val, 3))
                diam_major.append(round(d_val, 3))

        # Persist parallel arrays on self so _computeEllipseAxes can use them
        self._diam_minor = diam_minor
        self._diam_major = diam_major

        return diameters

    # ═══════════════════════════════════════════════════════════════════════════
    # REPLACEMENT: _computeEllipseAxes  (replaces lines 16999–17058)
    # ═══════════════════════════════════════════════════════════════════════════


    def _computeEllipseAxes(self):
        """Return per-point ellipse axes and compression (pancake) ratio.

        DiameterEllipseFix: now uses self._diam_minor / self._diam_major
        populated by _computeDiameters from the actual ray-cast geometry, rather
        than the previous sliding-window local-max proxy.

        Advantages over the old approach:
          • Directly measures the cross-section shape at each individual point —
            no neighbourhood contamination from adjacent trunk or branch geometry.
          • Correctly identifies focal compression (pancaking) even when the
            surrounding vessel is also abnormal (e.g. diffuse May-Thurner where
            the "neighbourhood max" was itself compressed, making ratio ≈ 1.0
            everywhere and masking the pathology).
          • The ratio is derived from the same ray pass that computes self.diameters,
            so there is no additional computational cost.

        Returns (minor_axes, major_axes, pancake_ratios) parallel to self.points.
          minor_axes[i]     — short diameter (mm) — compressed dimension
          major_axes[i]     — long  diameter (mm) — patent    dimension
          pancake_ratios[i] — major / minor        — 1.0 = circular, >1.5 = pancaked
        """
        n = len(self.points)

        # Fast path: use ray-derived axes if available (populated by
        # _computeDiameters in the same analysis run)
        if (
            hasattr(self, "_diam_minor")
            and hasattr(self, "_diam_major")
            and len(self._diam_minor) == n
            and len(self._diam_major) == n
        ):

            minor_axes = list(self._diam_minor)
            major_axes = list(self._diam_major)
            pancake_ratios = []
            for i in range(n):
                mn = minor_axes[i]
                mx = major_axes[i]
                if mn > 0.5:
                    ratio = min(mx / mn, 5.0)  # cap at 5× to suppress endpoint noise
                else:
                    ratio = 1.0
                pancake_ratios.append(ratio)

            # Log summary of worst compression per branch for auditability
            if hasattr(self, "branches") and self.branches:
                for bi, (bs, be) in enumerate(self.branches):
                    _ratios = [
                        pancake_ratios[gi]
                        for gi in range(bs, min(be, n))
                        if minor_axes[gi] > 0.5
                    ]
                    if _ratios:
                        worst = max(_ratios)
                        worst_gi = bs + _ratios.index(worst)
                        if worst >= 1.3:  # only log if at least mildly elliptical
                            label = (
                                "PANCAKING"
                                if worst >= 2.0
                                else (
                                    "Moderate compression"
                                    if worst >= 1.5
                                    else "Mild ellipticity"
                                )
                            )
                            mn_w = minor_axes[worst_gi]
                            mx_w = major_axes[worst_gi]
                            print(
                                f"[EllipseAxes] Branch {bi}: worst ratio={worst:.2f} "
                                f"@ gi={worst_gi}  minor={mn_w:.1f}mm  "
                                f"major={mx_w:.1f}mm  [{label}]"
                            )

            return minor_axes, major_axes, pancake_ratios

        # ── Fallback: sliding-window proxy (used if _computeDiameters has not ──
        # yet run in this session, e.g. VMTK-radius mode or reanalysis path   ──
        # where _diam_minor was not populated).                                ──
        # This is the original algorithm, preserved unchanged as a safety net.
        import statistics

        diams = self.diameters if self.diameters else [0.0] * n
        minor_axes = list(diams)
        major_axes = list(diams)
        pancake_ratios = [1.0] * n

        WIN = 20  # window half-width in points (~20mm at typical spacing)

        _gi_branch_lo = [0] * n
        _gi_branch_hi = [n] * n
        if hasattr(self, "branches") and self.branches:
            for bi, (bs, be) in enumerate(self.branches):
                for gi in range(max(0, bs), min(n, be)):
                    _gi_branch_lo[gi] = bs
                    _gi_branch_hi[gi] = be

        for i in range(n):
            d = diams[i]
            if d < 0.5:
                continue
            b_lo = _gi_branch_lo[i]
            b_hi = _gi_branch_hi[i]
            lo = max(b_lo, i - WIN)
            hi = min(b_hi, i + WIN)
            neighbours = [diams[j] for j in range(lo, hi) if diams[j] > 0.5]
            if len(neighbours) < 6:
                continue
            local_max = max(neighbours)
            ratio = min(local_max / d, 5.0) if d > 0 else 1.0
            pancake_ratios[i] = ratio
            minor_axes[i] = d
            major_axes[i] = local_max

        print(
            "[EllipseAxes] WARNING: using sliding-window fallback "
            "(ray-derived axes not available — run full analysis to activate "
            "DiameterEllipseFix)"
        )
        return minor_axes, major_axes, pancake_ratios

        # ═══════════════════════════════════════════════════════════════════════════
        # __init__ ADDITION — add these two lines wherever self.diameters = [] is set
        # ═══════════════════════════════════════════════════════════════════════════
        #
        #     self._diam_minor = []   # DiameterEllipseFix: short axis per centerline pt
        #     self._diam_major = []   # DiameterEllipseFix: long  axis per centerline pt
        #
        # ═══════════════════════════════════════════════════════════════════════════

        # ═══════════════════════════════════════════════════════════════════════════
        # OPTIONAL — surface color enhancement for pancaking
        # ═══════════════════════════════════════════════════════════════════════════
        #
        # After the fix, _detectFindings will correctly identify pancaking because
        # pancake_ratios will now be >> 1.0 at compression sites.
        # The existing EllipseGateRemoved logic (v257) already removed the ratio>=1.1
        # gate from the stenosis scanner, so compression findings will fire on the
        # mean diameter drop alone.
        #
        # If you want BOTH signals (mean drop AND ellipticity) to appear in the report,
        # update the phenotype classifier in _annotateTreeDiameters (_log_seg_diams):
        #
        #   Add a per-branch "max_ellipticity" field using diam_major/diam_minor, and
        #   if max_ellipticity >= 1.5 over the compression run, label it:
        #       "compression-like (pancaking, ratio=X.X)"
        #   instead of just "compression-like".
        #
        # This is cosmetic only — the detection and diameter values are correct after
        # the three-line fix above.
        # ═══════════════════════════════════════════════════════════════════════════


    def _detectFindings(self):
        """Detect vessel findings using consecutive-point run classification.

        Classification rules (per clinical protocol):
          Stenosis / compression (narrowing):
            1 pt only            → focal abnormality (ignore — artifact threshold)
            2–3 adjacent pts     → possible short lesion  → Mild Compression
            ≥ 3–5 consecutive    → definite lesion        → Pancaking (if ellipse ratio ≥ 1.5)
                                                            or Mild Compression
          Dilation (ectasia / aneurysm):
            Ectasia  : D_measured/D_ref ≥ 1.2  AND  Dmin/Dmax > 0.8  AND ≥ 3 consecutive pts
            Aneurysm : D_measured/D_ref ≥ 1.5  AND ≥ 3 consecutive pts

        Always uses ORIGINAL (pre-balloon) diameters so balloon expansion
        does not create false ectasia findings.
        """
        import statistics

        print("[VesselAnalyzer] Detecting findings (pancaking / aneurysm)...")

        # Use original diameters if balloon has modified them
        _orig = getattr(self, "_origDiameters", {})
        diams_to_use = list(self.diameters)
        for idx, orig_d in _orig.items():
            if idx < len(diams_to_use):
                diams_to_use[idx] = orig_d

        # Ellipse analysis for pancake (cross-section flattening) ratio
        try:
            self.minor_axes, self.major_axes, self.pancake_ratios = (
                self._computeEllipseAxes()
            )
        except Exception as ex:
            print(
                f"[VesselAnalyzer] Ellipse axes failed ({ex}), using diameter-only detection"
            )
            n = len(self.points)
            self.minor_axes = list(self.diameters)
            self.major_axes = list(self.diameters)
            self.pancake_ratios = [1.0] * n

        self.finding_type = [0] * len(self.points)
        self.findings = []

        # ── Vessel-type context ───────────────────────────────────────────────
        _venous = getattr(self, "vesselType", "arterial") == "venous"
        # Venous: raise dilation thresholds (bifurcation expansion is normal)
        _ANEURYSM_RATIO = 2.0 if _venous else 1.5  # arterial 1.5x / venous 2.0x
        _ECTASIA_RATIO = 1.5 if _venous else 1.2  # arterial 1.2x / venous 1.5x
        # Bifurcation dome: suppress ANEURYSM-grade smooth dilation inside this radius.
        # Reduced 25mm→10mm in venous mode: IVC/iliac confluence dome is much smaller
        # than an aortic bifurcation; 25mm was suppressing real ectasia at the carina.
        # Ectasia is NEVER suppressed by the dome gate — it is a clinically meaningful
        # finding at the venous confluence (post-thrombotic remodeling, chronic HTN).
        _BIF_DOME_MM = 10.0 if _venous else 0.0
        _bif_pt = getattr(self, "bifurcationPoint", None)
        _vtype_str = "venous" if _venous else "arterial"
        print(
            f"[VesselAnalyzer] Finding mode: {_vtype_str} "
            f"(aneurysm>={_ANEURYSM_RATIO:.2f}x, ectasia>={_ECTASIA_RATIO:.2f}x, "
            f"bif_dome={_BIF_DOME_MM:.0f}mm)"
        )
        # ─────────────────────────────────────────────────────────────────────

        trunk_end_gi = self.branches[0][1] - 1 if self.branches else -1
        trunk_end_z = (
            self.points[trunk_end_gi][2]
            if 0 <= trunk_end_gi < len(self.points)
            else float("-inf")
        )
        # trunk_root_z: proximal aorta inlet — used by per-branch skip gate below
        trunk_root_gi = self.branches[0][0] if self.branches else 0
        trunk_root_z = (
            self.points[trunk_root_gi][2]
            if trunk_root_gi < len(self.points)
            else trunk_end_z
        )

        # Healthy reference from the distal trunk (just above bifurcation)
        trunk_bs0, trunk_be0 = self.branches[0]
        _trk_ref_s = max(trunk_bs0, trunk_be0 - 30)
        _trk_ref_d = [
            diams_to_use[i]
            for i in range(_trk_ref_s, trunk_be0)
            if i < len(diams_to_use) and diams_to_use[i] > 1.0
        ]
        trunk_distal_ref = (
            statistics.median(_trk_ref_d) if len(_trk_ref_d) >= 3 else 0.0
        )

        # ── Run-detection helper ──────────────────────────────────────────────
        def _collect_runs(flags, min_run):
            """Given a list of (gi, True/False), return list of runs of True
            with length >= min_run.  Each run: {'start':gi, 'end':gi, 'indices':[gi,...]}
            """
            runs, cur = [], []
            for gi, flag in flags:
                if flag:
                    cur.append(gi)
                else:
                    if len(cur) >= min_run:
                        runs.append(
                            {"start": cur[0], "end": cur[-1], "indices": list(cur)}
                        )
                    cur = []
            if len(cur) >= min_run:
                runs.append({"start": cur[0], "end": cur[-1], "indices": list(cur)})
            return runs

        MIN_LESION_MM = 5.0  # minimum physical arc length for a run to be reported

        def _run_arc_mm(run):
            """Compute arc length of a run from self.points positions."""
            idxs = run["indices"]
            arc = 0.0
            for _k in range(len(idxs) - 1):
                a, b = idxs[_k], idxs[_k + 1]
                if a < len(self.points) and b < len(self.points):
                    arc += (
                        sum(
                            (self.points[b][d] - self.points[a][d]) ** 2
                            for d in range(3)
                        )
                        ** 0.5
                    )
            return arc

        def _filter_by_length(runs, label, bi):
            """Drop runs shorter than MIN_LESION_MM; log dropped ones."""
            kept = []
            for r in runs:
                arc = _run_arc_mm(r)
                r["arc_mm"] = round(arc, 1)
                if arc >= MIN_LESION_MM:
                    kept.append(r)
                else:
                    print(
                        f"[LengthGate] Branch {bi} {label} run "
                        f"gi{r['start']}–{r['end']} ({len(r['indices'])} pts, "
                        f"{arc:.1f}mm < {MIN_LESION_MM}mm) — too short, dropped"
                    )
            return kept

        def _worst_pt(indices, key_fn):
            """Return the index with the most extreme value of key_fn(gi)."""
            return min(indices, key=key_fn)

        for bi, (s, e) in enumerate(self.branches):
            branch_len = e - s
            if bi == 0:
                continue
            print(
                f"[FindDiag] bi={bi} name='{self.getBranchDisplayName(bi)}' "
                f"gi=[{s},{e-1}] pts={branch_len} "
                f"role={self.branchMeta.get(bi,{}).get('role','?')} "
                f"ogi={self.branchMeta.get(bi,{}).get('ostiumGi','?')} "
                f"sgi={self.branchMeta.get(bi,{}).get('stableStartGi','?')}"
            )
            # Gate: skip branches whose START point is above the trunk root.
            # Original guard used trunk_end_z (bifurcation end = distal aorta),
            # which incorrectly skipped renal branches — they originate on the
            # trunk wall ABOVE the bifurcation, so branch_start_z > trunk_end_z
            # is always true for renals.  Fix: use trunk_root_z (proximal aorta)
            # as the ceiling — only skip branches that start above the aortic
            # inlet, which are true artifacts, not anatomical side branches.
            #
            # IliacZGateFix: iliac branches are EXEMPT from this Z-gate entirely.
            # Root cause: after BIF_DOME_MM / IliacOstiumMaxZ, ostiumGi is set to the
            # cranial apex of the iliac dome (the max-Z point on the branch CL).
            # self.branches[bi] start is then TRIMMED to that ostiumGi (line ~12317),
            # so `s` == ostiumGi == the most cranial CL point on the iliac.  For the
            # Left Iliac this apex can equal or exceed trunk_root_z, causing the
            # gate to fire and silently skip the entire Left Iliac in lesion detection.
            # Confirmed iliac roles (iliac_left / iliac_right / main) are anatomically
            # anchored to the primary bifurcation — they are never artifacts — so we
            # bypass the Z-gate for them.  For other branches we use the RAW branch
            # start from _rawBranches (pre-trim) rather than the post-trim `s`, so
            # renal / side branches whose ostium trimming also shifts `s` cranially
            # are evaluated at their true graph-topology start Z, not their ostium Z.
            _bm_role = self.branchMeta.get(bi, {}).get("role", "")
            _is_confirmed_iliac = _bm_role in ("main", "iliac_left", "iliac_right")
            if not _is_confirmed_iliac:
                _raw_branches = getattr(self, "_rawBranches", self.branches)
                _raw_s = _raw_branches[bi][0] if bi < len(_raw_branches) else s
                branch_start_z = (
                    self.points[_raw_s][2] if _raw_s < len(self.points) else 0.0
                )
                if branch_start_z > trunk_root_z + 10.0:
                    print(
                        f"[DetectFindings] Branch {bi} ({_bm_role}): skipped — "
                        f"raw_start_z={branch_start_z:.1f} > trunk_root_z={trunk_root_z:.1f}+10"
                    )
                    continue

            # ── Geometry-grounded branch start ────────────────────────────────
            # ostiumGi is the best available index-based estimate of the carina,
            # but for main branches it points to gi=92 which is 16mm past the
            # true carina due to centerline sparsity.  The diameters in the first
            # ~20 pts after ostiumGi are inflated by the bifurcation flare zone.
            #
            # Strategy (Option A — local diameter minimum):
            #   1. Starting from ogi (or raw s if no ogi), walk forward to find
            #      the END of the flare zone: first point where dr/ds < FLARE_THR
            #      for FLARE_PERSIST consecutive points.
            #   2. That end-of-flare point is the geometry-grounded stable_start.
            #   3. If no flare is detected (branch joins cleanly), use ogi as-is.
            #   4. Hard cap: stable_start never exceeds s + MAX_FLARE_PTS.
            #
            # Additionally: diameter sampling uses a ±DIAM_WINDOW_PTS window
            # around each measurement point (trimmed mean) rather than raw point
            # values, making individual noisy points non-fatal.

            FLARE_THR = 0.08  # dr/ds threshold (fraction of mean diam per pt)
            FLARE_PERSIST = 3  # consecutive non-flare pts to confirm end
            DIAM_WINDOW_PTS = 3  # half-window for local diameter smoothing

            # Branch arc length (mm) — used to scale skip limits so short branches
            # (renals, 30–75 mm) are not consumed by limits tuned for 130–175 mm iliacs.
            _branch_arc_mm = (
                self.distances[e - 1] - self.distances[s]
                if e - 1 < len(self.distances) and s < len(self.distances)
                else (e - s) * 1.35
            )  # fallback: ~1.35 mm/pt

            # MAX_FLARE_PTS: cap at whichever is smaller — 20 pts or 25% of branch pts.
            MAX_FLARE_PTS = min(20, max(5, int((e - s) * 0.25)))

            meta = self.branchMeta.get(bi, {})
            ogi = meta.get("ostiumGi", None)
            raw_start = s
            scan_from = ogi if (ogi is not None and s <= ogi < e) else raw_start

            # Local smoothed diameter at index gi
            def _smooth_diam(gi):
                lo = max(raw_start, gi - DIAM_WINDOW_PTS)
                hi = min(e - 1, gi + DIAM_WINDOW_PTS)
                vals = [
                    diams_to_use[i] for i in range(lo, hi + 1) if diams_to_use[i] > 1.0
                ]
                if not vals:
                    return 0.0
                vals.sort()
                trim = max(1, len(vals) // 4)
                return sum(vals[trim : -trim or None]) / max(1, len(vals) - 2 * trim)

            # Estimate true vessel diameter from the DISTAL THIRD of the branch —
            # far enough from the ostium to be unaffected by the bifurcation flare.
            # Sampling proximal pts (e.g. scan_from+15..+35) gives an inflated
            # reference because those pts are still inside the flare cone, which
            # makes flare_ceil too high and causes flare detection to exit instantly.
            _branch_len = e - scan_from
            # Reference window: distal 55–85% of branch in point-space.
            # Hard floors of 20/25 pts crushed the window on short branches
            # (49-pt renal → ref_lo=scan_from+26, ref_hi=scan_from+41 after
            # clamp to e-5/e → 3-pt sliver → noisy _mean_ref_d → flare_ceil
            # too low → diameter-walk exits at stable_start==scan_from, so
            # stableStartGi is never advanced past the flare zone).
            # Fix: use only fractional windows so both floors scale with branch
            # length; a 5-pt minimum window is enforced by the fallback below.
            ref_lo = scan_from + int(_branch_len * 0.55)
            ref_hi = scan_from + int(_branch_len * 0.85)
            ref_lo = min(ref_lo, e - 5)
            ref_hi = min(ref_hi, e)
            if ref_hi - ref_lo < 5:  # very short branch — use distal third
                ref_lo = max(scan_from, e - max(10, _branch_len // 3))
                ref_hi = e
            _ref_vals = [
                diams_to_use[i] for i in range(ref_lo, ref_hi) if diams_to_use[i] > 1.0
            ]
            if _ref_vals:
                _ref_vals.sort()
                # Trimmed mean: discard top/bottom 10% to reject noise spikes
                _trim = max(1, len(_ref_vals) // 10)
                _mean_ref_d = sum(_ref_vals[_trim : -_trim or None]) / max(
                    1, len(_ref_vals) - 2 * _trim
                )
            else:
                _mean_ref_d = max(
                    (
                        diams_to_use[i]
                        for i in range(scan_from, e)
                        if diams_to_use[i] > 1.0
                    ),
                    default=10.0,
                )

            # Determine stable_start: the first point past the bifurcation flare.
            #
            # Strategy — two-pass:
            #   Pass 1 (diameter-walk): walk forward from scan_from until
            #     d_curr <= flare_ceil for FLARE_PERSIST consecutive pts.
            #     Cap at MAX_FLARE_PTS to bound the search.
            #   Pass 2 (hard floor): stable_start is at least FLARE_HARD_MM
            #     arc-distance from scan_from, regardless of pass 1 outcome.
            #     This handles cases where the vessel is large and the flare
            #     diameter exceeds flare_ceil for many more pts than MAX_FLARE_PTS.
            #

            stable_start = scan_from  # default
            if _mean_ref_d > 1.0:
                flare_ceil = _mean_ref_d * (1.0 + FLARE_THR)
                consec_stable = 0
                for _fi in range(scan_from, min(scan_from + MAX_FLARE_PTS, e - 1)):
                    d_curr = _smooth_diam(_fi)
                    if d_curr <= 0:
                        continue
                    if d_curr <= flare_ceil:
                        consec_stable += 1
                        if consec_stable >= FLARE_PERSIST:
                            stable_start = max(scan_from, _fi - FLARE_PERSIST + 1)
                            break
                    else:
                        consec_stable = 0

            # FLARE_HARD_MM: minimum arc to skip from ostium before scanning.
            # Original fixed value of 45 mm was correct for iliacs (130–175 mm)
            # but catastrophic for renals (30–75 mm) — it consumed the entire branch,
            # causing stable_start >= stable_end → continue, so stableStartGi was
            # never written and getBranchStats fell back to ostiumGi (+1 pt), leaving
            # the full proximal bleed zone in the mean calculation.
            #
            # Fix: cap at 30% of branch arc, with a 8 mm floor (minimum meaningful
            # skip past the ostium) and the original 45 mm ceiling (iliacs unchanged).
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

            # Store for debug visualization and downstream consumers.
            #
            # Two-path commit (v159):
            #   Path A — flare detected: stable_start advanced ≥2 pts beyond
            #     scan_from by either the diameter-walk or the hard-floor walk.
            #     Commit as authoritative flare end.
            #   Path B — no advance (clean join or flat diameter profile):
            #     Force a hard 10mm skip from scan_from so prox sampling never
            #     starts inside the ostium/junction zone regardless of flare shape.
            #     This is the fallback guarantee — without it, stableStartGi stays
            #     at ostiumGi and _annotateTreeDiameters pulls contaminated prox values
            #     (18.7mm instead of ~9mm on the 74mm renal).
            #
            # Threshold ≥2 (not >0) avoids committing trivial 1-pt advances that
            # are within centerline quantization noise.
            FLARE_COMMIT_MIN_PTS = 2  # minimum meaningful advance
            FLARE_FALLBACK_MM = 10.0  # hard skip when no flare detected

            _advanced = stable_start >= scan_from + FLARE_COMMIT_MIN_PTS

            if _advanced:
                self.branchMeta[bi]["flareEndGi"] = stable_start
                self.branchMeta[bi]["stableStartGi"] = stable_start
                print(
                    f"[VesselAnalyzer] Branch {bi} flare end: "
                    f"gi={stable_start} (+{stable_start - scan_from} pts from ostium, "
                    f"ref_diam={_mean_ref_d:.1f}mm)"
                )
            else:
                # Fallback: walk FLARE_FALLBACK_MM in arc distance from scan_from
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
                    self.branchMeta[bi]["stableStartGi"] = _fb_gi
                    print(
                        f"[VesselAnalyzer] Branch {bi} flare fallback: "
                        f"gi={_fb_gi} (+{_fb_gi - scan_from} pts, "
                        f"{_fb_arc:.1f}mm arc skip, ref_diam={_mean_ref_d:.1f}mm)"
                    )
            stable_end = e - 3
            if stable_start >= stable_end:
                print(
                    f"[FindDiag] bi={bi} SKIP stable_start>= stable_end: "
                    f"stable_start={stable_start} stable_end={stable_end} "
                    f"scan_from={scan_from} e={e} s={s}"
                )
                continue

            healthy_diams = [
                diams_to_use[i]
                for i in range(stable_start, stable_end)
                if diams_to_use[i] > 1.0
            ]
            if len(healthy_diams) < 5:
                print(
                    f"[FindDiag] bi={bi} SKIP healthy_diams<5: "
                    f"got {len(healthy_diams)} pts in [{stable_start},{stable_end}] "
                    f"stable_start={stable_start} stable_end={stable_end}"
                )
                continue

            # Reference diameter for finding detection.
            # Primary: diam_prox from branchMeta (post-flare proximal anchor set by
            # _annotateTreeDiameters) — this is always a pre-narrowing measurement.
            # For branches with diffuse compression spanning the distal window (e.g.
            # 51% narrowing over 58mm), _mean_ref_d sits INSIDE the narrowed zone
            # and self-references the compressed diameter → rel ≈ 1.0 everywhere →
            # no findings fired.  diam_prox avoids this by anchoring to clean tissue
            # just past the ostium flare.
            # Fallback chain: diam_prox → stenosisCandidate ref_diam → _mean_ref_d
            # → trunk-scaled branch median.
            _prox_ref = self.branchMeta.get(bi, {}).get("diam_prox", 0.0) or 0.0
            _sc_ref = (self.branchMeta.get(bi, {}).get("stenosisCandidate") or {}).get(
                "ref_diam", 0.0
            ) or 0.0
            # Use the highest reliable reference (prox anchor > sc_ref > distal window)
            _best_ref = max(_prox_ref, _sc_ref)
            if _best_ref > _mean_ref_d * 1.10:
                # Proximal anchor is materially higher — the distal window is inside
                # the narrowed zone; use the prox anchor as the clinical reference.
                median_d = _best_ref
                print(
                    f"[FindingsRef] Branch {bi}: using prox/sc ref={_best_ref:.1f}mm "
                    f"(distal _mean_ref_d={_mean_ref_d:.1f}mm was too low — "
                    f"likely inside narrowed zone)"
                )
            elif _mean_ref_d > 1.0:
                median_d = _mean_ref_d
            else:
                sorted_h_full = sorted(healthy_diams)
                n_h = len(sorted_h_full)
                branch_upper = statistics.median(
                    sorted_h_full[int(n_h * 0.20) : int(n_h * 0.80)] or sorted_h_full
                )
                if trunk_distal_ref > 0 and branch_upper > 0:
                    scale = branch_upper / trunk_distal_ref
                    ref_median = trunk_distal_ref * min(scale, 1.0)
                    ref_median = min(
                        ref_median,
                        sorted_h_full[int(n_h * 0.80)] if n_h > 4 else branch_upper,
                    )
                else:
                    ref_median = branch_upper
                median_d = ref_median

            if median_d <= 0:
                continue

            # ── Pre-flagged narrowing from _annotateTreeDiameters ─────────────
            # branchMeta may carry a 'stenosisCandidate' written by the
            # ⚠MIN confidence logic (HIGH/MED, non-distal-tip).  Seed a
            # finding here so it appears even if the pancake-ratio gate
            # below doesn't trigger (curved renals can have low ellipticity
            # even at genuine focal narrowings).
            # Only seed if no run-based finding overlaps this gi later
            # (checked by gi_min comparison at end of loop — for now, always
            # seed and let dedup handle overlap in the UI layer).
            _sc = self.branchMeta.get(bi, {}).get("stenosisCandidate")
            if _sc and _sc.get("confidence") in ("HIGH", "MED"):
                _sc_gi = _sc.get("gi_min", stable_start)
                _sc_gi_start = _sc.get("gi_run_start", _sc_gi)
                _sc_gi_end = _sc.get("gi_run_end", _sc_gi)
                if stable_start <= _sc_gi < stable_end:
                    _sc_d = _sc["min_diam"]
                    _sc_ref = max(_sc["ref_diam"], median_d)
                    _sc_ratio = round(_sc_ref / max(_sc_d, 0.1), 2)
                    _conf_str = _sc["confidence"]
                    _phenotype = _sc.get("phenotype", "")
                    _drop_str = f"{_sc['drop_pct']}%"
                    _run_str = f"{_sc['run_mm']}mm"
                    _n_run = max(1, _sc_gi_end - _sc_gi_start + 1)
                    _pheno_str = f", {_phenotype}" if _phenotype else ""
                    fdesc = (
                        f"Focal narrowing Ø{_sc_d:.1f}mm ({_drop_str} drop, "
                        f"{_run_str} run) [{_conf_str}{_pheno_str}]"
                    )
                    print(
                        f"[FindingsPreFlag] Branch {bi}: {fdesc} "
                        f"gi={_sc_gi_start}–{_sc_gi_end}"
                    )
                    self.findings.append(
                        {
                            "type": "Focal Narrowing",
                            "branchIdx": bi,
                            "pointIdx": _sc_gi,
                            "runStart": _sc_gi_start,
                            "runEnd": _sc_gi_end,
                            "runLen": _n_run,
                            "value": round(_sc_d, 2),
                            "ref": round(_sc_ref, 2),
                            "ratio": _sc_ratio,
                            "z": (
                                round(self.points[_sc_gi][2], 1)
                                if _sc_gi < len(self.points)
                                else 0.0
                            ),
                            "dist": (
                                round(self.distances[_sc_gi], 1)
                                if _sc_gi < len(self.distances)
                                else 0.0
                            ),
                            "desc": fdesc,
                            "dc_severe": _sc_ratio >= 2.0,
                            "dc_mild": _sc_ratio >= 1.5,
                            "source": "preFlag",
                        }
                    )
            # Previously used s+3 here, which caused the flare zone to be scanned
            # for dilation → false aneurysm detections at every branch proximal end.
            scan_range = list(range(stable_start, stable_end))

            # ── STENOSIS RUNS ─────────────────────────────────────────────────
            # Flag each point as a candidate stenosis point
            stenosis_flags = []
            for i in scan_range:
                d = diams_to_use[i]
                if d < 0.5:
                    stenosis_flags.append((i, False))
                    continue
                rel = d / median_d
                ratio = self.pancake_ratios[i]
                # Candidate: diameter < 83% of reference (any narrowing worth tracking).
                # NOTE: ellipse ratio gate (ratio >= 1.1) removed — venous compression
                # is often concentric (uniform narrowing → ratio ≈ 1.0) so requiring
                # non-circularity silently dropped all run-scanner findings for the
                # most common venous pathology.  The ratio is still used for
                # Pancaking vs Mild Compression classification below.
                is_candidate = rel < 0.83
                stenosis_flags.append((i, is_candidate))

            # Collect runs of ≥ 2 consecutive candidate points
            all_stenosis_runs = _filter_by_length(
                _collect_runs(stenosis_flags, min_run=2), "stenosis", bi
            )

            for run in all_stenosis_runs:
                idxs = run["indices"]
                n_run = len(idxs)
                # Worst point = smallest diameter in run
                worst_i = _worst_pt(idxs, lambda gi: diams_to_use[gi])
                d_min = diams_to_use[worst_i]
                d_max = max(diams_to_use[gi] for gi in idxs)
                ratio_w = self.pancake_ratios[worst_i]
                rel_min = d_min / median_d

                # ── detect_compression cross-validation ───────────────────
                # Build (dmin, dmax) pairs for this run from the minor/major
                # ellipse axes, then call the standalone detect_compression()
                # to get an independent severity screen.  If it flags "Severe"
                # on the worst point we upgrade a borderline 2-pt run to
                # Pancaking (overriding the n_run ≥ 3 requirement), and we
                # annotate the description so the clinician knows.
                _dc_pairs = [
                    (self.minor_axes[gi], self.major_axes[gi])
                    for gi in idxs
                    if gi < len(self.minor_axes) and gi < len(self.major_axes)
                ]
                try:
                    _dc_result = detect_compression(_dc_pairs)
                except Exception as _dce:
                    print(f"[FindDiag] detect_compression FAILED bi={bi}: {_dce}")
                    import traceback; traceback.print_exc()
                    _dc_result = []
                _dc_severe = any(sev == "Severe" for sev, _ in _dc_result)
                _dc_mild = any(sev == "Mild" for sev, _ in _dc_result)

                # Classify by run length and severity
                # 1 pt  → below min_run=2, never collected
                # 2–3 pts → possible short lesion → Mild Compression
                # ≥ 3 pts → definite lesion
                #   with ellipse ratio ≥ 1.5           → Pancaking
                #   or detect_compression flags Severe  → Pancaking (upgraded)
                #   otherwise                           → Mild Compression
                _run_mm = run.get("arc_mm", 0.0)
                if (n_run >= 3 and ratio_w >= 1.5) or (n_run >= 2 and _dc_severe):
                    ftype = "Pancaking"
                    _dc_note = (
                        " [dc:Severe]"
                        if _dc_severe and not (n_run >= 3 and ratio_w >= 1.5)
                        else ""
                    )
                    fdesc = (
                        f"Definite stenosis: {n_run} pts ({_run_mm:.1f}mm), "
                        f"min D={d_min:.1f}mm ({rel_min*100:.0f}% ref), "
                        f"ellipse ratio={ratio_w:.1f}{_dc_note}"
                    )
                    ftype_code = 2
                else:
                    ftype = "Mild Compression"
                    _dc_note = " [dc:Mild]" if _dc_mild else ""
                    fdesc = (
                        f"{'Definite' if n_run >= 3 else 'Possible'} narrowing: "
                        f"{n_run} pts ({_run_mm:.1f}mm), "
                        f"min D={d_min:.1f}mm ({rel_min*100:.0f}% ref){_dc_note}"
                    )
                    ftype_code = 1

                for gi in idxs:
                    if self.finding_type[gi] < ftype_code:
                        self.finding_type[gi] = ftype_code

                self.findings.append(
                    {
                        "type": ftype,
                        "branchIdx": bi,
                        "pointIdx": worst_i,
                        "runStart": run["start"],
                        "runEnd": run["end"],
                        "runLen": n_run,
                        "value": round(d_min, 2),
                        "ref": round(median_d, 2),
                        "ratio": round(ratio_w, 2),
                        "z": round(self.points[worst_i][2], 1),
                        "dist": round(self.distances[worst_i], 1),
                        "desc": fdesc,
                        "dc_severe": _dc_severe,
                        "dc_mild": _dc_mild,
                    }
                )

            # ── ILIAC ECCENTRIC COMPRESSION (pancake ratio scan) ──────────────
            # The standard stenosis run-scanner uses mean cross-section diameter
            # (average of ray hits).  Eccentric external compression — e.g. right
            # iliac artery crossing producing a slit-shaped left iliac lumen —
            # flattens the vessel without reducing the MEAN diameter proportionally:
            # minor axis collapses while major axis is preserved, so the mean stays
            # above 0.83 × ref and the run-scanner fires nothing.
            #
            # Fix: for confirmed iliac branches (role main/iliac_left/iliac_right),
            # run a SECOND scan on minor_axis / median_d.  A run of ≥3 points where
            # minor_axis < ILIAC_MINOR_THRESH × median_d AND pancake_ratio ≥
            # ILIAC_RATIO_THRESH (non-circular cross-section) is classified as
            # Eccentric Compression.  Thresholds are looser than the main stenosis
            # scanner to catch venous slit-compression that does not reduce lumen area
            # as severely as arterial stenosis.
            #
            # Only fires for iliac branches in venous mode (arterial iliacs have
            # symmetric muscular walls — pancake finding there is anatomy not disease).
            #
            # IliacEccentricGuard: additionally require that the branch meets minimum
            # anatomical criteria for a true iliac vein before running this scanner:
            #   1. Arc length >= ILIAC_MIN_ARC_MM (60 mm) — renal veins and short
            #      side branches mislabeled as 'main' by the pair-scorer are much
            #      shorter; true iliac veins are always > 80 mm.
            #   2. Z-extent (start Z − end Z) >= ILIAC_MIN_Z_DROP_MM (30 mm) —
            #      a true iliac vein descends well below the bifurcation; a branch
            #      that stays within 30 mm of its start Z is a renal / side branch
            #      that only briefly dips below the trunk, not an iliac.
            # Both guards are logged when they block the scanner so the audit trail
            # is clear.
            _bm_role_ec = self.branchMeta.get(bi, {}).get("role", "")
            _is_iliac_branch = _bm_role_ec in ("main", "iliac_left", "iliac_right")

            if _is_iliac_branch and _venous:
                # ── Guard 1: minimum arc length ───────────────────────────────
                ILIAC_MIN_ARC_MM = 60.0  # true iliac veins are always >= 80 mm
                if _branch_arc_mm < ILIAC_MIN_ARC_MM:
                    print(
                        f"[IliacEccentricGuard] Branch {bi} ({_bm_role_ec}): "
                        f"SKIP eccentric scan — arc={_branch_arc_mm:.1f}mm < "
                        f"{ILIAC_MIN_ARC_MM:.0f}mm (too short to be a true iliac vein)"
                    )
                    _is_iliac_branch = False

            if _is_iliac_branch and _venous:
                # ── Guard 2: minimum Z-descent from branch start to end ───────
                # True iliac veins descend well below the bifurcation.
                # Branches that stay near the trunk Z-level are renal / side branches.
                ILIAC_MIN_Z_DROP_MM = 30.0
                _br_start_z = (
                    self.points[s][2] if s < len(self.points) else 0.0
                )
                _br_end_z = (
                    self.points[e - 1][2] if e - 1 < len(self.points) else 0.0
                )
                _z_drop = _br_start_z - _br_end_z  # positive = descends caudally
                if _z_drop < ILIAC_MIN_Z_DROP_MM:
                    print(
                        f"[IliacEccentricGuard] Branch {bi} ({_bm_role_ec}): "
                        f"SKIP eccentric scan — Z-drop={_z_drop:.1f}mm < "
                        f"{ILIAC_MIN_Z_DROP_MM:.0f}mm (branch does not descend enough "
                        f"to be a true iliac vein; likely renal/side branch)"
                    )
                    _is_iliac_branch = False

            if _is_iliac_branch and _venous:
                ILIAC_MINOR_THRESH = 0.88  # minor axis < 88% of ref → candidate
                ILIAC_RATIO_THRESH = 1.25  # ellipse ratio > 1.25 → non-circular
                ILIAC_MIN_RUN_PTS = 3  # minimum consecutive points
                ILIAC_MIN_RUN_MM = 8.0  # minimum arc length (mm)

                eccentric_flags = []
                for _ei in scan_range:
                    _minor = (
                        self.minor_axes[_ei]
                        if _ei < len(self.minor_axes)
                        else diams_to_use[_ei]
                    )
                    _ratio = (
                        self.pancake_ratios[_ei]
                        if _ei < len(self.pancake_ratios)
                        else 1.0
                    )
                    if _minor < 0.5:
                        eccentric_flags.append((_ei, False))
                        continue
                    _rel_minor = _minor / median_d
                    _is_ec = (_rel_minor < ILIAC_MINOR_THRESH) and (
                        _ratio >= ILIAC_RATIO_THRESH
                    )
                    eccentric_flags.append((_ei, _is_ec))

                all_eccentric_runs = _collect_runs(
                    eccentric_flags, min_run=ILIAC_MIN_RUN_PTS
                )
                # length-filter
                all_eccentric_runs = _filter_by_length(
                    all_eccentric_runs, "eccentric", bi
                )
                # further filter: arc >= ILIAC_MIN_RUN_MM (already handled by _filter_by_length
                # with MIN_LESION_MM=5mm, but ILIAC_MIN_RUN_MM is stricter)
                all_eccentric_runs = [
                    r
                    for r in all_eccentric_runs
                    if r.get("arc_mm", 0.0) >= ILIAC_MIN_RUN_MM
                ]

                for ec_run in all_eccentric_runs:
                    # Skip if already covered by a standard stenosis finding in same range
                    _ec_idxs = ec_run["indices"]
                    _already_flagged = any(
                        self.finding_type[gi] >= 1
                        for gi in _ec_idxs
                        if gi < len(self.finding_type)
                    )
                    if _already_flagged:
                        continue

                    _ec_worst = _worst_pt(
                        _ec_idxs,
                        lambda gi: (
                            self.minor_axes[gi]
                            if gi < len(self.minor_axes)
                            else diams_to_use[gi]
                        ),
                    )
                    _ec_minor = (
                        self.minor_axes[_ec_worst]
                        if _ec_worst < len(self.minor_axes)
                        else diams_to_use[_ec_worst]
                    )
                    _ec_ratio = (
                        self.pancake_ratios[_ec_worst]
                        if _ec_worst < len(self.pancake_ratios)
                        else 1.0
                    )
                    _ec_rel = _ec_minor / median_d
                    _ec_run_mm = ec_run.get("arc_mm", 0.0)
                    _ec_n = len(_ec_idxs)

                    # Severity: severe if minor < 75% ref AND ratio > 1.5 (slit)
                    _ec_severe = (_ec_rel < 0.75) and (_ec_ratio >= 1.5)
                    _ec_ftype = "Pancaking" if _ec_severe else "Mild Compression"
                    _ec_code = 2 if _ec_severe else 1
                    _ec_label = (
                        "Eccentric compression (slit)"
                        if _ec_severe
                        else "Eccentric compression"
                    )
                    _ec_desc = (
                        f"{_ec_label}: {_ec_n} pts ({_ec_run_mm:.1f}mm), "
                        f"minor Ø={_ec_minor:.1f}mm ({_ec_rel*100:.0f}% ref), "
                        f"ellipse ratio={_ec_ratio:.2f}"
                    )
                    print(
                        f"[EccentricCompression] Branch {bi} ({_bm_role_ec}): {_ec_desc}"
                    )

                    for gi in _ec_idxs:
                        if (
                            gi < len(self.finding_type)
                            and self.finding_type[gi] < _ec_code
                        ):
                            self.finding_type[gi] = _ec_code

                    self.findings.append(
                        {
                            "type": _ec_ftype,
                            "branchIdx": bi,
                            "pointIdx": _ec_worst,
                            "runStart": ec_run["start"],
                            "runEnd": ec_run["end"],
                            "runLen": _ec_n,
                            "value": round(_ec_minor, 2),
                            "ref": round(median_d, 2),
                            "ratio": round(_ec_ratio, 2),
                            "z": (
                                round(self.points[_ec_worst][2], 1)
                                if _ec_worst < len(self.points)
                                else 0.0
                            ),
                            "dist": (
                                round(self.distances[_ec_worst], 1)
                                if _ec_worst < len(self.distances)
                                else 0.0
                            ),
                            "desc": _ec_desc,
                            "dc_severe": _ec_severe,
                            "dc_mild": not _ec_severe,
                            "source": "eccentric",
                        }
                    )

            # ── DILATION RUNS ─────────────────────────────────────────────────
            # Ectasia:  D/ref ≥ 1.2  AND  Dmin/Dmax > 0.8  AND ≥ 3 consecutive pts
            # Aneurysm: D/ref ≥ 1.5  AND ≥ 3 consecutive pts
            dilation_flags = []
            for i in scan_range:
                d = diams_to_use[i]
                if d < 0.5:
                    dilation_flags.append((i, False))
                    continue
                rel = d / median_d
                is_candidate = rel >= 1.2
                dilation_flags.append((i, is_candidate))

            all_dilation_runs = _filter_by_length(
                _collect_runs(dilation_flags, min_run=3), "dilation", bi
            )

            for run in all_dilation_runs:
                idxs = run["indices"]
                n_run = len(idxs)
                worst_i = _worst_pt(idxs, lambda gi: -diams_to_use[gi])  # largest D
                d_max = diams_to_use[worst_i]
                d_min_r = min(diams_to_use[gi] for gi in idxs)
                rel_max = d_max / median_d

                # Ectasia rule: D/ref ≥ _ECTASIA_RATIO AND Dmin/Dmax > 0.8 AND ≥ 3 pts
                # Aneurysm:     D/ref ≥ _ANEURYSM_RATIO AND ≥ 3 pts
                # (thresholds scaled by vessel type; see context block above)
                shape_uniform = (d_min_r / d_max) > 0.8  # diffuse, not focal spike

                # ── Bifurcation-zone suppression (venous) ─────────────────────
                # In venous mode, smooth dilation near the carina is physiologic
                # Y-expansion, not a true aneurysm.  Suppress unless it is a
                # focal spike (shape_uniform=True near bif = genuine finding).
                if _bif_pt is not None and _BIF_DOME_MM > 0:
                    import math as _math

                    _wp = self.points[worst_i]
                    _d_bif = _math.sqrt(
                        sum((_wp[k] - _bif_pt[k]) ** 2 for k in range(3))
                    )
                    # Ectasia near the bifurcation is clinically meaningful (post-thrombotic
                    # remodeling, chronic venous HTN) — never suppress it.
                    # Only suppress aneurysm-grade smooth expansion inside the dome zone.
                    _is_ectasia_grade = rel_max < _ANEURYSM_RATIO
                    if (
                        _d_bif < _BIF_DOME_MM
                        and not shape_uniform
                        and not _is_ectasia_grade
                    ):
                        print(
                            f"[FindingSuppress] Branch {bi} dilation gi={worst_i}: "
                            f"within {_d_bif:.1f}mm of bif, smooth aneurysm-grade expansion "
                            f"(venous physiology) — suppressed"
                        )
                        continue
                # ─────────────────────────────────────────────────────────────

                _run_mm_d = run.get("arc_mm", 0.0)
                if rel_max >= _ANEURYSM_RATIO:
                    ftype = "Aneurysm"
                    ftype_code = 3
                    fdesc = (
                        f"Aneurysm: {n_run} pts ({_run_mm_d:.1f}mm), max D={d_max:.1f}mm "
                        f"({rel_max:.1f}x ref, threshold={_ANEURYSM_RATIO:.2f}x)"
                    )
                elif rel_max >= _ECTASIA_RATIO and shape_uniform:
                    ftype = "Ectasia"
                    ftype_code = 2
                    fdesc = (
                        f"Ectasia: {n_run} pts ({_run_mm_d:.1f}mm), max D={d_max:.1f}mm "
                        f"({rel_max:.1f}x ref, threshold={_ECTASIA_RATIO:.2f}x), "
                        f"Dmin/Dmax={d_min_r/d_max:.2f}>0.8"
                    )
                else:
                    # below threshold or non-uniform shape — not reportable
                    continue

                for gi in idxs:
                    if self.finding_type[gi] < ftype_code:
                        self.finding_type[gi] = ftype_code

                self.findings.append(
                    {
                        "type": ftype,
                        "branchIdx": bi,
                        "pointIdx": worst_i,
                        "runStart": run["start"],
                        "runEnd": run["end"],
                        "runLen": n_run,
                        "value": round(d_max, 2),
                        "ref": round(median_d, 2),
                        "ratio": round(rel_max, 2),
                        "z": round(self.points[worst_i][2], 1),
                        "dist": round(self.distances[worst_i], 1),
                        "desc": fdesc,
                    }
                )

        # ── FindDiag: per-branch summary ─────────────────────────────────────
        _branch_findings = [f for f in self.findings if f.get("branchIdx") == bi]
        for _bi_check in range(1, len(self.branches)):
            _bi_f = [f for f in self.findings if f.get("branchIdx") == _bi_check]
            _name = self.getBranchDisplayName(_bi_check)
            print(
                f"[FindDiag] bi={_bi_check} '{_name}' findings so far: "
                f"{[f['type'] for f in _bi_f] if _bi_f else 'none'}"
            )

        # ── Bifurcation-level findings: symmetry and difficulty ───────────────
        # These are global (not per-point) findings stored with branchIdx=-1
        # and pointIdx=-1 so downstream report code can identify them.
        # Values are read from loadCenterline's stored attributes.
        _sym_ratio = getattr(self, "_bifSymmetryRatio", None)
        _sym_label = getattr(self, "_bifSymmetryLabel", None)
        _diff_str = getattr(self, "_bifDifficulty", None)
        _mean_ang = getattr(self, "_bifMeanAngleDeg", None)

        if _sym_ratio is not None and _sym_label is not None:
            if _sym_ratio > 1.2:
                _sym_type = "Asymmetry"
            elif _sym_ratio > 1.1:
                _sym_type = "Mild Asymmetry"
            else:
                _sym_type = "Symmetry"
            self.findings.append(
                {
                    "type": _sym_type,
                    "branchIdx": -1,
                    "pointIdx": -1,
                    "value": _sym_ratio,
                    "ref": 1.0,
                    "ratio": _sym_ratio,
                    "z": 0.0,
                    "dist": 0.0,
                    "desc": f"Branch diameter symmetry: {_sym_label}",
                }
            )

        if _diff_str is not None and _mean_ang is not None:
            if _mean_ang < 20.0:
                _diff_type = "Bifurcation: Easy"
            elif _mean_ang < 40.0:
                _diff_type = "Bifurcation: Moderate"
            else:
                _diff_type = "Bifurcation: Complex"
            self.findings.append(
                {
                    "type": _diff_type,
                    "branchIdx": -1,
                    "pointIdx": -1,
                    "value": _mean_ang,
                    "ref": 0.0,
                    "ratio": 1.0,
                    "z": 0.0,
                    "dist": 0.0,
                    "desc": f"Bifurcation difficulty: {_diff_str}",
                }
            )

        # ── Deduplicate: one finding per type-family per branch ───────────────
        # Stenosis and dilation are separate families.
        # Within each family, keep only the worst finding per 50mm window.
        def _type_family(ftype):
            return "dilation" if ftype in ("Aneurysm", "Ectasia") else "compression"

        self.findings.sort(
            key=lambda f: (
                f["branchIdx"],
                _type_family(f["type"]),
                -f.get("ratio", 1.0),
            )
        )

        deduped = []
        last_dist_per_key = {}
        for f in self.findings:
            # Global findings (branchIdx=-1) are always kept — they are unique
            # by type and have no spatial coordinate to deduplicate against.
            if f["branchIdx"] < 0:
                deduped.append(f)
                continue
            key = (f["branchIdx"], _type_family(f["type"]))
            last_d = last_dist_per_key.get(key, -999.0)
            sep_mm = 50.0
            if abs(f["dist"] - last_d) >= sep_mm:
                deduped.append(f)
                last_dist_per_key[key] = f["dist"]

        deduped.sort(key=lambda f: (f["branchIdx"], f["pointIdx"]))
        self.findings = deduped

        n_a = sum(1 for f in self.findings if f["type"] == "Aneurysm")
        n_e = sum(1 for f in self.findings if f["type"] == "Ectasia")
        n_p = sum(1 for f in self.findings if "Pancak" in f["type"])
        n_m = sum(1 for f in self.findings if "Compress" in f["type"])
        n_ec = sum(1 for f in self.findings if f.get("source") == "eccentric")
        # FIX: report all finding counts in a single clear line; pancaking is never hidden
        print(
            f"[VesselAnalyzer] Findings: {n_a} aneurysm, {n_e} ectasia, "
            f"{n_p} pancaking, {n_m} mild compression, {n_ec} eccentric iliac"
        )
        if n_p > 0 or n_ec > 0:
            _total_comp = n_p + n_ec
            if _venous:
                _ec_note = (
                    f", including {n_ec} eccentric iliac finding(s)" if n_ec > 0 else ""
                )
                print(
                    f"[VesselAnalyzer] *** COMPRESSION CONFIRMED: {_total_comp} location(s)"
                    f"{_ec_note} — bilateral iliac compression pattern (venous). "
                    f"Clinical review required before any intervention. ***"
                )
            else:
                print(
                    f"[VesselAnalyzer] *** PANCAKING CONFIRMED: {n_p} location(s) — kissing stent indicated ***"
                )
        for f in self.findings:
            bi = f["branchIdx"]
            bs = self.getBranchStartGi(bi)  # anatomical start
            loc_pt = f["pointIdx"] - bs
            loc_s = f.get("runStart", f["pointIdx"]) - bs
            loc_e = f.get("runEnd", f["pointIdx"]) - bs
            n_run = f.get("runLen", 1)
            run_tag = (
                f"run: local pt {loc_s}–{loc_e} ({n_run} pts)"
                if n_run > 1
                else f"local pt {loc_pt}"
            )
            print(
                f"  -> {f['type']} {self.getBranchDisplayName(bi)} "
                f"({run_tag}, worst D={f['value']:.2f}mm, ref={f['ref']:.2f}mm)"
            )
        return self.findings


# ── Slicer module-scanner guard ───────────────────────────────────────────────
# Slicer auto-scans all .py files in the module folder and expects a class
# matching the filename.  This stub satisfies that requirement without
# registering as a real loadable module (no ScriptedLoadableModule base).
class vessel_findings_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_findings_mixin"
            parent.hidden = True  # hide from Slicer module list
