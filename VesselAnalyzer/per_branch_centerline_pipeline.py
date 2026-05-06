# ─────────────────────────────────────────────────────────────────────────────
# Per-Branch Centerline Pipeline  (v282-TrunkInletSolve)
#
# Bug fixes in v276 (on top of v275 optimisations):
#
#   BUG 1 — Branch centerlines cross at the iliac bifurcation (continued fix)
#   ──────────────────────────────────────────────────────────────────────────
#   v276 partial fix: per-branch seeds nudged 25% of the full 3D bif→target
#   vector toward each target.  This fixed the upper (renal-level) branches
#   but the iliac branches (Z≈1665-1671, ~110mm below the bif) still crossed.
#
#   Root cause of v276 regression on iliacs: the 3D bif→target vector has a
#   large Z component (~-110mm for iliacs).  25% of that = -27mm of Z nudge,
#   pulling each seed far below the bifurcation into the iliac lumen itself.
#   VMTK then found a short spurious route (71mm / 82mm) from inside the iliac
#   instead of the correct full-arc path (165mm / 145mm).
#
#   v277 fix: per_branch_seeds() now uses an XY-only nudge of fixed magnitude
#   (lateral_mm=6mm ≈ half a typical iliac radius).  Zeroing the Z component
#   keeps the seed at the bifurcation Z level so VMTK solves the full branch
#   arc, while the XY bias is still large enough to push the Dijkstra start
#   node to the correct side of the bifurcation wall.
#
#   Expected log:
#     [PerBranchCL] per-branch seed: [xx.x  xx.x  1763.6]  (nudge_xy=[...])
#   Note Z stays near 1763.6 (original p_src Z) for all branches.
#   Root cause A: single shared p_src for all branch VMTK solves.
#     When every branch solve starts from the same upstream seed point, VMTK's
#     Dijkstra algorithm on the shared preprocessed mesh finds the globally
#     shortest geodesic from that common node.  For the two iliac outlets the
#     shortest paths share a long common-trunk mesh segment and then cross as
#     they diverge toward opposite sides of the bifurcation — producing the
#     visually interleaved/crossing centerlines seen in the Slicer viewport.
#
#     Fix: BranchSeedGenerator.per_branch_seeds() generates one proximal seed
#     per branch, each nudged laterally by 25 % of the bif→target vector.
#     This biases each Dijkstra solve onto the correct anatomical side of the
#     bifurcation wall from the first graph edge.
#     PerBranchSolver.solve_per_branch() uses these per-branch seeds.
#
#   Root cause B: endpoint positions not snapped to preprocessed mesh.
#     _extract_with_cached_mesh() passed the raw RAS fiducial positions to
#     VMTK.  The preprocessed (decimated) mesh has far fewer vertices than
#     the original; a raw position that lies between mesh vertices causes
#     VMTK to pick the globally nearest vertex, which can be on the wrong
#     branch wall when two outlets are anatomically close.
#
#     Fix: _extract_with_cached_mesh() now snaps every endpoint to its
#     nearest vertex on the preprocessed polydata via vtkPointLocator before
#     creating the fiducial node passed to VMTK.
#
# Performance improvements (unchanged from v275):
#
#   1. Phase 1 preprocessing SHARED with Phase 2
#   2. Phase 1 targetPoints lowered for bif-geometry extraction only
#   3. Phase 4 refinement is OFF by default (refine=False)
#   4. Phase 2 uses preprocess=False with the cached mesh
#
# Architecture (unchanged externally):
#   Phase 1  — global solve  →  bifurcation geometry only (snappedBifPt +
#               trunk direction).  Branch centerlines from this solve are
#               DISCARDED.
#   Phase 2  — per-branch independent VMTK solves, one seed pair per branch,
#               using the CACHED preprocessed mesh from Phase 1.
#   Phase 3  — tangent-preserving junction snap.
#   Phase 4  — optional light refinement (disabled by default).
#
# Integration (unchanged):
#   • Drop next to VesselAnalyzer.py.
#   • onExtractCenterline() creates PerBranchCenterlinePipeline and calls
#     pipeline.run(endpointsNode) — same API as before.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import vtk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Slicer module boilerplate — required so Slicer can load this file as a
#  scripted loadable module.  The class name MUST match the filename stem.
# ═════════════════════════════════════════════════════════════════════════════

class per_branch_centerline_pipeline(ScriptedLoadableModule):
    """Slicer module wrapper for the per-branch centerline pipeline."""

    def __init__(self, parent):
        super().__init__(parent)
        parent.title       = "Per-Branch Centerline Pipeline"
        parent.categories  = ["Vascular"]
        parent.dependencies = []
        parent.contributors = ["VesselAnalyzer"]
        parent.helpText    = (
            "Per-branch centerline extraction pipeline (v297-RenalDivergeTrunkOnly — "
            "renal divergence check compares against iliac_arrays only (not all "
            "prior branches); sibling renal trunk-prefix no longer causes false "
            "diverge_idx inflation; iliac_arrays tracked separately in the loop)."
        )
        parent.acknowledgementText = ""


class per_branch_centerline_pipelineWidget(ScriptedLoadableModuleWidget):
    """Minimal widget — this module is used programmatically, not via GUI."""

    def setup(self):
        super().setup()

    def cleanup(self):
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  Helper: thin numpy wrapper around a vtkMRMLMarkupsCurveNode
# ═════════════════════════════════════════════════════════════════════════════

def _curve_to_array(curve_node):
    """Return (N,3) float64 array of control-point positions (RAS mm)."""
    n = curve_node.GetNumberOfControlPoints()
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        curve_node.GetNthControlPointPosition(i, pts[i])
    return pts


def _array_to_curve(pts, name, scene=None):
    """Create a new vtkMRMLMarkupsCurveNode from an (N,3) array."""
    if scene is None:
        scene = slicer.mrmlScene
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
    node.GetDisplayNode().SetVisibility(False)
    for p in pts:
        node.AddControlPoint(vtk.vtkVector3d(p[0], p[1], p[2]))
    return node


def _arc_length(pts):
    """Cumulative arc-length array, same length as pts."""
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _resample_uniform(pts, step_mm=0.5):
    """Resample a polyline to uniform spacing step_mm."""
    arc = _arc_length(pts)
    total = arc[-1]
    if total < step_mm:
        return pts
    s_new = np.arange(0.0, total, step_mm)
    out = np.zeros((len(s_new), 3))
    for dim in range(3):
        out[:, dim] = np.interp(s_new, arc, pts[:, dim])
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Mesh preprocessing cache
#  Runs vtkCenterline preprocess() ONCE and shares the result across all solves
# ═════════════════════════════════════════════════════════════════════════════

class _PreprocessCache:
    """
    Runs ExtractCenterlineLogic.preprocess() once with the given parameters
    and hands back the cached vtkPolyData for all subsequent VMTK calls.

    Usage:
        cache = _PreprocessCache(logic, model_node, target_points=8000)
        pd    = cache.get()   # subsequent calls return the same object instantly
    """

    def __init__(self, model_node, target_points=8000,
                 decimation_aggressiveness=3.5, subdivide=True):
        self.model_node             = model_node
        self.target_points          = target_points
        self.decimation_aggressiveness = decimation_aggressiveness
        self.subdivide              = subdivide
        self._cached_pd             = None

    def get(self):
        if self._cached_pd is not None:
            return self._cached_pd

        print(f"[PerBranchCL] Preprocessing mesh (targetPoints={self.target_points}, "
              f"subdivide={self.subdivide}, decim={self.decimation_aggressiveness}) …")
        try:
            import ExtractCenterline
            logic = ExtractCenterline.ExtractCenterlineLogic()
            self._cached_pd = logic.preprocess(
                self.model_node.GetPolyData(),
                self.target_points,
                decimationAggressiveness=self.decimation_aggressiveness,
                subdivide=self.subdivide,
            )
        except Exception as e:
            print(f"[PerBranchCL]  ⚠  preprocess failed ({e}), using raw polydata")
            self._cached_pd = self.model_node.GetPolyData()

        n_pts = self._cached_pd.GetNumberOfPoints() if self._cached_pd else 0
        print(f"[PerBranchCL] Preprocessed mesh: {n_pts} points (cached for all solves)")
        return self._cached_pd


# ═════════════════════════════════════════════════════════════════════════════
#  Phase 1 — bifurcation geometry from the global solve
# ═════════════════════════════════════════════════════════════════════════════

class BifurcationGeometry:
    """
    Runs one global VMTK solve (all endpoints) to extract bifurcation
    geometry.  Only snappedBifPt and trunkDir are kept; the branch
    centerlines are discarded.

    OPTIMISATION: accepts a _PreprocessCache so the preprocessing step
    is NOT repeated by extractCenterline() internally — we pass the
    already-preprocessed polydata directly.
    """

    def __init__(self, logic, model_node, endpoints_node, preprocess_cache=None):
        self.logic              = logic
        self.model_node         = model_node
        self.endpoints_node     = endpoints_node
        self.preprocess_cache   = preprocess_cache  # accepted but unused (geometry-only path)

        self.snapped_bif_pt  = None
        self.trunk_dir        = None
        self.trunk_radius     = None

    # ------------------------------------------------------------------
    def compute(self):
        """
        Locate the bifurcation using a cheap 3-endpoint VMTK solve.

        FIX (v288-TrueBif): the previous geometric estimate (centroid of
        distal endpoints + medial-axis walk) was not reliable enough to
        anchor iliac_bif_seed() or the post-solve bif trim.  Small errors
        in snapped_bif_pt propagate into wrong lateral ray directions and
        wrong trim indices.

        New strategy:
          1. Pick the 3 most important endpoints: the proximal inlet
             (highest Z) + the 2 iliac tips (lowest Z pair).  Solving with
             only 3 endpoints is much faster than a full N-endpoint solve
             (~5-15 s vs 30-120 s for a 5-endpoint mesh).
          2. Run VMTK on those 3 endpoints with low target_points (4000)
             for speed.  VMTK's graph topology naturally places the
             bifurcation node at the true anatomical fork.
          3. Read the bifurcation coordinate from the split point in the
             returned polyline (the point shared by both branch paths).
          4. Fall back to the geometric estimate if the VMTK solve fails.
        """
        print("[PerBranchCL] Phase 1: locating bifurcation via 3-endpoint VMTK solve ...")

        pts_all = []
        n = self.endpoints_node.GetNumberOfControlPoints()
        for i in range(n):
            p = [0.0, 0.0, 0.0]
            self.endpoints_node.GetNthControlPointPosition(i, p)
            pts_all.append(np.array(p))

        if len(pts_all) < 2:
            raise RuntimeError("[PerBranchCL] Need >= 2 endpoints")

        pts_arr   = np.array(pts_all)
        z_vals    = pts_arr[:, 2]
        inlet_idx = int(np.argmax(z_vals))
        inlet_pt  = pts_arr[inlet_idx]
        distal    = [(i, p) for i, p in enumerate(pts_arr) if i != inlet_idx]

        # Try a fast 3-endpoint VMTK solve to get a true bifurcation point.
        vmtk_bif = self._vmtk_bif_from_3pts(inlet_pt, distal)

        if vmtk_bif is not None:
            self.snapped_bif_pt = vmtk_bif
            print(f"[PerBranchCL] bif_pt={np.round(self.snapped_bif_pt, 1)}  (VMTK 3-pt solve)")
        else:
            # Fallback: geometric estimate (original method)
            print("[PerBranchCL] Phase 1: VMTK solve failed, falling back to geometric estimate")
            bif_raw = np.array([p for _, p in distal]).mean(axis=0)
            self.snapped_bif_pt = self._snap_to_medial_axis(inlet_pt, bif_raw)
            print(f"[PerBranchCL] bif_pt={np.round(self.snapped_bif_pt, 1)}  (geometric fallback)")

        # Trunk direction: inlet -> snapped bif
        v    = self.snapped_bif_pt - inlet_pt
        norm = np.linalg.norm(v)
        self.trunk_dir = v / norm if norm > 1e-6 else np.array([0.0, 0.0, -1.0])

        # Trunk radius: measure 15 mm upstream of the bif.
        upstream_pt = self.snapped_bif_pt - self.trunk_dir * 15.0
        self.trunk_radius = self._estimate_trunk_radius_at(upstream_pt)

        print(f"[PerBranchCL] trunk_dir={np.round(self.trunk_dir, 3)}  "
              f"trunk_r={self.trunk_radius:.1f} mm")
        return self

    # ------------------------------------------------------------------
    def _vmtk_bif_from_3pts(self, inlet_pt, distal_list):
        """
        Run a fast VMTK extractCenterline with only 3 endpoints:
          - inlet_pt  (proximal, highest Z)
          - the 2 distal tips with lowest Z  (the two iliac outlets)

        Returns the bifurcation point as a numpy array, or None on failure.

        The bifurcation is detected as the point along the returned polylines
        that is shared (within BIF_SNAP_MM) by both branch paths — i.e. the
        last common point before the two paths diverge.
        """
        BIF_SNAP_MM = 3.0   # two paths are "shared" if closer than this
        try:
            import ExtractCenterline
            extract_logic = ExtractCenterline.ExtractCenterlineLogic()

            # Choose the 2 most distal (lowest-Z) targets as iliac proxies.
            sorted_distal = sorted(distal_list, key=lambda x: x[1][2])
            iliac_pts = [p for _, p in sorted_distal[:2]]

            # Build a 3-point preprocessed mesh (fast: low target_points).
            pd_raw = self.model_node.GetPolyData()
            try:
                pp_pd = extract_logic.preprocess(
                    pd_raw, 4000,
                    decimationAggressiveness=4.0,
                    subdivide=False,
                )
            except Exception:
                pp_pd = pd_raw

            # Build seed vtkPolyData AND an MRML fiducial node — probe both.
            seed_pd  = vtk.vtkPolyData()
            seed_pts = vtk.vtkPoints()
            for pt in [inlet_pt] + iliac_pts:
                seed_pts.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
            seed_pd.SetPoints(seed_pts)

            # Also build an MRML fiducial node for older Slicer builds.
            ep_node_3pt = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "_pb_bif3pt_ep")
            ep_node_3pt.GetDisplayNode().SetVisibility(False)
            for pt in [inlet_pt] + iliac_pts:
                ep_node_3pt.AddControlPoint(
                    vtk.vtkVector3d(float(pt[0]), float(pt[1]), float(pt[2])))

            def _run():
                errors = []
                # Form A — MRML fiducial node
                for kw in [{"curveSamplingDistance": 1.0}, {}]:
                    try:
                        r = extract_logic.extractCenterline(pp_pd, ep_node_3pt, **kw)
                        print(f"[PerBranchCL] _vmtk_bif_from_3pts: MRML form ok kw={kw}")
                        return r
                    except (TypeError, AttributeError) as e:
                        errors.append(f"MRML{kw}: {e}")
                # Form B — vtkPolyData seeds
                for kw in [{"curveSamplingDistance": 1.0}, {}]:
                    try:
                        r = extract_logic.extractCenterline(pp_pd, seed_pd, **kw)
                        print(f"[PerBranchCL] _vmtk_bif_from_3pts: PD form ok kw={kw}")
                        return r
                    except (TypeError, AttributeError) as e:
                        errors.append(f"PD{kw}: {e}")
                try:
                    r = extract_logic.extractCenterline(pp_pd, seed_pd, 1.0)
                    return r
                except Exception as e:
                    errors.append(f"pos: {e}")
                raise RuntimeError("\n".join(errors))

            try:
                result = _run()
            finally:
                slicer.mrmlScene.RemoveNode(ep_node_3pt)
            cl_pd  = result[0] if isinstance(result, tuple) else result

            if cl_pd is None or cl_pd.GetNumberOfPoints() == 0:
                print("[PerBranchCL] _vmtk_bif_from_3pts: empty result")
                return None

            # Extract all polyline cells from the result.
            paths = []
            lines = cl_pd.GetLines()
            vtk_pts = cl_pd.GetPoints()
            if lines and lines.GetNumberOfCells() > 0:
                lines.InitTraversal()
                id_list = vtk.vtkIdList()
                while lines.GetNextCell(id_list):
                    path = np.array([vtk_pts.GetPoint(id_list.GetId(k))
                                     for k in range(id_list.GetNumberOfIds())])
                    if len(path) > 1:
                        paths.append(path)

            if len(paths) < 2:
                print(f"[PerBranchCL] _vmtk_bif_from_3pts: only {len(paths)} paths, need ≥2")
                return None

            # Find the bifurcation: last point shared between any two paths.
            # Walk path[0] from its distal end (tip) toward the inlet and find
            # the first point that is also close to path[1].
            p0, p1 = paths[0], paths[1]

            # Build KD-tree on path[1] for fast nearest-point queries.
            from scipy.spatial import cKDTree
            kd1 = cKDTree(p1)

            bif_candidate = None
            # Walk p0 from distal → proximal; bifurcation = first shared point.
            for pt in p0[::-1]:
                dist, _ = kd1.query(pt)
                if dist < BIF_SNAP_MM:
                    bif_candidate = pt.copy()
                    break

            if bif_candidate is None:
                # Fallback: use the point on path[0] nearest to path[1]'s start.
                dists = np.linalg.norm(p0 - p1[0], axis=1)
                bif_candidate = p0[int(np.argmin(dists))].copy()
                print("[PerBranchCL] _vmtk_bif_from_3pts: shared-point fallback")

            print(f"[PerBranchCL] _vmtk_bif_from_3pts: bif={np.round(bif_candidate, 1)} "
                  f"({len(paths)} paths returned)")
            return bif_candidate

        except Exception as e:
            print(f"[PerBranchCL] _vmtk_bif_from_3pts failed: {e}")
            return None

    # ------------------------------------------------------------------
    def _snap_to_medial_axis(self, p_start, p_target, n_steps=40, n_rays=8):
        """
        Walk from p_start toward p_target in n_steps.
        At each step measure wall distance via radial rays.
        Return the point of maximum wall distance (= medial axis peak = bif).
        Falls back to p_target if the model has no polydata.
        """
        try:
            pd = self.model_node.GetPolyData()
            if pd is None or pd.GetNumberOfPoints() == 0:
                return p_target

            loc = vtk.vtkCellLocator()
            loc.SetDataSet(pd)
            loc.BuildLocator()

            best_pt   = p_target.copy()
            best_dist = -1.0

            for i in range(n_steps + 1):
                t   = i / n_steps
                pos = p_start + t * (p_target - p_start)

                # Build local frame
                tang = p_target - p_start
                tn   = np.linalg.norm(tang)
                if tn < 1e-6:
                    continue
                tang = tang / tn
                perp = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(perp, tang)) > 0.9:
                    perp = np.array([0.0, 1.0, 0.0])
                perp   = perp - np.dot(perp, tang) * tang
                perp  /= np.linalg.norm(perp)
                binorm = np.cross(tang, perp)

                hits = []
                for k in range(n_rays):
                    angle   = 2.0 * np.pi * k / n_rays
                    ray_dir = np.cos(angle) * perp + np.sin(angle) * binorm
                    p1 = (pos + ray_dir * 0.1).tolist()
                    p2 = (pos + ray_dir * 50.0).tolist()
                    t_hit  = vtk.mutable(0.0)
                    hit_pt = [0.0, 0.0, 0.0]
                    hc     = vtk.mutable(0)
                    si     = vtk.mutable(0)
                    if loc.IntersectWithLine(p1, p2, 0.001, t_hit, hit_pt,
                                             [0.0, 0.0, 0.0], si, hc):
                        d = np.linalg.norm(np.array(hit_pt) - pos)
                        if d > 0.5:
                            hits.append(d)
                if hits:
                    wd = float(np.median(hits))
                    if wd > best_dist:
                        best_dist = wd
                        best_pt   = pos.copy()

            return best_pt
        except Exception as e:
            print(f"[PerBranchCL] _snap_to_medial_axis fallback: {e}")
            return p_target

    # ------------------------------------------------------------------
    def _estimate_trunk_radius_at(self, pos, n_rays=8):
        """Estimate lumen radius at a single point via radial rays."""
        try:
            pd = self.model_node.GetPolyData()
            if pd is None or pd.GetNumberOfPoints() == 0:
                return 8.0
            loc = vtk.vtkCellLocator()
            loc.SetDataSet(pd)
            loc.BuildLocator()

            # Arbitrary frame (trunk is roughly Z-aligned)
            tang   = np.array([0.0, 0.0, 1.0])
            perp   = np.array([1.0, 0.0, 0.0])
            binorm = np.cross(tang, perp)

            hits = []
            for k in range(n_rays):
                angle   = 2.0 * np.pi * k / n_rays
                ray_dir = np.cos(angle) * perp + np.sin(angle) * binorm
                p1 = (pos + ray_dir * 0.1).tolist()
                p2 = (pos + ray_dir * 40.0).tolist()
                t_hit  = vtk.mutable(0.0)
                hit_pt = [0.0, 0.0, 0.0]
                hc     = vtk.mutable(0)
                si     = vtk.mutable(0)
                if loc.IntersectWithLine(p1, p2, 0.001, t_hit, hit_pt,
                                         [0.0, 0.0, 0.0], si, hc):
                    d = np.linalg.norm(np.array(hit_pt) - pos)
                    if d > 0.5:
                        hits.append(d)
            return float(np.median(hits)) if hits else 8.0
        except Exception as e:
            print(f"[PerBranchCL] _estimate_trunk_radius_at fallback: {e}")
            return 8.0



# ═════════════════════════════════════════════════════════════════════════════
#  Phase 2 — per-branch proximal seed generation
# ═════════════════════════════════════════════════════════════════════════════

class BranchSeedGenerator:
    # Back up only 1 × trunk radius from the bifurcation point.
    # 3× (the old value) placed p_src ~40 mm above the fork when the radius
    # estimate was inflated, causing both iliac geodesics to share a long
    # common-trunk segment and cross at the bifurcation.
    BACK_RADII    = 1.0
    ASCENT_STEP_MM = 0.5
    ASCENT_MAX_MM  = 20.0

    def __init__(self, bif_geom, model_node):
        self.bif_geom   = bif_geom
        self.model_node = model_node
        self._loc       = None
        self._obb       = None

    def _get_locator(self):
        if self._loc is None:
            pd = self.model_node.GetPolyData()
            self._loc = vtk.vtkCellLocator()
            self._loc.SetDataSet(pd)
            self._loc.BuildLocator()
        return self._loc

    def _get_obb(self):
        if self._obb is None:
            pd = self.model_node.GetPolyData()
            self._obb = vtk.vtkOBBTree()
            self._obb.SetDataSet(pd)
            self._obb.BuildLocator()
        return self._obb

    def _wall_dist(self, pos):
        loc     = self._get_locator()
        closest = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id  = vtk.reference(0)
        dist2   = vtk.reference(0.0)
        loc.FindClosestPoint(list(pos), closest, cell_id, sub_id, dist2)
        return float(dist2) ** 0.5

    def proximal_seed(self):
        bg     = self.bif_geom
        d_back = self.BACK_RADII * bg.trunk_radius
        p      = bg.snapped_bif_pt - bg.trunk_dir * d_back
        p      = self._ascend_to_medial(p, bg.trunk_dir)
        print(f"[PerBranchCL] p_src={np.round(p, 1)}  "
              f"(d_back={d_back:.1f} mm, wall_dist={self._wall_dist(p):.1f} mm)")
        return p

    def _ascend_to_medial(self, p0, direction):
        p      = p0.copy()
        prev   = self._wall_dist(p)
        step   = self.ASCENT_STEP_MM
        walked = 0.0
        while walked < self.ASCENT_MAX_MM:
            p_next = p + direction * step
            curr   = self._wall_dist(p_next)
            if curr <= prev:
                break
            prev    = curr
            p       = p_next
            walked += step
        return p

    def iliac_bif_seed(self, p_tgt):
        """
        Place a proximal seed well inside the vessel lumen on the same lateral
        side as p_tgt.  Used as the DISTAL endpoint of a tip-inward solve.

        FIX (v288-SeedValidation): the previous single-ray approach fired one
        ray at the bifurcation Z level and stepped back 1.5 mm from the wall.
        This was fragile because:
          (a) 1.5 mm is too close to the wall — coarse/decimated meshes may
              leave the seed outside the Dijkstra graph's interior nodes.
          (b) A bad snapped_bif_pt shifts the lateral direction, causing the
              ray to hit the wrong wall or miss entirely.
          (c) The OBBTree fallback placed the seed in open lumen near the
              bifurcation midline — the worst possible location.

        New strategy:
          1. Scan multiple Z levels (bif ± offsets) and fire a lateral ray at
             each, collecting all wall hits on the correct side.
          2. For each wall hit compute a candidate seed = wall_hit - ray * step,
             then measure wall_dist.  Accept the first candidate where
             MIN_WALL_DIST_MM ≤ wall_dist ≤ MAX_WALL_DIST_MM.
          3. If no candidate passes, use the LUMEN_FRAC fallback: place the
             seed at LUMEN_FRAC of the distance from bif to p_tgt — guaranteed
             to be inside the correct iliac lumen, far from the shared carina.
        """
        MIN_WALL_DIST_MM = 0.8   # seed must be at least this far from wall
        MAX_WALL_DIST_MM = 6.0   # but not so deep it crosses to other side
        STEPBACK_MM      = 4.0   # step back from wall hit before validating
        LUMEN_FRAC       = 0.25  # fallback: 25% of the way from bif → tip

        bif = self.bif_geom.snapped_bif_pt
        obb = self._get_obb()

        # Lateral XY direction from bifurcation toward the target iliac tip.
        lateral_xy = np.array([p_tgt[0] - bif[0], p_tgt[1] - bif[1], 0.0])
        lat_norm   = np.linalg.norm(lateral_xy)
        ray_dir    = (lateral_xy / lat_norm
                      if lat_norm > 1e-6 else np.array([1.0, 0.0, 0.0]))

        # Z offsets to scan: bifurcation level first, then ±5 mm, ±10 mm.
        z_offsets = [0.0, -5.0, 5.0, -10.0, 10.0]

        def _ray_hits(origin):
            p_back = (origin - ray_dir * 80.0).tolist()
            p_fwd  = (origin + ray_dir * 80.0).tolist()
            pts_vtk = vtk.vtkPoints()
            obb.IntersectWithLine(p_back, p_fwd, pts_vtk, None)
            hits = []
            for hi in range(pts_vtk.GetNumberOfPoints()):
                hp = np.array(pts_vtk.GetPoint(hi))
                t  = np.dot(hp - origin, ray_dir)
                if t > 0.5:           # forward hits only
                    hits.append((t, hp))
            hits.sort(key=lambda x: x[0])
            return hits

        best_seed = None
        best_mode = None

        for dz in z_offsets:
            origin = bif + np.array([0.0, 0.0, dz])
            hits   = _ray_hits(origin)
            for _, wall_pt in hits:
                candidate = wall_pt - ray_dir * STEPBACK_MM
                wd = self._wall_dist(candidate)
                if MIN_WALL_DIST_MM <= wd <= MAX_WALL_DIST_MM:
                    best_seed = candidate
                    best_mode = f"wall-raycast dz={dz:+.0f}mm wd={wd:.1f}mm"
                    break
            if best_seed is not None:
                break

        if best_seed is None:
            # Fallback: 25% of the way from bifurcation toward the iliac tip.
            # This point is always inside the correct iliac lumen, well past
            # the shared carina fan, with no dependence on ray casting.
            best_seed = bif + LUMEN_FRAC * (p_tgt - bif)
            wd        = self._wall_dist(best_seed)
            best_mode = f"lumen-frac({LUMEN_FRAC}) wd={wd:.1f}mm"

        # Ensure the seed has a minimum lateral separation from the midline
        # so VMTK cannot route this iliac through the other iliac's lumen.
        # If the seed is within 8mm lateral of bif, push it further along
        # the tip vector until it reaches that minimum.
        MIN_LAT_MM = 8.0
        lateral_dist = np.linalg.norm(best_seed[:2] - bif[:2])
        if lateral_dist < MIN_LAT_MM:
            push_seed = bif + 0.35 * (p_tgt - bif)
            push_seed[2] = best_seed[2]  # keep Z
            best_seed = push_seed
            best_mode += f"+lat_push(to {np.linalg.norm(push_seed[:2]-bif[:2]):.1f}mm)"

        print(f"[PerBranchCL] iliac_bif_seed ({best_mode}): "
              f"{np.round(best_seed, 1)}  tgt={np.round(p_tgt, 1)}")
        return best_seed

    def branch_trunk_seed(self, p_tgt):
        """
        Place a seed just inside the IVC wall at the renal (side-branch)
        takeoff level, for use as the PROXIMAL endpoint of a tip-inward solve.

        Strategy:
          1. At Z = p_tgt[2] (the renal tip Z level), fire a ray from a point
             on the trunk medial axis toward the target.  The first wall hit on
             the TARGET side is the IVC wall at the ostium level.
          2. Step back STEPBACK_MM from that wall hit toward the trunk center.
          3. Validate with wall_dist; fall back to a lumen-fraction point if no
             valid candidate is found within ±10 mm Z scan.

        Unlike iliac_bif_seed() (which fires from bif_pt at bif Z), this
        method fires from the trunk axis at the branch's own Z level so the
        seed is placed inside the IVC at the true ostium height.
        """
        MIN_WALL_DIST_MM = 0.8
        MAX_WALL_DIST_MM = 7.0
        STEPBACK_MM      = 3.0
        LUMEN_FRAC       = 0.10   # fallback: 10% of the way from target toward trunk

        bif   = self.bif_geom.snapped_bif_pt
        obb   = self._get_obb()

        # Lateral direction from trunk axis toward the target (XY only).
        lateral_xy = np.array([p_tgt[0] - bif[0], p_tgt[1] - bif[1], 0.0])
        lat_norm   = np.linalg.norm(lateral_xy)
        ray_dir    = (lateral_xy / lat_norm
                      if lat_norm > 1e-6 else np.array([1.0, 0.0, 0.0]))

        # Scan Z levels anchored at the target tip Z.
        # Venous renal/side branches enter the IVC at a cranial angle, so
        # the true ostium wall contact is typically 5-20 mm CRANIAL to the
        # ring centroid's Z.  Scan cranially first (+dz), then fall back to
        # the centroid level and caudal offsets.
        z_base    = p_tgt[2]
        z_offsets = [5.0, 10.0, 15.0, 20.0, 0.0, -5.0, -10.0]

        def _ray_hits(origin):
            p_back = (origin - ray_dir * 80.0).tolist()
            p_fwd  = (origin + ray_dir * 80.0).tolist()
            pts_vtk = vtk.vtkPoints()
            obb.IntersectWithLine(p_back, p_fwd, pts_vtk, None)
            hits = []
            for hi in range(pts_vtk.GetNumberOfPoints()):
                hp = np.array(pts_vtk.GetPoint(hi))
                t  = np.dot(hp - origin, ray_dir)
                if t > 0.5:
                    hits.append((t, hp))
            hits.sort(key=lambda x: x[0])
            return hits

        best_seed = None
        best_mode = None

        for dz in z_offsets:
            # Fire from the trunk medial axis (bif XY, shifted by dz from
            # target Z) — this guarantees the ray origin is inside the IVC.
            origin = np.array([bif[0], bif[1], z_base + dz])
            hits   = _ray_hits(origin)
            for _, wall_pt in hits:
                candidate = wall_pt - ray_dir * STEPBACK_MM
                wd = self._wall_dist(candidate)
                # Guard: seed must be at least MIN_SEP_MM from p_tgt,
                # otherwise VMTK gets a degenerate near-zero solve.
                sep = float(np.linalg.norm(candidate - p_tgt))
                if sep < 5.0:
                    continue
                if MIN_WALL_DIST_MM <= wd <= MAX_WALL_DIST_MM:
                    best_seed = candidate
                    best_mode = f"wall-raycast dz={dz:+.0f}mm wd={wd:.1f}mm"
                    break
            if best_seed is not None:
                break

        if best_seed is None:
            # Fallback: place seed 10% of the way from the renal tip toward bif.
            # This is inside the correct side-branch lumen near its ostium.
            best_seed = p_tgt + LUMEN_FRAC * (bif - p_tgt)
            wd        = self._wall_dist(best_seed)
            best_mode = f"lumen-frac({LUMEN_FRAC}) wd={wd:.1f}mm"

        print(f"[PerBranchCL] branch_trunk_seed ({best_mode}): "
              f"{np.round(best_seed, 1)}  tgt={np.round(p_tgt, 1)}")
        return best_seed


# ═════════════════════════════════════════════════════════════════════════════
#  Phase 3 — tangent-preserving junction snap
# ═════════════════════════════════════════════════════════════════════════════

class JunctionSnap:
    BLEND_RADII = 2.0

    @staticmethod
    def _smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _branch_radius_at_start(pts, model_loc, n_rays=6):
        if len(pts) < 2:
            return 5.0
        pos  = pts[0]
        tang = pts[min(1, len(pts) - 1)] - pts[0]
        n = np.linalg.norm(tang)
        if n < 1e-6:
            return 5.0
        tang /= n
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(perp, tang)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, tang) * tang
        perp /= np.linalg.norm(perp)
        binorm = np.cross(tang, perp)
        hits = []
        for k in range(n_rays):
            angle = 2.0 * np.pi * k / n_rays
            rdir  = np.cos(angle) * perp + np.sin(angle) * binorm
            p1    = (pos + rdir * 0.1).tolist()
            p2    = (pos + rdir * 30.0).tolist()
            t     = vtk.mutable(0.0)
            hit_pt = [0.0] * 3
            ci = vtk.mutable(0)
            si = vtk.mutable(0)
            d2 = vtk.mutable(0.0)
            if model_loc.IntersectWithLine(p1, p2, 0.001, t, hit_pt, [0.] * 3, si, ci):
                d = np.linalg.norm(np.array(hit_pt) - pos)
                if d > 0.2:
                    hits.append(d)
        return float(np.median(hits)) if hits else 5.0

    def apply(self, branch_pts_list, bif_pt, model_node, junction_pts=None):
        """
        Snap each branch's start point to its junction anchor and blend.

        FIX (v290-PerBranchJunction): the previous code used a single bif_pt
        for ALL branches.  For renal/side branches whose pts[0] is on the IVC
        wall at Z≈1877–1887, blending toward bif_pt (Z=1803) over 17–23mm
        dragged the renal start 74–87mm downward — producing a kinked path.

        junction_pts : list of (3,) arrays, one per branch.
          - Iliacs: bif_pt  (the iliac bifurcation)
          - Renals: pts[0]  (already the IVC wall seed at the correct Z level;
                             snap is effectively a no-op, just pins index 0)
        If junction_pts is None, falls back to using bif_pt for all branches
        (original behaviour, safe for iliac-only cases).
        """
        pd  = model_node.GetPolyData()
        loc = vtk.vtkCellLocator()
        loc.SetDataSet(pd)
        loc.BuildLocator()

        snapped = []
        for bi, pts in enumerate(branch_pts_list):
            if len(pts) < 3:
                snapped.append(pts)
                continue

            jpt = (junction_pts[bi]
                   if junction_pts is not None and bi < len(junction_pts)
                   else bif_pt)

            r       = self._branch_radius_at_start(pts, loc)
            L       = max(self.BLEND_RADII * r, 3.0)
            arc     = _arc_length(pts)
            new_pts = pts.copy()
            new_pts[0] = jpt

            # FIX (v297-BifSnapTopo): start blend at pts[2], not pts[1].
            #
            # Background: both iliac branches share pts[0] = jpt (the exact
            # same float triple).  The topology graph snaps with a 2 mm grid
            # cell; any two points that fall in the same cell are merged into
            # ONE node.  Before this fix, pts[1] of each iliac branch was also
            # blended toward jpt (weight ≈ smoothstep(arc[1]/L) ≈ 0.95), so
            # pts[1]_left ≈ pts[1]_right ≈ jpt + tiny offset — well within
            # the same 2 mm cell.  The bif node's two iliac neighbours then
            # collapsed into ONE node, reducing the bif node to degree=2, so
            # _isCritical() returned False and the chain walk ran straight
            # through the iliac bifurcation — concatenating both iliac paths
            # into a single 341.5 mm phantom edge (nodes 211-335 in the log).
            #
            # Skipping pts[1] preserves each branch's second VMTK-computed
            # point (~8-15 mm into the iliac wall), which lands in a different
            # grid cell for each iliac -> bif node stays degree=3 -> topology
            # is recovered.  The visual gap between pts[0] and pts[2] is
            # imperceptible at 0.5 mm sampling density.
            for i in range(2, len(pts)):
                if arc[i] >= L:
                    break
                w = self._smoothstep(arc[i] / L)
                new_pts[i] = (1.0 - w) * jpt + w * pts[i]

            snapped.append(new_pts)
            print(f"[PerBranchCL] Branch {bi}: junction snap @ {np.round(jpt, 1)}, "
                  f"blend_L={L:.1f} mm (r={r:.1f} mm)")
        return snapped


# ═════════════════════════════════════════════════════════════════════════════
#  Phase 2 core — per-branch VMTK solves (OPTIMISED: shared preprocessed mesh)
# ═════════════════════════════════════════════════════════════════════════════

class PerBranchSolver:
    """
    Runs one independent VMTK solve per (p_src, p_tgt) pair.

    OPTIMISATION: accepts a preprocessed vtkPolyData so the mesh
    preprocessing step (the most expensive part) is NOT repeated per
    branch.  Each VMTK call uses the cached polydata directly.
    """

    def __init__(self, logic, model_node, preprocessed_pd=None):
        self.logic          = logic
        self.model_node     = model_node
        self.preprocessed_pd = preprocessed_pd  # None → fall back to normal path

    # ------------------------------------------------------------------
    def solve_one(self, p_src, p_tgt, bi=0):
        """
        Solve a SINGLE branch (p_src -> p_tgt) and return its point array.

        FIX (v287-EpPolyData): The root cause of all 'SetInputData argument 1'
        errors is that this VMTK build's extractCenterline() expects:
            extractCenterline(vtkPolyData_mesh, vtkPolyData_seeds, ...)
        NOT a vtkMRMLMarkupsFiducialNode as the second argument.
        _vmtk_extract() now converts the endpoints node to a vtkPolyData of
        seed points before calling VMTK.
        """
        print(f"[PerBranchCL] Phase 2: branch {bi}  "
              f"src={np.round(p_src, 1)}  tgt={np.round(p_tgt, 1)} ...")

        ep_node = self._make_endpoints([p_src, p_tgt], f"_pb_ep_{bi}")
        cl_node = None
        try:
            cl_node = self._vmtk_extract(ep_node)
        except Exception as e:
            print(f"[PerBranchCL]   branch {bi} VMTK failed: {e}")
            cl_node = None
        finally:
            slicer.mrmlScene.RemoveNode(ep_node)

        if cl_node is None:
            raise RuntimeError(
                f"[PerBranchCL] branch {bi}: VMTK extraction failed "
                f"(src={np.round(p_src,1)}, tgt={np.round(p_tgt,1)}). "
                f"See console above for the underlying error."
            )

        pts = _curve_to_array(cl_node)
        slicer.mrmlScene.RemoveNode(cl_node)
        pts = _resample_uniform(pts, step_mm=0.5)
        print(f"[PerBranchCL]   branch {bi}: {len(pts)} pts, "
              f"arc={_arc_length(pts)[-1]:.1f} mm")
        return pts

    # ------------------------------------------------------------------
    def _vmtk_extract(self, ep_node):
        """
        Call VMTK extractCenterline with the correct argument types.

        Different Slicer/VMTK builds expect different second-argument types:
          Old builds: extractCenterline(vtkPolyData, vtkMRMLMarkupsFiducialNode)
          New builds: extractCenterline(vtkPolyData, vtkPolyData_seeds)

        We probe both.  MRML-node form first (this build expects it — the
        error 'PolyData has no attr GetNumberOfControlPoints' means the
        internal code called ep_arg.GetNumberOfControlPoints(), so the build
        that raised it expects an MRML node, not a PolyData).
        """
        import ExtractCenterline
        extract_logic = ExtractCenterline.ExtractCenterlineLogic()

        # ── Mesh: use cached preprocessed polydata if available ──────────────
        mesh_pd = (self.preprocessed_pd
                   if self.preprocessed_pd is not None
                   else self.model_node.GetPolyData())

        # ── Build seed vtkPolyData from ep_node (needed for PolyData form) ───
        seed_pd = vtk.vtkPolyData()
        seed_pts = vtk.vtkPoints()
        for i in range(ep_node.GetNumberOfControlPoints()):
            p = [0.0, 0.0, 0.0]
            ep_node.GetNthControlPointPositionWorld(i, p)
            seed_pts.InsertNextPoint(p[0], p[1], p[2])
        seed_pd.SetPoints(seed_pts)

        # ── Probe all known signatures ───────────────────────────────────────
        def _run():
            errors = []
            # Form A — MRML fiducial node (older builds)
            for kw in [{"curveSamplingDistance": 0.5}, {}]:
                try:
                    r = extract_logic.extractCenterline(mesh_pd, ep_node, **kw)
                    print(f"[PerBranchCL] _vmtk_extract: MRML-node form ok kw={kw}")
                    return r
                except (TypeError, AttributeError) as e:
                    errors.append(f"MRML{kw}: {e}")
            # Form B — vtkPolyData seeds (newer builds)
            for kw in [{"curveSamplingDistance": 0.5}, {}]:
                try:
                    r = extract_logic.extractCenterline(mesh_pd, seed_pd, **kw)
                    print(f"[PerBranchCL] _vmtk_extract: PolyData form ok kw={kw}")
                    return r
                except (TypeError, AttributeError) as e:
                    errors.append(f"PD{kw}: {e}")
            # Form C — positional fallback
            try:
                r = extract_logic.extractCenterline(mesh_pd, seed_pd, 0.5)
                print("[PerBranchCL] _vmtk_extract: positional fallback ok")
                return r
            except Exception as e:
                errors.append(f"pos: {e}")
            raise RuntimeError(
                "[PerBranchCL] All extractCenterline probes failed:\n"
                + "\n".join(f"  {e}" for e in errors)
            )

        result = _run()

        # Handle (polydata, voronoi) tuple return
        if isinstance(result, tuple):
            cl_pd = result[0]
        else:
            cl_pd = result

        if cl_pd is None or not hasattr(cl_pd, 'GetPoints') or cl_pd.GetNumberOfPoints() == 0:
            return None

        # Convert polydata to curve node
        cl_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", "_pb_branch_cl")
        cl_node.CreateDefaultDisplayNodes()
        cl_node.GetDisplayNode().SetVisibility(False)
        pts_vtk = cl_pd.GetPoints()
        lines = cl_pd.GetLines()
        if lines and lines.GetNumberOfCells() > 0:
            # Read first cell (the centerline path for a 2-endpoint solve)
            lines.InitTraversal()
            id_list = vtk.vtkIdList()
            lines.GetNextCell(id_list)
            for k in range(id_list.GetNumberOfIds()):
                p = pts_vtk.GetPoint(id_list.GetId(k))
                cl_node.AddControlPoint(vtk.vtkVector3d(p[0], p[1], p[2]))
        else:
            # Fallback: all points in order
            for k in range(pts_vtk.GetNumberOfPoints()):
                p = pts_vtk.GetPoint(k)
                cl_node.AddControlPoint(vtk.vtkVector3d(p[0], p[1], p[2]))

        if cl_node.GetNumberOfControlPoints() == 0:
            slicer.mrmlScene.RemoveNode(cl_node)
            return None
        return cl_node

    # ------------------------------------------------------------------
    def solve(self, p_src, ring_centroids):
        """Convenience wrapper: solve all branches and return list of arrays."""
        return [self.solve_one(p_src, p_tgt, bi)
                for bi, p_tgt in enumerate(ring_centroids)]

    def solve_per_branch(self, p_srcs, p_tgts):
        """
        Solve each branch with its OWN proximal seed.

        FIX (crossing branches): uses per-branch seeds (from
        BranchSeedGenerator.per_branch_seeds) instead of a single shared
        p_src.  Each seed is laterally offset toward its branch's distal
        target so VMTK starts on the correct side of the bifurcation.
        """
        return [self.solve_one(p_src, p_tgt, bi)
                for bi, (p_src, p_tgt) in enumerate(zip(p_srcs, p_tgts))]

    # ------------------------------------------------------------------
    def _extract_with_cached_mesh(self, preprocessed_pd, endpoints_node):
        """
        Run VMTK on an already-preprocessed polydata, avoiding an extra
        preprocess() call inside VesselAnalyzerLogic.extractCenterline().

        SNAPPING STRATEGY (v279-SnapRadiusFix):
        ─────────────────────────────────────────
        The proximal seed (index 0) has already been laterally nudged by
        per_branch_seeds() to sit on the correct anatomical side of the
        bifurcation wall.  A global vtkPointLocator.FindClosestPoint() snap
        of that seed is DANGEROUS: at the bifurcation level the two iliac
        walls are only ~15-20 mm apart, so the nearest mesh vertex to the
        nudged seed can land on the OPPOSITE wall, undoing the lateral bias
        and causing the geodesics to cross.

        Fix: snap each endpoint only if a mesh vertex exists within a tight
        radius (SNAP_RADIUS_MM).  If the closest vertex is farther than that
        threshold, pass the raw position directly — VMTK's internal Dijkstra
        seed resolution is robust to interior lumen points and will find the
        correct graph node without lateral ambiguity.

        The distal endpoint (index 1) is a ring centroid far from the
        bifurcation, so global snapping is safe there; the radius guard
        merely prevents snapping to a bogus vertex if the ring centroid
        happens to be slightly outside the decimated mesh boundary.
        """
        import ExtractCenterline
        extract_logic = ExtractCenterline.ExtractCenterlineLogic()

        # ── Radius-guarded snap ───────────────────────────────────────────────
        # Only snap an endpoint when the nearest mesh vertex is within
        # SNAP_RADIUS_MM.  This preserves the lateral XY bias of the proximal
        # seed while still correcting genuinely off-surface distal centroids.
        SNAP_RADIUS_MM = 8.0   # tighter than a typical iliac radius (~4-6 mm)

        pt_locator = vtk.vtkPointLocator()
        pt_locator.SetDataSet(preprocessed_pd)
        pt_locator.BuildLocator()

        snapped_ep_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", endpoints_node.GetName() + "_snapped")
        snapped_ep_node.GetDisplayNode().SetVisibility(False)

        for i in range(endpoints_node.GetNumberOfControlPoints()):
            p = [0.0, 0.0, 0.0]
            endpoints_node.GetNthControlPointPositionWorld(i, p)
            vid = pt_locator.FindClosestPoint(p)
            snapped = preprocessed_pd.GetPoint(vid)
            dist = float(np.linalg.norm(np.array(snapped) - np.array(p)))
            if dist <= SNAP_RADIUS_MM:
                snapped_ep_node.AddFiducial(snapped[0], snapped[1], snapped[2])
                print(f"[PerBranchCL]   ep[{i}] snap  dist={dist:.1f} mm → snapped  {np.round(snapped, 1)}")
            else:
                # Raw position is too far from any mesh vertex — pass through
                # unchanged so the lateral bias is preserved.
                snapped_ep_node.AddFiducial(p[0], p[1], p[2])
                print(f"[PerBranchCL]   ep[{i}] snap  dist={dist:.1f} mm → pass-through  {np.round(p, 1)}")

        def _run(ep):
            try:
                return extract_logic.extractCenterline(
                    preprocessed_pd, ep, curveSamplingDistance=0.5)
            except TypeError:
                try:
                    return extract_logic.extractCenterline(preprocessed_pd, ep, 0.5)
                except TypeError:
                    return extract_logic.extractCenterline(preprocessed_pd, ep)

        cl_pd, _ = _run(snapped_ep_node)
        slicer.mrmlScene.RemoveNode(snapped_ep_node)

        if cl_pd is None or cl_pd.GetNumberOfPoints() == 0:
            return None

        cl_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", f"_pb_branch_cl")
        cl_node.CreateDefaultDisplayNodes()
        cl_node.GetDisplayNode().SetVisibility(False)
        pts_vtk = cl_pd.GetPoints()
        for i in range(pts_vtk.GetNumberOfPoints()):
            p = pts_vtk.GetPoint(i)
            cl_node.AddControlPoint(vtk.vtkVector3d(p[0], p[1], p[2]))
        return cl_node

    # ------------------------------------------------------------------
    @staticmethod
    def _make_endpoints(positions, name):
        node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", name)
        node.GetDisplayNode().SetVisibility(False)
        for pos in positions:
            node.AddControlPoint(vtk.vtkVector3d(float(pos[0]), float(pos[1]), float(pos[2])))
        return node

    # ------------------------------------------------------------------
    @staticmethod
    def _straight_fallback(p_src, p_tgt, step_mm=0.5):
        d = np.linalg.norm(p_tgt - p_src)
        n = max(int(d / step_mm), 2)
        t = np.linspace(0.0, 1.0, n)
        return np.outer(1.0 - t, p_src) + np.outer(t, p_tgt)


# ═════════════════════════════════════════════════════════════════════════════
#  Phase 4 — light refinement (optional, disabled by default for speed)
# ═════════════════════════════════════════════════════════════════════════════

class LightRefiner:
    """
    Calls _refineSingleCurve() on each branch with conservative parameters.

    NOTE: This phase is disabled by default (refine=False in the pipeline)
    because it adds ~8 gradient-descent iterations per branch on top of an
    already high-quality VMTK result.  Enable only when extra accuracy at
    the junction is needed.
    """

    REFINE_PARAMS = {
        "n_iters"  : 8,
        "lr"       : 0.08,
        "lr_decay" : 1.0,
        "w_smooth" : 0.5,
        "w_center" : 1.0,
        "w_j"      : 0.0,
    }

    def __init__(self, widget, bif_pt):
        self.widget  = widget
        self.bif_pt  = bif_pt

    def refine(self, branch_curve_nodes):
        for bi, node in enumerate(branch_curve_nodes):
            if node.GetNumberOfControlPoints() < 4:
                continue
            print(f"[PerBranchCL] Phase 4: refining branch {bi} …")
            try:
                self.widget._refineSingleCurve(node, params=self.REFINE_PARAMS)
                self._pin_first_point(node)
            except Exception as e:
                print(f"[PerBranchCL]   ⚠  refine branch {bi}: {e}")

    def _pin_first_point(self, node):
        p = self.bif_pt
        node.SetNthControlPointPosition(0, float(p[0]), float(p[1]), float(p[2]))


# ═════════════════════════════════════════════════════════════════════════════
#  Merge: combine per-branch curves into a single merged CL node
# ═════════════════════════════════════════════════════════════════════════════

class BranchMerger:
    TRUNK_STUB_STEP_MM = 0.5

    def __init__(self, bif_pt, trunk_dir, trunk_radius):
        self.bif_pt      = bif_pt
        self.trunk_dir   = trunk_dir
        self.trunk_radius = trunk_radius

    def merge(self, branch_arrays, p_root, node_name="PerBranchCenterline",
              trunk_pts_override=None):
        # FIX (v284-VMTKTrunk): use the VMTK-computed trunk path when available.
        # trunk_pts_override is the pre-trim prefix from the first branch solve
        # (inlet → bifurcation, following the true medial axis).  Falling back
        # to a straight-line interpolation only when no VMTK path was captured.
        if trunk_pts_override is not None and len(trunk_pts_override) > 1:
            trunk_pts = _resample_uniform(trunk_pts_override,
                                          step_mm=self.TRUNK_STUB_STEP_MM)
            print(f"[PerBranchCL] Using VMTK trunk path ({len(trunk_pts)} pts, "
                  f"arc={_arc_length(trunk_pts)[-1]:.1f} mm)")
        else:
            trunk_pts = _resample_uniform(
                np.array([p_root, self.bif_pt]), step_mm=self.TRUNK_STUB_STEP_MM)
            print(f"[PerBranchCL] Using straight-line trunk fallback "
                  f"({len(trunk_pts)} pts)")

        # FIX (v285-MultiCell): output a vtkMRMLModelNode with one VTK polyline
        # cell per segment (trunk + each branch) instead of a flat concatenated
        # MarkupsCurveNode.  loadCenterline's graph-topology detector reads ALL
        # line cells independently, so each branch becomes a separate segment.
        # The old approach vstacked all arrays into one MarkupsCurveNode which
        # loadCenterline read as a single continuous polyline — branches were not
        # independent segments, causing the left iliac (and any non-first branch)
        # to be drawn as a zig-zag connector between branch endpoints rather than
        # as its own anatomically correct path.
        segments = [trunk_pts] + [arr for arr in branch_arrays if len(arr) > 1]

        vtkpts = vtk.vtkPoints()
        lines  = vtk.vtkCellArray()
        total_pts = 0

        for seg in segments:
            cell = vtk.vtkPolyLine()
            cell.GetPointIds().SetNumberOfIds(len(seg))
            for k, p in enumerate(seg):
                pid = vtkpts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
                cell.GetPointIds().SetId(k, pid)
            lines.InsertNextCell(cell)
            total_pts += len(seg)

        pd = vtk.vtkPolyData()
        pd.SetPoints(vtkpts)
        pd.SetLines(lines)

        merged_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode",
                                                         node_name)
        merged_node.SetAndObservePolyData(pd)
        merged_node.CreateDefaultDisplayNodes()
        merged_node.GetDisplayNode().SetVisibility(False)

        print(f"[PerBranchCL] Merged CL: {total_pts} pts total  "
              f"(trunk_stub={len(trunk_pts)}, "
              f"branches={[len(a) for a in branch_arrays if len(a) > 1]}) "
              f"→ {len(segments)} line cells")
        return merged_node


# ═════════════════════════════════════════════════════════════════════════════
#  Top-level orchestrator
# ═════════════════════════════════════════════════════════════════════════════

class PerBranchCenterlinePipeline:
    """
    Orchestrates all four phases.

    Key performance change vs v275 original:
      • Mesh preprocessing runs ONCE and is shared across Phase 1 and Phase 2.
      • refine=False by default — eliminates Phase 4 gradient descent.

    Parameters
    ----------
    logic      : VesselAnalyzerLogic instance
    widget     : VesselAnalyzerWidget instance (needed for _refineSingleCurve)
    model_node : vtkMRMLModelNode  — vessel surface
    refine     : bool  — run Phase 4 light refinement (default False for speed)
    """

    def __init__(self, logic, widget, model_node, refine=False):
        self.logic      = logic
        self.widget     = widget
        self.model_node = model_node
        self.do_refine  = refine

    # ------------------------------------------------------------------
    def run(self, endpoints_node):
        ring_centroids = self._read_centroids(endpoints_node)
        if len(ring_centroids) < 2:
            raise RuntimeError(
                f"[PerBranchCL] Need ≥2 endpoints; got {len(ring_centroids)}")

        # ── Build shared preprocessed mesh (runs ONCE) ────────────────
        print("[PerBranchCL] Building shared preprocessed mesh …")
        mesh_cache = _PreprocessCache(
            model_node=self.model_node,
            target_points=8000,
            decimation_aggressiveness=3.5,
            subdivide=True,
        )
        _cached = mesh_cache.get()
        if _cached is None or _cached.GetNumberOfPoints() == 0:
            raise RuntimeError(
                "[PerBranchCL] Mesh preprocessing returned an empty or null "
                "vtkPolyData — check that the vessel model node contains valid "
                "polydata before calling PerBranchCenterlinePipeline.run()."
            )

        # ── Phase 1: bif geometry ─────────────────────────────────────
        bif_geom = BifurcationGeometry(
            logic=self.logic,
            model_node=self.model_node,
            endpoints_node=endpoints_node,
            preprocess_cache=mesh_cache,
        ).compute()

        # ── Phase 2: trunk-inlet → branch-outlet solves ───────────────
        #
        # ARCHITECTURAL FIX (v282-TrunkInletSolve):
        # All previous attempts to fix crossing by nudging the proximal seed
        # failed because VMTK's extractCenterline() always resolves the seed
        # to the nearest Dijkstra graph node on the mesh surface, and at the
        # bifurcation level that graph node can be on the wrong iliac wall.
        #
        # Root cause: the bifurcation zone is the shared "fan" of the Dijkstra
        # graph — any seed near the midline will produce crossing geodesics.
        #
        # Fix: use the trunk INLET ring centroid (highest-Z endpoint) as the
        # proximal seed for every branch solve.  The trunk inlet is far from
        # the bifurcation (209mm), unambiguous, and on the boundary of the
        # mesh.  Each branch solve then runs trunk-inlet → branch-outlet,
        # giving VMTK two well-separated, on-boundary endpoints with no
        # ambiguity.  The resulting full-path centerlines DO cross (they share
        # a trunk segment), but we trim the shared trunk off each branch by
        # finding where each path diverges from the others.
        #
        # Trunk-segment trim: we find the bifurcation point on each raw
        # centerline as the point of minimum distance to the estimated
        # snapped_bif_pt, then discard everything proximal to that point.
        seed_gen     = BranchSeedGenerator(bif_geom, self.model_node)
        z_vals       = [c[2] for c in ring_centroids]
        trunk_idx    = int(np.argmax(z_vals))
        p_trunk      = ring_centroids[trunk_idx]
        p_tgts       = [c for i, c in enumerate(ring_centroids) if i != trunk_idx]
        p_src        = seed_gen.proximal_seed()   # medial-axis stub root for merge

        dist_to_tgt = np.linalg.norm(p_trunk - p_src)
        print(f"[PerBranchCL] Trunk inlet @ {np.round(p_trunk, 1)}, "
              f"dist_to_p_src={dist_to_tgt:.1f} mm")
        print(f"[PerBranchCL] Branch distal seeds ({len(p_tgts)}): "
              + ", ".join(f"c{i}={np.round(c, 1)}" for i, c in enumerate(p_tgts)))

        solver = PerBranchSolver(
            self.logic, self.model_node,
            preprocessed_pd=mesh_cache.get(),
        )

        # Solve trunk-inlet → each branch outlet
        branch_arrays  = []
        junction_pts   = []

        # Solve trunk first so it's available for renal junction snapping
        trunk_pts_vmtk = solver.solve_one(p_trunk, bif_geom.snapped_bif_pt, -1)
        print(f"[PerBranchCL]   trunk pre-solve: {len(trunk_pts_vmtk)} pts, "
              f"arc={_arc_length(trunk_pts_vmtk)[-1]:.1f} mm")

        # Bifurcation Z — targets below this are iliac (need tip-inward solve)
        bif_z = bif_geom.snapped_bif_pt[2]
        ILIAC_Z_THRESH = bif_z - 20.0

        for bi, p_tgt in enumerate(p_tgts):
            is_iliac = p_tgt[2] < ILIAC_Z_THRESH

            if is_iliac:
                bif_seed = seed_gen.iliac_bif_seed(p_tgt)
                pts_raw  = solver.solve_one(p_tgt, bif_seed, bi)
                pts      = pts_raw[::-1].copy()
                bif_idx    = self._find_divergence_idx(
                    pts, bif_geom.snapped_bif_pt, branch_arrays,
                    snap_mm=2.0)
                pts_branch = pts[bif_idx:]
                bif_dist   = np.linalg.norm(pts[bif_idx] - bif_geom.snapped_bif_pt)
                print(f"[PerBranchCL]   branch {bi}: tip-inward (reversed), "
                      f"diverge_idx={bif_idx} "
                      f"(bif_dist={bif_dist:.1f} mm), "
                      f"branch arc={_arc_length(pts_branch)[-1]:.1f} mm")
                junction_pts.append(bif_geom.snapped_bif_pt.copy())
            else:
                # FIX (v294-RenalViaInlet): target p_trunk (IVC inlet) not bif_pt.
                # Targeting bif_pt routed through the adjacent iliac lumen.
                pts_raw  = solver.solve_one(p_tgt, p_trunk, bi)
                pts      = pts_raw[::-1].copy()
                bif_idx  = self._find_divergence_idx(
                    pts, bif_geom.snapped_bif_pt, branch_arrays,
                    snap_mm=4.0, min_idx=1)
                pts_branch = pts[bif_idx:]
                bif_dist   = float(np.linalg.norm(
                    pts[bif_idx] - bif_geom.snapped_bif_pt))
                print(f"[PerBranchCL]   branch {bi}: renal via-inlet (reversed), "
                      f"diverge_idx={bif_idx} "
                      f"(bif_dist={bif_dist:.1f} mm), "
                      f"branch arc={_arc_length(pts_branch)[-1]:.1f} mm")
                from scipy.spatial import cKDTree as _KD
                _tkd = _KD(trunk_pts_vmtk)
                _d, _ti = _tkd.query(pts_branch[0])
                _jpt = trunk_pts_vmtk[_ti].copy()
                print(f"[PerBranchCL]   branch {bi}: renal junction snapped "
                      f"to trunk pt (d={_d:.2f} mm) @ {np.round(_jpt, 1)}")
                junction_pts.append(_jpt)

            branch_arrays.append(pts_branch)
        snapper       = JunctionSnap()
        branch_arrays = snapper.apply(
            branch_arrays, bif_geom.snapped_bif_pt, self.model_node,
            junction_pts=junction_pts)

        # ── Phase 4: light refinement (optional, off by default) ───────
        if self.do_refine:
            branch_nodes = []
            for bi, arr in enumerate(branch_arrays):
                n = _array_to_curve(arr, f"_pb_refine_{bi}")
                branch_nodes.append(n)
            refiner = LightRefiner(self.widget, bif_geom.snapped_bif_pt)
            refiner.refine(branch_nodes)
            branch_arrays = [_curve_to_array(n) for n in branch_nodes]
            for n in branch_nodes:
                slicer.mrmlScene.RemoveNode(n)

        # ── Merge ──────────────────────────────────────────────────────
        merger = BranchMerger(
            bif_geom.snapped_bif_pt,
            bif_geom.trunk_dir,
            bif_geom.trunk_radius,
        )
        # FIX (v284): pass p_trunk + VMTK trunk geometry to merger
        merged_node = merger.merge(
            branch_arrays, p_trunk, trunk_pts_override=trunk_pts_vmtk)

        print("[PerBranchCL] Pipeline complete ✓")
        return merged_node

    # ------------------------------------------------------------------
    def run_with_progress(self, endpoints_node,
                          status_cb=None, process_events_cb=None):
        """
        Trunk-inlet crossing fix (v282) exposed with UI progress callbacks.

        Identical logic to run() but calls:
          status_cb(msg, color=None)  — forwarded to the widget status label
          process_events_cb()         — called after each heavy phase so Qt
                                        can repaint without background threads
        (Both are optional; omitting them makes this behave like run().)

        Root cause of crossing (all prior v276-v281 attempts):
        ────────────────────────────────────────────────────────
        Per-branch proximal seeds generated near the bifurcation level all
        collapsed to the same Dijkstra graph node on the preprocessed mesh —
        the bifurcation wall mesh is sparse enough that two laterally-nudged
        interior seeds snap to the same surface vertex, giving VMTK identical
        start nodes and producing crossing geodesics.

        Fix: use the trunk INLET ring centroid (highest-Z endpoint, 209 mm
        from the bifurcation) as the proximal seed for EVERY branch solve.
        The inlet is far from the bifurcation, unambiguously on-boundary, and
        unique — each solve runs trunk-inlet → branch-outlet so VMTK always
        finds the full-arc path through the correct branch wall.  The shared
        trunk prefix is trimmed per-branch by finding the point on each
        full-arc path that is nearest to snapped_bif_pt.
        """
        def _status(msg, color=None):
            print(f"[PerBranchCL] {msg}")
            if status_cb:
                try:
                    status_cb(msg, color)
                except TypeError:
                    status_cb(msg)

        def _tick():
            if process_events_cb:
                process_events_cb()

        ring_centroids = self._read_centroids(endpoints_node)
        if len(ring_centroids) < 2:
            raise RuntimeError(
                f"[PerBranchCL] Need ≥2 endpoints; got {len(ring_centroids)}")

        # ── Phase 1a: shared preprocessed mesh ───────────────────────
        _status("Step 1/4: Preprocessing mesh (shared)...")
        mesh_cache = _PreprocessCache(
            model_node=self.model_node,
            target_points=8000,
            decimation_aggressiveness=3.5,
            subdivide=True,
        )
        _cached = mesh_cache.get()
        if _cached is None or _cached.GetNumberOfPoints() == 0:
            raise RuntimeError(
                "[PerBranchCL] Mesh preprocessing returned an empty or null "
                "vtkPolyData — check that the vessel model node contains valid "
                "polydata before calling run_with_progress()."
            )
        _tick()

        # ── Phase 1b: bifurcation geometry (instant, no VMTK solve) ──
        _status("Step 1/4: Locating bifurcation...")
        bif_geom = BifurcationGeometry(
            logic=self.logic,
            model_node=self.model_node,
            endpoints_node=endpoints_node,
            preprocess_cache=mesh_cache,
        ).compute()
        _tick()

        # ── Phase 2: trunk-inlet → per-branch-outlet VMTK solves ─────
        seed_gen  = BranchSeedGenerator(bif_geom, self.model_node)
        z_vals    = [c[2] for c in ring_centroids]
        trunk_idx = int(np.argmax(z_vals))
        p_trunk   = ring_centroids[trunk_idx]
        p_tgts    = [c for i, c in enumerate(ring_centroids) if i != trunk_idx]
        p_src     = seed_gen.proximal_seed()   # medial-axis stub root for merger

        dist_to_tgt = np.linalg.norm(p_trunk - p_src)
        print(f"[PerBranchCL] Trunk inlet @ {np.round(p_trunk, 1)}, "
              f"dist_to_p_src={dist_to_tgt:.1f} mm")
        print(f"[PerBranchCL] Branch distal seeds ({len(p_tgts)}): "
              + ", ".join(f"c{i}={np.round(c, 1)}"
                          for i, c in enumerate(p_tgts)))

        solver = PerBranchSolver(
            self.logic, self.model_node,
            preprocessed_pd=mesh_cache.get(),
        )

        branch_arrays  = []
        iliac_arrays   = []   # only iliac pts_branch arrays — used for renal divergence
        junction_pts   = []   # per-branch junction anchor for JunctionSnap

        # ── Solve trunk FIRST so it's available for renal junction snapping ──
        # trunk_pts_vmtk runs p_trunk → bif_pt along the IVC medial axis.
        # Solving it up front lets the renal junction-snap code below anchor
        # each renal start to the nearest point on the real trunk CL.
        print(f"[PerBranchCL] Pre-solving trunk: {np.round(p_trunk,1)} → "
              f"{np.round(bif_geom.snapped_bif_pt,1)}")
        trunk_pts_vmtk = solver.solve_one(p_trunk, bif_geom.snapped_bif_pt, -1)
        print(f"[PerBranchCL]   trunk: {len(trunk_pts_vmtk)} pts, "
              f"arc={_arc_length(trunk_pts_vmtk)[-1]:.1f} mm")

        # Bifurcation Z — targets below this are iliac (need tip-inward solve)
        bif_z = bif_geom.snapped_bif_pt[2]
        ILIAC_Z_THRESH = bif_z - 20.0   # >20 mm below bif = iliac

        for bi, p_tgt in enumerate(p_tgts):
            _status(f"Step 2/4: Branch {bi + 1} of {len(p_tgts)}...")

            is_iliac = p_tgt[2] < ILIAC_Z_THRESH

            if is_iliac:
                # FIX (v286-TipInward): solve tip → bif-wall seed to avoid the
                # shared Dijkstra fan at the carina.  Each iliac gets its own
                # unambiguous wall seed on the correct side; the result is then
                # reversed so the array runs bif→tip like all other branches.
                bif_seed = seed_gen.iliac_bif_seed(p_tgt)
                pts_raw  = solver.solve_one(p_tgt, bif_seed, bi)
                # Reverse so index 0 is near the bifurcation
                pts      = pts_raw[::-1].copy()
                # FIX (v288-DivergenceTrim): use divergence criterion.
                bif_idx    = self._find_divergence_idx(
                    pts, bif_geom.snapped_bif_pt, branch_arrays,
                    snap_mm=2.0)
                pts_branch = pts[bif_idx:]
                bif_dist   = np.linalg.norm(pts[bif_idx] - bif_geom.snapped_bif_pt)
                print(f"[PerBranchCL]   branch {bi}: tip-inward solve "
                      f"(reversed), arc={_arc_length(pts)[-1]:.1f} mm")
                print(f"[PerBranchCL]   branch {bi}: diverge_idx={bif_idx} "
                      f"(bif_dist={bif_dist:.1f} mm), "
                      f"branch arc={_arc_length(pts_branch)[-1]:.1f} mm")
                junction_pts.append(bif_geom.snapped_bif_pt.copy())
                iliac_arrays.append(pts_branch)
            else:
                # route through the adjacent iliac lumen (shorter Dijkstra path
                # than going up through the IVC trunk), producing a cell that
                # overlapped the iliac cells and fused them into one 343mm edge.
                #
                # Targeting p_trunk (IVC inlet, ~168mm above renals) forces
                # VMTK to route: renal_tip → ostium → IVC trunk → inlet.
                # _find_divergence_idx then trims the IVC trunk prefix leaving
                # only the renal arc from its ostium to its tip.
                pts_raw  = solver.solve_one(p_tgt, p_trunk, bi)
                pts      = pts_raw[::-1].copy()   # reverse: index 0 = inlet end
                # FIX (v297-RenalDivergeTrunkOnly): compare against iliac arrays
                # only, not all prior branches.  branch_arrays at this point also
                # contains the other renal's pts_branch, which starts at the renal
                # ostium on the IVC trunk.  That trunk-adjacent segment overlaps
                # this renal's IVC prefix for ~60 mm, falsely pushing diverge_idx
                # to ~120/244.  The iliacs don't overlap the IVC trunk, so comparing
                # against iliac_arrays gives a clean "where does this renal leave
                # the trunk?" answer.  If no iliacs have been solved yet (first
                # renal), falls through to the bif_pt fallback in _find_divergence_idx.
                bif_idx  = self._find_divergence_idx(
                    pts, bif_geom.snapped_bif_pt, iliac_arrays,
                    snap_mm=4.0, min_idx=1)
                pts_branch = pts[bif_idx:]
                bif_dist   = float(np.linalg.norm(
                    pts[bif_idx] - bif_geom.snapped_bif_pt))
                print(f"[PerBranchCL]   branch {bi}: renal via-inlet solve "
                      f"(reversed), arc={_arc_length(pts)[-1]:.1f} mm")
                print(f"[PerBranchCL]   branch {bi}: diverge_idx={bif_idx} "
                      f"(bif_dist={bif_dist:.1f} mm), "
                      f"branch arc={_arc_length(pts_branch)[-1]:.1f} mm")
                # Snap junction anchor to nearest trunk CL point so merged
                # polydata has a <0.5mm shared node (graph connectivity fix).
                from scipy.spatial import cKDTree as _KD
                _tkd = _KD(trunk_pts_vmtk)
                _d, _ti = _tkd.query(pts_branch[0])
                _jpt = trunk_pts_vmtk[_ti].copy()
                print(f"[PerBranchCL]   branch {bi}: renal junction snapped "
                      f"to trunk pt (d={_d:.2f} mm) @ {np.round(_jpt, 1)}")
                junction_pts.append(_jpt)

            branch_arrays.append(pts_branch)
            _tick()

        # ── Phase 3: junction snap ────────────────────────────────────
        _status("Step 3/4: Junction snap...")
        snapper       = JunctionSnap()
        branch_arrays = snapper.apply(
            branch_arrays, bif_geom.snapped_bif_pt, self.model_node,
            junction_pts=junction_pts)
        _tick()

        # ── Phase 4: optional light refinement ───────────────────────
        if self.do_refine:
            branch_nodes = []
            for bi, arr in enumerate(branch_arrays):
                n = _array_to_curve(arr, f"_pb_refine_{bi}")
                branch_nodes.append(n)
            refiner = LightRefiner(self.widget, bif_geom.snapped_bif_pt)
            refiner.refine(branch_nodes)
            branch_arrays = [_curve_to_array(n) for n in branch_nodes]
            for n in branch_nodes:
                import slicer as _slicer
                _slicer.mrmlScene.RemoveNode(n)

        # ── Merge ─────────────────────────────────────────────────────
        _status("Step 4/4: Merging...")
        merger = BranchMerger(
            bif_geom.snapped_bif_pt,
            bif_geom.trunk_dir,
            bif_geom.trunk_radius,
        )
        # FIX (v284): pass p_trunk + VMTK trunk geometry to merger
        merged_node = merger.merge(
            branch_arrays, p_trunk, trunk_pts_override=trunk_pts_vmtk)
        _tick()

        print("[PerBranchCL] Pipeline complete ✓")
        return merged_node

    # ------------------------------------------------------------------
    @staticmethod
    def _find_divergence_idx(pts, bif_pt, other_branch_arrays,
                             snap_mm=4.0, min_idx=1):
        """
        Find the index in `pts` where this branch diverges from all others.

        FIX (v288-DivergenceTrim): the previous trim used argmin(dist to
        snapped_bif_pt), which fails when snapped_bif_pt is slightly off —
        the nearest point on the path may be well into the branch lumen or
        still in the shared trunk, producing either too-short or too-long
        branch arrays.

        New criterion: walk pts from index 0 (bifurcation end) forward and
        find the first point that is NOT shared (within snap_mm) by any other
        branch path.  That is the true start of the unique branch geometry.

        If no other branch arrays are available yet (first branch), fall back
        to the nearest-to-bif_pt criterion but clamp to at least min_idx.

        Parameters
        ----------
        pts               : (N,3) array, index 0 = bifurcation end
        bif_pt            : (3,) bifurcation coordinate
        other_branch_arrays : list of (M,3) arrays of already-solved branches
        snap_mm           : distance threshold for "shared" point
        min_idx           : minimum trim index (never trim to index 0)
        """
        from scipy.spatial import cKDTree

        if other_branch_arrays:
            # Build a combined KD-tree of all already-solved branch points.
            all_other = np.vstack(other_branch_arrays)
            kd = cKDTree(all_other)
            for i, pt in enumerate(pts):
                dist, _ = kd.query(pt)
                if dist > snap_mm:
                    # First unique point found — trim here.
                    return max(i, min_idx)
            # All points shared — very unusual; fall back to bif trim.

        # Fallback: nearest point to bif_pt.
        dists = np.linalg.norm(pts - bif_pt, axis=1)
        return max(int(np.argmin(dists)), min_idx)

    # ------------------------------------------------------------------
    @staticmethod
    def _read_centroids(endpoints_node):
        n   = endpoints_node.GetNumberOfControlPoints()
        out = []
        for i in range(n):
            p = [0.0, 0.0, 0.0]
            endpoints_node.GetNthControlPointPosition(i, p)
            out.append(np.array(p))
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _assign_distal_seeds(ring_centroids, p_src):
        # Identify the trunk inlet ring as the one with the highest Z coordinate.
        # This matches the criterion used in BifurcationGeometry.compute() and is
        # robust regardless of where p_src sits.  The previous nearest-to-p_src
        # heuristic broke when p_src moved close to the bifurcation (low Z), at
        # which point the nearest centroid was an iliac outlet, not the IVC inlet.
        z_vals    = [c[2] for c in ring_centroids]
        trunk_idx = int(np.argmax(z_vals))
        p_tgts    = [c for i, c in enumerate(ring_centroids) if i != trunk_idx]

        dist_to_src = np.linalg.norm(ring_centroids[trunk_idx] - p_src)
        print(f"[PerBranchCL] Trunk inlet ring: centroid {trunk_idx} "
              f"@ {np.round(ring_centroids[trunk_idx], 1)}, "
              f"dist_to_src={dist_to_src:.1f} mm")
        print(f"[PerBranchCL] Branch distal seeds ({len(p_tgts)}): "
              + ", ".join(f"c{i}={np.round(c, 1)}"
                          for i, c in enumerate(p_tgts)))
        return p_tgts


# ═════════════════════════════════════════════════════════════════════════════
#  Integration shim — drop-in replacement for onExtractCenterline's CL call
# ═════════════════════════════════════════════════════════════════════════════

def extract_centerline_per_branch(widget, model_node, endpoints_node,
                                  seg_model_node=None, refine=False):
    """
    Drop-in replacement for the single extractCenterline() call inside
    VesselAnalyzerWidget.onExtractCenterline().

    refine=False by default for speed.  Pass refine=True to re-enable
    Phase 4 gradient-descent refinement.
    """
    pipeline = PerBranchCenterlinePipeline(
        logic=widget.logic,
        widget=widget,
        model_node=model_node,
        refine=refine,
    )
    return pipeline.run(endpoints_node)
