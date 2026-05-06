"""per_branch_centerline_pipeline_patch.py

Integration diff for PerBranchCenterlinePipeline.run().

This file is NOT meant to be run directly.  It shows the exact changes
needed to wire BranchCostField, find_divergence_idx, and
TopologyWarningTracker into the existing pipeline.

Search for each ── PATCH N ── comment in per_branch_centerline_pipeline.py
and apply the corresponding change.

All changes are additive (no existing logic is removed in this pass).
The cost field is built and updated, and instrumentation is logged, but
the cost array is not yet passed into VMTK as edge weights — that
requires a VMTK API probe which is left as a TODO (see PATCH 5).
This pass is enough to:
  • verify the field is computed correctly (via instrumentation logs)
  • enable degraded-mode detection
  • replace _find_divergence_idx with the combined distance+angle version
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 1 — import at the top of per_branch_centerline_pipeline.py
#  (after the existing imports block, before class definitions)
# ─────────────────────────────────────────────────────────────────────────────

# ADD after "import numpy as np":
from branch_cost_field import (
    BranchCostField,
    BranchType,
    TopologyWarningTracker,
    find_divergence_idx,          # replaces _find_divergence_idx static method
    validate_seed_outside_lumen,
)


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 2 — PerBranchCenterlinePipeline.__init__: add topology tracker
#
#  Original:
#      class PerBranchCenterlinePipeline:
#          def __init__(self, logic, widget, model_node, refine=False):
#              self.logic      = logic
#              self.widget     = widget
#              self.model_node = model_node
#              self.do_refine  = refine
#
#  Becomes:
# ─────────────────────────────────────────────────────────────────────────────

class PerBranchCenterlinePipeline:
    def __init__(self, logic, widget, model_node, refine=False):
        self.logic      = logic
        self.widget     = widget
        self.model_node = model_node
        self.do_refine  = refine
        # PATCH 2: topology warning tracker — collects snap-merge warnings
        # emitted during graph collapse and detects localised failure clusters.
        self._topo_tracker = None  # initialised in run() once n_verts is known


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 3 — PerBranchCenterlinePipeline.run(): wire cost field and tracker
#
#  Locate the existing run() method.  The changes below are inserted at
#  specific points; surrounding lines are shown for context (marked ···).
# ─────────────────────────────────────────────────────────────────────────────

def run_PATCHED(self, endpoints_node):
    """Patched version of PerBranchCenterlinePipeline.run().

    Changes vs original:
      1. TopologyWarningTracker initialised after mesh preprocessing.
      2. BranchCostField initialised after bif_geom.compute().
      3. Trunk solve → cost_field.update(trunk_pts, BranchType.TRUNK).
      4. Iliac solves → cost_field.update(..., BranchType.ILIAC) after each.
      5. Renal/side solves → cost_field.update(..., BranchType.RENAL/SIDE).
      6. _find_divergence_idx replaced with find_divergence_idx (imported).
      7. apply_to_solver_log called after each branch for instrumentation.
      8. Seed outside-lumen validation added before each VMTK call.
      9. Degraded-mode check gates solve parameters.
    """
    import numpy as _np

    def _status(msg):
        print(f"[PerBranchCL] {msg}")

    def _tick():
        pass  # progress tick — unchanged from original

    # ── Step 0: read endpoints (unchanged) ───────────────────────────────────
    ring_centroids = self._read_centroids(endpoints_node)
    if len(ring_centroids) < 2:
        raise RuntimeError("[PerBranchCL] Need >= 2 endpoints")

    # ── Step 1: preprocess mesh (shared) — unchanged ─────────────────────────
    _status("Step 1/4: Preprocessing mesh (shared)...")
    cache = _PreprocessCache(
        self.model_node,
        target_points=8000,
        subdivide=True,
        decimation_aggressiveness=3.5,
    )
    preprocessed_pd = cache.get()
    n_verts = preprocessed_pd.GetNumberOfPoints() if preprocessed_pd else 0

    # PATCH 3a: initialise topology tracker once n_verts is known.
    self._topo_tracker = TopologyWarningTracker(snap_tol_mm=2.0, n_verts=n_verts)

    # ── Step 1b: bifurcation geometry (unchanged) ─────────────────────────────
    _status("Step 1/4: Locating bifurcation...")
    bif_geom = BifurcationGeometry(
        self.logic, self.model_node, endpoints_node,
        preprocess_cache=cache,
    ).compute()

    # PATCH 3b: initialise cost field.
    # mesh_pts: convert vtkPoints to numpy — works with VTK 8+.
    try:
        import vtk
        vtk_pts  = preprocessed_pd.GetPoints()
        mesh_pts = _np.array([vtk_pts.GetPoint(i)
                               for i in range(vtk_pts.GetNumberOfPoints())])
    except Exception as _e:
        print(f"[PerBranchCL] Cost field: could not extract mesh pts ({_e}); "
              f"soft penalties disabled for this run.")
        mesh_pts = _np.empty((0, 3))

    cost_field = BranchCostField(
        mesh_pts,
        trunk_radius_mm  = bif_geom.trunk_radius,
        d_near_mm        = 1.0,    # one tunable param: near-zone boundary
        d_far_mm         = 5.0,    # one tunable param: far-zone boundary
        penalty_near     = 8.0,    # one tunable param: max cost multiplier
        onset_lateral_mm = 2.0,    # one tunable param: shared-origin exemption
    )

    # ── Trunk solve ───────────────────────────────────────────────────────────
    # (existing code; only the update() call is new)
    seed_gen     = BranchSeedGenerator(bif_geom, self.model_node)
    p_trunk      = seed_gen.proximal_seed()
    p_tgts       = self._assign_distal_seeds(ring_centroids, p_trunk)

    solver       = PerBranchSolver(self.logic, self.model_node, preprocessed_pd)

    _status("Pre-solving trunk: ...")
    trunk_pts_vmtk = solver.solve_one(
        ring_centroids[_np.argmax([c[2] for c in ring_centroids])],
        bif_geom.snapped_bif_pt,
        bi=-1,
    )

    # PATCH 3c: register trunk in cost field.
    cost_field.update(trunk_pts_vmtk, BranchType.TRUNK,
                      local_radius=bif_geom.trunk_radius)
    _instr = cost_field.apply_to_solver_log(trunk_pts_vmtk, BranchType.TRUNK)
    print(f"[PerBranchCL] [INSTR] trunk: {_instr}")

    # ── Per-branch solves ─────────────────────────────────────────────────────
    branch_arrays  = []
    junction_pts   = []
    iliac_arrays   = []   # tracked separately for renal divergence check

    # Determine branch types from endpoints (Z ordering: lowest Z = iliacs,
    # mid Z = renals, highest Z = trunk — already filtered out).
    # Simple heuristic: sort distal targets by Z; bottom 2 = iliac, rest = renal.
    sorted_tgts  = sorted(enumerate(p_tgts), key=lambda x: x[1][2])
    iliac_indices = {i for i, _ in sorted_tgts[:2]}

    for bi, p_tgt in enumerate(p_tgts):
        is_iliac = bi in iliac_indices
        b_type   = BranchType.ILIAC if is_iliac else BranchType.RENAL

        _status(f"Step 2/4: Branch {bi + 1} of {len(p_tgts)} ({b_type.value})...")

        # PATCH 3d: degraded-mode check.
        bif_z     = float(bif_geom.snapped_bif_pt[2])
        degraded  = self._topo_tracker.check_degraded(bif_z)
        snap_mm   = 3.0 if degraded else 4.0      # tighter in degraded mode
        angle_deg = 20.0 if degraded else 25.0
        if degraded:
            print(f"[PerBranchCL] [DEGRADED] branch {bi}: "
                  f"tightened snap_mm={snap_mm}, angle_deg={angle_deg}")

        # PATCH 3e: seed validation.
        if is_iliac:
            p_src = seed_gen.iliac_bif_seed(p_tgt)
        else:
            p_src = seed_gen.branch_trunk_seed(p_tgt)

        seed_valid, seed_reason = validate_seed_outside_lumen(
            p_src, self.model_node,
            log_prefix=f"[PerBranchCL] branch {bi}",
        )
        if not seed_valid:
            # Push seed 5% farther from bifurcation toward the target.
            p_src = p_src + 0.05 * (p_tgt - p_src)
            print(f"[PerBranchCL] branch {bi}: seed adjusted "
                  f"(was {seed_reason}) → {_np.round(p_src, 1)}")

        # ── VMTK solve (existing logic) ───────────────────────────────────────
        if is_iliac:
            pts_raw   = solver.solve_one(p_tgt, p_src, bi)
            pts       = pts_raw[::-1].copy()
            prior_arr = iliac_arrays
        else:
            pts_raw   = solver.solve_one(p_tgt, p_trunk, bi)
            pts       = pts_raw[::-1].copy()
            prior_arr = iliac_arrays  # renals compare against iliac only (v297 fix)

        # PATCH 3f: replace static _find_divergence_idx with the new version.
        bif_idx    = find_divergence_idx(
            pts, bif_geom.snapped_bif_pt, prior_arr,
            snap_mm=snap_mm,
            angle_threshold_deg=angle_deg,
            min_idx=1,
            log_prefix=f"[PerBranchCL]   branch {bi}",
        )
        pts_branch = pts[bif_idx:]
        bif_dist   = float(_np.linalg.norm(pts[bif_idx] - bif_geom.snapped_bif_pt))

        if is_iliac:
            print(f"[PerBranchCL]   branch {bi}: tip-inward solve (reversed), "
                  f"arc={_arc_length(pts)[-1]:.1f} mm")
            iliac_arrays.append(pts_branch)
        else:
            print(f"[PerBranchCL]   branch {bi}: renal via-inlet solve (reversed), "
                  f"arc={_arc_length(pts)[-1]:.1f} mm")
            # Snap renal junction to nearest trunk point (existing logic).
            from scipy.spatial import cKDTree as _KD
            _tkd = _KD(trunk_pts_vmtk)
            _d, _ti = _tkd.query(pts_branch[0])
            _jpt = trunk_pts_vmtk[_ti].copy()
            print(f"[PerBranchCL]   branch {bi}: renal junction snapped "
                  f"to trunk pt (d={_d:.2f} mm) @ {_np.round(_jpt, 1)}")
            junction_pts.append(_jpt)

        branch_arrays.append(pts_branch)

        # PATCH 3g: update cost field and log instrumentation.
        cost_field.update(pts_branch, b_type,
                          local_radius=bif_geom.trunk_radius * 0.6)
        _instr = cost_field.apply_to_solver_log(pts_branch, b_type)
        print(f"[PerBranchCL] [INSTR] branch {bi} ({b_type.value}): {_instr}")

        _tick()

    # ── Phase 3 and 4: unchanged ──────────────────────────────────────────────
    # (junction snap, optional refinement, merge — no changes needed here)

    # PATCH 3h: final topology warning summary.
    print(f"[PerBranchCL] {self._topo_tracker.summary()}")


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 4 — Remove the static _find_divergence_idx from PerBranchCenterlinePipeline
#
#  The static method is replaced by the imported find_divergence_idx.
#  Delete or comment out:
#
#      @staticmethod
#      def _find_divergence_idx(pts, bif_pt, other_branch_arrays,
#                               snap_mm=4.0, min_idx=1):
#          ...
#
#  All call sites in run() are updated in PATCH 3f above.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 5 — TODO: pass cost array into VMTK as edge weights
#
#  This is the remaining gap between the current implementation and the full
#  "soft ownership" model.  Currently, cost_field.update() computes the
#  per-vertex array and logs it, but the solver still uses unweighted
#  geodesic distance.
#
#  To complete the loop:
#    1. Probe whether VMTK's ExtractCenterlineLogic.extractCenterline()
#       (or the underlying vtkSlicerCenterlineComputationLogic) accepts a
#       vtkPolyData with a scalar array named "Cost" on its point data.
#
#    2. If yes: before each solve, write cost_field.query(b_type) into the
#       preprocessed_pd point data:
#
#           cost_arr = cost_field.query(b_type)
#           vtk_floats = vtk.vtkFloatArray()
#           vtk_floats.SetName("Cost")
#           for v in cost_arr:
#               vtk_floats.InsertNextValue(float(v))
#           preprocessed_pd.GetPointData().AddArray(vtk_floats)
#           preprocessed_pd.GetPointData().SetActiveScalars("Cost")
#
#    3. If VMTK does not respect the scalar array as edge weights, the
#       alternative is to post-filter: after solve, if
#       apply_to_solver_log()["pct_under_penalty"] > 30%, flag for re-solve
#       with a nudged seed or tighter divergence criterion.
#
#  Both approaches are instrumented by the existing apply_to_solver_log()
#  output, which will show activation_arc_mm and pct_under_penalty — the
#  primary signals for whether soft penalties are working.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 6 — Hook topology warnings from VesselAnalyzerLogic
#
#  The snap-merge warnings are emitted in VesselAnalyzerLogic (vessel_centerline_mixin.py),
#  not in PerBranchCenterlinePipeline.  To route them to the tracker:
#
#  In vessel_centerline_mixin.py, locate the code that prints:
#      [TopologyWarn] Node {nid} raw degree=3 but collapsed degree=2 ...
#
#  Add after that print():
#      if hasattr(self, '_pipeline') and self._pipeline is not None:
#          tracker = getattr(self._pipeline, '_topo_tracker', None)
#          if tracker is not None:
#              tracker.record(z_position=pos[2], node_id=nid, pos=pos)
#
#  Where self._pipeline is the PerBranchCenterlinePipeline instance stored
#  on the logic object during the current run.  Add:
#      self._pipeline = pipeline
#  in VesselAnalyzerLogic.extractCenterline() after pipeline.run() is called.
#
#  This keeps the warning source (mixin) decoupled from the consumer (pipeline)
#  without requiring the mixin to import branch_cost_field.
# ─────────────────────────────────────────────────────────────────────────────


# ── Slicer module-scanner stub ────────────────────────────────────────────────
# Prevents RuntimeError when Slicer's auto-discovery scans this file.

class per_branch_centerline_pipeline_patch:  # noqa: N801
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title  = "per_branch_centerline_pipeline_patch"
            parent.hidden = True
