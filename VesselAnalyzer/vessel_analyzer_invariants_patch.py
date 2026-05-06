"""vessel_analyzer_invariants_patch.py

Integration patches for VesselAnalyzer — session backlog items 1–4.

This file is NOT a standalone module.  It documents the exact code changes
to apply to existing files.  Each patch is marked with the target file and
the surrounding context lines needed to locate the insertion point.

Patches in this file
--------------------
PATCH A  branch_cost_field.py / run_PATCHED
         Replace ordering-based iliac detection with gap-based Z classification.

PATCH B  per_branch_centerline_pipeline.py / run_PATCHED (or run())
         Add trunk orientation guard (Point A invariant).

PATCH C  VesselAnalyzer.py / _refineOstia (post-trim block)
         sgi >= ostiumGi correction with MIN_STABLE_OFFSET_PTS fallback.

PATCH D  VesselAnalyzer.py / extractCenterline
         PipelineContext instantiation, input hash, and JSON write.

PATCH E  VesselAnalyzer.py / _computeOstiumConfidence
         Confidence grade floor from ctx.severity_floor().

Apply these in order A → E.  All patches are additive; no existing logic
is removed unless explicitly noted.
"""


# =============================================================================
#  PATCH A — Gap-based Z classification in run_PATCHED
#  Target: per_branch_centerline_pipeline_patch.py  (or live run() method)
#
#  REPLACE:
#      sorted_tgts  = sorted(enumerate(p_tgts), key=lambda x: x[1][2])
#      iliac_indices = {i for i, _ in sorted_tgts[:2]}
#
#  WITH the block below.
# =============================================================================

import numpy as _np_patch  # local alias — avoids shadowing outer np


def _classify_iliac_indices(p_tgts: list,
                             min_gap_mm: float = 30.0,
                             log_prefix: str = "[PerBranchCL]") -> set:
    """Classify which target indices are iliacs using the largest Z gap.

    Replaces the list-position heuristic with geometry-driven classification:

        1. Sort targets by Z (ascending = inferior first).
        2. Find the largest gap between adjacent Z values.
        3. Everything below the gap midpoint = iliac (inferior anatomy).
        4. Everything above = renal / other (superior anatomy).

    Falls back to "bottom 2 by Z" if the gap is below min_gap_mm, and
    registers a warning via the caller's log_prefix.

    Parameters
    ----------
    p_tgts      : list of (3,) arrays [x, y, z] — distal target coords.
    min_gap_mm  : minimum acceptable Z gap.  Below this the anatomy is
                  ambiguous (very short trunk, high iliac origins).
    log_prefix  : e.g. "[PerBranchCL]" for structured log output.

    Returns
    -------
    set of int — indices into p_tgts that should be treated as iliacs.
    """
    if len(p_tgts) < 2:
        return set()

    z_vals   = _np_patch.array([float(pt[2]) for pt in p_tgts])
    order    = _np_patch.argsort(z_vals)          # ascending Z (inferior first)
    z_sorted = z_vals[order]

    gaps     = _np_patch.diff(z_sorted)
    split_i  = int(_np_patch.argmax(gaps))        # index of largest gap
    max_gap  = float(gaps[split_i])

    if max_gap < min_gap_mm:
        # Degenerate: no clear separation — fall back to bottom-2-by-Z.
        print(
            f"{log_prefix} [WARN] Z gap ({max_gap:.1f}mm) < min_gap_mm "
            f"({min_gap_mm}mm) — iliac/renal Z ranges overlap or trunk "
            f"is very short.  Falling back to bottom-2-by-Z classification."
        )
        iliac_indices = {int(order[i]) for i in range(min(2, len(p_tgts)))}
    else:
        # Everything with Z below (z_sorted[split_i] + z_sorted[split_i+1])/2
        # is classified as iliac.
        z_thresh     = float(z_sorted[split_i] + z_sorted[split_i + 1]) / 2.0
        iliac_indices = {i for i, pt in enumerate(p_tgts)
                         if float(pt[2]) < z_thresh}

    print(
        f"{log_prefix} Iliac/renal Z split: max_gap={max_gap:.1f}mm "
        f"@ Z≈{z_sorted[split_i]:.1f}–{z_sorted[split_i+1]:.1f}  "
        f"iliac_indices={sorted(iliac_indices)}"
    )
    assert iliac_indices, "No targets classified as iliac — anatomy unexpected"
    return iliac_indices


# Usage in run_PATCHED — replace the two lines with:
#
#   iliac_indices = _classify_iliac_indices(
#       p_tgts,
#       min_gap_mm=30.0,
#       log_prefix="[PerBranchCL]",
#   )
#
# The rest of the loop (for bi, p_tgt in enumerate(p_tgts)) is unchanged.


# =============================================================================
#  PATCH B — Trunk orientation guard (Point A invariant)
#  Target: PerBranchCenterlinePipeline.run() or run_PATCHED
#
#  INSERT after bif_geom.compute() returns trunk_dir.
#  bif_geom.trunk_dir is the unit vector of the trunk at the bifurcation.
# =============================================================================

def _check_trunk_orientation(trunk_dir, ctx, log_prefix="[PerBranchCL]"):
    """Point A invariant: trunk must run predominantly in the -Z direction.

    In standard axial CT acquisitions (DICOM → RAS), superior is +Z and
    the IVC runs caudally, so trunk_dir[2] should be strongly negative.

    Threshold -0.7 corresponds to ≤ 45° tilt from vertical — generous
    enough to accommodate patient positioning variation but strict enough
    to catch incorrectly oriented volumes.

    Does NOT auto-correct.  Emits a warning so the operator can decide.
    """
    z_component = float(trunk_dir[2])
    ctx.expect(
        z_component < -0.7,
        key="trunk_orientation",
        msg=f"trunk_dir[2]={z_component:.3f}; expected < -0.7 for axial IVC CT. "
            f"Z-based branch ordering may be unreliable.",
        value=z_component,
    )
    if z_component >= -0.7:
        print(
            f"{log_prefix} [WARN] trunk_orientation: trunk_dir[2]={z_component:.3f} "
            f"— non-standard orientation detected.  Results may be unreliable."
        )


# Usage:
#   _check_trunk_orientation(bif_geom.trunk_dir, ctx)
# immediately after trunk_dir is available, before any Z-based logic.


# =============================================================================
#  PATCH C — sgi correction with MIN_STABLE_OFFSET_PTS
#  Target: VesselAnalyzer.py — the block after _trimBranches() runs
#          (currently around the [IliacOstium] POST-TRIM ostiumGi audit block)
#
#  This patch replaces the bare audit log with an audit + correction pass.
# =============================================================================

# Configuration constant — add to VesselAnalyzer.py module level or as a
# class attribute on VesselAnalyzerLogic:
MIN_STABLE_OFFSET_PTS = 7   # minimum pts from ostium to stable-start; adjust
                             # to match the existing StableStart walk offset

# INSERT this function into VesselAnalyzer.py (or vessel_ostium_mixin.py)
# and call it immediately after _trimBranches() and before any code that
# reads sgi for measurement purposes.

def _auditAndCorrectSgi(self):
    """Post-trim sgi invariant: enforce sgi >= ostiumGi for all branches.

    Context
    -------
    sgi (stableStartGi) is computed before _trimBranches() runs.
    _trimBranches() may advance ostiumGi by 1–6 pts.
    If the trim moves ostiumGi past the pre-trim sgi, the stable-start
    precedes the ostium — a semantically invalid state that can produce
    negative arc lengths downstream.

    Correction
    ----------
    When sgi < ostiumGi after trim, advance sgi to
        min(ostiumGi + MIN_STABLE_OFFSET_PTS, branch_end_gi)
    This preserves the semantic meaning of sgi as "a stable measurement
    point some distance inside the branch", rather than clamping it to
    the ostium (which is the noisiest measurement location).

    The correction is registered as DEGRADED in PipelineContext because
    it signals that the trim/ostium pipeline produced an inconsistent
    state — geometry is still valid, but something unexpected happened.
    """
    ctx = getattr(self, "_pipeline_ctx", None)

    for bi, meta in enumerate(self._branchMeta):
        role     = meta.get("role", "unknown")
        if role == "trunk":
            continue

        ogi      = meta.get("ostiumGi")
        sgi      = meta.get("stableStartGi")
        end_gi   = meta.get("endGi")

        if ogi is None or sgi is None:
            continue

        if sgi < ogi:
            old_sgi = sgi
            corrected = min(ogi + MIN_STABLE_OFFSET_PTS,
                            end_gi if end_gi is not None else ogi + MIN_STABLE_OFFSET_PTS)
            meta["stableStartGi"] = corrected

            z_val = None
            try:
                z_val = float(self._rawBranches[bi][corrected - meta.get("startGi", 0)][2])
            except Exception:
                pass

            print(
                f"[InvariantFix] bi={bi} role={role!r}: "
                f"sgi corrected {old_sgi}→{corrected} "
                f"(ostiumGi={ogi}, offset={MIN_STABLE_OFFSET_PTS})"
            )

            if ctx is not None:
                ctx.degrade(
                    False,   # condition=False → always fires (we are in the bad branch)
                    key="sgi_corrected",
                    msg=f"sgi {old_sgi}→{corrected} post-trim correction",
                    value=(old_sgi, corrected),
                    meta=PipelineContext.branch_meta(bi=bi, role=role, z=z_val),
                )

        # Final invariant check — should always pass after correction above.
        sgi_final = meta.get("stableStartGi")
        if ctx is not None:
            ctx.require(
                sgi_final >= ogi,
                key="sgi_ordering",
                msg=f"sgi={sgi_final} < ostiumGi={ogi} (uncorrectable)",
                value=(sgi_final, ogi),
                meta=PipelineContext.branch_meta(bi=bi, role=role),
            )


# =============================================================================
#  PATCH D — PipelineContext integration in extractCenterline()
#  Target: VesselAnalyzer.py — VesselAnalyzerLogic.extractCenterline()
#
#  INSERT at the top of extractCenterline(), before any mesh/pipeline work.
# =============================================================================

# ADD import at top of VesselAnalyzer.py:
#   from pipeline_context import PipelineContext, compute_input_hash

_PATCH_D_PSEUDOCODE = """
def extractCenterline(self):
    # ── PATCH D: context + input hash ────────────────────────────────────────
    import datetime
    ctx = PipelineContext()
    self._pipeline_ctx = ctx

    run_id    = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    build_tag = getattr(self, "_build_tag", "")  # already set at module load

    # Input hash — compute before any mesh modification.
    # stl_path is set in _exportMeshForRefinement(); if unavailable at this
    # point, compute it after export and before _startRefinementServer().
    # self._input_hash is set there; read it here if already available.
    input_hash = getattr(self, "_input_hash", "")

    # ── Existing extractCenterline code continues below ───────────────────────
    ...

    # ── PATCH D: Point A invariant checks ───────────────────────────────────
    # (insert after mesh preprocessing and trunk_dir are available)
    from vessel_analyzer_invariants_patch import _check_trunk_orientation
    _check_trunk_orientation(trunk_dir, ctx)

    ctx.require(n_verts > 1000, "mesh_size",
                msg=f"Preprocessed mesh too small: {n_verts} verts",
                value=n_verts)
    ctx.require(n_boundary_rings >= 2, "mesh_open",
                msg=f"Mesh has {n_boundary_rings} boundary ring(s); need >= 2",
                value=n_boundary_rings)

    ...  # existing pipeline ...

    # ── PATCH D: Point B — post-solve checks ────────────────────────────────
    # (insert after branch solves, before topology commit)
    ctx.degrade(max_z_gap > MIN_GAP_MM, "z_separability",
                msg=f"max_z_gap={max_z_gap:.1f}mm < {MIN_GAP_MM}mm",
                value=max_z_gap)

    topo_ok = not self._topo_tracker.is_degraded_critical()
    ctx.degrade(topo_ok, "topology_integrity",
                msg=self._topo_tracker.summary())

    ...  # sgi correction (PATCH C) runs here ...

    # ── PATCH D: Point C — post-commit directional checks ───────────────────
    DOT_THRESHOLD = -0.3
    MIN_LEN_MM    = 25.0
    for bi, meta in enumerate(self._branchMeta):
        role   = meta.get("role")
        length = meta.get("arcLength", 0.0)
        dot    = meta.get("dot_trunk")
        z_val  = meta.get("ostiumPt", [None, None, None])[2]
        if role in ("main", "side") and length >= MIN_LEN_MM and dot is not None:
            ctx.expect(
                dot < DOT_THRESHOLD,
                key="branch_direction",
                msg=f"bi={bi} dot={dot:.3f} — branch not diverging from trunk",
                value=dot,
                meta=ctx.branch_meta(bi=bi, role=role, z=z_val),
            )

    # ── PATCH D: raise + audit output ───────────────────────────────────────
    ctx.raise_if_invalid()

    runs_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    audit_path = os.path.join(runs_dir, f"{run_id}_{input_hash or 'nohash'}.json")
    with open(audit_path, "w") as f:
        f.write(ctx.to_json(run_id=run_id,
                             build_tag=build_tag,
                             input_hash=input_hash))
    print(f"[PipelineContext] Audit written → {audit_path}")
"""


# =============================================================================
#  PATCH E — Confidence grade floor
#  Target: VesselAnalyzer.py — _computeOstiumConfidence(), at grade commit
#
#  INSERT immediately before each branch's confidence grade is stored in
#  branchMeta.  The ctx bridge is backward-compatible: if _pipeline_ctx is
#  not set (e.g. older code paths), it silently skips.
# =============================================================================

_PATCH_E_SNIPPET = """
# ── PATCH E: apply severity floor from PipelineContext ───────────────────────
# Insert just before: branchMeta[bi]['confidence_grade'] = grade

ctx = getattr(self, "_pipeline_ctx", None)
if ctx is not None:
    floor = ctx.severity_floor()
    if floor:
        from pipeline_context import _grade_min
        old_grade = grade
        grade = _grade_min(grade, floor)
        if grade != old_grade:
            print(f"[PipelineContext] bi={bi} grade capped: "
                  f"{old_grade} → {grade} (pipeline floor={floor})")
"""


# =============================================================================
#  PATCH F — Input hash at STL export time
#  Target: VesselAnalyzer.py — the method that writes the STL for the
#          refinement server (likely _exportMeshForRefinement or similar)
#
#  INSERT immediately after the STL file is written, before the server starts.
# =============================================================================

_PATCH_F_SNIPPET = """
# ── PATCH F: compute input hash from raw STL bytes ───────────────────────────
from pipeline_context import compute_input_hash
self._input_hash = compute_input_hash(stl_path)
print(f"[PipelineContext] input_hash={self._input_hash}  stl={stl_path}")
"""


# =============================================================================
#  PATCH G — Renal model trust gate
#  Target: VesselAnalyzer.py — wherever ostium_logreg_data.json is loaded
#          and the LogReg weights are decided to be used.
#
#  This is the call-site for check_model_readiness().
# =============================================================================

_PATCH_G_SNIPPET = """
# ── PATCH G: renal model data-sufficiency gate ───────────────────────────────
from pipeline_context import check_model_readiness

# records = already-loaded JSON list from ostium_logreg_data.json
# feat_keys = list(records[0]['features'].keys())
ctx = getattr(self, "_pipeline_ctx", None) or PipelineContext()

renal_trusted = check_model_readiness(
    ctx, records, feat_keys,
    target_type="renal",
    min_unique_positives=10,
)

if renal_trusted:
    renal_conf = self._logRegPredict(features, weights["renal"])
else:
    # Heuristic: use anatomy-only rules (lat, Z, diam, topo)
    renal_conf = self._renalHeuristicScore(branch_meta)
    print(f"[LogReg] renal model untrusted (heuristic mode)")
"""


# ── Slicer module-scanner stub ────────────────────────────────────────────────
# Prevents RuntimeError when Slicer's auto-discovery scans this file.

class vessel_analyzer_invariants_patch:  # noqa: N801
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title  = "vessel_analyzer_invariants_patch"
            parent.hidden = True
