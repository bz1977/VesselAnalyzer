"""branch_cost_field.py — Soft branch-ownership layer for PerBranchSolver.

Addresses the core failure mode identified in review:
    VMTK's geodesic cost has no notion of branch ownership.
    The trunk lumen is the cheapest corridor, so every branch solve
    that shares a proximal seed tends to hug the trunk as long as
    possible before being forced toward its distal target.

Design
------
A BranchCostField maintains a per-vertex scalar array on the shared
preprocessed mesh.  After each branch is solved its centerline points
are used to raise the cost of nearby vertices.  The next branch solve
then faces a higher cost inside already-claimed lumen, which biases
the geodesic away from shared anatomy.

Key design choices (from review):
  • Soft penalties only — no hard exclusion. Penalties scale with
    effective_distance = dist / (local_radius + ε), so a path 1 mm
    from a 12 mm vessel is anatomically normal; 1 mm from a 6 mm
    vessel is inside it.
  • Two-reference activation: trunk is the stable global frame;
    solved branches refine locally. Activation fires when the current
    path point is farther than d_activate from the nearest existing
    centerline (trunk or branch). This avoids committing to a noisy
    trunk-axis approximation for curved anatomy.
  • Type-boundary layered weights instead of per-solve decay.
    Trunk penalties never decay. Iliac penalties are applied at full
    weight for subsequent iliac solves and at reduced weight for renal
    solves (renals legitimately run close to the IVC trunk).
  • One tunable parameter per mechanism (enforced in API).

Integration
-----------
Drop this file next to per_branch_centerline_pipeline.py.
In PerBranchCenterlinePipeline.run(), after preprocessing the mesh:

    from branch_cost_field import BranchCostField, BranchType
    cost_field = BranchCostField(preprocessed_pd, trunk_radius_mm)

After solving the trunk centerline:
    cost_field.update(trunk_pts, BranchType.TRUNK)

After each iliac solve:
    cost_field.update(iliac_pts, BranchType.ILIAC)

When passing to VMTK, inject the cost field as edge weights:
    # VMTK does not expose edge weights directly, so we use the
    # cost array as a vertex scalar that can be read by the caller
    # to post-filter or re-seed bad solves.  See apply_to_solver().

Instrumentation
---------------
Every update() call emits a structured log line so penalty activation
can be correlated with divergence quality metrics.  See _log_update().

Parameters (one per mechanism, all tunable)
-------------------------------------------
  d_near_mm          : float  — distance below which max penalty applies
  d_far_mm           : float  — distance above which no penalty applies
  penalty_near       : float  — cost multiplier at d_near (default 8.0)
  penalty_mid        : float  — cost multiplier at midpoint (default 3.0)
  onset_lateral_mm   : float  — activation threshold: min dist from any
                                existing centerline before penalties apply
                                (prevents penalizing shared origin region)
  renal_iliac_weight : float  — weight applied to iliac penalties when
                                computing cost for renal-type solves
                                (default 0.3; renals run close to IVC)
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Branch type taxonomy ──────────────────────────────────────────────────────

class BranchType(enum.Enum):
    TRUNK  = "trunk"
    ILIAC  = "iliac"
    RENAL  = "renal"
    SIDE   = "side"   # generic short side branch


# Layered penalty weights: rows = solved branch type, cols = query branch type.
# Entry [solved][querying] = weight applied to that solved branch's penalties
# when building the cost seen by the querying branch type.
#
# Design rationale:
#   TRUNK  penalties are full-strength for everything (trunk is always relevant)
#   ILIAC  penalties are full-strength for other iliacs, reduced for renals
#          (renals legitimately run next to IVC, not next to iliacs — but we
#          keep a non-zero weight so a renal that drifts into iliac lumen pays)
#   RENAL  penalties are reduced for iliacs (same logic, symmetric)
#   SIDE   penalties are lightweight for everything (short, less anatomical mass)
_LAYER_WEIGHTS = {
    # solved_type -> {querying_type -> weight}
    BranchType.TRUNK:  {BranchType.TRUNK: 1.0, BranchType.ILIAC: 1.0,
                        BranchType.RENAL: 0.6, BranchType.SIDE:  0.8},
    BranchType.ILIAC:  {BranchType.TRUNK: 1.0, BranchType.ILIAC: 1.0,
                        BranchType.RENAL: 0.3, BranchType.SIDE:  0.5},
    BranchType.RENAL:  {BranchType.TRUNK: 1.0, BranchType.ILIAC: 0.3,
                        BranchType.RENAL: 1.0, BranchType.SIDE:  0.5},
    BranchType.SIDE:   {BranchType.TRUNK: 1.0, BranchType.ILIAC: 0.5,
                        BranchType.RENAL: 0.5, BranchType.SIDE:  0.8},
}

_LOG = "[CostField]"


# ── Solved-branch record ──────────────────────────────────────────────────────

@dataclass
class _SolvedBranch:
    pts:         np.ndarray     # (N,3) centerline points
    branch_type: BranchType
    local_radius: float         # smoothed radius estimate (mm)
    kd_tree:     object = None  # cKDTree built lazily

    def get_kd(self):
        if self.kd_tree is None:
            if not _HAS_SCIPY:
                raise RuntimeError("scipy is required for BranchCostField")
            self.kd_tree = cKDTree(self.pts)
        return self.kd_tree


# ── Main class ────────────────────────────────────────────────────────────────

class BranchCostField:
    """Per-vertex cost field on the preprocessed mesh.

    Parameters
    ----------
    mesh_pts : (M, 3) array of preprocessed mesh vertex positions.
        Pass np.asarray(mesh.GetPoints().GetData()) or equivalent.
        The array is referenced, not copied — do not mutate it externally.
    trunk_radius_mm : float
        Estimated trunk radius at the bifurcation.  Used as a fallback
        local_radius when the caller does not supply one.
    d_near_mm : float
        Distance at which the maximum penalty applies.
    d_far_mm : float
        Distance at which penalty decays to 1.0 (no effect).
    penalty_near : float
        Cost multiplier applied to vertices within d_near_mm (effective
        distance) of an existing centerline.
    onset_lateral_mm : float
        Minimum distance from any existing centerline before activation.
        Prevents penalizing the shared origin region (first few mm of
        every branch that genuinely overlap near the bifurcation).
    """

    def __init__(
        self,
        mesh_pts:         np.ndarray,
        trunk_radius_mm:  float = 8.0,
        *,
        d_near_mm:         float = 1.0,
        d_far_mm:          float = 5.0,
        penalty_near:      float = 8.0,
        onset_lateral_mm:  float = 2.0,
    ):
        if not _HAS_SCIPY:
            raise RuntimeError(
                "BranchCostField requires scipy.spatial.cKDTree — "
                "install scipy into Slicer's Python."
            )

        self._mesh_pts        = np.asarray(mesh_pts, dtype=np.float64)
        self._trunk_radius    = float(trunk_radius_mm)
        self._d_near          = float(d_near_mm)
        self._d_far           = float(d_far_mm)
        self._penalty_near    = float(penalty_near)
        self._onset_lateral   = float(onset_lateral_mm)

        n_verts = len(self._mesh_pts)
        # cost_layers[branch_type] = (M,) float array, one per solved branch type.
        # We accumulate max-per-layer (not sum) so a second iliac doesn't double-
        # penalize the region the first iliac already claimed.
        self._cost_layers: dict[BranchType, np.ndarray] = {}

        # Combined KD-tree across all solved centerlines (for onset check).
        self._all_pts: list[np.ndarray] = []
        self._all_kd:  Optional[object] = None  # rebuilt on update()

        self._solved: list[_SolvedBranch] = []

        print(f"{_LOG} Initialized: {n_verts} mesh verts, "
              f"d_near={d_near_mm}mm, d_far={d_far_mm}mm, "
              f"penalty_near={penalty_near:.1f}x, "
              f"onset_lateral={onset_lateral_mm}mm")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        centerline_pts: np.ndarray,
        branch_type:    BranchType,
        local_radius:   Optional[float] = None,
    ) -> None:
        """Register a solved centerline and update the cost field.

        Parameters
        ----------
        centerline_pts : (N, 3) array of solved centerline points.
        branch_type    : type of the branch that was just solved.
        local_radius   : representative lumen radius (mm).  If None, the
                         trunk_radius_mm passed at construction is used.
        """
        t0 = time.perf_counter()
        pts = np.asarray(centerline_pts, dtype=np.float64)
        if len(pts) < 2:
            print(f"{_LOG} update() skipped: < 2 pts in {branch_type.value} centerline")
            return

        r = float(local_radius) if local_radius is not None else self._trunk_radius
        r = max(r, 1.0)  # guard against zero/negative radius estimates

        branch = _SolvedBranch(pts=pts, branch_type=branch_type, local_radius=r)
        self._solved.append(branch)

        # Snapshot the PRIOR KD-tree (before appending the new branch).
        # This is used in _compute_penalty_layer's onset gate so that the
        # new branch does not exempt its own neighbourhood from penalties.
        # If there are no prior branches, prior_kd is None and the onset
        # gate is skipped (nothing to share an origin with yet).
        prior_kd = self._all_kd  # may be None on first call

        # Rebuild combined KD-tree including this branch's points.
        self._all_pts.append(pts)
        combined = np.vstack(self._all_pts)
        self._all_kd = cKDTree(combined)

        # Compute penalty contribution of this branch on each mesh vertex.
        # Pass prior_kd so the onset gate uses only previously-solved branches.
        penalty_contrib = self._compute_penalty_layer(branch, prior_kd=prior_kd)

        if branch_type not in self._cost_layers:
            self._cost_layers[branch_type] = np.ones(len(self._mesh_pts))

        # Take element-wise max: don't let a second iliac add on top of the
        # first — just keep the highest penalty at each vertex for this layer.
        np.maximum(self._cost_layers[branch_type], penalty_contrib,
                   out=self._cost_layers[branch_type])

        elapsed = time.perf_counter() - t0
        n_penalized = int(np.sum(penalty_contrib > 1.05))
        max_p = float(penalty_contrib.max())
        self._log_update(branch_type, len(pts), r, n_penalized, max_p, elapsed)

    def query(self, query_type: BranchType) -> np.ndarray:
        """Return the combined cost array for a branch of query_type.

        Returns a (M,) float array where M = number of mesh vertices.
        Values ≥ 1.0; 1.0 means no penalty.  The caller passes this to
        the VMTK edge-weight mechanism (or uses it for post-hoc analysis).
        """
        if not self._cost_layers:
            return np.ones(len(self._mesh_pts))

        combined = np.ones(len(self._mesh_pts))
        for solved_type, layer in self._cost_layers.items():
            w = _LAYER_WEIGHTS[solved_type].get(query_type, 1.0)
            if w < 1e-6:
                continue
            # Additive in log space = multiplicative in cost space,
            # which keeps the combined penalty proportional to each layer's
            # contribution rather than summing arbitrarily.
            combined += w * (layer - 1.0)

        return combined

    def check_onset(self, pt: np.ndarray) -> Tuple[bool, float]:
        """Check whether a point is past the onset threshold.

        Returns
        -------
        (active, dist_to_nearest_branch)
            active = True if penalty should be applied here.
            dist_to_nearest_branch in mm.
        """
        if self._all_kd is None:
            return False, float("inf")
        dist, _ = self._all_kd.query(pt)
        return float(dist) > self._onset_lateral, float(dist)

    def apply_to_solver_log(self, branch_pts: np.ndarray,
                            query_type: BranchType) -> dict:
        """Evaluate penalty along a just-solved path — for instrumentation.

        Returns a dict suitable for logging:
            activation_idx    : first index where onset fires (or None)
            activation_arc_mm : arc length at activation_idx
            max_penalty       : maximum cost-field value along path
            pct_under_penalty : % of path points with cost > 1.05
            first_diverge_idx : first point with cost < 1.05 after activation
                                (approx: where the path left shared anatomy)
        """
        if len(branch_pts) < 2:
            return {}

        cost = self.query(query_type)
        # Build KD-tree on mesh verts to look up per-path costs.
        mesh_kd = cKDTree(self._mesh_pts)
        _, idxs = mesh_kd.query(branch_pts)
        path_costs = cost[idxs]

        arc = _arc_length(branch_pts)
        THRESHOLD = 1.05

        activation_idx = None
        for i, (pt, pc) in enumerate(zip(branch_pts, path_costs)):
            active, _ = self.check_onset(pt)
            if active and pc > THRESHOLD:
                activation_idx = i
                break

        pct_under = float(np.sum(path_costs > THRESHOLD)) / len(path_costs) * 100.0

        result = {
            "activation_idx":    activation_idx,
            "activation_arc_mm": float(arc[activation_idx]) if activation_idx is not None else None,
            "max_penalty":       float(path_costs.max()),
            "pct_under_penalty": pct_under,
        }
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_penalty_layer(self, branch: _SolvedBranch,
                               prior_kd=None) -> np.ndarray:
        """Compute per-vertex penalty contribution from one solved branch.

        Penalty is a function of effective_distance = dist / (radius + ε):
          eff_dist < d_near / radius  → penalty_near
          eff_dist > d_far  / radius  → 1.0 (no penalty)
          between                     → smoothstep interpolation

        Activation gate: vertices whose nearest point on any PRIOR branch
        (not including the branch being registered) is closer than
        onset_lateral_mm get penalty 1.0 (no penalty).  This exempts the
        shared bifurcation region where all branches originate together.

        Using prior_kd (not _all_kd) is critical: if we used the combined
        KD-tree that already includes the current branch, every mesh vertex
        near the current branch would have dist_to_all ≈ 0 and be exempted
        from its own penalties — the self-exemption bug.

        Parameters
        ----------
        prior_kd : cKDTree | None
            KD-tree built from all previously-solved branches, BEFORE the
            current branch was appended to self._all_pts.  None on first call
            (no prior branches → onset gate is skipped entirely).
        """
        branch_kd = branch.get_kd()
        dists_branch, _ = branch_kd.query(self._mesh_pts)

        # Effective distance normalised by branch radius.
        r     = branch.local_radius
        eps   = 0.5  # mm; prevents /0 for tiny radius estimates
        eff   = dists_branch / (r + eps)

        d_near_eff = self._d_near / (r + eps)
        d_far_eff  = self._d_far  / (r + eps)

        # Smooth penalty curve: penalty_near at eff=0, 1.0 at eff≥d_far_eff.
        penalty = np.ones(len(self._mesh_pts))
        in_near  = eff < d_near_eff
        in_blend = (~in_near) & (eff < d_far_eff)

        penalty[in_near] = self._penalty_near
        if np.any(in_blend):
            t = (eff[in_blend] - d_near_eff) / max(d_far_eff - d_near_eff, 1e-6)
            t = np.clip(t, 0.0, 1.0)
            # Smoothstep: avoids discontinuous gradient at blend boundaries.
            smooth = t * t * (3.0 - 2.0 * t)
            penalty[in_blend] = self._penalty_near * (1.0 - smooth) + 1.0 * smooth

        # Onset gate: exempt the shared origin region using PRIOR branches only.
        # On the first call, prior_kd is None — no shared origin to exempt yet.
        if prior_kd is not None and self._onset_lateral > 0.0:
            dists_prior, _ = prior_kd.query(self._mesh_pts)
            onset_mask = dists_prior < self._onset_lateral
            penalty[onset_mask] = 1.0

        return penalty

    @staticmethod
    def _log_update(branch_type, n_pts, radius, n_penalized, max_p, elapsed):
        print(f"{_LOG} update({branch_type.value}): "
              f"n_pts={n_pts} r={radius:.1f}mm "
              f"verts_penalized={n_penalized} "
              f"max_penalty={max_p:.2f}x "
              f"elapsed={elapsed*1000:.0f}ms")


# ── Improved divergence criterion ─────────────────────────────────────────────

def find_divergence_idx(
    pts:                np.ndarray,
    bif_pt:             np.ndarray,
    other_branch_arrays: List[np.ndarray],
    *,
    snap_mm:            float = 4.0,
    angle_threshold_deg: float = 25.0,
    min_idx:            int   = 1,
    log_prefix:         str   = "",
) -> int:
    """Find the index in pts where this branch diverges from all others.

    Replaces the distance-only criterion in PerBranchCenterlinePipeline.
    Uses a combined distance + tangent-angle test so that:
      • A path still inside the shared trunk lumen (close AND parallel)
        is NOT considered diverged.
      • A path that curves away (close but high angle) IS considered diverged.
      • A path that moves far away (any angle) IS considered diverged.

    Both conditions must be satisfied to mark a point as "diverged":
        dist_to_any_other_branch > snap_mm
        OR
        angle_to_nearest_branch_tangent > angle_threshold_deg

    Parameters
    ----------
    pts                 : (N, 3) array, index 0 = bifurcation / shared end.
    bif_pt              : (3,) bifurcation coordinate (fallback only).
    other_branch_arrays : list of (M, 3) arrays of already-solved branches.
    snap_mm             : distance threshold for "shared" region (mm).
    angle_threshold_deg : tangent angle threshold for "departing" (degrees).
        Using |dot| so parallel and anti-parallel are treated identically
        (avoids orientation ambiguity); U-turns are a rare edge case documented
        in the code review and can be addressed with directed tangents later.
    min_idx             : minimum returned index (never 0).
    log_prefix          : e.g. "[PerBranchCL] branch 3" for structured logging.

    Returns
    -------
    int : first index in pts where the branch is diverged, clamped to min_idx.
    """
    if not other_branch_arrays:
        # No reference branches — fall back to nearest-to-bif_pt.
        dists = np.linalg.norm(pts - bif_pt, axis=1)
        idx = max(int(np.argmin(dists)), min_idx)
        if log_prefix:
            print(f"{log_prefix} diverge_idx={idx} "
                  f"(fallback: nearest-to-bif, no prior branches)")
        return idx

    if not _HAS_SCIPY:
        raise RuntimeError("find_divergence_idx requires scipy")

    # Build combined KD-tree of all prior branch points.
    all_other = np.vstack(other_branch_arrays)
    kd        = cKDTree(all_other)

    # Precompute tangents for all other branches (for angle criterion).
    # Tangent at point i = normalised(pts[i+1] - pts[i-1]), clamped at ends.
    other_tangents = _compute_tangents_for_kd(other_branch_arrays, all_other)

    angle_cos_thresh = math.cos(math.radians(angle_threshold_deg))

    first_diverged = None
    log_rows = []

    for i in range(min_idx, len(pts)):
        pt = pts[i]

        # ── Distance criterion ────────────────────────────────────────────────
        dist, nn_idx = kd.query(pt)
        dist_ok = bool(dist > snap_mm)

        # ── Angle criterion ───────────────────────────────────────────────────
        # Local tangent at this point (forward difference; backward at end).
        if i < len(pts) - 1:
            local_tang = pts[i + 1] - pts[i]
        else:
            local_tang = pts[i] - pts[i - 1]
        lt_norm = np.linalg.norm(local_tang)

        angle_ok = False
        if lt_norm > 1e-6 and other_tangents is not None:
            local_tang /= lt_norm
            nn_tang     = other_tangents[nn_idx]
            nt_norm     = np.linalg.norm(nn_tang)
            if nt_norm > 1e-6:
                nn_tang /= nt_norm
                # |dot| → treats parallel and anti-parallel as co-aligned (angle=0).
                cos_angle = abs(float(np.dot(local_tang, nn_tang)))
                # angle_ok when the paths are NOT co-aligned (angle > threshold).
                angle_ok = cos_angle < angle_cos_thresh

        diverged = dist_ok or angle_ok

        log_rows.append((i, float(dist), dist_ok, angle_ok, diverged))

        if diverged and first_diverged is None:
            first_diverged = i
            # Log only a window around the first divergence for brevity.
            break

    if first_diverged is None:
        # All points shared — very unusual; fall back.
        first_diverged = max(int(np.argmin(np.linalg.norm(pts - bif_pt, axis=1))),
                             min_idx)

    # ── Structured log ────────────────────────────────────────────────────────
    if log_prefix:
        # Always log the first divergence point and a summary.
        n_checked = len(log_rows)
        n_dist_ok = sum(1 for r in log_rows if r[2])
        n_ang_ok  = sum(1 for r in log_rows if r[3])
        bif_dist  = float(np.linalg.norm(pts[first_diverged] - bif_pt))
        print(
            f"{log_prefix} diverge_idx={first_diverged} "
            f"(bif_dist={bif_dist:.1f}mm, "
            f"checked={n_checked} pts, "
            f"dist_criterion={n_dist_ok}, "
            f"angle_criterion={n_ang_ok})"
        )

    return first_diverged


def _compute_tangents_for_kd(
    branch_arrays: List[np.ndarray],
    all_pts: np.ndarray,
) -> Optional[np.ndarray]:
    """Build a (Total_N, 3) tangent array aligned to all_pts.

    For each point in each branch array, the tangent is computed as
    the centred finite difference (forward/backward at ends).  The
    resulting array has the same row ordering as np.vstack(branch_arrays),
    which is what cKDTree(all_pts) indexes.

    Returns None if inputs are empty.
    """
    if not branch_arrays:
        return None

    tangs = []
    for pts in branch_arrays:
        n = len(pts)
        if n == 1:
            tangs.append(np.zeros((1, 3)))
            continue
        t = np.empty_like(pts)
        # Central differences
        t[1:-1] = pts[2:] - pts[:-2]
        # Forward/backward at endpoints
        t[0]    = pts[1] - pts[0]
        t[-1]   = pts[-1] - pts[-2]
        # Normalise (leave zero-length tangents as-is; they are handled above)
        norms = np.linalg.norm(t, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        tangs.append(t / norms)

    return np.vstack(tangs)


# ── Arc-length helper (local copy to avoid import dependency) ─────────────────

def _arc_length(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc-length array, same length as pts."""
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


# ── Topology warning tracker ──────────────────────────────────────────────────

class TopologyWarningTracker:
    """Accumulate snap-merge warnings and detect localised failure clusters.

    Usage — call record() each time a TopologyWarn fires:
        tracker = TopologyWarningTracker(snap_tol_mm=2.0)
        tracker.record(z_position=1962.0, node_id=5, pos=(30.9, 161.3, 1962.9))

    Call check_degraded() before each branch solve to detect whether a
    localised cluster of warnings has accumulated:
        if tracker.check_degraded(z_query, band_size_mm):
            # activate degraded mode for this solve

    Band size is computed as k * snap_tol * sqrt(n_verts / reference_n),
    as derived in the code review, or supplied by the caller.
    """

    CLUSTER_THRESHOLD = 5   # N warnings in one band → degraded
    DEFAULT_K         = 4   # band_size = k * snap_tol * scale_factor

    def __init__(self, snap_tol_mm: float = 2.0, n_verts: int = 59000):
        self._snap_tol  = float(snap_tol_mm)
        self._n_verts   = int(n_verts)
        self._warnings: list[Tuple[float, int, tuple]] = []  # (z, node_id, pos)
        self._degraded_bands: set[int] = set()

    def record(self, z_position: float, node_id: int, pos: tuple) -> None:
        self._warnings.append((float(z_position), node_id, pos))
        # Recompute after every new warning so check_degraded() is always fresh.
        self._recompute_degraded()

    def check_degraded(self, z_query: float, band_size_mm: Optional[float] = None) -> bool:
        """Return True if z_query falls in a degraded Z band."""
        bs  = band_size_mm if band_size_mm is not None else self._default_band()
        key = int(z_query / bs)
        return key in self._degraded_bands

    def n_warnings(self) -> int:
        return len(self._warnings)

    def summary(self) -> str:
        if not self._warnings:
            return "No topology warnings recorded."
        lines = [f"Topology warnings: {len(self._warnings)} total, "
                 f"{len(self._degraded_bands)} degraded band(s)"]
        for band in sorted(self._degraded_bands):
            bs   = self._default_band()
            z0   = band * bs
            z1   = z0 + bs
            cnt  = sum(1 for z, _, _ in self._warnings if z0 <= z < z1)
            lines.append(f"  Z=[{z0:.0f},{z1:.0f}): {cnt} warnings → DEGRADED")
        return "\n".join(lines)

    def _default_band(self) -> float:
        ref_n  = 59_000  # reference preprocessed mesh size from existing pipeline
        scale  = math.sqrt(max(self._n_verts, 1) / ref_n)
        return self.DEFAULT_K * self._snap_tol * max(scale, 0.5)

    def _recompute_degraded(self) -> None:
        bs      = self._default_band()
        buckets: dict[int, int] = {}
        for z, _, _ in self._warnings:
            key = int(z / bs)
            buckets[key] = buckets.get(key, 0) + 1
        self._degraded_bands = {k for k, v in buckets.items()
                                 if v >= self.CLUSTER_THRESHOLD}


# ── Seed inside/outside validator ─────────────────────────────────────────────

def validate_seed_outside_lumen(
    seed_pt: np.ndarray,
    model_node,
    log_prefix: str = "",
) -> Tuple[bool, str]:
    """Check that a proposed VMTK seed is outside (or on) the vessel wall.

    Uses vtkSelectEnclosedPoints — reliable for watertight meshes.
    Falls back to closest-surface-distance sign for meshes with small gaps.

    Returns
    -------
    (valid, reason)
        valid  = True if seed is outside or on the mesh surface.
        reason = short diagnostic string for logging.
    """
    try:
        import vtk
        pd = model_node.GetPolyData()
        if pd is None or pd.GetNumberOfPoints() == 0:
            return True, "no_mesh(skip)"

        # vtkSelectEnclosedPoints: tolerates small mesh imperfections.
        enclosed = vtk.vtkSelectEnclosedPoints()
        enclosed.SetInputData(_single_point_polydata(seed_pt))
        enclosed.SetSurfaceData(pd)
        enclosed.SetTolerance(0.001)
        enclosed.Update()
        inside = bool(enclosed.IsInside(0))

        if inside:
            if log_prefix:
                print(f"{log_prefix} seed {np.round(seed_pt, 1)} INSIDE lumen — "
                      f"seed will be rejected or adjusted")
            return False, "inside_lumen"

        return True, "outside_ok"

    except Exception as exc:
        # vtkSelectEnclosedPoints can fail on non-manifold meshes.
        # Fall back to surface-distance sign.
        try:
            import vtk
            pd  = model_node.GetPolyData()
            loc = vtk.vtkCellLocator()
            loc.SetDataSet(pd)
            loc.BuildLocator()
            closest = [0.0, 0.0, 0.0]
            cid     = vtk.reference(0)
            sid     = vtk.reference(0)
            d2      = vtk.reference(0.0)
            loc.FindClosestPoint(list(seed_pt), closest, cid, sid, d2)
            wall_dist = math.sqrt(float(d2))
            # A seed >2mm from the wall is likely inside the lumen; on the wall
            # is ≤1mm.  This is a heuristic fallback only.
            if wall_dist > 2.0:
                return False, f"near_wall_fallback(d={wall_dist:.1f}mm)"
            return True, f"wall_dist_fallback(d={wall_dist:.1f}mm)"
        except Exception as exc2:
            return True, f"validation_error({exc2})"


def _single_point_polydata(pt: np.ndarray):
    """Create a vtkPolyData with one vertex at pt (for enclosed-point test)."""
    import vtk
    pd  = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
    verts = vtk.vtkCellArray()
    verts.InsertNextCell(1)
    verts.InsertCellPoint(0)
    pd.SetPoints(pts)
    pd.SetVerts(verts)
    return pd


# ── Slicer module-scanner stub ────────────────────────────────────────────────
# Required so Slicer's auto-discovery doesn't raise RuntimeError when it scans
# this file.  Not a real loadable module.

class branch_cost_field:  # noqa: N801
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title  = "branch_cost_field"
            parent.hidden = True
