"""
Branch validation is a pure, order-sensitive classifier.

Do not reorder checks without updating tests:
- neck_ratio and neck_abs must run before gradient
- gradient must run before angle

Tests explicitly depend on this evaluation order.
See tests/test_branch_validation.py for the full behavioral contract.

Debug output is gated by environment variables:
  VBRANCH_DEBUG=1    per-branch decision lines
  VBRANCH_VERBOSE=1  rejection lines at the call site (read in mixin, not here)
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as _np

__all__ = [
    "BranchView",
    "BranchThresholds",
    "_is_broken_branch",
    "REJECT_NECK_RATIO",
    "REJECT_NECK_ABS",
    "REJECT_GRAD",
    "REJECT_ANGLE",
]

_DEBUG = os.getenv("VBRANCH_DEBUG", "0") == "1"
# _VERBOSE intentionally not defined here — it gates call-site prints,
# not classifier internals. Read it at the boundary that owns those prints.

# ── Helpers ───────────────────────────────────────────────────────────────────

def _locator_radius(pt, loc):
    """VTK import is intentionally local — keeps tests free of VTK dependency."""
    if loc is None:
        return 5.0
    import vtk, math
    cp = [0., 0., 0.]
    ci, si, d2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.)
    loc.FindClosestPoint(list(pt), cp, ci, si, d2)
    return math.sqrt(float(d2))

def _cos_angle(u, v):
    n = _np.linalg.norm(u) * _np.linalg.norm(v)
    if n < 1e-6:
        return 1.0  # degenerate → treat as 0°, never reject
    return float(_np.clip(_np.dot(u, v) / n, -1.0, 1.0))

# ── BranchView ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class BranchView:
    points : list
    radii  : Optional[List[float]]  # None = locator fallback required
    bi     : int

    @classmethod
    def from_raw(cls, bi: int, branches, points, radii) -> "BranchView":
        bs, be = branches[bi]
        if radii is not None and len(radii) != len(points):
            print(f"[BranchView] WARNING bi={bi}: radii/points length mismatch "
                  f"({len(radii)} vs {len(points)}) — falling back to locator")
            r_slice = None
        else:
            r_slice = radii[bs:be] if radii is not None else None
        return cls(points=points[bs:be], radii=r_slice, bi=bi)

# ── Thresholds ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BranchThresholds:
    """
    Tune in this order (evaluation priority matches tuning priority):
    1. abs_min           physical vessel diameter floor; easiest to reason about
    2. drop_ratio        neck-collapse sensitivity; biggest impact on false positives
    3. grad_thresh       partial cuts vs smooth tapering
    4. angle_thresh_deg  last; geometry noise varies most by dataset
    """
    drop_ratio      : float = 0.35
    abs_min         : float = 2.5
    grad_thresh     : float = 0.5
    angle_thresh_deg: float = 120.0

# ── Rejection reasons ─────────────────────────────────────────────────────────

REJECT_NECK_RATIO = "neck_ratio"
REJECT_NECK_ABS   = "neck_abs"
REJECT_GRAD       = "gradient"
REJECT_ANGLE      = "direction_flip"

# ── Core classifier ───────────────────────────────────────────────────────────

def _is_broken_branch(bi, branches, points, radii, surf_locator,
                       thresholds: BranchThresholds = BranchThresholds()):
    """
    Returns a REJECT_* reason string if the branch looks like a reconstruction
    artifact, or None if it looks anatomically valid.

    Contracts:
    - pts and b_r remain positionally aligned throughout
    - Insufficient evidence (< 5 finite-radius points) returns None —
      false positives (removing real anatomy) are worse than false negatives
    - Evaluation is cost-ordered: pure Python → single NumPy array → second array
    - Bias toward false negatives: when in doubt, keep the branch
    """
    try:
        view = BranchView.from_raw(bi, branches, points, radii)

        if len(view.points) < 5:
            return None

        # ── Radii: fast path or locator fallback ─────────────────────────────
        if view.radii is not None:
            raw_r = view.radii
        else:
            raw_r = [_locator_radius(pt, surf_locator) for pt in view.points]

        # ── Sanitize: filter NaN/inf while preserving pt↔radius alignment ────
        pairs = [(pt, r) for pt, r in zip(view.points, raw_r) if _np.isfinite(r)]
        if len(pairs) < 5:
            return None  # bad locator output = insufficient evidence, not failure

        pts = [p for p, _ in pairs]
        b_r = [r for _, r in pairs]

        assert len(pts) == len(b_r), "pts/radii misalignment after NaN filter"

        # ── Cheap pre-NumPy rejection ─────────────────────────────────────────
        r_max  = max(b_r)
        r_min  = min(b_r)
        reason = None

        if r_max > 1e-3 and r_min < thresholds.drop_ratio * r_max:
            reason = REJECT_NECK_RATIO
        elif r_min < thresholds.abs_min:
            reason = REJECT_NECK_ABS
        else:
            # ── Gradient check ────────────────────────────────────────────────
            r_np = _np.array(b_r)
            if len(r_np) >= 2:
                denom    = _np.maximum(r_np[:-1], 1e-3)
                max_grad = float(_np.max(_np.abs(_np.diff(r_np) / denom)))
            else:
                max_grad = 0.0

            if max_grad > thresholds.grad_thresh:
                reason = REJECT_GRAD
            else:
                # ── Angle check (cosine avoids arccos in loop) ────────────────
                cos_thresh = _np.cos(_np.deg2rad(thresholds.angle_thresh_deg))
                for i in range(1, len(pts) - 1):
                    v1 = _np.subtract(pts[i],   pts[i-1])
                    v2 = _np.subtract(pts[i+1], pts[i])
                    if (_np.linalg.norm(v1) < 1e-6 or
                            _np.linalg.norm(v2) < 1e-6):
                        continue  # skip degenerate (nearly coincident) points
                    if _cos_angle(v1, v2) < cos_thresh:
                        reason = REJECT_ANGLE
                        break

        if _DEBUG:
            print(f"[BrokenBranch DEBUG] bi={bi} "
                  f"r_max={r_max:.2f} r_min={r_min:.2f} "
                  f"reason={reason or 'OK'}")
        return reason

    except Exception as e:
        import traceback
        print(f"[BrokenBranch] Error bi={bi}: {e}\n{traceback.format_exc()}")
        return None
# Slicer module boilerplate to prevent Qt plugin errors
try:
    import slicer
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule

    class vessel_branch_validation(ScriptedLoadableModule):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "Hidden"
            parent.hidden = True

except ImportError:
    pass
