"""
Behavioral contract for _is_broken_branch.

Covers:
- All four rejection reasons
- Threshold boundaries (< vs <=, eps-stabilised)
- Insufficient evidence → None (not failure)
- NaN/inf radius handling
- pt↔radius alignment invariant after NaN filtering
- Mismatch fallback (loud log, safe behavior)
- Degenerate geometry (zero-length segments)
- Evaluation order (neck checks shadow gradient)
- Custom BranchThresholds flow-through
- Debug output contract

No VTK, no Slicer, no pipeline state required.
"""

import math
import numpy as np
import pytest

from vessel_branch_validation import (
    _is_broken_branch,
    BranchThresholds,
    REJECT_NECK_RATIO,
    REJECT_NECK_ABS,
    REJECT_GRAD,
    REJECT_ANGLE,
)

# ── Synthetic geometry helpers ────────────────────────────────────────────────

def _straight_branch(n=10, radius=8.0):
    """Straight line along Z, uniform radius — canonical healthy branch."""
    pts   = [(0.0, 0.0, float(i)) for i in range(n)]
    radii = [radius] * n
    return pts, radii

def _make_branches(pts):
    """Minimal branches + points arrays for a single non-trunk branch."""
    # bi=0 trunk (ignored by caller), bi=1 = our branch under test
    branches = [(0, 0), (0, len(pts))]
    return branches, pts

def _call(straight, **overrides):
    return _is_broken_branch(**{**straight, **overrides})

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def straight():
    pts, radii = _straight_branch()
    branches, points = _make_branches(pts)
    return dict(bi=1, branches=branches, points=points,
                radii=radii, surf_locator=None)

# ── Healthy baseline ──────────────────────────────────────────────────────────

def test_healthy_branch_passes(straight):
    assert _call(straight) is None

# ── Rejection reasons ─────────────────────────────────────────────────────────

def test_neck_ratio(straight):
    radii = [8.0] * 10
    radii[5] = 0.5                            # r_min << drop_ratio * r_max
    assert _call(straight, radii=radii) == REJECT_NECK_RATIO

def test_neck_abs(straight):
    radii = [1.5] * 10                        # all below abs_min=2.5
    assert _call(straight, radii=radii) == REJECT_NECK_ABS

def test_gradient_spike(straight):
    radii = [8.0] * 10
    radii[5] = 0.8                            # sharp single-step drop
    # drop_ratio=0.0 prevents REJECT_NECK_RATIO from shadowing gradient check
    assert _call(straight, radii=radii,
                 thresholds=BranchThresholds(drop_ratio=0.0)) == REJECT_GRAD

def test_gradient_increase_does_not_reject(straight):
    """abs(diff) means a spike upward must not trigger gradient rejection."""
    radii = [8.0] * 10
    radii[5] = 12.0
    assert _call(straight, radii=radii,
                 thresholds=BranchThresholds(drop_ratio=0.0)) is None

def test_direction_flip(straight):
    pts = list(straight["points"])
    pts[5] = (50.0, 0.0, 5.0)               # sharp lateral kink
    assert _call(straight, points=pts) == REJECT_ANGLE

def test_small_angle_does_not_reject(straight):
    """Tiny deviation must not reject — guards against flipped cosine comparison."""
    pts = list(straight["points"])
    pts[5] = (0.1, 0.0, 5.0)
    assert _call(straight, points=pts) is None

# ── Threshold boundaries (< vs <=, eps-stabilised) ───────────────────────────

_EPS = 1e-9  # guards against numeric drift if r_max computation changes

def test_neck_ratio_at_boundary_does_not_reject(straight):
    radii = [8.0] * 10
    radii[5] = 0.35 * 8.0 + _EPS           # epsilon above threshold → no reject
    assert _call(straight, radii=radii) is None

def test_neck_ratio_just_below_boundary_rejects(straight):
    radii = [8.0] * 10
    radii[5] = 0.35 * 8.0 - _EPS           # epsilon below threshold → reject
    assert _call(straight, radii=radii) == REJECT_NECK_RATIO

def test_gradient_at_boundary_does_not_reject(straight):
    radii = [8.0] * 10
    radii[5] = 8.0 * (1 - 0.5) + _EPS
    assert _call(straight, radii=radii,
                 thresholds=BranchThresholds(drop_ratio=0.0)) is None

def test_angle_at_boundary_does_not_reject(straight):
    # v1 = (0,0,1); v2 rotated exactly 120° in XZ plane → dot(v1,v2) = cos(120°)
    # Angle between segments equals threshold → must NOT reject (strict <)
    pts = list(straight["points"])
    cos120 = math.cos(math.radians(120.0))
    pts[5] = (math.sin(math.radians(120.0)) + _EPS, 0.0, 5.0 + cos120)
    assert _call(straight, points=pts) is None

# ── Insufficient evidence → None (not failure) ────────────────────────────────

def test_too_short_returns_none():
    pts = [(0., 0., float(i)) for i in range(3)]
    branches, points = _make_branches(pts)
    assert _is_broken_branch(bi=1, branches=branches, points=points,
                              radii=[8.0] * 3, surf_locator=None) is None

def test_all_nan_radii_returns_none(straight):
    radii = [float("nan")] * 10
    assert _call(straight, radii=radii) is None

def test_mixed_nan_below_minimum_returns_none(straight):
    radii = [8.0] * 4 + [float("nan")] * 6  # only 4 finite → < 5
    assert _call(straight, radii=radii) is None

# ── Alignment invariant under partial NaN filtering ───────────────────────────

def test_pts_radii_alignment_after_nan_filter(straight):
    """9 finite values survive filtering — angle loop must not IndexError."""
    radii = [float("nan")] + [8.0] * 9
    assert _call(straight, radii=radii) is None

# ── Mismatch fallback (loud log, safe behavior) ───────────────────────────────

def test_radii_points_mismatch_falls_back_to_locator(straight):
    """Length mismatch → locator fallback (returns 5.0) → healthy → None."""
    radii = straight["radii"][:-1]           # one element short
    assert _call(straight, radii=radii) is None

# ── Degenerate geometry ───────────────────────────────────────────────────────

def test_duplicate_point_skipped_in_angle_loop(straight):
    """Zero-length segment must not produce NaN angle or spurious rejection."""
    pts = list(straight["points"])
    pts[5] = pts[4]                          # coincident → zero vector
    assert _call(straight, points=pts) is None

# ── Evaluation order ──────────────────────────────────────────────────────────

def test_neck_ratio_shadows_gradient(straight):
    """REJECT_NECK_RATIO must fire before gradient is evaluated."""
    radii = [8.0] * 10
    radii[5] = 0.1                           # triggers both ratio and gradient
    assert _call(straight, radii=radii) == REJECT_NECK_RATIO

def test_neck_abs_shadows_gradient(straight):
    """REJECT_NECK_ABS must fire before gradient is evaluated."""
    radii = [2.0] * 10                       # triggers both abs and gradient
    assert _call(straight, radii=radii) == REJECT_NECK_ABS

# ── Custom thresholds flow-through ────────────────────────────────────────────

def test_custom_thresholds_respected(straight):
    """BranchThresholds values must reach comparisons, not be shadowed."""
    radii = [8.0] * 10
    radii[5] = 3.0                           # passes default drop_ratio=0.35
    assert _call(straight, radii=radii) is None

    tight = BranchThresholds(drop_ratio=0.9) # 3.0 < 0.9*8.0 → reject
    assert _is_broken_branch(**{**straight, "radii": radii},
                              thresholds=tight) == REJECT_NECK_RATIO

# ── Debug output contract ─────────────────────────────────────────────────────

def test_debug_line_emitted_when_flag_set(straight, capsys, monkeypatch):
    monkeypatch.setattr("vessel_branch_validation._DEBUG", True)
    _call(straight)
    assert "[BrokenBranch DEBUG]" in capsys.readouterr().out

def test_debug_line_suppressed_by_default(straight, capsys, monkeypatch):
    monkeypatch.setattr("vessel_branch_validation._DEBUG", False)
    _call(straight)
    assert "[BrokenBranch DEBUG]" not in capsys.readouterr().out

def test_debug_line_contains_reason_on_rejection(straight, capsys, monkeypatch):
    monkeypatch.setattr("vessel_branch_validation._DEBUG", True)
    _call(straight, radii=[1.5] * 10)
    assert "reason=neck_abs" in capsys.readouterr().out

def test_debug_line_contains_ok_on_pass(straight, capsys, monkeypatch):
    monkeypatch.setattr("vessel_branch_validation._DEBUG", True)
    _call(straight)
    assert "reason=OK" in capsys.readouterr().out
