"""
vessel_geometry.py — Pure geometric computation helpers for VesselAnalyzer.

Extraction policy
-----------------
Only functions that satisfy ALL of the following live here:
  1. No Slicer, VTK, or Qt imports
  2. No closure over pipeline state (self, branchMeta, rawPoints, …)
  3. Deterministic given inputs — same args → same result, always
  4. Unit-testable standalone: ``python -c "from vessel_geometry import *"``

Functions that close over pipeline context (e.g. _renal_candidates, _eSurfLoc,
graphBranches) remain as nested defs inside loadCenterline / _refineOstia.
Do not move them here without first threading all context as explicit arguments
and writing the corresponding unit tests.

Usage in VesselAnalyzer.py
--------------------------
Replace the mid-file module-level block::

    import numpy as _np
    from collections import defaultdict as _defaultdict

    def _va_unit(v): ...
    def _va_angle(v1, v2): ...
    def _va_dist(p1, p2): ...

with::

    import numpy as _np
    from collections import defaultdict as _defaultdict
    from vessel_geometry import (
        geo_clamp, geo_unit, geo_angle_deg, geo_dist,
        geo_arc_length,
        renal_anatomy_gate, renal_composite_score,
        require_geo_keys,
    )
    # Legacy aliases so existing call-sites need zero changes:
    _va_unit   = geo_unit
    _va_angle  = geo_angle_deg
    _va_dist   = geo_dist

The nested _clamp / _renal_anatomy_gate / _renal_composite_score definitions
inside loadCenterline can then be deleted and replaced with calls to the module
versions (constants are passed as keyword arguments with identical defaults).
"""

import math
import numpy as _np

# ── Type alias ────────────────────────────────────────────────────────────────
# A "point" is any 3-element sequence (list, tuple, or np.ndarray).
# Functions return np.ndarray unless documented otherwise.

# ─────────────────────────────────────────────────────────────────────────────
# 1. Scalar helpers
# ─────────────────────────────────────────────────────────────────────────────

def geo_clamp(v, lo, hi):
    """Clamp *v* to [lo, hi].

    Identical to the local ``_clamp`` defined in several pipeline closures.
    Two-argument form ``geo_clamp(v, 0.0, 1.0)`` is the most common usage.

    Parameters
    ----------
    v  : float
    lo : float  lower bound
    hi : float  upper bound

    Returns
    -------
    float
    """
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vector helpers  (replaces _va_unit / _va_angle / _va_dist)
# ─────────────────────────────────────────────────────────────────────────────

def geo_unit(v):
    """Return the unit vector of *v*.

    Returns *v* unchanged (not normalised) when the norm is zero to avoid
    dividing by zero silently.  Callers that depend on a true unit vector
    should check ``np.linalg.norm(v) > 0`` first when that matters.

    Parameters
    ----------
    v : array-like, shape (N,)

    Returns
    -------
    np.ndarray, shape (N,)
    """
    v = _np.asarray(v, dtype=float)
    n = _np.linalg.norm(v)
    return v / n if n != 0.0 else v


def geo_angle_deg(v1, v2):
    """Angle in degrees between two vectors.

    Handles zero-length vectors safely (returns 0.0).

    Parameters
    ----------
    v1, v2 : array-like, shape (N,)

    Returns
    -------
    float  angle in [0, 180]
    """
    u1 = geo_unit(_np.asarray(v1, dtype=float))
    u2 = geo_unit(_np.asarray(v2, dtype=float))
    return float(_np.degrees(_np.arccos(_np.clip(_np.dot(u1, u2), -1.0, 1.0))))


def geo_dist(p1, p2):
    """Euclidean distance between two points.

    Parameters
    ----------
    p1, p2 : array-like, shape (3,)

    Returns
    -------
    float
    """
    return float(_np.linalg.norm(_np.asarray(p1, dtype=float) - _np.asarray(p2, dtype=float)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Polyline / arc helpers
# ─────────────────────────────────────────────────────────────────────────────

def geo_arc_length(pts):
    """Total arc length (mm) of a polyline.

    Parameters
    ----------
    pts : sequence of array-like, shape (N, 3) or list of 3-tuples

    Returns
    -------
    float  arc length in the same units as the input coordinates
    """
    if len(pts) < 2:
        return 0.0
    arr = _np.asarray(pts, dtype=float)
    return float(_np.sum(_np.linalg.norm(_np.diff(arr, axis=0), axis=1)))


def geo_arc_length_to_index(pts, target_mm, start=0):
    """Walk *pts* from index *start* until *target_mm* arc is covered.

    Returns the index of the first point whose cumulative arc from *start*
    meets or exceeds *target_mm*, or ``len(pts) - 1`` if the branch is
    shorter than requested.

    Parameters
    ----------
    pts       : sequence of 3-tuples / array-like shape (N, 3)
    target_mm : float
    start     : int  starting index (default 0)

    Returns
    -------
    int
    """
    arr = _np.asarray(pts, dtype=float)
    arc = 0.0
    for i in range(start, len(arr) - 1):
        arc += float(_np.linalg.norm(arr[i + 1] - arr[i]))
        if arc >= target_mm:
            return i + 1
    return len(arr) - 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Renal vein classification — pure math
# ─────────────────────────────────────────────────────────────────────────────
#
# These are the *mathematical cores* of the renal-tagging pipeline.
# They accept only plain Python scalars / floats and return plain scalars.
# All pipeline context (branchMeta, rawPoints, ostiumGi, …) is resolved by
# the caller before invoking these functions.
#
# The companion closures (_renalBranchPts, _is_symmetric_pair, _build_topology_score,
# _midptDiam, _localDiam, _nearestDistToProximalBranch) close over pipeline
# state and remain as nested defs inside loadCenterline.

# Default gate thresholds — documented here for single-source-of-truth.
# Override by passing keyword arguments; callers that reproduce the defaults
# need not be changed.
_RENAL_GATE_MIN_LEN_MM = 40.0    # absolute length floor
_RENAL_GATE_MIN_LAT_MM = 30.0    # minimum lateral tip offset
_RENAL_GATE_MIN_DZ_MM  = 20.0    # minimum height above iliac bif
_RENAL_GATE_MAX_DZ_MM  = 120.0   # maximum height above iliac bif

# Default composite-score weights — matches the v182 hand-tuned formula.
_RENAL_W_DIAM  = 0.25
_RENAL_W_LAT   = 0.25
_RENAL_W_Z     = 0.15
_RENAL_W_ANGLE = 0.10
_RENAL_W_TOPO  = 0.25
_RENAL_W_STAB  = 0.25   # penalty weight


def renal_anatomy_gate(
    blen,
    lat_mm,
    ostium_z,
    bif_z,
    *,
    min_len_mm=_RENAL_GATE_MIN_LEN_MM,
    min_lat_mm=_RENAL_GATE_MIN_LAT_MM,
    min_dz_mm=_RENAL_GATE_MIN_DZ_MM,
    max_dz_mm=_RENAL_GATE_MAX_DZ_MM,
):
    """Hard anatomical gate applied before composite scoring.

    Returns ``(passed: bool, reason: str)``.  A branch that fails ANY gate
    is immediately classified as ``side_branch`` regardless of composite score.

    Renal vein anatomy rules
    ------------------------
    * Not a tiny stub: length ≥ ``min_len_mm``
    * Extends meaningfully off the trunk: lateral tip offset ≥ ``min_lat_mm``
    * Originates clearly above the iliac bifurcation: dz ≥ ``min_dz_mm``
    * Does NOT originate excessively far from bif: dz ≤ ``max_dz_mm``

    Parameters
    ----------
    blen      : float  branch arc length (mm)
    lat_mm    : float  perpendicular tip offset from trunk axis (mm)
    ostium_z  : float  ostium Z coordinate (LPS mm)
    bif_z     : float  primary bifurcation Z coordinate (LPS mm)
    min_len_mm, min_lat_mm, min_dz_mm, max_dz_mm : float  gate thresholds

    Returns
    -------
    (bool, str)  (True, "ok") on pass; (False, reason_string) on fail
    """
    dz = ostium_z - bif_z
    if blen < min_len_mm:
        return False, f"len={blen:.1f}mm<{min_len_mm}mm"
    if lat_mm < min_lat_mm:
        return False, f"lat={lat_mm:.1f}mm<{min_lat_mm}mm"
    if dz < min_dz_mm:
        return False, f"dz={dz:.1f}mm<{min_dz_mm}mm"
    if dz > max_dz_mm:
        return False, f"dz={dz:.1f}mm>{max_dz_mm}mm"
    return True, "ok"


def renal_composite_score(
    diam_ratio,
    lat_mm,
    angle_deg,
    ostium_z,
    bif_z,
    prox_diam,
    dist_diam,
    topology_score=0.55,
    *,
    w_diam=_RENAL_W_DIAM,
    w_lat=_RENAL_W_LAT,
    w_z=_RENAL_W_Z,
    w_angle=_RENAL_W_ANGLE,
    w_topo=_RENAL_W_TOPO,
    w_stab=_RENAL_W_STAB,
    z_mean_mm=80.0,
    z_sigma_mm=40.0,
):
    """Composite renal-vein classifier (v182).

    Returns a named tuple of component scores so callers can log individual
    signals without recomputing.

    Score range: approximately [−0.25, 1.15].  Recommended threshold: ≥ 0.45
    → renal vein.

    Features
    --------
    diam_sc   – diameter ratio (strong candidate signal)
    lat_sc    – lateral tip offset from trunk axis (anatomical truth)
    z_sc      – relative Z: Gaussian anchored at ``bif_z + z_mean_mm``
    angle_sc  – departure angle (secondary support)
    topo_sc   – pair symmetry + dominance + connection quality (caller-supplied)
    stab_pen  – penalise proximal inflation from IVC wall leakage

    Parameters
    ----------
    diam_ratio     : float  branch-diam / trunk-diam at ostium Z
    lat_mm         : float  perpendicular tip offset from trunk axis (mm)
    angle_deg      : float  departure angle from trunk direction (degrees)
    ostium_z       : float  ostium Z coordinate (LPS mm)
    bif_z          : float  primary bifurcation Z coordinate (LPS mm)
    prox_diam      : float  branch diameter near ostium (mm)
    dist_diam      : float  branch diameter distally (mm) — 0 → no penalty
    topology_score : float  pre-computed topology score [0, 1], default 0.55
    w_*            : float  feature weights (keyword-only, use defaults)
    z_mean_mm      : float  expected renal ostium height above bif (mm)
    z_sigma_mm     : float  Gaussian half-width for Z score (mm)

    Returns
    -------
    (score, diam_sc, lat_sc, z_sc, angle_sc, topo_sc, stab_pen) : tuple of float
    """
    # 1. Diameter score: starts at ratio ≈ 0.40, saturates at ≈ 0.70
    diam_sc = geo_clamp((diam_ratio - 0.40) / 0.30, 0.0, 1.0)

    # 2. Lateral score: < 15 mm weak, > 40 mm strong renal
    lat_sc = geo_clamp((lat_mm - 15.0) / 25.0, 0.0, 1.0)

    # 3. Z-position: Gaussian centred at bif_z + z_mean_mm, σ = z_sigma_mm
    dz = ostium_z - bif_z
    z_sc = math.exp(-0.5 * ((dz - z_mean_mm) / z_sigma_mm) ** 2)

    # 4. Angle: below 20° weak, above 70° strong
    angle_sc = geo_clamp((angle_deg - 20.0) / 50.0, 0.0, 1.0)

    # 5. Topology score (caller-supplied from pair/dominance analysis)
    topo_sc = geo_clamp(topology_score, 0.0, 1.0)

    # 6. Proximal stability penalty (anti-fake / anti-IVC-bleed)
    if dist_diam and dist_diam > 1e-6:
        stab_ratio = prox_diam / dist_diam
    else:
        stab_ratio = 1.0
    stab_pen = geo_clamp((stab_ratio - 1.25) / 0.5, 0.0, 1.0)

    score = (
        w_diam  * diam_sc
        + w_lat   * lat_sc
        + w_z     * z_sc
        + w_angle * angle_sc
        + w_topo  * topo_sc
        - w_stab  * stab_pen
    )
    return score, diam_sc, lat_sc, z_sc, angle_sc, topo_sc, stab_pen


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pipeline contract guard
# ─────────────────────────────────────────────────────────────────────────────

def require_geo_keys(geo, required=("tip_x_sign", "tip_z", "bpts")):
    """Assert that a branch geometry dict contains all required keys.

    Call this at the top of any function that consumes a ``geo`` dict to catch
    missing keys at the producer boundary rather than with an obscure KeyError
    deep inside a computation.

    Parameters
    ----------
    geo      : dict   branch geometry dict
    required : tuple  keys that must be present

    Raises
    ------
    ValueError  listing all missing keys
    """
    missing = [k for k in required if k not in geo]
    if missing:
        raise ValueError(
            f"vessel_geometry: geo dict is missing required key(s): {missing}. "
            f"Present keys: {list(geo.keys())}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Legacy aliases  (zero migration cost for existing call-sites)
# ─────────────────────────────────────────────────────────────────────────────
# If VesselAnalyzer.py imports these aliases it can remove the local defs
# without touching any of the 100+ downstream call-sites.

_va_unit  = geo_unit        # previously: def _va_unit(v) at line 11207
_va_angle = geo_angle_deg   # previously: def _va_angle(v1, v2) at line 11213
_va_dist  = geo_dist        # previously: def _va_dist(p1, p2) at line 11219


# ─────────────────────────────────────────────────────────────────────────────
# 7. Point-list / edge helpers  (companion to centerline_graph.py)
# ─────────────────────────────────────────────────────────────────────────────
#
# These live here (not in centerline_graph) because they operate on plain
# 3-tuple geometry, not on graph node-id adjacency structures.

import math as _math

def geo_dist3(p: "sequence", q: "sequence") -> float:  # type: ignore[type-arg]
    """Euclidean distance between two 3-tuples / length-3 sequences.

    Identical to ``geo_dist`` for 3-D inputs, provided as a standalone helper
    so graph modules that don't want to import numpy can use it.

    Parameters
    ----------
    p, q : sequence of length 3

    Returns
    -------
    float
    """
    return _math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2)


_GEO_DEDUP_SNAP_MM: float = 5.0   # default endpoint proximity threshold


def geo_are_duplicate_edge(
    pts_a,
    pts_b,
    snap_mm: float = _GEO_DEDUP_SNAP_MM,
) -> bool:
    """Return ``True`` when *pts_a* and *pts_b* represent the same graph edge.

    Two edge polylines are considered duplicates when:

    * Their lengths differ by at most ``max(5, len(pts_a) // 5)`` points
      (allows for minor resampling differences), AND
    * Their endpoints match within *snap_mm* in either forward or reversed
      orientation.

    Extracted from the nested ``_areDuplicateEdge`` closure in
    ``VesselAnalyzer.loadCenterline`` (v275, line ~13961).

    Parameters
    ----------
    pts_a, pts_b : list of 3-tuples / array-like
    snap_mm      : float  endpoint distance threshold (default 5.0 mm)

    Returns
    -------
    bool
    """
    if abs(len(pts_a) - len(pts_b)) > max(5, len(pts_a) // 5):
        return False
    same_fwd = (
        geo_dist3(pts_a[0], pts_b[0]) < snap_mm
        and geo_dist3(pts_a[-1], pts_b[-1]) < snap_mm
    )
    same_rev = (
        geo_dist3(pts_a[0], pts_b[-1]) < snap_mm
        and geo_dist3(pts_a[-1], pts_b[0]) < snap_mm
    )
    return same_fwd or same_rev


# ── Slicer module-scanner guard ───────────────────────────────────────────────
# Slicer auto-scans all .py files in the module folder and expects a class
# matching the filename.  This stub satisfies that requirement without
# registering as an actual loadable module (no ScriptedLoadableModule base).
class vessel_geometry:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        pass



# ── Slicer module-scanner guard ───────────────────────────────────────────────
class vessel_geometry:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_geometry"
            parent.hidden = True
