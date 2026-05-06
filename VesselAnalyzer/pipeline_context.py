"""pipeline_context.py — Structured validation and audit layer for VesselAnalyzer.

Designed to be instantiated once per pipeline run in extractCenterline() and
threaded explicitly to each pipeline stage.  Accumulates named checks at three
severity levels, produces a versioned JSON audit trail, and drives the
confidence grade floor in the ostium confidence pipeline.

Usage
-----
    ctx = PipelineContext()
    self._pipeline_ctx = ctx           # bridge to mixin methods

    # Point A — pre-geometry (input validity)
    ctx.require(n_verts > 1000, "mesh_size", value=n_verts)
    ctx.require(n_rings >= 2,   "mesh_open", value=n_rings)
    ctx.expect(trunk_dir[2] < -0.7, "trunk_orientation",
               value=float(trunk_dir[2]))

    # Point B — post-solve, pre-topology commit (structural integrity)
    ctx.degrade(max_z_gap > MIN_GAP_MM, "z_separability", value=max_z_gap)
    ctx.degrade(not topo_tracker.is_degraded_critical(), "topology_integrity")

    # Point C — post-commit (directional diagnostics, warn-only)
    ctx.expect(dot < DOT_THRESHOLD, "branch_direction",
               value=dot, meta=ctx.branch_meta(bi, role, z))

    ctx.raise_if_invalid()             # aborts on ERROR-level failures

    # Confidence integration (inside _computeOstiumConfidence):
    ctx = getattr(self, "_pipeline_ctx", None)
    if ctx:
        floor = ctx.severity_floor()
        if floor:
            grade = min(grade, floor)

    # Audit output
    json_str = ctx.to_json(run_id=ts, build_tag=build_tag,
                            input_hash=self._input_hash)

Severity levels
---------------
ERROR    — fatal; raise_if_invalid() aborts the run.
DEGRADED — run completes but confidence grades are capped at MEDIUM.
WARN     — logged only; no downstream effect on output.

Polarity convention
-------------------
All three methods (require / expect / degrade) append an entry when their
condition argument is FALSE.  This matches standard assertion semantics and
is consistent across all three methods — no inversions.

    ctx.require(sgi >= ostiumGi, ...)   # fires if sgi < ostiumGi
    ctx.degrade(topo_ok, ...)           # fires if not topo_ok
    ctx.expect(trunk_dir[2] < -0.7, .) # fires if trunk nearly horizontal

Slicer-side only.  No dependencies beyond stdlib.
"""

from __future__ import annotations

import datetime
import enum
import json
from typing import Any, Dict, List, Optional


# ── Severity ──────────────────────────────────────────────────────────────────

class Severity(enum.Enum):
    WARN     = "warn"
    DEGRADED = "degraded"
    ERROR    = "error"


# ── Grade constants (match OstiumConfidence grade strings) ────────────────────

_GRADE_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "REJECT": 0}

def _grade_min(a: str, b: str) -> str:
    """Return the lower of two grade strings."""
    return a if _GRADE_ORDER.get(a, 0) <= _GRADE_ORDER.get(b, 0) else b


# ── PipelineContext ───────────────────────────────────────────────────────────

class PipelineContext:
    """Accumulates named validation checks and produces a versioned audit JSON.

    Thread this object explicitly through pipeline stages rather than storing
    on a global or relying on attribute lookup.  The ``self._pipeline_ctx``
    bridge attribute is a transitional compatibility shim only.
    """

    SCHEMA_VERSION = 1

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    # ── Core API (consistent polarity: fires when condition is False) ─────────

    def require(self,
                condition: bool,
                key: str,
                msg: str = "",
                value: Any = None,
                meta: Optional[Dict] = None) -> None:
        """Append an ERROR entry if condition is False.

        Use for invariants whose violation makes results unreliable:
        mesh integrity, endpoint count, sgi ordering after correction.
        raise_if_invalid() will abort the run if any ERROR is present.
        """
        if not condition:
            self._add(Severity.ERROR, key, msg, value, meta)

    def expect(self,
               condition: bool,
               key: str,
               msg: str = "",
               value: Any = None,
               meta: Optional[Dict] = None) -> None:
        """Append a WARN entry if condition is False.

        Use for soft expectations whose violation is notable but not fatal:
        trunk orientation, branch direction, model data sufficiency.
        Does not affect confidence grades.
        """
        if not condition:
            self._add(Severity.WARN, key, msg, value, meta)

    def degrade(self,
                condition: bool,
                key: str,
                msg: str = "",
                value: Any = None,
                meta: Optional[Dict] = None) -> None:
        """Append a DEGRADED entry if condition is False.

        Use for structural conditions that reduce reliability without making
        results wrong: topology clustering, poor Z separability, sgi correction
        applied.  Causes severity_floor() to return "MEDIUM", capping all
        confidence grades for this run.
        """
        if not condition:
            self._add(Severity.DEGRADED, key, msg, value, meta)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def errors(self) -> List[Dict]:
        return [e for e in self.entries if e["severity"] == Severity.ERROR.value]

    def degraded_entries(self) -> List[Dict]:
        return [e for e in self.entries if e["severity"] == Severity.DEGRADED.value]

    def warnings(self) -> List[Dict]:
        return [e for e in self.entries if e["severity"] == Severity.WARN.value]

    def is_valid(self) -> bool:
        """True if no ERROR-level entries are present."""
        return len(self.errors()) == 0

    def is_degraded(self) -> bool:
        """True if any DEGRADED-level entries are present."""
        return len(self.degraded_entries()) > 0

    def severity_floor(self) -> Optional[str]:
        """Grade cap string for the confidence pipeline, or None if no cap.

        Usage in _computeOstiumConfidence():
            floor = ctx.severity_floor()
            if floor:
                grade = _grade_min(grade, floor)
        """
        if self.errors():
            return "REJECT"
        if self.is_degraded():
            return "MEDIUM"
        return None

    def degradation_score(self) -> int:
        """Number of DEGRADED entries — for graduated grade caps (future use).

        Suggested thresholds (not yet active):
            score >= 1  → cap at MEDIUM
            score >= 3  → cap at LOW
        """
        return len(self.degraded_entries())

    # ── Flow control ──────────────────────────────────────────────────────────

    def raise_if_invalid(self) -> None:
        """Raise RuntimeError with a structured summary if any ERROR present."""
        if not self.is_valid():
            raise RuntimeError(
                f"[PipelineContext] Pipeline aborted — "
                f"{len(self.errors())} error(s):\n" +
                "\n".join(
                    f"  [{e['check']}] {e['msg']} (value={e['value']})"
                    for e in self.errors()
                )
            )

    def summary(self) -> Dict:
        return {
            "n_errors":   len(self.errors()),
            "n_degraded": len(self.degraded_entries()),
            "n_warnings": len(self.warnings()),
            "is_valid":   self.is_valid(),
        }

    # ── Meta helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def branch_meta(bi: Optional[int] = None,
                    role: Optional[str] = None,
                    z: Optional[float] = None) -> Dict:
        """Standard meta dict for branch-scoped checks.

        Always use this helper rather than constructing dicts ad hoc,
        so the shape stays consistent across all call sites.

        Example:
            ctx.degrade(sgi_ok, "sgi_ordering",
                        meta=ctx.branch_meta(bi=3, role="main", z=1903.1))
        """
        return {
            "bi":   bi,
            "role": role,
            "z":    float(z) if z is not None else None,
        }

    # ── JSON export ───────────────────────────────────────────────────────────

    def to_json(self,
                run_id: str = "",
                build_tag: str = "",
                input_hash: str = "") -> str:
        """Serialise to a versioned, stable-diff JSON string.

        Parameters
        ----------
        run_id      : timestamp string (e.g. "2026-05-01T22-17-03Z")
        build_tag   : VesselAnalyzer build tag (e.g. "v296-debug-VesselDebug")
        input_hash  : SHA-1[:12] of raw STL bytes before any processing

        Entries are sorted by (check, bi) for stable diffs between runs.
        """
        doc = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id":         run_id or "",
            "build_tag":      build_tag or "",
            "input_hash":     input_hash or "",
            "timestamp_utc":  datetime.datetime.utcnow().isoformat(),
            "entries": sorted(
                self.entries,
                key=lambda e: (
                    e["check"],
                    e.get("meta", {}).get("bi") or -1,
                )
            ),
            "summary": self.summary(),
        }
        return json.dumps(doc, indent=2)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _add(self,
             severity: Severity,
             key: str,
             msg: str,
             value: Any,
             meta: Optional[Dict]) -> None:
        self.entries.append({
            "check":    key,
            "severity": severity.value,
            "msg":      msg,
            "value":    value,
            "meta":     meta or {},
        })


# ── Input hash helper ─────────────────────────────────────────────────────────

def compute_input_hash(stl_path: str) -> str:
    """SHA-1[:12] of raw STL bytes — computed BEFORE any mesh processing.

    Call immediately after writing the STL for the refinement server,
    before _startRefinementServer().  Store result as self._input_hash
    and pass to ctx.to_json() at the end of the run.

    Returns empty string on read failure (non-fatal).
    """
    import hashlib
    try:
        with open(stl_path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except Exception as exc:
        print(f"[PipelineContext] compute_input_hash failed: {exc}")
        return ""


# ── LogReg data-sufficiency helpers ──────────────────────────────────────────

def count_unique_positive(records: list,
                           feat_keys: list,
                           target_type: str = "renal",
                           quantize_eps: float = 1e-4) -> int:
    """Count unique positive feature vectors for a given branch type.

    Uses quantized feature values to avoid false uniqueness from JSON
    float round-trip noise.  eps=1e-4 → 0.1 µm precision, well below
    any anatomically meaningful difference.

    Parameters
    ----------
    records     : list of dicts as loaded from ostium_logreg_data.json
    feat_keys   : ordered list of feature names (from records[0]['features'])
    target_type : one of 'main', 'renal', 'side', 'side_short'
    quantize_eps: rounding granularity for float features

    Returns
    -------
    int : number of unique positive feature vectors
    """
    def _q(v):
        return round(float(v) / quantize_eps) * quantize_eps

    unique = {
        tuple(_q(r["features"][k]) for k in feat_keys)
        for r in records
        if r.get("type") == target_type and r.get("y", 0) >= 0.5
    }
    return len(unique)


def check_model_readiness(ctx: PipelineContext,
                           records: list,
                           feat_keys: list,
                           target_type: str = "renal",
                           min_unique_positives: int = 10) -> bool:
    """Gate whether a trained model should be trusted or replaced by heuristic.

    Operates on raw dataset records (before training), not on fitted weights.
    This is intentional: the training loop will succeed even on 3 samples;
    the question is whether the resulting model is trustworthy.

    Registers appropriate ctx entries and returns True if model is trusted.

    Usage (at dataset load time, before deciding to use trained weights):
        records = json.load(open(dataset_path))
        feat_keys = list(records[0]['features'].keys())
        trusted = check_model_readiness(ctx, records, feat_keys, "renal")
        if trusted:
            # use trained weights
        else:
            # use heuristic scoring
    """
    n = count_unique_positive(records, feat_keys, target_type)

    if n == 0:
        ctx.require(
            False,
            f"{target_type}_model_no_positives",
            msg=f"No positive {target_type} samples in dataset — "
                f"labeling failure or empty class",
            value=0,
        )
        return False

    if n < min_unique_positives:
        ctx.expect(
            False,
            f"{target_type}_model_data_insufficient",
            msg=f"n_unique_pos={n} < threshold={min_unique_positives} "
                f"— using heuristic scoring",
            value=n,
        )
        return False

    return True


# ── Slicer module-scanner stub ────────────────────────────────────────────────
# Prevents RuntimeError when Slicer's auto-discovery scans this file.

class pipeline_context:  # noqa: N801
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title  = "pipeline_context"
            parent.hidden = True
