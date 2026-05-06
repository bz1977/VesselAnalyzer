"""
ostium_logreg.py — Logistic-regression ostium confidence scorer for VesselAnalyzer.

Extraction policy
-----------------
Only code that satisfies ALL of the following lives here:
  1. No Slicer, VTK, or Qt imports.
  2. No closure over pipeline state (self, branchMeta, …).
  3. Deterministic given inputs — same args → same result always (aside
     from the intentional stateful weight update in train_step).
  4. Unit-testable standalone: ``python -c "from ostium_logreg import OstiumLogReg"``.

Classes
-------
OstiumLogReg    Per-branch-type logistic regression confidence scorer.
                (Previously named ``_OstiumLogReg`` inside VesselAnalyzer.py.)

Data files (written/read relative to the module directory)
----------------------------------------------------------
ostium_logreg_weights.json   Learned per-branch-type weight vectors.
ostium_logreg_data.json      Accumulated training records (appended each run).

Usage in VesselAnalyzer.py
--------------------------
Replace the inline ``_OstiumLogReg`` class definition (lines ~10107–10430)
and its instantiation sites with::

    from ostium_logreg import OstiumLogReg as _OstiumLogReg

No other changes are required — the public API (classify_branch, build_features,
score, train_step, save) is identical.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence


class OstiumLogReg:
    """Lightweight logistic-regression ostium confidence scorer.

    Replaces hard-coded weight vectors in _computeOstiumConfidence with
    per-branch-type learned weights.  Weights are initialised from the
    hand-tuned priors already in the codebase, then updated from logged
    outcomes via online gradient descent.

    Branch types
    ─────────────
    main        Confirmed iliac (role iliac_left / iliac_right / main)
    renal       Confirmed renal vein (role renal_vein / renal_fragment)
    side_short  Non-main branch arc < 30 mm
    side        Everything else

    Feature vector (14 components)
    ───────────────────────────────────────────────────────────────────
    V  area/gradient score      (already 0-1)
    A  angle score              (already 0-1)
    L  length score             (already 0-1)
    T  topology score           (already 0-1)
    G  geometry score           (already 0-1)
    S  stability score          (already 0-1)
    C  curvature score          (already 0-1)
    Z  zone score               (already 0-1)
    D  div/lateral score        (already 0-1)
    AD interaction: A(lat) × D(div)
    VG interaction: V(area) × G(geom)
    fl_area_low      flag penalty (0 or 1)
    fl_zone_distant  flag penalty (0 or 1)
    fl_divergence_low flag penalty (0 or 1)

    Training
    ────────
    After each run _logOstiumTrainingRecord() appends a compact JSON
    record to  <module_dir>/ostium_logreg_data.json.
    Once ≥ MIN_TRAIN_N records are accumulated, train_step() runs one
    mini-batch SGD pass (last BATCH_SIZE records) and writes updated
    weights back to  <module_dir>/ostium_logreg_weights.json.

    Safeguards
    ──────────
    • Learning rate  LR = 0.02  (slow — prevents drift)
    • Weight clamp   [-4, +4]   (prevents runaway)
    • Bias clamp     [-3, +3]
    • Separate weight vectors per branch type — types don't pollute each other
    • Min-data guard: no training until MIN_TRAIN_N = 6 records per type
    """

    # ── Feature keys (order determines weight vector index) ─────────────
    FEATURE_KEYS = [
        "V",
        "A",
        "L",
        "T",
        "G",
        "S",
        "C",
        "Z",
        "D",  # 9 geometry signals
        "AD",  # interaction: A(lat) × D(div)
        "VG",  # interaction: V(area) × G(geom)
        "fl_area_low",  # flag penalties
        "fl_zone_distant",
        "fl_divergence_low",
    ]
    BRANCH_TYPES = ("main", "renal", "side_short", "side")

    # ── Prior weights ────────────────────────────────────────────────────
    # Calibrated so a "typical good branch" (features avg 0.65) gives
    # z ≈ 0 → sigmoid = 0.5, leaving room to discriminate.
    # A perfect branch (all 1.0) → z ≈ +2.5 → sigmoid ≈ 0.92 (max).
    # A clear negative (all 0.2) → z ≈ -2.5 → sigmoid ≈ 0.08 (min).
    # Order: V     A     L     T     G     S     C     Z     D
    #        AD    VG    fl_area  fl_zone  fl_div
    _PRIORS: Dict[str, Dict[str, Any]] = {
        "main": {
            "bias": -2.0,
            "w": [
                0.35, 0.50, 0.50, 0.75, 0.90,
                0.25, 0.20, 0.50, 0.15, 0.20,
                0.30, -0.50, -0.60, -0.20,
            ],
        },
        "renal": {
            "bias": -2.2,
            "w": [
                0.25, 0.45, 0.50, 0.60, 0.75,
                0.40, 0.15, 0.55, 0.90, 0.70,
                0.25, -0.60, -0.65, -0.30,
            ],
        },
        "side_short": {
            "bias": -2.1,
            "w": [
                0.20, 0.35, 0.20, 0.60, 0.60,
                0.45, 0.15, 0.45, 0.75, 0.50,
                0.20, -0.55, -0.55, -0.35,
            ],
        },
        "side": {
            "bias": -2.0,
            "w": [
                0.40, 0.50, 0.50, 0.50, 0.75,
                0.35, 0.25, 0.50, 0.40, 0.30,
                0.30, -0.55, -0.60, -0.25,
            ],
        },
    }

    LR = 0.02          # learning rate — slow, prevents drift
    BATCH_SIZE = 30    # recent records per training pass
    MIN_TRAIN_N = 6    # minimum records before first training
    W_CLAMP = 4.0      # weight clamp
    BIAS_CLAMP = 3.0   # bias clamp

    def __init__(self, weights_path: Optional[str] = None):
        # Deep-copy priors as mutable state
        self._weights: Dict[str, Dict[str, Any]] = {
            btype: {
                "bias": d["bias"],
                "w": list(d["w"]),
            }
            for btype, d in self._PRIORS.items()
        }
        self._weights_path = weights_path
        if weights_path:
            self._load(weights_path)

    # ── Public API ───────────────────────────────────────────────────────

    def classify_branch(self, role: str, arc_mm: Optional[float]) -> str:
        """Map pipeline role + arc_mm → one of the four branch type keys."""
        if role in ("main", "iliac_left", "iliac_right"):
            return "main"
        if role in ("renal_vein", "renal_fragment"):
            return "renal"
        if arc_mm is not None and arc_mm < 30.0:
            return "side_short"
        return "side"

    def build_features(
        self,
        components: Dict[str, float],
        flags: Sequence[str],
    ) -> Dict[str, float]:
        """Build the full 14-element feature dict from confidence components + flags.

        Parameters
        ----------
        components : dict   V/A/L/T/G/S/C/Z/D from ostium_confidence['components']
        flags      : list   list of flag name strings from ostium_confidence['flags']

        Returns
        -------
        dict with all 14 FEATURE_KEYS populated, values float in [0, 1].
        """
        f = {k: float(components.get(k, 0.0)) for k in "VALTGSCZD"}
        f["AD"] = f["A"] * f["D"]   # lateral × divergence
        f["VG"] = f["V"] * f["G"]   # area quality × geometry sanity
        f["fl_area_low"] = 1.0 if "area_low" in flags else 0.0
        f["fl_zone_distant"] = 1.0 if "zone_distant" in flags else 0.0
        f["fl_divergence_low"] = 1.0 if "divergence_low" in flags else 0.0
        return f

    def score(self, btype: str, features: Dict[str, float]) -> float:
        """Return a calibrated confidence float in [0, 1].

        Uses output compression p → 0.15 + 0.70 * sigmoid(z) so the range
        is [0.15, 0.85] rather than [0, 1].  This prevents saturation at
        either extreme during the early (few data) phase.

        Parameters
        ----------
        btype    : str   one of BRANCH_TYPES
        features : dict  keys per FEATURE_KEYS, values float
        """
        W = self._weights.get(btype, self._weights["side"])
        z = W["bias"]
        for i, k in enumerate(self.FEATURE_KEYS):
            z += W["w"][i] * float(features.get(k, 0.0))
        raw = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))
        return 0.15 + 0.70 * raw

    def train_step(self, records: List[Dict[str, Any]], btype: str) -> None:
        """One SGD pass over *records* for branch type *btype*.

        Each record must have:
            features : dict  all 14 FEATURE_KEYS populated
            y        : float target label.  True positives: 1.0.
                       Near-miss negatives: 0.3–0.5.  Hard negatives: 0.0.
        """
        if len(records) < self.MIN_TRAIN_N:
            return
        batch = records[-self.BATCH_SIZE:]
        W = self._weights.setdefault(
            btype,
            {
                "bias": self._PRIORS.get(btype, self._PRIORS["side"])["bias"],
                "w": list(self._PRIORS.get(btype, self._PRIORS["side"])["w"]),
            },
        )
        for rec in batch:
            feats = rec.get("features", {})
            y = float(rec.get("y", 1.0 if rec.get("accepted", False) else 0.0))
            z = W["bias"]
            for i, k in enumerate(self.FEATURE_KEYS):
                z += W["w"][i] * float(feats.get(k, 0.0))
            z = max(-20.0, min(20.0, z))
            pred_raw = 1.0 / (1.0 + math.exp(-z))
            err = y - pred_raw
            W["bias"] = max(
                -self.BIAS_CLAMP, min(self.BIAS_CLAMP, W["bias"] + self.LR * err)
            )
            for i, k in enumerate(self.FEATURE_KEYS):
                dw = self.LR * err * float(feats.get(k, 0.0))
                W["w"][i] = max(-self.W_CLAMP, min(self.W_CLAMP, W["w"][i] + dw))
        n_pos = sum(
            1 for r in batch
            if r.get("y", 1.0 if r.get("accepted", False) else 0.0) > 0.5
        )
        print(
            f"[LogReg] trained {btype} on {len(batch)} records "
            f"(pos={n_pos} neg={len(batch) - n_pos}) "
            f"bias={W['bias']:.3f}"
        )

    def save(self, path: str) -> None:
        """Write weights JSON to *path*."""
        try:
            with open(path, "w") as fh:
                json.dump(self._weights, fh, indent=2)
            print(f"[LogReg] weights saved → {path}")
        except Exception as exc:
            print(f"[LogReg] save failed: {exc}")

    # ── Private helpers ──────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as fh:
                loaded = json.load(fh)
            for btype in self.BRANCH_TYPES:
                if btype in loaded:
                    ld = loaded[btype]
                    self._weights[btype]["bias"] = float(ld["bias"])
                    w_loaded = [float(v) for v in ld["w"]]
                    n = len(self.FEATURE_KEYS)
                    if len(w_loaded) < n:
                        # Old 9-feature file: extend with prior values for new keys
                        prior_w = self._PRIORS.get(btype, self._PRIORS["side"])["w"]
                        w_loaded += prior_w[len(w_loaded):]
                    self._weights[btype]["w"] = w_loaded[:n]
            print(f"[LogReg] weights loaded ← {path}")
        except Exception as exc:
            print(f"[LogReg] load failed (using priors): {exc}")


# ── Backward-compatible alias ────────────────────────────────────────────────
# VesselAnalyzer.py instantiates this as _OstiumLogReg(weights_path=...).
# After swapping the import, no call sites need changing.
_OstiumLogReg = OstiumLogReg


# ── Slicer module-scanner guard ───────────────────────────────────────────────
class ostium_logreg:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "ostium_logreg"
            parent.hidden = True  # hide from Slicer module list
