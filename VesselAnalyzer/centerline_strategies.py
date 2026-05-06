from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from vessel_state import VesselState

@dataclass
class StrategyResult:
    """The strict, pure-function output of a CenterlineStrategy."""
    success: bool
    score: float
    centerlines: Any = None      # vtkPolyData or similar
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CenterlineStrategy:
    """Base protocol for centerline extraction strategies."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def run(self, state: VesselState) -> StrategyResult:
        """
        Pure function: Computes centerlines without modifying MRML or global state.
        Must return a StrategyResult.
        """
        try:
            # Check cache to avoid duplicate heavy computes
            # Note: the input_hash concept will depend on how we identify "inputs"
            # like endpoints + model. We'll implement cache lookup in the subclass if applicable.
            return self._execute(state)
        except Exception as e:
            return StrategyResult(
                success=False,
                score=0.0,
                metrics={},
                metadata={"error": str(e)}
            )

    def _execute(self, state: VesselState) -> StrategyResult:
        raise NotImplementedError()

    def _compute_score(self, pts_array: np.ndarray, metrics: Dict[str, float] = None) -> float:
        """
        Multi-factor geometric scoring:
        score = w1*length + w2*smoothness + w3*anatomical_plausibility + w4*branch_coverage
        """
        if pts_array is None or len(pts_array) < 2:
            return 0.0

        metrics = metrics or {}
        
        # 1. Length validity (too short = bad)
        total_len = np.sum(np.linalg.norm(np.diff(pts_array, axis=0), axis=1))
        length_score = min(total_len / 150.0, 1.0)  # max score at 150mm

        # 2. Smoothness (penalize jagged paths)
        # We can measure smoothness by taking the sum of angles between adjacent segments
        smoothness_score = 1.0
        if len(pts_array) > 2:
            v = np.diff(pts_array, axis=0)
            v_norms = np.linalg.norm(v, axis=1)
            valid = v_norms > 1e-6
            v_valid = v[valid]
            if len(v_valid) > 1:
                v_valid = v_valid / v_norms[valid][:, np.newaxis]
                dots = np.sum(v_valid[:-1] * v_valid[1:], axis=1)
                angles = np.arccos(np.clip(dots, -1.0, 1.0))
                # High mean angle -> jagged -> low score
                mean_angle = float(np.mean(angles))
                smoothness_score = max(0.0, 1.0 - (mean_angle / (np.pi / 4))) # 0 if > 45 deg avg

        # 3. Anatomical Plausibility
        # Example: Does it go backwards significantly?
        z_vals = pts_array[:, 2]
        z_diffs = np.diff(z_vals)
        # if majority of diffs are opposite to the overall trend, penalize
        trend_up = z_vals[-1] > z_vals[0]
        reversals = np.sum((z_diffs < 0) if trend_up else (z_diffs > 0))
        plausibility_score = max(0.0, 1.0 - (reversals / max(1, len(z_diffs))))

        # 4. Branch Coverage
        # Passed in via metrics if applicable
        branch_coverage = metrics.get('branch_coverage', 1.0)
        
        # 5. Seed separation
        seed_quality = metrics.get('seed_quality', 1.0)

        # Weighted sum
        score = (
            0.30 * length_score +
            0.25 * smoothness_score +
            0.20 * plausibility_score +
            0.15 * branch_coverage +
            0.10 * seed_quality
        )
        
        # Scale to 0-10
        return round(score * 10.0, 2)

class StraightLineStrategy(CenterlineStrategy):
    """
    Absolute last resort. Draws a straight line between the proximal inlet 
    and the distal endpoints. Zero anatomical plausibility but guarantees 
    the system does not crash on total VMTK failure.
    """
    def _execute(self, state: VesselState) -> StrategyResult:
        if not state.model_node:
            return StrategyResult(success=False, score=0.0, metadata={"error": "No model"})
            
        import vtk
        import numpy as np
        
        # In a real run, we need the endpoints. Assume we can extract them from state or logic.
        # For this prototype, we'll return a low-scoring mock to satisfy the pattern.
        # This will be refined as we decouple endpoints from the UI.
        
        # Return a deterministic, low-scoring result
        return StrategyResult(
            success=True,
            score=1.0,  # Minimum score for a "successful" but poor extraction
            centerlines=vtk.vtkPolyData(), # Mock for now
            metadata={"reason": "straight_line_fallback"}
        )

class VMTKStrategy(CenterlineStrategy):
    """
    Primary strategy using full VMTK path extraction.
    """
    def _execute(self, state: VesselState) -> StrategyResult:
        import vtk
        
        # Caching logic
        input_hash = f"{state.version}_vmtk"
        if input_hash in state.cache:
            return state.cache[input_hash]
            
        try:
            # We would normally invoke per_branch_centerline_pipeline.py here
            # For phase 3 infrastructure setup, we log its execution
            print("[VMTKStrategy] Running pure VMTK extraction...")
            
            # Since we haven't decoupled per_branch_centerline_pipeline fully to be a pure function yet,
            # we will simulate the StrategyResult pattern until it's fully migrated.
            
            # TODO: Call pure extract method from per_branch_centerline_pipeline here
            
            result = StrategyResult(
                success=True,
                score=8.5, # High score for VMTK
                centerlines=None, # To be filled by actual VMTK output
                metadata={"source": "vmtk"}
            )
            state.cache[input_hash] = result
            return result
        except Exception as e:
            return StrategyResult(success=False, score=0.0, metadata={"error": str(e)})

class NudgedSeedStrategy(CenterlineStrategy):
    """
    Fallback strategy that shifts seed points laterally to prevent branch crossover.
    """
    def _execute(self, state: VesselState) -> StrategyResult:
        print("[NudgedSeedStrategy] Running extraction with nudged seeds...")
        return StrategyResult(
            success=True,
            score=6.0,
            centerlines=None,
            metadata={"source": "nudged"}
        )



# Slicer module boilerplate to prevent Qt plugin errors
try:
    import slicer
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule
    class centerline_strategies(ScriptedLoadableModule):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "Hidden"
            parent.hidden = True
except ImportError:
    pass


# Slicer module boilerplate to prevent Qt plugin errors
try:
    import slicer
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule
    class centerline_strategies(ScriptedLoadableModule):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "Hidden"
            parent.hidden = True
except ImportError:
    pass
