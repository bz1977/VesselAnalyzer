from vessel_pipeline import PipelineStage
from vessel_state import VesselState

class CenterlineStage(PipelineStage):
    """Pipeline stage for extracting and loading centerlines.
    
    During Phase 1 of the migration, this stage wraps the existing 
    CenterlineMixin logic to ensure state transitions are correctly tracked.
    """
    
    def __init__(self, logic_instance, strategies=None, score_threshold=8.0):
        # We hold a reference to the main logic object to reuse existing 
        # complex VTK/VMTK routines during the incremental migration.
        self.logic = logic_instance
        self.strategies = strategies or []
        self.score_threshold = score_threshold


    def run(self, state: VesselState) -> VesselState:
        """Run centerline extraction and graph building.
        
        Note: The actual extraction often happens via user interaction or 
        per_branch_centerline_pipeline.py. This stage ensures that any 
        centerline loading updates the state correctly.
        """
        
        if not self.strategies:
            # Fallback to legacy path if no strategies provided
            centerline_node = getattr(self.logic, "_centerlineNode", None)
            if centerline_node:
                self.logic.loadCenterline(
                    centerline_node, 
                    modelNode=state.model_node,
                    segModelNode=state.seg_model_node
                )
            return state

        best_result = None
        comparisons = []

        for strategy in self.strategies:
            result = strategy.run(state)
            
            comparisons.append({
                "name": strategy.name,
                "score": result.score,
                "success": result.success
            })
            
            # Log individual result
            if self.logic.logger:
                self.logic.logger.log_event("STRATEGY_RESULT", comparisons[-1])
            
            if result.success:
                if best_result is None or result.score > best_result.score:
                    best_result = result
                    best_strategy_name = strategy.name
                    
            # Early exit if good enough
            if result.success and result.score >= self.score_threshold:
                break
                
        # Log comparison
        if self.logic.logger:
            self.logic.logger.log_event("STRATEGY_COMPARISON", {
                "strategies": comparisons,
                "selected": best_strategy_name if best_result else "none"
            })
            
        if best_result and best_result.success:
            # COMMIT phase: Only the winner mutates the state/MRML
            if hasattr(self.logic, "_commit_strategy_result"):
                self.logic._commit_strategy_result(best_result, state)
            else:
                # If no commit logic exists yet, just update state directly
                state.centerlines = best_result.centerlines
        else:
            print("[CenterlineStage] All strategies failed.")
        
        return state


# Slicer module boilerplate to prevent Qt plugin errors
try:
    import slicer
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule
    class vessel_centerline_stage(ScriptedLoadableModule):
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
    class vessel_centerline_stage(ScriptedLoadableModule):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "Hidden"
            parent.hidden = True
except ImportError:
    pass
