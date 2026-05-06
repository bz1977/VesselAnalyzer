from vessel_pipeline import PipelineStage
from vessel_state import VesselState

class OstiumStage(PipelineStage):
    """Pipeline stage for detecting branch ostia and roles.
    
    During Phase 1 of the migration, this stage wraps the existing 
    OstiumMixin logic.
    """
    
    def __init__(self, logic_instance):
        self.logic = logic_instance

    def run(self, state: VesselState) -> VesselState:
        """Run ostium detection algorithms.
        
        This delegates to the existing `computeOstia()` method in the 
        OstiumMixin. The mixin updates `self.points`, `self.ostiumConfidenceMap`, 
        etc. which are properties that mutate the `state` object.
        """
        
        # If there are no points, we can't run the ostium detector
        if not state.points:
            return state
            
        # Call the legacy computeOstia method
        # We check if it exists just to be safe
        if hasattr(self.logic, 'computeOstia'):
            self.logic.computeOstia()
            
        return state


# Slicer module boilerplate to prevent Qt plugin errors
try:
    import slicer
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule
    class vessel_ostium_stage(ScriptedLoadableModule):
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
    class vessel_ostium_stage(ScriptedLoadableModule):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "Hidden"
            parent.hidden = True
except ImportError:
    pass
