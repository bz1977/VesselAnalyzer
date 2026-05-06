from vessel_pipeline import PipelineStage
from vessel_state import VesselState

class RadiusMapStage(PipelineStage):
    """
    Computes deterministic geometric properties: radius mapping and wall thickness.
    Runs cleanly from state variables without modifying the MRML scene directly 
    (except where legacy logic hasn't been fully decoupled).
    """
    def __init__(self, logic_instance):
        self.logic = logic_instance
        
    @property
    def mode(self) -> str:
        return "compute"

    def run(self, state: VesselState) -> VesselState:
        if not state.points or not state.diameters:
            print("[RadiusMapStage] Missing points or diameters. Skipping.")
            return state
            
        print("[RadiusMapStage] Starting radius map computation...")
        # Currently wraps the mixin functions
        # This executes the multi-ray logic and surface map generation
        if hasattr(self.logic, 'updateSurfaceMap'):
            self.logic.updateSurfaceMap()
            
        return state


# Slicer module boilerplate to prevent Qt plugin errors
try:
    import slicer
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule
    class vessel_radius_map_stage(ScriptedLoadableModule):
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
    class vessel_radius_map_stage(ScriptedLoadableModule):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "Hidden"
            parent.hidden = True
except ImportError:
    pass
