from typing import Any, Protocol
from vessel_state import VesselState

class DebugLogger(Protocol):
    def log_event(self, event_type: str, data: dict):
        ...

class PipelineStage:
    """Base class for all VesselAnalyzer pipeline stages."""
    
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def mode(self) -> str:
        """Type of stage: 'compute', 'analysis', or 'visual'"""
        return "compute"

    def run(self, state: VesselState) -> VesselState:
        """Execute the stage on the given state and return the updated state."""
        raise NotImplementedError("Stages must implement run()")

class VesselPipeline:
    """Orchestrates sequential execution of pipeline stages with debug hooks."""
    
    def __init__(self, stages: list[PipelineStage], debug: DebugLogger = None):
        self.stages = stages
        self.debug = debug

    def run(self, state: VesselState) -> VesselState:
        """Run all stages in sequence on the state object."""
        for stage in self.stages:
            if self.debug:
                self.debug.log_event("STAGE_START", {"stage": stage.name})

            # Execute the stage
            state = stage.run(state)
            
            # Increment version to signify mutation
            state.version += 1

            if self.debug:
                self.debug.log_event("STAGE_END", {
                    "stage": stage.name,
                    "version": state.version
                })
        
        # Notify UI or other listeners that the pipeline has completed
        state.notify()
        return state


# Slicer module boilerplate to prevent Qt plugin errors
try:
    import slicer
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule
    class vessel_pipeline(ScriptedLoadableModule):
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
    class vessel_pipeline(ScriptedLoadableModule):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "Hidden"
            parent.hidden = True
except ImportError:
    pass
