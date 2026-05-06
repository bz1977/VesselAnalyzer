from vessel_pipeline import PipelineStage
from vessel_state import VesselState

class FindingsStage(PipelineStage):
    """
    Analyzes the computed geometry to identify clinical findings (stenosis, aneurysms, etc.).
    Does not mutate geometry or MRML representations directly.
    """
    def __init__(self, logic_instance):
        self.logic = logic_instance

    @property
    def mode(self) -> str:
        return "analysis"

    def run(self, state: VesselState) -> VesselState:
        if not state.points:
            print("[FindingsStage] Missing points. Skipping.")
            return state
            
        print("[FindingsStage] Running geometric analysis for findings...")
        if hasattr(self.logic, 'updateFindings'):
            self.logic.updateFindings()
            
        return state


# Slicer module boilerplate to prevent Qt plugin errors
try:
    import slicer
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule
    class vessel_findings_stage(ScriptedLoadableModule):
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
    class vessel_findings_stage(ScriptedLoadableModule):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "Hidden"
            parent.hidden = True
except ImportError:
    pass
