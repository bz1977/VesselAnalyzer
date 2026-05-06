from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable
import uuid

@dataclass
class VesselState:
    """Central mutable state for the VesselAnalyzer pipeline.
    
    Acts as the single source of truth for the Slicer UI and all pipeline
    stages. Increments version on mutation to allow UI synchronization.
    """
    model_node: Any = None
    seg_model_node: Any = None
    centerlines: Any = None
    
    points: List[tuple] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    diameters: List[float] = field(default_factory=list)
    _diam_minor: List[float] = field(default_factory=list)
    _diam_major: List[float] = field(default_factory=list)
    
    branches: List[tuple] = field(default_factory=list)
    active_branch: int = -1
    vessel_type: str = "arterial"
    
    branch_meta: Dict[int, Dict] = field(default_factory=dict)
    ostia: Dict = field(default_factory=dict)
    manual_branch_ranges: Dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)
    
    version: int = 0
    listeners: List[Callable] = field(default_factory=list)

    def notify(self):
        """Invoke all registered callbacks to signal a state change."""
        for cb in self.listeners:
            cb(self)

    def add_listener(self, cb: Callable):
        """Register a callback to be notified on state changes."""
        if cb not in self.listeners:
            self.listeners.append(cb)

    def remove_listener(self, cb: Callable):
        """Unregister a callback."""
        if cb in self.listeners:
            self.listeners.remove(cb)


# Slicer module boilerplate to prevent Qt plugin errors
try:
    import slicer
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule
    class vessel_state(ScriptedLoadableModule):
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
    class vessel_state(ScriptedLoadableModule):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "Hidden"
            parent.hidden = True
except ImportError:
    pass
