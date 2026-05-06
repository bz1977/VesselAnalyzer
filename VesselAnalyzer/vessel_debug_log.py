import json
import datetime
import os

class StructuredLogger:
    """Writes structured debug events to a per-run .jsonl file.
    
    This replaces the console print-based debugging with searchable, 
    diffable logs stored on disk, allowing retrospective analysis of pipeline runs.
    """
    
    def __init__(self, log_dir: str = None):
        if log_dir:
            self.log_dir = log_dir
        else:
            # Default to a 'logs' directory next to the current file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.log_dir = os.path.join(base_dir, "logs")
            
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filepath = os.path.join(self.log_dir, f"vessel_run_{timestamp}.jsonl")
        
    def log_event(self, event_type: str, data: dict = None):
        """Append an event to the JSONL log file."""
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event_type
        }
        if data:
            event.update(data)
            
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[StructuredLogger] Error writing to log: {e}")


# Slicer module boilerplate to prevent Qt plugin errors
try:
    import slicer
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule
    class vessel_debug_log(ScriptedLoadableModule):
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
    class vessel_debug_log(ScriptedLoadableModule):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "Hidden"
            parent.hidden = True
except ImportError:
    pass
