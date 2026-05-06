import os

files = [
    'vessel_centerline_stage.py',
    'vessel_debug_log.py',
    'vessel_findings_stage.py',
    'vessel_ostium_stage.py',
    'vessel_pipeline.py',
    'vessel_radius_map_stage.py',
    'vessel_state.py',
    'centerline_strategies.py',
    'vessel_branch_validation.py',
]

for f in files:
    if not os.path.exists(f):
        continue
    classname = f.replace('.py', '')
    boilerplate = f"""

# Slicer module boilerplate to prevent Qt plugin errors
try:
    import slicer
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule
    class {classname}(ScriptedLoadableModule):
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "Hidden"
            parent.hidden = True
except ImportError:
    pass
"""
    with open(f, 'a') as fp:
        fp.write(boilerplate)
    print(f"Appended boilerplate to {f}")
