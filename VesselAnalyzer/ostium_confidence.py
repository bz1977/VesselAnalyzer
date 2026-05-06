# ostium_confidence.py
#
# Slicer scripted-loadable module wrapper.
# Actual logistic-regression scoring is in ostium_logreg.py (OstiumLogReg).
#
# This file exists so Slicer's module auto-discovery finds a valid class
# whose name matches the filename stem.  Without it Slicer raises:
#   RuntimeError: class ostium_confidence was not found in file ostium_confidence.py
#
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
)


class ostium_confidence(ScriptedLoadableModule):
    """Slicer module wrapper — ostium confidence scoring utilities."""

    def __init__(self, parent):
        super().__init__(parent)
        parent.title        = "Ostium Confidence"
        parent.categories   = ["Vascular"]
        parent.dependencies = []
        parent.contributors = ["VesselAnalyzer"]
        parent.helpText     = (
            "Ostium detection confidence scoring for VesselAnalyzer. "
            "Logistic-regression scorer loaded from ostium_logreg_weights.json "
            "via OstiumLogReg in ostium_logreg.py."
        )
        parent.acknowledgementText = ""


class ostium_confidenceWidget(ScriptedLoadableModuleWidget):
    """Minimal widget — this module is used programmatically, not via GUI."""

    def setup(self):
        super().setup()

    def cleanup(self):
        pass
