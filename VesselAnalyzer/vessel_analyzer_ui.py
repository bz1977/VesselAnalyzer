"""
vessel_analyzer_ui.py — UI construction for VesselAnalyzerWidget.

Extracted from VesselAnalyzer.py to reduce main file size.
Contains the full _setupUI() method body — pure Qt/ctk widget creation,
no pipeline logic, no closures over external state.

Usage in VesselAnalyzer.py
--------------------------
Replace the _setupUI method body with a single delegation call::

    def _setupUI(self):
        from vessel_analyzer_ui import build_ui
        build_ui(self)

The ``build_ui(widget)`` function receives the widget instance as its only
argument and assigns all self.* attributes exactly as _setupUI did.
All call sites (``self._setupUI()`` in ``setup()``) require zero changes.
"""

import os
import sys
import math
import vtk
import slicer
import ctk
import qt
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


# ── Centerline extraction progress bar ───────────────────────────────────────
#
# CenterlineProgressInterceptor wraps sys.stdout so that every print() call
# from the pipeline (which lives in vessel_centerline_mixin.py) is examined for
# the known [PerBranchCL] stage markers.  When a marker is matched the progress
# bar on the widget is updated automatically — no changes to the mixin needed.
#
# Usage (from onExtractCenterline in vessel_centerline_widget_mixin.py):
#
#   from vessel_analyzer_ui import CenterlineProgressInterceptor
#   with CenterlineProgressInterceptor():
#       self.logic.extractCenterline(...)
#
# Or call the standalone helpers directly from wherever convenient:
#
#   from vessel_analyzer_ui import cl_progress_start, cl_progress_done

# Ordered list of (substring_to_watch, progress_pct, status_text) tuples.
# They are checked in order; first match wins for each printed line.
_CL_PROGRESS_STAGES = [
    # Stage 1 — preprocessing
    ("Step 1/4: Preprocessing mesh",        5,  "Preprocessing mesh…"),
    ("Preprocessed mesh:",                  15, "Mesh preprocessed ✓"),
    # Stage 1b — bifurcation solve
    ("Step 1/4: Locating bifurcation",      20, "Locating bifurcation…"),
    ("_vmtk_bif_from_3pts: bif=",           28, "Bifurcation located ✓"),
    # Stage 2 — per-branch solves
    ("Step 2/4: Branch 1 of",               35, "Solving branch 1 / 4…"),
    ("Step 2/4: Branch 2 of",               50, "Solving branch 2 / 4…"),
    ("Step 2/4: Branch 3 of",               63, "Solving branch 3 / 4…"),
    ("Step 2/4: Branch 4 of",               75, "Solving branch 4 / 4…"),
    # Stage 3 — junction snap
    ("Step 3/4: Junction snap",             83, "Junction snap…"),
    # Stage 4 — merging
    ("Step 4/4: Merging",                   90, "Merging branches…"),
    # Complete
    ("Pipeline complete",                   100, "Centerline extracted ✓"),
]


def _cl_get_bar_and_label():
    """Return (bar, label) widgets from the active VesselAnalyzerWidget, or (None, None)."""
    try:
        w = getattr(slicer.modules, "VesselAnalyzerWidget", None)
        if w is None:
            return None, None
        bar   = getattr(w, "extractProgressBar",  None)
        label = getattr(w, "extractStatusLabel",   None)
        return bar, label
    except Exception:
        return None, None


def _cl_set(pct, text=None):
    """Push *pct* (0-100) to the bar and optionally update the status label."""
    bar, label = _cl_get_bar_and_label()
    if bar is None:
        return
    bar.setValue(int(pct))
    bar.setVisible(pct > 0)
    if text and label:
        label.setText(text)
        label.setStyleSheet(
            "color: #27ae60; font-weight: bold;" if pct >= 100
            else "color: #2980b9;"
        )
    try:
        qt.QApplication.processEvents()
    except Exception:
        pass


def cl_progress_start():
    """Show the progress bar at 0 % and set the status label.  Call before extraction."""
    bar, label = _cl_get_bar_and_label()
    if bar:
        bar.setValue(0)
        bar.setVisible(True)
    if label:
        label.setText("Extracting centerline…")
        label.setStyleSheet("color: #2980b9;")
    try:
        qt.QApplication.processEvents()
    except Exception:
        pass


def cl_progress_done(success=True):
    """Set bar to 100 % (or 0 % on failure) and hide it."""
    if success:
        _cl_set(100, "Centerline extracted ✓")
        try:
            qt.QApplication.processEvents()
        except Exception:
            pass
    bar, label = _cl_get_bar_and_label()
    if bar:
        bar.setVisible(False)
        bar.setValue(0)
    if label and not success:
        label.setText("Extraction failed")
        label.setStyleSheet("color: #e74c3c; font-weight: bold;")


class CenterlineProgressInterceptor:
    """Context manager that intercepts print() output and drives the progress bar.

    Wraps sys.stdout; every line written is checked against _CL_PROGRESS_STAGES.
    The bar is reset on __enter__ and hidden on __exit__.

    Example
    -------
    from vessel_analyzer_ui import CenterlineProgressInterceptor
    with CenterlineProgressInterceptor():
        self.logic.extractCenterline(modelNode, endpointsNode)
    """

    class _Tee:
        """Write to both the real stdout and run the stage-detection callback."""
        def __init__(self, real, callback):
            self._real     = real
            self._callback = callback
            self._buf      = ""

        def write(self, text):
            self._real.write(text)
            self._buf += text
            # Process complete lines
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._callback(line)

        def flush(self):
            self._real.flush()

        def __getattr__(self, name):
            return getattr(self._real, name)

    def __enter__(self):
        cl_progress_start()
        self._real_stdout = sys.stdout
        sys.stdout = self._Tee(self._real_stdout, self._on_line)
        return self

    def __exit__(self, exc_type, *_):
        sys.stdout = self._real_stdout
        cl_progress_done(success=(exc_type is None))

    @staticmethod
    def _on_line(line):
        for substr, pct, text in _CL_PROGRESS_STAGES:
            if substr in line:
                _cl_set(pct, text)
                break


def build_ui(self):
    """Construct all UI widgets and assign them to *self*.

    Equivalent to ``VesselAnalyzerWidget._setupUI(self)``.
    Called by the thin delegation stub in VesselAnalyzer.py.
    """
    # =========================================================
    # STEP 1 — INPUT
    # =========================================================
    step1 = ctk.ctkCollapsibleButton()
    step1.text = "Step 1 — Select Vessel Model"
    self.layout.addWidget(step1)
    step1Layout = qt.QFormLayout(step1)

    self.modelSelector = slicer.qMRMLNodeComboBox()
    self.modelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.modelSelector.selectNodeUponCreation = False
    self.modelSelector.addEnabled = False
    self.modelSelector.removeEnabled = False
    self.modelSelector.noneEnabled = True
    self.modelSelector.setMRMLScene(slicer.mrmlScene)
    self.modelSelector.setToolTip("Select your vessel surface model")
    step1Layout.addRow("Vessel Model:", self.modelSelector)

    modelInfoLayout = qt.QHBoxLayout()
    self.modelInfoLabel = qt.QLabel("No model selected")
    self.modelInfoLabel.setStyleSheet("color: gray;")
    modelInfoLayout.addWidget(self.modelInfoLabel)
    step1Layout.addRow("Info:", modelInfoLayout)

    # Vessel type selector — placed here so it governs the full pipeline,
    # including finding thresholds, renal tagging, and stent suggestions.
    # Default is blank (unselected) — user must choose before analysis.
    self.vesselTypeCombo = qt.QComboBox()
    self.vesselTypeCombo.addItems(
        ["-- Select vessel type --", "Arterial", "Venous"]
    )
    self.vesselTypeCombo.setCurrentIndex(0)
    self.vesselTypeCombo.setToolTip(
        "Choose before running analysis.\n"
        "Arterial: standard thresholds (aneurysm >=1.5x, ectasia >=1.2x).\n"
        "Venous: raised thresholds + bifurcation expansion suppressed."
    )
    self.vesselTypeCombo.connect(
        "currentIndexChanged(int)", self.onVesselTypeChanged
    )
    step1Layout.addRow("Vessel Type:", self.vesselTypeCombo)

    self.vesselTypeHintLabel = qt.QLabel(
        "⚠  Set vessel type before running analysis"
    )
    self.vesselTypeHintLabel.setStyleSheet("color: #e67e22; font-size: 11px;")
    step1Layout.addRow("", self.vesselTypeHintLabel)

    # =========================================================
    # STEP 2 — EXTRACT CENTERLINE
    # =========================================================
    step2 = ctk.ctkCollapsibleButton()
    step2.text = "Step 2 — Extract Centerline"
    self.layout.addWidget(step2)
    step2Layout = qt.QFormLayout(step2)

    # Endpoints
    self.endpointsSelector = slicer.qMRMLNodeComboBox()
    self.endpointsSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.endpointsSelector.selectNodeUponCreation = True
    self.endpointsSelector.addEnabled = True
    self.endpointsSelector.removeEnabled = False
    self.endpointsSelector.noneEnabled = True
    self.endpointsSelector.setMRMLScene(slicer.mrmlScene)
    self.endpointsSelector.setToolTip("Select or create endpoint markups")
    step2Layout.addRow("Endpoints:", self.endpointsSelector)

    # Segmentation model selector — used only for endpoint detection.
    # Must point at the open-boundary segmentation surface (from Segment
    # Editor), NOT at the centerline tube model which is closed.
    # Kept separate from the Vessel Model selector in Step 1 so the user
    # Auto-detect button (full width)
    self.autoDetectButton = qt.QPushButton("🔍 Auto-Detect Endpoints")
    self.autoDetectButton.setStyleSheet(
        "background-color: #8e44ad; color: white; font-weight: bold; padding: 7px; font-size: 13px;"
    )
    self.autoDetectButton.setToolTip(
        "Detect vessel opening centroids from boundary rings of the "
        "Vessel Model. Uses open-boundary ring detection "
        "(vtkFeatureEdges) — accurate for all branch types including renal veins."
    )
    self.autoDetectButton.enabled = True
    step2Layout.addRow(self.autoDetectButton)

    # Manual place / cancel / clear buttons
    placeLayout = qt.QHBoxLayout()
    self.placeEndpointsButton = qt.QPushButton("✚ Place Manually")
    self.placeEndpointsButton.setStyleSheet(
        "background-color: #3498db; color: white; font-weight: bold; padding: 5px;"
    )
    self.placeEndpointsButton.setToolTip(
        "Manually click on vessel openings to place endpoints"
    )
    self.cancelPlaceButton = qt.QPushButton("■ Stop")
    self.cancelPlaceButton.setStyleSheet(
        "background-color: #e67e22; color: white; font-weight: bold; padding: 5px;"
    )
    self.cancelPlaceButton.setVisible(False)
    self.clearEndpointsButton = qt.QPushButton("✕ Clear All")
    self.clearEndpointsButton.setStyleSheet(
        "background-color: #e74c3c; color: white; padding: 5px;"
    )
    placeLayout.addWidget(self.placeEndpointsButton)
    placeLayout.addWidget(self.cancelPlaceButton)
    placeLayout.addWidget(self.clearEndpointsButton)
    step2Layout.addRow(placeLayout)

    self.endpointCountLabel = qt.QLabel("0 endpoints placed")
    self.endpointCountLabel.setStyleSheet("color: gray;")
    step2Layout.addRow("", self.endpointCountLabel)

    # Preprocess options
    self.preprocessCheck = qt.QCheckBox("Pre-process surface (recommended)")
    self.preprocessCheck.setChecked(True)
    step2Layout.addRow(self.preprocessCheck)

    # Extract button
    self.extractButton = qt.QPushButton("⚙ Extract Centerline")
    self.extractButton.setStyleSheet(
        "background-color: #2ecc71; color: white; font-weight: bold; padding: 8px; font-size: 13px;"
    )
    self.extractButton.enabled = False
    step2Layout.addRow(self.extractButton)

    self.extractStatusLabel = qt.QLabel("Ready")
    self.extractStatusLabel.setStyleSheet("color: gray;")
    step2Layout.addRow("Status:", self.extractStatusLabel)

    # ── Centerline extraction progress bar ───────────────────────────────
    self.extractProgressBar = qt.QProgressBar()
    self.extractProgressBar.setMinimum(0)
    self.extractProgressBar.setMaximum(100)
    self.extractProgressBar.setValue(0)
    self.extractProgressBar.setTextVisible(True)
    self.extractProgressBar.setFormat("%p%")
    self.extractProgressBar.setFixedHeight(16)
    self.extractProgressBar.setStyleSheet(
        "QProgressBar {"
        "  border: 1px solid #bdc3c7;"
        "  border-radius: 4px;"
        "  background: #ecf0f1;"
        "  font-size: 11px;"
        "  color: #2c3e50;"
        "  text-align: center;"
        "}"
        "QProgressBar::chunk {"
        "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        "    stop:0 #27ae60, stop:1 #2ecc71);"
        "  border-radius: 3px;"
        "}"
    )
    self.extractProgressBar.setVisible(False)
    step2Layout.addRow("Progress:", self.extractProgressBar)

    # ── Manual Centerline Drawing ─────────────────────────────────────────
    manualCLLabel = qt.QLabel("── Manual Centerline ──")
    manualCLLabel.setStyleSheet("color: #aaaaaa; font-size: 11px;")
    step2Layout.addRow(manualCLLabel)

    self.drawCenterlineButton = qt.QPushButton("\u270f Draw Centerline Manually")
    self.drawCenterlineButton.setStyleSheet(
        "background-color: #1a6fa8; color: white; font-weight: bold; padding: 7px; font-size: 12px;"
    )
    self.drawCenterlineButton.setToolTip(
        "Click points along the vessel lumen to draw a custom centerline.\n"
        "Place points from proximal to distal: IVC inlet -> bifurcation -> iliac tips.\n"
        "Use this to correct VMTK centerline errors at the bifurcation zone.\n"
        "When done, click 'Use This Centerline' to load it for analysis."
    )
    step2Layout.addRow(self.drawCenterlineButton)

    manualCLBtnLayout = qt.QHBoxLayout()
    self.stopDrawCenterlineButton = qt.QPushButton("\u25a0 Stop Drawing")
    self.stopDrawCenterlineButton.setStyleSheet(
        "background-color: #e67e22; color: white; font-weight: bold; padding: 5px;"
    )
    self.stopDrawCenterlineButton.setVisible(False)

    self.useManualCenterlineButton = qt.QPushButton("\u2714 Use This Centerline")
    self.useManualCenterlineButton.setStyleSheet(
        "background-color: #27ae60; color: white; font-weight: bold; padding: 5px;"
    )
    self.useManualCenterlineButton.setEnabled(False)

    self.clearManualCenterlineButton = qt.QPushButton("\u2715 Clear")
    self.clearManualCenterlineButton.setStyleSheet(
        "background-color: #e74c3c; color: white; padding: 5px;"
    )
    self.clearManualCenterlineButton.setEnabled(False)

    manualCLBtnLayout.addWidget(self.stopDrawCenterlineButton)
    manualCLBtnLayout.addWidget(self.useManualCenterlineButton)
    manualCLBtnLayout.addWidget(self.clearManualCenterlineButton)
    step2Layout.addRow(manualCLBtnLayout)

    self.refineCenterlineButton = qt.QPushButton("✨ Refine")
    self.refineCenterlineButton.setStyleSheet(
        "background-color: #8e44ad; color: white; font-weight: bold; padding: 5px;"
    )
    self.refineCenterlineButton.setToolTip(
        "Recenter each point onto the vessel lumen axis and smooth the curve.\n"
        "Run after drawing to correct any points placed off-centre."
    )
    self.refineCenterlineButton.setEnabled(False)
    step2Layout.addRow(self.refineCenterlineButton)

    self.autoTuneButton = qt.QPushButton("\U0001f52c Auto-Tune Parameters")
    self.autoTuneButton.setStyleSheet(
        "QPushButton { background-color: #5b2d8e; color: white; "
        "font-size: 12px; padding: 5px; border-radius: 4px; }"
        "QPushButton:disabled { background-color: #555; color: #999; }"
    )
    self.autoTuneButton.setToolTip(
        "Run refinement with 20 random parameter sets, score each trial,\n"
        "then apply the best weights automatically.\n"
        "Takes ~30-60 s for typical centerlines."
    )
    self.autoTuneButton.setEnabled(False)
    step2Layout.addRow(self.autoTuneButton)

    self.autoTuneStatusLabel = qt.QLabel("")
    self.autoTuneStatusLabel.setStyleSheet("color: #8e44ad; font-size: 11px;")
    self.autoTuneStatusLabel.setWordWrap(True)
    step2Layout.addRow("", self.autoTuneStatusLabel)

    self.manualCLStatusLabel = qt.QLabel("No manual centerline drawn")
    self.manualCLStatusLabel.setStyleSheet("color: gray; font-size: 11px;")
    step2Layout.addRow("", self.manualCLStatusLabel)

    self.refineMetricsLabel = qt.QLabel("")
    self.refineMetricsLabel.setStyleSheet(
        "color: #aaaaaa; font-size: 10px; font-family: monospace;"
    )
    self.refineMetricsLabel.setWordWrap(True)
    step2Layout.addRow("", self.refineMetricsLabel)

    self.seedQualityLabel = qt.QLabel("")
    self.seedQualityLabel.setStyleSheet("font-size: 11px;")
    self.seedQualityLabel.setWordWrap(True)
    self.seedQualityLabel.setTextFormat(qt.Qt.RichText)
    step2Layout.addRow("Seed quality:", self.seedQualityLabel)

    self._manualCurves = []  # list of vtkMRMLMarkupsCurveNode, one per stroke
    self._activeCurve = None  # curve node currently being drawn
    self._activeCurveObserver = None
    self._lastPlacedPoint = None  # last point of previous stroke for auto-snap
    self._allNodes = []  # [{pos, curve, idx}] — all placed control points
    self._snapRadiusMm = 3.0  # snap radius for branch-point detection
    self._drawProxy = None  # proxy curve used as click receiver
    self._drawProxyObserver = None
    # ─────────────────────────────────────────────────────────────────────

    # Create modelOpacitySlider early so it can be placed in Step 3
    self.modelOpacitySlider = ctk.ctkSliderWidget()
    self.modelOpacitySlider.minimum = 0.0
    self.modelOpacitySlider.maximum = 1.0
    self.modelOpacitySlider.singleStep = 0.05
    self.modelOpacitySlider.value = 1.0
    self.modelOpacitySlider.decimals = 2
    self.modelOpacitySlider.setToolTip("Adjust 3D vessel model opacity")

    # =========================================================
    # STEP 3 — IVUS NAVIGATION
    # =========================================================
    step3 = ctk.ctkCollapsibleButton()
    step3.text = "Step 3 — IVUS Navigation"
    self.layout.addWidget(step3)
    step3Layout = qt.QFormLayout(step3)

    # Manual centerline selector (if already extracted)
    self.centerlineSelector = slicer.qMRMLNodeComboBox()
    self.centerlineSelector.nodeTypes = [
        "vtkMRMLMarkupsCurveNode",
        "vtkMRMLModelNode",
    ]
    self.centerlineSelector.selectNodeUponCreation = False
    self.centerlineSelector.addEnabled = False
    self.centerlineSelector.removeEnabled = False
    self.centerlineSelector.noneEnabled = True
    self.centerlineSelector.setMRMLScene(slicer.mrmlScene)
    self.centerlineSelector.setToolTip(
        "Auto-filled after extraction, or select manually"
    )
    step3Layout.addRow("Centerline:", self.centerlineSelector)

    self.centerlineHintLabel = qt.QLabel(
        "💡 Tip: Use 'Network model' from SlicerVMTK for best results"
    )
    self.centerlineHintLabel.setStyleSheet("color: #8e44ad; font-size: 11px;")
    self.centerlineHintLabel.setWordWrap(True)
    step3Layout.addRow(self.centerlineHintLabel)

    self.centerlineVisButton = qt.QPushButton("👁 Show Centerline")
    self.centerlineVisButton.setCheckable(True)
    self.centerlineVisButton.setChecked(False)
    self.centerlineVisButton.setStyleSheet(
        "QPushButton { background-color: #555; color: white; padding: 4px 8px; border-radius: 3px; }"
        "QPushButton:checked { background-color: #27ae60; color: white; }"
    )
    self.centerlineVisButton.setToolTip(
        "Toggle centerline model visibility in 3D view"
    )
    step3Layout.addRow(self.centerlineVisButton)

    self.loadIVUSButton = qt.QPushButton("▶ Load for IVUS Navigation")
    self.loadIVUSButton.setStyleSheet(
        "background-color: #9b59b6; color: white; font-weight: bold; padding: 6px;"
    )
    step3Layout.addRow(self.loadIVUSButton)

    # Branch selector
    self.branchSelector = qt.QComboBox()
    self.branchSelector.addItem("All branches")
    self.branchSelector.setEnabled(False)
    step3Layout.addRow("Branch:", self.branchSelector)

    # Branch statistics box
    branchStatsBox = qt.QGroupBox("Branch Statistics")
    branchStatsBox.setStyleSheet(
        "QGroupBox { font-weight: bold; border: 1px solid #8e44ad; border-radius: 4px; margin-top: 6px; padding: 4px; }"
    )
    branchStatsBoxLayout = qt.QFormLayout(branchStatsBox)

    self.branchNameLabel = qt.QLabel("All branches")
    self.branchNameLabel.setStyleSheet("font-weight: bold; color: #8e44ad;")
    branchStatsBoxLayout.addRow("Selected:", self.branchNameLabel)

    self.branchLengthLabel = qt.QLabel("--")
    branchStatsBoxLayout.addRow("Length:", self.branchLengthLabel)

    self.branchMinLabel = qt.QLabel("--")
    self.branchMinLabel.setStyleSheet(
        "color: #e74c3c; font-weight: bold; font-size: 14px;"
    )
    branchStatsBoxLayout.addRow("Min Diameter:", self.branchMinLabel)

    self.branchMaxLabel = qt.QLabel("--")
    self.branchMaxLabel.setStyleSheet(
        "color: #3498db; font-weight: bold; font-size: 14px;"
    )
    branchStatsBoxLayout.addRow("Max Diameter:", self.branchMaxLabel)

    self.branchAvgLabel = qt.QLabel("--")
    self.branchAvgLabel.setStyleSheet("font-size: 13px;")
    branchStatsBoxLayout.addRow("Mean Diameter:", self.branchAvgLabel)

    branchMinMaxLayout = qt.QHBoxLayout()
    self.branchGoToMinButton = qt.QPushButton("⬤ Go to Min")
    self.branchGoToMinButton.setStyleSheet("color: #e74c3c; font-weight: bold;")
    self.branchGoToMaxButton = qt.QPushButton("⬤ Go to Max")
    self.branchGoToMaxButton.setStyleSheet("color: #3498db; font-weight: bold;")
    branchMinMaxLayout.addWidget(self.branchGoToMinButton)
    branchMinMaxLayout.addWidget(self.branchGoToMaxButton)
    branchStatsBoxLayout.addRow(branchMinMaxLayout)

    step3Layout.addRow(branchStatsBox)

    self.branchStatsLabel = qt.QLabel("")
    self.branchStatsLabel.setStyleSheet("font-size: 11px; color: #8e44ad;")
    self.branchStatsLabel.setWordWrap(True)
    step3Layout.addRow(self.branchStatsLabel)

    # Slider
    self.pointSlider = ctk.ctkSliderWidget()
    self.pointSlider.minimum = 0
    self.pointSlider.maximum = 0
    self.pointSlider.singleStep = 1
    self.pointSlider.decimals = 0
    self.pointSlider.enabled = False
    step3Layout.addRow("Model Opacity:", self.modelOpacitySlider)
    step3Layout.addRow("Navigate:", self.pointSlider)

    navLayout = qt.QHBoxLayout()
    self.prevButton = qt.QPushButton("◀ Prev")
    self.nextButton = qt.QPushButton("Next ▶")

    self.endoluminalButton = qt.QPushButton("🔭 Endoluminal View  OFF")
    self.endoluminalButton.setCheckable(True)
    self.endoluminalButton.setStyleSheet(
        "background-color: #2c3e50; color: white; font-weight: bold; padding: 5px;"
    )
    navLayout.addWidget(self.prevButton)
    navLayout.addWidget(self.nextButton)
    step3Layout.addRow(self.endoluminalButton)
    step3Layout.addRow(navLayout)

    # =========================================================
    # SURFACE MODEL  (moved here from bottom — logically follows
    # branch navigation, before Measurements)
    # =========================================================
    surfCollapsible = ctk.ctkCollapsibleButton()
    surfCollapsible.text = "🔧 Surface Model"
    surfCollapsible.collapsed = False
    self.layout.addWidget(surfCollapsible)
    surfLayout = qt.QFormLayout(surfCollapsible)

    self.surfModelSelector = slicer.qMRMLNodeComboBox()
    self.surfModelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.surfModelSelector.selectNodeUponCreation = False
    self.surfModelSelector.addEnabled = False
    self.surfModelSelector.removeEnabled = False
    self.surfModelSelector.noneEnabled = True
    self.surfModelSelector.showHidden = False
    self.surfModelSelector.setMRMLScene(slicer.mrmlScene)
    surfLayout.addRow("Model:", self.surfModelSelector)

    _surfBtnRow = qt.QHBoxLayout()
    self.surfaceBtn = qt.QPushButton("📐 Create Surface")
    self.surfaceBtn.setStyleSheet(
        "background-color: #1a6b3c; color: white; font-weight: bold; padding: 6px;"
    )
    self.surfaceBtn.setToolTip(
        "Clean + decimate mesh in 4 steps (~68% reduction), preserving topology"
    )
    _surfBtnRow.addWidget(self.surfaceBtn)

    self.surfCheckBtn = qt.QPushButton("🔍 Check Surface")
    self.surfCheckBtn.setStyleSheet(
        "background-color: #1a5276; color: white; font-weight: bold; padding: 6px;"
    )
    self.surfCheckBtn.setToolTip(
        "Check for open/non-manifold edges — 0 = watertight clean mesh"
    )
    _surfBtnRow.addWidget(self.surfCheckBtn)
    surfLayout.addRow(_surfBtnRow)

    self.surfResultLabel = qt.QLabel("")
    self.surfResultLabel.setWordWrap(True)
    self.surfResultLabel.setStyleSheet(
        "font-size: 12px; padding: 6px; border-radius: 4px;"
    )
    surfLayout.addRow(self.surfResultLabel)

    self.surfaceBtn.connect("clicked(bool)", self.onCreateSurface)
    self.surfCheckBtn.connect("clicked(bool)", self.onCheckSurface)

    # =========================================================
    # BRANCH SURFACE CLASSIFICATION
    # =========================================================
    branchSurfCollapsible = ctk.ctkCollapsibleButton()
    branchSurfCollapsible.text = "🫀 Branch Surface Classification"
    branchSurfCollapsible.collapsed = False
    self.layout.addWidget(branchSurfCollapsible)
    branchSurfLayout = qt.QFormLayout(branchSurfCollapsible)

    self.branchSurfModelSelector = slicer.qMRMLNodeComboBox()
    self.branchSurfModelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.branchSurfModelSelector.selectNodeUponCreation = False
    self.branchSurfModelSelector.addEnabled = False
    self.branchSurfModelSelector.removeEnabled = False
    self.branchSurfModelSelector.noneEnabled = True
    self.branchSurfModelSelector.showHidden = False
    self.branchSurfModelSelector.setMRMLScene(slicer.mrmlScene)
    branchSurfLayout.addRow("Vessel Model:", self.branchSurfModelSelector)

    self.curvatureThreshSlider = ctk.ctkSliderWidget()
    self.curvatureThreshSlider.minimum = 0.1
    self.curvatureThreshSlider.maximum = 3.0
    self.curvatureThreshSlider.singleStep = 0.05
    self.curvatureThreshSlider.value = 0.5
    self.curvatureThreshSlider.setToolTip(
        "Curvature threshold for ostium ring detection. "
        "Higher = only very sharp branch origins. Lower = more inclusive."
    )
    branchSurfLayout.addRow(
        "Ostium Curvature Threshold:", self.curvatureThreshSlider
    )

    self.mainLenGateSlider = ctk.ctkSliderWidget()
    self.mainLenGateSlider.minimum = 0.0
    self.mainLenGateSlider.maximum = 200.0
    self.mainLenGateSlider.singleStep = 5.0
    self.mainLenGateSlider.value = 0.0
    self.mainLenGateSlider.setToolTip(
        "Minimum length (mm) for a 'main' branch to get its own color. "
        "Set to 0 to show all main branches. Raise to demote short mains to side group."
    )
    branchSurfLayout.addRow("Main Branch Min Length (mm):", self.mainLenGateSlider)

    _bsRow = qt.QHBoxLayout()
    self.classifySurfaceBtn = qt.QPushButton("🫀 Classify Branches")
    self.classifySurfaceBtn.setStyleSheet(
        "background-color: #4a235a; color: white; font-weight: bold; padding: 6px;"
    )
    self.classifySurfaceBtn.setToolTip(
        "Detect branch emergence borders on the 3D surface, then "
        "color the mesh: blue = trunk, colors = branches"
    )
    _bsRow.addWidget(self.classifySurfaceBtn)

    self.clearSurfClassBtn = qt.QPushButton("✕ Clear")
    self.clearSurfClassBtn.setStyleSheet(
        "background-color: #5d6d7e; color: white; font-weight: bold; padding: 6px;"
    )
    self.clearSurfClassBtn.setToolTip(
        "Remove branch coloring and restore original model color"
    )
    _bsRow.addWidget(self.clearSurfClassBtn)
    branchSurfLayout.addRow(_bsRow)

    self.branchSurfResultLabel = qt.QLabel("")
    self.branchSurfResultLabel.setWordWrap(True)
    self.branchSurfResultLabel.setStyleSheet(
        "font-size: 12px; padding: 6px; border-radius: 4px;"
    )
    branchSurfLayout.addRow(self.branchSurfResultLabel)

    # ── Inspect Ostium Detection (P2 vs P3) ──────────────────────────
    self.debugOstiumBtn = qt.QPushButton("Inspect Ostium Detection (P2 vs P3)")
    self.debugOstiumBtn.setStyleSheet(
        "background-color: #1a3a4a; color: white; font-weight: bold; padding: 5px;"
    )
    self.debugOstiumBtn.setToolTip(
        "Place fiducials showing:\n"
        "  Red    -- raw bifurcation graph node\n"
        "  Yellow -- Pass-2 ostium (sibling-separation + radius)\n"
        "  Green  -- (not currently used)\n"
        "Run analysis first. Removes previous debug markers automatically."
    )
    branchSurfLayout.addRow(self.debugOstiumBtn)
    self.debugOstiumBtn.connect("clicked(bool)", self.onDebugOstium)

    # Model/Surface visibility toggle — placed here so it sits directly
    # below the Inspect Ostium Detection button (UIMove: was in Vis Options)
    self.modelVisButton = qt.QPushButton("👁 Showing: Model")
    self.modelVisButton.setCheckable(True)
    self.modelVisButton.setChecked(True)
    self.modelVisButton.setStyleSheet(
        "background-color: #27ae60; color: white; font-weight: bold; padding: 4px;"
    )
    self.modelVisButton.setToolTip(
        "Toggle between showing the vessel model (green) or the classified surface (blue).\n"
        "The two are shown exclusively — one hides when the other is shown."
    )
    branchSurfLayout.addRow(self.modelVisButton)

    self.classifySurfaceBtn.connect("clicked(bool)", self.onClassifySurface)
    self.clearSurfClassBtn.connect(
        "clicked(bool)", self.onClearSurfaceClassification
    )

    # =========================================================
    # MEASUREMENTS
    # =========================================================
    measCollapsible = ctk.ctkCollapsibleButton()
    measCollapsible.text = "Measurements"
    self.layout.addWidget(measCollapsible)
    measLayout = qt.QFormLayout(measCollapsible)

    self.pointIndexLabel = qt.QLabel("--")
    measLayout.addRow("Point:", self.pointIndexLabel)

    self.distanceLabel = qt.QLabel("--")
    self.distanceLabel.setStyleSheet("font-weight: bold;")
    measLayout.addRow("Distance from start:", self.distanceLabel)

    self.coordLabel = qt.QLabel("--")
    # Copy-coordinates button — lets user copy current point to clipboard
    coordWidget = qt.QWidget()
    coordRow = qt.QHBoxLayout(coordWidget)
    coordRow.setContentsMargins(0, 0, 0, 0)
    coordRow.addWidget(self.coordLabel)
    self.copyCoordButton = qt.QPushButton("📋")
    self.copyCoordButton.setFixedWidth(28)
    self.copyCoordButton.setToolTip(
        "Copy current coordinates + diameter to clipboard\n"
        "Format: R x.x, A y.y, S z.z  |  Ø d.dmm"
    )
    self.copyCoordButton.setStyleSheet(
        "padding: 1px; font-size: 13px; background: #2c3e50; color: white; "
        "border-radius: 3px;"
    )
    coordRow.addWidget(self.copyCoordButton)
    measLayout.addRow("Coordinates:", coordWidget)

    self.findingWarningLabel = qt.QLabel("")
    self.findingWarningLabel.setStyleSheet(
        "font-weight: bold; font-size: 13px; padding: 4px; border-radius: 4px;"
    )
    self.findingWarningLabel.setWordWrap(True)
    measLayout.addRow(self.findingWarningLabel)

    self.diameterLabel = qt.QLabel("--")
    self.diameterLabel.setStyleSheet(
        "font-size: 20px; font-weight: bold; color: #2ecc71;"
    )
    measLayout.addRow("Diameter:", self.diameterLabel)

    self.radiusLabel = qt.QLabel("--")
    self.radiusLabel.setStyleSheet("color: #27ae60;")
    measLayout.addRow(
        "Diameter (mm):", self.radiusLabel
    )  # FIX: was mislabelled "Radius"

    sep = qt.QFrame()
    sep.setFrameShape(qt.QFrame.HLine)
    measLayout.addRow(sep)

    self.minDiamLabel = qt.QLabel("--")
    self.minDiamLabel.setStyleSheet("color: #e74c3c; font-weight: bold;")
    measLayout.addRow("Min Diameter:", self.minDiamLabel)

    self.maxDiamLabel = qt.QLabel("--")
    self.maxDiamLabel.setStyleSheet("color: #3498db; font-weight: bold;")
    measLayout.addRow("Max Diameter:", self.maxDiamLabel)

    self.avgDiamLabel = qt.QLabel("--")
    measLayout.addRow("Mean Diameter:", self.avgDiamLabel)

    self.totalLengthLabel = qt.QLabel("--")
    measLayout.addRow("Total Length:", self.totalLengthLabel)

    minMaxLayout = qt.QHBoxLayout()
    self.goToMinButton = qt.QPushButton("⬤ Go to Min")
    self.goToMinButton.setStyleSheet("color: #e74c3c; font-weight: bold;")
    self.goToMaxButton = qt.QPushButton("⬤ Go to Max")
    self.goToMaxButton.setStyleSheet("color: #3498db; font-weight: bold;")
    minMaxLayout.addWidget(self.goToMinButton)
    minMaxLayout.addWidget(self.goToMaxButton)
    measLayout.addRow(minMaxLayout)

    # =========================================================
    # MANUAL BRANCH RANGES
    # =========================================================
    manualRangeCollapsible = ctk.ctkCollapsibleButton()
    manualRangeCollapsible.text = "📐 Manual Branch Ranges"
    manualRangeCollapsible.collapsed = True
    self.layout.addWidget(manualRangeCollapsible)
    manualRangeLayout = qt.QVBoxLayout(manualRangeCollapsible)

    _mrHint = qt.QLabel(
        "Pin each branch's start/end to exact navigator coordinates.\n"
        "Paste in the format:  R x.x, A y.y, S z.z  |  Ø d.d mm\n"
        "Leave a field blank to keep the auto-detected value."
    )
    _mrHint.setStyleSheet("color: #aaa; font-size: 11px;")
    _mrHint.setWordWrap(True)
    manualRangeLayout.addWidget(_mrHint)

    # Branch name → bi mapping (filled after analysis runs via onLoadIVUS)
    # We use a fixed set of rows: Trunk (0), Left Iliac, Right Iliac, B4, B5
    # The actual bi is resolved at apply-time from branchMeta roles.
    _mrGrid = qt.QGridLayout()
    _mrGrid.addWidget(qt.QLabel("<b>Branch</b>"), 0, 0)
    _mrGrid.addWidget(qt.QLabel("<b>Start coord  (paste from 📋)</b>"), 0, 1)
    _mrGrid.addWidget(qt.QLabel("<b>End coord  (paste from 📋)</b>"), 0, 2)

    # Row definitions: (display_label, role_key)
    # role_key: "trunk" | "iliac_left" | "iliac_right" | "branch_4" | "branch_5"
    _mr_rows = [
        ("Trunk", "trunk"),
        ("Left Iliac", "iliac_left"),
        ("Right Iliac", "iliac_right"),
        ("Branch 4", "branch_4"),
        ("Branch 5", "branch_5"),
    ]
    self._mrStartEdits = {}  # role_key → QLineEdit
    self._mrEndEdits = {}  # role_key → QLineEdit

    for row_i, (lbl, rkey) in enumerate(_mr_rows, start=1):
        _mrGrid.addWidget(qt.QLabel(lbl), row_i, 0)
        _se = qt.QLineEdit()
        _se.setPlaceholderText("R x.x, A y.y, S z.z  |  Ø d.d mm")
        _se.setStyleSheet("font-size: 11px;")
        _mrGrid.addWidget(_se, row_i, 1)
        self._mrStartEdits[rkey] = _se

        _ee = qt.QLineEdit()
        _ee.setPlaceholderText("R x.x, A y.y, S z.z  |  Ø d.d mm")
        _ee.setStyleSheet("font-size: 11px;")
        _mrGrid.addWidget(_ee, row_i, 2)
        self._mrEndEdits[rkey] = _ee

    manualRangeLayout.addLayout(_mrGrid)

    _mrBtnRow = qt.QHBoxLayout()
    self._mrApplyBtn = qt.QPushButton("✅ Apply Manual Ranges")
    self._mrApplyBtn.setStyleSheet(
        "background: #27ae60; color: white; font-weight: bold; padding: 4px;"
    )
    self._mrApplyBtn.setToolTip(
        "Snap each branch start/end to the nearest centerline point "
        "matching the pasted coordinates."
    )
    self._mrApplyBtn.clicked.connect(self.onApplyManualRanges)
    _mrBtnRow.addWidget(self._mrApplyBtn)

    self._mrClearBtn = qt.QPushButton("🗑 Clear All")
    self._mrClearBtn.setStyleSheet("padding: 4px;")
    self._mrClearBtn.clicked.connect(self.onClearManualRanges)
    _mrBtnRow.addWidget(self._mrClearBtn)
    manualRangeLayout.addLayout(_mrBtnRow)

    self._mrStatusLabel = qt.QLabel("")
    self._mrStatusLabel.setWordWrap(True)
    self._mrStatusLabel.setStyleSheet("font-size: 11px; color: #27ae60;")
    manualRangeLayout.addWidget(self._mrStatusLabel)

    # =========================================================
    # VISUALIZATION OPTIONS
    # =========================================================
    visCollapsible = ctk.ctkCollapsibleButton()
    visCollapsible.text = "Visualization Options"
    self.layout.addWidget(visCollapsible)
    visLayout = qt.QFormLayout(visCollapsible)

    self.showSphereCheck = qt.QCheckBox("Show MIS sphere")
    self.showSphereCheck.setChecked(True)
    visLayout.addRow(self.showSphereCheck)

    self.showRingCheck = qt.QCheckBox("Show cross-section ring")
    self.showRingCheck.setChecked(True)
    visLayout.addRow(self.showRingCheck)

    self.showLineCheck = qt.QCheckBox("Show diameter line")
    self.showLineCheck.setChecked(True)
    visLayout.addRow(self.showLineCheck)

    # Model/Surface visibility toggle — moved to Branch Surface Classification
    # collapsible, below Inspect Ostium Detection button (UIMove v242)

    self.sphereColorButton = qt.QPushButton("  Change Sphere Color")
    self.sphereColorButton.setStyleSheet(
        "background-color: rgb(46,204,113); color: white;"
    )
    visLayout.addRow(self.sphereColorButton)

    # =========================================================
    stentCollapsible = ctk.ctkCollapsibleButton()
    stentCollapsible.text = "Stent Planner"
    stentCollapsible.collapsed = False
    self.layout.addWidget(stentCollapsible)
    stentLayout = qt.QFormLayout(stentCollapsible)

    # Target branch selector — multi-select list
    branchLabelWidget = qt.QLabel("Target branch(es):")
    self.stentBranchList = qt.QListWidget()
    self.stentBranchList.setSelectionMode(
        qt.QAbstractItemView.ExtendedSelection
    )  # Ctrl/Shift for multi
    self.stentBranchList.setMaximumHeight(80)
    self.stentBranchList.setToolTip(
        "Select one or more branches to stent.\n"
        "Ctrl+click or Shift+click to select multiple."
    )
    stentLayout.addRow(branchLabelWidget, self.stentBranchList)
    # Keep a hidden QComboBox for backward-compatibility with all code
    # that reads self.stentBranchCombo.currentIndex
    self.stentBranchCombo = qt.QComboBox()
    self.stentBranchCombo.setVisible(False)
    stentLayout.addRow(self.stentBranchCombo)

    # Stent type
    self.stentTypeCombo = qt.QComboBox()
    self.stentTypeCombo.addItems(
        [
            "── Bare-Metal / Nitinol ──",
            "  Straight",
            "  Tapered",
            "  Bifurcated (Y / Trouser)",
            "── GORE VBX (Balloon-Expandable, Kissing) ──",
            "  VBX · Single Limb",
            "  VBX · Kissing (Parallel) — 7–10mm + POT",
            "── GORE Viabahn Venous (Self-Expanding) ──",
            "  Venous · Straight (10–28mm)",
            "  Venous · Tapered",
            "── Covered Stent Grafts ──",
            "  Covered · Straight (PTFE/ePTFE)",
            "  Covered · Bifurcated (EVAR Graft)",
            "── Drug-Eluting ──",
            "  DES · Sirolimus-Eluting",
            "  DES · Paclitaxel-Eluting",
            "  DES · Zotarolimus-Eluting",
            "── Specialty ──",
            "  Mesh Stent",
            "  Annular Coil Stent",
        ]
    )
    # Make separator items non-selectable
    for i in [0, 4, 7, 10, 13, 17]:
        _item = self.stentTypeCombo.model().item(i)
        if _item:
            _item.setEnabled(False)
            _item.setForeground(qt.QColor("#888888"))
    stentLayout.addRow("Stent type:", self.stentTypeCombo)

    # Landing zone — proximal
    self.stentProxSlider = qt.QSlider(qt.Qt.Horizontal)
    self.stentProxSlider.setMinimum(0)
    self.stentProxSlider.setMaximum(100)
    self.stentProxSlider.setValue(20)
    self.stentProxLabel = qt.QLabel("Pt 20")
    proxWidget = qt.QWidget()
    proxRow = qt.QHBoxLayout(proxWidget)
    proxRow.setContentsMargins(0, 0, 0, 0)
    proxRow.addWidget(self.stentProxSlider)
    proxRow.addWidget(self.stentProxLabel)
    stentLayout.addRow("Proximal landing:", proxWidget)

    # Landing zone — distal
    # Trunk extension spinbox (shown when prox slider is at 0)
    trunkWidget = qt.QWidget()
    trunkRow = qt.QHBoxLayout(trunkWidget)
    trunkRow.setContentsMargins(0, 0, 0, 0)
    self.stentTrunkSpin = qt.QDoubleSpinBox()
    self.stentTrunkSpin.setRange(0.0, 150.0)
    self.stentTrunkSpin.setSingleStep(5.0)
    self.stentTrunkSpin.setDecimals(1)
    self.stentTrunkSpin.setSuffix(" mm")
    self.stentTrunkSpin.setValue(0.0)
    self.stentTrunkSpin.setToolTip(
        "Extend stent proximally into the trunk vessel beyond the bifurcation"
    )
    trunkRow.addWidget(self.stentTrunkSpin)
    stentLayout.addRow("Trunk overlap:", trunkWidget)

    self.stentDistSlider = qt.QSlider(qt.Qt.Horizontal)
    self.stentDistSlider.setMinimum(0)
    self.stentDistSlider.setMaximum(100)
    self.stentDistSlider.setValue(80)
    self.stentDistLabel = qt.QLabel("Pt 80")
    distWidget = qt.QWidget()
    distRow = qt.QHBoxLayout(distWidget)
    distRow.setContentsMargins(0, 0, 0, 0)
    distRow.addWidget(self.stentDistSlider)
    distRow.addWidget(self.stentDistLabel)
    stentLayout.addRow("Distal landing:", distWidget)

    # Diameter inputs
    self.stentProxDiamSpin = qt.QDoubleSpinBox()
    self.stentProxDiamSpin.setRange(4.0, 40.0)
    self.stentProxDiamSpin.setSingleStep(0.5)
    self.stentProxDiamSpin.setValue(12.0)
    self.stentProxDiamSpin.setSuffix(" mm")
    stentLayout.addRow("Proximal diameter:", self.stentProxDiamSpin)

    self.stentDistDiamSpin = qt.QDoubleSpinBox()
    self.stentDistDiamSpin.setRange(4.0, 40.0)
    self.stentDistDiamSpin.setSingleStep(0.5)
    self.stentDistDiamSpin.setValue(10.0)
    self.stentDistDiamSpin.setSuffix(" mm")
    stentLayout.addRow("Distal diameter:", self.stentDistDiamSpin)

    # ── Kissing-only: VBX nominal size selector (replaces spinbox for kissing) ──
    self._vbxDiamWidget = qt.QWidget()
    _vbxRow = qt.QHBoxLayout(self._vbxDiamWidget)
    _vbxRow.setContentsMargins(0, 0, 0, 0)
    self.kissNomCombo = qt.QComboBox()
    for _nom in [5, 6, 7, 8, 9, 10, 11]:
        self.kissNomCombo.addItem(
            f"{_nom} mm  (max post-dilation: "
            f"{dict(zip([5,6,7,8,9,10,11],[8,9,11,13,13,13,16]))[_nom]} mm)"
        )
    self.kissNomCombo.setCurrentIndex(6)  # default 11mm
    _vbxRow.addWidget(self.kissNomCombo)
    self._vbxDiamWidget.setVisible(False)
    stentLayout.addRow("VBX nominal size:", self._vbxDiamWidget)

    # ── Kissing / Y-stent specific controls (shown only for those types) ──
    self._kissingControlsWidget = qt.QWidget()
    _kcLayout = qt.QFormLayout(self._kissingControlsWidget)
    _kcLayout.setContentsMargins(0, 0, 0, 0)

    # Proximal trunk extension (how far above bifurcation to start)
    self.kissProxExtSpin = qt.QDoubleSpinBox()
    self.kissProxExtSpin.setRange(5.0, 80.0)
    self.kissProxExtSpin.setSingleStep(5.0)
    self.kissProxExtSpin.setDecimals(0)
    self.kissProxExtSpin.setSuffix(" mm")
    self.kissProxExtSpin.setValue(10.0)
    self.kissProxExtSpin.setToolTip(
        "How far above the bifurcation the kissing stents extend into the trunk (IVC).\n"
        "Increase to cover more of the IVC. Typical: 20–40mm."
    )
    _kcLayout.addRow("Trunk coverage:", self.kissProxExtSpin)

    # Distal extension branch A
    _kissDistRow = qt.QHBoxLayout()
    self.kissDistASpin = qt.QDoubleSpinBox()
    self.kissDistASpin.setRange(10.0, 300.0)
    self.kissDistASpin.setSingleStep(10.0)
    self.kissDistASpin.setDecimals(0)
    self.kissDistASpin.setSuffix(" mm")
    self.kissDistASpin.setValue(60.0)
    self.kissDistASpin.setToolTip(
        "Length of stent in Branch A (left iliac) from bifurcation hub."
    )
    _kissDistRow.addWidget(qt.QLabel("A:"))
    _kissDistRow.addWidget(self.kissDistASpin)

    self.kissDistBSpin = qt.QDoubleSpinBox()
    self.kissDistBSpin.setRange(10.0, 300.0)
    self.kissDistBSpin.setSingleStep(10.0)
    self.kissDistBSpin.setDecimals(0)
    self.kissDistBSpin.setSuffix(" mm")
    self.kissDistBSpin.setValue(60.0)
    self.kissDistBSpin.setToolTip(
        "Length of stent in Branch B (right iliac) from bifurcation hub."
    )
    _kissDistRow.addWidget(qt.QLabel("  B:"))
    _kissDistRow.addWidget(self.kissDistBSpin)

    self.kissApplyBtn = qt.QPushButton("↵ Apply")
    self.kissApplyBtn.setFixedWidth(60)
    self.kissApplyBtn.setToolTip(
        "Re-place kissing stent with updated trunk/branch lengths."
    )
    self.kissApplyBtn.setStyleSheet(
        "background-color: #16a085; color: white; font-weight: bold; padding: 4px;"
    )
    _kissDistRow.addWidget(self.kissApplyBtn)
    _kcLayout.addRow("Branch depth (mm):", _kissDistRow)

    self._kissingControlsWidget.setVisible(False)
    stentLayout.addRow(self._kissingControlsWidget)

    # Auto-size button
    # Length spinbox row
    lengthWidget = qt.QWidget()
    lengthRow = qt.QHBoxLayout(lengthWidget)
    lengthRow.setContentsMargins(0, 0, 0, 0)
    self.stentLengthSpin = qt.QDoubleSpinBox()
    self.stentLengthSpin.setRange(10.0, 300.0)
    self.stentLengthSpin.setSingleStep(5.0)
    self.stentLengthSpin.setDecimals(1)
    self.stentLengthSpin.setSuffix(" mm")
    self.stentLengthSpin.setValue(40.0)
    self.stentLengthSpin.setToolTip(
        "Stent length.\n"
        "For kissing stents: sets Branch A (lesion side) depth in mm from hub.\n"
        "79mm = full GORE VBX maximum. Branch B set independently above.\n"
        "Snaps to available VBX lengths: 15/19/29/39/59/79mm."
    )
    self.stentLengthApplyBtn = qt.QPushButton("↵ Apply")
    self.stentLengthApplyBtn.setFixedWidth(60)
    lengthRow.addWidget(self.stentLengthSpin)
    lengthRow.addWidget(self.stentLengthApplyBtn)
    stentLayout.addRow("Stent length:", lengthWidget)

    self.stentAutoSizeButton = qt.QPushButton(
        "📐 Auto-size from Measurements   (optional)"
    )
    self.stentAutoSizeButton.setVisible(False)
    self.stentAutoSizeButton.setStyleSheet(
        "background-color: #8e44ad; color: white; font-weight: bold; padding: 5px;"
    )
    stentLayout.addRow(self.stentAutoSizeButton)

    self.stentAutoTypeButton = qt.QPushButton(
        "🧠 Auto-detect Stent Type   (optional)"
    )
    self.stentAutoTypeButton.setVisible(False)
    self.stentAutoTypeButton.setToolTip(
        "Analyze vessel geometry to recommend and configure the optimal stent type"
    )
    self.stentAutoTypeButton.setStyleSheet(
        "background-color: #1a6b3c; color: white; font-weight: bold; padding: 5px;"
    )
    stentLayout.addRow(self.stentAutoTypeButton)

    self.stentAutoTypeLabel = qt.QLabel("")
    self.stentAutoTypeLabel.setWordWrap(True)
    self.stentAutoTypeLabel.setStyleSheet(
        "font-size: 11px; color: #1a6b3c; padding: 3px; font-style: italic;"
    )
    stentLayout.addRow(self.stentAutoTypeLabel)

    # Summary + warning labels
    self.stentSummaryLabel = qt.QLabel("")
    self.stentSummaryLabel.setStyleSheet("font-size: 11px; color: #2c3e50;")
    self.stentSummaryLabel.setWordWrap(True)
    stentLayout.addRow(self.stentSummaryLabel)

    self.stentWarningLabel = qt.QLabel("")
    self.stentWarningLabel.setStyleSheet(
        "font-weight: bold; font-size: 12px; padding: 4px; border-radius: 4px;"
    )
    self.stentWarningLabel.setWordWrap(True)
    stentLayout.addRow(self.stentWarningLabel)

    # Red primary action button
    self.detectFindingsButton = qt.QPushButton("🎯 Find Lesions")
    self.detectFindingsButton.setStyleSheet(
        "background-color: #c0392b; color: white; font-weight: bold; "
        "padding: 8px; font-size: 13px;"
    )
    self.detectFindingsButton.setToolTip(
        "Detect all findings (pancaking, aneurysm, compression) and "
        "automatically place the optimal stent at each lesion"
    )
    stentLayout.addRow(self.detectFindingsButton)

    # ── Lesion Status Table ───────────────────────────────────────────
    self.lesionTableWidget = qt.QTableWidget(0, 5)
    self.lesionTableWidget.setHorizontalHeaderLabels(
        ["#", "Branch", "Type", "Min→Max", "Status"]
    )
    self.lesionTableWidget.horizontalHeader().setStretchLastSection(True)
    self.lesionTableWidget.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
    self.lesionTableWidget.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
    self.lesionTableWidget.setMaximumHeight(120)
    self.lesionTableWidget.setVisible(False)
    self.lesionTableWidget.setStyleSheet(
        "font-size: 11px; QHeaderView::section { background-color: #2c3e50; color: white; }"
    )
    self.lesionTableWidget.connect(
        "cellClicked(int,int)", self.onLesionTableClicked
    )
    # Set column widths
    self.lesionTableWidget.setColumnWidth(0, 25)
    self.lesionTableWidget.setColumnWidth(1, 70)
    self.lesionTableWidget.setColumnWidth(2, 100)
    self.lesionTableWidget.setColumnWidth(3, 90)
    stentLayout.addRow(self.lesionTableWidget)

    # ── Pre-dilation button (shown after Find Lesions detects stenosis) ──
    self.preDilateButton = qt.QPushButton(
        "🎈 Pre-Dilate Lesion  (balloon angioplasty)"
    )
    self.preDilateButton.setStyleSheet(
        "background-color: #8e44ad; color: white; font-weight: bold; "
        "padding: 7px; font-size: 12px;"
    )
    self.preDilateButton.setToolTip(
        "Visualize balloon pre-dilation at the detected stenosis before stent placement.\n"
        "In real practice: balloon inflated at lesion to expand it, then stent deployed."
    )
    self.preDilateButton.setEnabled(False)  # enabled after Find Lesions
    stentLayout.addRow(self.preDilateButton)

    self.preDilateStatusLabel = qt.QLabel("")
    self.preDilateStatusLabel.setStyleSheet(
        "font-size: 11px; color: #8e44ad; font-style: italic;"
    )
    self.preDilateStatusLabel.setWordWrap(True)
    self.preDilateCopyBtn = qt.QPushButton("📋 Copy")
    self.preDilateCopyBtn.setFixedWidth(70)
    self.preDilateCopyBtn.setToolTip("Copy balloon summary to clipboard")
    self.preDilateCopyBtn.setVisible(False)
    self.preDilateCopyBtn.connect("clicked(bool)", self.onCopyPreDilateStatus)
    _pdRow = qt.QHBoxLayout()
    _pdRow.addWidget(self.preDilateStatusLabel)
    _pdRow.addWidget(self.preDilateCopyBtn)
    stentLayout.addRow(_pdRow)

    # ── Manual balloon placement ─────────────────────────────────────
    _manualBalloonRow = qt.QHBoxLayout()
    self.manualBalloonButton = qt.QPushButton("📍 Place Balloon Manually")
    self.manualBalloonButton.setStyleSheet(
        "background-color: #c0392b; color: white; font-weight: bold; "
        "padding: 6px; font-size: 11px;"
    )
    self.manualBalloonButton.setToolTip(
        "Click two points on the vessel to define the balloon zone.\n"
        "Balloon diameter is set by the spinbox to the right."
    )
    self.manualBalloonButton.setEnabled(False)
    _manualBalloonRow.addWidget(self.manualBalloonButton)

    _manualBalloonRow.addWidget(qt.QLabel("Ø:"))
    self.manualBalloonDiamSpin = qt.QDoubleSpinBox()
    self.manualBalloonDiamSpin.setRange(2.0, 40.0)
    self.manualBalloonDiamSpin.setSingleStep(0.5)
    self.manualBalloonDiamSpin.setValue(10.0)
    self.manualBalloonDiamSpin.setSuffix(" mm")
    self.manualBalloonDiamSpin.setFixedWidth(90)
    self.manualBalloonDiamSpin.setToolTip("Balloon inflated diameter (mm)")
    _manualBalloonRow.addWidget(self.manualBalloonDiamSpin)

    self.manualBalloonCancelBtn = qt.QPushButton("✕")
    self.manualBalloonCancelBtn.setFixedWidth(30)
    self.manualBalloonCancelBtn.setEnabled(False)
    self.manualBalloonCancelBtn.setToolTip("Cancel manual balloon placement")
    self.manualBalloonCancelBtn.setStyleSheet("color: #c0392b; font-weight: bold;")
    _manualBalloonRow.addWidget(self.manualBalloonCancelBtn)
    stentLayout.addRow(_manualBalloonRow)

    self.manualBalloonStatusLabel = qt.QLabel("")
    self.manualBalloonStatusLabel.setStyleSheet(
        "font-size: 11px; color: #c0392b; font-style: italic;"
    )
    stentLayout.addRow(self.manualBalloonStatusLabel)

    self.manualBalloonButton.connect("clicked(bool)", self.onManualBalloonStart)
    self.manualBalloonCancelBtn.connect("clicked(bool)", self.onManualBalloonCancel)

    # ── POT (Proximal Optimization Technique) ─────────────────────────
    _potRow = qt.QHBoxLayout()
    self.potButton = qt.QPushButton("🔄 Apply POT")
    self.potButton.setStyleSheet(
        "background-color: #8e44ad; color: white; font-weight: bold; "
        "padding: 6px; font-size: 11px;"
    )
    self.potButton.setToolTip(
        "Proximal Optimization Technique: inflate a single larger balloon "
        "in the trunk to restore circular lumen after kissing stent deployment. "
        "Corrects carina crush and figure-8 deformation."
    )
    self.potButton.setEnabled(False)
    self.potRemoveButton = qt.QPushButton("✕ Remove POT")
    self.potRemoveButton.setStyleSheet(
        "background-color: #555; color: white; padding: 6px;"
    )
    self.potRemoveButton.setEnabled(False)
    _potRow.addWidget(self.potButton)
    _potRow.addWidget(self.potRemoveButton)
    stentLayout.addRow("POT after Kissing:", _potRow)

    self.potStatusLabel = qt.QLabel("")
    self.potStatusLabel.setStyleSheet(
        "font-size: 11px; color: #8e44ad; font-style: italic;"
    )
    stentLayout.addRow(self.potStatusLabel)

    self.potButton.connect("clicked(bool)", self.onApplyPOT)
    self.potRemoveButton.connect("clicked(bool)", self.onRemovePOT)

    # ── Carina Support ────────────────────────────────────────────────
    _carinaRow = qt.QHBoxLayout()
    self.carinaButton = qt.QPushButton("🔵 Carina Support")
    self.carinaButton.setStyleSheet(
        "background-color: #1a6b8a; color: white; font-weight: bold; "
        "padding: 6px; font-size: 11px;"
    )
    self.carinaButton.setToolTip(
        "Place a short flared support at the confluence (kissing point) "
        "to keep the lumen circular and prevent figure-8 compression."
    )
    self.carinaButton.setEnabled(False)
    self.carinaRemoveButton = qt.QPushButton("✕ Remove")
    self.carinaRemoveButton.setStyleSheet(
        "background-color: #555; color: white; padding: 6px;"
    )
    self.carinaRemoveButton.setEnabled(False)
    _carinaRow.addWidget(self.carinaButton)
    _carinaRow.addWidget(self.carinaRemoveButton)
    stentLayout.addRow("Carina support:", _carinaRow)

    # Carina diameter override spinbox
    _carinaDiamRow = qt.QHBoxLayout()
    _carinaDiamRow.addWidget(qt.QLabel("Confluence Ø:"))
    self.carinaDiamSpin = qt.QDoubleSpinBox()
    self.carinaDiamSpin.setMinimum(0.0)
    self.carinaDiamSpin.setMaximum(35.0)
    self.carinaDiamSpin.setSingleStep(0.5)
    self.carinaDiamSpin.setValue(0.0)
    self.carinaDiamSpin.setSuffix(" mm")
    self.carinaDiamSpin.setFixedWidth(90)
    self.carinaDiamSpin.setToolTip(
        "Override carina diameter. 0 = auto (branch stable median × 0.85, min 10mm). "
        "Try 10→11→12mm to test carina effect."
    )
    _carinaDiamRow.addWidget(self.carinaDiamSpin)
    _carinaDiamRow.addWidget(qt.QLabel("  Asymmetry:"))
    self.carinaAsymSpin = qt.QDoubleSpinBox()
    self.carinaAsymSpin.setMinimum(-3.0)
    self.carinaAsymSpin.setMaximum(3.0)
    self.carinaAsymSpin.setSingleStep(0.5)
    self.carinaAsymSpin.setValue(0.0)
    self.carinaAsymSpin.setSuffix(" mm")
    self.carinaAsymSpin.setFixedWidth(80)
    self.carinaAsymSpin.setToolTip(
        "Branch asymmetry offset: +N gives left branch N mm larger, right N mm smaller. "
        "Try ±1.0 to break symmetric collision."
    )
    _carinaDiamRow.addWidget(self.carinaAsymSpin)
    stentLayout.addRow("", _carinaDiamRow)

    self.carinaStatusLabel = qt.QLabel("")
    self.carinaStatusLabel.setStyleSheet(
        "font-size: 11px; color: #1a6b8a; font-style: italic;"
    )
    stentLayout.addRow(self.carinaStatusLabel)

    self.carinaButton.connect("clicked(bool)", self.onApplyCarinaSupportSIM)
    self.carinaRemoveButton.connect("clicked(bool)", self.onRemoveCarinaSupport)

    # ── Before/After compare button ───────────────────────────────────
    # ── Ruler button ─────────────────────────────────────────────────────
    _rulerRow = qt.QHBoxLayout()
    self.addRulerButton = qt.QPushButton("📏 Add Ruler")
    self.addRulerButton.setStyleSheet(
        "background-color: #16a085; color: white; font-weight: bold; padding: 6px; font-size: 11px;"
    )
    self.addRulerButton.setToolTip(
        "Place a ruler parallel to the lesion in the 3D view. Uses the current stent or balloon path. Offset 15mm to the side."
    )
    self.addRulerButton.setEnabled(False)
    self.removeRulerButton = qt.QPushButton("✕ Remove")
    self.removeRulerButton.setStyleSheet(
        "background-color: #555; color: white; padding: 6px;"
    )
    self.removeRulerButton.setEnabled(False)
    _rulerRow.addWidget(self.addRulerButton)
    _rulerRow.addWidget(self.removeRulerButton)
    stentLayout.addRow(_rulerRow)

    self.rulerStatusLabel = qt.QLabel("")
    self.rulerStatusLabel.setStyleSheet(
        "font-size: 11px; color: #16a085; font-style: italic;"
    )
    stentLayout.addRow(self.rulerStatusLabel)

    _compareRow = qt.QHBoxLayout()
    self.compareBeforeAfterButton = qt.QPushButton("🔀 Compare Before / After")
    self.compareBeforeAfterButton.setStyleSheet(
        "background-color: #2c3e50; color: white; font-weight: bold; "
        "padding: 6px; font-size: 11px;"
    )
    self.compareBeforeAfterButton.setToolTip(
        "Show a dual 3D view: left = original compressed vessel, "
        "right = balloon-expanded vessel after pre-dilation."
    )
    self.compareBeforeAfterButton.setEnabled(False)
    self.exitCompareButton = qt.QPushButton("✕ Exit Compare")
    self.exitCompareButton.setStyleSheet(
        "background-color: #c0392b; color: white; font-weight: bold; "
        "padding: 6px; font-size: 11px;"
    )
    self.exitCompareButton.setToolTip("Return to normal single 3D view.")
    self.exitCompareButton.setEnabled(False)
    _compareRow.addWidget(self.compareBeforeAfterButton)
    _compareRow.addWidget(self.exitCompareButton)
    stentLayout.addRow(_compareRow)

    # ── Workflow step divider ──────────────────────────────────────────
    sep1 = qt.QLabel("─── Manual Placement ───────────────────")
    sep1.setStyleSheet("color: #888; font-size: 10px;")
    stentLayout.addRow(sep1)

    # STEP 3 — Pick points manually
    self.stentPickButton = qt.QPushButton("① 📍 Straight Stent")
    self.stentPickButton.setToolTip(
        "Click two points on the 3D vessel — stent path is auto-traced between them"
    )
    self.stentPickButton.setStyleSheet(
        "background-color: #8e44ad; color: white; font-weight: bold; padding: 6px;"
    )

    # STEP 4 — Y-Stent picker
    self.stentPickYButton = qt.QPushButton("② 💋 Kissing / Y-Stent")
    self.stentPickYButton.setToolTip(
        "KISSING STENT — click 3 points in this order:\n\n"
        "  📍 Click 1: Anywhere on the IVC trunk ABOVE the bifurcation\n"
        "              (sets the proximal landing zone)\n\n"
        "  📍 Click 2: Deep in the LEFT iliac — at or PAST the lesion\n"
        "              (sets distal landing for left limb)\n\n"
        "  📍 Click 3: Deep in the RIGHT iliac — matching depth\n"
        "              (sets distal landing for right limb)\n\n"
        "⚠️  Do NOT cluster all clicks near the bifurcation!\n"
        "    Spread clicks along the full intended stent length.\n\n"
        "Change Stent type combo to Y/Trouser for bifurcation stent instead."
    )
    self.stentPickYButton.setStyleSheet(
        "background-color: #16a085; color: white; font-weight: bold; padding: 6px;"
    )

    # Auto kissing — no clicks needed
    self.autoKissButton = qt.QPushButton("⚡ Auto Kissing")
    self.autoKissButton.setToolTip(
        "Automatically place kissing stent.\n"
        "Proximal: Trunk coverage spinbox (default 20mm above hub).\n"
        "Distal: lesion location + 15mm margin, or spinbox, or 50% of branch.\n"
        "No clicks required — fine-tune with spinboxes after placement."
    )
    self.autoKissButton.setStyleSheet(
        "background-color: #8e44ad; color: white; font-weight: bold; padding: 6px;"
    )

    # Cancel
    self.stentPickCancelButton = qt.QPushButton("✕ Cancel")
    self.stentPickCancelButton.setStyleSheet(
        "background-color: #c0392b; color: white; padding: 6px;"
    )
    self.stentPickCancelButton.setEnabled(False)

    pickBtnWidget = qt.QWidget()
    pickBtnRow = qt.QHBoxLayout(pickBtnWidget)
    pickBtnRow.setContentsMargins(0, 0, 0, 0)
    pickBtnRow.addWidget(self.stentPickButton)
    pickBtnRow.addWidget(self.stentPickYButton)
    pickBtnRow.addWidget(self.autoKissButton)
    pickBtnRow.addWidget(self.stentPickCancelButton)

    # Kissing stent click guide
    _kissGuide = qt.QLabel(
        "<b>💋 Kissing stent (2 clicks):</b>  "
        "<span style='color:#c0392b'>① Deep in LEFT iliac (at/past lesion)</span>  →  "
        "<span style='color:#27ae60'>② Deep in RIGHT iliac</span>  "
        "<span style='color:#888'>(proximal IVC landing = automatic)</span>"
    )
    _kissGuide.setWordWrap(True)
    _kissGuide.setStyleSheet(
        "font-size:10px; padding:4px 2px; "
        "background:#EBF5FB; border-radius:4px; border:1px solid #AED6F1;"
    )
    stentLayout.addRow(_kissGuide)
    self.stentPickStatusLabel = qt.QLabel(
        "Click to pick start/end points on vessel"
    )
    self.stentPickStatusLabel.setStyleSheet(
        "font-size: 11px; color: #555; font-style: italic;"
    )
    self.stentPickStatusLabel.setWordWrap(True)
    stentLayout.addRow(pickBtnWidget)
    stentLayout.addRow(self.stentPickStatusLabel)

    # ── Y-stent branch diameter overrides ────────────────────────────
    _yDiamRow = qt.QHBoxLayout()
    _yDiamRow.addWidget(qt.QLabel("Left Ø:"))
    self.yLeftDiamSpin = qt.QDoubleSpinBox()
    self.yLeftDiamSpin.setRange(4.0, 40.0)
    self.yLeftDiamSpin.setSingleStep(0.5)
    self.yLeftDiamSpin.setValue(0.0)
    self.yLeftDiamSpin.setSuffix(" mm")
    self.yLeftDiamSpin.setFixedWidth(80)
    self.yLeftDiamSpin.setToolTip("Override left branch diameter. 0 = auto.")
    _yDiamRow.addWidget(self.yLeftDiamSpin)
    _yDiamRow.addWidget(qt.QLabel("  Right Ø:"))
    self.yRightDiamSpin = qt.QDoubleSpinBox()
    self.yRightDiamSpin.setRange(4.0, 40.0)
    self.yRightDiamSpin.setSingleStep(0.5)
    self.yRightDiamSpin.setValue(0.0)
    self.yRightDiamSpin.setSuffix(" mm")
    self.yRightDiamSpin.setFixedWidth(80)
    self.yRightDiamSpin.setToolTip("Override right branch diameter. 0 = auto.")
    _yDiamRow.addWidget(self.yRightDiamSpin)
    self.yReplacStentButton = qt.QPushButton("↺ Re-place")
    self.yReplacStentButton.setStyleSheet(
        "background-color: #2e86c1; color: white; font-weight: bold; padding: 5px;"
    )
    self.yReplacStentButton.setToolTip(
        "Re-place Y-stent using current diameter overrides without re-picking points.\n"
        "Use this to test asymmetric sizing (e.g. Left 16mm / Right 13.5mm)."
    )
    self.yReplacStentButton.setEnabled(False)
    _yDiamRow.addWidget(self.yReplacStentButton)
    stentLayout.addRow("Branch sizes:", _yDiamRow)
    self.yReplacStentButton.connect("clicked(bool)", self.onYStentReplace)

    # STEP 5 — Place / Remove
    self.stentPlaceButton = qt.QPushButton("③ 🔩 Place Stent in 3D")
    self.stentPlaceButton.setVisible(False)
    self.stentPlaceButton.setStyleSheet(
        "background-color: #e67e22; color: white; font-weight: bold; padding: 6px;"
    )
    self.stentRemoveButton = qt.QPushButton("✕ Remove Stent")
    self.stentRemoveButton.setStyleSheet(
        "background-color: #7f8c8d; color: white; padding: 4px;"
    )
    stentBtnWidget = qt.QWidget()
    stentBtnRow = qt.QHBoxLayout(stentBtnWidget)
    stentBtnRow.setContentsMargins(0, 0, 0, 0)
    stentBtnRow.addWidget(self.stentPlaceButton)
    stentBtnRow.addWidget(self.stentRemoveButton)
    stentLayout.addRow(stentBtnWidget)

    # ── Embolization Planner ──────────────────────────────────────────
    emboCollapsible = ctk.ctkCollapsibleButton()
    emboCollapsible.text = "Embolization Planner"
    emboCollapsible.collapsed = False
    self.layout.addWidget(emboCollapsible)
    emboLayout = qt.QFormLayout(emboCollapsible)

    # Device type selector
    self.emboDeviceCombo = qt.QComboBox()
    self.emboDeviceCombo.addItems(
        [
            "🌀 Coil Embolization",
            "🔵 Plug / Amplatzer Device",
            "💧 Liquid Embolic (Onyx/NBCA)",
            "🔧 Flow Diverter Stent",
        ]
    )
    emboLayout.addRow("Device Type:", self.emboDeviceCombo)

    # Target branch selector
    # Auto-detect button

    # Landing zone sliders
    emboLayout.addRow(qt.QLabel("── Landing Zones ──"))
    self.emboPickZoneButton = qt.QPushButton("📍 Pick Zone (2 points)")
    self.emboPickZoneButton.setStyleSheet(
        "background-color: #2980b9; color: white; font-weight: bold; padding: 5px;"
    )
    self.emboPickZoneButton.setEnabled(False)
    self.emboCancelPickButton = qt.QPushButton("✕ Cancel")
    self.emboCancelPickButton.setStyleSheet(
        "background-color: #555; color: white; padding: 5px;"
    )
    self.emboCancelPickButton.setEnabled(False)
    _emboPickRow = qt.QHBoxLayout()
    _emboPickRow.addWidget(self.emboPickZoneButton)
    _emboPickRow.addWidget(self.emboCancelPickButton)
    emboLayout.addRow(_emboPickRow)
    self.emboZoneLabel = qt.QLabel("No zone set")
    self.emboZoneLabel.setStyleSheet("color: #2980b9; font-size: 11px;")
    emboLayout.addRow(self.emboZoneLabel)

    # Device size
    self.emboSizeSpin = qt.QDoubleSpinBox()
    self.emboSizeSpin.setMinimum(1.0)
    self.emboSizeSpin.setMaximum(40.0)
    self.emboSizeSpin.setValue(8.0)
    self.emboSizeSpin.setSuffix(" mm")
    self.emboSizeSpin.setSingleStep(0.5)
    self.emboSizeSpin.setEnabled(False)
    emboLayout.addRow("Device Size:", self.emboSizeSpin)

    # Place / Remove buttons
    _emboPlaceRow = qt.QHBoxLayout()
    self.emboPlaceButton = qt.QPushButton("📍 Place Device")
    self.emboPlaceButton.setStyleSheet(
        "background-color: #c0392b; color: white; font-weight: bold; padding: 6px;"
    )
    self.emboPlaceButton.setEnabled(False)
    self.emboRemoveButton = qt.QPushButton("✕ Remove")
    self.emboRemoveButton.setStyleSheet(
        "background-color: #555; color: white; padding: 6px;"
    )
    self.emboRemoveButton.setEnabled(False)
    _emboPlaceRow.addWidget(self.emboPlaceButton)
    _emboPlaceRow.addWidget(self.emboRemoveButton)
    emboLayout.addRow(_emboPlaceRow)

    # Status label
    self.emboStatusLabel = qt.QLabel("")
    self.emboStatusLabel.setStyleSheet(
        "font-size: 11px; color: #c0392b; font-style: italic;"
    )
    self.emboStatusLabel.setWordWrap(True)
    emboLayout.addRow(self.emboStatusLabel)

    # =========================================================
    # CLOUDCOMPARE MAPPING
    # =========================================================
    ccCollapsible = ctk.ctkCollapsibleButton()
    ccCollapsible.text = "🗺 Open3D Distance Mapping"
    ccCollapsible.collapsed = True
    self.layout.addWidget(ccCollapsible)
    ccLayout = qt.QFormLayout(ccCollapsible)

    # ── Hint label ────────────────────────────────────────────────────────
    ccHint = qt.QLabel(
        "Compute per-point surface distances between two mesh models using "
        "Open3D — entirely in-process, no file export required. "
        "Result appears as a ColdToHot colour-mapped model node."
    )
    ccHint.setStyleSheet("color: #aaaaaa; font-size: 11px;")
    ccHint.setWordWrap(True)
    ccLayout.addRow(ccHint)

    # ── Installation note ─────────────────────────────────────────────────
    ccInstallNote = qt.QLabel(
        "⚠  Requires Open3D.  If not installed, run once in the Python console:\n"
        "    slicer.util.pip_install('open3d')"
    )
    ccInstallNote.setStyleSheet(
        "color: #e67e22; font-size: 10px; font-family: monospace;"
    )
    ccInstallNote.setWordWrap(True)
    ccLayout.addRow(ccInstallNote)

    # ── Comparison model selector ─────────────────────────────────────────
    self.ccCompareSelector = slicer.qMRMLNodeComboBox()
    self.ccCompareSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.ccCompareSelector.selectNodeUponCreation = False
    self.ccCompareSelector.addEnabled = False
    self.ccCompareSelector.removeEnabled = False
    self.ccCompareSelector.noneEnabled = True
    self.ccCompareSelector.setMRMLScene(slicer.mrmlScene)
    self.ccCompareSelector.setToolTip(
        "Model to compare against the vessel model selected in Step 1.\n"
        "Typically a follow-up scan, stent model, or external import."
    )
    ccLayout.addRow("Compare model:", self.ccCompareSelector)

    # ── Metric selector ───────────────────────────────────────────────────
    self.ccMetricCombo = qt.QComboBox()
    self.ccMetricCombo.addItems(["C2C Distance", "Hausdorff", "RMS"])
    self.ccMetricCombo.setCurrentIndex(0)
    self.ccMetricCombo.setToolTip(
        "C2C Distance : nearest-point distance for every vertex (default).\n"
        "Hausdorff    : symmetric max deviation — conservative worst-case.\n"
        "RMS          : global root-mean-square error shown as a flat colour."
    )
    ccLayout.addRow("Metric:", self.ccMetricCombo)

    # ── Action buttons ────────────────────────────────────────────────────
    _ccBtnRow = qt.QHBoxLayout()

    self.ccMapButton = qt.QPushButton("▶ Map with Open3D")
    self.ccMapButton.setStyleSheet(
        "background-color: #1a6fa8; color: white; font-weight: bold; padding: 6px;"
    )
    self.ccMapButton.setToolTip(
        "Compute per-point distances between the two models in-process\n"
        "and display the result as a ColdToHot colour map."
    )

    self.ccClearButton = qt.QPushButton("✕ Clear Mapping")
    self.ccClearButton.setStyleSheet(
        "background-color: #7f8c8d; color: white; padding: 6px;"
    )
    self.ccClearButton.setToolTip(
        "Remove all Open3D mapping result nodes from the scene."
    )

    _ccBtnRow.addWidget(self.ccMapButton)
    _ccBtnRow.addWidget(self.ccClearButton)
    ccLayout.addRow(_ccBtnRow)

    # ── Status label ──────────────────────────────────────────────────────
    self.ccStatusLabel = qt.QLabel("Ready")
    self.ccStatusLabel.setStyleSheet(
        "font-size: 11px; color: #1a6fa8; font-style: italic;"
    )
    self.ccStatusLabel.setWordWrap(True)
    ccLayout.addRow("Status:", self.ccStatusLabel)

    # Hidden stub so any legacy code referencing cloudComparePathEdit doesn't crash
    self.cloudComparePathEdit = qt.QLineEdit()
    self.cloudComparePathEdit.setVisible(False)

    # =========================================================
    # =========================================================
    # SURFACE SCALAR MAPS  (radius + wall thickness)
    # =========================================================
    smapCollapsible = ctk.ctkCollapsibleButton()
    smapCollapsible.text = "🌈 Surface Scalar Maps"
    smapCollapsible.collapsed = True
    self.layout.addWidget(smapCollapsible)
    smapLayout = qt.QFormLayout(smapCollapsible)

    # ── Section 1: Radius Map ─────────────────────────────────────────────
    _smapRadiusHdr = qt.QLabel("Radius Map")
    _smapRadiusHdr.setStyleSheet(
        "font-weight: bold; color: #d4ac0d; margin-top: 4px;"
    )
    smapLayout.addRow(_smapRadiusHdr)

    _smapRadiusHint = qt.QLabel(
        "Colours each vertex of the vessel surface by its distance to the "
        "nearest centerline point — a direct proxy for local vessel radius.  "
        "Run the centerline pipeline first (Step 2)."
    )
    _smapRadiusHint.setStyleSheet("color: #aaaaaa; font-size: 11px;")
    _smapRadiusHint.setWordWrap(True)
    smapLayout.addRow(_smapRadiusHint)

    _smapRadiusNote = qt.QLabel(
        "ℹ  Uses the vessel model selected in Step 1."
    )
    _smapRadiusNote.setStyleSheet("color: #5dade2; font-size: 10px;")
    smapLayout.addRow(_smapRadiusNote)

    self.smapRadiusButton = qt.QPushButton("▶ Radius Map")
    self.smapRadiusButton.setStyleSheet(
        "background-color: #1a5276; color: white; font-weight: bold; padding: 6px;"
    )
    self.smapRadiusButton.setToolTip(
        "Compute per-vertex radius on the Step-1 vessel surface.\n"
        "Result: <modelName>_RADIUS_MAP with ColdToHot colour map."
    )
    smapLayout.addRow(self.smapRadiusButton)

    # ── Divider ───────────────────────────────────────────────────────────
    _smapDiv = qt.QFrame()
    _smapDiv.setFrameShape(qt.QFrame.HLine)
    _smapDiv.setStyleSheet("color: #444444; margin: 6px 0px;")
    smapLayout.addRow(_smapDiv)

    # ── Section 2: Wall Thickness Map ─────────────────────────────────────
    _smapWallHdr = qt.QLabel("Wall Thickness Map")
    _smapWallHdr.setStyleSheet(
        "font-weight: bold; color: #d4ac0d; margin-top: 4px;"
    )
    smapLayout.addRow(_smapWallHdr)

    _smapWallHint = qt.QLabel(
        "Colours each vertex of the outer-wall mesh by its distance to the "
        "nearest vertex on the inner lumen surface — wall thickness per vertex.  "
        "Requires two separate segmentation meshes."
    )
    _smapWallHint.setStyleSheet("color: #aaaaaa; font-size: 11px;")
    _smapWallHint.setWordWrap(True)
    smapLayout.addRow(_smapWallHint)

    self.smapWallSelector = slicer.qMRMLNodeComboBox()
    self.smapWallSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.smapWallSelector.selectNodeUponCreation = False
    self.smapWallSelector.addEnabled = False
    self.smapWallSelector.removeEnabled = False
    self.smapWallSelector.noneEnabled = True
    self.smapWallSelector.setMRMLScene(slicer.mrmlScene)
    self.smapWallSelector.setToolTip("Outer vessel wall surface mesh.")
    smapLayout.addRow("Outer wall:", self.smapWallSelector)

    self.smapLumenSelector = slicer.qMRMLNodeComboBox()
    self.smapLumenSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.smapLumenSelector.selectNodeUponCreation = False
    self.smapLumenSelector.addEnabled = False
    self.smapLumenSelector.removeEnabled = False
    self.smapLumenSelector.noneEnabled = True
    self.smapLumenSelector.setMRMLScene(slicer.mrmlScene)
    self.smapLumenSelector.setToolTip("Inner lumen surface mesh.")
    smapLayout.addRow("Inner lumen:", self.smapLumenSelector)

    self.smapWallButton = qt.QPushButton("▶ Wall Thickness Map")
    self.smapWallButton.setStyleSheet(
        "background-color: #1a5276; color: white; font-weight: bold; padding: 6px;"
    )
    self.smapWallButton.setToolTip(
        "Compute per-vertex wall thickness (outer wall → nearest lumen vertex).\n"
        "Result: <wallModelName>_WALL_THICK with ColdToHot colour map."
    )
    smapLayout.addRow(self.smapWallButton)

    # ── Clear + Status (shared) ───────────────────────────────────────────
    _smapDiv2 = qt.QFrame()
    _smapDiv2.setFrameShape(qt.QFrame.HLine)
    _smapDiv2.setStyleSheet("color: #444444; margin: 6px 0px;")
    smapLayout.addRow(_smapDiv2)

    self.smapClearButton = qt.QPushButton("✕ Clear Maps")
    self.smapClearButton.setStyleSheet(
        "background-color: #7f8c8d; color: white; padding: 6px;"
    )
    self.smapClearButton.setToolTip(
        "Remove all _RADIUS_MAP and _WALL_THICK nodes from the scene."
    )
    smapLayout.addRow(self.smapClearButton)

    self.smapStatusLabel = qt.QLabel("Ready")
    self.smapStatusLabel.setStyleSheet(
        "font-size: 11px; color: #d4ac0d; font-style: italic;"
    )
    self.smapStatusLabel.setWordWrap(True)
    smapLayout.addRow("Status:", self.smapStatusLabel)

    # =========================================================
    # MESH REFINEMENT
    # =========================================================
    refineCollapsible = ctk.ctkCollapsibleButton()
    refineCollapsible.text = "🔧 Mesh Refinement"
    refineCollapsible.collapsed = False
    self.layout.addWidget(refineCollapsible)
    refineLayout = qt.QFormLayout(refineCollapsible)

    _brInfo = qt.QLabel(
        "Launches vessel_refinement_server.py as a background Python process, "
        "sends the vessel mesh for geometry repair and Taubin smoothing, then "
        "automatically loads the refined mesh and runs centerline extraction — "
        "one clean round-trip."
    )
    _brInfo.setWordWrap(True)
    _brInfo.setStyleSheet("font-size: 11px; color: #888888;")
    refineLayout.addRow(_brInfo)

    # Port + state dot row
    _rfPortRow = qt.QHBoxLayout()
    _rfPortLabel = qt.QLabel("Port:")
    _rfPortLabel.setStyleSheet("font-size: 11px;")
    self.refinePortSpin = qt.QSpinBox()
    self.refinePortSpin.setMinimum(1024)
    self.refinePortSpin.setMaximum(65535)
    self.refinePortSpin.setValue(6790)
    self.refinePortSpin.setFixedWidth(80)
    self.refinePortSpin.setToolTip(
        "TCP port the refinement server listens on (default 6790).\n"
        "Passed as argument when the server is started."
    )
    self.refineStateDot = qt.QLabel("●")
    self.refineStateDot.setStyleSheet("color: #c0392b; font-size: 14px;")
    self.refineStateDot.setToolTip("Server state: red = stopped, green = running")
    _rfPortRow.addWidget(_rfPortLabel)
    _rfPortRow.addWidget(self.refinePortSpin)
    _rfPortRow.addWidget(self.refineStateDot)
    _rfPortRow.addStretch(1)
    refineLayout.addRow(_rfPortRow)

    # Start / Stop server row
    _rfServerRow = qt.QHBoxLayout()
    self.refineServerStartButton = qt.QPushButton("▶ Start Server")
    self.refineServerStartButton.setStyleSheet(
        "background-color: #27ae60; color: white; font-weight: bold; padding: 6px;"
    )
    self.refineServerStartButton.setToolTip(
        "Launch vessel_refinement_server.py using Slicer's Python on the configured port.\n"
        "On first run, auto-installs open3d and pymeshfix (takes ~1 min).\n"
        "No-op if the server is already responding."
    )
    self.refineServerStopButton = qt.QPushButton("■ Stop Server")
    self.refineServerStopButton.setStyleSheet(
        "background-color: #c0392b; color: white; font-weight: bold; padding: 6px;"
    )
    self.refineServerStopButton.setToolTip(
        "Send POST /shutdown to the refinement server, then terminate the process."
    )
    _rfServerRow.addWidget(self.refineServerStartButton)
    _rfServerRow.addWidget(self.refineServerStopButton)
    refineLayout.addRow(_rfServerRow)

    # Server URL (auto-updated by port spin, also editable manually)
    self.refineApiUrlEdit = qt.QLineEdit("http://localhost:6790")
    self.refineApiUrlEdit.setToolTip(
        "Base URL of the refinement server (auto-updated from Port above).\n"
        "Default: http://localhost:6790"
    )
    self.refineApiUrlEdit.setPlaceholderText("http://localhost:6790")
    refineLayout.addRow("Server URL:", self.refineApiUrlEdit)

    # Keep URL in sync with port spin
    def _rf_sync_url(val):
        self.refineApiUrlEdit.setText(f"http://localhost:{val}")
    self.refinePortSpin.connect("valueChanged(int)", _rf_sync_url)

    # Refine button
    self.sendToRefineButton = qt.QPushButton("🔧 Refine Mesh")
    self.sendToRefineButton.setStyleSheet(
        "background-color: #d35400; color: white; font-weight: bold; padding: 6px;"
    )
    self.sendToRefineButton.setToolTip(
        "Export the current vessel model as STL, POST to the refinement server,\n"
        "load the refined STL as the active model, then run centerline extraction."
    )
    refineLayout.addRow(self.sendToRefineButton)

    # Status + progress
    self.refineStatusLabel = qt.QLabel("Server not started")
    self.refineStatusLabel.setStyleSheet(
        "font-size: 11px; color: #6c3483; font-style: italic;"
    )
    self.refineStatusLabel.setWordWrap(True)
    refineLayout.addRow("Status:", self.refineStatusLabel)

    self.refineProgressBar = qt.QProgressBar()
    self.refineProgressBar.setMinimum(0)
    self.refineProgressBar.setMaximum(0)   # indeterminate spinner while waiting
    self.refineProgressBar.setVisible(False)
    self.refineProgressBar.setStyleSheet(
        "QProgressBar { border: 1px solid #d35400; border-radius: 4px;"
        "  text-align: center; font-size: 11px; }"
        "QProgressBar::chunk { background-color: #d35400; border-radius: 3px; }"
    )
    refineLayout.addRow("Progress:", self.refineProgressBar)

    # MULTI-RAY RADIUS REFINEMENT
    # =========================================================
    bldCollapsible = ctk.ctkCollapsibleButton()
    bldCollapsible.text = "🔬 Multi-Ray Radius Refinement"
    bldCollapsible.collapsed = False
    self.layout.addWidget(bldCollapsible)
    bldLayout = qt.QFormLayout(bldCollapsible)

    _mrInfo = qt.QLabel(
        "Launches vessel_multiray_server.py as a background Python process using "
        "Slicer's own Python — no Blender required. "
        "Casts radial rays from each centerline point for accurate, "
        "confidence-scored radius estimates at bifurcations and elliptical vessels."
    )
    _mrInfo.setWordWrap(True)
    _mrInfo.setStyleSheet("font-size: 11px; color: #888888;")
    bldLayout.addRow(_mrInfo)

    # Port + state row
    _mrPortRow = qt.QHBoxLayout()
    _mrPortLabel = qt.QLabel("Port:")
    _mrPortLabel.setStyleSheet("font-size: 11px;")
    self.multirayPortSpin = qt.QSpinBox()
    self.multirayPortSpin.setMinimum(1024)
    self.multirayPortSpin.setMaximum(65535)
    self.multirayPortSpin.setValue(6789)
    self.multirayPortSpin.setToolTip(
        "TCP port the multi-ray server listens on (default 6789).\n"
        "Passed as argument when the server is started."
    )
    self.multirayPortSpin.setFixedWidth(80)
    self.multirayStateDot = qt.QLabel("●")
    self.multirayStateDot.setStyleSheet("color: #c0392b; font-size: 14px;")
    self.multirayStateDot.setToolTip("Server state: red = stopped, green = running")
    _mrPortRow.addWidget(_mrPortLabel)
    _mrPortRow.addWidget(self.multirayPortSpin)
    _mrPortRow.addWidget(self.multirayStateDot)
    _mrPortRow.addStretch(1)
    bldLayout.addRow(_mrPortRow)

    # Start / stop row
    _mrServerRow = qt.QHBoxLayout()
    self.blenderServerStartButton = qt.QPushButton("▶ Start Server")
    self.blenderServerStartButton.setStyleSheet(
        "background-color: #27ae60; color: white; font-weight: bold; padding: 6px;"
    )
    self.blenderServerStartButton.setToolTip(
        "Launch vessel_multiray_server.py using Slicer's Python on the configured port.\n"
        "No-op if the server is already responding."
    )
    self.blenderServerStopButton = qt.QPushButton("■ Stop Server")
    self.blenderServerStopButton.setStyleSheet(
        "background-color: #c0392b; color: white; font-weight: bold; padding: 6px;"
    )
    self.blenderServerStopButton.setToolTip(
        "Send POST /shutdown to the server, then terminate the process."
    )
    _mrServerRow.addWidget(self.blenderServerStartButton)
    _mrServerRow.addWidget(self.blenderServerStopButton)
    bldLayout.addRow(_mrServerRow)

    # Run / clear row
    _mrRunRow = qt.QHBoxLayout()
    self.multirayMapButton = qt.QPushButton("🔬 Run Multi-Ray Radii")
    self.multirayMapButton.setStyleSheet(
        "background-color: #1a6fa8; color: white; font-weight: bold; padding: 6px;"
    )
    self.multirayMapButton.setToolTip(
        "POST centerline branches to the multi-ray server and apply\n"
        "refined radius + confidence arrays to branchMeta.\n"
        "Requires: Extract Centerline + Run Analysis already completed.\n"
        "Requires: server running (click ▶ Start Server first)."
    )
    self.multirayClearButton = qt.QPushButton("✕ Clear Multi-Ray")
    self.multirayClearButton.setStyleSheet(
        "background-color: #7f8c8d; color: white; padding: 6px;"
    )
    self.multirayClearButton.setToolTip(
        "Remove multi-ray radius and confidence data from branchMeta."
    )
    _mrRunRow.addWidget(self.multirayMapButton)
    _mrRunRow.addWidget(self.multirayClearButton)
    bldLayout.addRow(_mrRunRow)

    self.multirayStatusLabel = qt.QLabel("Server not started")
    self.multirayStatusLabel.setStyleSheet(
        "font-size: 11px; color: #6c3483; font-style: italic;"
    )
    self.multirayStatusLabel.setWordWrap(True)
    bldLayout.addRow("Status:", self.multirayStatusLabel)

    self.multirayProgressBar = qt.QProgressBar()
    self.multirayProgressBar.setMinimum(0)
    self.multirayProgressBar.setMaximum(100)
    self.multirayProgressBar.setValue(0)
    self.multirayProgressBar.setFormat("%p%")
    self.multirayProgressBar.setTextVisible(True)
    self.multirayProgressBar.setVisible(False)   # hidden until a run starts
    self.multirayProgressBar.setStyleSheet(
        "QProgressBar { border: 1px solid #6c3483; border-radius: 4px;"
        "  text-align: center; font-size: 11px; }"
        "QProgressBar::chunk { background-color: #6c3483; border-radius: 3px; }"
    )
    bldLayout.addRow("Progress:", self.multirayProgressBar)

    # =========================================================
    # EXPORT
    # =========================================================
    exportCollapsible = ctk.ctkCollapsibleButton()
    exportCollapsible.text = "Export"
    self.layout.addWidget(exportCollapsible)
    exportLayout = qt.QFormLayout(exportCollapsible)

    self.exportButton = qt.QPushButton("💾 Export Measurements to CSV")
    exportLayout.addRow(self.exportButton)

    self.reportButton = qt.QPushButton("📄 Generate Clinical Report (DOCX)")
    self.reportButton.setStyleSheet(
        "background-color: #1F4E79; color: white; font-weight: bold; padding: 6px;"
    )
    exportLayout.addRow(self.reportButton)

    self.detectCollateralsButton = qt.QPushButton("🩸 Detect Collaterals")
    self.detectCollateralsButton.setVisible(False)
    self.detectCollateralsButton.setStyleSheet(
        "background-color: #00838f; color: white; font-weight: bold; padding: 6px;"
    )
    exportLayout.addRow(self.detectCollateralsButton)

    self.clearOverlayButton = qt.QPushButton("✕ Clear Color Overlay")
    self.clearOverlayButton.setVisible(False)
    self.clearOverlayButton.setStyleSheet(
        "background-color: #7f8c8d; color: white; padding: 4px;"
    )
    exportLayout.addRow(self.clearOverlayButton)

    self.layout.addStretch(1)
    # --- Connect signals ---
    self.autoDetectButton.connect("clicked(bool)", self.onAutoDetect)
    self.placeEndpointsButton.connect("clicked(bool)", self.onPlaceEndpoints)
    self.cancelPlaceButton.connect("clicked(bool)", self.onCancelPlace)
    self.clearEndpointsButton.connect("clicked(bool)", self.onClearEndpoints)
    self.extractButton.connect("clicked(bool)", self.onExtractCenterline)
    self.drawCenterlineButton.connect(
        "clicked(bool)", self.onDrawCenterlineManually
    )
    self.stopDrawCenterlineButton.connect(
        "clicked(bool)", self.onStopDrawCenterline
    )
    self.useManualCenterlineButton.connect(
        "clicked(bool)", self.onUseManualCenterline
    )
    self.clearManualCenterlineButton.connect(
        "clicked(bool)", self.onClearManualCenterline
    )
    self.refineCenterlineButton.connect("clicked(bool)", self.onRefineCenterline)
    self.autoTuneButton.connect("clicked(bool)", self.onAutoTune)
    self.loadIVUSButton.connect("clicked(bool)", self.onLoadIVUS)
    self.centerlineVisButton.connect("toggled(bool)", self.onCenterlineVisToggle)
    self.centerlineSelector.connect(
        "currentNodeChanged(vtkMRMLNode*)", self.onCenterlineNodeChanged
    )
    self.pointSlider.connect("valueChanged(double)", self.onSliderChanged)
    self.prevButton.connect("clicked(bool)", self.onPrevPoint)
    self.nextButton.connect("clicked(bool)", self.onNextPoint)
    self.endoluminalButton.connect("toggled(bool)", self.onEndoluminalToggle)
    self.goToMinButton.connect("clicked(bool)", self.onGoToMin)
    self.goToMaxButton.connect("clicked(bool)", self.onGoToMax)
    self.branchSelector.connect("currentIndexChanged(int)", self.onBranchChanged)
    self.branchGoToMinButton.connect("clicked(bool)", self.onGoToMin)
    self.branchGoToMaxButton.connect("clicked(bool)", self.onGoToMax)
    self.exportButton.connect("clicked(bool)", self.onExport)
    self.reportButton.connect("clicked(bool)", self.onGenerateReport)
    self.detectFindingsButton.connect("clicked(bool)", self.onDetectFindings)
    self.detectCollateralsButton.connect("clicked(bool)", self.onDetectCollaterals)
    self.clearOverlayButton.connect("clicked(bool)", self.onClearOverlay)
    self.stentBranchCombo.connect(
        "currentIndexChanged(int)", self.onStentBranchChanged
    )
    self.stentBranchList.connect(
        "itemSelectionChanged()", self.onStentBranchListChanged
    )
    self.stentProxSlider.connect("valueChanged(int)", self.onStentSliderChanged)
    self.stentDistSlider.connect("valueChanged(int)", self.onStentSliderChanged)
    self.stentProxDiamSpin.connect("valueChanged(double)", self.onStentParamChanged)
    self.stentDistDiamSpin.connect("valueChanged(double)", self.onStentParamChanged)
    self.stentTypeCombo.connect(
        "currentIndexChanged(int)", self.onStentParamChanged
    )
    self.stentAutoSizeButton.connect("clicked(bool)", self.onStentAutoSize)
    self.stentAutoTypeButton.connect("clicked(bool)", self.onAutoDetectStentType)

    self.stentLengthApplyBtn.connect("clicked(bool)", self.onStentLengthApply)
    self.stentTrunkSpin.connect("valueChanged(double)", self.onTrunkSpinChanged)
    self.stentLengthSpin.connect("valueChanged(double)", self.onStentLengthApply)
    self.kissApplyBtn.connect("clicked(bool)", self.onKissingApply)
    self.kissProxExtSpin.connect("valueChanged(double)", self.onKissingApply)
    self.kissDistASpin.connect("valueChanged(double)", self.onKissingApply)
    self.kissDistBSpin.connect("valueChanged(double)", self.onKissingApply)
    self.kissNomCombo.connect("currentIndexChanged(int)", self.onKissNomChanged)
    self.stentPlaceButton.connect("clicked(bool)", self.onStentPlace)
    self.stentPickButton.connect("clicked(bool)", self.onStentPickPoints)
    self.preDilateButton.connect("clicked(bool)", self.onPreDilate)
    self.copyCoordButton.connect("clicked(bool)", self.onCopyCoordinates)
    self.compareBeforeAfterButton.connect(
        "clicked(bool)", self.onCompareBeforeAfter
    )
    self.modelOpacitySlider.connect(
        "valueChanged(double)", self.onModelOpacityChanged
    )
    self.modelVisButton.connect("toggled(bool)", self.onModelVisToggle)
    self.emboDeviceCombo.connect(
        "currentIndexChanged(int)", self.onEmboDeviceChanged
    )

    self.emboPickZoneButton.connect("clicked(bool)", self.onEmboPickZone)
    self.emboCancelPickButton.connect("clicked(bool)", self.onEmboCancelPick)
    self.emboPlaceButton.connect("clicked(bool)", self.onEmboPlace)
    self.emboRemoveButton.connect("clicked(bool)", self.onEmboRemove)
    self.exitCompareButton.connect("clicked(bool)", self.onExitCompare)
    self.addRulerButton.connect("clicked(bool)", self.onAddRuler)
    self.removeRulerButton.connect("clicked(bool)", self.onRemoveRuler)
    self.stentPickYButton.connect("clicked(bool)", self.onStentPickY)
    self.autoKissButton.connect("clicked(bool)", self.onAutoPlaceKissing)
    self.stentPickCancelButton.connect("clicked(bool)", self.onStentPickCancel)
    self.stentRemoveButton.connect("clicked(bool)", self.onStentRemove)
    self.sphereColorButton.connect("clicked(bool)", self.onPickColor)
    self.modelSelector.connect(
        "currentNodeChanged(vtkMRMLNode*)", self.onModelSelected
    )
    # ── CloudCompare mapping ──────────────────────────────────────────────
    self.ccMapButton.connect("clicked(bool)", self.onCloudCompareMap)
    self.ccClearButton.connect("clicked(bool)", self.onCloudCompareClear)
    # ── Surface scalar maps ───────────────────────────────────────────────
    self.smapRadiusButton.connect("clicked(bool)", self.onRadiusMap)
    self.smapWallButton.connect("clicked(bool)",   self.onWallThicknessMap)
    self.smapClearButton.connect("clicked(bool)",  self.onSurfaceMapClear)

    # ── Mesh refinement ───────────────────────────────────────────────────────
    self.refineServerStartButton.connect("clicked(bool)", self.onRefineServerStart)
    self.refineServerStopButton.connect("clicked(bool)",  self.onRefineServerStop)
    self.sendToRefineButton.connect("clicked(bool)", self.onSendToBlender)

    # ── Multi-ray radius refinement ───────────────────────────────────────────
    self.blenderServerStartButton.connect("clicked(bool)", self.onBlenderServerStart)
    self.blenderServerStopButton.connect("clicked(bool)",  self.onBlenderServerStop)
    self.multirayMapButton.connect("clicked(bool)",        self.onMultiRayBlenderMap)
    self.multirayClearButton.connect("clicked(bool)",      self.onMultiRayBlenderClear)

    from VesselAnalyzer import VesselAnalyzerLogic
    self.logic = VesselAnalyzerLogic()
    self.logic._widget = self
    self.logic.cleanup()



# ── Slicer module-scanner guard ───────────────────────────────────────────────
# Slicer auto-scans all .py files in the module folder and expects a class
# matching the filename.  This stub satisfies that requirement without
# registering as an actual loadable module (no ScriptedLoadableModule base).
class vessel_analyzer_ui:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        pass



# ── Slicer module-scanner guard ───────────────────────────────────────────────
class vessel_analyzer_ui:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_analyzer_ui"
            parent.hidden = True
