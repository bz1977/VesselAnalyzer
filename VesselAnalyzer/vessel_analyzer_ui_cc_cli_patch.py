# vessel_analyzer_ui_cc_cli_patch.py
# Patch function — extends the existing Open3D Distance Mapping collapsible
# with a CloudCompare ICP Alignment sub-panel.
#
# HOW TO APPLY
# ─────────────
# In vessel_analyzer_ui.py, at the end of build_ui(self), BEFORE the
# "EXPORT" collapsible block, call:
#
#     from vessel_analyzer_ui_cc_cli_patch import build_cc_cli_panel
#     build_cc_cli_panel(self)
#
# The patch reuses the existing self.ccCompareSelector and self.ccMetricCombo
# from the Open3D panel (no duplication).  It adds:
#
#   self.ccExePathEdit          — QLineEdit  for the CloudCompare.exe path
#   self.ccExeBrowseButton      — QPushButton to browse for the exe
#   self.ccAlignButton          — QPushButton "⚙ Align & Map"
#   self.ccAlignClearButton     — QPushButton "✕ Clear Alignment"
#
# The existing self.ccStatusLabel is reused for status output.
#
# Signal connections (add to the signal-wiring block at the bottom of build_ui):
#   self.ccAlignButton.connect("clicked(bool)", self.onCCAlign)
#   self.ccAlignClearButton.connect("clicked(bool)", self.onCCAlignClear)
#   self.ccExeBrowseButton.connect("clicked(bool)", self._onBrowseCCExe)
# ──────────────────────────────────────────────────────────────────────────────

import os
import qt
import ctk


def build_cc_cli_panel(self):
    """Append the CC CLI alignment sub-panel to the build_ui(self) layout.

    Must be called after the main Open3D Distance Mapping collapsible block
    so that self.ccCompareSelector, self.ccMetricCombo, and self.ccStatusLabel
    already exist.
    """

    # ── Collapsible container ──────────────────────────────────────────────
    ccCliCollapsible = ctk.ctkCollapsibleButton()
    ccCliCollapsible.text = "⚙ CloudCompare ICP Alignment"
    ccCliCollapsible.collapsed = True
    self.layout.addWidget(ccCliCollapsible)
    ccCliLayout = qt.QFormLayout(ccCliCollapsible)

    # ── Description ───────────────────────────────────────────────────────
    ccCliHint = qt.QLabel(
        "Use CloudCompare CLI to ICP-align the comparison mesh to the "
        "vessel reference before running Open3D distance mapping.  "
        "The aligned mesh is loaded back into the Slicer scene automatically."
    )
    ccCliHint.setStyleSheet("color: #aaaaaa; font-size: 11px;")
    ccCliHint.setWordWrap(True)
    ccCliLayout.addRow(ccCliHint)

    # ── CC executable path ────────────────────────────────────────────────
    _exeRow = qt.QHBoxLayout()

    self.ccExePathEdit = qt.QLineEdit()
    self.ccExePathEdit.setPlaceholderText(
        r"C:\Program Files\CloudCompare\CloudCompare.exe"
    )
    self.ccExePathEdit.setToolTip(
        "Full path to the CloudCompare executable.\n"
        "Download the portable build from cloudcompare.org."
    )

    self.ccExeBrowseButton = qt.QPushButton("Browse…")
    self.ccExeBrowseButton.setMaximumWidth(72)
    self.ccExeBrowseButton.setToolTip(
        "Browse to CloudCompare.exe on disk."
    )

    _exeRow.addWidget(self.ccExePathEdit)
    _exeRow.addWidget(self.ccExeBrowseButton)
    ccCliLayout.addRow("CC Executable:", _exeRow)

    # ── ICP parameter summary (informational) ─────────────────────────────
    _icpNote = qt.QLabel(
        "ICP settings:  50 iterations · 80 % overlap · convergence 1 × 10⁻⁶ mm"
    )
    _icpNote.setStyleSheet(
        "color: #888888; font-size: 10px; font-family: monospace;"
    )
    ccCliLayout.addRow(_icpNote)

    # ── Note: compare model / metric reused from Open3D panel ────────────
    _reuseNote = qt.QLabel(
        "ℹ  Compare model and metric are shared with the Open3D panel above."
    )
    _reuseNote.setStyleSheet("color: #5dade2; font-size: 10px;")
    _reuseNote.setWordWrap(True)
    ccCliLayout.addRow(_reuseNote)

    # ── Action buttons ────────────────────────────────────────────────────
    _ccCliBtnRow = qt.QHBoxLayout()

    self.ccAlignButton = qt.QPushButton("⚙ Align & Map")
    self.ccAlignButton.setStyleSheet(
        "background-color: #117a65; color: white; font-weight: bold; padding: 6px;"
    )
    self.ccAlignButton.setToolTip(
        "Run CloudCompare ICP alignment, then apply Open3D distance mapping\n"
        "to the registered mesh.  Result appears as a colour-mapped model node."
    )

    self.ccAlignClearButton = qt.QPushButton("✕ Clear Alignment")
    self.ccAlignClearButton.setStyleSheet(
        "background-color: #7f8c8d; color: white; padding: 6px;"
    )
    self.ccAlignClearButton.setToolTip(
        "Remove all CC-aligned and O3D-mapped nodes from the scene."
    )

    _ccCliBtnRow.addWidget(self.ccAlignButton)
    _ccCliBtnRow.addWidget(self.ccAlignClearButton)
    ccCliLayout.addRow(_ccCliBtnRow)

    # Status is shared — reuse ccStatusLabel from the Open3D panel.
    # No second label needed.


def wire_cc_cli_signals(self):
    """Connect CC CLI panel signals.  Call at the end of the signal-wiring block.

    In vessel_analyzer_ui.py, at the end of build_ui(self) where other
    .connect() calls are made, add:

        from vessel_analyzer_ui_cc_cli_patch import wire_cc_cli_signals
        wire_cc_cli_signals(self)
    """
    self.ccAlignButton.connect("clicked(bool)",      self.onCCAlign)
    self.ccAlignClearButton.connect("clicked(bool)", self.onCCAlignClear)
    self.ccExeBrowseButton.connect("clicked(bool)",  self._onBrowseCCExe)


# ── Browser helper — bound into widget class (not a mixin method) ─────────────

def _onBrowseCCExe(self):
    """QFileDialog to select CloudCompare.exe.  Bind into widget class."""
    path = qt.QFileDialog.getOpenFileName(
        None,
        "Select CloudCompare Executable",
        self.ccExePathEdit.text or os.path.expanduser("~"),
        "Executables (*.exe);;All files (*)",
    )
    if path:
        self.ccExePathEdit.setText(path)


# ══════════════════════════════════════════════════════════════════════════════
# Slicer module stub
# ══════════════════════════════════════════════════════════════════════════════
# Slicer auto-scans every .py in the module directory and tries to instantiate
# a class whose name matches the file stem.  Mixins are plain Python files with
# no such class, which causes "Failed to load scripted loadable module" errors
# at startup.  This minimal stub satisfies the scanner without registering any
# real UI or logic — the file is still imported normally by VesselAnalyzer.py.

try:
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule

    class vessel_analyzer_ui_cc_cli_patch(ScriptedLoadableModule):
        """Slicer stub — this file is a mixin/patch, not a standalone module."""
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "vessel_analyzer_ui_cc_cli_patch"
            parent.hidden = True   # keeps it out of the Modules menu
except ImportError:
    pass  # outside Slicer (unit tests, CI) — stub not needed
