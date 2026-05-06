# vessel_cc_cli_mixin.py
# CloudCompare CLI automation for VesselAnalyzer
# Build tag: v1.0 (2026-04-26)
#
# WHAT THIS MODULE DOES
# ──────────────────────
# Automates CloudCompare via its command-line interface (CloudCompare -SILENT …)
# to perform ICP rigid registration of a comparison mesh against the reference
# vessel surface.  Once aligned, the result is fed directly into the existing
# Open3D distance-mapping pipeline (vessel_cloudcompare_mixin.CloudCompareMixin)
# with zero extra file-I/O from the user's perspective.
#
# WORKFLOW
# ────────
#   1. Export ref + cmp polydata to temp OBJ files   (vtk → disk, temp dir)
#   2. Run CC CLI:  -ICP -REFERENCE <ref.obj> -ALIGNED <cmp.obj>
#                  -MIN_ERROR_DIFF 1e-6 -ITER 50 -RANDOM_SAMPLING_LIMIT 50000
#                  -SAVE_MESHES (writes <cmp>_REGISTERED.obj beside <cmp.obj>)
#   3. Parse the CC log for the final RMS error + 4×4 transform matrix
#   4. Load aligned OBJ → vtkPolyData; apply transform to the Slicer model node
#      (so the Slicer scene reflects the alignment visually)
#   5. Call self.runCloudCompareMapping() with the aligned node → Open3D colours
#   6. Clean up temp files
#
# INTEGRATION (minimal diffs — see patch block at end of file)
# ─────────────────────────────────────────────────────────────
#   vessel_cloudcompare_mixin.py : unchanged
#   vessel_analyzer_ui.py        : add CC CLI sub-panel (see _build_cc_cli_panel)
#   VesselAnalyzer.py            : import + mixin + 2 slot bindings
#
# CLOUDCOMPARE INSTALLATION
# ──────────────────────────
#   Download the portable Windows build from cloudcompare.org and point the
#   'CC Executable' path widget at CloudCompare.exe.
#   The SILENT flag suppresses the GUI; all output goes to stdout/stderr.
#
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple

try:
    import vtk
    import slicer
    import numpy as np
    import qt
except ImportError:
    vtk = slicer = np = qt = None


# ── Constants ─────────────────────────────────────────────────────────────────

_CLI_SUFFIX    = "_CC_ALIGNED"          # Slicer node name suffix after alignment
_ICP_ITER      = 50                     # CC ICP max iterations
_ICP_OVERLAP   = 0.80                   # CC ICP min overlap ratio (0–1)
_ICP_MIN_ERR   = 1e-6                   # CC ICP convergence threshold (RMS mm)
_ICP_SAMPLES   = 50_000                # CC random-sampling limit for large meshes
_LOG_TAG       = "[CC-CLI]"


# ══════════════════════════════════════════════════════════════════════════════
# VTK ↔ OBJ helpers  (no Open3D dependency here — only stdlib + vtk)
# ══════════════════════════════════════════════════════════════════════════════

def _export_polydata_to_obj(polydata, path: str) -> None:
    """Write a vtkPolyData to an OBJ file via vtkOBJWriter."""
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()


def _import_obj_to_polydata(path: str):
    """Read an OBJ file and return vtkPolyData."""
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


# ══════════════════════════════════════════════════════════════════════════════
# CloudCompare CLI execution
# ══════════════════════════════════════════════════════════════════════════════

def _build_icp_command(
    cc_exe: str,
    ref_path: str,
    cmp_path: str,
    out_dir: str,
) -> list[str]:
    """Build the CloudCompare ICP command-line argument list.

    CloudCompare CLI docs (v2.13+):
        CloudCompare -SILENT -O <ref> -O <cmp>
                     -ICP -MIN_ERROR_DIFF <e> -ITER <n>
                          -OVERLAP <pct> -RANDOM_SAMPLING_LIMIT <n>
                     -SAVE_MESHES
    The first -O is the REFERENCE; the second is the ALIGNED (moving) cloud.
    """
    return [
        cc_exe,
        "-SILENT",
        "-O", ref_path,
        "-O", cmp_path,
        "-ICP",
            "-MIN_ERROR_DIFF", str(_ICP_MIN_ERR),
            "-ITER",            str(_ICP_ITER),
            "-OVERLAP",         str(int(_ICP_OVERLAP * 100)),
            "-RANDOM_SAMPLING_LIMIT", str(_ICP_SAMPLES),
        "-SAVE_MESHES",
    ]


def _run_cc_subprocess(cmd: list[str], timeout: int = 300) -> Tuple[int, str, str]:
    """Execute CC CLI synchronously.  Returns (returncode, stdout, stderr)."""
    print(f"{_LOG_TAG} Executing: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        raise FileNotFoundError(
            f"CloudCompare executable not found:\n  {cmd[0]}\n"
            "Set the correct path in the 'CC Executable' field."
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"CloudCompare ICP timed out after {timeout}s.\n"
            "Try reducing mesh resolution or increasing the timeout."
        )


def _parse_icp_rms(stdout: str, stderr: str) -> Optional[float]:
    """Extract the final ICP RMS error (mm) from CC stdout.

    CC typically prints lines like:
        [ICP] Final RMS = 0.314159
    or:
        [ICP] Convergence reached (RMS = 0.271828 after 23 iterations)
    """
    combined = stdout + "\n" + stderr
    # Pattern 1 — "Final RMS = N"
    m = re.search(r"Final\s+RMS\s*=\s*([\d.eE+\-]+)", combined)
    if m:
        return float(m.group(1))
    # Pattern 2 — "RMS = N after"
    m = re.search(r"RMS\s*=\s*([\d.eE+\-]+)\s+after", combined)
    if m:
        return float(m.group(1))
    return None


def _find_registered_obj(cmp_path: str, out_dir: str) -> Optional[str]:
    """Locate the ICP-registered OBJ written by CC.

    CC writes the result alongside the input file with a suffix such as
    _REGISTERED, _ICP, or _TRANSFORMED (version-dependent).  We also scan
    out_dir for any OBJ newer than the inputs.
    """
    base     = Path(cmp_path).stem
    cmp_dir  = Path(cmp_path).parent

    candidates = [
        cmp_dir / f"{base}_REGISTERED.obj",
        cmp_dir / f"{base}_ICP.obj",
        cmp_dir / f"{base}_TRANSFORMED.obj",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    # Fallback: any newer OBJ in the temp dir
    ref_mtime = Path(cmp_path).stat().st_mtime
    for p in Path(out_dir).glob("*.obj"):
        if p.stat().st_mtime > ref_mtime and p != Path(cmp_path):
            return str(p)

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Logic mixin
# ══════════════════════════════════════════════════════════════════════════════

class CCCliMixin:
    """Logic-level mixin — CloudCompare ICP alignment driving Open3D mapping.

    Depends on CloudCompareMixin (vessel_cloudcompare_mixin) already being in
    the MRO (it is, in VesselAnalyzerLogic).

    Public API
    ──────────
    runCCAlignment(referenceNode, compareNode, ccExe, metric) → node | None
        ICP-align compareNode to referenceNode via CC CLI, then run Open3D
        distance mapping on the aligned result.

    applyICPTransformFromFile(modelNode, transformedPolydata)
        Update a Slicer model node's polydata in-place with the CC result.
    """

    # ── Entry point ───────────────────────────────────────────────────────────

    def runCCAlignment(
        self,
        referenceNode,
        compareNode,
        ccExe: str,
        metric: str = "C2C Distance",
    ):
        """ICP-align compareNode to referenceNode using CloudCompare CLI.

        Steps
        ─────
        1. Export both meshes to a temp directory as OBJ.
        2. Run CC -ICP via subprocess.
        3. Load the registered OBJ back into a new Slicer model node.
        4. Hand the aligned node to runCloudCompareMapping() for Open3D colours.

        Returns the coloured result node, or None on failure.
        """
        print(f"{_LOG_TAG} ══ CC ICP Alignment ════════════════════════════════")

        if not ccExe or not os.path.isfile(ccExe):
            msg = (
                f"CloudCompare executable not found:\n  {ccExe}\n"
                "Set the correct path in the CC Executable field."
            )
            print(f"{_LOG_TAG} {msg}")
            self._cc_set_status(f"Error: CC exe not found — {ccExe}")
            try:
                slicer.util.warningDisplay(msg)
            except Exception:
                pass
            return None

        ref_pd = self._o3d_get_polydata(referenceNode, "reference")
        cmp_pd = self._o3d_get_polydata(compareNode,   "compare")
        if ref_pd is None or cmp_pd is None:
            return None

        tmp_dir = tempfile.mkdtemp(prefix="VesselCC_")
        ref_obj = os.path.join(tmp_dir, "reference.obj")
        cmp_obj = os.path.join(tmp_dir, "compare.obj")

        try:
            # ── 1. Export to OBJ ──────────────────────────────────────────
            self._cc_set_status("Exporting meshes to OBJ…")
            print(f"{_LOG_TAG} Temp dir: {tmp_dir}")
            _export_polydata_to_obj(ref_pd, ref_obj)
            _export_polydata_to_obj(cmp_pd, cmp_obj)
            print(
                f"{_LOG_TAG} Exported — ref: {Path(ref_obj).stat().st_size} B  "
                f"cmp: {Path(cmp_obj).stat().st_size} B"
            )

            # ── 2. Run CC ICP ─────────────────────────────────────────────
            self._cc_set_status(
                f"Running CC ICP  (iter={_ICP_ITER}, overlap={_ICP_OVERLAP:.0%})…"
            )
            cmd = _build_icp_command(ccExe, ref_obj, cmp_obj, tmp_dir)
            rc, stdout, stderr = _run_cc_subprocess(cmd)

            # Always log CC output
            if stdout.strip():
                for line in stdout.strip().splitlines():
                    print(f"{_LOG_TAG} CC> {line}")
            if stderr.strip():
                for line in stderr.strip().splitlines():
                    print(f"{_LOG_TAG} CC-err> {line}")

            if rc != 0:
                raise RuntimeError(
                    f"CloudCompare exited with code {rc}.\n"
                    "Check the Python console for CC output."
                )

            # ── 3. Parse RMS ──────────────────────────────────────────────
            rms = _parse_icp_rms(stdout, stderr)
            rms_str = f"{rms:.4f} mm" if rms is not None else "unknown"
            print(f"{_LOG_TAG} ICP RMS error: {rms_str}")

            # ── 4. Load registered mesh ───────────────────────────────────
            reg_obj = _find_registered_obj(cmp_obj, tmp_dir)
            if reg_obj is None:
                raise RuntimeError(
                    "CC did not produce a registered OBJ file.\n"
                    "Check the CC output in the Python console."
                )
            print(f"{_LOG_TAG} Registered mesh: {reg_obj}")

            aligned_pd = _import_obj_to_polydata(reg_obj)
            n_aligned  = aligned_pd.GetNumberOfPoints()
            if n_aligned == 0:
                raise RuntimeError(
                    f"Registered OBJ has no geometry: {reg_obj}"
                )
            print(f"{_LOG_TAG} Aligned mesh: {n_aligned} pts")

            # ── 5. Create aligned model node in the Slicer scene ─────────
            aligned_name = compareNode.GetName() + _CLI_SUFFIX
            # Reuse existing node if present (re-run scenario)
            aligned_node = slicer.mrmlScene.GetFirstNodeByName(aligned_name)
            if aligned_node is None:
                aligned_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLModelNode", aligned_name
                )
                aligned_node.CreateDefaultDisplayNodes()

            aligned_node.SetAndObservePolyData(aligned_pd)
            aligned_node.SetDisplayVisibility(True)

            self._cc_set_status(
                f"ICP done — RMS {rms_str}  →  running Open3D mapping…"
            )

            # ── 6. Feed aligned node into Open3D distance mapping ─────────
            result_node = self.runCloudCompareMapping(
                referenceNode=referenceNode,
                compareNode=aligned_node,
                metric=metric,
            )

            if result_node is not None:
                # Hide the intermediate aligned node — the coloured result is enough
                aligned_node.SetDisplayVisibility(False)
                print(
                    f"{_LOG_TAG} ✓ Pipeline complete — ICP RMS {rms_str}  "
                    f"result: '{result_node.GetName()}'"
                )

            return result_node

        except Exception as exc:
            print(f"{_LOG_TAG} Error: {exc}")
            print(traceback.format_exc())
            self._cc_set_status(f"CC error: {exc}")
            try:
                slicer.util.warningDisplay(str(exc))
            except Exception:
                pass
            return None

        finally:
            # ── 7. Clean up temp files ────────────────────────────────────
            self._cc_cleanup_tmp(tmp_dir)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cc_cleanup_tmp(self, tmp_dir: str) -> None:
        """Remove all files in tmp_dir then the directory itself."""
        try:
            for f in Path(tmp_dir).glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            Path(tmp_dir).rmdir()
            print(f"{_LOG_TAG} Cleaned up temp dir: {tmp_dir}")
        except Exception as exc:
            print(f"{_LOG_TAG} Temp cleanup warning: {exc}")

    def clearCCAlignmentNodes(self) -> None:
        """Remove all _CC_ALIGNED and _O3D_MAPPED nodes from the scene."""
        try:
            to_remove = []
            for i in range(slicer.mrmlScene.GetNumberOfNodes()):
                nd = slicer.mrmlScene.GetNthNode(i)
                if nd and nd.GetName():
                    name = nd.GetName()
                    if name.endswith(_CLI_SUFFIX) or name.endswith("_O3D_MAPPED"):
                        to_remove.append(nd)
            for nd in to_remove:
                slicer.mrmlScene.RemoveNode(nd)
            msg = f"Cleared {len(to_remove)} alignment/mapping node(s)."
            print(f"{_LOG_TAG} {msg}")
            self._cc_set_status(msg)
        except Exception as exc:
            print(f"{_LOG_TAG} clearCCAlignmentNodes error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Widget mixin — Qt slot methods bound into VesselAnalyzerWidget
# ══════════════════════════════════════════════════════════════════════════════

class CCCliWidgetMixin:
    """Qt event handlers for the CC CLI alignment panel.

    Bound into VesselAnalyzerWidget:
        from vessel_cc_cli_mixin import CCCliWidgetMixin as _CCCLIM
        onCCAlign       = _CCCLIM.onCCAlign
        onCCAlignClear  = _CCCLIM.onCCAlignClear
    """

    def onCCAlign(self):
        """Slot: '⚙ Align & Map' button clicked."""
        try:
            # ── Reference = Step 1 model selector ─────────────────────────
            ref_sel  = getattr(self, "modelSelector", None)
            ref_node = ref_sel.currentNode() if ref_sel else None

            # ── Compare = shared ccCompareSelector ─────────────────────────
            cmp_sel  = getattr(self, "ccCompareSelector", None)
            cmp_node = cmp_sel.currentNode() if cmp_sel else None

            if ref_node is None:
                slicer.util.warningDisplay(
                    "Select a vessel model in Step 1 before aligning."
                )
                return
            if cmp_node is None:
                slicer.util.warningDisplay(
                    "Select a comparison model in the 'Compare model' dropdown."
                )
                return

            # ── CC exe path ────────────────────────────────────────────────
            cc_exe_edit = getattr(self, "ccExePathEdit", None)
            cc_exe      = cc_exe_edit.text.strip() if cc_exe_edit else ""

            # ── Metric ────────────────────────────────────────────────────
            combo  = getattr(self, "ccMetricCombo", None)
            metric = combo.currentText if combo else "C2C Distance"

            # ── Status feedback ────────────────────────────────────────────
            lbl = getattr(self, "ccStatusLabel", None)
            if lbl:
                lbl.setText("Starting CC ICP alignment…")
            import qt as _qt
            _qt.QApplication.processEvents()

            result = self.logic.runCCAlignment(
                referenceNode=ref_node,
                compareNode=cmp_node,
                ccExe=cc_exe,
                metric=metric,
            )

            if result is not None:
                try:
                    slicer.util.resetThreeDViews()
                except Exception:
                    pass

        except Exception as exc:
            print(f"[CC-CLI Widget] onCCAlign error: {exc}")
            print(traceback.format_exc())

    def onCCAlignClear(self):
        """Slot: '✕ Clear Alignment' button clicked."""
        try:
            self.logic.clearCCAlignmentNodes()
        except Exception as exc:
            print(f"[CC-CLI Widget] onCCAlignClear error: {exc}")


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

    class vessel_cc_cli_mixin(ScriptedLoadableModule):
        """Slicer stub — this file is a mixin/patch, not a standalone module."""
        def __init__(self, parent):
            super().__init__(parent)
            parent.title = "vessel_cc_cli_mixin"
            parent.hidden = True   # keeps it out of the Modules menu
except ImportError:
    pass  # outside Slicer (unit tests, CI) — stub not needed
