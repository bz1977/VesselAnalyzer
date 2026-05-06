# VesselAnalyzer — see CHANGELOG.md for full version history.
# Build tag: v296-debug (2026-04-26)
import os
import math
import vtk
import slicer
import ctk
import qt
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# Graph and geometry helpers extracted to sibling modules
import numpy as _np
from collections import defaultdict as _defaultdict

# Graph and analysis helpers extracted to sibling modules
from centerline_graph import (
    graph_bfs_path          as _graph_bfs_path,
    graph_are_duplicate_edge as _graph_are_duplicate_edge,
    tree_count_nodes        as _tree_count_nodes,
    tree_log_nodes          as _tree_log_nodes,
)
from vessel_state import VesselState
from vessel_pipeline import VesselPipeline
from vessel_centerline_stage import CenterlineStage
from vessel_ostium_stage import OstiumStage
from vessel_radius_map_stage import RadiusMapStage
from vessel_findings_stage import FindingsStage
from centerline_strategies import VMTKStrategy, NudgedSeedStrategy, StraightLineStrategy
from vessel_debug_log import StructuredLogger
from centerline_analysis import (
    CenterlineGraph,
    build_graph,
    detect_bifurcations,
    score_bifurcation,
    extract_branches,
    find_trunk,
    refine_branch_origins,
    debug_branch_origins,
    detect_compression,
)
from ostium_logreg import OstiumLogReg as _OstiumLogReg


# ══════════════════════════════════════════════════════════════════════════════
# VesselDebug — comprehensive debug helper
# Covers four areas:
#   1. Mesh clipping / endpoint detection (white holes)
#   2. Iliac role assignment (IliacCorrect)
#   3. Ostium detection & confidence scoring
#   4. Centerline extraction / branch topology
#
# Log output  → tagged [VD-*] lines printed to the Python console
# Scene output → colour-coded fiducial markers placed in the 3D view
#
# Usage (call from VesselAnalyzerLogic methods or standalone):
#   VesselDebug.mesh_clipping(mesh, rings, endpoints)
#   VesselDebug.iliac_correct(branches, before, after)
#   VesselDebug.ostium(branches, conf_map)
#   VesselDebug.centerline(branches, bif_nodes)
#   VesselDebug.clear_markers()          # remove all VD markers from scene
# ══════════════════════════════════════════════════════════════════════════════
class VesselDebug:
    """Static debug helper — no instance needed."""

    # ── Marker node names ────────────────────────────────────────────────────
    _NODE_PREFIX = "VD_"
    _MARKER_NAMES = {
        "clip":    "VD_ClipRings",
        "iliac":   "VD_IliacRoles",
        "ostium":  "VD_Ostia",
        "cl":      "VD_Centerline",
        "bif":     "VD_Bifurcations",
    }

    # ── Colours (R,G,B 0–1) ─────────────────────────────────────────────────
    _COL = {
        "ring_open":   (1.00, 0.27, 0.00),   # orange-red  — clipped ring
        "ring_lateral":(1.00, 0.84, 0.00),   # yellow      — suspected lateral ostium
        "ep_trunk":    (0.20, 0.80, 0.20),   # green
        "ep_iliac":    (0.00, 0.60, 1.00),   # blue
        "ep_renal":    (1.00, 0.40, 0.80),   # pink
        "ep_unknown":  (0.80, 0.80, 0.80),   # grey
        "iliac_right": (0.10, 0.60, 1.00),   # blue
        "iliac_left":  (0.00, 0.90, 0.90),   # cyan
        "iliac_flip":  (1.00, 0.20, 0.20),   # red — role was changed
        "ostium_high": (0.10, 0.90, 0.10),   # green   — HIGH confidence
        "ostium_rev":  (1.00, 0.85, 0.00),   # yellow  — REVIEW
        "ostium_rej":  (1.00, 0.10, 0.10),   # red     — REJECT
        "bif_primary": (1.00, 0.55, 0.00),   # orange  — primary bifurcation
        "bif_second":  (0.70, 0.70, 0.70),   # grey    — secondary
        "cl_trunk":    (1.00, 1.00, 0.20),   # yellow  — trunk CL
        "cl_branch":   (0.30, 1.00, 0.60),   # mint    — branch CL
    }

    # ────────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_or_create_fiducial(name, colour):
        """Return an existing VD fiducial node or create a fresh one."""
        try:
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is None:
                node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsFiducialNode", name
                )
            dn = node.GetDisplayNode()
            if dn is None:
                node.CreateDefaultDisplayNodes()
                dn = node.GetDisplayNode()
            if dn:
                dn.SetSelectedColor(*colour)
                dn.SetColor(*colour)
                dn.SetGlyphScale(2.5)
                dn.SetTextScale(2.8)
                dn.SetVisibility(True)
            return node
        except Exception as exc:
            print(f"[VD] _get_or_create_fiducial({name}) error: {exc}")
            return None

    @staticmethod
    def _add_marker(node, xyz, label):
        """Add a single control point to a fiducial node."""
        try:
            if node is None:
                return
            idx = node.AddControlPoint(vtk.vtkVector3d(*xyz))
            node.SetNthControlPointLabel(idx, label)
            node.SetNthControlPointLocked(idx, True)
        except Exception as exc:
            print(f"[VD] _add_marker error: {exc}")

    @staticmethod
    def clear_markers():
        """Remove every VD_* debug node from the scene."""
        try:
            to_remove = []
            n = slicer.mrmlScene.GetNumberOfNodes()
            for i in range(n):
                nd = slicer.mrmlScene.GetNthNode(i)
                if nd and nd.GetName() and nd.GetName().startswith("VD_"):
                    to_remove.append(nd)
            for nd in to_remove:
                slicer.mrmlScene.RemoveNode(nd)
            print(f"[VD] Cleared {len(to_remove)} debug marker node(s) from scene.")
        except Exception as exc:
            print(f"[VD] clear_markers error: {exc}")

    # ────────────────────────────────────────────────────────────────────────
    # 1 ── MESH CLIPPING / ENDPOINT DETECTION
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def mesh_clipping(rings, endpoints, lateral_z_range=None):
        """
        Debug mesh clipping and endpoint detection.

        Parameters
        ----------
        rings : list of dict  {centroid:(x,y,z), radius:float, n_pts:int}
            Boundary rings detected after clipping.
        endpoints : list of (x,y,z)
            Final endpoint candidates.
        lateral_z_range : (z_min, z_max) or None
            Z range of the trunk — rings whose Z falls inside this range
            are flagged as suspected lateral ostia (renal-vein holes).
        """
        print("[VD-CLIP] ══ MESH CLIPPING DEBUG ══════════════════════════════")
        node_open = VesselDebug._get_or_create_fiducial(
            "VD_ClipRings_Open", VesselDebug._COL["ring_open"]
        )
        node_lat = VesselDebug._get_or_create_fiducial(
            "VD_ClipRings_Lateral", VesselDebug._COL["ring_lateral"]
        )

        for i, ring in enumerate(rings):
            c   = ring.get("centroid", (0, 0, 0))
            r   = ring.get("radius", 0.0)
            npt = ring.get("n_pts", 0)
            z   = c[2]

            # Is this ring inside the trunk Z band? → probably a lateral ostium
            is_lateral = (
                lateral_z_range is not None
                and lateral_z_range[0] < z < lateral_z_range[1]
            )
            flag = "⚠ LATERAL-OSTIUM?" if is_lateral else "tube-end"
            print(
                f"[VD-CLIP]   Ring {i}: centroid=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})"
                f"  r={r:.1f}mm  pts={npt}  → {flag}"
            )
            target_node = node_lat if is_lateral else node_open
            VesselDebug._add_marker(target_node, c, f"Ring{i}\n{flag}\nr={r:.1f}mm")

        print(f"[VD-CLIP]   {len(rings)} ring(s) total | "
              f"{sum(1 for r in rings if lateral_z_range and lateral_z_range[0]<r.get('centroid',(0,0,0))[2]<lateral_z_range[1])} suspected lateral")
        print(f"[VD-CLIP]   Endpoint candidates: {len(endpoints)}")
        for j, ep in enumerate(endpoints):
            print(f"[VD-CLIP]     ep{j}: ({ep[0]:.1f},{ep[1]:.1f},{ep[2]:.1f})")
        print("[VD-CLIP] ════════════════════════════════════════════════════")

    # ────────────────────────────────────────────────────────────────────────
    # 2 ── ILIAC ROLE ASSIGNMENT
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def iliac_correct(branch_meta_before, branch_meta_after):
        """
        Log and mark iliac role changes made by IliacCorrect.

        Parameters
        ----------
        branch_meta_before : dict  {bi: {'role':str, 'tip':(x,y,z), ...}}
        branch_meta_after  : dict  {bi: {'role':str, 'tip':(x,y,z), ...}}
        """
        print("[VD-ILIAC] ══ ILIAC ROLE ASSIGNMENT DEBUG ═══════════════════")
        node = VesselDebug._get_or_create_fiducial(
            "VD_IliacRoles", VesselDebug._COL["iliac_right"]
        )
        node_flip = VesselDebug._get_or_create_fiducial(
            "VD_IliacFlips", VesselDebug._COL["iliac_flip"]
        )

        flips = 0
        for bi in sorted(branch_meta_after.keys()):
            meta_a = branch_meta_after[bi]
            meta_b = branch_meta_before.get(bi, {})
            role_b = meta_b.get("role", "?")
            role_a = meta_a.get("role", "?")
            tip    = meta_a.get("tip", (0, 0, 0))
            x_cen  = meta_a.get("x_centroid", float("nan"))
            changed = role_b != role_a

            arrow = "→ FLIPPED ⚠" if changed else "  (unchanged)"
            print(
                f"[VD-ILIAC]   bi={bi}  before={role_b:15s}  after={role_a:15s}"
                f"  tip=({tip[0]:.1f},{tip[1]:.1f},{tip[2]:.1f})"
                f"  x_cen={x_cen:.1f}mm  {arrow}"
            )
            col = VesselDebug._COL["iliac_flip"] if changed else (
                VesselDebug._COL["iliac_right"] if "right" in role_a else
                VesselDebug._COL["iliac_left"]
            )
            label = f"bi{bi}\n{role_b}→{role_a}" if changed else f"bi{bi}\n{role_a}"
            target = node_flip if changed else node
            VesselDebug._add_marker(target, tip, label)
            if changed:
                flips += 1

        print(f"[VD-ILIAC]   Total role changes: {flips}")
        print("[VD-ILIAC] ════════════════════════════════════════════════════")

    # ────────────────────────────────────────────────────────────────────────
    # 3 ── OSTIUM DETECTION & CONFIDENCE SCORING
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def ostium(branch_meta, conf_map=None):
        """
        Log and mark ostium positions and confidence grades.

        Parameters
        ----------
        branch_meta : dict  {bi: {'role':str, 'ostium_pt':(x,y,z),
                                   'ostium_gi':int, 'stable_start_gi':int, ...}}
        conf_map    : dict  {bi: {'grade':str, 'eff':float, 'raw':float,
                                   'flags':list, 'spread':float}} or None
        """
        print("[VD-OSTIUM] ══ OSTIUM DETECTION DEBUG ══════════════════════")
        conf_map = conf_map or {}

        grade_col = {
            "HIGH":   VesselDebug._COL["ostium_high"],
            "REVIEW": VesselDebug._COL["ostium_rev"],
            "REJECT": VesselDebug._COL["ostium_rej"],
            "LOW":    VesselDebug._COL["ostium_rej"],
        }

        nodes_by_grade = {}

        for bi in sorted(branch_meta.keys()):
            meta  = branch_meta[bi]
            role  = meta.get("role", "?")
            opt   = meta.get("ostium_pt", None)
            ogi   = meta.get("ostium_gi", None)
            sgi   = meta.get("stable_start_gi", None)
            conf  = conf_map.get(bi, {})
            grade = conf.get("grade", "?")
            eff   = conf.get("eff",   float("nan"))
            raw   = conf.get("raw",   float("nan"))
            flags = conf.get("flags", [])
            spread= conf.get("spread", float("nan"))

            if opt is None:
                print(f"[VD-OSTIUM]   bi={bi} [{role}] ostium=NONE  grade={grade}")
                continue

            flag_str = ",".join(flags) if flags else "—"
            print(
                f"[VD-OSTIUM]   bi={bi} [{role:12s}]"
                f"  pt=({opt[0]:.1f},{opt[1]:.1f},{opt[2]:.1f})"
                f"  ogi={ogi}  sgi={sgi}"
                f"  eff={eff:.3f}  raw={raw:.3f}  ±{spread:.3f}"
                f"  grade={grade}  flags=[{flag_str}]"
            )

            col = grade_col.get(grade, VesselDebug._COL["ep_unknown"])
            if grade not in nodes_by_grade:
                nodes_by_grade[grade] = VesselDebug._get_or_create_fiducial(
                    f"VD_Ostium_{grade}", col
                )
            label = f"bi{bi}\n{role}\n{grade}\neff={eff:.2f}\n[{flag_str}]"
            VesselDebug._add_marker(nodes_by_grade[grade], opt, label)

        grades = [conf_map[bi].get("grade","?") for bi in conf_map]
        print(f"[VD-OSTIUM]   HIGH={grades.count('HIGH')}  "
              f"REVIEW={grades.count('REVIEW')}  "
              f"REJECT={grades.count('REJECT')}  "
              f"LOW={grades.count('LOW')}")
        print("[VD-OSTIUM] ════════════════════════════════════════════════════")

    # ────────────────────────────────────────────────────────────────────────
    # 4 ── CENTERLINE EXTRACTION / BRANCH TOPOLOGY
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def centerline(branches, bif_nodes=None):
        """
        Log and mark the centerline branch topology.

        Parameters
        ----------
        branches : list of dict
            Each: {'role':str, 'pts':list[(x,y,z)], 'arc':float, 'diam':float}
        bif_nodes : list of dict or None
            Each: {'pt':(x,y,z), 'score':float, 'is_primary':bool, 'degree':int}
        """
        print("[VD-CL] ══ CENTERLINE TOPOLOGY DEBUG ═══════════════════════")
        bif_nodes = bif_nodes or []

        # ── Branch summary ────────────────────────────────────────────────
        node_trunk  = VesselDebug._get_or_create_fiducial(
            "VD_CL_Trunk",  VesselDebug._COL["cl_trunk"]
        )
        node_branch = VesselDebug._get_or_create_fiducial(
            "VD_CL_Branch", VesselDebug._COL["cl_branch"]
        )

        for i, br in enumerate(branches):
            role = br.get("role", "?")
            pts  = br.get("pts",  [])
            arc  = br.get("arc",  0.0)
            diam = br.get("diam", 0.0)
            n    = len(pts)
            start = pts[0]  if pts else (0, 0, 0)
            end   = pts[-1] if pts else (0, 0, 0)

            print(
                f"[VD-CL]   Branch {i} [{role:12s}]"
                f"  pts={n}  arc={arc:.1f}mm  diam={diam:.1f}mm"
                f"  start=({start[0]:.1f},{start[1]:.1f},{start[2]:.1f})"
                f"  end=({end[0]:.1f},{end[1]:.1f},{end[2]:.1f})"
            )
            is_trunk = "trunk" in role.lower()
            tgt = node_trunk if is_trunk else node_branch
            VesselDebug._add_marker(tgt, start,
                f"B{i}[{role}]\nSTART arc={arc:.0f}mm\nØ={diam:.1f}mm")
            VesselDebug._add_marker(tgt, end,
                f"B{i}[{role}]\nEND")

        # ── Bifurcation summary ───────────────────────────────────────────
        node_bif_p = VesselDebug._get_or_create_fiducial(
            "VD_Bif_Primary",   VesselDebug._COL["bif_primary"]
        )
        node_bif_s = VesselDebug._get_or_create_fiducial(
            "VD_Bif_Secondary", VesselDebug._COL["bif_second"]
        )

        for j, bif in enumerate(bif_nodes):
            pt      = bif.get("pt",         (0, 0, 0))
            score   = bif.get("score",       0.0)
            primary = bif.get("is_primary",  False)
            degree  = bif.get("degree",      0)
            angle   = bif.get("max_angle",   float("nan"))

            tag = "★PRIMARY" if primary else "secondary"
            print(
                f"[VD-CL]   Bif {j} [{tag}]"
                f"  pt=({pt[0]:.1f},{pt[1]:.1f},{pt[2]:.1f})"
                f"  degree={degree}  score={score:.0f}  max_angle={angle:.1f}°"
            )
            tgt = node_bif_p if primary else node_bif_s
            VesselDebug._add_marker(tgt, pt,
                f"Bif{j} {tag}\ndeg={degree}\nscore={score:.0f}\nangle={angle:.1f}°")

        print(f"[VD-CL]   {len(branches)} branch(es)  {len(bif_nodes)} bifurcation(s)")
        print("[VD-CL] ════════════════════════════════════════════════════")

    # ────────────────────────────────────────────────────────────────────────
    # Convenience: dump everything at once from branchMeta + log context
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def dump_all(branch_meta, conf_map=None, bif_nodes=None,
                 rings=None, endpoints=None, lateral_z_range=None,
                 iliac_before=None):
        """
        Run all four debug passes in sequence.  Call once after the full
        pipeline completes to get a single comprehensive dump.
        """
        print("\n[VD] ╔══════════════════════════════════════════════════╗")
        print("[VD] ║          VesselDebug.dump_all()                  ║")
        print("[VD] ╚══════════════════════════════════════════════════╝")

        if rings is not None or endpoints is not None:
            VesselDebug.mesh_clipping(
                rings or [], endpoints or [], lateral_z_range
            )

        if iliac_before is not None:
            VesselDebug.iliac_correct(iliac_before, branch_meta)

        VesselDebug.ostium(branch_meta, conf_map)

        branches_list = [
            {
                "role": meta.get("role", "?"),
                "pts":  meta.get("pts",  []),
                "arc":  meta.get("arc",  0.0),
                "diam": meta.get("diam", 0.0),
            }
            for meta in branch_meta.values()
        ]
        VesselDebug.centerline(branches_list, bif_nodes)
        print("[VD] ══ dump_all complete ═══════════════════════════════\n")


class VesselAnalyzer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Vessel Analyzer"
        self.parent.icon = qt.QIcon(
            os.path.join(os.path.dirname(__file__), "VesselAnalyzer_icon.png")
        )
        self.parent.categories = ["Vascular"]
        self.parent.dependencies = []
        self.parent.contributors = ["VesselAnalyzer"]
        self.parent.helpText = """
Unified vessel analysis tool combining centerline extraction and IVUS-like diameter visualization.
Step 1: Segment your vessel and export to a Model.
Step 2: Extract the centerline with one click.
Step 3: Navigate along the centerline and view diameter at each point in 3D.
"""
        self.parent.acknowledgementText = ""


class VesselAnalyzerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._updatingGUI = False
        self._sphereColor = (0.18, 0.80, 0.44)
        self._pendingVesselType = (
            None  # set by onVesselTypeChanged, applied on logic creation
        )

    def setup(self):
        # ── FILE IDENTITY SENTINEL ──────────────────────────────────────────
        import os as _os

        print(f"[VesselAnalyzer] *** LOADED FROM: {_os.path.abspath(__file__)} ***")
        print(f"[VesselAnalyzer] *** BUILD TAG: v296-debug-VesselDebug ***")
        # ───────────────────────────────────────────────────────────────────
        ScriptedLoadableModuleWidget.setup(self)
        self._layoutObserverTag = None
        self._importObserverTag = None
        self._nodeAddedObserverTag = None
        self._nodeRemovedObserverTag = None
        self._sliceNodeObserverTags = []  # per-slice-node visibility lock observers
        self._sliceLockSuppressed = False  # set True only during Before/After compare
        self._setupUI()
        
        # ── Pipeline State Sync ─────────────────────────────────────────────
        if hasattr(self, 'logic') and hasattr(self.logic, 'state'):
            self.logic.state.add_listener(self._onStateChanged)
            
        self._installLayoutObserver()
        self._installSceneObservers()
        self._force3DLayout()

    def _onStateChanged(self, state):
        """Called whenever the VesselPipeline mutates the state object."""
        print(f"[VesselAnalyzerWidget] State updated -> Version {state.version}")
        # Future UI refresh calls go here (e.g., updating branch selectors, lists)
        import slicer
        slicer.app.processEvents()

    def _recoverManualCurves(self):
        """Scan the scene for ManualCenterline_* curve nodes and repopulate
        self._manualCurves so Refine/Clear/Auto-Tune work after a scene load
        or when the module panel is opened with an existing scene.
        Also restores modelSelector if it has lost its selection (has_model=False
        on all branches after reload)."""
        try:
            # ── 1. Recover curve nodes ────────────────────────────────────
            recovered = []
            n = slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLMarkupsCurveNode")
            for i in range(n):
                node = slicer.mrmlScene.GetNthNodeByClass(i, "vtkMRMLMarkupsCurveNode")
                if node is not None and node.GetName().startswith("ManualCenterline_"):
                    recovered.append(node)
            if not recovered:
                return
            recovered.sort(key=lambda nd: nd.GetName())
            self._manualCurves = recovered
            n_pts = sum(c.GetNumberOfControlPoints() for c in self._manualCurves)
            n_branches = len(self._manualCurves)
            can_refine = n_branches >= 1 and n_pts >= n_branches * 2
            self.refineCenterlineButton.setEnabled(can_refine)
            self.autoTuneButton.setEnabled(can_refine)
            if hasattr(self, "clearManualCenterlineButton"):
                self.clearManualCenterlineButton.setEnabled(True)
            print(
                f"[ManualCenterline] Recovered {n_branches} curve(s) "
                f"({n_pts} pts) -> Refine/Auto-Tune {'enabled' if can_refine else 'disabled'}"
            )

            # ── 2. Restore modelSelector if selection was lost ────────────
            # After scene reload qMRMLNodeComboBox often loses its selection.
            # If it is already pointing at a valid model with polydata, leave
            # it alone.  Otherwise find the model node with the most points
            # (the segmentation surface) and auto-select it.
            try:
                current = self.modelSelector.currentNode()
                has_valid = (
                    current is not None
                    and current.GetPolyData() is not None
                    and current.GetPolyData().GetNumberOfPoints() > 0
                )
                if not has_valid:
                    best_node = None
                    best_pts = 0
                    nm = slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLModelNode")
                    for i in range(nm):
                        mn = slicer.mrmlScene.GetNthNodeByClass(i, "vtkMRMLModelNode")
                        if mn is None:
                            continue
                        pd = mn.GetPolyData()
                        if pd is None:
                            continue
                        npts = pd.GetNumberOfPoints()
                        if npts > best_pts:
                            best_pts = npts
                            best_node = mn
                    if best_node is not None:
                        self.modelSelector.setCurrentNode(best_node)
                        print(
                            f"[ManualCenterline] modelSelector restored -> "
                            f"'{best_node.GetName()}' ({best_pts} pts)"
                        )
            except Exception as me:
                print(f"[ManualCenterline] modelSelector restore error: {me}")
        except Exception as e:
            print(f"[ManualCenterline] Recovery error: {e}")
        self._updateSeedQualityPanel()

    # ── Anatomical endpoint label mapping ────────────────────────────────────
    #
    # Image reference (POSTERIOR 3D view, matching the reference overlay):
    #
    #   Endpoints-5  →  IVC inlet / top of Trunk          (superior-most, highest Z)
    #   Endpoints-3  →  Right Renal vein ostium            (mid-height, viewer's RIGHT)
    #   Endpoints-4  →  Left Renal vein ostium             (mid-height, viewer's LEFT)
    #   Endpoints-1  →  Right iliac outlet                 (inferior,   viewer's RIGHT)
    #   Endpoints-2  →  Left iliac outlet                  (inferior,   viewer's LEFT)
    #
    # ── CRITICAL — RAS X-axis convention in POSTERIOR view ──────────────────
    #
    # In Slicer's POSTERIOR 3D view the camera looks from behind the patient
    # toward the front.  This MIRRORS the X-axis relative to the patient:
    #
    #   Viewer's RIGHT  = Patient's LEFT  = NEGATIVE X in RAS
    #   Viewer's LEFT   = Patient's RIGHT = POSITIVE X in RAS
    #
    # The reference image labels match the VIEWER'S perspective, so:
    #
    #   Endpoints-1 (Right iliac, viewer's RIGHT)  → MORE NEGATIVE X (RAS)
    #   Endpoints-2 (Left iliac,  viewer's LEFT)   → MORE POSITIVE X (RAS)
    #   Endpoints-3 (Right Renal, viewer's RIGHT)  → MORE NEGATIVE X (RAS)
    #   Endpoints-4 (Left Renal,  viewer's LEFT)   → MORE POSITIVE X (RAS)
    #
    # Assignment algorithm:
    #   1. Sort all endpoints by Z (superior-to-inferior in RAS = descending Z).
    #   2. The top point → Endpoints-5  (IVC inlet / trunk top).
    #   3. The two bottom points → Endpoints-1/2  (iliacs).
    #      - More NEGATIVE X → Endpoints-1  (Right iliac, viewer's right)
    #      - More POSITIVE X → Endpoints-2  (Left iliac,  viewer's left)
    #   4. Remaining mid-height points → Endpoints-3/4  (renals).
    #      X is compared relative to the trunk inlet (Endpoints-5) X to handle
    #      cases where the IVC is not centred at X=0.
    #      - X < trunk_X  → Endpoints-3  (Right Renal vein, viewer's right)
    #      - X ≥ trunk_X  → Endpoints-4  (Left Renal vein,  viewer's left)
    #   5. If only 4 endpoints exist (no renal on one side), the single
    #      mid-height point is assigned based on its X relative to trunk_X.
    #   6. Any count other than 4–5 falls back to generic Endpoints-N naming.
    #
    # The method renames control points on an existing
    # vtkMRMLMarkupsFiducialNode in-place and returns a dict mapping
    # point index → anatomical label string.

    # Canonical label set (1-indexed, matching the image overlay)
    _ENDPOINT_LABELS = {
        1: "Right Iliac",
        2: "Left Iliac",
        3: "Right Renal Vein",
        4: "Left Renal Vein",
        5: "IVC inlet (Trunk)",
    }

    def _assignAnatomicEndpointLabels(self, fiducialNode):
        """Rename control points on *fiducialNode* according to the anatomical
        layout shown in the reference image (Endpoints-1 … Endpoints-5).

        POSTERIOR view convention (camera behind patient, looking forward):
          viewer's RIGHT = patient's LEFT = NEGATIVE X in RAS
          viewer's LEFT  = patient's RIGHT = POSITIVE X in RAS

        The image labels (Right/Left iliac, Right/Left Renal) refer to the
        viewer's perspective, so Right → more negative X, Left → more positive X.

        Left/right discrimination for renal veins uses the trunk inlet X position
        as the midline reference so the assignment is robust when the IVC is not
        centred at X=0.

        Parameters
        ----------
        fiducialNode : vtkMRMLMarkupsFiducialNode
            The node whose control points will be relabelled in-place.

        Returns
        -------
        dict
            Mapping of {point_index: anatomical_label} for all assigned points,
            or an empty dict if the count was outside the supported range.
        """
        try:
            n = fiducialNode.GetNumberOfControlPoints()

            # ── DEBUG: dump raw coordinates before any assignment ────────
            print(f"[EndpointLabel] ── RAW ENDPOINT COORDINATES ──────────────────")
            print(f"[EndpointLabel] Total control points: {n}")
            raw_ras = [0.0, 0.0, 0.0]
            for i in range(n):
                fiducialNode.GetNthControlPointPositionWorld(i, raw_ras)
                print(
                    f"[EndpointLabel]   Point {i}: X={raw_ras[0]:+.2f}  Y={raw_ras[1]:+.2f}"
                    f"  Z={raw_ras[2]:+.2f}  label='{fiducialNode.GetNthControlPointLabel(i)}'"
                )
            print(f"[EndpointLabel] ────────────────────────────────────────────────")

            if n not in (4, 5):
                print(
                    f"[EndpointLabel] {n} endpoints found — expected 4 or 5. "
                    "Skipping anatomical labelling; using generic Endpoints-N names."
                )
                # Still apply generic names so display is consistent
                for i in range(n):
                    fiducialNode.SetNthControlPointLabel(i, f"Endpoints-{i + 1}")
                return {}

            # ── 1. Collect RAS coordinates ───────────────────────────────
            pts = []
            ras = [0.0, 0.0, 0.0]
            for i in range(n):
                fiducialNode.GetNthControlPointPositionWorld(i, ras)
                pts.append((i, float(ras[0]), float(ras[1]), float(ras[2])))
            # pts: list of (original_index, X, Y, Z)

            # ── 2. Sort by Z descending (superior first in RAS) ──────────
            by_z = sorted(pts, key=lambda p: p[3], reverse=True)
            print(f"[EndpointLabel] Z-sorted order (high→low):")
            for rank, p in enumerate(by_z):
                print(f"[EndpointLabel]   rank={rank}  pt_idx={p[0]}  X={p[1]:+.2f}  Z={p[3]:+.2f}")

            # ── 3. Assign superior endpoint → Endpoints-5 (IVC inlet) ────
            ep5_entry = by_z[0]
            ep5_idx   = ep5_entry[0]
            trunk_x   = ep5_entry[1]   # IVC inlet X used as anatomical midline
            print(f"[EndpointLabel] IVC inlet (Endpoints-5): pt_idx={ep5_idx}  X={trunk_x:+.2f}  Z={ep5_entry[3]:+.2f}")

            # ── 4. Classify the 4 non-trunk endpoints as iliacs vs renals ────
            #
            # The old approach (lowest-Z = iliacs) breaks when renal vein openings
            # are more caudal than iliac openings (short infra-renal segment, or
            # renal veins entering the IVC below the iliac bifurcation).
            #
            # Robust classifier: use LATERAL OFFSET from the trunk midline.
            #   - Iliac openings are near the midline (small |X - trunk_X|)
            #   - Renal vein openings are far lateral (large |X - trunk_X|)
            #
            # Among 4 non-trunk points, the two with SMALLEST lateral offset = iliacs,
            # the two with LARGEST lateral offset = renals.
            # Within each pair, sort ascending X: more negative X = Right (viewer's right).
            #
            # Fallback to Z-sort if lateral offsets are ambiguous (all within 10mm
            # of each other), which handles degenerate or single-renal cases.
            non_trunk = [p for p in pts if p[0] != ep5_idx]
            # non_trunk: list of (orig_idx, X, Y, Z)

            lat_offsets = [(p, abs(p[1] - trunk_x)) for p in non_trunk]
            lat_offsets.sort(key=lambda x: x[1])   # ascending lateral offset

            # Check if lateral spread is sufficient to discriminate
            offsets_only = [lo for _, lo in lat_offsets]
            _spread = offsets_only[-1] - offsets_only[0] if len(offsets_only) == 4 else 0
            _mid_gap = offsets_only[2] - offsets_only[1] if len(offsets_only) == 4 else 0

            if _spread >= 20.0 and _mid_gap >= 5.0:
                # Clear lateral separation: small-offset pair = iliacs, large-offset = renals
                iliac_pair = sorted([lat_offsets[0][0], lat_offsets[1][0]], key=lambda p: p[1])
                renal_pair = sorted([lat_offsets[2][0], lat_offsets[3][0]], key=lambda p: p[1])
                print(
                    f"[EndpointLabel] Lateral-offset classification "
                    f"(spread={_spread:.1f}mm, mid_gap={_mid_gap:.1f}mm):"
                )
            else:
                # Ambiguous lateral spread — fall back to Z-sort (original logic)
                print(
                    f"[EndpointLabel] Lateral spread insufficient "
                    f"(spread={_spread:.1f}mm, mid_gap={_mid_gap:.1f}mm) "
                    f"— falling back to Z-sort"
                )
                by_z_non = sorted(non_trunk, key=lambda p: p[3])  # ascending Z
                iliac_pair = sorted(by_z_non[:2], key=lambda p: p[1])   # inferior two
                renal_pair = sorted(by_z_non[2:], key=lambda p: p[1])   # mid/superior

            # POSTERIOR view convention: camera looks from behind patient toward front,
            # which mirrors the X-axis relative to the patient.
            # iliac_pair is sorted ascending X, so [0]=most negative X, [1]=most positive X.
            #   Viewer's RIGHT = Patient's LEFT = NEGATIVE X  → Endpoints-1 (Right iliac, viewer's right)
            #   Viewer's LEFT  = Patient's RIGHT = POSITIVE X → Endpoints-2 (Left iliac,  viewer's left)
            ep1_idx = iliac_pair[0][0]   # more negative X → Right iliac (viewer's right, Endpoints-1)
            ep2_idx = iliac_pair[1][0]   # more positive X → Left iliac  (viewer's left,  Endpoints-2)
            print(
                f"[EndpointLabel] Iliac pair:"
                f"  Right iliac ep1=pt{ep1_idx}(X={iliac_pair[0][1]:+.2f}, lat={abs(iliac_pair[0][1]-trunk_x):.1f}mm)"
                f"  Left iliac  ep2=pt{ep2_idx}(X={iliac_pair[1][1]:+.2f}, lat={abs(iliac_pair[1][1]-trunk_x):.1f}mm)"
            )

            # ── 5. Remaining two → Endpoints-3/4  (renals) ───────────────────
            ep3_idx = ep4_idx = None
            if len(renal_pair) == 2:
                ep3_idx = renal_pair[0][0]   # more negative X → Right Renal vein (viewer's right, Endpoints-3)
                ep4_idx = renal_pair[1][0]   # more positive X → Left Renal vein  (viewer's left,  Endpoints-4)
                print(
                    f"[EndpointLabel] Renal pair:"
                    f"  Right renal ep3=pt{ep3_idx}(X={renal_pair[0][1]:+.2f}, lat={abs(renal_pair[0][1]-trunk_x):.1f}mm)"
                    f"  Left renal  ep4=pt{ep4_idx}(X={renal_pair[1][1]:+.2f}, lat={abs(renal_pair[1][1]-trunk_x):.1f}mm)"
                )
            # len == 0: both renal labels remain None (guard prevents KeyError below).

            # ── 6. Build assignment map and rename control points ─────────
            assignment = {
                ep5_idx: (5, self._ENDPOINT_LABELS[5]),
                ep1_idx: (1, self._ENDPOINT_LABELS[1]),
                ep2_idx: (2, self._ENDPOINT_LABELS[2]),
            }
            if ep3_idx is not None:
                assignment[ep3_idx] = (3, self._ENDPOINT_LABELS[3])
            if ep4_idx is not None:
                assignment[ep4_idx] = (4, self._ENDPOINT_LABELS[4])

            print(f"[EndpointLabel] ── FINAL ASSIGNMENT ──────────────────────────")
            result = {}
            for pt_i, (ep_num, label) in assignment.items():
                fiducialNode.SetNthControlPointLabel(
                    pt_i, f"Endpoints-{ep_num}"
                )
                # Store description as control point description for tooltip/report use
                try:
                    fiducialNode.SetNthControlPointDescription(pt_i, label)
                except Exception:
                    pass
                result[pt_i] = label
                # Look up the RAS coords for this point for verification
                fiducialNode.GetNthControlPointPositionWorld(pt_i, raw_ras)
                print(
                    f"[EndpointLabel]   pt_idx={pt_i}  →  Endpoints-{ep_num}  ({label})"
                    f"  RAS=({raw_ras[0]:+.2f}, {raw_ras[1]:+.2f}, {raw_ras[2]:+.2f})"
                )
            print(f"[EndpointLabel] ────────────────────────────────────────────────")

            # ── VesselDebug area 1: record endpoints from final assignment ─
            try:
                logic = getattr(slicer.modules, "VesselAnalyzerWidget", None)
                logic = getattr(logic, "logic", None) if logic else None
                if logic is not None:
                    ep_list = []
                    tmp = [0.0, 0.0, 0.0]
                    for pt_i in range(fiducialNode.GetNumberOfControlPoints()):
                        fiducialNode.GetNthControlPointPositionWorld(pt_i, tmp)
                        ep_list.append(tuple(tmp))
                    logic._vd_endpoints = ep_list
                    print(f"[VD] Recorded {len(ep_list)} endpoints → _vd_endpoints")

                    # Derive trunk Z range from the superior endpoint (IVC inlet)
                    if ep5_idx < len(ep_list):
                        z_top = ep_list[ep5_idx][2]
                        # iliac endpoints give the bottom bound
                        z_bots = [ep_list[ep1_idx][2], ep_list[ep2_idx][2]]
                        z_bot  = min(z_bots)
                        # Lateral "renal vein" zone = 20–80% of trunk Z span
                        span = z_top - z_bot
                        logic._vd_lateral_z_range = (
                            z_bot  + span * 0.20,
                            z_top  - span * 0.20,
                        )
                        print(f"[VD] Lateral Z range → ({logic._vd_lateral_z_range[0]:.1f}"
                              f", {logic._vd_lateral_z_range[1]:.1f})")
            except Exception as _vd_exc:
                print(f"[VD] Endpoint record error: {_vd_exc}")

            return result

        except Exception as exc:
            import traceback
            print(f"[EndpointLabel] Error during anatomical label assignment: {exc}")
            print(traceback.format_exc())
            return {}

    def enter(self):
        """Called when module tab is selected — force single 3D view and install all observers."""
        self._force3DLayout()
        self._installLayoutObserver()
        self._installSceneObservers()
        self._recoverManualCurves()
        # Hide the DataProbe panel
        try:
            dp = slicer.util.findChild(
                slicer.util.mainWindow(), "DataProbeCollapsibleWidget"
            )
            if dp:
                dp.setVisible(False)
        except Exception:
            pass

    def exit(self):
        """Called when module tab is deselected — remove all observers."""
        self._removeLayoutObserver()
        self._removeSceneObservers()
        self._removeSliceNodeObservers()

    # ------------------------------------------------------------------
    # Layout enforcement
    # ------------------------------------------------------------------

    def _lockSlicePlanes(self, caller=None, event=None):
        """Observer callback — fires whenever a slice node is modified.
        Re-entrancy guarded so our own Set* calls don't retrigger it."""
        if getattr(self, "_sliceLockSuppressed", False):
            return
        if getattr(self, "_sliceLockActive", False):
            return
        self._sliceLockActive = True
        try:
            if caller:
                try:
                    if caller.GetSliceVisible():
                        caller.SetSliceVisible(False)
                except Exception:
                    pass
                try:
                    if caller.GetSlabReconstructionThickness() != 0:
                        caller.SetSlabReconstructionThickness(0)
                except Exception:
                    pass
                # Use the display node for intersection visibility (non-deprecated API)
                try:
                    dn = (
                        caller.GetDisplayNode()
                        if hasattr(caller, "GetDisplayNode")
                        else None
                    )
                    if dn and hasattr(dn, "SetIntersectingSlicesVisibility"):
                        dn.SetIntersectingSlicesVisibility(False)
                except Exception:
                    pass
            # Hide "Volume Slice" model nodes — the solid CT slab rectangles in 3D
            try:
                for node in slicer.util.getNodes().values():
                    if "Volume Slice" in node.GetName() and node.GetDisplayVisibility():
                        node.SetDisplayVisibility(False)
            except Exception:
                pass
        finally:
            self._sliceLockActive = False

    def _lockVRCropping(self, caller=None, event=None):
        """Observer callback on VR display nodes — keeps cropping disabled."""
        if getattr(self, "_sliceLockSuppressed", False):
            return
        try:
            if caller and caller.GetCroppingEnabled():
                caller.SetCroppingEnabled(False)
        except Exception:
            pass

    def _installSliceNodeObservers(self):
        """Attach ModifiedEvent observers to every slice node and VR display node
        so any attempt to show slice planes or enable VR cropping is reversed."""
        self._removeSliceNodeObservers()
        tags = []
        # Slice nodes — lock slice visibility and intersection lines
        try:
            for name in ["Red", "Green", "Yellow"]:
                sliceNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode" + name)
                if sliceNode:
                    tag = sliceNode.AddObserver(
                        vtk.vtkCommand.ModifiedEvent, self._lockSlicePlanes
                    )
                    tags.append((sliceNode, tag))
        except Exception:
            pass
        # Volume rendering display nodes — lock cropping off
        try:
            for vrNode in slicer.util.getNodesByClass(
                "vtkMRMLVolumeRenderingDisplayNode"
            ):
                tag = vrNode.AddObserver(
                    vtk.vtkCommand.ModifiedEvent, self._lockVRCropping
                )
                tags.append((vrNode, tag))
        except Exception:
            pass
        self._sliceNodeObserverTags = tags

    def _removeSliceNodeObservers(self):
        """Remove per-slice-node observers."""
        try:
            for sliceNode, tag in getattr(self, "_sliceNodeObserverTags", []):
                try:
                    sliceNode.RemoveObserver(tag)
                except Exception:
                    pass
        except Exception:
            pass
        self._sliceNodeObserverTags = []

    def _hideAllSlicePlanes(self):
        """Hide slice planes, slab reconstruction, intersection lines, VR cropping
        and ROI nodes. Uses exact IDs and APIs confirmed to work in the Slicer
        Python console."""
        for name in ["Red", "Green", "Yellow"]:
            # --- SliceNode: plane visibility, slab thickness, intersections ---
            try:
                sliceNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode" + name)
                if sliceNode:
                    sliceNode.SetSliceVisible(False)
                    # Slab reconstruction thickness > 0 renders CT panels as
                    # thick volumes in 3D even when SetSliceVisible is False
                    try:
                        sliceNode.SetSlabReconstructionThickness(0)
                    except Exception:
                        pass
            except Exception:
                pass
            # --- SliceDisplayNode: intersection visibility (new non-deprecated API) ---
            try:
                sliceNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNode" + name)
                if sliceNode:
                    dn = (
                        sliceNode.GetDisplayNode()
                        if hasattr(sliceNode, "GetDisplayNode")
                        else None
                    )
                    if dn and hasattr(dn, "SetIntersectingSlicesVisibility"):
                        dn.SetIntersectingSlicesVisibility(False)
            except Exception:
                pass
            # --- SliceCompositeNode: unlink only (intersection API removed) ---
            try:
                compNode = slicer.mrmlScene.GetNodeByID(
                    "vtkMRMLSliceCompositeNode" + name
                )
                if compNode:
                    try:
                        compNode.SetLinkedControl(False)
                    except Exception:
                        pass
            except Exception:
                pass
        # Via applicationLogic — catches any additional slice views
        try:
            appLogic = slicer.app.applicationLogic()
            for name in ["Red", "Green", "Yellow"]:
                try:
                    sl = appLogic.GetSliceLogic(name)
                    if sl:
                        sn = sl.GetSliceNode()
                        sn.SetSliceVisible(False)
                        try:
                            sn.SetSlabReconstructionThickness(0)
                        except Exception:
                            pass
                        # Use display node for intersection visibility
                        try:
                            dn = (
                                sn.GetDisplayNode()
                                if hasattr(sn, "GetDisplayNode")
                                else None
                            )
                            if dn and hasattr(dn, "SetIntersectingSlicesVisibility"):
                                dn.SetIntersectingSlicesVisibility(False)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        # Disable volume rendering cropping and visibility
        try:
            for vrNode in slicer.util.getNodesByClass(
                "vtkMRMLVolumeRenderingDisplayNode"
            ):
                try:
                    vrNode.SetCroppingEnabled(False)
                except Exception:
                    pass
                try:
                    vrNode.SetVisibility(False)
                except Exception:
                    pass
        except Exception:
            pass
        # Hide ROI nodes (cropping box frames visible in 3D)
        for cls in ["vtkMRMLMarkupsROINode", "vtkMRMLAnnotationROINode"]:
            try:
                for roi in slicer.util.getNodesByClass(cls):
                    roi.SetDisplayVisibility(False)
            except Exception:
                pass
        # Hide "Volume Slice" model nodes — the solid CT slab rectangles in 3D
        try:
            for node in slicer.util.getNodes().values():
                if "Volume Slice" in node.GetName():
                    node.SetDisplayVisibility(False)
        except Exception:
            pass

    def _force3DLayout(self):
        """Force single 3D-only view and suppress ALL slice plane visibility in 3D."""
        try:
            lm = slicer.app.layoutManager()
            lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
            # Lock via layout node
            layoutNode = slicer.mrmlScene.GetNodeByID("vtkMRMLLayoutNodeSingleton")
            if layoutNode:
                layoutNode.SetViewArrangement(
                    slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView
                )
            # Maximize 3D widget
            try:
                w = lm.threeDWidget(0)
                if w:
                    w.setMaximized(True)
            except Exception:
                pass
            # Hide slice panel widgets (2D panels alongside 3D view)
            try:
                for name in lm.sliceViewNames():
                    sw = lm.sliceWidget(name)
                    if sw:
                        sw.setVisible(False)
            except Exception:
                pass
            # Hide all slice planes using confirmed-working APIs
            self._hideAllSlicePlanes()
            # Disconnect volumes from slice composite nodes so DICOM load
            # does not re-activate slice planes
            try:
                for name in ["Red", "Green", "Yellow"]:
                    try:
                        sl = slicer.app.applicationLogic().GetSliceLogic(name)
                        if sl:
                            cn = sl.GetSliceCompositeNode()
                            if cn:
                                cn.SetBackgroundVolumeID(None)
                                cn.SetForegroundVolumeID(None)
                                cn.SetLabelVolumeID(None)
                    except Exception:
                        pass
            except Exception:
                pass
            # Disable crosshair navigation mode
            try:
                crosshairNode = slicer.mrmlScene.GetFirstNodeByClass(
                    "vtkMRMLCrosshairNode"
                )
                if crosshairNode:
                    crosshairNode.SetCrosshairMode(
                        slicer.vtkMRMLCrosshairNode.NoCrosshair
                    )
            except Exception:
                pass
            # Install per-slice-node observers — persistent lock against
            # anything re-enabling slice planes after this call returns
            self._installSliceNodeObservers()
        except Exception:
            pass

    def _installLayoutObserver(self):
        """Lock layout node: any change back to a non-allowed layout is immediately reversed."""
        try:
            if getattr(self, "_layoutObserverTag", None) is not None:
                return  # already installed
            layoutNode = slicer.mrmlScene.GetNodeByID("vtkMRMLLayoutNodeSingleton")
            if layoutNode:
                self._layoutObserverTag = layoutNode.AddObserver(
                    slicer.vtkMRMLLayoutNode.LayoutModifiedEvent, self._onLayoutChanged
                )
        except Exception:
            pass

    def _removeLayoutObserver(self):
        """Remove layout lock observer."""
        try:
            tag = getattr(self, "_layoutObserverTag", None)
            if tag is not None:
                layoutNode = slicer.mrmlScene.GetNodeByID("vtkMRMLLayoutNodeSingleton")
                if layoutNode:
                    layoutNode.RemoveObserver(tag)
                self._layoutObserverTag = None
        except Exception:
            pass

    def _installSceneObservers(self):
        """Hook into scene events that trigger layout changes:
        - EndImportEvent   : fired after DICOM / scene load completes
        - NodeAddedEvent   : fired when a volume node is added (DICOM series)
        - NodeRemovedEvent : fired when nodes are removed (stent removal etc.)
        """
        try:
            if getattr(self, "_importObserverTag", None) is None:
                self._importObserverTag = slicer.mrmlScene.AddObserver(
                    slicer.mrmlScene.EndImportEvent, self._onSceneImported
                )
        except Exception:
            pass
        try:
            if getattr(self, "_nodeAddedObserverTag", None) is None:
                self._nodeAddedObserverTag = slicer.mrmlScene.AddObserver(
                    slicer.mrmlScene.NodeAddedEvent, self._onNodeAdded
                )
        except Exception:
            pass
        try:
            if getattr(self, "_nodeRemovedObserverTag", None) is None:
                self._nodeRemovedObserverTag = slicer.mrmlScene.AddObserver(
                    slicer.mrmlScene.NodeRemovedEvent, self._onNodeRemoved
                )
        except Exception:
            pass

    def _removeSceneObservers(self):
        """Remove scene observers installed by _installSceneObservers."""
        try:
            tag = getattr(self, "_importObserverTag", None)
            if tag is not None:
                slicer.mrmlScene.RemoveObserver(tag)
                self._importObserverTag = None
        except Exception:
            pass
        try:
            tag = getattr(self, "_nodeAddedObserverTag", None)
            if tag is not None:
                slicer.mrmlScene.RemoveObserver(tag)
                self._nodeAddedObserverTag = None
        except Exception:
            pass
        try:
            tag = getattr(self, "_nodeRemovedObserverTag", None)
            if tag is not None:
                slicer.mrmlScene.RemoveObserver(tag)
                self._nodeRemovedObserverTag = None
        except Exception:
            pass

    def _onSceneImported(self, caller=None, event=None):
        """Fired after any DICOM / scene import — restore 3D view and
        re-populate _manualCurves from any ManualCenterline_* nodes that
        were saved in the scene, so Refine / Clear / Auto-Tune work after
        a save-and-reload cycle."""
        self._force3DLayout()
        self._recoverManualCurves()

    def _onNodeAdded(self, caller=None, event=None):
        """Fired when any node is added to the scene.
        Only react to scalar volume nodes (DICOM); debounce with a shared timer
        so VMTK's many node additions don't flood the event loop."""
        try:
            node = caller.GetNodeByID(caller.GetLastAddedNodeID()) if caller else None
            if node is None or not node.IsA("vtkMRMLScalarVolumeNode"):
                return
            if not hasattr(self, "_nodeAddedTimer"):
                self._nodeAddedTimer = qt.QTimer()
                self._nodeAddedTimer.setSingleShot(True)
                self._nodeAddedTimer.connect("timeout()", self._force3DLayout)
            self._nodeAddedTimer.start(300)
        except Exception:
            pass

    def _onNodeRemoved(self, caller=None, event=None):
        """Fired when any node is removed from the scene (e.g. stent/balloon removal).
        Debounce and schedule a layout correction so Slicer's post-removal
        layout switch back to conventional view is overridden."""
        try:
            if not hasattr(self, "_nodeRemovedTimer"):
                self._nodeRemovedTimer = qt.QTimer()
                self._nodeRemovedTimer.setSingleShot(True)
                self._nodeRemovedTimer.connect("timeout()", self._force3DLayout)
            self._nodeRemovedTimer.start(200)
        except Exception:
            pass

    def _onLayoutChanged(self, caller, event):
        """Re-enforce single 3D view whenever layout changes while module is active.
        Uses a re-entrancy guard + debounce timer to avoid infinite observer loops."""
        if getattr(self, "_layoutChanging", False):
            return
        try:
            lm = slicer.app.layoutManager()
            cur = lm.layout
            target = slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView
            ENDO = 502  # endoluminal side-by-side layout
            two3d = getattr(self, "_TWO3D_LAYOUT_ID", None)
            allowed = {target, ENDO}
            if two3d is not None:
                allowed.add(two3d)
            if cur not in allowed:
                # Debounce: use a single timer so rapid repeated events
                # don't stack — just schedule one deferred correction
                if not hasattr(self, "_layoutCorrectTimer"):
                    self._layoutCorrectTimer = qt.QTimer()
                    self._layoutCorrectTimer.setSingleShot(True)
                    self._layoutCorrectTimer.connect(
                        "timeout()", self._applyLayoutCorrection
                    )
                self._layoutCorrectTimer.start(50)
        except Exception:
            pass

    def _applyLayoutCorrection(self):
        """Deferred layout correction — called 50ms after an unwanted layout change."""
        if getattr(self, "_layoutChanging", False):
            return
        self._layoutChanging = True
        try:
            self._force3DLayout()
        finally:
            self._layoutChanging = False

    def _setupUI(self):
        from vessel_analyzer_ui import build_ui
        build_ui(self)

    # ── Centerline extraction & refinement ──────────────────────────────
    # Methods: onModelSelected, onAutoDetect, onPlaceEndpoints, onCancelPlace,
    #          onInteractionModeChanged, _cleanupObservers, onEndpointAdded,
    #          onClearEndpoints, onExtractCenterline, onDrawCenterlineManually,
    #          _update_manual_cl_ui, onStopDrawCenterline, onUseManualCenterline,
    #          onClearManualCenterline, _computeSeedQuality, _updateSeedQualityPanel,
    #          _densifyCurve, onRefineCenterline, onAutoTune, _computeCurveMetrics,
    #          _refineSingleCurve, _refineAutoTune, onCenterlineNodeChanged,
    #          onCenterlineVisToggle
    from vessel_centerline_widget_mixin import CenterlineWidgetMixin as _CWM
    onModelSelected            = _CWM.onModelSelected
    onAutoDetect               = _CWM.onAutoDetect
    onPlaceEndpoints           = _CWM.onPlaceEndpoints
    onCancelPlace              = _CWM.onCancelPlace
    onInteractionModeChanged   = _CWM.onInteractionModeChanged
    _cleanupObservers          = _CWM._cleanupObservers
    onEndpointAdded            = _CWM.onEndpointAdded
    onClearEndpoints           = _CWM.onClearEndpoints
    onExtractCenterline        = _CWM.onExtractCenterline
    onDrawCenterlineManually   = _CWM.onDrawCenterlineManually
    _update_manual_cl_ui       = _CWM._update_manual_cl_ui

    def onExtractCenterline(self):
        """Override: wrap the mixin extraction with a progress-bar interceptor.

        Intercepts every print() call and matches [PerBranchCL] stage markers
        to drive the extractProgressBar in the UI automatically.
        No changes to vessel_centerline_mixin.py or vessel_centerline_widget_mixin.py needed.
        """
        from vessel_centerline_widget_mixin import CenterlineWidgetMixin as _CWM_local
        from vessel_analyzer_ui import CenterlineProgressInterceptor
        with CenterlineProgressInterceptor():
            _CWM_local.onExtractCenterline(self)

    onStopDrawCenterline       = _CWM.onStopDrawCenterline
    onUseManualCenterline      = _CWM.onUseManualCenterline
    onClearManualCenterline    = _CWM.onClearManualCenterline
    _computeSeedQuality        = _CWM._computeSeedQuality
    _updateSeedQualityPanel    = _CWM._updateSeedQualityPanel
    _densifyCurve              = _CWM._densifyCurve
    onRefineCenterline         = _CWM.onRefineCenterline
    onAutoTune                 = _CWM.onAutoTune
    _computeCurveMetrics       = _CWM._computeCurveMetrics
    _refineSingleCurve         = _CWM._refineSingleCurve
    _refineAutoTune            = _CWM._refineAutoTune
    onCenterlineNodeChanged    = _CWM.onCenterlineNodeChanged
    onCenterlineVisToggle      = _CWM.onCenterlineVisToggle

    def onAutoDetect(self):
        """Override wrapper: run the mixin's auto-detect, then apply anatomical
        endpoint labels so the 3D view matches the reference image overlay."""
        if self.logic is None:
            slicer.util.errorDisplay(
                "VesselAnalyzer: logic is not initialised.\n"
                "The module may not have loaded correctly — check the Python "
                "console for setup errors and reload the module."
            )
            return
        from vessel_centerline_widget_mixin import CenterlineWidgetMixin as _CWM_local
        _CWM_local.onAutoDetect(self)
        self._labelEndpointsAfterAutoDetect()

    def _labelEndpointsAfterAutoDetect(self):
        """Find the active endpoints fiducial node and apply anatomical labels."""
        try:
            # The endpoint node is stored on the widget as self._endpointsNode
            # (set by onPlaceEndpoints / onAutoDetect in the mixin).
            ep_node = getattr(self, "_endpointsNode", None)
            if ep_node is None:
                # Fall back: search the scene for the Endpoints markup node
                n = slicer.mrmlScene.GetNumberOfNodesByClass(
                    "vtkMRMLMarkupsFiducialNode"
                )
                for i in range(n):
                    node = slicer.mrmlScene.GetNthNodeByClass(
                        i, "vtkMRMLMarkupsFiducialNode"
                    )
                    if node and node.GetName().startswith("Endpoints"):
                        ep_node = node
                        break
            if ep_node is None:
                print("[EndpointLabel] No Endpoints node found after auto-detect.")
                return
            labels = self._assignAnatomicEndpointLabels(ep_node)
            if labels:
                # Update the seed-quality panel to reflect new labels
                self._updateSeedQualityPanel()
        except Exception as exc:
            print(f"[EndpointLabel] _labelEndpointsAfterAutoDetect error: {exc}")

    # ── IVUS navigation & measurements ──────────────────────────────────
    # Methods: onLoadIVUS (overridden below), onBranchChanged, updateBranchStatsBox,
    #          onSliderChanged, _getSphereColor, onEndoluminalToggle,
    #          onPrevPoint, onNextPoint, onGoToMin, onGoToMax, onPickColor,
    #          updateMeasurementsAtIndex, updateStats, onExport
    from vessel_navigation_widget_mixin import NavigationWidgetMixin as _NWM
    # onLoadIVUS is overridden below to apply iliac L/R correction after the mixin runs
    onBranchChanged            = _NWM.onBranchChanged
    updateBranchStatsBox       = _NWM.updateBranchStatsBox
    onSliderChanged            = _NWM.onSliderChanged
    _getSphereColor            = _NWM._getSphereColor
    onEndoluminalToggle        = _NWM.onEndoluminalToggle
    onPrevPoint                = _NWM.onPrevPoint
    onNextPoint                = _NWM.onNextPoint
    onGoToMin                  = _NWM.onGoToMin
    onGoToMax                  = _NWM.onGoToMax
    onPickColor                = _NWM.onPickColor
    updateMeasurementsAtIndex  = _NWM.updateMeasurementsAtIndex
    updateStats                = _NWM.updateStats
    onExport                   = _NWM.onExport

    def onLoadIVUS(self):
        """Override: run the mixin's full analysis pipeline, then fix iliac
        Left/Right labels that the centerline_analysis module assigns using the
        wrong X-sign convention for a posterior 3D view.

        centerline_analysis.find_trunk uses:
            positive X → Right iliac   (anterior-view convention)
        Correct for posterior view (camera behind patient):
            NEGATIVE X → viewer's RIGHT → anatomical RIGHT iliac
            POSITIVE X → viewer's LEFT  → anatomical LEFT  iliac
        """
        from vessel_navigation_widget_mixin import NavigationWidgetMixin as _NWM_local
        _NWM_local.onLoadIVUS(self)
        self._correctIliacLabels()

    def _correctIliacLabels(self):
        """Post-analysis hook: use labeled iliac endpoints as ground-truth anchors
        to identify the true iliac branches, regardless of graph topology errors.

        Strategy
        --------
        1. Read "Right iliac" / "Left iliac" RAS coordinates from the Endpoints
           fiducial node (set by _assignAnatomicEndpointLabels).
        2. For every non-trunk branch, compute the Euclidean distance from the
           branch TIP to each iliac endpoint.
        3. The branch whose tip is closest to each endpoint (within 15 mm) is
           that iliac — regardless of its current role (main / side / etc.).
        4. Previously-labeled iliac branches that don't match either endpoint
           are demoted to 'side'.
        5. Falls back to an X-centroid heuristic if endpoint labels are absent.

        This handles the case where VMTK routes the iliac centerline to a
        renal-junction node instead of the primary bifurcation, causing the
        graph walk to label the wrong short stub as "iliac_left/right" while
        the real long iliac vein ends up as role='side'.
        """
        try:
            logic = getattr(self, "logic", None)
            if logic is None:
                print("[IliacCorrect] No logic object — skipping.")
                return

            # ── Fast-path: trust IliacLabel assignments ───────────────────
            # IliacLabel (in vessel_centerline_mixin) uses endpoint-description
            # matching — the most reliable source.  If it already resolved both
            # iliac branches, verify the current branchMeta roles are consistent
            # and skip the rest of this function.
            _il_right_bi = getattr(logic, "_iliacRightBi", None)
            _il_left_bi  = getattr(logic, "_iliacLeftBi",  None)
            if _il_right_bi is not None and _il_left_bi is not None:
                _bm = getattr(logic, "branchMeta", {})
                _r_role = _bm.get(_il_right_bi, {}).get("role", "")
                _l_role = _bm.get(_il_left_bi,  {}).get("role", "")
                _r_ok = "right" in _r_role.lower()
                _l_ok = "left"  in _l_role.lower()
                if _r_ok and _l_ok:
                    print(f"[IliacCorrect] IliacLabel assignments confirmed "
                          f"(right_bi={_il_right_bi} role={_r_role!r}, "
                          f"left_bi={_il_left_bi} role={_l_role!r}) — skipping.")
                    return
                else:
                    # IliacLabel stores its bi assignments in _iliacRightBi/_iliacLeftBi
                    # but it probes the ostium point to match endpoints. Both main iliac
                    # branches share the SAME ostium (the primary bifurcation), so
                    # IliacLabel's L/R resolution can be wrong. Do NOT blindly force
                    # its assignments onto branchMeta when they conflict with the current
                    # roles (which may have already been corrected by ClassifySurface).
                    # Fall through to the endpoint-anchored distance logic below, which
                    # uses true 3-D branch-tip distances to the labelled iliac endpoints
                    # and is the most reliable disambiguator.
                    print(
                        f"[IliacCorrect] IliacLabel stored bi ("
                        f"right_bi={_il_right_bi}, left_bi={_il_left_bi}) "
                        f"conflicts with current roles "
                        f"({_r_role!r}/{_l_role!r}). "
                        f"IliacLabel may be wrong (shared-ostium probe). "
                        f"Skipping fast-path force — deferring to endpoint-anchored logic."
                    )
                    # Clear stale IliacLabel bi attributes so the endpoint-anchored
                    # path below does a full fresh assignment.
                    try:
                        logic._iliacRightBi = None
                        logic._iliacLeftBi  = None
                    except Exception:
                        pass
                    # fall through — do NOT return

            branchMeta_raw = getattr(logic, "branchMeta", None)
            if not branchMeta_raw:
                print("[IliacCorrect] branchMeta empty — skipping.")
                return

            # ── Normalise branchMeta to a list of (bi, meta) tuples ──────
            print(f"[IliacCorrect] branchMeta type={type(branchMeta_raw).__name__}  "
                  f"len={len(branchMeta_raw)}  "
                  f"first_key_type={type(next(iter(branchMeta_raw))).__name__}")

            if isinstance(branchMeta_raw, dict):
                meta_items = list(branchMeta_raw.items())
            else:
                first = branchMeta_raw[0] if branchMeta_raw else None
                if isinstance(first, dict):
                    meta_items = list(enumerate(branchMeta_raw))
                else:
                    print(f"[IliacCorrect] Unexpected branchMeta structure "
                          f"(item type {type(first).__name__}) — skipping.")
                    return

            branches = getattr(logic, "_rawBranches", None) or getattr(logic, "branches", [])
            points   = getattr(logic, "points", [])

            # ── Step 1: Read iliac endpoint RAS coords from scene ─────────
            right_ep_ras = None   # anatomical RIGHT iliac endpoint (x,y,z)
            left_ep_ras  = None   # anatomical LEFT  iliac endpoint (x,y,z)
            try:
                ep_node = getattr(self, "_endpointsNode", None)
                if ep_node is None:
                    n = slicer.mrmlScene.GetNumberOfNodesByClass(
                        "vtkMRMLMarkupsFiducialNode")
                    for i in range(n):
                        nd = slicer.mrmlScene.GetNthNodeByClass(
                            i, "vtkMRMLMarkupsFiducialNode")
                        if nd and nd.GetName().startswith("Endpoints"):
                            ep_node = nd
                            break
                if ep_node is not None:
                    p = [0.0, 0.0, 0.0]
                    for ci in range(ep_node.GetNumberOfControlPoints()):
                        desc = ep_node.GetNthControlPointDescription(ci).lower()
                        ep_node.GetNthControlPointPositionWorld(ci, p)
                        if "right iliac" in desc:
                            right_ep_ras = (float(p[0]), float(p[1]), float(p[2]))
                        elif "left iliac" in desc:
                            left_ep_ras  = (float(p[0]), float(p[1]), float(p[2]))
            except Exception as ee:
                print(f"[IliacCorrect] Endpoint lookup error: {ee}")

            print(f"[IliacCorrect] Iliac endpoints from scene:")
            print(f"[IliacCorrect]   right_ep={right_ep_ras}")
            print(f"[IliacCorrect]   left_ep ={left_ep_ras}")

            # Branches that terminate at endpoint fiducials labelled renal are
            # not iliac candidates, even if their tips happen to be close in X
            # or their role was temporarily demoted to side by a later pass.
            renal_endpoint_bis = set()
            try:
                ep_map = getattr(logic, "branchEndpointMap", {})
                if ep_node is not None and ep_map:
                    for bi_ep, ep_idx in ep_map.items():
                        if ep_idx >= ep_node.GetNumberOfControlPoints():
                            continue
                        ep_label = ep_node.GetNthControlPointLabel(ep_idx)
                        ep_desc = ep_node.GetNthControlPointDescription(ep_idx)
                        if "renal" in f"{ep_label} {ep_desc}".lower():
                            renal_endpoint_bis.add(bi_ep)
                if renal_endpoint_bis:
                    print(
                        f"[IliacCorrect] Renal endpoint anchors excluded from "
                        f"iliac candidates: {sorted(renal_endpoint_bis)}"
                    )
            except Exception as ree:
                print(f"[IliacCorrect] Renal endpoint exclusion failed: {ree}")

            # ── Step 2: Compute tip distances for all non-trunk branches ──
            # IMPORTANT: branches are stored in varying directions depending on
            # role.  Side branches are stored tip→bif (branches[bi][0] = iliac
            # tip, branches[bi][1] = bif end), while main branches are stored
            # bif→distal.  We check BOTH endpoints and take the minimum so the
            # correct iliac-tip end is always found regardless of direction.
            def _pt_dist(p, ep_ras):
                dx = float(p[0]) - ep_ras[0]
                dy = float(p[1]) - ep_ras[1]
                dz = float(p[2]) - ep_ras[2]
                return float(_np.sqrt(dx*dx + dy*dy + dz*dz))

            def _tip_dist(bi, ep_ras):
                """Min distance from either branch endpoint to ep_ras.

                branches[bi] = (startIdx, endIdx) where endIdx is EXCLUSIVE
                (i.e. one past the last valid point, matching Python slice convention).
                The last valid point index is therefore endIdx - 1.
                """
                try:
                    if not branches or bi >= len(branches):
                        return 9999.0
                    bs, be = branches[bi][0], branches[bi][1] - 1  # be is now inclusive last
                    d_s = _pt_dist(points[bs], ep_ras) if bs < len(points) else 9999.0
                    d_e = _pt_dist(points[be], ep_ras) if be < len(points) else 9999.0
                    return min(d_s, d_e)
                except Exception:
                    return 9999.0

            def _tip_ras(bi):
                """Return the branch's more-inferior (lower-Z) endpoint for logs.

                branches[bi] = (startIdx, endIdx) where endIdx is EXCLUSIVE.
                Last valid index = endIdx - 1.
                """
                try:
                    if not branches or bi >= len(branches):
                        return (0, 0, 0)
                    bs, be = branches[bi][0], branches[bi][1] - 1  # be is now inclusive last
                    ps = points[bs] if bs < len(points) else None
                    pe = points[be] if be < len(points) else None
                    if ps is None and pe is None:
                        return (0, 0, 0)
                    if ps is None:
                        tp = pe
                    elif pe is None:
                        tp = ps
                    else:
                        tp = ps if float(ps[2]) < float(pe[2]) else pe
                    return (float(tp[0]), float(tp[1]), float(tp[2]))
                except Exception:
                    return (0, 0, 0)

            # Only attempt endpoint-matching if we found at least one endpoint
            endpoints_available = (right_ep_ras is not None or left_ep_ras is not None)

            if endpoints_available:
                print("[IliacCorrect] ── TIP DISTANCES TO ILIAC ENDPOINTS ─────────────")
                right_cands = []   # (dist, bi, meta)
                left_cands  = []
                for bi, meta in meta_items:
                    if not isinstance(meta, dict):
                        continue
                    if meta.get("role") == "trunk":
                        continue
                    tip = _tip_ras(bi)
                    d_r = _tip_dist(bi, right_ep_ras) if right_ep_ras else 9999.0
                    d_l = _tip_dist(bi, left_ep_ras)  if left_ep_ras  else 9999.0
                    role = meta.get("role", "?")
                    print(f"[IliacCorrect]   bi={bi} role={role!r:16s} "
                          f"tip=({tip[0]:.1f},{tip[1]:.1f},{tip[2]:.1f}) "
                          f"d_right={d_r:.1f}mm  d_left={d_l:.1f}mm")
                    right_cands.append((d_r, bi, meta))
                    left_cands.append((d_l, bi, meta))

                MATCH_MM = 15.0    # within 15 mm → high-confidence snap
                ASSIGN_MM = 999.0  # closest-wins threshold when CL doesn't reach tips
                # IVC anatomy: iliac centerlines often end 100-250mm from the
                # labeled scene endpoints (VMTK stops at the bifurcation region).
                # Use closest-wins among branches that already have an iliac role
                # so the labeled endpoints still drive assignment even when the
                # centerline is far away.

                right_cands.sort(key=lambda t: t[0])
                left_cands.sort(key=lambda t: t[0])

                def _is_endpoint_candidate(bi, meta):
                    if bi in renal_endpoint_bis:
                        return False
                    role = meta.get("role", "")
                    if role == "trunk":
                        return False
                    if role in ("renal_vein", "renal_right", "renal_left", "renal_fragment"):
                        return False
                    return True

                # Hard endpoint snaps are ground-truth: if a non-renal branch
                # tip is within MATCH_MM of a labelled iliac endpoint, accept it
                # regardless of its current role or arc length.  Short true iliac
                # endpoint stubs can otherwise be missed and replaced by a longer
                # but farther branch in the lateral-X fallback.
                right_endpoint_cands = [(d, bi, meta) for d, bi, meta in right_cands
                                        if _is_endpoint_candidate(bi, meta)]
                left_endpoint_cands  = [(d, bi, meta) for d, bi, meta in left_cands
                                        if _is_endpoint_candidate(bi, meta)]

                # Filter candidate lists to iliac-role branches only for
                # closest-wins fallback; non-iliac branches (side/renal) should
                # not be re-labelled iliac by the loose fallback path.
                # Exception: side branches whose length exceeds the RenalTag soft
                # ceiling (120mm) are iliac candidates that RenalTag correctly
                # rejected but that the graph walk never promoted to 'main' because
                # the true bifurcation node was not found.  Include them here so the
                # endpoint-anchored and lateral-X fallback paths can still assign them.
                _ILIAC_MIN_LEN_MM = 120.0
                iliac_roles = ("iliac_right", "iliac_left", "main")
                def _is_iliac_candidate(bi, meta):
                    if bi in renal_endpoint_bis:
                        return False
                    role = meta.get("role", "")
                    if role in ("renal_vein", "renal_right", "renal_left", "renal_fragment"):
                        return False
                    if role in iliac_roles:
                        return True
                    if role == "side":
                        try:
                            bs, be = branches[bi][0], branches[bi][1]
                            segs = [points[gi] for gi in range(bs, be) if gi < len(points)]
                            arc = sum(
                                math.sqrt(sum((segs[k+1][j]-segs[k][j])**2 for j in range(3)))
                                for k in range(len(segs)-1)
                            ) if len(segs) >= 2 else 0.0
                            return arc >= _ILIAC_MIN_LEN_MM
                        except Exception:
                            return False
                    return False
                right_iliac_cands = [(d, bi, meta) for d, bi, meta in right_cands
                                     if _is_iliac_candidate(bi, meta)]
                left_iliac_cands  = [(d, bi, meta) for d, bi, meta in left_cands
                                     if _is_iliac_candidate(bi, meta)]

                def _best_match(cands, threshold):
                    return cands[0][1] if cands and cands[0][0] < threshold else None

                right_bi_new = _best_match(right_endpoint_cands, MATCH_MM)
                left_bi_new  = _best_match(left_endpoint_cands,  MATCH_MM)

                # ── Lateral-X ranking fallback ───────────────────────────────
                # When the hard snap fails (CL doesn't reach scene endpoints),
                # rank the iliac-role branches by their distal CL tip X coordinate.
                # In RAS: right iliac endpoint has the most-negative X, left has
                # the most-positive.  Among the CL branches heading toward these
                # endpoints, the one whose distal CL tip has the most-negative X
                # is heading rightward → assign to right iliac, and vice versa.
                # This is identical in logic to the X-centroid fallback below,
                # but restricted to branches that already carry an iliac/main role
                # and uses the labeled endpoint X coordinates to validate the
                # assignment direction (confirming right_ep.X < 0, left_ep.X > 0).
                if (right_bi_new is None or left_bi_new is None):
                    seen_bis = {}  # bi → distal_tip_x
                    for _, bi_c, meta_c in right_iliac_cands + left_iliac_cands:
                        if bi_c in seen_bis:
                            continue
                        try:
                            bs_c, be_c = branches[bi_c][0], branches[bi_c][1] - 1  # be_c inclusive last
                            n_c = be_c - bs_c + 1
                            tip_s = bs_c + max(0, n_c * 3 // 4)
                            xs_c = [float(points[gi][0])
                                    for gi in range(tip_s, be_c + 1)
                                    if gi < len(points)]
                            cx = sum(xs_c) / len(xs_c) if xs_c else float(points[be_c][0])
                        except Exception:
                            cx = 0.0
                        seen_bis[bi_c] = cx

                    if len(seen_bis) >= 2:
                        sorted_bx = sorted(seen_bis.items(), key=lambda t: t[1])
                        # Validate direction: right_ep should have negative X,
                        # left_ep positive X — if flipped, swap the convention.
                        right_x_ref = float(right_ep_ras[0]) if right_ep_ras else -1.0
                        left_x_ref  = float(left_ep_ras[0])  if left_ep_ras  else +1.0
                        convention_normal = (right_x_ref < left_x_ref)
                        # convention_normal=True → most-neg CL X → right (standard)
                        # convention_normal=False → most-neg CL X → left  (unusual)
                        if convention_normal:
                            r_bi, r_cx = sorted_bx[0]    # most-negative → right
                            l_bi, l_cx = sorted_bx[-1]   # most-positive → left
                        else:
                            r_bi, r_cx = sorted_bx[-1]   # most-positive → right
                            l_bi, l_cx = sorted_bx[0]    # most-negative → left

                        if right_bi_new is None:
                            right_bi_new = r_bi
                            print(f"[IliacCorrect] Lateral-X rank assign: "
                                  f"right_bi={right_bi_new} "
                                  f"(CL_tip_x={r_cx:+.1f}mm, "
                                  f"ep_x={right_x_ref:+.1f}mm, "
                                  f"convention={'normal' if convention_normal else 'flipped'})")
                        if left_bi_new is None and l_bi != right_bi_new:
                            left_bi_new = l_bi
                            print(f"[IliacCorrect] Lateral-X rank assign: "
                                  f"left_bi={left_bi_new} "
                                  f"(CL_tip_x={l_cx:+.1f}mm, "
                                  f"ep_x={left_x_ref:+.1f}mm)")

                # ── Collision resolution ──────────────────────────────────
                # Case A: same bi matched to both endpoints — this happens when
                # VMTK returned a cross-bif path that spans both iliac tips.
                # Strategy: keep the assignment for the endpoint where this bi
                # is uniquely the closest; try to find another branch for the
                # second endpoint.  If none exists within threshold, flag the
                # cross-bif branch as 'iliac_both' so callers know it covers
                # both sides.
                if right_bi_new is not None and right_bi_new == left_bi_new:
                    cross_bif_bi = right_bi_new   # this branch spans both tips
                    # Check whether start or end of cross-bif branch is closer
                    # to right vs left endpoint, to decide which "side" it is
                    try:
                        bs_c, be_c = branches[cross_bif_bi][0], branches[cross_bif_bi][1] - 1  # be_c inclusive last
                        d_start_right = (_pt_dist(points[bs_c], right_ep_ras)
                                         if bs_c < len(points) and right_ep_ras else 9999.0)
                        d_end_right   = (_pt_dist(points[be_c], right_ep_ras)
                                         if be_c < len(points) and right_ep_ras else 9999.0)
                        d_start_left  = (_pt_dist(points[bs_c], left_ep_ras)
                                         if bs_c < len(points) and left_ep_ras  else 9999.0)
                        d_end_left    = (_pt_dist(points[be_c], left_ep_ras)
                                         if be_c < len(points) and left_ep_ras  else 9999.0)
                        # start end closer to left → bi starts at left iliac
                        start_is_left = (d_start_left < d_start_right)
                    except Exception:
                        start_is_left = True

                    print(f"[IliacCorrect] Cross-bif path detected: bi={cross_bif_bi} "
                          f"spans both iliac endpoints "
                          f"(start_is_left={start_is_left})")

                    # The cross-bif branch covers both — try to find exclusive
                    # matches for each side among the remaining branches
                    next_r = next((t for t in right_iliac_cands[1:] if t[0] < MATCH_MM), None)
                    next_l = next((t for t in left_iliac_cands[1:] if t[0] < MATCH_MM), None)

                    if next_r is not None and next_l is None:
                        # Another branch exclusively reaches right
                        right_bi_new = next_r[1]
                        # left stays as cross-bif bi
                        left_bi_new  = cross_bif_bi
                    elif next_l is not None and next_r is None:
                        # Another branch exclusively reaches left
                        left_bi_new  = next_l[1]
                        right_bi_new = cross_bif_bi
                    elif next_r is not None and next_l is not None:
                        right_bi_new = next_r[1]
                        left_bi_new  = next_l[1]
                    else:
                        # No exclusive matches — assign cross-bif branch to one
                        # side based on which end is its start, mark it specially
                        if start_is_left:
                            left_bi_new  = cross_bif_bi
                            right_bi_new = None   # will fall to X-centroid for right
                        else:
                            right_bi_new = cross_bif_bi
                            left_bi_new  = None

                    print(f"[IliacCorrect]   After cross-bif resolution: "
                          f"right_bi={right_bi_new}  left_bi={left_bi_new}")

                print(f"[IliacCorrect] ── ENDPOINT-ANCHORED ASSIGNMENT ────────────────")
                print(f"[IliacCorrect]   right_bi={right_bi_new}  left_bi={left_bi_new}")

                if right_bi_new is not None or left_bi_new is not None:
                    # Apply roles for whichever sides were resolved
                    assigned_bis = {b for b in (right_bi_new, left_bi_new)
                                    if b is not None}
                    for bi, meta in meta_items:
                        if not isinstance(meta, dict):
                            continue
                        old_role  = meta.get("role", "")
                        was_iliac = "iliac" in old_role or old_role == "main"
                        if bi == right_bi_new:
                            meta["role"]  = "iliac_right"
                            meta["label"] = "Right Iliac"
                            meta["_ep_anchored"] = True   # ← lock: fallback must not override
                            for k in ("display_name", "name", "branchName"):
                                if k in meta:
                                    meta[k] = "Right Iliac"
                            print(f"[IliacCorrect]   bi={bi} → iliac_right "
                                  f"(was {old_role!r})")
                        elif bi == left_bi_new:
                            meta["role"]  = "iliac_left"
                            meta["label"] = "Left Iliac"
                            meta["_ep_anchored"] = True   # ← lock: fallback must not override
                            for k in ("display_name", "name", "branchName"):
                                if k in meta:
                                    meta[k] = "Left Iliac"
                            print(f"[IliacCorrect]   bi={bi} → iliac_left "
                                  f"(was {old_role!r})")
                        elif was_iliac and bi not in assigned_bis:
                            # Stale iliac label — no endpoint matched it.
                            # If one side is still unresolved (None), keep the
                            # existing iliac label so X-centroid can use it.
                            if (right_bi_new is None or left_bi_new is None):
                                print(f"[IliacCorrect]   bi={bi} kept {old_role!r} "
                                      f"(candidate for unresolved side)")
                            else:
                                d_right = _tip_dist(bi, right_ep_ras) if right_ep_ras else 9999.0
                                d_left  = _tip_dist(bi, left_ep_ras)  if left_ep_ras  else 9999.0
                                same_side_role = None
                                if d_right < d_left and d_right <= 80.0:
                                    same_side_role = "iliac_right"
                                elif d_left < d_right and d_left <= 80.0:
                                    same_side_role = "iliac_left"

                                if same_side_role is not None:
                                    parent_bi = (
                                        right_bi_new
                                        if same_side_role == "iliac_right"
                                        else left_bi_new
                                    )
                                    meta["role"] = "iliac_fragment"
                                    meta["fragment_of"] = parent_bi
                                    meta["iliac_role"] = same_side_role
                                    meta["label"] = (
                                        "Right Iliac Fragment"
                                        if same_side_role == "iliac_right"
                                        else "Left Iliac Fragment"
                                    )
                                    meta["lateral_label"] = (
                                        "Right"
                                        if same_side_role == "iliac_right"
                                        else "Left"
                                    )
                                    for k in ("display_name", "name", "branchName"):
                                        if k in meta:
                                            meta[k] = meta["label"]
                                    try:
                                        _surf_colors = getattr(logic, "_branch_surface_colors", None)
                                        if isinstance(_surf_colors, dict) and parent_bi in _surf_colors:
                                            _surf_colors[bi] = _surf_colors[parent_bi]
                                    except Exception:
                                        pass
                                    print(f"[IliacCorrect]   bi={bi} marked "
                                          f"iliac_fragment of bi={parent_bi} "
                                          f"(was {old_role!r}, "
                                          f"d_right={d_right:.1f}mm, "
                                          f"d_left={d_left:.1f}mm)")
                                else:
                                    meta["role"]  = "side"
                                    print(f"[IliacCorrect]   bi={bi} demoted "
                                          f"{old_role!r} → 'side' (no endpoint match)")

                    print("[IliacCorrect] ────────────────────────────────────────────")

                    # If one side is still unresolved, supplement with X-centroid
                    if right_bi_new is None or left_bi_new is None:
                        print("[IliacCorrect] One side unresolved — "
                              "supplementing with X-centroid fallback")
                        self._correctIliacLabels_XCentroidFallback(
                            meta_items, branches, points)

                    # ── Sync corrected bi values back to logic so the navigation
                    # mixin resolves the right branch index.  Without this,
                    # logic._iliacRightBi / _iliacLeftBi still hold the stale
                    # mixin-assigned values and the navigator lands on the wrong
                    # branch's points even though branchMeta labels are correct.
                    try:
                        if right_bi_new is not None:
                            logic._iliacRightBi = right_bi_new
                            print(f"[IliacCorrect] Synced logic._iliacRightBi = {right_bi_new}")
                        if left_bi_new is not None:
                            logic._iliacLeftBi = left_bi_new
                            print(f"[IliacCorrect] Synced logic._iliacLeftBi  = {left_bi_new}")
                    except Exception as se:
                        print(f"[IliacCorrect] logic bi-sync skipped: {se}")

                    try:
                        self._refreshBranchSelectorAfterLabelCorrection()
                    except Exception as re:
                        print(f"[IliacCorrect] Branch selector refresh skipped: {re}")
                    return   # done — endpoint path succeeded

                # If we got here both matched None despite endpoints being present:
                print("[IliacCorrect] Endpoint match threshold not met "
                      f"(best right={right_cands[0][0]:.1f}mm "
                      f"best left={left_cands[0][0]:.1f}mm) — "
                      f"falling back to X-centroid.")

            # ── Fallback: X-centroid on already-labeled iliac branches ────
            self._correctIliacLabels_XCentroidFallback(meta_items, branches, points)

        except Exception as exc:
            import traceback
            print(f"[IliacCorrect] ERROR: {exc}")
            print(traceback.format_exc())

    def _correctIliacLabels_XCentroidFallback(self, meta_items, branches, points):
        """Fallback: swap iliac labels using distal-tip X coordinate when endpoint
        RAS coordinates are unavailable.  Uses the distal 25% of each branch
        (tip-side) so the centroid is far from the shared bifurcation point."""
        print("[IliacCorrect] ── X-CENTROID FALLBACK ──────────────────────────────")
        iliac_items = []
        renal_endpoint_bis = set()
        try:
            logic = getattr(self, "logic", None)
            ep_map = getattr(logic, "branchEndpointMap", {}) if logic is not None else {}
            ep_node = getattr(self, "_endpointsNode", None)
            if ep_node is None:
                ep_node = slicer.mrmlScene.GetFirstNodeByName("Endpoints")
            if ep_node is not None and ep_map:
                for bi_ep, ep_idx in ep_map.items():
                    if ep_idx >= ep_node.GetNumberOfControlPoints():
                        continue
                    ep_label = ep_node.GetNthControlPointLabel(ep_idx)
                    ep_desc = ep_node.GetNthControlPointDescription(ep_idx)
                    if "renal" in f"{ep_label} {ep_desc}".lower():
                        renal_endpoint_bis.add(bi_ep)
        except Exception:
            renal_endpoint_bis = set()
        for bi, meta in meta_items:
            if not isinstance(meta, dict):
                continue
            role  = meta.get("role", "?")
            if bi in renal_endpoint_bis or role in (
                "renal_vein", "renal_right", "renal_left", "renal_fragment"
            ):
                print(f"[IliacCorrect]   bi={bi}  role={role!r:16s}  "
                      "X=RENAL-ENDPOINT  → skipped")
                continue
            # ── Skip branches already locked by endpoint-anchored assignment ──
            if meta.get("_ep_anchored", False):
                tip_x_str = "n/a"
                try:
                    if branches and bi < len(branches):
                        be = branches[bi][1] - 1  # exclusive→inclusive
                        if be < len(points):
                            tip_x_str = f"{points[be][0]:+.2f}"
                except Exception:
                    pass
                print(f"[IliacCorrect]   bi={bi}  role={role!r:16s}  "
                      f"X=LOCKED (ep_anchored)  tip_X={tip_x_str}mm  → skipped")
                continue
            x_val = None
            x_src = "n/a"
            try:
                if branches and bi < len(branches):
                    bs, be = branches[bi][0], branches[bi][1] - 1  # be inclusive last
                    n_pts  = be - bs + 1
                    # Take the TIP-SIDE 25% so we're away from the bif
                    tip_start = bs + max(0, n_pts * 3 // 4)
                    xs = [float(points[gi][0])
                          for gi in range(tip_start, be + 1)
                          if gi < len(points)]
                    if xs:
                        x_val = sum(xs) / len(xs)
                        x_src = f"tip25%centroid({len(xs)}pts)"
                    elif be < len(points):
                        x_val = float(points[be][0])
                        x_src = "tip_only"
            except Exception:
                pass
            # ostiumPt last resort
            if x_val is None:
                try:
                    op = meta.get("ostiumPt") or meta.get("ostium_pt")
                    if op is not None:
                        x_val = float(op[0])
                        x_src = "ostiumPt"
                except Exception:
                    pass
            tip_x_str = "n/a"
            try:
                if branches and bi < len(branches):
                    be = branches[bi][1] - 1  # exclusive→inclusive
                    if be < len(points):
                        tip_x_str = f"{points[be][0]:+.2f}"
            except Exception:
                pass
            print(f"[IliacCorrect]   bi={bi}  role={role!r:16s}  "
                  f"X={f'{x_val:+.2f}mm' if x_val is not None else 'n/a':12s}  "
                  f"(src={x_src})  tip_X={tip_x_str}mm")
            # Include long side branches (>= 120mm) as iliac candidates.
            # These are iliacs that RenalTag rejected but the graph walk
            # never promoted to "main" due to a missed bifurcation node.
            _is_long_side = False
            if role == "side" and x_val is not None:
                try:
                    _bs, _be = branches[bi][0], branches[bi][1]
                    _spts = [points[_gi] for _gi in range(_bs, _be) if _gi < len(points)]
                    _arc = sum(
                        math.sqrt(sum((_spts[_k+1][_j]-_spts[_k][_j])**2 for _j in range(3)))
                        for _k in range(len(_spts)-1)
                    ) if len(_spts) >= 2 else 0.0
                    _is_long_side = _arc >= 120.0
                except Exception:
                    pass
            if (role in ("iliac_right", "iliac_left", "main") or _is_long_side) \
                    and x_val is not None:
                iliac_items.append((bi, meta, x_val))

        if len(iliac_items) < 2:
            print(f"[IliacCorrect] Fallback: only {len(iliac_items)} iliac branch(es) "
                  f"— nothing to swap.")
            return

        # Most negative X = anatomical RIGHT (posterior-view convention)
        iliac_items.sort(key=lambda t: t[2])
        right_bi, right_meta, right_x = iliac_items[0]
        left_bi,  left_meta,  left_x  = iliac_items[-1]

        print(f"[IliacCorrect]   Most negative X: bi={right_bi}  "
              f"X={right_x:+.2f}mm → RIGHT iliac")
        print(f"[IliacCorrect]   Most positive X: bi={left_bi}   "
              f"X={left_x:+.2f}mm → LEFT  iliac")

        right_already = "right" in right_meta.get("role","").lower()
        left_already  = "left"  in left_meta.get("role","").lower()
        if right_already and left_already:
            print("[IliacCorrect] ✓ Labels already correct — no swap needed.")
            print("[IliacCorrect] ────────────────────────────────────────────────")
            return

        right_meta["role"]  = "iliac_right"
        right_meta["label"] = "Right Iliac"
        left_meta["role"]   = "iliac_left"
        left_meta["label"]  = "Left Iliac"
        for key in ("display_name", "name", "branchName"):
            if key in right_meta:
                right_meta[key] = "Right Iliac"
            if key in left_meta:
                left_meta[key]  = "Left Iliac"

        print(f"[IliacCorrect]   bi={right_bi} → iliac_right")
        print(f"[IliacCorrect]   bi={left_bi}  → iliac_left")
        print("[IliacCorrect] ────────────────────────────────────────────────────")

        # Sync corrected bi values back to logic so the navigation mixin
        # resolves the right branch index after the label swap.
        try:
            logic = getattr(self, "logic", None)
            if logic is not None:
                logic._iliacRightBi = right_bi
                logic._iliacLeftBi  = left_bi
                print(f"[IliacCorrect] Synced logic._iliacRightBi={right_bi}  "
                      f"_iliacLeftBi={left_bi}")
        except Exception as se:
            print(f"[IliacCorrect] logic bi-sync skipped: {se}")

        try:
            self._refreshBranchSelectorAfterLabelCorrection()
        except Exception as re:
            print(f"[IliacCorrect] Branch selector refresh skipped: {re}")

    def _refreshBranchSelectorAfterLabelCorrection(self):
        """Re-populate branch combo boxes after iliac label correction so the
        UI reflects the corrected role names rather than the stale mixin output."""
        try:
            logic = self.logic
            branchMeta = getattr(logic, "branchMeta", [])
            if not branchMeta:
                return

            # Navigation branch selector (branchSelector combo)
            try:
                combo = self.branchSelector
                current_idx = combo.currentIndex
                try:
                    current_bi = combo.itemData(current_idx)
                except Exception:
                    current_bi = getattr(logic, "activeBranch", -1)
                combo.blockSignals(True)
                combo.clear()
                combo.addItem("All branches", -1)
                _bm_items = (
                    branchMeta.items()
                    if isinstance(branchMeta, dict)
                    else enumerate(branchMeta)
                )
                for bi, meta in sorted(_bm_items):
                    role  = meta.get("role", f"Branch {bi+1}")
                    if role in ("trunk", "noise", "renal_fragment", "iliac_fragment"):
                        continue
                    try:
                        label = logic.getBranchDisplayName(bi)
                    except Exception:
                        label = meta.get("label") or role
                    combo.addItem(label, bi)
                restored_idx = 0
                for _ci in range(combo.count):
                    if combo.itemData(_ci) == current_bi:
                        restored_idx = _ci
                        break
                combo.setCurrentIndex(restored_idx)
                restored_bi = combo.itemData(restored_idx)
                if restored_bi is None:
                    restored_bi = -1
                try:
                    logic.setActiveBranch(restored_bi)
                except Exception:
                    logic.activeBranch = restored_bi
                try:
                    if restored_bi == -1:
                        _cur_idx = int(self.pointSlider.value)
                        _cur_gi = logic.localToGlobal(_cur_idx)
                        _trav_branch_ids = getattr(
                            logic, "_allBranchTraversalBranchIds", None
                        )
                        if (
                            _trav_branch_ids
                            and 0 <= _cur_idx < len(_trav_branch_ids)
                            and isinstance(_trav_branch_ids[_cur_idx], int)
                        ):
                            _cur_bi = _trav_branch_ids[_cur_idx]
                        else:
                            _cur_bi = logic.getBranchForPoint(_cur_gi)
                        _bm = getattr(logic, "branchMeta", {})
                        _role = _bm.get(_cur_bi, {}).get("role", "")
                        if _role in ("iliac_fragment", "renal_fragment"):
                            _parent_bi = _bm.get(_cur_bi, {}).get("fragment_of")
                            if isinstance(_parent_bi, int) and _parent_bi >= 0:
                                _cur_bi = _parent_bi
                        if _cur_bi >= 0:
                            self.branchNameLabel.setText(logic.getBranchDisplayName(_cur_bi))
                        else:
                            self.branchNameLabel.setText("All branches")
                    else:
                        self.branchNameLabel.setText(logic.getBranchDisplayName(restored_bi))
                except Exception:
                    pass
                print(
                    f"[NavigatorCombo DEBUG] restored selection: "
                    f"row={restored_idx} bi={restored_bi} "
                    f"activeBranch={getattr(logic, 'activeBranch', '?')}"
                )
                try:
                    _bn_text = self.branchNameLabel.text
                except Exception as _bne:
                    _bn_text = f"ERR:{_bne}"
                try:
                    _combo_idx_now = combo.currentIndex
                    _combo_data_now = combo.itemData(_combo_idx_now)
                    _combo_text_now = combo.itemText(_combo_idx_now)
                except Exception as _cne:
                    _combo_idx_now, _combo_data_now, _combo_text_now = "ERR", "ERR", f"ERR:{_cne}"
                print(
                    f"[NavigatorLabel DEBUG] after-refresh-visible-state: "
                    f"comboIndex={_combo_idx_now} comboData={_combo_data_now!r} "
                    f"comboText={_combo_text_now!r} "
                    f"activeBranch={getattr(logic, 'activeBranch', '?')!r} "
                    f"branchNameLabel={_bn_text!r}"
                )
                print("[NavigatorCombo DEBUG] after IliacCorrect refresh:")
                _br_dbg = getattr(logic, "branches", [])
                for _row in range(combo.count):
                    _data = combo.itemData(_row)
                    _txt = combo.itemText(_row)
                    if _data is None or _data == -1:
                        print(f"[NavigatorCombo DEBUG]   row={_row} text={_txt!r} data={_data!r}")
                        continue
                    _meta = branchMeta.get(_data, {}) if isinstance(branchMeta, dict) else branchMeta[_data]
                    _rng = _br_dbg[_data] if 0 <= _data < len(_br_dbg) else ("?", "?")
                    _mark = "  <-- restored" if _row == restored_idx else ""
                    try:
                        _disp = logic.getBranchDisplayName(_data)
                    except Exception as _de:
                        _disp = f"ERROR:{_de}"
                    print(
                        f"[NavigatorCombo DEBUG]   row={_row} data_bi={_data} "
                        f"text={_txt!r} role={_meta.get('role','?')!r} "
                        f"label={_meta.get('label','?')!r} "
                        f"display={_disp!r} range={_rng}{_mark}"
                    )
                combo.blockSignals(False)
                print(f"[IliacCorrect] branchSelector refreshed ({combo.count} items)")
            except AttributeError:
                pass   # widget may not exist in all UI configurations

            # Stent branch list (stentBranchList widget)
            try:
                self.onStentBranchListChanged()
            except Exception:
                pass

            # Re-apply direct RGB mapper — IliacCorrect runs AFTER onClassifySurface,
            # and the selector refresh above pumps Qt events which cause Slicer's
            # displayable manager to rebuild the mapper in LUT mode, wiping the
            # direct-scalar setting that onClassifySurface applied.
            self._reapplyDirectRGBMapper()

        except Exception as exc:
            print(f"[IliacCorrect] _refreshBranchSelectorAfterLabelCorrection: {exc}")

    def _reapplyDirectRGBMapper(self):
        """Re-apply the identity LUT on the vessel model mapper after MRML resets it.

        Called after any operation that pumps Qt events (selector refresh,
        infoDisplay, etc.) which cause Slicer's vtkMRMLModelDisplayableManager
        to rebuild the mapper, discarding the LUT set in onClassifySurface.

        Reuses self._branchClassLUT built by onClassifySurface.  Safe to call
        when no BranchClassification array exists — exits silently.
        """
        try:
            node = getattr(self, "branchSurfModelSelector", None)
            node = node.currentNode() if node else None
            if node is None:
                node = getattr(self, "modelSelector", None)
                node = node.currentNode() if node else None
            if node is None or not node.GetPolyData():
                return
            poly = node.GetPolyData()
            if poly.GetPointData().GetArray("BranchClassification") is None:
                return
            poly.GetPointData().SetActiveScalars("BranchClassification")
            dn = node.GetDisplayNode()
            if dn:
                dn.SetActiveScalarName("BranchClassification")
                dn.SetScalarVisibility(True)
                dn.SetScalarRangeFlag(0)
                dn.SetScalarRange(0, 255)
            # Retrieve the LUT built by onClassifySurface (stored on the widget)
            _lut = getattr(self, "_branchClassLUT", None)
            import slicer as _sl
            _sl.app.processEvents()
            try:
                view = _sl.app.layoutManager().threeDWidget(0).threeDView()
                renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
                n_target = poly.GetNumberOfPoints()
                for ac in renderer.GetActors():
                    mp = ac.GetMapper()
                    if mp is None:
                        continue
                    try:
                        inp = mp.GetInputDataObject(0, 0)
                    except Exception:
                        continue
                    if inp is None:
                        continue
                    if getattr(inp, "GetNumberOfPoints", lambda: -1)() != n_target:
                        continue
                    if inp.GetPointData().GetArray("BranchClassification") is None:
                        continue
                    inp.GetPointData().SetActiveScalars("BranchClassification")
                    mp.SetColorModeToMapScalars()
                    mp.SetScalarModeToUsePointData()
                    mp.ScalarVisibilityOn()
                    mp.SetScalarRange(0, 255)
                    if _lut is not None:
                        mp.SetLookupTable(_lut)
                        mp.UseLookupTableScalarRangeOn()
                    view.renderWindow().Render()
                    print("[IliacCorrect] _reapplyDirectRGBMapper: LUT restored ✓")
                    return
            except Exception as _me:
                print(f"[IliacCorrect] _reapplyDirectRGBMapper mapper scan: {_me}")
        except Exception as _e:
            print(f"[IliacCorrect] _reapplyDirectRGBMapper: {_e}")

    # ── Surface classification & findings ────────────────────────────────
    # Methods: onFindAndPlaceStents, onVesselTypeChanged, onDetectFindings,
    #          onClearOverlay, onCreateSurface, onCheckSurface,
    #          onClassifySurface, onClearSurfaceClassification,
    #          onDebugOstium, onDetectCollaterals
    from vessel_surface_findings_widget_mixin import SurfaceFindingsWidgetMixin as _SFWM
    onFindAndPlaceStents          = _SFWM.onFindAndPlaceStents
    onVesselTypeChanged           = _SFWM.onVesselTypeChanged
    onDetectFindings              = _SFWM.onDetectFindings
    onClearOverlay                = _SFWM.onClearOverlay
    onCreateSurface               = _SFWM.onCreateSurface
    onCheckSurface                = _SFWM.onCheckSurface
    onClassifySurface             = _SFWM.onClassifySurface
    onClearSurfaceClassification  = _SFWM.onClearSurfaceClassification
    onDebugOstium                 = _SFWM.onDebugOstium
    onDetectCollaterals           = _SFWM.onDetectCollaterals

    # ── Stent planning ───────────────────────────────────────────────────
    # Methods: _updateStentSliderRanges, _resetStentSlidersForBranch,
    #          onStentBranchListChanged, onStentBranchChanged,
    #          onStentSliderChanged, onStentParamChanged, onStentLengthApply,
    #          onKissNomChanged, onTrunkSpinChanged, onKissingApply,
    #          _liveUpdateStent, _refreshStentSummary, onStentAutoSize,
    #          onAutoPlaceAllStents, onAutoPlaceKissing, onAutoDetectStentType,
    #          _updateStentSummaryOnly, _restoreDefaultLayout, _snapStentLength
    from vessel_stent_widget_mixin import StentWidgetMixin as _SWM
    _updateStentSliderRanges      = _SWM._updateStentSliderRanges
    _resetStentSlidersForBranch   = _SWM._resetStentSlidersForBranch
    onStentBranchListChanged      = _SWM.onStentBranchListChanged
    onStentBranchChanged          = _SWM.onStentBranchChanged
    onStentSliderChanged          = _SWM.onStentSliderChanged
    onStentParamChanged           = _SWM.onStentParamChanged
    onStentLengthApply            = _SWM.onStentLengthApply
    onKissNomChanged              = _SWM.onKissNomChanged
    onTrunkSpinChanged            = _SWM.onTrunkSpinChanged
    onKissingApply                = _SWM.onKissingApply
    _liveUpdateStent              = _SWM._liveUpdateStent
    _refreshStentSummary          = _SWM._refreshStentSummary
    onStentAutoSize               = _SWM.onStentAutoSize
    onAutoPlaceAllStents          = _SWM.onAutoPlaceAllStents
    onAutoPlaceKissing            = _SWM.onAutoPlaceKissing
    onAutoDetectStentType         = _SWM.onAutoDetectStentType
    _updateStentSummaryOnly       = _SWM._updateStentSummaryOnly
    _restoreDefaultLayout         = _SWM._restoreDefaultLayout
    _snapStentLength              = _SWM._snapStentLength

    # ── Rulers, embolization, comparison & lesion table ──────────────────
    # Methods: onAddRuler, onRemoveRuler, onModelOpacityChanged,
    #          onModelVisToggle, onEmboDeviceChanged, onEmboPickZone,
    #          _pollEmboPickPoints, _finishEmboPick, onEmboCancelPick,
    #          onEmboPlace, _emboPlaceMarker, _emboMakeTube, _emboMakeNode,
    #          _emboRadii, _placeCoilEmbo, _placePlugEmbo, _placeLiquidEmbo,
    #          _placeFlowDiverter, onEmboRemove, onExitCompare,
    #          onCompareBeforeAfter, _compareAfterLayoutSwitch,
    #          _addStenosisMarker, _buildLesionTable, onLesionTableClicked,
    #          onCopyPreDilateStatus, onCopyCoordinates, onApplyManualRanges,
    #          onClearManualRanges
    from vessel_embo_compare_widget_mixin import EmboCompareWidgetMixin as _ECWM
    onAddRuler                    = _ECWM.onAddRuler
    onRemoveRuler                 = _ECWM.onRemoveRuler
    onModelOpacityChanged         = _ECWM.onModelOpacityChanged
    onModelVisToggle              = _ECWM.onModelVisToggle
    onEmboDeviceChanged           = _ECWM.onEmboDeviceChanged
    onEmboPickZone                = _ECWM.onEmboPickZone
    _pollEmboPickPoints           = _ECWM._pollEmboPickPoints
    _finishEmboPick               = _ECWM._finishEmboPick
    onEmboCancelPick              = _ECWM.onEmboCancelPick
    onEmboPlace                   = _ECWM.onEmboPlace
    _emboPlaceMarker              = _ECWM._emboPlaceMarker
    _emboMakeTube                 = _ECWM._emboMakeTube
    _emboMakeNode                 = _ECWM._emboMakeNode
    _emboRadii                    = _ECWM._emboRadii
    _placeCoilEmbo                = _ECWM._placeCoilEmbo
    _placePlugEmbo                = _ECWM._placePlugEmbo
    _placeLiquidEmbo              = _ECWM._placeLiquidEmbo
    _placeFlowDiverter            = _ECWM._placeFlowDiverter
    onEmboRemove                  = _ECWM.onEmboRemove
    onExitCompare                 = _ECWM.onExitCompare
    onCompareBeforeAfter          = _ECWM.onCompareBeforeAfter
    _compareAfterLayoutSwitch     = _ECWM._compareAfterLayoutSwitch
    _addStenosisMarker            = _ECWM._addStenosisMarker
    _buildLesionTable             = _ECWM._buildLesionTable
    onLesionTableClicked          = _ECWM.onLesionTableClicked
    onCopyPreDilateStatus         = _ECWM.onCopyPreDilateStatus
    onCopyCoordinates             = _ECWM.onCopyCoordinates
    onApplyManualRanges           = _ECWM.onApplyManualRanges
    onClearManualRanges           = _ECWM.onClearManualRanges

    # ── Pre-dilate, POT, carina, balloon, stent-pick, place & report ─────
    # Methods: onPreDilate, onApplyPOT, onRemovePOT, onApplyCarinaSupportSIM,
    #          onYStentReplace, onRemoveCarinaSupport, onManualBalloonStart,
    #          _pollBalloonPickPoints, _finishBalloonPick, _finishBalloonPickImpl,
    #          onManualBalloonCancel, onStentPickPoints, _pollStentPickPoints,
    #          _finishStentPick, onStentPickCancel, onStentPickY,
    #          _finishStentPickY, _snapToCenterline, _traceCenterlinePath,
    #          _removeBalloons, _removeStentedMarkers, onStentPlace,
    #          onStentRemove, onGenerateReport
    from vessel_intervention_widget_mixin import InterventionWidgetMixin as _IWM
    onPreDilate                   = _IWM.onPreDilate
    onApplyPOT                    = _IWM.onApplyPOT
    onRemovePOT                   = _IWM.onRemovePOT
    onApplyCarinaSupportSIM       = _IWM.onApplyCarinaSupportSIM
    onYStentReplace               = _IWM.onYStentReplace
    onRemoveCarinaSupport         = _IWM.onRemoveCarinaSupport
    onManualBalloonStart          = _IWM.onManualBalloonStart
    _pollBalloonPickPoints        = _IWM._pollBalloonPickPoints
    _finishBalloonPick            = _IWM._finishBalloonPick
    _finishBalloonPickImpl        = _IWM._finishBalloonPickImpl
    onManualBalloonCancel         = _IWM.onManualBalloonCancel
    onStentPickPoints             = _IWM.onStentPickPoints
    _pollStentPickPoints          = _IWM._pollStentPickPoints
    _finishStentPick              = _IWM._finishStentPick
    onStentPickCancel             = _IWM.onStentPickCancel
    onStentPickY                  = _IWM.onStentPickY
    _finishStentPickY             = _IWM._finishStentPickY
    _snapToCenterline             = _IWM._snapToCenterline
    _traceCenterlinePath          = _IWM._traceCenterlinePath
    _removeBalloons               = _IWM._removeBalloons
    _removeStentedMarkers         = _IWM._removeStentedMarkers
    onStentPlace                  = _IWM.onStentPlace
    onStentRemove                 = _IWM.onStentRemove
    onGenerateReport              = _IWM.onGenerateReport

    # ── CloudCompare mapping (Open3D) ────────────────────────────────────────
    # Methods: onCloudCompareMap, onCloudCompareClear
    from vessel_cloudcompare_mixin import CloudCompareWidgetMixin as _CCWM
    onCloudCompareMap   = _CCWM.onCloudCompareMap
    onCloudCompareClear = _CCWM.onCloudCompareClear

    # ── Surface scalar maps (radius + wall thickness) ─────────────────────────
    from vessel_surface_map_mixin import SurfaceMapWidgetMixin as _SMWM
    onRadiusMap        = _SMWM.onRadiusMap
    onWallThicknessMap = _SMWM.onWallThicknessMap
    onSurfaceMapClear  = _SMWM.onSurfaceMapClear

    # ── Multi-ray adaptive radius ─────────────────────────────────────────────
    from vessel_blender_multiray_mixin import MultiRayBlenderWidgetMixin as _MRW
    onSendToBlender        = _MRW.onSendToBlender
    onBlenderServerStart   = _MRW.onBlenderServerStart
    onBlenderServerStop    = _MRW.onBlenderServerStop
    onRefineServerStart    = _MRW.onRefineServerStart
    onRefineServerStop     = _MRW.onRefineServerStop
    onMultiRayBlenderMap   = _MRW.onMultiRayBlenderMap
    onMultiRayBlenderClear = _MRW.onMultiRayBlenderClear


# ── Logic mixins ─────────────────────────────────────────────────────────────
try:
    from vessel_centerline_mixin import CenterlineMixin
    from vessel_ostium_mixin import OstiumMixin
    from vessel_findings_mixin import FindingsMixin
    from vessel_visualization_mixin import VisualizationMixin
    from vessel_stent_mixin import StentMixin
    from vessel_branch_accessor_mixin import BranchAccessorMixin
    from vessel_report_mixin import ReportMixin
    from vessel_cloudcompare_mixin import CloudCompareMixin          # CloudCompare mapping (Open3D)
    from vessel_blender_multiray_mixin import MultiRayBlenderMixin   # Multi-ray adaptive radius refinement
    from vessel_surface_map_mixin import SurfaceMapMixin             # Radius map + wall thickness map
except ImportError as _e:
    raise ImportError(
        f"VesselAnalyzer: failed to import a logic mixin — {_e}\n"
        "Make sure all vessel_*_mixin.py files are in the same directory as VesselAnalyzer.py"
    ) from _e


class VesselAnalyzerLogic(
    CenterlineMixin,
    OstiumMixin,
    FindingsMixin,
    VisualizationMixin,
    StentMixin,
    BranchAccessorMixin,
    ReportMixin,
    CloudCompareMixin,               # CloudCompare distance mapping (Open3D)
    MultiRayBlenderMixin,            # Multi-ray adaptive radius refinement
    SurfaceMapMixin,                 # Radius map + wall thickness map
    ScriptedLoadableModuleLogic,
):
    # ── Legacy State Shims ───────────────────────────────────────────────────
    @property
    def points(self): return self.state.points
    @points.setter
    def points(self, value): self.state.points = value

    @property
    def distances(self): return self.state.distances
    @distances.setter
    def distances(self, value): self.state.distances = value

    @property
    def diameters(self): return self.state.diameters
    @diameters.setter
    def diameters(self, value): self.state.diameters = value

    @property
    def _diam_minor(self): return self.state._diam_minor
    @_diam_minor.setter
    def _diam_minor(self, value): self.state._diam_minor = value

    @property
    def _diam_major(self): return self.state._diam_major
    @_diam_major.setter
    def _diam_major(self, value): self.state._diam_major = value

    @property
    def branches(self): return self.state.branches
    @branches.setter
    def branches(self, value): self.state.branches = value

    @property
    def activeBranch(self): return self.state.active_branch
    @activeBranch.setter
    def activeBranch(self, value): self.state.active_branch = value

    @property
    def modelNode(self): return self.state.model_node
    @modelNode.setter
    def modelNode(self, value): self.state.model_node = value
    
    @property
    def segModelNode(self): return self.state.seg_model_node
    @segModelNode.setter
    def segModelNode(self, value): self.state.seg_model_node = value

    @property
    def vesselType(self): return self.state.vessel_type
    @vesselType.setter
    def vesselType(self, value): self.state.vessel_type = value

    @property
    def manualBranchRanges(self): return self.state.manual_branch_ranges
    @manualBranchRanges.setter
    def manualBranchRanges(self, value): self.state.manual_branch_ranges = value

    @property
    def branchMeta(self): return self.state.branch_meta
    @branchMeta.setter
    def branchMeta(self, value): self.state.branch_meta = value

    @property
    def ostiumConfidenceMap(self): return self.state.ostia
    @ostiumConfidenceMap.setter
    def ostiumConfidenceMap(self, value): self.state.ostia = value

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.state = VesselState()
        self.logger = StructuredLogger()
        
        self.pipeline = VesselPipeline([
            CenterlineStage(self, strategies=[
                VMTKStrategy(),
                NudgedSeedStrategy(),
                StraightLineStrategy()
            ]),
            OstiumStage(self),
            RadiusMapStage(self),
            FindingsStage(self)
        ], debug=self.logger)
        
        self._sphereModelNode = None
        self._ringModelNode = None
        self._lineMarkupNode = None

        # ── Logistic-regression ostium scorer ────────────────────────────
        # Weights file sits alongside VesselAnalyzer.py in the same directory.
        import os as _os

        _va_dir = _os.path.dirname(_os.path.abspath(__file__))
        self._logreg_weights_path = _os.path.join(_va_dir, "ostium_logreg_weights.json")
        self._logreg_data_path = _os.path.join(_va_dir, "ostium_logreg_data.json")
        self._logreg = _OstiumLogReg(weights_path=self._logreg_weights_path)

        # ── VesselDebug state snapshots ───────────────────────────────────
        # Populated incrementally by the pipeline so VesselDebug.dump_all()
        # can be called at any point with a consistent view of the state.
        #
        # Area 1 — Mesh clipping / endpoint detection
        self._vd_rings           = []    # list of {centroid,radius,n_pts}
        self._vd_endpoints       = []    # list of (x,y,z) final endpoints
        self._vd_lateral_z_range = None  # (z_min, z_max) of IVC trunk body
        # Area 2 — Iliac role assignment
        self._vd_iliac_before    = {}    # {bi: {role, tip, x_centroid}} snapshot
        # Area 3 — Ostium confidence
        self._vd_conf_map        = {}    # {bi: {grade, eff, raw, flags, spread}}
        # Area 4 — Centerline topology
        self._vd_bif_nodes       = []    # [{pt, score, is_primary, degree, max_angle}]

        # Install post-run auto-populate wrapper
        self._vd_wrap_run()

    def debug_dump(self):
        """Public convenience: run VesselDebug.dump_all() with all accumulated
        state.  Call from the Python console at any time after pipeline completion:

            slicer.modules.VesselAnalyzerWidget.logic.debug_dump()
        """
        # Auto-populate from branchMeta if the mixin fields weren't filled yet
        self._vd_sync_from_branch_meta()
        VesselDebug.dump_all(
            branch_meta      = getattr(self, "branchMeta", {}),
            conf_map         = self._vd_conf_map,
            bif_nodes        = self._vd_bif_nodes,
            rings            = self._vd_rings,
            endpoints        = self._vd_endpoints,
            lateral_z_range  = self._vd_lateral_z_range,
            iliac_before     = self._vd_iliac_before if self._vd_iliac_before else None,
        )

    def _vd_sync_from_branch_meta(self):
        """Derive all _vd_* fields from branchMeta + graph state produced by the
        mixin pipeline.  Safe to call repeatedly — overwrites stale snapshots.

        This is the zero-integration path: no mixin edits needed.  Everything
        is reconstructed from the data that already exists on self after a run.
        """
        try:
            bm = getattr(self, "branchMeta", {})
            if not bm:
                print("[VD] _vd_sync: branchMeta is empty — run the pipeline first.")
                return

            # ── Area 1: rings & endpoints ─────────────────────────────────
            # Re-derive from branchMeta ostium points and trunk Z extent.
            # The actual ring data lives in the centerline mixin; approximate
            # from known branch tip / ostium positions.
            if not self._vd_endpoints:
                eps = []
                for bi, meta in bm.items():
                    tip = meta.get("tip") or meta.get("ostium_pt")
                    if tip:
                        eps.append(tuple(tip))
                self._vd_endpoints = eps

            # Trunk Z range from trunk branch points
            trunk_meta = next((m for m in bm.values()
                               if m.get("role","").lower() == "trunk"), None)
            if trunk_meta and self._vd_lateral_z_range is None:
                pts = trunk_meta.get("pts", [])
                if pts:
                    zs = [p[2] for p in pts]
                    # lateral range = body of trunk (exclude top 10%)
                    z_lo = min(zs)
                    z_hi = max(zs)
                    margin = (z_hi - z_lo) * 0.10
                    self._vd_lateral_z_range = (z_lo + margin, z_hi - margin)

            # Rings: build synthetic entries from each non-trunk ostium
            if not self._vd_rings:
                for bi, meta in bm.items():
                    role = meta.get("role","")
                    if "trunk" in role.lower():
                        continue
                    opt = meta.get("ostium_pt")
                    if opt:
                        diam = meta.get("diam", 0.0)
                        self._vd_rings.append({
                            "centroid": tuple(opt),
                            "radius":   diam / 2.0,
                            "n_pts":    0,   # not available without mixin data
                        })

            # ── Area 2: iliac-before snapshot ─────────────────────────────
            # Build a "before" that mirrors current state (best we can do
            # without intercepting IliacCorrect at runtime).
            if not self._vd_iliac_before:
                for bi, meta in bm.items():
                    self._vd_iliac_before[bi] = {
                        "role":       meta.get("iliac_role_before", meta.get("role", "?")),
                        "tip":        meta.get("tip", (0,0,0)),
                        "x_centroid": meta.get("x_centroid", float("nan")),
                    }

            # ── Area 3: confidence map ────────────────────────────────────
            if not self._vd_conf_map:
                for bi, meta in bm.items():
                    self._vd_conf_map[bi] = {
                        "grade":  meta.get("conf_grade",  "?"),
                        "eff":    meta.get("conf_eff",    float("nan")),
                        "raw":    meta.get("conf_raw",    float("nan")),
                        "flags":  meta.get("conf_flags",  []),
                        "spread": meta.get("conf_spread", float("nan")),
                    }

            # ── Area 4: bifurcation nodes ─────────────────────────────────
            if not self._vd_bif_nodes:
                bif_scores = getattr(self, "_bifScores", {})
                primary_bi = getattr(self, "_primaryBifNode", None)
                for node_id, score in bif_scores.items():
                    pt = getattr(self, "_bifPositions", {}).get(node_id, (0,0,0))
                    self._vd_bif_nodes.append({
                        "pt":         pt,
                        "score":      score,
                        "is_primary": (node_id == primary_bi),
                        "degree":     3,
                        "max_angle":  float("nan"),
                    })

            print(f"[VD] _vd_sync complete: "
                  f"{len(self._vd_rings)} rings, "
                  f"{len(self._vd_endpoints)} endpoints, "
                  f"{len(self._vd_conf_map)} conf entries, "
                  f"{len(self._vd_bif_nodes)} bif nodes")

        except Exception as exc:
            import traceback
            print(f"[VD] _vd_sync error: {exc}")
            print(traceback.format_exc())

    # ── Auto-hook: wrap runAnalysis to populate _vd_* after every run ────────
    # This wraps whatever method the mixin exposes as the top-level entry point.
    # Works even though runAnalysis is defined in CenterlineMixin.
    def _vd_wrap_run(self):
        """Install a post-run wrapper on the logic entry point (called once from __init__).
        Tries several candidate method names used across CenterlineMixin versions."""
        # Candidate names the mixin might use as its top-level entry point
        candidates = ["runAnalysis", "extractCenterline", "computeCenterline",
                      "run", "analyze", "processCenterline"]
        for name in candidates:
            orig = getattr(self.__class__, name, None)
            if orig is None or getattr(orig, "_vd_wrapped", False):
                continue
            def _make_wrapper(orig_fn, method_name):
                def _wrapped(self_inner, *args, **kwargs):
                    result = orig_fn(self_inner, *args, **kwargs)
                    try:
                        self_inner._iliac_correct_fix()
                    except Exception as exc:
                        print(f"[IliacFix] Post-run fix error: {exc}")
                    try:
                        self_inner._vd_rings.clear()
                        self_inner._vd_endpoints.clear()
                        self_inner._vd_lateral_z_range = None
                        self_inner._vd_iliac_before.clear()
                        self_inner._vd_conf_map.clear()
                        self_inner._vd_bif_nodes.clear()
                        self_inner._vd_sync_from_branch_meta()
                        print(f"[VD] Auto-populated debug state after {method_name}() ✓")
                    except Exception as exc:
                        print(f"[VD] Post-run sync error: {exc}")
                    return result
                _wrapped._vd_wrapped = True
                return _wrapped
            self.__class__.__dict__  # force dict lookup
            setattr(self.__class__, name, _make_wrapper(orig, name))
            print(f"[VD] Wrapped logic entry point: {name}()")
            return
        print("[VD] _vd_wrap_run: no known entry point found on class — "
              "call logic.debug_dump() manually after each run")

    # ── IliacCorrect fix ─────────────────────────────────────────────────────
    # Root cause: the X-centroid fallback uses the proximal 25% of branch
    # points to compute X.  For the long side branch (bi=1, 341mm) that runs
    # from the bifurcation all the way to the left-iliac tip, the proximal 25%
    # curls toward negative-X, so the fallback wrongly concludes it is the
    # RIGHT iliac — even though its tip is 1.7mm from the left-iliac endpoint.
    #
    # Fix: after IliacCorrect runs, check every iliac-role branch against the
    # scene endpoint fiducials.  Any branch whose tip is within SNAP_MM of a
    # named iliac endpoint overrides the centroid-fallback assignment.
    # This runs automatically via the runAnalysis wrapper above.

    _ILIAC_CORRECT_FIX_SNAP_MM = 10.0   # mm — tip must be within this to lock

    def _iliac_correct_fix(self):
        """Post-IliacCorrect tip-anchor override.

        Re-reads the Endpoints fiducial node and for every branch whose tip
        is within _ILIAC_CORRECT_FIX_SNAP_MM of a named iliac endpoint,
        forces the correct role regardless of what the X-centroid fallback
        decided.

        Logs every change with [IliacFix] tag.
        """
        try:
            bm = getattr(self, "branchMeta", {})
            if not bm:
                return

            # ── 1. Read scene iliac endpoints by label ────────────────────
            ep_node = slicer.mrmlScene.GetFirstNodeByName("Endpoints")
            if ep_node is None:
                print("[IliacFix] No 'Endpoints' node found — skipping fix")
                return

            right_ep = None
            left_ep  = None
            tmp = [0.0, 0.0, 0.0]
            for i in range(ep_node.GetNumberOfControlPoints()):
                desc  = ep_node.GetNthControlPointDescription(i)
                label = ep_node.GetNthControlPointLabel(i)
                ep_node.GetNthControlPointPositionWorld(i, tmp)
                pt = tuple(tmp)
                if "Right iliac" in desc or "Right iliac" in label:
                    right_ep = pt
                elif "Left iliac" in desc or "Left iliac" in label:
                    left_ep  = pt

            if right_ep is None and left_ep is None:
                print("[IliacFix] Could not identify iliac endpoints by label — skipping")
                return

            print(f"[IliacFix] right_ep={right_ep}  left_ep={left_ep}")

            # ── 2. For each branch, compute tip→endpoint distances ─────────
            snap = self._ILIAC_CORRECT_FIX_SNAP_MM
            changes = 0
            for bi, meta in bm.items():
                role = meta.get("role", "")
                if "iliac" not in role.lower() and role not in ("side", "main"):
                    continue

                # Use the actual far tip of the branch (last point)
                pts = meta.get("pts", [])
                tip = tuple(pts[-1]) if pts else meta.get("tip", None)
                if tip is None:
                    continue

                def dist3(a, b):
                    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

                d_right = dist3(tip, right_ep) if right_ep else float("inf")
                d_left  = dist3(tip, left_ep)  if left_ep  else float("inf")
                best_d  = min(d_right, d_left)

                if best_d > snap:
                    continue   # tip not close to either endpoint — leave alone

                correct_role = "iliac_right" if d_right < d_left else "iliac_left"
                old_role     = role

                if old_role != correct_role:
                    meta["role"] = correct_role
                    # Also fix display name if present
                    if "display_name" in meta:
                        meta["display_name"] = (
                            "Right iliac" if correct_role == "iliac_right"
                            else "Left iliac"
                        )
                    changes += 1
                    print(
                        f"[IliacFix] bi={bi}  tip=({tip[0]:.1f},{tip[1]:.1f},{tip[2]:.1f})"
                        f"  d_right={d_right:.1f}mm  d_left={d_left:.1f}mm"
                        f"  {old_role} → {correct_role}  ✓ FIXED"
                    )
                else:
                    print(
                        f"[IliacFix] bi={bi}  role={correct_role} already correct"
                        f"  (d_right={d_right:.1f}  d_left={d_left:.1f})"
                    )

            print(f"[IliacFix] Done: {changes} role(s) corrected")

        except Exception as exc:
            import traceback
            print(f"[IliacFix] Error: {exc}")
            print(traceback.format_exc())

    def debug_clear(self):
        """Remove all VD_* debug markers from the 3D scene."""
        VesselDebug.clear_markers()

    # ── Hooks callable from mixin code ───────────────────────────────────────
    # These are thin setters so the mixin files can populate debug state
    # without importing VesselDebug themselves.

    def vd_record_rings(self, rings):
        """Area 1: record boundary rings detected after mesh clipping.

        Call from vessel_centerline_mixin after the clipping step, e.g.:
            self.vd_record_rings([
                {"centroid": (x,y,z), "radius": r, "n_pts": n},
                ...
            ])
        Then VesselDebug will colour-code lateral vs tube-end rings in 3D.
        """
        try:
            self._vd_rings = list(rings)
            print(f"[VD] vd_record_rings: {len(self._vd_rings)} ring(s) stored")
            # Immediate visual — run clip debug now with whatever Z range we have
            VesselDebug.mesh_clipping(
                self._vd_rings,
                self._vd_endpoints,
                self._vd_lateral_z_range,
            )
        except Exception as exc:
            print(f"[VD] vd_record_rings error: {exc}")

    def vd_snapshot_iliac_before(self):
        """Area 2: snapshot branchMeta roles *before* IliacCorrect runs.

        Call from vessel_ostium_mixin (or vessel_visualization_mixin) immediately
        before the IliacCorrect block:
            self.vd_snapshot_iliac_before()
            # ... IliacCorrect code ...
        """
        try:
            bm = getattr(self, "branchMeta", {})
            self._vd_iliac_before = {
                bi: {
                    "role":       meta.get("role", "?"),
                    "tip":        meta.get("tip",  (0,0,0)),
                    "x_centroid": meta.get("x_centroid", float("nan")),
                }
                for bi, meta in bm.items()
            }
            print(f"[VD] vd_snapshot_iliac_before: {len(self._vd_iliac_before)} branches snapshotted")
        except Exception as exc:
            print(f"[VD] vd_snapshot_iliac_before error: {exc}")

    def vd_record_conf(self, bi, grade, eff, raw, flags, spread):
        """Area 3: record ostium confidence result for one branch.

        Call from vessel_ostium_mixin inside OstiumConfidence loop:
            self.vd_record_conf(bi, grade, eff, raw, flags, spread)
        """
        try:
            self._vd_conf_map[bi] = {
                "grade":  grade,
                "eff":    eff,
                "raw":    raw,
                "flags":  list(flags) if flags else [],
                "spread": spread,
            }
        except Exception as exc:
            print(f"[VD] vd_record_conf error: {exc}")

    def vd_record_bif(self, pt, score, is_primary, degree, max_angle=float("nan")):
        """Area 4: record one bifurcation node.

        Call from vessel_centerline_mixin after bifurcation detection:
            self.vd_record_bif(pt, score, is_primary, degree, max_angle)
        """
        try:
            self._vd_bif_nodes.append({
                "pt":         tuple(pt),
                "score":      score,
                "is_primary": is_primary,
                "degree":     degree,
                "max_angle":  max_angle,
            })
        except Exception as exc:
            print(f"[VD] vd_record_bif error: {exc}")

    def cleanup(self):
        for node in [self._sphereModelNode, self._ringModelNode, self._lineMarkupNode]:
            if node and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self._sphereModelNode = None
        self._ringModelNode = None
        self._lineMarkupNode = None
