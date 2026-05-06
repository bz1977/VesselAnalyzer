"""
centerline_analysis.py — Centerline graph analysis helpers for VesselAnalyzer.

Extraction policy
-----------------
Only code that satisfies ALL of the following lives here:
  1. No Slicer or Qt imports at module level (slicer may be imported lazily
     inside individual functions where strictly required, e.g. debug helpers).
  2. No closure over pipeline state (self, nodePos, cAdj, cEdge, …).
  3. Deterministic given inputs — same args → same result, always.
  4. Unit-testable standalone: ``python -c "from centerline_analysis import *"``.

Classes / functions
-------------------
CenterlineGraph          Lightweight adjacency-list graph with radii
build_graph              Build graph by connecting consecutive points (legacy)
build_graph_from_polydata Build topology-correct graph from VMTK vtkPolyData
detect_bifurcations      Candidate bifurcation nodes by angle + radius-ratio
score_bifurcation        Quality score [0, 1] for a single bifurcation node
extract_branches         Grow branches outward from high-quality bifurcations
find_trunk               Identify trunk branch (highest trunk-score, radius-gated)
find_trunk_from_cells    Identify trunk from VMTK cells (preferred over find_trunk)
refine_branch_origins    Walk backward to true anatomical branch origin
debug_branch_origins     Place Slicer fiducials to validate origin locations
detect_compression       Screen diameter pairs for vessel compression

Usage in VesselAnalyzer.py
--------------------------
Replace the inline block (lines ~9698–10106) with::

    from centerline_analysis import (
        CenterlineGraph,
        build_graph,
        build_graph_from_polydata,
        detect_bifurcations,
        score_bifurcation,
        extract_branches,
        find_trunk,
        find_trunk_from_cells,
        refine_branch_origins,
        debug_branch_origins,
        detect_compression,
    )

The ``import numpy as _np`` and ``from collections import defaultdict as
_defaultdict`` at line ~9680 can be kept in VesselAnalyzer.py for the rest of
the file, OR removed if nothing else in that file uses them directly.
"""

from __future__ import annotations

import math
from collections import defaultdict as _defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as _np

from vessel_geometry import geo_angle_deg as _va_angle  # public alias; _va_angle kept for readability


# ─────────────────────────────────────────────────────────────────────────────
# CenterlineGraph
# ─────────────────────────────────────────────────────────────────────────────

class CenterlineGraph:
    """Lightweight adjacency-list graph of centerline points with radii.

    Parameters
    ----------
    points : array-like, shape (N, 3)
    radii  : array-like, shape (N,)
    """

    def __init__(self, points, radii):
        self.points = _np.asarray(points, dtype=float)
        self.radii = _np.asarray(radii, dtype=float)
        self.edges = _defaultdict(list)  # node_id → [node_id, ...]

    def add_edge(self, i, j):
        self.edges[i].append(j)
        self.edges[j].append(i)


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(points, radii):
    """Build a CenterlineGraph by connecting consecutive points.

    This is the legacy fallback for when a raw flattened point array is
    supplied (e.g. a single straight-line centerline with no branching).
    For multi-branch VMTK output use ``build_graph_from_polydata`` instead —
    it respects the cell-per-branch topology that VMTK produces.

    Parameters
    ----------
    points : list/array of (x, y, z) tuples or array shape (N, 3)
    radii  : list/array of N radius values

    Returns
    -------
    CenterlineGraph
    """
    g = CenterlineGraph(points, radii)
    for i in range(len(g.points) - 1):
        g.add_edge(i, i + 1)
    return g


def build_graph_from_polydata(centerline_polydata):
    """Build a CenterlineGraph that correctly reflects VMTK branch topology.

    VMTK's centerline output is a vtkPolyData where **each cell is one
    branch** (a separate polyline from one endpoint to another, passing
    through bifurcation nodes that are shared between cells).  The legacy
    ``build_graph`` ignored cell structure entirely, chaining all points
    consecutively — which caused bifurcation detection to operate on a
    flattened, non-topological array and trunk identification to fail.

    This function:
      1. Iterates over every cell (branch polyline) in the polydata.
      2. Collects all unique points (de-duplicated by position to a 0.01 mm
         tolerance) into a global point list.
      3. Adds edges between consecutive points within each cell, giving the
         graph correct branching topology.
      4. Reads per-point radius from the ``MaximumInscribedSphereRadius``
         point-data array (VMTK's standard name), falling back to
         ``Radius`` or a uniform 1.0 if neither is present.

    Parameters
    ----------
    centerline_polydata : vtk.vtkPolyData
        The polydata returned by VMTK's centerline extractor.

    Returns
    -------
    CenterlineGraph
        Graph whose edges reflect the true branch topology.
    list of list of int
        ``cells`` — each inner list is the sequence of global point indices
        for one VMTK cell/branch.  Pass directly to ``find_trunk_from_cells``
        instead of using ``extract_branches`` + ``find_trunk``.
    """
    import vtk

    pd = centerline_polydata

    # ── 1. Radius array ──────────────────────────────────────────────────────
    radius_arr = None
    for name in ("MaximumInscribedSphereRadius", "Radius"):
        radius_arr = pd.GetPointData().GetArray(name)
        if radius_arr is not None:
            break

    # ── 2. Collect unique points via spatial hash ─────────────────────────────
    #    VMTK duplicates bifurcation nodes across cells; merge them by position.
    TOL = 0.01          # mm — points closer than this are the same node
    pt_coords: List[_np.ndarray] = []
    pt_radii:  List[float]       = []
    coord_to_idx: Dict[Tuple[int, int, int], int] = {}

    def _key(x, y, z):
        return (int(round(x / TOL)), int(round(y / TOL)), int(round(z / TOL)))

    n_pts = pd.GetNumberOfPoints()
    raw_xyz   = _np.zeros((n_pts, 3))
    raw_radii = _np.ones(n_pts)
    ras = [0.0, 0.0, 0.0]
    for i in range(n_pts):
        pd.GetPoint(i, ras)
        raw_xyz[i] = ras
        if radius_arr is not None:
            raw_radii[i] = radius_arr.GetTuple1(i)

    vtk_to_global: Dict[int, int] = {}
    for vi in range(n_pts):
        k = _key(*raw_xyz[vi])
        if k not in coord_to_idx:
            gi = len(pt_coords)
            coord_to_idx[k] = gi
            pt_coords.append(raw_xyz[vi])
            pt_radii.append(float(raw_radii[vi]))
        vtk_to_global[vi] = coord_to_idx[k]

    # ── 3. Build graph edges from cells ──────────────────────────────────────
    g = CenterlineGraph(
        _np.array(pt_coords, dtype=float),
        _np.array(pt_radii,  dtype=float),
    )

    cells: List[List[int]] = []
    id_list = vtk.vtkIdList()
    n_cells = pd.GetNumberOfCells()
    for ci in range(n_cells):
        pd.GetCellPoints(ci, id_list)
        cell_global = [vtk_to_global[id_list.GetId(j)]
                       for j in range(id_list.GetNumberOfIds())]
        cells.append(cell_global)
        for j in range(len(cell_global) - 1):
            a, b = cell_global[j], cell_global[j + 1]
            if b not in g.edges[a]:   # avoid duplicate edges at shared bifurcation nodes
                g.add_edge(a, b)

    print(f"[build_graph_from_polydata] {n_cells} cells, "
          f"{len(pt_coords)} unique nodes, "
          f"r_max={max(pt_radii):.1f}mm r_mean={float(_np.mean(pt_radii)):.1f}mm")
    return g, cells


# ─────────────────────────────────────────────────────────────────────────────
# Bifurcation detection and scoring
# ─────────────────────────────────────────────────────────────────────────────

def detect_bifurcations(graph, angle_thr=30.0, radius_ratio_thr=0.9):
    """Detect candidate bifurcation nodes by angle and radius-ratio criteria.

    A node is a bifurcation candidate when at least one pair of its neighbours
    subtends an angle > *angle_thr* AND the two branch radii are sufficiently
    different (radius_ratio < *radius_ratio_thr*).

    Parameters
    ----------
    graph            : CenterlineGraph
    angle_thr        : float, degrees — minimum inter-branch angle
    radius_ratio_thr : float — maximum min/max radius ratio (lower = more asymmetric)

    Returns
    -------
    list of node indices that are bifurcation candidates
    """
    bifurcations = []
    for i, neighbors in graph.edges.items():
        if len(neighbors) < 2:
            continue
        p = graph.points[i]
        for j in neighbors:
            for k in neighbors:
                if j >= k:
                    continue
                v1 = graph.points[j] - p
                v2 = graph.points[k] - p
                ang = _va_angle(v1, v2)
                rj = graph.radii[j]
                rk = graph.radii[k]
                if max(rj, rk) == 0:
                    continue
                ratio = min(rj, rk) / max(rj, rk)
                if ang > angle_thr and ratio < radius_ratio_thr:
                    bifurcations.append(i)
    return list(set(bifurcations))


def score_bifurcation(graph, idx):
    """Compute a quality score in [0, 1] for a single bifurcation node.

    Score components (weighted sum):
      0.4 × mean inter-branch angle / 180°        — stronger divergence → higher
      0.3 × mean radius drop (1 − r_child/r_hub)  — greater taper → higher
      0.2 × mean edge length ratio                 — more equidistant → higher
      0.1 × −std of inter-branch angles / 180°    — noisy → lower

    Returns 0 if the node has fewer than 2 neighbours.
    """
    neighbors = graph.edges[idx]
    if len(neighbors) < 2:
        return 0.0

    p = graph.points[idx]
    r = graph.radii[idx]

    # --- per-neighbour metrics ------------------------------------------
    edge_lengths = []
    r_drops = []
    for j in neighbors:
        v = graph.points[j] - p
        edge_lengths.append(_np.linalg.norm(v))
        r_drops.append(graph.radii[j] / r if r > 0 else 1.0)

    # --- pairwise angle metrics -----------------------------------------
    angles = []
    for a in range(len(neighbors)):
        for b in range(a + 1, len(neighbors)):
            va = graph.points[neighbors[a]] - p
            vb = graph.points[neighbors[b]] - p
            angles.append(_va_angle(va, vb))

    if not angles:
        return 0.0

    max_edge = max(edge_lengths) if edge_lengths else 1.0
    angle_arr = _np.asarray(angles)

    angle_score = float(_np.mean(angle_arr)) / 180.0
    radius_score = 1.0 - float(_np.mean(r_drops))
    dist_score = float(_np.mean(edge_lengths)) / (max_edge + 1e-5)
    curvature_noise = float(_np.std(angle_arr)) / 180.0

    score = (
        0.4 * angle_score
        + 0.3 * radius_score
        + 0.2 * dist_score
        - 0.1 * curvature_noise
    )

    return float(_np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Branch extraction
# ─────────────────────────────────────────────────────────────────────────────

def _grow_branch(graph, start, visited):
    """Walk from *start* along unvisited degree-1 edges until blocked.

    Returns the ordered list of node indices forming the branch.
    """
    branch = [start]
    current = start
    while True:
        candidates = [n for n in graph.edges[current] if n not in visited]
        if len(candidates) != 1:
            break
        nxt = candidates[0]
        visited.add(nxt)
        branch.append(nxt)
        current = nxt
    return branch


def extract_branches(graph, bifurcations, score_thr=0.3):
    """Grow branches outward from each high-quality bifurcation node.

    Parameters
    ----------
    graph        : CenterlineGraph
    bifurcations : list of bifurcation node indices (from detect_bifurcations)
    score_thr    : float — skip bifurcations whose score < this threshold

    Returns
    -------
    list of branches, each branch being a list of node indices
    """
    visited = set()
    branches = []
    for b in bifurcations:
        if score_bifurcation(graph, b) < score_thr:
            continue
        for n in graph.edges[b]:
            if n in visited:
                continue
            visited.add(n)
            branch = _grow_branch(graph, n, visited)
            branches.append(branch)
    return branches


# ─────────────────────────────────────────────────────────────────────────────
# Trunk identification
# ─────────────────────────────────────────────────────────────────────────────

def _branch_metrics(graph, branch):
    """Return a composite trunk score for *branch* (higher = more trunk-like).

    Score = arc_length × mean_radius / (mean_curvature + ε)

    A long, wide, straight vessel segment scores highest — which correctly
    identifies the trunk/IVC/aorta over short, curved daughter branches.
    """
    pts = graph.points[branch]
    radii = graph.radii[branch]
    if len(pts) < 2:
        return 0.0
    segs = _np.diff(pts, axis=0)
    length = float(_np.sum(_np.linalg.norm(segs, axis=1)))
    mean_r = float(_np.mean(radii))
    # curvature proxy = mean turning angle between consecutive segments
    dirs = segs / (_np.linalg.norm(segs, axis=1, keepdims=True) + 1e-8)
    if len(dirs) > 1:
        dot_vals = _np.clip((_np.sum(dirs[:-1] * dirs[1:], axis=1)), -1.0, 1.0)
        curvature = float(_np.mean(_np.degrees(_np.arccos(dot_vals))))
    else:
        curvature = 0.0
    return length * mean_r / (curvature + 1e-5)


def find_trunk(graph, branches):
    """Identify the trunk branch (highest length × radius / curvature score).

    Parameters
    ----------
    graph    : CenterlineGraph
    branches : list of branches from extract_branches

    Returns
    -------
    The branch (list of node indices) with the highest trunk score,
    or None if *branches* is empty.

    .. note::
        Prefer ``find_trunk_from_cells`` when the graph was built with
        ``build_graph_from_polydata`` — it operates directly on VMTK cells
        and is immune to the consecutive-chain artefacts that can fool this
        function.
    """
    if not branches:
        return None
    scores = [_branch_metrics(graph, b) for b in branches]
    # Radius guard: a candidate whose mean_r is less than 40 % of the
    # widest candidate cannot be the trunk regardless of length/curvature.
    # This prevents a short renal-vein path from winning when the IVC body
    # is present as a wide, long segment.
    max_r = max(float(_np.mean(graph.radii[b])) for b in branches)
    r_floor = 0.40 * max_r
    for i, b in enumerate(branches):
        if float(_np.mean(graph.radii[b])) < r_floor:
            scores[i] = -1.0   # disqualify
    best = int(_np.argmax(scores))
    print(f"[find_trunk] winner idx={best} "
          f"r={float(_np.mean(graph.radii[branches[best]])):.1f}mm "
          f"score={scores[best]:.1f}")
    return branches[best]


def find_trunk_from_cells(graph, cells):
    """Identify the trunk cell from VMTK polydata cells.

    Operates directly on the per-cell lists returned by
    ``build_graph_from_polydata``, bypassing ``extract_branches`` and
    ``detect_bifurcations`` entirely.  This is the correct entry point when
    the graph was built from VMTK polydata.

    The trunk is the cell with the highest ``_branch_metrics`` score subject
    to the same radius guard used in ``find_trunk``: cells whose mean radius
    is below 40 % of the widest cell are disqualified.

    Parameters
    ----------
    graph : CenterlineGraph
        Built by ``build_graph_from_polydata``.
    cells : list of list of int
        Per-cell global node-index sequences (second return value of
        ``build_graph_from_polydata``).

    Returns
    -------
    trunk_cell : list of int
        Node-index sequence of the trunk cell.
    trunk_idx : int
        Index into *cells* of the winning cell.
    """
    if not cells:
        return None, -1

    scores = [_branch_metrics(graph, c) for c in cells]
    cell_r = [float(_np.mean(graph.radii[c])) for c in cells]
    max_r  = max(cell_r)
    r_floor = 0.40 * max_r

    for i, r in enumerate(cell_r):
        if r < r_floor:
            scores[i] = -1.0
        print(f"[find_trunk_from_cells] cell={i} "
              f"len={len(cells[i])} r={r:.1f}mm score={scores[i]:.1f}"
              + (" [disqualified]" if scores[i] < 0 else ""))

    best = int(_np.argmax(scores))
    print(f"[find_trunk_from_cells] trunk=cell[{best}] "
          f"r={cell_r[best]:.1f}mm score={scores[best]:.1f}")
    return cells[best], best


# ─────────────────────────────────────────────────────────────────────────────
# Branch origin refinement
# ─────────────────────────────────────────────────────────────────────────────

def refine_branch_origins(
    graph, bifurcations, asym_threshold=1.3, persistence=3, smooth_window=5
):
    """Walk backward from each detected bifurcation to find the true branch origin.

    The graph-topology node (bifurcation) is typically mid-branch — where
    shape deformation is *strongest*, not where it *starts*.  This function
    backtracks along the centerline until the vessel returns to single-lumen
    asymmetry, then interpolates to sub-voxel accuracy.

    Parameters
    ----------
    graph          : CenterlineGraph
    bifurcations   : list of node indices from detect_bifurcations()
    asym_threshold : float — A = max_r / min_r; below this → single lumen
    persistence    : int   — consecutive normal points required to confirm origin
    smooth_window  : int   — moving-average window for asymmetry signal

    Returns
    -------
    dict: { bif_node_idx : origin_point (x,y,z as np.array) }
    """
    origins = {}

    for bif_idx in bifurcations:
        # ── 1. Build a backward walk: gather the chain of nodes leading
        #       into this bifurcation from the upstream (trunk) side.
        #       Strategy: follow the neighbor whose radius is largest
        #       (trunk heuristic) iteratively, up to 60 steps.
        chain = []
        visited = {bif_idx}
        current = bif_idx
        for _ in range(60):
            nbrs = [n for n in graph.edges[current] if n not in visited]
            if not nbrs:
                break
            # pick neighbor with largest radius = trunk direction
            nxt = max(nbrs, key=lambda n: graph.radii[n])
            visited.add(nxt)
            chain.append(nxt)
            current = nxt
        chain = list(reversed(chain))  # now upstream → bif_idx order
        chain.append(bif_idx)

        if len(chain) < 2:
            origins[bif_idx] = graph.points[bif_idx]
            continue

        # ── 2. Compute asymmetry proxy at each step.
        raw_asym = []
        for node in chain:
            r_self = graph.radii[node]
            nbr_r = [graph.radii[n] for n in graph.edges[node]]
            if not nbr_r or r_self <= 0:
                raw_asym.append(1.0)
                continue
            asym = max(nbr_r) / (min(nbr_r) + 1e-8)
            raw_asym.append(asym)

        # ── 3. Smooth the asymmetry signal.
        w = min(smooth_window, len(raw_asym))
        kernel = _np.ones(w) / w
        if len(raw_asym) >= w:
            smoothed = list(_np.convolve(raw_asym, kernel, mode="same"))
        else:
            smoothed = raw_asym

        # ── 4. Walk forward through the chain; find the first sustained
        #       crossing of asym_threshold (persistence points in a row).
        origin_idx_in_chain = len(chain) - 1  # default: the bif node itself
        consec = 0
        for ci in range(len(chain)):
            if smoothed[ci] >= asym_threshold:
                consec += 1
                if consec >= persistence:
                    origin_idx_in_chain = max(0, ci - persistence + 1)
                    break
            else:
                consec = 0

        # ── 5. Sub-voxel interpolation between last-normal and first-abnormal.
        i_abnormal = origin_idx_in_chain
        i_normal = max(0, i_abnormal - 1)

        A_normal = smoothed[i_normal]
        A_abnormal = smoothed[i_abnormal]
        dA = A_abnormal - A_normal

        if dA > 1e-6:
            t = (asym_threshold - A_normal) / dA
            t = float(_np.clip(t, 0.0, 1.0))
        else:
            t = 0.0

        P_normal = graph.points[chain[i_normal]]
        P_abnormal = graph.points[chain[i_abnormal]]
        origin_pt = P_normal + t * (P_abnormal - P_normal)

        origins[bif_idx] = origin_pt

    return origins


# ─────────────────────────────────────────────────────────────────────────────
# Debug helper (lazily imports slicer — safe to call only inside Slicer)
# ─────────────────────────────────────────────────────────────────────────────

def debug_branch_origins(origins, bifurcations, graph, scene=None):
    """Place coloured markup fiducials in Slicer to validate origin locations.

    Colours
    -------
    Red    — P_detected  (raw bifurcation node from detect_bifurcations)
    Green  — P_origin    (refined origin from refine_branch_origins)

    Parameters
    ----------
    origins      : dict returned by refine_branch_origins()
    bifurcations : list of node indices (same list passed to refine_branch_origins)
    graph        : CenterlineGraph
    scene        : slicer.mrmlScene (optional, uses slicer.mrmlScene if None)
    """
    import slicer  # lazy — only available inside 3D Slicer

    if scene is None:
        scene = slicer.mrmlScene

    def _make_fiducial_node(name, color_rgb):
        node = scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        node.CreateDefaultDisplayNodes()
        dn = node.GetDisplayNode()
        dn.SetSelectedColor(*color_rgb)
        dn.SetColor(*color_rgb)
        dn.SetPointLabelsVisibility(True)
        dn.SetTextScale(2.5)
        return node

    detected_node = _make_fiducial_node("BifOrigin_Detected", (1.0, 0.15, 0.15))
    refined_node = _make_fiducial_node("BifOrigin_Refined", (0.18, 0.80, 0.44))

    for bif_idx in bifurcations:
        p_det = graph.points[bif_idx]
        detected_node.AddControlPoint(p_det[0], p_det[1], p_det[2], f"Det_{bif_idx}")

        if bif_idx in origins:
            p_orig = origins[bif_idx]
            refined_node.AddControlPoint(
                p_orig[0], p_orig[1], p_orig[2], f"Orig_{bif_idx}"
            )

    print(
        f"[BifOrigin] Placed {detected_node.GetNumberOfControlPoints()} "
        f"detected / {refined_node.GetNumberOfControlPoints()} refined markers."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Compression screening
# ─────────────────────────────────────────────────────────────────────────────

def detect_compression(diameters):
    """Screen a sequence of (dmin, dmax) diameter pairs for compression.

    Classification rules (matching VesselAnalyzer's clinical protocol):
      ratio = dmin / dmax
        ratio < 0.5  → Severe compression  (pancaking threshold)
        ratio < 0.7  → Mild compression

    Parameters
    ----------
    diameters : list of (dmin, dmax) tuples — minimum and maximum cross-
                sectional diameters at each centerline point.

    Returns
    -------
    list of (severity_label, ratio) tuples, one per input pair.
    Non-compressed points are omitted (sparse output).
    """
    compression = []
    for dmin, dmax in diameters:
        if dmax <= 0:
            continue
        ratio = dmin / (dmax + 1e-5)
        if ratio < 0.5:
            compression.append(("Severe", round(ratio, 3)))
        elif ratio < 0.7:
            compression.append(("Mild", round(ratio, 3)))
    return compression


# ── Slicer module-scanner guard ───────────────────────────────────────────────
class centerline_analysis:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "centerline_analysis"
            parent.hidden = True  # hide from Slicer module list
