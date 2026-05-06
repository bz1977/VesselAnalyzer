"""
centerline_graph.py — Pure graph-algorithm helpers for VesselAnalyzer.

Extraction policy
-----------------
Only functions that satisfy ALL of the following live here:
  1. No Slicer, VTK, or Qt imports.
  2. No closure over pipeline state (self, nodePos, cAdj, cEdge, …).
  3. Deterministic given inputs — same args → same result, always.
  4. Unit-testable standalone: ``python -c "from centerline_graph import *"``.

Functions that close over pipeline context (``_bfsPathLen``, ``_longestPath``,
``_trunkScore``, ``_branchScore``, etc.) remain as nested defs inside
``loadCenterline``.  Do not move them here without first threading all context
as explicit arguments and writing the corresponding unit tests.

Usage in VesselAnalyzer.py
--------------------------
Replace the nested ``_bfsPath`` definition at line ~13306 with::

    from centerline_graph import graph_bfs_path as _bfsPath

and replace the nested ``_areDuplicateEdge`` definition at line ~13961 with::

    from centerline_graph import graph_are_duplicate_edge as _areDuplicateEdge

The nested ``_count_tree`` and ``_log_tree`` definitions at lines ~14875 /
~14908 can be replaced with::

    from centerline_graph import tree_count_nodes, tree_log_nodes

passing ``MICRO_SEG_MM`` explicitly:
    _ts, _tb, _te = tree_count_nodes(self.branchTree, micro_seg_mm=MICRO_SEG_MM)
    tree_log_nodes(self.branchTree, micro_seg_mm=MICRO_SEG_MM)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────
# Graph nodes are plain ints (VMTK critical-node indices).
# Adjacency maps: Dict[int, Set[int]] (or any iterable of int).
# Tree nodes: the dict produced by _build_tree_node inside loadCenterline.

NodeId = int
Adjacency = Dict[NodeId, Any]   # Any iterable of NodeId
TreeNode = Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# 1. BFS shortest path
# ─────────────────────────────────────────────────────────────────────────────

def graph_bfs_path(
    src: NodeId,
    dst: NodeId,
    adj: Adjacency,
) -> List[NodeId]:
    """Return the shortest undirected path from *src* to *dst* in *adj*.

    Uses BFS so the returned path has the minimum number of hops.  When
    *src == dst* a single-element list ``[src]`` is returned.  When no path
    exists an empty list is returned.

    This is a pure function extracted from the nested ``_bfsPath`` closure
    inside ``loadCenterline`` (VesselAnalyzer v275, line ~13306).  It is
    identical in behaviour; only the name and scope have changed.

    Parameters
    ----------
    src : int   source node id
    dst : int   destination node id
    adj : dict  adjacency map ``{node_id: iterable_of_neighbour_ids}``
                (supports both ``dict[int, set]`` and ``defaultdict``)

    Returns
    -------
    list[int]  node path from *src* to *dst* inclusive, or ``[]`` if none.

    Examples
    --------
    >>> adj = {0: {1, 2}, 1: {0, 3}, 2: {0, 3}, 3: {1, 2}}
    >>> graph_bfs_path(0, 3, adj)
    [0, 1, 3]
    >>> graph_bfs_path(0, 0, adj)
    [0]
    >>> graph_bfs_path(0, 99, adj)
    []
    """
    if src == dst:
        return [src]

    visited: Dict[NodeId, Optional[NodeId]] = {src: None}
    queue: List[NodeId] = [src]

    while queue:
        nq: List[NodeId] = []
        for n in queue:
            for nb in adj.get(n, set()):
                if nb not in visited:
                    visited[nb] = n
                    if nb == dst:
                        # Reconstruct path by walking parent pointers.
                        path: List[NodeId] = [dst]
                        cur: NodeId = dst
                        while visited[cur] is not None:
                            cur = visited[cur]  # type: ignore[assignment]
                            path.append(cur)
                        return list(reversed(path))
                    nq.append(nb)
        queue = nq

    return []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Duplicate-edge detector
# ─────────────────────────────────────────────────────────────────────────────

_DEDUP_SNAP_MM: float = 5.0   # endpoint proximity threshold for duplicate detection


def graph_are_duplicate_edge(
    pts_a: List[Any],
    pts_b: List[Any],
    snap_mm: float = _DEDUP_SNAP_MM,
) -> bool:
    """Return ``True`` when *pts_a* and *pts_b* represent the same graph edge.

    Two edge polylines are considered duplicates when:

    * Their lengths differ by at most ``max(5, len(pts_a) // 5)`` points
      (allows for minor resampling differences), AND
    * Their endpoints match within *snap_mm* in either forward or reversed
      orientation.

    This is a pure function extracted from the nested ``_areDuplicateEdge``
    closure inside ``loadCenterline`` (VesselAnalyzer v275, line ~13961).
    It is identical in behaviour; only the name and scope have changed.

    Parameters
    ----------
    pts_a, pts_b : list of 3-tuples / array-like   polyline point sequences
    snap_mm      : float  endpoint distance threshold (default 5.0 mm)

    Returns
    -------
    bool

    Examples
    --------
    >>> a = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    >>> b = [(0.1, 0, 0), (1, 0, 0), (2.1, 0, 0)]
    >>> graph_are_duplicate_edge(a, b)
    True
    >>> c = [(10, 0, 0), (11, 0, 0), (12, 0, 0)]
    >>> graph_are_duplicate_edge(a, c)
    False
    """
    if abs(len(pts_a) - len(pts_b)) > max(5, len(pts_a) // 5):
        return False

    def _d3(p: Any, q: Any) -> float:
        return math.sqrt(
            (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
        )

    same_fwd = _d3(pts_a[0], pts_b[0]) < snap_mm and _d3(pts_a[-1], pts_b[-1]) < snap_mm
    same_rev = _d3(pts_a[0], pts_b[-1]) < snap_mm and _d3(pts_a[-1], pts_b[0]) < snap_mm
    return same_fwd or same_rev


def graph_dedup_edges(
    branches: List[List[Any]],
    snap_mm: float = _DEDUP_SNAP_MM,
) -> Tuple[List[List[Any]], int]:
    """Remove duplicate edges from *branches* and return ``(deduped, n_removed)``.

    Convenience wrapper that applies :func:`graph_are_duplicate_edge` to an
    entire branch list in O(N²) — acceptable for the typical graph sizes
    produced by VMTK (< 50 branches).

    Parameters
    ----------
    branches : list of polyline point-lists
    snap_mm  : float  forwarded to :func:`graph_are_duplicate_edge`

    Returns
    -------
    (deduped_branches, n_removed) : (list, int)
    """
    deduped: List[List[Any]] = []
    for pts in branches:
        if not any(graph_are_duplicate_edge(pts, seen, snap_mm) for seen in deduped):
            deduped.append(pts)
    return deduped, len(branches) - len(deduped)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Branch-tree summary helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MICRO_SEG_MM: float = 10.0  # topology-artifact segment length threshold


def tree_count_nodes(
    node: TreeNode,
    micro_seg_mm: float = _DEFAULT_MICRO_SEG_MM,
) -> Tuple[int, int, int]:
    """Recursively count segments, bifurcations, and endpoints in *node*.

    A node is counted as a **bifurcation** only when its connecting segment
    has ``arc_mm >= micro_seg_mm``; sub-threshold segments are graph-compression
    artifacts and should not inflate the anatomical bifurcation count.

    This is a pure function extracted from the nested ``_count_tree`` closure
    inside ``loadCenterline`` (VesselAnalyzer v275, line ~14875).

    Parameters
    ----------
    node         : TreeNode  root of the branch tree (as built by _build_tree_node)
    micro_seg_mm : float     artifact threshold (default 10.0 mm)

    Returns
    -------
    (n_segments, n_bifurcations, n_endpoints) : (int, int, int)
    """
    segs: int = 0 if node["segment"] is None else 1
    seg_arc: float = node["segment"]["arc_mm"] if node["segment"] else 0.0
    bifs: int = (
        1 if node["type"] == "bifurcation" and seg_arc >= micro_seg_mm else 0
    )
    eps: int = 1 if node["type"] == "endpoint" else 0

    for ch in node["children"]:
        s, b, e = tree_count_nodes(ch, micro_seg_mm)
        segs += s
        bifs += b
        eps += e

    return segs, bifs, eps


def tree_log_nodes(
    node: TreeNode,
    prefix: str = "",
    micro_seg_mm: float = _DEFAULT_MICRO_SEG_MM,
) -> None:
    """Print a visual representation of the branch tree to stdout.

    Output is identical to the nested ``_log_tree`` closure that appeared at
    VesselAnalyzer v275 line ~14908.

    Parameters
    ----------
    node         : TreeNode  current node (start with tree root)
    prefix       : str       indentation string (caller should pass ``""``
                             for the root and the function recurses with
                             ``prefix + "  "``)
    micro_seg_mm : float     artifact threshold used to add ``[micro]`` tag
    """
    seg = node["segment"]
    if seg:
        label = f"seg {seg['arc_mm']:.1f}mm [{seg.get('role', '?')}]"
    else:
        label = "ROOT"

    ntype: str = node["type"]
    pb_tag: str = " ★PRIMARY_BIF" if node.get("is_primary_bif") else ""
    micro_tag: str = (
        " [micro]"
        if seg and seg["arc_mm"] < micro_seg_mm and ntype == "bifurcation"
        else ""
    )

    print(
        f"[BranchTree]   {prefix}node {node['node_id']} "
        f"({ntype}{pb_tag}{micro_tag}) ← {label}"
    )

    for ch in node["children"]:
        tree_log_nodes(ch, prefix + "  ", micro_seg_mm)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Legacy aliases  (zero migration cost for existing call-sites)
# ─────────────────────────────────────────────────────────────────────────────
# Drop-in replacements: swap the nested def for these imports in loadCenterline.

_bfsPath           = graph_bfs_path            # was nested def at line ~13306
_areDuplicateEdge  = graph_are_duplicate_edge  # was nested def at line ~13961
_count_tree        = tree_count_nodes          # was nested def at line ~14875
_log_tree          = tree_log_nodes            # was nested def at line ~14908


# ── Slicer module-scanner guard ───────────────────────────────────────────────
# Slicer auto-scans all .py files in the module folder and expects a class
# matching the filename.  This stub satisfies that requirement without
# registering as an actual loadable module (no ScriptedLoadableModule base).
class centerline_graph:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "centerline_graph"
            parent.hidden = True

