"""
test_vessel_pure.py — Unit tests for VesselAnalyzer's Slicer-free helper modules.

Run with:
    python -m pytest test_vessel_pure.py -v
or:
    python test_vessel_pure.py

Covers:
    centerline_graph   — graph_bfs_path, graph_are_duplicate_edge,
                         graph_dedup_edges, tree_count_nodes, tree_log_nodes
    ostium_logreg      — OstiumLogReg: classify_branch, build_features,
                         score, train_step, save/_load, weight clamping
    vessel_geometry    — geo_clamp, geo_unit, geo_angle_deg, geo_dist,
                         geo_arc_length, renal_anatomy_gate,
                         renal_composite_score
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import unittest

# ── Make sure sibling modules are importable when running from the project dir ──
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from centerline_graph import (
    graph_bfs_path,
    graph_are_duplicate_edge,
    graph_dedup_edges,
    tree_count_nodes,
    tree_log_nodes,
)
from ostium_logreg import OstiumLogReg
from vessel_geometry import (
    geo_clamp,
    geo_unit,
    geo_angle_deg,
    geo_dist,
    geo_arc_length,
)

try:
    from vessel_geometry import renal_anatomy_gate, renal_composite_score
    _HAS_RENAL = True
except ImportError:
    _HAS_RENAL = False


# ═══════════════════════════════════════════════════════════════════════════════
# centerline_graph tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphBfsPath(unittest.TestCase):

    def _adj(self):
        """Simple diamond graph:  0─1─3
                                   └─2─┘  """
        return {0: {1, 2}, 1: {0, 3}, 2: {0, 3}, 3: {1, 2}}

    # ── Basic traversal ──────────────────────────────────────────────────

    def test_self_loop(self):
        self.assertEqual(graph_bfs_path(0, 0, self._adj()), [0])

    def test_direct_neighbour(self):
        path = graph_bfs_path(0, 1, self._adj())
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 1)
        self.assertEqual(len(path), 2)

    def test_shortest_path_length(self):
        """BFS must find the 2-hop path (0→1→3 or 0→2→3), not a longer one."""
        path = graph_bfs_path(0, 3, self._adj())
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 3)

    def test_path_is_connected(self):
        """Every consecutive pair in the returned path must be neighbours."""
        adj = self._adj()
        path = graph_bfs_path(0, 3, adj)
        for a, b in zip(path, path[1:]):
            self.assertIn(b, adj[a], f"Edge {a}-{b} does not exist in adj")

    # ── Disconnected / missing nodes ─────────────────────────────────────

    def test_unreachable_node(self):
        self.assertEqual(graph_bfs_path(0, 99, self._adj()), [])

    def test_empty_adj(self):
        self.assertEqual(graph_bfs_path(0, 1, {}), [])

    def test_isolated_node(self):
        adj = {0: set(), 1: set()}
        self.assertEqual(graph_bfs_path(0, 1, adj), [])

    # ── Linear chain ─────────────────────────────────────────────────────

    def test_linear_chain(self):
        adj = {i: {i - 1, i + 1} for i in range(1, 9)}
        adj[0] = {1}
        adj[9] = {8}
        path = graph_bfs_path(0, 9, adj)
        self.assertEqual(path, list(range(10)))

    # ── Result is list, not other iterable ──────────────────────────────

    def test_returns_list(self):
        self.assertIsInstance(graph_bfs_path(0, 3, self._adj()), list)

    # ── Adjacency with list values (not just sets) ───────────────────────

    def test_list_adjacency(self):
        adj = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
        path = graph_bfs_path(0, 3, adj)
        self.assertEqual(len(path), 3)


class TestGraphAreDuplicateEdge(unittest.TestCase):

    def _line(self, n=5, start=0):
        """n collinear points along X starting at `start`."""
        return [(start + i, 0.0, 0.0) for i in range(n)]

    # ── Exact duplicate ──────────────────────────────────────────────────

    def test_identical_edges(self):
        a = self._line(5)
        self.assertTrue(graph_are_duplicate_edge(a, a))

    def test_slightly_offset_same_dir(self):
        a = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        b = [(0.1, 0, 0), (1, 0, 0), (2.1, 0, 0)]
        self.assertTrue(graph_are_duplicate_edge(a, b))

    def test_reversed_duplicate(self):
        a = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        b = list(reversed(a))
        self.assertTrue(graph_are_duplicate_edge(a, b))

    # ── Not duplicates ───────────────────────────────────────────────────

    def test_far_apart_edges(self):
        a = self._line(5, start=0)
        b = self._line(5, start=100)
        self.assertFalse(graph_are_duplicate_edge(a, b))

    def test_very_different_lengths(self):
        a = self._line(5)
        b = self._line(50)
        self.assertFalse(graph_are_duplicate_edge(a, b))

    def test_same_start_different_end(self):
        a = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        c = [(0, 0, 0), (1, 0, 0), (100, 0, 0)]
        self.assertFalse(graph_are_duplicate_edge(a, c))

    # ── snap_mm sensitivity ──────────────────────────────────────────────

    def test_just_within_snap(self):
        a = [(0, 0, 0), (5, 0, 0)]
        b = [(4.9, 0, 0), (5, 0, 0)]   # start 4.9mm away < default 5mm
        self.assertTrue(graph_are_duplicate_edge(a, b, snap_mm=5.0))

    def test_just_outside_snap(self):
        a = [(0, 0, 0), (10, 0, 0)]
        b = [(6, 0, 0), (10, 0, 0)]    # start 6mm away > snap=5mm
        self.assertFalse(graph_are_duplicate_edge(a, b, snap_mm=5.0))


class TestGraphDedupEdges(unittest.TestCase):

    def test_no_duplicates(self):
        branches = [
            [(0, 0, 0), (1, 0, 0)],
            [(10, 0, 0), (11, 0, 0)],
        ]
        deduped, n_removed = graph_dedup_edges(branches)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(n_removed, 0)

    def test_removes_one_duplicate(self):
        a = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        branches = [a, list(reversed(a)), [(10, 0, 0), (11, 0, 0)]]
        deduped, n_removed = graph_dedup_edges(branches)
        self.assertEqual(n_removed, 1)
        self.assertEqual(len(deduped), 2)

    def test_all_duplicates(self):
        a = [(0, 0, 0), (1, 0, 0)]
        branches = [a, a, a]
        deduped, n_removed = graph_dedup_edges(branches)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(n_removed, 2)

    def test_empty_input(self):
        deduped, n_removed = graph_dedup_edges([])
        self.assertEqual(deduped, [])
        self.assertEqual(n_removed, 0)


class TestTreeCountNodes(unittest.TestCase):

    def _leaf(self, arc_mm):
        return {
            "node_id": 1,
            "type": "endpoint",
            "is_primary_bif": False,
            "segment": {"arc_mm": arc_mm, "role": "trunk"},
            "children": [],
        }

    def _bif(self, arc_mm, children):
        return {
            "node_id": 2,
            "type": "bifurcation",
            "is_primary_bif": False,
            "segment": {"arc_mm": arc_mm, "role": "trunk"},
            "children": children,
        }

    def _root(self, children):
        return {
            "node_id": 0,
            "type": "root",
            "is_primary_bif": False,
            "segment": None,
            "children": children,
        }

    # ── Trivial trees ────────────────────────────────────────────────────

    def test_single_leaf(self):
        node = self._leaf(20.0)
        segs, bifs, eps = tree_count_nodes(node)
        self.assertEqual(segs, 1)
        self.assertEqual(bifs, 0)
        self.assertEqual(eps, 1)

    def test_root_no_segment(self):
        node = self._root([])
        segs, bifs, eps = tree_count_nodes(node)
        self.assertEqual(segs, 0)
        self.assertEqual(bifs, 0)
        self.assertEqual(eps, 0)

    # ── Bifurcation counting ─────────────────────────────────────────────

    def test_bif_above_threshold_counted(self):
        leaf1 = self._leaf(20.0)
        leaf2 = self._leaf(20.0)
        bif = self._bif(arc_mm=15.0, children=[leaf1, leaf2])
        segs, bifs, eps = tree_count_nodes(bif, micro_seg_mm=10.0)
        self.assertEqual(bifs, 1)
        self.assertEqual(eps, 2)

    def test_micro_bif_not_counted(self):
        """A bifurcation node whose segment is < micro_seg_mm must not be counted."""
        leaf1 = self._leaf(20.0)
        leaf2 = self._leaf(20.0)
        bif = self._bif(arc_mm=3.0, children=[leaf1, leaf2])   # 3mm < default 10mm
        segs, bifs, eps = tree_count_nodes(bif, micro_seg_mm=10.0)
        self.assertEqual(bifs, 0)

    def test_two_bifurcations(self):
        """Y with one branch itself bifurcating → 2 anatomical bifs."""
        e1 = self._leaf(20.0)
        e2 = self._leaf(20.0)
        e3 = self._leaf(20.0)
        inner_bif = self._bif(arc_mm=15.0, children=[e2, e3])
        outer_bif = self._bif(arc_mm=20.0, children=[e1, inner_bif])
        root = self._root([outer_bif])
        segs, bifs, eps = tree_count_nodes(root, micro_seg_mm=10.0)
        self.assertEqual(bifs, 2)
        self.assertEqual(eps, 3)

    # ── Segment counting ─────────────────────────────────────────────────

    def test_segment_count(self):
        """3-branch tree (root→bif, bif→leaf×2) → 3 segments."""
        leaf1 = self._leaf(20.0)
        leaf2 = self._leaf(20.0)
        bif = self._bif(arc_mm=20.0, children=[leaf1, leaf2])
        root = self._root([bif])
        segs, _, _ = tree_count_nodes(root)
        self.assertEqual(segs, 3)


class TestTreeLogNodes(unittest.TestCase):

    def _simple_tree(self):
        leaf = {
            "node_id": 1,
            "type": "endpoint",
            "is_primary_bif": False,
            "segment": {"arc_mm": 30.0, "role": "trunk"},
            "children": [],
        }
        return {
            "node_id": 0,
            "type": "root",
            "is_primary_bif": False,
            "segment": None,
            "children": [leaf],
        }

    def test_produces_output(self):
        """tree_log_nodes must write at least one line to stdout."""
        captured = io.StringIO()
        sys.stdout = captured
        try:
            tree_log_nodes(self._simple_tree())
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("[BranchTree]", output)

    def test_micro_tag_present(self):
        """Bifurcation node with arc < micro_seg_mm must carry [micro] tag."""
        micro_bif = {
            "node_id": 2,
            "type": "bifurcation",
            "is_primary_bif": False,
            "segment": {"arc_mm": 3.0, "role": "trunk"},
            "children": [],
        }
        captured = io.StringIO()
        sys.stdout = captured
        try:
            tree_log_nodes(micro_bif, micro_seg_mm=10.0)
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("[micro]", captured.getvalue())

    def test_primary_bif_star(self):
        """Primary bifurcation node must carry the ★PRIMARY_BIF marker."""
        node = {
            "node_id": 5,
            "type": "bifurcation",
            "is_primary_bif": True,
            "segment": {"arc_mm": 20.0, "role": "trunk"},
            "children": [],
        }
        captured = io.StringIO()
        sys.stdout = captured
        try:
            tree_log_nodes(node)
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("★PRIMARY_BIF", captured.getvalue())


# ═══════════════════════════════════════════════════════════════════════════════
# OstiumLogReg tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOstiumLogRegClassifyBranch(unittest.TestCase):

    def setUp(self):
        self.lr = OstiumLogReg()

    def test_iliac_roles(self):
        for role in ("main", "iliac_left", "iliac_right"):
            self.assertEqual(self.lr.classify_branch(role, 100.0), "main")

    def test_renal_roles(self):
        for role in ("renal_vein", "renal_fragment"):
            self.assertEqual(self.lr.classify_branch(role, 100.0), "renal")

    def test_side_short(self):
        self.assertEqual(self.lr.classify_branch("side_branch", 20.0), "side_short")

    def test_side_long(self):
        self.assertEqual(self.lr.classify_branch("side_branch", 50.0), "side")

    def test_arc_none_defaults_to_side(self):
        self.assertEqual(self.lr.classify_branch("collateral", None), "side")

    def test_arc_exactly_30_is_side(self):
        """arc_mm=30 is NOT < 30, so it must be 'side' not 'side_short'."""
        self.assertEqual(self.lr.classify_branch("side_branch", 30.0), "side")


class TestOstiumLogRegBuildFeatures(unittest.TestCase):

    def setUp(self):
        self.lr = OstiumLogReg()

    def test_keys_present(self):
        feats = self.lr.build_features(
            {k: 0.5 for k in "VALTGSCZD"},
            [],
        )
        for k in OstiumLogReg.FEATURE_KEYS:
            self.assertIn(k, feats, f"Key '{k}' missing from build_features output")

    def test_interactions_computed(self):
        feats = self.lr.build_features({"A": 0.8, "D": 0.5, "V": 0.6, "G": 0.4}, [])
        self.assertAlmostEqual(feats["AD"], 0.8 * 0.5)
        self.assertAlmostEqual(feats["VG"], 0.6 * 0.4)

    def test_flag_area_low(self):
        feats = self.lr.build_features({}, ["area_low"])
        self.assertEqual(feats["fl_area_low"], 1.0)
        self.assertEqual(feats["fl_zone_distant"], 0.0)

    def test_flag_zone_distant(self):
        feats = self.lr.build_features({}, ["zone_distant"])
        self.assertEqual(feats["fl_zone_distant"], 1.0)

    def test_flag_divergence_low(self):
        feats = self.lr.build_features({}, ["divergence_low"])
        self.assertEqual(feats["fl_divergence_low"], 1.0)

    def test_missing_components_default_zero(self):
        feats = self.lr.build_features({}, [])
        for k in "VALTGSCZD":
            self.assertEqual(feats[k], 0.0)


class TestOstiumLogRegScore(unittest.TestCase):

    def setUp(self):
        self.lr = OstiumLogReg()

    def _feats(self, val):
        return {k: val for k in OstiumLogReg.FEATURE_KEYS}

    def test_output_range_all_zero(self):
        s = self.lr.score("main", self._feats(0.0))
        self.assertGreaterEqual(s, 0.15)
        self.assertLessEqual(s, 0.85)

    def test_output_range_all_one(self):
        s = self.lr.score("main", self._feats(1.0))
        self.assertGreaterEqual(s, 0.15)
        self.assertLessEqual(s, 0.85)

    def test_higher_features_give_higher_score(self):
        low  = self.lr.score("side", self._feats(0.1))
        high = self.lr.score("side", self._feats(0.9))
        self.assertGreater(high, low)

    def test_all_branch_types_score(self):
        feats = self._feats(0.5)
        for btype in OstiumLogReg.BRANCH_TYPES:
            s = self.lr.score(btype, feats)
            self.assertGreaterEqual(s, 0.15)
            self.assertLessEqual(s, 0.85)

    def test_unknown_branch_type_falls_back(self):
        """Unknown branch type should not crash — falls back to 'side' weights."""
        s = self.lr.score("invented_type", self._feats(0.5))
        self.assertGreaterEqual(s, 0.15)
        self.assertLessEqual(s, 0.85)


class TestOstiumLogRegTrainStep(unittest.TestCase):

    def setUp(self):
        self.lr = OstiumLogReg()

    def _record(self, y):
        feats = {k: 0.7 for k in OstiumLogReg.FEATURE_KEYS}
        return {"features": feats, "y": y}

    def test_no_train_below_min(self):
        """train_step must do nothing when fewer than MIN_TRAIN_N records provided."""
        before = list(self.lr._weights["side"]["w"])
        self.lr.train_step([self._record(1.0)] * 2, "side")  # 2 < 6
        self.assertEqual(self.lr._weights["side"]["w"], before)

    def test_weights_change_after_training(self):
        """After a full training pass the weights must differ from priors."""
        records = [self._record(1.0)] * 10
        before = list(self.lr._weights["side"]["w"])
        self.lr.train_step(records, "side")
        after = self.lr._weights["side"]["w"]
        self.assertNotEqual(before, after)

    def test_weight_clamp_respected(self):
        """No weight component should exceed W_CLAMP after many extreme updates."""
        records = [self._record(1.0)] * 100
        for _ in range(50):
            self.lr.train_step(records, "renal")
        for w in self.lr._weights["renal"]["w"]:
            self.assertLessEqual(abs(w), OstiumLogReg.W_CLAMP + 1e-9)

    def test_bias_clamp_respected(self):
        records = [self._record(0.0)] * 100
        for _ in range(50):
            self.lr.train_step(records, "main")
        bias = self.lr._weights["main"]["bias"]
        self.assertLessEqual(abs(bias), OstiumLogReg.BIAS_CLAMP + 1e-9)

    def test_convergence_direction(self):
        """Training on all-positive labels should push the score upward."""
        feats = {k: 0.5 for k in OstiumLogReg.FEATURE_KEYS}
        s_before = self.lr.score("side", feats)
        records = [{"features": feats, "y": 1.0}] * 30
        self.lr.train_step(records, "side")
        s_after = self.lr.score("side", feats)
        self.assertGreater(s_after, s_before)

    def test_negative_label_pushes_score_down(self):
        feats = {k: 0.5 for k in OstiumLogReg.FEATURE_KEYS}
        s_before = self.lr.score("side", feats)
        records = [{"features": feats, "y": 0.0}] * 30
        self.lr.train_step(records, "side")
        s_after = self.lr.score("side", feats)
        self.assertLess(s_after, s_before)


class TestOstiumLogRegSaveLoad(unittest.TestCase):

    def test_round_trip(self):
        lr = OstiumLogReg()
        # Modify a weight so we can detect it after reload
        lr._weights["main"]["w"][0] = 3.14
        lr._weights["renal"]["bias"] = -1.23

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            lr.save(path)
            lr2 = OstiumLogReg(weights_path=path)
            self.assertAlmostEqual(lr2._weights["main"]["w"][0], 3.14, places=5)
            self.assertAlmostEqual(lr2._weights["renal"]["bias"], -1.23, places=5)
        finally:
            os.unlink(path)

    def test_missing_file_uses_priors(self):
        """Loading a non-existent weights file must silently fall back to priors."""
        lr = OstiumLogReg(weights_path="/nonexistent/path/weights.json")
        # Priors must still be intact
        for btype in OstiumLogReg.BRANCH_TYPES:
            self.assertIn(btype, lr._weights)

    def test_old_9_feature_file_extended(self):
        """A weights file with only 9 features (old format) must be extended
        with prior values for the 5 new features without crashing."""
        old_weights = {
            "side": {
                "bias": -1.5,
                "w": [0.1] * 9,   # old 9-feature format
            }
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(old_weights, f)
            path = f.name
        try:
            lr = OstiumLogReg(weights_path=path)
            self.assertEqual(len(lr._weights["side"]["w"]), len(OstiumLogReg.FEATURE_KEYS))
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# vessel_geometry tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeoClamp(unittest.TestCase):

    def test_below_range(self):
        self.assertEqual(geo_clamp(-5.0, 0.0, 1.0), 0.0)

    def test_above_range(self):
        self.assertEqual(geo_clamp(2.0, 0.0, 1.0), 1.0)

    def test_within_range(self):
        self.assertAlmostEqual(geo_clamp(0.5, 0.0, 1.0), 0.5)

    def test_at_boundary_low(self):
        self.assertEqual(geo_clamp(0.0, 0.0, 1.0), 0.0)

    def test_at_boundary_high(self):
        self.assertEqual(geo_clamp(1.0, 0.0, 1.0), 1.0)

    def test_negative_range(self):
        self.assertEqual(geo_clamp(-3.0, -5.0, -1.0), -3.0)


class TestGeoUnit(unittest.TestCase):

    def test_already_unit_x(self):
        v = geo_unit([1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(v[0]), 1.0)
        self.assertAlmostEqual(float(v[1]), 0.0)

    def test_normalisation(self):
        v = geo_unit([3.0, 4.0, 0.0])
        mag = math.sqrt(sum(float(x) ** 2 for x in v))
        self.assertAlmostEqual(mag, 1.0, places=6)

    def test_zero_vector_returns_zero(self):
        v = geo_unit([0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(float(x) ** 2 for x in v), 0.0)

    def test_negative_components(self):
        v = geo_unit([-1.0, -1.0, -1.0])
        mag = math.sqrt(sum(float(x) ** 2 for x in v))
        self.assertAlmostEqual(mag, 1.0, places=6)


class TestGeoAngleDeg(unittest.TestCase):

    def test_parallel_zero(self):
        self.assertAlmostEqual(geo_angle_deg([1, 0, 0], [1, 0, 0]), 0.0, places=4)

    def test_antiparallel_180(self):
        self.assertAlmostEqual(geo_angle_deg([1, 0, 0], [-1, 0, 0]), 180.0, places=4)

    def test_perpendicular_90(self):
        self.assertAlmostEqual(geo_angle_deg([1, 0, 0], [0, 1, 0]), 90.0, places=4)

    def test_45_degrees(self):
        v = [1.0, 1.0, 0.0]
        self.assertAlmostEqual(geo_angle_deg([1, 0, 0], v), 45.0, places=4)

    def test_output_in_range(self):
        import random
        rng = random.Random(42)
        for _ in range(50):
            a = [rng.gauss(0, 1) for _ in range(3)]
            b = [rng.gauss(0, 1) for _ in range(3)]
            if all(x == 0 for x in a) or all(x == 0 for x in b):
                continue
            ang = geo_angle_deg(a, b)
            self.assertGreaterEqual(ang, 0.0)
            self.assertLessEqual(ang, 180.0)


class TestGeoDist(unittest.TestCase):

    def test_origin_to_unit_x(self):
        self.assertAlmostEqual(geo_dist([0, 0, 0], [1, 0, 0]), 1.0)

    def test_known_distance(self):
        self.assertAlmostEqual(geo_dist([0, 0, 0], [3, 4, 0]), 5.0)

    def test_symmetric(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        self.assertAlmostEqual(geo_dist(a, b), geo_dist(b, a))

    def test_self_distance_zero(self):
        self.assertAlmostEqual(geo_dist([7, 8, 9], [7, 8, 9]), 0.0)


class TestGeoArcLength(unittest.TestCase):
    """geo_arc_length returns the TOTAL arc length as a single float."""

    def test_two_points(self):
        total = geo_arc_length([[0, 0, 0], [5, 0, 0]])
        self.assertAlmostEqual(total, 5.0)

    def test_three_collinear_points(self):
        # Two segments of 5mm each → total 10mm
        total = geo_arc_length([[0, 0, 0], [3, 4, 0], [6, 8, 0]])
        self.assertAlmostEqual(total, 10.0)

    def test_length_equals_sum_of_segments(self):
        # (0,0,0)→(1,0,0)=1mm, →(1,1,0)=1mm, →(2,1,0)=1mm → total 3mm
        total = geo_arc_length([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]])
        self.assertAlmostEqual(total, 3.0)

    def test_single_point_zero(self):
        total = geo_arc_length([[1, 2, 3]])
        self.assertAlmostEqual(total, 0.0)

    def test_empty_returns_zero(self):
        total = geo_arc_length([])
        self.assertAlmostEqual(total, 0.0)

    def test_returns_float(self):
        self.assertIsInstance(geo_arc_length([[0, 0, 0], [1, 0, 0]]), float)

    def test_known_pythagorean(self):
        # 3-4-5 right triangle in XY
        total = geo_arc_length([[0, 0, 0], [3, 4, 0]])
        self.assertAlmostEqual(total, 5.0)

    def test_longer_path_larger_arc(self):
        short = geo_arc_length([[0, 0, 0], [1, 0, 0]])
        long_ = geo_arc_length([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        self.assertGreater(long_, short)


@unittest.skipUnless(_HAS_RENAL, "renal_anatomy_gate / renal_composite_score not exported")
class TestRenalGate(unittest.TestCase):
    """renal_anatomy_gate(blen, lat_mm, ostium_z, bif_z) -> (bool, reason_str)
    dz = ostium_z - bif_z must be in [min_dz_mm=20, max_dz_mm=120]."""

    # bif_z=0 so ostium_z == dz
    def _call(self, blen=80.0, lat_mm=45.0, dz=70.0):
        passed, _ = renal_anatomy_gate(blen, lat_mm, ostium_z=dz, bif_z=0.0)
        return passed

    def test_long_lateral_passes(self):
        self.assertTrue(self._call())

    def test_short_arc_fails(self):
        self.assertFalse(self._call(blen=20.0))

    def test_low_lateral_fails(self):
        self.assertFalse(self._call(lat_mm=5.0))

    def test_dz_too_low_fails(self):
        self.assertFalse(self._call(dz=10.0))

    def test_dz_too_high_fails(self):
        self.assertFalse(self._call(dz=150.0))

    def test_returns_tuple(self):
        result = renal_anatomy_gate(80.0, 45.0, ostium_z=70.0, bif_z=0.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_reason_string_on_fail(self):
        passed, reason = renal_anatomy_gate(10.0, 45.0, ostium_z=70.0, bif_z=0.0)
        self.assertFalse(passed)
        self.assertIsInstance(reason, str)
        self.assertGreater(len(reason), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Slicer scripted-module stub
# ═══════════════════════════════════════════════════════════════════════════════
# Slicer auto-discovers every .py file in the module folder and requires a class
# whose name matches the filename exactly.  This stub satisfies that requirement
# without interfering with the unittest logic above.

try:
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule

    class test_vessel_pure(ScriptedLoadableModule):
        """Slicer stub — this file is a pure-Python unittest suite.
        It has no UI and is not intended to be used interactively inside Slicer.
        Run tests from the command line:
            python -m pytest test_vessel_pure.py -v
        """

        def __init__(self, parent):
            ScriptedLoadableModule.__init__(self, parent)
            self.parent.title = "test_vessel_pure (unit tests)"
            self.parent.categories = ["Developer Tools"]
            self.parent.dependencies = []
            self.parent.contributors = ["VesselAnalyzer"]
            self.parent.helpText = (
                "Pure-Python unit tests for VesselAnalyzer helper modules. "
                "Run from the command line with pytest, not from the Slicer GUI."
            )
            self.parent.acknowledgementText = ""

except ImportError:
    # Running outside Slicer (e.g. plain pytest) — stub is not needed.
    pass
