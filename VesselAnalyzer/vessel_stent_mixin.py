"""
vessel_stent_mixin.py — 3-D stent, balloon, kissing-stent, and Y-stent placement; surface classification.

Part of the VesselAnalyzer mixin decomposition.
These methods are mixed into VesselAnalyzerLogic via multiple inheritance:

    class VesselAnalyzerLogic(
        StentMixin,
        ...
        ScriptedLoadableModuleLogic,
    ): ...

All methods use ``self`` normally — no changes to call sites required.
"""
# ruff: noqa  (this file is auto-extracted; formatting is inherited from VesselAnalyzer.py)


import slicer


class StentMixin:
    """Mixin: 3-D stent, balloon, kissing-stent, and Y-stent placement; surface classification."""

    def _expandMeshAtLesion(self, gi_list, r_compressed, r_healthy):
        """Show expanded vessel as a NEW semi-transparent model node.

        Creates a deep copy of the original mesh, deforms the copy using
        vtkPointLocator, and adds it as a separate node — never touching
        the original polydata, which prevents the VTK segfault crash.
        """
        import vtk, math

        try:
            if not self.modelNode:
                return
            orig_pd = self.modelNode.GetPolyData()
            if not orig_pd or orig_pd.GetNumberOfPoints() == 0:
                return

            # Remove ALL previous expanded mesh nodes
            for _n in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
                if (
                    _n.GetName() == "ExpandedVessel"
                    or _n.GetAttribute("VesselAnalyzerExpanded") == "1"
                ):
                    slicer.mrmlScene.RemoveNode(_n)

            # Deep copy — never modify the original
            copy_pd = vtk.vtkPolyData()
            copy_pd.DeepCopy(orig_pd)

            # Resolve path points
            _interp = getattr(self, "_interpPoints", {})
            path_pts = []
            for gi in gi_list:
                if gi < 0:
                    p = _interp.get(gi)
                else:
                    p = self.points[gi] if gi < len(self.points) else None
                if p:
                    path_pts.append(p)
            if not path_pts:
                return

            cl_vtk = vtk.vtkPoints()
            # Centerline points are already in LPS — same space as mesh
            path_pts_lps = list(path_pts)
            for p in path_pts_lps:
                cl_vtk.InsertNextPoint(p[0], p[1], p[2])
            cl_pd = vtk.vtkPolyData()
            cl_pd.SetPoints(cl_vtk)
            locator = vtk.vtkPointLocator()
            locator.SetDataSet(cl_pd)
            locator.BuildLocator()

            mesh_pts = copy_pd.GetPoints()
            n_mesh = mesh_pts.GetNumberOfPoints()
            branch_max_d = max(
                (
                    self.diameters[gi]
                    for gi in gi_list
                    if 0 <= gi < len(self.diameters) and self.diameters[gi] > 0
                ),
                default=r_healthy * 2,
            )
            search_r = branch_max_d / 2.0 * 1.3
            expand_ratio = (
                r_healthy / r_compressed
                if r_compressed > 0 and r_healthy > r_compressed
                else 1.0
            )

            modified = 0
            for vi in range(n_mesh):
                vp = mesh_pts.GetPoint(vi)
                cl_idx = locator.FindClosestPoint(vp)
                cp = path_pts_lps[cl_idx]
                vec = [vp[k] - cp[k] for k in range(3)]
                dist = math.sqrt(sum(x * x for x in vec)) or 1e-6
                if dist > search_r:
                    continue
                unit = [x / dist for x in vec]
                new_dist = min(dist * expand_ratio, r_healthy * 1.05)
                new_pt = [cp[k] + unit[k] * new_dist for k in range(3)]
                mesh_pts.SetPoint(vi, new_pt[0], new_pt[1], new_pt[2])
                modified += 1

            if modified == 0:
                return

            copy_pd.GetPoints().Modified()
            copy_pd.Modified()

            # Create expanded vessel node — start hidden, shown only if needed
            exp_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", "ExpandedVessel"
            )
            exp_node.SetAndObservePolyData(copy_pd)
            exp_node.SetAttribute("VesselAnalyzerExpanded", "1")
            exp_node.CreateDefaultDisplayNodes()
            exp_dn = exp_node.GetDisplayNode()
            exp_dn.SetColor(0.3, 0.8, 1.0)
            exp_dn.SetOpacity(0.65)
            exp_dn.SetBackfaceCulling(0)
            exp_dn.SetVisibility(0)  # hidden — balloon tube is the visual
            exp_dn.SetRepresentation(2)
            print(
                f"[ExpandMesh] expanded copy: {modified} vertices moved, "
                f"r {r_compressed:.2f}→{r_healthy:.2f}mm"
            )

        except Exception as _e:
            import traceback

            print(f"[ExpandMesh] error: {_e}")
            traceback.print_exc()


    def _restoreMesh(self):
        """Remove expanded vessel copy and restore original mesh visibility."""
        try:
            # Remove expanded vessel copy
            for node in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
                if node.GetName() == "ExpandedVessel":
                    slicer.mrmlScene.RemoveNode(node)
            # Restore original vessel
            if self.modelNode:
                orig_dn = self.modelNode.GetDisplayNode()
                if orig_dn:
                    orig_dn.SetVisibility(1)
                    orig_dn.SetOpacity(1.0)
            # Remove ring marker nodes
            for node in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
                if "_Ring" in node.GetName():
                    slicer.mrmlScene.RemoveNode(node)
        except Exception as e:
            import traceback

            traceback.print_exc()


    def _placeBalloon3D(self, gi_list, radius, name="BalloonDilate"):
        """Place a semi-transparent inflated balloon cylinder at the stenosis.
        The balloon is wider than the stent (healthy vessel diam) and shorter,
        representing the angioplasty balloon fully inflated to expand the lesion.
        Color: translucent orange-yellow, distinct from cyan stents.
        """
        import vtk, math

        try:
            pts = self.points
            # Resolve path indices — real indices (>=0) come from self.points;
            # synthetic negative indices come from _interpPoints (gap bridging).
            _interp = getattr(self, "_interpPoints", {})
            resolved_coords = []
            for g in gi_list:
                if g < 0:
                    if g in _interp:
                        resolved_coords.append(_interp[g])
                elif g < len(pts):
                    resolved_coords.append(pts[g])
            if len(resolved_coords) < 2:
                return None

            # ── Build smooth tube with Hermite bridging at branch gaps ──────
            # Large gaps (>8mm) between consecutive path points mean the path
            # crosses a bifurcation where no centerline points exist.
            # We bridge these gaps with a Hermite cubic arc using one-sided
            # tangents (purely from each branch's own direction), producing a
            # smooth anatomical curve through the bifurcation space.
            GAP_THRESH = 8.0  # mm — gaps above this get Hermite bridging
            ARC_PER_MM = 1.5  # interpolated points per mm of gap

            raw_coords = resolved_coords
            n_raw = len(raw_coords)

            def _onesided_tangent(coords, idx, forward):
                """One-sided tangent — never crosses a gap."""
                if forward:
                    i0, i1 = idx, min(len(coords) - 1, idx + 2)
                else:
                    i0, i1 = max(0, idx - 2), idx
                d = [coords[i1][k] - coords[i0][k] for k in range(3)]
                dn = math.sqrt(sum(x * x for x in d)) or 1.0
                return [x / dn for x in d]

            # Build dense path: original points + Hermite arc at each large gap
            dense = [list(raw_coords[0])]
            for i in range(1, n_raw):
                p0 = raw_coords[i - 1]
                p1 = raw_coords[i]
                gap = math.sqrt(sum((p1[k] - p0[k]) ** 2 for k in range(3)))
                if gap > GAP_THRESH:
                    # Exit tangent: look backward along path before gap
                    t0 = _onesided_tangent(raw_coords, i - 1, forward=False)
                    # Entry tangent: look forward along path after gap
                    t1 = _onesided_tangent(raw_coords, i, forward=True)
                    n_steps = max(8, int(gap * ARC_PER_MM))
                    chord = gap
                    for si in range(1, n_steps):
                        t = si / n_steps
                        t2, t3 = t * t, t * t * t
                        h00 = 2 * t3 - 3 * t2 + 1
                        h10 = t3 - 2 * t2 + t
                        h01 = -2 * t3 + 3 * t2
                        h11 = t3 - t2
                        p = [
                            h00 * p0[k]
                            + h10 * chord * t0[k]
                            + h01 * p1[k]
                            + h11 * chord * t1[k]
                            for k in range(3)
                        ]
                        dense.append(p)
                dense.append(list(p1))

            n_dense = len(dense)

            # Arc length of dense path for taper calculation
            arc = [0.0]
            for i in range(1, n_dense):
                d = math.sqrt(
                    sum((dense[i][k] - dense[i - 1][k]) ** 2 for k in range(3))
                )
                arc.append(arc[-1] + d)
            total_len = arc[-1] or 1.0

            # Build polyline and smooth with spline
            cp_r = vtk.vtkPoints()
            cl_r = vtk.vtkCellArray()
            cl_r.InsertNextCell(n_dense)
            for i, p in enumerate(dense):
                cp_r.InsertNextPoint(p[0], p[1], p[2])
                cl_r.InsertCellPoint(i)
            cpd_r = vtk.vtkPolyData()
            cpd_r.SetPoints(cp_r)
            cpd_r.SetLines(cl_r)

            spline = vtk.vtkSplineFilter()
            spline.SetInputData(cpd_r)
            spline.SetNumberOfSubdivisions(max(20, int(total_len)))
            spline.Update()
            cpd_smooth = spline.GetOutput()

            # ── Single smooth angioplasty balloon ────────────────────────
            # Matches image: one elongated transparent teal/cyan balloon with
            # gentle taper at both ends, and a thin guidewire through the centre.
            n_smooth = cpd_smooth.GetNumberOfPoints()

            # ── Balloon body: tapered tube along centreline ───────────────
            bal_pts = vtk.vtkPoints()
            bal_lines = vtk.vtkCellArray()
            bal_radii = vtk.vtkFloatArray()
            bal_radii.SetName("TubeRadius")
            bal_lines.InsertNextCell(n_smooth)
            for si in range(n_smooth):
                pt = cpd_smooth.GetPoint(si)
                bal_pts.InsertNextPoint(pt[0], pt[1], pt[2])
                bal_lines.InsertCellPoint(si)
                # Smooth sine taper: full radius at centre, tapers to 20% at tips
                t = si / max(n_smooth - 1, 1)
                taper = math.sin(math.pi * t)
                bal_radii.InsertNextValue(radius * (0.20 + 0.80 * taper))
            bal_pd = vtk.vtkPolyData()
            bal_pd.SetPoints(bal_pts)
            bal_pd.SetLines(bal_lines)
            bal_pd.GetPointData().AddArray(bal_radii)
            bal_pd.GetPointData().SetActiveScalars("TubeRadius")
            bal_tf = vtk.vtkTubeFilter()
            bal_tf.SetInputData(bal_pd)
            bal_tf.SetNumberOfSides(32)
            bal_tf.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
            bal_tf.CappingOn()
            bal_tf.Update()

            # ── Guidewire: thin tube along same path ─────────────────────
            wire_pts = vtk.vtkPoints()
            wire_lines = vtk.vtkCellArray()
            wire_lines.InsertNextCell(n_smooth)
            for si in range(n_smooth):
                pt = cpd_smooth.GetPoint(si)
                wire_pts.InsertNextPoint(pt[0], pt[1], pt[2])
                wire_lines.InsertCellPoint(si)
            wire_pd = vtk.vtkPolyData()
            wire_pd.SetPoints(wire_pts)
            wire_pd.SetLines(wire_lines)
            wire_tf = vtk.vtkTubeFilter()
            wire_tf.SetInputData(wire_pd)
            wire_tf.SetRadius(radius * 0.06)  # thin guidewire
            wire_tf.SetNumberOfSides(10)
            wire_tf.SetVaryRadiusToVaryRadiusOff()
            wire_tf.CappingOn()
            wire_tf.Update()

            # ── Balloon node (transparent teal) ──────────────────────────
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
            node.SetAndObservePolyData(bal_tf.GetOutput())
            node.CreateDefaultDisplayNodes()
            dn = node.GetDisplayNode()
            dn.SetScalarVisibility(False)
            dn.SetColor(0.20, 0.82, 0.85)  # teal/cyan — matches image
            dn.SetOpacity(0.38)  # transparent so vessel shows through
            dn.SetRepresentation(2)
            dn.SetFrontfaceCulling(0)
            dn.SetBackfaceCulling(0)
            dn.SetAmbient(0.10)
            dn.SetDiffuse(0.40)
            dn.SetSpecular(0.80)
            dn.SetVisibility(1)
            dn.SetVisibility2D(1)

            # ── Guidewire node (dark teal, opaque) ───────────────────────
            wire_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", name + "_wire"
            )
            wire_node.SetAndObservePolyData(wire_tf.GetOutput())
            wire_node.CreateDefaultDisplayNodes()
            wdn = wire_node.GetDisplayNode()
            wdn.SetColor(0.05, 0.35, 0.45)  # dark teal wire
            wdn.SetOpacity(1.0)
            wdn.SetRepresentation(2)
            wdn.SetVisibility(1)

            print(f"[Balloon] {name}: {len(gi_list)}pts r={radius:.1f}mm placed")
            return node
        except Exception as e:
            import traceback

            traceback.print_exc()
            return None


    def _placeKissingStent3D(self, plan):
        """Kissing stent: two stents diverging from a shared proximal trunk path.
        Each stent runs: prox_landing → bifurcation hub → its own branch distal.
        The trunk segment is shared (overlapping), the branch segments diverge.
        Uses _traceCenterlinePath so gap bridging handles any vessel topology.
        """
        import vtk, math

        try:
            bi = plan["branchIdx"]
            prox_gi = plan["proxPt"]
            branches = self.branches
            pts = self.points
            diams = self.diameters
            _interp = getattr(self, "_interpPoints", {})

            # ── Ensure _interpPoints dict exists for bridge points ────────────
            if not hasattr(self, "_interpPoints"):
                self._interpPoints = {}
            _interp = self._interpPoints  # live reference
            print(
                f"[KissingStent] ENTRY: proxDiam={plan.get('proxDiam','?')} "
                f"distPt={plan.get('distPt','?')} distPt_B={plan.get('distPt_B','?')} "
                f"bi={bi} sibling={plan.get('siblingIdx','?')}"
            )

            # If branchIdx is -1 or invalid, find the longest downstream branch
            if bi < 0 or bi >= len(branches):
                best_len = 0.0
                bi = 1  # default to branch 1
                for _bj, (_bjs, _bje) in enumerate(branches):
                    if _bj == 0:
                        continue
                    _blen = sum(
                        math.sqrt(
                            sum(
                                (pts[_bjs + i + 1][k] - pts[_bjs + i][k]) ** 2
                                for k in range(3)
                            )
                        )
                        for i in range(min(_bje - _bjs - 1, 200))
                        if _bjs + i + 1 < len(pts)
                    )
                    if _blen > best_len:
                        best_len = _blen
                        bi = _bj
                print(
                    f"[KissingStent] branchIdx was invalid, resolved to bi={bi} ({best_len:.0f}mm)"
                )

            def _resolve(gi):
                if gi < 0:
                    return _interp.get(gi)
                return pts[gi] if gi < len(pts) else None

            # ── Hub = trunk end (bifurcation point) ──────────────────────────
            hub_gi = branches[0][1] - 1
            trunk_bs = branches[0][0]

            # prox_gi must be IN the trunk and strictly before hub.
            # If caller passed hub itself or anything downstream, walk back 20mm.
            # Do NOT re-walk if prox_gi is already a valid trunk point — that
            # would double the trunk length and place the stent above the hub.
            if prox_gi >= hub_gi:
                acc_k = 0.0
                prox_gi = hub_gi
                for _ti in range(hub_gi - 1, trunk_bs - 1, -1):
                    if _ti + 1 < len(pts):
                        _p0, _p1 = pts[_ti], pts[_ti + 1]
                        acc_k += (
                            (_p1[0] - _p0[0]) ** 2
                            + (_p1[1] - _p0[1]) ** 2
                            + (_p1[2] - _p0[2]) ** 2
                        ) ** 0.5
                    prox_gi = _ti
                    if acc_k >= 20.0:
                        break
                print(
                    f"[KissingStent] prox_gi was at/past hub — walked back to gi{prox_gi} ({acc_k:.1f}mm)"
                )

            # ── Find sibling branch ───────────────────────────────────────────
            # Use plan["siblingIdx"] when the caller already resolved the sibling
            # (e.g. _finishStentPickY passes bi_right directly).  Only run the
            # expensive distance-based detection when siblingIdx is absent.
            bs_A, be_A = branches[bi]
            sibling_bi = plan.get("siblingIdx", None)
            if sibling_bi is not None and (
                sibling_bi <= 0 or sibling_bi >= len(branches)
            ):
                sibling_bi = None  # invalid — fall through to auto-detect
            if sibling_bi is None:
                sibling_dist = float("inf")
                for bj, (bsj, bej) in enumerate(branches):
                    if bj == bi or bj == 0:
                        continue
                    # Compute full branch length (up to 200 pts for accuracy)
                    bj_mm = sum(
                        math.sqrt(
                            sum(
                                (pts[bsj + i + 1][k] - pts[bsj + i][k]) ** 2
                                for k in range(3)
                            )
                        )
                        for i in range(min(bej - bsj - 1, 200))
                        if bsj + i + 1 < len(pts)
                    )
                    if bj_mm < 150.0:
                        continue  # skip renal veins and short side branches
                    d = math.sqrt(
                        sum((pts[bsj][k] - pts[bs_A][k]) ** 2 for k in range(3))
                    )
                    if d < sibling_dist:
                        sibling_bi = bj
                        sibling_dist = d
                # Fallback to ≥80mm if no ≥150mm branch found
                if sibling_bi is None:
                    for bj, (bsj, bej) in enumerate(branches):
                        if bj == bi or bj == 0:
                            continue
                        bj_mm = sum(
                            math.sqrt(
                                sum(
                                    (pts[bsj + i + 1][k] - pts[bsj + i][k]) ** 2
                                    for k in range(3)
                                )
                            )
                            for i in range(min(bej - bsj - 1, 60))
                            if bsj + i + 1 < len(pts)
                        )
                        if bj_mm < 80.0:
                            continue
                        d = math.sqrt(
                            sum((pts[bsj][k] - pts[bs_A][k]) ** 2 for k in range(3))
                        )
                        if d < sibling_dist:
                            sibling_bi = bj
                            sibling_dist = d
                # Last resort
                if sibling_bi is None:
                    for bj, (bsj, bej) in enumerate(branches):
                        if bj == bi or bj == 0:
                            continue
                        d = math.sqrt(
                            sum((pts[bsj][k] - pts[bs_A][k]) ** 2 for k in range(3))
                        )
                        if d < sibling_dist:
                            sibling_bi = bj
                            sibling_dist = d
            if sibling_bi is None:
                print("[KissingStent] No sibling branch found")
                return False
            try:
                _sdist_str = f"{sibling_dist:.1f}mm"
            except NameError:
                _sdist_str = "from-plan"
            print(f"[KissingStent] bi={bi} sibling=bi{sibling_bi} dist={_sdist_str}")
            bs_B, be_B = branches[sibling_bi]

            # ── Trace trunk segment ───────────────────────────────────────
            trunk_seg = list(range(prox_gi, hub_gi + 1))

            # ── Distal points ─────────────────────────────────────────────
            # Use user-picked distal points from plan when available.
            # Fall back to VBX 59mm walk only when plan has no distPt.
            user_dist_A = plan.get("distPt", None)
            user_dist_B = plan.get("distPt_B", None)

            VBX_LENGTHS = [15, 19, 29, 39, 59, 79]
            trunk_mm_val = sum(
                math.sqrt(
                    sum(
                        (pts[trunk_seg[i + 1]][k] - pts[trunk_seg[i]][k]) ** 2
                        for k in range(3)
                    )
                )
                for i in range(len(trunk_seg) - 1)
                if trunk_seg[i] < len(pts) and trunk_seg[i + 1] < len(pts)
            )
            # Branch target: use actual budget remaining after trunk, not hardcoded 59mm
            VBX_HARD_MAX_LEN = 79.0
            target_total = VBX_HARD_MAX_LEN
            branch_target_mm = max(10.0, target_total - trunk_mm_val)

            def walk_branch_mm(bs, be, target):
                """Walk branch pts until target mm reached, return end gi.
                Stops before the step that would overshoot so total never exceeds target.
                """
                acc = 0.0
                gi_end = bs
                for _bi in range(bs, be - 1):
                    if _bi + 1 < len(pts):
                        step = math.sqrt(
                            sum((pts[_bi + 1][k] - pts[_bi][k]) ** 2 for k in range(3))
                        )
                        if acc + step > target:
                            break
                        acc += step
                    gi_end = _bi + 1
                return min(gi_end, be - 1)

            # Helper: does a pancaking finding exist on branch bi at/before gi?
            def _has_pancaking(branch_bi, gi_end):
                return any(
                    f.get("branchIdx") == branch_bi
                    and "Pancak" in f.get("type", "")
                    and branches[branch_bi][0] <= f["pointIdx"] <= gi_end
                    for f in getattr(self, "findings", [])
                )

            # Branch A (primary iliac, bi)
            # Rule: honour user pick if it covers a pancaking lesion, even past budget.
            # Only cap to budget when the pick is arbitrary (no lesion to cover).
            VBX_HARD_MAX = 79.0
            branch_budget_mm = max(10.0, VBX_HARD_MAX - trunk_mm_val)

            if user_dist_A is not None and bs_A <= user_dist_A < be_A:
                _pick_A_mm = sum(
                    math.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
                    for i in range(bs_A, min(user_dist_A, len(pts) - 1))
                    if i + 1 < len(pts)
                )
                if _pick_A_mm <= branch_budget_mm:
                    dist_A_gi = user_dist_A
                    print(
                        f"[KissingStent] Branch A: user pick gi{dist_A_gi} ({_pick_A_mm:.0f}mm ≤ budget)"
                    )
                elif _has_pancaking(bi, user_dist_A):
                    # Pick exceeds budget but covers a pancaking lesion — honour it
                    dist_A_gi = user_dist_A
                    print(
                        f"[KissingStent] Branch A: pancaking pick gi{dist_A_gi} "
                        f"({_pick_A_mm:.0f}mm > budget {branch_budget_mm:.0f}mm — lesion coverage)"
                    )
                else:
                    dist_A_gi = walk_branch_mm(bs_A, be_A, branch_budget_mm)
                    print(
                        f"[KissingStent] Branch A: user pick {_pick_A_mm:.0f}mm > budget "
                        f"→ capped to gi{dist_A_gi}"
                    )
            else:
                dist_A_gi = walk_branch_mm(bs_A, be_A, branch_target_mm)
                print(f"[KissingStent] Branch A: VBX walk → gi{dist_A_gi}")

            # Branch B (sibling iliac) — same logic
            if user_dist_B is not None and bs_B <= user_dist_B < be_B:
                _pick_B_mm = sum(
                    math.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
                    for i in range(bs_B, min(user_dist_B, len(pts) - 1))
                    if i + 1 < len(pts)
                )
                if _pick_B_mm <= branch_budget_mm:
                    dist_B_gi = user_dist_B
                    print(
                        f"[KissingStent] Branch B: user pick gi{dist_B_gi} ({_pick_B_mm:.0f}mm ≤ budget)"
                    )
                elif _has_pancaking(sibling_bi, user_dist_B):
                    dist_B_gi = user_dist_B
                    print(
                        f"[KissingStent] Branch B: pancaking pick gi{dist_B_gi} "
                        f"({_pick_B_mm:.0f}mm > budget {branch_budget_mm:.0f}mm — lesion coverage)"
                    )
                else:
                    dist_B_gi = walk_branch_mm(bs_B, be_B, branch_budget_mm)
                    print(
                        f"[KissingStent] Branch B: user pick {_pick_B_mm:.0f}mm > budget "
                        f"→ capped to gi{dist_B_gi}"
                    )
            else:
                dist_B_gi = walk_branch_mm(bs_B, be_B, branch_target_mm)
                print(f"[KissingStent] Branch B: VBX walk → gi{dist_B_gi}")

            # ── LENGTH DEBUG ──────────────────────────────────────────────
            def _mm(a, b):
                return sum(
                    math.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
                    for i in range(a, min(b, len(pts) - 1))
                )

            _brA_mm = _mm(bs_A, dist_A_gi)
            _brB_mm = _mm(bs_B, dist_B_gi)
            _using_user_picks = user_dist_A is not None or user_dist_B is not None
            _total_A = trunk_mm_val + _brA_mm
            if _using_user_picks:
                _len_status = f"user-pick ({_total_A:.0f}mm)"
            else:
                _len_status = "OK ✓" if _total_A <= 79 else "TOO LONG ✗"
            # FIX: report branch-local pt indices alongside global gi for clarity
            _local_A_end = dist_A_gi - bs_A
            _local_B_end = dist_B_gi - bs_B
            print(
                f"[LenDebug] trunk_mm={trunk_mm_val:.1f}  target_total={target_total:.0f}  branch_target={branch_target_mm:.1f}"
            )
            print(
                f"[LenDebug] Branch A: gi{bs_A}→gi{dist_A_gi} (local pt 0→{_local_A_end}) "
                f"branch_mm={_brA_mm:.1f}  TOTAL={_total_A:.1f}mm"
            )
            print(
                f"[LenDebug] Branch B: gi{bs_B}→gi{dist_B_gi} (local pt 0→{_local_B_end}) "
                f"branch_mm={_brB_mm:.1f}  TOTAL={trunk_mm_val+_brB_mm:.1f}mm"
            )
            print(f"[LenDebug] {_len_status} (max VBX=79mm)")

            # ── Branch segments ───────────────────────────────────────────
            branch_A_path = list(range(bs_A, dist_A_gi + 1))

            # Branch B: bs_B may NOT be adjacent to hub_gi (topology gap).
            # Example: hub=gi95, bs_B=386 — these are spatially close but
            # index-distant. Build a linear bridge using intermediate branch points
            # so the tube doesn't jump across anatomy.
            if bs_B == hub_gi + 1 or bs_B == bs_A:
                # Directly adjacent — no bridge needed
                branch_B_path = list(range(bs_B, dist_B_gi + 1))
            else:
                # Find the connecting path: walk from hub spatially to bs_B.
                # Use the branch that starts closest to hub_gi as a connector.
                # In this anatomy: trunk ends at gi95 (S=1871), Branch 3 starts
                # at gi386 (S=1817) — there is no index bridge, but they are
                # the same bifurcation. We walk Branch 3 from its start.
                # The trunk_seg already ends at hub_gi; append bs_B onwards.
                branch_B_path = list(range(bs_B, dist_B_gi + 1))
                print(
                    f"[KissingStent] Branch B gap bridge: hub=gi{hub_gi} → bs_B=gi{bs_B} "
                    f"(gap={bs_B - hub_gi} indices)"
                )

            # Remove any duplicate of hub at junction
            branch_A_path = [g for g in branch_A_path if g != hub_gi]
            branch_B_path = [g for g in branch_B_path if g != hub_gi]

            # ── Assemble full paths ───────────────────────────────────────
            path_A = trunk_seg + branch_A_path
            _trunk_for_B = trunk_seg

            # For Branch B: if bs_B is not adjacent to hub_gi, the tube will
            # draw a straight line across the gap. Fix by inserting the hub
            # point's 3D coordinates as a synthetic negative index, so the
            # renderer transitions smoothly from hub to bs_B.
            if bs_B != hub_gi + 1 and bs_B != bs_A:
                _interp_gi = -(bs_B + 1000)
                hub_pt = pts[hub_gi] if hub_gi < len(pts) else None
                bsB_pt = pts[bs_B] if bs_B < len(pts) else None
                if hub_pt and bsB_pt:
                    mid_pt = tuple((hub_pt[k] + bsB_pt[k]) / 2.0 for k in range(3))
                    self._interpPoints[_interp_gi] = hub_pt
                    self._interpPoints[_interp_gi - 1] = mid_pt
                    path_B = _trunk_for_B + [_interp_gi, _interp_gi - 1] + branch_B_path
                    print(
                        f"[KissingStent] Branch B: synthetic bridge gi{hub_gi}→gi{bs_B} "
                        f"via interp keys {_interp_gi},{_interp_gi-1}"
                    )
                else:
                    path_B = _trunk_for_B + branch_B_path
            else:
                path_B = _trunk_for_B + branch_B_path
            print(
                f"[KissingStent] path_A: {len(trunk_seg)} trunk + {len(branch_A_path)} branchA "
                f"= {len(path_A)}pts  gi{path_A[0]}→gi{path_A[-1]}"
            )
            print(
                f"[KissingStent] path_B: {len(trunk_seg)} trunk + {len(branch_B_path)} branchB "
                f"= {len(path_B)}pts  gi{path_B[0]}→gi{path_B[-1]}"
            )

            # ── Per-branch diameter sizing ────────────────────────────────────
            def branch_avg_diam(bs, be):
                stable_start = bs + 5
                stable_end = be - 3
                vals = sorted(
                    [
                        diams[i]
                        for i in range(stable_start, stable_end)
                        if i < len(diams) and diams[i] > 1.0
                    ]
                )
                if not vals:
                    return 8.0
                return vals[int(len(vals) * 0.75)]  # 75th pct = healthy reference

            vd_A = branch_avg_diam(bs_A, be_A)
            vd_B = branch_avg_diam(bs_B, be_B)

            # ── GORE VBX kissing stent sizing ─────────────────────────────
            # Priority: use plan["proxDiam"] / plan["distDiam"] when the user
            # has explicitly set them via the stent planner spinboxes.
            # Fall back to auto-sizing from vessel diameter only when absent.
            VBX_NOMINAL = [5, 6, 7, 8, 9, 10, 11]
            VBX_MAX_POST = {5: 8, 6: 9, 7: 11, 8: 13, 9: 13, 10: 13, 11: 16}
            VBX_LENGTHS = [15, 19, 29, 39, 59, 79]

            def snap_vbx_to_vessel(vd):
                """VBX nominal for kissing: use ~80% of vessel diameter (post-dilation rule).
                GORE VBX expands to max_post which should match vessel diameter.
                Work backwards: nom where max_post ≈ vessel_d → nom ≈ vessel_d * 0.80.
                """
                target_nom = vd * 0.80
                return float(min(VBX_NOMINAL, key=lambda s: abs(s - target_nom)))

            def snap_vbx_length(target_mm):
                """Snap to nearest available VBX length."""
                return min(VBX_LENGTHS, key=lambda l: abs(l - target_mm))

            # User override from spinboxes (proxDiam = symmetric kissing diameter)
            user_prox = plan.get("proxDiam", 0.0) or 0.0
            user_dist = plan.get("distDiam", 0.0) or 0.0

            if user_prox >= 5.0:
                # User set a specific diameter — honour it directly
                # Snap to nearest VBX nominal
                nom_kiss = min(VBX_NOMINAL, key=lambda s: abs(s - user_prox))
                print(
                    f"[KissingStent] diameter override: proxDiam={user_prox:.1f}mm "
                    f"→ snapped to nom={nom_kiss}mm"
                )
            else:
                # Auto-size from vessel
                nom_A = snap_vbx_to_vessel(vd_A)
                nom_B = snap_vbx_to_vessel(vd_B)
                nom_kiss = min(nom_A, nom_B)
                print(
                    f"[KissingStent] auto-sizing: vd_A={vd_A:.1f}→{nom_A:.0f}mm  "
                    f"vd_B={vd_B:.1f}→{nom_B:.0f}mm  nom_kiss={nom_kiss:.0f}mm"
                )

            r_A = nom_kiss / 2.0
            r_B = nom_kiss / 2.0
            max_post_A = VBX_MAX_POST.get(int(nom_kiss), 13)
            r_trunk = min(r_A * 2.0, 7.0)

            # VBX length — snap total stent length to nearest available
            branch_A_mm = sum(
                math.sqrt(sum((pts[i + 1][k] - pts[i][k]) ** 2 for k in range(3)))
                for i in range(bs_A, min(dist_A_gi, len(pts) - 1))
                if i + 1 < len(pts)
            )
            snapped_len = snap_vbx_length(trunk_mm_val + branch_A_mm)

            print(
                f"[KissingStent] VBX: nom={nom_kiss:.0f}mm×{snapped_len}mm  "
                f"max_post={max_post_A}mm  r_each={r_A:.1f}mm"
            )

            # ── Offset: separate the two tubes in the trunk zone ─────────────
            # Use the TRUE FAR endpoints of each branch (be_A-1, be_B-1) to compute
            # the lateral divergence axis.  The VBX-capped dist_A/B_gi are only
            # ~37mm into the branch and still nearly co-linear — they give a
            # misleading sep vector (almost pure-Y in this vessel).
            # The full branch endpoints are hundreds of mm apart and capture the
            # real left-right (X-axis) divergence of the iliacs.
            ep_A_far = pts[be_A - 1] if be_A - 1 < len(pts) else pts[bs_A]
            ep_B_far = pts[be_B - 1] if be_B - 1 < len(pts) else pts[bs_B]
            sep = [ep_A_far[k] - ep_B_far[k] for k in range(3)]
            sep_len = math.sqrt(sum(x * x for x in sep)) or 1.0
            sep = [x / sep_len for x in sep]

            # Project out the vertical (Z) component so sep is truly lateral
            sep[2] = 0.0
            sep_len2 = math.sqrt(sep[0] ** 2 + sep[1] ** 2) or 1.0
            sep = [sep[0] / sep_len2, sep[1] / sep_len2, 0.0]

            print(
                f"[SepDebug] ep_A_far={[round(x,1) for x in ep_A_far]}  "
                f"ep_B_far={[round(x,1) for x in ep_B_far]}"
            )
            print(
                f"[SepDebug] sep={[round(x,2) for x in sep]}  (lateral only, Z zeroed)"
            )

            # hub_set spans all trunk + any synthetic bridge points
            hub_set = set(range(trunk_bs, hub_gi + 1))
            for _gk in path_B:
                if _gk < 0:
                    hub_set.add(_gk)
            print(
                f"[KissingStent] hub_set: trunk_bs={trunk_bs}→hub_gi={hub_gi} ({len(hub_set)} pts)"
            )
            overlap_A = sum(1 for g in path_A if g in hub_set)
            overlap_B = sum(1 for g in path_B if g in hub_set)
            print(f"[KissingStent] trunk overlap: A={overlap_A}pts  B={overlap_B}pts")

            def make_offset_coords(path, offset_sign, r_render, off_dist):
                TAPER_PTS = 12
                result, radii = [], []
                n_trunk_pts = 0
                branch_start_idx = None
                for idx, gi in enumerate(path):
                    if gi >= 0 and gi not in hub_set:
                        branch_start_idx = idx
                        break
                for idx, gi in enumerate(path):
                    p = _resolve(gi)
                    if p is None:
                        continue
                    in_trunk = gi >= 0 and gi in hub_set
                    if in_trunk:
                        off = off_dist * offset_sign
                        p = (
                            p[0] + sep[0] * off,
                            p[1] + sep[1] * off,
                            p[2] + sep[2] * off,
                        )
                        n_trunk_pts += 1
                    elif (
                        branch_start_idx is not None
                        and idx < branch_start_idx + TAPER_PTS
                    ):
                        taper = 1.0 - (idx - branch_start_idx) / TAPER_PTS
                        off = off_dist * offset_sign * taper
                        p = (
                            p[0] + sep[0] * off,
                            p[1] + sep[1] * off,
                            p[2] + sep[2] * off,
                        )
                    radii.append(r_render)
                    result.append(p)
                if n_trunk_pts == 0:
                    print(
                        f"[KissingStent] WARNING: path sign={offset_sign:+.0f} "
                        f"has 0 trunk points — hub_set may not overlap path"
                    )
                return result, radii

            # ── Visual rendering ──────────────────────────────────────────────
            # Each tube radius must fit inside half the vessel lumen after
            # accounting for the lateral offset between the two tubes.
            # Two tubes of radius r separated by 2r need 4r total → each needs
            # r ≤ vessel_radius / 2.  Cap to 92% of that for visual clearance.
            _sample_gis = list(trunk_seg)
            _sample_gis += list(range(bs_A, min(bs_A + 20, be_A)))
            _sample_gis += list(range(bs_B, min(bs_B + 20, be_B)))
            _vessel_radii = [
                diams[g] / 2.0
                for g in _sample_gis
                if 0 <= g < len(diams) and diams[g] > 1.0
            ]
            _vessel_r_avg = (
                (sum(_vessel_radii) / len(_vessel_radii)) if _vessel_radii else 8.0
            )

            _max_post = float(VBX_MAX_POST.get(int(nom_kiss), 13))
            # Each tube gets at most half the vessel lumen (two tubes side by side)
            _r_max_fit = _vessel_r_avg * 0.92 / 2.0
            render_r = min(_max_post / 2.0, _r_max_fit)
            render_r = max(render_r, 1.5)
            # Offset: separate tubes so both limbs are clearly distinct visually
            render_off = render_r * 1.5
            print(
                f"[KissingStent] render: r={render_r:.2f}mm offset={render_off:.2f}mm "
                f"(nom={nom_kiss:.0f}mm max_post={_max_post:.0f}mm "
                f"vessel_r={_vessel_r_avg:.1f}mm)"
            )

            coords_A, radii_A = make_offset_coords(path_A, +1.0, render_r, render_off)
            coords_B, radii_B = make_offset_coords(path_B, -1.0, render_r, render_off)

            # ── Build GORE VBX kissing stent graft tubes ─────────────────
            # VBX = covered stent: ePTFE graft + cobalt-chrome diamond-zig frame
            # Rendered as ivory/white semi-opaque tubes (ePTFE appearance)

            def make_vbx_tube(coords, radii, color=(0.96, 0.95, 0.90)):
                """VBX covered stent graft tube: ivory ePTFE appearance."""
                if len(coords) < 2:
                    return None
                cp = vtk.vtkPoints()
                cl = vtk.vtkCellArray()
                cr = vtk.vtkFloatArray()
                cr.SetName("TubeRadius")
                cl.InsertNextCell(len(coords))
                for i, (p, rad) in enumerate(zip(coords, radii)):
                    cp.InsertNextPoint(p[0], p[1], p[2])
                    cl.InsertCellPoint(i)
                    cr.InsertNextValue(rad)
                cpd = vtk.vtkPolyData()
                cpd.SetPoints(cp)
                cpd.SetLines(cl)
                cpd.GetPointData().AddArray(cr)
                cpd.GetPointData().SetActiveScalars("TubeRadius")
                tf = vtk.vtkTubeFilter()
                tf.SetInputData(cpd)
                tf.SetNumberOfSides(24)
                tf.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
                tf.CappingOn()
                tf.Update()
                return tf.GetOutput()

            def make_nitinol_frame(coords, radii):
                """GORE-style zigzag/chevron nitinol wire frame.
                Replicates the W/Z sinusoidal chevron pattern visible on GORE
                Viabahn/Excluder: repeating circumferential zigzag rings with
                alternating peak offsets, connected by short axial bridges."""
                if len(coords) < 2:
                    return None
                import math as _mf

                frame_app = vtk.vtkAppendPolyData()

                # Arc-length parameterisation
                arc = [0.0]
                for i in range(1, len(coords)):
                    d = _mf.sqrt(
                        sum((coords[i][k] - coords[i - 1][k]) ** 2 for k in range(3))
                    )
                    arc.append(arc[-1] + d)
                total_len = arc[-1]
                if total_len < 1.0:
                    return None

                def sample(s):
                    s = max(0.0, min(s, arc[-1]))
                    for i in range(1, len(arc)):
                        if arc[i] >= s:
                            t = (s - arc[i - 1]) / max(arc[i] - arc[i - 1], 1e-9)
                            return tuple(
                                coords[i - 1][k] * (1 - t) + coords[i][k] * t
                                for k in range(3)
                            )
                    return coords[-1]

                def sample_r(s):
                    frac = s / max(arc[-1], 1e-9)
                    idx = min(int(frac * (len(radii) - 1)), len(radii) - 1)
                    return radii[idx] * 1.035  # just outside graft

                def local_frame(s):
                    ds = arc[-1] / 400
                    n = sample(s + ds * 0.5)
                    p = sample(max(0, s - ds * 0.5))
                    tx = n[0] - p[0]
                    ty = n[1] - p[1]
                    tz = n[2] - p[2]
                    tl = _mf.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
                    tx, ty, tz = tx / tl, ty / tl, tz / tl
                    ax = (0, 0, 1) if abs(tz) < 0.9 else (1, 0, 0)
                    ux = ty * ax[2] - tz * ax[1]
                    uy = tz * ax[0] - tx * ax[2]
                    uz = tx * ax[1] - ty * ax[0]
                    ul = _mf.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
                    ux, uy, uz = ux / ul, uy / ul, uz / ul
                    vx = ty * uz - tz * uy
                    vy = tz * ux - tx * uz
                    vz = tx * uy - ty * ux
                    return (tx, ty, tz), (ux, uy, uz), (vx, vy, vz)

                # Chevron ring parameters
                n_struts = 8  # zigzag peaks per ring circumference
                ring_pitch = 6.0  # axial distance between rings (mm)
                peak_amp = 2.5  # axial half-amplitude of chevron (mm)
                wire_r = 0.40  # wire cross-section radius
                N_WIRE = 6  # wire cross-section sides
                pts_per_seg = 8  # polyline points per zigzag segment

                n_rings = max(2, int(total_len / ring_pitch))

                def make_wire_tube(spine_pts):
                    """Extrude a thin round tube along a polyline spine."""
                    if len(spine_pts) < 2:
                        return None
                    sp_arc = [0.0]
                    for i in range(1, len(spine_pts)):
                        p, q = spine_pts[i - 1], spine_pts[i]
                        sp_arc.append(
                            sp_arc[-1]
                            + _mf.sqrt(sum((q[k] - p[k]) ** 2 for k in range(3)))
                        )
                    wpts = vtk.vtkPoints()
                    wpolys = vtk.vtkCellArray()
                    N = len(spine_pts)
                    for si, (px, py, pz) in enumerate(spine_pts):
                        # tangent
                        if si < N - 1:
                            nx, ny, nz = spine_pts[si + 1]
                            tx, ty, tz = nx - px, ny - py, nz - pz
                        else:
                            nx, ny, nz = spine_pts[si - 1]
                            tx, ty, tz = px - nx, py - ny, pz - nz
                        tl = _mf.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
                        tx, ty, tz = tx / tl, ty / tl, tz / tl
                        ax2 = (0, 0, 1) if abs(tz) < 0.9 else (1, 0, 0)
                        ux2 = ty * ax2[2] - tz * ax2[1]
                        uy2 = tz * ax2[0] - tx * ax2[2]
                        uz2 = tx * ax2[1] - ty * ax2[0]
                        ul2 = _mf.sqrt(ux2 * ux2 + uy2 * uy2 + uz2 * uz2) or 1.0
                        ux2, uy2, uz2 = ux2 / ul2, uy2 / ul2, uz2 / ul2
                        vx2 = ty * uz2 - tz * uy2
                        vy2 = tz * ux2 - tx * uz2
                        vz2 = tx * uy2 - ty * ux2
                        for ci in range(N_WIRE):
                            theta = 2 * _mf.pi * ci / N_WIRE
                            ct, st = _mf.cos(theta), _mf.sin(theta)
                            wpts.InsertNextPoint(
                                px + wire_r * (ct * ux2 + st * vx2),
                                py + wire_r * (ct * uy2 + st * vy2),
                                pz + wire_r * (ct * uz2 + st * vz2),
                            )
                    for si in range(N - 1):
                        for ci in range(N_WIRE):
                            a = si * N_WIRE + ci
                            b = si * N_WIRE + (ci + 1) % N_WIRE
                            c = (si + 1) * N_WIRE + (ci + 1) % N_WIRE
                            d = (si + 1) * N_WIRE + ci
                            q = vtk.vtkQuad()
                            q.GetPointIds().SetId(0, a)
                            q.GetPointIds().SetId(1, b)
                            q.GetPointIds().SetId(2, c)
                            q.GetPointIds().SetId(3, d)
                            wpolys.InsertNextCell(q)
                    wpd = vtk.vtkPolyData()
                    wpd.SetPoints(wpts)
                    wpd.SetPolys(wpolys)
                    return wpd

                for ri in range(n_rings):
                    s_center = ring_pitch * (ri + 0.5)
                    if s_center > arc[-1]:
                        break
                    R = sample_r(s_center)
                    cx, cy, cz = sample(s_center)
                    (_t, _u, _v) = local_frame(s_center)
                    ux, uy, uz = _u
                    vx, vy, vz = _v

                    # Offset alternating rings axially by half peak_amp
                    phase_offset = peak_amp * 0.5 if ri % 2 == 1 else 0.0

                    # Build one chevron ring: zigzag around circumference
                    ring_spine = []
                    for si_seg in range(n_struts):
                        phi0 = 2 * _mf.pi * si_seg / n_struts
                        phi1 = 2 * _mf.pi * (si_seg + 1) / n_struts
                        # Each segment: valley → peak → valley (sinusoidal axial)
                        for k in range(pts_per_seg + 1):
                            t = k / pts_per_seg
                            phi = phi0 + (phi1 - phi0) * t
                            ax_off = phase_offset + peak_amp * _mf.sin(_mf.pi * t * 2)
                            # Sample axial position on centreline
                            s_pt = s_center + ax_off
                            s_pt = max(0.0, min(arc[-1], s_pt))
                            cx2, cy2, cz2 = sample(s_pt)
                            R2 = sample_r(s_pt)
                            (_t2, _u2, _v2) = local_frame(s_pt)
                            u2x, u2y, u2z = _u2
                            v2x, v2y, v2z = _v2
                            cp2 = _mf.cos(phi)
                            sp2 = _mf.sin(phi)
                            ring_spine.append(
                                (
                                    cx2 + R2 * (cp2 * u2x + sp2 * v2x),
                                    cy2 + R2 * (cp2 * u2y + sp2 * v2y),
                                    cz2 + R2 * (cp2 * u2z + sp2 * v2z),
                                )
                            )

                    wpd = make_wire_tube(ring_spine)
                    if wpd:
                        frame_app.AddInputData(wpd)

                frame_app.Update()
                return frame_app.GetOutput()

            # ── Build and assemble tubes ──────────────────────────────────
            tube_A = make_vbx_tube(coords_A, radii_A)
            tube_B = make_vbx_tube(coords_B, radii_B)

            if not tube_A and not tube_B:
                return False

            app_AB = vtk.vtkAppendPolyData()
            if tube_A:
                app_AB.AddInputData(tube_A)
            if tube_B:
                app_AB.AddInputData(tube_B)
            app_AB.Update()

            self._stentNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", "StentModel"
            )
            self._stentNode.SetAndObservePolyData(app_AB.GetOutput())
            self._stentNode.CreateDefaultDisplayNodes()
            dn = self._stentNode.GetDisplayNode()
            dn.SetScalarVisibility(False)
            # VBX is a COVERED stent graft — ivory/white ePTFE appearance
            # (NOT silver/metallic — that would be a bare-metal stent)
            dn.SetColor(0.96, 0.95, 0.90)  # ivory-white ePTFE graft
            dn.SetOpacity(0.82)  # semi-opaque: graft visible, vessel visible through
            dn.SetAmbient(0.35)
            dn.SetDiffuse(0.60)
            dn.SetSpecular(0.25)  # low specular — ePTFE is matte, not shiny
            dn.SetRepresentation(2)
            dn.SetVisibility(1)

            for _en in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
                if _en.GetAttribute("VesselAnalyzerExpanded") == "1":
                    slicer.mrmlScene.RemoveNode(_en)
            if hasattr(self, "modelNode") and self.modelNode:
                _od = self.modelNode.GetDisplayNode()
                if _od:
                    _od.SetVisibility(1)
            print(
                f"[KissingStent] ✓ Placed: trunk={len(trunk_seg)}pts "
                f"A={len(branch_A_path)}pts (local pt 0→{len(branch_A_path)-1}) "
                f"B={len(branch_B_path)}pts (local pt 0→{len(branch_B_path)-1}) "
                f"nom={nom_kiss:.0f}mm×{snapped_len}mm VBX "
                f"max_post={max_post_A}mm r_each={r_A:.1f}mm"
            )
            # FIX: verify ≤79mm and print pancaking coverage status
            _pancake_gis_A = [
                f["pointIdx"]
                for f in getattr(self, "findings", [])
                if f.get("branchIdx") == bi and "Pancak" in f.get("type", "")
            ]
            _pancake_gis_B = [
                f["pointIdx"]
                for f in getattr(self, "findings", [])
                if f.get("branchIdx") == sibling_bi and "Pancak" in f.get("type", "")
            ]
            if _pancake_gis_A:
                _p_covered = all(bs_A <= g <= dist_A_gi for g in _pancake_gis_A)
                print(
                    f"[KissingStent] Pancaking Branch A: {'COVERED ✓' if _p_covered else 'NOT COVERED ✗'} "
                    f"(lesion local pts {[g-bs_A for g in _pancake_gis_A]}, stent ends local pt {_local_A_end})"
                )
            if _pancake_gis_B:
                _p_covered = all(bs_B <= g <= dist_B_gi for g in _pancake_gis_B)
                print(
                    f"[KissingStent] Pancaking Branch B: {'COVERED ✓' if _p_covered else 'NOT COVERED ✗'} "
                    f"(lesion local pts {[g-bs_B for g in _pancake_gis_B]}, stent ends local pt {_local_B_end})"
                )
            _total_placed = trunk_mm_val + max(_brA_mm, _brB_mm)
            if _total_placed > 79.0:
                print(
                    f"[KissingStent] ✗ WARNING: total {_total_placed:.1f}mm EXCEEDS 79mm VBX limit"
                )
            else:
                print(f"[KissingStent] ✓ Total {_total_placed:.1f}mm ≤ 79mm VBX limit")
            if coords_A:
                p_end = coords_A[-1]
                # FIX: report diameter (2×|R|) instead of raw signed R coordinate
                print(
                    f"[KissingStent] Limb A end: D={abs(p_end[0])*2:.1f} A={p_end[1]:.1f} S={p_end[2]:.1f}"
                )
            if coords_B:
                p_end = coords_B[-1]
                print(
                    f"[KissingStent] Limb B end: D={abs(p_end[0])*2:.1f} A={p_end[1]:.1f} S={p_end[2]:.1f}"
                )
            return True
        except Exception as e:
            import traceback

            print(f"[KissingStent] ERROR: {e}")
            traceback.print_exc()
            return False


    def _placeYStent3D(self, plan):
        """Build and display a Y/trouser bifurcation stent from 3 limb paths."""
        import vtk

        try:
            _interp = getattr(self, "_interpPoints", {})

            def _resolve(gi):
                if gi < 0:
                    return _interp.get(gi)
                if gi < len(self.points):
                    return self.points[gi]
                return None

            appender = vtk.vtkAppendPolyData()
            n_added = 0
            for seg_name, seg_pts, seg_r in [
                ("trunk", plan["y_trunk"], plan["r_trunk"]),
                ("left", plan["y_left"], plan["r_left"]),
                ("right", plan["y_right"], plan["r_right"]),
            ]:
                if not seg_pts or len(seg_pts) < 2:
                    continue
                resolved = [_resolve(gi) for gi in seg_pts]
                resolved = [p for p in resolved if p is not None]
                if len(resolved) < 2:
                    continue
                cp = vtk.vtkPoints()
                cl = vtk.vtkCellArray()
                cr = vtk.vtkFloatArray()
                cr.SetName("TubeRadius")
                cl.InsertNextCell(len(resolved))
                for ri, pt in enumerate(resolved):
                    cp.InsertNextPoint(pt[0], pt[1], pt[2])
                    cl.InsertCellPoint(ri)
                    cr.InsertNextValue(seg_r)
                cpd = vtk.vtkPolyData()
                cpd.SetPoints(cp)
                cpd.SetLines(cl)
                cpd.GetPointData().AddArray(cr)
                cpd.GetPointData().SetActiveScalars("TubeRadius")
                tf = vtk.vtkTubeFilter()
                tf.SetInputData(cpd)
                tf.SetNumberOfSides(24)
                tf.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
                tf.CappingOn()
                tf.Update()
                appender.AddInputData(tf.GetOutput())
                n_added += 1
            if n_added == 0:
                print("[StentY] ERROR: no segments to render")
                return
            appender.Update()
            pd_mesh = appender.GetOutput()

            # ── Type-specific appearance (mirrors placeStent3D logic) ──────
            stype = plan.get("type", "Bifurcated").strip().lower()
            stent_color = (0.9, 0.75, 0.1)  # default gold (BMS/Bifurcated)
            stent_opacity = 0.85
            stent_repr = 2
            stent_ambient = 0.2
            stent_specular = 0.5
            if (
                "covered" in stype
                or "ptfe" in stype
                or "evar" in stype
                or "graft" in stype
            ):
                stent_color = (0.95, 0.95, 0.95)  # white PTFE graft
                stent_opacity = 0.75
            elif "des" in stype or "eluting" in stype:
                stent_color = (0.2, 0.6, 0.9)  # blue drug-eluting
                stent_opacity = 0.90
            elif "tapered" in stype:
                stent_color = (1.0, 0.5, 0.1)  # orange-gold
            elif "venous" in stype or "viabahn" in stype:
                stent_color = (0.6, 0.85, 0.95)  # light blue Viabahn
                stent_opacity = 0.80
            elif "vbx" in stype or "kissing" in stype:
                stent_color = (0.75, 0.78, 0.82)  # cobalt-chrome metallic
                stent_opacity = 0.88
                stent_specular = 0.7
            print(
                f"[StentY] Y-stent type='{stype}' color={stent_color} opacity={stent_opacity}"
            )

            self._stentNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", "StentModel"
            )
            self._stentNode.SetAndObservePolyData(pd_mesh)
            self._stentNode.CreateDefaultDisplayNodes()
            dn = self._stentNode.GetDisplayNode()
            dn.SetScalarVisibility(False)
            dn.SetColor(*stent_color)
            dn.SetOpacity(stent_opacity)
            dn.SetRepresentation(stent_repr)
            dn.SetAmbient(stent_ambient)
            dn.SetSpecular(stent_specular)
            dn.SetVisibility(1)
            print("[StentY] Y-stent placed in 3D")
        except Exception as e:
            import traceback

            print(f"[StentY] ERROR: {e}")
            traceback.print_exc()


    def placeStent3D(self, plan):
        """Build a 3D tube mesh along centerline from proxPt to distPt."""
        import math

        # Only remove existing stent if NOT placing an additional manual straight stent.
        # custom_path picks accumulate; all other routes (slider, auto) replace.
        is_additional = plan and plan.get("_additional", False)
        if not is_additional:
            self.removeStent3D()
        if not plan:
            return
        gp = plan["proxPt"]
        gd = plan["distPt"]
        pd = plan["proxDiam"] / 2.0
        dd = plan["distDiam"] / 2.0

        # Sanity check: stent should not span more than 300mm or full point range
        if gd <= gp:
            print(f"[placeStent3D] Invalid range gp={gp} gd={gd} — aborting")
            return
        if len(self.distances) > gd and len(self.distances) > gp:
            stent_mm = self.distances[gd] - self.distances[gp]
            if stent_mm > 350.0:
                print(
                    f"[placeStent3D] Stent span {stent_mm:.0f}mm > 350mm — likely wrong branch, aborting"
                )
                return
        stype = plan["type"]
        # Cross-bifurcation: prepend trunk points walking backwards from bifurcation
        trunk_mm = plan.get("trunk_mm", 0.0)
        # Use explicit custom path if provided (point-to-point pick mode)
        if plan.get("custom_path"):
            pts_range = plan["custom_path"]
        else:
            pts_range = list(range(gp, gd + 1))
        if not plan.get("custom_path") and trunk_mm > 0.0 and len(self.branches) > 1:
            # Branch 0 is the trunk; find its end point (= bifurcation = gp)
            trunk_start, trunk_end = self.branches[0]  # pts 0..116
            # Walk backwards from trunk_end-1 accumulating distance until trunk_mm
            trunk_pts = []
            acc = 0.0
            for ti in range(trunk_end - 1, trunk_start - 1, -1):
                if ti + 1 < len(self.points):
                    p0 = self.points[ti]
                    p1 = self.points[ti + 1]
                    acc += (
                        (p1[0] - p0[0]) ** 2
                        + (p1[1] - p0[1]) ** 2
                        + (p1[2] - p0[2]) ** 2
                    ) ** 0.5
                trunk_pts.append(ti)
                if acc >= trunk_mm:
                    break
            trunk_pts.reverse()  # proximal → distal order
            pts_range = trunk_pts + pts_range
        # Normalise stype — strip leading spaces from combo items
        stype = stype.strip()
        # Kissing stent: two parallel tubes across bifurcation
        if "Kissing" in stype or "VBX" in stype:
            placed = self._placeKissingStent3D(plan)
            if placed:
                return
            # fallthrough to single tube if no sibling found
        # For bifurcated/Y-stent: skip regular tube, go straight to Y rendering
        if "Bifurcated" in stype and plan.get("y_trunk") is not None:
            self._placeYStent3D(plan)
            return
        # If plan came from a Kissing/Y pick and type switched to a single-vessel type,
        # remap proxPt/distPt to the primary branch only so the single tube
        # doesn't run from deep trunk through bifurcation (anatomically wrong).
        # branchIdx / bi_left is the primary branch — use its start as the new prox.
        if plan.get("branchIdx", -1) > 0 and not plan.get("custom_path"):
            bi_single = plan["branchIdx"]
            if bi_single < len(self.branches):
                bs_s, be_s = self.branches[bi_single]
                # Only remap if current gp is in the trunk (before branch start)
                if gp < bs_s:
                    print(
                        f"[placeStent3D] Remapping prox from trunk gi{gp} "
                        f"to branch{bi_single} start gi{bs_s} for single-tube type '{stype}'"
                    )
                    gp = bs_s
                    pts_range = list(range(gp, gd + 1))
        # If plan came from a Y/Kissing pick but type switched to something plain,
        # ensure pts_range covers a sensible path (proxPt→distPt sequential)
        if not plan.get("custom_path") and gp > gd:
            gp, gd = gd, gp
            pts_range = list(range(gp, gd + 1))
        n = len(pts_range)
        if n < 2:
            return
        N_CIRC = 24  # circumferential segments
        # Use vtkTubeFilter for robust tube rendering — avoids degenerate ring issues
        import vtk

        cline_pts = vtk.vtkPoints()
        cline_lines = vtk.vtkCellArray()
        cline_radii = vtk.vtkFloatArray()
        cline_radii.SetName("TubeRadius")
        cline_lines.InsertNextCell(n)
        pdmap = getattr(self, "preDilationMap", {})
        _min_r = float("inf")
        _max_r = 0.0
        _balloon_hits = 0
        for ri, gi in enumerate(pts_range):
            pt = self.points[gi]
            cline_pts.InsertNextPoint(pt[0], pt[1], pt[2])
            cline_lines.InsertCellPoint(ri)
            t = ri / (n - 1)
            r = pd + (dd - pd) * t if stype == "Tapered" else (pd + dd) / 2.0
            r_before = r
            if gi in pdmap:
                _entry = pdmap[gi]
                balloon_r = _entry["balloon_diam"] / 2.0
                r = max(r, balloon_r)
                _balloon_hits += 1
            _min_r = min(_min_r, r)
            _max_r = max(_max_r, r)
            cline_radii.InsertNextValue(r)
        cline_pd = vtk.vtkPolyData()
        cline_pd.SetPoints(cline_pts)
        cline_pd.SetLines(cline_lines)
        cline_pd.GetPointData().AddArray(cline_radii)
        cline_pd.GetPointData().SetActiveScalars("TubeRadius")
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(cline_pd)
        tube.SetNumberOfSides(24)
        tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        tube.CappingOn()
        tube.Update()
        pd_mesh = tube.GetOutput()

        # ── Type-specific appearance ────────────────────────────────────
        stype_clean = stype.strip().lower()
        stent_color = (0.9, 0.75, 0.1)  # default gold
        stent_opacity = 0.85
        stent_repr = 2  # 0=points,1=wireframe,2=surface
        stent_ambient = 0.2
        stent_specular = 0.5

        if "mesh" in stype_clean:
            # ── Abbott/BSc open-cell diamond nitinol stent ───────────────
            # Rows of diamond cells with alternating offsets, like the
            # Abbott Xience / BSc Synergy open-cell stent structure.
            import math as _ms

            pts_list = [self.points[gi] for gi in pts_range]
            stent_r = (pd + dd) / 2.0  # mean radius
            wire_r_s = 0.30  # wire cross-section
            N_CELLS_CIRC = 8  # diamonds around circumference
            CELL_LEN_MM = 3.0  # axial length of one diamond row
            N_WIRE_S = 6  # wire tube cross-section sides

            # Arc-length of centreline
            arc_s = [0.0]
            for i in range(1, len(pts_list)):
                d = _ms.sqrt(
                    sum((pts_list[i][k] - pts_list[i - 1][k]) ** 2 for k in range(3))
                )
                arc_s.append(arc_s[-1] + d)
            total_s = arc_s[-1] or 1.0

            def samp_s(s):
                s = max(0.0, min(s, arc_s[-1]))
                for i in range(1, len(arc_s)):
                    if arc_s[i] >= s:
                        t = (s - arc_s[i - 1]) / max(arc_s[i] - arc_s[i - 1], 1e-9)
                        return tuple(
                            pts_list[i - 1][k] * (1 - t) + pts_list[i][k] * t
                            for k in range(3)
                        )
                return pts_list[-1]

            def frame_s(s):
                ds = total_s / 400
                n = samp_s(s + ds * 0.5)
                p = samp_s(max(0, s - ds * 0.5))
                tx = n[0] - p[0]
                ty = n[1] - p[1]
                tz = n[2] - p[2]
                tl = _ms.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
                tx, ty, tz = tx / tl, ty / tl, tz / tl
                ax = (0, 0, 1) if abs(tz) < 0.9 else (1, 0, 0)
                ux = ty * ax[2] - tz * ax[1]
                uy = tz * ax[0] - tx * ax[2]
                uz = tx * ax[1] - ty * ax[0]
                ul = _ms.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
                ux, uy, uz = ux / ul, uy / ul, uz / ul
                vx = ty * uz - tz * uy
                vy = tz * ux - tx * uz
                vz = tx * uy - ty * ux
                return (ux, uy, uz), (vx, vy, vz)

            def pt_on_surf(s, phi):
                cx, cy, cz = samp_s(s)
                (ux, uy, uz), (vx, vy, vz) = frame_s(s)
                cp = _ms.cos(phi)
                sp = _ms.sin(phi)
                return (
                    cx + stent_r * (cp * ux + sp * vx),
                    cy + stent_r * (cp * uy + sp * vy),
                    cz + stent_r * (cp * uz + sp * vz),
                )

            def make_strut(p0, p1, npts=10):
                """Build a thin tube segment between two surface points."""
                spine = [
                    (p0[k] + (p1[k] - p0[k]) * i / (npts - 1))
                    for i in range(npts)
                    for k in range(3)
                ]
                spine = [
                    (spine[i * 3], spine[i * 3 + 1], spine[i * 3 + 2])
                    for i in range(npts)
                ]
                wpts = vtk.vtkPoints()
                wpolys = vtk.vtkCellArray()
                N = len(spine)
                for si, (px, py, pz) in enumerate(spine):
                    if si < N - 1:
                        nx, ny, nz = spine[si + 1]
                    else:
                        nx, ny, nz = spine[si - 1]
                        nx, ny, nz = 2 * px - nx, 2 * py - ny, 2 * pz - nz
                    tx = nx - px
                    ty = ny - py
                    tz = nz - pz
                    tl = _ms.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
                    tx, ty, tz = tx / tl, ty / tl, tz / tl
                    ax2 = (0, 0, 1) if abs(tz) < 0.9 else (1, 0, 0)
                    ux2 = ty * ax2[2] - tz * ax2[1]
                    uy2 = tz * ax2[0] - tx * ax2[2]
                    uz2 = tx * ax2[1] - ty * ax2[0]
                    ul2 = _ms.sqrt(ux2 * ux2 + uy2 * uy2 + uz2 * uz2) or 1.0
                    ux2, uy2, uz2 = ux2 / ul2, uy2 / ul2, uz2 / ul2
                    vx2 = ty * uz2 - tz * uy2
                    vy2 = tz * ux2 - tx * uz2
                    vz2 = tx * uy2 - ty * ux2
                    for ci in range(N_WIRE_S):
                        th = 2 * _ms.pi * ci / N_WIRE_S
                        ct, st = _ms.cos(th), _ms.sin(th)
                        wpts.InsertNextPoint(
                            px + wire_r_s * (ct * ux2 + st * vx2),
                            py + wire_r_s * (ct * uy2 + st * vy2),
                            pz + wire_r_s * (ct * uz2 + st * vz2),
                        )
                for si in range(N - 1):
                    for ci in range(N_WIRE_S):
                        a = si * N_WIRE_S + ci
                        b = si * N_WIRE_S + (ci + 1) % N_WIRE_S
                        c = (si + 1) * N_WIRE_S + (ci + 1) % N_WIRE_S
                        d = (si + 1) * N_WIRE_S + ci
                        q = vtk.vtkQuad()
                        q.GetPointIds().SetId(0, a)
                        q.GetPointIds().SetId(1, b)
                        q.GetPointIds().SetId(2, c)
                        q.GetPointIds().SetId(3, d)
                        wpolys.InsertNextCell(q)
                wpd = vtk.vtkPolyData()
                wpd.SetPoints(wpts)
                wpd.SetPolys(wpolys)
                return wpd

            mesh_app = vtk.vtkAppendPolyData()
            n_rows = max(2, int(total_s / CELL_LEN_MM))
            for row in range(n_rows):
                s0 = total_s * row / n_rows
                s1 = total_s * (row + 1) / n_rows
                s_mid = (s0 + s1) / 2.0
                offset = (
                    _ms.pi / N_CELLS_CIRC if row % 2 == 1 else 0.0
                )  # alternate rows
                for ci in range(N_CELLS_CIRC):
                    phi_left = 2 * _ms.pi * (ci + 0.0) / N_CELLS_CIRC + offset
                    phi_right = 2 * _ms.pi * (ci + 1.0) / N_CELLS_CIRC + offset
                    phi_mid = (phi_left + phi_right) / 2.0
                    # Diamond: bottom → left_mid → top → right_mid → bottom
                    p_bot = pt_on_surf(s0, phi_mid)
                    p_left = pt_on_surf(s_mid, phi_left)
                    p_top = pt_on_surf(s1, phi_mid)
                    p_right = pt_on_surf(s_mid, phi_right)
                    for seg in [
                        (p_bot, p_left),
                        (p_left, p_top),
                        (p_top, p_right),
                        (p_right, p_bot),
                    ]:
                        spd = make_strut(seg[0], seg[1])
                        if spd:
                            mesh_app.AddInputData(spd)
            mesh_app.Update()
            pd_mesh = mesh_app.GetOutput()
            stent_color = (0.82, 0.82, 0.85)  # cobalt-chrome silver
            stent_opacity = 1.0
            stent_repr = 2
            stent_ambient = 0.3
            stent_specular = 0.8
        elif "coil" in stype_clean or "annular" in stype_clean:
            # ── Annular coil stent: helical spring coil (like image) ─────
            # A continuous helix wrapped around the vessel axis
            import math as _m

            appender_coil = vtk.vtkAppendPolyData()
            pts_list = [self.points[gi] for gi in pts_range]
            coil_radius = pd  # stent radius
            wire_radius = 0.6  # wire thickness
            turns_per_mm = 0.35  # coil density
            N_WIRE_CIRC = 10  # cross-section segments
            # Compute total length
            total_len = sum(
                _m.sqrt(
                    sum((pts_list[i + 1][k] - pts_list[i][k]) ** 2 for k in range(3))
                )
                for i in range(len(pts_list) - 1)
            )
            n_turns = int(total_len * turns_per_mm)
            n_steps = max(n_turns * 24, 120)
            # Sample centerline by arc length for smooth parameterisation
            arc = [0.0]
            for i in range(1, len(pts_list)):
                d = _m.sqrt(
                    sum((pts_list[i][k] - pts_list[i - 1][k]) ** 2 for k in range(3))
                )
                arc.append(arc[-1] + d)

            def sample_cl(s):
                s = max(0.0, min(s, arc[-1]))
                for i in range(1, len(arc)):
                    if arc[i] >= s:
                        t = (s - arc[i - 1]) / max(arc[i] - arc[i - 1], 1e-9)
                        return tuple(
                            pts_list[i - 1][k] * (1 - t) + pts_list[i][k] * t
                            for k in range(3)
                        )
                return pts_list[-1]

            # Build helix spine points
            helix_pts = []
            for si in range(n_steps + 1):
                s = arc[-1] * si / n_steps
                phi = 2 * _m.pi * n_turns * si / n_steps
                cx, cy, cz = sample_cl(s)
                # Tangent
                ds = arc[-1] / n_steps
                nx, ny, nz = sample_cl(s + ds * 0.5)
                px, py, pz = sample_cl(max(0, s - ds * 0.5))
                tx = nx - px
                ty = ny - py
                tz = nz - pz
                tl = _m.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
                tx, ty, tz = tx / tl, ty / tl, tz / tl
                ax = (0, 0, 1) if abs(tz) < 0.9 else (1, 0, 0)
                ux = ty * ax[2] - tz * ax[1]
                uy = tz * ax[0] - tx * ax[2]
                uz = tx * ax[1] - ty * ax[0]
                ul = _m.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
                ux, uy, uz = ux / ul, uy / ul, uz / ul
                vx = ty * uz - tz * uy
                vy = tz * ux - tx * uz
                vz = tx * uy - ty * ux
                cp, sp = _m.cos(phi), _m.sin(phi)
                helix_pts.append(
                    (
                        cx + coil_radius * (cp * ux + sp * vx),
                        cy + coil_radius * (cp * uy + sp * vy),
                        cz + coil_radius * (cp * uz + sp * vz),
                        tx,
                        ty,
                        tz,
                    )
                )
            # Extrude circular cross-section along helix spine
            h_pts = vtk.vtkPoints()
            h_poly = vtk.vtkCellArray()
            for si, (hx, hy, hz, tx, ty, tz) in enumerate(helix_pts):
                # local frame at helix point
                ax2 = (0, 0, 1) if abs(tz) < 0.9 else (1, 0, 0)
                ux2 = ty * ax2[2] - tz * ax2[1]
                uy2 = tz * ax2[0] - tx * ax2[2]
                uz2 = tx * ax2[1] - ty * ax2[0]
                ul2 = _m.sqrt(ux2 * ux2 + uy2 * uy2 + uz2 * uz2) or 1.0
                ux2, uy2, uz2 = ux2 / ul2, uy2 / ul2, uz2 / ul2
                vx2 = ty * uz2 - tz * uy2
                vy2 = tz * ux2 - tx * uz2
                vz2 = tx * uy2 - ty * ux2
                for ci in range(N_WIRE_CIRC):
                    theta = 2 * _m.pi * ci / N_WIRE_CIRC
                    ct, st = _m.cos(theta), _m.sin(theta)
                    h_pts.InsertNextPoint(
                        hx + wire_radius * (ct * ux2 + st * vx2),
                        hy + wire_radius * (ct * uy2 + st * vy2),
                        hz + wire_radius * (ct * uz2 + st * vz2),
                    )
            for si in range(len(helix_pts) - 1):
                for ci in range(N_WIRE_CIRC):
                    a = si * N_WIRE_CIRC + ci
                    b = si * N_WIRE_CIRC + (ci + 1) % N_WIRE_CIRC
                    c = (si + 1) * N_WIRE_CIRC + (ci + 1) % N_WIRE_CIRC
                    d = (si + 1) * N_WIRE_CIRC + ci
                    quad = vtk.vtkQuad()
                    quad.GetPointIds().SetId(0, a)
                    quad.GetPointIds().SetId(1, b)
                    quad.GetPointIds().SetId(2, c)
                    quad.GetPointIds().SetId(3, d)
                    h_poly.InsertNextCell(quad)
            h_pd = vtk.vtkPolyData()
            h_pd.SetPoints(h_pts)
            h_pd.SetPolys(h_poly)
            appender_coil.AddInputData(h_pd)
            appender_coil.Update()
            pd_mesh = appender_coil.GetOutput()
            stent_color = (0.75, 0.75, 0.75)  # silver wire
            stent_opacity = 1.0
            stent_repr = 2

        elif "des" in stype_clean or "eluting" in stype_clean:
            stent_color = (0.2, 0.6, 0.9)  # blue drug-eluting
            stent_opacity = 0.9
        elif "covered" in stype_clean or "ptfe" in stype_clean or "evar" in stype_clean:
            stent_color = (0.95, 0.95, 0.95)  # white graft
            stent_opacity = 0.75
        elif "tapered" in stype_clean:
            stent_color = (1.0, 0.5, 0.1)  # orange-gold
        # BMS / Bare-metal / Straight: default gold already set

        stent_node_name = (
            "StentModel_" + str(plan.get("_stent_idx", 0))
            if plan.get("_additional")
            else "StentModel"
        )
        self._stentNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", stent_node_name
        )
        self._stentNode.SetAndObservePolyData(pd_mesh)
        self._stentNode.CreateDefaultDisplayNodes()
        dn = self._stentNode.GetDisplayNode()
        dn.SetScalarVisibility(False)
        dn.SetColor(*stent_color)
        dn.SetOpacity(stent_opacity)
        dn.SetRepresentation(stent_repr)
        dn.SetAmbient(stent_ambient)
        dn.SetSpecular(stent_specular)
        dn.SetVisibility(1)
        print("[VesselAnalyzer] Stent placed in 3D")


    def removeStent3D(self):
        # Remove all StentModel* nodes (handles multiple placed stents)
        for node in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
            if node.GetName().startswith("StentModel"):
                slicer.mrmlScene.RemoveNode(node)
        if (
            hasattr(self, "_stentNode")
            and self._stentNode
            and slicer.mrmlScene.IsNodePresent(self._stentNode)
        ):
            slicer.mrmlScene.RemoveNode(self._stentNode)
            for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
                dispNode = node.GetDisplayNode()
                if dispNode:
                    dispNode.SetOpacity(1.0)
        self._stentNode = None
        if (
            hasattr(self, "_kissFrameNode")
            and self._kissFrameNode
            and slicer.mrmlScene.IsNodePresent(self._kissFrameNode)
        ):
            slicer.mrmlScene.RemoveNode(self._kissFrameNode)
        self._kissFrameNode = None


    def classifySurfaceBranches(
        self, poly_data, curvature_threshold=0.5, main_len_gate_mm=0.0
    ):
        """
        Classify every surface vertex by proximity to the nearest centerline
        branch, then colorize the mesh per branch.

        The vessel mesh is one connected piece — we do NOT rely on connectivity
        splitting.  Instead we build a KD-tree over all centerline points
        (tagged by branch index) and assign each surface vertex to whichever
        branch centerline is closest in 3D.  The ostium curvature threshold is
        used only for counting / display; the color boundaries emerge naturally
        from the Voronoi-like nearest-centerline assignment.

        Returns dict:
          'poly'            : vtkPolyData with 'BranchClassification' RGB point array
          'classifications' : {bi: {'label', 'length_mm', 'mean_radius_mm'}}
          'ostium_count'    : number of high-curvature surface pts (informational)
        """
        import vtk

        try:
            import numpy as _np
            from scipy.spatial import cKDTree as _cKDTree
        except ImportError:
            raise RuntimeError(
                "scipy is required. Install via: "
                "pip install scipy --break-system-packages"
            )

        # ── 1. Mean curvature (informational — counts ostium ring pts) ────────
        curv = vtk.vtkCurvatures()
        curv.SetInputData(poly_data)
        curv.SetCurvatureTypeToMean()
        curv.Update()
        curved_poly = curv.GetOutput()

        curv_arr = curved_poly.GetPointData().GetArray("Mean_Curvature")
        ostium_count = 0
        if curv_arr:
            for i in range(curv_arr.GetNumberOfTuples()):
                if abs(curv_arr.GetValue(i)) >= curvature_threshold:
                    ostium_count += 1
        print(
            f"[ClassifySurface] Ostium ring pts "
            f"(|curv|>={curvature_threshold}): {ostium_count}"
        )

        # ── 2. Build per-branch centerline segment arrays ────────────────────
        #
        # New approach: centerline projection + axial gating (replaces KD-tree
        # nearest-point + heuristic overrides).
        #
        # For each surface vertex we:
        #   a) Project onto every branch centerline (line-segment projection)
        #   b) Gate: only accept projections that fall WITHIN the branch's
        #      valid arc window [ostiumGi .. branch_end]
        #   c) Among valid candidates, pick the one with smallest perpendicular
        #      distance (+ direction alignment tie-break)
        #   d) Fall back to trunk ONLY when no branch wins — not as a first step
        #
        # This eliminates the twin problems of the old approach:
        #   • 15mm seed-skip → coarse coverage near ostia
        #   • trunk-first bias → branches losing their own surface area
        #
        PERP_MAX_MM = 35.0  # max perpendicular distance to claim a vertex
        DOT_MIN = -0.3  # min dot(vertex→proj, branch_dir): rejects back-projections
        BIF_DOME_MM = 10.0  # tight bifurcation dome (reduced from 22mm)

        meta = getattr(self, "branchMeta", {})

        # Collect per-branch centerline point arrays (dense — all points, no skip)
        trunk_bs, trunk_be = self.branches[0]
        trunk_pts_arr = _np.array(
            [
                self.points[gi]
                for gi in range(trunk_bs, trunk_be)
                if gi < len(self.points)
            ],
            dtype=float,
        )

        if len(trunk_pts_arr) < 2:
            raise RuntimeError("No centerline points. Load centerline first.")

        # Per-branch: store (pts_array, ostiumGi, start_gi, end_gi)
        branch_cls = []  # index matches self.branches index (0=trunk)
        for bi, (bs, be) in enumerate(self.branches):
            pts_arr = _np.array(
                [self.points[gi] for gi in range(bs, be) if gi < len(self.points)],
                dtype=float,
            )
            ogi = meta.get(bi, {}).get("ostiumGi", bs)
            if ogi is None:
                ogi = bs  # suppressed branch — treat raw start as ostium
            # Convert ostiumGi (global) to local index within pts_arr
            ogi_local = max(0, ogi - bs)
            # CaudalDepartureSnap: for branches that snap ostiumGi to the bif
            # node (ki=0) because they depart caudally, classifySurfaceBranches
            # must still start Voronoi competition PAST the shared dome zone
            # (at first_k, the BIF_DOME_MM crossing).  Without this, the
            # caudal-departure branch competes from ki=0 (the exact bif node)
            # and wins surface vertices that belong to the sibling iliac,
            # erasing its surface coloring near the bifurcation.
            surf_ogi_local = meta.get(bi, {}).get("surf_classify_ogi_local")
            if surf_ogi_local is not None:
                ogi_local = max(ogi_local, surf_ogi_local)
            branch_cls.append(
                {"pts": pts_arr, "ogi_local": ogi_local, "bs": bs, "be": be}
            )

        # ── 3. Project every surface vertex onto every branch ─────────────────
        n_pts = curved_poly.GetNumberOfPoints()
        surf_pts = _np.array([curved_poly.GetPoint(i) for i in range(n_pts)])

        total_cl_pts = sum(len(b["pts"]) for b in branch_cls)
        total_branch_pts = sum(len(b["pts"]) for b in branch_cls[1:])
        print(
            f"[ClassifySurface] Projection: {total_cl_pts} total centerline pts "
            f"({len(trunk_pts_arr)} trunk, {total_branch_pts} branch, dense), "
            f"{len(self.branches)} branches"
        )

        def _project_to_segments(surf_v, pts):
            """Vectorised projection of surf_v (N,3) onto polyline pts (M,3).
            Returns (min_perp_dist per vertex, local_t index of closest segment)."""
            segs_a = pts[:-1]  # (M-1, 3)
            segs_b = pts[1:]  # (M-1, 3)
            ab = segs_b - segs_a  # (M-1, 3)
            ab_len2 = (ab * ab).sum(axis=1)  # (M-1,)
            ab_len2 = _np.maximum(ab_len2, 1e-12)

            # For each vertex, for each segment: t = dot(v-a, ab) / |ab|^2
            # surf_v: (N,3), segs_a: (M-1,3) → broadcast → (N, M-1)
            v_minus_a = (
                surf_v[:, _np.newaxis, :] - segs_a[_np.newaxis, :, :]
            )  # (N,M-1,3)
            t = (v_minus_a * ab[_np.newaxis, :, :]).sum(axis=2) / ab_len2[
                _np.newaxis, :
            ]
            t = _np.clip(t, 0.0, 1.0)  # (N, M-1)

            # Closest point on each segment
            proj = (
                segs_a[_np.newaxis, :, :] + t[:, :, _np.newaxis] * ab[_np.newaxis, :, :]
            )
            diff = surf_v[:, _np.newaxis, :] - proj  # (N, M-1, 3)
            perp2 = (diff * diff).sum(axis=2)  # (N, M-1)

            best_seg = _np.argmin(perp2, axis=1)  # (N,)
            best_perp = _np.sqrt(perp2[_np.arange(len(surf_v)), best_seg])
            return best_perp, best_seg

        # Assign: vert_bi = -1 means unassigned (falls to trunk)
        vert_bi = _np.full(n_pts, -1, dtype=int)
        vert_best_d = _np.full(n_pts, _np.inf)

        # ── Pre-compute ghost distance for noise-disqualification ────────────
        # Discarded noise branches leave ghost centerline points in self.points.
        # Instead of overriding results after classification (which fights the
        # projection), we disqualify ghost-dominated vertices DURING competition:
        # a branch cannot claim a vertex if that vertex is closer to a ghost
        # centerline than to the branch itself.  The vertex stays unassigned
        # and falls to trunk as a natural fallback — not a hard override.
        # ALSO: beyond-trunk ghost pts (e.g. suprarenal IVC) accurately
        # trace vessel surface above trunk gi0.  We concatenate them onto
        # trunk_pts_arr so the trunk seeding pass covers that region —
        # otherwise 2000+ vertices fall unassigned instead of seeded directly.
        _discarded_ranges = getattr(self, "_discarded_gi_ranges", [])
        _d_ghost = None
        if _discarded_ranges:
            _ghost_pts = []
            for dbs, dbe in _discarded_ranges:
                for gi in range(dbs, dbe):
                    if gi < len(self.points):
                        _ghost_pts.append(self.points[gi])
            if _ghost_pts:
                from scipy.spatial import cKDTree as _cKDTree

                _ghost_arr = _np.array(_ghost_pts, dtype=float)
                _ghost_tree = _cKDTree(_ghost_arr)
                _d_ghost, _ = _ghost_tree.query(surf_pts)
                print(
                    f"[ClassifySurface] Ghost centerlines: {len(_ghost_pts)} pts from "
                    f"{len(_discarded_ranges)} discarded branch(es) — used as disqualifier"
                )
                # Extend trunk centerline with ghost pts so suprarenal
                # surface vertices are seeded by the trunk pass rather
                # than left unassigned.  Sort Z descending (proximal-first)
                # so the polyline is coherent for segment projection.
                _ghost_sorted = _ghost_arr[_np.argsort(_ghost_arr[:, 2])[::-1]]
                trunk_pts_arr = _np.vstack([trunk_pts_arr, _ghost_sorted])
                print(
                    f"[ClassifySurface] Trunk extended with {len(_ghost_pts)} ghost pts "
                    f"(Z={_ghost_arr[:,2].min():.0f}→{_ghost_arr[:,2].max():.0f})"
                )

        # Ghost Z bounds — used to exempt co-located real branches from Gate 4
        _ghost_z_min = float(_ghost_arr[:, 2].min()) if _d_ghost is not None else None
        _ghost_z_max = float(_ghost_arr[:, 2].max()) if _d_ghost is not None else None
        # --- Trunk pass: seed trunk ownership for all vertices ---
        trunk_perp, trunk_seg = _project_to_segments(surf_pts, trunk_pts_arr)
        trunk_mask = trunk_perp <= PERP_MAX_MM
        vert_bi[trunk_mask] = 0
        vert_best_d[trunk_mask] = trunk_perp[trunk_mask]

        # --- Branch pass: each branch competes from ostium onward ---
        for bi in range(1, len(self.branches)):
            b = branch_cls[bi]
            pts = b["pts"]
            ogi_l = b["ogi_local"]
            if len(pts) < 2 or ogi_l >= len(pts) - 1:
                continue
            # Only use centerline points from ostium onward (no proximal skip
            # needed — axial gating via segment index handles overlap)
            pts_valid = pts[ogi_l:]
            if len(pts_valid) < 2:
                continue
            perp, seg = _project_to_segments(surf_pts, pts_valid)
            # Gate 1: perpendicular distance
            cand = perp < PERP_MAX_MM
            # Gate 2: vertex must project onto a segment WITHIN the valid arc
            # (seg index relative to pts_valid — no upstream bleed)
            cand &= seg < len(pts_valid) - 1
            # Gate 3: beat current winner
            cand &= perp < vert_best_d
            # Gate 4: ghost disqualification — if a vertex is closer to a
            # discarded noise centerline than to this branch, the branch cannot
            # claim it.  Exemption: branches whose ostium lies inside the ghost
            # Z band are real vessels co-located with the ghost (e.g. renal vein
            # overlapping a beyond-trunk stub) and must compete freely there.
            if _d_ghost is not None:
                _ogi_z = float(pts[ogi_l, 2]) if ogi_l < len(pts) else None
                _in_ghost_band = (
                    _ogi_z is not None
                    and _ghost_z_min is not None
                    and (_ghost_z_min - 15.0) <= _ogi_z <= (_ghost_z_max + 15.0)
                )
                if not _in_ghost_band:
                    cand &= perp <= _d_ghost
                else:
                    print(
                        f"[ClassifySurface] Branch {bi}: ghost gate exempted "
                        f"(ostium Z={_ogi_z:.0f} inside ghost band "
                        f"Z={_ghost_z_min:.0f}–{_ghost_z_max:.0f})"
                    )
            vert_bi[cand] = bi
            vert_best_d[cand] = perp[cand]

        # Any still-unassigned vertices → trunk (trunk is fallback, not a sink)
        _unassigned = int((vert_bi == -1).sum())
        vert_bi[vert_bi == -1] = 0
        if _unassigned:
            print(
                f"[ClassifySurface] Trunk fallback: {_unassigned} unassigned vertices → TRUNK"
            )

        # ── Tight bifurcation dome (trunk reclaims carina only) ───────────────
        # Much smaller than before (10mm vs 22mm) — only captures the true
        # carina where branch ostia genuinely overlap.
        # IliacDomeExemption: vertices closer to a confirmed iliac wall-snap
        # point (ostium_p3 from IliacSurfaceSnap) than to the bifurcation are
        # exempted from reclaim — they are iliac surface, not carina.
        _bif_pt = getattr(self, "bifurcationPoint", None)
        if _bif_pt is not None:
            _bif_arr = _np.array(_bif_pt, dtype=float)
            _d_bif = _np.linalg.norm(surf_pts - _bif_arr, axis=1)
            _dome_mask = _d_bif <= BIF_DOME_MM

            # Build exemption mask: vertices closer to any main branch's
            # confirmed wall-snap point than to the bifurcation point are
            # kept as that branch, not forced to trunk.
            _exempt_mask = _np.zeros(n_pts, dtype=bool)
            for _ebi, (_ebs, _ebe) in enumerate(self.branches):
                _emeta = meta.get(_ebi, {})
                _erole = _emeta.get("role", "")
                if _erole not in ("main", "iliac_left", "iliac_right"):
                    continue
                _ep3 = _emeta.get("ostium_p3")
                if _ep3 is None:
                    continue
                _ep3_arr = _np.array(_ep3, dtype=float)
                _d_wall = _np.linalg.norm(surf_pts - _ep3_arr, axis=1)
                # Exempt if closer to the wall-snap point than to the bif
                # AND currently assigned to this branch by Voronoi
                _this_exempt = _dome_mask & (_d_wall < _d_bif) & (vert_bi == _ebi)
                if _this_exempt.any():
                    _exempt_mask |= _this_exempt
                    print(
                        f"[ClassifySurface] IliacDomeExemption bi={_ebi} "
                        f"({self.getBranchDisplayName(_ebi)}): "
                        f"{int(_this_exempt.sum())} verts exempted "
                        f"(closer to wall_pt {[round(float(x),1) for x in _ep3]} "
                        f"than to bif)"
                    )

            _reclaim_mask = _dome_mask & ~_exempt_mask
            _forced = int(_reclaim_mask.sum())
            if _forced:
                vert_bi[_reclaim_mask] = 0
                print(
                    f"[ClassifySurface] Bifurcation dome: "
                    f"r={BIF_DOME_MM}mm around bif {_bif_arr.round(1).tolist()} "
                    f"→ reclaimed {_forced} carina vertices to TRUNK "
                    f"({int(_exempt_mask.sum())} iliac verts exempted)"
                )
            elif _dome_mask.any():
                print(
                    f"[ClassifySurface] Bifurcation dome: "
                    f"all {int(_dome_mask.sum())} dome verts exempted (iliac wall-snap)"
                )
        # ─────────────────────────────────────────────────────────────────────

        print(f"[ClassifySurface] Surface vertices: {n_pts}")
        from collections import Counter as _Counter

        bi_counts = _Counter(vert_bi.tolist())
        for bi_k, cnt in sorted(bi_counts.items()):
            print(f"  {self.getBranchDisplayName(bi_k)}: {cnt} vertices")

        # ── 4. Per-branch geometry from assigned surface vertices ──────────────
        classifications = {}
        for bi in range(len(self.branches)):
            mask = vert_bi == bi
            if not mask.any():
                continue
            pts = surf_pts[mask]
            centroid = pts.mean(axis=0)

            _, _, Vt = _np.linalg.svd(pts - centroid, full_matrices=False)
            main_axis = Vt[0]
            projected = pts @ main_axis
            length_mm = float(projected.max() - projected.min())

            perp_dists = _np.linalg.norm(
                pts
                - centroid
                - _np.outer(pts @ main_axis - centroid @ main_axis, main_axis),
                axis=1,
            )
            mean_radius = float(perp_dists.mean())

            # Classification: trust branchMeta role for all known values.
            # 'trunk' → trunk, 'main'/'side'/'noise'/'renal_vein' → branch.
            # Geometric fallback only when role is absent entirely.
            # NOTE: 'noise' and 'renal_vein' must be listed here explicitly —
            # omitting them causes elongated noise stubs (e.g. beyond-trunk
            # branches with L>80mm, elongation>6) to be falsely promoted to
            # 'trunk' by the geometric fallback, producing blue surface coloring
            # for vessels that should be orange/cyan.
            role = meta.get(bi, {}).get("role", "")
            if role == "trunk":
                label = "trunk"
            elif role in (
                "main",
                "iliac_right",
                "iliac_left",
                "side",
                "noise",
                "renal_vein",
                "renal_fragment",
            ):
                label = "branch"
            elif bi == 0:
                label = "trunk"  # index 0 is always trunk
            else:
                # No role metadata — use geometry
                elongation = length_mm / max(mean_radius, 0.1)
                label = "trunk" if elongation > 6.0 and length_mm > 80 else "branch"

            classifications[bi] = {
                "label": label,
                "length_mm": round(length_mm, 1),
                "mean_radius_mm": round(mean_radius, 1),
            }
            print(
                f"[ClassifySurface]   {self.getBranchDisplayName(bi)}: "
                f"{label.upper()}, L={length_mm:.0f}mm, "
                f"r≈{mean_radius:.1f}mm, verts={mask.sum()}"
            )

        # ── 5. Assign colors — 4 display groups ──────────────────────────────
        #
        # Rules (driven purely by branchMeta role + optional length gate):
        #   role='trunk'               → cornflower blue
        #   role='main' AND len>=gate  → one distinct color each (iliac limbs)
        #   role='main' but len<gate   → demoted to side group (orange)
        #   role='side' / anything else→ side group (orange)
        #

        TRUNK_COLOR = (100, 149, 237)  # cornflower blue
        RENAL_COLOR = (80, 200, 220)  # cyan — renal veins
        SIDE_COLOR = (180, 120, 60)  # muted orange
        MAIN_COLORS = [
            (220, 80, 60),  # red   — first main limb
            (50, 180, 100),  # green — second main limb
        ]
        main_color_idx = 0
        bi_to_color = {}

        # ── Geometric rescue thresholds ───────────────────────────────────────
        # branchMeta roles can be absent or wrong when the tagging pipeline runs
        # before IliacLabel / RenalTag propagate (e.g. suprarenal IVC continuation
        # gets role="" or role="side" but has r≈37mm >> iliac r≈9mm).
        # Compute mean radius of confirmed main/iliac branches as a baseline.
        _iliac_radii = [
            info["mean_radius_mm"]
            for bi, info in classifications.items()
            if meta.get(bi, {}).get("role", "") in ("main", "iliac_right", "iliac_left")
            and info["mean_radius_mm"] > 0
        ]
        if _iliac_radii:
            _mean_iliac_r = sum(_iliac_radii) / len(_iliac_radii)
        else:
            # No tagged iliacs yet — derive a proxy from the two largest non-trunk
            # branches by radius (they are most likely the true iliacs).
            # This prevents the hardcoded 9.0 fallback from producing a threshold
            # that catches mid-sized side branches as renal.
            _proxy_radii = sorted(
                [
                    info["mean_radius_mm"]
                    for bi, info in classifications.items()
                    if bi != 0 and info["mean_radius_mm"] > 0
                    and classifications[bi]["length_mm"] >= 40.0
                ],
                reverse=True,
            )[:2]
            _mean_iliac_r = sum(_proxy_radii) / len(_proxy_radii) if _proxy_radii else 9.0
            print(
                f"[ClassifySurface] _mean_iliac_r: no tagged iliacs — "
                f"proxy from top-2 branches = {_mean_iliac_r:.1f}mm "
                f"(radii={[round(r,1) for r in _proxy_radii]})"
            )

        # Renal rescue: branches with no iliac/trunk/renal role, short arc (<50mm),
        # and small radius (<2/3 of iliac mean) that depart laterally are likely
        # renal veins that were tagged "side" by the pipeline.
        _RENAL_ROLES = {"renal_vein", "renal_right", "renal_left", "renal_fragment"}
        _MAIN_ROLES  = {"main", "iliac_right", "iliac_left"}

        # Endpoint labels are the strongest signal for renal veins.  The
        # morphology/GeoRescue passes below are geometric fallbacks; if the
        # centerline endpoint is explicitly labelled renal, tag it as renal now
        # so display coloring and downstream correction agree with the scene.
        _endpoint_renal_anchors = {}
        try:
            _ep_map = getattr(self, "branchEndpointMap", {})
            _ep_node = None
            import slicer as _sl
            for _n in _sl.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
                if "endpoint" in _n.GetName().lower():
                    _ep_node = _n
                    break
            if _ep_node and _ep_map:
                for _bi_ep, _ep_idx in _ep_map.items():
                    if _ep_idx >= _ep_node.GetNumberOfControlPoints():
                        continue
                    _ep_label_raw = _ep_node.GetNthControlPointLabel(_ep_idx)
                    _ep_desc_raw = _ep_node.GetNthControlPointDescription(_ep_idx)
                    _ep_text = f"{_ep_label_raw} {_ep_desc_raw}".lower()
                    if "renal" not in _ep_text or _bi_ep == 0:
                        continue
                    _lat = "Right" if "right" in _ep_text else (
                        "Left" if "left" in _ep_text else ""
                    )
                    _endpoint_renal_anchors[_bi_ep] = _lat
                    _old_role = meta.get(_bi_ep, {}).get("role", "")
                    _rv_name = f"{_lat} Renal Vein" if _lat else f"Renal Vein {_bi_ep}"
                    meta.setdefault(_bi_ep, {}).update({
                        "role":          "renal_vein",
                        "label":         _rv_name,
                        "display_name":  _rv_name,
                        "name":          _rv_name,
                    })
                    if _lat:
                        meta[_bi_ep]["lateral_label"] = _lat
                    if hasattr(self, "branchMeta"):
                        self.branchMeta.setdefault(_bi_ep, {}).update(meta[_bi_ep])
                    print(
                        f"[ClassifySurface] RenalEndpoint: bi={_bi_ep} "
                        f"endpoint='{_ep_label_raw or _ep_desc_raw}' "
                        f"{_old_role!r} -> 'renal_vein'"
                    )
        except Exception as _ree:
            print(f"[ClassifySurface] RenalEndpoint check failed: {_ree}")

        # ── Fix iliac left/right role assignment by tip X coordinate ─────────
        # Branches are detected structurally, but left/right roles were stamped
        # without comparing the two iliacs against each other — both ended up as
        # 'iliac_left'.  Re-derive roles now from tip X sign (RAS: negative X =
        # patient's left, positive X = patient's right).
        _iliac_bis = [
            bi for bi in classifications
            if meta.get(bi, {}).get("role", "") in _MAIN_ROLES
            and classifications[bi]["length_mm"] >= main_len_gate_mm
        ]
        if len(_iliac_bis) >= 2:
            # Sort ascending by tip X: index 0 = most-right (negative X / smaller X), index 1 = most-left (positive X / larger X)
            _iliac_bis.sort(key=lambda b: self.points[self.branches[b][1] - 1][0], reverse=False)
            _role_seq    = ["iliac_right", "iliac_left"]
            _lateral_seq = ["Right",       "Left"]
            for _rank, _bi in enumerate(_iliac_bis):
                _new_role = _role_seq[min(_rank, 1)]
                _new_lat  = _lateral_seq[min(_rank, 1)]
                _old_role = meta.get(_bi, {}).get("role", "")
                meta.setdefault(_bi, {})["role"]          = _new_role
                meta.setdefault(_bi, {})["lateral_label"] = _new_lat
                
                # Update UI labels so getBranchDisplayName() reflects the fix
                _new_name = f"{_new_lat} Iliac"
                meta[_bi]["label"] = _new_name
                meta[_bi]["display_name"] = _new_name
                meta[_bi]["name"] = _new_name
                
                print(
                    f"[ClassifySurface] IliacFix: bi={_bi}"
                    f"  tip_X={self.points[self.branches[_bi][1]-1][0]:+.1f}"
                    f"  {_old_role!r} → {_new_role!r}  lateral={_new_lat!r}  name={_new_name!r}"
                )
            # Propagate corrected roles into branchMeta (same object via reference)
            if hasattr(self, "branchMeta"):
                for _bi in _iliac_bis:
                    self.branchMeta.setdefault(_bi, {}).update(meta[_bi])
        # ── end IliacFix ──────────────────────────────────────────────────────

        # ── Morphological iliac promotion ─────────────────────────────────────
        # IliacFix only corrects left/right *within* the set of branches already
        # carrying an iliac/main role in branchMeta.  It cannot fix the upstream
        # error where the centerline pipeline tagged the wrong branches as iliacs
        # (e.g. a short stub gets role='iliac_left' while a long, fat limb stays
        # role='side').  Detect this by comparing the 2 largest non-trunk
        # branches by mean_radius_mm against the currently-tagged iliacs.
        # If the top-2-by-radius set is NOT already the iliac set AND their
        # mean radius exceeds the tagged-iliac mean by >15%, promote them and
        # demote the mis-tagged ones to 'side'.
        # Gate: candidates must be ≥40 mm long (true iliacs are never stubs).
        _MORPH_RADIUS_RATIO = 1.15   # 15 % larger mean radius → suspect mismatch
        _MORPH_MIN_LEN_MM   = 40.0   # minimum length to be considered an iliac

        # MorphRescue must never promote a branch that is endpoint-anchored to a
        # renal vein.  RenalEndpoint above already tags these branches; keep the
        # explicit set here as a second guard for iliac promotion.
        _morph_renal_anchored = set(_endpoint_renal_anchors)
        try:
            for _bi_ep in sorted(_morph_renal_anchored):
                print(
                    f"[MorphRescue] bi={_bi_ep} endpoint-anchored to renal vein "
                    f"→ excluded from iliac promotion"
                )
        except Exception as _mre:
            print(f"[MorphRescue] endpoint-anchor check failed: {_mre}")

        _all_branch_bis = [
            bi for bi in classifications
            if bi != 0
            and classifications[bi]["length_mm"] >= _MORPH_MIN_LEN_MM
            and bi not in _morph_renal_anchored   # never promote a confirmed renal
        ]
        _sorted_by_r = sorted(
            _all_branch_bis,
            key=lambda b: classifications[b]["mean_radius_mm"],
            reverse=True,
        )
        _top2 = _sorted_by_r[:2]

        if len(_top2) == 2:
            _top2_roles = {meta.get(b, {}).get("role", "") for b in _top2}
            _top2_already_iliac = _top2_roles <= _MAIN_ROLES  # both in main roles

            if not _top2_already_iliac:
                _top2_r_mean = sum(
                    classifications[b]["mean_radius_mm"] for b in _top2
                ) / 2.0
                # _mean_iliac_r was computed above from currently-tagged iliacs
                if _top2_r_mean > _mean_iliac_r * _MORPH_RADIUS_RATIO and _mean_iliac_r > 0:
                    print(
                        f"[ClassifySurface] MorphRescue: top-2-by-radius "
                        f"(bi={_top2}, r_mean={_top2_r_mean:.1f}mm) > "
                        f"{_MORPH_RADIUS_RATIO}× tagged-iliac mean "
                        f"({_mean_iliac_r:.1f}mm) — re-assigning iliac roles."
                    )
                    # Demote existing iliac-tagged branches → side.
                    # Also clear any stale "R-Iliac"/"L-Iliac" display names that
                    # IliacFix wrote earlier so they don't duplicate the promoted
                    # branches in the debug output.
                    for _b in list(_iliac_bis):
                        _old = meta.get(_b, {}).get("role", "")
                        _demoted_name = f"Branch-{_b}"
                        meta.setdefault(_b, {}).update({
                            "role":         "side",
                            "label":        _demoted_name,
                            "display_name": _demoted_name,
                            "name":         _demoted_name,
                        })
                        if hasattr(self, "branchMeta"):
                            self.branchMeta.setdefault(_b, {}).update(meta[_b])
                        print(
                            f"[ClassifySurface] MorphRescue: demoted bi={_b} "
                            f"({_demoted_name}) "
                            f"{_old!r} → 'side'  "
                            f"(r={classifications[_b]['mean_radius_mm']:.1f}mm, "
                            f"L={classifications[_b]['length_mm']:.0f}mm)"
                        )

                    # Promote top-2 by radius; assign left/right by tip X (RAS)
                    _top2_by_x = sorted(
                        _top2,
                        key=lambda b: self.points[self.branches[b][1] - 1][0],
                    )
                    _role_seq    = ["iliac_right", "iliac_left"]
                    _lateral_seq = ["Right",        "Left"]
                    for _rank, _bi in enumerate(_top2_by_x):
                        _new_role = _role_seq[_rank]
                        _new_lat  = _lateral_seq[_rank]
                        _new_name = f"{_new_lat} Iliac"
                        meta.setdefault(_bi, {}).update({
                            "role":          _new_role,
                            "lateral_label": _new_lat,
                            "label":         _new_name,
                            "display_name":  _new_name,
                            "name":          _new_name,
                        })
                        if hasattr(self, "branchMeta"):
                            self.branchMeta.setdefault(_bi, {}).update(meta[_bi])
                        print(
                            f"[ClassifySurface] MorphRescue: promoted bi={_bi} "
                            f"({self.getBranchDisplayName(_bi)}) → {_new_role!r}  "
                            f"tip_X={self.points[self.branches[_bi][1]-1][0]:+.1f}  "
                            f"r={classifications[_bi]['mean_radius_mm']:.1f}mm, "
                            f"L={classifications[_bi]['length_mm']:.0f}mm"
                        )

                    # Rebuild _iliac_bis and _mean_iliac_r for the color loop
                    _iliac_bis = _top2_by_x
                    _mean_iliac_r = sum(
                        classifications[b]["mean_radius_mm"] for b in _iliac_bis
                    ) / len(_iliac_bis)
        # ── end MorphRescue ───────────────────────────────────────────────────

        # ── IliacEndpoint correction ──────────────────────────────────────────
        # Surface coloring runs before the widget-level IliacCorrect hook, so do
        # the same hard endpoint snap here.  This keeps the printed color groups
        # and surface RGB map aligned with the final anatomical branch set.
        try:
            _ie_right_ep = None
            _ie_left_ep = None
            _ie_node = None
            import slicer as _sl
            for _n in _sl.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
                if "endpoint" in _n.GetName().lower():
                    _ie_node = _n
                    break
            if _ie_node is not None:
                _p = [0.0, 0.0, 0.0]
                for _ci in range(_ie_node.GetNumberOfControlPoints()):
                    _txt = (
                        f"{_ie_node.GetNthControlPointLabel(_ci)} "
                        f"{_ie_node.GetNthControlPointDescription(_ci)}"
                    ).lower()
                    _ie_node.GetNthControlPointPositionWorld(_ci, _p)
                    if "right iliac" in _txt:
                        _ie_right_ep = (float(_p[0]), float(_p[1]), float(_p[2]))
                    elif "left iliac" in _txt:
                        _ie_left_ep = (float(_p[0]), float(_p[1]), float(_p[2]))

            def _ie_dist(a, b):
                return (
                    (float(a[0]) - b[0]) ** 2
                    + (float(a[1]) - b[1]) ** 2
                    + (float(a[2]) - b[2]) ** 2
                ) ** 0.5

            def _ie_tip_dist(_bi, _ep):
                if _ep is None or _bi >= len(self.branches):
                    return 9999.0
                _bs, _be = self.branches[_bi][0], self.branches[_bi][1] - 1
                _ds = _ie_dist(self.points[_bs], _ep) if _bs < len(self.points) else 9999.0
                _de = _ie_dist(self.points[_be], _ep) if _be < len(self.points) else 9999.0
                return min(_ds, _de)

            def _ie_candidate(_bi):
                _role = meta.get(_bi, {}).get("role", "")
                return (
                    _bi != 0
                    and _bi not in _endpoint_renal_anchors
                    and _role not in _RENAL_ROLES
                    and _role != "trunk"
                )

            _IE_MATCH_MM = 15.0
            _ie_right_bi = None
            _ie_left_bi = None
            _ie_right_cands = sorted(
                [(_ie_tip_dist(_bi, _ie_right_ep), _bi) for _bi in classifications if _ie_candidate(_bi)],
                key=lambda _t: _t[0],
            )
            _ie_left_cands = sorted(
                [(_ie_tip_dist(_bi, _ie_left_ep), _bi) for _bi in classifications if _ie_candidate(_bi)],
                key=lambda _t: _t[0],
            )
            if _ie_right_cands and _ie_right_cands[0][0] < _IE_MATCH_MM:
                _ie_right_bi = _ie_right_cands[0][1]
            if _ie_left_cands and _ie_left_cands[0][0] < _IE_MATCH_MM:
                _ie_left_bi = _ie_left_cands[0][1]

            if _ie_right_bi is not None or _ie_left_bi is not None:
                print(
                    f"[ClassifySurface] IliacEndpoint: "
                    f"right_bi={_ie_right_bi}  left_bi={_ie_left_bi}"
                )
                if _ie_right_bi is not None:
                    meta.setdefault(_ie_right_bi, {}).update({
                        "role":          "iliac_right",
                        "lateral_label": "Right",
                        "label":         "Right Iliac",
                        "display_name":  "Right Iliac",
                        "name":          "Right Iliac",
                    })
                if _ie_left_bi is not None:
                    meta.setdefault(_ie_left_bi, {}).update({
                        "role":          "iliac_left",
                        "lateral_label": "Left",
                        "label":         "Left Iliac",
                        "display_name":  "Left Iliac",
                        "name":          "Left Iliac",
                    })

                _assigned = {_b for _b in (_ie_right_bi, _ie_left_bi) if _b is not None}
                for _bi in list(classifications):
                    if _bi in _assigned or _bi == 0 or _bi in _endpoint_renal_anchors:
                        continue
                    _role = meta.get(_bi, {}).get("role", "")
                    if _role not in _MAIN_ROLES:
                        continue
                    _dr = _ie_tip_dist(_bi, _ie_right_ep)
                    _dl = _ie_tip_dist(_bi, _ie_left_ep)
                    _frag_role = None
                    _parent = None
                    if _ie_right_bi is not None and _dr < _dl and _dr <= 80.0:
                        _frag_role, _parent = "iliac_right", _ie_right_bi
                    elif _ie_left_bi is not None and _dl < _dr and _dl <= 80.0:
                        _frag_role, _parent = "iliac_left", _ie_left_bi
                    if _frag_role is None:
                        continue
                    _lat = "Right" if _frag_role == "iliac_right" else "Left"
                    meta.setdefault(_bi, {}).update({
                        "role":          "iliac_fragment",
                        "fragment_of":   _parent,
                        "iliac_role":    _frag_role,
                        "lateral_label": _lat,
                        "label":         f"{_lat} Iliac Fragment",
                        "display_name":  f"{_lat} Iliac Fragment",
                        "name":          f"{_lat} Iliac Fragment",
                    })
                    print(
                        f"[ClassifySurface] IliacEndpoint: bi={_bi} "
                        f"marked iliac_fragment of bi={_parent} "
                        f"(d_right={_dr:.1f}mm, d_left={_dl:.1f}mm)"
                    )

                if hasattr(self, "branchMeta"):
                    for _bi in classifications:
                        self.branchMeta.setdefault(_bi, {}).update(meta.get(_bi, {}))
                _iliac_bis = [
                    _b for _b in (_ie_right_bi, _ie_left_bi)
                    if _b is not None
                ]
                if _iliac_bis:
                    _mean_iliac_r = sum(
                        classifications[_b]["mean_radius_mm"] for _b in _iliac_bis
                    ) / len(_iliac_bis)
        except Exception as _iee:
            print(f"[ClassifySurface] IliacEndpoint correction failed: {_iee}")
        # ── end IliacEndpoint correction ──────────────────────────────────────

        # ── GeoRescue RENAL pre-computation ──────────────────────────────────
        # Derive the renal Z band and trunk X centroid from trunk branch (bi=0).
        # Renal veins originate in the middle 20–80 % of the IVC trunk span —
        # outside that band the branch is a bifurcation stub or iliac fragment.
        # These values are used inside the color loop below.
        _gr_trunk_bs, _gr_trunk_be = self.branches[0]
        _gr_trunk_gi_range = [
            gi for gi in range(_gr_trunk_bs, _gr_trunk_be)
            if gi < len(self.points)
        ]
        if _gr_trunk_gi_range:
            _gr_z_vals = [self.points[gi][2] for gi in _gr_trunk_gi_range]
            _gr_z_min  = min(_gr_z_vals)
            _gr_z_max  = max(_gr_z_vals)
            _gr_span   = _gr_z_max - _gr_z_min
            _gr_z_lo   = _gr_z_min + 0.20 * _gr_span   # 20 % above bif
            _gr_z_hi   = _gr_z_min + 0.80 * _gr_span   # 80 % below inlet
            _gr_x_cen  = sum(
                self.points[gi][0] for gi in _gr_trunk_gi_range
            ) / len(_gr_trunk_gi_range)
        else:
            _gr_z_lo, _gr_z_hi, _gr_x_cen = 0.0, 9999.0, 0.0
            _gr_z_min,  _gr_z_max = 0.0, 9999.0

        _GR_RENAL_MIN_LEN = 40.0   # mm — stubs shorter than this cannot be renal
        _GR_RENAL_MIN_LAT = 8.0    # mm — minimum lateral offset from trunk axis
        _gr_renal_promoted = {}    # bi → score, used by post-loop cap

        print(
            f"[GeoRescue] Renal Z band: [{_gr_z_lo:.1f}, {_gr_z_hi:.1f}]  "
            f"trunk span=[{_gr_z_min:.1f}, {_gr_z_max:.1f}]  "
            f"trunk_x_cen={_gr_x_cen:.1f}mm"
        )
        # ── end GeoRescue pre-computation ────────────────────────────────────

        for bi, info in classifications.items():
            role   = meta.get(bi, {}).get("role", "")
            len_mm = info["length_mm"]
            r_mm   = info["mean_radius_mm"]

            if role == "trunk":
                bi_to_color[bi] = TRUNK_COLOR

            elif role in _RENAL_ROLES:
                bi_to_color[bi] = RENAL_COLOR

            elif role in _MAIN_ROLES and len_mm >= main_len_gate_mm:
                # Color by tip-X rank so left iliac is always MAIN_COLORS[0]
                # and right iliac is always MAIN_COLORS[1], regardless of bi order.
                _rank = _iliac_bis.index(bi) if bi in _iliac_bis else main_color_idx
                bi_to_color[bi] = MAIN_COLORS[_rank % len(MAIN_COLORS)]
                main_color_idx += 1

            elif role == "iliac_fragment":
                _parent = meta.get(bi, {}).get("fragment_of")
                _rank = _iliac_bis.index(_parent) if _parent in _iliac_bis else 0
                bi_to_color[bi] = MAIN_COLORS[_rank % len(MAIN_COLORS)]

            else:
                # ── Geometric rescue ─────────────────────────────────────────
                # 1. Trunk rescue: role is missing/wrong but radius is >> iliac
                #    → suprarenal IVC continuation mis-labelled as side/noise.
                if r_mm > _mean_iliac_r * 1.8 and role not in _MAIN_ROLES:
                    bi_to_color[bi] = TRUNK_COLOR
                    print(
                        f"[ClassifySurface] GeoRescue TRUNK: "
                        f"{self.getBranchDisplayName(bi)} "
                        f"r={r_mm:.1f}mm > 1.8×{_mean_iliac_r:.1f}mm iliac mean "
                        f"(role='{role}')"
                    )

                # 2. Renal rescue: multi-constraint (replaces radius-only check).
                #    All four conditions must hold:
                #      a) length floor  — rules out bifurcation stubs
                #      b) Z-range gate  — ostium must be in the renal band
                #      c) lateral gate  — tip must be offset from trunk axis
                #      d) radius gate   — smaller than iliac mean (keeps original)
                elif (
                    len_mm >= _GR_RENAL_MIN_LEN
                    and r_mm < _mean_iliac_r * 0.85
                    and role not in _MAIN_ROLES
                    and bi != 0
                ):
                    # Ostium Z: prefer ostiumGi from metadata, fall back to branch start
                    _gr_ogi = meta.get(bi, {}).get("ostiumGi") or self.branches[bi][0]
                    _gr_ogi = min(int(_gr_ogi), len(self.points) - 1)
                    _gr_oz  = self.points[_gr_ogi][2]
                    # Lateral offset: tip X relative to trunk centroid
                    _gr_tip_x = self.points[self.branches[bi][1] - 1][0]
                    _gr_lat   = abs(_gr_tip_x - _gr_x_cen)

                    if _gr_z_lo <= _gr_oz <= _gr_z_hi and _gr_lat >= _GR_RENAL_MIN_LAT:
                        bi_to_color[bi] = RENAL_COLOR
                        _lat = "Right" if _gr_tip_x < _gr_x_cen else "Left"
                        _rescued_name = f"{_lat[0]}-RV-{bi}"
                        meta.setdefault(bi, {}).update({
                            "role":          "renal_vein",
                            "lateral_label": _lat,
                            "label":         _rescued_name,
                            "display_name":  _rescued_name,
                            "name":          _rescued_name,
                        })
                        if hasattr(self, "branchMeta"):
                            self.branchMeta.setdefault(bi, {}).update(meta[bi])
                        _gr_renal_promoted[bi] = len_mm * 0.6 + r_mm * 0.4
                        print(
                            f"[ClassifySurface] GeoRescue RENAL: "
                            f"{self.getBranchDisplayName(bi)} "
                            f"L={len_mm:.0f}mm r={r_mm:.1f}mm "
                            f"oz={_gr_oz:.1f} lat={_gr_lat:.1f}mm "
                            f"(role='{role}' → 'renal_vein')"
                        )
                    else:
                        bi_to_color[bi] = SIDE_COLOR
                        print(
                            f"[ClassifySurface] GeoRescue RENAL rejected: bi={bi} "
                            f"L={len_mm:.0f}mm r={r_mm:.1f}mm "
                            f"oz={_gr_oz:.1f} band=[{_gr_z_lo:.1f},{_gr_z_hi:.1f}] "
                            f"lat={_gr_lat:.1f}mm (need>={_GR_RENAL_MIN_LAT:.0f}mm)"
                        )
                else:
                    bi_to_color[bi] = SIDE_COLOR

                print(
                    f"[DEBUG GeoRescue] bi={bi} {self.getBranchDisplayName(bi)} "
                    f"r={r_mm:.2f}mm  "
                    f"live={_mean_iliac_r:.2f}  "
                    f"renal_thr={_mean_iliac_r * 0.85:.2f}  "
                    f"→ {bi_to_color.get(bi, 'unset')}"
                )

        # ── GeoRescue renal cap: at most 1 per side ───────────────────────────
        # If more than 2 branches were promoted by GeoRescue, keep only the
        # highest-scoring one on each side (left/right of trunk centroid).
        # This is a backstop — with the Z and lateral gates above, over-promotion
        # should be rare; the cap prevents any residual ambiguous cases.
        if len(_gr_renal_promoted) > 2:
            _gr_left  = {b: s for b, s in _gr_renal_promoted.items()
                         if self.points[self.branches[b][1] - 1][0] >= _gr_x_cen}
            _gr_right = {b: s for b, s in _gr_renal_promoted.items()
                         if self.points[self.branches[b][1] - 1][0] <  _gr_x_cen}
            _gr_keep  = set()
            if _gr_left:  _gr_keep.add(max(_gr_left,  key=_gr_left.get))
            if _gr_right: _gr_keep.add(max(_gr_right, key=_gr_right.get))
            for _b in list(_gr_renal_promoted):
                if _b not in _gr_keep:
                    _demoted = f"Side-{_b}"
                    meta.setdefault(_b, {}).update({
                        "role":         "side",
                        "label":        _demoted,
                        "display_name": _demoted,
                        "name":         _demoted,
                    })
                    if hasattr(self, "branchMeta"):
                        self.branchMeta.setdefault(_b, {}).update(meta[_b])
                    bi_to_color[_b] = SIDE_COLOR
                    print(
                        f"[ClassifySurface] GeoRescue cap: bi={_b} demoted "
                        f"(>{len(_gr_renal_promoted)} renal candidates, kept {_gr_keep})"
                    )
        # ── end GeoRescue renal cap ───────────────────────────────────────────

        print(f"[ClassifySurface] Color groups (main_len_gate={main_len_gate_mm}mm):")
        print(f"[DEBUG] bi_to_color raw: { {b: self.getBranchDisplayName(b) for b in bi_to_color} }")
        print(f"[DEBUG] MAIN entries: { [(b, self.getBranchDisplayName(b), c) for b,c in bi_to_color.items() if c in MAIN_COLORS and meta.get(b, {}).get('role') != 'iliac_fragment'] }")
        print(
            f"  Trunk  : {[self.getBranchDisplayName(b) for b,c in bi_to_color.items() if c == TRUNK_COLOR]}"
        )
        print(
            f"  Main   : {[self.getBranchDisplayName(b) for b,c in bi_to_color.items() if c in MAIN_COLORS and meta.get(b, {}).get('role') != 'iliac_fragment']}"
        )
        print(
            f"  Renal  : {[self.getBranchDisplayName(b) for b,c in bi_to_color.items() if c == RENAL_COLOR]}"
        )
        print(
            f"  Side   : {[self.getBranchDisplayName(b) for b,c in bi_to_color.items() if c == SIDE_COLOR]}"
        )

        # ── 6. Write RGB array directly onto the (curvature-enriched) poly ────
        color_arr = vtk.vtkUnsignedCharArray()
        color_arr.SetName("BranchClassification")
        color_arr.SetNumberOfComponents(3)
        color_arr.SetNumberOfTuples(n_pts)
        for i in range(n_pts):
            bi = int(vert_bi[i])
            rgb = bi_to_color.get(bi, (128, 128, 128))
            color_arr.SetTuple3(i, *rgb)

        curved_poly.GetPointData().AddArray(color_arr)
        # Explicitly make BranchClassification the active scalar so that
        # vtkCurvatures' Mean_Curvature array (added in step 1) does not
        # remain active and drive the mapper's rainbow colormap.
        curved_poly.GetPointData().SetActiveScalars("BranchClassification")

        # Persist branch→RGB map so the navigator sphere can match surface color.
        # Keys are branch indices (int), values are (R,G,B) tuples in 0-255 range.
        self._branch_surface_colors = {
            bi: bi_to_color.get(bi, (128, 128, 128)) for bi in range(len(self.branches))
        }

        return {
            "poly": curved_poly,
            "classifications": classifications,
            "ostium_count": ostium_count,
            "bi_to_color": bi_to_color,          # (R,G,B) 0-255 per branch index
            "TRUNK_COLOR": TRUNK_COLOR,
            "RENAL_COLOR": RENAL_COLOR,
            "MAIN_COLORS": MAIN_COLORS,
            "SIDE_COLOR": SIDE_COLOR,
        }



# ── Slicer module-scanner guard ───────────────────────────────────────────────
# Slicer auto-scans all .py files in the module folder and expects a class
# matching the filename.  This stub satisfies that requirement without
# registering as a real loadable module (no ScriptedLoadableModule base).
class vessel_stent_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_stent_mixin"
            parent.hidden = True  # hide from Slicer module list
