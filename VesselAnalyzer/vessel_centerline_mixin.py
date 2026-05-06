"""
vessel_centerline_mixin.py — Centerline extraction, loading, and graph-pipeline methods.

Part of the VesselAnalyzer mixin decomposition.
These methods are mixed into VesselAnalyzerLogic via multiple inheritance:

    class VesselAnalyzerLogic(
        CenterlineMixin,
        ...
        ScriptedLoadableModuleLogic,
    ): ...

All methods use ``self`` normally — no changes to call sites required.
"""

# ruff: noqa  (this file is auto-extracted; formatting is inherited from VesselAnalyzer.py)

import slicer
from centerline_analysis import build_graph, score_bifurcation
from vessel_geometry import renal_anatomy_gate, renal_composite_score, geo_clamp
import math
import vtk


class CenterlineMixin:
    """Mixin: Centerline extraction, loading, and graph-pipeline methods."""

    def autoDetectEndpoints(self, modelNode):
        """Detect VMTK centerline endpoints from vessel opening centroids.

        Strategy:
          1. Run vtkFeatureEdges on the raw mesh.
             If boundary rings exist (open mesh from Segment Editor with
             cropped vessel ends) → cluster them directly.  Done.
          2. If the mesh is closed (Segment Editor default export caps all
             ends) → use _findTipPoints to locate approximate tip regions,
             clip a sphere at each tip to open the surface, then run
             vtkFeatureEdges on the clipped result.  The resulting boundary
             ring centroid sits inside the vessel lumen (not on the IVC wall)
             because the clip sphere removes the cap and the ring forms at
             the cross-section inside the lumen.
        """
        import vtk
        import math

        polydata = modelNode.GetPolyData()
        if not polydata or polydata.GetNumberOfPoints() == 0:
            return []

        def _extract_rings(pd):
            """Return list of (centroid, radius, npts) for each boundary ring."""
            fe = vtk.vtkFeatureEdges()
            fe.SetInputData(pd)
            fe.BoundaryEdgesOn()
            fe.NonManifoldEdgesOff()
            fe.FeatureEdgesOff()
            fe.ManifoldEdgesOff()
            fe.Update()
            bpd = fe.GetOutput()
            nBPts = bpd.GetNumberOfPoints()
            if nBPts == 0:
                return []

            # BFS connected-component clustering
            bpd.BuildLinks()
            visited = [False] * nBPts
            rings = []
            for start in range(nBPts):
                if visited[start]:
                    continue
                comp = []
                queue = [start]
                visited[start] = True
                while queue:
                    pid = queue.pop()
                    comp.append(pid)
                    cellIds = vtk.vtkIdList()
                    bpd.GetPointCells(pid, cellIds)
                    for ci in range(cellIds.GetNumberOfIds()):
                        cell = bpd.GetCell(cellIds.GetId(ci))
                        for vi in range(cell.GetNumberOfPoints()):
                            nb = cell.GetPointId(vi)
                            if not visited[nb]:
                                visited[nb] = True
                                queue.append(nb)
                rings.append(comp)

            bPts = bpd.GetPoints()
            results = []
            MIN_RING_PTS = 4
            for ri, ring in enumerate(rings):
                if len(ring) < MIN_RING_PTS:
                    continue
                xs = [bPts.GetPoint(p)[0] for p in ring]
                ys = [bPts.GetPoint(p)[1] for p in ring]
                zs = [bPts.GetPoint(p)[2] for p in ring]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                cz = sum(zs) / len(zs)
                r = sum(
                    math.sqrt((xs[i] - cx) ** 2 + (ys[i] - cy) ** 2 + (zs[i] - cz) ** 2)
                    for i in range(len(xs))
                ) / len(xs)
                results.append(((cx, cy, cz), r, len(ring)))
            return results

        # ── 1. Try direct boundary detection on raw mesh ─────────────────────
        rings = _extract_rings(polydata)
        if rings:
            print(
                f"[AutoEndpoints] Open mesh: {len(rings)} boundary rings detected directly"
            )
            centroids = []
            for i, (c, r, n) in enumerate(rings):
                print(
                    f"  Ring {i}: {n} pts  centroid=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})  r≈{r:.1f}mm"
                )
                centroids.append(c)
        else:
            # ── 2. Closed mesh — clip open at tips then re-detect ─────────────
            print(
                "[AutoEndpoints] Closed mesh — clipping at tip locations to open vessel ends"
            )
            tips = self._findTipPoints(polydata)
            if not tips:
                print("[AutoEndpoints] No tip points found — returning empty")
                return []

            # Clip radius: large enough to remove the cap and expose a clean
            # cross-section ring inside the lumen, small enough not to eat
            # into adjacent anatomy.  8mm works well for IVC/iliac diameters.
            CLIP_R = 8.0
            clipped = polydata
            for tip in tips:
                sphere = vtk.vtkSphere()
                sphere.SetCenter(tip[0], tip[1], tip[2])
                sphere.SetRadius(CLIP_R)
                clipper = vtk.vtkClipPolyData()
                clipper.SetInputData(clipped)
                clipper.SetClipFunction(sphere)
                clipper.SetInsideOut(False)
                clipper.Update()
                clipped = clipper.GetOutput()

            rings = _extract_rings(clipped)
            if not rings:
                print(
                    "[AutoEndpoints] No rings after clipping — falling back to tip points"
                )
                return tips

            print(f"[AutoEndpoints] Clipped mesh: {len(rings)} boundary rings detected")
            centroids = []
            for i, (c, r, n) in enumerate(rings):
                print(
                    f"  Ring {i}: {n} pts  centroid=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})  r≈{r:.1f}mm"
                )
                centroids.append(c)

        # ── 3. Deduplicate centroids closer than MIN_SEP_MM ──────────────────
        MIN_SEP_MM = 10.0
        deduped = []
        for c in centroids:
            if not deduped:
                deduped.append(c)
                continue
            d = min(
                math.sqrt(sum((c[k] - d[k]) ** 2 for k in range(3))) for d in deduped
            )
            if d >= MIN_SEP_MM:
                deduped.append(c)

        print(
            f"[AutoEndpoints] {len(centroids)} rings → {len(deduped)} endpoints after dedup"
        )
        for i, p in enumerate(deduped):
            print(f"  Endpoint {i}: X={p[0]:.1f} Y={p[1]:.1f} Z={p[2]:.1f}")

        return deduped

    def _findTipPoints(self, polydata):
        """Find vessel tip points using widest-cross-section and extreme point method"""
        import math

        pts = polydata.GetPoints()
        nPts = pts.GetNumberOfPoints()
        if nPts == 0:
            return []

        allPoints = [pts.GetPoint(i) for i in range(nPts)]
        bounds = polydata.GetBounds()
        xMin, xMax = bounds[0], bounds[1]
        zMin, zMax = bounds[4], bounds[5]
        zRange = zMax - zMin
        cx = sum(p[0] for p in allPoints) / nPts

        # Top inlet
        topPt = max(allPoints, key=lambda p: p[2])

        # Bottom left and right iliac tips
        leftHalf = [p for p in allPoints if p[0] < cx]
        rightHalf = [p for p in allPoints if p[0] >= cx]
        botLeftPt = min(leftHalf, key=lambda p: p[2]) if leftHalf else None
        botRightPt = min(rightHalf, key=lambda p: p[2]) if rightHalf else None

        # Horizontal branches: scan slices, find local X-width peak above bottom 40%
        scanStep = zRange / 60
        sliceData = []
        for i in range(60):
            sliceZ = zMin + i * scanStep
            slicePts = [p for p in allPoints if abs(p[2] - sliceZ) < scanStep * 1.5]
            if len(slicePts) < 5:
                sliceData.append((sliceZ, 0, None, None))
                continue
            w = max(p[0] for p in slicePts) - min(p[0] for p in slicePts)
            sliceData.append(
                (
                    sliceZ,
                    w,
                    min(slicePts, key=lambda p: p[0]),
                    max(slicePts, key=lambda p: p[0]),
                )
            )

        # Find local width peak in upper 60% — split left/right halves
        searchZMin = zMin + zRange * 0.40
        bestLeftPt = bestRightPt = None
        bestLeftW = bestRightW = 0

        for i in range(1, len(sliceData) - 1):
            sliceZ, w, lPt, rPt = sliceData[i]
            if sliceZ < searchZMin or lPt is None:
                continue
            prevW = sliceData[i - 1][1]
            nextW = sliceData[i + 1][1]

            # Left half extreme
            slicePts = [
                p
                for p in allPoints
                if abs(p[2] - sliceZ) < scanStep * 1.5 and p[0] < cx
            ]
            if slicePts:
                leftExtreme = min(slicePts, key=lambda p: p[0])
                leftW = cx - leftExtreme[0]
                if leftW > bestLeftW:
                    bestLeftW = leftW
                    bestLeftPt = leftExtreme

            # Right half extreme
            slicePts = [
                p
                for p in allPoints
                if abs(p[2] - sliceZ) < scanStep * 1.5 and p[0] >= cx
            ]
            if slicePts:
                rightExtreme = max(slicePts, key=lambda p: p[0])
                rightW = rightExtreme[0] - cx
                if rightW > bestRightW:
                    bestRightW = rightW
                    bestRightPt = rightExtreme

        rawCandidates = [
            p
            for p in [topPt, botLeftPt, botRightPt, bestLeftPt, bestRightPt]
            if p is not None
        ]

        # Deduplicate - reduce threshold since branches are at same level
        filtered = []
        for ep in rawCandidates:
            if not filtered:
                filtered.append(ep)
                continue
            minDist = min(
                math.sqrt(sum((ep[k] - f[k]) ** 2 for k in range(3))) for f in filtered
            )
            if minDist > 10.0:  # reduced from 15mm to catch same-level branches
                filtered.append(ep)

        # Debug output to Python console
        print(f"[VesselAnalyzer] Raw candidates: {len(rawCandidates)}")
        for i, p in enumerate(rawCandidates):
            print(f"  Candidate {i}: X={p[0]:.1f} Y={p[1]:.1f} Z={p[2]:.1f}")
        print(f"[VesselAnalyzer] After dedup: {len(filtered)} endpoints")
        for i, p in enumerate(filtered):
            print(f"  Final {i}: X={p[0]:.1f} Y={p[1]:.1f} Z={p[2]:.1f}")

        return filtered

    def _autoDetectEndpointsFallback(self, modelNode):
        """Fallback using geometry-based tip detection"""
        polydata = modelNode.GetPolyData()
        if not polydata:
            return []
        return self._findTipPoints(polydata)

        # NOTE: Alternative geodesic-extremes method was removed (dead code after return).

    def extractCenterline(
        self, modelNode, endpointsNode, preprocess=True, targetPoints=5000
    ):
        """Extract centerline using VMTK via Slicer's ExtractCenterline module"""
        try:
            import ExtractCenterline

            extractLogic = ExtractCenterline.ExtractCenterlineLogic()

            # Preprocess — use higher target points for better medial axis
            # accuracy at the bifurcation zone (old default 5000 was too coarse)
            if preprocess:
                preprocessed = extractLogic.preprocess(
                    modelNode.GetPolyData(),
                    max(targetPoints, 8000),  # minimum 8000 for bifurcation accuracy
                    decimationAggressiveness=3.5,  # less aggressive = better bifurcation geometry
                    subdivide=True,  # subdivide helps medial axis stay centered
                )
            else:
                preprocessed = modelNode.GetPolyData()

            # Build endpoint positions
            endpointPositions = []
            for i in range(endpointsNode.GetNumberOfControlPoints()):
                p = [0, 0, 0]
                endpointsNode.GetNthControlPointPositionWorld(i, p)
                endpointPositions.append(p)

            # ── Seed nudging ──────────────────────────────────────────────────
            # VMTK collapses paths to nearby endpoints into a single segment
            # when seeds are too close together or sit on the mesh surface
            # rather than inside the lumen. This produces fewer centerline
            # segments than expected (e.g. 4 segments from 5 endpoints).
            #
            # Fix: for any seed within CLUSTER_THRESH_MM of another seed,
            # nudge it inward along the surface normal direction so it sits
            # NUDGE_MM deeper inside the lumen. VMTK then has a distinct
            # intra-lumen target for each branch and traces separate paths.
            #
            # The inward direction is estimated as the vector from the seed
            # toward the mesh centroid (always points into the lumen for
            # convex or mildly concave vessels).
            CLUSTER_THRESH_MM = 20.0  # seeds closer than this are candidates
            NUDGE_MM = 6.0  # how far to push each seed inward

            import math as _math

            def _dist3ep(a, b):
                return _math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))

            # Compute mesh centroid for inward-direction reference
            bounds = [0.0] * 6
            preprocessed.GetBounds(bounds)
            mesh_cx = (bounds[0] + bounds[1]) * 0.5
            mesh_cy = (bounds[2] + bounds[3]) * 0.5
            mesh_cz = (bounds[4] + bounds[5]) * 0.5

            # Find which seeds are clustered (within CLUSTER_THRESH_MM of
            # at least one other seed) — these are nudge candidates.
            n_ep = len(endpointPositions)
            nudge_candidates = set()
            for _i in range(n_ep):
                for _j in range(_i + 1, n_ep):
                    if (
                        _dist3ep(endpointPositions[_i], endpointPositions[_j])
                        < CLUSTER_THRESH_MM
                    ):
                        nudge_candidates.add(_i)
                        nudge_candidates.add(_j)

            # Nudge each candidate seed inward
            nudged_positions = [list(p) for p in endpointPositions]
            if nudge_candidates:
                print(
                    f"[SeedNudge] {len(nudge_candidates)} clustered seeds within "
                    f"{CLUSTER_THRESH_MM}mm — nudging {NUDGE_MM}mm inward"
                )
                for _ni in nudge_candidates:
                    sp = endpointPositions[_ni]
                    # Direction: seed → mesh centroid (inward)
                    _dx = mesh_cx - sp[0]
                    _dy = mesh_cy - sp[1]
                    _dz = mesh_cz - sp[2]
                    _dl = _math.sqrt(_dx * _dx + _dy * _dy + _dz * _dz)
                    if _dl > 1e-6:
                        _dx /= _dl
                        _dy /= _dl
                        _dz /= _dl
                    nudged_positions[_ni] = [
                        sp[0] + _dx * NUDGE_MM,
                        sp[1] + _dy * NUDGE_MM,
                        sp[2] + _dz * NUDGE_MM,
                    ]
                    print(
                        f"[SeedNudge]   ep{_ni}: "
                        f"({sp[0]:.1f},{sp[1]:.1f},{sp[2]:.1f}) → "
                        f"({nudged_positions[_ni][0]:.1f},"
                        f"{nudged_positions[_ni][1]:.1f},"
                        f"{nudged_positions[_ni][2]:.1f})"
                    )

                # Build a new temporary endpoints node with nudged positions
                nudgedEndpointsNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsFiducialNode", "NudgedEndpoints"
                )
                nudgedEndpointsNode.CreateDefaultDisplayNodes()
                nudgedEndpointsNode.GetDisplayNode().SetVisibility(0)
                for _np in nudged_positions:
                    nudgedEndpointsNode.AddControlPoint(vtk.vtkVector3d(_np))
                vmtkEndpointsNode = nudgedEndpointsNode
            else:
                print(f"[SeedNudge] No clustered seeds — using original positions")
                vmtkEndpointsNode = endpointsNode
            # ── End seed nudging ──────────────────────────────────────────────

            # Create output curve node
            centerlineCurve = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode", "CenterlineCurve"
            )
            centerlineCurve.CreateDefaultDisplayNodes()
            _cdn = centerlineCurve.GetDisplayNode()
            if _cdn:
                _cdn.SetVisibility(0)

            # Extract - finer sampling (0.5mm vs old 1.0mm) for better
            # medial axis accuracy especially at the bifurcation zone.
            # Try different API signatures for different VMTK versions.
            def _run_extract(ep_node):
                try:
                    return extractLogic.extractCenterline(
                        preprocessed, ep_node, curveSamplingDistance=0.5
                    )
                except TypeError:
                    try:
                        return extractLogic.extractCenterline(
                            preprocessed, ep_node, 0.5
                        )
                    except TypeError:
                        return extractLogic.extractCenterline(preprocessed, ep_node)

            centerlinePolyData, voronoiDiagram = _run_extract(vmtkEndpointsNode)

            # Check segment count — if VMTK still collapsed paths, retry
            # with doubled nudge distance for the clustered seeds.
            n_segs = (
                centerlinePolyData.GetLines().GetNumberOfCells()
                if centerlinePolyData.GetLines()
                else 0
            )
            expected_segs = n_ep - 1  # fully connected tree has n-1 unique paths
            if nudge_candidates and n_segs < expected_segs:
                print(
                    f"[SeedNudge] VMTK returned {n_segs} segments (expected ≥{expected_segs}) "
                    f"— retrying with doubled nudge ({NUDGE_MM*2:.0f}mm)"
                )
                nudgedEndpointsNode2 = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsFiducialNode", "NudgedEndpoints2"
                )
                nudgedEndpointsNode2.CreateDefaultDisplayNodes()
                nudgedEndpointsNode2.GetDisplayNode().SetVisibility(0)
                for _ni2, _sp2 in enumerate(endpointPositions):
                    if _ni2 in nudge_candidates:
                        _dx2 = mesh_cx - _sp2[0]
                        _dy2 = mesh_cy - _sp2[1]
                        _dz2 = mesh_cz - _sp2[2]
                        _dl2 = _math.sqrt(_dx2 * _dx2 + _dy2 * _dy2 + _dz2 * _dz2)
                        if _dl2 > 1e-6:
                            _dx2 /= _dl2
                            _dy2 /= _dl2
                            _dz2 /= _dl2
                        _np2 = [
                            _sp2[0] + _dx2 * NUDGE_MM * 2,
                            _sp2[1] + _dy2 * NUDGE_MM * 2,
                            _sp2[2] + _dz2 * NUDGE_MM * 2,
                        ]
                    else:
                        _np2 = list(_sp2)
                    nudgedEndpointsNode2.AddControlPoint(vtk.vtkVector3d(_np2))
                centerlinePolyData2, voronoiDiagram2 = _run_extract(
                    nudgedEndpointsNode2
                )
                n_segs2 = (
                    centerlinePolyData2.GetLines().GetNumberOfCells()
                    if centerlinePolyData2.GetLines()
                    else 0
                )
                if n_segs2 > n_segs:
                    print(
                        f"[SeedNudge] Retry improved: {n_segs} → {n_segs2} segments — using retry result"
                    )
                    centerlinePolyData = centerlinePolyData2
                    voronoiDiagram = voronoiDiagram2
                else:
                    print(
                        f"[SeedNudge] Retry did not improve ({n_segs2} segments) — keeping first result"
                    )
                try:
                    slicer.mrmlScene.RemoveNode(nudgedEndpointsNode2)
                except Exception:
                    pass

            # Clean up temporary nudged endpoints node
            if nudge_candidates:
                try:
                    slicer.mrmlScene.RemoveNode(nudgedEndpointsNode)
                except Exception:
                    pass

            # Convert polydata to curve node
            pts = centerlinePolyData.GetPoints()
            lines = centerlinePolyData.GetLines()
            if lines and lines.GetNumberOfCells() > 0:
                lines.InitTraversal()
                idList = vtk.vtkIdList()
                lines.GetNextCell(idList)
                for i in range(idList.GetNumberOfIds()):
                    pid = idList.GetId(i)
                    p = pts.GetPoint(pid)
                    centerlineCurve.AddControlPoint(vtk.vtkVector3d(p))

            # Also store as model node
            centerlineModel = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", "CenterlineModel"
            )
            centerlineModel.CreateDefaultDisplayNodes()
            centerlineModel.SetAndObservePolyData(centerlinePolyData)
            dispNode = centerlineModel.GetDisplayNode()
            dispNode.SetColor(1, 0.5, 0)
            dispNode.SetLineWidth(3)
            dispNode.SetVisibility(0)

            return centerlineCurve

        except Exception as e:
            print(f"ExtractCenterline module error: {e}")
            # Fallback: use skeleton/simple centerline
            return self._fallbackCenterline(modelNode, endpointsNode)

    def _fallbackCenterline(self, modelNode, endpointsNode):
        """Simple fallback: straight line between endpoints"""
        curveNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", "CenterlineCurve_simple"
        )
        curveNode.CreateDefaultDisplayNodes()
        _cdn2 = curveNode.GetDisplayNode()
        if _cdn2:
            _cdn2.SetVisibility(0)
        for i in range(endpointsNode.GetNumberOfControlPoints()):
            p = [0, 0, 0]
            endpointsNode.GetNthControlPointPositionWorld(i, p)
            curveNode.AddControlPoint(vtk.vtkVector3d(p))
        return curveNode

    def loadCenterline(self, centerlineNode, modelNode=None, segModelNode=None):
        self.points = []
        self.distances = []
        self.diameters = []
        self.branches = []
        self.activeBranch = -1
        self.modelNode = modelNode
        self.segModelNode = segModelNode  # segmentation surface (open-boundary mesh)
        # used for ring-normal ostium detection

        rawPoints = []

        if centerlineNode.GetClassName() == "vtkMRMLMarkupsCurveNode":
            n = centerlineNode.GetNumberOfControlPoints()
            for i in range(n):
                p = [0, 0, 0]
                centerlineNode.GetNthControlPointPositionWorld(i, p)
                rawPoints.append(tuple(p))

        elif centerlineNode.GetClassName() == "vtkMRMLModelNode":
            polydata = centerlineNode.GetPolyData()
            if not polydata:
                return False
            pts = polydata.GetPoints()
            lines = polydata.GetLines()
            if lines and lines.GetNumberOfCells() > 0:
                # Read ALL line cells (all branches)
                # Check for VMTK MaximumInscribedSphereRadius array
                radiusArray = polydata.GetPointData().GetArray(
                    "MaximumInscribedSphereRadius"
                )
                if radiusArray:
                    print(
                        f"[VesselAnalyzer] Found MaximumInscribedSphereRadius array — using VMTK radii"
                    )
                else:
                    print(
                        f"[VesselAnalyzer] No radius array found — will use surface distance method"
                    )

                lines.InitTraversal()
                idList = vtk.vtkIdList()
                allSegments = []
                allSegmentRadii = []  # parallel array: radius per point
                while lines.GetNextCell(idList):
                    seg = []
                    radii = []
                    for i in range(idList.GetNumberOfIds()):
                        pid = idList.GetId(i)
                        seg.append(pts.GetPoint(pid))
                        if radiusArray:
                            radii.append(radiusArray.GetValue(pid))
                        else:
                            radii.append(None)
                    if seg:
                        allSegments.append(seg)
                        allSegmentRadii.append(radii)

                print(f"[VesselAnalyzer] Found {len(allSegments)} centerline segments")
                for i, s in enumerate(allSegments):
                    pass  # segment listed above

                # Use VMTK centerline array data for proper branch splitting
                # The CellData contains branch topology info
                self.branches = []

                # Check if polydata has centerline branch info arrays
                cellData = polydata.GetCellData()
                hasBranchArray = (
                    cellData.HasArray("CenterlineSectionBifurcation")
                    or cellData.HasArray("Blanking")
                    or cellData.HasArray("GroupIds")
                )

                if hasBranchArray:
                    # Use VMTK GroupIds to split branches properly
                    groupArray = cellData.GetArray("GroupIds") or cellData.GetArray(
                        "CenterlineSectionGroupIds"
                    )
                    if groupArray:
                        # Group line cells by GroupId
                        groupSegs = {}
                        lines.InitTraversal()
                        idList = vtk.vtkIdList()
                        cellIdx = 0
                        while lines.GetNextCell(idList):
                            gid = int(groupArray.GetValue(cellIdx))
                            seg = [
                                pts.GetPoint(idList.GetId(j))
                                for j in range(idList.GetNumberOfIds())
                            ]
                            if gid not in groupSegs:
                                groupSegs[gid] = []
                            groupSegs[gid].extend(seg)
                            cellIdx += 1
                        for gid in sorted(groupSegs.keys()):
                            seg = groupSegs[gid]
                            startIdx = len(rawPoints)
                            rawPoints.extend(seg)
                            self.branches.append((startIdx, len(rawPoints)))
                    else:
                        hasBranchArray = False

                if not hasBranchArray:
                    # ── Graph-based topology detection (topology + geometry) ───────
                    #
                    # Pipeline (all four stages required for clinical reliability):
                    #
                    #   1. Build graph — snap ALL points with LOW tolerance (0.5 mm)
                    #      so only truly co-located points merge.  1.5 mm was causing
                    #      parallel centerlines to cross-connect → fake bifurcations.
                    #
                    #   2. Collapse degree-2 chains — a degree-2 node is a pass-through
                    #      on a straight vessel segment, not an anatomical event.
                    #      Collapsing them leaves only critical nodes (deg-1 / deg≥3)
                    #      and direct edges between them.
                    #
                    #   3. Validate bifurcations geometrically — a real bifurcation
                    #      has at least one edge pair with angle > MIN_BIF_ANGLE (45°).
                    #      Noise produces near-collinear degree-3 nodes that fail this
                    #      test and are demoted to degree-2 (pass-through).
                    #
                    #   4. Prune short branches — any branch < MIN_BRANCH_MM is a
                    #      sampling artifact, not an anatomical vessel.

                    print(
                        f"[VesselAnalyzer] Detecting branches via topology graph "
                        f"(snap + collapse + angle filter + length prune)"
                    )

                    import statistics as _stats

                    # ── tuning constants ──────────────────────────────────────────
                    SNAP_TOL = 2.0  # mm — relaxed to handle float drift at per-branch pipeline junction points
                    INV_TOL = 1.0 / SNAP_TOL
                    MIN_BIF_ANGLE = (
                        45.0  # degrees — minimum angle to be a real bifurcation
                    )
                    MIN_BRANCH_MM = 25.0  # mm — shorter branches are artifacts
                    #   Raised from 15mm: edges like 113-116 (7.7mm) and 52-54 (1.9mm)
                    #   are graph junction stubs, not anatomical vessels. 25mm keeps
                    #   all real iliac side branches (≥32mm in this dataset) while
                    #   eliminating the micro-stubs that inflate the branch count.

                    # ── helpers ───────────────────────────────────────────────────

                    def _dist3(a, b):
                        return math.sqrt(
                            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
                        )

                    def _seg_length(pts_list):
                        t = 0.0
                        for k in range(1, len(pts_list)):
                            t += _dist3(pts_list[k - 1], pts_list[k])
                        return t

                    # ── step 1: build graph with tight snap ───────────────────────
                    cellMap = {}  # grid-cell key → node_id
                    nodePos = []  # node_id → (x,y,z) centroid
                    nodeCount = []  # points merged into this node
                    nodeAdj = []  # node_id → set of neighbour node_ids

                    def _snap(p):
                        ix = int(math.floor(p[0] * INV_TOL))
                        iy = int(math.floor(p[1] * INV_TOL))
                        iz = int(math.floor(p[2] * INV_TOL))
                        key = (ix, iy, iz)
                        if key in cellMap:
                            nid = cellMap[key]
                            c = nodeCount[nid]
                            pos = nodePos[nid]
                            nodePos[nid] = (
                                (pos[0] * c + p[0]) / (c + 1),
                                (pos[1] * c + p[1]) / (c + 1),
                                (pos[2] * c + p[2]) / (c + 1),
                            )
                            nodeCount[nid] = c + 1
                            return nid
                        nid = len(nodePos)
                        cellMap[key] = nid
                        nodePos.append(p)
                        nodeCount.append(1)
                        nodeAdj.append(set())
                        return nid

                    for si, seg in enumerate(allSegments):
                        prev_nid = None
                        for p in seg:
                            nid = _snap(p)
                            if prev_nid is not None and prev_nid != nid:
                                nodeAdj[prev_nid].add(nid)
                                nodeAdj[nid].add(prev_nid)
                            prev_nid = nid

                    nNodes = len(nodePos)
                    print(
                        f"[VesselAnalyzer] Graph raw: {nNodes} nodes, "
                        f"{sum(len(s) for s in allSegments)} input pts, "
                        f"snap={SNAP_TOL}mm"
                    )

                    # ── step 2: collapse degree-2 chains ─────────────────────────
                    # Replace every maximal chain of degree-2 nodes with a single edge
                    # between the two critical nodes at its ends.
                    # We do this by building a compressed graph: cNodes / cAdj / cEdgePts
                    # where cEdgePts[(a,b)] = ordered list of nodePos along that edge.

                    def _degree(n):
                        return len(nodeAdj[n])

                    def _isCritical(n):
                        return _degree(n) != 2

                    # Walk a degree-2 chain from startNode coming from prevNode.
                    # Returns (endNode, [intermediate node positions including startNode])
                    def _walkChain(startNode, prevNode):
                        pts = [nodePos[startNode]]
                        cur = startNode
                        prev = prevNode
                        while not _isCritical(cur):
                            nxts = [nb for nb in nodeAdj[cur] if nb != prev]
                            if not nxts:
                                break
                            nxt = nxts[0]
                            pts.append(nodePos[nxt])
                            prev = cur
                            cur = nxt
                        return cur, pts

                    critSet = [n for n in range(nNodes) if _isCritical(n)]
                    # compressed adjacency: crit node → set of crit neighbours
                    cAdj = {n: set() for n in critSet}
                    cEdge = {}  # (a,b) sorted tuple → list of (x,y,z) along edge

                    visitedEdges = set()
                    for startNode in critSet:
                        for nb in list(nodeAdj[startNode]):
                            key = (min(startNode, nb), max(startNode, nb))
                            if key in visitedEdges:
                                continue
                            visitedEdges.add(key)
                            if _isCritical(nb):
                                endNode = nb
                                edgePts = [nodePos[startNode], nodePos[nb]]
                            else:
                                endNode, midPts = _walkChain(nb, startNode)
                                edgePts = [nodePos[startNode]] + midPts
                            if endNode == startNode:
                                continue  # loop — skip
                            cAdj[startNode].add(endNode)
                            cAdj[endNode].add(startNode)
                            ekey = (min(startNode, endNode), max(startNode, endNode))
                            if ekey not in cEdge:
                                cEdge[ekey] = edgePts

                    print(
                        f"[VesselAnalyzer] After collapse: {len(critSet)} critical nodes, "
                        f"{len(cEdge)} edges"
                    )

                    # FIX (v297-BifSnapTopo): warn when a raw degree>=3 node
                    # collapses to degree<3 in cAdj — this means the snap
                    # merged two of its neighbours, hiding a true bifurcation.
                    # Paired with the JunctionSnap pts[2] fix in
                    # per_branch_centerline_pipeline.py which prevents the
                    # over-merge from happening in the first place.
                    for _cn in critSet:
                        raw_deg = len(nodeAdj[_cn])
                        col_deg = len(cAdj.get(_cn, set()))
                        if raw_deg >= 3 and col_deg < 3:
                            print(
                                f"[TopologyWarn] Node {_cn} raw degree={raw_deg} "
                                f"but collapsed degree={col_deg} — "
                                f"snap over-merge near bifurcation at "
                                f"pos={tuple(round(x,1) for x in nodePos[_cn])}. "
                                f"If iliac bif is missing, check JunctionSnap "
                                f"blend radius vs SNAP_TOL ({SNAP_TOL}mm)."
                            )

                    # ── step 3: validate bifurcations (3-condition rule) ──────────
                    #
                    # Angle alone is clinically unreliable:
                    #   - Iliac bifurcations:  15–25°  (shallow)
                    #   - Carotid bifurcation: 30–60°
                    #   - Coronary:            10–40°
                    # A hard angle threshold demotes real bifurcations in compressed
                    # or diseased vessels.
                    #
                    # Three-condition rule (ALL three must pass to DEMOTE a node,
                    # i.e. a node survives if it passes ANY of the three conditions):
                    #
                    #   Condition 1 — topology: degree ≥ 3                (mandatory)
                    #   Condition 2 — angle:    max pairwise angle > 15°  (loose)
                    #   Condition 3 — spatial divergence: downstream paths
                    #                 separate by > DIVERG_MM within LOOK_MM  (key)
                    #
                    # A node is demoted only if ALL THREE conditions jointly fail,
                    # i.e. if it is: low angle AND no spatial divergence AND the
                    # edges heading away from it converge rather than spread.
                    # This makes the filter safe for shallow-angle anatomy.

                    MIN_BIF_ANGLE = 15.0  # degrees — very loose; angle alone not gating
                    DIVERG_MM = 3.0  # mm — paths must be this far apart downstream
                    # Collect (nodeId, [hub_pos, nb_pos…]) for deferred bif scoring
                    _pendingBifScores = []
                    LOOK_MM = (
                        20.0  # mm — look this far downstream to measure divergence
                    )

                    def _edgeDir(hubNode, nbNode, lookMm=5.0):
                        """Unit vector from hub toward nbNode, sampled lookMm downstream."""
                        ekey = (min(hubNode, nbNode), max(hubNode, nbNode))
                        pts = cEdge.get(ekey, [nodePos[hubNode], nodePos[nbNode]])
                        hub = nodePos[hubNode]
                        # Orient pts so pts[0] is nearest hub
                        if len(pts) > 1 and _dist3(pts[-1], hub) < _dist3(pts[0], hub):
                            pts = list(reversed(pts))
                        # Walk forward until lookMm accumulated
                        acc = 0.0
                        far = pts[0]
                        for k in range(1, len(pts)):
                            step = _dist3(pts[k - 1], pts[k])
                            if acc + step >= lookMm:
                                frac = (lookMm - acc) / step if step > 0 else 0
                                far = (
                                    pts[k - 1][0] + frac * (pts[k][0] - pts[k - 1][0]),
                                    pts[k - 1][1] + frac * (pts[k][1] - pts[k - 1][1]),
                                    pts[k - 1][2] + frac * (pts[k][2] - pts[k - 1][2]),
                                )
                                break
                            acc += step
                            far = pts[k]
                        d = (far[0] - hub[0], far[1] - hub[1], far[2] - hub[2])
                        ln = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
                        return (
                            (d[0] / ln, d[1] / ln, d[2] / ln) if ln > 0 else (0, 0, 1)
                        )

                    def _angleBetween(u, v):
                        dot = max(
                            -1.0, min(1.0, u[0] * v[0] + u[1] * v[1] + u[2] * v[2])
                        )
                        return math.degrees(math.acos(abs(dot)))

                    def _spatialDivergence(hubNode, nbA, nbB):
                        """
                        Sample point pA on edge hubNode→nbA at LOOK_MM downstream.
                        Sample point pB on edge hubNode→nbB at LOOK_MM downstream.
                        Return distance between pA and pB.
                        A real bifurcation has pA-pB >> DIVERG_MM.
                        A curved pass-through has pA-pB ≈ 0 (same direction).
                        """

                        def _sampleAlongEdge(hubN, nbN, targetMm):
                            ekey = (min(hubN, nbN), max(hubN, nbN))
                            pts = cEdge.get(ekey, [nodePos[hubN], nodePos[nbN]])
                            hub = nodePos[hubN]
                            if len(pts) > 1 and _dist3(pts[-1], hub) < _dist3(
                                pts[0], hub
                            ):
                                pts = list(reversed(pts))
                            acc = 0.0
                            prev = pts[0]
                            for k in range(1, len(pts)):
                                step = _dist3(pts[k - 1], pts[k])
                                if acc + step >= targetMm:
                                    frac = (targetMm - acc) / step if step > 0 else 0
                                    return (
                                        pts[k - 1][0]
                                        + frac * (pts[k][0] - pts[k - 1][0]),
                                        pts[k - 1][1]
                                        + frac * (pts[k][1] - pts[k - 1][1]),
                                        pts[k - 1][2]
                                        + frac * (pts[k][2] - pts[k - 1][2]),
                                    )
                                acc += step
                            return pts[-1]

                        pA = _sampleAlongEdge(hubNode, nbA, LOOK_MM)
                        pB = _sampleAlongEdge(hubNode, nbB, LOOK_MM)
                        return _dist3(pA, pB)

                    demoted = set()
                    for n in list(critSet):
                        nbs = list(cAdj[n])
                        if len(nbs) < 3:
                            continue  # degree < 3 — not a bifurcation candidate

                        dirs = [_edgeDir(n, nb) for nb in nbs]

                        # Condition 2: angle
                        maxAngle = 0.0
                        for i in range(len(dirs)):
                            for j in range(i + 1, len(dirs)):
                                maxAngle = max(
                                    maxAngle, _angleBetween(dirs[i], dirs[j])
                                )
                        angleOk = maxAngle >= MIN_BIF_ANGLE

                        # Condition 3: spatial divergence — check ALL pairs of edges
                        maxDiverg = 0.0
                        for i in range(len(nbs)):
                            for j in range(i + 1, len(nbs)):
                                maxDiverg = max(
                                    maxDiverg, _spatialDivergence(n, nbs[i], nbs[j])
                                )
                        divergOk = maxDiverg >= DIVERG_MM

                        # Demote only if BOTH geometry conditions fail
                        # (topology condition = degree≥3 is already true here)
                        if not angleOk and not divergOk:
                            demoted.add(n)
                            print(
                                f"[VesselAnalyzer] Demoted node {n}: "
                                f"degree={len(nbs)}, "
                                f"max_angle={maxAngle:.1f}° < {MIN_BIF_ANGLE}°, "
                                f"max_diverg={maxDiverg:.1f}mm < {DIVERG_MM}mm"
                            )
                        else:
                            print(
                                f"[VesselAnalyzer] Bifurcation node {n} confirmed: "
                                f"degree={len(nbs)}, "
                                f"max_angle={maxAngle:.1f}°, "
                                f"max_diverg={maxDiverg:.1f}mm "
                                f"({'angle+diverg' if angleOk and divergOk else 'angle only' if angleOk else 'diverg only'})"
                            )
                            # Store (nodePos, neighbour positions) for deferred
                            # score_bifurcation() call once _lookupRadius is available
                            # (radius lookup requires segLocators built in step 8).
                            _pendingBifScores.append(
                                (n, [nodePos[n]] + [nodePos[nb] for nb in nbs])
                            )

                    # Re-classify after demotion
                    realEndpoints = [
                        n for n in critSet if len(cAdj[n]) == 1 and n not in demoted
                    ]
                    realBifurcations = [
                        n for n in critSet if len(cAdj[n]) >= 3 and n not in demoted
                    ]
                    realCritical = set(realEndpoints + realBifurcations)

                    print(
                        f"[VesselAnalyzer] Validated: "
                        f"{len(realEndpoints)} endpoints, "
                        f"{len(realBifurcations)} bifurcations "
                        f"(demoted {len(demoted)} noise nodes)"
                    )

                    # ── step 4: walk branches in compressed graph ─────────────────
                    # Walk from every real critical node along every unvisited edge
                    # until the next real critical node (skipping demoted pass-throughs).
                    visitedCEdges = set()
                    graphBranches = []  # list of (pts_list) — positions along edge

                    def _walkCompressed(startNode, firstNb):
                        """Walk compressed graph from startNode through firstNb,
                        skipping demoted nodes, until next real critical node."""
                        ekey = (min(startNode, firstNb), max(startNode, firstNb))
                        pts = list(
                            cEdge.get(ekey, [nodePos[startNode], nodePos[firstNb]])
                        )
                        if nodePos[startNode] != pts[0]:
                            pts = list(reversed(pts))
                        cur = firstNb
                        prev = startNode
                        while cur not in realCritical:
                            nxts = [nb for nb in cAdj[cur] if nb != prev]
                            if not nxts:
                                break
                            nxt = nxts[0]
                            ek2 = (min(cur, nxt), max(cur, nxt))
                            seg2 = cEdge.get(ek2, [nodePos[cur], nodePos[nxt]])
                            if seg2[0] != nodePos[cur]:
                                seg2 = list(reversed(seg2))
                            pts.extend(seg2[1:])
                            prev = cur
                            cur = nxt
                        return pts, cur

                    for startNode in sorted(realCritical):
                        for nb in sorted(cAdj[startNode]):
                            ekey = (min(startNode, nb), max(startNode, nb))
                            if ekey in visitedCEdges:
                                continue
                            visitedCEdges.add(ekey)
                            pts, endNode = _walkCompressed(startNode, nb)
                            if len(pts) >= 2:
                                graphBranches.append(pts)

                    print(
                        f"[VesselAnalyzer] Graph walk: {len(graphBranches)} branches "
                        f"before length prune"
                    )

                    # ── step 5: prune short branches ──────────────────────────────
                    graphBranches = [
                        b for b in graphBranches if _seg_length(b) >= MIN_BRANCH_MM
                    ]
                    print(
                        f"[VesselAnalyzer] After prune (≥{MIN_BRANCH_MM}mm): "
                        f"{len(graphBranches)} branches"
                    )

                    # ── step 5a: loop / noise suppression ─────────────────────────
                    #
                    # Drops redundant loop edges that arise from graph cycles
                    # (e.g. two paths between the same pair of bifurcation nodes).
                    # Two-tier logic:
                    #
                    #   Tier 1 — topology + ratio:
                    #     Both endpoints degree >= 3, an alternative path exists
                    #     without this branch, AND branch arc < SHORT_RATIO × alt path.
                    #
                    #   Tier 2 — absolute safety net:
                    #     Both endpoints degree >= 3, endpoint pair shared with
                    #     another branch, AND branch arc < LOOP_ABS_MAX_MM.
                    #
                    # Arc lengths use _seg_length() (true Euclidean arc) so
                    # SHORT_RATIO comparisons are geometrically meaningful.
                    # Must run BEFORE trunk detection so usage/score metrics
                    # are computed on clean topology only.

                    _LOOP_SHORT_RATIO = 0.60  # tier-1: branch < 60% of alt path
                    _LOOP_ABS_MAX_MM = 30.0  # tier-2: absolute cap (mm)
                    # 30 mm keeps real side branches (renals ~30-80mm) while
                    # still dropping true graph-cycle duplicates (always shorter
                    # than their alternate path partner).

                    def _closestCritNode(p):
                        best_n, best_d = None, 1e18
                        for _cn in realCritical:
                            _d = _dist3(nodePos[_cn], p)
                            if _d < best_d:
                                best_d = _d
                                best_n = _cn
                        return best_n

                    def _bfsPathLen(src, dst, adj_exclude_edge):
                        """BFS arc-length from src to dst, excluding one directed edge."""
                        # adj_exclude_edge = frozenset({src, dst}) — the edge to skip
                        from collections import deque as _deque

                        if src == dst:
                            return 0.0
                        visited = {src: (None, 0.0)}
                        queue = _deque([src])
                        while queue:
                            cur = queue.popleft()
                            for nb in cAdj.get(cur, set()):
                                ekey = frozenset({cur, nb})
                                if ekey == adj_exclude_edge:
                                    continue
                                if nb not in visited:
                                    ek2 = (min(cur, nb), max(cur, nb))
                                    seg = _seg_length(
                                        cEdge.get(ek2, [nodePos[cur], nodePos[nb]])
                                    )
                                    visited[nb] = (cur, visited[cur][1] + seg)
                                    if nb == dst:
                                        return visited[nb][1]
                                    queue.append(nb)
                        return None  # no path

                    # Build endpoint-pair index for tier-2 shared-endpoint detection
                    _epPairs = {}
                    for _bi, _bpts in enumerate(graphBranches):
                        _na = _closestCritNode(_bpts[0])
                        _nb = _closestCritNode(_bpts[-1])
                        if _na is not None and _nb is not None and _na != _nb:
                            _key = (min(_na, _nb), max(_na, _nb))
                            _epPairs.setdefault(_key, []).append(_bi)

                    _loopKept = []
                    _loopDropped = []
                    for _bi, _bpts in enumerate(graphBranches):
                        _na = _closestCritNode(_bpts[0])
                        _nb = _closestCritNode(_bpts[-1])
                        _arc = _seg_length(_bpts)
                        _key = (min(_na, _nb), max(_na, _nb)) if (_na and _nb) else None

                        # degree check — both endpoints must be bifurcation nodes
                        # (required for Tier 1 and Tier 2; Tier 3 runs independently)
                        _deg_a = len(cAdj.get(_na, set())) if _na is not None else 0
                        _deg_b = len(cAdj.get(_nb, set())) if _nb is not None else 0
                        _both_junctions = _deg_a >= 3 and _deg_b >= 3

                        # ── Tier 3: orphaned short stub ───────────────────────────
                        # Catches stubs whose loop partner was already dropped,
                        # leaving them as degree-1 danglers with no redundancy signal.
                        #
                        # Key insight: real side branches (renal veins etc.) have their
                        # HIGH-DEGREE root snapped tightly onto a trunk/bifurcation node.
                        # Orphaned noise stubs have their high-degree root sitting on a
                        # graph node that is NOT part of any retained trunk edge — it's
                        # a floating junction from the original loop that lost its partner.
                        #
                        # Rule: drop if ALL hold:
                        #   a) short (< STUB_MAX_MM)
                        #   b) exactly one degree-1 endpoint (true dangling tip)
                        #   c) the HIGH-DEGREE root node is NOT a confirmed bifurcation
                        #      node AND is far (> STUB_ROOT_DIST_MM) from all bifurcations
                        #      — real branches root at confirmed bifs; orphans don't
                        #   d) NOT a lateral departure (angle to trunk axis < LATERAL_SAFE_DEG)
                        #      — real side branches (renals) depart at large angles and
                        #      are never noise regardless of sibling count
                        _STUB_MAX_MM = 50.0  # mm
                        _STUB_ROOT_DIST_MM = (
                            15.0  # mm from nearest confirmed bifurcation
                        )
                        _STUB_LATERAL_DEG = 45.0  # branches departing >= this are kept
                        _PROX_SAMPLE_MM = 15.0  # mm to sample for departure direction
                        # 15mm (was 8mm): short branches rooted at trunk/bif nodes have
                        # their first ~8mm dominated by trunk curvature, making them
                        # appear falsely axial.  15mm reliably escapes the curvature
                        # zone and captures the true lateral departure of renal vessels.
                        if _arc < _STUB_MAX_MM:
                            _root_node = None  # the high-degree junction end
                            if _deg_a == 1 and _deg_b >= 3:
                                _root_node = _nb
                            elif _deg_b == 1 and _deg_a >= 3:
                                _root_node = _na
                            if _root_node is not None:
                                # ── lateral angle guard — keep real side branches ──
                                # Use proximal departure direction (first _PROX_SAMPLE_MM
                                # from the root end) rather than the end-to-end vector.
                                # End-to-end vectors for short branches rooted at bif
                                # nodes are dominated by the graph walk direction and
                                # appear falsely axial even for lateral vessels.
                                #
                                # Identify which end of _bpts is the root (high-degree) end:
                                # it is the end closest to nodePos[_root_node].
                                _rpos = nodePos[_root_node]
                                _d0 = _dist3(_bpts[0], _rpos)
                                _d1 = _dist3(_bpts[-1], _rpos)
                                if _d0 <= _d1:
                                    _prox_pts = _bpts  # root at [0], walk forward
                                    _step_dir = 1
                                else:
                                    _prox_pts = _bpts  # root at [-1], walk backward
                                    _step_dir = -1
                                # Walk _PROX_SAMPLE_MM from root end
                                _root_pt = (
                                    _prox_pts[0] if _step_dir == 1 else _prox_pts[-1]
                                )
                                _sample_pt = _root_pt
                                _acc = 0.0
                                _idxs = (
                                    range(len(_prox_pts) - 1)
                                    if _step_dir == 1
                                    else range(len(_prox_pts) - 1, 0, -1)
                                )
                                for _pi in _idxs:
                                    _nxt = _pi + 1 if _step_dir == 1 else _pi - 1
                                    _acc += _dist3(_prox_pts[_pi], _prox_pts[_nxt])
                                    if _acc >= _PROX_SAMPLE_MM:
                                        _sample_pt = _prox_pts[_nxt]
                                        break
                                    _sample_pt = _prox_pts[_nxt]
                                _bvec = [_sample_pt[k] - _root_pt[k] for k in range(3)]
                                _bvl = math.sqrt(sum(x * x for x in _bvec))
                                if _bvl > 1e-6:
                                    _bvec = [x / _bvl for x in _bvec]
                                    # Trunk axis proxy: Z-axis (craniocaudal in axial CT).
                                    _stub_dot = abs(_bvec[2])
                                    _stub_ang = math.degrees(
                                        math.acos(min(1.0, _stub_dot))
                                    )
                                else:
                                    _stub_ang = 0.0
                                if _stub_ang >= _STUB_LATERAL_DEG:
                                    # Lateral — real side branch, never drop in T3
                                    _loopKept.append(_bpts)
                                    continue

                                _is_confirmed_bif = _root_node in realBifurcations
                                if not _is_confirmed_bif:
                                    # Root is not a confirmed bif — classic orphaned stub
                                    _root_pos = nodePos[_root_node]
                                    _min_bif_dist = min(
                                        (
                                            _dist3(_root_pos, nodePos[_bn])
                                            for _bn in realBifurcations
                                        ),
                                        default=1e18,
                                    )
                                    if _min_bif_dist > _STUB_ROOT_DIST_MM:
                                        print(
                                            f"[LoopFilter T3] Branch {_bi}: "
                                            f"{_arc:.1f}mm stub, root {_root_node} "
                                            f"not confirmed bif, {_min_bif_dist:.1f}mm away"
                                            f" → orphaned stub, dropped"
                                        )
                                        _loopDropped.append(_bi)
                                        continue
                                else:
                                    # Root IS a confirmed bif — only drop if the node
                                    # has very high degree (>= 5), meaning a genuine
                                    # redundant stub in a dense loop cluster.
                                    # DO NOT drop based on sibling count alone:
                                    # a renal vein at a degree-3/4 trunk bif always has
                                    # 3+ longer siblings (trunk-in, trunk-out, iliac)
                                    # and would be wrongly eliminated.  The lateral guard
                                    # above is the primary renal-protection mechanism;
                                    # this path is a last-resort catch for pathological
                                    # graph noise only.
                                    _root_deg = len(cAdj.get(_root_node, set()))
                                    # Count retained branches (not yet dropped) that
                                    # share this root node AND are longer than this stub.
                                    # NOTE: parentheses are required — without them the
                                    # 'or' binds looser than 'and', causing length check
                                    # to only apply to the end-node match branch.
                                    _longer_siblings = sum(
                                        1
                                        for _oi, _opts in enumerate(graphBranches)
                                        if _oi != _bi
                                        and _oi not in _loopDropped
                                        and _seg_length(_opts) > _arc
                                        and (
                                            _closestCritNode(_opts[0]) == _root_node
                                            or _closestCritNode(_opts[-1]) == _root_node
                                        )
                                    )
                                    if _root_deg >= 5:
                                        print(
                                            f"[LoopFilter T3] Branch {_bi}: "
                                            f"{_arc:.1f}mm stub at confirmed bif {_root_node} "
                                            f"(deg={_root_deg}, {_longer_siblings} longer siblings)"
                                            f" → excess stub, dropped"
                                        )
                                        _loopDropped.append(_bi)
                                        continue

                        if not _both_junctions:
                            _loopKept.append(_bpts)
                            continue

                        # ── Tier 1: topology + ratio ──────────────────────────────
                        _excl = frozenset({_na, _nb})
                        _altLen = _bfsPathLen(_na, _nb, _excl)
                        if _altLen is not None and _arc < _LOOP_SHORT_RATIO * _altLen:
                            print(
                                f"[LoopFilter T1] Branch {_bi}: "
                                f"{_arc:.1f}mm < {_LOOP_SHORT_RATIO}×{_altLen:.1f}mm "
                                f"alt path → loop noise, dropped"
                            )
                            _loopDropped.append(_bi)
                            continue

                        # ── Tier 2: absolute safety net ───────────────────────────
                        if _key is not None:
                            _siblings = [i for i in _epPairs.get(_key, []) if i != _bi]
                            if _siblings and _arc < _LOOP_ABS_MAX_MM:
                                print(
                                    f"[LoopFilter T2] Branch {_bi}: "
                                    f"{_arc:.1f}mm < {_LOOP_ABS_MAX_MM}mm, "
                                    f"shares endpoints with branch(es) {_siblings}"
                                    f" → loop noise, dropped"
                                )
                                _loopDropped.append(_bi)
                                continue

                        _loopKept.append(_bpts)

                    if _loopDropped:
                        print(
                            f"[LoopFilter] Suppressed {len(_loopDropped)} loop "
                            f"branch(es): indices {_loopDropped}"
                        )
                        graphBranches = _loopKept
                    else:
                        print(f"[LoopFilter] No loop branches detected")

                    # ── step 5b: trunk detection via usage × length score ─────────
                    #
                    # Correct trunk definition (robust for any anatomy):
                    #   Trunk = the connected set of edges that maximises score,
                    #   where score(edge) = path_usage × arc_length.
                    #
                    # Why usage×length and not usage alone:
                    #   A central hub node can have high usage simply because every
                    #   path passes through it — but the hub segment itself may be
                    #   only a few mm long (a bifurcation junction).  Pure usage
                    #   selects that short hub as "trunk".  Multiplying by length
                    #   correctly rewards the long shared vessel (IVC, common iliac)
                    #   over the short junction region.
                    #
                    # Algorithm:
                    #   1. BFS all endpoint pairs → edge usage counts.
                    #   2. Score each edge = usage × arc_length.
                    #   3. Trunk = connected component of edges with score ≥ 70th
                    #      percentile of all edge scores (percentile avoids having to
                    #      tune an absolute threshold).
                    #   4. Within the trunk component, find the longest simple path
                    #      (its ordered points become the committed trunk branch).
                    #   5. Deduplicate graphBranches: remove any branch whose points
                    #      are the reverse of another (VMTK sometimes generates both
                    #      directions of the same path).
                    #   6. Order: trunk first, then limbs by arc length descending.

                    from collections import Counter as _Counter

                    # BFS shortest path on compressed graph
                    from centerline_graph import graph_bfs_path as _bfsPath

                    # Step 1: count how many endpoint-pair paths use each edge
                    edgeUsage = _Counter()
                    epNodes = realEndpoints
                    for i in range(len(epNodes)):
                        for j in range(i + 1, len(epNodes)):
                            path = _bfsPath(epNodes[i], epNodes[j], cAdj)
                            for k in range(len(path) - 1):
                                edgeUsage[frozenset({path[k], path[k + 1]})] += 1

                    # Step 2: score = usage × arc × mean_diameter
                    #
                    # Diameter-aware scoring ensures the trunk (wide, shared vessel)
                    # scores higher than long narrow side branches.  We sample diameter
                    # at the edge midpoint using the surface mesh so the score reflects
                    # true vessel calibre, not just geometric length.
                    #
                    # Fallback when no modelNode: diameter = 1.0 (neutral, score = usage×arc).

                    def _edgeArc(ekey):
                        a, b = tuple(ekey)
                        ek2 = (min(a, b), max(a, b))
                        ep = cEdge.get(ek2, [nodePos[a], nodePos[b]])
                        return _seg_length(ep)

                    def _edgeMidpt(ekey):
                        a, b = tuple(ekey)
                        ek2 = (min(a, b), max(a, b))
                        ep = cEdge.get(ek2, [nodePos[a], nodePos[b]])
                        if not ep:
                            return nodePos[a]
                        mid_i = len(ep) // 2
                        return ep[mid_i]

                    # Build a temporary surface locator for diameter sampling
                    # (lightweight closest-face; full ray casting runs post-commit)
                    _eSurfLoc = None
                    if modelNode:
                        _epd = modelNode.GetPolyData()
                        if _epd:
                            _eSurfLoc = vtk.vtkCellLocator()
                            _eSurfLoc.SetDataSet(_epd)
                            _eSurfLoc.BuildLocator()

                    def _edgeDiam(ekey):
                        """Mean diameter of an edge via closest-face sampling at midpoint."""
                        if _eSurfLoc is None:
                            return 1.0  # neutral fallback
                        mid = _edgeMidpt(ekey)
                        cp = [0.0, 0.0, 0.0]
                        ci = vtk.reference(0)
                        si = vtk.reference(0)
                        d2 = vtk.reference(0.0)
                        _eSurfLoc.FindClosestPoint(list(mid), cp, ci, si, d2)
                        r = math.sqrt(float(d2))
                        return max(r * 2.0, 0.5)  # diameter, min 0.5mm

                    edgeScore = {
                        ek: edgeUsage[ek] * _edgeArc(ek) * _edgeDiam(ek)
                        for ek in edgeUsage
                    }

                    print(f"[VesselAnalyzer] Edge scores (usage x arc x diam):")
                    for ek, sc in sorted(edgeScore.items(), key=lambda x: -x[1]):
                        a, b = tuple(ek)
                        print(
                            f"  edge {a}-{b}: usage={edgeUsage[ek]}, "
                            f"arc={_edgeArc(ek):.1f}mm, "
                            f"diam={_edgeDiam(ek):.1f}mm, score={sc:.0f}"
                        )

                    # Step 3: select trunk edges by USAGE FIRST, arc second.
                    #
                    # Why usage×arc fails here:
                    #   A long leaf-limb (usage=4, arc=200mm → score=800) outscores
                    #   the true shared trunk (usage=6, arc=70mm → score=420) because
                    #   the arc factor dominates.  The true trunk is the edge traversed
                    #   by the MOST endpoint-pair paths — highest usage count.
                    #
                    # Algorithm:
                    #   1. Find max_usage across all edges.
                    #   2. Trunk candidate edges = those with usage >= max_usage - 1
                    #      (allow one below max so a near-equally-shared hub isn't missed).
                    #   3. If that gives an isolated set (no connected component of ≥ 2
                    #      nodes), fall back to the top-30% by usage×arc score.
                    if edgeUsage:
                        max_usage = max(edgeUsage.values())
                        # Trunk edges: high usage AND minimum arc length.
                        # Micro-stubs (e.g. 7.7mm, 1.9mm junction artefacts) have
                        # max usage but are NOT anatomical trunk segments — filtering
                        # them here keeps trunkNodePath clean and avoids inflating the
                        # stitched trunk with tiny connector segments.
                        TRUNK_MIN_ARC_MM = (
                            MIN_BRANCH_MM  # same threshold as branch prune
                        )
                        trunkEks = {
                            ek
                            for ek, u in edgeUsage.items()
                            if u >= max_usage - 1 and _edgeArc(ek) >= TRUNK_MIN_ARC_MM
                        }
                        # If filtering eliminates all candidates, relax to arc-only check
                        if not trunkEks:
                            trunkEks = {
                                ek for ek, u in edgeUsage.items() if u >= max_usage - 1
                            }
                            print(
                                f"[VesselAnalyzer] Trunk min-arc filter relaxed "
                                f"(no edges ≥ {TRUNK_MIN_ARC_MM}mm with max usage)"
                            )
                        print(
                            f"[VesselAnalyzer] Trunk edges (usage >= {max_usage-1}, "
                            f"arc >= {TRUNK_MIN_ARC_MM}mm): "
                            f"{len(trunkEks)} (max_usage={max_usage})"
                        )
                    else:
                        trunkEks = set()
                        print(
                            f"[VesselAnalyzer] Trunk edges: none (no edge usage data)"
                        )

                    # Step 4: within trunk edges, find longest simple path.
                    #
                    # Key: micro-stub edges (< TRUNK_MIN_ARC_MM) are excluded from
                    # trunkEks to avoid inflating the stitched trunk, but they are
                    # needed to CONNECT the trunk components in trunkAdj.  Without
                    # them, high-usage long segments that share a micro-stub junction
                    # appear as disconnected islands and the DFS only sees one segment.
                    #
                    # Solution: build trunkAdj from ALL max-usage edges (ignoring arc
                    # length), so the graph is always connected.  The micro-stub edges
                    # are traversed in the DFS but contribute negligible arc weight —
                    # they don't distort the longest-path selection.
                    trunkNodes = set()
                    for ek in trunkEks:
                        a, b = tuple(ek)
                        trunkNodes.add(a)
                        trunkNodes.add(b)

                    # ── Suprarenal chain extension ────────────────────────────
                    # Problem: the trunk stitch only picks up edge 112-318
                    # (usage=6, arc=87.8mm).  Beyond node 318 the suprarenal
                    # IVC continues through micro-stub connectors:
                    #   318→322 (3.1mm, usage=6) → 333 (13.7mm, usage=6) → 492 (99.1mm, usage=4)
                    # Both connector edges have max usage but arc < TRUNK_MIN_ARC_MM,
                    # so they are excluded from trunkEks.  The leaf edge 333-492
                    # has usage=4 (below max_usage-1=5), so it also stays out.
                    # Result: nodes 322, 333, 492 never enter trunkNodes and the
                    # suprarenal IVC is mistakenly treated as a side branch.
                    #
                    # Fix: from each current trunk endpoint (degree-1 node in
                    # trunkAdj after the main edges are added), walk outward
                    # through any high-usage (>= max_usage-1) edge chain as
                    # long as it is a simple continuation (degree ≤ 2 in that
                    # sub-graph — no branching).  Branching means the chain
                    # reaches a real bifurcation; a non-branching chain is just
                    # a segmented representation of the same tube and belongs in
                    # the trunk.
                    #
                    # Build adjacency for ALL edges (any usage, any arc length).
                    _allEdgeAdj = {}  # node → {neighbor: arc_mm}
                    _edgeUsageAdj = {}  # node → {neighbor: usage_count}
                    for ek, u in edgeUsage.items():
                        a, b = tuple(ek)
                        arc = _edgeArc(ek)
                        _allEdgeAdj.setdefault(a, {})[b] = arc
                        _allEdgeAdj.setdefault(b, {})[a] = arc
                        _edgeUsageAdj.setdefault(a, {})[b] = u
                        _edgeUsageAdj.setdefault(b, {})[a] = u

                    # ── Suprarenal chain extension ────────────────────────────
                    # Walk outward from trunk endpoints to absorb any linear
                    # continuation that the usage-threshold filter missed.
                    #
                    # Trunk continuation direction is identified by usage count:
                    # the edge with the HIGHEST usage leaving a node is the trunk
                    # backbone.  For a genuine side branch, the trunk-backbone edge
                    # has higher usage than the departing branch edge.
                    #
                    # At each candidate node _nb reached via edge (_tn→_nb):
                    #   1. Collect all outside exits from _nb (not in trunkNodes).
                    #   2. If none → leaf endpoint, absorb only if the incoming
                    #      arc is the max-usage edge from _tn outward.
                    #   3. If the max-usage outside exit from _nb has usage ≥
                    #      the incoming edge usage → clear continuation, absorb.
                    #   4. If two exits have equal max usage → Y-bif, stop.
                    #
                    # Walk trace for this anatomy:
                    #   318→322: usage=6, 322's outside exits: 333(u=6), 417(u=4)
                    #     max outside usage=6 ≥ incoming usage=6 → absorb 322 ✓
                    #   322→333: usage=6, 333's outside exits: 492(u=4), 367(u=4)
                    #     both exits have same usage=4 < incoming=6 → 333 is a
                    #     terminal bif with only short side branches → absorb 333,
                    #     then follow dominant arc (99.1mm to 492) ✓
                    #   322→417: usage=4 < max outgoing from 322 (=6 via 333) →
                    #     417 is a side branch, skip ✓
                    #   333→492: leaf, arc=99.1mm = max-arc from 333 outward → absorb ✓
                    #   333→367: leaf, arc=31mm < max-arc (99.1mm) → skip ✓
                    _newly_added = True
                    while _newly_added:
                        _newly_added = False
                        for _tn in list(trunkNodes):
                            # Max usage of any edge leaving _tn outside trunk
                            _tn_out = {
                                x: (
                                    _edgeUsageAdj.get(_tn, {}).get(x, 0),
                                    _allEdgeAdj.get(_tn, {}).get(x, 0.0),
                                )
                                for x in _allEdgeAdj.get(_tn, {})
                                if x not in trunkNodes
                            }
                            if not _tn_out:
                                continue
                            _tn_max_u = max(v[0] for v in _tn_out.values())
                            _tn_max_arc = max(v[1] for v in _tn_out.values())

                            for _nb, (_nb_u, _nb_arc) in _tn_out.items():
                                # Skip edges whose usage is below the max-usage
                                # leaving _tn — those are side branches, not trunk.
                                if _nb_u < _tn_max_u:
                                    continue

                                # Outside exits from _nb
                                _nb_out_u = {
                                    x: _edgeUsageAdj.get(_nb, {}).get(x, 0)
                                    for x in _allEdgeAdj.get(_nb, {})
                                    if x not in trunkNodes and x != _tn
                                }
                                _nb_out_arc = {
                                    x: _allEdgeAdj.get(_nb, {}).get(x, 0.0)
                                    for x in _allEdgeAdj.get(_nb, {})
                                    if x not in trunkNodes and x != _tn
                                }

                                if len(_nb_out_u) == 0:
                                    # Leaf endpoint: absorb only if this is the
                                    # dominant-arc edge from _tn (not a short stub)
                                    if _nb_arc < _tn_max_arc:
                                        continue
                                elif len(_nb_out_u) >= 2:
                                    # Check: are there two equally dominant exits?
                                    _sorted_u = sorted(_nb_out_u.values(), reverse=True)
                                    if _sorted_u[0] == _sorted_u[1]:
                                        # Equal-usage Y-bif: check arc ratio
                                        _sorted_arc = sorted(
                                            _nb_out_arc.values(), reverse=True
                                        )
                                        _ratio = (
                                            _sorted_arc[0] / _sorted_arc[1]
                                            if _sorted_arc[1] > 0
                                            else float("inf")
                                        )
                                        if _ratio < 2.5:
                                            continue  # genuine equal Y-split — stop

                                # ── Directional gate ─────────────────────────
                                # Reject extension if the candidate edge points
                                # strongly AWAY from the current trunk direction.
                                # This prevents a high-usage side branch from
                                # leaking into the trunk chain in noisy graphs.
                                # Gate: dot(trunk_dir, edge_dir) > -0.3
                                # (allows up to ~107° — very permissive, only
                                # blocks u-turns and near-perpendicular exits).
                                # Requires ≥2 confirmed trunk nodes for direction.
                                if (
                                    len(trunkNodes) >= 2
                                    and _nb in nodePos
                                    and _tn in nodePos
                                ):
                                    _tn_pos = nodePos[_tn]
                                    _nb_pos = nodePos[_nb]
                                    # Current trunk direction: last trunk node → _tn
                                    # (approximate from the set; use the node that
                                    # was added most recently — we use nodePos diff)
                                    _prev = next(
                                        (
                                            n
                                            for n in trunkNodes
                                            if n != _tn and n in nodePos
                                        ),
                                        None,
                                    )
                                    if _prev is not None:
                                        _td = tuple(
                                            nodePos[_tn][k] - nodePos[_prev][k]
                                            for k in range(3)
                                        )
                                        _ed = tuple(
                                            _nb_pos[k] - _tn_pos[k] for k in range(3)
                                        )
                                        _td_len = sum(x * x for x in _td) ** 0.5
                                        _ed_len = sum(x * x for x in _ed) ** 0.5
                                        if _td_len > 1e-6 and _ed_len > 1e-6:
                                            _dot = sum(
                                                _td[k] * _ed[k] for k in range(3)
                                            ) / (_td_len * _ed_len)
                                            if _dot < -0.3:
                                                continue  # u-turn — not trunk

                                trunkNodes.add(_nb)
                                _newly_added = True
                                print(
                                    f"[VesselAnalyzer] Trunk chain extended: "
                                    f"node {_tn} → {_nb} "
                                    f"(arc={_nb_arc:.1f}mm, usage={_nb_u})"
                                )

                    # Add connecting edges: any max-usage edge between trunk nodes
                    # (including micro-stubs filtered out of trunkEks)
                    trunkAdj = {n: set() for n in trunkNodes}
                    for ek in trunkEks:
                        a, b = tuple(ek)
                        trunkAdj[a].add(b)
                        trunkAdj[b].add(a)
                    # Bridge ALL edges between trunk nodes — including lower-usage
                    # leaf edges (e.g. 333-492, usage=4) that were absorbed into
                    # trunkNodes by the chain extension above.  The extension
                    # already validated they are linear continuations; excluding
                    # them from trunkAdj here would leave the terminal trunk node
                    # disconnected so _longestPath could never reach it.
                    for ek in edgeUsage:
                        a, b = tuple(ek)
                        if a in trunkNodes and b in trunkNodes:
                            trunkAdj[a].add(b)
                            trunkAdj[b].add(a)

                    # DFS to find longest arc-weighted simple path (trunk graph is tiny)
                    # We weight by arc length so the longest *physical* path wins,
                    # not just the one with the most intermediate nodes.
                    def _longestPath(adj, nodes):
                        best_path = []
                        best_arc = -1.0

                        def _dfs(cur, visited, path, arc):
                            nonlocal best_path, best_arc
                            if arc > best_arc:
                                best_arc = arc
                                best_path = list(path)
                            for nb in adj.get(cur, set()):
                                if nb not in visited:
                                    ek2 = (min(cur, nb), max(cur, nb))
                                    seg = _seg_length(
                                        cEdge.get(ek2, [nodePos[cur], nodePos[nb]])
                                    )
                                    visited.add(nb)
                                    path.append(nb)
                                    _dfs(nb, visited, path, arc + seg)
                                    path.pop()
                                    visited.discard(nb)

                        for start in nodes:
                            _dfs(start, {start}, [start], 0.0)
                        return best_path  # list of crit node IDs

                    trunkNodePath = _longestPath(trunkAdj, trunkNodes)
                    print(f"[VesselAnalyzer] Trunk node path: {trunkNodePath}")

                    # ── Highest-Z endpoint trunk correction ───────────────────
                    # Problem: in venous anatomy (IVC + iliacs) the path with
                    # max-usage often runs along one iliac limb rather than the
                    # true IVC trunk above the bifurcation.  Result: the IVC
                    # inlet (highest-Z endpoint) is treated as a side branch.
                    #
                    # Invariant: the true trunk MUST pass through the highest-Z
                    # endpoint (IVC inlet / aortic root) because that endpoint
                    # is the proximal anchor of the vessel tree.  If it is not
                    # in trunkNodePath, rebuild the trunk by finding the path
                    # from that endpoint through the primary bifurcation node
                    # using BFS over the full compressed graph (_allEdgeAdj).
                    #
                    # This runs BEFORE trunkPtsOrdered is built, so no geometric
                    # data needs to change — only the node path list is updated.
                    # ─────────────────────────────────────────────────────────
                    if realEndpoints and nodePos:
                        _ep_z = {
                            ep: nodePos[ep][2]
                            for ep in realEndpoints
                            if ep < len(nodePos)
                        }
                        if _ep_z:
                            _maxZ_ep = max(_ep_z, key=lambda n: _ep_z[n])
                            _trunkNodeSet = set(trunkNodePath)

                            if _maxZ_ep not in _trunkNodeSet:
                                # BFS from _maxZ_ep through full graph to find
                                # shortest node path to any node in trunkNodePath.
                                from collections import deque as _deque

                                _bfs_q = _deque([[_maxZ_ep]])
                                _bfs_visited = {_maxZ_ep}
                                _bfs_found = None
                                while _bfs_q and _bfs_found is None:
                                    _bfs_path = _bfs_q.popleft()
                                    _bfs_cur = _bfs_path[-1]
                                    for _bfs_nb in _allEdgeAdj.get(_bfs_cur, {}):
                                        if _bfs_nb in _bfs_visited:
                                            continue
                                        _bfs_new_path = _bfs_path + [_bfs_nb]
                                        if _bfs_nb in _trunkNodeSet:
                                            _bfs_found = (_bfs_new_path, _bfs_nb)
                                            break
                                        _bfs_visited.add(_bfs_nb)
                                        _bfs_q.append(_bfs_new_path)

                                if _bfs_found is not None:
                                    _bridge_path, _bridge_join = _bfs_found
                                    # _bridge_path runs: maxZ_ep → ... → join_node
                                    # trunkNodePath runs: [some_end ... join_node ... other_end]
                                    # We want new trunk:  maxZ_ep → ... → join → ... → far_end
                                    # where far_end is whichever trunk end is NOT join_node.
                                    #
                                    # Find join index and pick the sub-path AWAY from join
                                    # (i.e. the trunk segment past the join, toward the other end).
                                    _join_idx = trunkNodePath.index(_bridge_join)
                                    # Trunk tail = nodes after join (toward far end)
                                    # Trunk head = nodes before join (toward near end, discarded)
                                    _trunk_tail = trunkNodePath[
                                        _join_idx:
                                    ]  # join → far_end
                                    # Also consider trunk going the other direction from join
                                    _trunk_head_rev = list(
                                        reversed(trunkNodePath[: _join_idx + 1])
                                    )  # join → near_end

                                    # Pick the trunk direction that goes AWAY from maxZ_ep.
                                    # The correct direction is the one whose first step
                                    # (after join) has a lower Z than the bridge arrival.
                                    # Simple heuristic: pick the longer tail.
                                    if len(_trunk_tail) >= len(_trunk_head_rev):
                                        _chosen_tail = _trunk_tail
                                    else:
                                        _chosen_tail = _trunk_head_rev

                                    # New path: bridge (maxZ_ep → join) + tail (join → far end)
                                    # _bridge_path already ends with _bridge_join, so skip it
                                    # at the start of _chosen_tail (which also starts with join).
                                    _new_trunk = _bridge_path + _chosen_tail[1:]

                                    print(
                                        f"[TrunkZ] Highest-Z endpoint node {_maxZ_ep} "
                                        f"(Z={_ep_z[_maxZ_ep]:.1f}) not in trunk path — "
                                        f"rebuilding via bridge through node {_bridge_join}; "
                                        f"new path={_new_trunk} (length={len(_new_trunk)} nodes)"
                                    )
                                    trunkNodePath = _new_trunk
                                    # Absorb bridge nodes into trunkNodes so
                                    # the geometric stitching loop below can
                                    # find cEdge segments for them.
                                    for _bn in _bridge_path:
                                        trunkNodes.add(_bn)
                                        for _bnn in _allEdgeAdj.get(_bn, {}):
                                            if _bnn in trunkNodes:
                                                trunkAdj.setdefault(_bn, set()).add(
                                                    _bnn
                                                )
                                                trunkAdj.setdefault(_bnn, set()).add(
                                                    _bn
                                                )
                                    # Verify all consecutive node pairs in new path
                                    # have cEdge entries — log any gaps
                                    for _pi in range(len(_new_trunk) - 1):
                                        _pa, _pb = _new_trunk[_pi], _new_trunk[_pi + 1]
                                        _pek = (min(_pa, _pb), max(_pa, _pb))
                                        _has_edge = _pek in cEdge
                                        _za = (
                                            nodePos[_pa][2]
                                            if _pa < len(nodePos)
                                            else "?"
                                        )
                                        _zb = (
                                            nodePos[_pb][2]
                                            if _pb < len(nodePos)
                                            else "?"
                                        )
                                        print(
                                            f"[TrunkZ]   edge {_pa}({_za:.1f})→{_pb}({_zb:.1f}): cEdge={'✓' if _has_edge else '✗ MISSING'}"
                                        )
                                else:
                                    print(
                                        f"[TrunkZ] Highest-Z endpoint node {_maxZ_ep} "
                                        f"(Z={_ep_z[_maxZ_ep]:.1f}) not reachable from "
                                        f"trunk — keeping original trunk path"
                                    )
                            else:
                                print(
                                    f"[TrunkZ] Highest-Z endpoint node {_maxZ_ep} "
                                    f"(Z={_ep_z[_maxZ_ep]:.1f}) already in trunk path ✓"
                                )

                    # Collect the geometric points for this trunk node path
                    # by concatenating cEdge segments in order.
                    # Start with nodePos[trunkNodePath[0]] so the trunk begins
                    # exactly at the primary bif node position (not at the first
                    # interior cEdge sample which may be a few mm away).
                    trunkPtsOrdered = []
                    trunkRadiiOrdered = []
                    _tn0 = trunkNodePath[0]
                    if 0 <= _tn0 < len(nodePos):
                        trunkPtsOrdered.append(nodePos[_tn0])  # pin bif end exactly
                    for k in range(len(trunkNodePath) - 1):
                        a = trunkNodePath[k]
                        b = trunkNodePath[k + 1]
                        ek2 = (min(a, b), max(a, b))
                        epts = list(cEdge.get(ek2, [nodePos[a], nodePos[b]]))
                        if (
                            epts
                            and trunkPtsOrdered
                            and _dist3(epts[0], trunkPtsOrdered[-1])
                            > _dist3(epts[-1], trunkPtsOrdered[-1])
                        ):
                            epts = list(reversed(epts))
                        if trunkPtsOrdered:
                            # Skip points within 0.5mm of the last committed point
                            # to avoid duplicates from the pinned bif-node anchor
                            epts = [
                                p for p in epts if _dist3(p, trunkPtsOrdered[-1]) > 0.5
                            ]
                        trunkPtsOrdered.extend(epts)
                    # Pin the root end too (last node in path)
                    _tnN = trunkNodePath[-1]
                    if (
                        0 <= _tnN < len(nodePos)
                        and trunkPtsOrdered
                        and _dist3(nodePos[_tnN], trunkPtsOrdered[-1]) > 0.5
                    ):
                        trunkPtsOrdered.append(nodePos[_tnN])

                    # Pre-compute _lockedBifNode here (before step 4b) using the same
                    # scoring logic that flow-norm applies later.  Flow-norm will
                    # recompute and overwrite — results are identical because edgeScore,
                    # realBifurcations, and cAdj are all available at this point.
                    # This is needed because step 4b (trunk truncation) must run BEFORE
                    # graphBranches is assembled, but _lockedBifNode is otherwise only
                    # assigned inside the downstream `if graphBranches:` block.
                    _lockedBifNode = None
                    _lockedBifScore = -1.0
                    for _tn in realBifurcations:
                        _sc = sum(
                            edgeScore.get(frozenset({_tn, _nb}), 0)
                            for _nb in cAdj.get(_tn, set())
                        )
                        if _sc > _lockedBifScore:
                            _lockedBifScore = _sc
                            _lockedBifNode = _tn
                    if _lockedBifNode is None:
                        _trunkPathSet_pre = set(trunkNodePath)
                        for _tn in trunkNodePath:
                            if (
                                len(
                                    [
                                        n
                                        for n in cAdj.get(_tn, set())
                                        if n not in _trunkPathSet_pre
                                    ]
                                )
                                > 0
                            ):
                                _lockedBifNode = _tn
                                break

                    # ── Step 4b: terminal-bifurcation trunk truncation ────────────
                    #
                    # The trunk chain extension (step 4a) greedily absorbs the
                    # highest-scoring path through every bifurcation node, including
                    # the PRIMARY bifurcation (aorto-iliac split).  This is correct
                    # for scoring purposes but semantically wrong: the aorta ends AT
                    # the primary bifurcation — it does not continue into one iliac.
                    #
                    # Fix (post-truncation, not stop-during-extension):
                    #   1. Locate _lockedBifNode in trunkNodePath.
                    #   2. Everything PAST that node in trunkNodePath is a terminal
                    #      iliac continuation — strip it from trunk.
                    #   3. Truncate trunkPtsOrdered at the geometric position of
                    #      _lockedBifNode (using trunkPtsOrdered directly, not nodePos).
                    #   4. The stripped continuation edge points are re-inserted into
                    #      graphBranches as a peer branch so both iliacs are co-equal.
                    #
                    # Robustness: we require _lockedBifNode to be an interior node
                    # (not at position 0 or -1 of trunkNodePath) — if the scored
                    # primary bif is at the trunk tip it was already a terminal node
                    # and no truncation is needed.
                    #
                    # This runs BEFORE dedup/sort/stitch so the stripped continuation
                    # participates normally in all downstream filtering.

                    _continuation_pts = None  # will hold stripped iliac segment if any
                    if _lockedBifNode is not None and _lockedBifNode in trunkNodePath:
                        _bif_idx = trunkNodePath.index(_lockedBifNode)
                        # Only truncate if bif is interior (not already the terminal node)
                        if 0 < _bif_idx < len(trunkNodePath) - 1:
                            # ── find geometric cut point in trunkPtsOrdered ──────────
                            # trunkPtsOrdered is built from cEdge geometry — not nodePos.
                            # We find the point in trunkPtsOrdered closest to nodePos of
                            # _lockedBifNode (the only reliable position anchor here),
                            # then split there.
                            _bif_geom = nodePos[_lockedBifNode]
                            _best_cut = min(
                                range(len(trunkPtsOrdered)),
                                key=lambda i: _dist3(trunkPtsOrdered[i], _bif_geom),
                            )
                            # Collect continuation points (past the bif cut)
                            _continuation_pts = trunkPtsOrdered[_best_cut:]
                            # Truncate trunk to aortic segment only
                            trunkPtsOrdered = trunkPtsOrdered[: _best_cut + 1]
                            trunkNodePath = trunkNodePath[: _bif_idx + 1]

                            _cont_arc = _seg_length(_continuation_pts)
                            _trk_z0 = trunkPtsOrdered[0][2] if trunkPtsOrdered else "?"
                            _trk_zN = trunkPtsOrdered[-1][2] if trunkPtsOrdered else "?"
                            print(
                                f"[BifTrunc] Primary bif node {_lockedBifNode} is interior "
                                f"(idx={_bif_idx}/{len(trunkNodePath)-1}) — "
                                f"trunk truncated to aorta only "
                                f"({_seg_length(trunkPtsOrdered):.1f}mm), "
                                f"Z range=[{_trk_z0:.1f}..{_trk_zN:.1f}], "
                                f"continuation ({_cont_arc:.1f}mm) already in graphBranches "
                                f"via graph walk — no prepend (avoids duplicate)"
                            )
                            # NOTE: do NOT prepend _continuation_pts to graphBranches.
                            # The edge (lockedBifNode→continuation) was already emitted
                            # by the graph walk and exists in graphBranches. After BifTrunc
                            # removes the continuation node from trunkNodePath, that edge
                            # is no longer eaten by _isTrunkSub (both endpoints must be in
                            # _trunkNodeSet, but the distal node is now outside it).
                            # Prepending would create a second copy → duplicate iliac.
                        else:
                            print(
                                f"[BifTrunc] Primary bif node {_lockedBifNode} "
                                f"already at trunk terminal (idx={_bif_idx}) — "
                                f"no truncation needed"
                            )

                    # Step 5: deduplicate graphBranches
                    # Two branches are duplicates if they share both endpoints regardless
                    # of traversal order (same-direction OR reversed).
                    from centerline_graph import (
                        graph_are_duplicate_edge as _areDuplicateEdge,
                    )

                    deduped = []
                    for pts in graphBranches:
                        if not any(_areDuplicateEdge(pts, seen) for seen in deduped):
                            deduped.append(pts)
                    if len(deduped) < len(graphBranches):
                        print(
                            f"[VesselAnalyzer] Deduplicated: "
                            f"{len(graphBranches)} → {len(deduped)} branches"
                        )
                    graphBranches = deduped

                    # Step 6: identify trunk branch in graphBranches
                    # The trunk branch is the one whose endpoints match the trunk node path ends
                    trunkEkSet = trunkEks

                    def _ptsToEdgeKey(pts):
                        def _closestCrit(p):
                            best_n = None
                            best_d = 1e18
                            for n in realCritical:
                                pos = nodePos[n]
                                d = (
                                    (p[0] - pos[0]) ** 2
                                    + (p[1] - pos[1]) ** 2
                                    + (p[2] - pos[2]) ** 2
                                )
                                if d < best_d:
                                    best_d = d
                                    best_n = n
                            return best_n

                        na = _closestCrit(pts[0])
                        nb = _closestCrit(pts[-1])
                        return frozenset({na, nb}) if na != nb else None

                    def _trunkScore(pts):
                        ek = _ptsToEdgeKey(pts)
                        sc = edgeScore.get(ek, 0) if ek else 0
                        return (-sc, -_seg_length(pts))

                    graphBranches.sort(key=_trunkScore)

                    # Replace branch[0] points with the stitched trunk path if it's longer
                    # (the stitched path spans multiple compressed edges; individual
                    # graphBranch entries only span one edge each)
                    if trunkPtsOrdered and len(trunkPtsOrdered) >= 2:
                        # Remove any graphBranch whose BOTH endpoints are trunk-path
                        # nodes — that means it's a sub-segment of the trunk.
                        # Using endpoint-node membership rather than edge-key lookup
                        # is more robust when short stubs (< MIN_BRANCH_MM) were pruned
                        # from graphBranches before trunk stitching.
                        _trunkNodeSet = set(trunkNodePath)
                        # Build a "trunk-adjacent" set: trunk nodes + any endpoint node
                        # that lies within TRUNK_SNAP_MM of any trunk node.
                        # This is needed because the VMTK graph can produce two nearby
                        # critical nodes at the same anatomical root location — e.g.
                        # node 113 (trunk path end) and node 510 (endpoint leaf) may
                        # both sit at the proximal aortic root but be 5-15mm apart
                        # geometrically (beyond the 3mm snap used previously).
                        # A branch spanning trunk-node-21 → endpoint-510 is a trunk
                        # return artifact and must be removed here.
                        TRUNK_SNAP_MM = 20.0
                        _trunkAdjacentSet = set(_trunkNodeSet)
                        for _ep in realEndpoints:
                            if _ep in _trunkAdjacentSet:
                                continue
                            _ep_pos = nodePos[_ep]
                            _nearest_trunk_d = min(
                                _dist3(_ep_pos, nodePos[_tn]) for _tn in _trunkNodeSet
                            )
                            if _nearest_trunk_d < TRUNK_SNAP_MM:
                                _trunkAdjacentSet.add(_ep)

                        nonTrunkBranches = []
                        for pts in graphBranches:
                            # Resolve both endpoints to their nearest node in all of
                            # realCritical (not just trunkNodeSet) so that endpoint
                            # leaves adjacent to the trunk root are correctly identified.
                            _ek_sub = _ptsToEdgeKey(pts)
                            if _ek_sub is not None:
                                _sub_nodes = list(_ek_sub)
                                _isTrunkSub = (
                                    len(_sub_nodes) == 2
                                    and _sub_nodes[0] in _trunkAdjacentSet
                                    and _sub_nodes[1] in _trunkAdjacentSet
                                    and _sub_nodes[0] != _sub_nodes[1]
                                )
                                if _isTrunkSub:
                                    pass  # diagnostic only when needed
                            else:
                                _isTrunkSub = False
                            if _isTrunkSub:
                                pass  # skip — sub-segment of (or return-path along) trunk
                            else:
                                nonTrunkBranches.append(pts)
                        graphBranches = [trunkPtsOrdered] + nonTrunkBranches

                        # ── Trunk-return artifact suppression ──────────────────────
                        # The _isTrunkSub check above requires BOTH endpoints to be in
                        # _trunkNodeSet (compressed critical nodes, no leaf endpoints).
                        # A limb that runs from a mid-trunk bifurcation node back to
                        # the proximal aortic root endpoint node survives because the
                        # leaf endpoint node is absent from _trunkNodeSet.
                        #
                        # Detection: use edgeUsage (already computed) as the definitive
                        # signal.  A trunk-return artifact has:
                        #   - usage < max_usage (it is NOT a trunk edge by definition)
                        #   - exactly ONE endpoint node in _trunkNodeSet (it departs
                        #     from the trunk topology but the other end is a leaf)
                        #
                        # _ptsToEdgeKey resolves each end to its nearest node in
                        # realCritical, so the frozenset edge key is reliable for
                        # both trunk-node and leaf-endpoint ends.
                        # Trunk geometric Z range — reliable regardless of nodePos values.
                        # trunkPtsOrdered is oriented proximal→distal at this point
                        # (trunk flip happens in flow norm below, but the key anchor
                        # is the primary bif node which we can look up directly).
                        # We use the bif node geometric position for the distal Z gate.
                        _trunkProxZ = max(trunkPtsOrdered[0][2], trunkPtsOrdered[-1][2])
                        _trunkDistZ = min(trunkPtsOrdered[0][2], trunkPtsOrdered[-1][2])

                        # Two-pass trunk-return suppression:
                        # Pass 1: evaluate every limb independently and record
                        #         a provisional keep/drop decision plus its Z band.
                        # Pass 2: before finalizing any DROP, check whether ANY
                        #         other limb (provisionally kept OR dropped) shares
                        #         the same Z band — that is the sibling-witness.
                        #         A true trunk-return artifact has no peer at its Z;
                        #         a genuine branch at a multi-branch confluence does.
                        # This two-pass design avoids ordering sensitivity: the
                        # witness may appear before or after the candidate in the
                        # iteration order.
                        _trunkReturnKept = []
                        _trunkReturnDropped = []
                        # Pass-1 results: list of (pts, provisional_drop:bool, minZ, maxZ, log_str)
                        _tr_pass1 = []
                        for _tri, _trPts in enumerate(graphBranches[1:], start=1):
                            if not _trPts or len(_trPts) < 2:
                                _tr_pass1.append((_tri, _trPts, False, None, None))
                                continue
                            _ek = _ptsToEdgeKey(_trPts)
                            if _ek is None:
                                _tr_pass1.append((_tri, _trPts, False, None, None))
                                continue
                            _usage = edgeUsage.get(_ek, 0)
                            # Must be below max_usage (trunk edges have max_usage)
                            if _usage >= max_usage:
                                _tr_pass1.append((_tri, _trPts, False, None, None))
                                continue
                            # Exactly one endpoint must be a trunk critical node
                            _nodes = list(_ek)
                            _inTrunk = [n for n in _nodes if n in _trunkNodeSet]
                            _notInTrunk = [n for n in _nodes if n not in _trunkNodeSet]
                            # Geometric tip: the branch end with the HIGHER Z
                            # (more proximal in axial CT where Z decreases caudally).
                            # For a trunk-return artifact, the tip runs back toward the
                            # aortic root — it will be the high-Z end of the branch.
                            _branchMaxZ = max(_trPts[0][2], _trPts[-1][2])
                            _branchMinZ = min(_trPts[0][2], _trPts[-1][2])
                            print(
                                f"[TrunkReturn] Limb {_tri}: arc={_seg_length(_trPts):.1f}mm "
                                f"usage={_usage} edge={set(_ek)} "
                                f"inTrunk={_inTrunk} "
                                f"branchZ=[{_branchMinZ:.1f},{_branchMaxZ:.1f}]"
                            )
                            if len(_inTrunk) != 1:
                                _tr_pass1.append((_tri, _trPts, False, None, None))
                                continue
                            # Trunk-return criterion (no nodePos dependency):
                            # A branch is a return artifact if its tip reaches back into
                            # the PROXIMAL trunk zone — i.e. its high-Z end exceeds the
                            # trunk's distal (bif) Z by a meaningful margin.
                            # Real side branches depart distally and run laterally or
                            # further distally; they do not climb back above the bif level.
                            #
                            # Lateral escape hatch: if the branch departs at a large
                            # angle to the trunk axis (>= LATERAL_ANGLE_DEG) it is a
                            # genuine side branch (renal / collateral), not a trunk-return
                            # artifact, even if its Z range overlaps the trunk.
                            _PROXIMAL_MARGIN_MM = 10.0
                            _LATERAL_ANGLE_DEG = (
                                45.0  # branches > 45° to trunk are side
                            )
                            _PROX_SAMPLE_MM_TR = (
                                15.0  # mm for departure direction sample
                            )
                            # 15mm (was 8mm): first ~8mm at a trunk node still follows
                            # trunk curvature; 15mm reliably escapes it for renal vessels.
                            #
                            # Trunk axis proxy: computed directly from trunkPtsOrdered
                            # (root→bif direction) because _trunkDir is not yet assigned
                            # at this point in the pipeline — it is set after flow-norm,
                            # which runs after trunk-return.  Using '_trunkDir' in dir()
                            # always fell back to [0,0,-1] (pure Z), making the lateral
                            # escape hatch unable to distinguish lateral from axial branches.
                            _tr_p0 = trunkPtsOrdered[0]
                            _tr_p1 = trunkPtsOrdered[-1]
                            _tr_td = [_tr_p1[k] - _tr_p0[k] for k in range(3)]
                            _tr_tl = math.sqrt(sum(x * x for x in _tr_td))
                            _trunk_axis_tr = (
                                [x / _tr_tl for x in _tr_td]
                                if _tr_tl > 1e-6
                                else [0.0, 0.0, -1.0]
                            )
                            _is_return_candidate = (
                                _branchMaxZ > _trunkDistZ + _PROXIMAL_MARGIN_MM
                            )
                            _angle_deg = -1.0  # sentinel: not yet measured
                            if _is_return_candidate:
                                # Measure branch departure angle using proximal direction
                                # (first _PROX_SAMPLE_MM_TR from the trunk-node end).
                                # End-to-end vectors are unreliable for short branches —
                                # the graph walk direction contaminates the result.
                                #
                                # The trunk-node end is the end of _trPts closest to
                                # trunkPtsOrdered (reliable geometry, no nodePos dependency).
                                # nodePos after snap-and-collapse is unreliable — it does not
                                # reflect the true geometric position of the trunk node, causing
                                # the wrong branch end to be selected as the root, which then
                                # measures the departure direction *away* from the trunk instead
                                # of *from* it, producing a near-axial dot that fails the
                                # lateral escape hatch even for genuine renal veins.
                                # Fix: find the branch end geometrically closest to any point
                                # in trunkPtsOrdered — this is always the trunk-junction end.
                                _td0 = min(
                                    _dist3(_trPts[0], _tp) for _tp in trunkPtsOrdered
                                )
                                _td1 = min(
                                    _dist3(_trPts[-1], _tp) for _tp in trunkPtsOrdered
                                )
                                _tr_root = _trPts[0] if _td0 <= _td1 else _trPts[-1]
                                _tr_fwd = 1 if _td0 <= _td1 else -1
                                _tr_samp = _tr_root
                                _tr_acc = 0.0
                                _tr_idxs = (
                                    range(len(_trPts) - 1)
                                    if _tr_fwd == 1
                                    else range(len(_trPts) - 1, 0, -1)
                                )
                                for _tpi in _tr_idxs:
                                    _tnxt = _tpi + 1 if _tr_fwd == 1 else _tpi - 1
                                    _tr_acc += _dist3(_trPts[_tpi], _trPts[_tnxt])
                                    if _tr_acc >= _PROX_SAMPLE_MM_TR:
                                        _tr_samp = _trPts[_tnxt]
                                        break
                                    _tr_samp = _trPts[_tnxt]
                                _bdir = [_tr_samp[k] - _tr_root[k] for k in range(3)]
                                _bdl = math.sqrt(sum(x * x for x in _bdir))
                                if _bdl > 1e-6:
                                    _bdir = [x / _bdl for x in _bdir]
                                    _dot = sum(
                                        _bdir[k] * _trunk_axis_tr[k] for k in range(3)
                                    )
                                    _angle_deg = math.degrees(
                                        math.acos(max(-1.0, min(1.0, abs(_dot))))
                                    )
                                else:
                                    _angle_deg = 0.0
                                if _angle_deg >= _LATERAL_ANGLE_DEG:
                                    print(
                                        f"[TrunkReturn] KEEP Limb {_tri}: "
                                        f"branchMaxZ={_branchMaxZ:.1f} > trunkDistZ+margin "
                                        f"BUT angle={_angle_deg:.1f}°>={_LATERAL_ANGLE_DEG}° "
                                        f"→ lateral side branch, kept"
                                    )
                                    _tr_pass1.append(
                                        (_tri, _trPts, False, _branchMinZ, _branchMaxZ)
                                    )
                                    continue
                                # ── Tip lateral displacement check ─────────────────────
                                # Proximal angle fails for renal veins that run parallel
                                # to the IVC for the first 15–20mm before diverging.
                                # Ground truth: project the branch TIP onto the trunk
                                # axis and measure perpendicular (radial) offset.
                                # A real IVC duplicate stays close to the trunk axis at
                                # its tip; a renal vein tip is 30–80mm off-axis.
                                # Trunk axis: unit vector _trunk_axis_tr, anchor = _tr_p0.
                                # Tip = the branch end FURTHEST from trunkPtsOrdered
                                # (opposite of _tr_root, the trunk-junction end).
                                _TIP_LATERAL_MM = (
                                    25.0  # mm — renal tips are >>25mm off-axis
                                )
                                _tr_tip = _trPts[-1] if _td0 <= _td1 else _trPts[0]
                                # Vector from trunk anchor to tip
                                _v_tip = [_tr_tip[k] - _tr_p0[k] for k in range(3)]
                                # Axial projection (scalar)
                                _axial = sum(
                                    _v_tip[k] * _trunk_axis_tr[k] for k in range(3)
                                )
                                # Lateral component = tip_vec − axial_projection * axis
                                _lat_vec = [
                                    _v_tip[k] - _axial * _trunk_axis_tr[k]
                                    for k in range(3)
                                ]
                                _lat_mm = math.sqrt(sum(x * x for x in _lat_vec))
                                if _lat_mm >= _TIP_LATERAL_MM:
                                    print(
                                        f"[TrunkReturn] KEEP Limb {_tri}: "
                                        f"angle={_angle_deg:.1f}°<{_LATERAL_ANGLE_DEG}° (proximal axial) "
                                        f"BUT tip_lateral={_lat_mm:.1f}mm>={_TIP_LATERAL_MM}mm "
                                        f"→ curved/renal side branch, kept"
                                    )
                                    _tr_pass1.append(
                                        (_tri, _trPts, False, _branchMinZ, _branchMaxZ)
                                    )
                                    continue
                                else:
                                    print(
                                        f"[TrunkReturn] axial confirm: "
                                        f"angle={_angle_deg:.1f}° tip_lateral={_lat_mm:.1f}mm<{_TIP_LATERAL_MM}mm "
                                        f"→ confirmed IVC duplicate"
                                    )
                            # Record provisional decision for pass 2
                            _tr_pass1.append(
                                (
                                    _tri,
                                    _trPts,
                                    _is_return_candidate,
                                    _branchMinZ,
                                    _branchMaxZ,
                                )
                            )

                        # ── Pass 2: sibling-witness rescue ─────────────────────────
                        # A true trunk-return artifact has no peer branch at the same Z.
                        # If any other limb (kept OR provisionally dropped) overlaps the
                        # candidate's Z band by >= _Z_OVERLAP_THRESH_MM, the candidate
                        # is at a genuine multi-branch confluence and must be kept.
                        _Z_OVERLAP_THRESH_MM = 5.0
                        for _tri, _trPts, _prov_drop, _pMinZ, _pMaxZ in _tr_pass1:
                            if not _prov_drop:
                                # Always-keep branch (passed an early guard or lateral/tip escape)
                                _trunkReturnKept.append(_trPts)
                                continue
                            # Look for a sibling witness among ALL other pass-1 entries
                            _sibling_witness = None
                            for _oi, _oPts, _oDrop, _oMinZ, _oMaxZ in _tr_pass1:
                                if _oPts is _trPts:
                                    continue  # skip self
                                if _oMinZ is None or _oMaxZ is None:
                                    continue  # early-guard entry — no Z data
                                _overlap = min(_pMaxZ, _oMaxZ) - max(_pMinZ, _oMinZ)
                                if _overlap >= _Z_OVERLAP_THRESH_MM:
                                    _sibling_witness = (_oi, _oMinZ, _oMaxZ)
                                    break
                            if _sibling_witness is not None:
                                _sw_i, _sw_minZ, _sw_maxZ = _sibling_witness
                                print(
                                    f"[TrunkReturn] KEEP Limb {_tri}: "
                                    f"sibling-witness Limb {_sw_i} "
                                    f"(Z=[{_sw_minZ:.1f},{_sw_maxZ:.1f}]) overlaps "
                                    f"candidate Z=[{_pMinZ:.1f},{_pMaxZ:.1f}] "
                                    f"by ≥{_Z_OVERLAP_THRESH_MM}mm "
                                    f"→ genuine multi-branch confluence, kept"
                                )
                                _trunkReturnKept.append(_trPts)
                            else:
                                # Emit the DROP log here (was inside pass 1 before)
                                _trunkReturnDropped.append(_tri)
                                print(
                                    f"[TrunkReturn] DROP Limb {_tri}: "
                                    f"branchMaxZ={_pMaxZ:.1f} > trunkDistZ={_trunkDistZ:.1f}+{_PROXIMAL_MARGIN_MM}"
                                    f" no sibling witness → trunk-return artifact, dropped"
                                )
                        # Always sort kept limbs by descending arc length so that long
                        # anatomical branches (iliacs, renals) get lower branch indices
                        # than short rescued siblings, regardless of whether any limb
                        # was ultimately dropped.
                        _trunkReturnKept.sort(
                            key=lambda _p: _seg_length(_p), reverse=True
                        )
                        graphBranches = [trunkPtsOrdered] + _trunkReturnKept
                        # ───────────────────────────────────────────────────────────

                        print(
                            f"[VesselAnalyzer] Trunk stitched: "
                            f"{_seg_length(trunkPtsOrdered):.1f}mm, "
                            f"{len(trunkPtsOrdered)} pts "
                            f"({len(graphBranches)-1} limbs remaining)"
                        )

                    print(f"[VesselAnalyzer] Branch order after trunk detection:")
                    for i, pts in enumerate(graphBranches):
                        ek = _ptsToEdgeKey(pts)
                        sc = edgeScore.get(ek, 0) if ek else 0
                        lbl = "TRUNK" if i == 0 else "limb "
                        print(
                            f"  [{i}] {lbl}  {_seg_length(pts):.1f}mm  score={sc:.0f}"
                        )

                    # ── step 5c: flow normalization ───────────────────────────────
                    #
                    # After trunk detection, graphBranches[0] is the trunk with its
                    # points stored in whatever order the graph walk produced.  All
                    # downstream direction logic (branch scoring, _outwardDir, angle_deg,
                    # ostium detection) assumes:
                    #   graphBranches[0] is oriented proximal → distal
                    #   graphBranches[bi] for bi≥1 is oriented bif-end → tip
                    #
                    # Neither guarantee is met without explicit normalization.
                    #
                    # Algorithm:
                    #   1. Trunk: the proximal end has the LARGER Y coordinate for
                    #      axial CT (IVC runs inferior→superior, so larger Y = proximal).
                    #      Generic fallback: orient so the trunk's first point has the
                    #      larger mean-Z distance from all endpoints (i.e. is furthest
                    #      from the distal tips).  We use a simple heuristic: the trunk
                    #      end closest to the centroid of ALL endpoint positions is proximal
                    #      (the common iliac / IVC root) — flip if needed.
                    #      If no endpoints, orient by length: longer sub-segment → proximal.
                    #
                    #   2. Limbs: orient so their first point is closest to the
                    #      bifurcation end of the trunk (graphBranches[0][-1]).
                    #
                    # Both flips operate on the raw point lists (in-place reversal).

                    if graphBranches:
                        # ── 1. Orient trunk — anchor on primary bifurcation node ──
                        #
                        # The trunk path spans from the vessel root to the primary
                        # bifurcation node.  We identify the primary bif node as the
                        # node in trunkNodePath with the highest degree in cAdj — it
                        # is the most branched, therefore the true split point.
                        # trunk[0] = proximal root; trunk[-1] = primary bif end.
                        #
                        # This replaces the old tip-centroid heuristic which failed
                        # when the primary bif node is closer to the tip centroid than
                        # the root (common in short aortic trunks with long iliacs).

                        trk = graphBranches[0]
                        # PRIMARY BIFURCATION — locked here, used everywhere downstream.
                        # Rank ALL known bifurcation nodes by sum of departing edge scores
                        # (usage × arc × diam).  The node with the highest outflow score
                        # is the anatomical primary bifurcation (aorto-iliac split).
                        # This single selection drives flow norm, _trunkDir, main-branch
                        # selection, and ostium detection — no secondary overrides.
                        _lockedBifNode = None
                        _lockedBifScore = -1.0
                        for _tn in realBifurcations:
                            _sc = sum(
                                edgeScore.get(frozenset({_tn, _nb}), 0)
                                for _nb in cAdj.get(_tn, set())
                            )
                            if _sc > _lockedBifScore:
                                _lockedBifScore = _sc
                                _lockedBifNode = _tn
                        if _lockedBifNode is None:
                            # Fallback: trunk-path node with most non-trunk departures
                            _trunkPathSet = set(trunkNodePath)
                            for _tn in trunkNodePath:
                                if (
                                    len(
                                        [
                                            n
                                            for n in cAdj.get(_tn, set())
                                            if n not in _trunkPathSet
                                        ]
                                    )
                                    > 0
                                ):
                                    _lockedBifNode = _tn
                                    break
                        _bifInTrunk = _lockedBifNode  # alias used by bif_end below
                        if _bifInTrunk is not None:
                            print(
                                f"[VesselAnalyzer] Flow norm: locked primary bif = {_bifInTrunk} "
                                f"(score={_lockedBifScore:.0f}), "
                                f"pos=Z={nodePos[_bifInTrunk][2]:.1f}"
                            )
                        else:
                            print(
                                "[VesselAnalyzer] Flow norm: no primary bif node found"
                            )

                        # ── 2. Orient limbs correctly ────────────────────────────
                        # Main branches (iliacs): orient bif-end-first so [0] is at
                        # the primary bifurcation and [-1] is the distal tip.
                        #
                        # Side branches (renals, collaterals): orient ostium-first so
                        # [0] is at the trunk wall junction and [-1] is the distal tip.
                        # Using bif_end proximity for side branches is wrong — their
                        # ostium is on the trunk, not at the iliac bifurcation, so
                        # both ends may be far from bif_end and the flip is arbitrary.
                        #
                        # Classification: a limb is a "side branch" if its closest
                        # endpoint to trunkPtsOrdered is < SIDE_TRUNK_SNAP_MM and
                        # neither end is within MAIN_BIF_SNAP_MM of bif_end.
                        bif_end = (
                            nodePos[_bifInTrunk]
                            if _bifInTrunk is not None
                            and 0 <= _bifInTrunk < len(nodePos)
                            else (trk[-1] if trk else (0.0, 0.0, 0.0))
                        )
                        _trunkBifPos = bif_end
                        _trunkRootPos = None
                        MAIN_BIF_SNAP_MM = 30.0  # end within this → main branch end
                        SIDE_TRUNK_SNAP_MM = (
                            30.0  # end within this of trunk → side ostium
                        )
                        flipped = 0
                        for _bi in range(1, len(graphBranches)):
                            _bp = graphBranches[_bi]
                            if not _bp:
                                continue
                            _d_start_bif = _dist3(_bp[0], bif_end)
                            _d_end_bif = _dist3(_bp[-1], bif_end)
                            _is_main_candidate = (
                                _d_start_bif < MAIN_BIF_SNAP_MM
                                or _d_end_bif < MAIN_BIF_SNAP_MM
                            )
                            if _is_main_candidate:
                                # Main branch — orient bif-end first (standard)
                                if _d_end_bif < _d_start_bif:
                                    graphBranches[_bi] = list(reversed(_bp))
                                    flipped += 1
                            else:
                                # Side branch — orient ostium-first using trunkPtsOrdered
                                # proximity.  The end closest to the trunk is the ostium.
                                _d_start_trunk = min(
                                    _dist3(_bp[0], _tp) for _tp in trunkPtsOrdered
                                )
                                _d_end_trunk = min(
                                    _dist3(_bp[-1], _tp) for _tp in trunkPtsOrdered
                                )
                                if _d_end_trunk < _d_start_trunk:
                                    graphBranches[_bi] = list(reversed(_bp))
                                    flipped += 1
                        if flipped:
                            print(
                                f"[VesselAnalyzer] Flow norm: {flipped} limb(s) "
                                f"flipped to bif-end-first / ostium-first orientation"
                            )
                        else:
                            print(
                                f"[VesselAnalyzer] Flow norm: all limbs already "
                                f"correctly oriented — no flips needed"
                            )

                        # Pin each limb's bif-end ([0]) to the exact bif node position
                        # so branches connect seamlessly to the trunk tip.
                        # Only snap if the limb start is within BIF_SNAP_MM of bif_end
                        # (avoids snapping side branches that genuinely start elsewhere).
                        BIF_SNAP_MM = 20.0
                        _pinned = 0
                        for _bi in range(1, len(graphBranches)):
                            _bp = graphBranches[_bi]
                            if not _bp:
                                continue
                            _d = _dist3(_bp[0], bif_end)
                            if 0 < _d <= BIF_SNAP_MM:
                                # Prepend the exact bif node position
                                graphBranches[_bi] = [bif_end] + list(_bp)
                                _pinned += 1
                        if _pinned:
                            print(
                                f"[VesselAnalyzer] Flow norm: {_pinned} limb(s) "
                                f"pinned to bif node position"
                            )

                        # Find trunk root = trunk endpoint furthest from bif node.
                        # Also enforce trunk list orientation: root at [0], bif at [-1].
                        if graphBranches:
                            _trk0 = graphBranches[0]
                            if _trk0:
                                _da = _dist3(_trk0[0], _trunkBifPos)
                                _db = _dist3(_trk0[-1], _trunkBifPos)
                                # _trunkBifPos = nodePos[_bifInTrunk] = caudal bif end
                                # _trunkRootPos = the trunk endpoint FURTHER from the bif node
                                # _da = dist(trk[0], bif),  _db = dist(trk[-1], bif)
                                # Further end (root) = trk[0] if _da > _db, else trk[-1]
                                _trunkRootPos = _trk0[0] if _da > _db else _trk0[-1]
                                print(
                                    f"[VesselAnalyzer] Flow norm: "
                                    f"root=Z={_trunkRootPos[2]:.1f} "
                                    f"bif=Z={_trunkBifPos[2]:.1f}"
                                )

                    segLocators = []
                    for si, seg in enumerate(allSegments):
                        spd = vtk.vtkPolyData()
                        spts = vtk.vtkPoints()
                        for p in seg:
                            spts.InsertNextPoint(p[0], p[1], p[2])
                        spd.SetPoints(spts)
                        sloc = vtk.vtkPointLocator()
                        sloc.SetDataSet(spd)
                        sloc.BuildLocator()
                        segLocators.append((sloc, spts, allSegmentRadii[si]))

                    def _lookupRadius(p):
                        best_d2 = 1e18
                        best_r = None
                        for sloc, spts, sradii in segLocators:
                            cid = sloc.FindClosestPoint([p[0], p[1], p[2]])
                            cp = spts.GetPoint(cid)
                            d2 = (
                                (p[0] - cp[0]) ** 2
                                + (p[1] - cp[1]) ** 2
                                + (p[2] - cp[2]) ** 2
                            )
                            if d2 < best_d2:
                                best_d2 = d2
                                best_r = sradii[cid]
                        return best_r

                    # ── step 7: ep assignment ─────────────────────────────────────
                    def _epDist(a, b):
                        return math.sqrt(
                            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
                        )

                    epNode = slicer.mrmlScene.GetFirstNodeByName("Endpoints")
                    epPositions = []
                    if epNode:
                        for _ei in range(epNode.GetNumberOfControlPoints()):
                            _p = [0, 0, 0]
                            epNode.GetNthControlPointPositionWorld(_ei, _p)
                            epPositions.append(tuple(_p))

                    _used_ep_indices = set()
                    _ep_overflow_counter = [0]  # mutable counter for overflow branches

                    def _assignEp(tip):
                        if not epPositions:
                            # No endpoint node — sequential unique indices.
                            fb = len(_used_ep_indices)
                            _used_ep_indices.add(fb)
                            return fb
                        ranked = sorted(
                            range(len(epPositions)),
                            key=lambda ei: _epDist(tip, epPositions[ei]),
                        )
                        for ei in ranked:
                            if ei not in _used_ep_indices:
                                _used_ep_indices.add(ei)
                                return ei
                        # All real endpoints consumed — assign a unique synthetic index
                        # beyond the real ep range so the display name doesn't collide.
                        synthetic = len(epPositions) + _ep_overflow_counter[0]
                        _ep_overflow_counter[0] += 1
                        _used_ep_indices.add(synthetic)
                        return synthetic

                    # ── step 8: commit ────────────────────────────────────────────
                    self.rawRadii = []
                    self.branchMeta = {}
                    self.branchEndpointMap = {}
                    trunkPts = graphBranches[0] if graphBranches else []
                    for branchN, pts_list in enumerate(graphBranches):
                        radii_list = [_lookupRadius(p) for p in pts_list]
                        if len(pts_list) < 2:
                            continue
                        # For the trunk (branchN==0), enforce root-first orientation.
                        # List-mutation in flow-norm scope doesn't reliably persist here,
                        # so we re-check at commit time using the stored _trunkBifPos.
                        if (
                            branchN == 0
                            and _trunkBifPos is not None
                            and len(pts_list) >= 2
                        ):
                            _c0 = _dist3(pts_list[0], _trunkBifPos)
                            _cN = _dist3(pts_list[-1], _trunkBifPos)
                            if _c0 < _cN:
                                # pts_list[0] is closer to bif — reverse to root-first
                                pts_list = list(reversed(pts_list))
                                radii_list = list(reversed(radii_list))
                        startIdx = len(rawPoints)
                        rawPoints.extend(pts_list)
                        self.rawRadii.extend(radii_list)
                        self.branches.append((startIdx, len(rawPoints)))
                        # For the trunk (branchN==0), the canonical tip for endpoint
                        # matching is the PROXIMAL ROOT end (pts_list[0] = root after
                        # orientation), not the bif end (pts_list[-1]).  The IVC inlet
                        # fiducial (ep5) is placed at the cranial root, so using [-1]
                        # (the bifurcation) would match it to an iliac endpoint instead.
                        _ep_probe = pts_list[0] if branchN == 0 else pts_list[-1]
                        self.branchEndpointMap[branchN] = _assignEp(_ep_probe)

                        origin_pt = pts_list[0]
                        tip_pt = pts_list[-1]
                        branch_length = _seg_length(pts_list)

                        if branchN == 0 or not trunkPts:
                            angle_deg = 0.0
                        else:
                            # Sample branch direction at +15 mm from the origin,
                            # not just the first 5 pts.  Branches that share a
                            # parallel zone immediately at the carina (dot≈0.99)
                            # will show near-zero angle with the naïve approach;
                            # sampling further downstream reveals the true
                            # anatomical divergence angle.
                            ANGLE_SAMPLE_MM = 15.0
                            _acc = 0.0
                            _p_sample = pts_list[-1]  # fallback: tip
                            for _ai in range(len(pts_list) - 1):
                                _acc += _dist3(pts_list[_ai], pts_list[_ai + 1])
                                if _acc >= ANGLE_SAMPLE_MM:
                                    _p_sample = pts_list[_ai + 1]
                                    break
                            bdir = [_p_sample[k] - pts_list[0][k] for k in range(3)]
                            blen = math.sqrt(sum(x * x for x in bdir))
                            bdir = [x / blen for x in bdir] if blen > 0 else [0, 0, 1]
                            lo = max(0, len(trunkPts) - 4)
                            hi = len(trunkPts) - 1
                            tdir = [trunkPts[hi][k] - trunkPts[lo][k] for k in range(3)]
                            tlen = math.sqrt(sum(x * x for x in tdir))
                            tdir = [x / tlen for x in tdir] if tlen > 0 else [0, 0, 1]
                            dot = max(
                                -1.0, min(1.0, sum(bdir[k] * tdir[k] for k in range(3)))
                            )
                            angle_deg = math.degrees(math.acos(abs(dot)))

                        self.branchMeta[branchN] = {
                            "id": __import__("uuid").uuid4().hex,
                            "origin": origin_pt,
                            "tip": tip_pt,
                            "originZ": origin_pt[2],
                            "angle_deg": round(angle_deg, 1),
                            "length_mm": round(branch_length, 1),
                            "tipZ": tip_pt[2],
                            "bifIdx": 0,
                            "trimIdx": 0,
                        }

                    print(
                        f"[VesselAnalyzer] Committed {len(self.branches)} branches "
                        f"({len(realEndpoints)} endpoints, "
                        f"{len(realBifurcations)} bifurcations)"
                    )

                    # ── step 8 post: resolve deferred bifurcation scores ──────────
                    # score_bifurcation() needs radius data from _lookupRadius, which
                    # is only available after segLocators is built (step 8).  We
                    # stored the pending (nodeId, pts_list) tuples during step 3 and
                    # resolve them now.
                    self._bifScoreMap = {}
                    for _bsn, _bsPts in _pendingBifScores:
                        _micro_r_hub = _lookupRadius(_bsPts[0]) or 1.0
                        _micro_radii = [_micro_r_hub]
                        for _bsPt in _bsPts[1:]:
                            _r = _lookupRadius(_bsPt) or _micro_r_hub
                            _micro_radii.append(_r if _r else _micro_r_hub)
                        _micro_g = build_graph(_bsPts, _micro_radii)
                        for _ei in range(1, len(_bsPts)):
                            _micro_g.add_edge(0, _ei)
                        _bif_score = score_bifurcation(_micro_g, 0)
                        self._bifScoreMap[_bsn] = _bif_score
                        print(
                            f"[VesselAnalyzer] bifScore node {_bsn}: {_bif_score:.3f}"
                        )

                    # Attach bifScore to branchMeta for every committed branch.
                    # We map each branch's origin point back to the nearest scored
                    # bifurcation node so downstream logic can gate on score quality.
                    for _bi, (_bs, _be) in enumerate(self.branches):
                        if _bi == 0:
                            continue  # trunk has no bifurcation origin
                        _bOrigin = self.branchMeta.get(_bi, {}).get("origin")
                        if not _bOrigin:
                            continue
                        _bestSN = None
                        _bestSD = 1e18
                        for _sn, _sc in self._bifScoreMap.items():
                            _sd = math.sqrt(
                                sum(
                                    (_bOrigin[k] - nodePos[_sn][k]) ** 2
                                    for k in range(3)
                                )
                            )
                            if _sd < _bestSD:
                                _bestSD = _sd
                                _bestSN = _sn
                        if _bestSN is not None and _bestSD < 20.0:
                            self.branchMeta[_bi]["bifScore"] = round(
                                self._bifScoreMap[_bestSN], 3
                            )

                    # ── step 8b-pre: build parallel branch tree ───────────────────
                    #
                    # self.branchTree is a parallel data structure to self.branches.
                    # It preserves the full topology of the compressed graph as a
                    # rooted tree, with one node per critical graph node and one
                    # segment per graph edge.  self.branches / self.branchMeta are
                    # NOT modified — all existing downstream logic is unaffected.
                    #
                    # Tree node schema:
                    #   {
                    #     'node_id'  : int           graph node id
                    #     'pos'      : (x,y,z)       spatial position (from nodePos)
                    #     'type'     : 'root'|'bifurcation'|'endpoint'
                    #     'depth'    : int           0 = root
                    #     'segment'  : {             edge from parent to this node
                    #       'pts'    : [(x,y,z),…]  ordered centerline points
                    #       'arc_mm' : float         arc length
                    #       'diam_mm': float         mean diameter (placeholder, filled later)
                    #     }                          None for root
                    #     'children' : [tree_node, …]
                    #   }
                    #
                    def _seg_arc(pts):
                        arc = 0.0
                        for _k in range(len(pts) - 1):
                            arc += math.sqrt(
                                sum(
                                    (pts[_k + 1][d] - pts[_k][d]) ** 2 for d in range(3)
                                )
                            )
                        return round(arc, 1)

                    # Trunk-path node set — all nodes on trunkNodePath are
                    # anatomically intermediate even if cAdj degree looks like endpoint.
                    _trunkPathSet_tree = set(trunkNodePath)

                    # Last node of trunk path = true endpoint (no trunk continuation).
                    # All other trunk-path nodes are intermediate bifurcations even if
                    # they appear as degree-2 after snap-collapse (e.g. node 52, 1.9mm edge).
                    _trunkPathIntermediate = set(trunkNodePath[:-1])  # all except last

                    def _build_tree_node(nid, parent_nid, depth, _visited=None):
                        if _visited is None:
                            _visited = set()
                        if nid in _visited:
                            # Cycle detected — skip to prevent infinite recursion
                            return None
                        _visited.add(nid)

                        pos = nodePos[nid] if nid < len(nodePos) else (0, 0, 0)
                        ntype = (
                            "root"
                            if parent_nid is None
                            else (
                                "bifurcation"
                                if (
                                    nid in realBifurcations
                                    or nid in _trunkPathIntermediate
                                )
                                else "endpoint"
                            )
                        )
                        # Segment from parent to this node
                        seg = None
                        if parent_nid is not None:
                            ekey = (min(parent_nid, nid), max(parent_nid, nid))
                            pts = list(cEdge.get(ekey, [nodePos[parent_nid], pos]))
                            # Orient pts: parent first
                            if pts and math.sqrt(
                                sum(
                                    (pts[0][d] - nodePos[parent_nid][d]) ** 2
                                    for d in range(3)
                                )
                            ) > math.sqrt(
                                sum(
                                    (pts[-1][d] - nodePos[parent_nid][d]) ** 2
                                    for d in range(3)
                                )
                            ):
                                pts = list(reversed(pts))
                            seg = {"pts": pts, "arc_mm": _seg_arc(pts), "diam_mm": 0.0}

                        node = {
                            "node_id": nid,
                            "pos": pos,
                            "type": ntype,
                            "depth": depth,
                            "segment": seg,
                            "children": [],
                        }
                        # Recurse into children (all neighbours except parent)
                        for nb in sorted(cAdj.get(nid, [])):
                            if nb == parent_nid:
                                continue
                            child = _build_tree_node(nb, nid, depth + 1, _visited)
                            if child is not None:
                                node["children"].append(child)
                        return node

                    # Root = first node of trunkNodePath (inflow / aortic root)
                    _tree_root_nid = trunkNodePath[0] if trunkNodePath else 0
                    self.branchTree = _build_tree_node(_tree_root_nid, None, 0)

                    # Annotate each segment with its trunk/branch classification
                    # by matching to trunkNodePath edges.
                    _trunk_edges = set()
                    for _tk in range(len(trunkNodePath) - 1):
                        _trunk_edges.add(
                            (
                                min(trunkNodePath[_tk], trunkNodePath[_tk + 1]),
                                max(trunkNodePath[_tk], trunkNodePath[_tk + 1]),
                            )
                        )

                    def _annotate_tree(node, parent_nid):
                        """Pass 1: trunk/branch role from trunkNodePath edges only.
                        No bifScores dependency — safe to call before step 8b."""
                        if node["segment"] is not None and parent_nid is not None:
                            ekey = (
                                min(parent_nid, node["node_id"]),
                                max(parent_nid, node["node_id"]),
                            )
                            node["segment"]["role"] = (
                                "trunk" if ekey in _trunk_edges else "branch"
                            )
                        for ch in node["children"]:
                            _annotate_tree(ch, node["node_id"])

                    _annotate_tree(self.branchTree, None)

                    # Log summary
                    MICRO_SEG_MM = (
                        10.0  # segments shorter than this are topology artifacts
                    )

                    from centerline_graph import tree_count_nodes as _count_tree
                    from vessel_geometry import (
                        renal_anatomy_gate,
                        renal_composite_score,
                        geo_clamp,
                    )

                    _ts, _tb, _te = _count_tree(self.branchTree)
                    _trunk_segs = [
                        n
                        for n in self.branchTree["children"]
                        if n["segment"] and n["segment"].get("role") == "trunk"
                    ]

                    print(
                        f"[BranchTree] Built: root=node {_tree_root_nid}, "
                        f"{_ts} segments, {_tb} bifurcations, {_te} endpoints"
                    )

                    from centerline_graph import tree_log_nodes as _log_tree

                    # (tree logged after pass-2 primary-bif annotation below)

                    # ── step 8b: semantic branch classification ───────────────────
                    #
                    # Goal: identify ONE true primary bifurcation and enforce
                    #   1 trunk  +  2 main daughter branches  +  N side branches.
                    #
                    # If the graph produced multiple bifurcation nodes (due to
                    # noise or genuine multi-level branching), we rank them by
                    # "bifurcation importance score" and keep only the highest.
                    # Importance = sum of (length × mean_diameter) for all limbs
                    # connected to that node — the node with the most hemodynamically
                    # significant outflow is the primary bifurcation.
                    #
                    # Classification:
                    #   trunk  — branch 0 (the stitched longest path, always)
                    #   main   — top-2 limbs at the primary bifurcation by score
                    #   side   — everything else
                    #
                    # Fallback when no diameter available: score = length alone.
                    # All branches remain committed — none are deleted.

                    # ── mean diameter helper ──────────────────────────────────────
                    def _meanDiam(branchN):
                        bs, be = self.branches[branchN]
                        # Prefer VMTK rawRadii; fall back to surface diameters
                        radii = self.rawRadii[bs:be]
                        valid = [r * 2.0 for r in radii if r and r > 0]
                        if valid:
                            return sum(valid) / len(valid)
                        # Surface diameters computed post-commit — use if available
                        if (
                            getattr(self, "diameters", None)
                            and len(self.diameters) > be
                        ):
                            dv = [d for d in self.diameters[bs:be] if d > 0]
                            return sum(dv) / len(dv) if dv else 0.0
                        return 0.0

                    # ── trunk direction at bifurcation ───────────────────────────
                    # Measured 10–40 mm PROXIMAL to the bif end of the trunk
                    # (i.e. well away from the blended junction zone).
                    # After flow normalization: trunk[0]=proximal, trunk[-1]=bif end.
                    # We walk backwards from trunk[-1] by TRUNK_DIR_SKIP_MM and
                    # sample the direction over the next TRUNK_DIR_WINDOW_MM segment.
                    TRUNK_DIR_SKIP_MM = 10.0  # skip this far back from bif end
                    TRUNK_DIR_WINDOW_MM = 30.0  # measure direction over this window

                    # Trunk direction = root → bif = proximal-to-distal = flow direction.
                    # Computed directly from the stored topology positions (_trunkRootPos,
                    # _trunkBifPos) rather than from graphBranches[0] point ordering —
                    # the list-flip may not persist across Python's list-mutation semantics.
                    # Trunk direction = root → bif = proximal-to-distal flow direction.
                    # _trunkRootPos = trunk endpoint furthest from _trunkBifPos (nodePos[bif]).
                    # _trunkBifPos  = nodePos of primary bifurcation node.
                    # Both established in the flow-norm block above; immune to list order.
                    _trunkDir = [0.0, 0.0, 1.0]  # safe fallback
                    if _trunkRootPos is not None and _trunkBifPos is not None:
                        _td = [_trunkBifPos[k] - _trunkRootPos[k] for k in range(3)]
                        _tl = math.sqrt(sum(x * x for x in _td))
                        if _tl > 1e-6:
                            _trunkDir = [x / _tl for x in _td]
                    elif graphBranches and len(graphBranches[0]) >= 2:
                        _p0 = graphBranches[0][0]
                        _p1 = graphBranches[0][-1]
                        _td = [_p1[k] - _p0[k] for k in range(3)]
                        _tl = math.sqrt(sum(x * x for x in _td))
                        if _tl > 1e-6:
                            _trunkDir = [x / _tl for x in _td]

                    # Store root/bif positions in branchMeta[0] so downstream code
                    # (stent planning, diameter traversal) can orient the trunk correctly
                    # without relying on the point list order.
                    if self.branchMeta and 0 in self.branchMeta:
                        self.branchMeta[0]["trunkRootPos"] = _trunkRootPos
                        self.branchMeta[0]["trunkBifPos"] = _trunkBifPos

                    print(
                        f"[VesselAnalyzer] Trunk direction at bif: "
                        f"({_trunkDir[0]:+.2f},{_trunkDir[1]:+.2f},{_trunkDir[2]:+.2f})"
                    )

                    def _branchScore(bi):
                        """Scoring for bifurcation node ranking only (not main selection).
                        length^1.2 * diameter^0.8 — simpler here because pairwise
                        signed-dot is used for the actual main branch selection.

                        Diameter is sampled at the branch midpoint using the surface
                        locator (_eSurfLoc) built during edge scoring — available
                        before commit, unlike rawRadii and self.diameters.
                        """
                        length = self.branchMeta[bi]["length_mm"]
                        if length <= 0:
                            return 0.0
                        # Try VMTK radii first, then surface locator, then length only
                        diam = _meanDiam(bi)
                        if (
                            diam <= 0
                            and _eSurfLoc is not None
                            and bi < len(graphBranches)
                        ):
                            # Sample diameter at graphBranches[bi] midpoint
                            bpts = graphBranches[bi]
                            if bpts:
                                mid = bpts[len(bpts) // 2]
                                cp = [0.0, 0.0, 0.0]
                                ci = vtk.reference(0)
                                si = vtk.reference(0)
                                d2 = vtk.reference(0.0)
                                _eSurfLoc.FindClosestPoint(list(mid), cp, ci, si, d2)
                                diam = max(math.sqrt(float(d2)) * 2.0, 0.5)
                        if diam > 0:
                            return (length**1.2) * (diam**0.8)
                        return length**1.2

                    # ── map each non-trunk limb to its nearest bifurcation node ──
                    #
                    # Uses the compressed graph (cAdj / cEdge) so candidate extraction
                    # is topology-exact: only branches that share an edge with the bif
                    # node in the compressed graph are considered.  This replaces the
                    # old proximity-distance heuristic which excluded true outflow
                    # branches whose origin point happened to sit > 25 mm from the node.
                    bifNodePos = {n: nodePos[n] for n in realBifurcations}

                    # For each graphBranches[bi], record which compressed-graph nodes
                    # are its two endpoints (start / end critical nodes).
                    def _branchEndNodes(bi):
                        """Return (startNode, endNode) — the two compressed-graph
                        critical nodes at the ends of graphBranches[bi].  Uses
                        nearest-node lookup on the branch's first and last point."""
                        if bi >= len(graphBranches) or not graphBranches[bi]:
                            return None, None
                        bPts = graphBranches[bi]
                        p0, p1 = bPts[0], bPts[-1]

                        def _nearestCrit(p):
                            best_n, best_d = None, 1e18
                            for n in list(cAdj.keys()):  # critical nodes only
                                d = _dist3(nodePos[n], p)
                                if d < best_d:
                                    best_d = d
                                    best_n = n
                            return best_n, best_d

                        sn, sd = _nearestCrit(p0)
                        en, ed = _nearestCrit(p1)
                        return sn, en

                    # Build bifLimbs via graph-edge membership:
                    # A branch belongs to bifurcation node N if N is one of its endpoints
                    # in the compressed graph.
                    bifLimbs = {n: [] for n in realBifurcations}
                    for bi in range(1, len(self.branches)):
                        sn, en = _branchEndNodes(bi)
                        for bn in realBifurcations:
                            if sn == bn or en == bn:
                                bifLimbs[bn].append(bi)
                                break

                    # ── score each bifurcation node ───────────────────────────────
                    # Score = sum of branch scores for all limbs departing from node.
                    # The highest-scoring node is the primary (true) bifurcation.
                    bifScores = {}
                    for bn, limbList in bifLimbs.items():
                        bifScores[bn] = sum(_branchScore(bi) for bi in limbList)

                    if bifScores:
                        # Use the locked primary bif from flow norm — do NOT re-select.
                        # _lockedBifNode was chosen by edge-score ranking before flow norm
                        # and is the single source of truth for the primary bifurcation.
                        trueBifNode = (
                            _lockedBifNode
                            if _lockedBifNode in bifScores
                            else max(bifScores, key=bifScores.get)
                        )
                        otherBifs = [n for n in realBifurcations if n != trueBifNode]
                        print(
                            f"[VesselAnalyzer] Bifurcation scores: "
                            f"{ {n: f'{s:.0f}' for n,s in sorted(bifScores.items(), key=lambda x:-x[1])} }"
                        )
                        print(
                            f"[VesselAnalyzer] Primary bifurcation: node {trueBifNode} "
                            f"(score={bifScores.get(trueBifNode,0):.0f}), "
                            f"demoting {len(otherBifs)} secondary nodes to side-branch level"
                        )
                    else:
                        trueBifNode = _lockedBifNode  # use locked even if no bifScores

                    # ── Pass 2: annotate primary bifurcation in branchTree ───────
                    # Now that bifScores and trueBifNode are resolved, mark the
                    # primary bifurcation node and upgrade its entry segment role.
                    if hasattr(self, "branchTree") and self.branchTree is not None:
                        _pb_node = trueBifNode

                        def _annotate_primary_bif(node):
                            if node["node_id"] == _pb_node:
                                node["is_primary_bif"] = True
                                if node["segment"] is not None:
                                    node["segment"]["role"] = "primary_bif_entry"
                            for ch in node["children"]:
                                _annotate_primary_bif(ch)

                        _annotate_primary_bif(self.branchTree)
                        # Final tree log — reflects both pass-1 and pass-2 annotations
                        _log_tree(self.branchTree)
                    # ─────────────────────────────────────────────────────────────

                    # ── select top-2 main branches at primary bifurcation ─────────
                    #
                    # Core insight: at a true bifurcation the two main branches are
                    # nearly colinear continuations of the trunk — they point in
                    # OPPOSITE directions away from the split point.  A side branch
                    # departs at a large angle to one or both main branches, so its
                    # pairwise dot product with any main branch is NOT strongly negative.
                    #
                    # Algorithm — pairwise signed-dot scoring:
                    #
                    #   1. Compute the direction of every candidate branch by walking
                    #      DIVERGE_SKIP_MM from the bifurcation node OUTWARD along
                    #      that branch (i.e. away from the bif).  Direction is the
                    #      unit vector from the bif node to the sampled point.
                    #
                    #   2. Pre-filter: remove branches shorter than
                    #      LENGTH_FRAC × max_candidate_length.  This kills short
                    #      junction stubs before pair evaluation.
                    #
                    #   3. Evaluate every pair (i, j) with:
                    #         pair_score = W_angle * (−dot(d_i, d_j))   ← want opposite → dot≈−1
                    #                    + W_len   * (len_i + len_j) / max_len
                    #                    + W_flow  * (|dot(d_i, trunk)| + |dot(d_j, trunk)|)
                    #      The trunk-alignment bonus (W_flow) rewards pairs that are
                    #      both roughly aligned with the trunk axis — i.e. the true
                    #      flow-continuation pair, not a side branch plus anything.
                    #
                    #   4. Select the highest-scoring pair.  If fewer than 2 survive
                    #      the length pre-filter, relax to all graph-adjacent limbs.
                    #
                    # Weights (tuned for iliac/carotid anatomy — adjust if needed):
                    #   W_angle = 3.0  (dominant: oppose direction is primary signal)
                    #   W_len   = 1.5  (secondary: prefer longer branches)
                    #   W_flow  = 1.0  (tertiary: trunk-axis alignment bonus)
                    #
                    # Trunk-continuation suppression (SIGNED dot, not abs):
                    #   A branch with dot(outward_dir, trunk_dir) > +SAME_DIR_THRESH
                    #   is running in the SAME direction as the trunk — it is a trunk
                    #   continuation artifact, not a true bifurcation limb.  Suppress it.
                    #   IMPORTANT: do NOT suppress branches with dot ≈ −1 (opposite to
                    #   trunk) — those are valid main branches running "the other way."
                    #   This is the key distinction abs(dot) gets wrong.
                    #
                    # For aorto-iliac anatomy: the trunk runs inferiorly; both iliac
                    # branches run inferiorly too (dot < 0 against the proximal-to-distal
                    # trunk direction), so SAME_DIR_THRESH should be kept fairly high
                    # (0.90+) to avoid suppressing real branches.

                    DIVERGE_SKIP_MM = 15.0  # mm from bif node before sampling direction
                    LENGTH_FRAC = 0.35  # pre-filter: drop branches < this × max_len
                    SAME_DIR_THRESH = (
                        0.92  # suppress if dot(branch, trunk) > this value
                    )
                    # FIX (v297-CranialExcl): exclude strongly cranial short
                    # branches from the main-branch pair candidates.  A branch
                    # with dot(outward, trunk) < −CRANIAL_EXCL_THRESH is departing
                    # OPPOSITE to the trunk direction — i.e. heading cranially when
                    # the trunk runs caudally.  If it is also short enough to be a
                    # renal vein (< _RENAL_MAX_LEN_MM), it should not compete for
                    # the main-branch pair slots.
                    #
                    # This is a symptom-level guard; the root cause (iliac bif topo
                    # collapse) is fixed in per_branch_centerline_pipeline.py.  The
                    # guard fires only when ≥ 2 limbs remain after cranial exclusion,
                    # so it cannot leave us with fewer than 2 candidates.
                    CRANIAL_EXCL_THRESH = 0.70
                    CRANIAL_EXCL_MAX_MM = 120.0  # same ceiling as _RENAL_MAX_LEN_MM
                    W_ANGLE = 3.0
                    W_LEN = 1.5
                    W_FLOW = 1.0

                    def _outwardDir(bi, bifNodeIdx, skipMm=DIVERGE_SKIP_MM):
                        """Unit vector from the bif node OUTWARD along branch bi,
                        measured at skipMm from the bif node.
                        After flow normalization, graphBranches[bi][0] is guaranteed
                        to be the bif-end, so we always walk from pts[0] forward."""
                        if bi >= len(graphBranches) or not graphBranches[bi]:
                            return list(_trunkDir)
                        pts = graphBranches[bi]
                        bifPos = nodePos[bifNodeIdx]
                        # Walk skipMm from the start (bif end) along the branch
                        acc = 0.0
                        p_sample = pts[-1]
                        for i in range(len(pts) - 1):
                            acc += _dist3(pts[i], pts[i + 1])
                            if acc >= skipMm:
                                p_sample = pts[i + 1]
                                break
                        # Outward direction: from bif node position to sampled point
                        v = [p_sample[k] - bifPos[k] for k in range(3)]
                        vl = math.sqrt(sum(x * x for x in v))
                        return [x / vl for x in v] if vl > 1e-6 else list(_trunkDir)

                    mainSet = set()
                    if trueBifNode is not None:
                        primaryLimbs = bifLimbs[trueBifNode]

                        # Pre-compute directions and lengths for logging
                        _dirs = {
                            bi: _outwardDir(bi, trueBifNode) for bi in primaryLimbs
                        }
                        _lens = {
                            bi: self.branchMeta[bi]["length_mm"] for bi in primaryLimbs
                        }

                        # Diameter: prefer VMTK rawRadii, fall back to surface locator
                        def _limbDiam(bi):
                            d = _meanDiam(bi)
                            if d > 0:
                                return d
                            if _eSurfLoc is not None and bi < len(graphBranches):
                                bpts = graphBranches[bi]
                                if bpts:
                                    mid = bpts[len(bpts) // 2]
                                    cp = [0.0, 0.0, 0.0]
                                    ci = vtk.reference(0)
                                    si = vtk.reference(0)
                                    d2 = vtk.reference(0.0)
                                    _eSurfLoc.FindClosestPoint(
                                        list(mid), cp, ci, si, d2
                                    )
                                    return max(math.sqrt(float(d2)) * 2.0, 0.5)
                            return 0.0

                        _diams = {bi: _limbDiam(bi) for bi in primaryLimbs}
                        max_len = max(_lens.values()) if _lens else 1.0

                        _will_suppress = (
                            len(primaryLimbs) > 2
                        )  # suppression only for 3+ limbs
                        print(
                            f"[VesselAnalyzer] Main-branch pairwise evaluation "
                            f"({len(primaryLimbs)} graph-adjacent limbs at bif node {trueBifNode}):"
                        )
                        for bi in primaryLimbs:
                            d_trunk = sum(_dirs[bi][k] * _trunkDir[k] for k in range(3))
                            suppressed = _will_suppress and d_trunk > SAME_DIR_THRESH
                            print(
                                f"  Branch {bi}: len={_lens[bi]:.1f}mm "
                                f"diam={_diams[bi]:.1f}mm "
                                f"dot_trunk={d_trunk:+.2f} "
                                f"{'SUPPRESSED (trunk continuation)' if suppressed else 'candidate'}"
                            )

                        # Trunk-continuation suppression (SIGNED dot — not abs).
                        #
                        # Purpose: at a trifurcation or noisy junction, exclude the limb
                        # that is simply a straight continuation of the trunk rather than
                        # a true bifurcation outflow.
                        #
                        # When there are only 2 limbs at the primary bif node (standard
                        # Y-bifurcation — e.g. aortic/IVC split into two iliacs), both
                        # limbs MUST be the main branches regardless of their direction
                        # relative to the trunk.  Suppression would eliminate one valid
                        # branch.  Skip suppression in this case.
                        #
                        # When there are 3+ limbs, one may be a spurious trunk extension
                        # artifact.  Apply suppression only then.
                        if len(primaryLimbs) <= 2:
                            valid_limbs = list(primaryLimbs)
                            print(
                                f"[VesselAnalyzer] Trunk suppression skipped "
                                f"(only {len(primaryLimbs)} limb(s) — Y-bifurcation)"
                            )
                        else:
                            valid_limbs = [
                                bi
                                for bi in primaryLimbs
                                if sum(_dirs[bi][k] * _trunkDir[k] for k in range(3))
                                <= SAME_DIR_THRESH
                            ]
                            if len(valid_limbs) < 2:
                                valid_limbs = list(primaryLimbs)
                                print(
                                    f"[VesselAnalyzer] Trunk suppression relaxed "
                                    f"(< 2 valid after threshold {SAME_DIR_THRESH:.2f})"
                                )
                            else:
                                print(
                                    f"[VesselAnalyzer] After trunk suppression: "
                                    f"{len(valid_limbs)}/{len(primaryLimbs)} remain"
                                )

                        # FIX (v297-CranialExcl): remove strongly cranial
                        # short-branch candidates — they are likely renal veins
                        # that slipped past trunk suppression because their
                        # dot(branch, trunk) is strongly NEGATIVE (heading
                        # opposite = cranially), not positive like a trunk
                        # continuation.  Only apply when it leaves ≥ 2 limbs.
                        cranial_excluded = [
                            bi for bi in valid_limbs
                            if (
                                sum(_dirs[bi][k] * _trunkDir[k] for k in range(3))
                                < -CRANIAL_EXCL_THRESH
                                and _lens[bi] < CRANIAL_EXCL_MAX_MM
                            )
                        ]
                        if cranial_excluded and len(valid_limbs) - len(cranial_excluded) >= 2:
                            for _bi in cranial_excluded:
                                _d = sum(_dirs[_bi][k] * _trunkDir[k] for k in range(3))
                                print(
                                    f"[VesselAnalyzer] CranialExcl: branch {_bi} "
                                    f"excluded from main-branch pair "
                                    f"(dot_trunk={_d:+.2f} < -{CRANIAL_EXCL_THRESH}, "
                                    f"len={_lens[_bi]:.1f}mm < {CRANIAL_EXCL_MAX_MM:.0f}mm)"
                                )
                            valid_limbs = [
                                bi for bi in valid_limbs
                                if bi not in cranial_excluded
                            ]
                        elif cranial_excluded:
                            print(
                                f"[VesselAnalyzer] CranialExcl: would remove "
                                f"{len(cranial_excluded)} cranial branch(es) but "
                                f"only {len(valid_limbs)} limbs total — skipped"
                            )

                        # Length pre-filter: keep branches ≥ LENGTH_FRAC × max length
                        long_limbs = [
                            bi
                            for bi in valid_limbs
                            if _lens[bi] >= LENGTH_FRAC * max_len
                        ]
                        if len(long_limbs) < 2:
                            long_limbs = list(valid_limbs)
                            print(
                                f"[VesselAnalyzer] Length pre-filter relaxed "
                                f"(< 2 branches ≥ {LENGTH_FRAC:.0%} of {max_len:.0f}mm)"
                            )
                        else:
                            print(
                                f"[VesselAnalyzer] Length pre-filter: "
                                f"{len(long_limbs)}/{len(valid_limbs)} branches pass "
                                f"(≥ {LENGTH_FRAC:.0%} × {max_len:.0f}mm = "
                                f"{LENGTH_FRAC*max_len:.0f}mm)"
                            )

                        # Pairwise scoring — signed dot rewards opposite-pointing pairs.
                        #
                        # Low-angle fallback: when the inter-branch angle is < 10°
                        # (dot > cos(10°) ≈ 0.985), both branches are nearly parallel
                        # and the angle signal is unreliable.  In that regime, weight
                        # length + trunk-alignment more heavily (W_LEN and W_FLOW
                        # double, W_ANGLE halved) so the longer, better-aligned pair wins.
                        LOW_ANGLE_DOT = 0.985  # cos(10°)

                        from itertools import combinations as _combinations

                        best_pair = None
                        best_score = -1e18
                        for bi, bj in _combinations(long_limbs, 2):
                            di = _dirs[bi]
                            dj = _dirs[bj]
                            dot_ij = sum(di[k] * dj[k] for k in range(3))
                            angle_score = -dot_ij

                            # Adaptive weights for low-angle pairs
                            low_angle = dot_ij > LOW_ANGLE_DOT
                            wa = W_ANGLE * (0.5 if low_angle else 1.0)
                            wl = W_LEN * (2.0 if low_angle else 1.0)
                            wf = W_FLOW * (2.0 if low_angle else 1.0)

                            len_score = (_lens[bi] + _lens[bj]) / max_len
                            flow_score = abs(
                                sum(di[k] * _trunkDir[k] for k in range(3))
                            ) + abs(sum(dj[k] * _trunkDir[k] for k in range(3)))
                            pair_score = (
                                wa * angle_score + wl * len_score + wf * flow_score
                            )
                            la_tag = " [low-angle]" if low_angle else ""
                            print(
                                f"  [PAIR] {bi}–{bj}: dot={dot_ij:+.2f} "
                                f"angle_sc={angle_score:+.2f} "
                                f"len_sc={len_score:.2f} "
                                f"flow_sc={flow_score:.2f} "
                                f"→ pair_score={pair_score:.2f}{la_tag}"
                            )
                            if pair_score > best_score:
                                best_score = pair_score
                                best_pair = (bi, bj)
                        _GATE_MIN_LAT_MM = 15.0  # was 20.0
                        if best_pair:
                            mainSet.update(best_pair)
                            print(
                                f"[VesselAnalyzer] Selected main branches: "
                                f"{sorted(mainSet)} (pair_score={best_score:.2f})"
                            )
                        else:
                            # Absolute fallback: take two longest
                            ranked = sorted(
                                primaryLimbs, key=lambda b: _lens[b], reverse=True
                            )
                            mainSet.update(ranked[:2])
                            print(
                                f"[VesselAnalyzer] Fallback: selected two longest "
                                f"branches: {sorted(mainSet)}"
                            )

                    # ── detect broken branches ────────────────────────────────────
                    def _is_broken_branch(bi, drop_ratio=0.35, abs_min=2.5, grad_thresh=0.5, angle_thresh_deg=120.0):
                        try:
                            bs, be = self.branches[bi]
                            b_points = self.points[bs:be]
                            
                            def _get_radius(pt):
                                if hasattr(self, "rawRadii") and len(self.rawRadii) == len(self.points):
                                    # Fallback index mapping if needed, but since we are iterating
                                    # we'll just use the surface locator for robustness.
                                    pass
                                
                                if _eSurfLoc is None:
                                    return 5.0
                                import vtk
                                import math
                                cp = [0.0, 0.0, 0.0]
                                ci = vtk.reference(0)
                                si = vtk.reference(0)
                                d2 = vtk.reference(0.0)
                                _eSurfLoc.FindClosestPoint(list(pt), cp, ci, si, d2)
                                return math.sqrt(float(d2))

                            b_radii = [_get_radius(pt) for pt in b_points]
                            if len(b_radii) < 5:
                                return False
                            
                            r_max = max(b_radii)
                            r_min = min(b_radii)
                                
                            r_np = _np.array(b_radii)
                            grad = _np.abs(_np.diff(r_np) / (r_np[:-1] + 1e-6))
                            max_grad = _np.max(grad) if len(grad) > 0 else 0
                            
                            def angle(u, v):
                                nrm = _np.linalg.norm(u) * _np.linalg.norm(v)
                                if nrm < 1e-6: return 0.0
                                cosang = _np.dot(u, v) / nrm
                                return _np.degrees(_np.arccos(_np.clip(cosang, -1, 1)))
                                
                            pts_np = _np.array(b_points)
                            max_angle = 0
                            for i in range(1, len(pts_np)-1):
                                v1 = pts_np[i] - pts_np[i-1]
                                v2 = pts_np[i+1] - pts_np[i]
                                max_angle = max(max_angle, angle(v1, v2))

                            print(f"[BrokenBranch DEBUG] bi={bi} r_max={r_max:.2f} r_min={r_min:.2f} "
                                  f"max_grad={max_grad:.2f} max_angle={max_angle:.1f}")

                            if r_min < drop_ratio * r_max:
                                print(f"[BrokenBranch] bi={bi} rejected due to neck collapse "
                                      f"(r_min={r_min:.2f} < {drop_ratio}*r_max={drop_ratio*r_max:.2f})")
                                return True
                            if r_min < abs_min:
                                print(f"[BrokenBranch] bi={bi} rejected due to absolute neck collapse "
                                      f"(r_min={r_min:.2f} < {abs_min})")
                                return True
                                
                            if max_grad > grad_thresh:
                                print(f"[BrokenBranch] bi={bi} rejected due to sharp radius drop "
                                      f"(max_grad={max_grad:.2f} > {grad_thresh})")
                                return True
                                
                            for i in range(1, len(pts_np)-1):
                                v1 = pts_np[i] - pts_np[i-1]
                                v2 = pts_np[i+1] - pts_np[i]
                                if angle(v1, v2) > angle_thresh_deg:
                                    print(f"[BrokenBranch] bi={bi} rejected due to direction flip > {angle_thresh_deg} deg")
                                    return True
                            return False
                        except Exception as e:
                            import traceback
                            print(f"[BrokenBranch] Error checking bi={bi}: {e}\n{traceback.format_exc()}")
                            return False

                    # ── assign roles ──────────────────────────────────────────────
                    for bi in range(len(self.branches)):
                        if bi == 0:
                            self.branchMeta[bi]["role"] = "trunk"
                        elif _is_broken_branch(bi):
                            self.branchMeta[bi]["role"] = "noise"
                        elif bi in mainSet:
                            self.branchMeta[bi]["role"] = "main"
                        else:
                            self.branchMeta[bi]["role"] = "side"


                    # ── renal vein / artery tagging ───────────────────────────────
                    #
                    # Tag side branches as 'renal_vein' if they meet the origin +
                    # length criteria AND score above the composite threshold.
                    #
                    # Vessel-type gating (VesselTypeGate, v297):
                    #   "venous"  — full renal-vein classifier active.
                    #               Threshold 0.45 is tuned for IVC geometry where
                    #               renal veins are wide (≥45% of trunk diameter).
                    #   "arterial" — classifier is SKIPPED entirely.
                    #               The composite scorer was tuned for venous anatomy;
                    #               renal arteries are far narrower relative to the
                    #               aorta so diam_ratio is structurally low, causing
                    #               false positives at threshold 0.45.  Until a
                    #               dedicated arterial classifier exists, all lateral
                    #               branches on arterial cases stay 'side'.
                    #
                    # Tagged branches are kept in anatomy output (real vessels)
                    # but downstream stent planning excludes them.

                    _vessel_type = getattr(self, "vesselType", "venous") or "venous"
                    _is_arterial = _vessel_type.lower().startswith("art")

                    if _is_arterial:
                        # Skip the entire renal classifier for arterial cases.
                        # All side branches remain role='side'.
                        print(
                            f"[RenalTag] vesselType={_vessel_type!r} → "
                            f"renal classifier SKIPPED (arterial pipeline)"
                        )

                    _RENAL_MAX_LEN_MM      = 120.0   # soft ceiling — flagged but admitted
                    _RENAL_MAX_LEN_HARD_MM = 280.0   # hard ceiling — trunk continuations
                    _RENAL_DIAM_RATIO = (
                        0.45  # branch_avg / trunk_avg (legacy, kept for log)
                    )
                    _RENAL_TRUNK_SNAP_MM = (
                        20.0  # mm — origin within this of trunk (was 10)
                    )
                    _RENAL_ANGLE_DEG = 45.0  # departure angle — kept for log compat
                    _RENAL_SCORE_THRESH = (
                        0.45  # composite score gate (replaces OR-gate)
                    )

                    # _clamp / _renal_anatomy_gate / _renal_composite_score
                    # removed — now imported from vessel_geometry; local aliases
                    # preserve call-site names without any other changes.
                    _clamp = lambda v, lo=0.0, hi=1.0: geo_clamp(v, lo, hi)
                    _renal_anatomy_gate    = renal_anatomy_gate
                    _renal_composite_score = renal_composite_score

                    # self.branches[bi] = (startIdx, endIdx) where endIdx is
                    # exclusive (= len(rawPoints) after extend).
                    # self.points is not yet assigned at this stage — use rawPoints.
                    def _renalBranchPts(bi):
                        bs, be = self.branches[bi]
                        return rawPoints[bs:be]

                    _trunkPts = _renalBranchPts(0)

                    # ── trunk X-centroid (used for lateral pairing) ───────────────
                    # Computed ONCE here so every candidate gets the same reference.
                    # If _trunkPts is empty the centroid is None; the per-branch
                    # tip_x_sign assignment will detect this and log explicitly
                    # rather than silently falling back to 0 (which disables pairing).
                    if _trunkPts:
                        _trunk_x_centroid = sum(p[0] for p in _trunkPts) / len(
                            _trunkPts
                        )
                    else:
                        _trunk_x_centroid = None
                        print(
                            "[RenalTag] WARNING: trunk branch (bi=0) has no points — "
                            "tip_x_sign cannot be computed; lateral pairing will be disabled. "
                            "Check that rawPoints and self.branches[0] are populated before RenalTag."
                        )

                    def _midptDiam(bpts):
                        """Sample diameter at branch midpoint via surface locator."""
                        if not bpts:
                            return 0.0
                        mid = bpts[len(bpts) // 2]
                        if _eSurfLoc is not None:
                            cp = [0.0, 0.0, 0.0]
                            ci = vtk.reference(0)
                            si = vtk.reference(0)
                            d2 = vtk.reference(0.0)
                            _eSurfLoc.FindClosestPoint(list(mid), cp, ci, si, d2)
                            return max(math.sqrt(float(d2)) * 2.0, 0.5)
                        return 0.0

                    _trunkDiam = _midptDiam(_trunkPts)

                    # Trunk Z extent — used to reject stubs that originate
                    # beyond either trunk endpoint (above root or below bif).
                    # Use ALL committed branch points, not just branch 0.
                    # Reason: second-order side branches (e.g. suprarenal IVC
                    # collaterals rooted beyond node 318) originate from segments
                    # that failed the trunk-edge criteria and were folded into
                    # non-trunk branches — their origin Z is legitimately outside
                    # branch-0's Z range even though they are real vessels.
                    # Covering all branch points gives the true anatomical extent.
                    _allZs = [
                        p[2]
                        for bi in range(len(self.branches))
                        for p in _renalBranchPts(bi)
                    ]
                    _trunkZmin = min(_allZs)
                    _trunkZmax = max(_allZs)
                    _trunkOnlyZmin = min(p[2] for p in _trunkPts)
                    _trunkOnlyZmax = max(p[2] for p in _trunkPts)
                    print(
                        f"[NoiseGate] Vessel Z extent: [{_trunkZmin:.1f}, {_trunkZmax:.1f}] "
                        f"(trunk-only was [{_trunkOnlyZmin:.1f}, {_trunkOnlyZmax:.1f}])"
                    )

                    # Trunk axis direction (root→bif, computed in flow-norm above)
                    # _trunkDir is always set before this block — initialized to a
                    # safe fallback [0,0,1] at line ~9798 and overwritten with the
                    # real value if trunkPtsOrdered is valid.
                    _renalTrunkAxis = _trunkDir

                    # snappedBifPt is assigned later; use trunk distal end as proxy.
                    _bif_z_ref = (
                        trunkPtsOrdered[-1][2] if trunkPtsOrdered else _trunkOnlyZmin
                    )

                    # ── Noise gate: prune stubs that start beyond trunk Z extent ──
                    # Runs for ALL vessel types — this is a topology filter, not
                    # anatomy-specific.  Moved out of Pass 1 so it fires even when
                    # the renal classifier is skipped (arterial cases).
                    _TRUNK_Z_MARGIN = 5.0
                    for _ng_bi in range(1, len(self.branches)):
                        if self.branchMeta[_ng_bi]["role"] != "side":
                            continue
                        _ng_bpts = _renalBranchPts(_ng_bi)
                        if not _ng_bpts:
                            continue
                        _ng_z = _ng_bpts[0][2]
                        if (
                            _ng_z > _trunkZmax + _TRUNK_Z_MARGIN
                            or _ng_z < _trunkZmin - _TRUNK_Z_MARGIN
                        ):
                            print(
                                f"[NoiseGate] Branch {_ng_bi}: origin Z={_ng_z:.1f} "
                                f"outside trunk Z=[{_trunkZmin:.1f},{_trunkZmax:.1f}] "
                                f"→ beyond-trunk stub, noise"
                            )
                            self.branchMeta[_ng_bi]["role"] = "noise"

                    # ── Noise gate: bifurcation stubs ─────────────────────────────
                    # VMTK sometimes inserts a short extra edge at the iliac bifurcation
                    # node, producing a graph branch that:
                    #   • originates within a few mm of the bif Z (ostium ≈ snappedBifPt)
                    #   • is very short (< 40 mm)
                    #   • has iliac-scale diameter (≥ 10 mm) — not a collateral
                    #   • sits close to the trunk centerline (low min_trunk_dist)
                    # These stubs are not anatomical vessels; they are topology artefacts
                    # from the junction.  Marking them noise here prevents them from
                    # appearing as a named colored region on the surface and from
                    # entering the renal classifier.
                    # Safety guard: if a branch at bif Z is strongly lateral (lat>30mm,
                    # ang>25°) it is a real vessel — leave it alone.
                    _BIF_STUB_Z_TOL_MM   = 8.0    # ostium within this of _bif_z_ref
                    _BIF_STUB_LEN_MAX_MM = 40.0
                    _BIF_STUB_DIAM_MIN   = 10.0   # ignore sub-collateral stubs
                    _BIF_STUB_SNAP_MM    = 8.0    # origin close to trunk centerline
                    _BIF_STUB_LAT_SAFE   = 30.0   # lateral escape → real branch
                    _BIF_STUB_ANG_SAFE   = 25.0   # departure angle → real branch

                    for _ng_bi in range(1, len(self.branches)):
                        if self.branchMeta[_ng_bi]["role"] != "side":
                            continue
                        _ng_bpts = _renalBranchPts(_ng_bi)
                        if not _ng_bpts:
                            continue
                        _ng_len  = self.branchMeta[_ng_bi].get("length_mm", 0.0)
                        _ng_oz   = _ng_bpts[0][2]
                        _ng_diam = _midptDiam(_ng_bpts)

                        if not (
                            abs(_ng_oz - _bif_z_ref) <= _BIF_STUB_Z_TOL_MM
                            and _ng_len <= _BIF_STUB_LEN_MAX_MM
                            and _ng_diam >= _BIF_STUB_DIAM_MIN
                        ):
                            continue

                        # Compute trunk snap distance for this branch
                        _ng_snap = min(_dist3(_ng_bpts[0], _tp) for _tp in _trunkPts)
                        if _ng_snap > _BIF_STUB_SNAP_MM:
                            continue

                        # Safety: real lateral branch at bif level → leave it
                        _ng_ang = 0.0
                        _ng_lat = 0.0
                        if len(_ng_bpts) >= 2:
                            _ng_rv  = [_ng_bpts[-1][k] - _ng_bpts[0][k] for k in range(3)]
                            _ng_rvl = math.sqrt(sum(x*x for x in _ng_rv))
                            if _ng_rvl > 1e-6:
                                _ng_rv  = [x/_ng_rvl for x in _ng_rv]
                                _ng_ang = math.degrees(math.acos(
                                    max(-1.0, min(1.0, abs(sum(
                                        _ng_rv[k]*_renalTrunkAxis[k] for k in range(3)
                                    ))))
                                ))
                            try:
                                _ta_r2 = trunkPtsOrdered[0]
                                _ta_b2 = trunkPtsOrdered[-1]
                                _ta_v2 = [_ta_b2[k]-_ta_r2[k] for k in range(3)]
                                _ta_l2 = math.sqrt(sum(x*x for x in _ta_v2))
                                if _ta_l2 > 1e-6:
                                    _ta_h2   = [x/_ta_l2 for x in _ta_v2]
                                    _tip_r2  = [_ng_bpts[-1][k]-_ta_r2[k] for k in range(3)]
                                    _pr2     = sum(_tip_r2[k]*_ta_h2[k] for k in range(3))
                                    _pe2     = [_tip_r2[k]-_pr2*_ta_h2[k] for k in range(3)]
                                    _ng_lat  = math.sqrt(sum(x*x for x in _pe2))
                            except Exception:
                                pass

                        if _ng_lat > _BIF_STUB_LAT_SAFE and _ng_ang > _BIF_STUB_ANG_SAFE:
                            print(
                                f"[NoiseGate] Branch {_ng_bi}: near bif (dZ={abs(_ng_oz-_bif_z_ref):.1f}mm)"
                                f" but lateral (lat={_ng_lat:.1f}mm, ang={_ng_ang:.1f}°) → kept"
                            )
                            continue

                        print(
                            f"[NoiseGate] Branch {_ng_bi}: bifurcation stub"
                            f" (dZ={abs(_ng_oz-_bif_z_ref):.1f}mm, len={_ng_len:.1f}mm,"
                            f" diam={_ng_diam:.1f}mm, snap={_ng_snap:.1f}mm,"
                            f" lat={_ng_lat:.1f}mm, ang={_ng_ang:.1f}°) → noise"
                        )
                        self.branchMeta[_ng_bi]["role"] = "noise"

                    # ── Pass 1: gather per-branch geometry for all side candidates ──
                    # We need geometry from ALL candidates before scoring so we can run
                    # pair-symmetry and dominance analysis (topology layer, v182).
                    # Skipped entirely for arterial cases (see VesselTypeGate above).
                    _renal_candidates = {}  # bi → dict of pre-computed geometry

                    for bi in range(1, len(self.branches)) if not _is_arterial else []:
                        if self.branchMeta[bi]["role"] != "side":
                            continue

                        _blen = self.branchMeta[bi].get("length_mm", 0.0)
                        _bpts = _renalBranchPts(bi)
                        _bdiam = _midptDiam(_bpts)
                        _bstart = _bpts[0]

                        # ── noise gate: stub that starts beyond trunk endpoints ──
                        # (already handled above — kept as a no-op guard)
                        _bstart_z = _bstart[2]

                        # ── length gate (soft + hard) ─────────────────────────────
                        # Compute diagnostic angle/lat for all length-flagged branches
                        # (same block serves both the hard-reject log and the soft-flag log).
                        _length_flagged = False
                        if _blen >= _RENAL_MAX_LEN_MM:
                            _rej_ang = 0.0
                            _rej_lat = 0.0
                            if len(_bpts) >= 2:
                                _rv = [_bpts[-1][k] - _bpts[0][k] for k in range(3)]
                                _rvl = math.sqrt(sum(x*x for x in _rv))
                                if _rvl > 1e-6:
                                    _rv = [x/_rvl for x in _rv]
                                    _rej_ang = math.degrees(math.acos(
                                        max(-1.0, min(1.0, abs(sum(_rv[k]*_renalTrunkAxis[k] for k in range(3)))))
                                    ))
                            try:
                                _ta_r = trunkPtsOrdered[0]
                                _ta_b = trunkPtsOrdered[-1]
                                _ta_v = [_ta_b[k]-_ta_r[k] for k in range(3)]
                                _ta_l = math.sqrt(sum(x*x for x in _ta_v))
                                if _ta_l > 1e-6:
                                    _ta_h = [x/_ta_l for x in _ta_v]
                                    _tip_r = [_bpts[-1][k]-_ta_r[k] for k in range(3)]
                                    _pr = sum(_tip_r[k]*_ta_h[k] for k in range(3))
                                    _pe = [_tip_r[k]-_pr*_ta_h[k] for k in range(3)]
                                    _rej_lat = math.sqrt(sum(x*x for x in _pe))
                            except Exception:
                                pass

                            if _blen >= _RENAL_MAX_LEN_HARD_MM:
                                # Absolute ceiling — true trunk continuation, no rescue possible.
                                print(
                                    f"[RenalTag] Branch {bi}: len={_blen:.1f}mm"
                                    f" >= hard_max={_RENAL_MAX_LEN_HARD_MM}mm → hard-rejected"
                                    f" [diagnostic: angle={_rej_ang:.1f}° lat={_rej_lat:.1f}mm]"
                                )
                                continue
                            else:
                                # Soft ceiling — flag for topo penalty; pair-bypass can still rescue.
                                _length_flagged = True
                                print(
                                    f"[RenalTag] Branch {bi}: len={_blen:.1f}mm"
                                    f" >= soft_max={_RENAL_MAX_LEN_MM}mm → length-flagged"
                                    f" [diagnostic: angle={_rej_ang:.1f}° lat={_rej_lat:.1f}mm]"
                                )

                        # Check origin: start of branch within snap distance of trunk.
                        # IMPORTANT: _trunkPts is branch-0 only, but the trunk centerline
                        # may span multiple branch indices when the graph folds (trunk-only
                        # Z range vs full vessel Z range can differ by 300mm+).  Use ALL
                        # committed branch points so ostia on the lower IVC are not silently
                        # dropped.  Role filter excludes already-classified noise/renal/main
                        # to avoid false proximity from iliac branches.
                        _all_trunk_ref_pts = [
                            _pt
                            for _tbi in range(len(self.branches))
                            if self.branchMeta.get(_tbi, {}).get("role") not in ("side", "noise")
                            for _pt in _renalBranchPts(_tbi)
                        ] or _trunkPts  # fallback to branch-0 if filter yields nothing
                        _min_trunk_dist = min(
                            _dist3(_bstart, _tp) for _tp in _all_trunk_ref_pts
                        )
                        # Length-flagged branches (long laterals) get a wider snap
                        # allowance because their ostium may lie on a trunk segment
                        # represented by a non-zero branch index.
                        _snap_thresh = (
                            _RENAL_TRUNK_SNAP_MM * 2.0
                            if _length_flagged
                            else _RENAL_TRUNK_SNAP_MM
                        )
                        if _min_trunk_dist > _snap_thresh:
                            print(
                                f"[RenalTag] Branch {bi}: origin too far from trunk"
                                f" ({_min_trunk_dist:.1f}mm > {_snap_thresh:.0f}mm)"
                                f" → skipped"
                            )
                            continue

                        # ── departure angle from trunk axis ───────────────────────
                        _RENAL_PROX_MM = 15.0
                        if len(_bpts) >= 2:
                            _r_root = _bpts[0]
                            _r_samp = _bpts[-1]
                            _r_acc = 0.0
                            for _rpi in range(len(_bpts) - 1):
                                _r_acc += _dist3(_bpts[_rpi], _bpts[_rpi + 1])
                                if _r_acc >= _RENAL_PROX_MM:
                                    _r_samp = _bpts[_rpi + 1]
                                    break
                                _r_samp = _bpts[_rpi + 1]
                            _bvec = [_r_samp[k] - _r_root[k] for k in range(3)]
                            _bvl = math.sqrt(sum(x * x for x in _bvec))
                            if _bvl > 1e-6:
                                _bvec = [x / _bvl for x in _bvec]
                                _dot = sum(
                                    _bvec[k] * _renalTrunkAxis[k] for k in range(3)
                                )
                                _ang = math.degrees(
                                    math.acos(max(-1.0, min(1.0, abs(_dot))))
                                )
                            else:
                                _ang = 0.0
                        else:
                            _ang = 0.0

                        # ── lateral tip offset from trunk axis ────────────────────
                        _btip = _bpts[-1]
                        try:
                            _ta_root = trunkPtsOrdered[0]
                            _ta_bif = trunkPtsOrdered[-1]
                            _ta_vec = [_ta_bif[k] - _ta_root[k] for k in range(3)]
                            _ta_len = math.sqrt(sum(x * x for x in _ta_vec))
                            if _ta_len > 1e-6:
                                _ta_hat = [x / _ta_len for x in _ta_vec]
                                _tip_rel = [_btip[k] - _ta_root[k] for k in range(3)]
                                _proj = sum(_tip_rel[k] * _ta_hat[k] for k in range(3))
                                _perp = [
                                    _tip_rel[k] - _proj * _ta_hat[k] for k in range(3)
                                ]
                                _lat_offset_mm = math.sqrt(sum(x * x for x in _perp))
                            else:
                                _lat_offset_mm = 0.0
                        except Exception:
                            _lat_offset_mm = 0.0

                        # ── tip X-sign relative to trunk centroid (for pairing) ───
                        # _trunk_x_centroid is pre-computed once above _renal_candidates.
                        # If it is None the trunk had no points — pairing is impossible
                        # for this case and we set 0 (neutral) with an explicit log so
                        # the failure is visible rather than silently disabling pairing.
                        if _trunk_x_centroid is not None:
                            _tip_x_sign = (
                                1 if (_btip[0] - _trunk_x_centroid) >= 0 else -1
                            )
                        else:
                            _tip_x_sign = 0
                            print(
                                f"[RenalTag] Branch {bi}: tip_x_sign=0 (no trunk centroid); "
                                f"lateral pairing disabled for this branch"
                            )

                        # ── proximal / distal diameter for stability penalty ───────
                        _prox_d = self.branchMeta[bi].get("prox_diam", _bdiam)
                        _dist_d = self.branchMeta[bi].get("dist_diam", _bdiam)

                        _diam_ratio = (_bdiam / _trunkDiam) if _trunkDiam > 0 else 0.0
                        _ostium_z = _bpts[0][2]

                        # ── terminal alignment with trunk axis ────────────────────
                        # Measures whether the distal end of the branch is still
                        # running axially (iliac continuation) or has turned lateral
                        # (renal termination).  Sampled over the last 20mm of arc
                        # rather than a fixed point count — avoids resolution artefacts.
                        # Result: 0.0 = purely lateral terminus, 1.0 = purely axial.
                        _TERMINAL_SAMPLE_MM = 20.0
                        _term_pts = []
                        _t_acc = 0.0
                        for _ti in range(len(_bpts) - 1, 0, -1):
                            _t_acc += _dist3(_bpts[_ti], _bpts[_ti - 1])
                            _term_pts.append(_bpts[_ti])
                            if _t_acc >= _TERMINAL_SAMPLE_MM:
                                break
                        if len(_term_pts) >= 2:
                            _tvec = [_term_pts[0][k] - _term_pts[-1][k] for k in range(3)]
                            _tvl  = math.sqrt(sum(x*x for x in _tvec))
                            if _tvl > 1e-6:
                                _tvec = [x/_tvl for x in _tvec]
                                _terminal_align = abs(sum(
                                    _tvec[k] * _renalTrunkAxis[k] for k in range(3)
                                ))
                            else:
                                _terminal_align = 0.5  # degenerate direction → neutral
                        else:
                            _terminal_align = 0.5  # branch too short to sample → neutral

                        _renal_candidates[bi] = {
                            "blen": _blen,
                            "bdiam": _bdiam,
                            "diam_ratio": _diam_ratio,
                            "lat_mm": _lat_offset_mm,
                            "ang": _ang,
                            "ostium_z": _ostium_z,
                            "prox_d": _prox_d,
                            "dist_d": _dist_d,
                            "tip_x_sign": _tip_x_sign,
                            "tip_z": _btip[2],
                            "min_trunk_dist": _min_trunk_dist,
                            "bpts": _bpts,
                            "length_flagged": _length_flagged,
                            "terminal_align": _terminal_align,
                        }

                    # ── Pass 2: topology layer (v182) ─────────────────────────────
                    # 2a. Anatomy gate — remove obvious non-renals before pair logic.
                    #     Branches failing the gate are immediately side_branch.
                    print(
                        f"[RenalTag] Pass 2 entry: {len(_renal_candidates)} candidates "
                        f"(trunk_x_centroid={_trunk_x_centroid:.1f}mm)"
                        if _trunk_x_centroid is not None
                        else f"[RenalTag] Pass 2 entry: {len(_renal_candidates)} candidates "
                        f"(trunk_x_centroid=None — pairing disabled)"
                    )
                    for _bi, _geo in _renal_candidates.items():
                        print(
                            f"[RenalTag]   bi={_bi} keys={sorted(_geo.keys())} "
                            f"tip_x_sign={_geo['tip_x_sign']}"
                        )
                    # FIX (v297-RenalGate): _RENAL_SHORT_ZONE_MM is a softer
                    # floor that overrides the anatomy-gate len<40mm threshold
                    # when the branch is genuinely in the renal Z band and long
                    # enough to not be a stub (>= 25mm).  The Z band is defined
                    # as the stretch between the iliac bifurcation (bif_z_ref)
                    # and the trunk root; branches originating in this zone and
                    # departing laterally are renal-vein candidates regardless of
                    # whether their arc just misses the 40mm length threshold.
                    _RENAL_SHORT_ZONE_MM = 25.0

                    _gate_passed = {}
                    for _bi, _geo in _renal_candidates.items():

                        _gate_ok, _gate_reason = _renal_anatomy_gate(
                            _geo["blen"], _geo["lat_mm"], _geo["ostium_z"], _bif_z_ref
                        )

                        # Override A: short branch in renal Z zone — relax len floor.
                        # Applies only when the sole failure reason is length.
                        if (not _gate_ok
                                and "len=" in _gate_reason
                                and "<" in _gate_reason
                                and _geo["blen"] >= _RENAL_SHORT_ZONE_MM):
                            # Confirm origin is in the renal band (above bif, below root)
                            _oz = _geo["ostium_z"]
                            if _bif_z_ref <= _oz <= _trunkZmax:
                                _gate_ok = True
                                _gate_reason = (
                                    f"len_override (blen={_geo['blen']:.1f}mm "
                                    f">={_RENAL_SHORT_ZONE_MM:.0f}mm, in renal zone)"
                                )
                                print(
                                    f"[RenalTag] Branch {_bi}: "
                                    f"short-zone override → gate PASS ({_gate_reason})"
                                )

                        # Override B: dz-only failure with strong lateral evidence.
                        #
                        # The dz gate (abs(ostium_z - bif_z_ref) >= 20mm) rejects branches
                        # that originate too close to the iliac bifurcation in Z.  The intent
                        # is to exclude iliac continuations, which depart axially and have
                        # near-zero lateral offset.  However, a renal vein that enters the IVC
                        # near the bif level (short infra-renal segment) has ostium_z ≈ bif_z_ref,
                        # so dz=0 and the gate fires as a false positive.
                        #
                        # Rescue when ALL of:
                        #   1. dz is the ONLY failing reason ("dz=" in reason, "len=" not in reason)
                        #   2. lateral offset is large (>= 25mm) — proves sideways departure,
                        #      not a trunk continuation
                        #   3. branch length is in the renal range (>= 25mm) — safety redundancy,
                        #      already enforced by Pass 1 filters
                        #   4. departure angle is not trunk-like (>= 25° AND <= 70°)
                        #      — near-0° = axial continuation (iliac or trunk stub);
                        #        the MINIMUM of 25° is critical: a branch with lat=37mm
                        #        but angle=15° is running nearly parallel to the trunk
                        #        (e.g. an iliac that starts lateral) and must not be rescued.
                        #      EXCEPTION: when lat_mm >= 60mm the lateral displacement is
                        #      conclusive by itself — no iliac reaches 60mm lateral offset —
                        #      so the angle floor is relaxed to 15° to rescue renal veins
                        #      with a shallow proximal departure that curves laterally.
                        _DZ_OVERRIDE_LAT_MM         = 25.0
                        _DZ_OVERRIDE_ANG_MIN        = 25.0   # standard floor
                        _DZ_OVERRIDE_ANG_MIN_LAT    = 15.0   # relaxed floor when lat is conclusive
                        _DZ_OVERRIDE_LAT_CONCLUSIVE = 60.0   # mm — above this, lat alone is strong
                        _DZ_OVERRIDE_ANG_MAX        = 70.0
                        _DZ_OVERRIDE_LEN_MIN        = 25.0

                        _eff_ang_min = (
                            _DZ_OVERRIDE_ANG_MIN_LAT
                            if _geo["lat_mm"] >= _DZ_OVERRIDE_LAT_CONCLUSIVE
                            else _DZ_OVERRIDE_ANG_MIN
                        )

                        if (not _gate_ok
                                and "dz=" in _gate_reason
                                and "len=" not in _gate_reason
                                and not _geo["length_flagged"]   # long branch at bif Z = iliac, never renal
                                and _geo["lat_mm"] >= _DZ_OVERRIDE_LAT_MM
                                and _geo["blen"]   >= _DZ_OVERRIDE_LEN_MIN
                                and _geo["ang"]    >= _eff_ang_min
                                and _geo["ang"]    <= _DZ_OVERRIDE_ANG_MAX):
                            if _geo["ostium_z"] <= _trunkZmax:
                                _gate_ok = True
                                _gate_reason = (
                                    f"dz_lateral_override "
                                    f"(dz={abs(_geo['ostium_z'] - _bif_z_ref):.1f}mm<20 "
                                    f"but lat={_geo['lat_mm']:.1f}mm>={_DZ_OVERRIDE_LAT_MM:.0f}mm "
                                    f"ang={_geo['ang']:.1f}°∈[{_eff_ang_min:.0f}°,{_DZ_OVERRIDE_ANG_MAX:.0f}°] "
                                    f"len={_geo['blen']:.1f}mm)"
                                )
                                print(
                                    f"[RenalTag] Branch {_bi}: "
                                    f"dz-lateral override → gate PASS ({_gate_reason})"
                                )
                        elif (not _gate_ok
                                and "dz=" in _gate_reason
                                and _geo["length_flagged"]):
                            print(
                                f"[RenalTag] Branch {_bi}: "
                                f"dz-lateral override BLOCKED — length_flagged "
                                f"(len={_geo['blen']:.1f}mm, lat={_geo['lat_mm']:.1f}mm) "
                                f"→ long branch at bif Z is iliac, not renal"
                            )
                        elif (not _gate_ok
                                and "dz=" in _gate_reason
                                and _geo["ang"] < _eff_ang_min):
                            print(
                                f"[RenalTag] Branch {_bi}: "
                                f"dz-lateral override BLOCKED — angle={_geo['ang']:.1f}° "
                                f"< min={_eff_ang_min:.0f}° (near-axial branch, "
                                f"lat={_geo['lat_mm']:.1f}mm)"
                            )

                        _gate_passed[_bi] = _gate_ok

                        if not _gate_ok:
                            print(
                                f"[RenalTag] Branch {_bi}: GATE FAIL — {_gate_reason} → side_branch"
                            )

                    # Pair-bypass A: two gate-failed branches on opposite sides
                    # within 30mm Z are almost certainly a bilateral renal pair.
                    # Exception: if both are length_flagged AND both ostia are within
                    # 20mm of bif_z_ref, they are the iliac branches — reinstatement
                    # would undo the gate that just correctly rejected them.
                    _failed_bis = [_bi for _bi, _ok in _gate_passed.items() if not _ok]
                    for _i, _bi_a in enumerate(_failed_bis):
                        for _bi_b in _failed_bis[_i + 1 :]:
                            _ga = _renal_candidates[_bi_a]
                            _gb = _renal_candidates[_bi_b]
                            _x_opp = (
                                _ga["tip_x_sign"] != _gb["tip_x_sign"]
                                and _ga["tip_x_sign"] != 0
                                and _gb["tip_x_sign"] != 0
                            )
                            _dz_close = abs(_ga["ostium_z"] - _gb["ostium_z"]) < 30.0
                            # Block: both long AND opposite sides at same Z → iliacs
                            _both_length_flagged = (
                                _ga["length_flagged"] and _gb["length_flagged"]
                            )
                            if _x_opp and _dz_close:
                                if _both_length_flagged:
                                    print(
                                        f"[RenalTag] Pair-bypass A BLOCKED: Branch {_bi_a} + Branch {_bi_b} "
                                        f"both length_flagged (len={_ga['blen']:.1f}mm, {_gb['blen']:.1f}mm) "
                                        f"opposite sides at same Z → iliac pair, not renal"
                                    )
                                    continue
                                _gate_passed[_bi_a] = True
                                _gate_passed[_bi_b] = True
                                print(
                                    f"[RenalTag] Pair-bypass A: Branch {_bi_a} + Branch {_bi_b} "
                                    f"reinstated (both failed, opposite sides, "
                                    f"dZ={abs(_ga['ostium_z']-_gb['ostium_z']):.1f}mm)"
                                )

                    # Pair-bypass B: one gate-passed + one gate-failed on opposite sides.
                    # Occurs when asymmetric anatomy causes one branch to clear the gate
                    # while its contralateral partner fails (e.g. different proximal angle
                    # due to branching geometry).  Rescue the failed branch when:
                    #   • opposite X side from the passed branch
                    #   • ostia within 50mm Z (wider than bypass A — these are long branches)
                    #   • both have strong lateral offset (>= 40mm) — confirms bilateral anatomy
                    #   • similar length (ratio < 2.5 — prevents iliac from riding along)
                    _passed_bis = [_bi for _bi, _ok in _gate_passed.items() if _ok]
                    _failed_bis_now = [_bi for _bi, _ok in _gate_passed.items() if not _ok]
                    _BYPASS_B_Z_MM    = 50.0
                    _BYPASS_B_LAT_MM  = 40.0
                    _BYPASS_B_LEN_RATIO = 2.5
                    for _bi_p in _passed_bis:
                        for _bi_f in _failed_bis_now:
                            _gp = _renal_candidates[_bi_p]
                            _gf = _renal_candidates[_bi_f]
                            _x_opp = (
                                _gp["tip_x_sign"] != _gf["tip_x_sign"]
                                and _gp["tip_x_sign"] != 0
                                and _gf["tip_x_sign"] != 0
                            )
                            _dz_close = abs(_gp["ostium_z"] - _gf["ostium_z"]) < _BYPASS_B_Z_MM
                            _both_lateral = (
                                _gp["lat_mm"] >= _BYPASS_B_LAT_MM
                                and _gf["lat_mm"] >= _BYPASS_B_LAT_MM
                            )
                            _len_similar = (
                                max(_gp["blen"], _gf["blen"])
                                / max(min(_gp["blen"], _gf["blen"]), 1e-6)
                                < _BYPASS_B_LEN_RATIO
                            )
                            if _x_opp and _dz_close and _both_lateral and _len_similar:
                                _gate_passed[_bi_f] = True
                                print(
                                    f"[RenalTag] Pair-bypass B: Branch {_bi_f} reinstated "
                                    f"(contralateral to passed Branch {_bi_p}, "
                                    f"opposite sides, dZ={abs(_gp['ostium_z']-_gf['ostium_z']):.1f}mm, "
                                    f"lat={_gf['lat_mm']:.1f}mm, "
                                    f"len_ratio={max(_gp['blen'],_gf['blen'])/max(min(_gp['blen'],_gf['blen']),1e-6):.2f})"
                                )

                    _gated_bis = [_bi for _bi, _ok in _gate_passed.items() if _ok]

                    # 2b. Pair symmetry detection.
                    #   Renal veins typically appear as a bilateral pair:
                    #   • opposite lateral sides (tip_x_sign differs)
                    #   • ostia at similar Z levels (|ΔZ| < 25 mm)
                    _PAIR_Z_DIFF_MM = 25.0
                    _renal_pairs = []  # list of frozensets {bi_a, bi_b}
                    for _i, _bi_a in enumerate(_gated_bis):
                        for _bi_b in _gated_bis[_i + 1 :]:
                            _ga = _renal_candidates[_bi_a]
                            _gb = _renal_candidates[_bi_b]
                            _opposite_side = (
                                _ga["tip_x_sign"] != _gb["tip_x_sign"]
                                and _ga["tip_x_sign"] != 0
                                and _gb["tip_x_sign"] != 0
                            )
                            _z_close = (
                                abs(_ga["ostium_z"] - _gb["ostium_z"]) < _PAIR_Z_DIFF_MM
                            )
                            if _opposite_side and _z_close:
                                _renal_pairs.append(frozenset({_bi_a, _bi_b}))
                                print(
                                    f"[RenalTag] Pair detected: Branch {_bi_a} ↔ Branch {_bi_b} "
                                    f"(opposite sides, ΔZ={abs(_ga['ostium_z']-_gb['ostium_z']):.1f}mm)"
                                )

                    _paired_bis = set()
                    for _p in _renal_pairs:
                        _paired_bis |= _p

                    # Check pair length symmetry (symmetric pair = stronger signal).
                    def _is_symmetric_pair(bi_a, bi_b):
                        _la = _renal_candidates[bi_a]["blen"]
                        _lb = _renal_candidates[bi_b]["blen"]
                        _lr = max(_la, _lb) / max(min(_la, _lb), 1e-6)
                        return _lr < 2.0  # within 2:1 length ratio

                    # 2c. Dominance detection.
                    #   If one gated branch scores much higher than all others, it is
                    #   the dominant candidate and gets a bonus; weak companions get
                    #   a mild penalty to prevent drag-along classification.
                    def _raw_score_no_topo(bi):
                        """Quick score without topology (used for dominance ranking)."""
                        g = _renal_candidates[bi]
                        sc, *_ = _renal_composite_score(
                            g["diam_ratio"],
                            g["lat_mm"],
                            g["ang"],
                            g["ostium_z"],
                            _bif_z_ref,
                            g["prox_d"],
                            g["dist_d"],
                            topology_score=0.55,
                        )  # neutral prior
                        return sc

                    _raw_scores = {_bi: _raw_score_no_topo(_bi) for _bi in _gated_bis}
                    _DOMINANT_MARGIN = 0.20  # score gap to call one branch dominant
                    _dominant_bi = None
                    if len(_raw_scores) >= 2:
                        _sorted_bi = sorted(
                            _raw_scores, key=lambda b: _raw_scores[b], reverse=True
                        )
                        if (
                            _raw_scores[_sorted_bi[0]] - _raw_scores[_sorted_bi[1]]
                            >= _DOMINANT_MARGIN
                        ):
                            _dominant_bi = _sorted_bi[0]
                            print(
                                f"[RenalTag] Dominant candidate: Branch {_dominant_bi} "
                                f"(margin={_raw_scores[_sorted_bi[0]]-_raw_scores[_sorted_bi[1]]:.3f})"
                            )
                    elif len(_raw_scores) == 1:
                        _dominant_bi = next(iter(_raw_scores))

                    # 2d. Build per-branch topology score.
                    def _build_topology_score(bi):
                        topo = 0.55  # neutral prior

                        # Pair membership (strongest positive signal)
                        _in_any_pair = any(bi in _p for _p in _renal_pairs)
                        if _in_any_pair:
                            topo += 0.30
                            # Symmetric pair (similar lengths) = additional bonus
                            for _p in _renal_pairs:
                                if bi in _p:
                                    _other = next(x for x in _p if x != bi)
                                    if _is_symmetric_pair(bi, _other):
                                        topo += 0.15
                                    break
                        elif _renal_candidates[bi].get("length_flagged"):
                            # Long branch with no symmetric partner → almost certainly
                            # a trunk continuation, not a renal vein.
                            topo -= 0.40
                            print(
                                f"[RenalTag] Branch {bi}: length_flagged + unpaired"
                                f" → topo penalty -0.40"
                            )

                        # Dominance bonus
                        if _dominant_bi is not None and bi == _dominant_bi:
                            topo += 0.15

                        # Terminal alignment: smooth penalty/bonus on axial vs lateral
                        # distal direction.  Iliac continuations stay axial (align≈1);
                        # renal veins turn lateral before terminating (align≈0).
                        # Scaled linearly around the 0.5 neutral point; clamped to
                        # avoid over-weighting a single noisy direction vector.
                        _ta = _renal_candidates[bi].get("terminal_align", 0.5)
                        _ta_mod = (_ta - 0.5) * 0.6   # maps [0,1] → [-0.30, +0.30]
                        topo -= _clamp(_ta_mod, -0.25, 0.25)
                        if abs(_ta_mod) > 0.05:        # only log when signal is non-trivial
                            print(
                                f"[RenalTag] Branch {bi}: terminal_align={_ta:.2f}"
                                f" → topo_mod={-_clamp(_ta_mod,-0.25,0.25):+.2f}"
                            )

                        # Unstable trunk connection penalty
                        _snap = _renal_candidates[bi]["min_trunk_dist"]
                        if _snap > 5.0:
                            topo -= _clamp((_snap - 5.0) / 10.0, 0.0, 0.40)

                        return _clamp(topo, 0.0, 1.0)

                    # ── Pass 3: final composite score + classification ─────────────
                    for bi in _renal_candidates:
                        _geo = _renal_candidates[bi]

                        # Hard gate — gate failures become side_branch immediately.
                        if not _gate_passed[bi]:
                            self.branchMeta[bi]["renal_score"] = 0.0
                            continue  # role stays 'side'

                        _topo_sc = _build_topology_score(bi)

                        (
                            _rscore,
                            _diam_sc,
                            _lat_sc,
                            _z_sc,
                            _angle_sc,
                            _topo_sc_out,
                            _stab_pen,
                        ) = _renal_composite_score(
                            diam_ratio=_geo["diam_ratio"],
                            lat_mm=_geo["lat_mm"],
                            angle_deg=_geo["ang"],
                            ostium_z=_geo["ostium_z"],
                            bif_z=_bif_z_ref,
                            prox_diam=_geo["prox_d"],
                            dist_diam=_geo["dist_d"],
                            topology_score=_topo_sc,
                        )

                        # Legacy flags (kept for log readability)
                        _diam_ok = _geo["diam_ratio"] >= _RENAL_DIAM_RATIO
                        _angle_ok = _geo["ang"] >= _RENAL_ANGLE_DEG

                        print(
                            f"[RenalTag] Branch {bi}: len={_geo['blen']:.1f}mm "
                            f"diam={_geo['bdiam']:.1f}mm diam_ratio={_geo['diam_ratio']:.2f} "
                            f"lat={_geo['lat_mm']:.1f}mm angle={_geo['ang']:.1f}° "
                            f"trunk_snap={_geo['min_trunk_dist']:.1f}mm "
                            f"score={_rscore:.3f} "
                            f"(diam_sc={_diam_sc:.2f} lat_sc={_lat_sc:.2f} "
                            f"z_sc={_z_sc:.2f} angle_sc={_angle_sc:.2f} "
                            f"topo_sc={_topo_sc_out:.2f} stab_pen={_stab_pen:.2f})"
                        )

                        if _rscore >= _RENAL_SCORE_THRESH:
                            self.branchMeta[bi]["role"] = "renal_vein"
                            self.branchMeta[bi]["renal_score"] = round(_rscore, 3)
                            # Assign lateral_label now using tip X (LPS: larger X = Right)
                            _rbs, _rbe = self.branches[bi]
                            if _rbe > _rbs and _rbe - 1 < len(rawPoints):
                                _rtip_x = rawPoints[_rbe - 1][0]
                                # In RAS: positive X = anatomical Right, negative X = Left
                                _rlat = "Right" if _rtip_x > 0 else "Left"
                                self.branchMeta[bi]["lateral_label"] = _rlat
                            _reasons = []
                            if _diam_ok:
                                _reasons.append(
                                    f"diam_ratio={_geo['diam_ratio']:.2f}>={_RENAL_DIAM_RATIO}"
                                )
                            if _angle_ok:
                                _reasons.append(
                                    f"angle={_geo['ang']:.1f}°>={_RENAL_ANGLE_DEG}°"
                                )
                            _reasons.append(
                                f"composite={_rscore:.3f}>={_RENAL_SCORE_THRESH}"
                            )
                            print(
                                f"[RenalTag] Branch {bi}: → renal_vein ({', '.join(_reasons)})"
                            )
                        else:
                            self.branchMeta[bi]["renal_score"] = round(_rscore, 3)
                            print(
                                f"[RenalTag] Branch {bi}: → side branch "
                                f"(composite={_rscore:.3f} < {_RENAL_SCORE_THRESH})"
                            )

                    # ── step 8c-pre: renal branch consolidation ──────────────────
                    #
                    # Over-segmentation at shallow-angle renal junctions causes one
                    # anatomical renal vein to appear as two separate branches in the
                    # graph.  Signature: two renal_vein branches whose origin points
                    # are very close in 3D space AND on the same lateral side (same X
                    # sign relative to trunk centerline).
                    #
                    # Detection criteria (all must pass):
                    #   1. Both branches are role == 'renal_vein'
                    #   2. Ostium-to-ostium 3D distance < MERGE_DIST_MM (25mm)
                    #   3. Same lateral side: both tips have same X sign
                    #   4. Angle between their departure vectors < MERGE_ANGLE_DEG (50°)
                    #
                    # Resolution: mark the shorter branch as 'renal_fragment'
                    # (excluded from display, report, and stent logic) and store a
                    # reference to the primary in its branchMeta.  The primary branch
                    # absorbs the shorter into its 'renal_group' list.
                    # The underlying self.branches and gi ranges are NOT modified —
                    # only the role tag changes.
                    #
                    MERGE_DIST_MM = 25.0
                    MERGE_ANGLE_DEG = 50.0

                    _renal_bis = [
                        bi
                        for bi in range(len(self.branches))
                        if self.branchMeta.get(bi, {}).get("role") == "renal_vein"
                    ]

                    if len(_renal_bis) >= 2:
                        # Build (bi, ostium_pt, tip_pt, departure_vec) for each renal
                        _renal_info = {}
                        for _rbi in _renal_bis:
                            _rs, _re = self.branches[_rbi]
                            _ogi = self.branchMeta.get(_rbi, {}).get("ostiumGi", _rs)
                            _ogi = max(_rs, min(_re - 1, _ogi))
                            _ost = (
                                tuple(self.points[_ogi])
                                if _ogi < len(self.points)
                                else None
                            )
                            _tip = (
                                tuple(self.points[_re - 1])
                                if _re - 1 < len(self.points)
                                else None
                            )
                            # Departure vector: first DEPART_MM arc from ostium
                            DEPART_MM = 15.0
                            _arc_d = 0.0
                            _dep = None
                            for _di in range(_ogi, min(_re - 1, _ogi + 30)):
                                if _di + 1 >= len(self.points):
                                    break
                                _arc_d += (
                                    sum(
                                        (self.points[_di + 1][k] - self.points[_di][k])
                                        ** 2
                                        for k in range(3)
                                    )
                                    ** 0.5
                                )
                                if _arc_d >= DEPART_MM:
                                    _dep = tuple(
                                        self.points[_di + 1][k] - self.points[_ogi][k]
                                        for k in range(3)
                                    )
                                    break
                            if _dep is None and _tip and _ost:
                                _dep = tuple(_tip[k] - _ost[k] for k in range(3))
                            _renal_info[_rbi] = {
                                "ostium": _ost,
                                "tip": _tip,
                                "dep": _dep,
                                "len": (
                                    self.distances[_re - 1] - self.distances[_rs]
                                    if _re - 1 < len(self.distances)
                                    and _rs < len(self.distances)
                                    else 0.0
                                ),
                            }

                        # Pairwise merge check
                        _merged = set()
                        for _i, _rbi_a in enumerate(_renal_bis):
                            for _rbi_b in _renal_bis[_i + 1 :]:
                                if _rbi_a in _merged or _rbi_b in _merged:
                                    continue
                                _ia = _renal_info[_rbi_a]
                                _ib = _renal_info[_rbi_b]
                                if not (_ia["ostium"] and _ib["ostium"]):
                                    continue

                                # Criterion 1: 3D distance between ostia
                                _ost_dist = (
                                    sum(
                                        (_ia["ostium"][k] - _ib["ostium"][k]) ** 2
                                        for k in range(3)
                                    )
                                    ** 0.5
                                )
                                if _ost_dist > MERGE_DIST_MM:
                                    continue

                                # Criterion 2: same lateral side
                                # For short branches (<40mm), the tip endpoint can
                                # wander to the opposite side due to VMTK path routing
                                # through thin/curved geometry.  Use ostium X for
                                # side determination on short branches; tip X for
                                # longer branches where the tip reliably indicates side.
                                # Reference: trunk centerline X interpolated at each
                                # branch's ostium Z (not root X — IVC curves).
                                def _trunk_x_at_z(z_val):
                                    """Trunk centerline X nearest to a given Z."""
                                    _ts, _te = self.branches[0]
                                    _best_x = self.points[_ts][0]
                                    _best_dz = abs(self.points[_ts][2] - z_val)
                                    for _ti in range(_ts, _te):
                                        if _ti >= len(self.points):
                                            break
                                        _dz = abs(self.points[_ti][2] - z_val)
                                        if _dz < _best_dz:
                                            _best_dz = _dz
                                            _best_x = self.points[_ti][0]
                                    return _best_x

                                SHORT_BRANCH_MM = 40.0
                                _ref_a = (
                                    _ia["ostium"]
                                    if _ia["len"] < SHORT_BRANCH_MM
                                    else _ia["tip"]
                                )
                                _ref_b = (
                                    _ib["ostium"]
                                    if _ib["len"] < SHORT_BRANCH_MM
                                    else _ib["tip"]
                                )
                                _tx_a = (
                                    _trunk_x_at_z(_ref_a[2])
                                    if self.branches and _ref_a
                                    else 0.0
                                )
                                _tx_b = (
                                    _trunk_x_at_z(_ref_b[2])
                                    if self.branches and _ref_b
                                    else 0.0
                                )
                                _lat_a = (_ref_a[0] - _tx_a) if _ref_a else 0.0
                                _lat_b = (_ref_b[0] - _tx_b) if _ref_b else 0.0
                                # Ostia are nearly coincident → skip side check when
                                # ostia distance is very small (< 15mm both are same origin)
                                if _ost_dist >= 15.0 and _lat_a * _lat_b < 0:
                                    continue  # genuinely opposite sides

                                # Criterion 3: angle between departure vectors
                                _da = _ia["dep"]
                                _db = _ib["dep"]
                                _ang_ab = 0.0
                                if _da and _db:
                                    _la = sum(x * x for x in _da) ** 0.5
                                    _lb = sum(x * x for x in _db) ** 0.5
                                    if _la > 1e-6 and _lb > 1e-6:
                                        _dot = sum(
                                            _da[k] * _db[k] for k in range(3)
                                        ) / (_la * _lb)
                                        _ang_ab = math.degrees(
                                            math.acos(max(-1.0, min(1.0, _dot)))
                                        )
                                        if _ang_ab > MERGE_ANGLE_DEG:
                                            print(
                                                f"[RenalConsolidate] Skip {_rbi_a}/{_rbi_b}: "
                                                f"ostia={_ost_dist:.1f}mm angle={_ang_ab:.1f}° "
                                                f"(>{MERGE_ANGLE_DEG}°)"
                                            )
                                            continue
                                print(
                                    f"[RenalConsolidate] Candidate pair {_rbi_a}/{_rbi_b}: "
                                    f"ostia={_ost_dist:.1f}mm angle={_ang_ab:.1f}° — merging"
                                )

                                # All criteria passed — consolidate
                                _primary = (
                                    _rbi_a if _ia["len"] >= _ib["len"] else _rbi_b
                                )
                                _fragment = _rbi_b if _primary == _rbi_a else _rbi_a
                                _plen = _renal_info[_primary]["len"]
                                _flen = _renal_info[_fragment]["len"]

                                self.branchMeta.setdefault(_fragment, {})[
                                    "role"
                                ] = "renal_fragment"
                                self.branchMeta.setdefault(_fragment, {})[
                                    "fragment_of"
                                ] = _primary
                                self.branchMeta.setdefault(_primary, {}).setdefault(
                                    "renal_group", []
                                ).append(_fragment)
                                _merged.add(_fragment)

                                print(
                                    f"[RenalConsolidate] Branch {_fragment} "
                                    f"({_flen:.1f}mm) → fragment of Branch {_primary} "
                                    f"({_plen:.1f}mm); "
                                    f"ostia_dist={_ost_dist:.1f}mm "
                                    f"angle={_ang_ab:.1f}°"
                                )

                    # ── SideBranch→ReanalFragment consolidation ────────────────
                    # Second pass: find side_branch stubs that are geographically
                    # adjacent to a renal_vein but failed the len>=40mm gate so never
                    # got tagged renal_vein.  These are VMTK over-segmentation artifacts
                    # at shallow-angle renal junctions (e.g. Branch5 = 22mm horizontal
                    # entry stub of the same vessel as Branch4 = 74mm renal_vein).
                    #
                    # Criteria (all must pass):
                    #   1. Branch is role == 'side_branch' (not already renal/fragment)
                    #   2. Its ostium is within SIDE_MERGE_DIST_MM of a renal_vein ostium
                    #   3. Departure angle between them < SIDE_MERGE_ANG_DEG
                    #   4. Branch arc < SIDE_MAX_LEN_MM (short stubs only)
                    SIDE_MERGE_DIST_MM = 20.0  # ostium proximity gate
                    SIDE_MERGE_ANG_DEG = 65.0  # departure angle gate
                    SIDE_MAX_LEN_MM = 35.0  # only short stubs qualify

                    _renal_bis_now = [
                        bi
                        for bi in range(len(self.branches))
                        if self.branchMeta.get(bi, {}).get("role") == "renal_vein"
                    ]
                    _side_bis = [
                        bi
                        for bi in range(len(self.branches))
                        if (
                            self.branchMeta.get(bi, {}).get("role") == "side_branch"
                            and bi != 0
                        )
                    ]

                    for _sbi in _side_bis:
                        _smeta = self.branchMeta.get(_sbi, {})
                        _ss, _se = self.branches[_sbi]
                        _s_arc = (
                            self.distances[_se - 1] - self.distances[_ss]
                            if _se - 1 < len(self.distances)
                            and _ss < len(self.distances)
                            else 999.0
                        )
                        if _s_arc > SIDE_MAX_LEN_MM:
                            continue
                        _s_ogi = _smeta.get("ostiumGi", _ss)
                        _s_ogi = max(_ss, min(_se - 1, _s_ogi))
                        if _s_ogi >= len(self.points):
                            continue
                        _s_ost = self.points[_s_ogi]

                        # Departure vector of side branch (first 10mm arc)
                        _s_dep = None
                        _s_arc_d = 0.0
                        for _sdi in range(_s_ogi, min(_se - 1, _s_ogi + 20)):
                            if _sdi + 1 >= len(self.points):
                                break
                            _s_arc_d += (
                                sum(
                                    (self.points[_sdi + 1][k] - self.points[_sdi][k])
                                    ** 2
                                    for k in range(3)
                                )
                                ** 0.5
                            )
                            if _s_arc_d >= 10.0:
                                _s_dep = tuple(
                                    self.points[_sdi + 1][k] - _s_ost[k]
                                    for k in range(3)
                                )
                                break
                        if _s_dep is None and _se - 1 < len(self.points):
                            _s_dep = tuple(
                                self.points[_se - 1][k] - _s_ost[k] for k in range(3)
                            )

                        _best_renal = None
                        _best_dist = SIDE_MERGE_DIST_MM + 1.0

                        for _rbi2 in _renal_bis_now:
                            _r_ogi2 = self.branchMeta.get(_rbi2, {}).get(
                                "ostiumGi", self.branches[_rbi2][0]
                            )
                            if _r_ogi2 >= len(self.points):
                                continue
                            _r_ost2 = self.points[_r_ogi2]
                            _dist2 = (
                                sum((_s_ost[k] - _r_ost2[k]) ** 2 for k in range(3))
                                ** 0.5
                            )
                            if _dist2 > SIDE_MERGE_DIST_MM:
                                continue

                            # Check departure angle
                            _rs2, _re2 = self.branches[_rbi2]
                            _r_dep2 = None
                            _r_arc_d2 = 0.0
                            for _rdi2 in range(_r_ogi2, min(_re2 - 1, _r_ogi2 + 20)):
                                if _rdi2 + 1 >= len(self.points):
                                    break
                                _r_arc_d2 += (
                                    sum(
                                        (
                                            self.points[_rdi2 + 1][k]
                                            - self.points[_rdi2][k]
                                        )
                                        ** 2
                                        for k in range(3)
                                    )
                                    ** 0.5
                                )
                                if _r_arc_d2 >= 10.0:
                                    _r_dep2 = tuple(
                                        self.points[_rdi2 + 1][k] - _r_ost2[k]
                                        for k in range(3)
                                    )
                                    break
                            if _r_dep2 is None and _re2 - 1 < len(self.points):
                                _r_dep2 = tuple(
                                    self.points[_re2 - 1][k] - _r_ost2[k]
                                    for k in range(3)
                                )
                            if _s_dep and _r_dep2:
                                _la = sum(x * x for x in _s_dep) ** 0.5
                                _lb = sum(x * x for x in _r_dep2) ** 0.5
                                if _la > 1e-6 and _lb > 1e-6:
                                    _dot2 = sum(
                                        _s_dep[k] * _r_dep2[k] for k in range(3)
                                    ) / (_la * _lb)
                                    _ang2 = math.degrees(
                                        math.acos(max(-1.0, min(1.0, _dot2)))
                                    )
                                    if _ang2 > SIDE_MERGE_ANG_DEG:
                                        continue

                            if _dist2 < _best_dist:
                                _best_dist = _dist2
                                _best_renal = _rbi2

                        if _best_renal is not None:
                            _rlen = (
                                self.distances[self.branches[_best_renal][1] - 1]
                                - self.distances[self.branches[_best_renal][0]]
                                if (
                                    self.branches[_best_renal][1] - 1
                                    < len(self.distances)
                                    and self.branches[_best_renal][0]
                                    < len(self.distances)
                                )
                                else 0.0
                            )
                            _smeta["role"] = "renal_fragment"
                            _smeta["fragment_of"] = _best_renal
                            self.branchMeta.setdefault(_best_renal, {}).setdefault(
                                "renal_group", []
                            ).append(_sbi)
                            print(
                                f"[RenalConsolidate] SideFrag Branch {_sbi} "
                                f"({_s_arc:.1f}mm) → fragment of renal Branch {_best_renal} "
                                f"({_rlen:.1f}mm); ostia_dist={_best_dist:.1f}mm"
                            )

                    # ── step 8c: bifurcation snapping ────────────────────────────
                    #
                    # The graph topology node (trueBifNode / _trunkBifPos) is already
                    # correct in connectivity and close in space — it is the grid-cell
                    # centroid after snap+collapse, typically 1–5 mm from the true
                    # anatomical split.
                    #
                    # All previous composite-score approaches failed because:
                    #   • from any trunk point within 25mm of the bif, the 20mm
                    #     branch lookahead hits nearly the same territory — rawDivg
                    #     is flat and rawGrad ≈ 0, so the scorer has no signal.
                    #   • allowing the winner to move into branch midline territory
                    #     caused 35mm overcorrection (v130).
                    #   • the trunk list orientation is not guaranteed, causing the
                    #     walker to anchor at the wrong end (v131/v132).
                    #
                    # Correct approach — two steps:
                    #
                    #  1. NEAREST-POINT SNAP: find the trunk centerline point
                    #     closest to _trunkBifPos.  This is the primary snapped
                    #     position.  It corrects the centroid offset without risk
                    #     of drifting upstream or downstream.
                    #
                    #  2. DISTAL NUDGE (optional, capped):
                    #     Walk distally from the snapped point up to NUDGE_MAX_MM
                    #     (10 mm) collecting inter-branch divergence.  Stop at the
                    #     first point where divg >= MIN_DIVG_MM OR we reach
                    #     NUDGE_MAX_MM.  This nudges the point to where branches
                    #     have genuinely separated without risking the point passing
                    #     the ostia (which are ~20mm away).
                    #
                    # Falls back to nodePos[trueBifNode] if trunk is empty.

                    NUDGE_MAX_MM = 10.0  # max distal nudge from nearest-point snap
                    MIN_DIVG_MM = 5.0  # target inter-branch separation

                    mainBranches = [
                        bi
                        for bi in range(len(self.branches))
                        if self.branchMeta[bi]["role"] == "main"
                    ]

                    snappedBifPt = (
                        nodePos[trueBifNode]
                        if trueBifNode is not None
                        else (0.0, 0.0, 0.0)
                    )

                    if trueBifNode is not None and len(mainBranches) >= 2:
                        brA_gb = mainBranches[0]
                        brB_gb = mainBranches[1]
                        trunkPtsFull = graphBranches[0]

                        # ── Step 1: nearest-point snap ────────────────────────────
                        # Anchor = _trunkBifPos (topology-confirmed bif end of trunk)
                        _bifAnchor = (
                            _trunkBifPos
                            if "_trunkBifPos" in dir() and _trunkBifPos is not None
                            else nodePos[trueBifNode]
                        )
                        _snapIdx = 0
                        _snapDist = 1e18
                        for _ti, _tp in enumerate(trunkPtsFull):
                            _d = _dist3(_tp, _bifAnchor)
                            if _d < _snapDist:
                                _snapDist = _d
                                _snapIdx = _ti

                        # ── Step 2: determine distal direction from snap point ────
                        # Distal = toward _trunkBifPos (away from _trunkRootPos).
                        # The trunk list may be oriented either way, so detect by
                        # comparing distances of neighbouring indices to _bifAnchor.
                        _rootAnchor = (
                            _trunkRootPos
                            if "_trunkRootPos" in dir() and _trunkRootPos is not None
                            else None
                        )
                        if _rootAnchor is not None:
                            _d_prev = (
                                _dist3(trunkPtsFull[_snapIdx - 1], _rootAnchor)
                                if _snapIdx > 0
                                else 1e18
                            )
                            _d_next = (
                                _dist3(trunkPtsFull[_snapIdx + 1], _rootAnchor)
                                if _snapIdx < len(trunkPtsFull) - 1
                                else 1e18
                            )
                            # Distal = step that moves AWAY from root (larger dist to root)
                            _distalStep = 1 if _d_next > _d_prev else -1
                        else:
                            # Fallback: distal = toward the end of the list that is
                            # closer to _bifAnchor
                            _d_start = _dist3(trunkPtsFull[0], _bifAnchor)
                            _d_end = _dist3(trunkPtsFull[-1], _bifAnchor)
                            _distalStep = 1 if _d_end < _d_start else -1

                        # ── helper: inter-branch divergence at a trunk point ──────
                        def _branchDivgAt(nearPt, lookMm):
                            """Distance between branch A and branch B, each sampled
                            at lookMm downstream of nearPt."""

                            def _sample(brIdx):
                                if brIdx >= len(graphBranches):
                                    return nearPt
                                bPts = graphBranches[brIdx]
                                if not bPts:
                                    return nearPt
                                best_i = 0
                                best_d = 1e18
                                for i, bp in enumerate(bPts):
                                    d = _dist3(bp, nearPt)
                                    if d < best_d:
                                        best_d = d
                                        best_i = i
                                acc = 0.0
                                for i in range(best_i, len(bPts) - 1):
                                    acc += _dist3(bPts[i], bPts[i + 1])
                                    if acc >= lookMm:
                                        return bPts[i + 1]
                                return bPts[-1]

                            return _dist3(_sample(brA_gb), _sample(brB_gb))

                        # ── Step 2: distal nudge up to NUDGE_MAX_MM ───────────────
                        _nudgeIdx = _snapIdx
                        _nudgeAcc = 0.0
                        _nudged = False
                        _snapDivg = _branchDivgAt(trunkPtsFull[_snapIdx], 20.0)
                        _curDivg = _snapDivg
                        if _snapDivg < MIN_DIVG_MM:
                            _ni = _snapIdx
                            while True:
                                _ni += _distalStep
                                if _ni < 0 or _ni >= len(trunkPtsFull):
                                    break
                                _nudgeAcc += _dist3(
                                    trunkPtsFull[_ni - _distalStep], trunkPtsFull[_ni]
                                )
                                if _nudgeAcc > NUDGE_MAX_MM:
                                    break
                                _d = _branchDivgAt(trunkPtsFull[_ni], 20.0)
                                if _d >= MIN_DIVG_MM:
                                    _nudgeIdx = _ni
                                    _curDivg = _d
                                    _nudged = True
                                    break

                        snappedBifPt = trunkPtsFull[_nudgeIdx]
                        tag = f" nudged+{_nudgeAcc:.1f}mm" if _nudged else ""
                        print(
                            f"[VesselAnalyzer] Bifurcation snapped: "
                            f"({snappedBifPt[0]:.1f},{snappedBifPt[1]:.1f},"
                            f"{snappedBifPt[2]:.1f}), "
                            f"snap_dist={_snapDist:.1f}mm "
                            f"divg={_curDivg:.1f}mm{tag}"
                        )

                    # Store snapped position in branchMeta for downstream use
                    self.bifurcationPoint = snappedBifPt

                    # ── step 8d: ostium detection — branch-separation algorithm ──
                    #
                    # Based on findTrueBranchStart: compare each branch against all
                    # sibling branches at the same step index.  The ostium is the first
                    # point where THIS branch has diverged far enough from EVERY sibling.
                    #
                    # Two-pass algorithm:
                    #
                    # Pass 1 — spatial separation:
                    #   Skip first OSTIUM_SKIP_PTS points (noisy junction zone).
                    #   At each subsequent step k, compute the minimum distance from
                    #   branch[i][k] to the corresponding point on every other branch:
                    #     sep = min over siblings j of dist(branch_i[k], branch_j[k])
                    #   Ostium = first k where sep > SEP_THRESH for 2 consecutive steps.
                    #
                    # Pass 2 — radius stabilization (refinement):
                    #   From the pass-1 ostium forward, walk up to 5 more points and
                    #   find the first where the local diameter has stabilised to within
                    #   RADIUS_STABLE_FRAC of the at-bifurcation diameter.  This pushes
                    #   the ostium past the funnel/flare zone at the split.
                    #
                    # Tuning parameters (large-vessel iliac anatomy):
                    #   SEP_THRESH_MM    2.5 mm   — increase for wide vessels
                    #   OSTIUM_SKIP_PTS  3        — skip noisy junction region
                    #   RADIUS_STABLE    0.15     — 15% diameter change = unstable

                    # Separation threshold: detect ONSET of divergence, not full
                    # independence.  At a Y-bifurcation two iliac branches share a
                    # common wall funnel — they never achieve "half a diameter" of
                    # daylight at the true ostium.  Use a small absolute floor and a
                    # tiny fractional threshold so the first point where the branches
                    # start pulling apart is captured.
                    #
                    # SEP_FRAC × local_diam or SEP_MIN_MM, whichever is larger.
                    # Reduced from 0.4/3.0 → 0.08/1.5 to catch early divergence.
                    # OSTIUM_MAX_STEPS reduced from 40 → 15: the true ostium of an
                    # iliac Y-bif is always within the first 10–15 centerline points
                    # (~10–20 mm).  Walking 40 pts was the source of the 43mm error.
                    SEP_FRAC = 0.08  # 8% of local diameter (~1mm for 13mm vessel)
                    SEP_MIN_MM = 1.5  # absolute floor (mm)
                    OSTIUM_SKIP_PTS = 1  # skip only 1 point (the shared bif node)
                    RADIUS_STABLE_FRAC = 0.15  # radius stabilisation tolerance
                    OSTIUM_MAX_STEPS = 15  # hard cap: true ostium is within 15 pts

                    def _localDiam(pt):
                        """Local vessel diameter at pt via surface locator (closest-face)."""
                        if _eSurfLoc is None:
                            return 8.0  # anatomical fallback for no-model case
                        cp = [0.0, 0.0, 0.0]
                        ci = vtk.reference(0)
                        si = vtk.reference(0)
                        d2 = vtk.reference(0.0)
                        _eSurfLoc.FindClosestPoint(list(pt), cp, ci, si, d2)
                        return max(math.sqrt(float(d2)) * 2.0, 1.0)

                    # Collect sibling branch point lists for cross-comparison.
                    # Include ALL non-trunk branches so side branches also get an
                    # ostiumGi — this lets getBranchStartGi work correctly for them.
                    _allNonTrunkBis = list(range(1, len(self.branches)))
                    _siblingPts = {
                        bi: (graphBranches[bi] if bi < len(graphBranches) else [])
                        for bi in _allNonTrunkBis
                    }

                    # Pass 1 fix: only compare against the PROXIMAL portion of each
                    # sibling (first PROX_COMPARE_PTS points), so distal segments of
                    # long branches don't prevent the threshold from ever being met.
                    # Also only compare against branches that START near the same
                    # bifurcation node (within BIF_PROXIMITY_MM of snappedBifPt).
                    PROX_COMPARE_PTS = 25  # only scan this many proximal sibling pts
                    BIF_PROXIMITY_MM = (
                        60.0  # sib must start within this of bif to qualify
                    )

                    def _nearestDistToProximalBranch(branchIdx, queryPt):
                        """Distance from queryPt to the proximal PROX_COMPARE_PTS pts of branch."""
                        if branchIdx >= len(graphBranches):
                            return 1e18
                        bPtsS = graphBranches[branchIdx]
                        # Check that this sibling starts near the bifurcation
                        if not bPtsS:
                            return 1e18
                        d_start = _dist3(bPtsS[0], snappedBifPt)
                        if d_start > BIF_PROXIMITY_MM:
                            return 1e18  # different bifurcation — skip
                        best = 1e18
                        for p in bPtsS[:PROX_COMPARE_PTS]:
                            d = _dist3(p, queryPt)
                            if d < best:
                                best = d
                        return best

                    for bi in _allNonTrunkBis:
                        bPts = _siblingPts[bi]
                        ostiumPt = bPts[0] if bPts else snappedBifPt
                        found_k = OSTIUM_SKIP_PTS  # default fallback step

                        # ── Main-branch shortcut ──────────────────────────────────
                        # [IliacOstium] Debug: log every branch entry so we can see
                        # whether the right iliac reaches the dome-walk path.
                        _is_main_branch = bi in mainSet
                        _role_now = self.branchMeta.get(bi, {}).get("role", "?")
                        _bPts0 = bPts[0] if bPts else None
                        print(
                            f"[IliacOstium] Loop bi={bi} role={_role_now} "
                            f"in_mainSet={_is_main_branch} "
                            f"mainSet={sorted(mainSet)} "
                            f"bPts_len={len(bPts)} "
                            f"bPts[0]=({_bPts0[0]:.1f},{_bPts0[1]:.1f},{_bPts0[2]:.1f})"
                            f" snappedBif=({snappedBifPt[0]:.1f},{snappedBifPt[1]:.1f},{snappedBifPt[2]:.1f})"
                            if _bPts0
                            else f"[IliacOstium] Loop bi={bi} role={_role_now} in_mainSet={_is_main_branch} bPts_len=0"
                        )
                        if _is_main_branch and bPts:
                            BIF_DOME_MM = 9.0  # walk until this far from snappedBifPt
                            DIVERGE_ARC_MM = 25.0  # hard arc cap (safety net)

                            # Pass A: first_k = first point >= BIF_DOME_MM from bif.
                            first_k = 0
                            _arc = 0.0
                            for _ki in range(1, len(bPts)):
                                if _ki > 1:
                                    _arc += _dist3(bPts[_ki - 1], bPts[_ki])
                                if _arc > DIVERGE_ARC_MM:
                                    first_k = _ki
                                    _d_bif = _dist3(bPts[_ki], snappedBifPt)
                                    print(
                                        f"[IliacOstium] bi={bi} PassA: arc cap at ki={_ki} "
                                        f"arc={_arc:.1f}mm d_bif={_d_bif:.1f}mm"
                                    )
                                    break
                                _d_bif = _dist3(bPts[_ki], snappedBifPt)
                                if _d_bif >= BIF_DOME_MM:
                                    first_k = _ki
                                    print(
                                        f"[IliacOstium] bi={bi} PassA: first_k={_ki} "
                                        f"d_bif={_d_bif:.1f}mm >= {BIF_DOME_MM}mm "
                                        f"pt=({bPts[_ki][0]:.1f},{bPts[_ki][1]:.1f},{bPts[_ki][2]:.1f})"
                                    )
                                    break

                            # Pass B: find display_k = point with MAXIMUM Z (most cranial)
                            # within the arc cap.  This is the true anatomical apex of the
                            # iliac dome — the point the navigator should anchor to.
                            #
                            # Why not max 3D distance from bif?  The iliac CL curves cranially
                            # then swings laterally and descends, so a later point can be
                            # further in 3D from bif (e.g. 23mm at ki=21, Z=1776) while being
                            # LOWER in Z than the true apex (ki=7, Z=1779).  Max-3D picks the
                            # wrong point.  Max-Z always selects the highest point on the dome
                            # regardless of lateral drift.
                            #
                            # Guard: only consider points that are already >= BIF_DOME_MM from
                            # bif (same gate as PassA) so we stay past the shared dome zone.
                            _best_dome_k = first_k
                            _best_dome_z = bPts[first_k][2]  # Z of first-crossing
                            _arc2 = 0.0
                            for _ki in range(1, len(bPts)):
                                if _ki > 1:
                                    _arc2 += _dist3(bPts[_ki - 1], bPts[_ki])
                                if _arc2 > DIVERGE_ARC_MM:
                                    break
                                _d_bif = _dist3(bPts[_ki], snappedBifPt)
                                _z_ki = bPts[_ki][2]
                                if _d_bif >= BIF_DOME_MM and _z_ki > _best_dome_z:
                                    _best_dome_z = _z_ki
                                    _best_dome_k = _ki

                            # ── CaudalDepartureSnap ───────────────────────────────
                            # If the PassB max-Z winner is BELOW snappedBifPt Z, the
                            # branch departs caudally without a shared cranial dome
                            # (e.g. Right Iliac going immediately downward from the
                            # carina).  The branch CL has no cranial apex to walk to.
                            #
                            # Fix (v248-SiblingZOstium): the anatomical ostium Z of a
                            # caudally-departing iliac equals the shared dome apex Z,
                            # which was already found by the sibling iliac's PassB
                            # max-Z walk (bi=1 → Z=1780.5).  Find the trunk CL point
                            # whose Z is closest to the sibling's committed ostium Z;
                            # that trunk point is the ostium rim on this side.
                            #
                            # ostiumGi stays at bs_gi (no branch-start trim).
                            # surf_classify_ogi_local = first_k (Voronoi floor).
                            # ostiumPt (display only) = trunk point at sibling ostium Z.
                            #
                            # Threshold: if best_dome_Z < bif_Z - CAUDAL_THRESH, the
                            # branch departs caudally.  2mm tolerance handles jitter.
                            _CAUDAL_THRESH_MM = 2.0
                            _bif_z = snappedBifPt[2]
                            if _best_dome_z < _bif_z - _CAUDAL_THRESH_MM:
                                # Pull sibling's committed ostium Z from branchMeta.
                                _sib_z = None
                                for _sbi in mainSet:
                                    if _sbi == bi:
                                        continue
                                    _sib_meta = self.branchMeta.get(_sbi, {})
                                    _sib_ostium = _sib_meta.get("ostium")
                                    if _sib_ostium is not None:
                                        _sib_z = _sib_ostium[2]
                                        break

                                # Find trunk CL point with Z closest to sibling's Z.
                                # Search the last 50 points of trunkPtsFull (bif at [-1]).
                                _target_z = _sib_z if _sib_z is not None else _bif_z
                                _tpts = trunkPtsFull
                                _trunk_cand = _tpts[-1]  # fallback = bif node
                                _trunk_dz = abs(_tpts[-1][2] - _target_z)
                                for _ti in range(
                                    len(_tpts) - 2, max(len(_tpts) - 50, -1), -1
                                ):
                                    _dz = abs(_tpts[_ti][2] - _target_z)
                                    if _dz < _trunk_dz:
                                        _trunk_dz = _dz
                                        _trunk_cand = _tpts[_ti]

                                _ostium_display_pt = _trunk_cand
                                print(
                                    f"[IliacOstium] bi={bi} CaudalDepartureSnap: "
                                    f"PassB max_Z={_best_dome_z:.1f} < bif_Z={_bif_z:.1f} - {_CAUDAL_THRESH_MM}mm "
                                    f"→ sib_Z={_target_z:.1f} trunk-Z ostium at "
                                    f"({_trunk_cand[0]:.1f},{_trunk_cand[1]:.1f},{_trunk_cand[2]:.1f}) "
                                    f"dz={_trunk_dz:.1f}mm"
                                )

                                display_k = 0  # ostiumGi stays at bs_gi (branch start)
                                # Surface classification must still start PAST the shared
                                # dome zone — use first_k (the BIF_DOME_MM crossing) as
                                # the Voronoi competition floor, not ki=0 (the bif node).
                                # This prevents bi=2 from claiming Left Iliac dome verts.
                                self.branchMeta[bi]["surf_classify_ogi_local"] = first_k
                                # Override ostiumPt for display with the trunk-Z point.
                                self.branchMeta[bi][
                                    "_causal_departure_ostium_pt"
                                ] = _ostium_display_pt
                            else:
                                display_k = _best_dome_k
                                _ostium_display_pt = (
                                    None  # normal path — uses bPts[display_k]
                                )

                            _best_d = _dist3(bPts[display_k], snappedBifPt)
                            print(
                                f"[IliacOstium] bi={bi} PassB: "
                                f"first_k={first_k} "
                                f"first_k_pt=({bPts[first_k][0]:.1f},{bPts[first_k][1]:.1f},{bPts[first_k][2]:.1f}) "
                                f"display_k={display_k} "
                                f"display_k_pt=({bPts[display_k][0]:.1f},{bPts[display_k][1]:.1f},{bPts[display_k][2]:.1f}) "
                                f"max_Z={_best_dome_z:.1f} d_bif={_best_d:.1f}mm"
                            )

                            _best_d = _dist3(bPts[display_k], snappedBifPt)
                            # CaudalDepartureSnap: use trunk-rim point for display if found
                            _cds_override = self.branchMeta[bi].get(
                                "_causal_departure_ostium_pt"
                            )
                            ostiumPt = (
                                _cds_override
                                if _cds_override is not None
                                else bPts[display_k]
                            )
                            bs_gi = self.branches[bi][0]
                            be_gi = self.branches[bi][1]
                            ostium_gi = min(bs_gi + display_k, be_gi - 1)
                            self.branchMeta[bi]["ostiumGi"] = ostium_gi
                            self.branchMeta[bi]["stableStartGi"] = ostium_gi

                            self.branchMeta[bi]["ostium_p2"] = snappedBifPt
                            self.branchMeta[bi]["ostium_p3"] = tuple(
                                float(x) for x in ostiumPt
                            )
                            self.branchMeta[bi]["ostium"] = tuple(
                                float(x) for x in ostiumPt
                            )
                            od = _dist3(ostiumPt, snappedBifPt)
                            print(
                                f"[IliacOstium] bi={bi} COMMITTED: "
                                f"ostiumGi={ostium_gi} bs_gi={bs_gi} be_gi={be_gi} "
                                f"ostiumPt=({ostiumPt[0]:.1f},{ostiumPt[1]:.1f},{ostiumPt[2]:.1f}) "
                                f"d_bif={od:.1f}mm (IliacSurfaceSnap deferred to _refineOstia)"
                            )
                            continue  # next branch

                        # ── Pass 1: sibling-separation scan ──────────────────────
                        # Compare each branch point against the NEAREST point on each
                        # sibling (not same step index) — more robust when branches
                        # have different point densities after bif-node pinning.
                        # Threshold scales with local diameter so wide vessels need
                        # proportionally more separation to be considered independent.
                        for k in range(
                            OSTIUM_SKIP_PTS, min(len(bPts), OSTIUM_MAX_STEPS)
                        ):
                            p = bPts[k]
                            local_diam = _localDiam(p)
                            sep_thresh = max(SEP_FRAC * local_diam, SEP_MIN_MM)

                            # All siblings must be farther than sep_thresh
                            sep_ok = True
                            for bj, sjPts in _siblingPts.items():
                                if bj == bi or not sjPts:
                                    continue
                                d_sib = _nearestDistToProximalBranch(bj, p)
                                if d_sib < sep_thresh:
                                    sep_ok = False
                                    break

                            if not sep_ok:
                                continue

                            # Confirm: next step also separated (noise filter)
                            k2 = k + 1
                            if k2 < len(bPts):
                                p2 = bPts[k2]
                                for bj, sjPts in _siblingPts.items():
                                    if bj == bi or not sjPts:
                                        continue
                                    if (
                                        _nearestDistToProximalBranch(bj, p2)
                                        < sep_thresh
                                    ):
                                        sep_ok = False
                                        break

                            if sep_ok:
                                found_k = k
                                ostiumPt = p
                                break

                        # Log Pass-1 result with per-step separations

                        # ── Pass 2: radius stabilisation refinement ───────────────
                        # Walk BACKWARD from found_k toward the bif to find the first
                        # point where the local diameter has stabilised relative to a
                        # far-downstream reference (past the flare zone).
                        #
                        # CRITICAL GUARD: Pass 2 may only DECREASE found_k (move the
                        # ostium closer to the bif), never increase it.  Previously the
                        # reference was sampled at found_k+5..+15, which was still in
                        # the flare, giving an inflated r_stable and pushing the ostium
                        # ~20mm downstream.  Now we sample at found_k+15..+30 to skip
                        # the flare, and we only commit the result if it is ≤ found_k.
                        _p2_k0 = found_k  # Pass-1 result — guard baseline
                        if bPts and found_k < len(bPts):
                            _ref_start = min(found_k + 15, len(bPts) - 1)
                            _ref_end = min(found_k + 30, len(bPts))
                            _ref_diams = [
                                _localDiam(bPts[_ki])
                                for _ki in range(_ref_start, _ref_end)
                            ]
                            if _ref_diams:
                                _ref_diams.sort()
                                r_stable = _ref_diams[len(_ref_diams) // 2]  # median
                                _p2_best_k = found_k  # default: no change
                                for k in range(
                                    found_k, max(OSTIUM_SKIP_PTS - 1, 0), -1
                                ):
                                    r_k = _localDiam(bPts[k])
                                    if (
                                        r_stable > 0
                                        and abs(r_k - r_stable) / r_stable
                                        < RADIUS_STABLE_FRAC
                                    ):
                                        _p2_best_k = k
                                        break
                                if _p2_best_k <= _p2_k0:  # only accept if bif-ward
                                    ostiumPt = bPts[_p2_best_k]
                                    found_k = _p2_best_k

                        self.branchMeta[bi]["ostium"] = ostiumPt
                        od = _dist3(ostiumPt, snappedBifPt)

                        # Map found_k (index into graphBranches[bi]) to the committed
                        # global point index (gi into self.points / rawPoints).
                        #
                        # graphBranches[bi] is committed sequentially into rawPoints
                        # starting at bs_gi, so:
                        #   self.points[bs_gi + k] == graphBranches[bi][k]
                        #
                        # This is exact — no nearest-distance search needed, which
                        # was the bug: searching self.points for a coordinate that
                        # came from graphBranches returned the raw start (bs_gi)
                        # because the ostium point wasn't in self.points verbatim.
                        bs_gi = self.branches[bi][0]  # raw global start
                        be_gi = self.branches[bi][1]
                        ostium_gi = min(bs_gi + found_k, be_gi - 1)
                        self.branchMeta[bi]["ostiumGi"] = ostium_gi
                        self.branchMeta[bi][
                            "stableStartGi"
                        ] = ostium_gi  # refined by _detectFindings

                        print(
                            f"[VesselAnalyzer] Branch {bi} ostium: "
                            f"({ostiumPt[0]:.1f},{ostiumPt[1]:.1f},{ostiumPt[2]:.1f}), "
                            f"{od:.1f}mm from bifurcation, "
                            f"gi={ostium_gi} (skips {ostium_gi - bs_gi} pts from raw start)"
                        )

                    # ── apply ostiumGi back to self.branches ──────────────────────
                    #
                    # Until now ostiumGi was only stored in branchMeta — the raw
                    # self.branches[bi] tuple still started at the graph topology node
                    # (the bifurcation).  We now rewrite self.branches[bi][0] to
                    # ostiumGi so every consumer that reads branches[bi][0] directly
                    # gets the anatomical start without needing getBranchStartGi().
                    #
                    # self.branches is a list of (startGi, endGi) tuples stored as
                    # a plain Python list — we replace each entry in-place.
                    # The trunk (bi==0) is never trimmed — it has no ostium.
                    #
                    # Also update branchMeta['length_mm'] to reflect the trimmed length
                    # so FINAL ANATOMY reports accurate branch lengths.

                    # ── PRE-TRIM snapshot for stable traversal ────────────────────
                    # Capture EVERYTHING the traversal builder needs BEFORE any
                    # ostiumGi trimming or refinement (both happen below / in
                    # _stabilizeRenalOstia / _refineOstia called later).
                    #
                    # Snapshotted fields:
                    #   _rawBranches  — (start_gi, end_gi) from graph topology
                    #   _travSnapshot — per-branch frozen dict:
                    #       role        : branch role (unchanged after this point)
                    #       ostium_pt   : best available ostium coordinate RIGHT NOW
                    #                     (used for renal trunk-walk anchor)
                    #
                    # The traversal builder reads ONLY these frozen values so the
                    # traversal point count is completely insulated from any future
                    # ostium refinement.  Measurements / display continue to read
                    # live branchMeta as before.
                    self._rawBranches = list(self.branches)
                    self._travSnapshot = {}
                    for _sbi in range(len(self.branches)):
                        _smeta = self.branchMeta.get(_sbi, {})
                        # Prefer explicit ostium point; fall back to raw branch start
                        _sbs = self.branches[_sbi][0]
                        _opt = _smeta.get("ostium") or (
                            tuple(rawPoints[_sbs]) if _sbs < len(rawPoints) else None
                        )
                        self._travSnapshot[_sbi] = {
                            "role": _smeta.get("role", ""),
                            "ostium_pt": _opt,
                        }
                    print(
                        f"[VesselAnalyzer] _rawBranches snapshot: "
                        f"{len(self._rawBranches)} branches, "
                        f"total pts={sum(e-s for s,e in self._rawBranches)}"
                    )

                    _trim_log = []
                    for bi in range(1, len(self.branches)):
                        ogi = self.branchMeta.get(bi, {}).get("ostiumGi")
                        if ogi is None:
                            continue
                        raw_start, raw_end = self.branches[bi]
                        if ogi <= raw_start or ogi >= raw_end:
                            continue  # no change needed or out of range

                        # Rewrite the branch start
                        self.branches[bi] = (ogi, raw_end)

                        # Recompute length from ostium to end
                        trimmed_len = (
                            self.distances[raw_end - 1] - self.distances[ogi]
                            if raw_end - 1 < len(self.distances)
                            and ogi < len(self.distances)
                            else self.branchMeta[bi]["length_mm"]
                        )
                        skipped = ogi - raw_start
                        self.branchMeta[bi]["length_mm"] = round(trimmed_len, 1)
                        _trim_log.append(
                            f"  Branch {bi}: trimmed {skipped} pts "
                            f"(gi {raw_start}→{ogi}), "
                            f"new length={trimmed_len:.1f}mm"
                        )

                    if _trim_log:
                        print(
                            f"[VesselAnalyzer] Branch trim applied ({len(_trim_log)} branches):"
                        )
                        for _tl in _trim_log:
                            print(_tl)
                    else:
                        print(
                            f"[VesselAnalyzer] Branch trim: no branches trimmed "
                            f"(ostiumGi == raw start for all)"
                        )

                    # [IliacOstium] Post-trim audit: log final ostiumGi / ostiumPt for
                    # every branch so the console shows exactly what survived the trim.
                    print(f"[IliacOstium] POST-TRIM ostiumGi audit:")
                    for _dbi in range(len(self.branches)):
                        _dm = self.branchMeta.get(_dbi, {})
                        _d_ogi = _dm.get("ostiumGi")
                        _d_opt = _dm.get("ostium")
                        _d_role = _dm.get("role", "?")
                        _d_coord = (
                            f"({_d_opt[0]:.1f},{_d_opt[1]:.1f},{_d_opt[2]:.1f})"
                            if _d_opt is not None
                            else "None"
                        )
                        print(
                            f"  bi={_dbi} role={_d_role} ostiumGi={_d_ogi} "
                            f"ostiumPt={_d_coord}"
                        )

                    # ── discard branches too short after trimming OR tagged noise ──
                    # A branch shorter than MIN_CLINICAL_MM after ostium trimming is
                    # a junction stub or sampling artefact, not a clinical vessel.
                    # Branches tagged 'noise' by the NoiseGate (origin outside trunk
                    # Z-extent) are also discarded here so they are fully removed from
                    # self.branches before surface classification — preventing their
                    # surface area from being absorbed by adjacent real branches.
                    MIN_CLINICAL_MM = 25.0
                    _kept = [0]  # always keep trunk (bi=0)
                    _discarded = []
                    for bi in range(1, len(self.branches)):
                        role = self.branchMeta.get(bi, {}).get("role", "")
                        length = self.branchMeta.get(bi, {}).get("length_mm", 0)
                        if role == "noise":
                            _discarded.append(bi)
                            print(
                                f"[NoiseGate] Branch {bi}: discarding from branch list "
                                f"(role=noise, len={length:.1f}mm)"
                            )
                        elif length >= MIN_CLINICAL_MM:
                            _kept.append(bi)
                        else:
                            _discarded.append(bi)

                    if _discarded:
                        print(
                            f"[VesselAnalyzer] Discarding {len(_discarded)} sub-clinical "
                            f"branches (< {MIN_CLINICAL_MM}mm after trim): {_discarded}"
                        )
                        # Rebuild self.branches, branchMeta keeping only _kept.
                        # NOTE: branchNames is NOT carried through here — it will be
                        # rebuilt from branchMeta roles by _roleName() after compaction.
                        # Carrying stale names through produces index-shifted display
                        # names when a branch is discarded mid-list.
                        new_branches = [self.branches[bi] for bi in _kept]
                        new_meta = {
                            new_i: self.branchMeta[old_bi]
                            for new_i, old_bi in enumerate(_kept)
                        }
                        self._discarded_gi_ranges = [
                            self.branches[bi] for bi in _discarded
                        ]
                        self.branches = new_branches
                        self.branchMeta = new_meta
                        # branchNames rebuilt from roles by _roleName() below
                        # Reset branchEndpointMap to sequential new indices so
                        # getBranchDisplayName produces correct names after compaction.
                        self.branchEndpointMap = {
                            new_i: new_i for new_i in range(len(new_branches))
                        }
                        print(
                            f"[VesselAnalyzer] After discard: {len(self.branches)} branches remain"
                        )
                        # Re-derive mainBranches from updated branchMeta after compaction
                        # (old indices in mainSet are now stale — roles are the ground truth)
                        mainBranches = [
                            bi
                            for bi in range(len(self.branches))
                            if self.branchMeta[bi]["role"] == "main"
                        ]
                        print(
                            f"[VesselAnalyzer] mainBranches remapped after discard: {mainBranches}"
                        )
                        # Store the old-index list so the traversal builder can map
                        # _rawBranches (pre-discard) to the post-discard branch order.
                        self._kept_branch_old_bis = list(_kept)
                    else:
                        # No discard — _rawBranches already aligns 1:1 with self.branches
                        self._kept_branch_old_bis = list(range(len(self.branches)))

                    # ── trunk continuation detection ──────────────────────────────
                    # A branch is a trunk continuation if it has:
                    #   1. diameter >= CONT_DIAM_RATIO × trunk mean diameter
                    #   2. angle to trunk direction < CONT_ANGLE_DEG
                    #   3. connected to the trunk bif end (within CONT_CONN_MM)
                    # Such branches are reclassified as 'side' (not 'main') but kept,
                    # since they represent the true proximal trunk extension beyond the
                    # bifurcation.  Future versions could stitch them into trunk[0].
                    CONT_DIAM_RATIO = 0.85  # 85% of trunk diameter
                    CONT_ANGLE_DEG = 30.0  # degrees
                    _trunk_diam = (
                        sum(
                            d
                            for d in self.diameters[
                                self.branches[0][0] : self.branches[0][1]
                            ]
                            if d > 0
                        )
                        / max(
                            1,
                            sum(
                                1
                                for d in self.diameters[
                                    self.branches[0][0] : self.branches[0][1]
                                ]
                                if d > 0
                            ),
                        )
                        if getattr(self, "diameters", [])
                        else 0.0
                    )
                    _trunk_dir = _trunkDir  # unit vector from flow norm

                    _cont_promoted = []
                    if _trunk_diam > 0:
                        for bi in range(1, len(self.branches)):
                            meta = self.branchMeta.get(bi, {})
                            if meta.get("role") == "main":
                                continue  # main branches are never continuations
                            b_diam = (
                                meta.get("length_mm", 0)
                                and (
                                    sum(
                                        d
                                        for d in self.diameters[
                                            self.branches[bi][0] : self.branches[bi][1]
                                        ]
                                        if d > 0
                                    )
                                    / max(
                                        1,
                                        sum(
                                            1
                                            for d in self.diameters[
                                                self.branches[bi][0] : self.branches[
                                                    bi
                                                ][1]
                                            ]
                                            if d > 0
                                        ),
                                    )
                                )
                                if getattr(self, "diameters", [])
                                else 0.0
                            )
                            if b_diam < CONT_DIAM_RATIO * _trunk_diam:
                                continue
                            # Direction of this branch (first 5 pts from ostium)
                            bs, be = self.branches[bi]
                            if be - bs < 3:
                                continue
                            _k = min(5, be - bs - 1)
                            _p0 = self.points[bs]
                            _p1 = self.points[bs + _k]
                            _bdir = [_p1[k] - _p0[k] for k in range(3)]
                            _bl = math.sqrt(sum(x * x for x in _bdir))
                            if _bl < 1e-6:
                                continue
                            _bdir = [x / _bl for x in _bdir]
                            _dot = sum(_bdir[k] * _trunk_dir[k] for k in range(3))
                            _ang = math.degrees(
                                math.acos(max(-1.0, min(1.0, abs(_dot))))
                            )
                            if _ang < CONT_ANGLE_DEG:
                                _cont_promoted.append(bi)
                                print(
                                    f"[VesselAnalyzer] Branch {bi} flagged as trunk continuation "
                                    f"(diam={b_diam:.1f}mm={b_diam/_trunk_diam:.0%} trunk, "
                                    f"angle={_ang:.1f}°)"
                                )
                    if not _cont_promoted:
                        print(
                            f"[VesselAnalyzer] Trunk continuation check: none detected"
                        )

                    # ── Vessel-type geometry validator (VesselTypeValidate, v297) ──
                    #
                    # Checks whether the geometry of the classified trunk is consistent
                    # with the user-selected vessel type.  Prints a clear warning if not.
                    # Does NOT override the user's selection — just surfaces the mismatch
                    # so it's visible in the log and (optionally) in the UI.
                    #
                    # Heuristics (all approximate, tuned for IVC vs aorta):
                    #
                    #   Trunk diameter:
                    #     IVC (venous):    typically 15–35 mm
                    #     Aorta (arterial): typically 12–28 mm — overlapping range,
                    #     so diameter alone is weak.  Use as secondary signal only.
                    #
                    #   Trunk straightness (curvature proxy):
                    #     IVC: generally straighter than aorta (lower tortuosity)
                    #     Aorta: more tortuous, especially infrarenal
                    #
                    #   Renal branch count vs selected type:
                    #     If selected=arterial AND renal_vein branches exist → impossible
                    #     (renal_vein role is blocked for arterial, so this would only
                    #     fire if the gate is misconfigured)
                    #     If selected=venous AND no renal_vein branches found → warn
                    #     (expected for single-kidney or truncated scans, not an error)
                    #
                    #   Primary bifurcation angle:
                    #     IVC bifurcation into iliacs: typically wider (100–130°)
                    #     Aortic bifurcation: typically narrower (50–80°)

                    try:
                        _vt_type = getattr(self, "vesselType", "venous") or "venous"
                        _vt_is_art = _vt_type.lower().startswith("art")

                        _vt_warns = []

                        # ── Signal 1: trunk diameter ──────────────────────────────
                        # Venous IVC is typically ≥ 15 mm; aorta can be as narrow as
                        # 12 mm post-aneurysm repair.  Only flag extreme mismatches.
                        _vt_tdiam = _trunk_diam if "_trunk_diam" in dir() else _trunkDiam
                        if _vt_tdiam > 0:
                            if _vt_is_art and _vt_tdiam > 38.0:
                                _vt_warns.append(
                                    f"trunk diameter {_vt_tdiam:.1f}mm is very large for aorta "
                                    f"(typical ≤28mm) — may be IVC"
                                )
                            if not _vt_is_art and _vt_tdiam < 10.0:
                                _vt_warns.append(
                                    f"trunk diameter {_vt_tdiam:.1f}mm is very small for IVC "
                                    f"(typical ≥15mm) — may be aorta"
                                )

                        # ── Signal 2: renal branch presence vs vessel type ─────────
                        _vt_renal_bis = [
                            bi for bi, m in self.branchMeta.items()
                            if m.get("role") == "renal_vein"
                        ]
                        if _vt_is_art and _vt_renal_bis:
                            # Should be impossible with VesselTypeGate active — flag if seen
                            _vt_warns.append(
                                f"renal_vein branches {_vt_renal_bis} found on arterial case "
                                f"— VesselTypeGate may not have fired"
                            )

                        # ── Signal 3: main branch count ───────────────────────────
                        # IVC → 2 iliac branches expected
                        # Aorta → 2 iliac branches expected (same, so not discriminating)
                        # But arterial aorta also has renal arteries — no renal_vein →
                        # side branches are the correct residual; no warn needed.

                        # ── Signal 4: bifurcation angle ───────────────────────────
                        _vt_main_bis = [
                            bi for bi, m in self.branchMeta.items()
                            if m.get("role") in ("main", "iliac_left", "iliac_right")
                        ]
                        if len(_vt_main_bis) == 2:
                            _vt_d0 = [
                                self.branchMeta[_vt_main_bis[0]].get("angle_deg", -1),
                                self.branchMeta[_vt_main_bis[1]].get("angle_deg", -1),
                            ]
                            _vt_bif_angle = sum(a for a in _vt_d0 if a >= 0)
                            if _vt_bif_angle > 10:  # only if angles are available
                                if _vt_is_art and _vt_bif_angle > 110:
                                    _vt_warns.append(
                                        f"bifurcation angle {_vt_bif_angle:.0f}° is wide "
                                        f"(typical aorta ≤90°) — may be IVC"
                                    )
                                if not _vt_is_art and _vt_bif_angle < 60:
                                    _vt_warns.append(
                                        f"bifurcation angle {_vt_bif_angle:.0f}° is narrow "
                                        f"(typical IVC ≥80°) — may be aorta"
                                    )

                        # ── Emit results ──────────────────────────────────────────
                        if _vt_warns:
                            print(
                                f"[VesselTypeValidate] *** GEOMETRY MISMATCH WARNING ***\n"
                                f"[VesselTypeValidate] Selected vessel type: {_vt_type!r}\n"
                                + "\n".join(
                                    f"[VesselTypeValidate]   ⚠  {w}" for w in _vt_warns
                                )
                            )
                            # Surface to UI status label if widget is reachable
                            try:
                                _vt_widget = getattr(
                                    slicer.modules, "VesselAnalyzerWidget", None
                                )
                                _vt_lbl = getattr(_vt_widget, "extractStatusLabel", None)
                                if _vt_lbl is not None:
                                    _vt_lbl.setText(
                                        "⚠ Vessel type may be wrong — check log"
                                    )
                                    _vt_lbl.setStyleSheet(
                                        "color: #e67e22; font-weight: bold;"
                                    )
                            except Exception:
                                pass
                        else:
                            print(
                                f"[VesselTypeValidate] vesselType={_vt_type!r} "
                                f"consistent with geometry ✓"
                            )

                    except Exception as _vt_exc:
                        print(f"[VesselTypeValidate] skipped (error: {_vt_exc})")

                    # ── post-ostium angle refinement ──────────────────────────────
                    # angle_deg was computed at branch-commit time (step 8), before
                    # ostium detection.  Both main branches shared bPts[0] = bif_end,
                    # so the initial angle vector started from the same shared node
                    # and landed in the parallel co-location zone → near-zero angle.
                    #
                    # Now that ostiumGi is resolved, recompute using a chord that is
                    # well past the bifurcation flare zone so the vector reflects the
                    # true individual branch direction.
                    #
                    # Algorithm (adaptive chord-skip):
                    #   skip  = min(20mm, 25% of branch length) — clears bif flare
                    #   sample = min(20mm, 25% of branch length) — stable direction window
                    #   Both values scale down for short branches (renals ~30mm)
                    #   so the sample stays in the proximal-mid zone, not the tip.
                    _trk = rawPoints[self.branches[0][0] : self.branches[0][1]]
                    if len(_trk) >= 2:
                        _tlo = _trk[max(0, len(_trk) - 4)]
                        _thi = _trk[-1]
                        _td = [_thi[k] - _tlo[k] for k in range(3)]
                        _tl = math.sqrt(sum(x * x for x in _td))
                        _tdir_ref = [x / _tl for x in _td] if _tl > 1e-6 else [0, 0, 1]
                    else:
                        _tdir_ref = list(_trunkDir)

                    for bi in range(1, len(self.branches)):
                        meta = self.branchMeta.get(bi, {})
                        ogi = meta.get("ostiumGi")
                        if ogi is None:
                            continue
                        bs, be = self.branches[bi]
                        if (
                            ogi < bs
                            or ogi >= be
                            or ogi >= len(rawPoints)
                            or be - ogi < 2
                        ):
                            continue

                        # Adaptive chord parameters — scale to branch length so short
                        # branches (renals ~31mm) aren't measured near their tip.
                        # Cap at fixed maximums for long branches (iliacs ~130-175mm).
                        #   skip  = min(20mm, 25% of branch arc) — clears bif flare dome
                        #   sample = min(20mm, 25% of branch arc) — stable direction window
                        _branch_arc = meta.get(
                            "length_mm", _seg_length(rawPoints[bs:be])
                        )
                        _adapt_mm = min(20.0, 0.25 * _branch_arc)
                        _skip_mm = _adapt_mm
                        _sample_mm = _adapt_mm

                        # ── Phase 1: skip _skip_mm past ostium ───────────────────
                        _skip_acc = 0.0
                        _chord_start_gi = ogi
                        for _si in range(ogi, be - 1):
                            _skip_acc += _dist3(rawPoints[_si], rawPoints[_si + 1])
                            if _skip_acc >= _skip_mm:
                                _chord_start_gi = _si + 1
                                break

                        # ── Phase 2: walk _sample_mm from chord_start ─────────────
                        _samp_acc = 0.0
                        _p_start = rawPoints[_chord_start_gi]
                        _p_sample = rawPoints[be - 1]  # fallback: tip
                        for _ai in range(_chord_start_gi, be - 1):
                            _samp_acc += _dist3(rawPoints[_ai], rawPoints[_ai + 1])
                            if _samp_acc >= _sample_mm:
                                _p_sample = rawPoints[_ai + 1]
                                break

                        _bv = [_p_sample[k] - _p_start[k] for k in range(3)]
                        _bl = math.sqrt(sum(x * x for x in _bv))
                        if _bl < 1e-6:
                            continue
                        _bv = [x / _bl for x in _bv]
                        _dot_ref = max(
                            -1.0, min(1.0, sum(_bv[k] * _tdir_ref[k] for k in range(3)))
                        )
                        new_angle = math.degrees(math.acos(abs(_dot_ref)))
                        old_angle = meta.get("angle_deg", 0.0)
                        self.branchMeta[bi]["angle_deg"] = round(new_angle, 1)
                        print(
                            f"[VesselAnalyzer] Branch {bi} angle refined: "
                            f"{old_angle:.1f}° → {new_angle:.1f}° "
                            f"(chord skip={_skip_mm:.0f}mm + sample={_sample_mm:.0f}mm "
                            f"from ostiumGi={ogi}, branch={_branch_arc:.0f}mm)"
                        )

                    # ── print structured anatomy summary ──────────────────────────
                    trunkMeta = self.branchMeta[0]
                    sideBranches = [
                        bi
                        for bi in range(len(self.branches))
                        if self.branchMeta[bi]["role"] == "side"
                    ]

                    print(f"[VesselAnalyzer] ── FINAL ANATOMY ──────────────────────")
                    print(f"  Trunk:        {trunkMeta['length_mm']:.1f} mm")
                    print(
                        f"  Bifurcation:  ({snappedBifPt[0]:.1f},"
                        f"{snappedBifPt[1]:.1f},{snappedBifPt[2]:.1f})"
                    )
                    for bi in mainBranches:
                        m = self.branchMeta[bi]
                        ost = m.get("ostium", snappedBifPt)
                        _lat_lbl = m.get("lateral_label", "")
                        _lbl = f"{_lat_lbl} Iliac" if _lat_lbl else f"Main [{bi}]"
                        print(
                            f"  {_lbl}:   {m['length_mm']:.1f} mm, "
                            f"angle={m['angle_deg']:.1f}°, "
                            f"ostium=({ost[0]:.1f},{ost[1]:.1f},{ost[2]:.1f})"
                        )
                    for bi in sideBranches:
                        m = self.branchMeta[bi]
                        print(f"  Side [{bi}]:   {m['length_mm']:.1f} mm")

                    # ── Diameter symmetry between main branches ───────────────────
                    if len(mainBranches) >= 2:
                        _d = [
                            self.branchMeta[bi].get("length_mm", 0.0)
                            for bi in mainBranches
                        ]
                        # Use mean diameter from branchMeta if available,
                        # else fall back to length as proxy
                        _md = [
                            self.branchMeta[bi].get("mean_diam", 0.0)
                            for bi in mainBranches
                        ]
                        if all(d > 0 for d in _md):
                            _sym_vals = _md
                            _sym_label = "diam"
                        else:
                            _sym_vals = _d
                            _sym_label = "len"
                        _sym_ratio = (
                            max(_sym_vals) / min(_sym_vals)
                            if min(_sym_vals) > 0
                            else 1.0
                        )
                        if _sym_ratio > 1.2:
                            _sym_str = (
                                f"ASYMMETRIC ({_sym_label} ratio={_sym_ratio:.2f})"
                            )
                        elif _sym_ratio > 1.1:
                            _sym_str = (
                                f"mild asymmetry ({_sym_label} ratio={_sym_ratio:.2f})"
                            )
                        else:
                            _sym_str = (
                                f"symmetric ({_sym_label} ratio={_sym_ratio:.2f})"
                            )
                        self._bifSymmetryRatio = round(_sym_ratio, 3)
                        self._bifSymmetryLabel = _sym_str
                        print(f"  Symmetry:     {_sym_str}")

                    # ── Bifurcation difficulty classification ─────────────────────
                    if len(mainBranches) >= 2:
                        _angles = [
                            self.branchMeta[bi].get("angle_deg", 0.0)
                            for bi in mainBranches
                        ]
                        _mean_angle = sum(_angles) / len(_angles)
                        _angle_spread = max(_angles) - min(_angles)
                        if _mean_angle < 20.0:
                            _diff_str = f"easy — low-angle bifurcation (mean {_mean_angle:.1f}°)"
                        elif _mean_angle < 40.0:
                            _diff_str = f"moderate (mean {_mean_angle:.1f}°)"
                        else:
                            _diff_str = (
                                f"complex — wide bifurcation (mean {_mean_angle:.1f}°)"
                            )
                        # Skewed bifurcation: one limb much wider than the other
                        # (common in May-Thurner anatomy — right iliac often steeper)
                        if _angle_spread > 30.0:
                            _diff_str += f", skewed (spread {_angle_spread:.1f}°)"
                            self._bifSkewed = True
                        else:
                            self._bifSkewed = False
                        self._bifDifficulty = _diff_str
                        self._bifMeanAngleDeg = round(_mean_angle, 1)
                        self._bifAngleSpreadDeg = round(_angle_spread, 1)
                        print(f"  Difficulty:   {_diff_str}")

                    # ── L/R iliac labeling ────────────────────────────────────────
                    # Strategy — prefer Endpoints node descriptions over raw tip X:
                    #
                    # The Endpoints fiducial node was labelled by
                    # _assignAnatomicEndpointLabels (VesselAnalyzer.py) before
                    # centerline extraction, so each control point already carries
                    # the correct Description: "Right iliac", "Left iliac", etc.
                    # We match each main branch to its nearest Endpoints control
                    # point and read the description to decide laterality.
                    #
                    # Why not raw tip X?
                    # The last point of the branch in rawPoints is the internal
                    # graph tip (where the VMTK solve terminated), which for
                    # a renal-path branch selected as "main" by the pair-scorer
                    # can land at the renal junction Z (~1915mm), far from the
                    # true iliac distal opening.  Its X sign is therefore
                    # unreliable for lateral assignment.
                    #
                    # Fallback: if the Endpoints node is unavailable or no
                    # description is set, revert to comparing the X-coordinates
                    # of the branch ostium points (branchMeta ostiumPt), which
                    # are placed at the primary bifurcation and are reliably
                    # comparable between the two main branches.
                    #
                    # Guard: only assign if exactly 2 main branches are present.
                    if len(mainBranches) == 2:
                        # Try Endpoints-node description matching first
                        _ep_node_label = slicer.mrmlScene.GetFirstNodeByName(
                            "Endpoints"
                        )
                        _bi_to_lat = {}   # bi → "Right" | "Left"

                        if _ep_node_label is not None:
                            _n_ep = _ep_node_label.GetNumberOfControlPoints()
                            # Build list of (cp_idx, description, position)
                            _ep_descs = []
                            for _ei in range(_n_ep):
                                _desc = ""
                                try:
                                    _desc = (
                                        _ep_node_label
                                        .GetNthControlPointDescription(_ei)
                                        .strip()
                                    )
                                except Exception:
                                    pass
                                _ep_p = [0.0, 0.0, 0.0]
                                _ep_node_label.GetNthControlPointPositionWorld(
                                    _ei, _ep_p
                                )
                                _ep_descs.append((_ei, _desc, tuple(_ep_p)))

                            # For each main branch, find the nearest endpoint
                            # control point by 3-D distance from the branch's
                            # raw graph tip (closest anatomical opening).
                            #
                            # IMPORTANT: matching is exclusive — once an endpoint
                            # is claimed by one branch it is removed from the pool
                            # for the remaining branches.  Without exclusion, both
                            # branches can independently win the same endpoint
                            # (e.g. both closer to "Right iliac" than "Left iliac")
                            # producing two iliac_right with no iliac_left.
                            def _ep_dist3(a, b):
                                return (
                                    (a[0] - b[0]) ** 2
                                    + (a[1] - b[1]) ** 2
                                    + (a[2] - b[2]) ** 2
                                ) ** 0.5

                            _claimed_ei = set()   # endpoint indices already assigned

                            for _mbi in mainBranches:
                                _mbs_g, _mbe_g = self.branches[_mbi]
                                # Probe priority:
                                #   1. raw graph tip — the iliac main branches share
                                #      the SAME ostium point (the primary bifurcation),
                                #      so ostium-based distances are identical for both
                                #      branches and cannot discriminate L from R.
                                #      The raw graph tip is near the true iliac distal
                                #      opening and gives unambiguous 3-D distances to
                                #      the labelled iliac endpoints.
                                #   2. ostium — fallback only when no tip is available
                                #      (e.g. degenerate branch with 0 raw points).
                                if _mbe_g > _mbs_g and _mbe_g - 1 < len(rawPoints):
                                    _probe = rawPoints[_mbe_g - 1]
                                    _probe_src = "raw_tip"
                                else:
                                    _ost_pt = self.branchMeta.get(_mbi, {}).get("ostium")
                                    if _ost_pt is not None:
                                        _probe = tuple(_ost_pt)
                                        _probe_src = "ostium_fallback"
                                    else:
                                        _probe = None
                                        _probe_src = "none"
                                if _probe is None:
                                    print(
                                        f"[IliacLabel] bi={_mbi}: probe is None"
                                        f" (branches[{_mbi}]=({_mbs_g},{_mbe_g})"
                                        f" rawPoints len={len(rawPoints)}) — skipped"
                                    )
                                    continue
                                print(
                                    f"[IliacLabel] bi={_mbi}: probe src={_probe_src}"
                                    f" ({_probe[0]:+.1f},{_probe[1]:+.1f},{_probe[2]:+.1f})"
                                )
                                _best_ei, _best_d, _best_desc = -1, 1e18, ""
                                for _ei, _desc, _epos in _ep_descs:
                                    if _ei in _claimed_ei:
                                        continue   # already owned by another branch
                                    _d = _ep_dist3(_probe, _epos)
                                    print(
                                        f"[IliacLabel] bi={_mbi}"
                                        f"  probe=({_probe[0]:+.1f},{_probe[1]:+.1f},{_probe[2]:+.1f})"
                                        f"  ep{_ei} desc={_desc!r}"
                                        f"  dist={_d:.1f}mm"
                                    )
                                    if _d < _best_d:
                                        _best_d = _d
                                        _best_ei = _ei
                                        _best_desc = _desc
                                print(
                                    f"[IliacLabel] bi={_mbi} → best ep{_best_ei}"
                                    f" desc={_best_desc!r} dist={_best_d:.1f}mm"
                                )
                                # Map description → laterality
                                if "right iliac" in _best_desc.lower():
                                    _bi_to_lat[_mbi] = "Right"
                                    _claimed_ei.add(_best_ei)
                                elif "left iliac" in _best_desc.lower():
                                    _bi_to_lat[_mbi] = "Left"
                                    _claimed_ei.add(_best_ei)
                                else:
                                    print(
                                        f"[IliacLabel] bi={_mbi}: best ep{_best_ei}"
                                        f" desc={_best_desc!r} is not an iliac endpoint"
                                        f" — laterality unresolved, fallback will handle"
                                    )
                                # Non-iliac descriptions (renal etc.) leave
                                # _bi_to_lat entry absent → fallback below

                        # Fallback: if description matching didn't resolve both,
                        # compare ostium X-coordinates (both start at the primary
                        # bifurcation, so X difference reflects which side each
                        # branch departs toward).
                        if len(_bi_to_lat) < 2:
                            print(
                                f"[IliacLabel] Endpoint description matching resolved"
                                f" {len(_bi_to_lat)}/2 branches: {_bi_to_lat}"
                                f" — falling back to X-coordinate comparison"
                            )
                            _mb_x = {}
                            for _mbi in mainBranches:
                                if _mbi in _bi_to_lat:
                                    continue   # already resolved — don't overwrite
                                # Prefer ostium (set by IliacOstium pipeline),
                                # fall back to raw graph tip.
                                _opt = self.branchMeta.get(_mbi, {}).get(
                                    "ostium"
                                )
                                if _opt is not None:
                                    _mb_x[_mbi] = _opt[0]
                                    print(
                                        f"[IliacLabel] bi={_mbi}: X from ostium"
                                        f" = {_opt[0]:+.1f}mm"
                                    )
                                else:
                                    _mbs_g, _mbe_g = self.branches[_mbi]
                                    if _mbe_g > _mbs_g and _mbe_g - 1 < len(rawPoints):
                                        _mb_x[_mbi] = rawPoints[_mbe_g - 1][0]
                                        print(
                                            f"[IliacLabel] bi={_mbi}: X from raw tip"
                                            f" = {rawPoints[_mbe_g - 1][0]:+.1f}mm"
                                            f" (no ostium in meta)"
                                        )
                                    else:
                                        print(
                                            f"[IliacLabel] bi={_mbi}: cannot read X"
                                            f" (branches=({_mbs_g},{_mbe_g})"
                                            f" rawPoints len={len(rawPoints)})"
                                        )
                            # If one branch was already resolved by description matching,
                            # assign the opposite side to the remaining one.
                            if len(_mb_x) == 1 and len(_bi_to_lat) == 1:
                                _resolved_lat = next(iter(_bi_to_lat.values()))
                                _remaining_bi = next(iter(_mb_x.keys()))
                                _opposite = "Left" if _resolved_lat == "Right" else "Right"
                                _bi_to_lat[_remaining_bi] = _opposite
                                print(
                                    f"[IliacLabel] bi={_remaining_bi}: assigned"
                                    f" {_opposite!r} as complement to already-resolved"
                                    f" {_resolved_lat!r} on the other main branch"
                                )
                            elif len(_mb_x) == 2:
                                _mbi_list = list(_mb_x.keys())
                                _x0, _x1 = _mb_x[_mbi_list[0]], _mb_x[_mbi_list[1]]
                                print(
                                    f"[IliacLabel] X fallback:"
                                    f" bi={_mbi_list[0]} X={_x0:+.1f}mm"
                                    f"  bi={_mbi_list[1]} X={_x1:+.1f}mm"
                                    f" (RAS: larger X = anatomical Right)"
                                )
                                # In RAS coordinates: more positive X = anatomical Right,
                                # more negative X = anatomical Left.
                                _bi_to_lat[_mbi_list[0]] = (
                                    "Right" if _x0 >= _x1 else "Left"
                                )
                                _bi_to_lat[_mbi_list[1]] = (
                                    "Left" if _x0 >= _x1 else "Right"
                                )
                            else:
                                print(
                                    f"[IliacLabel] X fallback: _mb_x has"
                                    f" {len(_mb_x)} entries — cannot assign"
                                )

                        # Apply labels if both branches were resolved
                        if len(_bi_to_lat) == 2:
                            for _mbi, _lat in _bi_to_lat.items():
                                self.branchMeta[_mbi]["lateral_label"] = _lat
                                _role = (
                                    "iliac_right"
                                    if _lat == "Right"
                                    else "iliac_left"
                                )
                                self.branchMeta[_mbi]["role"] = _role
                                # Determine which X to log: use ostiumPt if available
                                _opt = self.branchMeta[_mbi].get("ostiumPt")
                                _log_x = (
                                    _opt[0]
                                    if _opt is not None
                                    else self.branchMeta[_mbi].get(
                                        "angle_deg", 0.0
                                    )
                                )
                                _mbs_g, _mbe_g = self.branches[_mbi]
                                _tip_x = (
                                    rawPoints[_mbe_g - 1][0]
                                    if _mbe_g > _mbs_g
                                    and _mbe_g - 1 < len(rawPoints)
                                    else float("nan")
                                )
                                print(
                                    f"[IliacLabel] Branch {_mbi} → {_lat} iliac "
                                    f"(tip X={_tip_x:.1f}mm)"
                                )
                            # Store convenience attributes
                            _right_bi = next(
                                (b for b, l in _bi_to_lat.items() if l == "Right"),
                                None,
                            )
                            _left_bi = next(
                                (b for b, l in _bi_to_lat.items() if l == "Left"),
                                None,
                            )
                            if _right_bi is not None:
                                self._iliacRightBi = _right_bi
                            if _left_bi is not None:
                                self._iliacLeftBi = _left_bi
                        else:
                            print(
                                "[IliacLabel] Could not resolve L/R from endpoint "
                                "descriptions or ostium X — labeling skipped"
                            )
                    else:
                        print(
                            f"[IliacLabel] {len(mainBranches)} main branch(es) — "
                            f"L/R labeling requires exactly 2"
                        )

                    # ── Renal orthogonality score ─────────────────────────────────
                    # A lightweight 0–1 score combining:
                    #   angle_score     = angle_deg / 90.0  (1.0 = fully orthogonal)
                    #   lateral_score   = tip lateral offset / REF_MM  capped at 1.0
                    #     where lateral offset = distance from tip to trunk axis line
                    #   ortho_score = 0.5 * angle_score + 0.5 * lateral_score
                    # Stored as branchMeta[bi]['ortho_score'] (0.0–1.0, higher = more
                    # orthogonal = more renal-like).
                    # Reference for lateral normalisation: 50mm (typical renal offset).
                    ORTHO_LAT_REF_MM = 50.0
                    if len(_trk) >= 2:
                        # Trunk axis: unit vector from root to bif end
                        _tax_p0 = rawPoints[self.branches[0][0]]
                        _tax_p1 = rawPoints[self.branches[0][1] - 1]
                        _tax_v = [_tax_p1[k] - _tax_p0[k] for k in range(3)]
                        _tax_l = math.sqrt(sum(x * x for x in _tax_v))
                        _trunk_axis = (
                            [x / _tax_l for x in _tax_v] if _tax_l > 1e-6 else [0, 0, 1]
                        )
                    else:
                        _trunk_axis = list(_trunkDir)

                    _renal_bis = [
                        bi
                        for bi in range(1, len(self.branches))
                        if self.branchMeta[bi].get("role") == "renal_vein"
                    ]
                    for _rbi in _renal_bis:
                        _rm = self.branchMeta[_rbi]
                        _ang = _rm.get("angle_deg", 0.0)
                        _angle_score = min(1.0, _ang / 90.0)

                        # Lateral offset: project tip onto trunk axis, measure perp dist
                        _rbs, _rbe = self.branches[_rbi]
                        if _rbe > _rbs and _rbe <= len(rawPoints):
                            _tip = rawPoints[_rbe - 1]
                            # Vector from trunk root to tip
                            _v2tip = [_tip[k] - _tax_p0[k] for k in range(3)]
                            # Scalar projection onto trunk axis
                            _proj = sum(_v2tip[k] * _trunk_axis[k] for k in range(3))
                            # Perpendicular component
                            _perp = [
                                _v2tip[k] - _proj * _trunk_axis[k] for k in range(3)
                            ]
                            _lat_mm = math.sqrt(sum(x * x for x in _perp))
                        else:
                            _lat_mm = 0.0

                        _lat_score = min(1.0, _lat_mm / ORTHO_LAT_REF_MM)
                        _ortho = round(0.5 * _angle_score + 0.5 * _lat_score, 3)
                        _rm["ortho_score"] = _ortho
                        _rm["tip_lateral_mm"] = round(_lat_mm, 1)
                        print(
                            f"[OrthoScore] Branch {_rbi} (renal): "
                            f"angle={_ang:.1f}° → angle_sc={_angle_score:.2f}  "
                            f"lat={_lat_mm:.1f}mm → lat_sc={_lat_score:.2f}  "
                            f"ortho={_ortho:.3f}"
                        )

                    # Compute tip_lateral_mm for all remaining non-main branches
                    # (side_branch etc.) so venous confidence lateral score is valid.
                    _main_roles = ("main", "iliac_right", "iliac_left", "renal_vein")
                    for _sbi in range(1, len(self.branches)):
                        if self.branchMeta[_sbi].get("role") in _main_roles:
                            continue
                        if "tip_lateral_mm" in self.branchMeta[_sbi]:
                            continue  # already computed above
                        _sm = self.branchMeta[_sbi]
                        _sbs, _sbe = self.branches[_sbi]
                        if _sbe > _sbs and _sbe <= len(rawPoints):
                            _tip = rawPoints[_sbe - 1]
                            _v2tip = [_tip[k] - _tax_p0[k] for k in range(3)]
                            _proj = sum(_v2tip[k] * _trunk_axis[k] for k in range(3))
                            _perp = [
                                _v2tip[k] - _proj * _trunk_axis[k] for k in range(3)
                            ]
                            _lat_mm_s = math.sqrt(sum(x * x for x in _perp))
                        else:
                            _lat_mm_s = 0.0
                        _sm["tip_lateral_mm"] = round(_lat_mm_s, 1)
                        print(
                            f"[OrthoScore] Branch {_sbi} (side): "
                            f"lat={_lat_mm_s:.1f}mm"
                        )

                    print(f"[VesselAnalyzer] ──────────────────────────────────────")

            else:
                for i in range(pts.GetNumberOfPoints()):
                    rawPoints.append(pts.GetPoint(i))

        if len(rawPoints) < 2:
            return False

        self.points = rawPoints

        # ── Build generic branch structure from detected segments ────────────────
        # Works for any bifurcated vessel: aorto-iliac, carotid, coronary, etc.
        # Branches are numbered sequentially; the trunk (longest segment) is Branch 1.
        # No hardcoded coordinates, indices, or vessel-specific assumptions.

        # Assign semantic names from role + lateral_label.
        # lateral_label ("Right" | "Left") is written by the IliacLabel block
        # before this function runs; roles "iliac_right" / "iliac_left" always
        # carry a matching lateral_label, but we fall back gracefully if absent.
        #
        # Side branches and renal veins are numbered sequentially (Side-1, Side-2…)
        # rather than by raw bi so the displayed number is stable and compact.
        _side_counter = [0]   # mutable cell for closure
        _rv_counter   = [0]

        def _roleName(i):
            meta  = getattr(self, "branchMeta", {}).get(i, {})
            role  = meta.get("role", "")
            lat   = meta.get("lateral_label", "")   # "Right" | "Left" | ""

            if role == "trunk":
                name = "Trunk"
            elif role == "iliac_right":
                name = "Right Iliac"
            elif role == "iliac_left":
                name = "Left Iliac"
            elif role == "main":
                # main = pair member IliacLabel didn't resolve yet
                name = f"{lat} Iliac" if lat else f"Main-{i}"
            elif role == "side":
                _side_counter[0] += 1
                name = f"Side-{_side_counter[0]}"
            elif role == "renal_vein":
                _rv_counter[0] += 1
                name = f"{lat} Renal Vein" if lat else f"Renal Vein {_rv_counter[0]}"
            elif role == "renal_fragment":
                name = f"RF-{i}"
            elif role == "iliac_fragment":
                parent = meta.get("fragment_of", "")
                side = meta.get("lateral_label", "")
                prefix = side[0] if side else "I"
                name = f"{prefix}-IliacFrag-{parent}"
            elif role == "noise":
                name = f"Noise-{i}"
            else:
                name = f"B-{i}"   # unknown role — never silently blank

            print(
                f"[RoleName] bi={i}  role={role!r}  lateral={lat!r}"
                f"  → name={name!r}"
            )
            return name

        self.branchNames = [_roleName(i) for i in range(len(self.branches))]
        print(f"[RoleName] Final branchNames: {self.branchNames}")

        # ── Sync names into branchMeta so the UI reads the right labels ──────
        # The navigation widget mixin and IliacCorrect both read branchMeta keys
        # ("label", "display_name", "name") to populate the branch selector combo
        # and stats box.  _roleName only builds self.branchNames (a standalone
        # list); without this sync those keys are absent, so the combo falls back
        # to raw role strings ("iliac_right") or shows nothing at all.
        print("[BranchNameSync] Writing names into branchMeta:")
        for _sni, _sname in enumerate(self.branchNames):
            if _sni not in self.branchMeta:
                print(f"[BranchNameSync]   bi={_sni}: NOT in branchMeta \u2014 skipped")
                continue
            _sm = self.branchMeta[_sni]
            _old_label   = _sm.get("label",        "<absent>")
            _old_display = _sm.get("display_name", "<absent>")
            _old_name    = _sm.get("name",         "<absent>")
            _sm["label"]        = _sname
            _sm["display_name"] = _sname
            _sm["name"]         = _sname
            print(
                f"[BranchNameSync]   bi={_sni}"
                f"  role={_sm.get('role', '?')!r}"
                f"  lateral={_sm.get('lateral_label', '')!r}"
                f"  name={_sname!r}"
                f"  (was label={_old_label!r}"
                f" display_name={_old_display!r}"
                f" name={_old_name!r})"
            )

        # No sub-branches for generic vessels — all branches are independent
        self.branchSubRanges = []
        self.branchConnections = {}

        # ── Snapshot raw branch ranges for traversal ──────────────────────────
        # self._rawBranches was captured BEFORE the ostium-trim step above, so
        # it holds the full graph-topology point ranges (not shifted by ostiumGi).
        # Discarded branches were removed from self.branches before this point,
        # so we re-filter _rawBranches/_travSnapshot to match kept branch indices.
        _kept_old_bis = getattr(self, "_kept_branch_old_bis", None)
        if _kept_old_bis is not None:
            self._rawBranches = [
                self._rawBranches[old_bi]
                for old_bi in _kept_old_bis
                if old_bi < len(self._rawBranches)
            ]
            _old_snap = getattr(self, "_travSnapshot", {})
            self._travSnapshot = {
                new_i: _old_snap[old_bi]
                for new_i, old_bi in enumerate(_kept_old_bis)
                if old_bi in _old_snap
            }
        # Safety: ensure length matches self.branches
        if len(self._rawBranches) != len(self.branches):
            self._rawBranches = list(self.branches)
            print(
                f"[VesselAnalyzer] WARNING: _rawBranches length mismatch — "
                f"falling back to trimmed branches for traversal"
            )
        print(
            f"[VesselAnalyzer] Traversal will use _rawBranches: "
            f"{len(self._rawBranches)} branches, "
            f"total raw pts={sum(e-s for s,e in self._rawBranches)}"
        )

        # ── Frozen traversal snapshot ─────────────────────────────────────────
        # _travSnapshot was captured BEFORE any ostium trimming or refinement.
        # The traversal builder reads ONLY from this frozen dict — never from
        # live branchMeta — so roles, ostium coordinates, and point counts are
        # 100% invariant to any ostium-detection change made after this point.
        _trav_snap = getattr(self, "_travSnapshot", {})

        def _tsnap_role(bi):
            return _trav_snap.get(bi, {}).get("role", "")

        def _tsnap_ostium_pt(bi):
            return _trav_snap.get(bi, {}).get("ostium_pt")

        # _allBranchTraversal: anatomical IVUS navigation order
        #
        #   1. Trunk reversed: bif-end (gi_be-1) → proximal root (gi_bs)
        #      → slider 0 = bifurcation, feels like entering from the lesion site
        #   2. For each main/side branch (bi >= 1, role != renal_vein):
        #      a. ostium → distal tip  (forward through branch)
        #      b. distal tip → bif     (retrace back to bifurcation)
        #   3. Renal veins: walk trunk from iliac-bif up to renal ostium,
        #      enter branch (ostium → tip), retrace (tip → ostium),
        #      walk trunk back down to iliac-bif.
        #      This ensures the suprarenal IVC above the renal ostia is
        #      navigated during the renal pullback sequence, not skipped.
        #
        # This mirrors the clinical IVUS pullback sequence: start at bif,
        # sweep each iliac out and back, then for each renal walk up the
        # suprarenal IVC, sweep the renal, and walk back down.

        # Trunk reversed (bif first) — use raw snapshot so trunk range is stable
        _t_bs, _t_be = self._rawBranches[0]
        _trunk_fwd = list(range(_t_bs, _t_be))  # proximal→bif
        _trunk_rev = list(reversed(_trunk_fwd))  # bif→proximal (slider starts here)
        self._allBranchTraversal = _trunk_rev[:]
        self._allBranchTraversalBranchIds = [0] * len(_trunk_rev)

        # Separate main/side branches from renals — use frozen roles
        _main_branches = []
        _renal_branches = []
        for _bi in range(1, len(self._rawBranches)):
            _role = _tsnap_role(_bi)
            if _role == "renal_vein":
                _renal_branches.append(_bi)
            elif _role == "renal_fragment":
                pass  # fragment: skip from traversal (duplicate of primary)
            else:
                _main_branches.append(_bi)

        # bif global index (last point of trunk = bifurcation hub)
        _bif_gi = _t_be - 1

        for _bi in _main_branches:
            _bs, _be = self._rawBranches[
                _bi
            ]  # raw snapshot — stable across ostium refines
            _fwd = list(range(_bs, _be))  # ostium → distal
            _ret = list(reversed(_fwd))  # distal → ostium
            # Only iliac-type main branches are actually connected to the primary
            # bifurcation and need a bridge point back to bif_gi.  Side/short
            # branches that ended up in _main_branches (not renal_vein / fragment)
            # are attached to the trunk wall — appending bif_gi would create a
            # spurious duplicate of that gi in the traversal, breaking globalToTraversal.
            _role = _tsnap_role(_bi)  # frozen — never changes between runs
            _is_iliac = _role in ("main", "iliac_left", "iliac_right")
            _ret_to_bif = [_bif_gi] if _is_iliac else []
            self._allBranchTraversal += _fwd + _ret + _ret_to_bif
            self._allBranchTraversalBranchIds += (
                [_bi] * (len(_fwd) + len(_ret) + len(_ret_to_bif))
            )
            print(
                f"[Nav] Main/side {_bi} role={_role}: "
                f"raw gi {_bs}..{_be-1} ({_be-_bs} pts), "
                f"fwd+ret={len(_fwd)*2}pts, is_iliac={_is_iliac}, "
                f"running_total={len(self._allBranchTraversal)}"
            )

        # Sort renals by raw branch start gi so visit order is deterministic
        # and independent of ostiumGi (which can drift between runs).
        def _renal_sort_key(bi):
            return self._rawBranches[bi][0]  # raw start — frozen, never drifts

        _renal_branches_sorted = sorted(_renal_branches, key=_renal_sort_key)

        for _bi in _renal_branches_sorted:
            _bs, _be = self._rawBranches[
                _bi
            ]  # raw snapshot — stable across ostium refines

            # ── Walk up trunk from iliac-bif to renal ostium ──────────────
            # IMPORTANT: The renal branch gi range (_bs.._be) is SEPARATE from
            # the trunk gi range (_t_bs.._t_be) — they are concatenated
            # sequentially in rawPoints.  _ogi is a BRANCH gi and will NEVER
            # fall inside the trunk gi range, so the old gi-range check always
            # produced walk_up=0.
            #
            # Fix: find the trunk point whose COORDINATE is nearest to the
            # renal ostium coordinate, walk from bif to that trunk point.
            # _trunk_rev[0]=_bif_gi, _trunk_rev[k]=_bif_gi-k (reversed fwd).
            #
            # The ostium coordinate comes from the FROZEN _travSnapshot so the
            # trunk walk length never changes even if _refineOstia shifts the
            # ostium point later.

            # Ostium coordinate — read from frozen snapshot only
            _ogi_pt = _tsnap_ostium_pt(_bi)

            _walk_up = []
            _walk_down = []
            _best_trunk_gi = _t_bs
            _best_trunk_dist = float("inf")
            if _ogi_pt is not None and len(_trunk_fwd) > 1:
                # Find trunk gi with coordinate nearest to renal ostium
                for _tgi in range(_t_bs, _t_be):
                    _tp = self.points[_tgi]
                    _d = (
                        (_tp[0] - _ogi_pt[0]) ** 2
                        + (_tp[1] - _ogi_pt[1]) ** 2
                        + (_tp[2] - _ogi_pt[2]) ** 2
                    ) ** 0.5
                    if _d < _best_trunk_dist:
                        _best_trunk_dist = _d
                        _best_trunk_gi = _tgi

                # trunk_rev position of nearest trunk gi
                _steps = _bif_gi - _best_trunk_gi
                if 0 < _steps <= len(_trunk_rev) - 1:
                    _walk_up = _trunk_rev[1 : _steps + 1]  # skip bif (already there)
                    _walk_down = list(reversed(_walk_up))
                print(
                    f"[Nav] Renal {_bi}: nearest trunk gi={_best_trunk_gi} "
                    f"dist={_best_trunk_dist:.1f}mm walk_steps={_steps} "
                    f"ostium_pt=({_ogi_pt[0]:.1f},{_ogi_pt[1]:.1f},{_ogi_pt[2]:.1f})"
                )

            _fwd = list(range(_bs, _be))  # renal: ostium → tip
            _ret = list(reversed(_fwd))  # renal: tip → ostium

            # Full sequence: walk up trunk → enter renal → retrace renal → walk down trunk
            self._allBranchTraversal += _walk_up + _fwd + _ret + _walk_down
            self._allBranchTraversalBranchIds += (
                [0] * len(_walk_up)
                + [_bi] * (len(_fwd) + len(_ret))
                + [0] * len(_walk_down)
            )
            print(
                f"[Nav] Renal {_bi}: walk_up={len(_walk_up)}pts "
                f"(bif_gi={_bif_gi}→trunk_gi={_best_trunk_gi}), "
                f"branch={len(_fwd)}pts (raw gi {_bs}..{_be-1}), "
                f"walk_down={len(_walk_down)}pts, running_total={len(self._allBranchTraversal)}"
            )

        print(
            f"[Nav] Traversal built: {len(self._allBranchTraversal)} steps "
            f"(trunk_rev={len(_trunk_rev)}, "
            f"main={len(_main_branches)}, renal={len(_renal_branches)})"
        )
        print(
            f"[Nav] Raw branch breakdown (frozen snapshot — invariant to ostium changes):"
        )
        for _dbi, (_dbs, _dbe) in enumerate(self._rawBranches):
            _role = _tsnap_role(_dbi)
            _opt = _tsnap_ostium_pt(_dbi)
            _opt_str = (
                f"({_opt[0]:.1f},{_opt[1]:.1f},{_opt[2]:.1f})" if _opt else "none"
            )
            print(
                f"[Nav]   branch {_dbi} role={_role}: "
                f"raw gi {_dbs}..{_dbe-1} ({_dbe-_dbs} pts), "
                f"frozen_ostium_pt={_opt_str}"
            )

        # Log the ep_map so numbering can be verified
        _ep_map = getattr(self, "branchEndpointMap", {})
        _ep_display = {
            k: ("Trunk" if k == 0 else f"ep{v}→Branch {v+1}")
            for k, v in _ep_map.items()
        }
        print(f"[VesselAnalyzer] branchEndpointMap: {_ep_display}")

        # Display names for logging
        for _bi, (_bs, _be) in enumerate(self.branches):
            _p0 = self.points[_bs]
            _p1 = self.points[_be - 1]
            _n = _be - _bs
            print(
                f"[VesselAnalyzer] {self.getBranchDisplayName(_bi)}: "
                f"local pt 0-{_n-1} ({_n} pts) "
                f"D={abs(_p0[0])*2:.1f},A={_p0[1]:.1f},S={_p0[2]:.1f} → "
                f"D={abs(_p1[0])*2:.1f},A={_p1[1]:.1f},S={_p1[2]:.1f}"
            )

        # ─────────────────────────────────────────────────────────────────────

        self.distances = [0.0]
        for i in range(1, len(self.points)):
            dx = self.points[i][0] - self.points[i - 1][0]
            dy = self.points[i][1] - self.points[i - 1][1]
            dz = self.points[i][2] - self.points[i - 1][2]
            self.distances.append(
                self.distances[-1] + math.sqrt(dx * dx + dy * dy + dz * dz)
            )

        # Use VMTK MaximumInscribedSphereRadius if available (most accurate)
        # This is the radius of the largest inscribed sphere at each centerline point
        if (
            hasattr(self, "rawRadii")
            and len(self.rawRadii) == len(self.points)
            and all(r is not None for r in self.rawRadii)
        ):
            self.diameters = [r * 2.0 for r in self.rawRadii]
            print(f"[VesselAnalyzer] Using VMTK inscribed sphere diameters")
        else:
            self.diameters = (
                self._computeDiameters() if modelNode else [0.0] * len(self.points)
            )
            print(
                f"[VesselAnalyzer] Using surface distance diameters (VMTK radius array not available)"
            )

            # ── Per-branch median smoothing ───────────────────────────────────
            # Surface-distance ray-casting produces isolated single-point spikes
            # (±30–50 % of local diameter) when a ray exits through a thin wall
            # and re-enters an adjacent vessel — or when mesh triangulation is
            # locally coarse.  These spikes corrupt min/max stats and finding
            # detection even after the outlier-K filter in _computeDiameters.
            #
            # Strategy: sliding median window of ±DIAM_SMOOTH_WIN pts applied
            # independently within each branch's gi range so that the ostium /
            # bifurcation boundary never bleeds across branches.
            #
            # WIN=4 → ±4 pts = 9-pt window ≈ ±6 mm at typical 1.4 mm/pt spacing.
            # This removes single-ray-hit spikes (1–2 pts wide) while preserving
            # 10 mm-scale taper gradients and focal stenoses (≥5 pts = ≥7 mm).
            #
            # Raw values preserved in self._rawDiameters for debugging.
            # ─────────────────────────────────────────────────────────────────
            if getattr(self, "branches", None) and len(self.diameters) > 0:
                self._rawDiameters = dict(enumerate(self.diameters))  # gi → raw_d
                DIAM_SMOOTH_WIN = 4
                smoothed = list(self.diameters)  # copy; we write back here
                for bs, be in self.branches:
                    seg = list(range(bs, be))
                    n_seg = len(seg)
                    if n_seg < 3:
                        continue
                    for local_i, gi in enumerate(seg):
                        lo = max(0, local_i - DIAM_SMOOTH_WIN)
                        hi = min(n_seg, local_i + DIAM_SMOOTH_WIN + 1)
                        window_vals = [
                            self.diameters[seg[j]]
                            for j in range(lo, hi)
                            if self.diameters[seg[j]] > 0.5
                        ]
                        if len(window_vals) >= 3:
                            window_vals.sort()
                            smoothed[gi] = window_vals[len(window_vals) // 2]
                        # else: leave raw value (thin/short segment, don't corrupt)
                self.diameters = smoothed
                print(
                    f"[VesselAnalyzer] Diameter smoothing: WIN={DIAM_SMOOTH_WIN} "
                    f"({2*DIAM_SMOOTH_WIN+1}-pt median per branch)"
                )
        # ── Flare-start computation (must precede _annotateTreeDiameters) ────
        # _detectFindings computes stableStartGi as part of its per-branch
        # flare-walk, but it is only called when the user clicks "Find Lesions".
        # _annotateTreeDiameters uses stableStartGi to set the prox window —
        # if it hasn't been set yet, gi_s_prox stays at ostiumGi and prox is
        # contaminated by the junction/flare zone (prox=18.7mm on 74mm renal).
        # Fix: run the flare-walk unconditionally here so stableStartGi is
        # always populated before _annotateTreeDiameters executes.
        self._computeStableStarts()
        self._stabilizeRenalOstia()
        self._refineOstia()
        self._computeOstiumConfidence()
        self._logOstiumTrainingRecord()  # LogReg: log features+outcome, train if enough data
        # ── Post-confidence REJECT suppression ───────────────────────────────
        # Gate 3 in _refineOstia could not check confidence grades because
        # _computeOstiumConfidence runs after _refineOstia.  Apply suppression
        # here now that grades are available.  Side effect: suppressed branches
        # get ostiumGi=None so downstream display skips the ostium marker.
        self._suppressRejectedOstia()
        self._applyManualRanges()

        # [IliacOstium] FINAL audit — print every branch's ostiumGi, stableStartGi,
        # and the coordinate at that gi so we can compare against what the navigator
        # shows.  This is the ground truth of what loadCenterline committed.
        print(f"[IliacOstium] FINAL PIPELINE ostiumGi / stableStartGi / coord audit:")
        for _fbi in range(len(self.branches)):
            _fm = self.branchMeta.get(_fbi, {})
            _f_role = _fm.get("role", "?")
            _f_ogi = _fm.get("ostiumGi")
            _f_sgi = _fm.get("stableStartGi")
            _f_lat = _fm.get("lateral_label", "")
            _f_label = f"{_f_role}" + (f"/{_f_lat}" if _f_lat else "")
            if _f_ogi is not None and _f_ogi < len(self.points):
                _fp = self.points[_f_ogi]
                _f_coord = f"R {_fp[0]:.1f} A {_fp[1]:.1f} S {_fp[2]:.1f}"
                _f_diam = (
                    f"Ø {self.diameters[_f_ogi]:.2f}mm"
                    if self.diameters and _f_ogi < len(self.diameters)
                    else "Ø ?"
                )
            else:
                _f_coord = "gi=None or OOB"
                _f_diam = ""
            print(
                f"  bi={_fbi} [{_f_label}] ostiumGi={_f_ogi} sgi={_f_sgi} "
                f"-> {_f_coord}  {_f_diam}"
            )
        # Annotate tree segments with diameter statistics now that
        # self.points and self.diameters are both fully populated.
        if getattr(self, "branchTree", None) is not None:
            self._annotateTreeDiameters()
        # Correct artifact-like diameter drops in self.diameters in-place so
        # downstream stats (getBranchStats, report, stent sizing) are clean.
        # Re-annotate after correction so [TreeDiam] log reflects fixed values.
        if self._applyArtifactCorrection():
            if getattr(self, "branchTree", None) is not None:
                self._annotateTreeDiameters()

        # Rename VMTK 'Endpoints-N' markup nodes to anatomical names so the
        # 3D view shows "Trunk", "Left Renal vein", "Right iliac", etc.
        self._labelEndpointNodes()

        # ── [BranchStatsDebug] Dump everything the stats box will read ────────
        # This block exposes the full state of branchMeta, branchNames,
        # branchEndpointMap, and lateral_label so mismatches between what the
        # stats box shows and what the classifier assigned are immediately visible.
        print(f"[BranchStatsDebug] ══ BRANCH STATS INPUT DUMP ══════════════════")
        print(f"[BranchStatsDebug] Total branches: {len(self.branches)}")
        print(f"[BranchStatsDebug] branchNames: {getattr(self, 'branchNames', [])}")
        _ep_map_d = getattr(self, "branchEndpointMap", {})
        print(f"[BranchStatsDebug] branchEndpointMap: {_ep_map_d}")
        print(f"[BranchStatsDebug] ── Per-branch detail ──────────────────────────")
        for _dbi in range(len(self.branches)):
            _dm = self.branchMeta.get(_dbi, {})
            _d_role     = _dm.get("role", "MISSING")
            _d_lat      = _dm.get("lateral_label", "")
            _d_len      = _dm.get("length_mm", 0.0)
            _d_angle    = _dm.get("angle_deg", 0.0)
            _d_ogi      = _dm.get("ostiumGi")
            _d_ost      = _dm.get("ostium")
            _d_name     = getattr(self, "branchNames", [None]*(_dbi+1))[_dbi] \
                          if _dbi < len(getattr(self, "branchNames", [])) else "?"
            _d_ep_idx   = _ep_map_d.get(_dbi, "?")
            _d_bs, _d_be = self.branches[_dbi]
            # Raw tip X (last committed point — what _assignEp and L/R logic use)
            _d_tip_x = "?"
            if _d_be > _d_bs and _d_be - 1 < len(self.points):
                _d_tip_x = f"{self.points[_d_be - 1][0]:.1f}"
            # Raw root X (first committed point)
            _d_root_x = "?"
            if _d_bs < len(self.points):
                _d_root_x = f"{self.points[_d_bs][0]:.1f}"
            _d_label   = _dm.get("label",        "<absent>")
            _d_dname   = _dm.get("display_name", "<absent>")
            print(
                f"[BranchStatsDebug]  bi={_dbi}"
                f"  name={_d_name!r}"
                f"  label={_d_label!r}"
                f"  display_name={_d_dname!r}"
                f"  role={_d_role!r}"
                f"  lateral={_d_lat!r}"
                f"  ep_idx={_d_ep_idx}"
                f"  len={_d_len:.1f}mm"
                f"  angle={_d_angle:.1f}°"
                f"  root_X={_d_root_x}"
                f"  tip_X={_d_tip_x}"
                f"  ostiumGi={_d_ogi}"
            )
        print(f"[BranchStatsDebug] ── Endpoint node control points ─────────────")
        try:
            import slicer as _sl
            _ep_nd = _sl.mrmlScene.GetFirstNodeByName("Endpoints")
            if _ep_nd:
                for _ci in range(_ep_nd.GetNumberOfControlPoints()):
                    _cp = [0.0, 0.0, 0.0]
                    _ep_nd.GetNthControlPointPositionWorld(_ci, _cp)
                    _cp_lbl  = _ep_nd.GetNthControlPointLabel(_ci)
                    try:
                        _cp_desc = _ep_nd.GetNthControlPointDescription(_ci)
                    except Exception:
                        _cp_desc = ""
                    print(
                        f"[BranchStatsDebug]   cp{_ci}"
                        f"  label={_cp_lbl!r}"
                        f"  desc={_cp_desc!r}"
                        f"  X={_cp[0]:+.1f} Y={_cp[1]:+.1f} Z={_cp[2]:+.1f}"
                    )
            else:
                print("[BranchStatsDebug]   No 'Endpoints' node found in scene")
        except Exception as _dbg_e:
            print(f"[BranchStatsDebug]   (endpoint node read failed: {_dbg_e})")
        print(f"[BranchStatsDebug] ════════════════════════════════════════════════")
        # ── end BranchStatsDebug ──────────────────────────────────────────────

        # ── Final label audit: confirm branchMeta label keys right before return
        # If something between BranchNameSync and here overwrites or clears them
        # it will show up as a mismatch between this dump and the sync log above.
        print("[LabelAudit] branchMeta label keys at runAnalysis return:")
        for _lai in range(len(self.branches)):
            _lam = self.branchMeta.get(_lai, {})
            print(
                f"[LabelAudit]   bi={_lai}"
                f"  role={_lam.get('role','?')!r}"
                f"  label={_lam.get('label','<absent>')!r}"
                f"  display_name={_lam.get('display_name','<absent>')!r}"
                f"  name={_lam.get('name','<absent>')!r}"
            )
        print("[LabelAudit] Done — if labels above are correct but UI still blank,")
        print("[LabelAudit] the combo is populated AFTER runAnalysis returns and")
        print("[LabelAudit] reads from a different source. Upload vessel_navigation_widget_mixin.py.")

        return True


    def _labelEndpointNodes(self):
        """Sync control-point labels to their descriptions on the Endpoints node.

        _assignAnatomicEndpointLabels (called before centerline extraction) writes
        the correct anatomical name into each control point's Description field:
            desc = "Right iliac" | "Left iliac" | "Right Renal vein" |
                   "Left Renal vein" | "IVC inlet (Trunk)"

        The Label field is what shows in the 3D view.  Any subsequent processing
        (centerline extraction, graph walk, branch-matching) must not override
        these descriptions — but the label may have been reset to a generic
        "Endpoints-N" string.  This method simply copies desc → label for every
        control point that has a non-empty description, which is the ground truth.

        The old branch-matching approach (matching via branchEndpointMap proximity)
        was unreliable because VMTK routes centerlines to whichever graph node is
        nearest — often the wrong anatomical endpoint — causing renal labels to
        overwrite iliac control points and vice-versa.
        """
        import slicer

        # ── 1. Find the Endpoints fiducial node ──────────────────────────────
        ep_node = None
        for cls in ("vtkMRMLMarkupsFiducialNode", "vtkMRMLMarkupsCurveNode"):
            n = slicer.mrmlScene.GetNumberOfNodesByClass(cls)
            for i in range(n):
                node = slicer.mrmlScene.GetNthNodeByClass(i, cls)
                if node and node.GetName().startswith("Endpoints"):
                    ep_node = node
                    break
            if ep_node:
                break

        if ep_node is None:
            print("[LabelEndpoints] No Endpoints node found — skipping.")
            return

        n_cp = ep_node.GetNumberOfControlPoints()
        if n_cp == 0:
            print("[LabelEndpoints] Endpoints node has 0 control points — skipping.")
            return

        # ── 2. Copy desc → label for every cp that has a description ─────────
        # The description was set by _assignAnatomicEndpointLabels and is the
        # single source of truth for the anatomical name of each endpoint.
        # We never overwrite descriptions here — only labels.
        renamed = 0
        for ci in range(n_cp):
            try:
                desc = ep_node.GetNthControlPointDescription(ci).strip()
            except Exception:
                desc = ""
            if not desc:
                continue  # no ground-truth description — leave label unchanged

            old_label = ep_node.GetNthControlPointLabel(ci)
            if old_label == desc:
                continue  # already correct

            ep_node.SetNthControlPointLabel(ci, desc)
            renamed += 1
            p = [0.0, 0.0, 0.0]
            ep_node.GetNthControlPointPosition(ci, p)
            print(
                f"[LabelEndpoints] cp{ci} '{old_label}' → '{desc}'  "
                f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})"
            )

        print(f"[LabelEndpoints] Done: {renamed}/{n_cp} control points relabeled.")

    def getBranchDisplayName(self, bi):
        """Return the human-readable display name for branch index *bi*.

        Priority:
          1. branchMeta[bi]["label"]  — written by BranchNameSync after every
             runAnalysis; the canonical source of truth.
          2. branchNames[bi]          — the same value, kept in sync, but
             checked as a belt-and-suspenders fallback.
          3. Role-based fallback      — role + lateral_label, never blank.
          4. Generic index fallback   — "Branch N" so the UI always gets a string.
        """
        if bi is None or bi < 0:
            return "All branches"

        meta = getattr(self, "branchMeta", {}).get(bi, {})

        # 1. branchMeta label key (written by BranchNameSync)
        label = meta.get("label", "")
        if label:
            return label

        # 2. branchNames list
        bn = getattr(self, "branchNames", [])
        if 0 <= bi < len(bn) and bn[bi]:
            return bn[bi]

        # 3. Reconstruct from role + lateral_label
        role = meta.get("role", "")
        lat  = meta.get("lateral_label", "")
        if role == "trunk":
            return "Trunk"
        if role in ("iliac_right", "iliac_left"):
            if lat:
                return f"{lat} Iliac"
            return "Right Iliac" if "right" in role else "Left Iliac"
        if role == "main":
            return f"{lat} Iliac" if lat else f"Main-{bi}"
        if role == "renal_vein":
            return f"{lat} Renal Vein" if lat else f"Renal Vein {bi}"
        if role == "renal_fragment":
            return f"RF-{bi}"
        if role == "iliac_fragment":
            parent = meta.get("fragment_of", "")
            side = meta.get("lateral_label", "")
            prefix = side[0] if side else "I"
            return f"{prefix}-IliacFrag-{parent}"
        if role == "side":
            return f"Side-{bi}"  # fallback only; BranchNameSync sets sequential label

        # 4. Generic fallback — never return blank
        return f"Branch {bi + 1}"


# ── Slicer module-scanner guard ───────────────────────────────────────────────
# Slicer auto-scans all .py files in the module folder and expects a class
# matching the filename.  This stub satisfies that requirement without
# registering as a real loadable module (no ScriptedLoadableModule base).
class vessel_centerline_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""

    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_centerline_mixin"
            parent.hidden = True  # hide from Slicer module list
