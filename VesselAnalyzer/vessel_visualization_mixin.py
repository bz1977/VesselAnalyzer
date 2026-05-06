"""
vessel_visualization_mixin.py — Color overlays, collateral detection, endoluminal camera, and scene updates.

Part of the VesselAnalyzer mixin decomposition.
These methods are mixed into VesselAnalyzerLogic via multiple inheritance:

    class VesselAnalyzerLogic(
        VisualizationMixin,
        ...
        ScriptedLoadableModuleLogic,
    ): ...

All methods use ``self`` normally — no changes to call sites required.
"""
# ruff: noqa  (this file is auto-extracted; formatting is inherited from VesselAnalyzer.py)


import slicer
import vtk
import math


class VisualizationMixin:
    """Mixin: Color overlays, collateral detection, endoluminal camera, and scene updates."""

    def applyColorOverlay(self):
        """Show findings as colored markup spheres at each finding location."""
        if not hasattr(self, "findings") or not self.findings:
            return
        self._removeOverlayNodes()
        TYPE_COLORS = {
            "Aneurysm": (0.90, 0.10, 0.10),
            "Ectasia": (1.00, 0.55, 0.00),
            "Pancaking": (0.20, 0.60, 1.00),
            "Mild Compression": (1.00, 0.85, 0.00),
            "Eccentric Compression": (0.95, 0.40, 0.10),  # orange-red: eccentric iliac
            "Focal Narrowing": (0.90, 0.10, 0.10),  # red: focal stenosis / May-Thurner
        }
        pdmap = getattr(self, "preDilationMap", {})
        for f in self.findings:
            idx = f["pointIdx"]
            # Skip ballooned points — their markers were removed in onPreDilate
            if idx in pdmap:
                continue
            pt = self.points[idx]
            # Eccentric iliac findings are stored as "Mild Compression" type (for stent
            # sizing compatibility) but should display with the eccentric label and color
            ftype_display = f["type"]
            if f.get("source") == "eccentric":
                ftype_display = "Eccentric Compression"
            color = TYPE_COLORS.get(ftype_display, (0.5, 0.5, 0.5))
            # Use original (pre-balloon) diameter for marker offset
            orig_diams = getattr(self, "_origDiameters", {})
            diam = (
                orig_diams.get(idx)
                or f.get("value")
                or (
                    self.diameters[idx]
                    if self.diameters and idx < len(self.diameters)
                    else 10.0
                )
            )
            # Offset marker along the vessel's local normal (perpendicular to centerline)
            # so it sits beside the vessel regardless of orientation
            import math as _math

            prev_idx = max(0, idx - 1)
            next_idx = min(len(self.points) - 1, idx + 1)
            tang = [
                self.points[next_idx][k] - self.points[prev_idx][k] for k in range(3)
            ]
            tlen = _math.sqrt(sum(x * x for x in tang)) or 1.0
            tang = [x / tlen for x in tang]
            # Cross with world-up (0,0,1) to get a radial direction
            ref = [0, 0, 1] if abs(tang[2]) < 0.9 else [1, 0, 0]
            normal = [
                tang[1] * ref[2] - tang[2] * ref[1],
                tang[2] * ref[0] - tang[0] * ref[2],
                tang[0] * ref[1] - tang[1] * ref[0],
            ]
            nlen = _math.sqrt(sum(x * x for x in normal)) or 1.0
            normal = [x / nlen for x in normal]
            offset = diam * 0.75
            marker_pt = (
                pt[0] + normal[0] * offset,
                pt[1] + normal[1] * offset,
                pt[2] + normal[2] * offset,
            )
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", f"Finding_{ftype_display}_{f['pointIdx']}"
            )
            node.AddControlPoint(
                vtk.vtkVector3d(marker_pt[0], marker_pt[1], marker_pt[2])
            )
            node.SetNthControlPointLabel(0, ftype_display)
            dn = node.GetDisplayNode()
            if dn is None:
                node.CreateDefaultDisplayNodes()
                dn = node.GetDisplayNode()
            dn.SetSelectedColor(color[0], color[1], color[2])
            dn.SetColor(color[0], color[1], color[2])
            dn.SetGlyphScale(8.0)
            dn.SetGlyphType(13)
            dn.SetTextScale(4.0)
            dn.SetOpacity(1.0)
            dn.SetVisibility(1)
            node.SetAttribute("VesselAnalyzerOverlay", "1")
        # Do NOT change model opacity — keep vessel at full opacity
        # Finding markers are placed beside the vessel (offset by radius) so they're
        # visible without making the model transparent
        # Hide IVUS ring — it's inside the lumen and appears as a disk when visible
        if self._ringModelNode and slicer.mrmlScene.IsNodePresent(self._ringModelNode):
            self._ringModelNode.GetDisplayNode().SetVisibility(0)


    def _detectCollaterals(self):
        """Detect collateral vessels: short + small diameter + high branch-off angle."""
        KNOWN_BIF_Z = []
        TOL = 25.0  # no hardcoded bifurcation Z
        labels = [self.getBranchDisplayName(i) for i in range(20)]
        self.collaterals = []
        meta = getattr(self, "branchMeta", {})

        _sub_indices = {
            sr["branch_idx"]
            for sr in getattr(self, "branchSubRanges", [])
            if "branch_idx" in sr
        }
        for bi in range(len(self.branches)):
            if bi in _sub_indices:
                continue
            stats = self.getBranchStats(bi)
            bm = meta.get(bi, {})
            length = stats["length"]
            maxD = stats["max"]
            avgD = stats["avg"]
            angle = bm.get("angle_deg", 0.0)
            originZ = bm.get("originZ", 0.0)
            label = labels[bi] if bi < len(labels) else f"Branch {bi+1}"

            score = 0
            reasons = []
            if length < 80.0:
                score += 1
                reasons.append(f"short ({length:.0f}mm)")
            if maxD < 8.0:
                score += 1
                reasons.append(f"small (max {maxD:.1f}mm)")
            elif maxD < 10.0:
                score += 0.5
                reasons.append(f"borderline diam ({maxD:.1f}mm)")
            if angle > 60.0:
                score += 1
                reasons.append(f"high angle ({angle:.0f} deg)")
            elif angle > 45.0:
                score += 0.5
                reasons.append(f"moderate angle ({angle:.0f} deg)")
            if any(abs(originZ - bz) < TOL for bz in KNOWN_BIF_Z):
                score -= 1.5
                reasons.append(f"near bifurcation Z={originZ:.0f}")

            if score >= 2.0:
                self.collaterals.append(
                    {
                        "branchIdx": bi,
                        "label": label,
                        "confidence": "High" if score >= 2.5 else "Moderate",
                        "score": round(score, 1),
                        "length_mm": round(length, 1),
                        "maxD": round(maxD, 2),
                        "avgD": round(avgD, 2),
                        "angle_deg": round(angle, 1),
                        "originZ": round(originZ, 1),
                        "tipZ": round(bm.get("tipZ", 0.0), 1),
                        "origin_pt": bm.get("origin", None),
                        "reasons": ", ".join(reasons),
                    }
                )

        print(f"[VesselAnalyzer] Collaterals: {len(self.collaterals)} found")
        for c in self.collaterals:
            print(f"  {c['label']}: score={c['score']}, {c['reasons']}")
        return self.collaterals


    def applyCollateralOverlay(self):
        """Show collateral branches as cyan markup spheres."""
        if not hasattr(self, "collaterals"):
            return
        self._removeOverlayNodes()
        CYAN = (0.00, 0.78, 0.86)
        for c in self.collaterals:
            pt = c.get("origin_pt")
            if pt is None:
                si, _ = self.branches[c["branchIdx"]]
                pt = self.points[si]
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", f"Collateral_{c['label']}"
            )
            node.AddControlPoint(vtk.vtkVector3d(pt[0], pt[1], pt[2]))
            node.SetNthControlPointLabel(0, f"Collateral\n{c['label']}")
            dn = node.GetDisplayNode()
            dn.SetSelectedColor(*CYAN)
            dn.SetColor(*CYAN)
            dn.SetGlyphScale(4.0)
            dn.SetTextScale(2.5)
            dn.SetVisibility(1)
            node.SetAttribute("VesselAnalyzerOverlay", "1")
        print(
            f"[VesselAnalyzer] Collateral overlay: {len(self.collaterals)} markers placed"
        )


    def _removeOverlayNodes(self):
        """Remove all previously placed overlay markup nodes."""
        to_remove = []
        for i in range(slicer.mrmlScene.GetNumberOfNodes()):
            n = slicer.mrmlScene.GetNthNode(i)
            if n and n.GetAttribute("VesselAnalyzerOverlay") == "1":
                to_remove.append(n)
        for n in to_remove:
            slicer.mrmlScene.RemoveNode(n)


    def clearColorOverlay(self):
        """Remove all overlay markers and restore vessel opacity."""
        self._removeOverlayNodes()
        if self.modelNode:
            _mdn = self.modelNode.GetDisplayNode()
            if _mdn:
                _mdn.SetOpacity(1.0)


    def updateVisualizations(
        self,
        idx,
        showSphere=True,
        showRing=True,
        showLine=True,
        sphereColor=(0.18, 0.80, 0.44),
    ):
        if idx < 0 or idx >= len(self.points):
            print(f"[VIZ] REJECTED idx={idx} len={len(self.points)}")
            return
        pt = self.points[idx]

        radius = (
            self.diameters[idx] / 2.0
            if self.diameters and self.diameters[idx] > 0
            else 3.0
        )

        # Tangent direction
        nextIdx = min(idx + 1, len(self.points) - 1)
        prevIdx = max(idx - 1, 0)
        tx = self.points[nextIdx][0] - self.points[prevIdx][0]
        ty = self.points[nextIdx][1] - self.points[prevIdx][1]
        tz = self.points[nextIdx][2] - self.points[prevIdx][2]
        tlen = math.sqrt(tx * tx + ty * ty + tz * tz)
        if tlen > 0:
            tx /= tlen
            ty /= tlen
            tz /= tlen

        # --- Sphere ---
        if showSphere:
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(pt[0], pt[1], pt[2])
            sphere.SetRadius(radius)
            sphere.SetPhiResolution(20)
            sphere.SetThetaResolution(20)
            sphere.Update()
            if not self._sphereModelNode or not slicer.mrmlScene.IsNodePresent(
                self._sphereModelNode
            ):
                self._sphereModelNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLModelNode", "IVUS_Sphere"
                )
                self._sphereModelNode.CreateDefaultDisplayNodes()
            self._sphereModelNode.SetAndObservePolyData(sphere.GetOutput())

            d = self._sphereModelNode.GetDisplayNode()
            d.SetColor(sphereColor[0], sphereColor[1], sphereColor[2])
            d.SetOpacity(0.4)
            d.SetVisibility(1)
            # Show sphere in all views — both normal and endoluminal
            d.RemoveAllViewNodeIDs()
        elif self._sphereModelNode and slicer.mrmlScene.IsNodePresent(
            self._sphereModelNode
        ):
            self._sphereModelNode.GetDisplayNode().SetVisibility(0)

        # --- Ring ---
        if showRing:
            disk = vtk.vtkDiskSource()
            disk.SetInnerRadius(radius * 0.85)
            disk.SetOuterRadius(radius)
            disk.SetCircumferentialResolution(40)
            disk.SetRadialResolution(1)
            disk.Update()
            transform = vtk.vtkTransform()
            transform.Translate(pt[0], pt[1], pt[2])
            cross = [ty * 1 - tz * 0, tz * 0 - tx * 1, tx * 0 - ty * 0]
            if abs(tx) < 0.9:
                cross = [ty * 0 - tz * 1, tz * tx - tx * 0, tx * 1 - ty * tx]
            clen = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
            if clen > 0.001:
                dot = tz  # dot with [0,0,1]
                angle = math.degrees(math.acos(max(-1, min(1, dot))))
                transform.RotateWXYZ(
                    angle, cross[0] / clen, cross[1] / clen, cross[2] / clen
                )
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(disk.GetOutput())
            tf.SetTransform(transform)
            tf.Update()
            if not self._ringModelNode or not slicer.mrmlScene.IsNodePresent(
                self._ringModelNode
            ):
                self._ringModelNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLModelNode", "IVUS_Ring"
                )
                self._ringModelNode.CreateDefaultDisplayNodes()
            self._ringModelNode.SetAndObservePolyData(tf.GetOutput())
            d = self._ringModelNode.GetDisplayNode()
            d.SetColor(1.0, 0.8, 0.0)
            d.SetOpacity(0.9)
            d.SetVisibility(1)
        elif self._ringModelNode and slicer.mrmlScene.IsNodePresent(
            self._ringModelNode
        ):
            self._ringModelNode.GetDisplayNode().SetVisibility(0)

        # --- Diameter Line ---
        if showLine and self.diameters and self.diameters[idx] > 0:
            if abs(tx) < 0.9:
                perp = [ty * 0 - tz * 1, tz * tx - tx * 0, tx * 1 - ty * tx]
            else:
                perp = [ty * 1 - tz * 0, tz * 0 - tx * 1, tx * 0 - ty * 0]
            plen = math.sqrt(perp[0] ** 2 + perp[1] ** 2 + perp[2] ** 2)
            if plen > 0:
                perp = [p / plen for p in perp]
            p1 = [
                pt[0] + perp[0] * radius,
                pt[1] + perp[1] * radius,
                pt[2] + perp[2] * radius,
            ]
            p2 = [
                pt[0] - perp[0] * radius,
                pt[1] - perp[1] * radius,
                pt[2] - perp[2] * radius,
            ]
            if not self._lineMarkupNode or not slicer.mrmlScene.IsNodePresent(
                self._lineMarkupNode
            ):
                self._lineMarkupNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsLineNode", "IVUS_DiameterLine"
                )
                self._lineMarkupNode.CreateDefaultDisplayNodes()
                d = self._lineMarkupNode.GetDisplayNode()
                d.SetSelectedColor(1, 0.2, 0.2)
                d.SetColor(1, 0.2, 0.2)
                d.SetLineThickness(0.5)
                d.SetTextScale(3)
                for _hide_label in (
                    "SetPropertiesLabelVisibility",
                    "SetPointLabelsVisibility",
                ):
                    if hasattr(d, _hide_label):
                        getattr(d, _hide_label)(False)
            self._lineMarkupNode.RemoveAllControlPoints()
            self._lineMarkupNode.AddControlPoint(vtk.vtkVector3d(p1))
            self._lineMarkupNode.AddControlPoint(vtk.vtkVector3d(p2))
            d = self._lineMarkupNode.GetDisplayNode()
            for _hide_label in (
                "SetPropertiesLabelVisibility",
                "SetPointLabelsVisibility",
            ):
                if hasattr(d, _hide_label):
                    getattr(d, _hide_label)(False)
            d.SetVisibility(1)
        elif self._lineMarkupNode and slicer.mrmlScene.IsNodePresent(
            self._lineMarkupNode
        ):
            self._lineMarkupNode.GetDisplayNode().SetVisibility(0)


    def setEndoluminalCamera(self, idx):
        """Position the 3D camera inside the vessel at centerline point idx,
        looking forward along the tangent direction."""
        import math

        globalIdx = idx + self.getActiveBranchOffset()
        if globalIdx < 0 or globalIdx >= len(self.points):
            return

        pt = self.points[globalIdx]

        # Tangent: look-ahead direction
        ni = min(globalIdx + 3, len(self.points) - 1)
        pi = max(globalIdx - 1, 0)
        tx = self.points[ni][0] - self.points[pi][0]
        ty = self.points[ni][1] - self.points[pi][1]
        tz = self.points[ni][2] - self.points[pi][2]
        tlen = math.sqrt(tx * tx + ty * ty + tz * tz)
        if tlen < 1e-6:
            return
        tx /= tlen
        ty /= tlen
        tz /= tlen

        # Camera sits slightly behind current point, looks ahead
        CAM_OFFSET = 2.0  # mm behind current point
        LOOK_AHEAD = 15.0  # mm ahead to focus on

        cam_pos = [
            pt[0] - tx * CAM_OFFSET,
            pt[1] - ty * CAM_OFFSET,
            pt[2] - tz * CAM_OFFSET,
        ]
        focal_pt = [
            pt[0] + tx * LOOK_AHEAD,
            pt[1] + ty * LOOK_AHEAD,
            pt[2] + tz * LOOK_AHEAD,
        ]

        # Up vector: world Z if tangent not too vertical, else world Y
        if abs(tz) < 0.9:
            view_up = [0.0, 0.0, 1.0]
        else:
            view_up = [0.0, 1.0, 0.0]

        # Use widget(1) for endoluminal camera
        lm = slicer.app.layoutManager()
        threeDWidget = (
            lm.threeDWidget(1) if lm.threeDViewCount > 1 else lm.threeDWidget(0)
        )
        if not threeDWidget:
            return
        threeDView = threeDWidget.threeDView()
        renderer = threeDView.renderWindow().GetRenderers().GetFirstRenderer()
        if not renderer:
            return

        camera = renderer.GetActiveCamera()
        camera.SetPosition(cam_pos[0], cam_pos[1], cam_pos[2])
        camera.SetFocalPoint(focal_pt[0], focal_pt[1], focal_pt[2])
        camera.SetViewUp(view_up[0], view_up[1], view_up[2])
        camera.SetViewAngle(90)  # wide FOV for endoluminal feel

        renderer.ResetCameraClippingRange()
        threeDView.renderWindow().Render()

        # Also update widget(0) external view to keep current point centered
        if lm.threeDViewCount > 1:
            w0 = lm.threeDWidget(0)
            if w0:
                r0 = w0.threeDView().renderWindow().GetRenderers().GetFirstRenderer()
                if r0:
                    # Move focal point to current position, keep camera angle
                    c0 = r0.GetActiveCamera()
                    cur_pos = list(c0.GetPosition())
                    cur_foc = list(c0.GetFocalPoint())
                    # Shift both by delta to center on current point
                    delta = [pt[k] - cur_foc[k] for k in range(3)]
                    c0.SetFocalPoint(pt[0], pt[1], pt[2])
                    c0.SetPosition(
                        cur_pos[0] + delta[0],
                        cur_pos[1] + delta[1],
                        cur_pos[2] + delta[2],
                    )
                    r0.ResetCameraClippingRange()
                    w0.threeDView().renderWindow().Render()


    def resetCamera(self):
        """Restore default camera view and model opacity."""
        threeDWidget = slicer.app.layoutManager().threeDWidget(0)
        if not threeDWidget:
            return
        threeDView = threeDWidget.threeDView()
        renderer = threeDView.renderWindow().GetRenderers().GetFirstRenderer()
        if renderer:
            renderer.GetActiveCamera().SetViewAngle(30)
            renderer.ResetCamera()
            threeDView.renderWindow().Render()


# ── Slicer module-scanner guard ───────────────────────────────────────────────
# Slicer auto-scans all .py files in the module folder and expects a class
# matching the filename.  This stub satisfies that requirement without
# registering as a real loadable module (no ScriptedLoadableModule base).
class vessel_visualization_mixin:  # noqa: E302
    """Slicer module-scanner stub — not a real loadable module."""
    def __init__(self, parent=None):
        if parent:
            parent.title = "vessel_visualization_mixin"
            parent.hidden = True  # hide from Slicer module list
