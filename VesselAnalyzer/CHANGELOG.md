# VesselAnalyzer Changelog

> This file was extracted from the build-note comment in VesselAnalyzer.py.
> Future entries should be added here as `## vNNN — Title` sections.

---

## v294 — AnatomicEndpointLabels (2026-04-25)

### Summary
Add automatic anatomical labelling of the 5 seed endpoints produced by
**Auto-Detect** so the 3D viewport labels match the reference image overlay:

| Slicer label   | Anatomy               | Assignment rule                        |
|----------------|-----------------------|----------------------------------------|
| Endpoints-5    | IVC inlet / Trunk top | Highest Z (most superior point)        |
| Endpoints-4    | Left Renal vein       | Mid-height, most negative X (left)     |
| Endpoints-3    | Right Renal vein      | Mid-height, most positive X (right)    |
| Endpoints-2    | Left iliac            | Inferior pair, most negative X (left)  |
| Endpoints-1    | Right iliac           | Inferior pair, most positive X (right) |

### Changes
- **`VesselAnalyzerWidget._ENDPOINT_LABELS`** — class-level dict mapping
  endpoint number → human-readable anatomical name (used for control-point
  description tooltip and report context).
- **`VesselAnalyzerWidget._assignAnatomicEndpointLabels(fiducialNode)`** —
  new method; sorts control points by Z and X to assign canonical
  `Endpoints-N` names and anatomical descriptions in-place.  Supports 4 or
  5 endpoints (one renal absent → graceful degradation).  Falls back to
  generic `Endpoints-N` naming for counts outside this range.
- **`VesselAnalyzerWidget.onAutoDetect()`** — thin override that calls the
  original `CenterlineWidgetMixin.onAutoDetect` then invokes
  `_labelEndpointsAfterAutoDetect()` to apply anatomical labels without
  modifying the mixin.
- **`VesselAnalyzerWidget._labelEndpointsAfterAutoDetect()`** — locates the
  active `vtkMRMLMarkupsFiducialNode` (via `self._endpointsNode` or scene
  scan for nodes named `Endpoints*`), calls `_assignAnatomicEndpointLabels`,
  and refreshes the seed-quality panel.
- Build tag bumped to **v294-AnatomicEndpointLabels**.

### Log tags
```
[EndpointLabel] Point N → Endpoints-M (Anatomy name)
[EndpointLabel] No Endpoints node found after auto-detect.
[EndpointLabel] N endpoints found — expected 4 or 5. Skipping …
[EndpointLabel] Error during anatomical label assignment: <exc>
```

### No downstream changes
Centerline extraction, diameter pipeline, stent/report logic, and all
existing widget mixin methods are unchanged.

---



### PerBranchCenterline: replace single global VMTK solve with hybrid per-branch pipeline;
Phase 1: global solve used only to extract snappedBifPt + trunk direction (branch centerlines discarded);
Phase 2: independent VMTK solve per branch with (p_src=snappedBifPt-trunkDir*3r, p_tgt=boundary ring centroid)

### no bifurcation cost-field competition, no trunk stealing iliac, no renal drift along IVC wall;
Phase 3: tangent-preserving junction snap forces all branch curves to exact shared bif point via smoothstep blend over 2*branch_radius

### C1 at junction, no kinks;
Phase 4: light refinement via existing _refineSingleCurve (n_iters=8, lr_decay=1.0, w_j=0, first point re-pinned after each iter);
integration: onExtractCenterline now imports PerBranchCenterlinePipeline from per_branch_centerline_pipeline.py (sibling file) and calls pipeline.run(endpointsNode)

### merged node passed to loadCenterline unchanged;
downstream pipeline (diameter, navigator, stent, report) zero changes;
eliminates root cause of most v236-v274 compensating heuristics (TrunkZFix, RenalConsolidate, FlowNormSideBranch, BifTrunc complexity now largely redundant);
v274-RefineGoodBudgetFix: fix plain Refine button diverging on freshly-densified GOOD curves;
root cause: GOOD branch call passed n_iters=15 overriding the function default of 30

### 15 iters is sufficient for AutoTune trials that start near-converged but not for fresh geometry starting far from the wall (CL_1 arc=192mm, 13 free pts still moving at 0.82mm/iter at iter 14);
optimizer exhausted budget mid-settling producing tortuosity 1.11→4.22 and wall_dist_mean 1.23→4.66mm;
fix: remove n_iters=15 override from GOOD branch call so _refineSingleCurve uses its default n_iters=30;
WEAK branch at line 2905 retains n_iters=15 (intentional reduced budget for unstable seeds);
consistent with AutoTune final-apply path which already restores n_iters=30 at best_params restore;
fix plain Refine button diverging on freshly-densified GOOD curves;
root cause: GOOD branch call passed n_iters=15 overriding the function default of 30

### 15 iters is sufficient for AutoTune trials that start near-converged but not for fresh geometry starting far from the wall (CL_1 arc=192mm, 13 free pts still moving at 0.82mm/iter at iter 14);
optimizer exhausted budget mid-settling producing tortuosity 1.11→4.22 and wall_dist_mean 1.23→4.66mm;
fix: remove n_iters=15 override from GOOD branch call so _refineSingleCurve uses its default n_iters=30;
WEAK branch at line 2905 retains n_iters=15 (intentional reduced budget for unstable seeds);
consistent with AutoTune final-apply path which already restores n_iters=30 at best_params restore;
v268 LrDecayTrialFix: fix CL_1 divergence in AutoTune final apply pass and CL_5 never converging;
root cause: lr_decay=0.92 was applied uniformly to all calls including AutoTune trials

### trials start near-converged (small residual) so decay is irrelevant, but the final apply starts from fresh geometry with large initial displacement;
with high w_j=7.24 selected by AutoTune, the junction anchor fights the wall-centering term and the decaying lr shrinks the corrective step while junction restoring force stays proportional, causing monotonic divergence from iter 5 onward;
fix: (1) lr_decay is now read from params dict via p.get('lr_decay', 0.94)

### default 0.94 for plain Refine button;
(2) AutoTune trial sample dict sets lr_decay=1.0 (no decay)

### trials are 8 iters starting near-converged, decay only corrupts scoring fidelity;
(3) best_params gains lr_decay=0.94 alongside n_iters=20 when restored for final apply

### full decay re-enabled for the geometrically meaningful final pass;
decay also slowed 0.92→0.94 to give more room in the tail for branches like CL_5 that approach threshold asymptotically;
v267 ModelSelectorRestore unchanged

### v267

### fix has_model=False on all branches after scene reload causing avg_movement=0.0000 and AutoTune score=+0.000 on every trial;
root cause: qMRMLNodeComboBox loses its selection after scene reload so modelSelector.currentNode() returns None in _refineSingleCurve

### wall-centering term has no polydata to project against, gradient is zero, optimizer exits immediately;
fix: extend _recoverManualCurves (called from enter() and _onSceneImported) with a second pass that checks if modelSelector.currentNode() is valid;
if not, scans all vtkMRMLModelNode nodes in the scene and auto-selects the one with the most polydata points (the segmentation surface);
[ManualCenterline] modelSelector restored -> 'NodeName' (N pts) log confirms;
left alone if selector already has a valid node;
v266 ManualCurveRestore unchanged

### v266

### fix Refine/Clear/Auto-Tune always disabled until a new curve is drawn;
root cause: _manualCurves lived only in Python widget memory

### lost on scene reload AND on fresh Slicer open with existing scene;
fix: (1) new _recoverManualCurves() helper scans mrmlScene for all vtkMRMLMarkupsCurveNode nodes named ManualCenterline_*, sorts by name, repopulates self._manualCurves, and re-enables buttons;
(2) enter() calls _recoverManualCurves() every time the module tab is selected

### covers fresh open, tab-switch, and any other entry;
(3) _onSceneImported now calls _recoverManualCurves() after _force3DLayout() instead of duplicating the logic;
[ManualCenterline] Recovered N curve(s) log confirms state on every entry;
v265 SceneReloadRestore folded into this fix

### v264

### fix Refine/Clear/Auto-Tune disabled after save-and-reload;
root cause: _manualCurves lived only in Python widget memory

### on scene reload the ManualCenterline_* markup nodes came back from the .mrb file but _manualCurves was empty so all three buttons stayed disabled;
fix: extend _onSceneImported (already fires on EndImportEvent) to scan mrmlScene for all vtkMRMLMarkupsCurveNode nodes whose name starts with "ManualCenterline_", sort by name to preserve draw order, repopulate self._manualCurves, and re-enable refineCenterlineButton/autoTuneButton/clearManualCenterlineButton when ≥1 curve with ≥2pts each is recovered;
[ManualCenterline] log line confirms recovery count and enable state;
v264 LrDecay unchanged

### v264

### fix CL_1 divergence in AutoTune final apply pass and CL_5 never converging;
root cause: lr_decay=0.92 was applied uniformly to all calls including AutoTune trials

### trials start near-converged (small residual) so decay is irrelevant, but the final apply starts from fresh geometry with large initial displacement;
with high w_j=7.24 selected by AutoTune, the junction anchor fights the wall-centering term and the decaying lr shrinks the corrective step while junction restoring force stays proportional, causing monotonic divergence from iter 5 onward;
fix: (1) lr_decay is now read from params dict via p.get('lr_decay', 0.94)

### default 0.94 for plain Refine button;
(2) AutoTune trial sample dict sets lr_decay=1.0 (no decay)

### trials are 8 iters starting near-converged, decay only corrupts scoring fidelity;
(3) best_params gains lr_decay=0.94 alongside n_iters=20 when restored for final apply

### full decay re-enabled for the geometrically meaningful final pass;
decay also slowed 0.92→0.94 to give more room in the tail for branches like CL_5 that approach threshold asymptotically;
v267 ModelSelectorRestore unchanged

### v267

### fix has_model=False on all branches after scene reload causing avg_movement=0.0000 and AutoTune score=+0.000 on every trial;
root cause: qMRMLNodeComboBox loses its selection after scene reload so modelSelector.currentNode() returns None in _refineSingleCurve

### wall-centering term has no polydata to project against, gradient is zero, optimizer exits immediately;
fix: extend _recoverManualCurves (called from enter() and _onSceneImported) with a second pass that checks if modelSelector.currentNode() is valid;
if not, scans all vtkMRMLModelNode nodes in the scene and auto-selects the one with the most polydata points (the segmentation surface);
[ManualCenterline] modelSelector restored -> 'NodeName' (N pts) log confirms;
left alone if selector already has a valid node;
v266 ManualCurveRestore unchanged

### v266

### fix Refine/Clear/Auto-Tune always disabled until a new curve is drawn;
root cause: _manualCurves lived only in Python widget memory

### lost on scene reload AND on fresh Slicer open with existing scene;
fix: (1) new _recoverManualCurves() helper scans mrmlScene for all vtkMRMLMarkupsCurveNode nodes named ManualCenterline_*, sorts by name, repopulates self._manualCurves, and re-enables buttons;
(2) enter() calls _recoverManualCurves() every time the module tab is selected

### covers fresh open, tab-switch, and any other entry;
(3) _onSceneImported now calls _recoverManualCurves() after _force3DLayout() instead of duplicating the logic;
[ManualCenterline] Recovered N curve(s) log confirms state on every entry;
v265 SceneReloadRestore folded into this fix

### v264

### fix Refine/Clear/Auto-Tune disabled after save-and-reload;
root cause: _manualCurves lived only in Python widget memory

### on scene reload the ManualCenterline_* markup nodes came back from the .mrb file but _manualCurves was empty so all three buttons stayed disabled;
fix: extend _onSceneImported (already fires on EndImportEvent) to scan mrmlScene for all vtkMRMLMarkupsCurveNode nodes whose name starts with "ManualCenterline_", sort by name to preserve draw order, repopulate self._manualCurves, and re-enable refineCenterlineButton/autoTuneButton/clearManualCenterlineButton when ≥1 curve with ≥2pts each is recovered;
[ManualCenterline] log line confirms recovery count and enable state;
v264 LrDecay unchanged

### v264

### TrunkZFix2: fix bridge path orientation bug in TrunkZ correction;
in v260 the bridge path [1096,822,734,207] was being reversed before prepending to trunkNodePath

### producing [734,822,1096,207,547] (wrong, starting from iliac tip);
fix: use bridge path as-is (it already runs maxZ_ep→join correctly from BFS) and append the trunk tail from join onward, giving correct [1096,822,734,207,547];
also add [TrunkZ] edge-by-edge cEdge validation debug log and [BifTrunc] Z-range log so the trunk geometry can be fully audited in console;
v260 TrunkZFix+BranchStartNavFix+ManualBranchRanges unchanged

### fix trunk misidentification in venous anatomy where one iliac limb has higher usage than the true IVC trunk above the bifurcation;
root cause: max-usage edge scoring anchors the trunk on whichever path is traversed by the most endpoint-pair BFS routes

### in a 5-endpoint IVC tree with one iliac endpoint, the iliac→bif path gets usage=6 while the true IVC inlet→bif segment gets usage=4, so the trunk runs caudal-to-cranial (iliac tip→bif) and the IVC inlet becomes a "side branch";
invariant violated: the true trunk MUST pass through the highest-Z endpoint (IVC inlet / aortic root) because that is the proximal anchor of the vessel tree;
fix: after _longestPath builds trunkNodePath from usage-based edges, check if the highest-Z realEndpoint node is in the path;
if not, run BFS over _allEdgeAdj from that endpoint to find the shortest node path to any existing trunk node, then prepend the bridge to rebuild trunkNodePath so it runs from the IVC inlet through the primary bifurcation;
bridge nodes are absorbed into trunkNodes/trunkAdj so the geometric cEdge stitching loop finds the correct segments;
[TrunkZ] log tag shows whether the correction fired and which node/Z was the inlet;
expected: trunk now runs S=1969.9→S=1780.1 (IVC inlet→bif), Right Iliac runs S=1779.8→S=1666.8, Left Iliac runs S=1779.5→S=1672.2, side branches correctly identified;
BranchStartNavFix: getBranchStartGi priority manualStartGi→ostiumGi→raw, stableStartGi excluded;
ManualBranchRanges: per-branch coordinate override UI + _applyManualRanges pipeline step

### (1) BranchStartNavFix: fix branch navigator start point showing too far distal;
root cause: getBranchStartGi was using stableStartGi as highest priority

### stableStartGi is the post-flare diameter anchor (several mm distal to the anatomical ostium) and was never intended as the navigation start;
fix: reorder priority

### manualStartGi first, then ostiumGi, then raw branches[bi][0];
stableStartGi explicitly excluded from getBranchStartGi and used only in _annotateTreeDiameters/getBranchStats for diameter measurement;
(2) ManualBranchRanges: per-branch start/end coordinate override system;
new self.manualBranchRanges dict on VesselAnalyzerLogic stores bi → {start:(x,y,z), end:(x,y,z)} in RAS mm;
new _applyManualRanges() method runs after _suppressRejectedOstia in the pipeline

### for each entry it searches within the branch's raw gi range for the nearest point to the specified coordinate and stores manualStartGi/manualEndGi in branchMeta;
getBranchStartGi checks manualStartGi first (highest authority above ostiumGi);
getActiveBranchPointCount checks manualEndGi first (uses manualEndGi+1 as exclusive end);
new setManualBranchRanges(dict) public method stores the dict and calls _applyManualRanges immediately if points are already loaded;
new UI collapsible "📐 Manual Branch Ranges" between Measurements and Visualization Options

### five rows (Trunk, Left Iliac, Right Iliac, Branch 4, Branch 5) each with Start and End QLineEdit fields accepting paste from the 📋 copy button in R x.x, A y.y, S z.z | Ø d.d mm format;
Apply button calls onApplyManualRanges which parses coords, resolves bi from branchMeta roles (trunk=0, iliac_left/right by role, branch_4/5 by index), calls setManualBranchRanges, and refreshes the slider maximum;
Clear button wipes all manualStartGi/manualEndGi from branchMeta;
status label shows applied ranges or parse errors;
[ManualRange] log tag shows gi snap result with distance in mm for every branch

### ManualCenterlineAndExtractionFix: (1) ManualCenterlineDraw: new UI section in Step 2

### "Draw Centerline Manually";
user clicks points along vessel lumen in 3D view using Slicer markup placement mode;
curve is stored as vtkMRMLMarkupsCurveNode "ManualCenterline";
resampled to 1mm spacing on commit;
"Use This Centerline" button sets it as the active centerline node and loads it directly into loadCenterline (already accepts vtkMRMLMarkupsCurveNode);
use case: correct VMTK medial axis errors at the bifurcation zone where the centerline drifts to one wall;
workflow: Draw -> place points proximal-to-distal -> Stop Drawing -> Use This Centerline -> Load for IVUS Navigation;
(2) ExtractionQualityFix: VMTK extractCenterline preprocessing improved for better bifurcation medial axis accuracy;
targetPoints raised min(user,8000) (was fixed 5000)

### more mesh detail = more accurate Voronoi diagram at the carina;
decimationAggressiveness reduced 4.0->3.5

### less decimation preserves bifurcation geometry;
subdivide changed False->True

### subdivision helps medial axis stay centered in the lumen;
curveSamplingDistance reduced 1.0mm->0.5mm

### finer sampling reduces centerline drift artifacts between sample points;
EctasiaAndVesselTypeFix: (1) EctasiaDomeExempt: bifurcation dome suppression was silently hiding real ectasia at the IVC/iliac confluence;
root cause: _BIF_DOME_MM=25mm covered the entire carina region and suppressed any smooth dilation within that radius regardless of severity grade;
two fixes: (a) dome radius reduced 25mm→10mm in venous mode

### IVC confluence dome is anatomically ~9mm (BIF_DOME_MM=9mm for ostium walk), not 25mm;
the old 25mm radius was borrowed from aortic bifurcation geometry and was far too large for venous anatomy;
(b) ectasia-grade dilation (rel_max < _ANEURYSM_RATIO) is now EXEMPT from dome suppression entirely

### venous ectasia at the confluence is a clinically meaningful finding indicating post-thrombotic remodeling or chronic venous hypertension, not physiologic Y-expansion;
only aneurysm-grade smooth expansion (rel_max >= 2.0x) is still suppressible within the dome;
expected: ectasia between Z=1769–1780 (5mm from bif) now detected and rendered;
(2) VesselTypePrint: vessel type now printed prominently as [VesselAnalyzer] *** VESSEL TYPE: VENOUS *** at every analysis run, immediately after logic instantiation;
prevents ambiguity when reviewing console logs;
two rendering bugs in applyColorOverlay prevented Focal Narrowing and Eccentric Compression findings from displaying correctly in the 3D view;
(1) FocalNarrowingColor: "Focal Narrowing" type (produced by FindingsPreFlag for compression-like narrowings e.g. May-Thurner on Branch 4) was absent from TYPE_COLORS dict

### fallback color (0.5,0.5,0.5) grey was used and the label rendered with no clinical color coding;
fix: added "Focal Narrowing": (0.90,0.10,0.10) red to TYPE_COLORS, consistent with stenosis severity;
(2) EccentricLabelFix: eccentric iliac compression findings are stored with type="Mild Compression" (intentional

### stent sizing logic keys on this type) but source="eccentric";
applyColorOverlay was rendering them as yellow "Mild Compression" markers rather than the distinct orange-red "Eccentric Compression" color already defined in TYPE_COLORS;
fix: derive ftype_display from source field

### if source=="eccentric" override display type to "Eccentric Compression" for both color lookup and SetNthControlPointLabel;
the stored f["type"] is unchanged so stent sizing and report generation are unaffected;
node name also updated to use ftype_display for scene clarity;
expected: Focal Narrowing on Branch 4 (May-Thurner lesion, 51% drop, 58.5mm run) now shows as red marker with "Focal Narrowing" label;
Left/Right Iliac eccentric compressions now show as orange-red markers with "Eccentric Compression" label instead of yellow "Mild Compression";
IliacZGateFix: Left Iliac (and Right Iliac) were silently skipped by lesion detection;
root cause: after IliacOstiumMaxZ/BIF_DOME_MM walk, ostiumGi is set to the MOST CRANIAL (max-Z) point on the iliac CL, and self.branches[bi] start is then trimmed to that ostiumGi

### so `s` in the _detectFindings loop == the cranial apex Z, which can equal or exceed trunk_root_z (the proximal aorta Z);
the guard `branch_start_z > trunk_root_z + 10.0` then fires and skips the entire Left Iliac (Right Iliac may survive if its apex is slightly lower);
fix: (1) confirmed iliac roles (main/iliac_left/iliac_right) are EXEMPT from the Z-gate entirely

### they are anatomically anchored to the primary bifurcation and can never be artifacts;
(2) for all other branch roles the Z-gate now uses _rawBranches[bi][0] (pre-trim graph-topology start) instead of the post-trim `s`

### this prevents renal/side branches whose ostium trimming also shifts `s` cranially from being incorrectly gated;
fix applied in both _detectFindings and _computeStableStarts which carry identical Z-gate logic;
log tag [DetectFindings] Branch N: skipped

### raw_start_z=X > trunk_root_z=Y+10 now printed for any branch that IS skipped so the gate is auditable;
expected: Left Iliac lesion detection now runs and reports narrowings;
Right Iliac unaffected (was passing before, still passes);
— v257

### FindingsDetectionFix: two root-cause fixes for lesion scanner producing zero findings despite known 51% compression-like narrowing on Branch 3 (renal vein);
(1) EllipseGateRemoved: stenosis candidate gate `(rel < 0.83) AND (ratio >= 1.1)` changed to `(rel < 0.83)` only

### venous compression is typically concentric (uniform lumen reduction → ellipse ratio ≈ 1.0), so requiring ratio >= 1.1 silently blocked all run-scanner findings for the most common venous pathology;
ellipse ratio is still used for classification (Pancaking vs Mild Compression) but no longer as an entry gate;
(2) ProxRefAnchor: reference diameter for run-scanner changed from `_mean_ref_d` (distal 55-85% trimmed mean) to `diam_prox` (post-flare proximal anchor from _annotateTreeDiameters) when diam_prox > _mean_ref_d × 1.10;
root cause: for diffuse compression spanning the distal reference window (Branch 3: 51% narrowing, 58.5mm run, 29% of branch arc) the distal window sits inside the narrowed zone and returns the compressed diameter as the reference

### rel = d / compressed_ref ≈ 1.0 everywhere → no flags;
diam_prox is anchored to clean post-flare tissue proximal to the narrowing and is always the pre-compression calibre;
fallback chain unchanged (sc_ref → _mean_ref_d → trunk-scaled median);
[FindingsRef] log tag shows when prox/sc ref overrides the distal window

### v246

### IliacSurfaceSnapSelectionFix: fix two wrong-vertex bugs in v245 IliacSurfaceSnap;
(1) bi=1 Left Iliac far-wall bug: argmax(dot) selected the outermost iliac wall vertex (R=8.7 far side) instead of the proximal rim facing the IVC

### fix: among candidates with positive lateral dot (correct hemisphere), select the one nearest to bifurcationPoint in 3D (proximal rim = smallest distance to bif, not max dot);
label changed to iliac_surf_proxrim;
(2) bi=2 Right Iliac degenerate radial: CaudalDepartureSnap sets meta['ostium'] to a trunk CL point

### using it as the radial origin gave trunk→trunk = zero vector, triggering degenerate-radial fallback;
fix: always compute radial from self.points[ogi] (branch CL at ostiumGi, always offset from trunk axis) regardless of whether CaudalDepartureSnap ran;
Z-band raised 5→6mm (iliac dome is shallower than renal takeoff);
IliacSurfaceSnapFix: move IliacSurfaceSnap from loadCenterline into _refineOstia Gate-1 block;
root cause of v244 CL-fallback: snap was running inside loadCenterline before _refineOstia/_computeDiameters/classifySurfaceBranches

### the Voronoi surface KDTrees (_surf_kd_per_branch) do not exist yet at that point, so the try-block silently fell through to CL fallback;
fix: snap now runs inside _refineOstia when is_main=True (Gate 1 skip), where _surf_kd_per_branch[bi] is already built from the correct segmentation surface;
loadCenterline commit block reverted to CL point only (ostium/ostium_p3 = ostiumPt tuple), _refineOstia overwrites ostium/ostium_p3 with the wall vertex;
IliacSurfaceSnap: snap iliac ostium display coordinate to the nearest iliac-classified surface vertex so the green marker overlaps the branch segmentation;
root cause of marker/surface mismatch: the iliac dome-walk committed ostiumPt=bPts[display_k] (VMTK medial-axis CL point, ~5-7mm medial to the wall in R) and set ostium_p3=None, meaning the marker rendered at the lumen centre, not on the wall;
fix: after locking ostiumGi/stableStartGi (diameter pipeline unchanged), build a local Voronoi surface assignment (same logic as classifySurfaceBranches/_refineOstia), collect iliac-classified verts (NOT trunk-classified

### trunk surface at ostium Z is the shared carina face), apply +-5mm Z-band filter at ostiumPt[2], then select the candidate most lateral to the IVC trunk axis in the XY plane via radial dot product (trunk CL at that Z -> ostiumPt, Z-component zeroed);
result written to both ostium and ostium_p3;
20mm cap rejects misassigned verts with CL fallback;
log tag [IliacOstium] bi=N IliacSurfaceSnap shows surface=iliac_surf_lateral/CL_fallback, CL_pt, wall_pt, snap_d

### IliacOstiumMaxZ: fix Pass B of the iliac dome walk;
root cause confirmed from [IliacOstium] debug log: Pass B was selecting the point FURTHEST in 3D distance from snappedBifPt within the arc cap, but the right iliac CL curves cranially (Z increases) then swings laterally and descends, so bPts[21] at Z=1776 is 23.1mm from bif in 3D yet LOWER in Z than bPts[7] at Z=1779 (the true cranial apex);
the old max-3D criterion picked bPts[21]=(7.6,150.9,1776.0) instead of the correct apex;
fix: Pass B now selects the point with MAXIMUM Z (most cranial) within the arc cap, gated by d_bif >= BIF_DOME_MM to stay past the shared dome zone;
expected: right iliac display_k advances to the ki with Z≈1783.5 (the user-confirmed ostium), left iliac unaffected (its dome walks monotonically in 3D so max-Z and max-3D agree);
all [IliacOstium] debug tags from v242-IliacDebug retained so the PassB log now shows max_Z= instead of d_bif_apex=

### IliacDebug: add dense [IliacOstium] debug tags throughout the iliac ostium pipeline to diagnose why the right iliac ostium still shows Z=1774.9 instead of Z=1783.5 after the v242 IliacOstiumApexFix;
(1) [IliacOstium] Loop: printed for every branch at dome-walk entry

### shows bi, role, in_mainSet, full mainSet contents, bPts_len, bPts[0] coord, snappedBifPt so we can confirm whether the right iliac branch reaches the _is_main_branch path;
(2) [IliacOstium] PassA: printed at the first_k commit

### shows ki, arc, d_bif at crossing;
(3) [IliacOstium] PassB: printed after display_k resolves

### shows both first_k coord and display_k coord so the apex vs first-crossing difference is immediately visible;
(4) [IliacOstium] COMMITTED: printed at ostiumGi write

### shows ostiumGi, bs_gi, be_gi, final ostiumPt;
(5) [IliacOstium] POST-TRIM: full ostiumGi audit printed for every branch after the trim block;
(6) [IliacOstium] FINAL PIPELINE: printed after _suppressRejectedOstia

### shows every branch's final ostiumGi, stableStartGi, and the actual 3D coordinate + diameter at that gi (ground truth of what the navigator will display);
all existing IliacOstiumApexFix logic from v242 retained unchanged

### IliacOstiumApexFix: right iliac ostium was landing at the first-crossing point (Z=1774.9, Ø=12.06mm) instead of the anatomically correct cranial apex (Z=1783.5, Ø=13.31mm);
root cause: the BIF_DOME_MM dome walk correctly computes display_k (furthest point from bif within arc cap = true cranial apex) and first_k (first point ≥9mm from bif = first crossing), but was using first_k for ostiumGi/branch-trim and display_k only for the visual marker

### so the navigator coordinates came from first_k, not the apex;
fix: use display_k for both ostiumGi and ostiumPt;
_rawBranches freezes the traversal pre-trim so this is navigation-safe;
expected: right iliac navigator shows R 25.9, A 146.2, S 1783.5 Ø 13.31mm;
UIMove: move "👁 Showing: Model/Surface" toggle button from Visualization Options collapsible to Branch Surface Classification collapsible, directly below the "Inspect Ostium Detection" button

### visually logical (surface toggle lives with surface controls);
NavRobust guards from v241 retained unchanged

### NavRobust: defensive guards added to navigator code paths so that any future change to ostium/branch/traversal logic cannot crash or silently corrupt navigation;
changes are PURELY defensive

### zero logic changes to traversal, _rawBranches, _travSnapshot, getBranchForPoint, localToGlobal, or globalToTraversal;
(1) getMeasurementAtIndex: guard distances/diameters array bounds before indexing (len check before distances[realIdx] and distances[traversal[0]]);
if distances is shorter than points (can happen if loadCenterline partially fails), returns a safe zero-distance result rather than IndexError;
(2) onSliderChanged jump guard: the prevIdx localToGlobal call is now wrapped in try/except so a crash inside the guard cannot block the slider update

### navigation always proceeds even if the jump-guard check throws;
(3) getBranchForPoint: explicit fallback log suppressed (already returns -1, now documented);
(4) localToGlobal: added guard so a None traversal or out-of-range localIdx always returns localIdx rather than raising;
all changes are additions only

### no existing logic deleted or altered

### TraversalFullyFrozen: make traversal count 100% invariant to any ostium-detection change;
root cause of remaining drift: traversal builder was still reading live branchMeta for (1) branch roles

### _refineOstia can demote/promote roles after traversal build;
(2) renal ostium coordinates

### _stabilizeRenalOstia/_refineOstia rewrite branchMeta['ostium'] after traversal is built, shifting the trunk walk-up length;
fix: (1) add _travSnapshot dict captured PRE-TRIM alongside _rawBranches

### stores frozen role + frozen ostium_pt (branchMeta ostium if set, else raw branch start coordinate) for every branch;
(2) traversal builder reads ONLY _travSnapshot via _tsnap_role()/_tsnap_ostium_pt() helpers

### zero live branchMeta reads remain in the traversal code path;
(3) renal sort order changed from ostiumGi (can drift) to raw branch start gi (frozen);
(4) _travSnapshot re-filtered through _kept_branch_old_bis after discard compaction same as _rawBranches;
net result: traversal count is now determined solely by (a) graph topology output and (b) the raw branch start coordinate used as renal ostium anchor

### both fixed at graph-walk time, completely unaffected by BIF_DOME_MM / DIVERGE_ARC_MM / _refineOstia / _stabilizeRenalOstia or any other ostium algorithm

### v238

### RenalWalkUpFix: fix renal walk_up=0 bug that caused traversal count to be 835 instead of ~971;
root cause confirmed from [Nav] debug: renal branch gi range (407..456) is SEPARATE from trunk gi range (0..150)

### they are concatenated sequentially in rawPoints, so the old check `if _t_bs <= _ogi < _t_be` ALWAYS evaluated False (408 is not in 0..150) and walk_up/walk_down were always set to [] regardless of anatomy;
fix: replace gi-range check with coordinate-based nearest-trunk-point lookup

### find the trunk gi whose 3D coordinate is nearest to the renal ostium coordinate (from branchMeta['ostium'], fallback to raw branch start point), compute walk_steps = bif_gi - nearest_trunk_gi, slice trunk_rev accordingly;
expected new total ~971 = 835 (old) + ~68 walk_up + ~68 walk_down;
new [Nav] debug line shows nearest_trunk_gi / dist_mm / walk_steps so the walk geometry is fully auditable

### v237

### TraversalDebug: fix _rawBranches snapshot timing (was after discard, now before ostium-trim) + add _kept_branch_old_bis mapping so traversal builder correctly re-filters snapshot after discard compaction + add per-branch [Nav] debug lines showing raw gi range / ostiumGi trim / running total so the source of any count drift is immediately visible in the console

### v236

### fix centerline point count changing between edits;
root cause: _allBranchTraversal was built from self.branches[bi] start indices AFTER the ostium-trim step (which sets branches[bi][0]=ostiumGi)

### any change that shifts ostiumGi (e.g. BIF_DOME_MM walk, DIVERGE_ARC_MM cap, _refineOstia) directly changed how many points each branch contributed to the traversal and the total count drifted from the baseline 934;
fix: snapshot self.branches into self._rawBranches immediately before the traversal build (post-discard, pre-refine), and use _rawBranches exclusively in the traversal construction for trunk, main/side, and renal branches;
all subsequent ostium-refinement trims continue to write to self.branches[bi] as before

### only the traversal is insulated from those changes;
traversal count is now determined solely by the raw graph topology output and will remain 934 regardless of which ostium-detection algorithm is active

### v235

### IliacOstiumBifDist: replace sibling-separation walk (DIVERGE_SEP_MM) with direct bif-distance walk (BIF_DOME_MM=9mm) for main Y-bif branches;
root cause of v234 failure: both iliacs share a cranial dome and enter the same cranial zone before diverging

### the closest sibling point at the right iliac cranial apex (26.2,145.9,1782.6) is only ~3.9mm from the left iliac proximal pts (which also curve cranially to Z≈1779), so the 8mm sep threshold fails to fire at the anatomically correct apex and instead fires on the caudal leg of the right iliac (Z=1767.6, 17.9mm from target);
sibling-separation is fundamentally unsuitable for shared-dome geometry where both branches approach each other before diverging;
fix: walk bPts until d(bPts[k], snappedBifPt) >= BIF_DOME_MM=9mm

### snappedBifPt is the fixed carina centroid (bif geometry, not topology);
the first point beyond 9mm is the edge of the shared dome on each branch's own CL path;
right iliac target at d=9.1mm ≥ 9mm fires at correct cranial apex (Z≈1782);
left iliac fires at similar arc;
hard cap DIVERGE_ARC_MM=25mm retained as safety net;
all sibling pre-collection code removed (no longer needed)

### IliacOstiumSepFix: fix Right Iliac ostium still landing at bPts[1] (Z=1773.3) despite v233 Z-guard patch;
root cause of v233 failure: the correct Right Iliac ostium at Z=1782.6 is 7.7mm CRANIAL to snappedBifPt Z=1774.9

### the Z-guard `if _p[2] >= _bif_z: continue` blocks the correct answer because it is proximal in Z;
simultaneously the guard forced Left Iliac to over-walk to arc=25.7mm;
both failures traced to same incorrect assumption that the ostium must be caudal to bif_Z;
actual constraint: the two branches must be physically separated in 3D space;
fix: (1) remove Z guard entirely;
(2) raise DIVERGE_SEP_MM 2mm→8mm (≈half the bif-zone vessel diameter of 13-15mm);
at 2mm both bPts[1] values (d=2.2mm and d=2.7mm from bif) immediately satisfy the threshold

### same as before;
at 8mm both are rejected and the walk continues until the branches are genuinely independent in 3D space;
Right Iliac correct ostium at d=9.1mm from bif → committed at first point ≥8mm → expected Z≈1782;
Left Iliac commits at d≈8mm → arc≈7-10mm, well short of the 25.7mm over-walk;
DIVERGE_ARC_MM=25mm hard cap unchanged as safety net

### ThreeBugFix: (1) RightIliacOstiumDistalGuard: diverge walk now requires candidate point to be distal (lower Z) to snappedBifPt Z before accepting sibling-separation;
root cause: bPts[1] at the shared graph node satisfied DIVERGE_SEP_MM=2mm via topology alone while still being 1.6mm proximal to bif Z

### adding `if _p[2] >= _bif_z: continue` forces the walk to advance past the true anatomical carina before committing;
expected: Right Iliac ostium advances from Z=1773.3 to Z≈1782.6 (~8pts further distal, gi≈303);
(2) RenalSnapSourceAdaptive: snap query origin now switches from ogi to detect_gi when detect_gi is ≥8mm distal to ogi in arc distance;
root cause: ogi=416 (RenalStabilizer wall-contact anchor) is inside the IVC wall zone where the radial direction is nearly tangential (top-5 dots≤0.11)

### snap lands on a trunk-wall face 5mm proximal to snap_src;
detect_gi=423 (confirmed lumen-separation point) is 8.2mm past ogi and correctly in the lateral takeoff zone;
switching snap_src to detect_gi gives a valid outward radial and the trunk surface candidate at the true ostial rim;
expected: Branch 4 snap Z advances from 1870.5 toward 1892.8;
diameter at ostium corrects from 15mm toward 9.7mm;
(3) NavTraversalDualFix: (a) side/short branches in _main_branches no longer receive a ret_to_bif=[bif_gi] bridge

### only iliac-role branches (main/iliac_left/iliac_right) are anatomically connected to the primary bifurcation;
appending bif_gi for side branches created duplicate traversal entries causing globalToTraversal to resolve to position 0 (trunk start) instead of the correct slider position;
(b) globalToTraversal rewritten to prefer the first outbound occurrence of gi (where traversal[i+1]>gi) rather than absolute first

### eliminates up/down asymmetry when bif_gi still appears multiple times as iliac ret_to_bif bridges;
fallback to absolute first if no outbound occurrence found

### SnapDecoupling: decouple snap query origin from diameter anchor;
root cause of v231 failure confirmed from log: shifting detect_gi 423→434 moved Z-matched trunk CL reference past the lateral takeoff zone

### radial direction flipped inward (all top-5 dots negative: -0.67 to -0.73), eff collapsed 0.428→0.072 REJECT;
the core problem is using detect_gi as BOTH the diameter pipeline anchor AND the wall snap query origin;
fix: _snap_gi always uses ogi (geometric wall-contact anchor

### last point within 2.5mm of trunk KDTree, set by RenalStabilizer for renals or _refineOstia for sides) while _detect_gi remains the diameter/stableStart anchor;
at ogi the branch CL is still in the IVC wall zone so the Z-matched trunk CL reference is correctly lateral to the branch CL and the radial direction points outward;
[OstiumSnap] log now shows both snap_gi (ogi) and detect_gi separately for full auditability;
WEAK_ENTER drift block removed (reverted from v231);
all other v230 debug layers (DetectGI/OstiumSnap/OstiumQuality) retained unchanged

### WeakEnterDrift: controlled detect_gi shift for WEAK_ENTER branches;
root cause confirmed by [DetectGI] debug: Branch3 gap=18gi (enter=423,min=441), Branch4 gap=14gi (enter=468,min=482)

### ratio crosses 0.75 at the flare rim then keeps dropping to true lumen floor;
jumping to min_ratio_gi would overshoot into branch lumen;
fix: when mode==WEAK_ENTER AND gap>GAP_THRESH(10gi) AND enter_ratio>0.70, shift detect_gi forward by alpha*gap where alpha=clamp((enter_ratio-min_ratio)/enter_ratio, 0.20, 0.60)

### adaptive: bigger ratio drop → bolder shift;
safety clamp to min_ratio_gi-SAFETY_MARGIN(3gi) prevents overshoot;
expected results: Branch3 423→~430 (gap=18,drop=0.42,alpha=0.59→ceil@438);
Branch4 468→~474 (gap=14,drop=0.22,alpha=0.30→+4gi);
ENTER-mode and FALLBACK-mode detections left completely unchanged;
[DetectGI] drift log shows gap/alpha/enter_gi/shifted_gi/ceiling for full auditability

### DecisionDebug: three structured debug layers added to ostium pipeline;
(1) [DetectGI] block after detect_gi resolution: fields branch/mode/enter_gi/enter_ratio/min_ratio_gi/min_ratio/thresh/scan/confirm_pts

### mode=ENTER (confirmed crossing), FALLBACK (ratio never dropped, min-ratio used), WEAK_ENTER (confirmed crossing before global minimum

### noisy scan);
replaces manual ratio reading and immediately shows if fallback fires;
(2) [OstiumSnap] block restructured with fields branch/gi/ratio/thresh/lat/surface/CL_pt/wall_pt/snap_d/delta_mm/stability

### delta_mm=CL→wall displacement, stability=stable(<2mm)/drifting(2–8mm)/far(>8mm);
exposes snap drift for horizontal-ring branches like Branch5;
(3) [OstiumQuality] block in _suppressRejectedOstia: fields branch/eff/raw/pen/spread/reject_thresh/state/dominant_flag/all_flags

### state=LOW_CONF or VERY_LOW, dominant_flag=highest-penalty flag driving the REJECT grade;
makes confidence formula auditable without reading the per-branch venous breakdown lines;
all three layers are log-only with zero impact on computed values

### OstiumProximalTwoFix: (1) DIAM_ENTER_THRESH lowered 0.80→0.75: Branch4 ratio was exactly 0.80 at the boundary so < 0.80 never fired and the min-ratio fallback placed detect_gi ~4mm too proximal (S=1889.5 vs target S=1893.9);
at 0.75 the forward scan walks further into clean branch lumen before committing;
(2) REJECT suppression removed from _suppressRejectedOstia: previously ostiumGi and ostium_p3 were cleared to None for REJECT-grade branches causing no marker at all

### for Branch5 this left the display anchored at ogi inside the IVC flare zone (too proximal);
new behaviour keeps ostiumGi and ostium_p3 intact and copies to ostium_p3_lowconf for dim-yellow rendering, matching the existing low-confidence display path;
self.branches[bi] and diameter pipeline unaffected by both changes

### OstiumSnapZMatchedRadial: fix radial direction using Z-matched trunk CL point;
root cause of v227 failure: nearest-3D trunk CL point to detect_gi was axially offset (trunk_cl Z=1881.7 vs detect_gi Z=1883.4 is fine, but XY position X=26 Y=154.9 vs detect_gi X=19.3 Y=159.0 gave radial [-0.84, 0.51, 0.21]

### the 0.21 Z component was small but more critically the XY direction was wrong because the nearest-3D trunk point is not the trunk point at the same Z cross-section);
fix: use _trunk_zs/_trunk_gis (already built for _trunk_diam_at_z) to find the trunk CL point at the exact same Z as detect_gi, then compute radial in XY plane only (zero out Z component before normalising)

### this gives a purely lateral direction from the IVC axis to detect_gi independent of any axial offset;
expected: radial dir with near-zero Z and XY pointing toward the lateral renal wall;
OstiumSnapRadial: replace branch-tangent direction with trunk-CL radial vector for lateral snap selection;
root cause of cranial snap: _t_fwd at detect_gi points axially (branch CL runs parallel to IVC near junction before curving lateral) so argmax(dot) selected the most cranial trunk vert not the lateral wall;
fix: compute radial = detect_gi_point minus nearest_trunk_CL_point (already available via _snap_trunk_kd built earlier in scope);
this vector always points radially outward from the IVC axis toward the lateral wall regardless of branch CL orientation;
fallback to branch tangent if _snap_trunk_kd unavailable;
log now shows radial dir + trunk_cl anchor point;
ExtractDiag: add [ExtractCenterline] print statements to surface the silent failure

### the button fires but the console will now show exactly which guard (no model / no endpoints / n<2) is blocking execution;
OstiumSnapLateral: fix anterior snap by replacing argmin(distance) with argmax(dot(vec,_t_fwd)) for trunk candidate selection;
root cause confirmed from 3D screenshot: nearest trunk vert from CL origin is always the anterior IVC wall face regardless of branch direction;
fix: among all trunk-classified verts within SEARCH_R_MM=15mm, project each onto _t_fwd (branch tangent at detect_gi, pointing laterally toward the renal takeoff wall) and pick the one with maximum dot product

### this selects the trunk vert furthest in the branch departure direction = the lateral IVC wall where the renal vein actually enters;
stale belt-and-suspenders comment updated;
log now shows trunk_surf_lateral + dot value;
CenterlineVisToggle: add "👁 Show/Hide Centerline" toggle button to Step 3 UI below the hint label;
button is a checkable QPushButton (grey=hidden, green=visible);
onCenterlineVisToggle sets dn.SetVisibility(0/1) on the selected centerline node's display node;
onCenterlineNodeChanged syncs button state when the user changes the centerline selector;
both handlers use blockSignals to avoid recursive toggling;
allows user to inspect centerline orientation at the renal takeoff to diagnose OstiumSnap direction

### OstiumSnapCapFix+StaleComments: two fixes on top of v222;
(1) SEARCH_R_MM raised 8→15mm: v222 trunk-surf snap was immediately cap-rejected because the IVC trunk wall sits ~8-9mm from the renal CL at the takeoff zone (IVC mean radius ~10mm)

### snap_d=8.7mm and 8.1mm both exceeded the old 8mm cap and fell back to raw CL;
15mm comfortably reaches the trunk wall while still excluding the contralateral IVC wall (~20+mm away);
MAX_SNAP_MM=SEARCH_R_MM so the cap rises automatically;
(2) OFFSET_FRAC tangent-shift removed: the 4mm offset along branch tangent was designed for the old branch-surf query and pushed the sphere in the wrong direction for trunk-surf queries;
querying directly from the CL point finds the nearest trunk wall in all directions which for a lateral renal is the lateral IVC wall rim;
(3) three stale comment blocks describing superseded algorithms (snap-from-sgi, branch-surf-first, proximal-branch-surf) removed from lines 14481-14539

### they contradicted the current trunk-surf-first strategy and would cause confusion on future edits

### OstiumSnapTrunkPool: fix renal/side ostium snapping to IVC wall;
root cause: _surf_kd_per_branch[bi] (Voronoi branch-classified verts) contains IVC wall faces at the junction zone that are geometrically closer to the renal CL than to the trunk CL

### querying branch verts therefore returned an IVC wall vertex regardless of offset;
fix: swap primary snap candidate pool to _trunk_surf_kd (trunk-classified verts) within SEARCH_R_MM of detect_gi;
the trunk surface AT the takeoff Z is the lateral IVC wall that forms the anatomical ostial rim

### exactly where the marker should land;
fallback chain: trunk KDTree (primary) → branch KDTree (fallback, previous primary) → raw CL[detect_gi];
log now shows trunk_surf_nearest / branch_surf_fallback / CL fallback to confirm which path fired;
detect_gi/ostiumGi/self.branches commit unchanged

### OstiumSnapProximal: find most-proximal branch-surface vertex within SEARCH_R_MM=12mm of detect_gi;
root cause of all previous failures now understood from debug screenshot: (1) trunk KDTree → cranial IVC wall;
(2) nearest branch vertex from detect_gi → same junction-zone vertex;
(3) walking into branch then nearest branch vertex → distal branch wall (P3_3 marker too far into branch lumen visible in screenshot);
correct definition confirmed by user: ostium = where branch color changes from IVC-green to cyan/orange = PROXIMAL EDGE of branch-classified surface;
algorithm: collect all branch-classified vertices within 12mm of detect_gi CL point, project each onto branch tangent, pick the one with minimum dot product (most proximal toward IVC)

### this is the branch surface vertex closest to the IVC/branch boundary in the proximal direction;
fallback chain: nearest branch vertex → trunk KDTree → CL[detect_gi];
log shows branch_surf_prox bi=N n_cand=N

### OstiumSnapOffset: walk SNAP_OFFSET_MM=8mm along branch CL past detect_gi before querying branch-surface KDTree;
root cause of persistent identical wall coordinate: detect_gi==sgi is at the proximal edge of independent lumen but still geometrically inside the junction zone

### the nearest branch-classified surface vertex from there is the same IVC wall patch as before;
fix: walk 8mm further along the branch CL from detect_gi to place the snap query origin inside clean branch lumen, then the nearest branch-surface vertex found is the proximal wall of the branch as seen from inside = the ostial rim at the IVC/branch color boundary;
log shows snap_gi=N(+Xmm) to confirm offset applied;
detect_gi/ostiumGi/self.branches commit unchanged

### OstiumSnapBranchSurf: replace trunk-surface KDTree snap with per-branch Voronoi surface KDTree;
root cause of all previous cranial marker placement: snapping to nearest TRUNK-classified surface vertex always returned the cranial IVC wall face at the junction Z level regardless of snap origin (sgi or detect_gi), because trunk KDTree vertices at that Z are the proximal face of the opening, not the ostial rim;
fix: at start of _refineOstia build a Voronoi assignment of all surface vertices to their nearest branch CL (same logic as classifySurfaceBranches), then build a per-branch cKDTree (_surf_kd_per_branch[bi]) of vertices assigned to each branch;
in OstiumSnap, query _surf_kd_per_branch[bi] with detect_gi CL point

### the nearest branch-classified vertex is by definition on the branch side of the IVC/branch color boundary, exactly where the surface transitions from green (IVC) to cyan/orange (branch) which is the anatomical ostium the user identified;
fallback chain: branch KDTree → trunk KDTree → raw CL[detect_gi];
log shows snap_label=branch_surf bi=N or trunk_surf fallback;
_snap_coord/ostium/ostium_p3 commit unchanged

### OstiumSnapDetectGi: change snap source from sgi to detect_gi;
root cause of cranial marker placement visible in 3D screenshot: snapping from sgi (which equals detect_gi after sgi-floor) still finds the PROXIMAL IVC wall face because the CL point at sgi is at or just inside the vessel wall

### nearest-surface from there is the cranial rim of the opening, not the ostial rim;
fix: snap from detect_gi (confirmed post-flare lumen-separation point, guaranteed >= sgi by v217 floor) so nearest trunk-surface vertex is the lateral IVC wall at the true ostial opening;
SNAP_DOT_MIN tightened -0.2→0.0: only accepts snaps with net lateral/distal component, rejects purely cranial snaps (dot<0) pointing back toward proximal IVC wall;
fallback changed from CL[sgi] to CL[detect_gi] (already in correct anatomical zone);
log now shows snap_from=detect_gi=N vs sgi=N to confirm which path fired;
ostiumGi/self.branches unaffected (diameter pipeline unchanged)

### OstiumSnapConfirm+SgiFloor: two fixes for Branch4/Branch5 detect_gi landing too proximally (~1mm past ogi, inside IVC flare zone);
(1) CONFIRM_PTS=3: forward scan now requires ratio<0.80 to hold for 3 consecutive points before committing

### single-point dips in the shared-lumen zone (which caused the early commit) are skipped;
(2) SgiFloor: _detect_gi is floored to stableStartGi after the forward scan

### sgi is guaranteed to be in post-flare clean tissue by the flare-walk, so no anatomically valid ostium can be proximal to it;
both fixes are additive: CONFIRM_PTS catches transient dips, sgi-floor catches cases where ratio stays depressed through the entire junction zone;
expected: Branch4 detect_gi advances from 421 (≈gi_ogi+1) to ≥423 (sgi), display snap shifts distally from Z≈1883 to Z≈1884–1886;
Branch5 similarly advances past its sgi;
diameter pipeline, report, and stent sizing unaffected (ostiumGi/self.branches committed to updated detect_gi)

### OstiumSnapSgi: switch snap source from ogi to sgi (stableStartGi);
root cause of persistent proximal bias: ogi (Z≈1876) is inside shared IVC/branch lumen

### nearest trunk surface from there is the PROXIMAL IVC wall, correctly on-wall but cranial to the true ostium;
sgi (Z≈1883, first clean independent branch tissue) snaps to the DISTAL rim of the IVC opening which is the anatomical ostium;
directional constraint retained (dot<-0.2 rejects strongly backward snaps);
log now shows snap_from=sgi=N;
diameter pipeline unaffected (still commits to detect_gi)

### OstiumSnapRelaxed: two fixes for Branch3 green marker collapsing onto trunk CL;
(1) SNAP_DOT_MIN relaxed 0.2→-0.2: renal veins enter IVC nearly perpendicular (dot≈0.08 on Branch3) so the 0.2 threshold was rejecting valid ostial snaps

### threshold now only rejects strongly backward projections (>~101° from branch direction);
dot=0.08 now ACCEPTED → marker lands on IVC wall at ostium rim instead of falling back;
(2) FallbackFix: if snap still rejected, fallback uses raw CL[ogi] coordinate (junction-region point) instead of a trunk CL projection which was collapsing the marker onto the trunk centerline upstream of the true ostium;
[SnapCheck] log updated to show REJECT→CL[ogi] to distinguish from the old REJECT→CL trunk-projection behavior

### OstiumSnapDirectional: revert snap source to ogi + add directional dot constraint to reject backward projections;
root cause of v213 proximal bias: snapping from detect_gi (5mm inside branch lumen) projects backward onto proximal IVC wall because nearest-surface from a distal point finds the parent vessel wall, not the ostial rim;
fix: (1) snap source reverted to ogi (junction-region anchor);
(2) directional reject: compute v_branch=unit vector along CL at ogi (3pt forward avg), v_snap=unit vector from CL[ogi] to nearest surface vertex, reject if dot(v_snap,v_branch)<SNAP_DOT_MIN=0.2 (snap points more than ~78° against branch direction = toward trunk wall);
fallback: if rejected use raw CL coordinate at ogi so marker is in correct neighbourhood;
[SnapCheck] log line shows dot and KEEP/REJECT for every branch;
ostiumGi/self.branches[bi] still commit to detect_gi so diameter pipeline unaffected

### OstiumSnapUnified: snap display coordinates from detect_gi for ALL branch types including renal_stabilized;
root cause of Branch3 green marker being cranial to the true funnel rim: ogi (RenalStabilizer wall-contact point, last CL point within 2.5mm of trunk) is still inside the shared lumen

### detect_gi (first point where ratio<0.80 = proximal edge of independent branch lumen) geometrically corresponds to the visible funnel rim on the IVC wall, confirmed by user 3D view review;
renal_stabilized distinction removed from snap-source selection, both branch types now unconditionally snap from detect_gi;
ostiumGi/self.branches[bi] unaffected (always committed to detect_gi);
log shows snap_from=detect_gi=N for all branches

### ExtractButtonFixRegression: duplicate modelSelector.connect("currentNodeChanged", onModelSelected) at line 472 (before extractButton creation at line 588) reintroduced;
Slicer silently swallows the AttributeError on the early connect, leaving the signal table in a corrupted state so the authoritative connect at line 1616 never fires

### extractButton stays disabled permanently;
fix: remove the early line-472 connect;
single connect retained in the consolidated signal block after all widgets built;
same root cause as ExtractButtonFix originally applied in prior build

### OstiumProximalFix: two patches for Branch3/Branch4 ostia placed too proximally;
(1) RingNormalReналClamp: _geo_ostium clamp was pinning renal ring-normal commits to sgi even when GEO_RENAL_EXT_MM extension found a valid failure boundary past sgi

### fix: for is_renal branches clamp to _geo_end (the extension ceiling) instead of sgi so the extra 15mm scan range can actually commit distal results;
(2) OstiumSnapSourceSelect: snap display coordinates from detect_gi for non-renal-stabilized branches (side_branch, collaterals);
ogi is only a valid wall-contact anchor when RenalStabilizer set the renal_stabilized flag

### for other branches ogi is the composite-scorer result which may still be inside the flare zone;
detect_gi (first point where ratio<0.80 = proximal edge of true independent lumen) is the correct snap source;
RenalStabilizer-flagged branches keep ogi-anchored snap unchanged;
log now shows snap_from=ogi=N vs snap_from=detect_gi=N to confirm which path fired

### OstiumSnapWallAnchor: fix marker too-proximal on renal vein ostia;
root cause: OstiumSnap was snapping the display coordinate from self.points[detect_gi] (the backward-scan crossing point, already 5-10mm inside the branch lumen past the true wall junction) to the nearest trunk surface vertex

### this placed the marker on the distal rim of the IVC opening, appearing too proximal in the 3D view;
fix: snap display coordinates from self.points[ogi] (the RenalStabilizer wall-contact anchor, last point where CL was within 2.5mm of trunk)

### ogi is the correct anatomical anchor;
ostiumGi and self.branches[bi] still commit to detect_gi so diameter pipeline and finding detection are unaffected;
only branchMeta ostium / ostium_p3 display coordinates change;
log now shows snap_from=ogi=N to confirm which point was used

### LogRegCalibration: three fixes for v208 saturation (all branches → 0.99): (1) PriorRescale: halved all prior weights so a typical good branch maps to sigmoid≈0.5 not ≈1.0;
perfect branch→0.92 max, clear negative→0.08 min;
(2) OutputCompression: score() applies p→0.15+0.70×sigmoid so range is [0.15,0.85] preventing hard saturation at either extreme;
(3) FlagFeatures+Interactions: 9→14 features

### added AD=lat×div (renal/side separation), VG=area×geometry (sanity cross-term), fl_area_low/fl_zone_distant/fl_divergence_low with negative priors (-0.5 to -0.65) so these flags now depress confidence inside the model not just as post-hoc penalties;
call sites updated to use build_features(comps, flags);
(4) NegativeSampling: _logOstiumTrainingRecord now auto-generates near-miss (y=0.35) and hard (y=0.0) negatives by shifting observation window ±8–40 pts along branch CL with diameter/stability/zone signals re-extracted at shifted gi;
cross-branch negatives (another branch type's ostium signature scored against current type) also added;
dataset now has pos:neg ratio ~1:8 per run;
total capped at 2000 records;
MIN_TRAIN_N lowered 10→6 to allow first training pass after a single run

### replace manual additive weighted sums in _computeOstiumConfidence (arterial path and venous path, including ShortBranchBoost recompute) with _OstiumLogReg logistic-regression scorer;
per-branch-type weight vectors (main/renal/side_short/side) initialised from hand-tuned formula priors then updated via online mini-batch SGD;
new _logOstiumTrainingRecord() method appends V/A/L/T/G/S/C/Z/D feature vectors + accepted bool to ostium_logreg_data.json after each run and triggers train_step() once MIN_TRAIN_N=10 records per type are available;
weights persisted to ostium_logreg_weights.json next to module;
_OstiumLogReg class added above VesselAnalyzerLogic with classify_branch/score/train_step/save/_load API;
all existing flag/penalty/spread/grading/override/ShortBranchBoost logic untouched

### only the weighted-sum step is replaced;
expected: same grades as v207 on first run (priors match hand-tuned formula), gradually improving over 20-50 cases as weights adapt;
logs [LogReg-A]/[LogReg-V] per branch showing raw score and type, [LogReg] trained after each SGD pass, [LogReg] dataset after each run

### RenalConsolidateExtended+ShortBranchBoost: two targeted fixes for Branch5-type anatomy;
(1) RenalConsolidateExtended: extend RenalConsolidate pairwise check to include side_branch candidates that are geographically adjacent to a renal_vein;
root cause: Branch5 (side_branch, 22mm) is the VMTK over-segmented entry stub of the same vessel as Branch4 (renal_vein, 74mm)

### RenalConsolidate only checked role==renal_vein pairs so Branch5 was invisible to it;
fix: after the existing renal_vein pairwise merge, run a second pass that checks every side_branch whose ostium is within SIDE_MERGE_DIST_MM=20mm of a renal_vein ostium AND angle<SIDE_MERGE_ANG_DEG=60°

### mark it renal_fragment of that renal_vein;
logs [RenalConsolidate] SideFrag Branch N (Xmm) → fragment of renal Branch M;
(2) ShortBranchBoost in _computeOstiumConfidence: for branches arc<30mm in venous mode, boost topo_v by +0.15 (floored at existing value), reduce area and curvature penalties to 0, and apply safety floor: if topo>=0.6 AND stab>=0.8 → eff floor 0.45 (MEDIUM);
this ensures short side-branch stubs that survive consolidation still get MEDIUM not LOW/REJECT;
logs [ShortBranchBoost] Branch N: topo boosted / floor applied;
retains all v206 logic unchanged

### OstiumSnapFullBranch: extend OstiumSnap scan window from [ogi,sgi] to [ogi,e-2] (full branch);
root cause: for B4(31mm) and B5(31mm) sgi is only ~10pts past ogi and still inside IVC junction zone

### the true ostium where diameter drops to branch scale is past sgi (B4 target S=1892.5 vs sgi S=1881.4);
clamping to sgi guaranteed the backward scan never reached the crossing point;
fix: scan full branch [ogi,e-2], backward walk from tip finds first ratio>0.80 crossing, commit one step distal;
retains v201 RingNormalBackwardScan and v200 OstiumSnapBackwardScan

### RingNormalBackwardScan: replace forward ring-normal scan from ogi with backward scan from sgi;
root cause of B4/B5 wrong ostium via sphere navigator: forward scan started at ogi (inside IVC flare zone) and ring-dot fired immediately on the oblique IVC wall geometry → marker placed inside the shared lumen, skipping OstiumSnap via `continue`;
fix: collect ring results for all gi in [ogi, geo_end], then walk backward from sgi

### find the first gi where ring fails (not perp OR not separating), commit the gi one step distal (still in the good zone);
if no failure found (whole window good) commit ogi;
clamp to [ogi, sgi];
log tag now says ring-normal(bwd);
also retains v200 OstiumSnapBackwardScan fix

### OstiumSnapBackwardScan: replace forward scored-scan from ogi with backward scan from sgi;
root cause of B4/B5 marker being too proximal (B4: S=1883→1892, Ø=15→10mm;
B5: S=1878→1882, Ø=16→12mm): forward scan started at ogi which is inside the trunk-contaminated flare zone

### diameter signal was ≈0 across the whole shared-lumen region so composite score peaked at ogi driven by divergence alone;
fix: scan [ogi,sgi] backward from sgi (the reliable clean-lumen anchor), stop at the first gi where ratio=branch_diam/trunk_diam_at_z rises above DIAM_ENTER_THRESH=0.80 (entering trunk influence), commit the gi one step distal of that crossing;
fallback to ogi if branch was already separated at ogi (no crossing);
clamp to [ogi,sgi];
snap committed detect_gi to nearest trunk surface vertex as before;
log tag [OstiumSnap] now shows ratio/thresh instead of composite score

### OstiumDualGate: replace naive nearest-surface-snap with anatomically correct two-signal walk for renal ostium detection;
algorithm: (1) walk branch CL forward from ogi;
(2) find FIRST point where BOTH (A) branch diameter < 70% of trunk diameter at same Z AND (B) branch direction diverges >30° from trunk direction (dot < cos(30°)=0.866);
(3) snap that CL point to nearest trunk surface vertex;
(4) commit snapped coordinate as actual ostiumGi+ostium (not just display P3);
trunk diameter at Z looked up by nearest-Z trunk gi from self.diameters;
trunk direction from trunk CL root→bif unit vector;
fallback to ogi if no dual-gate hit within [ogi, sgi+5];
tangents array already computed in this scope;
log [OstiumSnap] shows detect gi, ratio, dot, CL coord, IVC wall coord, snap_d;
fix snap-to-IVC-wall using ogi not best_gi;
v197 snapped from self.points[best_gi] (gi=416, Z=1875.8) which is already ~14mm into the branch lumen, producing a snap at Z=1877

### confirmed correct on IVC wall but ~15mm too cranial;
fix: snap from self.points[ogi] (gi=411, the stabilizer-advanced wall-contact exit point) instead

### ogi is the last point where the branch CL was still in contact with the IVC wall, making it the correct anatomical anchor for the surface snap;
best_gi unchanged (still used for ostiumGi/diameter pipeline);
only ostium_p3 (green marker) coordinate changes;
fix renal ostium marker placement on trunk wall;
root cause: self.points[ostiumGi] is the VMTK medial-axis centerline coordinate which for a renal branch in the first 10-15mm runs inside the IVC lumen

### not on the IVC wall

### so the marker renders at a point that is visually "on the trunk";
the anatomically correct ostium coordinate is the nearest trunk-surface vertex to self.points[ostiumGi] = where the renal vein first intersects the IVC wall;
fix: (1) build a trunk-surface KDTree once at the top of _refineOstia using all surface vertices within 20mm of the trunk centerline;
(2) after committing best_gi for any renal branch, query the KDTree with self.points[best_gi] and store the nearest trunk-surface vertex as branchMeta[bi]['ostium_p3'];
(3) this populates the green P3 marker (previously always 0) with the correct IVC-wall coordinate;
ostiumGi/best_gi index unchanged

### only the display coordinate (ostium_p3) is corrected;
diameter sampling and finding detection still anchor to sgi;
log tag [OstiumSnap] Branch N: CL pt [...] → IVC wall [...] (snap_d=X.Xmm);
also retains v196 segModelNode pass-through and renal ring-detector parameter adjustments;
ThreeFixes: (1) currentIndexFix: vesselTypeCombo.currentIndex() TypeError on onReAnalyze

### Qt property accessed as method call fails in some Slicer builds where currentIndex is an int property not a callable;
fix: replace .currentIndex() with .currentIndex (property access) at all 3 call sites (onReAnalyze, onLoadIVUS entry, stent check);
(2) RingNormalOstiumClamp: ring-normal detector could overshoot sgi by scanning beyond stableStart (Branch4 ogi=458→476 sgi=468 = +8pts past sgi);
two-part fix: cap _geo_end=min(_geo_end,sgi) so scan loop never enters post-stable zone;
add _geo_ostium=min(_geo_ostium,sgi) commit guard so result is always <=sgi;
(3) ImageNote: red coloring in Image1 is IVUS diameter heat map not branch misclassification

### Image2 confirms surface classification is correct (green trunk/iliacs, cyan renal, orange side);
v194

### RingNormalOstium: replace vertex-normal geometric detector with ring-plane-normal detector;
new _computeRingNormal() method: sphere-clip+vtkCutter+vtkStripper yields largest contour by perimeter, numpy SVD on ring point cloud gives smallest singular vector=best-fit plane normal, ellipticity=S[0]/S[1] from singular values;
detector signals: (A) dot(ring_normal,branch_tangent)>RING_DOT_THRESH=0.85

### sphere navigator ring plane aligns with branch axis = wall is perpendicular;
(B) ellipticity<RING_ELIP_THRESH=1.6

### ring is near-circular = cut is orthogonal not oblique;
(C) d_trunk(i+1)-d_trunk(i)>SEP_THRESH=0.3mm

### branch separating;
stable for STABLE_N=2 consecutive;
ostium_method=ring_normal;
log shows dot/elip/ring_pts/sep per hit;
physics: oblique cut at IVC junction produces elliptical ring with normal misaligned to branch axis;
perpendicular cut past ostium produces circular ring with normal=branch tangent

### exactly what the sphere navigator shows the user;
fallback to composite scorer unchanged;
v193

### NoneGuard: fix TypeError crash in classifySurfaceBranches when ostiumGi=None on suppressed-REJECT branches;
line `ogi_local=max(0,ogi-bs)` crashed with NoneType-int arithmetic because _suppressRejectedOstia sets ostiumGi=None after _computeOstiumConfidence but classifySurfaceBranches reads it unconditionally;
fix: add `if ogi is None: ogi = bs` guard immediately after ostiumGi retrieval so suppressed branches use raw topology start for surface classification (correct

### they have no anatomical ostium);
all other ogi extraction sites already None-safe (11 sites checked);
v192

### GeometricOstiumDetector: replace length-based Gate2 with pure geometry;
FIRST point where (A) surface normal perp to branch tangent: 1-|dot(n,v)|>PERP_THRESH=0.80 AND (B) separation gradient d_trunk(i+1)-d_trunk(i)>SEP_THRESH=0.3mm;
stable for STABLE_N=2 consecutive pts;
normals via vtkPolyDataNormals+vtkPointLocator;
trunk dist via scipy cKDTree<=60pts;
ostium_method=geometric;
fallback to composite scorer if no hit;
Gate1 main/iliac skip unchanged;
REJECT suppression in _suppressRejectedOstia unchanged;
v191

### BranchTypeRouter: three hard gates in _refineOstia enforce type-aware ostium strategy;
Gate1 main/iliac branches skipped (bifurcation-anchored, refinement over-shifts);
Gate2 non-renal branches <REFINE_MIN_MM=40mm skipped (flow-norm ogi IS the attachment point, no reliable geometry for scoring);
Gate3 REJECT suppression moved to new _suppressRejectedOstia() called after _computeOstiumConfidence (Gate3 in _refineOstia was dead

### confidence not yet computed at that point);
_suppressRejectedOstia: post-confidence pass sets ostiumGi=None for REJECT-grade non-main non-exempt branches;
exempt from suppression: attachment_short (reliable attachment point) and already-suppressed;
logs [OstiumSuppress] Branch N: REJECT (eff=X)

### ostium suppressed;
self.branches[bi] left intact so diameter pipeline still runs;
ExpectedBehavior: Iliacs->skipped(bifurcation-anchored);
Branch4(58mm,MEDIUM)->refined by attachment scorer;
Branch5(25mm,REJECT)->suppressed(ostium=None,no marker);
v190: UnifiedOstiumScore: replace global-max scorer with 4-signal change-point detector;
(1) W_LAT=0.50/0.30 lateral separation rate d_lat=lat[gi]-lat[gi-1] normalised at LAT_NORM_MM=5mm PRIMARY for short branches;
(2) W_ANG=0.20/0.40 delayed angle to trunk with ANGLE_OFFSET_MM=12mm skip past dome contamination;
(3) W_STAB_DIR=0.30 direction stability=dot(dir[gi],dir[gi+delta]) gates chaotic junction zone;
(4) W_PROX penalty=exp(-arc/10mm) prevents early picks fixing proximal-bias bug;
AdaptiveWeights: short branch <60mm uses w_lat=0.50 w_ang=0.20 w_stab=0.30 long branch >=60mm uses w_lat=0.30 w_ang=0.40 w_stab=0.30 fixes Branch4+Branch5 without hacks;
TrunkKDTree: lateral separation measured via scipy cKDTree on subsampled trunk CL (<=60pts);
TrunkDir: per-branch trunk direction vector for delayed angle;
ForwardLocalPeak: selection replaced from global argmax to forward scan picking first candidate exceeding PEAK_THRESHOLD=0.35 that beats PEAK_LOOKBACK=3 preceding pts;
hard safety guards: gi>=ogi+5pts AND gi<=sgi;
fallback to global argmax if no peak found;
LegacyCorroboration: area/diam/topo/curv/drop/cons retained at reduced weights (sum ~0.45);
LogUpdated: new columns lat ang stab_dir prox_pen plus weight-regime tag;
ostium_scores_v2 gains lat/ang/stab_dir/prox_pen/short_branch/w_lat/w_ang/w_stab_dir keys;
ExpectedResults: Branch5(short) lat dominates->correct ostium;
Branch4(curved) stability low at false peak->penalised->real ostium wins

### v189: RefineOstiumV2Signals: three physiological signals from refineOstium_v2 merged into _refineOstia;
(1) DiamDrop W=0.20 PRIMARY: drop_sc=1-d/trunk_diam forces ostium to land where lumen has left trunk influence (eliminates ~18mm picks on renal branches);
(2) DistalConsistency W=0.12: cons_sc checks 5-15pts ahead are stable (avoids noisy flare zone);
(3) TrunkProximityPenalty coeff=0.35: hard penalty when d>1.3xbranch_mean kills trunk-bleed-inflated candidates;
(4) WinExtension: win_end now sgi+EXT_MM (20mm default 30mm renal) vs old cap at sgi (true ostium is 10-25mm past stableStart);
branch_mean anchored to first 10 post-stableStart diameters;
trunk_diam=median of trunk gi range;
W_AREA reduced 0.30 to 0.20 W_DIAM 0.20 to 0.10 to make room;
log gains drop/cons/trunk_pen columns;
ExpectedResults: Branch4 18.8mm->~11-12mm Branch5 16.5mm->~12-13mm position shifts 10-25mm distal

### OstiumConfidenceVenousFormula: full venous confidence formula replacing v187 weight-profile tweak with a dedicated dual-path fusion;
VENOUS PATH: (A) lateral_score=clamp(tip_lateral_mm/40mm) PRIMARY signal weight=0.30

### renal veins at 30-80mm off-axis immediately score high;
(B) z_pos_score=Gaussian(dz,centre=70mm,σ=40mm) above bifurcation weight=0.20

### branches mid-IVC score near 1.0;
(C) diam_ratio_score=1.0 in [0.45,0.90] else soft ramp weight=0.15;
(D) topo_v=max(topology_score,0.60) softened default weight=0.15

### diffuse venous attachments floor at 0.60 not near-zero;
(E) stability_score weight=0.10 unchanged;
(F) curv_v=0.50 neutral unless erratic weight=0.05;
(G) area_v=0.50 neutral if signal weak else 0.50+0.50*area_sc weight=0.05;
VenousOverrideRule: if lat>30mm AND diam_ratio>0.45 AND length>50mm → floor confidence at 0.75 (HIGH), logged as [OVERRIDE];
ARTERIAL PATH: byte-for-byte unchanged from v186;
FlagUpdates: topology_uncertain suppressed in venous (topo already floored);
divergence_low suppressed in venous;
gradient_weak/area_low/curvature_flat already zero-penalty in venous (v187);
ResultDict: new venous_override bool key;
LogUpdated: venous branches print dedicated lat/z/diam/topo/stab/curv/area breakdown line;
[OVERRIDE] tag on log when override fires;
ExpectedResults: Branch4(lat=54.7mm,ratio=0.52,len=74mm) override fires → raw≥0.75 → eff≥0.60 → HIGH;
iliacs floor→0.60→HIGH;
small fragment excluded via renal_fragment role OstiumConfidenceVenousMode: full venous-aware ostium confidence scoring;
root cause of LOW/REJECT on IVC iliacs: arterial priors (strong area gradient, high curvature, sharp taper) applied to venous confluence geometry which physiologically has flat area profiles, low curvature, and gradual taper;
(1) VenousWeightProfile: in venous mode wT=0.22 wG=0.22 (topology+geometry dominant, up from 0.12/0.20) wC=0.04 (curvature de-weighted from 0.09) gradient slot halved (wV+wArea≈0.12 vs 0.24 arterial);
arterial profile unchanged;
(2) VenousFlagSuppression: gradient_weak, area_low, curvature_flat penalties set to 0.0 in venous mode

### flags still logged for visibility but carry no penalty since these features are expected physiology not detection failures;
topology_uncertain/lumen_unstable/zone_distant retain full penalties in both modes;
(3) VenousMainFloor: main-branch effective floor raised 0.50→0.60 in venous mode;
(4) VenousGradeThresholds: venous thresholds −0.10 across board: HIGH≥0.60 MEDIUM≥0.40 LOW≥0.25 (vs arterial 0.70/0.50/0.35)

### eff=0.60 in a vein represents same anatomical confidence as eff=0.70 in an artery;
(5) AreaWeakThresholdRelaxed: main-branch gradient_weak threshold already relaxed in v186 (range<0.15, peak<0.10 vs side branch 0.25/0.15);
(6) LogUpdated: header now prints mode=venous/arterial;
per-branch line appends [venous] tag;
expected results venous mode: Left Iliac floor→0.60→HIGH Right Iliac floor→0.60→HIGH Renal topology-dominant→MEDIUM-HIGH OstiumConfidenceCalibration: three targeted fixes for smooth iliac anatomy producing false LOW/REJECT grades;
(1) RelaxedAreaWeakThreshold: gradient_weak threshold for main branches lowered (range<0.15, peak<0.10 vs 0.25/0.15 for side branches)

### aortic Y has gradual area transition by design, not a detection failure;
side branch thresholds unchanged;
(2) RoleAwarePenaltyScale: gradient_weak/curvature_flat/geometry_weak/area_low penalties halved for confirmed iliac branches (_pen_scale=0.5 for is_main)

### smooth anatomy at the aortic Y is expected physiology;
topology/zone/lumen_unstable/zone_distant penalties remain full-weight as they are role-agnostic;
(3) MainBranchEffectiveFloor: confirmed iliac (is_main=True) effective_score clamped to minimum 0.50 so a proven anatomical branch cannot grade below MEDIUM regardless of signal spread

### pipeline already confirmed these branches via graph topology;
floor logged with [main-floor] tag;
expected results on current dataset: Left Iliac LOW(0.45)→MEDIUM or HIGH;
Right Iliac REJECT(0.29)→MEDIUM;
renal/side branch grades unchanged (side branch thresholds and full penalties intact) OstiumConfidenceLowerBound: replace raw-score grading with 3-stage lower-bound pipeline;
(1) FlagEnforcement: flags (gradient_weak, topology_uncertain, curvature_flat, geometry_weak, area_low, lumen_unstable, zone_distant, divergence_low) now carry explicit penalties (0.04–0.08) subtracted from raw confidence before grading, capped at 0.30 total

### previously flags were log-only and had no effect on grade;
(2) UncertaintyRedefined: uncertainty changed from meaningless (1−score) to std of the 9 component values

### measures actual signal agreement spread;
(3) LowerBoundGrading: effective_score = penalised_confidence − component_spread;
grade thresholds HIGH≥0.70 / MEDIUM≥0.50 / LOW≥0.35 / REJECT applied to effective_score not raw score;
(4) StrongCountBoostRemoved: strong_count≥3→HIGH promotion removed

### this was the primary source of "HIGH but shaky" grades (e.g. Right Iliac 0.728±0.272→HIGH now correctly→MEDIUM);
strong_count retained in result dict for diagnostics only;
(5) ResultDictExpanded: new keys penalised_score, effective_score, flag_penalty alongside existing score/grade/uncertainty/components/flags;
(6) DisplayUpdated: debug dialog, Word doc detail table, and AI prompt meas_summary all updated to show eff= as the primary score;
(7) QABucketFixed: LOW grade now routes to qa_review (not qa_reject);
REJECT label updated to "effective<0.35";
applied to your example data: Left Iliac eff=0.72→HIGH✓ Right Iliac eff≈0.46→MEDIUM✓ Branch4 eff≈0.52→MEDIUM✓ Branch5(T=0.01) eff≈0.34→REJECT✓

### ExtractButtonFix: remove duplicate modelSelector.connect at line 472 that fired onModelSelected before extractButton was created;
Slicer silently swallowed the AttributeError leaving extractButton permanently disabled;
single connect retained in the consolidated signal-connection block after all widgets are built RefineOstiumPhase1: replace single-point proximity-dominated scorer (_refineOstia v4) with a multi-point window scanning system;
three new methods added: _computeBranchTangents (central-difference ±2pt smoothed tangents per branch, computed inline at scan time), _computeCrossSectionArea (vtkCutter+vtkStripper+shoelace on 2D projected ring, sphere pre-clip at 12mm to prevent adjacent-branch contamination, largest strip by perimeter not point count), and a rewritten _refineOstia (replaces entire old body);
scoring now runs over window [ostiumGi−BACK_MM, stableStartGi] per branch

### BACK_MM=8mm for main/side branches, RENAL_BACK_MM=5mm for renals;
four normalised signals per candidate gi: area_gradient (vtkCutter area change across ±LOOK pts, normalised at 30% relative increase, weight 0.40

### PRIMARY signal replacing radius_consistency KDTree approach), diam_gradient (self.diameters[gi] symmetric difference, normalised at 15%, weight 0.25), topology_prior (exp(-|gi−ogi|/8pts), weight 0.20

### replaces hard proximity penalty), stability_score (diameter CV over 6 pts distal, weight 0.15);
renal branches: previously skipped after _stabilizeRenalOstia, now refined inside ±5mm window using stabilizer as prior (is_renal=True flag selects RENAL_BACK_MM);
confidence upgraded to Confidence 2.0: mean(window_scores)−std(window_scores) stored in branchMeta[bi]['ostium_confidence_v2'] alongside per-component breakdown;
ostium_method tag changed 'scored'→'scored_v2';
log format: [RefineOstium] Branch N: [renal ]ogi=X→Y (+Npts, +Ymm) sgi=Z score=S conf=C window=[a,b](Npts) [area=0.xx diam=0.xx topo=0.xx stab=0.xx];
graceful degradation: if modelNode is None, area_sc=0.0 for all candidates and scorer falls back to diam+topo+stability;
existing _computeOstiumConfidence and OstiumConfidence grade system untouched

### RenalTagV2: replace single-pass composite scorer with three-pass anatomy-gated topology-aware classifier;
Pass 1 collects geometry for ALL side candidates before any decision;
Pass 2 applies a hard anatomical gate (len>=40mm, lat>=30mm, 20mm<=dz<=120mm

### all non-negotiable, gate failures become side_branch before scoring) then runs pair-symmetry detection (opposite X-sign + |ΔZ|<25mm → renal pair) and dominance analysis (largest score-margin branch flagged dominant);
Pass 3 builds per-branch topology_score (0.55 neutral prior, +0.30 pair membership, +0.15 symmetric-pair bonus, +0.15 dominance bonus, −up-to-0.40 unstable-snap penalty) and feeds it into the composite scorer replacing the old fixed-weight slot;
composite weights redistributed: diam 0.40→0.25, lat 0.30→0.25, new topo 0.25, z unchanged 0.15, angle unchanged 0.10;
Z-score upgraded from linear ramp to Gaussian centred at bif_z+80mm σ=40mm (Gaussian anchors to the anatomical centre of the renal Z window;
branches too near or too far from bif both penalised);
net effect on current dataset: Branch 3 (len=74mm, lat=54.7mm, dz=87mm) passes gate, strong pair candidate, dominant → renal_vein ✓;
Branch 4 (len=25mm) fails gate at len<40mm → side_branch ✓ (no longer misclassified as renal);
[RenalTag] log adds GATE FAIL line for blocked branches and Pair detected / Dominant candidate lines for positive topology signals;
topo_sc column added to per-branch score log;
all downstream code unchanged (role='renal_vein' tag drives everything)

### RenalStabilizer: add _stabilizeRenalOstia() called between _computeStableStarts and _refineOstia;
for each renal_vein branch walk forward from ostiumGi along centerline and find the last point within CONTACT_THRESH_MM=2.5mm of trunk KDTree

### this is the anatomical exit point where renal lumen separates from IVC wall;
flags branch as renal_stabilized → _refineOstia skips it (proximity w=2.5 would pull toward primary bif);
guards: Z-sanity clamp 120mm from bif_z, no-contact→no-change, no-advance→no-change, never moves past stableStartGi;
expected logs: [RenalStabilizer] Branch N: contact_len=X pts → ostium_gi=Y (was Z, +N pts, stabilized) and [RefineOstium] Branch N: skipped (renal_stabilized, gi=Y);
confidence grade expected to rise MEDIUM→HIGH as topology_uncertain resolves with better ostium placement

### RefineOstiumV4: replace surface-centroid approach (v179) with per-CL-point composite scoring model;
primary signal=proximity exp(-arc_from_bif/4mm) w=2.5 prevents renal drift (88-100mm);
neck=diameter-drop max(0,(D[i-k]-D[i+k])/D[i]) w=2.0;
normal=1-|dot(branch_dir,trunk_dir)| w=1.5;
curvature=1-dot(v_back,v_fwd) w=1.0;
radius_consistency=exp(-local_std_radial/1mm) w=1.0 (surface KDTree, lightweight, no normals/curvature filter);
penalties: dist_pen ramps after 10mm w=2.0, stable_zone_pen hard-1 at sgi w=1.5, cluster_pen=min(1,surf_pts_8mm/200) w=1.5;
search window hard-capped at 15mm arc from ogi;
surface used only for radius_consistency (no centroid, avoids v179 large-cluster drift);
score_log per branch for full debug;
ostium_method=scored in branchMeta;
log format [RefineOstium] Branch N: ogi=X->Y (+Npts, +N.Nmm) sgi=Z score=S.SSS [prox=... neck=... nrml=... curv=... rcns=... pen_d=... pen_s=... pen_c=...]

### RefineOstiumV3: RefineOstiumV3: surface-based ostium detection (primary path) with v178 CL-transition fallback;
primary path: extract 14mm surface patch around topological ogi, score vertices on C=curvature-magnitude w=0.35, N=normal-divergence-from-trunk-dir w=0.45, R=radial-expansion-gradient w=0.20 (all independently normalised), weighted centroid of score>0.55 cluster projected to nearest CL pt in [ogi,sgi], accept if proj_d<10mm and cluster>=6pts (auto-relaxes to p75 threshold on small patches);
trunk_dir from self.branches[0] geometry;
surface normals via vtkPolyDataNormals (SplittingOff);
curvature via vtkCurvatures Mean;
radial dist via scipy cKDTree on subsampled trunk CL;
commits ostium_method=surface in branchMeta;
Path B fallback=v178 CL logic (12mm cap, diam-grad+dir-vs-trunk+curvature) fires if modelNode absent, scipy missing, or Path A cluster/projection fails;
log tags: [RefineOstium] Branch N: ogi=X->Y surface cluster=Kpts proj_d=N.Nmm scores[c=... n=... r=...] for surface path, appends [CL-fallback] for centerline path;
carina dome NOT excluded from patch (classifySurfaceBranches dome-reclaim is coloring-only and runs later)

### RefineOstiumV2: RefineOstiumV2: replace "find stable region downstream" with "find strongest transition near bifurcation";
hard MAX_SHIFT_MM=12.0 cap on search window prevents +28-34mm drift;
signals redesigned: D=diameter gradient |d[i+2]-d[i-2]|/max(d[i],1e-3) w=0.45, A=direction-change-vs-trunk 1-|dot(branch_dir,trunk_dir)| w=0.35 (trunk_dir from self.branches[0] geometry), C=local-curvature 1-dot(v_back,v_fwd) w=0.20;
old signals removed (one-sided drop, direction-stability, trunk-separation-distance);
detection=max-change, confidence=downstream unchanged;
expected shift: iliacs ~2-8mm, renals ~1-5mm

### OstiumConfidenceV3: replace multiplicative-penalty confidence scoring with additive probabilistic fusion;
root cause of all-zero D scores: after branch trimming self.branches[bi][0]==ogi so shift=ogi-s=0 always;
new D signal uses flare_window size (sgi-s) as stability proxy, blended with 0.55 prior in surface-distance mode so no signal can hard-zero;
new A: sigmoid centred at 30° width 15° (was linear /90 giving low scores for iliacs at 24°/64°);
new L: length/(0.4×trunk_len)

### iliacs now score 0.7–1.0 vs 0 before;
new T: soft degradation only (no binary collapse), reference 8mm (was 5), neutral default 0.6 (was 0.5);
new G (weight 0.25, strongest): geometric sanity

### +0.40 ostium near bif/trunk, +0.30 length>60mm, +0.30 diam_ratio>0.40;
weights wD=0.20 wA=0.20 wL=0.20 wT=0.15 wG=0.25;
grades HIGH≥0.75 MEDIUM≥0.50 LOW<0.50 REJECT<0.30;
log adds ±uncertainty;
component keys D/A/L/T/G (was diameter/angle/lateral/topology);
penalties removed, flags=soft-warnings only;
report table and debug dialog updated;
results stored in branchMeta[bi][ostium_confidence];
QA bucket summary logged under [OstiumConfidence];
getBranchStats return dict gains confidence key;
generateReport bdata gains ost_conf, meas_summary includes grade+score per branch, per-branch Word doc detail table gains colour-coded Ostium confidence row (green=HIGH, amber=UNCERTAIN/LOW);
onDebugOstium dialog appended with per-branch confidence lines

### RefineOstia+OrangeMarker: add _refineOstia() called after _computeStableStarts;
for each non-trunk branch finds the true anatomical ostium (first cross-section where branch lumen is no longer shared with parent) by scoring every candidate pt in [ostiumGi, stableStartGi) on 3 normalised signals: (1) diameter drop (weight 0.40)

### negative gradient = leaving the shared funnel;
(2) direction stabilisation (weight 0.30)

### angle between dir[i] and dir[i+5], low angle-change = past the dome;
(3) separation from trunk centerline (weight 0.30)

### perpendicular distance to nearest trunk pt, increasing = branch is separating;
all 3 signals normalised independently to [0,1] over the window before weighting;
argmax of composite score → ostiumGiRefined stored in branchMeta;
ostiumGi overwritten to refined value;
self.branches[bi] start trimmed to match;
stableStartGi unchanged;
guard: window must be ≥3 pts, result always clamped to [ostiumGi, stableStartGi];
trunk (bi==0) skipped;
trunk pts subsampled ≤50 pts for speed;
expected: iliacs (window ≤3 pts) kept as-is;
renal branches (window ~10–15mm, ~10–15 pts) refined by ~1–2 mm;
log tag [RefineOstium] Branch N: ogi=X→Y (+Npts, +N.Nmm) sgi=Z window=Wpts scores[d=... a=... t=...]+BifZFix: snappedBifPt UnboundLocalError patched;
_bif_z_ref now uses trunkPtsOrdered[-1][2] (already built by BifTrunc/flow-norm above RenalTag) instead of snappedBifPt (assigned later at ostium-detection step);
fallback to _trunkOnlyZmin if trunkPtsOrdered is empty

### RenalCompositeScore: replace OR-gate renal classifier (diam_ratio>=0.45 OR angle>=45°) with a five-feature composite score (threshold 0.45);
features: diam_sc (ratio starts at 0.40, saturates at 0.70, weight 0.40), lat_sc (perpendicular tip offset from trunk axis: <15mm=0, >40mm=1, weight 0.30), z_sc (ostium must be ≥20mm above iliac bif, weight 0.15), angle_sc (departure angle: <20°=0, >70°=1, weight 0.10), stab_pen (proximal/distal ratio >1.25 → penalty up to 0.25 to suppress IVC-wall-bleed inflation);
lateral offset computed inline in RenalTag using trunkPtsOrdered (already available post flow-norm) identical to OrthoScore geometry;
branchMeta['renal_score'] written for every candidate (auditable);
old _RENAL_DIAM_RATIO/ANGLE_DEG constants retained for log compatibility;
expected on current dataset: Branch 3 (diam_ratio=0.52, lat=54.7mm, angle=31.5°) → score≈0.67 → renal ✓;
Branch 4 (diam_ratio=0.74, lat=20.9mm, angle=80.7°) → score≈0.42 → side (borderline, not blindly accepted) ✓;
snappedBifPt used as bif_z reference with safe fallback to _trunkOnlyZmin

### fix ref-window collapse for short branches in _detectFindings;
hard floors of max(int(len*0.55),20) / max(int(len*0.85),25) pts crushed the distal reference window on 49-pt renals (ref_lo=scan_from+26, ref_hi=scan_from+41 after clamps → 3-pt sliver → noisy _mean_ref_d → flare_ceil too low → diameter-walk exits instantly at stable_start==scan_from → stableStartGi never written → _annotateTreeDiameters falls back to ostiumGi → prox=18.7mm artifact survives v158 fix);
root cause: absolute pt floors don't scale with branch length;
fix: remove hard floors, use pure fractional windows (55%/85% of _branch_len), with a 5-pt minimum fallback using distal third

### on 49-pt renal this gives ref_lo=scan_from+26→27, ref_hi=scan_from+41 → but after clamp ref_hi stays valid since 49*0.85≈41 and e=49+scan_from, giving a proper ~14-pt reference window in clean distal tissue

### TreeDiamProxClamp: fix prox=18.7mm artifact in _annotateTreeDiameters for shallow-angle renal branches;
root cause: prox window uses first n/10 pts of seg_diams starting at raw gi_s

### for Branch 3 (renal, 49pts) that is pts 0-4 inside the IVC junction zone, giving prox=18.7mm even after median smoothing (the contamination spans many consecutive pts, not isolated spikes);
fix: look up stableStartGi from branchMeta for the matching branch and advance gi_s_prox to that index before computing the prox window;
full seg_diams (mean/min/max) unchanged;
prox now samples from first post-flare pts (~10-11mm), taper drops from +8.7mm to physiological ~1-2mm;
dist window unchanged (last 10% is always in clean distal zone)

### [v157] DiamSmooth: per-branch median smooth (WIN=4, 9-pt window ≈±6mm) applied to self.diameters after surface-distance _computeDiameters;
eliminates single-ray-hit spikes (1–2 pts wide) from thin-wall leakage and coarse mesh triangulation without blurring 10mm-scale taper or focal stenoses (≥5 pts=≥7mm survive);
applied per branch so ostium/bifurcation boundaries never bleed;
raw values preserved in self._rawDiameters for debugging;
skipped entirely in VMTK-radius mode (those are already smooth);
also retains v155 StableDiamFix and v156 getBranchStatsDistalCap

### [v156] getBranchStatsDistalCap: fix residual diameter contamination in getBranchStats for shallow-angle renal branches;
root cause: OUTLIER_K=2.0 in _computeDiameters uses per-point r_min as its reference, but at proximal end of Branch 4 (angle=31.5°) the centerline overlaps IVC lumen so r_min≈11mm (IVC radius), cap=22mm

### trunk-wall hits at 18–22mm pass;
fix: Pass A distal-anchored cap in getBranchStats using distal 55–85% trimmed mean × DISTAL_CAP_K=1.6, safety net keeps only if ≥50% pts survive;
Pass B IQR unchanged;
StableDiamFix also retained (v155)

### [legacy build note] v155-RayOutlierReject: fix trunk-wall contamination in _computeDiameters;
at renal branch points near the IVC junction some of the 8 radial rays exit through the thin renal wall and hit the adjacent IVC wall, returning a radius 3–5× larger than the true lumen (e.g. true r≈5mm, trunk-wall hit≈15mm);
these outlier hits inflate the per-point mean dramatically (prox=18mm instead of ~9mm), producing the Branch 4 taper anomaly (+8.7mm) and max=23.56mm spike;
fix: after collecting all ray hits, compute r_min = min(radii_hits) as the reference "true" radius (the closest wall is always the branch wall), then reject any hit > OUTLIER_K × r_min where OUTLIER_K=2.0;
this tolerates ±100% eccentricity for non-circular lumens (compressed IVC) while rejecting trunk-wall hits that are 3–5× the minimum;
fallback to full set if fewer than 2 hits survive rejection (prevents silent data loss on genuinely eccentric cross-sections);
root cause for asymmetry: Branch 4 angle=31.5° → shallow departure → majority of ray fan still overlaps IVC, Branch 5 angle=80.7° → perpendicular → rays exit IVC immediately;
OUTLIER_K=2.0 chosen conservatively: a 2:1 min:max ratio covers elliptical compression (IVC minor/major ~0.5 in May-Thurner) without allowing trunk bleed

### ReportEnrich: wire iliac L/R labels and renal ortho scores into generateReport;
bdata now collects role, angle_deg, lateral_label, ortho_score, tip_lateral_mm from branchMeta per branch;
AI prompt receives dedicated iliac_ctx (Left/Right Iliac: length, avg diam, angle) and renal_ctx (length, avg diam, angle, lateral offset mm, ortho_score) blocks in addition to meas_summary;
meas_summary rows now append angle and ortho inline;
ortho score legend added to prompt notes (>0.65 = strong lateral, <0.5 = soft/curved);
iliac naming propagates automatically via getBranchDisplayName which already returns 'Left Iliac'/'Right Iliac'

###  classifySurfaceBranches was treating iliac_right/iliac_left roles as SIDE_COLOR (orange) because the color assignment check only matched role=='main';
after v152 IliacLabel overwrites role to 'iliac_right'/'iliac_left' AFTER mainBranches selection, so the surface classifier saw an unknown role and fell to the else branch;
fix: extend both the branch-label classification (label='branch') and the MAIN_COLORS assignment to include 'iliac_right' and 'iliac_left' roles;
also extend widget UI main_names check;
all role=='main' checks in loadCenterline (mainBranches list, trunk continuation, etc.) run BEFORE IliacLabel and are unaffected

###  (1) IliacLabel: L/R iliac labeling for main branches using distal tip X-coordinate in LPS convention (larger X = anatomical Right, smaller X = anatomical Left);
requires exactly 2 main branches;
stores lateral_label ('Left'/'Right') and updates role to 'iliac_right'/'iliac_left' in branchMeta;
_iliacRightBi/_iliacLeftBi convenience attributes on logic;
getBranchDisplayName now returns 'Left Iliac'/'Right Iliac' when lateral_label is set;
FINAL ANATOMY print shows label inline;
_roleName handles new roles;
(2) OrthoScore: renal orthogonality score (0–1) stored as branchMeta[bi]['ortho_score'] and 'tip_lateral_mm';
combines angle_score (angle_deg/90, capped 1.0) and lateral_score (tip perp offset from trunk axis / 50mm ref, capped 1.0) at equal weight 0.5+0.5;
trunk axis computed from rawPoints root→bif;
applied to all renal_vein branches;
[OrthoScore] log shows per-component breakdown

###  make chord-skip angle sampling adaptive to branch length;
fixed 20mm skip+sample underestimates lateral angle for short branches (31mm renal: 77° → 41° with fixed 20mm because sample lands near the tip which curves back toward trunk);
fix: _adapt_mm = min(20mm, 0.25 × branch_length_mm), both skip and sample use _adapt_mm;
for 174mm iliac → 20mm (unchanged), for 74mm renal → 18mm, for 31mm renal → 7.8mm;
this keeps the sample in the proximal-quarter of each branch where departure direction is cleanest;
log updated to show per-branch adaptive skip+sample values and branch length

###  replace 15mm proximal angle sampling with chord-skip method;
previous approach sampled 15mm from ostiumGi which for iliac branches at the aortic bifurcation samples through the bif dome geometry

### one iliac departs nearly perpendicular to trunk (reads ~86°) while the other initially continues axially then bends (reads ~18°), both wrong;
root cause: the first 20-30mm at the bif is shared carina/dome geometry before each iliac takes its individual direction;
fix: two-phase chord

### Phase 1 skips CHORD_SKIP_MM=20mm past ostiumGi (past the dome), Phase 2 walks CHORD_SAMPLE_MM=20mm from there to get the direction vector;
this places the measurement window in stable individual vessel geometry for both iliacs;
reference direction unchanged (_tdir_ref = last 4 pts of trunk, axial);
fallback to tip if branch shorter than skip+sample;
applies to all branches (renals also benefit from the skip since their first few mm hug the trunk wall);
log updated to show chord skip+sample parameters

###  three fixes for BifTrunc fallout;
(1) DuplicateIliac: removed erroneous prepend of _continuation_pts to graphBranches in BifTrunc

### the continuation edge (lockedBifNode→distal) already exists in graphBranches from the graph walk;
after BifTrunc removes the distal node from trunkNodePath the edge survives _isTrunkSub naturally (distal node no longer in _trunkNodeSet);
prepending created a second identical copy producing duplicate iliac branches;
(2) DeupBothDir: _areSameReversed renamed _areDuplicateEdge and extended to catch same-direction duplicates in addition to reversed ones

### checks both A[0]≈B[0]+A[-1]≈B[-1] (forward) and A[0]≈B[-1]+A[-1]≈B[0] (reversed) within 5mm snap;
(3) BifDomeAngle: angle refinement for main branches whose ostium is at the primary bifurcation now skips a 10mm dome-clearance zone before walking the 15mm sample window

### the first ~10mm at the bif runs axially (shared dome geometry before divergence) causing falsely near-axial vectors (~10° instead of ~50°);
fix: if role==main and ostium is within 5mm of snappedBifPt, advance sample start by BIF_DOME_SKIP_MM=10mm before measuring departure direction

###  post-truncation of trunk at primary bifurcation node;
trunk chain extension greedily absorbs one iliac into the trunk path (correct for scoring) but semantically wrong

### aorta ends at the bifurcation, not inside an iliac;
fix: after trunkPtsOrdered is fully built, locate _lockedBifNode in trunkNodePath;
if it is an interior node (not already terminal), find the geometric cut point in trunkPtsOrdered using nearest-point to nodePos[_lockedBifNode] (trunkPtsOrdered geometry, not unreliable nodePos directly);
truncate trunkPtsOrdered and trunkNodePath at that cut;
re-expose the stripped continuation segment by prepending it to graphBranches so it survives dedup/sort/stitch as a peer iliac branch;
both iliacs now appear as co-equal terminal branches at the bifurcation;
trunk length, diameter stats, IVUS traversal, surface classification, and kissing stent logic all correct automatically;
guard: if _lockedBifNode is already at trunk terminal position (idx == len-1) no truncation is applied

###  ClipThenRing: autoDetectEndpoints now handles both open and closed meshes;
open mesh (boundary edges present) → cluster rings directly as before;
closed mesh (Segment Editor default export, boundary edges=0) → clip CLIP_R=8mm spheres at _findTipPoints locations to open the vessel ends, then run vtkFeatureEdges on the clipped result;
the boundary ring formed after clipping has its centroid inside the vessel lumen cross-section (not on the IVC outer wall) because the sphere removes the cap and the ring forms at the interior cross-section;
this is the key improvement over the old 3mm sphere approach which placed openings on the IVC wall surface

### 8mm radius reaches into the lumen;
full ring log shows centroid and radius for each detected opening regardless of path taken

### SegModelSelector: add dedicated Segmentation Model selector to Step 2 UI for endpoint detection;
Auto-detect was using self.modelSelector (Step 1 Vessel Model) which the user had pointed at the centerline tube model

### that mesh is closed by VMTK construction so vtkFeatureEdges returned no boundary rings and fell back to _findTipPoints;
fix: separate qMRMLNodeComboBox (segModelSelector) in Step 2 above the Auto-detect button, labeled "Segmentation Model", with hint label "⚠ Select segmentation surface for accurate endpoint detection" that turns green on selection;
onSegModelSelected handler enables autoDetectButton and updates hint;
onAutoDetect now reads from segModelSelector;
onModelSelected no longer controls autoDetectButton enable state;
the segmentation surface (open-boundary mesh from Segment Editor) and the vessel model (centerline tube or analysis model) are now cleanly separated

### BoundaryRingEndpoints: replace autoDetectEndpoints sphere-clip approach with direct mesh boundary ring detection;
previous method used _findTipPoints to find surface-extreme X/Z points, clipped 3mm spheres there to create fake openings, then ran VMTK autoDetectEndpoints on the opened surface

### for renal veins this placed fake openings on the IVC outer wall rather than inside the renal lumen because _findTipPoints picks widest-X-extent surface points, not vessel opening centroids;
VMTK then traced paths along the IVC wall instead of entering the renal vein;
fix: use vtkFeatureEdges(BoundaryEdgesOn) on the original mesh

### the segmentation already has real open-boundary rings at every vessel opening (IVC top, iliac tips, renal vein distal cuts);
BFS clusters boundary edge points into connected rings;
each ring centroid is the true VMTK seed point sitting inside the vessel lumen;
rings with <4 points are filtered as mesh artifacts;
centroids within 10mm are deduplicated;
full ring log shows count, centroid, and radius for each detected opening

### FlowNormSideBranch: fix flow normalization orientation for side branches (renal veins, collaterals);
main branches (iliacs) correctly use bif-end-first (closest end to primary bif node at [0]);
side branches were also flipped by bif proximity which is wrong

### their ostium is on the trunk wall, not at the iliac bifurcation, so both ends are far from bif_end and the flip was arbitrary;
fix: side branches detected by absence of any end within MAIN_BIF_SNAP_MM=30mm of bif_end are instead oriented ostium-first using min(dist(end, tp) for tp in trunkPtsOrdered)

### the trunk-closest end becomes [0] (ostium), distal tip becomes [-1];
this fixes Branch 1 (renal) running distal→proximal, which caused the renal tagger proximal departure sample to measure from tip toward IVC (near-axial, wrong direction) and the NoiseGate _bstart_z check to see the tip Z instead of the ostium Z

### TrunkReturnTipLateral: add tip lateral displacement as second escape hatch in TrunkReturn filter;
proximal angle (15mm sample) fails for renal veins that run parallel to IVC near the ostium before curving lateral

### the angle reads 6–20° even for genuine renals because the first 15mm is still dominated by IVC wall curvature;
fix: after proximal angle fails, project the branch tip onto the trunk axis and measure perpendicular offset (_lat_mm);
renal tips are 30–80mm off-axis (confirmed: X=−29.9 and X=+52.4 vs IVC centerline ~X=21–43);
IVC duplicate tips stay within 10–15mm of axis;
threshold=25mm;
log shows both angle and tip_lateral for every evaluated limb so future failures are fully diagnosable

### TrunkReturnNodePos: fix remaining nodePos usage in trunk-return lateral escape hatch;
root-end identification for the 15mm proximal departure sample was using nodePos[_tn_node] to find which end of _trPts is closest to the trunk junction

### nodePos after VMTK snap-and-collapse is unreliable (known issue throughout pipeline) so the wrong branch end was selected as root in many cases, causing the departure vector to be measured in the wrong direction (away from trunk instead of from it), producing a near-axial dot product that failed the 45° lateral gate even for genuine renal veins;
fix: replace nodePos[_tn_node] proximity with min(dist(end, tp) for tp in trunkPtsOrdered)

### trunkPtsOrdered is real geometric data built from cEdge samples and is always reliable;
this is the same pattern used everywhere else in the pipeline that previously depended on nodePos

### TrunkReturnFix: two bugs in the trunk-return lateral escape hatch that caused renal veins to be dropped even after the T3 fix;
(1) _trunkDir scope bug: '_trunkDir' in dir() does not look up local variables in nested scopes

### it was always falling back to [0,0,-1] (pure Z axis) instead of the real trunk direction, making the lateral angle check measure against Z and fail for renal branches that run mostly in Z;
fix: compute _trunk_axis_tr directly from trunkPtsOrdered (root→bif) inside the trunk-return block, which is always available at that point

### _trunkDir itself is not yet assigned until after flow-norm which runs after trunk-return;
(2) proximal sample window widened 8mm→15mm in trunk-return to match T3 fix

### first 8mm at a trunk node still follows trunk curvature and gives a falsely axial departure vector;
same fix applied to renal tagging _renalTrunkAxis which now uses _trunkDir directly (it is reliably set before renal tagging runs at step 8+)

### VesselTypeUX: three follow-up fixes for the unchosen vessel type;
(1) vesselTypeHintLabel added below combo

### shows orange warning "⚠ Set vessel type before running analysis" when unset, turns green "✓ Arterial/Venous thresholds active" on selection;
(2) _pendingVesselType stored on widget in onVesselTypeChanged so the chosen type survives logic re-instantiation

### onLoadIVUS and onReAnalyze both apply it to the fresh VesselAnalyzerLogic();
initialized to None in __init__;
(3) report header: Modality now reads "CT Venography" for venous or "CT Angiography" for arterial (was hardcoded "CT Venography"), and a "Vessel type:" row is added to the right header column showing "Arterial" or "Venous"

### report is now unambiguous about which thresholds were applied

### UIReorder: (1) Vessel Type combo moved to Step 1 immediately after Vessel Model selector, default index 0 = "-- Select vessel type --" (unchosen) so user must explicitly choose Arterial or Venous before analysis;
onVesselTypeChanged updated for new index mapping (0=blank, 1=Arterial, 2=Venous);
duplicate vesselTypeCombo creation removed from Stent Planner;
(2) Re-run Surface & Analysis button added to Step 1, enabled once a model is selected, calls onCreateSurface then onLoadIVUS if a centerline is already present

### fixes the "no way to re-model after surfacing" gap;
(3) Surface Model, Branch Surface Classification, and Inspect Ostium Detection collapsibles moved from bottom of panel (after Export) to immediately after Step 3 navigation controls, before Measurements

### logical flow: navigate → inspect surface → measure

### RenalFix: three-part fix to recover renal veins eliminated by the T3 loop filter;
(1) T3 proximal sample window widened 8mm→15mm so lateral-angle guard escapes trunk curvature zone at short-branch roots (first 8mm at a trunk/bif node is dominated by trunk direction, causing renal branches to appear falsely axial);
(2) T3 confirmed-bif drop condition changed from _root_deg>=4 OR _longer_siblings>=3 to _root_deg>=5 only

### sibling-count path removed because a renal vein at a degree-3 trunk bif always has 3+ longer siblings (trunk-in, trunk-out, iliac) and was systematically eliminated;
(3) renal tagging angle computation upgraded from end-to-end vector (bpts[-1]-bpts[0]) to proximal departure vector (first 15mm walk, matching T3/trunk-return style) so angle is measured from the true lateral departure, not the global displacement direction

### these three fixes together allow short lateral branches (20–80mm, 45°+ departure) to survive the loop filter and be correctly tagged as renal vessels

### LateralAngle: fix end-to-end direction vector bias in both T3 loop filter and trunk-return lateral escape hatch;
both now use proximal departure direction (first 8mm from the root/trunk node end, identified by nearest-point to nodePos) instead of bpts[-1]-bpts[0];
end-to-end vectors for short branches rooted at bif nodes are dominated by graph walk direction and appear falsely axial even for lateral vessels (renals)

### TreeDiam: fix cross-branch gi contamination;
per-branch KD-trees + best_branch_for_seg matching ensure each segment queries only its own branch gi range, preventing trunk-return segments from mapping to trunk diameters

### Segment-level diameter annotation: _annotateTreeDiameters() populates diam_mean/min/max/prox/dist/taper/gi_start/gi_end on every branchTree segment;
KD-tree nearest-point lookup;
called automatically after _computeDiameters;
[TreeDiam] log shows per-segment profile with taper and min-diameter warnings

### BranchTree: fix node 52 type (use _trunkPathIntermediate set, not nb_count);
move _log_tree call to after pass-2 so log shows final primary_bif_entry role

### Fix BranchTree NameError: removed bifScores from _annotate_tree (pass 1);
added _annotate_primary_bif (pass 2) after trueBifNode is resolved

### no more free-variable error on loadCenterline

### BranchTree fixes: trunk-path intermediate nodes correctly typed as bifurcation;
primary bifurcation node marked with is_primary_bif + primary_bif_entry segment role;
log shows star marker on primary bif

### Parallel branch tree (self.branchTree): DFS build from compressed graph, trunk/branch segment annotation, arc_mm per segment;
self.branches unchanged

### zero downstream impact

### Length-gated finding detection: runs shorter than MIN_LESION_MM=5.0mm dropped regardless of point count (replaces implicit point-count heuristic);
arc_mm shown in all finding descriptions for clinical confidence assessment

### Finding detection: FLARE_HARD_MM 30→45mm;
median_d now uses distal _mean_ref_d (flare-free) as primary reference instead of branch_upper from healthy_diams which could still include flare bleed

### Flare detection: two-pass stable_start (diameter walker + 30mm hard arc floor);
eliminates false aneurysm at branch proximal end regardless of MAX_FLARE_PTS cap

### Fix flare reference: sampled from distal 55-85% of branch (not proximal 15-35pts) so flare_ceil is correct and stable_start advances past flare zone;
venous thresholds 2.0x aneurysm / 1.5x ectasia;
vessel-type-aware clinical message (venous: compression pattern, not kissing stent)

### Fix: dilation scan now starts from stable_start (post-flare) instead of s+3;
eliminates false aneurysm detection in bifurcation flare zone

### Vessel-type flag (Arterial/Venous): UI combo in Stent Planner;
venous mode raises aneurysm threshold to 1.95x and ectasia to 1.56x;
bifurcation-zone dilation suppression (25mm dome, smooth expansion only) prevents false aneurysm at IVC/iliac Y-junction

### FallbackCommit: replace single-path commit guard (stable_start>scan_from) with two-path logic;
Path A: advance ≥2pts → commit as authoritative flare end;
Path B: no advance (clean join/flat profile) → walk hard 10mm arc skip from scan_from and commit as fallback stableStartGi;
guarantees prox sampling never starts inside ostium zone regardless of flare shape;
new log tag [VesselAnalyzer] Branch N flare fallback: gi=X (+N pts, Y.Ymm arc skip) distinguishes Path B commits from Path A

### StableStartEarly: root-cause fix for prox=18.7mm artifact;
flare-walk that sets stableStartGi lived inside _detectFindings (button-click only), so _annotateTreeDiameters always ran with stableStartGi==ostiumGi → guard (gi_s < _sgi) always False → gi_s_prox never advanced → prox sampled from flare zone;
fix: extract flare-walk into _computeStableStarts() and call it from _computeDiameters after smoothing, before _annotateTreeDiameters

### runs unconditionally on every loadCenterline;
also fix guard in _annotateTreeDiameters from (gi_s < _sgi) to (_sgi > gi_s) which is identical but explicit;
new log tag [StableStart] Branch N: gi=X for Path A, [StableStart] Branch N: fallback gi=X for Path B

### ProxWindowRefine: two improvements to prox diameter estimation in _annotateTreeDiameters;
(1) PostStableOffset: after advancing gi_s_prox to stableStartGi, walk a further PROX_POST_OFFSET_MM=5mm of arc distance (capped at 20% of remaining segment) to clear residual non-cylindrical junction geometry

### stableStartGi marks end of flare zone but the next few pts can still be elliptical/oblique, pulling mean to 15mm instead of ~10mm on the 74mm renal;
(2) TrimmedMinimum: replace _trimmed_mean on prox window with lower-50%-median (sorted window, take median of bottom half)

### more robust at proximal end where occasional inflated samples from wall-blending can dominate a 3-pt mean;
together these should reduce prox from 15mm to ~9-11mm on the 74mm renal without affecting dist (dist uses full seg end window, unchanged)

### RenalSizing: two follow-up changes after prox diameter fix;
(1) ProxSanityGuard: after computing prox in _annotateTreeDiameters, clamp to PROX_SANITY_K=1.4×dist if exceeded

### no vessel widens >40% proximally in a single segment;
logs [TreeDiam] Branch N prox inflation: Xmm → clamped to Ymm when triggered;
protects against future regressions without masking real anatomy;
(2) DistAnchor: write proxD/distD back to branchMeta from _annotateTreeDiameters so bdata can expose them in generateReport;
renal context string now reports distal Ø as (sizing) anchor and proximal Ø for context

### AI prompt receives the clinically correct sizing diameter instead of the mildly-inflated avg;
iliac context unchanged (avg is reliable there, no proximal bias)

### MinLocation: add min-diameter location to ⚠MIN log tag in _annotateTreeDiameters;
finds index of minimum in seg_diams, computes % along branch (0%=proximal, 100%=distal);
labels as distal-tip (≥80%), proximal (≤20%), or N% along otherwise;
distinguishes distal mesh cap artifact from mid-branch stenosis without any extra computation cost;
example: ⚠MIN=6.8mm@distal-tip vs ⚠MIN=6.8mm@62% along

### StenosisPreFlag: consume stenosisCandidate from branchMeta in _detectFindings;
HIGH/MED confidence narrowings (non-distal-tip, ≥20% drop) written by _annotateTreeDiameters ⚠MIN block are now seeded as Focal Narrowing findings before the run-scanner loop;
this catches genuine focal narrowings in curved renals where pancake_ratios<1.1 (low ellipticity) prevents the main stenosis run-scanner from triggering;
finding dict has source=preFlag to distinguish from run-scanner findings;
log tag [FindingsPreFlag] Branch N: desc;
for Branch 4 in current dataset: 6.8mm@69% along, drop=42%, run~5mm → HIGH confidence → appears in findings list and UI

### ContiguousRun: fix non-contiguous run counting in ⚠MIN stenosis detection;
previous sum(1 for d in _seg_d if d<thresh) counted all below-threshold pts regardless of adjacency, inflating run_mm (21.2mm was scatter not a solid zone);
fix: track longest unbroken streak of below-threshold pts, store gi_run_start/gi_run_end of that streak in stenosisCandidate;
FindingsPreFlag updated to use gi_run_start/gi_run_end for runStart/runEnd/runLen so the finding correctly spans the narrowing zone rather than pointing at a single gi;
run_mm is now the true physical extent of the worst contiguous narrowing

### TrunkDotGate: add directional dot-product gate to trunk chain extension;
previously only usage==max_usage and Y-bif arc-ratio checks were applied;
a high-usage side branch that happens to share max-usage with the trunk continuation could leak into the trunk chain in noisier graphs;
fix: after usage gate, compute dot(current_trunk_direction, candidate_edge_direction) using nodePos;
reject extension if dot<-0.3 (allows up to ~107°, only blocks u-turns and near-perpendicular exits);
requires ≥2 confirmed trunk nodes for direction estimate;
threshold is intentionally permissive (not 0.7) to avoid rejecting curved trunks

### only prevents geometric u-turns from being absorbed;
no change expected on current dataset (all extensions are axial), provides safety for future cases

### NarrowingPhenotype+BifSkew: (1) NarrowingPhenotype: add phenotype classifier to ⚠MIN block;
artifact-like = drop>35% AND run<25mm (sharp localized drop, likely mesh pinch or ray cluster);
compression-like = drop>=30% AND run>=25% of branch arc (diffuse extrinsic narrowing, e.g. May-Thurner);
focal = drop>=20%;
mild otherwise;
phenotype stored in stenosisCandidate and shown in ⚠MIN log tag and FindingsPreFlag description;
current case (42% drop, 21.6mm run, branch=74mm, fraction=29%) → compression-like;
(2) BifSkew: detect asymmetric bifurcation angle spread in difficulty classification;
if max_angle - min_angle > 30° append ", skewed (spread N°)" to difficulty string;
store _bifSkewed and _bifAngleSpreadDeg;
current case (64.4° vs 24.9° = spread 39.5°) → complex + skewed, clinically consistent with May-Thurner anatomy where the right iliac artery crossing angle influences bilateral iliac departure angles

### ArtifactCorrection: add _applyArtifactCorrection() called after first _annotateTreeDiameters pass;
for each branch with stenosisCandidate.phenotype==artifact-like, replace the contiguous below-threshold run (gi_run_start..gi_run_end) in self.diameters with the median of clean flanking windows (FLANK_PTS=8 pts each side, MIN_FLANK=3 pts);
re-runs _annotateTreeDiameters after correction so [TreeDiam] log shows corrected values;
only artifact-like is corrected

### compression-like and focal phenotypes preserved as clinically meaningful;
expected: Branch 3 6.8mm run replaced with ~10mm flank median, ⚠MIN warning disappears on second annotation pass, getBranchStats/report/stent sizing all see corrected values;
correction is auditable via [ArtifactCorr] log tag showing before/after and flank sample count

### MicroSegFilter: suppress micro-segments (<10mm) from bifurcation count in _count_tree;
node 318 (3.1mm) was inflating BranchTree bifurcation count to 4 (vs 3 anatomical);
fix: _count_tree only counts a bifurcation if its connecting segment arc >= MICRO_SEG_MM=10mm;
node 318 (3.1mm) → excluded;
node 322 (13.7mm) → still counted (marginally above threshold, is a real trunk-renal junction);
BranchTree log adds [micro] tag to short bifurcation nodes for visual identification without suppressing them;
tree structure and traversal are unchanged

### only the reported bifurcation count changes

### RenalConsolidate: add post-RenalTag pairwise merge step that detects over-segmented renal junctions;
signature: two renal_vein branches whose ostia are within MERGE_DIST_MM=25mm, tips on same lateral side (same X sign vs trunk), departure vectors within MERGE_ANGLE_DEG=50°;
resolution: mark shorter branch role=renal_fragment with fragment_of=primary_bi, primary gets renal_group list;
self.branches/gi ranges unchanged

### role tag only;
renal_fragment excluded from: nav traversal, OrthoScore, surface color groups, report renal context, getBranchDisplayName (returns RF{i});
expected: Branch 5 (31mm) tagged as fragment of Branch 4 (74mm), [Nav] traversal drops from 2 renals to 1, report shows one renal vein instead of two;
log tag [RenalConsolidate] Branch N (Xmm) → fragment of Branch M (Ymm);
criteria logged for auditability

### RenalConsolidateFix: fix same-side criterion that blocked merge;
root cause: tip X vs single root X gave false opposite-side result for short branch whose VMTK endpoint wanders;
fixes: (1) Z-interpolated trunk reference X at each branch ostium/tip level;
(2) for branches <40mm use ostium X instead of tip X (short branch tips unreliable);
(3) waive side check entirely when ostia are <15mm apart (coincident origin means same-side by definition);
add diagnostic log [RenalConsolidate] Skip or Candidate pair showing which criterion fires;
expected: Branch 4 (31mm) → fragment of Branch 3 (74mm)
