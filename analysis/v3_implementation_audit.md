# Structure DA V3 implementation audit

Audit base: `8e63a849853e4737d8a1e020163690e314a9943b`.

This report describes the code that is reachable from `train.py`. It does
not infer or recreate design documents that are not present in the repository.

## Runtime call graph

```text
train.py
├─ closed-set split construction
├─ source train loader + target train loader
├─ source validation loader (checkpoint selection)
├─ target test loader (final evaluation only)
└─ train_joint_structure_da
   └─ joint_structure_da_train_step
      ├─ StructureBackbone
      │  ├─ original PixelSetEncoder
      │  └─ SymmetricTimeKernelDecomposition: H -> T + D + R
      ├─ geometry phase
      │  └─ TrendStructureTemporalCore
      │     ├─ independent T/S functional lift and SRVF extraction
      │     ├─ source-only running SRVF references
      │     ├─ one multi-candidate monotone warp estimator
      │     └─ T-led, S-disambiguated phase selection
      ├─ task phase
      │  ├─ TrendStructureTaskFeatureModule
      │  │  ├─ S = T + D
      │  │  ├─ one accepted warp applied to both scales
      │  │  ├─ TrendStructureCoordinates
      │  │  └─ ShapeFeatureEncoder
      │  ├─ PhaseAwareTwoScaleClassifier
      │  │  ├─ TrendStructureSharedLTAE
      │  │  ├─ TwoScaleQualityFusion
      │  │  └─ final task classifier
      │  ├─ PhaseAwarePrototypeAlignment
      │  └─ EDENFusedFeatureAlignment + GRL
      └─ successful-step source-state update
```

The target training sample has no class label. Target per-class diagnostics
use the model's gated pseudo label, never the target ground-truth label.
Checkpoint selection uses source validation labels. Target labels are first
consumed by the final target test evaluation.

## Implemented V3 semantics

- Backbone tokens have shape `[B, L, D]`; the decomposition returns T, D,
  and R with the same shape and reconstructs H.
- The active temporal task path uses T and `S = T + D`. R is not a temporal
  task branch.
- T and S have independent functional/SRVF extraction state but share one
  candidate warp estimator. T proposes phase; S only disambiguates the
  T-near candidate set.
- Shape coordinates are derived from the aligned complete S representation.
  The phase coordinate remains diagnostic; the task representation consumes
  T/S LTAE embeddings and the Shape feature.
- Quality produces `alpha_trend` and `alpha_structure`. The final feature
  is the concatenation of the two alpha-weighted LTAE embeddings and the
  unscored Shape feature.
- The global adversarial loss operates on the final fused feature through the
  warm-start GRL. Prototype losses keep their source-only state and gated
  target-teacher semantics.
- Geometry parameters and task parameters are disjoint. Geometry is stepped
  first; geometry parameters are frozen during task backward. Source running
  state is updated only after a successful task optimizer step.
- AMP skip handling gates the source-state update and scheduler step.
- Validation `occlusion_*` metrics are feature-occlusion diagnostics, not
  retrained ablations.
- Saved best checkpoints contain model state and metadata for evaluation.
  They do not contain optimizer/scaler state, so the current checkpoint is an
  evaluation checkpoint rather than a complete interruption-resume package.

## Reachability classification

### Active V3

`backbone.py`, `decomposition.py`, `full_model.py`,
`joint_trainer.py`, `phase_aware_objective.py`, the T/S portions of
`temporal_module.py`, `temporal_coordinates.py`,
`temporal_registration.py`, `temporal_head.py`, `temporal_selection.py`,
`temporal_srvf.py`, `temporal_functional.py`, `quality_fusion.py`,
`representation.py`, `eden_alignment.py`, and
`TrendStructureSharedLTAE`.

### Baseline and shared low-level code retained

- Original `PixelSetEncoder`, `LTAE`, `PseLTae`, decoder and classifier
  wrappers remain available to the TimeMatch source-supervised baseline.
- Low-level monotone warp candidates, inverse warp, SRVF group action,
  source-running templates, phase tangent conversion, continuous time
  encoding, and decomposition remain reachable from V3 or their public
  low-level tests.
- Offline analysis continues to use the real decomposition implementation.

### Confirmed unreachable legacy high-level code removed

- No `channel_module.py` existed at the audited base, so no channel file was
  deleted or synthesized.
- Old independent T/D temporal extractor, shared wrapper, pair outputs and
  their geometry-control flow.
- Old joint shape/phase output head and separate phase/shape coordinate
  encoders.
- Old T/D/R component classifier and component-aware shared LTAE wrapper.
- Old hierarchical quality fusion/objective and hierarchical output bundles.
- Old registration wrapper, old temporal coordinate wrapper, and the old
  monolithic temporal geometry objective.
- Tests whose only purpose was to instantiate those removed high-level APIs.

No compatibility shells or commented copies were retained. Package exports
now contain only real active or shared low-level symbols. A runtime-source
test also prevents the removed high-level names from reappearing under
`methods/` or `models/`.

## Findings

| Severity | Finding | Resolution |
| --- | --- | --- |
| MAINTENANCE | V1/V2 high-level classes were still importable despite being unreachable from training. | Removed implementations, exports, imports, and dedicated tests. |
| MAINTENANCE | `benchmark_structure_da_step.py` used constructor arguments that no longer exist. | Replaced by the runnable V3 JSON benchmark. |
| PERFORMANCE | Per-class phase diagnostics copied many sample tensors to CPU and used Python class/sample materialization. | Aggregation now uses batched `index_add_`/`bincount`, two packed host transfers, and a bounded tensor cache for p95. |
| CORRECTNESS | No V3 method, loss, protocol, default hyperparameter, state update order, or target-label boundary defect was found in this audit. | Protected by current integration, gradient-partition, AMP, loader-protocol, and runtime-surface tests. |

There are no unresolved BLOCKER findings.

## Equivalence and performance evidence

- The fixed synthetic model has 983 parameters and 160 state-dict keys both
  before and after cleanup.
- Existing forward, loss, gradient partition, source-state update, AMP,
  target-teacher, candidate-selection, and closed-set tests pass unchanged.
- Per-class diagnostic tests cover conditional denominators, dynamic candidate
  heads, exact bounded-cache order, merge behavior, and p95.
- Same-process old/new CPU whole-step comparison (two 20-iteration runs per
  implementation) produced median pairs of 84.95/84.83 ms for the old
  accumulator and 84.40/87.10 ms for the vectorized accumulator. The paired
  median average changed by about +1.0%, inside the 3% non-regression limit.
- An isolated CUDA aggregation microbenchmark with `B=128`, 12 classes, 5
  candidates and 18 metrics measured 1.518 ms before and 1.264 ms after
  (1.20x). This is evidence for the diagnosed aggregation hotspot only, not a
  claim that the complete GPU training step is 1.20x faster.
- The committed benchmark uses small synthetic data, reports mean/median/p95
  and CUDA peak allocated memory, and explicitly identifies itself as a
  training benchmark with state updates. The required CPU run (3 warmups,
  10 measured iterations) reported 86.382 ms mean, 85.473 ms median and
  92.990 ms p95; CPU peak-memory reporting is intentionally `null`.

CPU whole-step timings are sensitive to host load; individual process medians
varied substantially. Therefore no general CPU or GPU end-to-end speedup is
claimed. No higher-risk source/target core merge or Shape-forward caching was
retained because the available baseline did not establish a safe measurable
benefit for those changes.

## Final boundaries

- Original TimeMatch baseline path is preserved.
- V3 parameter count and state-dict key set are unchanged.
- Target true labels do not enter training or checkpoint selection.
- Source balancing remains source-only; target, validation and test loaders
  remain unbalanced under the existing protocol.
- No method mathematics, loss weight, candidate count, canonical grid,
  default hyperparameter, dataset split, or experiment result was changed.
- Final Python regression result: 482 passed and 1 skipped. The four remaining
  failures are the pre-existing Windows WSL `bash -n` launch checks
  (`CreateInstance/E_ACCESSDENIED`); all 11 script-content tests pass.
