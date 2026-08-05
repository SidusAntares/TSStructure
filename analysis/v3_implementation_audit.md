# Structure DA V3 implementation audit

This report describes code reachable from `train.py` after the compact-log and
feature-snapshot update.

## Runtime call graph

```text
train.py
|- closed-set split construction
|- source train + target train loaders
|- source validation loader (checkpoint selection)
|- target test loader (final evaluation only)
`- train_joint_structure_da
   `- joint_structure_da_train_step
      |- StructureBackbone
      |  |- PixelSetEncoder
      |  `- SymmetricTimeKernelDecomposition: H -> T + D + R
      |- TrendStructureTemporalCore
      |  |- independent T/S functional lift and SRVF extraction
      |  |- source-only running SRVF references
      |  `- T-led, S-disambiguated phase selection
      |- TrendStructureTaskFeatureModule
      |- PhaseAwareTwoScaleClassifier
      |  |- TrendStructureSharedLTAE
      |  |- TwoScaleQualityFusion
      |  `- final task classifier
      |- PhaseAwarePrototypeAlignment
      `- successful-step source-state update
```

The target training sample has no class label. Target per-class diagnostics use
the model's gated pseudo label, never target ground truth. Checkpoint selection
uses source validation labels; target labels are first consumed by final test
evaluation.

The final fused feature has no global domain discriminator or gradient-reversal
branch. `TwoScaleQualityFusion` retains its internal domain classifiers because
they produce quality scores and are a distinct part of the quality objective.

## Active V3 modules

`backbone.py`, `decomposition.py`, `feature_snapshots.py`, `full_model.py`,
`joint_trainer.py`, `phase_aware_objective.py`, `temporal_module.py`,
`temporal_coordinates.py`, `temporal_registration.py`, `temporal_head.py`,
`temporal_selection.py`, `temporal_srvf.py`, `temporal_functional.py`,
`quality_fusion.py`, `representation.py`, and `TrendStructureSharedLTAE`.

Feature snapshots use deterministic train-split parcels, `model.eval()`, and
`torch.inference_mode()`. They store raw PSE token curves and accepted aligned
structure SRVFs without updating model state.
