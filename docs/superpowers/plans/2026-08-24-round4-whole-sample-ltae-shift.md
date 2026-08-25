# Round 4 Whole-Sample LTAE Shift Implementation Plan

> **For agentic workers:** Execute inline with strict red/green tests. Do not commit, push, merge, remove PSE, or launch the full 12-task run.

**Goal:** Move the TimeMatch global temporal translation from ReIMTS/mTAN observation times to one whole-sample LTAE sequence while caching all pre-LTAE computation during candidate search.

**Architecture:** PSE and ReIMTS/mTAN consume only real observation timestamps. The lowest 4×8 latent tokens receive absolute calendar reference positions, flatten chronologically to 32 tokens, and enter one LTAE/classifier call with a single sample-wide shift. TimeMatch detects this cache capability while preserving the baseline PseLTAE path.

**Tech Stack:** Python, PyTorch, pytest, Bash launcher.

**Spec:** User-provided Round 4 requirements in the current task.

## Global Constraints

- Preserve PSE and the 1→2→4 ReIMTS/IARF topology.
- Defaults remain levels=3, scale_factor=2, period=365, reference_points=8, latent_dim=128, heads=1.
- No shift may enter patch membership, mTAN, IARF, or recursive fusion.
- No temporal pooling before the one whole-sample LTAE call.
- Formal source and target losses are sample-level; patch occupancy remains diagnostic only.
- Baseline PseLTAE numerical shift behavior is unchanged.
- Do not run the complete 12-task launcher or perform git commits.

---

### Task 1: Real-Time Patch Metadata and Absolute References

**Files:**
- Modify: `models/reimts/recursive_temporal.py`
- Test: `tests/test_reimts_mtan.py`

**Interfaces:**
- `gather_period_patches(features, observation_positions, patches, period) -> PeriodPatchBatch`
- `RecursiveTemporalEncoder.forward(spatial_features, observation_positions) -> RecursiveTemporalOutput`
- Output exposes lowest tokens, validity, absolute reference positions, and per-scale observation positions/validity.

- [ ] Add tests using `[3,17,41,96,103,188,249,360]` to prove gathered timestamps are an exact subset of the input.
- [ ] Add tests proving candidate shifts cannot alter membership or cached ReIMTS output.
- [ ] Add literal expected tests for four ordered blocks of eight absolute calendar positions.
- [ ] Run the focused tests and confirm failures are caused by the Round 3 API/coordinates.
- [ ] Remove shifted encoder positions from recursive encoding and construct each patch's absolute reference coordinates.
- [ ] Re-run the focused tests to green.

### Task 2: One Chronological Sequence and One LTAE

**Files:**
- Modify: `models/reimts_classifier.py`
- Test: `tests/test_reimts_mtan.py`

**Interfaces:**
- `ReIMTSShiftFeatures(tokens, positions, patch_valid, timing)` stores `[B,32,D]` and `[B,32]`.
- `prepare_shift_features(...)` runs PSE plus ReIMTS once.
- `forward_from_shift_features(features, shift, return_feats=False)` runs only `positions + shift`, LTAE, and classifier.

- [ ] Add tests for 11111111/22222222/33333333/44444444 flatten order, tensor-position alignment, and no pre-LTAE pooling.
- [ ] Add call-count tests proving ordinary forward calls LTAE and classifier once.
- [ ] Add tests proving shifts modify only LTAE positions and `return_feats` is the final sample embedding.
- [ ] Confirm RED against the Round 3 patch decoder.
- [ ] Implement preparation, chronological flattening, whole-sample decode, and sample-level diagnostic output.
- [ ] Make patch loss mode fail fast and verify strict state-dict round-trip equivalence.

### Task 3: Cached Candidate Search and Runtime Logging

**Files:**
- Modify: `timematch.py`
- Test: `tests/test_timematch_reimts.py`
- Test: `tests/test_structured_logging.py`

**Interfaces:**
- Capability path calls `prepare_shift_features` once per sampled batch and `forward_from_shift_features` once per candidate.
- Baseline fallback retains `spatial_encoder → temporal_encoder(positions + shift) → decoder` exactly.

- [ ] Add a 121-candidate call-count test for PSE=1, ReIMTS/mTAN=1, LTAE=121, classifier=121.
- [ ] Add structured-log assertions for spatial, ReIMTS/mTAN, total preparation, candidate total, mean candidate, and total runtime.
- [ ] Confirm RED against the PSE-only cache.
- [ ] Implement capability detection, accumulated timings, and the expanded shift diagnostic block.
- [ ] Re-run TimeMatch and logging tests, including baseline regression.

### Task 4: Sample-Level Training and Launchers

**Files:**
- Modify: `train.py`
- Modify: `timematch.py`
- Modify: `closedset_scripts/run_reimts_mtan_closedset_12tasks_4gpu.sh`
- Modify: `closedset_scripts/smoke_reimts_mtan_closedset.sh`
- Test: `tests/test_train_reimts_registration.py`
- Test: `tests/test_reimts_12task_launcher.py`
- Test: `tests/test_reimts_smoke_script.py`

**Interfaces:**
- Formal ReIMTS loss mode defaults to and requires `sample`.
- Source/teacher/student logits remain `[B,C]`; no labels are repeated.

- [ ] Add tests proving source and target criteria receive `[B,C]` and `[B]`, and patch mode fails fast.
- [ ] Update CLI default/help, training helpers, launcher flags, and smoke flags.
- [ ] Preserve occupancy collection through `patch_valid` without using it as loss weights.
- [ ] Run registration, launcher, TimeMatch debug, and progress-bar tests.

### Task 5: Fresh Verification and Runtime Evidence

**Files:**
- No production changes unless a fresh failure receives its own regression test.

- [ ] Run the six required focused test files plus smoke/progress tests.
- [ ] Run `python -m pytest -q`, `git diff --check`, `git diff --cached --check`, `python train.py --help`, and Bash syntax validation.
- [ ] Inspect available data, CUDA, checkpoints, and Round 3 baseline artifacts without launching all tasks.
- [ ] If available, run DK1 source 1 epoch and DK1→FR1 1 epoch/2 steps/sample_size=1 with progress off.
- [ ] Benchmark identical batch/range/sample_size on Round 3 and Round 4; otherwise report the exact missing prerequisite.
- [ ] Report branch/HEAD, status, diff stat, implementation invariants, verification evidence, smoke/runtime evidence, and remaining risks.
