# ReIMTS+mTAN TimeMatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-level ReIMTS+mTAN classifier to the closed-set TimeMatch debug branch while preserving all baseline protocol and temporal-shift semantics.

**Architecture:** Run the existing PSE once, split its `[B,T,128]` output into timestamp-defined 1/2/4 patches, encode each scale with an independent ReIMTS-compatible mTAN encoder, and recursively fuse only temporal representations. Decode the four lowest-scale representations through one shared LTAE and one shared classifier, then average patch logits to return `[B,C]`.

**Tech Stack:** Python, PyTorch, NumPy, pytest, existing TSStructure PSE/LTAE/TimeMatch utilities.

**Spec:** User-provided “给 Codex 的完整实现要求” in the 2026-08-23 task.

## Global Constraints

- Work only on `dec-test-ReIMTS`, based on `benchmark/closedset-timematch`; do not switch to or merge `main`.
- Preserve unrelated working-tree changes and do not modify the baseline branch.
- Preserve the closed-set mapping, folds, pseudo threshold, EMA, focal loss, shift estimator formula, augmentation, dates, and baseline defaults.
- PSE runs exactly once per model forward and once per target batch during shift estimation.
- Patch membership uses unshifted raw timestamps; mTAN receives shifted encoder timestamps.
- Fixed first-round hierarchy is 3 levels with 1/2/4 patches over a 365-day period.
- Scale encoders are independent; patches within a scale share their encoder.
- No interpolation/resampling, additional decomposition, attention branch, DA loss, mTAN decoder/classifier/loss, or copied PyOmniTS training stack.
- Shared LTAE and classifier operate on the four lowest patches; sample output is the mean of patch logits.
- All public forwards used by TimeMatch return `[B,C]` in train, validation, and test.
- Do not push or merge after implementation.

---

## File Structure

- `models/reimts/__init__.py`: public exports for the temporal encoder package.
- `models/reimts/mtan_encoder.py`: minimal mTAN attention encoder returning the official sampled `z0` temporal representation semantics, adapted without reconstruction/classification heads.
- `models/reimts/recursive_temporal.py`: timestamp patch membership/gathering, masks, SplitOrDuplicate, IARF, and three-scale recursion.
- `models/reimts_classifier.py`: one PSE, recursive encoder, one LTAE, one classifier, patch-logit averaging, and shifted-forward capabilities.
- `timematch.py`: capability dispatch for all shifted paths while retaining the exact baseline call path.
- `train.py`: model construction and CLI arguments only.
- `tests/test_reimts_mtan.py`: split, mTAN, IARF, decoder, checkpoint, and model-shape contracts.
- `tests/test_timematch_reimts.py`: baseline regression, shift topology/coordinates, PSE call count, and pseudo-label contracts.
- `closedset_scripts/smoke_reimts_mtan_closedset.sh`: one-fold minimal source + TimeMatch run using the existing dataset/task naming.

---

### Task 1: Timestamp Period Splitting

**Files:**
- Create: `models/reimts/__init__.py`
- Create: `models/reimts/recursive_temporal.py`
- Create: `tests/test_reimts_mtan.py`

**Interfaces:**
- Consumes: `features: Tensor[B,T,D]`, `split_positions: Tensor[B,T]`, `encoder_positions: Tensor[B,T]`.
- Produces: `period_patch_ids(positions, patches, period) -> Tensor[B,T]` and `gather_period_patches(...) -> PatchBatch` with tensors `[B,P,L,*]` plus validity mask.

- [ ] **Step 1: Write failing split tests**

```python
def test_period_patch_ids_use_timestamps_not_observation_counts():
    positions = torch.tensor([[1, 2, 200, 201]])
    ids = period_patch_ids(positions, patches=2, period=365)
    assert ids.tolist() == [[0, 0, 1, 1]]

def test_shift_changes_encoder_positions_not_membership():
    base = torch.tensor([[90, 180, 270, 360]])
    patch = gather_period_patches(torch.randn(1, 4, 8), base, base + 30, 4, 365)
    assert torch.equal(patch.patch_ids, period_patch_ids(base, 4, 365))
    assert torch.equal(patch.encoder_positions[patch.valid], (base + 30).flatten())
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_reimts_mtan.py -k 'period or membership'`
Expected: collection/import failure because the package and functions do not exist.

- [ ] **Step 3: Implement boundary assignment and padded gather**

```python
ids = torch.div(positions.clamp(0, period - 1) * patches, period, rounding_mode="floor")
valid = ids.unsqueeze(-1) == torch.arange(patches, device=ids.device).view(1, patches, 1)
```

Pack observations in original order, pad only storage slots, and carry a boolean mask so every observation appears exactly once per scale.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q tests/test_reimts_mtan.py -k 'period or membership'`
Expected: all selected tests pass.

### Task 2: Minimal Official-Semantics mTAN Encoders

**Files:**
- Create: `models/reimts/mtan_encoder.py`
- Modify: `models/reimts/__init__.py`
- Modify: `tests/test_reimts_mtan.py`

**Interfaces:**
- Consumes: padded patch values `[BP,L,128]`, timestamps `[BP,L]`, observation validity `[BP,L]`.
- Produces: `E_time: Tensor[BP,R,D]` and reference positions `[BP,R]`; encoder instances are stored in a three-entry `ModuleList`.

- [ ] **Step 1: Write failing independence/reference/coordinate tests**

```python
assert len({id(module) for module in encoder.scale_encoders}) == 3
assert encoder.reference_points == (32, 16, 8)
e = encoder.scale_encoders[1](values, shifted_positions, valid)
assert e.shape == (batch_patches, 16, latent_dim)
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_reimts_mtan.py -k mtan`
Expected: missing `ReIMTSTemporalEncoder`/`MTANEncoder`.

- [ ] **Step 3: Adapt the minimal official encoder**

Implement learned time embedding plus multi-time attention from `layers/mTAN/models.py`, concatenate value and observation masks as the official encoder does, split encoder output into mean/log-variance, sample `z0`, and expose `z0` as `E_time` with attribution comments. Do not include mTAN reconstruction decoder, classifier, or losses.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q tests/test_reimts_mtan.py -k mtan`
Expected: independent modules, shared intra-scale calls, reference counts, and shifted coordinates pass.

### Task 3: Recursive Temporal IARF

**Files:**
- Modify: `models/reimts/recursive_temporal.py`
- Modify: `tests/test_reimts_mtan.py`

**Interfaces:**
- Consumes: parent `H_time` split into child patches and child `E_time`, both `[BP,R,D]`, plus `[BP,R]` validity.
- Produces: `alpha = relu(FF(masked_H))` and `G = E - alpha * masked_H`, each `[BP,R,D]`.

- [ ] **Step 1: Write failing shape/mask tests**

```python
alpha, g = fusion(e, h, mask)
assert alpha.shape == g.shape == e.shape == h.shape
assert torch.equal(g[~mask], e[~mask])
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_reimts_mtan.py -k 'iarf or recursive'`
Expected: missing fusion/recursion implementation.

- [ ] **Step 3: Implement SplitOrDuplicate and masked IARF**

Use exact halving/duplication dictated by the 1→2→4 topology and reference counts; assert shape equality before fusion. Zero padded parent representations before `FF` and multiplication.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q tests/test_reimts_mtan.py -k 'iarf or recursive'`
Expected: mask and `E/H/G` shape contracts pass.

### Task 4: Shared PSE/LTAE/Classifier Model

**Files:**
- Create: `models/reimts_classifier.py`
- Modify: `tests/test_reimts_mtan.py`

**Interfaces:**
- Produces: `PseReIMTSMTANLTAE.forward(...)`, `.forward_with_shift(...)`, and `.forward_from_spatial_with_shift(...)`, all returning `[B,C]`.

- [ ] **Step 1: Write failing sharing/logit/checkpoint tests**

```python
patch_features, patch_logits, logits = model.forward_debug(...)
assert patch_features.shape == (B, 4, 128)
assert patch_logits.shape == (B, 4, C)
assert torch.allclose(logits, patch_logits.mean(1))
clone.load_state_dict(model.state_dict(), strict=True)
```

Use a non-linear classifier fixture to prove `mean(classifier(features))` is used instead of `classifier(mean(features))`.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_reimts_mtan.py -k 'decoder or logits or checkpoint'`
Expected: model import failure.

- [ ] **Step 3: Implement the frozen tensor flow**

Run PSE once, recursively produce `[B,4,8,D]`, flatten to `[4B,8,D]`, invoke one LTAE and one decoder, reshape logits to `[B,4,C]`, and average dimension 1. Pass lowest-scale reference indices to LTAE as valid temporal positions.

- [ ] **Step 4: Run GREEN and refactor**

Run: `pytest -q tests/test_reimts_mtan.py`
Expected: all model tests pass.

### Task 5: TimeMatch Shift Capability

**Files:**
- Modify: `timematch.py`
- Create: `tests/test_timematch_reimts.py`
- Modify: `tests/test_timematch_debug.py`

**Interfaces:**
- Produces: `forward_model_with_shift(model, pixels, mask, positions, extra, shift)` and capability-aware spatial shift estimation.

- [ ] **Step 1: Write failing baseline/ReIMTS regression tests**

```python
expected = baseline(pixels, mask, positions + shift, extra)
actual = forward_model_with_shift(baseline, pixels, mask, positions, extra, shift)
assert torch.allclose(actual, expected)
assert pse.call_count == 1
assert reimts.seen_split_positions == positions
assert reimts.seen_encoder_positions == positions + shift
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_timematch_reimts.py tests/test_timematch_debug.py`
Expected: capability helper is missing.

- [ ] **Step 3: Implement capability dispatch everywhere TimeMatch shifts**

For capable models call `forward_with_shift`; for baselines call the unchanged `model(..., positions + shift, ...)`. During estimation compute PSE once per batch, call `forward_from_spatial_with_shift` when present, otherwise retain `decoder(temporal_encoder(spatial_feats, positions + shift))`. Update pseudo labels and teacher/student paths without changing estimator formulas.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q tests/test_timematch_reimts.py tests/test_timematch_debug.py`
Expected: baseline equivalence, topology invariance, coordinate shift, one-PSE, and `[B,C]` pass.

### Task 6: Registration and Closed-Set Smoke Script

**Files:**
- Modify: `train.py`
- Create: `closedset_scripts/smoke_reimts_mtan_closedset.sh`
- Modify: `tests/test_reimts_mtan.py`

**Interfaces:**
- CLI model choice: `psereimtsmtanltae`.
- CLI parameters: `--reimts_levels 3`, `--reimts_scale_factor 2`, `--reimts_period 365`, plus explicit mTAN latent/reference configuration used by model construction.

- [ ] **Step 1: Write failing parser/constructor tests**

Assert the default remains `pseltae` and the new choice/arguments construct the new class without changing existing branches.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_reimts_mtan.py -k registration`
Expected: parser rejects `psereimtsmtanltae`.

- [ ] **Step 3: Register model and add smoke script**

Reuse `denmark/32VNH/2017 → france/30TXT/2017`, source one fold/one epoch, then TimeMatch one epoch with minimal steps/sample size. Keep class mapping and task names controlled by the existing closed-set pipeline.

- [ ] **Step 4: Run GREEN/help/syntax checks**

Run: `pytest -q tests/test_reimts_mtan.py -k registration`
Run: `python train.py --help`
Run: `bash -n closedset_scripts/smoke_reimts_mtan_closedset.sh`
Expected: tests pass, help lists new options, shell syntax exits 0.

### Task 7: Fresh Verification and Handoff

**Files:**
- Modify only files required to fix verification failures attributable to this change.

- [ ] **Step 1: Run focused suites**

Run: `pytest -q tests/test_reimts_mtan.py tests/test_timematch_reimts.py tests/test_timematch_debug.py tests/test_closed_set_protocol.py`
Expected: all pass.

- [ ] **Step 2: Run full repository suite**

Run: `pytest -q`
Expected: all available tests pass; record unrelated/environment failures verbatim.

- [ ] **Step 3: Run data smoke when configured data exists**

Run: `bash closedset_scripts/smoke_reimts_mtan_closedset.sh`
Expected: one supervised fold and one TimeMatch epoch complete with shift estimation. If the dataset is unavailable, record the exact missing path and still report syntax/unit verification separately.

- [ ] **Step 4: Inspect final repository state**

Run: `git branch --show-current`, `git status --short`, `git diff --stat`, `git diff --name-status`, and `git diff`.

- [ ] **Step 5: Deliver the requested 17-part report**

Include initial HEAD, current branch/state, reference files and official `pred_repr_time` tensor, reference counts/rationale, exact tensor flow, split/encoder time distinction, TimeMatch compatibility, baseline regression status, every command/result, smoke output, full diff, and remaining risks. Do not push or merge.
