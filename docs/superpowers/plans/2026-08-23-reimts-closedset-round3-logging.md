# ReIMTS Closed-set Round 3 Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable structured logs and a failure-isolated four-GPU launcher for the 12 directed ReIMTS closed-set tasks.

**Architecture:** Keep training mathematics unchanged and collect scalar diagnostics around existing calls. Central formatting helpers render stable multi-line blocks; the launcher assigns one serial source worker to each GPU and redirects each Python run into its own task log.

**Tech Stack:** Bash, Python, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-reimts-closedset-round3-logging-design.md`

## Global Constraints

- Do not modify ReIMTS, mTAN, LTAE, loss, aggregation, TimeMatch, EMA, shift, augmentation, or closed-set mathematics.
- Do not pass `--data_root` from the formal launcher.
- Do not launch CUDA training locally.
- Do not commit, push, or merge.
- Every production behavior follows RED → verified RED → GREEN.

---

### Task 1: Generic log formatting

**Files:**
- Modify: `utils/train_utils.py`
- Test: `tests/test_structured_logging.py`

**Interfaces:**
- Produces: `format_duration(seconds) -> str`, `format_log_block(title, lines, border='=') -> str`, `log_block(title, lines, border='=') -> None`.

- [ ] Write tests for hour/minute/second duration formatting and multi-line blocks.
- [ ] Run those tests and verify missing-symbol failures.
- [ ] Implement pure formatting plus a thin printing helper.
- [ ] Run tests and verify GREEN.

### Task 2: Diagnostic formatters

**Files:**
- Modify: `models/reimts_classifier.py`
- Modify: `timematch.py`
- Test: `tests/test_structured_logging.py`

**Interfaces:**
- Produces patch occupancy summaries, pseudo histograms, TimeMatch epoch text, and shift ranking text from already-computed tensors/scalars.

- [ ] Write tests asserting required fields and that histogram/top-five entries occupy separate lines.
- [ ] Run tests and verify failures identify missing formatters.
- [ ] Implement pure formatter/accumulator helpers without changing model outputs.
- [ ] Run tests and verify GREEN.

### Task 3: Source and validation epoch logging

**Files:**
- Modify: `train.py`
- Modify: `evaluation.py`
- Test: `tests/test_structured_logging.py`

**Interfaces:**
- Consumes: log/duration and patch summary helpers.
- Produces: one source summary per epoch and one validation summary per validation call.

- [ ] Add behavior tests with synthetic metrics and disabled progress bars.
- [ ] Verify RED.
- [ ] Aggregate patch validity across existing source batches, time epochs/validation, and render blocks.
- [ ] Verify GREEN and existing evaluation regressions.

### Task 4: TimeMatch and shift logging

**Files:**
- Modify: `timematch.py`
- Test: `tests/test_structured_logging.py`
- Test: `tests/test_timematch_reimts.py`

**Interfaces:**
- Produces separate source/target/total loss averages, pseudo confidence/acceptance/histogram diagnostics, unchanged score selection with ranked diagnostics, and epoch timing fields.

- [ ] Add tests for zero accepted pseudo-labels, loss separation, score direction, top-five formatting, and tqdm-off output.
- [ ] Verify RED.
- [ ] Instrument existing loop and estimator without modifying selected shifts or losses.
- [ ] Verify GREEN and TimeMatch regressions.

### Task 5: Run completion logging

**Files:**
- Modify: `train.py`
- Test: `tests/test_structured_logging.py`

**Interfaces:**
- Produces `[RUN COMPLETE]` with experiment, formatted total runtime, best validation macro-F1, and final test macro-F1.

- [ ] Add a pure formatter test and verify RED.
- [ ] Thread existing returned metrics into the final block without retraining or reevaluation.
- [ ] Verify GREEN.

### Task 6: Four-GPU launcher

**Files:**
- Create: `closedset_scripts/run_reimts_mtan_closedset_12tasks_4gpu.sh`
- Create: `tests/test_reimts_12task_launcher.py`

**Interfaces:**
- Consumes environment variables `EXPERIMENT_NAME`, `SEED`, and `PYTHON_BIN`.
- Produces four source runs, twelve serial DA runs, 17 prescribed logs, checkpoint checks, per-task shell timing, and aggregate failure status.

- [ ] Write static tests for domains, mapping, counts, flags, paths, weights, and failure handling.
- [ ] Verify RED because launcher is absent.
- [ ] Implement the Bash launcher without `--data_root`.
- [ ] Verify GREEN and `bash -n`.

### Task 7: Fresh verification

**Files:**
- Verify all modified files.

- [ ] Run `python -m pytest -q`.
- [ ] Run `git diff --check` and `git diff --cached --check`.
- [ ] Run `bash -n closedset_scripts/run_reimts_mtan_closedset_12tasks_4gpu.sh`.
- [ ] Run `python train.py --help` when imports are available.
- [ ] Run synthetic/mock logging smoke and confirm no tqdm control sequences.
- [ ] Record that no local CUDA or real-data training was started.

