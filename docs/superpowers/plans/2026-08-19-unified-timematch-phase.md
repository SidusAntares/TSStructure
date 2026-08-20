# Unified TimeMatch Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one TimeMatch CLI supporting exact alpha-zero degeneration and optional AM-selected nonlinear residual Domain Phase, then remove unreachable historical StructureDA code.

**Architecture:** The canonical semantic model remains PSE -> LTAE -> classifier. A frozen geometry copy is constructed only when the alpha bank contains a nonzero value. Shared runtime helpers replace imports from historical diagnostic scripts.

**Tech Stack:** Python, PyTorch, pytest, Bash, Git.

---

### Task 1: Freeze the unified CLI contract

**Files:**
- Modify: `tests/structure_da/test_timematch_nonlinear_phase_training_contract.py`
- Modify: `tests/structure_da/test_timematch_nonlinear_phase.py`

- [ ] Add tests requiring a single training entry, explicit alpha candidates, zero-mode geometry bypass, exact scalar timestamps, and four unique GPU/task assignments.
- [ ] Run the two tests and verify they fail because the current CLI still exposes three modes and separate Original entrypoints.

### Task 2: Consolidate runtime helpers

**Files:**
- Modify: `scripts/timematch_runtime.py`
- Modify: `methods/structure_da/original_timematch.py`
- Create: `methods/structure_da/timematch_phase.py`

- [ ] Move split reconstruction, selected loaders, registration configuration and device-loader behavior into focused current modules.
- [ ] Add tests for label-free target scans and geometry construction only when a nonzero alpha exists.
- [ ] Run focused tests and verify they pass.

### Task 3: Create the single adaptation trainer

**Files:**
- Create: `scripts/train_timematch_phase.py`
- Keep and simplify: `scripts/train_original_timematch_source.py`
- Delete: `scripts/train_stage2_original_timematch.py`
- Delete: `scripts/train_stage2_timematch_nonlinear_phase_search.py`

- [ ] Keep one source-supervised Stage 1 producer and one Stage 2 implementation behind `--alpha-candidates`; the launcher composes the complete run.
- [ ] Preserve original TimeMatch source training and adaptation control flow.
- [ ] Verify `{0}` bypasses registration and the full bank performs delta-then-alpha AM selection.

### Task 4: Replace launchers

**Files:**
- Create: `scripts/run_timematch_phase_4tasks_4gpu_seed1.sh`
- Delete: obsolete Original/P0/P1 launchers and comparator.

- [ ] Map GPU 0..3 to AT1->DK1, DK1->FR2, FR1->AT1 and FR2->FR1.
- [ ] Default to alpha zero and permit the full bank through an environment setting.
- [ ] Verify independent flat logs, outputs, exit codes and shell syntax.

### Task 5: Remove unreachable historical implementation

**Files:**
- Modify: `methods/structure_da/__init__.py`
- Delete: unreachable old StructureDA methods, scripts and their dedicated tests.

- [ ] Search imports from the canonical trainer and retained tests.
- [ ] Keep only the decomposition/functional/SRVF/registration modules needed by nonlinear phase plus the canonical TimeMatch modules.
- [ ] Delete 13A/13B/14/15, oracle, V3, old model/trainer/objective/diagnostic chains and dangling tests.

### Task 6: Final verification

**Files:**
- Test: retained `tests/structure_da/*.py`

- [ ] Run focused alpha/contract tests.
- [ ] Run all retained tests.
- [ ] Run `python -m compileall methods scripts tests`.
- [ ] Run `bash -n scripts/run_timematch_phase_4tasks_4gpu_seed1.sh`.
- [ ] Run legacy-reference search and require zero dangling imports.
- [ ] Run `git diff --check` and report status without starting real training.
