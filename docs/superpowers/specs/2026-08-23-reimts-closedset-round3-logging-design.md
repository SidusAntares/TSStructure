# ReIMTS Closed-set Round 3 Logging and Launcher Design

## Scope

Freeze all ReIMTS, mTAN, LTAE, loss, aggregation, TimeMatch, EMA, pseudo-label,
shift-estimator, augmentation, and closed-set mathematics. This round changes
only experiment orchestration, human-readable logging, diagnostic aggregation,
and runtime accounting.

Local work ends after unit/static/synthetic validation. Real-data smoke and the
16 formal Python runs execute later on the CUDA server after human review.

## Components

1. `utils/train_utils.py` owns generic block and duration formatting.
2. `evaluation.py` emits compact multi-line validation blocks and returns the
   same best-F1 value used by existing callers.
3. `train.py` aggregates source patch occupancy per epoch, prints one source
   epoch block, and prints one run-complete block after final testing.
4. `timematch.py` aggregates source/target/total loss, confidence, accepted
   pseudo-labels, class histograms, patch occupancy, shift timing, training
   timing, validation timing, and elapsed time. Shift selection math remains
   byte-for-byte equivalent in meaning; only ranked diagnostics are added.
5. `closedset_scripts/run_reimts_mtan_closedset_12tasks_4gpu.sh` launches four
   independent source workers. Each worker trains one source once and then
   executes its three targets serially on its fixed GPU.

## Launcher Contract

- Domains: DK1, FR1, FR2, AT1; GPUs: 0, 1, 2, 3 respectively.
- Seed defaults to 1 and `PYTHON_BIN` defaults to `python`.
- `EXPERIMENT_NAME` defaults to `reimts_mtan_closedset_12tasks_seed1`.
- Do not pass `--data_root`.
- Logs are exactly `logs/<experiment>/<task>.log`; outputs are
  `outputs/<experiment>/<task>`; TensorBoard roots are
  `runs/<experiment>/<task>`.
- A source failure or missing `fold_0/model.pt` stops only that worker. A DA
  failure is recorded and the worker continues. Launcher exits nonzero and
  summarizes failed tasks if any worker reports a failure.
- Task logs contain shell `[RUN_START]` and `[RUN_END]` records. Launcher log
  contains scheduling only and never duplicates model output.

## Logging Contract

- All formal commands explicitly pass `--progress_bar off`.
- Source logs contain one `[SOURCE] Epoch n/N` block per epoch with loss, LR,
  ReIMTS patch occupancy, epoch seconds, and elapsed duration.
- Validation logs contain loss, accuracy, macro-F1, kappa, best before/after,
  checkpoint status, and validation seconds; classification reports remain
  final-test only.
- TimeMatch logs contain one epoch block with shift metadata, pseudo-label
  acceptance/confidence/debug F1/histogram, three loss averages, patch
  occupancy, and timing.
- Shift estimation logs contain estimator, range, candidate/sample counts,
  selected and second-best score, gap, top five, debug-oracle accuracy, and
  runtime. True target labels are diagnostic only and never affect selection.
- Run completion contains experiment, total runtime, best validation macro-F1,
  and test macro-F1.

## Testing

Use TDD for formatters, source/validation/TimeMatch blocks, shift rankings, and
launcher structure. Tests assert stable fields and multi-line structure rather
than entire rendered strings. No CUDA or real dataset is launched locally.

