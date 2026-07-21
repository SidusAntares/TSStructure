# TimeMatch CUDA Debug Notes

## What the four `run_timematch*.log` files actually contain

- The logs contain 12 attempted TimeMatch transfers across the four domains:
  - `FR1 = france/30TXT/2017`
  - `FR2 = france/31TCJ/2017`
  - `DK1 = denmark/32VNH/2017`
  - `AT1 = austria/33UVP/2017`
- Only 8 transfers finish and print a final test `macro_f1`.
- 4 transfers abort during `TimeMatch Epoch 1/20` with `CUDA error: device-side assert triggered`:
  - `FR1 -> DK1`
  - `FR2 -> FR1`
  - `FR2 -> DK1`
  - `AT1 -> FR2`

## Evidence from the logs

- `run_timematch_30TXT_gpu1.log` fails for `timematch_30TXT_to_32VNH` (`FR1 -> DK1`).
- `run_timematch_31TCJ_gpu2.log` fails for:
  - `timematch_31TCJ_to_32VNH` (`FR2 -> DK1`)
  - `timematch_31TCJ_to_30TXT` (`FR2 -> FR1`)
- `run_timematch_33UVP_gpu3.log` fails for `timematch_33UVP_to_31TCJ` (`AT1 -> FR2`).

The earliest CUDA assertion in all three failure patterns is:

```text
../aten/src/ATen/native/cuda/Indexing.cu:1289:
indexSelectLargeIndex ... Assertion `srcIndex < srcSelectDimSize` failed.
```

The Python stack then reports the failure while executing:

```text
train.py -> train_timematch(...) -> timematch.py line 114
student.forward(...) -> models/stclassifier.py -> models/ltae.py
```

## Most likely root cause

The failure is most likely caused by **temporal position indices going out of range in the LTAE positional embedding**, not by the classifier loss itself.

Why this is the leading hypothesis:

1. `train_timematch()` crashes on the source branch before loss computation:
   - `timematch.py:114`
   - `logits_source = student.forward(pixels_s, mask_s, position_s + source_to_target_shift, extra_s)`
2. `LTAE.forward()` applies an embedding lookup:
   - `models/ltae.py:87`
   - `enc_output = x + self.positional_enc(positions + self.max_temporal_shift)`
3. The embedding table is finite:
   - `models/ltae.py:64`
   - `max_position + 2 * max_temporal_shift`
   - With current defaults this becomes `365 + 2 * 100 = 565`, which matches the log line `Embedding(565, 256)`.
4. The CUDA assertion is an indexing assertion (`srcIndex < srcSelectDimSize`), which is exactly what you would expect from an out-of-range embedding index.

## Why `AT1 -> FR2` is especially suspicious

`AT1 -> FR2` is one of the failing directions, but it is not unique. The same failure pattern also appears in three other directions.

What these failed directions have in common is that TimeMatch adds an estimated temporal shift to the date positions during adaptation:

- target-to-source shift estimation in `timematch.py`
- source forward pass with `position_s + source_to_target_shift`
- target forward pass with `position_t_weak + target_to_source_shift`

If any sample already has a large day index, adding a shift can push it beyond the embedding limit.

## Why this can happen in this codebase

`PixelSetData.days_after()` computes positions as absolute day differences from `metadata["start_date"]`:

```python
date_positions = [interval_days(d, start_date) for d in dates]
```

But the temporal encoder assumes a fixed maximum position of 365 days plus a shift buffer. If the dataset dates span more than that assumed range, or if the shift adds enough offset, the positional embedding lookup can overflow.

## How to debug it quickly

1. Re-run the failing task with synchronous CUDA:

```bash
CUDA_LAUNCH_BLOCKING=1 python train.py -e timematch_33UVP_to_31TCJ --source austria/33UVP/2017 --target france/31TCJ/2017 timematch --weights outputs/pseltae_33UVP
```

2. Print min/max positions right before the failing forward call in `timematch.py`:

```python
print("source shift", source_to_target_shift)
print("source pos min/max", position_s.min().item(), position_s.max().item())
print("shifted source pos min/max", (position_s + source_to_target_shift).min().item(), (position_s + source_to_target_shift).max().item())
```

3. Add the same checks in `models/ltae.py` before the embedding lookup:

```python
idx = positions + self.max_temporal_shift
print("embedding idx min/max", idx.min().item(), idx.max().item(), "table size", self.positional_enc.num_embeddings)
```

4. Also print the estimated shift from `estimate_temporal_shift()` each epoch to verify whether the first-epoch shift is already too large.

## Practical fixes to try

### Fix A: enlarge the positional embedding range

This is the most direct fix if the issue is indeed position overflow.

- Increase `max_position` and/or `max_temporal_shift` in `models/ltae.py` and the model constructors.
- For example, make the embedding table comfortably larger than the largest observed `date_position + abs(shift)`.

### Fix B: clamp shifted positions before the embedding lookup

Safer for quick experiments:

```python
idx = torch.clamp(positions + self.max_temporal_shift, 0, self.positional_enc.num_embeddings - 1)
enc_output = x + self.positional_enc(idx)
```

This prevents crashes, though it changes the intended temporal encoding near the boundaries.

### Fix C: normalize date positions to a safer range

If `date_positions` are larger than expected because of dataset-specific start dates, normalize or remap them so they fit the encoder's assumed range before adaptation shifts are added.

## Bottom line

- The current logs do **not** show 11 successful TimeMatch macro-F1 results; they show 8 successes and 4 CUDA-aborted transfers.
- `AT1 -> FR2` does fail, but it fails with the same low-level indexing pattern as `FR1 -> DK1`, `FR2 -> FR1`, and `FR2 -> DK1`.
- The strongest code-level hypothesis is **out-of-range positional embedding indices after temporal shifting in `LTAE`**.
