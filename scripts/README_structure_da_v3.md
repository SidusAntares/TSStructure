# TSStructure V3 offline experiment workflow

These scripts run entirely from local code and local TimeMatch data. They do
not manage source control, contact a network service, install packages, or
download data or weights. Set paths for the current server rather than editing
the scripts:

```bash
export DATA_ROOT=/path/to/local/timematch_data
export OUTPUT_ROOT=/path/to/local/structure_da_v3_outputs
export CONDA_ENV=timematch
```

If Conda is already active, leave `PYTHON_BIN=python`. Otherwise activate the
environment first, or set `PYTHON_BIN` to that environment's Python executable.

Run the stages in this order:

1. Check the offline environment:

   ```bash
   bash scripts/check_server_env.sh
   ```

2. Run the short AT1 to DK1 smoke. It checks execution, validation,
   checkpointing, finite losses and optimizer/state updates; it does not
   evaluate DA performance.

   ```bash
   bash scripts/smoke_structure_da_v3.sh
   python scripts/check_structure_da_smoke.py \
     "$OUTPUT_ROOT/smoke/AT1_DK1/seed_1"
   ```

3. Run and analyze the fixed AT1 to DK1 seed-1 diagnostic pilot. This pilot
   uses formal model/loss settings but only 25 epochs, so it is not a final
   experiment result.

   ```bash
   bash scripts/run_structure_da_diagnostic_at1_dk1.sh
   python scripts/analyze_structure_da_diagnostic.py \
     "$OUTPUT_ROOT/diagnostic_pilot/AT1_DK1/seed_1"
   ```

4. Only after the diagnostic reports `PASS_FOR_PILOT4`, run the four-GPU pilot
   and aggregate it:

   ```bash
   GPU0=0 GPU1=1 GPU2=2 GPU3=3 \
     bash scripts/run_structure_da_pilot4_4gpu.sh
   python scripts/analyze_structure_da_pilot4.py "$OUTPUT_ROOT/pilot4"
   ```

5. Build a full experiment launcher only after the pilot reports
   `READY_FOR_FULL_EXPERIMENT`.

Existing directories are never silently overwritten. Set `OVERWRITE=1` only
when deliberately replacing a run. X2 (classic numerical SRVF comparison) and
X3 (direct S-shape versus S-reference residual ablation) remain later work;
class-specific reference training also depends on X1. Validation `occlusion_*`
metrics are feature-occlusion diagnostics, not formal retrained ablations.
