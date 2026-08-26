"""TensorBoard and text diagnostics for FreDN models."""

import torch

from models.fredn.nufft import centered_modes


_SCALAR_KEYS = {
    "mask_mean": "fredn/mask_mean",
    "mask_std": "fredn/mask_std",
    "mask_lt_0.1": "fredn/mask_lt_0.1",
    "mask_gt_0.9": "fredn/mask_gt_0.9",
    "trend_energy_ratio": "fredn/trend_energy_ratio",
    "seasonal_energy_ratio": "fredn/seasonal_energy_ratio",
    "reconstruction_error": "fredn/reconstruction_error",
    "additivity_error": "fredn/additivity_error",
    "solver_iterations": "fredn/nufft_solver_iterations",
    "solver_residual": "fredn/nufft_solver_residual",
    "shared_points_rate": "fredn/nufft_shared_points_rate",
    "analysis_time": "fredn/nufft_analysis_time",
    "synthesis_time": "fredn/nufft_synthesis_time",
    "imaginary_residual": "fredn/imaginary_residual",
    "trend_logit_rms": "fredn/trend_logit_rms",
    "seasonal_logit_rms": "fredn/seasonal_logit_rms",
    "trend_feature_rms": "fredn/trend_feature_rms",
    "seasonal_feature_rms": "fredn/seasonal_feature_rms",
    "mask_min": "fredn/mask_min",
    "mask_max": "fredn/mask_max",
    "mask_p05": "fredn/mask_p05",
    "mask_p25": "fredn/mask_p25",
    "mask_p50": "fredn/mask_p50",
    "mask_p75": "fredn/mask_p75",
    "mask_p95": "fredn/mask_p95",
    "mask_near_half": "fredn/mask_near_half",
    "mask_low025": "fredn/mask_low025",
    "mask_high075": "fredn/mask_high075",
    "feature_freq_std_mean": "fredn/feature_freq_std_mean",
    "feature_freq_std_median": "fredn/feature_freq_std_median",
    "feature_freq_std_p90": "fredn/feature_freq_std_p90",
    "feature_freq_std_max": "fredn/feature_freq_std_max",
    "feature_freq_range_mean": "fredn/feature_freq_range_mean",
    "feature_freq_range_median": "fredn/feature_freq_range_median",
    "feature_freq_range_p90": "fredn/feature_freq_range_p90",
    "feature_freq_range_max": "fredn/feature_freq_range_max",
    "abs_freq_corr": "fredn/abs_freq_corr",
    "branch_trend_energy_ratio": "fredn/branch_trend_energy_ratio",
    "branch_seasonal_energy_ratio": "fredn/branch_seasonal_energy_ratio",
    "fourier_condition_mean": "fredn/fourier_condition_mean",
    "fourier_condition_median": "fredn/fourier_condition_median",
    "fourier_condition_p95": "fredn/fourier_condition_p95",
    "fourier_condition_max": "fredn/fourier_condition_max",
}


def _to_scalar(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
    return float(value)


def _format_scalar(value):
    return f"{_to_scalar(value):.6f}"


def _format_vector(values):
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    return ",".join(f"{float(value):.6f}" for value in values)


def _has_all(diagnostics, keys):
    return all(key in diagnostics for key in keys)


def log_fredn_diagnostics(
    model,
    writer,
    step,
    frequency_log_interval=100,
):
    diagnostics = getattr(model, "last_diagnostics", None)
    if not diagnostics or writer is None:
        return
    for source_name, tensorboard_name in _SCALAR_KEYS.items():
        if source_name in diagnostics:
            value = diagnostics[source_name]
            if torch.is_tensor(value):
                value = value.detach().cpu()
            writer.add_scalar(tensorboard_name, value, step)

    frequency_means = diagnostics.get("frequency_mask_mean")
    if frequency_means is None or step % frequency_log_interval != 0:
        return
    print(
        f"FREDN_MASK_BY_FREQUENCY|step={step}|"
        f"means={_format_vector(frequency_means)}"
    )

    distribution_keys = (
        "frequency_mask_std",
        "frequency_mask_p05",
        "frequency_mask_p25",
        "frequency_mask_p50",
        "frequency_mask_p75",
        "frequency_mask_p95",
        "frequency_mask_near_half",
        "frequency_mask_low025",
        "frequency_mask_high075",
    )
    if _has_all(diagnostics, distribution_keys):
        print(
            f"FREDN_MASK_DISTRIBUTION|step={step}|"
            f"std={_format_vector(diagnostics['frequency_mask_std'])}|"
            f"p05={_format_vector(diagnostics['frequency_mask_p05'])}|"
            f"p25={_format_vector(diagnostics['frequency_mask_p25'])}|"
            f"p50={_format_vector(diagnostics['frequency_mask_p50'])}|"
            f"p75={_format_vector(diagnostics['frequency_mask_p75'])}|"
            f"p95={_format_vector(diagnostics['frequency_mask_p95'])}|"
            f"near_half={_format_vector(diagnostics['frequency_mask_near_half'])}|"
            f"low025={_format_vector(diagnostics['frequency_mask_low025'])}|"
            f"high075={_format_vector(diagnostics['frequency_mask_high075'])}"
        )

    global_keys = (
        "mask_mean",
        "mask_std",
        "mask_min",
        "mask_max",
        "mask_p05",
        "mask_p25",
        "mask_p50",
        "mask_p75",
        "mask_p95",
        "mask_near_half",
        "mask_low025",
        "mask_high075",
    )
    if _has_all(diagnostics, global_keys):
        fields = "|".join(
            f"{key.removeprefix('mask_')}={_format_scalar(diagnostics[key])}"
            for key in global_keys
        )
        print(f"FREDN_MASK_GLOBAL|step={step}|{fields}")

    specialization_keys = (
        "feature_freq_std_mean",
        "feature_freq_std_median",
        "feature_freq_std_p90",
        "feature_freq_std_max",
        "feature_freq_range_mean",
        "feature_freq_range_median",
        "feature_freq_range_p90",
        "feature_freq_range_max",
    )
    if _has_all(diagnostics, specialization_keys):
        fields = "|".join(
            f"{key.removeprefix('feature_')}={_format_scalar(diagnostics[key])}"
            for key in specialization_keys
        )
        print(f"FREDN_MASK_FEATURE_SPECIALIZATION|step={step}|{fields}")

    if "abs_freq_corr" in diagnostics:
        print(
            f"FREDN_MASK_PRIOR_ALIGNMENT|step={step}|"
            f"abs_freq_corr={_format_scalar(diagnostics['abs_freq_corr'])}"
        )

    energy_keys = (
        "input_energy_by_frequency",
        "trend_energy_by_frequency",
        "seasonal_energy_by_frequency",
    )
    if _has_all(diagnostics, energy_keys):
        print(
            f"FREDN_ENERGY_BY_FREQUENCY|step={step}|"
            f"input={_format_vector(diagnostics['input_energy_by_frequency'])}|"
            f"trend={_format_vector(diagnostics['trend_energy_by_frequency'])}|"
            f"seasonal={_format_vector(diagnostics['seasonal_energy_by_frequency'])}"
        )

    if _has_all(
        diagnostics,
        ("branch_trend_energy_ratio", "branch_seasonal_energy_ratio"),
    ):
        print(
            f"FREDN_BRANCH_ENERGY|step={step}|"
            f"trend_ratio={_format_scalar(diagnostics['branch_trend_energy_ratio'])}|"
            f"seasonal_ratio={_format_scalar(diagnostics['branch_seasonal_energy_ratio'])}"
        )

    branch_output_keys = (
        "trend_feature_rms",
        "seasonal_feature_rms",
        "trend_logit_rms",
        "seasonal_logit_rms",
    )
    if _has_all(diagnostics, branch_output_keys):
        fields = "|".join(
            f"{key}={_format_scalar(diagnostics[key])}"
            for key in branch_output_keys
        )
        print(f"FREDN_BRANCH_OUTPUT|step={step}|{fields}")

    if _has_all(diagnostics, ("reconstruction_error", "additivity_error")):
        print(
            f"FREDN_FOURIER_DIAGNOSTICS|step={step}|"
            f"reconstruction_error={_format_scalar(diagnostics['reconstruction_error'])}|"
            f"additivity_error={_format_scalar(diagnostics['additivity_error'])}"
        )

    condition_keys = (
        "fourier_condition_mean",
        "fourier_condition_median",
        "fourier_condition_p95",
        "fourier_condition_max",
    )
    if _has_all(diagnostics, condition_keys):
        print(
            f"FREDN_FOURIER_CONDITION|step={step}|"
            f"cond_mean={_format_scalar(diagnostics['fourier_condition_mean'])}|"
            f"cond_median={_format_scalar(diagnostics['fourier_condition_median'])}|"
            f"cond_p95={_format_scalar(diagnostics['fourier_condition_p95'])}|"
            f"cond_max={_format_scalar(diagnostics['fourier_condition_max'])}"
        )


def log_fredn_checkpoint_mask(model, stage, output_path=None):
    """Log the expanded mask from the model state that will be evaluated."""
    unwrapped_model = getattr(model, "module", model)
    disentangler = getattr(unwrapped_model, "frequency_disentangler", None)
    if disentangler is None:
        return False

    with torch.no_grad():
        mask = disentangler.expanded_mask().detach()
        diagnostics = disentangler.mask_diagnostics(mask)
        frequencies = getattr(unwrapped_model, "fredn_modes", None)
        if frequencies is None:
            frequencies = centered_modes(
                mask.shape[0],
                device=mask.device,
                dtype=mask.dtype,
            )
        else:
            frequencies = frequencies.detach()

    print(
        f"FREDN_CHECKPOINT_MASK|stage={stage}|"
        f"frequency_mean={_format_vector(diagnostics['frequency_mask_mean'])}|"
        f"frequency_std={_format_vector(diagnostics['frequency_mask_std'])}|"
        f"global_std={_format_scalar(diagnostics['mask_std'])}|"
        f"near_half={_format_scalar(diagnostics['mask_near_half'])}|"
        f"low025={_format_scalar(diagnostics['mask_low025'])}|"
        f"high075={_format_scalar(diagnostics['mask_high075'])}|"
        f"feature_freq_std_mean={_format_scalar(diagnostics['feature_freq_std_mean'])}|"
        f"abs_freq_corr={_format_scalar(diagnostics['abs_freq_corr'])}"
    )
    if output_path is not None:
        torch.save(
            {
                "mask": mask.cpu(),
                "frequencies": frequencies.cpu(),
            },
            output_path,
        )
    return True
