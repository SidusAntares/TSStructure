"""TensorBoard and text diagnostics for FreDN models."""

import torch


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
}


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
            writer.add_scalar(tensorboard_name, diagnostics[source_name], step)

    frequency_means = diagnostics.get("frequency_mask_mean")
    if frequency_means is None or step % frequency_log_interval != 0:
        return
    if torch.is_tensor(frequency_means):
        values = frequency_means.detach().cpu().tolist()
    else:
        values = list(frequency_means)
    formatted = ",".join(f"{float(value):.6f}" for value in values)
    print(f"FREDN_MASK_BY_FREQUENCY|step={step}|means={formatted}")
