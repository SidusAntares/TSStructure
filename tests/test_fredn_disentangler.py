import torch

from models.fredn.disentangler import FrequencyDisentangler
from models.fredn.nufft import centered_modes


def test_mask_is_symmetric_for_positive_and_negative_frequencies():
    module = FrequencyDisentangler(num_modes=5, channels=3)

    mask = module.expanded_mask()

    assert mask.shape == (5, 3)
    assert torch.equal(mask[0], mask[4])
    assert torch.equal(mask[1], mask[3])


def test_trend_and_seasonal_coefficients_are_exact_complements():
    torch.manual_seed(5)
    module = FrequencyDisentangler(num_modes=5, channels=2)
    coeffs = torch.randn(3, 5, 2, dtype=torch.complex64)

    trend, seasonal, mask = module(coeffs)

    assert torch.allclose(trend + seasonal, coeffs, rtol=1e-6, atol=1e-6)
    assert torch.equal(mask[0], mask[4])


def test_initial_mask_follows_smooth_trend_spectral_decay():
    module = FrequencyDisentangler(num_modes=7, channels=2)

    nonnegative_mask = torch.sigmoid(module.nonnegative_logits.detach())

    assert torch.all(nonnegative_mask[0] > nonnegative_mask[1])
    assert torch.all(nonnegative_mask[1] > nonnegative_mask[2])
    assert torch.all(nonnegative_mask[2] > nonnegative_mask[3])


def test_branch_sensitive_loss_reaches_mask_parameters():
    torch.manual_seed(7)
    module = FrequencyDisentangler(num_modes=5, channels=2)
    coeffs = torch.randn(2, 5, 2, dtype=torch.complex64)

    trend, _, _ = module(coeffs)
    trend.abs().square().mean().backward()

    assert module.nonnegative_logits.grad is not None
    assert torch.count_nonzero(module.nonnegative_logits.grad).item() > 0


def test_diagnostics_include_frequency_means_and_distinct_branch_energy():
    module = FrequencyDisentangler(num_modes=3, channels=1)
    coeffs = torch.ones(1, 3, 1, dtype=torch.complex64)
    trend, seasonal, mask = module(coeffs)

    diagnostics = module.diagnostics(coeffs, trend, seasonal, mask)

    assert diagnostics["frequency_mask_mean"].shape == (3,)
    assert diagnostics["mask_mean"] == torch.mean(mask).item()
    assert diagnostics["trend_energy_ratio"] != diagnostics["seasonal_energy_ratio"]


def test_mask_diagnostics_distinguish_collapsed_and_polarized_half_means():
    collapsed = torch.full((3, 4), 0.5)
    polarized = torch.tensor([[0.1, 0.1, 0.9, 0.9]]).expand(3, -1)

    collapsed_diagnostics = FrequencyDisentangler.mask_diagnostics(collapsed)
    polarized_diagnostics = FrequencyDisentangler.mask_diagnostics(polarized)

    assert collapsed_diagnostics["mask_mean"] == torch.tensor(0.5)
    assert collapsed_diagnostics["mask_std"] == torch.tensor(0.0)
    assert collapsed_diagnostics["mask_near_half"] == torch.tensor(1.0)
    assert polarized_diagnostics["mask_mean"] == torch.tensor(0.5)
    assert torch.allclose(polarized_diagnostics["mask_std"], torch.tensor(0.4))
    assert polarized_diagnostics["mask_near_half"] == torch.tensor(0.0)
    assert polarized_diagnostics["mask_low025"] == torch.tensor(0.5)
    assert polarized_diagnostics["mask_high075"] == torch.tensor(0.5)


def test_mask_diagnostics_capture_frequency_and_feature_specialization():
    frequency_ordered = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.5, 0.5, 0.5],
            [0.8, 0.8, 0.8],
        ]
    )
    feature_ordered = torch.tensor(
        [
            [0.1, 0.5, 0.9],
            [0.5, 0.5, 0.5],
            [0.9, 0.5, 0.1],
        ]
    )

    frequency_diagnostics = FrequencyDisentangler.mask_diagnostics(
        frequency_ordered
    )
    feature_diagnostics = FrequencyDisentangler.mask_diagnostics(feature_ordered)

    assert torch.allclose(
        frequency_diagnostics["frequency_mask_mean"],
        torch.tensor([0.2, 0.5, 0.8]),
    )
    assert frequency_diagnostics["feature_freq_std_mean"] > 0
    assert torch.allclose(
        frequency_diagnostics["feature_freq_range_mean"],
        torch.tensor(0.6),
    )
    assert torch.allclose(
        feature_diagnostics["frequency_mask_mean"],
        torch.full((3,), 0.5),
    )
    assert feature_diagnostics["feature_freq_std_mean"] > 0
    assert feature_diagnostics["feature_freq_range_max"] > 0


def test_mask_prior_alignment_detects_decay_and_reversal():
    module = FrequencyDisentangler(num_modes=7, channels=4)
    initialized = module.mask_diagnostics(module.expanded_mask())
    magnitudes = centered_modes(7).abs().float().unsqueeze(1)
    reversed_mask = (0.1 + 0.2 * magnitudes).expand(-1, 4)
    reversed_diagnostics = module.mask_diagnostics(reversed_mask)

    assert initialized["abs_freq_corr"] < 0
    assert reversed_diagnostics["abs_freq_corr"] > 0


def test_energy_routing_matches_analytic_frequency_energies():
    module = FrequencyDisentangler(num_modes=3, channels=1)
    coeffs = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.complex64)
    mask = torch.tensor([[0.5], [0.25], [0.75]])
    trend = coeffs * mask.unsqueeze(0)
    seasonal = coeffs * (1.0 - mask).unsqueeze(0)

    diagnostics = module.diagnostics(coeffs, trend, seasonal, mask)

    input_energy = torch.tensor([1.0, 4.0, 9.0])
    trend_energy = input_energy * mask.squeeze(1).square()
    seasonal_energy = input_energy * (1.0 - mask.squeeze(1)).square()
    branch_total = trend_energy.sum() + seasonal_energy.sum()
    assert torch.allclose(
        diagnostics["input_energy_by_frequency"],
        input_energy / input_energy.sum(),
    )
    assert torch.allclose(
        diagnostics["trend_energy_by_frequency"],
        trend_energy / trend_energy.sum(),
    )
    assert torch.allclose(
        diagnostics["seasonal_energy_by_frequency"],
        seasonal_energy / seasonal_energy.sum(),
    )
    assert torch.allclose(
        diagnostics["branch_trend_energy_ratio"],
        trend_energy.sum() / branch_total,
    )
    assert torch.allclose(
        diagnostics["branch_seasonal_energy_ratio"],
        seasonal_energy.sum() / branch_total,
    )


def test_symmetric_mask_has_symmetric_per_frequency_statistics():
    module = FrequencyDisentangler(num_modes=5, channels=6)
    diagnostics = module.mask_diagnostics(module.expanded_mask())

    for key in (
        "frequency_mask_mean",
        "frequency_mask_std",
        "frequency_mask_p05",
        "frequency_mask_p95",
        "frequency_mask_near_half",
        "frequency_mask_low025",
        "frequency_mask_high075",
    ):
        values = diagnostics[key]
        assert torch.equal(values, values.flip(0)), key
