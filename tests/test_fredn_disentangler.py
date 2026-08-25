import torch

from models.fredn.disentangler import FrequencyDisentangler


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
