import math
import sys
import types

import pytest
import torch

from models.fredn.nufft import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
    DenseFourierBackend,
    IrregularFourierAnalyzer,
    IrregularFourierSynthesizer,
    centered_modes,
    positions_to_periodic_points,
)
from models.fredn.disentangler import FrequencyDisentangler


def _known_real_series(dtype=torch.complex128):
    negative_two = torch.tensor([[0.2 - 0.1j, -0.1 + 0.05j]], dtype=dtype)
    negative_one = torch.tensor([[0.4 + 0.2j, 0.3 - 0.15j]], dtype=dtype)
    zero = torch.tensor([[1.0 + 0.0j, -0.5 + 0.0j]], dtype=dtype)
    return torch.stack(
        [negative_two, negative_one, zero, negative_one.conj(), negative_two.conj()],
        dim=1,
    )


def test_centered_modes_reject_even_count_and_return_paired_indices():
    with pytest.raises(ValueError, match="positive odd"):
        centered_modes(4)

    modes = centered_modes(5)

    assert modes.tolist() == [-2, -1, 0, 1, 2]


def test_positions_map_to_shared_physical_period_without_sample_normalization():
    positions = torch.tensor([[0.0, 91.25, 365.0], [30.0, 121.25, 395.0]])

    points = positions_to_periodic_points(positions, period_days=365.0)

    expected = torch.tensor(
        [
            [0.0, math.pi / 2.0, 0.0],
            [2.0 * math.pi * 30.0 / 365.0, 2.0 * math.pi * 121.25 / 365.0, 2.0 * math.pi * 30.0 / 365.0],
        ]
    )
    assert torch.allclose(points, expected, atol=1e-6)


def test_dense_type1_with_opposite_sign_is_adjoint_of_type2():
    torch.manual_seed(4)
    backend = DenseFourierBackend()
    points = torch.tensor([-2.1, -0.7, 0.2, 1.4], dtype=torch.float64)
    coeffs = torch.randn(2, 5, 3, dtype=torch.complex128)
    values = torch.randn(2, 4, 3, dtype=torch.complex128)

    evaluated = backend.type2(points, coeffs, isign=1)
    adjoint_values = backend.type1(points, values, num_modes=5, isign=-1)

    lhs = torch.vdot(evaluated.reshape(-1), values.reshape(-1))
    rhs = torch.vdot(coeffs.reshape(-1), adjoint_values.reshape(-1))
    assert torch.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_synthesizer_has_semantic_real_output_interface():
    backend = DenseFourierBackend()
    synthesizer = IrregularFourierSynthesizer(
        num_modes=3,
        period_days=365.0,
        backend=backend,
    )
    positions = torch.tensor([[0.0, 30.0, 70.0]], dtype=torch.float64)
    positive = torch.tensor([[[0.5 + 0.25j]]], dtype=torch.complex128)
    zero = torch.tensor([[[1.0 + 0.0j]]], dtype=torch.complex128)
    coeffs = torch.cat([positive.conj(), zero, positive], dim=1)

    reconstructed = synthesizer(coeffs, positions)

    assert reconstructed.shape == (1, 3, 1)
    assert not reconstructed.is_complex()
    assert synthesizer.last_diagnostics["imaginary_residual"] < 1e-12


def test_official_backend_dependency_error_names_required_package():
    from models.fredn.nufft import PytorchFinufftBackend

    try:
        backend = PytorchFinufftBackend()
    except ImportError as exc:
        assert "pytorch-finufft" in str(exc)
    else:
        assert backend is not None


def test_official_backend_preserves_centered_mode_order_and_explicit_signs(monkeypatch):
    calls = []

    def fake_type2(points, targets, **kwargs):
        calls.append(("type2", kwargs))
        return torch.zeros(
            targets.shape[0],
            targets.shape[1],
            points.shape[-1],
            dtype=targets.dtype,
        )

    def fake_type1(points, values, output_shape, **kwargs):
        calls.append(("type1", kwargs))
        return torch.zeros(
            values.shape[0],
            values.shape[1],
            *output_shape,
            dtype=values.dtype,
        )

    package = types.ModuleType("pytorch_finufft")
    functional = types.ModuleType("pytorch_finufft.functional")
    functional.finufft_type1 = fake_type1
    functional.finufft_type2 = fake_type2
    package.functional = functional
    monkeypatch.setitem(sys.modules, "pytorch_finufft", package)
    monkeypatch.setitem(sys.modules, "pytorch_finufft.functional", functional)
    from models.fredn.nufft import PytorchFinufftBackend

    backend = PytorchFinufftBackend()
    points = torch.tensor([-0.5, 0.5])
    coeffs = torch.zeros(1, 3, 1, dtype=torch.complex64)
    values = torch.zeros(1, 2, 1, dtype=torch.complex64)
    backend.type2(points, coeffs, isign=1)
    backend.type1(points, values, num_modes=3, isign=-1)

    assert calls == [
        ("type2", {"eps": 1e-6, "modeord": 0, "isign": 1}),
        ("type1", {"eps": 1e-6, "modeord": 0, "isign": -1}),
    ]


def test_official_type1_adapter_passes_1d_output_shape_as_tuple():
    from models.fredn.nufft import PytorchFinufftBackend

    output_shapes = []

    def fake_type1(points, values, output_shape, **kwargs):
        output_shapes.append(output_shape)
        size = output_shape[0] if isinstance(output_shape, tuple) else output_shape
        return torch.zeros(
            values.shape[0],
            values.shape[1],
            size,
            dtype=values.dtype,
        )

    backend = object.__new__(PytorchFinufftBackend)
    backend._type1 = fake_type1
    backend.eps = 1e-6
    points = torch.tensor([-0.5, 0.5])
    values = torch.zeros(1, 2, 1, dtype=torch.complex64)

    result = backend.type1(points, values, num_modes=3, isign=-1)

    assert output_shapes == [(3,)]
    assert result.shape == (1, 3, 1)


def test_regularized_analysis_recovers_known_irregular_finite_series():
    backend = DenseFourierBackend()
    positions = torch.tensor(
        [[5.0, 40.0, 90.0, 140.0, 210.0, 280.0, 340.0]],
        dtype=torch.float64,
    )
    known_coeffs = _known_real_series()
    points = positions_to_periodic_points(positions, period_days=365.0)
    features = backend.type2(points[0], known_coeffs, isign=1).real
    analyzer = IrregularFourierAnalyzer(
        num_modes=5,
        period_days=365.0,
        reg=1e-10,
        tol=1e-10,
        max_iter=30,
        backend=backend,
    )
    synthesizer = IrregularFourierSynthesizer(5, 365.0, backend=backend)

    recovered, diagnostics = analyzer(features, positions)
    reconstructed = synthesizer(recovered, positions)

    coefficient_error = torch.linalg.vector_norm(recovered - known_coeffs) / torch.linalg.vector_norm(known_coeffs)
    reconstruction_error = torch.linalg.vector_norm(reconstructed - features) / torch.linalg.vector_norm(features)
    print("synthetic_coefficient_error", coefficient_error.item())
    print("synthetic_reconstruction_error", reconstruction_error.item())
    assert coefficient_error < 1e-7
    assert reconstruction_error < 1e-8
    assert diagnostics["solver_converged"] is True
    assert diagnostics["solver_iterations"] <= 30


def test_analysis_groups_equal_timestamp_rows_and_reports_shared_rate():
    backend = DenseFourierBackend()
    analyzer = IrregularFourierAnalyzer(
        num_modes=3,
        period_days=365.0,
        reg=1e-4,
        tol=1e-7,
        max_iter=10,
        backend=backend,
    )
    positions = torch.tensor(
        [[0.0, 30.0, 60.0], [0.0, 30.0, 60.0], [5.0, 35.0, 65.0]]
    )
    features = torch.randn(3, 3, 2)

    coeffs, diagnostics = analyzer(features, positions)
    synthesizer = IrregularFourierSynthesizer(3, 365.0, backend=backend)
    synthesizer(coeffs, positions)

    assert diagnostics["num_point_groups"] == 2
    assert diagnostics["shared_points_rate"] == pytest.approx(2.0 / 3.0)
    assert synthesizer.last_diagnostics["num_point_groups"] == 2
    assert synthesizer.last_diagnostics["shared_points_rate"] == pytest.approx(2.0 / 3.0)


def test_analysis_and_synthesis_propagate_gradients_to_real_features():
    backend = DenseFourierBackend()
    analyzer = IrregularFourierAnalyzer(
        num_modes=3,
        period_days=365.0,
        reg=1e-3,
        tol=1e-7,
        max_iter=10,
        backend=backend,
    )
    synthesizer = IrregularFourierSynthesizer(3, 365.0, backend=backend)
    positions = torch.tensor([[0.0, 50.0, 120.0, 200.0]])
    features = torch.randn(1, 4, 2, requires_grad=True)

    coeffs, _ = analyzer(features, positions)
    reconstructed = synthesizer(coeffs, positions)
    reconstructed.square().mean().backward()

    assert features.grad is not None
    assert torch.count_nonzero(features.grad).item() > 0


def test_dense_direct_matches_cg_reference_for_different_timestamp_rows():
    torch.manual_seed(23)
    positions = torch.tensor(
        [
            [3.0, 31.0, 76.0, 129.0, 190.0, 254.0, 333.0],
            [8.0, 42.0, 81.0, 145.0, 202.0, 271.0, 350.0],
            [1.0, 19.0, 68.0, 121.0, 184.0, 247.0, 319.0],
        ],
        dtype=torch.float64,
    )
    features = torch.randn(3, 7, 2, dtype=torch.float64)
    reference = IrregularFourierAnalyzer(
        num_modes=5,
        period_days=365.0,
        reg=1e-2,
        tol=1e-12,
        max_iter=100,
        backend=DenseFourierBackend(),
    )
    direct = BatchedDirectFourierAnalyzer(
        num_modes=5,
        period_days=365.0,
        reg=1e-2,
    )

    reference_coeffs, reference_diagnostics = reference(features, positions)
    direct_coeffs, direct_diagnostics = direct(features, positions)

    relative_error = torch.linalg.vector_norm(
        direct_coeffs - reference_coeffs
    ) / torch.linalg.vector_norm(reference_coeffs)
    assert relative_error < 1e-7
    assert reference_diagnostics["solver_converged"] is True
    assert direct_diagnostics["solver_info"].shape == (3,)
    assert torch.count_nonzero(direct_diagnostics["solver_info"]) == 0


def test_dense_direct_uses_one_batched_solve_ex_for_distinct_timestamps(monkeypatch):
    torch.manual_seed(29)
    positions = torch.tensor(
        [
            [0.0, 20.0, 60.0, 130.0, 220.0, 330.0],
            [5.0, 35.0, 80.0, 150.0, 245.0, 350.0],
            [9.0, 47.0, 95.0, 175.0, 260.0, 359.0],
        ],
        dtype=torch.float64,
    )
    features = torch.randn(3, 6, 2, dtype=torch.float64)
    calls = []
    original_solve_ex = torch.linalg.solve_ex

    def recording_solve_ex(gram, rhs, **kwargs):
        calls.append((gram.shape, rhs.shape, kwargs))
        return original_solve_ex(gram, rhs, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve_ex", recording_solve_ex)
    analyzer = BatchedDirectFourierAnalyzer(5, 365.0, 1e-3)

    coeffs, _ = analyzer(features, positions)

    assert coeffs.shape == (3, 5, 2)
    assert calls == [
        (torch.Size([3, 5, 5]), torch.Size([3, 5, 2]), {"check_errors": False})
    ]


def test_dense_direct_trend_and_seasonal_match_cg_reference_path():
    torch.manual_seed(31)
    positions = torch.tensor(
        [
            [4.0, 37.0, 88.0, 146.0, 213.0, 282.0, 344.0],
            [7.0, 45.0, 101.0, 158.0, 226.0, 294.0, 356.0],
        ],
        dtype=torch.float64,
    )
    features = torch.randn(2, 7, 3, dtype=torch.float64)
    backend = DenseFourierBackend()
    reference_analyzer = IrregularFourierAnalyzer(
        5, 365.0, 1e-2, 1e-12, 100, backend=backend
    )
    reference_synthesizer = IrregularFourierSynthesizer(
        5, 365.0, backend=backend
    )
    direct_analyzer = BatchedDirectFourierAnalyzer(5, 365.0, 1e-2)
    direct_synthesizer = BatchedDirectFourierSynthesizer(5, 365.0)
    disentangler = FrequencyDisentangler(5, 3).double()

    reference_coeffs, _ = reference_analyzer(features, positions)
    direct_coeffs, _ = direct_analyzer(features, positions)
    reference_trend, reference_seasonal, _ = disentangler(reference_coeffs)
    direct_trend, direct_seasonal, _ = disentangler(direct_coeffs)

    reference_trend_output = reference_synthesizer(reference_trend, positions)
    reference_seasonal_output = reference_synthesizer(reference_seasonal, positions)
    direct_trend_output = direct_synthesizer(direct_trend, positions)
    direct_seasonal_output = direct_synthesizer(direct_seasonal, positions)

    assert torch.allclose(direct_trend_output, reference_trend_output, rtol=1e-7, atol=1e-9)
    assert torch.allclose(
        direct_seasonal_output,
        reference_seasonal_output,
        rtol=1e-7,
        atol=1e-9,
    )


def test_dense_direct_preserves_conjugate_symmetry_and_real_synthesis():
    torch.manual_seed(37)
    positions = torch.tensor(
        [[2.0, 29.0, 73.0, 118.0, 177.0, 238.0, 301.0, 358.0]],
        dtype=torch.float64,
    )
    features = torch.randn(1, 8, 2, dtype=torch.float64)
    analyzer = BatchedDirectFourierAnalyzer(5, 365.0, 1e-3)
    synthesizer = BatchedDirectFourierSynthesizer(5, 365.0)

    coeffs, _ = analyzer(features, positions)
    reconstructed_complex = synthesizer.synthesize_complex(coeffs, positions)

    assert torch.allclose(coeffs[:, :2], coeffs[:, 3:].flip(1).conj(), rtol=1e-10, atol=1e-10)
    assert torch.linalg.vector_norm(reconstructed_complex.imag) < 1e-10
    assert not synthesizer(coeffs, positions).is_complex()
