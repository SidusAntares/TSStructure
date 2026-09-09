"""Parameter-free finite Fourier reconstruction on irregular timestamps."""

import math
from typing import Dict

import torch
from torch import nn


def centered_modes(num_modes: int, device=None, dtype=None) -> torch.Tensor:
    """Return the shared odd Fourier index set ``[-K, ..., K]``."""
    if num_modes <= 0 or num_modes % 2 == 0:
        raise ValueError("fourier_num_modes must be a positive odd integer")
    half = num_modes // 2
    return torch.arange(-half, half + 1, device=device, dtype=dtype)


def positions_to_periodic_points(
    positions: torch.Tensor,
    period_days: float,
) -> torch.Tensor:
    """Map physical day coordinates to the shared periodic interval."""
    if period_days <= 0:
        raise ValueError("fourier_period_days must be positive")
    points = 2.0 * math.pi * positions / period_days
    return torch.remainder(points + math.pi, 2.0 * math.pi) - math.pi


def _batched_fourier_matrix(
    points: torch.Tensor,
    num_modes: int,
    isign: int,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    if isign not in (-1, 1):
        raise ValueError("isign must be explicitly set to -1 or 1")
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    modes = centered_modes(num_modes, device=points.device, dtype=real_dtype)
    phase = points.to(real_dtype).unsqueeze(-1) * modes
    return torch.exp((isign * 1j) * phase).to(complex_dtype)


class BatchedDirectFourierAnalyzer(nn.Module):
    """Solve a batched irregular Fourier ridge system with ``solve_ex``."""

    def __init__(
        self,
        num_modes: int,
        period_days: float = 365.0,
        reg: float = 1e-3,
        synthesis_isign: int = 1,
    ):
        super().__init__()
        centered_modes(num_modes)
        if reg < 0:
            raise ValueError("fourier_reg must be non-negative")
        if period_days <= 0:
            raise ValueError("fourier_period_days must be positive")
        if synthesis_isign not in (-1, 1):
            raise ValueError("synthesis_isign must be explicitly set to -1 or 1")
        self.num_modes = num_modes
        self.period_days = period_days
        self.reg = reg
        self.synthesis_isign = synthesis_isign
        self.last_diagnostics: Dict[str, object] = {}

    @staticmethod
    def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
        return torch.complex128 if real_dtype == torch.float64 else torch.complex64

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        collect_diagnostics: bool = False,
    ):
        if features.ndim != 3 or positions.ndim != 2:
            raise ValueError("features must be [B,L,D] and positions must be [B,L]")
        if features.is_complex():
            raise ValueError("features must be real-valued")
        if features.shape[:2] != positions.shape:
            raise ValueError("features and positions batch/time dimensions must match")

        complex_dtype = self._complex_dtype(features.dtype)
        points = positions_to_periodic_points(positions, self.period_days)
        matrix = _batched_fourier_matrix(
            points, self.num_modes, self.synthesis_isign, complex_dtype
        )
        adjoint = matrix.conj().transpose(-2, -1)
        gram = torch.matmul(adjoint, matrix)
        identity = torch.eye(
            self.num_modes, device=features.device, dtype=complex_dtype
        )
        gram = gram + self.reg * identity
        rhs = torch.matmul(adjoint, features.to(complex_dtype))
        solution, info = torch.linalg.solve_ex(gram, rhs, check_errors=False)
        diagnostics = {"solver_info": info.detach()}
        if collect_diagnostics:
            normal_residual = torch.matmul(gram, solution) - rhs
            residual = torch.linalg.vector_norm(normal_residual, dim=(-2, -1))
            residual = residual / torch.linalg.vector_norm(
                rhs, dim=(-2, -1)
            ).clamp_min(torch.finfo(features.dtype).eps)
            diagnostics.update(
                {
                    "per_sample_solver_residual": residual.detach(),
                    "per_sample_solver_converged": (
                        (info == 0) & torch.isfinite(residual)
                    ).detach(),
                }
            )
        self.last_diagnostics = diagnostics
        return solution, diagnostics


class BatchedDirectFourierSynthesizer(nn.Module):
    """Evaluate centered Fourier modes at irregular timestamp rows."""

    def __init__(
        self,
        num_modes: int,
        period_days: float = 365.0,
        synthesis_isign: int = 1,
    ):
        super().__init__()
        centered_modes(num_modes)
        if period_days <= 0:
            raise ValueError("fourier_period_days must be positive")
        if synthesis_isign not in (-1, 1):
            raise ValueError("synthesis_isign must be explicitly set to -1 or 1")
        self.num_modes = num_modes
        self.period_days = period_days
        self.synthesis_isign = synthesis_isign

    def synthesize_complex(
        self,
        coeffs: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if coeffs.ndim != 3 or positions.ndim != 2:
            raise ValueError("coeffs must be [B,F,D] and positions must be [B,L]")
        if coeffs.shape[0] != positions.shape[0]:
            raise ValueError("coeffs and positions must have the same batch size")
        if coeffs.shape[1] != self.num_modes:
            raise ValueError("coeffs frequency dimension does not match num_modes")
        if not coeffs.is_complex():
            raise ValueError("coeffs must be complex-valued")
        points = positions_to_periodic_points(positions, self.period_days)
        matrix = _batched_fourier_matrix(
            points, self.num_modes, self.synthesis_isign, coeffs.dtype
        )
        return torch.matmul(matrix, coeffs)

    def forward(self, coeffs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.synthesize_complex(coeffs, positions).real
