"""Irregular Fourier operators used by FreDN.

Type 2 evaluates Fourier coefficients at irregular points. Type 1 with the
opposite ``isign`` is its adjoint; it is deliberately not described as an
inverse because irregular Fourier analysis requires a linear solve.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


def centered_modes(num_modes: int, device=None, dtype=None) -> torch.Tensor:
    """Return the shared odd Fourier index set ``[-K, ..., K]``."""
    if num_modes <= 0 or num_modes % 2 == 0:
        raise ValueError("fredn_num_modes must be a positive odd integer")
    half = num_modes // 2
    return torch.arange(-half, half + 1, device=device, dtype=dtype)


def positions_to_periodic_points(
    positions: torch.Tensor,
    period_days: float,
) -> torch.Tensor:
    """Map shared physical day coordinates to FINUFFT's ``[-pi, pi)`` box."""
    if period_days <= 0:
        raise ValueError("fredn_period_days must be positive")
    points = 2.0 * math.pi * positions / period_days
    return torch.remainder(points + math.pi, 2.0 * math.pi) - math.pi


def _batched_fourier_matrix(
    points: torch.Tensor,
    num_modes: int,
    isign: int,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    """Build ``A[b,l,k] = exp(isign * i * x[b,l] * k)`` in one batch."""
    if isign not in (-1, 1):
        raise ValueError("isign must be explicitly set to -1 or 1")
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    modes = centered_modes(num_modes, device=points.device, dtype=real_dtype)
    phase = points.to(real_dtype).unsqueeze(-1) * modes
    return torch.exp((isign * 1j) * phase).to(complex_dtype)


class BatchedDirectFourierAnalyzer(nn.Module):
    """Solve the batched irregular Fourier ridge system with a dense direct solve."""

    def __init__(
        self,
        num_modes: int,
        period_days: float,
        reg: float,
        synthesis_isign: int = 1,
    ):
        super().__init__()
        centered_modes(num_modes)
        if reg < 0:
            raise ValueError("fredn_nufft_reg must be non-negative")
        if period_days <= 0:
            raise ValueError("fredn_period_days must be positive")
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
            points,
            self.num_modes,
            self.synthesis_isign,
            complex_dtype,
        )
        adjoint = matrix.conj().transpose(-2, -1)
        gram = torch.matmul(adjoint, matrix)
        identity = torch.eye(
            self.num_modes,
            device=features.device,
            dtype=complex_dtype,
        )
        gram = gram + self.reg * identity
        rhs = torch.matmul(adjoint, features.to(complex_dtype))
        solution, info = torch.linalg.solve_ex(
            gram,
            rhs,
            check_errors=False,
        )
        diagnostics = {"solver_info": info.detach()}
        if collect_diagnostics:
            normal_residual = torch.matmul(gram, solution) - rhs
            solver_residual = torch.linalg.vector_norm(
                normal_residual,
                dim=(-2, -1),
            ) / torch.linalg.vector_norm(rhs, dim=(-2, -1)).clamp_min(
                torch.finfo(features.dtype).eps
            )
            identical_rows = (
                positions.unsqueeze(1) == positions.unsqueeze(0)
            ).all(dim=-1)
            shared_points_rate = (
                identical_rows.sum(dim=1) > 1
            ).to(features.dtype).mean()
            condition_numbers = torch.linalg.cond(gram.detach())
            diagnostics.update(
                {
                    "solver_residual": solver_residual.max().detach(),
                    "per_sample_solver_residual": solver_residual.detach(),
                    "per_sample_solver_converged": (
                        (info == 0) & torch.isfinite(solver_residual)
                    ).detach(),
                    "shared_points_rate": shared_points_rate.detach(),
                    "condition_numbers": condition_numbers.detach(),
                }
            )
        self.last_diagnostics = diagnostics
        return solution, diagnostics


class BatchedDirectFourierSynthesizer(nn.Module):
    """Evaluate centered Fourier modes for all timestamp rows without grouping."""

    def __init__(
        self,
        num_modes: int,
        period_days: float,
        synthesis_isign: int = 1,
    ):
        super().__init__()
        centered_modes(num_modes)
        if period_days <= 0:
            raise ValueError("fredn_period_days must be positive")
        if synthesis_isign not in (-1, 1):
            raise ValueError("synthesis_isign must be explicitly set to -1 or 1")
        self.num_modes = num_modes
        self.period_days = period_days
        self.synthesis_isign = synthesis_isign
        self.last_diagnostics: Dict[str, object] = {}

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
            points,
            self.num_modes,
            self.synthesis_isign,
            coeffs.dtype,
        )
        return torch.matmul(matrix, coeffs)

    def forward(
        self,
        coeffs: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.synthesize_complex(coeffs, positions).real


class DenseFourierBackend:
    """Differentiable exact Fourier sums used as a test reference backend."""

    @staticmethod
    def _matrix(
        points: torch.Tensor,
        num_modes: int,
        isign: int,
        complex_dtype: torch.dtype,
    ) -> torch.Tensor:
        if isign not in (-1, 1):
            raise ValueError("isign must be explicitly set to -1 or 1")
        real_dtype = (
            torch.float64 if complex_dtype == torch.complex128 else torch.float32
        )
        modes = centered_modes(
            num_modes,
            device=points.device,
            dtype=real_dtype,
        )
        phase = points.to(real_dtype).unsqueeze(-1) * modes
        return torch.exp((isign * 1j) * phase).to(complex_dtype)

    def type2(
        self,
        points: torch.Tensor,
        coeffs: torch.Tensor,
        *,
        isign: int,
    ) -> torch.Tensor:
        matrix = self._matrix(points, coeffs.shape[1], isign, coeffs.dtype)
        return torch.einsum("lf,bfd->bld", matrix, coeffs)

    def type1(
        self,
        points: torch.Tensor,
        values: torch.Tensor,
        *,
        num_modes: int,
        isign: int,
    ) -> torch.Tensor:
        matrix = self._matrix(points, num_modes, isign, values.dtype)
        return torch.einsum("lf,bld->bfd", matrix, values)


class PytorchFinufftBackend:
    """Official autograd-capable pytorch-finufft backend adapter."""

    def __init__(self, eps: float = 1e-6):
        try:
            from pytorch_finufft.functional import finufft_type1, finufft_type2
        except ImportError as exc:
            raise ImportError(
                "FreDN requires pytorch-finufft (server package "
                "`pytorch-finufft==0.1.0`) and a FINUFFT/cuFINUFFT backend"
            ) from exc
        self._type1 = finufft_type1
        self._type2 = finufft_type2
        self.eps = eps

    def type2(
        self,
        points: torch.Tensor,
        coeffs: torch.Tensor,
        *,
        isign: int,
    ) -> torch.Tensor:
        if isign not in (-1, 1):
            raise ValueError("isign must be explicitly set to -1 or 1")
        targets = coeffs.permute(0, 2, 1).contiguous()
        values = self._type2(
            points.unsqueeze(0),
            targets,
            eps=self.eps,
            modeord=0,
            isign=isign,
        )
        return values.permute(0, 2, 1).contiguous()

    def type1(
        self,
        points: torch.Tensor,
        values: torch.Tensor,
        *,
        num_modes: int,
        isign: int,
    ) -> torch.Tensor:
        if isign not in (-1, 1):
            raise ValueError("isign must be explicitly set to -1 or 1")
        strengths = values.permute(0, 2, 1).contiguous()
        output_shape = (int(num_modes),)
        coeffs = self._type1(
            points.unsqueeze(0),
            strengths,
            output_shape,
            eps=self.eps,
            modeord=0,
            isign=isign,
        )
        return coeffs.permute(0, 2, 1).contiguous()


def _group_identical_rows(
    positions: torch.Tensor,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Group batch indices whose timestamp rows are exactly equal."""
    groups = {}
    for batch_index, row in enumerate(positions.detach().cpu().tolist()):
        groups.setdefault(tuple(row), []).append(batch_index)
    result = []
    for indices in groups.values():
        index_tensor = torch.tensor(indices, device=positions.device, dtype=torch.long)
        result.append((index_tensor, positions[indices[0]]))
    return result


def _shared_points_rate(groups, batch_size: int) -> float:
    shared_samples = sum(len(indices) for indices, _ in groups if len(indices) > 1)
    return float(shared_samples) / float(batch_size)


def _synchronize_if_cuda(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


class IrregularFourierAnalyzer(nn.Module):
    """Solve regularized irregular Fourier least squares with complex CG."""

    def __init__(
        self,
        num_modes: int,
        period_days: float,
        reg: float,
        tol: float,
        max_iter: int,
        backend: Optional[object] = None,
        synthesis_isign: int = 1,
    ):
        super().__init__()
        centered_modes(num_modes)
        if reg < 0:
            raise ValueError("fredn_nufft_reg must be non-negative")
        if tol <= 0:
            raise ValueError("fredn_nufft_tol must be positive")
        if max_iter <= 0:
            raise ValueError("fredn_nufft_max_iter must be positive")
        if synthesis_isign not in (-1, 1):
            raise ValueError("synthesis_isign must be explicitly set to -1 or 1")
        self.num_modes = num_modes
        self.period_days = period_days
        self.reg = reg
        self.tol = tol
        self.max_iter = max_iter
        self.backend = backend if backend is not None else PytorchFinufftBackend()
        self.synthesis_isign = synthesis_isign
        self.adjoint_isign = -synthesis_isign
        self.last_diagnostics: Dict[str, object] = {}

    @staticmethod
    def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
        return torch.complex128 if real_dtype == torch.float64 else torch.complex64

    def _solve_group(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        int,
        float,
        bool,
        List[int],
        List[float],
        List[bool],
    ]:
        values = features.to(self._complex_dtype(features.dtype))
        rhs = self.backend.type1(
            points,
            values,
            num_modes=self.num_modes,
            isign=self.adjoint_isign,
        )

        def normal_operator(coeffs):
            evaluated = self.backend.type2(
                points,
                coeffs,
                isign=self.synthesis_isign,
            )
            adjointed = self.backend.type1(
                points,
                evaluated,
                num_modes=self.num_modes,
                isign=self.adjoint_isign,
            )
            return adjointed + self.reg * coeffs

        solution = torch.zeros_like(rhs)
        residual = rhs.clone()
        direction = residual.clone()
        residual_squared = (residual.conj() * residual).sum(dim=1).real
        rhs_norm = residual_squared.sqrt().clamp_min(torch.finfo(features.dtype).eps)
        relative_residual = residual_squared.sqrt() / rhs_norm
        sample_converged = torch.all(relative_residual <= self.tol, dim=1)
        sample_iterations = torch.full(
            (features.shape[0],),
            -1,
            device=features.device,
            dtype=torch.long,
        )
        sample_iterations[sample_converged] = 0
        converged = bool(torch.all(sample_converged).item())
        iterations = 0
        tiny = torch.finfo(features.dtype).eps

        while iterations < self.max_iter and not converged:
            operator_direction = normal_operator(direction)
            denominator = (
                (direction.conj() * operator_direction).sum(dim=1).real
            ).clamp_min(tiny)
            alpha = residual_squared / denominator
            solution = solution + alpha.unsqueeze(1) * direction
            new_residual = residual - alpha.unsqueeze(1) * operator_direction
            new_residual_squared = (
                (new_residual.conj() * new_residual).sum(dim=1).real
            )
            iterations += 1
            relative_residual = new_residual_squared.sqrt() / rhs_norm
            sample_converged = torch.all(relative_residual <= self.tol, dim=1)
            newly_converged = (sample_iterations < 0) & sample_converged
            sample_iterations[newly_converged] = iterations
            converged = bool(torch.all(sample_converged).item())
            if converged:
                residual = new_residual
                residual_squared = new_residual_squared
                break
            beta = new_residual_squared / residual_squared.clamp_min(tiny)
            direction = new_residual + beta.unsqueeze(1) * direction
            residual = new_residual
            residual_squared = new_residual_squared

        max_relative_residual = float(relative_residual.max().detach().cpu())
        sample_residuals = relative_residual.max(dim=1).values
        sample_iterations[sample_iterations < 0] = iterations
        return (
            solution,
            iterations,
            max_relative_residual,
            converged,
            sample_iterations.detach().cpu().tolist(),
            sample_residuals.detach().cpu().tolist(),
            sample_converged.detach().cpu().tolist(),
        )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
    ):
        if features.ndim != 3 or positions.ndim != 2:
            raise ValueError("features must be [B,L,D] and positions must be [B,L]")
        if features.is_complex():
            raise ValueError("features must be real-valued")
        if features.shape[:2] != positions.shape:
            raise ValueError("features and positions batch/time dimensions must match")

        points = positions_to_periodic_points(positions, self.period_days)
        groups = _group_identical_rows(points)
        _synchronize_if_cuda(features)
        started = time.perf_counter()
        per_sample = [None] * features.shape[0]
        per_sample_iterations = [None] * features.shape[0]
        per_sample_residuals = [None] * features.shape[0]
        per_sample_converged = [None] * features.shape[0]
        group_diagnostics = []
        for indices, shared_points in groups:
            group_features = features.index_select(0, indices)
            (
                solution,
                iterations,
                residual,
                converged,
                group_sample_iterations,
                group_sample_residuals,
                group_sample_converged,
            ) = self._solve_group(shared_points, group_features)
            for offset, batch_index in enumerate(indices.detach().cpu().tolist()):
                per_sample[batch_index] = solution[offset]
                per_sample_iterations[batch_index] = group_sample_iterations[offset]
                per_sample_residuals[batch_index] = group_sample_residuals[offset]
                per_sample_converged[batch_index] = group_sample_converged[offset]
            group_diagnostics.append((iterations, residual, converged))
        coeffs = torch.stack(per_sample, dim=0)
        _synchronize_if_cuda(coeffs)
        elapsed = time.perf_counter() - started
        diagnostics = {
            "solver_iterations": max(item[0] for item in group_diagnostics),
            "solver_residual": max(item[1] for item in group_diagnostics),
            "solver_converged": all(item[2] for item in group_diagnostics),
            "num_point_groups": len(groups),
            "shared_points_rate": _shared_points_rate(groups, features.shape[0]),
            "analysis_time": elapsed,
            "per_sample_solver_iterations": per_sample_iterations,
            "per_sample_solver_residual": per_sample_residuals,
            "per_sample_solver_converged": per_sample_converged,
        }
        self.last_diagnostics = diagnostics
        return coeffs, diagnostics


class IrregularFourierSynthesizer(nn.Module):
    """Evaluate complex Fourier coefficients and return their real signal."""

    def __init__(
        self,
        num_modes: int,
        period_days: float,
        backend: Optional[object] = None,
        synthesis_isign: int = 1,
    ):
        super().__init__()
        centered_modes(num_modes)
        if synthesis_isign not in (-1, 1):
            raise ValueError("synthesis_isign must be explicitly set to -1 or 1")
        self.num_modes = num_modes
        self.period_days = period_days
        self.backend = backend if backend is not None else PytorchFinufftBackend()
        self.synthesis_isign = synthesis_isign
        self.last_diagnostics: Dict[str, object] = {}

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

        points = positions_to_periodic_points(positions, self.period_days)
        groups = _group_identical_rows(points)
        _synchronize_if_cuda(coeffs)
        started = time.perf_counter()
        per_sample = [None] * coeffs.shape[0]
        for indices, shared_points in groups:
            group_coeffs = coeffs.index_select(0, indices)
            group_result = self.backend.type2(
                shared_points,
                group_coeffs,
                isign=self.synthesis_isign,
            )
            for offset, batch_index in enumerate(indices.detach().cpu().tolist()):
                per_sample[batch_index] = group_result[offset]
        result = torch.stack(per_sample, dim=0)
        _synchronize_if_cuda(result)
        elapsed = time.perf_counter() - started
        denominator = torch.linalg.vector_norm(result.real).clamp_min(1e-12)
        imaginary_residual = torch.linalg.vector_norm(result.imag) / denominator
        self.last_diagnostics = {
            "synthesis_time": elapsed,
            "imaginary_residual": float(imaginary_residual.detach().cpu()),
            "num_point_groups": len(groups),
            "shared_points_rate": _shared_points_rate(groups, coeffs.shape[0]),
        }
        return result

    def forward(
        self,
        coeffs: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.synthesize_complex(coeffs, positions).real
