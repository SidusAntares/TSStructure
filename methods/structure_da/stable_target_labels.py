"""Three-way confirmed-phase evidence scan for stable target labels."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .confirmed_phase_view import (
    IDENTITY_PHASE_GROUP_ID,
    ConfirmedPhaseView,
    build_confirmed_class_to_group_map,
    build_confirmed_phase_view,
    build_phase_calibrated_view,
)
from .domain_phase_state import (
    CandidatePhaseCompatibilityStatus,
    DomainPhaseConfig,
    DomainPhaseState,
    PhaseDecisionStatus,
    PhaseGroup,
    PhaseGroupStatus,
    evaluate_sample_class_phase_compatibility,
)
from .ema_teacher import Stage2EMATeacher
from .prototype_bank import SourcePrototypeBank, support_aware_q_distance
from .target_hypothesis_scan import CandidatePseudoLabel, TargetHypothesisScanResult


@dataclass(frozen=True)
class StableLabelConfig:
    tau_f: float
    tau_q: float
    cls_confidence_min: float | None
    cls_margin_min: float | None
    fused_confidence_min: float | None
    fused_margin_min: float | None
    q_confidence_min: float | None
    q_margin_min: float | None

    def __post_init__(self) -> None:
        for name in ("tau_f", "tau_q"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        for branch, confidence, margin in (
            ("classifier", self.cls_confidence_min, self.cls_margin_min),
            ("fused", self.fused_confidence_min, self.fused_margin_min),
            ("q", self.q_confidence_min, self.q_margin_min),
        ):
            if confidence is None and margin is None:
                raise ValueError(f"{branch} requires a confidence or margin gate")
            for name, value in (("confidence", confidence), ("margin", margin)):
                if value is not None and (
                    not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError(f"{branch} {name} gate must lie in [0,1]")


@dataclass(frozen=True)
class StableTargetCandidate:
    sample_id: int
    class_id: int
    group_id: int
    cls_confidence: float
    cls_margin: float
    fused_distance: float
    fused_confidence: float
    fused_margin: float
    q_distance: float
    q_confidence: float
    q_margin: float
    q_common_support: float
    passed_classifier: bool
    passed_fused: bool
    passed_q: bool
    accepted: bool
    reject_reason: str | None
    phase_compatible: bool = False
    phase_distance_to_group: float | None = None
    candidate_q_shape_distance: float | None = None
    candidate_ambiguous: bool = False


@dataclass(frozen=True)
class StableTargetLabel:
    sample_id: int
    class_id: int
    group_id: int
    aligned_q_shape: Tensor
    aligned_q_support: Tensor
    fused_repr: Tensor
    confidence_summary: float


@dataclass(frozen=True)
class StableTargetLabelScanResult:
    candidates: tuple[StableTargetCandidate, ...]
    stable_labels: tuple[StableTargetLabel, ...]
    num_samples: int
    num_without_confirmed_phase: int
    num_candidate_views: int
    num_classifier_pass: int
    num_fused_pass: int
    num_q_pass: int
    num_stable_labels: int
    num_ambiguous_rejected: int
    stable_class_counts: tuple[int, ...]
    num_phase_compatible: int = 0
    num_phase_incompatible: int = 0


def _passes_gate(
    confidence: float,
    margin: float,
    confidence_min: float | None,
    margin_min: float | None,
) -> bool:
    return (
        (confidence_min is None or confidence >= confidence_min)
        and (margin_min is None or margin >= margin_min)
    )


def _probability_margin(probabilities: Tensor, candidate_index: int) -> tuple[float, float]:
    candidate = float(probabilities[candidate_index].item())
    others = torch.cat(
        (probabilities[:candidate_index], probabilities[candidate_index + 1 :])
    )
    if others.numel() == 0:
        return candidate, float("nan")
    return candidate, candidate - float(others.max().item())


def _integration_weights(grid_size: int, *, device, dtype) -> Tensor:
    weights = torch.ones(grid_size, device=device, dtype=dtype)
    weights[[0, -1]] *= 0.5
    return weights / weights.sum()


@torch.no_grad()
def evaluate_stable_target_candidate(
    *,
    view: ConfirmedPhaseView,
    view_index: int,
    class_id: int,
    source_prototype_bank: SourcePrototypeBank,
    config: StableLabelConfig,
) -> StableTargetCandidate:
    sample_id = int(view.sample_ids[view_index].item())
    if class_id not in view.member_classes:
        return StableTargetCandidate(
            sample_id, class_id, view.group_id,
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), float("nan"),
            False, False, False, False, "class_not_in_confirmed_group",
        )

    probabilities = view.probabilities[view_index]
    cls_confidence, cls_margin = _probability_margin(probabilities, class_id)
    passed_classifier = (
        int(probabilities.argmax().item()) == class_id
        and _passes_gate(
            cls_confidence,
            cls_margin,
            config.cls_confidence_min,
            config.cls_margin_min,
        )
    )

    bank = source_prototype_bank
    ready = bank.ready.to(device=view.fused_repr.device)
    ready_indices = torch.nonzero(ready, as_tuple=False).flatten()
    fused_distance = fused_confidence = fused_margin = float("nan")
    passed_fused = False
    if (
        0 <= class_id < ready.numel()
        and bool(ready[class_id].item())
        and ready_indices.numel() >= 2
        and bank.f_distance_samples[class_id] is not None
        and bank.f_distance_samples[class_id].numel() > 0
        and math.isfinite(float(bank.f_quantiles[class_id, 2].item()))
    ):
        query = view.fused_repr[view_index].unsqueeze(0)
        prototypes = bank.fused.to(device=query.device, dtype=query.dtype)[ready_indices]
        distances = 1.0 - F.cosine_similarity(query, prototypes, dim=-1)
        fused_probabilities = torch.softmax(-distances / config.tau_f, dim=0)
        local = int(torch.nonzero(ready_indices == class_id, as_tuple=False)[0].item())
        fused_distance = float(distances[local].item())
        fused_confidence, fused_margin = _probability_margin(fused_probabilities, local)
        passed_fused = (
            int(fused_probabilities.argmax().item()) == local
            and fused_distance <= float(bank.f_quantiles[class_id, 2].item())
            and _passes_gate(
                fused_confidence,
                fused_margin,
                config.fused_confidence_min,
                config.fused_margin_min,
            )
        )

    q_distance = q_confidence = q_margin = q_common_support = float("nan")
    passed_q = False
    if 0 <= class_id < ready.numel() and bool(view.q_valid[view_index].item()):
        q_query = view.aligned_q_shape[view_index : view_index + 1]
        support_query = view.aligned_q_support[view_index : view_index + 1]
        q_prototypes = bank.shape_srvf.to(device=q_query.device, dtype=q_query.dtype)[ready_indices]
        support_prototypes = bank.shape_support.to(
            device=q_query.device, dtype=q_query.dtype
        )[ready_indices]
        distances = support_aware_q_distance(
            q_query,
            q_prototypes,
            support_query,
            support_prototypes,
            _integration_weights(q_query.shape[1], device=q_query.device, dtype=q_query.dtype),
        )
        valid_local = distances.valid[0]
        candidate_positions = torch.nonzero(
            ready_indices == class_id, as_tuple=False
        ).flatten()
        if candidate_positions.numel() and valid_local.sum().item() >= 2:
            local = int(candidate_positions[0].item())
            if (
                bool(valid_local[local].item())
                and bank.q_distance_samples[class_id] is not None
                and bank.q_distance_samples[class_id].numel() > 0
                and math.isfinite(float(bank.q_quantiles[class_id, 2].item()))
            ):
                valid_indices = torch.nonzero(valid_local, as_tuple=False).flatten()
                valid_distances_sq = distances.distance_sq[0, valid_indices]
                q_probabilities = torch.softmax(-valid_distances_sq / config.tau_q, dim=0)
                candidate_valid_position = int(
                    torch.nonzero(valid_indices == local, as_tuple=False)[0].item()
                )
                q_distance = float(distances.distance[0, local].item())
                q_common_support = float(distances.common_support[0, local].item())
                q_confidence, q_margin = _probability_margin(
                    q_probabilities, candidate_valid_position
                )
                passed_q = (
                    int(q_probabilities.argmax().item()) == candidate_valid_position
                    and q_distance <= float(bank.q_quantiles[class_id, 2].item())
                    and _passes_gate(
                        q_confidence,
                        q_margin,
                        config.q_confidence_min,
                        config.q_margin_min,
                    )
                )

    accepted = passed_classifier and passed_fused and passed_q
    reject_reason = None
    if not accepted:
        failed = []
        if not passed_classifier:
            failed.append("classifier")
        if not passed_fused:
            failed.append("fused")
        if not passed_q:
            failed.append("q")
        reject_reason = "+".join(failed)
    return StableTargetCandidate(
        sample_id=sample_id,
        class_id=class_id,
        group_id=view.group_id,
        cls_confidence=cls_confidence,
        cls_margin=cls_margin,
        fused_distance=fused_distance,
        fused_confidence=fused_confidence,
        fused_margin=fused_margin,
        q_distance=q_distance,
        q_confidence=q_confidence,
        q_margin=q_margin,
        q_common_support=q_common_support,
        passed_classifier=passed_classifier,
        passed_fused=passed_fused,
        passed_q=passed_q,
        accepted=accepted,
        reject_reason=reject_reason,
    )


def _empty_result(num_samples: int, num_classes: int) -> StableTargetLabelScanResult:
    return StableTargetLabelScanResult(
        candidates=(),
        stable_labels=(),
        num_samples=num_samples,
        num_without_confirmed_phase=num_samples,
        num_candidate_views=0,
        num_classifier_pass=0,
        num_fused_pass=0,
        num_q_pass=0,
        num_stable_labels=0,
        num_ambiguous_rejected=0,
        stable_class_counts=(0,) * num_classes,
    )


def _subset_batch(batch: dict, rows: list[int], batch_size: int) -> dict:
    subset = {}
    index = torch.tensor(rows, dtype=torch.long)
    for name in ("pixels", "valid_pixels", "positions", "extra", "time_mask"):
        value = batch.get(name)
        if value is None:
            continue
        if not isinstance(value, Tensor):
            raise ValueError(f"batch[{name!r}] must be a tensor")
        if value.ndim > 0 and value.shape[0] == batch_size and not (
            name == "positions" and value.ndim == 1
        ):
            subset[name] = value.index_select(0, index.to(value.device))
        else:
            subset[name] = value
    return subset


def _candidate_phase_rejection(
    candidate: CandidatePseudoLabel,
    *,
    phase_status: CandidatePhaseCompatibilityStatus,
    phase_distance: float | None,
) -> StableTargetCandidate:
    nan = float("nan")
    return StableTargetCandidate(
        sample_id=int(candidate.sample_id),
        class_id=int(candidate.class_id),
        group_id=IDENTITY_PHASE_GROUP_ID,
        cls_confidence=nan,
        cls_margin=nan,
        fused_distance=nan,
        fused_confidence=nan,
        fused_margin=nan,
        q_distance=nan,
        q_confidence=nan,
        q_margin=nan,
        q_common_support=nan,
        passed_classifier=False,
        passed_fused=False,
        passed_q=False,
        accepted=False,
        reject_reason=f"phase_{phase_status.value}",
        phase_compatible=False,
        phase_distance_to_group=phase_distance,
        candidate_q_shape_distance=float(candidate.q_shape_distance),
        candidate_ambiguous=bool(candidate.ambiguous),
    )


def _identity_gamma_from_compatibility(compatibility) -> Tensor:
    for item in compatibility:
        if item.gamma is not None:
            return torch.linspace(0.0, 1.0, item.gamma.numel(), dtype=torch.float64)
    raise RuntimeError("identity-confirmed stable-label scan requires cached candidate gammas")


@torch.no_grad()
def scan_stable_target_labels_from_candidates(
    *,
    ema_teacher: Stage2EMATeacher,
    target_loader,
    hypothesis_result: TargetHypothesisScanResult,
    phase_state: DomainPhaseState,
    phase_config: DomainPhaseConfig,
    source_prototype_bank: SourcePrototypeBank,
    config: StableLabelConfig,
    sample_ids: tuple[int, ...] | None = None,
) -> StableTargetLabelScanResult:
    """Confirm or correct the Round-A candidate set under confirmed Domain Phase.

    The formal path never searches arbitrary classes.  A non-ambiguous sample
    contributes only its primary geometry-first candidate.  A near-tie may also
    contribute the single Round-A secondary candidate; confirmed Domain Phase
    can therefore resolve that *existing ambiguity* without allowing the
    classifier to invent a new class.

    Individual sample/class gammas are used only for Phase compatibility.  Any
    Shape stored in a stable label is recomputed with the confirmed group center
    (or explicit identity), preserving target-domain intrinsic phase variation.
    """
    num_classes = int(source_prototype_bank.ready.numel())
    requested = None if sample_ids is None else set(int(item) for item in sample_ids)
    requested_count = hypothesis_result.num_samples if requested is None else len(requested)
    if phase_state.decision_status is PhaseDecisionStatus.UNCONFIRMED:
        return _empty_result(requested_count, num_classes)

    primary_by_sample: dict[int, CandidatePseudoLabel] = {}
    for candidate in hypothesis_result.candidate_pseudo_labels:
        sample_id = int(candidate.sample_id)
        if requested is not None and sample_id not in requested:
            continue
        if sample_id in primary_by_sample:
            raise ValueError(f"duplicate primary candidate pseudo-label for sample {sample_id}")
        primary_by_sample[sample_id] = candidate

    alignment_lookup = {
        (int(item.sample_id), int(item.class_id)): item
        for item in hypothesis_result.pairwise_alignments
    }
    confirmed_groups = {
        group.group_id: group
        for group in phase_state.groups
        if group.status is PhaseGroupStatus.CONFIRMED
    }
    ready_classes = tuple(
        int(i)
        for i in torch.nonzero(
            source_prototype_bank.ready.detach().cpu(), as_tuple=False
        ).flatten().tolist()
    )

    option_by_key: dict[tuple[int, int], CandidatePseudoLabel] = {}
    compatibility_by_key = {}
    rejected_candidates: list[StableTargetCandidate] = []
    view_spec_by_key: dict[tuple[int, int], tuple[int, Tensor, tuple[int, ...]]] = {}

    for sample_id, primary in primary_by_sample.items():
        options = [primary]
        if (
            primary.ambiguous
            and primary.secondary_class_id is not None
            and primary.secondary_q_shape_distance is not None
        ):
            options.append(
                replace(
                    primary,
                    class_id=int(primary.secondary_class_id),
                    q_shape_distance=float(primary.secondary_q_shape_distance),
                    phase_evidence_eligible=bool(
                        primary.secondary_phase_evidence_eligible
                        if primary.secondary_phase_evidence_eligible is not None
                        else False
                    ),
                    secondary_class_id=None,
                    secondary_q_shape_distance=None,
                    secondary_phase_evidence_eligible=None,
                )
            )

        for option in options:
            key = (sample_id, int(option.class_id))
            option_by_key[key] = option
            compatibility = evaluate_sample_class_phase_compatibility(
                sample_id=sample_id,
                class_id=int(option.class_id),
                alignment=alignment_lookup.get(key),
                state=phase_state,
                config=phase_config,
                # Estimating Domain Phase required stricter evidence.  Once the
                # domain-level Phase is fixed, numerically valid candidates may
                # be checked against it and then judged on the calibrated view.
                require_phase_evidence_eligible=False,
            )
            compatibility_by_key[key] = compatibility
            if compatibility.status is not CandidatePhaseCompatibilityStatus.COMPATIBLE:
                rejected_candidates.append(
                    _candidate_phase_rejection(
                        option,
                        phase_status=compatibility.status,
                        phase_distance=compatibility.phase_distance_to_group,
                    )
                )
                continue

            if phase_state.decision_status is PhaseDecisionStatus.IDENTITY_CONFIRMED:
                if compatibility.gamma is None:
                    rejected_candidates.append(
                        _candidate_phase_rejection(
                            option,
                            phase_status=CandidatePhaseCompatibilityStatus.UNUSABLE,
                            phase_distance=compatibility.phase_distance_to_group,
                        )
                    )
                    continue
                identity_gamma = torch.linspace(
                    0.0, 1.0, compatibility.gamma.numel(), dtype=torch.float64
                )
                view_spec_by_key[key] = (
                    IDENTITY_PHASE_GROUP_ID,
                    identity_gamma,
                    ready_classes,
                )
                continue

            group_id = compatibility.assigned_group_id
            group = None if group_id is None else confirmed_groups.get(int(group_id))
            if group is None or int(option.class_id) not in group.member_classes:
                rejected_candidates.append(
                    _candidate_phase_rejection(
                        option,
                        phase_status=CandidatePhaseCompatibilityStatus.NO_CONFIRMED_GROUP,
                        phase_distance=compatibility.phase_distance_to_group,
                    )
                )
                view_spec_by_key.pop(key, None)
                continue
            view_spec_by_key[key] = (
                group.group_id,
                group.center_gamma,
                group.member_classes,
            )

    teacher = ema_teacher.model()
    teacher.eval()
    view_lookup: dict[tuple[int, int], tuple[ConfirmedPhaseView, int]] = {}
    considered_samples: set[int] = set()
    candidate_view_count = 0
    fallback_index = 0

    for batch in target_loader:
        pixels = batch.get("pixels")
        if not isinstance(pixels, Tensor):
            raise ValueError("target batch must contain tensor pixels")
        batch_size = int(pixels.shape[0])
        batch_ids = batch.get("index")
        if not isinstance(batch_ids, Tensor) or batch_ids.shape != (batch_size,):
            batch_ids = torch.arange(
                fallback_index, fallback_index + batch_size, dtype=torch.long
            )
        batch_ids = batch_ids.detach().to(device="cpu", dtype=torch.long)
        fallback_index += batch_size
        if requested is None:
            considered_samples.update(int(item) for item in batch_ids.tolist())
        else:
            considered_samples.update(
                int(item) for item in batch_ids.tolist() if int(item) in requested
            )

        rows_by_group: dict[int, list[int]] = {}
        group_specs: dict[int, tuple[Tensor, tuple[int, ...]]] = {}
        for row in range(batch_size):
            sample_id = int(batch_ids[row].item())
            seen_groups: set[int] = set()
            for (candidate_sample, _class_id), spec in view_spec_by_key.items():
                if candidate_sample != sample_id:
                    continue
                group_id, center_gamma, member_classes = spec
                if group_id in seen_groups:
                    continue
                seen_groups.add(group_id)
                rows_by_group.setdefault(group_id, []).append(row)
                group_specs[group_id] = (center_gamma, member_classes)

        for group_id in sorted(rows_by_group):
            rows = rows_by_group[group_id]
            center_gamma, member_classes = group_specs[group_id]
            batch_sample_ids = torch.tensor(
                [int(batch_ids[row].item()) for row in rows], dtype=torch.long
            )
            view = build_phase_calibrated_view(
                model=teacher,
                batch=_subset_batch(batch, rows, batch_size),
                sample_ids=batch_sample_ids,
                group_id=group_id,
                member_classes=member_classes,
                center_gamma=center_gamma,
            )
            candidate_view_count += len(rows)
            for view_index, sample_id in enumerate(view.sample_ids.tolist()):
                view_lookup[(int(sample_id), group_id)] = (view, view_index)

    if requested is not None and considered_samples != requested:
        missing = sorted(requested - considered_samples)
        raise ValueError(
            "target statistics loader is missing requested sample ids: "
            + ",".join(str(item) for item in missing[:10])
        )

    candidates = list(rejected_candidates)
    accepted_by_sample: dict[int, list[tuple[StableTargetCandidate, ConfirmedPhaseView, int]]] = {}
    phase_compatible_samples: set[int] = set()
    for key in sorted(view_spec_by_key):
        sample_id, class_id = key
        group_id = view_spec_by_key[key][0]
        source = view_lookup.get((sample_id, group_id))
        if source is None:
            continue
        option = option_by_key[key]
        compatibility = compatibility_by_key[key]
        view, view_index = source
        evaluated = evaluate_stable_target_candidate(
            view=view,
            view_index=view_index,
            class_id=class_id,
            source_prototype_bank=source_prototype_bank,
            config=config,
        )
        evaluated = replace(
            evaluated,
            phase_compatible=True,
            phase_distance_to_group=compatibility.phase_distance_to_group,
            candidate_q_shape_distance=float(option.q_shape_distance),
            candidate_ambiguous=bool(option.ambiguous),
        )
        candidates.append(evaluated)
        phase_compatible_samples.add(sample_id)
        if evaluated.accepted:
            accepted_by_sample.setdefault(sample_id, []).append(
                (evaluated, view, view_index)
            )

    stable_labels: list[StableTargetLabel] = []
    class_counts = [0] * num_classes
    ambiguous_rejected = 0
    for sample_id in sorted(accepted_by_sample):
        accepted = accepted_by_sample[sample_id]
        if len(accepted) != 1:
            ambiguous_rejected += 1
            continue
        evaluated, view, view_index = accepted[0]
        stable_labels.append(
            StableTargetLabel(
                sample_id=evaluated.sample_id,
                class_id=evaluated.class_id,
                group_id=evaluated.group_id,
                aligned_q_shape=view.aligned_q_shape[view_index].detach(),
                aligned_q_support=view.aligned_q_support[view_index].detach(),
                fused_repr=view.fused_repr[view_index].detach(),
                confidence_summary=min(
                    evaluated.cls_confidence,
                    evaluated.fused_confidence,
                    evaluated.q_confidence,
                ),
            )
        )
        class_counts[evaluated.class_id] += 1

    candidates.sort(key=lambda item: (item.sample_id, item.class_id))
    return StableTargetLabelScanResult(
        candidates=tuple(candidates),
        stable_labels=tuple(stable_labels),
        num_samples=requested_count,
        num_without_confirmed_phase=requested_count - len(phase_compatible_samples),
        num_candidate_views=candidate_view_count,
        num_classifier_pass=sum(item.passed_classifier for item in candidates),
        num_fused_pass=sum(item.passed_fused for item in candidates),
        num_q_pass=sum(item.passed_q for item in candidates),
        num_stable_labels=len(stable_labels),
        num_ambiguous_rejected=ambiguous_rejected,
        stable_class_counts=tuple(class_counts),
        num_phase_compatible=len(phase_compatible_samples),
        num_phase_incompatible=max(0, requested_count - len(phase_compatible_samples)),
    )


@torch.no_grad()
def scan_stable_target_labels(
    *,
    ema_teacher: Stage2EMATeacher,
    target_loader,
    hypothesis_result: TargetHypothesisScanResult,
    phase_state: DomainPhaseState,
    source_prototype_bank: SourcePrototypeBank,
    config: StableLabelConfig,
) -> StableTargetLabelScanResult:
    class_to_group = build_confirmed_class_to_group_map(phase_state)
    num_classes = int(source_prototype_bank.ready.numel())
    if not class_to_group:
        return _empty_result(hypothesis_result.num_samples, num_classes)

    teacher = ema_teacher.model()
    teacher.eval()
    hypotheses_by_sample: dict[int, list] = {}
    for hypothesis in hypothesis_result.hypotheses:
        hypotheses_by_sample.setdefault(int(hypothesis.sample_id), []).append(hypothesis)

    view_lookup: dict[tuple[int, int], tuple[ConfirmedPhaseView, int]] = {}
    fallback_index = 0
    candidate_view_count = 0
    for batch in target_loader:
        pixels = batch.get("pixels")
        if not isinstance(pixels, Tensor):
            raise ValueError("target batch must contain tensor pixels")
        batch_size = int(pixels.shape[0])
        batch_ids = batch.get("index")
        if not isinstance(batch_ids, Tensor) or batch_ids.shape != (batch_size,):
            batch_ids = torch.arange(
                fallback_index, fallback_index + batch_size, dtype=torch.long
            )
        batch_ids = batch_ids.detach().to(device="cpu", dtype=torch.long)
        fallback_index += batch_size
        rows_by_group: dict[int, list[int]] = {}
        groups: dict[int, PhaseGroup] = {}
        for row in range(batch_size):
            sample_id = int(batch_ids[row].item())
            seen_groups: set[int] = set()
            for hypothesis in hypotheses_by_sample.get(sample_id, ()):
                group = class_to_group.get(int(hypothesis.class_id))
                if group is None or group.group_id in seen_groups:
                    continue
                seen_groups.add(group.group_id)
                rows_by_group.setdefault(group.group_id, []).append(row)
                groups[group.group_id] = group
        for group_id in sorted(rows_by_group):
            rows = rows_by_group[group_id]
            sample_ids = torch.tensor(
                [int(batch_ids[row].item()) for row in rows], dtype=torch.long
            )
            view = build_confirmed_phase_view(
                model=teacher,
                batch=_subset_batch(batch, rows, batch_size),
                sample_ids=sample_ids,
                group=groups[group_id],
            )
            candidate_view_count += len(rows)
            for view_index, sample_id in enumerate(view.sample_ids.tolist()):
                view_lookup[(int(sample_id), group_id)] = (view, view_index)

    candidates: list[StableTargetCandidate] = []
    candidate_sources: dict[tuple[int, int, int], tuple[ConfirmedPhaseView, int]] = {}
    samples_with_phase: set[int] = set()
    for hypothesis in hypothesis_result.hypotheses:
        sample_id = int(hypothesis.sample_id)
        class_id = int(hypothesis.class_id)
        group = class_to_group.get(class_id)
        if group is None:
            continue
        source = view_lookup.get((sample_id, group.group_id))
        if source is None:
            continue
        samples_with_phase.add(sample_id)
        view, view_index = source
        candidate = evaluate_stable_target_candidate(
            view=view,
            view_index=view_index,
            class_id=class_id,
            source_prototype_bank=source_prototype_bank,
            config=config,
        )
        candidates.append(candidate)
        candidate_sources[(sample_id, class_id, group.group_id)] = source

    accepted_by_sample: dict[int, list[StableTargetCandidate]] = {}
    for candidate in candidates:
        if candidate.accepted:
            accepted_by_sample.setdefault(candidate.sample_id, []).append(candidate)
    stable_labels: list[StableTargetLabel] = []
    class_counts = [0] * num_classes
    ambiguous_rejected = 0
    for sample_id in sorted(accepted_by_sample):
        accepted = accepted_by_sample[sample_id]
        if len(accepted) != 1:
            ambiguous_rejected += 1
            continue
        candidate = accepted[0]
        view, view_index = candidate_sources[
            (candidate.sample_id, candidate.class_id, candidate.group_id)
        ]
        stable_labels.append(
            StableTargetLabel(
                sample_id=candidate.sample_id,
                class_id=candidate.class_id,
                group_id=candidate.group_id,
                aligned_q_shape=view.aligned_q_shape[view_index].detach(),
                aligned_q_support=view.aligned_q_support[view_index].detach(),
                fused_repr=view.fused_repr[view_index].detach(),
                confidence_summary=min(
                    candidate.cls_confidence,
                    candidate.fused_confidence,
                    candidate.q_confidence,
                ),
            )
        )
        class_counts[candidate.class_id] += 1
    return StableTargetLabelScanResult(
        candidates=tuple(candidates),
        stable_labels=tuple(stable_labels),
        num_samples=hypothesis_result.num_samples,
        num_without_confirmed_phase=hypothesis_result.num_samples - len(samples_with_phase),
        num_candidate_views=candidate_view_count,
        num_classifier_pass=sum(candidate.passed_classifier for candidate in candidates),
        num_fused_pass=sum(candidate.passed_fused for candidate in candidates),
        num_q_pass=sum(candidate.passed_q for candidate in candidates),
        num_stable_labels=len(stable_labels),
        num_ambiguous_rejected=ambiguous_rejected,
        stable_class_counts=tuple(class_counts),
    )


@torch.no_grad()
def scan_stable_target_labels_from_confirmed_phase(
    *,
    ema_teacher: Stage2EMATeacher,
    target_loader,
    phase_state: DomainPhaseState,
    source_prototype_bank: SourcePrototypeBank,
    config: StableLabelConfig,
    sample_ids: tuple[int, ...] | None = None,
) -> StableTargetLabelScanResult:
    """Scan stable labels directly under confirmed domain-level Phase groups.

    Exact sample/class registration hypotheses are deliberately *not* required
    here.  They are evidence for estimating Domain Phase.  Once a group is
    confirmed, its center gamma defines the calibrated view; classifier, fused
    source prototypes and q-Shape prototypes then provide the three independent
    class gates.  This keeps Shape estimation from inheriting the cost and
    sample-level bias of individual DP registration.
    """
    confirmed_groups = tuple(
        group
        for group in phase_state.groups
        if group.status is PhaseGroupStatus.CONFIRMED
    )
    num_classes = int(source_prototype_bank.ready.numel())
    requested = None if sample_ids is None else set(int(item) for item in sample_ids)
    requested_count = None if requested is None else len(requested)
    if not confirmed_groups:
        return _empty_result(0 if requested_count is None else requested_count, num_classes)

    teacher = ema_teacher.model()
    teacher.eval()
    candidates: list[StableTargetCandidate] = []
    candidate_sources: dict[tuple[int, int, int], tuple[ConfirmedPhaseView, int]] = {}
    considered_samples: set[int] = set()
    candidate_view_count = 0
    fallback_index = 0

    for batch in target_loader:
        pixels = batch.get("pixels")
        if not isinstance(pixels, Tensor):
            raise ValueError("target batch must contain tensor pixels")
        batch_size = int(pixels.shape[0])
        batch_ids = batch.get("index")
        if not isinstance(batch_ids, Tensor) or batch_ids.shape != (batch_size,):
            batch_ids = torch.arange(
                fallback_index, fallback_index + batch_size, dtype=torch.long
            )
        batch_ids = batch_ids.detach().to(device="cpu", dtype=torch.long)
        fallback_index += batch_size
        rows = [
            row
            for row in range(batch_size)
            if requested is None or int(batch_ids[row].item()) in requested
        ]
        if not rows:
            continue
        row_sample_ids = torch.tensor(
            [int(batch_ids[row].item()) for row in rows], dtype=torch.long
        )
        considered_samples.update(int(item) for item in row_sample_ids.tolist())

        for group in confirmed_groups:
            view = build_confirmed_phase_view(
                model=teacher,
                batch=_subset_batch(batch, rows, batch_size),
                sample_ids=row_sample_ids,
                group=group,
            )
            candidate_view_count += len(rows)
            for view_index, sample_id_value in enumerate(view.sample_ids.tolist()):
                sample_id = int(sample_id_value)
                for class_id in group.member_classes:
                    candidate = evaluate_stable_target_candidate(
                        view=view,
                        view_index=view_index,
                        class_id=int(class_id),
                        source_prototype_bank=source_prototype_bank,
                        config=config,
                    )
                    candidates.append(candidate)
                    candidate_sources[(sample_id, int(class_id), group.group_id)] = (
                        view, view_index
                    )

    if requested is not None and considered_samples != requested:
        missing = sorted(requested - considered_samples)
        raise ValueError(
            "target statistics loader is missing requested sample ids: "
            + ",".join(str(item) for item in missing[:10])
        )

    accepted_by_sample: dict[int, list[StableTargetCandidate]] = {}
    for candidate in candidates:
        if candidate.accepted:
            accepted_by_sample.setdefault(candidate.sample_id, []).append(candidate)

    stable_labels: list[StableTargetLabel] = []
    class_counts = [0] * num_classes
    ambiguous_rejected = 0
    for sample_id in sorted(accepted_by_sample):
        accepted = accepted_by_sample[sample_id]
        if len(accepted) != 1:
            ambiguous_rejected += 1
            continue
        candidate = accepted[0]
        view, view_index = candidate_sources[
            (candidate.sample_id, candidate.class_id, candidate.group_id)
        ]
        stable_labels.append(
            StableTargetLabel(
                sample_id=candidate.sample_id,
                class_id=candidate.class_id,
                group_id=candidate.group_id,
                aligned_q_shape=view.aligned_q_shape[view_index].detach(),
                aligned_q_support=view.aligned_q_support[view_index].detach(),
                fused_repr=view.fused_repr[view_index].detach(),
                confidence_summary=min(
                    candidate.cls_confidence,
                    candidate.fused_confidence,
                    candidate.q_confidence,
                ),
            )
        )
        class_counts[candidate.class_id] += 1

    num_samples = len(considered_samples)
    return StableTargetLabelScanResult(
        candidates=tuple(candidates),
        stable_labels=tuple(stable_labels),
        num_samples=num_samples,
        num_without_confirmed_phase=0,
        num_candidate_views=candidate_view_count,
        num_classifier_pass=sum(candidate.passed_classifier for candidate in candidates),
        num_fused_pass=sum(candidate.passed_fused for candidate in candidates),
        num_q_pass=sum(candidate.passed_q for candidate in candidates),
        num_stable_labels=len(stable_labels),
        num_ambiguous_rejected=ambiguous_rejected,
        stable_class_counts=tuple(class_counts),
    )
