import importlib

import numpy as np
import pandas as pd
import pytest


def _module():
    return importlib.import_module("analysis.structure_da.domain_style_diagnostics")


def _peak_curve(grid, peak):
    return np.exp(-0.5 * ((np.asarray(grid) - peak) / 22.0) ** 2)


def test_canonical_interpolation_stays_inside_hull_and_masks_large_gaps():
    module = _module()
    grid = np.array([1.0, 20.0, 50.0, 80.0, 120.0])
    values, valid = module.interpolate_canonical_curve(
        np.array([10.0, 30.0, 100.0]), np.array([0.1, 0.3, 0.9]),
        np.ones(3, dtype=bool), grid, max_gap_days=40.0,
    )
    assert not valid[0]
    assert valid[1]
    assert not valid[2:4].any()
    assert not valid[4]
    assert np.isnan(values[~valid]).all()


def test_phase_shift_does_not_bridge_masked_canonical_gap():
    module = _module()
    values = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    aligned, valid = module._shift_curve(values, np.arange(5.0), 0.5)
    assert not valid[3]
    assert np.isnan(aligned[3])


def test_robust_location_reports_mean_median_iqr_and_rejects_under_three():
    module = _module()
    curves = np.array([[1.0, 1.0], [1.1, 2.0], [0.9, 3.0], [20.0, np.nan]])
    masks = np.isfinite(curves)
    result = module.robust_pointwise_location(curves, masks)
    assert result["n_valid"].tolist() == [4, 3]
    assert result["robust"][0] < result["mean"][0]
    assert np.isfinite(result["median"]).all()
    sparse = module.robust_pointwise_location(curves[:2], masks[:2])
    assert np.isnan(sparse["robust"]).all()


def test_target_peak_thirty_days_later_recovers_negative_shift():
    module = _module()
    grid = np.linspace(1.0, 365.0, 365)
    result = module.estimate_phase_shift(
        _peak_curve(grid, 150.0), _peak_curve(grid, 180.0), grid,
        peak_search_start=45.0, peak_search_end=330.0,
        min_peak_prominence_ratio=0.15, max_shift_days=90.0,
        shift_refine_radius_days=14.0, min_common_support=0.65,
    )
    assert result["valid"]
    assert abs(result["shift_days"] + 30.0) <= 1.0
    assert result["phase_available"]
    assert result["phase_status"] == "valid_nonidentity"
    assert result["search_mode"] == "peak_guided"
    assert 0.0 in result["candidate_shifts"]


def test_unreliable_peak_uses_full_trend_search_and_recovers_shift():
    module = _module()
    grid = np.linspace(1.0, 365.0, 365)
    result = module.estimate_phase_shift(
        _peak_curve(grid, 60.0), _peak_curve(grid, 90.0), grid,
        peak_search_start=45.0, peak_search_end=330.0,
        min_peak_prominence_ratio=0.15, max_shift_days=90.0,
        shift_refine_radius_days=14.0, min_common_support=0.65,
    )
    assert not result["source_peak"]["valid"]
    assert result["phase_available"]
    assert result["search_mode"] == "full_trend_search"
    assert result["phase_status"] == "valid_nonidentity"
    assert abs(result["shift_days"] + 30.0) <= 1.0


def test_peak_guided_search_clamps_out_of_range_initial_shift_to_boundary():
    module = _module()
    grid = np.linspace(1.0, 365.0, 365)
    result = module.estimate_phase_shift(
        _peak_curve(grid, 220.0), _peak_curve(grid, 100.0), grid,
        max_shift_days=90.0, shift_refine_radius_days=14.0,
        min_common_support=0.65,
    )
    assert result["phase_available"]
    assert result["search_mode"] == "peak_guided"
    assert result["phase_status"] == "valid_nonidentity"
    assert np.isclose(result["shift_days"], 90.0)
    assert result["shift_at_boundary"]


def test_identity_is_selected_when_relative_gain_is_below_threshold():
    module = _module()
    grid = np.linspace(1.0, 365.0, 128)
    curve = _peak_curve(grid, 160.0)
    result = module.estimate_phase_shift(curve, curve.copy(), grid)
    assert result["phase_available"]
    assert result["phase_status"] == "valid_identity"
    assert result["shift_days"] == 0.0
    assert result["relative_gain"] < 0.02


def test_invalid_canonical_grid_has_explicit_unavailable_status():
    module = _module()
    grid = np.array([1.0, 10.0, 8.0, 20.0])
    source = np.array([0.1, 0.6, 0.4, 0.2])
    result = module.estimate_phase_shift(source, source.copy(), grid)
    assert not result["phase_available"]
    assert result["phase_status"] == "invalid_grid"
    assert np.isnan(result["common_support"])


def test_phase_refinement_uses_actual_canonical_grid_step():
    module = _module()
    grid = np.linspace(1.0, 365.0, 128)
    source = _peak_curve(grid, 150.0)
    target = _peak_curve(grid, 177.0)
    result = module.estimate_phase_shift(source, target, grid)
    assert result["valid"]
    step = grid[1] - grid[0]
    offset_steps = (result["shift_days"] - result["initial_shift_days"]) / step
    assert np.isclose(offset_steps, round(offset_steps), atol=1e-10)


def test_vertical_offset_does_not_change_phase_shift():
    module = _module()
    grid = np.linspace(1.0, 365.0, 365)
    source = _peak_curve(grid, 140.0)
    target = _peak_curve(grid, 165.0) + 0.4
    result = module.estimate_phase_shift(source, target, grid)
    assert result["valid"]
    assert abs(result["shift_days"] + 25.0) <= 1.0


def test_flat_trend_is_phase_invalid():
    module = _module()
    grid = np.linspace(1.0, 365.0, 128)
    result = module.detect_trend_peak(np.full(128, 0.4), grid)
    assert not result["valid"]
    assert result["reason"] == "insufficient_dynamic_range"
    phase = module.estimate_phase_shift(
        np.full(128, 0.4), np.full(128, 0.4), grid,
    )
    assert not phase["phase_available"]
    assert phase["phase_status"] == "invalid_flat_trend"
    assert phase["shift_days"] == 0.0
    assert np.isnan(phase["common_support"])


def test_invalid_trend_never_falls_back_to_structure_peak():
    module = _module()
    grid = np.linspace(1.0, 365.0, 128)
    result = module.estimate_class_phase(
        np.full(128, 0.4), np.full(128, 0.4),
        _peak_curve(grid, 140.0), _peak_curve(grid, 170.0), grid,
    )
    assert not result["phase_available"]
    assert result["shift_days"] == 0.0
    assert result["structure_source_peak_valid"]
    assert result["structure_target_peak_valid"]


def test_stable_seed_is_process_independent_and_class_specific():
    module = _module()
    first = module.stable_class_seed("DK1", "FR2", "corn", 1)
    assert first == module.stable_class_seed("DK1", "FR2", "corn", 1)
    assert first != module.stable_class_seed("DK1", "FR2", "wheat", 1)


def test_bootstrap_phase_is_numerically_deterministic_for_fixed_seed():
    module = _module()
    grid = np.linspace(1.0, 365.0, 128)

    def records(domain, peak):
        items = []
        for index, offset in enumerate((-0.01, 0.0, 0.01, 0.02)):
            curve = _peak_curve(grid, peak) + offset
            valid = np.ones(grid.shape, dtype=bool)
            items.append(module.CanonicalParcelRecord(
                domain, "corn", index, grid.copy(), curve.copy(), curve.copy(),
                curve.copy(), valid.copy(), valid.copy(), valid.copy(),
                grid.copy(), valid.copy(),
            ))
        return items

    config = module.DomainStyleConfig(
        "DK1", "FR2", bootstrap_repeats=8, min_class_samples=3,
    )
    first = module.bootstrap_phase_discrepancy(
        records("DK1", 150.0), records("FR2", 180.0), config, "corn"
    )
    second = module.bootstrap_phase_discrepancy(
        records("DK1", 150.0), records("FR2", 180.0), config, "corn"
    )
    assert first["shift_median"] == second["shift_median"]
    assert np.allclose(first["deltas"], second["deltas"], equal_nan=True)


def test_loco_style_excludes_held_out_class():
    module = _module()
    classes = [f"c{i}" for i in range(6)]
    deltas = {name: np.full(8, float(index)) for index, name in enumerate(classes)}
    masks = {name: np.ones(8, dtype=bool) for name in classes}
    reliability = {name: 1.0 for name in classes}
    loco = module.compute_loco_domain_styles(
        classes, classes, deltas, masks, reliability
    )
    assert loco["c5"]["valid"]
    assert "c5" not in loco["c5"]["classes_used"]
    assert np.all(loco["c5"]["style"] < 5.0)


def test_loco_requires_five_remaining_classes():
    module = _module()
    classes = [f"c{i}" for i in range(5)]
    values = {name: np.ones(4) for name in classes}
    masks = {name: np.ones(4, bool) for name in classes}
    result = module.compute_loco_domain_styles(
        classes, classes, values, masks, {name: 1.0 for name in classes}
    )
    assert not result["c0"]["valid"]
    assert result["c0"]["reason"] == "insufficient_loco_classes"


def test_loco_evaluates_noncontributor_without_leaking_held_class():
    module = _module()
    evaluation = [f"c{i}" for i in range(6)]
    contributors = evaluation[:5]
    deltas = {name: np.full(8, float(index)) for index, name in enumerate(evaluation)}
    masks = {name: np.ones(8, dtype=bool) for name in evaluation}
    result = module.compute_loco_domain_styles(
        evaluation, contributors, deltas, masks,
        {name: 1.0 for name in contributors},
    )
    assert result["c5"]["valid"]
    assert "c5" not in result["c5"]["classes_used"]
    assert result["c5"]["classes_used"] == contributors


def test_consensus_irls_downweights_outlying_class():
    module = _module()
    classes = [f"c{i}" for i in range(6)]
    deltas = {name: np.zeros(12) for name in classes}
    deltas["c5"] = np.full(12, 20.0)
    result = module.fit_robust_domain_style(
        classes, deltas, {name: np.ones(12, bool) for name in classes},
        {name: 1.0 for name in classes},
    )
    assert result["valid"]
    assert result["consensus"]["c5"] < result["consensus"]["c0"]
    assert np.nanmax(np.abs(result["style"])) < 5.0


def test_conflicting_discrepancies_do_not_claim_universal_improvement():
    module = _module()
    classes = [f"c{i}" for i in range(6)]
    deltas = {name: np.full(10, 1.0 if i < 3 else -1.0) for i, name in enumerate(classes)}
    masks = {name: np.ones(10, bool) for name in classes}
    result = module.fit_robust_domain_style(classes, deltas, masks, {name: 1.0 for name in classes})
    explained = [module.style_explained_fraction(deltas[name], result["style"], masks[name]) for name in classes]
    assert min(explained) <= 1e-8


def test_same_style_on_h_t_s_preserves_d_and_r_without_clipping():
    module = _module()
    components = {
        "original": np.array([1.2, -1.1, 0.5]),
        "trend": np.array([0.8, -0.4, 0.2]),
        "structure": np.array([1.0, -0.7, 0.3]),
    }
    style = np.array([0.3, -0.2, 0.1])
    styled = module.apply_shared_style(components, style, style_lambda=1.0)
    assert np.allclose(styled["structure"] - styled["trend"], components["structure"] - components["trend"])
    assert np.allclose(styled["original"] - styled["structure"], components["original"] - components["structure"])
    assert styled["original"][0] > 1.0
    assert module.physical_violation_fraction(styled["original"]) > 0.0
    assert module.hierarchy_max_errors(components, styled)["max_dynamics_error"] < 1e-10


def test_style_lambda_zero_is_exact_identity_even_when_style_has_nan():
    module = _module()
    components = {
        "original": np.array([0.2, 0.4, 0.6]),
        "trend": np.array([0.1, 0.3, 0.5]),
        "structure": np.array([0.15, 0.35, 0.55]),
    }
    styled = module.apply_shared_style(
        components, np.array([np.nan, 0.5, -0.2]), 0.0
    )
    for name in components:
        assert np.array_equal(styled[name], components[name])
    assert max(module.hierarchy_max_errors(components, styled).values()) == 0.0


def test_shared_style_lambda_one_reduces_matching_t_and_s_discrepancy():
    module = _module()
    source = {"original": np.zeros(5), "trend": np.zeros(5), "structure": np.ones(5)}
    style = np.full(5, 0.5)
    styled = module.apply_shared_style(source, style, 1.0)
    target_t = np.full(5, 0.5)
    target_s = np.full(5, 1.5)
    assert module.support_weighted_rmse(styled["trend"], target_t) == 0.0
    assert module.support_weighted_rmse(styled["structure"], target_s) == 0.0


def test_energy_distance_is_finite_with_partial_support():
    module = _module()
    source = np.array([[0.0, 1.0, np.nan], [0.2, 1.1, 0.4]])
    target = np.array([[0.1, 0.9, 0.5], [0.0, np.nan, 0.6]])
    value = module.support_weighted_energy_distance(
        source, np.isfinite(source), target, np.isfinite(target), min_common_support=0.5
    )
    assert np.isfinite(value)
    assert value >= 0.0


def test_collector_default_domains_are_unchanged_and_domain_filter_is_ordered(tmp_path):
    from analysis.structure_da import raw_timeseries as raw

    calls = []

    class FakeDataset:
        metadata = {"dates": [20170101, 20170201, 20170301]}
        def __len__(self): return 1
        def __getitem__(self, index):
            pixels = np.zeros((3, 10, 2), dtype=np.float32)
            pixels[:, 2, :] = 10000
            pixels[:, 3, :] = 20000
            return {"pixels": pixels, "label": 0}

    def factory(root, name, classes):
        calls.append(name)
        return FakeDataset()

    raw.collect_ndvi_diagnostic_parcels(
        tmp_path, 1, 1, ("corn",), factory
    )
    assert calls == list(raw.DOMAIN_DATASETS.values())
    calls.clear()
    frame, sampled, common = raw.collect_ndvi_diagnostic_parcels(
        tmp_path, 1, 1, ("corn",), factory, domains=("FR2", "DK1", "FR2")
    )
    assert calls == [raw.DOMAIN_DATASETS["FR2"], raw.DOMAIN_DATASETS["DK1"]]
    assert list(dict.fromkeys(frame["domain"])) == ["DK1", "FR2"] or set(frame["domain"]) == {"FR2", "DK1"}
    assert {row["domain"] for row in sampled} == {"FR2", "DK1"}
    assert common == ["corn"]


@pytest.mark.parametrize("domains", [("DK1",), ("DK1", "BAD")])
def test_collector_rejects_invalid_domain_selection(tmp_path, domains):
    from analysis.structure_da import raw_timeseries as raw
    with pytest.raises(ValueError):
        raw.collect_ndvi_diagnostic_parcels(
            tmp_path, 1, 1, ("corn",), lambda *args: object(), domains=domains
        )


def test_domain_style_cli_defaults_and_oracle_command():
    from scripts.analyze_structure_da import build_parser
    args = build_parser().parse_args([
        "ndvi-domain-style", "--data-root", "data", "--output-dir", "out",
        "--source-domain", "DK1", "--target-domain", "FR2",
    ])
    assert args.command == "ndvi-domain-style"
    assert args.samples_per_group == 100
    assert args.bootstrap_repeats == 200
    assert args.canonical_grid_size == 128
    assert args.style_lambdas == [0.0, 0.5, 1.0, 1.5]


def test_domain_style_config_always_adds_lambda_zero():
    module = _module()
    config = module.DomainStyleConfig(
        "DK1", "FR2", style_lambdas=(1.0, 0.5, 1.0)
    )
    assert config.style_lambdas == (0.0, 0.5, 1.0)


def test_phase_plot_target_label_distinguishes_alignment_from_identity_fallback():
    module = _module()
    assert module._phase_target_label("FR2", True) == "FR2 T-phase aligned"
    assert "identity fallback" in module._phase_target_label("FR2", False)
    assert "not aligned" in module._phase_target_label("FR2", False)


def test_domain_style_cli_rejects_same_domain_and_invalid_ratios():
    from scripts.analyze_structure_da import build_parser, validate_domain_style_args
    parser = build_parser()
    args = parser.parse_args([
        "ndvi-domain-style", "--data-root", "data", "--output-dir", "out",
        "--source-domain", "DK1", "--target-domain", "DK1",
    ])
    with pytest.raises(ValueError, match="different"):
        validate_domain_style_args(args)
    args.target_domain = "FR2"
    args.min_common_support = 1.1
    with pytest.raises(ValueError, match="min_common_support"):
        validate_domain_style_args(args)


def test_canonicalize_decomposes_each_parcel_exactly_once():
    module = _module()
    calls = []
    def decomposition(values, doys, valid):
        calls.append(1)
        values = np.asarray(values, dtype=float)
        return {"doys": np.asarray(doys, float), "valid": np.asarray(valid, bool),
                "original": values, "trend": values * 0.8,
                "structure": values * 0.9}
    parcel = {"domain": "DK1", "class_name": "corn", "parcel_index": 7,
              "doys": np.array([10., 40., 70.]), "ndvi": np.array([.1, .8, .2]),
              "valid": np.ones(3, bool)}
    record = module.canonicalize_parcel(parcel, np.linspace(10, 70, 7), 40, decomposition)
    assert len(calls) == 1
    assert isinstance(record, module.CanonicalParcelRecord)
    assert record.original_h.shape == (7,)


def test_oracle_runner_writes_declared_tables_and_figures_on_synthetic_data(tmp_path):
    module = _module()
    classes = tuple(f"crop_{index}" for index in range(6))
    dates = [20170110, 20170210, 20170310, 20170410, 20170510, 20170610,
             20170710, 20170810, 20170910, 20171010, 20171110, 20171210]

    class FakeDataset:
        metadata = {"dates": dates}
        def __init__(self, target, class_count):
            self.target = target
            self.class_count = class_count
        def __len__(self): return self.class_count * 4
        def __getitem__(self, index):
            label, repeat = divmod(index, 4)
            doys = np.linspace(10, 344, len(dates))
            peak = 145 + 4 * label + (25 if self.target else 0)
            ndvi = 0.08 + 0.70 * np.exp(-0.5 * ((doys - peak) / 42) ** 2) + repeat * 0.002
            red = np.full((len(dates), 2), 10000.0)
            nir = red * (1 + ndvi[:, None]) / (1 - ndvi[:, None])
            pixels = np.zeros((len(dates), 10, 2), dtype=np.float32)
            pixels[:, 2, :] = red; pixels[:, 3, :] = nir
            return {"pixels": pixels, "label": label}

    def factory(root, name, labels):
        return FakeDataset("30TXT" in name, len(labels))

    config = module.DomainStyleConfig(
        "DK1", "FR2", samples_per_group=4, bootstrap_repeats=3,
        canonical_grid_size=48, min_class_samples=3,
        min_common_support=0.5, min_bootstrap_valid_rate=0.0,
        max_interpolation_gap_days=40, style_lambdas=(1.0,),
    )
    stale = (
        tmp_path / "out/figures/raw_timeseries/domain_style_oracle/"
        "DK1_to_FR2/per_class/stale"
    )
    stale.mkdir(parents=True)
    (stale / "old.png").write_bytes(b"old")
    result = module.run_ndvi_domain_style_diagnostic(
        tmp_path, tmp_path / "out", config, classes=classes, dataset_factory=factory
    )
    table_dir = tmp_path / "out/tables/domain_style_oracle/DK1_to_FR2"
    figure_dir = tmp_path / "out/figures/raw_timeseries/domain_style_oracle/DK1_to_FR2"
    for filename in (
        "phase_alignment.csv", "class_domain_discrepancy_long.csv",
        "class_style_weights.csv", "domain_style_curves.csv",
        "loco_domain_style_curves.csv", "style_compensation_metrics.csv",
        "task_style_summary.csv", "manifest.json",
    ):
        assert (table_dir / filename).is_file()
    assert (figure_dir / "task_summary/phase_diagnostics.png").is_file()
    assert (figure_dir / "01_phase_aligned_before_style/crop_0.png").is_file()
    assert (figure_dir / "02_style_compensation_lambda_sweep/crop_0.png").is_file()
    assert (figure_dir / "03_class_discrepancy_vs_style/crop_0.png").is_file()
    assert not (figure_dir / "per_class").exists()
    assert result["manifest"]["oracle_target_labels"] is True
    assert result["manifest"]["not_for_training"] is True
    assert result["manifest"]["deployable_uda"] is False
    assert result["eligible_classes"] == list(classes)
    assert result["style_contributor_classes"] == list(classes)
    phase = pd.read_csv(table_dir / "phase_alignment.csv")
    assert {"source_domain", "target_domain", "source_t_peak_doy",
            "source_t_peak_prominence_ratio", "common_support_fraction",
            "shift_median_days", "shift_mad_days", "shift_q25_days",
            "shift_q75_days", "phase_available", "phase_status",
            "search_mode", "identity_objective", "best_objective",
            "relative_gain", "second_best_objective", "objective_margin",
            "shift_at_boundary", "bootstrap_search_success_rate",
            "bootstrap_identity_rate", "bootstrap_nonidentity_rate",
            "bootstrap_relative_gain_median",
            "bootstrap_objective_margin_median", "style_contributor",
            "style_evaluable"} <= set(phase.columns)
    assert phase["style_contributor"].all()
    assert phase["style_evaluable"].all()
    metrics = pd.read_csv(table_dir / "style_compensation_metrics.csv")
    assert {"lambda", "center_rmse_relative_change", "energy_relative_change",
            "nearest_class_margin_before", "nearest_class_margin_after",
            "physical_violation_fraction", "hierarchy_error",
            "invalid_reason"} <= set(metrics.columns)
    baseline = metrics[metrics["lambda"] == 0.0]
    assert not baseline.empty
    assert np.array_equal(
        baseline["center_rmse_before"].to_numpy(),
        baseline["center_rmse_after"].to_numpy(), equal_nan=True,
    )
    assert np.array_equal(
        baseline["iqr_distance_before"].to_numpy(),
        baseline["iqr_distance_after"].to_numpy(), equal_nan=True,
    )
    assert np.array_equal(
        baseline["energy_distance_before"].to_numpy(),
        baseline["energy_distance_after"].to_numpy(), equal_nan=True,
    )
    assert (baseline["hierarchy_error"] == 0.0).all()
    assert result["manifest"]["phase_search"]["peak_is_hard_requirement"] is False
    assert result["manifest"]["figure_layout"] == "figure_type_then_class"
    for filename in (
        "class_trend_discrepancies_and_style.png", "class_style_weights.png",
        "discrepancy_cosine_matrix.png", "lambda_summary.png",
        "phase_diagnostics.png",
    ):
        assert (figure_dir / "task_summary" / filename).is_file()

    ineligible_config = module.DomainStyleConfig(
        "DK1", "FR2", samples_per_group=4, bootstrap_repeats=1,
        canonical_grid_size=48, min_class_samples=5,
        min_common_support=0.5, min_bootstrap_valid_rate=0.0,
        max_interpolation_gap_days=40, style_lambdas=(1.0,),
    )
    ineligible_result = module.run_ndvi_domain_style_diagnostic(
        tmp_path, tmp_path / "ineligible", ineligible_config,
        classes=classes, dataset_factory=factory,
    )
    assert ineligible_result["eligible_classes"] == []
    assert (
        tmp_path / "ineligible/figures/raw_timeseries/domain_style_oracle/"
        "DK1_to_FR2/task_summary/discrepancy_cosine_matrix.png"
    ).is_file()

    five_class_result = module.run_ndvi_domain_style_diagnostic(
        tmp_path, tmp_path / "five_classes",
        module.DomainStyleConfig(
            "DK1", "FR2", samples_per_group=4, bootstrap_repeats=1,
            canonical_grid_size=48, min_class_samples=3,
            min_common_support=0.5, min_bootstrap_valid_rate=0.0,
            max_interpolation_gap_days=40, style_lambdas=(1.0,),
        ),
        classes=classes[:5], dataset_factory=factory,
    )
    assert five_class_result["style_contributor_classes"] == list(classes[:5])
    five_phase = pd.read_csv(
        tmp_path / "five_classes/tables/domain_style_oracle/DK1_to_FR2/"
        "phase_alignment.csv"
    )
    assert five_phase["phase_available"].all()
    assert not five_phase["style_evaluable"].any()
    five_metrics = pd.read_csv(
        tmp_path / "five_classes/tables/domain_style_oracle/DK1_to_FR2/"
        "style_compensation_metrics.csv"
    )
    five_baseline = five_metrics[five_metrics["lambda"] == 0.0]
    assert not five_baseline.empty
    assert np.isfinite(five_baseline["center_rmse_before"]).all()
    assert np.array_equal(
        five_baseline["center_rmse_before"].to_numpy(),
        five_baseline["center_rmse_after"].to_numpy(), equal_nan=True,
    )
    assert np.array_equal(
        five_baseline["iqr_distance_before"].to_numpy(),
        five_baseline["iqr_distance_after"].to_numpy(), equal_nan=True,
    )
    assert np.array_equal(
        five_baseline["energy_distance_before"].to_numpy(),
        five_baseline["energy_distance_after"].to_numpy(), equal_nan=True,
    )
    assert (five_baseline["hierarchy_error"] == 0.0).all()
