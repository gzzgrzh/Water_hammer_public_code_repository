import numpy as np
import pytest
import torch

from water16_reproduction.parametric_fssi_hybrid import (
    DEFAULT_CONFIG,
    deterministic_indices,
    event_aligned_time_indices,
    anchor_loss,
    AnchorBatch,
    balanced_event_aligned_time_indices,
    characteristic_event_candidates,
    response_peak_enriched_time_indices,
    load_config,
    nested_maximin_case_subset,
    nested_time_subset,
)
from water16_reproduction.parametric_fssi_forward import load_config as load_design_config
from water16_reproduction.parametric_fssi_reference import case_parameters, load_baseline
from water16_reproduction.parametric_fssi_model import wave_speed_magnitudes


def test_hybrid_is_registered_only_after_physics_only_failure() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config["status"] == "pilot_registered_after_physics_only_failure"
    assert "maximum P NRMSE 0.0871" in config["registration_reason"]


def test_anchor_policy_excludes_validation_and_test_data() -> None:
    config = load_config(DEFAULT_CONFIG)
    policy = config["anchor_policy"]
    assert policy["allowed_split"] == "train"
    assert policy["validation_and_test_labels_forbidden"] is True
    assert policy["total_anchor_vectors"] == 12 * 8 * 48
    assert policy["total_scalar_labels"] == 4 * policy["total_anchor_vectors"]


def test_deterministic_indices_include_grid_endpoints() -> None:
    indices = deterministic_indices(101, 8)
    assert len(indices) == 8
    assert indices[0] == 0
    assert indices[-1] == 100
    np.testing.assert_array_equal(indices, deterministic_indices(101, 8))


def test_deterministic_indices_reject_invalid_count() -> None:
    with pytest.raises(ValueError):
        deterministic_indices(10, 11)


def test_registered_anchor_coordinates_match_the_actual_index_rule() -> None:
    config = load_config(DEFAULT_CONFIG)
    policy = config["anchor_policy"]
    x_indices = deterministic_indices(
        policy["saved_reference_grid_per_case"]["spatial_points"],
        policy["spatial_points_per_case"],
    )
    t_indices = deterministic_indices(
        policy["saved_reference_grid_per_case"]["time_points"],
        policy["time_points_per_case"],
    )
    np.testing.assert_array_equal(x_indices, policy["spatial_indices"])
    np.testing.assert_array_equal(t_indices, policy["time_indices"])
    np.testing.assert_allclose(x_indices / 160.0, policy["spatial_positions_over_L"])
    np.testing.assert_allclose(t_indices * 0.001, policy["time_positions_s"])


def test_event_aligned_indices_are_fixed_count_deterministic_and_event_focused() -> None:
    design_path = DEFAULT_CONFIG.parents[0] / "parametric_fssi_forward_v1.json"
    design = load_design_config(design_path)
    case = design["case_design"]["pilot"]["cases"][0]
    params, _ = case_parameters(load_baseline(design), case, 0.8)
    times = np.linspace(0.0, 0.8, 801)
    first = event_aligned_time_indices(times, 50.0, params, 48)
    second = event_aligned_time_indices(times, 50.0, params, 48)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 48
    assert first[0] == 0
    assert first[-1] == 800
    closure_index = int(np.rint(params.valve_close_time_s / 0.001))
    assert min(abs(first - closure_index)) <= 1


def test_all_formal_cases_have_two_paired_wave_speed_magnitudes() -> None:
    design_path = DEFAULT_CONFIG.parents[0] / "parametric_fssi_forward_v1.json"
    design = load_design_config(design_path)
    baseline = load_baseline(design)
    from water16_reproduction.parametric_fssi_forward import materialize_cases

    for case in materialize_cases(design, "formal"):
        params, _ = case_parameters(baseline, case, 0.8)
        speeds = wave_speed_magnitudes(params)
        assert speeds.shape == (2,)
        assert 0.0 < speeds[0] < speeds[1]


def test_tail_anchor_objective_penalizes_concentrated_large_errors(monkeypatch) -> None:
    prediction = torch.zeros((10, 4), dtype=torch.float64)
    prediction[-1] = 5.0
    monkeypatch.setattr(
        "water16_reproduction.parametric_fssi_hybrid.characteristic_state",
        lambda model, positions, times, case: prediction,
    )
    batch = AnchorBatch(
        case={},
        positions=torch.zeros((10, 1), dtype=torch.float64),
        times=torch.zeros((10, 1), dtype=torch.float64),
        targets=torch.zeros((10, 4), dtype=torch.float64),
        scales=torch.ones(4, dtype=torch.float64),
        observation_weights=torch.ones(4, dtype=torch.float64),
        source_file=DEFAULT_CONFIG,
    )
    mean_only = anchor_loss(None, batch)
    with_tail = anchor_loss(None, batch, tail_fraction=0.1, tail_weight=1.0)
    np.testing.assert_allclose(float(mean_only), 2.5)
    np.testing.assert_allclose(float(with_tail), 27.5)


def test_balanced_event_indices_cover_both_wave_families_and_full_time_window() -> None:
    design_path = DEFAULT_CONFIG.parents[0] / "parametric_fssi_forward_v1.json"
    design = load_design_config(design_path)
    case = design["case_design"]["pilot"]["cases"][0]
    params, _ = case_parameters(load_baseline(design), case, 0.8)
    times = np.linspace(0.0, 0.8, 801)
    selected = balanced_event_aligned_time_indices(times, 50.0, params, 48)
    assert len(selected) == 48
    assert selected[0] == 0 and selected[-1] == 800
    assert np.all(np.diff(selected) > 0)
    speeds = wave_speed_magnitudes(params)
    family_sets = [
        set(characteristic_event_candidates(times, 50.0, params, float(speed)))
        for speed in speeds
    ]
    overlap_counts = [len(set(selected) & family) for family in family_sets]
    assert min(overlap_counts) >= 15
    assert np.count_nonzero(selected < 200) >= 5
    assert np.count_nonzero(selected > 600) >= 5


def test_peak_enrichment_keeps_physics_events_and_adds_response_peak() -> None:
    design_path = DEFAULT_CONFIG.parents[0] / "parametric_fssi_forward_v1.json"
    design = load_design_config(design_path)
    case = design["case_design"]["pilot"]["cases"][0]
    params, _ = case_parameters(load_baseline(design), case, 0.8)
    times = np.linspace(0.0, 0.8, 801)
    traces = np.zeros((801, 4))
    traces[613, 3] = 10.0
    traces[287, 2] = 8.0
    physics = balanced_event_aligned_time_indices(times, 50.0, params, 32)
    selected = response_peak_enriched_time_indices(
        times, 50.0, params, traces, np.zeros(4), 48, 32
    )
    assert len(selected) == 48
    assert set(physics).issubset(set(selected))
    assert 613 in selected
    assert 287 in selected


def test_nested_physics_event_designs_are_strictly_nested() -> None:
    design_path = DEFAULT_CONFIG.parents[0] / "parametric_fssi_forward_v1.json"
    design = load_design_config(design_path)
    case = design["case_design"]["pilot"]["cases"][0]
    params, _ = case_parameters(load_baseline(design), case, 0.8)
    times = np.linspace(0.0, 0.8, 801)
    candidates = balanced_event_aligned_time_indices(times, 50.0, params, 48)
    design_12 = nested_time_subset(candidates, 12)
    design_24 = nested_time_subset(candidates, 24)
    assert set(design_12).issubset(set(design_24))
    assert set(design_24).issubset(set(candidates))


def test_nested_maximin_training_case_design() -> None:
    design_path = DEFAULT_CONFIG.parents[0] / "parametric_fssi_forward_v1.json"
    design = load_design_config(design_path)
    from water16_reproduction.parametric_fssi_forward import materialize_cases

    cases = [case for case in materialize_cases(design, "formal") if case["split"] == "train"]
    subset_18 = nested_maximin_case_subset(cases, 18, design)
    subset_36 = nested_maximin_case_subset(cases, 36, design)
    subset_72 = nested_maximin_case_subset(cases, 72, design)
    ids_18 = {case["case_id"] for case in subset_18}
    ids_36 = {case["case_id"] for case in subset_36}
    ids_72 = {case["case_id"] for case in subset_72}
    assert len(ids_18) == 18 and len(ids_36) == 36 and len(ids_72) == 72
    assert ids_18.issubset(ids_36)
    assert ids_36.issubset(ids_72)
