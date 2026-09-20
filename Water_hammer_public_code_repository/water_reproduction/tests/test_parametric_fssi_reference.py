import copy

import numpy as np
import pytest

from water16_reproduction.parametric_fssi_forward import DEFAULT_CONFIG, load_config
from water16_reproduction.parametric_fssi_reference import (
    audit_reference_dataset,
    case_parameters,
    downsample_truth,
    engineering_metrics,
    generate_reference_dataset,
    load_baseline,
    normalized_field_rmse,
)


def test_case_parameters_apply_all_three_registered_coordinates() -> None:
    config = load_config(DEFAULT_CONFIG)
    baseline = load_baseline(config)
    case = config["case_design"]["pilot"]["cases"][0]
    params, fluid_speed = case_parameters(baseline, case, 0.2)
    assert params.soil_E_pa / params.pipe_E_pa == pytest.approx(
        case["soil_to_pipe_modulus_ratio"]
    )
    assert params.initial_velocity_m_s == pytest.approx(case["initial_velocity_m_s"])
    assert params.valve_close_time_s == pytest.approx(
        case["closure_time_over_L_cf"] * params.length_m / fluid_speed
    )
    assert params.head_difference_m == pytest.approx(20.0)


def test_engineering_metrics_identify_pressure_and_stress_locations() -> None:
    config = load_config(DEFAULT_CONFIG)
    params, _ = case_parameters(
        load_baseline(config), config["case_design"]["pilot"]["cases"][0], 0.2
    )
    pressure0 = params.water_density_kg_m3 * params.gravity_m_s2 * params.head_difference_m
    area_f = np.pi * params.inner_radius_m**2
    area_t = np.pi * (
        (params.inner_radius_m + params.wall_thickness_m) ** 2
        - params.inner_radius_m**2
    )
    stress0 = area_f * pressure0 / area_t
    truth = {
        "z": np.asarray([0.0, params.length_m]),
        "t": np.asarray([0.0, 0.1, 0.2]),
        "V": np.zeros((2, 3)),
        "uz": np.zeros((2, 3)),
        "P": pressure0 + np.asarray([[0.0, 2.0, -1.0], [0.0, 5.0, -3.0]]),
        "sigma_z": stress0 + np.asarray([[0.0, -7.0, 1.0], [0.0, 4.0, 2.0]]),
    }
    policy = {
        "pressure_convention": "gauge",
        "atmospheric_pressure_pa": 101325.0,
        "vapor_pressure_pa": 2339.0,
        "minimum_cavitation_margin_pa": 20000.0,
    }
    metrics = engineering_metrics(truth, params, policy)
    assert metrics["maximum_pressure_increment_pa"] == pytest.approx(5.0)
    assert metrics["maximum_pressure_location_over_L"] == pytest.approx(1.0)
    assert metrics["maximum_absolute_axial_stress_increment_pa"] == pytest.approx(7.0)
    assert metrics["stress_critical_location_over_L"] == pytest.approx(0.0)


def test_downsample_truth_preserves_endpoints_and_state_shape() -> None:
    truth = {
        "z": np.linspace(0.0, 1.0, 11),
        "t": np.linspace(0.0, 0.4, 9),
    }
    for index, state in enumerate(("V", "uz", "P", "sigma_z"), start=1):
        truth[state] = index * truth["z"][:, None] + truth["t"][None, :]
    saved = downsample_truth(truth, spatial_points=5, time_stride=3)
    assert saved["V"].shape == (5, 4)
    assert saved["t"][0] == pytest.approx(0.0)
    assert saved["t"][-1] == pytest.approx(0.4)
    assert saved["x"][0] == pytest.approx(0.0)
    assert saved["x"][-1] == pytest.approx(1.0)


def test_normalized_field_rmse_is_zero_for_identical_fields() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert normalized_field_rmse(values, values.copy()) == pytest.approx(0.0)
    assert normalized_field_rmse(values, values + 0.3) == pytest.approx(0.075)


def test_smoke_reference_dataset_is_independent_and_auditable(tmp_path) -> None:
    config = copy.deepcopy(load_config(DEFAULT_CONFIG))
    config["truth_solver"]["smoke"].update(
        {"n_cells": 8, "output_points": 21, "t_final_s": 0.01, "case_limit": 1}
    )
    config["truth_solver"]["storage"].update(
        {"saved_spatial_points": 5, "saved_time_stride": 2}
    )
    report = generate_reference_dataset(config, "smoke", tmp_path)
    assert report["status"] == "pass"
    assert report["case_count"] == 1
    audit = audit_reference_dataset(config, "smoke", tmp_path)
    assert audit == {"status": "pass", "mode": "smoke", "case_count": 1, "issues": []}
    resumed = generate_reference_dataset(config, "smoke", tmp_path)
    assert resumed == report
