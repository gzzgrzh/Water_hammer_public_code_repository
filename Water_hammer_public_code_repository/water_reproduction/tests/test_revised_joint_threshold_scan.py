from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from water16_reproduction.revised_joint_threshold_scan import (
    registered_cases,
    robust_minimum_closure,
    utilizations,
)


ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    return json.loads(
        (ROOT / "PINN_FSSI_research_plan/configs/revised_joint_threshold_assessment_v1.json").read_text(encoding="utf-8")
    )


def test_registered_scan_is_unique_and_inside_domain() -> None:
    config = load_config()
    cases = registered_cases(config)
    keys = {
        (
            row["soil_to_pipe_modulus_ratio"],
            row["closure_time_over_L_cf"],
            row["initial_velocity_m_s"],
        )
        for row in cases
    }
    assert len(cases) == len(keys)
    assert len(cases) >= 12000
    for row in cases:
        assert 0.0003 <= row["soil_to_pipe_modulus_ratio"] <= 0.1
        assert 0.1 <= row["closure_time_over_L_cf"] <= 1.2
        assert 0.03 <= row["initial_velocity_m_s"] <= 0.09


def test_joint_utilization_and_control_are_auditable() -> None:
    thresholds = load_config()["threshold_scenarios"]["nominal"]
    row = {
        "maximum_pressure_increment_pa": thresholds["maximum_pressure_increment_pa"] * 1.1,
        "maximum_absolute_axial_stress_increment_pa": thresholds["maximum_absolute_axial_stress_increment_pa"] * 0.8,
        "minimum_absolute_pressure_pa": thresholds["minimum_absolute_pressure_pa"] / 0.7,
    }
    values, safe, control = utilizations(row, thresholds)
    assert np.allclose(values, [1.1, 0.8, 0.7])
    assert not safe
    assert control == "pressure"


def test_minimum_closure_requires_all_slower_cases_to_be_safe() -> None:
    rows = [
        {"closure_time_over_L_cf": 0.1, "nominal_safe": False},
        {"closure_time_over_L_cf": 0.2, "nominal_safe": True},
        {"closure_time_over_L_cf": 0.3, "nominal_safe": False},
        {"closure_time_over_L_cf": 0.4, "nominal_safe": True},
        {"closure_time_over_L_cf": 0.5, "nominal_safe": True},
    ]
    assert np.isclose(robust_minimum_closure(rows, "nominal"), 0.4)

