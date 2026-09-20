import json
from pathlib import Path

import torch

from water16_reproduction.parametric_fssi_conventional_pinn import (
    ParametricFullDomainPINN,
    load_config,
)
from water16_reproduction.parametric_fssi_forward import materialize_cases
from water16_reproduction.parametric_fssi_model import load_design
from water16_reproduction.parametric_fssi_reference import case_parameters, load_baseline


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "PINN_FSSI_research_plan/configs/parametric_fssi_conventional_pinn_baseline_v1.json"


def test_registered_baseline_has_no_characteristic_features() -> None:
    config = load_config(CONFIG)
    assert config["network"]["inputs"] == [
        "x_over_L", "t_over_T", "log_Es_over_E", "closure_time_over_L_cf", "initial_velocity_m_s"
    ]
    assert config["network"]["uses_characteristic_speeds"] is False
    assert config["network"]["uses_travel_time_features"] is False
    assert config["evaluation"]["test_is_read_once_after_the_frozen_600_epoch_run"] is True


def test_model_parameter_count_and_hard_constraints() -> None:
    torch.set_default_dtype(torch.float64)
    config = load_config(CONFIG)
    design = load_design(config)
    baseline = load_baseline(design)
    model = ParametricFullDomainPINN(design, baseline, config)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 28_000 <= count <= 32_000
    case = next(case for case in materialize_cases(design, "formal") if case["split"] == "train")
    params, _ = case_parameters(baseline, case, model.t_final_s)
    points = torch.tensor([[0.0, 0.0], [params.length_m, 0.0], [0.0, 0.2]], dtype=torch.float64)
    state = model(points, case)
    assert torch.allclose(state[0, 0], torch.tensor(params.initial_velocity_m_s), atol=1e-12)
    assert torch.allclose(state[0, 1], torch.tensor(0.0), atol=1e-12)
    assert torch.allclose(state[0, 2], state[2, 2], atol=1e-10)
    assert torch.allclose(state[2, 1], torch.tensor(0.0), atol=1e-12)
