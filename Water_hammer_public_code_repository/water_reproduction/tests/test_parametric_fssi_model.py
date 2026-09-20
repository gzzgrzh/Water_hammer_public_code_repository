import copy

import numpy as np
import pytest
import torch

from water16_reproduction.parametric_fssi_model import (
    DEFAULT_CONFIG,
    ThreeParameterBoundaryModel,
    boundary_losses,
    characteristic_state,
    hard_boundary_diagnostics,
    load_config,
    load_design,
)
from water16_reproduction.parametric_fssi_reference import load_baseline
from water16_reproduction.parametric_fssi_forward import materialize_cases
from water16_reproduction.parametric_fssi_model import evaluate_reference_cases


def build_model():
    config = load_config(DEFAULT_CONFIG)
    design = load_design(config)
    model = ThreeParameterBoundaryModel(design, config, load_baseline(design)).double()
    return config, design, model


def test_model_config_forbids_all_reference_labels() -> None:
    config = load_config(DEFAULT_CONFIG)
    policy = config["training_data_policy"]
    assert policy["uses_moc_labels"] is False
    assert policy["uses_validation_cases"] is False
    assert policy["uses_test_cases"] is False


def test_features_change_with_each_registered_parameter() -> None:
    _, design, model = build_model()
    base = copy.deepcopy(design["case_design"]["pilot"]["cases"][0])
    times = torch.linspace(0.0, 0.2, 7)[:, None]
    from water16_reproduction.parametric_fssi_reference import case_parameters
    params, _ = case_parameters(model.baseline, base, model.t_final_s)
    baseline_features = model.features(times, base, params, "left")
    replacements = {
        "soil_to_pipe_modulus_ratio": 0.01,
        "closure_time_over_L_cf": 0.8,
        "initial_velocity_m_s": 0.08,
    }
    for name, value in replacements.items():
        changed = copy.deepcopy(base)
        changed[name] = value
        changed_params, _ = case_parameters(model.baseline, changed, model.t_final_s)
        features = model.features(times, changed, changed_params, "left")
        assert not torch.equal(features, baseline_features), name


def test_raw_parameter_ablation_keeps_width_and_zeros_selected_channel() -> None:
    config = load_config(DEFAULT_CONFIG)
    design = load_design(config)
    config["network"]["input_ablation"] = {
        "raw_parameter_channels": {
            "soil_to_pipe_modulus_ratio": False,
            "closure_time_over_L_cf": True,
            "initial_velocity_m_s": True,
        },
        "closure_phase_features": True,
        "travel_time_features": True,
    }
    model = ThreeParameterBoundaryModel(design, config, load_baseline(design)).double()
    case = design["case_design"]["pilot"]["cases"][0]
    from water16_reproduction.parametric_fssi_reference import case_parameters

    params, _ = case_parameters(model.baseline, case, model.t_final_s)
    features = model.features(torch.linspace(0.0, 0.2, 7)[:, None], case, params, "left")
    assert features.shape[1] == model.network[0].in_features
    assert torch.count_nonzero(features[:, 1]) == 0
    assert torch.count_nonzero(features[:, 2]) > 0
    assert torch.count_nonzero(features[:, 3]) > 0


def test_equations_only_ablation_removes_all_parameter_dependent_network_features() -> None:
    config = load_config(DEFAULT_CONFIG)
    design = load_design(config)
    config["network"]["input_ablation"] = {
        "raw_parameter_channels": {name: False for name in (
            "soil_to_pipe_modulus_ratio",
            "closure_time_over_L_cf",
            "initial_velocity_m_s",
        )},
        "closure_phase_features": False,
        "travel_time_features": False,
    }
    model = ThreeParameterBoundaryModel(design, config, load_baseline(design)).double()
    base = copy.deepcopy(design["case_design"]["pilot"]["cases"][0])
    changed = copy.deepcopy(base)
    changed.update({
        "soil_to_pipe_modulus_ratio": 0.01,
        "closure_time_over_L_cf": 0.8,
        "initial_velocity_m_s": 0.08,
    })
    from water16_reproduction.parametric_fssi_reference import case_parameters

    times = torch.linspace(0.0, 0.2, 7)[:, None]
    base_params, _ = case_parameters(model.baseline, base, model.t_final_s)
    changed_params, _ = case_parameters(model.baseline, changed, model.t_final_s)
    base_features = model.features(times, base, base_params, "left")
    changed_features = model.features(times, changed, changed_params, "left")
    torch.testing.assert_close(base_features, changed_features)


def test_characteristic_state_has_four_finite_outputs() -> None:
    _, design, model = build_model()
    case = design["case_design"]["pilot"]["cases"][0]
    state = characteristic_state(
        model,
        torch.tensor([[0.0], [50.0], [100.0]], dtype=torch.float64),
        torch.tensor([[0.0], [0.1], [0.2]], dtype=torch.float64),
        case,
    )
    assert state.shape == (3, 4)
    assert torch.all(torch.isfinite(state))


def test_hard_boundaries_and_losses_are_finite_at_initialization() -> None:
    _, design, model = build_model()
    case = design["case_design"]["pilot"]["cases"][0]
    times = torch.linspace(0.001, 0.2, 21, dtype=torch.float64)[:, None]
    diagnostics = hard_boundary_diagnostics(model, times, case)
    assert max(diagnostics.values()) < 1.0e-10
    losses = boundary_losses(model, times, case)
    assert set(losses) == {"transport_consistency", "valve_dynamic"}
    assert all(np.isfinite(float(value.detach())) for value in losses.values())


def test_evaluation_stage_selects_formal_case_definitions(tmp_path) -> None:
    config, design, model = build_model()
    root = tmp_path / "reference"
    x = np.linspace(0.0, 100.0, 5)
    t = np.linspace(0.0, 0.8, 9)
    from water16_reproduction.parametric_fssi_model import predict_field

    formal_validation = [
        case for case in materialize_cases(design, "formal")
        if case["split"] == "validation"
    ]
    for formal_case in formal_validation:
        case_dir = root / "validation" / formal_case["case_id"]
        case_dir.mkdir(parents=True)
        fields = predict_field(model, formal_case, x, t, torch.device("cpu"), 128)
        np.savez_compressed(case_dir / "truth_evaluation_grid.npz", x=x, t=t, **fields)
    evaluation_config = dict(config)
    evaluation_config["reference_root"] = str(root)
    evaluation_config["evaluation"] = {
        "stage": "formal", "splits": ["validation"], "batch_size": 128
    }
    summary, rows = evaluate_reference_cases(
        model, design, evaluation_config, torch.device("cpu")
    )
    assert summary["case_count"] == len(formal_validation) == 12
    assert rows[0]["case_id"].startswith("formal_validation_")
    assert "reference_pressure_peak_increment_pa" in rows[0]
