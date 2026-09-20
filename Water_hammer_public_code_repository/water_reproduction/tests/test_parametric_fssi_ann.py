from pathlib import Path

import torch

from water16_reproduction.parametric_fssi_ann import ParametricCoordinateANN, load_config
from water16_reproduction.parametric_fssi_model import load_design
from water16_reproduction.parametric_fssi_reference import load_baseline


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "PINN_FSSI_research_plan/configs/parametric_fssi_ann_baseline_v1.json"


def test_ann_is_strictly_data_only_and_uses_non_oracle_anchors() -> None:
    config = load_config(CONFIG)
    network = config["network"]
    assert network["uses_governing_equations"] is False
    assert network["uses_boundary_or_initial_losses"] is False
    assert network["uses_hard_physical_constraints"] is False
    assert network["uses_characteristic_speeds"] is False
    assert network["uses_travel_time_features"] is False
    assert config["anchor_config"].endswith("parametric_fssi_hybrid_formal_v3_balanced_events.json")


def test_ann_parameter_count_matches_coordinate_pinn_scale() -> None:
    torch.set_default_dtype(torch.float64)
    config = load_config(CONFIG)
    design = load_design(config)
    model = ParametricCoordinateANN(design, load_baseline(design), config)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 28_000 <= count <= 32_000
