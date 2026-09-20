from __future__ import annotations

import numpy as np
import torch

from water16_reproduction.xu2025_external_three_model import (
    FieldData,
    FieldNetwork,
    initial_head_line,
    registered_time_indices,
    variable_metrics,
)


def test_registered_split_is_fixed_and_disjoint() -> None:
    time_s = np.linspace(0.0, 6.0, 601)
    fitting, validation = registered_time_indices(time_s)
    assert fitting.size == 96
    assert validation.size == 24
    assert not set(fitting) & set(validation)


def test_initial_head_line_recovers_linear_profile() -> None:
    x = np.array([80.0, 160.0, 240.0])
    field = FieldData(np.array([0.0]), x, (50.0 - 0.0026 * x)[None, :], np.ones((1, 3)))
    slope, intercept = initial_head_line(field)
    assert np.isclose(slope, -0.0026)
    assert np.isclose(intercept, 50.0)


def test_characteristic_features_and_outputs_are_finite() -> None:
    model = FieldNetwork("characteristic_pinn", -0.0026, 50.0).double()
    x = torch.tensor([[50.0], [200.0]], dtype=torch.float64)
    t = torch.tensor([[0.1], [1.2]], dtype=torch.float64)
    features = model.features(x, t)
    head, velocity = model(x, t)
    assert features.shape == (2, 20)
    assert head.shape == velocity.shape == (2, 1)
    assert torch.isfinite(head).all() and torch.isfinite(velocity).all()


def test_metrics_identical_curves_are_exact() -> None:
    reference = np.array([0.0, 1.0, -1.0, 0.5])
    result = variable_metrics(reference, reference.copy())
    assert result["rmse"] == 0.0
    assert result["nrmse_dynamic_range"] == 0.0
    assert np.isclose(result["correlation"], 1.0)
