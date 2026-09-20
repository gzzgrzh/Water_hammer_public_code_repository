"""Three-parameter characteristic physics model for forward FSSI prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from water16_reproduction.common.physics import (
    characteristic_basis,
    cross_section_areas,
    initial_pressure_pa,
    state_scales,
    system_matrices,
)
from water16_reproduction.parametric_fssi_forward import (
    PARAMETER_NAMES,
    load_config as load_design_config,
    materialize_cases,
)
from water16_reproduction.parametric_fssi_reference import (
    case_parameters,
    first_local_peak_index,
    load_baseline,
)
from water16_reproduction.wp1_verification import STATE_NAMES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "PINN_FSSI_research_plan" / "configs" / "parametric_fssi_model_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "PINN_FSSI_research_plan" / "outputs" / "parametric_fssi_model_v1"
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("parametric FSSI model config schema_version must be 1")
    if config.get("method") != "three_parameter_hard_projected_characteristic_physics_model":
        raise ValueError("unexpected parametric FSSI method")
    policy = config["training_data_policy"]
    if any(
        policy[name]
        for name in (
            "uses_moc_labels",
            "uses_validation_cases",
            "uses_test_cases",
            "uses_digitised_literature_curves",
        )
    ):
        raise ValueError("v1 training must remain independent of validation data")
    for mode in ("smoke", "pilot"):
        spec = config["training"][mode]
        if int(spec["epochs"]) < 1 or int(spec["boundary_points"]) < 3:
            raise ValueError(f"invalid {mode} training controls")
    return config


def load_design(model_config: dict[str, Any]) -> dict[str, Any]:
    return load_design_config(ROOT / model_config["design_config"])


def canonical_basis(params) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    speeds, right, _, scales = characteristic_basis(params)
    right = right.copy()
    for column in range(right.shape[1]):
        pivot = int(np.argmax(np.abs(right[:, column])))
        if right[pivot, column] < 0.0:
            right[:, column] *= -1.0
    return speeds, right, np.linalg.inv(right), scales


def wave_speed_magnitudes(params) -> np.ndarray:
    """Return the two physical speed magnitudes after pairing +/- eigenvalues."""

    magnitudes = np.sort(np.abs(canonical_basis(params)[0]))
    if len(magnitudes) != 4 or magnitudes[0] <= 0.0:
        raise ValueError("the FSSI system requires four nonzero characteristic speeds")
    paired = magnitudes.reshape(2, 2)
    pair_error = np.abs(paired[:, 1] - paired[:, 0]) / np.maximum(
        np.mean(paired, axis=1), 1.0e-30
    )
    if np.max(pair_error) > 1.0e-8:
        raise ValueError("the FSSI characteristic speeds do not form +/- pairs")
    speeds = np.mean(paired, axis=1)
    if speeds[1] <= speeds[0]:
        raise ValueError("the FSSI feature map requires two distinct wave speeds")
    return speeds


class ThreeParameterBoundaryModel(nn.Module):
    """Shared outgoing boundary traces over time and three engineering inputs."""

    def __init__(self, design: dict[str, Any], config: dict[str, Any], baseline) -> None:
        super().__init__()
        self.design = design
        self.baseline = baseline
        self.t_final_s = float(design["truth_solver"]["pilot"]["t_final_s"])
        network = config["network"]
        self.time_frequencies = tuple(float(v) for v in network["time_fourier_frequencies"])
        self.closure_frequencies = tuple(float(v) for v in network["closure_phase_frequencies"])
        self.travel_frequencies = tuple(
            float(v) for v in network.get("travel_time_frequencies", [])
        )
        ablation = network.get("input_ablation", {})
        raw_channels = ablation.get("raw_parameter_channels", {})
        self.raw_parameter_channels = {
            name: bool(raw_channels.get(name, True)) for name in PARAMETER_NAMES
        }
        self.use_closure_phase_features = bool(
            ablation.get("closure_phase_features", True)
        )
        self.use_travel_time_features = bool(
            ablation.get("travel_time_features", True)
        )
        input_width = (
            7
            + 2 * len(self.time_frequencies)
            + 2 * len(self.closure_frequencies)
            + 4 * len(self.travel_frequencies)
        )
        width = int(network["hidden_width"])
        modules: list[nn.Module] = [nn.Linear(input_width, width), nn.Tanh()]
        for _ in range(int(network["hidden_layers"]) - 1):
            modules.extend([nn.Linear(width, width), nn.Tanh()])
        modules.append(nn.Linear(width, 3))
        self.network = nn.Sequential(*modules)
        output = self.network[-1]
        assert isinstance(output, nn.Linear)
        if network.get("zero_initialize_output", False):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def normalized_parameter(self, name: str, value: float) -> float:
        definition = self.design["parameter_domain"][name]
        lower, upper = float(definition["lower"]), float(definition["upper"])
        if not lower <= float(value) <= upper:
            raise ValueError(f"{name} lies outside the registered domain")
        if definition["coordinate"] == "natural_log":
            value, lower, upper = math.log(float(value)), math.log(lower), math.log(upper)
        return 2.0 * (float(value) - lower) / (upper - lower) - 1.0

    def features(
        self, times: torch.Tensor, case: dict[str, Any], params, side: str
    ) -> torch.Tensor:
        if times.ndim == 1:
            times = times[:, None]
        time_unit = times / self.t_final_s
        ratio = self.normalized_parameter(
            "soil_to_pipe_modulus_ratio", case["soil_to_pipe_modulus_ratio"]
        )
        closure = self.normalized_parameter(
            "closure_time_over_L_cf", case["closure_time_over_L_cf"]
        )
        velocity = self.normalized_parameter(
            "initial_velocity_m_s", case["initial_velocity_m_s"]
        )
        side_value = -1.0 if side == "left" else 1.0
        closure_phase = torch.clamp(times / params.valve_close_time_s, 0.0, 1.0)
        post_closure = torch.clamp(
            (times - params.valve_close_time_s) / self.t_final_s, min=0.0
        )
        parts = [
            2.0 * time_unit - 1.0,
            torch.full_like(
                time_unit,
                ratio if self.raw_parameter_channels["soil_to_pipe_modulus_ratio"] else 0.0,
            ),
            torch.full_like(
                time_unit,
                closure if self.raw_parameter_channels["closure_time_over_L_cf"] else 0.0,
            ),
            torch.full_like(
                time_unit,
                velocity if self.raw_parameter_channels["initial_velocity_m_s"] else 0.0,
            ),
            torch.full_like(time_unit, side_value),
            2.0 * closure_phase - 1.0
            if self.use_closure_phase_features
            else torch.zeros_like(time_unit),
            post_closure
            if self.use_closure_phase_features
            else torch.zeros_like(time_unit),
        ]
        for frequency in self.time_frequencies:
            phase = 2.0 * math.pi * frequency * time_unit
            parts.extend([torch.sin(phase), torch.cos(phase)])
        for frequency in self.closure_frequencies:
            phase = math.pi * frequency * closure_phase
            if self.use_closure_phase_features:
                parts.extend([torch.sin(phase), torch.cos(phase)])
            else:
                parts.extend([torch.zeros_like(time_unit), torch.zeros_like(time_unit)])
        for speed in wave_speed_magnitudes(params):
            round_trip_phase = times * float(speed) / (2.0 * params.length_m)
            for frequency in self.travel_frequencies:
                phase = 2.0 * math.pi * frequency * round_trip_phase
                if self.use_travel_time_features:
                    parts.extend([torch.sin(phase), torch.cos(phase)])
                else:
                    parts.extend([torch.zeros_like(time_unit), torch.zeros_like(time_unit)])
        return torch.cat(parts, dim=1)

    def raw_trace(
        self, times: torch.Tensor, case: dict[str, Any], params, side: str
    ) -> torch.Tensor:
        return self.network(self.features(times, case, params, side))

    def gate(self, times: torch.Tensor, params) -> torch.Tensor:
        scale = max(params.valve_close_time_s, self.t_final_s / 200.0)
        return 1.0 - torch.exp(-torch.clamp(times, min=0.0) / scale)


def _basis(model: ThreeParameterBoundaryModel, case: dict[str, Any], reference: torch.Tensor):
    params, _ = case_parameters(model.baseline, case, model.t_final_s)
    speeds_np, right_np, left_np, scales_np = canonical_basis(params)
    device, dtype = reference.device, reference.dtype
    right = torch.as_tensor(right_np, device=device, dtype=dtype)
    left = torch.as_tensor(left_np, device=device, dtype=dtype)
    scales = torch.as_tensor(scales_np, device=device, dtype=dtype)
    pressure0 = initial_pressure_pa(params)
    area_f, area_t = cross_section_areas(params)
    q0 = torch.tensor(
        [params.initial_velocity_m_s, 0.0, pressure0, area_f * pressure0 / area_t],
        device=device,
        dtype=dtype,
    )
    initial_w = left @ (q0 / scales)
    return params, speeds_np, right, scales, initial_w, scales[:, None] * right


def valve_relative_velocity(params, times: torch.Tensor) -> torch.Tensor:
    phase = torch.clamp(times / params.valve_close_time_s, 0.0, 1.0)
    return 0.5 * params.initial_velocity_m_s * (1.0 + torch.cos(math.pi * phase))


def projected_boundary_amplitudes(
    model: ThreeParameterBoundaryModel,
    times: torch.Tensor,
    case: dict[str, Any],
    side: str,
) -> torch.Tensor:
    if times.ndim == 1:
        times = times[:, None]
    params, speeds, _, scales, initial_w, physical_modes = _basis(model, case, times)
    positive = np.flatnonzero(speeds > 0.0)
    negative = np.flatnonzero(speeds < 0.0)
    incoming = positive if side == "left" else negative
    outgoing = negative if side == "left" else positive
    raw = model.raw_trace(times, case, params, side)
    gate = model.gate(times, params)
    amplitudes = initial_w.reshape(1, 4).expand(len(times), -1).clone()
    amplitudes[:, outgoing] = initial_w[outgoing].reshape(1, 2) + gate * raw[:, :2]

    conditions = torch.zeros((2, 4), device=times.device, dtype=times.dtype)
    if side == "left":
        conditions[0, 2] = 1.0
        conditions[1, 1] = 1.0
        targets = torch.tensor(
            [initial_pressure_pa(params), 0.0], device=times.device, dtype=times.dtype
        ).reshape(1, 2).expand(len(times), -1)
    else:
        conditions[0, 0] = 1.0
        conditions[0, 1] = -1.0
        conditions[1, 1] = 1.0
        axial_velocity = gate * scales[1] * raw[:, 2:3]
        targets = torch.cat([valve_relative_velocity(params, times), axial_velocity], dim=1)
    matrix = conditions @ physical_modes[:, incoming]
    without_incoming = (
        amplitudes @ physical_modes.T
        - amplitudes[:, incoming] @ physical_modes[:, incoming].T
    )
    rhs = targets - without_incoming @ conditions.T
    amplitudes[:, incoming] = torch.linalg.solve(matrix, rhs.T).T
    return amplitudes


def characteristic_state(
    model: ThreeParameterBoundaryModel,
    positions: torch.Tensor,
    times: torch.Tensor,
    case: dict[str, Any],
) -> torch.Tensor:
    if positions.ndim == 1:
        positions = positions[:, None]
    if times.ndim == 1:
        times = times[:, None]
    if positions.shape != times.shape:
        raise ValueError("positions and times must have matching shapes")
    _, speeds, right, scales, initial_w, _ = _basis(model, case, times)
    amplitudes: list[torch.Tensor] = []
    for index, speed in enumerate(speeds):
        if speed > 0.0:
            boundary_time = times - positions / float(speed)
            boundary = projected_boundary_amplitudes(
                model, torch.clamp(boundary_time, min=0.0), case, "left"
            )[:, index : index + 1]
        else:
            boundary_time = times - (model.baseline.length_m - positions) / float(-speed)
            boundary = projected_boundary_amplitudes(
                model, torch.clamp(boundary_time, min=0.0), case, "right"
            )[:, index : index + 1]
        amplitudes.append(
            torch.where(
                boundary_time >= 0.0,
                boundary,
                torch.full_like(times, float(initial_w[index])),
            )
        )
    return (torch.cat(amplitudes, dim=1) @ right.T) * scales


def boundary_residuals(
    model: ThreeParameterBoundaryModel, times: torch.Tensor, case: dict[str, Any]
) -> dict[str, torch.Tensor]:
    times = times.detach().clone().requires_grad_(True)
    params, speeds, right, scales, initial_w, _ = _basis(model, case, times)
    left_w = projected_boundary_amplitudes(model, times, case, "left")
    right_w = projected_boundary_amplitudes(model, times, case, "right")
    right_q = (right_w @ right.T) * scales
    uz_t = torch.autograd.grad(
        right_q[:, 1:2],
        times,
        grad_outputs=torch.ones_like(right_q[:, 1:2]),
        create_graph=True,
        retain_graph=True,
    )[0]
    area_f, area_t = cross_section_areas(params)
    dynamic = (
        params.valve_mass_kg * uz_t - area_f * right_q[:, 2:3] + area_t * right_q[:, 3:4]
    ) / max(area_f * float(scales[2]), area_t * float(scales[3]), 1.0)
    consistency: list[torch.Tensor] = []
    for index, speed in enumerate(speeds):
        source_time = times - params.length_m / abs(float(speed))
        if speed > 0.0:
            source = projected_boundary_amplitudes(
                model, torch.clamp(source_time, min=0.0), case, "left"
            )[:, index : index + 1]
            target = right_w[:, index : index + 1]
        else:
            source = projected_boundary_amplitudes(
                model, torch.clamp(source_time, min=0.0), case, "right"
            )[:, index : index + 1]
            target = left_w[:, index : index + 1]
        consistency.append(
            target
            - torch.where(
                source_time >= 0.0,
                source,
                torch.full_like(source, float(initial_w[index])),
            )
        )
    return {"transport_consistency": torch.cat(consistency, dim=1), "valve_dynamic": dynamic}


def boundary_losses(
    model: ThreeParameterBoundaryModel, times: torch.Tensor, case: dict[str, Any]
) -> dict[str, torch.Tensor]:
    return {name: torch.mean(value**2) for name, value in boundary_residuals(model, times, case).items()}


def hard_boundary_diagnostics(
    model: ThreeParameterBoundaryModel, times: torch.Tensor, case: dict[str, Any]
) -> dict[str, float]:
    params, _, right, scales, _, _ = _basis(model, case, times)
    left = (projected_boundary_amplitudes(model, times, case, "left") @ right.T) * scales
    right_q = (projected_boundary_amplitudes(model, times, case, "right") @ right.T) * scales
    upstream = torch.stack(
        [(left[:, 2] - initial_pressure_pa(params)) / scales[2], left[:, 1] / scales[1]], dim=1
    )
    kinematic = (
        right_q[:, 0:1] - right_q[:, 1:2] - valve_relative_velocity(params, times)
    ) / scales[0]
    return {
        "upstream_nrmse": float(torch.sqrt(torch.mean(upstream**2)).detach().cpu()),
        "valve_kinematic_nrmse": float(torch.sqrt(torch.mean(kinematic**2)).detach().cpu()),
    }


def transport_residual_nrmse(
    model: ThreeParameterBoundaryModel,
    positions: torch.Tensor,
    times: torch.Tensor,
    case: dict[str, Any],
) -> float:
    positions = positions.detach().clone().requires_grad_(True)
    times = times.detach().clone().requires_grad_(True)
    state = characteristic_state(model, positions, times, case)
    q_x, q_t = [], []
    for index in range(4):
        q_x.append(torch.autograd.grad(state[:, index:index+1], positions, torch.ones_like(state[:, index:index+1]), retain_graph=True)[0])
        q_t.append(torch.autograd.grad(state[:, index:index+1], times, torch.ones_like(state[:, index:index+1]), retain_graph=True)[0])
    params, _ = case_parameters(model.baseline, case, model.t_final_s)
    matrix_a_np, matrix_b_np = system_matrices(params)
    matrix_a = torch.as_tensor(matrix_a_np, device=times.device, dtype=times.dtype)
    matrix_b = torch.as_tensor(matrix_b_np, device=times.device, dtype=times.dtype)
    residual = torch.cat(q_t, dim=1) @ matrix_a.T + torch.cat(q_x, dim=1) @ matrix_b.T
    scales = torch.as_tensor(state_scales(params), device=times.device, dtype=times.dtype)
    row_scale = torch.abs(matrix_a) @ scales / params.t_final_s + torch.abs(matrix_b) @ scales / params.length_m
    return float(torch.sqrt(torch.mean((residual / row_scale) ** 2)).detach().cpu())


def dynamic_state_scale(reference: np.ndarray, initial: float) -> float:
    perturbation = reference - initial
    return max(float(np.ptp(reference)), float(np.max(np.abs(perturbation))), 1.0e-30)


def predict_field(
    model: ThreeParameterBoundaryModel,
    case: dict[str, Any],
    x: np.ndarray,
    t: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    xx, tt = np.meshgrid(x, t, indexing="ij")
    points_x, points_t = xx.ravel(), tt.ravel()
    values = np.empty((len(points_x), 4), dtype=float)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points_x), batch_size):
            stop = min(start + batch_size, len(points_x))
            values[start:stop] = characteristic_state(
                model,
                torch.as_tensor(points_x[start:stop, None], device=device),
                torch.as_tensor(points_t[start:stop, None], device=device),
                case,
            ).cpu().numpy()
    model.train()
    return {state: values[:, index].reshape(xx.shape) for index, state in enumerate(STATE_NAMES)}


def evaluate_reference_cases(
    model: ThreeParameterBoundaryModel,
    design: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_root = ROOT / config["reference_root"]
    requested_splits = set(config["evaluation"]["splits"])
    evaluation_stage = config["evaluation"].get("stage", "pilot")
    cases = [
        case
        for case in materialize_cases(design, evaluation_stage)
        if case["split"] in requested_splits
    ]
    rows: list[dict[str, Any]] = []
    batch_size = int(config["evaluation"]["batch_size"])
    for case in cases:
        path = reference_root / case["split"] / case["case_id"] / "truth_evaluation_grid.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            x, t = archive["x"], archive["t"]
            reference = {state: archive[state] for state in STATE_NAMES}
        prediction = predict_field(model, case, x, t, device, batch_size)
        params, _ = case_parameters(model.baseline, case, model.t_final_s)
        pressure0 = initial_pressure_pa(params)
        area_f, area_t = cross_section_areas(params)
        initials = {
            "V": params.initial_velocity_m_s,
            "uz": 0.0,
            "P": pressure0,
            "sigma_z": area_f * pressure0 / area_t,
        }
        state_metrics: dict[str, float] = {}
        for state in STATE_NAMES:
            scale = dynamic_state_scale(reference[state], initials[state])
            error = prediction[state] - reference[state]
            state_metrics[f"{state}_nrmse"] = float(np.sqrt(np.mean(error**2)) / scale)
        reference_pressure_delta = reference["P"] - pressure0
        prediction_pressure_delta = prediction["P"] - pressure0
        reference_stress_delta = reference["sigma_z"] - initials["sigma_z"]
        prediction_stress_delta = prediction["sigma_z"] - initials["sigma_z"]
        reference_pressure_peak = float(np.max(reference_pressure_delta))
        prediction_pressure_peak = float(np.max(prediction_pressure_delta))
        reference_stress_peak = float(np.max(np.abs(reference_stress_delta)))
        prediction_stress_peak = float(np.max(np.abs(prediction_stress_delta)))
        pressure_peak_error = abs(prediction_pressure_peak - reference_pressure_peak) / max(
            abs(reference_pressure_peak), 1.0e-30
        )
        stress_peak_error = abs(prediction_stress_peak - reference_stress_peak) / max(
            reference_stress_peak, 1.0e-30
        )
        reference_pressure_index = np.unravel_index(
            int(np.argmax(reference_pressure_delta)), reference_pressure_delta.shape
        )
        prediction_pressure_index = np.unravel_index(
            int(np.argmax(prediction_pressure_delta)), prediction_pressure_delta.shape
        )
        reference_stress_index = np.unravel_index(
            int(np.argmax(np.abs(reference_stress_delta))), reference_stress_delta.shape
        )
        prediction_stress_index = np.unravel_index(
            int(np.argmax(np.abs(prediction_stress_delta))), prediction_stress_delta.shape
        )
        start_index = int(np.searchsorted(t, params.valve_close_time_s))
        reference_pressure_first_index = first_local_peak_index(reference["P"][-1], start_index)
        prediction_pressure_first_index = first_local_peak_index(prediction["P"][-1], start_index)
        reference_stress_first_index = first_local_peak_index(
            np.abs(reference_stress_delta[-1]), start_index
        )
        prediction_stress_first_index = first_local_peak_index(
            np.abs(prediction_stress_delta[-1]), start_index
        )
        rows.append({
            "case_id": case["case_id"],
            "split": case["split"],
            "test_class": case["test_class"],
            **{name: case[name] for name in PARAMETER_NAMES},
            **state_metrics,
            "pressure_peak_relative_error": pressure_peak_error,
            "stress_peak_relative_error": stress_peak_error,
            "reference_pressure_peak_increment_pa": reference_pressure_peak,
            "prediction_pressure_peak_increment_pa": prediction_pressure_peak,
            "reference_stress_peak_increment_pa": reference_stress_peak,
            "prediction_stress_peak_increment_pa": prediction_stress_peak,
            "reference_pressure_first_peak_time_s": float(t[reference_pressure_first_index]),
            "prediction_pressure_first_peak_time_s": float(t[prediction_pressure_first_index]),
            "reference_stress_first_peak_time_s": float(t[reference_stress_first_index]),
            "prediction_stress_first_peak_time_s": float(t[prediction_stress_first_index]),
            "reference_pressure_critical_location_over_L": float(
                x[reference_pressure_index[0]] / params.length_m
            ),
            "prediction_pressure_critical_location_over_L": float(
                x[prediction_pressure_index[0]] / params.length_m
            ),
            "reference_stress_critical_location_over_L": float(
                x[reference_stress_index[0]] / params.length_m
            ),
            "prediction_stress_critical_location_over_L": float(
                x[prediction_stress_index[0]] / params.length_m
            ),
        })
        print(
            f"evaluated {case['case_id']}: P={state_metrics['P_nrmse']:.3%}, "
            f"sigma={state_metrics['sigma_z_nrmse']:.3%}", flush=True
        )
    summary = {
        "case_count": len(rows),
        "maximum_by_metric": {
            metric: max(row[metric] for row in rows)
            for metric in (
                "V_nrmse", "uz_nrmse", "P_nrmse", "sigma_z_nrmse",
                "pressure_peak_relative_error", "stress_peak_relative_error",
            )
        },
        "mean_by_metric": {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in (
                "V_nrmse", "uz_nrmse", "P_nrmse", "sigma_z_nrmse",
                "pressure_peak_relative_error", "stress_peak_relative_error",
            )
        },
    }
    return summary, rows


def run_training(config: dict[str, Any], mode: str, output_dir: Path) -> dict[str, Any]:
    design = load_design(config)
    baseline = load_baseline(design)
    spec = config["training"][mode]
    torch.set_default_dtype(torch.float64 if spec["dtype"] == "float64" else torch.float32)
    torch.manual_seed(int(spec["seed"]))
    np.random.seed(int(spec["seed"]))
    device = torch.device("cuda" if spec["device"] == "auto" and torch.cuda.is_available() else ("cpu" if spec["device"] == "auto" else spec["device"]))
    run_dir = output_dir / mode
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {run_dir}")
    run_dir.mkdir(parents=True)
    model = ThreeParameterBoundaryModel(design, config, baseline).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["learning_rate"]))
    train_cases = [case for case in materialize_cases(design, "pilot") if case["split"] == "train"][: int(spec["case_limit"])]
    times = torch.linspace(0.0, model.t_final_s, int(spec["boundary_points"]), device=device)[:, None]
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, int(spec["epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = {name: torch.zeros((), device=device) for name in config["loss_weights"]}
        for case in train_cases:
            losses = boundary_losses(model, times, case)
            for name, value in losses.items():
                accumulated[name] += value / len(train_cases)
        total = sum(float(config["loss_weights"][name]) * value for name, value in accumulated.items())
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip"])).detach().cpu())
        optimizer.step()
        row = {"epoch": float(epoch), "loss_total": float(total.detach().cpu()), **{f"loss_{name}": float(value.detach().cpu()) for name, value in accumulated.items()}, "gradient_norm": gradient_norm}
        history.append(row)
        if epoch == 1 or epoch == int(spec["epochs"]) or epoch % int(spec["log_every"]) == 0:
            print(f"[{mode}] epoch={epoch} loss={row['loss_total']:.3e} transport={row['loss_transport_consistency']:.3e} dynamic={row['loss_valve_dynamic']:.3e}", flush=True)
        if epoch % int(spec["checkpoint_every"]) == 0 or epoch == int(spec["epochs"]):
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "history": history}, run_dir / "checkpoint.pt")

    lbfgs_steps = int(spec.get("lbfgs_steps", 0))
    if lbfgs_steps:
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=float(spec.get("lbfgs_learning_rate", 0.3)), max_iter=lbfgs_steps, line_search_fn="strong_wolfe")
        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            accumulated = {name: torch.zeros((), device=device) for name in config["loss_weights"]}
            for case in train_cases:
                losses = boundary_losses(model, times, case)
                for name, value in losses.items():
                    accumulated[name] += value / len(train_cases)
            total = sum(float(config["loss_weights"][name]) * value for name, value in accumulated.items())
            total.backward()
            return total
        lbfgs.step(closure)
        torch.save({"state_dict": model.state_dict(), "epoch": int(spec["epochs"]), "history": history}, run_dir / "checkpoint.pt")

    diagnostic_times = torch.linspace(0.01, 0.79, 97, device=device)[:, None]
    rng = np.random.default_rng(int(spec["seed"]))
    diagnostic_positions = torch.as_tensor(rng.uniform(5.0, 95.0, (128, 1)), device=device)
    diagnostic_field_times = torch.as_tensor(rng.uniform(0.04, 0.76, (128, 1)), device=device)
    boundary_values = [hard_boundary_diagnostics(model, diagnostic_times, case) for case in train_cases]
    transport_values = [transport_residual_nrmse(model, diagnostic_positions, diagnostic_field_times, case) for case in train_cases]
    structural = {
        "maximum_transport_nrmse": max(transport_values),
        "maximum_upstream_nrmse": max(value["upstream_nrmse"] for value in boundary_values),
        "maximum_valve_kinematic_nrmse": max(value["valve_kinematic_nrmse"] for value in boundary_values),
    }
    structural_pass = structural["maximum_transport_nrmse"] <= float(config["acceptance"]["maximum_transport_nrmse"]) and max(structural["maximum_upstream_nrmse"], structural["maximum_valve_kinematic_nrmse"]) <= float(config["acceptance"]["maximum_hard_boundary_nrmse"])
    report: dict[str, Any] = {
        "status": "pass" if structural_pass else "failed",
        "mode": mode,
        "model_id": config["model_id"],
        "method": config["method"],
        "device": str(device),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_case_ids": [case["case_id"] for case in train_cases],
        "training_uses_moc_labels": False,
        "training_seconds": time.perf_counter() - started,
        "structural_diagnostics": structural,
        "final_training_loss": history[-1],
    }
    if mode == "pilot":
        evaluation, rows = evaluate_reference_cases(model, design, config, device)
        limits = config["acceptance"]
        accuracy_pass = (
            evaluation["maximum_by_metric"]["P_nrmse"] <= float(limits["pilot_pressure_nrmse"])
            and evaluation["maximum_by_metric"]["sigma_z_nrmse"] <= float(limits["pilot_axial_stress_nrmse"])
            and evaluation["maximum_by_metric"]["pressure_peak_relative_error"] <= float(limits["pilot_pressure_peak_relative_error"])
            and evaluation["maximum_by_metric"]["stress_peak_relative_error"] <= float(limits["pilot_stress_peak_relative_error"])
        )
        report["evaluation"] = evaluation
        report["accuracy_acceptance"] = "pass" if accuracy_pass else "failed"
        report["status"] = "pass" if structural_pass and accuracy_pass else "failed"
        with (run_dir / "held_out_case_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "config": config, "report": report}, run_dir / "model.pt")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("audit", "smoke", "pilot"), default="audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "audit":
        design = load_design(config)
        report = {
            "status": "pass",
            "mode": "audit",
            "model_id": config["model_id"],
            "parameter_names": list(PARAMETER_NAMES),
            "training_data_policy": config["training_data_policy"],
            "pilot_counts": design["case_design"]["pilot"]["expected_counts"],
        }
    else:
        report = run_training(config, args.mode, args.output_dir)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
