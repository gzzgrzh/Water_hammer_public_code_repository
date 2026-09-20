"""Train one preregistered conventional full-space--time parametric PINN baseline."""

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
from torch import nn

from water16_reproduction.common.physics import (
    cross_section_areas,
    initial_pressure_pa,
    state_scales,
    system_matrices,
)
from water16_reproduction.parametric_fssi_forward import PARAMETER_NAMES, materialize_cases
from water16_reproduction.parametric_fssi_hybrid import load_config as load_hybrid_config
from water16_reproduction.parametric_fssi_hybrid import load_training_anchors
from water16_reproduction.parametric_fssi_model import dynamic_state_scale, load_design
from water16_reproduction.parametric_fssi_reference import (
    case_parameters,
    first_local_peak_index,
    load_baseline,
)
from water16_reproduction.wp1_verification import STATE_NAMES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "PINN_FSSI_research_plan/configs/parametric_fssi_conventional_pinn_baseline_v1.json"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/parametric_fssi_conventional_pinn_baseline_v1"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    network = config["network"]
    if network["uses_characteristic_speeds"] or network["uses_travel_time_features"]:
        raise ValueError("conventional baseline may not use characteristic or travel-time features")
    splits = config["evaluation"]["splits"]
    if "validation" not in splits or not set(splits).issubset({"validation", "test"}):
        raise ValueError("conventional baseline must evaluate validation and optionally test")
    return config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def normalized_parameter_values(case: dict[str, Any], design: dict[str, Any]) -> list[float]:
    values = []
    for name in PARAMETER_NAMES:
        definition = design["parameter_domain"][name]
        lower, upper = float(definition["lower"]), float(definition["upper"])
        value = float(case[name])
        if definition["coordinate"] == "natural_log":
            lower, upper, value = math.log(lower), math.log(upper), math.log(value)
        values.append(2.0 * (value - lower) / (upper - lower) - 1.0)
    return values


class ParametricFullDomainPINN(nn.Module):
    """Five-coordinate MLP with no characteristic or travel-time features."""

    def __init__(self, design: dict[str, Any], baseline, config: dict[str, Any]) -> None:
        super().__init__()
        self.design = design
        self.baseline = baseline
        self.t_final_s = float(design["truth_solver"]["formal"]["t_final_s"])
        width = int(config["network"]["hidden_width"])
        layers = int(config["network"]["hidden_layers"])
        modules: list[nn.Module] = [nn.Linear(5, width), nn.Tanh()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(width, width), nn.Tanh()])
        modules.append(nn.Linear(width, 4))
        self.network = nn.Sequential(*modules)

    def forward(self, points: torch.Tensor, case: dict[str, Any]) -> torch.Tensor:
        params, _ = case_parameters(self.baseline, case, self.t_final_s)
        xi = points[:, 0:1] / params.length_m
        tau = points[:, 1:2] / self.t_final_s
        mu = torch.as_tensor(
            normalized_parameter_values(case, self.design), device=points.device, dtype=points.dtype
        ).expand(len(points), -1)
        inputs = torch.cat([2.0 * xi - 1.0, 2.0 * tau - 1.0, mu], dim=1)
        raw = self.network(inputs)
        scales = torch.as_tensor(state_scales(params), device=points.device, dtype=points.dtype)
        correction = raw * scales
        pressure0 = initial_pressure_pa(params)
        area_f, area_t = cross_section_areas(params)
        stress0 = area_f * pressure0 / area_t
        gate = tau
        uz = gate * xi * correction[:, 1:2]
        pressure = pressure0 + gate * xi * correction[:, 2:3]
        stress = stress0 + gate * correction[:, 3:4]
        free_velocity = params.initial_velocity_m_s + gate * correction[:, 0:1]
        phase = torch.clamp(points[:, 1:2] / params.valve_close_time_s, 0.0, 1.0)
        relative = 0.5 * params.initial_velocity_m_s * (1.0 + torch.cos(math.pi * phase))
        valve_velocity = uz + relative
        velocity = (1.0 - xi) * free_velocity + xi * valve_velocity
        return torch.cat([velocity, uz, pressure, stress], dim=1)


def sample_points(params, specification: dict[str, Any], rng: np.random.Generator, device: torch.device):
    pde_count = int(specification["pde_points_per_case"])
    boundary_count = int(specification["boundary_points_per_case"])
    pde = np.column_stack([
        rng.uniform(0.0, params.length_m, pde_count),
        rng.uniform(0.0, params.t_final_s, pde_count),
    ])
    right = np.column_stack([
        np.full(boundary_count, params.length_m),
        np.linspace(params.t_final_s / boundary_count, params.t_final_s, boundary_count),
    ])
    return (
        torch.as_tensor(pde, device=device),
        torch.as_tensor(right, device=device),
    )


def pde_loss(model: ParametricFullDomainPINN, points: torch.Tensor, case: dict[str, Any]) -> torch.Tensor:
    params, _ = case_parameters(model.baseline, case, model.t_final_s)
    zt = points.detach().clone().requires_grad_(True)
    state = model(zt, case)
    derivatives = [
        torch.autograd.grad(
            state[:, index:index + 1], zt, torch.ones_like(state[:, index:index + 1]),
            create_graph=True, retain_graph=True,
        )[0]
        for index in range(4)
    ]
    state_x = torch.cat([value[:, 0:1] for value in derivatives], dim=1)
    state_t = torch.cat([value[:, 1:2] for value in derivatives], dim=1)
    matrix_a_np, matrix_b_np = system_matrices(params)
    matrix_a = torch.as_tensor(matrix_a_np, device=zt.device, dtype=zt.dtype)
    matrix_b = torch.as_tensor(matrix_b_np, device=zt.device, dtype=zt.dtype)
    residual = state_t @ matrix_a.T + state_x @ matrix_b.T
    scales = torch.as_tensor(state_scales(params), device=zt.device, dtype=zt.dtype)
    row_scale = (
        torch.abs(matrix_a) @ scales / model.t_final_s
        + torch.abs(matrix_b) @ scales / params.length_m
    )
    return torch.mean((residual / torch.clamp(row_scale, min=1.0e-30)) ** 2)


def valve_dynamic_loss(
    model: ParametricFullDomainPINN, points: torch.Tensor, case: dict[str, Any]
) -> torch.Tensor:
    params, _ = case_parameters(model.baseline, case, model.t_final_s)
    right = points.detach().clone().requires_grad_(True)
    state = model(right, case)
    uz_t = torch.autograd.grad(
        state[:, 1:2], right, torch.ones_like(state[:, 1:2]), create_graph=True, retain_graph=True
    )[0][:, 1:2]
    area_f, area_t = cross_section_areas(params)
    force = params.valve_mass_kg * uz_t - area_f * state[:, 2:3] + area_t * state[:, 3:4]
    scales = state_scales(params)
    force_scale = max(area_f * scales[2], area_t * scales[3], 1.0)
    return torch.mean((force / force_scale) ** 2)


def anchor_objective(model: ParametricFullDomainPINN, batch, config: dict[str, Any]) -> torch.Tensor:
    points = torch.cat([batch.positions, batch.times], dim=1)
    prediction = model(points, batch.case)
    squared = ((prediction - batch.targets) / batch.scales) ** 2
    weights = batch.observation_weights[None, :]
    mean_loss = torch.sum(squared * weights) / (squared.shape[0] * torch.sum(weights))
    fraction = float(config["anchor_objective"]["tail_fraction"])
    weight = float(config["anchor_objective"]["tail_weight"])
    count = max(1, int(math.ceil(fraction * squared.shape[0])))
    observed_indices = torch.nonzero(batch.observation_weights > 0.0, as_tuple=False).flatten().tolist()
    tail = torch.stack([
        torch.topk(squared[:, state], count).values.mean() for state in observed_indices
    ]).mean()
    return mean_loss + weight * tail


def predict_field(
    model: ParametricFullDomainPINN,
    case: dict[str, Any],
    x: np.ndarray,
    t: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    xx, tt = np.meshgrid(x, t, indexing="ij")
    points = np.column_stack([xx.ravel(), tt.ravel()])
    values = np.empty((len(points), 4), dtype=float)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            stop = min(start + batch_size, len(points))
            values[start:stop] = model(
                torch.as_tensor(points[start:stop], device=device), case
            ).cpu().numpy()
    model.train()
    return {state: values[:, index].reshape(xx.shape) for index, state in enumerate(STATE_NAMES)}


def evaluate(model, config: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases = [
        case for case in materialize_cases(model.design, "formal")
        if case["split"] in set(config["evaluation"]["splits"])
    ]
    root = ROOT / config["reference_root"]
    rows = []
    for case in cases:
        path = root / case["split"] / case["case_id"] / "truth_evaluation_grid.npz"
        with np.load(path, allow_pickle=False) as archive:
            x, t = archive["x"], archive["t"]
            reference = {state: archive[state] for state in STATE_NAMES}
        prediction = predict_field(model, case, x, t, device, int(config["evaluation"]["batch_size"]))
        params, _ = case_parameters(model.baseline, case, model.t_final_s)
        pressure0 = initial_pressure_pa(params)
        area_f, area_t = cross_section_areas(params)
        initials = {"V": params.initial_velocity_m_s, "uz": 0.0, "P": pressure0, "sigma_z": area_f * pressure0 / area_t}
        metrics = {}
        for state in STATE_NAMES:
            scale = dynamic_state_scale(reference[state], initials[state])
            metrics[f"{state}_nrmse"] = float(np.sqrt(np.mean((prediction[state] - reference[state]) ** 2)) / scale)
        reference_pressure = reference["P"] - pressure0
        prediction_pressure = prediction["P"] - pressure0
        reference_stress = reference["sigma_z"] - initials["sigma_z"]
        prediction_stress = prediction["sigma_z"] - initials["sigma_z"]
        reference_pressure_peak = float(np.max(reference_pressure))
        prediction_pressure_peak = float(np.max(prediction_pressure))
        reference_stress_peak = float(np.max(np.abs(reference_stress)))
        prediction_stress_peak = float(np.max(np.abs(prediction_stress)))
        pressure_index = np.unravel_index(int(np.argmax(reference_pressure)), reference_pressure.shape)
        pressure_prediction_index = np.unravel_index(int(np.argmax(prediction_pressure)), prediction_pressure.shape)
        stress_index = np.unravel_index(int(np.argmax(np.abs(reference_stress))), reference_stress.shape)
        stress_prediction_index = np.unravel_index(int(np.argmax(np.abs(prediction_stress))), prediction_stress.shape)
        start = int(np.searchsorted(t, params.valve_close_time_s))
        p_first = first_local_peak_index(reference["P"][-1], start)
        pp_first = first_local_peak_index(prediction["P"][-1], start)
        s_first = first_local_peak_index(np.abs(reference_stress[-1]), start)
        sp_first = first_local_peak_index(np.abs(prediction_stress[-1]), start)
        rows.append({
            "case_id": case["case_id"], "split": case["split"], "test_class": case["test_class"],
            **{name: case[name] for name in PARAMETER_NAMES}, **metrics,
            "pressure_peak_relative_error": abs(prediction_pressure_peak - reference_pressure_peak) / max(abs(reference_pressure_peak), 1e-30),
            "stress_peak_relative_error": abs(prediction_stress_peak - reference_stress_peak) / max(reference_stress_peak, 1e-30),
            "reference_pressure_peak_increment_pa": reference_pressure_peak,
            "prediction_pressure_peak_increment_pa": prediction_pressure_peak,
            "reference_stress_peak_increment_pa": reference_stress_peak,
            "prediction_stress_peak_increment_pa": prediction_stress_peak,
            "reference_pressure_first_peak_time_s": float(t[p_first]),
            "prediction_pressure_first_peak_time_s": float(t[pp_first]),
            "reference_stress_first_peak_time_s": float(t[s_first]),
            "prediction_stress_first_peak_time_s": float(t[sp_first]),
            "reference_pressure_critical_location_over_L": float(x[pressure_index[0]] / params.length_m),
            "prediction_pressure_critical_location_over_L": float(x[pressure_prediction_index[0]] / params.length_m),
            "reference_stress_critical_location_over_L": float(x[stress_index[0]] / params.length_m),
            "prediction_stress_critical_location_over_L": float(x[stress_prediction_index[0]] / params.length_m),
        })
        print(f"[baseline/eval] {case['case_id']} P={metrics['P_nrmse']:.3%} stress={metrics['sigma_z_nrmse']:.3%}", flush=True)
    names = ("V_nrmse", "uz_nrmse", "P_nrmse", "sigma_z_nrmse", "pressure_peak_relative_error", "stress_peak_relative_error")
    summary = {}
    for split in ("validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            continue
        summary[split] = {
            "case_count": len(subset),
            "mean_by_metric": {name: float(np.mean([row[name] for row in subset])) for name in names},
            "maximum_by_metric": {name: max(row[name] for row in subset) for name in names},
        }
    return summary, rows


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    epoch: int,
    config_path: Path,
    history: list[dict[str, float]],
    rng: np.random.Generator,
) -> None:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.random.get_rng_state(),
        "config": str(config_path),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def train(config_path: Path, output_dir: Path, mode: str) -> dict[str, Any]:
    config = load_config(config_path)
    specification = config["training"][mode]
    run_dir = output_dir / mode
    report_path = run_dir / "model_report.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    if run_dir.exists() and not checkpoint_path.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"conventional PINN output exists without a resumable checkpoint: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(int(specification["seed"]))
    np.random.seed(int(specification["seed"]))
    device = resolve_device(specification["device"])
    design = load_design(config)
    baseline = load_baseline(design)
    model = ParametricFullDomainPINN(design, baseline, config).to(device)
    hybrid = load_hybrid_config(ROOT / config["anchor_config"])
    anchors = load_training_anchors(
        hybrid, design, baseline, device, torch.float64, int(specification["case_limit"])
    )
    anchor_by_id = {batch.case["case_id"]: batch for batch in anchors}
    cases = [batch.case for batch in anchors]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(specification["learning_rate"]))
    rng = np.random.default_rng(int(specification["seed"]) + 503)
    weights = config["loss_weights"]
    history: list[dict[str, float]] = []
    resumed_from_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        resumed_from_epoch = int(checkpoint["epoch"])
        if "numpy_rng_state" in checkpoint:
            rng.bit_generator.state = checkpoint["numpy_rng_state"]
        if "torch_rng_state" in checkpoint:
            torch.random.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(resumed_from_epoch + 1, int(specification["epochs"]) + 1):
        indices = rng.choice(len(cases), size=min(int(specification["case_batch_size"]), len(cases)), replace=False)
        optimizer.zero_grad(set_to_none=True)
        totals = {"pde": torch.zeros((), device=device), "valve_dynamic": torch.zeros((), device=device), "training_anchor": torch.zeros((), device=device)}
        for index in indices:
            case = cases[int(index)]
            params, _ = case_parameters(baseline, case, model.t_final_s)
            pde_points, right_points = sample_points(params, specification, rng, device)
            totals["pde"] = totals["pde"] + pde_loss(model, pde_points, case)
            totals["valve_dynamic"] = totals["valve_dynamic"] + valve_dynamic_loss(model, right_points, case)
            totals["training_anchor"] = totals["training_anchor"] + anchor_objective(model, anchor_by_id[case["case_id"]], config)
        for name in totals:
            totals[name] = totals[name] / len(indices)
        loss = sum(float(weights[name]) * totals[name] for name in totals)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(specification["gradient_clip"])).detach().cpu())
        optimizer.step()
        record = {"epoch": epoch, "loss_total": float(loss.detach().cpu()), **{f"loss_{name}": float(value.detach().cpu()) for name, value in totals.items()}, "gradient_norm": gradient_norm}
        history.append(record)
        if epoch == 1 or epoch % int(specification["log_every"]) == 0:
            print(f"[conventional/{mode}] epoch={epoch} total={record['loss_total']:.3e} pde={record['loss_pde']:.3e} anchor={record['loss_training_anchor']:.3e}", flush=True)
        if epoch % int(specification["checkpoint_every"]) == 0 or epoch == int(specification["epochs"]):
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch, config_path, history, rng
            )
    training_seconds = time.perf_counter() - started
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    evaluation, rows = ({}, []) if mode == "smoke" else evaluate(model, config, device)
    if rows:
        with (run_dir / "held_out_case_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    acceptance = config["acceptance"]
    passed = mode == "smoke" or (
        evaluation["validation"]["maximum_by_metric"]["P_nrmse"] <= float(acceptance["validation_pressure_nrmse"])
        and evaluation["validation"]["maximum_by_metric"]["sigma_z_nrmse"] <= float(acceptance["validation_axial_stress_nrmse"])
        and evaluation["validation"]["maximum_by_metric"]["pressure_peak_relative_error"] <= float(acceptance["validation_pressure_peak_relative_error"])
        and evaluation["validation"]["maximum_by_metric"]["stress_peak_relative_error"] <= float(acceptance["validation_stress_peak_relative_error"])
    )
    report = {
        "status": "pass" if passed else "failed",
        "mode": mode,
        "model_id": config["model_id"],
        "method": config["method"],
        "device": str(device),
        "seed": int(specification["seed"]),
        "resumed_from_epoch": resumed_from_epoch,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_case_count": len(cases),
        "case_batch_size": int(specification["case_batch_size"]),
        "anchor_vectors_per_case": int(hybrid["anchor_policy"]["anchor_vectors_per_case"]),
        "training_seconds_this_invocation": training_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "final_training_loss": history[-1],
        "evaluation": evaluation,
        "accuracy_acceptance": "passed" if passed else "failed",
        "stop_rule_applied": mode == "formal",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.config, args.output_dir, args.mode)


if __name__ == "__main__":
    main()
