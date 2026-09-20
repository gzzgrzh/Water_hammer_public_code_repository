"""Train a data-only coordinate ANN on the registered sparse FSSI labels.

The ANN receives the same five coordinates and the same sparse four-state labels
as the coordinate PINN.  It uses no governing-equation, boundary, initial-state,
characteristic-speed, or travel-time loss/feature.  Checkpoints are resumable so
an interrupted formal run is never restarted from epoch zero.
"""

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
)
from water16_reproduction.parametric_fssi_conventional_pinn import (
    evaluate,
    normalized_parameter_values,
    resolve_device,
)
from water16_reproduction.parametric_fssi_hybrid import (
    load_config as load_anchor_config,
    load_training_anchors,
)
from water16_reproduction.parametric_fssi_model import load_design
from water16_reproduction.parametric_fssi_reference import case_parameters, load_baseline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "PINN_FSSI_research_plan/configs/parametric_fssi_ann_baseline_v1.json"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/parametric_fssi_ann_baseline_v1"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("ANN config schema_version must be 1")
    if config.get("method") != "data_only_parametric_coordinate_ann":
        raise ValueError("unexpected ANN method")
    network = config["network"]
    forbidden = (
        "uses_governing_equations",
        "uses_boundary_or_initial_losses",
        "uses_hard_physical_constraints",
        "uses_characteristic_speeds",
        "uses_travel_time_features",
    )
    if any(bool(network[name]) for name in forbidden):
        raise ValueError("the ANN baseline must remain data-only")
    splits = config["evaluation"]["splits"]
    if "validation" not in splits or not set(splits).issubset({"validation", "test"}):
        raise ValueError("ANN must evaluate validation and optionally test")
    return config


class ParametricCoordinateANN(nn.Module):
    """Five-coordinate MLP trained only against sparse state labels."""

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
            normalized_parameter_values(case, self.design),
            device=points.device,
            dtype=points.dtype,
        ).expand(len(points), -1)
        inputs = torch.cat([2.0 * xi - 1.0, 2.0 * tau - 1.0, mu], dim=1)
        raw = self.network(inputs)
        pressure0 = initial_pressure_pa(params)
        area_f, area_t = cross_section_areas(params)
        initial = torch.as_tensor(
            [params.initial_velocity_m_s, 0.0, pressure0, area_f * pressure0 / area_t],
            device=points.device,
            dtype=points.dtype,
        )
        scales = torch.as_tensor(
            state_scales(params), device=points.device, dtype=points.dtype
        )
        return initial + raw * scales


def sparse_label_loss(
    model: ParametricCoordinateANN, batch, tail_fraction: float, tail_weight: float
) -> torch.Tensor:
    points = torch.cat([batch.positions, batch.times], dim=1)
    prediction = model(points, batch.case)
    squared = ((prediction - batch.targets) / batch.scales) ** 2
    weights = batch.observation_weights[None, :]
    mean_loss = torch.sum(squared * weights) / (squared.shape[0] * torch.sum(weights))
    if tail_fraction <= 0.0 or tail_weight <= 0.0:
        return mean_loss
    count = max(1, int(math.ceil(tail_fraction * squared.shape[0])))
    observed_indices = torch.nonzero(batch.observation_weights > 0.0, as_tuple=False).flatten().tolist()
    tail = torch.stack(
        [torch.topk(squared[:, state], count).values.mean() for state in observed_indices]
    ).mean()
    return mean_loss + tail_weight * tail


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    rng: np.random.Generator,
    config_path: Path,
) -> dict[str, Any]:
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
    return payload


def _restore_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[int, list[dict[str, float]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "numpy_rng_state" in checkpoint:
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
    if "torch_rng_state" in checkpoint:
        torch.random.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return int(checkpoint["epoch"]), list(checkpoint.get("history", []))


def train(config_path: Path, output_dir: Path, mode: str) -> dict[str, Any]:
    config = load_config(config_path)
    specification = config["training"][mode]
    run_dir = output_dir / mode
    report_path = run_dir / "model_report.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    if run_dir.exists() and not checkpoint_path.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"ANN output exists without a resumable checkpoint: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float64 if specification["dtype"] == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    seed = int(specification["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(specification["device"])
    design = load_design(config)
    baseline = load_baseline(design)
    model = ParametricCoordinateANN(design, baseline, config).to(device)
    anchor_config = load_anchor_config(ROOT / config["anchor_config"])
    anchors = load_training_anchors(
        anchor_config,
        design,
        baseline,
        device,
        dtype,
        int(specification["case_limit"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(specification["learning_rate"]))
    rng = np.random.default_rng(seed + 701)
    history: list[dict[str, float]] = []
    resumed_from_epoch = 0
    if checkpoint_path.exists():
        resumed_from_epoch, history = _restore_checkpoint(
            checkpoint_path, model, optimizer, rng, device
        )

    objective = config["anchor_objective"]
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(resumed_from_epoch + 1, int(specification["epochs"]) + 1):
        indices = rng.choice(
            len(anchors),
            size=min(int(specification["case_batch_size"]), len(anchors)),
            replace=False,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device, dtype=dtype)
        for index in indices:
            loss = loss + sparse_label_loss(
                model,
                anchors[int(index)],
                float(objective["tail_fraction"]),
                float(objective["tail_weight"]),
            ) / len(indices)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(specification["gradient_clip"])
            ).detach().cpu()
        )
        optimizer.step()
        row = {
            "epoch": float(epoch),
            "loss_total": float(loss.detach().cpu()),
            "loss_training_anchor": float(loss.detach().cpu()),
            "gradient_norm": gradient_norm,
        }
        history.append(row)
        if epoch == 1 or epoch == int(specification["epochs"]) or epoch % int(specification["log_every"]) == 0:
            print(
                f"[ann/{mode}] epoch={epoch} anchor={row['loss_training_anchor']:.3e}",
                flush=True,
            )
        if epoch % int(specification["checkpoint_every"]) == 0 or epoch == int(specification["epochs"]):
            torch.save(
                _checkpoint_payload(model, optimizer, epoch, history, rng, config_path),
                checkpoint_path,
            )

    training_seconds = time.perf_counter() - started
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    evaluation, rows = ({}, []) if mode == "smoke" else evaluate(model, config, device)
    if rows:
        with (run_dir / "held_out_case_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    limits = config["acceptance"]
    accuracy_pass = mode == "smoke" or (
        evaluation["validation"]["mean_by_metric"]["P_nrmse"]
        <= float(limits["validation_mean_pressure_nrmse"])
        and evaluation["validation"]["mean_by_metric"]["sigma_z_nrmse"]
        <= float(limits["validation_mean_axial_stress_nrmse"])
    )
    report = {
        "status": "pass" if accuracy_pass else "completed_below_target",
        "mode": mode,
        "model_id": config["model_id"],
        "method": config["method"],
        "device": str(device),
        "seed": seed,
        "resumed_from_epoch": resumed_from_epoch,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_case_count": len(anchors),
        "case_batch_size": int(specification["case_batch_size"]),
        "anchor_vectors_per_case": int(anchor_config["anchor_policy"]["anchor_vectors_per_case"]),
        "anchor_selection": anchor_config["anchor_policy"]["selection"],
        "uses_governing_equations": False,
        "uses_boundary_or_initial_losses": False,
        "uses_hard_physical_constraints": False,
        "training_seconds_this_invocation": training_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "final_training_loss": history[-1],
        "evaluation": evaluation,
        "accuracy_acceptance": "passed" if accuracy_pass else "failed",
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
