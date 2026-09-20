"""Three-model sparse reconstruction on the public Xu et al. (2025) MOC fields.

The development stage reads only ``train.mat``.  The published ``test.mat``
locations are loaded only in the release stage.  This benchmark is numerical
and two-state; it does not constitute an experimental four-state FSSI test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "PINN_FSSI_research_plan/outputs/external_xu2025_github_v1/repository/pinn_for_hydraulic_transients"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/xu2025_external_three_model_v1"
DATASETS = ("SFM", "TVB")
MODELS = ("ann", "coordinate_pinn", "characteristic_pinn")
SEEDS = (31041, 31053, 31059)

PIPE_LENGTH_M = 300.0
PIPE_DIAMETER_M = 0.05
WAVE_SPEED_M_S = 1000.0
GRAVITY_M_S2 = 9.806
FRICTION_FACTOR = 0.015
TIME_END_S = 6.0
HEAD_SCALE_M = 45.0
VELOCITY_SCALE_M_S = 0.42


def settings(mode: str) -> dict[str, Any]:
    if mode == "smoke":
        return {"epochs": 8, "checkpoint_interval": 4, "collocation": 48, "device": "cuda", "dtype": "float64"}
    if mode == "formal":
        return {"epochs": 1800, "checkpoint_interval": 100, "collocation": 384, "device": "cuda", "dtype": "float64"}
    if mode == "pilot":
        return {"epochs": 600, "checkpoint_interval": 50, "collocation": 256, "device": "cuda", "dtype": "float64"}
    raise ValueError(mode)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


@dataclass(frozen=True)
class FieldData:
    time_s: np.ndarray
    x_m: np.ndarray
    head_m: np.ndarray
    velocity_m_s: np.ndarray


def data_path(dataset: str, stage: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(dataset)
    name = "train.mat" if stage == "develop" else "test.mat"
    return REPOSITORY / f"Case_MOC_{dataset}" / name


def load_field(dataset: str, stage: str) -> FieldData:
    raw = scipy.io.loadmat(data_path(dataset, stage))
    if stage == "develop":
        x_key, h_key, v_key = "X_star", "H_star", "V_star"
    elif stage == "release":
        x_key, h_key, v_key = "X_test", "H_test", "V_test"
    else:
        raise ValueError(stage)
    time_s = np.asarray(raw["t"], dtype=float).reshape(-1)
    x_m = np.asarray(raw[x_key], dtype=float).reshape(-1)
    head_m = np.asarray(raw[h_key], dtype=float)
    velocity_m_s = np.asarray(raw[v_key], dtype=float)
    expected = (time_s.size, x_m.size)
    if head_m.shape != expected or velocity_m_s.shape != expected:
        raise ValueError(f"unexpected {dataset} {stage} field shape")
    return FieldData(time_s, x_m, head_m, velocity_m_s)


def registered_time_indices(time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 96 fitting and 24 validation times without looking at responses."""
    candidates = np.unique(np.rint(np.linspace(0, time_s.size - 1, 120)).astype(int))
    if candidates.size != 120:
        raise RuntimeError("registered time grid is not unique")
    validation = candidates[4::5]
    fitting = np.setdiff1d(candidates, validation, assume_unique=True)
    if fitting.size != 96 or validation.size != 24:
        raise RuntimeError("registered 96/24 split was not constructed")
    return fitting, validation


def initial_head_line(field: FieldData) -> tuple[float, float]:
    slope, intercept = np.polyfit(field.x_m, field.head_m[0], 1)
    return float(slope), float(intercept)


class FieldNetwork(nn.Module):
    def __init__(self, model_name: str, initial_slope: float, initial_intercept: float, width: int = 64, depth: int = 4) -> None:
        super().__init__()
        if model_name not in MODELS:
            raise ValueError(model_name)
        self.model_name = model_name
        self.initial_slope = initial_slope
        self.initial_intercept = initial_intercept
        feature_count = 2 if model_name != "characteristic_pinn" else 20
        layers: list[nn.Module] = [nn.Linear(feature_count, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 2))
        self.network = nn.Sequential(*layers)

    def features(self, x_m: torch.Tensor, time_s: torch.Tensor) -> torch.Tensor:
        x_norm = 2.0 * x_m / PIPE_LENGTH_M - 1.0
        t_norm = 2.0 * time_s / TIME_END_S - 1.0
        values = [x_norm, t_norm]
        if self.model_name == "characteristic_pinn":
            characteristic_period = 4.0 * PIPE_LENGTH_M / WAVE_SPEED_M_S
            minus = (time_s - x_m / WAVE_SPEED_M_S) / characteristic_period
            plus = (time_s + x_m / WAVE_SPEED_M_S) / characteristic_period
            values.extend([minus, plus])
            for harmonic in (1.0, 2.0, 3.0, 4.0):
                values.extend(
                    [
                        torch.sin(2.0 * math.pi * harmonic * minus),
                        torch.cos(2.0 * math.pi * harmonic * minus),
                        torch.sin(2.0 * math.pi * harmonic * plus),
                        torch.cos(2.0 * math.pi * harmonic * plus),
                    ]
                )
        return torch.cat(values, dim=1)

    def forward(self, x_m: torch.Tensor, time_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(self.features(x_m, time_s))
        initial_head = self.initial_intercept + self.initial_slope * x_m
        head = initial_head + HEAD_SCALE_M * raw[:, 0:1]
        velocity = 0.412529612 + VELOCITY_SCALE_M_S * raw[:, 1:2]
        return head, velocity


def tensors_for_indices(field: FieldData, indices: np.ndarray, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    x_grid, t_grid = np.meshgrid(field.x_m, field.time_s[indices])
    head = field.head_m[indices, :]
    velocity = field.velocity_m_s[indices, :]
    return (
        torch.as_tensor(x_grid.reshape(-1, 1), device=device, dtype=dtype),
        torch.as_tensor(t_grid.reshape(-1, 1), device=device, dtype=dtype),
        torch.as_tensor(head.reshape(-1, 1), device=device, dtype=dtype),
        torch.as_tensor(velocity.reshape(-1, 1), device=device, dtype=dtype),
    )


def normalized_data_loss(model: FieldNetwork, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    x_m, time_s, head, velocity = batch
    predicted_head, predicted_velocity = model(x_m, time_s)
    return torch.mean(((predicted_head - head) / HEAD_SCALE_M) ** 2) + torch.mean(
        ((predicted_velocity - velocity) / VELOCITY_SCALE_M_S) ** 2
    )


def physics_loss(model: FieldNetwork, samples: int, generator: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if model.model_name == "ann":
        return torch.zeros((), device=device, dtype=dtype)
    x_m = torch.rand((samples, 1), generator=generator, device=device, dtype=dtype) * PIPE_LENGTH_M
    time_s = 0.001 + torch.rand((samples, 1), generator=generator, device=device, dtype=dtype) * (TIME_END_S - 0.001)
    x_m.requires_grad_(True)
    time_s.requires_grad_(True)
    head, velocity = model(x_m, time_s)
    head_t = torch.autograd.grad(head, time_s, torch.ones_like(head), create_graph=True)[0]
    head_x = torch.autograd.grad(head, x_m, torch.ones_like(head), create_graph=True)[0]
    velocity_t = torch.autograd.grad(velocity, time_s, torch.ones_like(velocity), create_graph=True)[0]
    velocity_x = torch.autograd.grad(velocity, x_m, torch.ones_like(velocity), create_graph=True)[0]
    momentum = velocity_t + GRAVITY_M_S2 * head_x + FRICTION_FACTOR * velocity * torch.abs(velocity) / (2.0 * PIPE_DIAMETER_M)
    continuity = head_t + WAVE_SPEED_M_S**2 / GRAVITY_M_S2 * velocity_x
    return torch.mean((momentum * TIME_END_S / VELOCITY_SCALE_M_S) ** 2) + torch.mean(
        (continuity * TIME_END_S / HEAD_SCALE_M) ** 2
    )


def initial_loss(model: FieldNetwork, samples: int, generator: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if model.model_name == "ann":
        return torch.zeros((), device=device, dtype=dtype)
    x_m = torch.rand((samples, 1), generator=generator, device=device, dtype=dtype) * PIPE_LENGTH_M
    time_s = torch.zeros_like(x_m)
    head, velocity = model(x_m, time_s)
    target_head = model.initial_intercept + model.initial_slope * x_m
    target_velocity = torch.full_like(velocity, 0.412529612)
    return torch.mean(((head - target_head) / HEAD_SCALE_M) ** 2) + torch.mean(
        ((velocity - target_velocity) / VELOCITY_SCALE_M_S) ** 2
    )


def predict_field(model: FieldNetwork, field: FieldData, device: torch.device, dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray]:
    x_grid, t_grid = np.meshgrid(field.x_m, field.time_s)
    x_m = torch.as_tensor(x_grid.reshape(-1, 1), device=device, dtype=dtype)
    time_s = torch.as_tensor(t_grid.reshape(-1, 1), device=device, dtype=dtype)
    chunks_h, chunks_v = [], []
    with torch.no_grad():
        for start in range(0, x_m.shape[0], 8192):
            h, v = model(x_m[start : start + 8192], time_s[start : start + 8192])
            chunks_h.append(h.cpu().numpy())
            chunks_v.append(v.cpu().numpy())
    shape = (field.time_s.size, field.x_m.size)
    return np.concatenate(chunks_h).reshape(shape), np.concatenate(chunks_v).reshape(shape)


def variable_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - reference
    dynamic_range = max(float(np.ptp(reference)), 1.0e-12)
    correlation = float(np.corrcoef(reference, prediction)[0, 1]) if np.std(prediction) > 0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nrmse_dynamic_range": float(np.sqrt(np.mean(error**2)) / dynamic_range),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "peak_absolute_error": float(abs(np.max(prediction) - np.max(reference))),
        "trough_absolute_error": float(abs(np.min(prediction) - np.min(reference))),
        "correlation": correlation,
    }


def validation_score(model: FieldNetwork, field: FieldData, validation_indices: np.ndarray, device: torch.device, dtype: torch.dtype) -> float:
    batch = tensors_for_indices(field, validation_indices, device, dtype)
    with torch.no_grad():
        h, v = model(batch[0], batch[1])
    h_nrmse = torch.sqrt(torch.mean((h - batch[2]) ** 2)) / HEAD_SCALE_M
    v_nrmse = torch.sqrt(torch.mean((v - batch[3]) ** 2)) / VELOCITY_SCALE_M_S
    return float((h_nrmse + v_nrmse).cpu() / 2.0)


def run_directory(output_root: Path, mode: str, dataset: str, model_name: str, seed: int) -> Path:
    return output_root / mode / "development" / dataset / model_name / f"seed_{seed}"


def train_one(dataset: str, model_name: str, seed: int, mode: str, output_root: Path) -> dict[str, Any]:
    run_dir = run_directory(output_root, mode, dataset, model_name, seed)
    report_path = run_dir / "development_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = settings(mode)
    device = resolve_device(cfg["device"])
    dtype = torch.float64 if cfg["dtype"] == "float64" else torch.float32
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1701)
    field = load_field(dataset, "develop")
    fitting_indices, validation_indices = registered_time_indices(field.time_s)
    slope, intercept = initial_head_line(field)
    model = FieldNetwork(model_name, slope, intercept).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=8.0e-4)
    fit_batch = tensors_for_indices(field, fitting_indices, device, dtype)
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_data = normalized_data_loss(model, fit_batch)
        loss_pde = physics_loss(model, int(cfg["collocation"]), generator, device, dtype)
        loss_initial = initial_loss(model, 96, generator, device, dtype)
        loss = loss_data + 0.0025 * loss_pde + 0.10 * loss_initial
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        optimizer.step()
        if epoch == 1 or epoch % int(cfg["checkpoint_interval"]) == 0 or epoch == int(cfg["epochs"]):
            score = validation_score(model, field, validation_indices, device, dtype)
            row = {
                "epoch": epoch,
                "loss_total": float(loss.detach().cpu()),
                "loss_data": float(loss_data.detach().cpu()),
                "loss_pde": float(loss_pde.detach().cpu()),
                "loss_initial": float(loss_initial.detach().cpu()),
                "validation_score": score,
            }
            history.append(row)
            if score < best_score:
                best_score = score
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            checkpoint = run_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            if not checkpoint.exists():
                torch.save({"epoch": epoch, "state_dict": model.state_dict(), "best_score": best_score, "best_state": best_state}, checkpoint)
            print(f"[xu/{dataset}/{model_name}/seed={seed}] epoch={epoch} loss={row['loss_total']:.3e} val={score:.3%}", flush=True)
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    best_path = run_dir / "best_model.pt"
    if not best_path.exists():
        torch.save(
            {
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "initial_slope": slope,
                "initial_intercept": intercept,
                "best_validation_score": best_score,
                "state_dict": best_state,
            },
            best_path,
        )
    report = {
        "status": "pass",
        "stage": "develop",
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "mode": mode,
        "development_file_loaded": str(data_path(dataset, "develop").relative_to(ROOT)),
        "sealed_test_file_loaded": False,
        "training_positions_m": field.x_m.tolist(),
        "fitting_times_per_position": int(fitting_indices.size),
        "validation_times_per_position": int(validation_indices.size),
        "fitting_state_vectors": int(fitting_indices.size * field.x_m.size),
        "best_validation_score": best_score,
        "history": history,
        "runtime_s": time.perf_counter() - started,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def plot_development(reports: list[dict[str, Any]], target: Path) -> None:
    colors = {"ann": "#777777", "coordinate_pinn": "#E47832", "characteristic_pinn": "#2878B5"}
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        for model_name in MODELS:
            selected = [row for row in reports if row["dataset"] == dataset and row["model"] == model_name]
            if not selected:
                continue
            for index, row in enumerate(selected):
                epochs = [point["epoch"] for point in row["history"]]
                scores = [100.0 * point["validation_score"] for point in row["history"]]
                axis.plot(epochs, scores, color=colors[model_name], alpha=0.28, lw=1.2)
            common_epochs = [point["epoch"] for point in selected[0]["history"]]
            mean_scores = np.mean([[100.0 * point["validation_score"] for point in row["history"]] for row in selected], axis=0)
            axis.plot(common_epochs, mean_scores, color=colors[model_name], lw=2.7, label=model_name.replace("_", " "))
        axis.set_title(dataset)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out-time score (%)")
    axes[1].legend(frameon=False)
    figure.suptitle("Xu 2025 public numerical benchmark: development convergence")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(target.with_suffix(f".{suffix}"), dpi=320 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def develop(output_root: Path, mode: str, requested_models: tuple[str, ...], requested_seeds: tuple[int, ...]) -> dict[str, Any]:
    mode_dir = output_root / mode
    matrix_path = mode_dir / "development_matrix_report.json"
    if matrix_path.exists():
        return json.loads(matrix_path.read_text(encoding="utf-8"))
    reports = []
    for dataset in DATASETS:
        for model_name in requested_models:
            for seed in requested_seeds:
                reports.append(train_one(dataset, model_name, seed, mode, output_root))
                progress_path = mode_dir / f"development_progress_{len(reports):02d}.json"
                if not progress_path.exists():
                    progress_path.write_text(
                        json.dumps({"completed": [{key: row[key] for key in ("dataset", "model", "seed", "best_validation_score")} for row in reports]}, indent=2),
                        encoding="utf-8",
                    )
    report = {"status": "pass", "stage": "develop", "mode": mode, "runs": reports, "sealed_test_opened": False}
    mode_dir.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_development(reports, mode_dir / "F37_xu2025_development_convergence")
    return report


def load_best_model(path: Path, device: torch.device, dtype: torch.dtype) -> FieldNetwork:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = FieldNetwork(payload["model"], payload["initial_slope"], payload["initial_intercept"]).to(device=device, dtype=dtype)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def release(output_root: Path, mode: str) -> dict[str, Any]:
    development_path = output_root / mode / "development_matrix_report.json"
    if not development_path.exists():
        raise FileNotFoundError("development stage must finish before release")
    release_dir = output_root / mode / "sealed_release"
    report_path = release_dir / "sealed_test_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    release_dir.mkdir(parents=True, exist_ok=True)
    cfg = settings(mode)
    device = resolve_device(cfg["device"])
    dtype = torch.float64 if cfg["dtype"] == "float64" else torch.float32
    rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, str, int], tuple[FieldData, np.ndarray, np.ndarray]] = {}
    for dataset in DATASETS:
        field = load_field(dataset, "release")
        for model_name in MODELS:
            for seed in SEEDS if mode == "formal" else (SEEDS[0],):
                model_path = run_directory(output_root, mode, dataset, model_name, seed) / "best_model.pt"
                model = load_best_model(model_path, device, dtype)
                predicted_head, predicted_velocity = predict_field(model, field, device, dtype)
                curves[(dataset, model_name, seed)] = (field, predicted_head, predicted_velocity)
                for sensor_index, x_m in enumerate(field.x_m):
                    for variable, reference, prediction in (
                        ("head", field.head_m[:, sensor_index], predicted_head[:, sensor_index]),
                        ("velocity", field.velocity_m_s[:, sensor_index], predicted_velocity[:, sensor_index]),
                    ):
                        rows.append(
                            {
                                "dataset": dataset,
                                "model": model_name,
                                "seed": seed,
                                "test_position_m": float(x_m),
                                "variable": variable,
                                **variable_metrics(reference, prediction),
                            }
                        )
    table_path = release_dir / "T13_xu2025_sealed_test_metrics.csv"
    with table_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = {"ann": "#777777", "coordinate_pinn": "#E47832", "characteristic_pinn": "#2878B5"}
    primary_seed = SEEDS[0]
    for figure_number, dataset in ((38, "SFM"), (39, "TVB")):
        field = curves[(dataset, "ann", primary_seed)][0]
        figure, axes = plt.subplots(2, 2, figsize=(13.0, 7.2), sharex=True)
        for column, x_m in enumerate(field.x_m):
            axes[0, column].plot(field.time_s, field.head_m[:, column], color="black", lw=1.8, label="reference")
            axes[1, column].plot(field.time_s, field.velocity_m_s[:, column], color="black", lw=1.8, label="reference")
            for model_name in MODELS:
                _, head, velocity = curves[(dataset, model_name, primary_seed)]
                axes[0, column].plot(field.time_s, head[:, column], color=colors[model_name], lw=1.25, label=model_name.replace("_", " "))
                axes[1, column].plot(field.time_s, velocity[:, column], color=colors[model_name], lw=1.25, label=model_name.replace("_", " "))
            axes[0, column].set_title(f"{dataset}, x={x_m:.0f} m")
            axes[1, column].set_xlabel("Time (s)")
            for axis in axes[:, column]:
                axis.grid(alpha=0.18)
        axes[0, 0].set_ylabel("Head (m)")
        axes[1, 0].set_ylabel("Velocity (m/s)")
        axes[0, 1].legend(frameon=False, ncol=2, fontsize=8)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(release_dir / f"F{figure_number}_xu2025_{dataset.lower()}_sealed_histories.{suffix}", dpi=320 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for axis, dataset in zip(axes, DATASETS):
        summary[dataset] = {}
        x = np.arange(len(MODELS))
        head_values, velocity_values = [], []
        for model_name in MODELS:
            selected = [row for row in rows if row["dataset"] == dataset and row["model"] == model_name]
            head_nrmse = float(np.mean([row["nrmse_dynamic_range"] for row in selected if row["variable"] == "head"]))
            velocity_nrmse = float(np.mean([row["nrmse_dynamic_range"] for row in selected if row["variable"] == "velocity"]))
            head_values.append(100.0 * head_nrmse)
            velocity_values.append(100.0 * velocity_nrmse)
            summary[dataset][model_name] = {
                "mean_head_nrmse_dynamic_range": head_nrmse,
                "mean_velocity_nrmse_dynamic_range": velocity_nrmse,
                "mean_head_correlation": float(np.nanmean([row["correlation"] for row in selected if row["variable"] == "head"])),
                "mean_velocity_correlation": float(np.nanmean([row["correlation"] for row in selected if row["variable"] == "velocity"])),
            }
        axis.bar(x - 0.18, head_values, 0.36, label="head", color="#4C78A8")
        axis.bar(x + 0.18, velocity_values, 0.36, label="velocity", color="#F58518")
        axis.set_xticks(x, [name.replace("_", "\n") for name in MODELS])
        axis.set_title(dataset)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Sealed-location NRMSE (%)")
    axes[1].legend(frameon=False)
    figure.suptitle("Xu 2025 public MOC fields: sparse spatial reconstruction")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(release_dir / f"F40_xu2025_sealed_error_summary.{suffix}", dpi=320 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    report = {
        "status": "pass",
        "stage": "sealed_release",
        "mode": mode,
        "test_files_loaded": [str(data_path(dataset, "release").relative_to(ROOT)) for dataset in DATASETS],
        "test_positions_m": {dataset: load_field(dataset, "release").x_m.tolist() for dataset in DATASETS},
        "summary": summary,
        "metrics_table": table_path.name,
        "figures": [
            "F38_xu2025_sfm_sealed_histories.png",
            "F39_xu2025_tvb_sealed_histories.png",
            "F40_xu2025_sealed_error_summary.png",
        ],
        "scope_limit": "Independent public numerical two-state MOC benchmark; not experimental or four-state FSSI validation.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "release"), default="develop")
    parser.add_argument("--mode", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "develop":
        requested_models = tuple(value.strip() for value in args.models.split(",") if value.strip())
        requested_seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
        unknown = set(requested_models) - set(MODELS)
        if unknown:
            raise ValueError(f"unknown models: {sorted(unknown)}")
        result = develop(args.output_dir, args.mode, requested_models, requested_seeds)
    else:
        result = release(args.output_dir, args.mode)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
