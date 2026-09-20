"""Sparse-sensor external validation on the public Perugia pipe network.

The development stage is structurally prevented from loading the three sealed
test sensors.  ANN, coordinate PINN, and characteristic PINN use identical
pressure labels.  PINN flow is a latent state constrained by the early-time
frictionless network water-hammer equations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "PINN_FSSI_research_plan/outputs/external_perugia_zenodo_5535442/5u_q015.mat"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/perugia_external_three_model_v1"

G = 9.80665
Q0 = 0.00015  # q015: 0.15 L/s at the active end user.
TIME_SCALE_S = 2.0
HEAD_SCALE_M = 20.0
FLOW_SCALE_M3_S = Q0

TRAIN_SENSORS = ("h5u", "h5", "h54", "h56", "h58", "h32")
VALIDATION_SENSORS = ("h4", "h6")
SEALED_TEST_SENSORS = ("h8", "h7", "h1")
ALL_SENSORS = TRAIN_SENSORS + VALIDATION_SENSORS + SEALED_TEST_SENSORS
MODEL_NAMES = ("ann", "coordinate_pinn", "characteristic_pinn")
SEEDS = (29041, 29053, 29059)


@dataclass(frozen=True)
class Edge:
    name: str
    start: str
    end: str
    length_m: float
    diameter_m: float
    wave_speed_m_s: float

    @property
    def area_m2(self) -> float:
        return math.pi * self.diameter_m**2 / 4.0


@dataclass(frozen=True)
class SensorLocation:
    name: str
    edge: int
    coordinate_m: float


EDGES = (
    Edge("R-3", "R", "3", 42.3, 0.0933, 398.82),
    Edge("3-4", "3", "4", 100.0, 0.0638, 387.89),
    Edge("3-6", "3", "6", 100.0, 0.0638, 387.89),
    Edge("4-5", "4", "5", 100.0, 0.0638, 387.89),
    Edge("6-5", "6", "5", 100.0, 0.0638, 387.89),
    Edge("4-7", "4", "7", 100.0, 0.0426, 379.81),
    Edge("7-8", "7", "8", 100.0, 0.0426, 379.81),
    Edge("5-8", "5", "8", 100.0, 0.0426, 379.81),
    Edge("5-5u", "5", "5u", 23.6, 0.0200, 455.91),
)

SENSORS = {
    "h1": SensorLocation("h1", 0, 0.0),
    "h32": SensorLocation("h32", 0, 33.9),
    "h4": SensorLocation("h4", 1, 100.0),
    "h5": SensorLocation("h5", 8, 0.0),
    "h54": SensorLocation("h54", 3, 99.0),
    "h56": SensorLocation("h56", 4, 99.0),
    "h58": SensorLocation("h58", 7, 1.0),
    "h5u": SensorLocation("h5u", 8, 23.6),
    "h6": SensorLocation("h6", 2, 100.0),
    "h7": SensorLocation("h7", 5, 100.0),
    "h8": SensorLocation("h8", 7, 100.0),
}

INTERNAL_NODES = ("3", "4", "5", "6", "7", "8")


def variables_for_stage(stage: str) -> tuple[str, ...]:
    if stage == "develop":
        return ("t",) + TRAIN_SENSORS + VALIDATION_SENSORS
    if stage == "release":
        return ("t",) + SEALED_TEST_SENSORS
    raise ValueError(stage)


def load_stage_data(path: Path, stage: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    allowed = variables_for_stage(stage)
    raw = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=list(allowed))
    time_vector = np.asarray(raw["t"], dtype=float).ravel()
    signals = {
        name: np.asarray(raw[name], dtype=float).ravel()
        for name in allowed
        if name != "t"
    }
    for name, values in signals.items():
        if values.shape != time_vector.shape:
            raise ValueError(f"shape mismatch for {name}")
    return time_vector, signals


def sparse_time_indices(time_vector: np.ndarray, labels_per_sensor: int = 96) -> np.ndarray:
    if labels_per_sensor != 96:
        raise ValueError("the registered external experiment currently requires 96 labels per sensor")
    requested = np.concatenate(
        [
            np.linspace(-0.20, 0.0, 8, endpoint=False),
            np.linspace(0.0, 1.0, 56, endpoint=False),
            np.linspace(1.0, 2.0, 24, endpoint=False),
            np.linspace(2.0, 3.0, 8, endpoint=True),
        ]
    )
    indices = np.asarray([int(np.argmin(np.abs(time_vector - value))) for value in requested], dtype=int)
    if len(indices) != labels_per_sensor or len(np.unique(indices)) != labels_per_sensor:
        raise RuntimeError("could not create 96 unique registered time labels")
    return indices


def baseline_disturbance(time_vector: np.ndarray, signal: np.ndarray) -> np.ndarray:
    baseline = float(np.mean(signal[(time_vector >= -0.8) & (time_vector <= -0.05)]))
    return signal - baseline


def node_travel_times() -> dict[str, float]:
    nodes = sorted({edge.start for edge in EDGES} | {edge.end for edge in EDGES})
    times = {node: float("inf") for node in nodes}
    times["5u"] = 0.0
    unvisited = set(nodes)
    while unvisited:
        node = min(unvisited, key=lambda value: times[value])
        unvisited.remove(node)
        for edge in EDGES:
            if edge.start == node:
                neighbour = edge.end
            elif edge.end == node:
                neighbour = edge.start
            else:
                continue
            candidate = times[node] + edge.length_m / edge.wave_speed_m_s
            if candidate < times[neighbour]:
                times[neighbour] = candidate
    return times


NODE_TRAVEL_TIMES = node_travel_times()


def incident_edges(node: str) -> list[tuple[int, float, float]]:
    result = []
    for index, edge in enumerate(EDGES):
        if edge.start == node:
            result.append((index, 0.0, 1.0))
        if edge.end == node:
            result.append((index, edge.length_m, -1.0))
    return result


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


class NetworkSurrogate(nn.Module):
    def __init__(self, model_name: str, width: int = 64, layers: int = 4) -> None:
        super().__init__()
        if model_name not in MODEL_NAMES:
            raise ValueError(model_name)
        self.model_name = model_name
        feature_count = 2 + len(EDGES)
        if model_name == "characteristic_pinn":
            feature_count += 6
        output_count = 1 if model_name == "ann" else 2
        modules: list[nn.Module] = [nn.Linear(feature_count, width), nn.Tanh()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(width, width), nn.Tanh()])
        modules.append(nn.Linear(width, output_count))
        self.network = nn.Sequential(*modules)
        self.register_buffer("lengths", torch.as_tensor([edge.length_m for edge in EDGES]))
        self.register_buffer("speeds", torch.as_tensor([edge.wave_speed_m_s for edge in EDGES]))
        self.register_buffer("tau_start", torch.as_tensor([NODE_TRAVEL_TIMES[edge.start] for edge in EDGES]))
        self.register_buffer("tau_end", torch.as_tensor([NODE_TRAVEL_TIMES[edge.end] for edge in EDGES]))

    def travel_time(self, edge_index: torch.Tensor, coordinate_m: torch.Tensor) -> torch.Tensor:
        lengths = self.lengths[edge_index].unsqueeze(1)
        speeds = self.speeds[edge_index].unsqueeze(1)
        from_start = self.tau_start[edge_index].unsqueeze(1) + coordinate_m / speeds
        from_end = self.tau_end[edge_index].unsqueeze(1) + (lengths - coordinate_m) / speeds
        return torch.minimum(from_start, from_end)

    def forward(
        self,
        edge_index: torch.Tensor,
        coordinate_m: torch.Tensor,
        time_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        lengths = self.lengths[edge_index].unsqueeze(1)
        local = 2.0 * coordinate_m / lengths - 1.0
        time_coordinate = 2.0 * time_s / TIME_SCALE_S - 1.0
        one_hot = torch.nn.functional.one_hot(edge_index, num_classes=len(EDGES)).to(coordinate_m.dtype)
        features = [local, time_coordinate, one_hot]
        if self.model_name == "characteristic_pinn":
            tau = self.travel_time(edge_index, coordinate_m)
            minus = (time_s - tau) / TIME_SCALE_S
            plus = (time_s + tau) / TIME_SCALE_S
            features.extend(
                [minus, plus, torch.sin(math.pi * minus), torch.cos(math.pi * minus), torch.sin(math.pi * plus), torch.cos(math.pi * plus)]
            )
        raw = self.network(torch.cat(features, dim=1))
        if self.model_name == "ann":
            return HEAD_SCALE_M * raw[:, 0:1], None
        head = HEAD_SCALE_M * raw[:, 0:1]
        flow = FLOW_SCALE_M3_S * raw[:, 1:2]
        return head, flow


@dataclass
class PressureBatch:
    sensors: list[str]
    edge_index: torch.Tensor
    coordinate_m: torch.Tensor
    time_s: torch.Tensor
    target_head_m: torch.Tensor
    sensor_scale_m: torch.Tensor


def build_pressure_batch(
    time_vector: np.ndarray,
    signals: dict[str, np.ndarray],
    sensors: Iterable[str],
    device: torch.device,
    dtype: torch.dtype,
    sparse: bool,
) -> PressureBatch:
    sensor_names = list(sensors)
    if any(name not in signals for name in sensor_names):
        raise PermissionError("requested sensor was not loaded for this stage")
    if sparse:
        time_indices = sparse_time_indices(time_vector)
    else:
        event_indices = np.flatnonzero((time_vector >= 0.0) & (time_vector <= 2.0))
        time_indices = event_indices[::8]
        if event_indices[-1] not in time_indices:
            time_indices = np.append(time_indices, event_indices[-1])
    edges: list[int] = []
    coordinates: list[float] = []
    times: list[float] = []
    targets: list[float] = []
    scales: list[float] = []
    rows: list[str] = []
    for sensor in sensor_names:
        disturbance = baseline_disturbance(time_vector, signals[sensor])
        registered_event = disturbance[(time_vector >= 0.0) & (time_vector <= 2.0)]
        sensor_scale = max(float(np.ptp(registered_event)), 0.05)
        location = SENSORS[sensor]
        for index in time_indices:
            edges.append(location.edge)
            coordinates.append(location.coordinate_m)
            times.append(float(time_vector[index]))
            targets.append(float(disturbance[index]))
            scales.append(sensor_scale)
            rows.append(sensor)
    return PressureBatch(
        sensors=rows,
        edge_index=torch.as_tensor(edges, device=device, dtype=torch.long),
        coordinate_m=torch.as_tensor(coordinates, device=device, dtype=dtype).unsqueeze(1),
        time_s=torch.as_tensor(times, device=device, dtype=dtype).unsqueeze(1),
        target_head_m=torch.as_tensor(targets, device=device, dtype=dtype).unsqueeze(1),
        sensor_scale_m=torch.as_tensor(scales, device=device, dtype=dtype).unsqueeze(1),
    )


def data_loss(model: NetworkSurrogate, batch: PressureBatch) -> torch.Tensor:
    predicted, _ = model(batch.edge_index, batch.coordinate_m, batch.time_s)
    return torch.mean(((predicted - batch.target_head_m) / batch.sensor_scale_m) ** 2)


def pde_loss(model: NetworkSurrogate, samples_per_edge: int, generator: torch.Generator) -> torch.Tensor:
    if model.model_name == "ann":
        return torch.zeros((), device=model.lengths.device, dtype=model.lengths.dtype)
    losses = []
    for edge_index, edge in enumerate(EDGES):
        coordinate = torch.rand((samples_per_edge, 1), generator=generator, device=model.lengths.device, dtype=model.lengths.dtype) * edge.length_m
        time_s = 0.002 + torch.rand((samples_per_edge, 1), generator=generator, device=model.lengths.device, dtype=model.lengths.dtype) * (TIME_SCALE_S - 0.002)
        coordinate.requires_grad_(True)
        time_s.requires_grad_(True)
        indices = torch.full((samples_per_edge,), edge_index, device=model.lengths.device, dtype=torch.long)
        head, flow = model(indices, coordinate, time_s)
        assert flow is not None
        head_t = torch.autograd.grad(head, time_s, torch.ones_like(head), create_graph=True)[0]
        head_s = torch.autograd.grad(head, coordinate, torch.ones_like(head), create_graph=True)[0]
        flow_t = torch.autograd.grad(flow, time_s, torch.ones_like(flow), create_graph=True)[0]
        flow_s = torch.autograd.grad(flow, coordinate, torch.ones_like(flow), create_graph=True)[0]
        residual_h = (head_t + edge.wave_speed_m_s**2 / (G * edge.area_m2) * flow_s) * TIME_SCALE_S / HEAD_SCALE_M
        residual_q = (flow_t + G * edge.area_m2 * head_s) * TIME_SCALE_S / FLOW_SCALE_M3_S
        losses.append(torch.mean(residual_h**2) + torch.mean(residual_q**2))
    return torch.stack(losses).mean()


def constraint_losses(
    model: NetworkSurrogate,
    samples: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model.model_name == "ann":
        zero = torch.zeros((), device=model.lengths.device, dtype=model.lengths.dtype)
        return zero, zero, zero
    device = model.lengths.device
    dtype = model.lengths.dtype
    times = 0.002 + torch.rand((samples, 1), generator=generator, device=device, dtype=dtype) * (TIME_SCALE_S - 0.002)
    node_terms = []
    for node in INTERNAL_NODES:
        heads = []
        signed_flows = []
        for edge_index, coordinate, sign in incident_edges(node):
            indices = torch.full((samples,), edge_index, device=device, dtype=torch.long)
            positions = torch.full((samples, 1), coordinate, device=device, dtype=dtype)
            head, flow = model(indices, positions, times)
            assert flow is not None
            heads.append(head)
            signed_flows.append(sign * flow)
        for head in heads[1:]:
            node_terms.append(torch.mean(((head - heads[0]) / HEAD_SCALE_M) ** 2))
        node_terms.append(torch.mean((torch.stack(signed_flows).sum(dim=0) / FLOW_SCALE_M3_S) ** 2))
    node_loss = torch.stack(node_terms).mean()

    reservoir_edge = torch.zeros(samples, device=device, dtype=torch.long)
    reservoir_s = torch.zeros((samples, 1), device=device, dtype=dtype)
    reservoir_head, _ = model(reservoir_edge, reservoir_s, times)
    valve_edge = torch.full((samples,), 8, device=device, dtype=torch.long)
    valve_s = torch.full((samples, 1), EDGES[8].length_m, device=device, dtype=dtype)
    _, valve_flow = model(valve_edge, valve_s, times)
    assert valve_flow is not None
    boundary_loss = torch.mean((reservoir_head / HEAD_SCALE_M) ** 2) + torch.mean(((valve_flow + Q0) / FLOW_SCALE_M3_S) ** 2)

    edge_index = torch.arange(len(EDGES), device=device, dtype=torch.long).repeat_interleave(samples)
    coordinate = torch.cat(
        [
            torch.rand((samples, 1), generator=generator, device=device, dtype=dtype) * edge.length_m
            for edge in EDGES
        ],
        dim=0,
    )
    initial_time = torch.zeros_like(coordinate)
    initial_head, initial_flow = model(edge_index, coordinate, initial_time)
    assert initial_flow is not None
    initial_loss = torch.mean((initial_head / HEAD_SCALE_M) ** 2) + torch.mean((initial_flow / FLOW_SCALE_M3_S) ** 2)
    return node_loss, boundary_loss, initial_loss


def metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    time_vector: np.ndarray,
    baseline_noise_sigma: float,
) -> dict[str, float]:
    scale = max(float(np.ptp(reference)), 1e-12)
    rmse = float(np.sqrt(np.mean((prediction - reference) ** 2)))
    ref_positive = float(np.max(reference))
    pred_positive = float(np.max(prediction))
    ref_negative = float(np.min(reference))
    pred_negative = float(np.min(prediction))
    peak_index_reference = int(np.argmax(reference))
    peak_index_prediction = int(np.argmax(prediction))
    baseline_sigma = max(float(baseline_noise_sigma), 1e-12)
    arrival_threshold = max(5.0 * baseline_sigma, 0.03 * float(np.max(np.abs(reference))))

    def first_arrival(values: np.ndarray) -> float:
        indices = np.flatnonzero(np.abs(values) >= arrival_threshold)
        return float(time_vector[indices[0]]) if len(indices) else float("nan")

    reference_arrival = first_arrival(reference)
    prediction_arrival = first_arrival(prediction)
    if np.isfinite(reference_arrival) and np.isfinite(prediction_arrival):
        arrival_error = abs(prediction_arrival - reference_arrival)
    elif np.isfinite(reference_arrival):
        arrival_error = float(time_vector[-1] - time_vector[0])
    else:
        arrival_error = float("nan")
    correlation = float(np.corrcoef(reference, prediction)[0, 1]) if np.std(prediction) > 0 else 0.0
    return {
        "rmse_m": rmse,
        "nrmse_dynamic_range": rmse / scale,
        "positive_peak_absolute_error_m": abs(pred_positive - ref_positive),
        "positive_peak_relative_error": abs(pred_positive - ref_positive) / max(abs(ref_positive), 1e-12),
        "negative_peak_absolute_error_m": abs(pred_negative - ref_negative),
        "negative_peak_relative_error": abs(pred_negative - ref_negative) / max(abs(ref_negative), 1e-12),
        "positive_peak_time_absolute_error_s": abs(float(time_vector[peak_index_prediction] - time_vector[peak_index_reference])),
        "arrival_threshold_m": arrival_threshold,
        "reference_first_arrival_s": reference_arrival,
        "prediction_first_arrival_s": prediction_arrival,
        "first_arrival_absolute_error_s": arrival_error,
        "correlation": correlation,
        "rmse_over_baseline_noise": rmse / baseline_sigma,
    }


@torch.no_grad()
def predict_sensor(
    model: NetworkSurrogate,
    sensor: str,
    time_vector: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    location = SENSORS[sensor]
    count = len(time_vector)
    edge_index = torch.full((count,), location.edge, device=device, dtype=torch.long)
    coordinate = torch.full((count, 1), location.coordinate_m, device=device, dtype=dtype)
    times = torch.as_tensor(time_vector, device=device, dtype=dtype).unsqueeze(1)
    prediction, _ = model(edge_index, coordinate, times)
    return prediction[:, 0].detach().cpu().numpy()


def evaluate_sensors(
    model: NetworkSurrogate,
    time_vector: np.ndarray,
    signals: dict[str, np.ndarray],
    sensors: Iterable[str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    mask = (time_vector >= 0.0) & (time_vector <= 2.0)
    evaluation_time = time_vector[mask]
    results = {}
    references = {}
    predictions = {}
    for sensor in sensors:
        full_disturbance = baseline_disturbance(time_vector, signals[sensor])
        reference = full_disturbance[mask]
        prediction = predict_sensor(model, sensor, evaluation_time, device, dtype)
        references[sensor] = reference
        predictions[sensor] = prediction
        baseline_mask = (time_vector >= -0.8) & (time_vector <= -0.05)
        baseline_noise_sigma = float(np.std(full_disturbance[baseline_mask]))
        results[sensor] = metrics(reference, prediction, evaluation_time, baseline_noise_sigma)
    return results, references, predictions, evaluation_time


def mean_validation_score(results: dict[str, dict[str, float]]) -> float:
    return float(np.mean([row["nrmse_dynamic_range"] for row in results.values()]))


def model_settings(mode: str) -> dict[str, Any]:
    if mode == "smoke":
        return {"epochs": 4, "checkpoint_interval": 2, "validation_interval": 2, "collocation_per_edge": 4, "constraint_samples": 4, "device": "cpu", "dtype": "float64"}
    if mode == "formal":
        return {"epochs": 1200, "checkpoint_interval": 50, "validation_interval": 50, "collocation_per_edge": 24, "constraint_samples": 24, "device": "cuda", "dtype": "float64"}
    raise ValueError(mode)


def checkpoint_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / f"checkpoint_epoch{epoch:06d}.pt"


def latest_checkpoint(run_dir: Path) -> Path | None:
    paths = sorted(run_dir.glob("checkpoint_epoch*.pt"))
    return paths[-1] if paths else None


def save_checkpoint(
    path: Path,
    model: NetworkSurrogate,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    best_score: float,
    best_state: dict[str, torch.Tensor] | None,
    generator: torch.Generator,
) -> None:
    if path.exists():
        return
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_score": best_score,
            "best_state": best_state,
            "generator_state": generator.get_state(),
        },
        path,
    )


def train_one(
    input_path: Path,
    output_root: Path,
    model_name: str,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    settings = model_settings(mode)
    run_dir = output_root / mode / model_name / f"seed_{seed}"
    report_path = run_dir / "development_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(settings["device"])
    dtype = torch.float64 if settings["dtype"] == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 911)
    time_vector, signals = load_stage_data(input_path, "develop")
    if set(signals) & set(SEALED_TEST_SENSORS):
        raise RuntimeError("sealed test sensor entered development memory")
    train_batch = build_pressure_batch(time_vector, signals, TRAIN_SENSORS, device, dtype, sparse=True)
    model = NetworkSurrogate(model_name).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=7.5e-4)
    history: list[dict[str, float]] = []
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    start_epoch = 0
    resume_path = latest_checkpoint(run_dir)
    if resume_path is not None:
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history = list(saved["history"])
        best_score = float(saved["best_score"])
        best_state = saved["best_state"]
        generator.set_state(saved["generator_state"])
        start_epoch = int(saved["epoch"])
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, int(settings["epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_data = data_loss(model, train_batch)
        if model_name == "ann":
            loss_pde = torch.zeros_like(loss_data)
            loss_node = torch.zeros_like(loss_data)
            loss_boundary = torch.zeros_like(loss_data)
            loss_initial = torch.zeros_like(loss_data)
            loss_total = loss_data
        else:
            loss_pde = pde_loss(model, int(settings["collocation_per_edge"]), generator)
            loss_node, loss_boundary, loss_initial = constraint_losses(model, int(settings["constraint_samples"]), generator)
            loss_total = loss_data + 0.08 * loss_pde + 0.25 * loss_node + 0.25 * loss_boundary + 0.15 * loss_initial
        loss_total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
        optimizer.step()
        should_validate = epoch == 1 or epoch % int(settings["validation_interval"]) == 0 or epoch == int(settings["epochs"])
        if should_validate:
            validation, _, _, _ = evaluate_sensors(model, time_vector, signals, VALIDATION_SENSORS, device, dtype)
            score = mean_validation_score(validation)
            if score < best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            row = {
                "epoch": float(epoch),
                "loss_total": float(loss_total.detach().cpu()),
                "loss_data": float(loss_data.detach().cpu()),
                "loss_pde": float(loss_pde.detach().cpu()),
                "loss_node": float(loss_node.detach().cpu()),
                "loss_boundary": float(loss_boundary.detach().cpu()),
                "loss_initial": float(loss_initial.detach().cpu()),
                "gradient_norm": gradient_norm,
                "validation_mean_nrmse": score,
            }
            history.append(row)
            print(f"[perugia/{model_name}/seed={seed}] epoch={epoch} loss={row['loss_total']:.3e} val={score:.3%}", flush=True)
        if epoch % int(settings["checkpoint_interval"]) == 0 or epoch == int(settings["epochs"]):
            save_checkpoint(checkpoint_path(run_dir, epoch), model, optimizer, epoch, history, best_score, best_state, generator)
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    validation, references, predictions, evaluation_time = evaluate_sensors(model, time_vector, signals, VALIDATION_SENSORS, device, dtype)
    best_path = run_dir / "best_model.pt"
    if not best_path.exists():
        torch.save({"model_name": model_name, "seed": seed, "state_dict": best_state, "best_validation_score": best_score}, best_path)
    validation_rows = []
    for sensor in VALIDATION_SENSORS:
        validation_path = run_dir / f"validation_{sensor}.npz"
        if not validation_path.exists():
            np.savez_compressed(validation_path, time_s=evaluation_time, reference=references[sensor], prediction=predictions[sensor])
        validation_rows.append({"sensor": sensor, **validation[sensor]})
    report = {
        "status": "pass",
        "stage": "develop",
        "model": model_name,
        "seed": seed,
        "mode": mode,
        "development_variables_loaded": list(variables_for_stage("develop")),
        "sealed_test_variables_loaded": [],
        "training_sensors": list(TRAIN_SENSORS),
        "validation_sensors": list(VALIDATION_SENSORS),
        "sealed_test_sensors": list(SEALED_TEST_SENSORS),
        "labels_per_training_sensor": 96,
        "training_pressure_labels": 96 * len(TRAIN_SENSORS),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "best_validation_mean_nrmse": best_score,
        "validation_metrics": validation,
        "history": history,
        "training_seconds_this_invocation": time.perf_counter() - started,
        "best_model": best_path.name,
    }
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    validation_table = run_dir / "validation_metrics.csv"
    if not validation_table.exists():
        with validation_table.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(validation_rows[0]))
            writer.writeheader()
            writer.writerows(validation_rows)
    return report


def develop(input_path: Path, output_root: Path, mode: str, models: Iterable[str], seeds: Iterable[int]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    matrix_path = output_root / mode / "development_matrix_report.json"
    if matrix_path.exists():
        return json.loads(matrix_path.read_text(encoding="utf-8"))
    reports = []
    for model_name in models:
        for seed in seeds:
            reports.append(train_one(input_path, output_root, model_name, seed, mode))
            progress = {
                "stage": "develop",
                "mode": mode,
                "completed": [{"model": row["model"], "seed": row["seed"], "best_validation_mean_nrmse": row["best_validation_mean_nrmse"]} for row in reports],
            }
            progress_path = output_root / mode / f"development_progress_{len(reports):02d}.json"
            if not progress_path.exists():
                progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    report = {"status": "pass", "stage": "develop", "mode": mode, "runs": reports, "sealed_test_opened": False}
    matrix_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    mode_dir = output_root / mode
    colors = {"ann": "#777777", "coordinate_pinn": "#E07B39", "characteristic_pinn": "#2878B5"}
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    for model_name in dict.fromkeys(row["model"] for row in reports):
        model_runs = [row for row in reports if row["model"] == model_name]
        for run_report in model_runs:
            history = run_report["history"]
            axis.plot(
                [row["epoch"] for row in history],
                [100.0 * row["validation_mean_nrmse"] for row in history],
                color=colors[model_name],
                alpha=0.32,
                lw=0.85,
            )
        common_epochs = [row["epoch"] for row in model_runs[0]["history"]]
        mean_curve = np.mean(
            [[100.0 * row["validation_mean_nrmse"] for row in run_report["history"]] for run_report in model_runs],
            axis=0,
        )
        axis.plot(common_epochs, mean_curve, color=colors[model_name], lw=1.8, label=model_name.replace("_", " "))
    axis.set(xlabel="Epoch", ylabel="Validation pressure NRMSE (%)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        target = mode_dir / f"F24_perugia_development_convergence.{suffix}"
        if not target.exists():
            figure.savefig(target, dpi=320 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    primary_runs = {}
    for row in reports:
        primary_runs.setdefault(row["model"], row)
    figure, axes = plt.subplots(len(VALIDATION_SENSORS), 1, figsize=(11.2, 5.8), sharex=True)
    for axis, sensor in zip(np.atleast_1d(axes), VALIDATION_SENSORS):
        plotted_reference = False
        for model_name, run_report in primary_runs.items():
            curve_path = mode_dir / model_name / f"seed_{run_report['seed']}" / f"validation_{sensor}.npz"
            with np.load(curve_path) as archive:
                curve_time = archive["time_s"]
                reference = archive["reference"]
                prediction = archive["prediction"]
            if not plotted_reference:
                axis.plot(curve_time, reference, color="black", lw=1.1, label="experiment")
                plotted_reference = True
            axis.plot(curve_time, prediction, color=colors[model_name], lw=0.95, label=model_name.replace("_", " "))
        axis.set_ylabel(f"{sensor}: $H-H_0$ (m)")
        axis.grid(alpha=0.2)
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=4, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        target = mode_dir / f"F25_perugia_validation_sensor_histories.{suffix}"
        if not target.exists():
            figure.savefig(target, dpi=320 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return report


def release(input_path: Path, output_root: Path, mode: str) -> dict[str, Any]:
    development_path = output_root / mode / "development_matrix_report.json"
    if not development_path.exists():
        raise FileNotFoundError("development matrix must be completed before sealed release")
    release_dir = output_root / mode / "sealed_release"
    report_path = release_dir / "sealed_test_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    release_dir.mkdir(parents=True, exist_ok=True)
    time_vector, test_signals = load_stage_data(input_path, "release")
    device = resolve_device(model_settings(mode)["device"])
    dtype = torch.float64 if model_settings(mode)["dtype"] == "float64" else torch.float32
    mask = (time_vector >= 0.0) & (time_vector <= 2.0)
    all_rows = []
    saved_curves: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for model_name in MODEL_NAMES:
        for seed in SEEDS if mode == "formal" else (SEEDS[0],):
            best_path = output_root / mode / model_name / f"seed_{seed}" / "best_model.pt"
            if not best_path.exists():
                raise FileNotFoundError(best_path)
            payload = torch.load(best_path, map_location=device, weights_only=False)
            model = NetworkSurrogate(model_name).to(device=device, dtype=dtype)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            results, references, predictions, evaluation_time = evaluate_sensors(model, time_vector, test_signals, SEALED_TEST_SENSORS, device, dtype)
            for sensor in SEALED_TEST_SENSORS:
                row = {"model": model_name, "seed": seed, "sensor": sensor, **results[sensor]}
                all_rows.append(row)
                saved_curves[(model_name, seed, sensor)] = (references[sensor], predictions[sensor])
    table_path = release_dir / "T09_perugia_sealed_test_metrics.csv"
    if table_path.exists():
        raise FileExistsError(table_path)
    with table_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    primary_seed = SEEDS[0]
    figure, axes = plt.subplots(len(SEALED_TEST_SENSORS), 1, figsize=(11.4, 8.2), sharex=True)
    colors = {"ann": "#777777", "coordinate_pinn": "#E07B39", "characteristic_pinn": "#2878B5"}
    for axis, sensor in zip(axes, SEALED_TEST_SENSORS):
        reference, _ = saved_curves[("ann", primary_seed, sensor)]
        axis.plot(time_vector[mask], reference, color="black", lw=1.1, label="experiment")
        for model_name in MODEL_NAMES:
            _, prediction = saved_curves[(model_name, primary_seed, sensor)]
            axis.plot(time_vector[mask], prediction, lw=0.95, color=colors[model_name], label=model_name.replace("_", " "))
        axis.set_ylabel(f"{sensor}: $H-H_0$ (m)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        target = release_dir / f"F26_perugia_sealed_test_histories.{suffix}"
        if target.exists():
            raise FileExistsError(target)
        figure.savefig(target, dpi=320 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), gridspec_kw={"hspace": 0.34, "wspace": 0.28})
    axis = axes[0, 0]
    positions = np.arange(len(MODEL_NAMES))
    values = [[100.0 * row["nrmse_dynamic_range"] for row in all_rows if row["model"] == model_name] for model_name in MODEL_NAMES]
    boxes = axis.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showmeans=True)
    for patch, model_name in zip(boxes["boxes"], MODEL_NAMES):
        patch.set_facecolor(colors[model_name])
        patch.set_alpha(0.75)
    axis.set_xticks(positions, [name.replace("_", " ") for name in MODEL_NAMES])
    axis.set_ylabel("Sealed-sensor pressure NRMSE (%)")
    axis.grid(axis="y", alpha=0.2)
    axis.set_title("(a) Full-history error")

    axis = axes[0, 1]
    width = 0.24
    sensor_positions = np.arange(len(SEALED_TEST_SENSORS))
    for model_index, model_name in enumerate(MODEL_NAMES):
        model_values = []
        for sensor in SEALED_TEST_SENSORS:
            selected = [100.0 * row["positive_peak_relative_error"] for row in all_rows if row["model"] == model_name and row["sensor"] == sensor]
            model_values.append(float(np.mean(selected)))
        axis.bar(sensor_positions + (model_index - 1) * width, model_values, width, color=colors[model_name], label=model_name.replace("_", " "))
    axis.set_xticks(sensor_positions, SEALED_TEST_SENSORS)
    axis.set_ylabel("Positive-peak relative error (%)")
    axis.set_title("(b) Peak reconstruction")
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 0]
    for model_index, model_name in enumerate(MODEL_NAMES):
        model_values = []
        for sensor in SEALED_TEST_SENSORS:
            selected = [row["first_arrival_absolute_error_s"] for row in all_rows if row["model"] == model_name and row["sensor"] == sensor]
            model_values.append(float(np.nanmean(selected)))
        axis.bar(sensor_positions + (model_index - 1) * width, model_values, width, color=colors[model_name], label=model_name.replace("_", " "))
    axis.set_xticks(sensor_positions, SEALED_TEST_SENSORS)
    axis.set_ylabel("First-arrival absolute error (s)")
    axis.set_title("(c) Wave-arrival reconstruction")
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 1]
    residuals = np.stack(
        [
            saved_curves[("characteristic_pinn", primary_seed, sensor)][1]
            - saved_curves[("characteristic_pinn", primary_seed, sensor)][0]
            for sensor in SEALED_TEST_SENSORS
        ]
    )
    limit = max(float(np.quantile(np.abs(residuals), 0.99)), 1.0e-6)
    image = axis.imshow(
        residuals,
        aspect="auto",
        origin="lower",
        extent=[float(evaluation_time[0]), float(evaluation_time[-1]), -0.5, len(SEALED_TEST_SENSORS) - 0.5],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_yticks(np.arange(len(SEALED_TEST_SENSORS)), SEALED_TEST_SENSORS)
    axis.set(xlabel="Time (s)", ylabel="Sealed sensor", title="(d) Characteristic-PINN residual (primary seed)")
    figure.colorbar(image, ax=axis, pad=0.02, label="Prediction − experiment (m)")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        target = release_dir / f"F27_perugia_sealed_test_engineering_metrics.{suffix}"
        if target.exists():
            raise FileExistsError(target)
        figure.savefig(target, dpi=320 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    summary = {}
    for model_name in MODEL_NAMES:
        selected = [row for row in all_rows if row["model"] == model_name]
        summary[model_name] = {
            key: float(np.nanmean([row[key] for row in selected]))
            for key in ("rmse_m", "nrmse_dynamic_range", "positive_peak_relative_error", "negative_peak_relative_error", "positive_peak_time_absolute_error_s", "first_arrival_absolute_error_s", "correlation")
        }
    report = {
        "status": "pass",
        "stage": "sealed_release",
        "mode": mode,
        "variables_loaded": list(variables_for_stage("release")),
        "test_sensors": list(SEALED_TEST_SENSORS),
        "summary": summary,
        "metrics_table": table_path.name,
        "figures": ["F26_perugia_sealed_test_histories.png", "F26_perugia_sealed_test_histories.pdf", "F27_perugia_sealed_test_engineering_metrics.png", "F27_perugia_sealed_test_engineering_metrics.pdf"],
        "scope_limit": "Pressure-only reconstruction in an independent looped HDPE network; not a four-state buried-FSSI experiment.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("develop", "release"), default="develop")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--seeds", default=str(SEEDS[0]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "develop":
        models = tuple(value.strip() for value in args.models.split(",") if value.strip())
        seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
        unknown = set(models) - set(MODEL_NAMES)
        if unknown:
            raise ValueError(f"unknown models: {sorted(unknown)}")
        report = develop(args.input, args.output_dir, args.mode, models, seeds)
    else:
        report = release(args.input, args.output_dir, args.mode)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
