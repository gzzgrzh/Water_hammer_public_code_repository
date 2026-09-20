"""Physics-data hybrid refinement of the three-parameter FSSI surrogate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from water16_reproduction.parametric_fssi_forward import PARAMETER_NAMES, materialize_cases
from water16_reproduction.parametric_fssi_model import (
    ThreeParameterBoundaryModel,
    boundary_losses,
    evaluate_reference_cases,
    hard_boundary_diagnostics,
    load_config as load_base_model_config,
    load_design,
    characteristic_state,
    wave_speed_magnitudes,
)
from water16_reproduction.parametric_fssi_reference import case_parameters, load_baseline
from water16_reproduction.wp1_verification import STATE_NAMES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "PINN_FSSI_research_plan" / "configs" / "parametric_fssi_hybrid_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "PINN_FSSI_research_plan" / "outputs" / "parametric_fssi_hybrid_v1"
)


@dataclass
class AnchorBatch:
    case: dict[str, Any]
    positions: torch.Tensor
    times: torch.Tensor
    targets: torch.Tensor
    scales: torch.Tensor
    observation_weights: torch.Tensor
    source_file: Path


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("hybrid config schema_version must be 1")
    if config.get("method") != "three_parameter_characteristic_physics_data_hybrid_surrogate":
        raise ValueError("unexpected hybrid method")
    policy = config["anchor_policy"]
    if policy["allowed_split"] != "train" or not policy["validation_and_test_labels_forbidden"]:
        raise ValueError("hybrid anchors must be restricted to the train split")
    expected = int(policy["spatial_points_per_case"]) * int(policy["time_points_per_case"])
    if expected != int(policy["anchor_vectors_per_case"]):
        raise ValueError("anchor vector count is inconsistent")
    return config


def deterministic_indices(size: int, count: int) -> np.ndarray:
    if not 1 <= count <= size:
        raise ValueError("anchor count must lie between one and grid size")
    return np.unique(np.rint(np.linspace(0, size - 1, count)).astype(int))


def nested_time_subset(candidate_indices: np.ndarray, count: int) -> np.ndarray:
    """Return a deterministic nested subset of an already selected anchor design."""

    candidates = np.asarray(candidate_indices, dtype=int)
    if candidates.ndim != 1 or len(np.unique(candidates)) != len(candidates):
        raise ValueError("nested anchor candidates must be a unique one-dimensional array")
    if not 1 <= count <= len(candidates):
        raise ValueError("nested anchor count must lie within the candidate design")
    selected_positions = [0]
    if len(candidates) > 1:
        selected_positions.append(len(candidates) - 1)
    while len(selected_positions) < count:
        available = [index for index in range(len(candidates)) if index not in selected_positions]
        next_position = max(
            available,
            key=lambda index: (
                min(abs(index - selected) for selected in selected_positions),
                -index,
            ),
        )
        selected_positions.append(next_position)
    return np.sort(candidates[np.asarray(selected_positions[:count], dtype=int)])


def nested_maximin_case_subset(
    cases: list[dict[str, Any]], count: int, design: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return a deterministic nested space-filling prefix in parameter space."""

    if count < 1 or count > len(cases):
        raise ValueError("case count must lie within the registered training design")
    ordered = sorted(cases, key=lambda case: case["case_id"])
    coordinates = []
    for case in ordered:
        row = []
        for name in PARAMETER_NAMES:
            definition = design["parameter_domain"][name]
            lower, upper = float(definition["lower"]), float(definition["upper"])
            value = float(case[name])
            if definition["coordinate"] == "natural_log":
                lower, upper, value = math.log(lower), math.log(upper), math.log(value)
            row.append(2.0 * (value - lower) / (upper - lower) - 1.0)
        coordinates.append(row)
    points = np.asarray(coordinates, dtype=float)
    selected = [int(np.argmin(np.sum(points**2, axis=1)))]
    remaining = set(range(len(ordered))) - set(selected)
    while len(selected) < count:
        candidates = sorted(remaining)
        minimum_distance = np.asarray(
            [
                np.min(np.sum((points[index] - points[selected]) ** 2, axis=1))
                for index in candidates
            ]
        )
        best = candidates[int(np.argmax(minimum_distance))]
        selected.append(best)
        remaining.remove(best)
    return [ordered[index] for index in selected]


def event_aligned_time_indices(
    times: np.ndarray,
    position_m: float,
    params,
    count: int,
) -> np.ndarray:
    """Select fixed-count local wave-event anchors without using field values."""

    if not 2 <= count <= len(times):
        raise ValueError("event-aligned count must lie between two and grid size")
    dt = float(np.mean(np.diff(times)))
    if dt <= 0.0 or not np.allclose(np.diff(times), dt):
        raise ValueError("event-aligned selection requires a uniform time grid")
    maximum_index = len(times) - 1
    ordered: list[int] = [0, maximum_index]

    def add_time(value: float) -> None:
        if not 0.0 <= value <= float(times[-1]):
            return
        center = int(np.rint(value / dt))
        for offset in (0, -1, 1, -2, 2, -4, 4):
            index = min(max(center + offset, 0), maximum_index)
            if index not in ordered:
                ordered.append(index)

    add_time(float(params.valve_close_time_s))
    for speed in wave_speed_magnitudes(params):
        speed = float(speed)
        period = 2.0 * params.length_m / speed
        path_lengths = (
            position_m,
            params.length_m - position_m,
            params.length_m + position_m,
            2.0 * params.length_m - position_m,
        )
        maximum_order = int(math.ceil(float(times[-1]) / period))
        for order in range(maximum_order + 1):
            for path_length in path_lengths:
                arrival = path_length / speed + order * period
                add_time(arrival)
                add_time(float(params.valve_close_time_s) + arrival)
                if len(ordered) >= count:
                    return np.asarray(sorted(ordered[:count]), dtype=int)
    for index in deterministic_indices(len(times), count):
        if int(index) not in ordered:
            ordered.append(int(index))
        if len(ordered) >= count:
            break
    if len(ordered) != count:
        raise RuntimeError("could not materialise the requested event-aligned anchors")
    return np.asarray(sorted(ordered), dtype=int)


def characteristic_event_candidates(
    times: np.ndarray,
    position_m: float,
    params,
    speed: float,
) -> np.ndarray:
    """Return a temporally complete candidate set for one characteristic family."""

    dt = float(np.mean(np.diff(times)))
    maximum_index = len(times) - 1
    candidates: set[int] = set()
    period = 2.0 * params.length_m / speed
    path_lengths = (
        position_m,
        params.length_m - position_m,
        params.length_m + position_m,
        2.0 * params.length_m - position_m,
    )
    maximum_order = int(math.ceil(float(times[-1]) / period))
    for order in range(maximum_order + 1):
        for path_length in path_lengths:
            arrival = path_length / speed + order * period
            for event_time in (arrival, float(params.valve_close_time_s) + arrival):
                if not 0.0 <= event_time <= float(times[-1]):
                    continue
                center = int(np.rint(event_time / dt))
                for offset in (0, -1, 1, -2, 2):
                    candidates.add(min(max(center + offset, 0), maximum_index))
    return np.asarray(sorted(candidates), dtype=int)


def balanced_event_aligned_time_indices(
    times: np.ndarray,
    position_m: float,
    params,
    count: int,
) -> np.ndarray:
    """Select equal slow/fast wave-family coverage over the complete time window."""

    if not 12 <= count <= len(times):
        raise ValueError("balanced event count must lie between 12 and the grid size")
    dt = float(np.mean(np.diff(times)))
    if dt <= 0.0 or not np.allclose(np.diff(times), dt):
        raise ValueError("balanced event selection requires a uniform time grid")
    maximum_index = len(times) - 1
    closure = int(np.rint(float(params.valve_close_time_s) / dt))
    selected: list[int] = []

    def add(index: int) -> None:
        bounded = min(max(int(index), 0), maximum_index)
        if bounded not in selected:
            selected.append(bounded)

    for index in (0, maximum_index, closure, closure - 1, closure + 1, closure - 2, closure + 2):
        add(index)
    candidates_by_family = [
        characteristic_event_candidates(times, position_m, params, float(speed))
        for speed in wave_speed_magnitudes(params)
    ]
    remaining = count - len(selected)
    family_quota = remaining // 2
    for candidates in candidates_by_family:
        available = np.asarray([index for index in candidates if int(index) not in selected], dtype=int)
        quota = min(family_quota, len(available))
        for candidate_index in deterministic_indices(len(available), quota):
            add(int(available[candidate_index]))

    merged = np.unique(np.concatenate(candidates_by_family))
    for candidate_index in deterministic_indices(len(merged), min(len(merged), count)):
        add(int(merged[candidate_index]))
        if len(selected) >= count:
            break
    for index in deterministic_indices(len(times), count):
        add(int(index))
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise RuntimeError("could not materialise balanced dual-family event anchors")
    return np.asarray(sorted(selected), dtype=int)


def response_peak_enriched_time_indices(
    times: np.ndarray,
    position_m: float,
    params,
    state_traces: np.ndarray,
    initial_states: np.ndarray,
    count: int,
    physics_count: int,
) -> np.ndarray:
    """Combine dual-family physics events with train-only response peaks."""

    if state_traces.shape != (len(times), 4):
        raise ValueError("peak enrichment requires one four-state trace per time")
    if not 12 <= physics_count < count <= len(times):
        raise ValueError("invalid physics/total anchor counts for peak enrichment")
    selected = list(
        balanced_event_aligned_time_indices(
            times, position_m, params, physics_count
        )
    )
    scales = np.maximum(
        np.maximum(np.ptp(state_traces, axis=0), np.max(np.abs(state_traces - initial_states), axis=0)),
        1.0e-30,
    )
    normalized = np.abs((state_traces - initial_states) / scales)
    gradient = np.abs(np.gradient(state_traces, times, axis=0))
    gradient /= np.maximum(np.max(gradient, axis=0), 1.0e-30)

    def ranked_separated(values: np.ndarray) -> list[int]:
        chosen: list[int] = []
        for index in np.argsort(values)[::-1]:
            index = int(index)
            if all(abs(index - previous) >= 3 for previous in chosen):
                chosen.append(index)
        return chosen

    candidates: list[int] = []
    for state in range(4):
        amplitude = ranked_separated(normalized[:, state])
        slope = ranked_separated(gradient[:, state])
        candidates.extend(amplitude[:3])
        candidates.extend(slope[:2])
    for index in candidates:
        if index not in selected:
            selected.append(index)
        if len(selected) >= count:
            break
    for index in deterministic_indices(len(times), count):
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise RuntimeError("could not materialise response-peak enriched anchors")
    return np.asarray(sorted(selected), dtype=int)


def _state_scale(values: np.ndarray, initial: float) -> float:
    return max(float(np.ptp(values)), float(np.max(np.abs(values - initial))), 1.0e-30)


def load_training_anchors(
    config: dict[str, Any],
    design: dict[str, Any],
    baseline,
    device: torch.device,
    dtype: torch.dtype,
    case_limit: int,
) -> list[AnchorBatch]:
    root = ROOT / config["reference_root"]
    policy = config["anchor_policy"]
    observation_policy = config.get("observation_policy", {})
    observed_states = tuple(observation_policy.get("observed_states", STATE_NAMES))
    unknown_states = sorted(set(observed_states) - set(STATE_NAMES))
    if unknown_states or not observed_states:
        raise ValueError(f"invalid observed states: {observed_states}")
    observation_weights = np.asarray(
        [1.0 if state in observed_states else 0.0 for state in STATE_NAMES],
        dtype=float,
    )
    noise_fraction = float(observation_policy.get("noise_std_fraction_of_dynamic_scale", 0.0))
    if noise_fraction < 0.0:
        raise ValueError("observation noise fraction must be non-negative")
    noise_seed = int(observation_policy.get("noise_seed", 29041))
    data_stage = config.get("data_stage", "pilot")
    eligible_cases = [
        case for case in materialize_cases(design, data_stage)
        if case["split"] == policy["allowed_split"]
    ]
    case_selection = policy.get("case_selection", "registered_order_prefix")
    if case_selection == "nested_maximin_parameter_space_v1":
        cases = nested_maximin_case_subset(eligible_cases, case_limit, design)
    elif case_selection == "registered_order_prefix":
        cases = eligible_cases[:case_limit]
    else:
        raise ValueError(f"unsupported training-case selection: {case_selection}")
    batches: list[AnchorBatch] = []
    for case in cases:
        source = root / "train" / case["case_id"] / "truth_evaluation_grid.npz"
        if not source.exists():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as archive:
            x = archive["x"]
            t = archive["t"]
            fields = {state: archive[state] for state in STATE_NAMES}
        params, _ = case_parameters(
            baseline,
            case,
            float(design["truth_solver"][data_stage]["t_final_s"]),
        )
        x_indices = deterministic_indices(len(x), int(policy["spatial_points_per_case"]))
        if policy["selection"] == "deterministic_uniform_indices_in_saved_evaluation_grid":
            common_time_indices = deterministic_indices(
                len(t), int(policy["time_points_per_case"])
            )
            time_indices_by_x = [common_time_indices for _ in x_indices]
        elif policy["selection"] == "local_characteristic_event_aligned_v1":
            time_indices_by_x = [
                event_aligned_time_indices(
                    t,
                    float(x[x_index]),
                    params,
                    int(policy["time_points_per_case"]),
                )
                for x_index in x_indices
            ]
        elif policy["selection"] == "balanced_dual_characteristic_event_aligned_v2":
            time_indices_by_x = [
                balanced_event_aligned_time_indices(
                    t,
                    float(x[x_index]),
                    params,
                    int(policy["time_points_per_case"]),
                )
                for x_index in x_indices
            ]
        elif policy["selection"] == "balanced_dual_characteristic_event_aligned_nested_v1":
            candidate_count = int(policy["candidate_time_points_per_case"])
            time_indices_by_x = [
                nested_time_subset(
                    balanced_event_aligned_time_indices(
                        t,
                        float(x[x_index]),
                        params,
                        candidate_count,
                    ),
                    int(policy["time_points_per_case"]),
                )
                for x_index in x_indices
            ]
        elif policy["selection"] == "train_response_peak_enriched_v3":
            pressure0 = params.water_density_kg_m3 * params.gravity_m_s2 * params.head_difference_m
            area_f = np.pi * params.inner_radius_m**2
            area_t = np.pi * ((params.inner_radius_m + params.wall_thickness_m)**2 - params.inner_radius_m**2)
            initials = np.asarray(
                [params.initial_velocity_m_s, 0.0, pressure0, area_f * pressure0 / area_t]
            )
            time_indices_by_x = [
                response_peak_enriched_time_indices(
                    t,
                    float(x[x_index]),
                    params,
                    np.stack([fields[state][x_index] for state in STATE_NAMES], axis=1),
                    initials,
                    int(policy["time_points_per_case"]),
                    int(policy["physics_time_points_per_case"]),
                )
                for x_index in x_indices
            ]
        elif policy["selection"] == "train_response_peak_enriched_nested_ablation_v4":
            pressure0 = params.water_density_kg_m3 * params.gravity_m_s2 * params.head_difference_m
            area_f = np.pi * params.inner_radius_m**2
            area_t = np.pi * ((params.inner_radius_m + params.wall_thickness_m)**2 - params.inner_radius_m**2)
            initials = np.asarray(
                [params.initial_velocity_m_s, 0.0, pressure0, area_f * pressure0 / area_t]
            )
            candidate_count = int(policy["candidate_time_points_per_case"])
            candidate_physics = int(policy["candidate_physics_time_points_per_case"])
            time_indices_by_x = [
                nested_time_subset(
                    response_peak_enriched_time_indices(
                        t,
                        float(x[x_index]),
                        params,
                        np.stack([fields[state][x_index] for state in STATE_NAMES], axis=1),
                        initials,
                        candidate_count,
                        candidate_physics,
                    ),
                    int(policy["time_points_per_case"]),
                )
                for x_index in x_indices
            ]
        else:
            raise ValueError(f"unsupported anchor selection: {policy['selection']}")
        positions = np.concatenate(
            [np.full(len(t_indices), x[x_index]) for x_index, t_indices in zip(x_indices, time_indices_by_x)]
        )
        selected_times = np.concatenate([t[t_indices] for t_indices in time_indices_by_x])
        targets = np.concatenate(
            [
                np.stack(
                    [fields[state][x_index, t_indices] for state in STATE_NAMES],
                    axis=1,
                )
                for x_index, t_indices in zip(x_indices, time_indices_by_x)
            ],
            axis=0,
        )
        pressure0 = params.water_density_kg_m3 * params.gravity_m_s2 * params.head_difference_m
        area_f = np.pi * params.inner_radius_m**2
        area_t = np.pi * ((params.inner_radius_m + params.wall_thickness_m)**2 - params.inner_radius_m**2)
        initials = [params.initial_velocity_m_s, 0.0, pressure0, area_f * pressure0 / area_t]
        scales = np.asarray(
            [_state_scale(fields[state], initial) for state, initial in zip(STATE_NAMES, initials)],
            dtype=float,
        )
        if noise_fraction > 0.0:
            case_digest = int.from_bytes(
                hashlib.sha256(case["case_id"].encode("utf-8")).digest()[:8],
                byteorder="little",
                signed=False,
            )
            noise_rng = np.random.default_rng((noise_seed + case_digest) % (2**32))
            noise = noise_rng.normal(size=targets.shape) * scales[None, :] * noise_fraction
            targets = targets + noise * observation_weights[None, :]
        batches.append(
            AnchorBatch(
                case=case,
                positions=torch.as_tensor(positions[:, None], device=device, dtype=dtype),
                times=torch.as_tensor(selected_times[:, None], device=device, dtype=dtype),
                targets=torch.as_tensor(targets, device=device, dtype=dtype),
                scales=torch.as_tensor(scales, device=device, dtype=dtype),
                observation_weights=torch.as_tensor(
                    observation_weights, device=device, dtype=dtype
                ),
                source_file=source,
            )
        )
    return batches


def anchor_loss(
    model: ThreeParameterBoundaryModel,
    batch: AnchorBatch,
    tail_fraction: float = 0.0,
    tail_weight: float = 0.0,
) -> torch.Tensor:
    prediction = characteristic_state(model, batch.positions, batch.times, batch.case)
    squared = ((prediction - batch.targets) / batch.scales) ** 2
    weights = batch.observation_weights[None, :]
    mean_loss = torch.sum(squared * weights) / (squared.shape[0] * torch.sum(weights))
    if tail_fraction <= 0.0 or tail_weight <= 0.0:
        return mean_loss
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("anchor tail_fraction must lie in (0, 1]")
    count = max(1, int(math.ceil(tail_fraction * squared.shape[0])))
    observed_indices = torch.nonzero(batch.observation_weights > 0.0, as_tuple=False).flatten().tolist()
    statewise_tail = torch.stack(
        [torch.topk(squared[:, state], count).values.mean() for state in observed_indices]
    ).mean()
    return mean_loss + tail_weight * statewise_tail


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_compatible_state_dict(
    model: ThreeParameterBoundaryModel, source: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Copy a v1 state and zero only newly appended first-layer features."""

    target = model.state_dict()
    adapted: list[str] = []
    for name, source_value in source.items():
        if name not in target:
            raise ValueError(f"unexpected checkpoint tensor: {name}")
        if target[name].shape == source_value.shape:
            target[name] = source_value
            continue
        if (
            name == "network.0.weight"
            and target[name].shape[0] == source_value.shape[0]
            and target[name].shape[1] > source_value.shape[1]
        ):
            target[name].zero_()
            target[name][:, : source_value.shape[1]] = source_value
            adapted.append(name)
            continue
        raise ValueError(
            f"incompatible checkpoint tensor {name}: "
            f"source={tuple(source_value.shape)}, target={tuple(target[name].shape)}"
        )
    model.load_state_dict(target)
    return {"adapted_tensors": adapted}


def run_training(config: dict[str, Any], mode: str, output_dir: Path) -> dict[str, Any]:
    base_config = copy.deepcopy(
        load_base_model_config(ROOT / config["base_model_config"])
    )
    if "network_input_ablation" in config:
        base_config["network"]["input_ablation"] = copy.deepcopy(
            config["network_input_ablation"]
        )
    design = load_design(base_config)
    baseline = load_baseline(design)
    spec = config["training"][mode]
    dtype = torch.float64 if spec["dtype"] == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(int(spec["seed"]))
    device = torch.device(
        "cuda" if spec["device"] == "auto" and torch.cuda.is_available()
        else ("cpu" if spec["device"] == "auto" else spec["device"])
    )
    run_dir = output_dir / mode
    report_path = run_dir / "model_report.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    model = ThreeParameterBoundaryModel(design, base_config, baseline).to(device)
    warm_path = ROOT / config["warm_start_checkpoint"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["learning_rate"]))
    history: list[dict[str, float]] = []
    resumed_from_epoch = 0
    checkpoint_adaptation: dict[str, Any] = {"adapted_tensors": []}
    if checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        resumed_from_epoch = int(checkpoint["epoch"])
        start_epoch = resumed_from_epoch + 1
    else:
        checkpoint = _load_checkpoint(warm_path, device)
        checkpoint_adaptation = load_compatible_state_dict(
            model, checkpoint["state_dict"]
        )
        start_epoch = 1
    anchors = load_training_anchors(
        config, design, baseline, device, dtype, int(spec["case_limit"])
    )
    times = torch.linspace(
        0.0, model.t_final_s, int(spec["boundary_points"]), device=device, dtype=dtype
    )[:, None]
    weights = config["loss_weights"]
    anchor_objective = config.get("anchor_objective", {})
    started = time.perf_counter()
    for epoch in range(start_epoch, int(spec["epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = {
            "transport_consistency": torch.zeros((), device=device),
            "valve_dynamic": torch.zeros((), device=device),
            "training_anchor": torch.zeros((), device=device),
        }
        for batch in anchors:
            physics = boundary_losses(model, times, batch.case)
            accumulated["transport_consistency"] += physics["transport_consistency"] / len(anchors)
            accumulated["valve_dynamic"] += physics["valve_dynamic"] / len(anchors)
            accumulated["training_anchor"] += anchor_loss(
                model,
                batch,
                float(anchor_objective.get("tail_fraction", 0.0)),
                float(anchor_objective.get("tail_weight", 0.0)),
            ) / len(anchors)
        total = sum(float(weights[name]) * value for name, value in accumulated.items())
        total.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip"]))
            .detach().cpu()
        )
        optimizer.step()
        row = {
            "epoch": float(epoch),
            "loss_total": float(total.detach().cpu()),
            **{f"loss_{name}": float(value.detach().cpu()) for name, value in accumulated.items()},
            "gradient_norm": gradient_norm,
        }
        history.append(row)
        if epoch == 1 or epoch == int(spec["epochs"]) or epoch % int(spec["log_every"]) == 0:
            print(
                f"[hybrid/{mode}] epoch={epoch} total={row['loss_total']:.3e} "
                f"anchor={row['loss_training_anchor']:.3e}", flush=True
            )
        if epoch % int(spec["checkpoint_every"]) == 0 or epoch == int(spec["epochs"]):
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "history": history,
                },
                checkpoint_path,
            )

    diagnostic_times = torch.linspace(0.01, 0.79, 97, device=device, dtype=dtype)[:, None]
    boundaries = [hard_boundary_diagnostics(model, diagnostic_times, batch.case) for batch in anchors]
    hard_max = max(max(values.values()) for values in boundaries)
    report: dict[str, Any] = {
        "status": "pass" if hard_max <= float(config["acceptance"]["maximum_hard_boundary_nrmse"]) else "failed",
        "mode": mode,
        "model_id": config["model_id"],
        "method": config["method"],
        "data_stage": config.get("data_stage", "pilot"),
        "device": str(device),
        "warm_start_checkpoint": str(warm_path.resolve()),
        "resumed_from_epoch": resumed_from_epoch,
        "checkpoint_adaptation": checkpoint_adaptation,
        "network_input_ablation": base_config["network"].get(
            "input_ablation",
            {
                "raw_parameter_channels": {
                    name: True for name in PARAMETER_NAMES
                },
                "closure_phase_features": True,
                "travel_time_features": True,
            },
        ),
        "training_case_ids": [batch.case["case_id"] for batch in anchors],
        "anchor_source_files": [str(batch.source_file.resolve()) for batch in anchors],
        "validation_or_test_labels_used": False,
        "anchor_vectors_per_case": int(config["anchor_policy"]["anchor_vectors_per_case"]),
        "observation_policy": config.get(
            "observation_policy",
            {"observed_states": list(STATE_NAMES), "noise_std_fraction_of_dynamic_scale": 0.0},
        ),
        "anchor_objective": anchor_objective,
        "training_seconds": time.perf_counter() - started,
        "maximum_hard_boundary_nrmse": hard_max,
        "final_training_loss": history[-1],
    }
    if mode in ("pilot", "formal"):
        evaluation_config = dict(base_config)
        evaluation_config["reference_root"] = config["reference_root"]
        evaluation_config["evaluation"] = config["evaluation"]
        evaluation, rows = evaluate_reference_cases(model, design, evaluation_config, device)
        limits = config["acceptance"]
        prefix = "formal" if mode == "formal" else "pilot"
        accuracy_pass = (
            evaluation["maximum_by_metric"]["P_nrmse"] <= float(limits[f"{prefix}_pressure_nrmse"])
            and evaluation["maximum_by_metric"]["sigma_z_nrmse"] <= float(limits[f"{prefix}_axial_stress_nrmse"])
            and evaluation["maximum_by_metric"]["pressure_peak_relative_error"] <= float(limits[f"{prefix}_pressure_peak_relative_error"])
            and evaluation["maximum_by_metric"]["stress_peak_relative_error"] <= float(limits[f"{prefix}_stress_peak_relative_error"])
        )
        report["evaluation"] = evaluation
        report["accuracy_acceptance"] = "pass" if accuracy_pass else "failed"
        report["status"] = "pass" if report["status"] == "pass" and accuracy_pass else "failed"
        import csv
        with (run_dir / "held_out_case_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "config": config, "report": report}, run_dir / "model.pt")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode", choices=("audit", "smoke", "pilot", "formal"), default="audit"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "audit":
        report = {
            "status": "pass",
            "mode": "audit",
            "model_id": config["model_id"],
            "registration_reason": config["registration_reason"],
            "anchor_policy": config["anchor_policy"],
        }
    else:
        report = run_training(config, args.mode, args.output_dir)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
