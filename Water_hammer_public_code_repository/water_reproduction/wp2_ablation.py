"""WP2 four-state full-domain PINN ablation protocol.

The module deliberately keeps MOC data outside the training loss.  P0--P4
share one MLP and one physical case; only the explicitly declared method
components change.  Audit and smoke modes do not establish PINN accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from water16_reproduction.common.physics import (
    PhysicalParameters,
    characteristic_basis,
    cross_section_areas,
    initial_pressure_pa,
    state_scales,
    system_matrices,
)
from water16_reproduction.research_baseline import (
    DEFAULT_CONFIG as WP0_CONFIG,
    DEFAULT_PROVENANCE as WP0_PROVENANCE,
    load_config as load_wp0_config,
    validate_config as validate_wp0_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT / "PINN_FSSI_research_plan"
DEFAULT_CONFIG = RESEARCH_ROOT / "configs" / "wp2_ablation_v1.json"
DEFAULT_V2_CONFIG = RESEARCH_ROOT / "configs" / "wp2_ablation_v2.json"
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "outputs" / "wp2"
VARIANT_ORDER = ("P0", "P1", "P2", "P3", "P4")
STATE_NAMES = ("V", "uz", "P", "sigma_z")
STATE_UNITS = ("m/s", "m/s", "Pa", "Pa")
WP2_MODEL_REVISION = 2


@dataclass
class TrainingConfig:
    epochs: int
    lbfgs_steps: int
    hidden_width: int
    hidden_layers: int
    pde_points: int
    initial_points: int
    boundary_points: int
    learning_rate: float
    gradient_balance_every: int
    adaptive_resample_every: int
    adaptive_pool_factor: int
    checkpoint_every: int
    plot_x: int
    plot_t: int
    dtype: str
    device: str
    seed: int


def load_wp2_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_revision",
        "experiment_id",
        "status",
        "pre_formal_amendments",
        "physical_case_config",
        "reference",
        "shared_network",
        "formal_training",
        "smoke_training",
        "pilot_training",
        "formal_seeds",
        "variants",
        "fairness_rules",
        "evaluation",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"WP2 config missing keys: {missing}")
    if data["schema_version"] != 1:
        raise ValueError("WP2 schema_version must be 1")
    if not isinstance(data["protocol_revision"], int) or data["protocol_revision"] < 1:
        raise ValueError("WP2 protocol_revision must be a positive integer")
    if data["status"] != "frozen_protocol":
        raise ValueError("WP2 status must be frozen_protocol")
    variant_order = configured_variant_order(data)
    if tuple(data["variants"]) != variant_order:
        raise ValueError("WP2 variants must follow variant_order")
    if len(data["formal_seeds"]) < 5 or len(set(data["formal_seeds"])) < 5:
        raise ValueError("WP2 formal protocol requires at least five unique seeds")
    if len(data["pilot_training"]["seeds"]) != 1:
        raise ValueError("WP2 pilot protocol must use exactly one seed")
    if data["shared_network"]["state_order"] != list(STATE_NAMES):
        raise ValueError(f"WP2 state order must be {list(STATE_NAMES)}")
    variants = data["variants"]
    cumulative = (
        "input_normalization",
        "output_normalization",
        "residual_normalization",
        "gradient_balance",
        "hard_determinable_constraints",
        "adaptive_resampling",
    )
    for name in cumulative:
        seen = False
        for variant in variant_order:
            value = bool(variants[variant][name])
            if seen and not value:
                raise ValueError(f"WP2 component {name} is not cumulative at {variant}")
            seen = seen or value
    causal_bins = [int(variants[variant]["causal_bins"]) for variant in variant_order]
    if any(value < 1 for value in causal_bins) or any(
        fine < coarse for coarse, fine in zip(causal_bins[:-1], causal_bins[1:])
    ):
        raise ValueError("WP2 causal hierarchy must be positive and cumulative")
    allowed_sampling = {
        "uniform_latin_hypercube",
        "characteristic_uniform_mixture",
    }
    for variant, spec in variants.items():
        if spec["sampling"] not in allowed_sampling:
            raise ValueError(f"unsupported sampling for {variant}")
    required_variants = data["evaluation"]["formal_acceptance"].get(
        "required_variants", []
    )
    if not required_variants or not set(required_variants) <= set(variant_order):
        raise ValueError("formal_acceptance.required_variants must name WP2 variants")
    return data


def configured_variant_order(config: dict[str, Any]) -> tuple[str, ...]:
    order = tuple(config.get("variant_order", config.get("variants", {})))
    if not order or len(set(order)) != len(order):
        raise ValueError("WP2 variant_order must be a non-empty unique list")
    return order


def load_physical_case(config: dict[str, Any]) -> PhysicalParameters:
    expected = (DEFAULT_CONFIG.parent / config["physical_case_config"]).resolve()
    if expected != WP0_CONFIG.resolve():
        raise ValueError(
            "WP2 physical_case_config must resolve to the frozen WP0 baseline"
        )
    return validate_wp0_config(load_wp0_config(expected), WP0_PROVENANCE)


def training_config(
    config: dict[str, Any], mode: str, seed: int
) -> TrainingConfig:
    formal = config["formal_training"]
    shared = config["shared_network"]
    values = {
        "epochs": formal["epochs"],
        "lbfgs_steps": formal["lbfgs_steps"],
        "hidden_width": shared["hidden_width"],
        "hidden_layers": shared["hidden_layers"],
        "pde_points": formal["pde_points"],
        "initial_points": formal["initial_points"],
        "boundary_points": formal["boundary_points"],
        "learning_rate": formal["learning_rate"],
        "gradient_balance_every": formal["gradient_balance_every"],
        "adaptive_resample_every": formal["adaptive_resample_every"],
        "adaptive_pool_factor": formal["adaptive_pool_factor"],
        "checkpoint_every": formal["checkpoint_every"],
        "plot_x": formal["plot_x"],
        "plot_t": formal["plot_t"],
        "dtype": formal["dtype"],
        "device": formal["device"],
        "seed": seed,
    }
    if mode == "smoke":
        smoke = config["smoke_training"]
        for name in (
            "epochs",
            "lbfgs_steps",
            "hidden_width",
            "hidden_layers",
            "pde_points",
            "initial_points",
            "boundary_points",
            "gradient_balance_every",
            "adaptive_resample_every",
            "adaptive_pool_factor",
            "checkpoint_every",
            "plot_x",
            "plot_t",
        ):
            values[name] = smoke[name]
    elif mode == "pilot":
        pilot = config["pilot_training"]
        for name in (
            "epochs",
            "lbfgs_steps",
            "checkpoint_every",
            "plot_x",
            "plot_t",
        ):
            values[name] = pilot[name]
    result = TrainingConfig(**values)
    for name in (
        "epochs",
        "hidden_width",
        "hidden_layers",
        "pde_points",
        "initial_points",
        "boundary_points",
        "plot_x",
        "plot_t",
        "checkpoint_every",
    ):
        if getattr(result, name) < 1:
            raise ValueError(f"training setting {name} must be positive")
    return result


def select_device(config: TrainingConfig) -> torch.device:
    torch.set_default_dtype(
        torch.float64 if config.dtype == "float64" else torch.float32
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def valve_relative_velocity(
    params: PhysicalParameters, time_values: torch.Tensor
) -> torch.Tensor:
    close_time = params.valve_close_time_s
    if close_time <= 0.0:
        return torch.zeros_like(time_values)
    phase = torch.clamp(time_values / close_time, 0.0, 1.0)
    return 0.5 * params.initial_velocity_m_s * (
        1.0 + torch.cos(math.pi * phase)
    )


def initial_state(params: PhysicalParameters) -> np.ndarray:
    pressure = initial_pressure_pa(params)
    area_f, area_t = cross_section_areas(params)
    return np.asarray(
        [params.initial_velocity_m_s, 0.0, pressure, area_f * pressure / area_t],
        dtype=float,
    )


class FullDomainPINN(nn.Module):
    """One shared MLP with optional nondimensionalization and hard transforms."""

    def __init__(
        self,
        params: PhysicalParameters,
        train: TrainingConfig,
        variant: dict[str, Any],
    ) -> None:
        super().__init__()
        self.params = params
        self.variant = variant
        frequencies = tuple(float(value) for value in variant.get("fourier_frequencies", []))
        if any(value <= 0.0 for value in frequencies):
            raise ValueError("fourier_frequencies must contain positive values")
        self.fourier_frequencies = frequencies
        input_width = 2 + 4 * len(frequencies)
        modules: list[nn.Module] = [
            nn.Linear(input_width, train.hidden_width),
            nn.Tanh(),
        ]
        for _ in range(train.hidden_layers - 1):
            modules.extend([nn.Linear(train.hidden_width, train.hidden_width), nn.Tanh()])
        modules.append(nn.Linear(train.hidden_width, 4))
        self.network = nn.Sequential(*modules)
        self.register_buffer(
            "scales", torch.as_tensor(state_scales(params), dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "q0", torch.as_tensor(initial_state(params), dtype=torch.get_default_dtype())
        )

    def network_input(self, points: torch.Tensor) -> torch.Tensor:
        x_unit = points[:, 0:1] / self.params.length_m
        t_unit = points[:, 1:2] / self.params.t_final_s
        base = (
            torch.cat([2.0 * x_unit - 1.0, 2.0 * t_unit - 1.0], dim=1)
            if self.variant["input_normalization"]
            else points
        )
        if not self.fourier_frequencies:
            return base
        features = [base]
        for frequency in self.fourier_frequencies:
            phase_x = 2.0 * math.pi * frequency * x_unit
            phase_t = 2.0 * math.pi * frequency * t_unit
            features.extend(
                [
                    torch.sin(phase_x),
                    torch.cos(phase_x),
                    torch.sin(phase_t),
                    torch.cos(phase_t),
                ]
            )
        return torch.cat(features, dim=1)

    def initial_gate(self, time_values: torch.Tensor) -> torch.Tensor:
        scale = self.variant.get("initial_gate_time_scale_s")
        if scale is None:
            return time_values / self.params.t_final_s
        scale_value = float(scale)
        if scale_value <= 0.0:
            raise ValueError("initial_gate_time_scale_s must be positive")
        return 1.0 - torch.exp(-time_values / scale_value)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        raw = self.network(self.network_input(points))
        correction = raw * self.scales if self.variant["output_normalization"] else raw
        if not self.variant["hard_determinable_constraints"]:
            return correction

        xi = points[:, 0:1] / self.params.length_m
        gate = self.initial_gate(points[:, 1:2])
        uz = gate * xi * correction[:, 1:2]
        pressure = self.q0[2] + gate * xi * correction[:, 2:3]
        stress = self.q0[3] + gate * correction[:, 3:4]
        free_velocity = self.q0[0] + gate * correction[:, 0:1]
        valve_velocity = uz + valve_relative_velocity(
            self.params, points[:, 1:2]
        )
        velocity = (1.0 - xi) * free_velocity + xi * valve_velocity
        return torch.cat([velocity, uz, pressure, stress], dim=1)


def latin_hypercube(
    count: int,
    params: PhysicalParameters,
    rng: np.random.Generator,
    time_horizon_s: float | None = None,
) -> np.ndarray:
    x = (rng.permutation(count) + rng.random(count)) / count
    t = (rng.permutation(count) + rng.random(count)) / count
    horizon = params.t_final_s if time_horizon_s is None else float(time_horizon_s)
    return np.column_stack([x * params.length_m, t * horizon])


def reflected_characteristic_points(
    count: int,
    params: PhysicalParameters,
    rng: np.random.Generator,
    time_horizon_s: float | None = None,
) -> np.ndarray:
    speeds = np.unique(np.round(np.abs(characteristic_basis(params)[0]), 12))
    horizon = params.t_final_s if time_horizon_s is None else float(time_horizon_s)
    times = rng.uniform(0.0, horizon, count)
    events = rng.uniform(0.0, 1.0, count) * np.minimum(
        times, params.valve_close_time_s
    )
    selected = rng.choice(speeds, size=count)
    travel = selected * np.maximum(times - events, 0.0)
    folded = np.mod(params.length_m - travel, 2.0 * params.length_m)
    positions = np.where(
        folded <= params.length_m,
        folded,
        2.0 * params.length_m - folded,
    )
    positions += rng.normal(0.0, 0.01 * params.length_m, count)
    return np.column_stack(
        [np.clip(positions, 0.0, params.length_m), times]
    )


def sample_pde_points(
    count: int,
    params: PhysicalParameters,
    variant: dict[str, Any],
    rng: np.random.Generator,
    time_horizon_s: float | None = None,
) -> np.ndarray:
    if variant["sampling"] == "uniform_latin_hypercube":
        return latin_hypercube(count, params, rng, time_horizon_s)
    characteristic_count = count // 2
    return np.vstack(
        [
            latin_hypercube(
                count - characteristic_count, params, rng, time_horizon_s
            ),
            reflected_characteristic_points(
                characteristic_count, params, rng, time_horizon_s
            ),
        ]
    )


def make_training_points(
    params: PhysicalParameters,
    train: TrainingConfig,
    variant: dict[str, Any],
    device: torch.device,
    time_horizon_s: float | None = None,
    rng_offset: int = 0,
) -> dict[str, torch.Tensor]:
    horizon = params.t_final_s if time_horizon_s is None else float(time_horizon_s)
    if not 0.0 < horizon <= params.t_final_s:
        raise ValueError("training time horizon must lie in (0, t_final_s]")
    rng = np.random.default_rng(train.seed + 101 + int(rng_offset))
    pde = sample_pde_points(
        train.pde_points, params, variant, rng, time_horizon_s=horizon
    )
    initial = np.column_stack(
        [np.linspace(0.0, params.length_m, train.initial_points), np.zeros(train.initial_points)]
    )
    boundary_time = np.linspace(
        horizon / max(train.boundary_points, 2),
        horizon,
        train.boundary_points,
    )
    left = np.column_stack([np.zeros_like(boundary_time), boundary_time])
    right = np.column_stack(
        [np.full_like(boundary_time, params.length_m), boundary_time]
    )
    return {
        name: torch.as_tensor(values, device=device)
        for name, values in {
            "pde": pde,
            "initial": initial,
            "left": left,
            "right": right,
        }.items()
    }


def pde_residuals(
    model: FullDomainPINN,
    points: torch.Tensor,
    params: PhysicalParameters,
    normalized: bool,
    create_graph: bool = True,
) -> torch.Tensor:
    zt = points.detach().clone().requires_grad_(True)
    state = model(zt)
    derivatives = [
        torch.autograd.grad(
            state[:, index : index + 1],
            zt,
            grad_outputs=torch.ones_like(state[:, index : index + 1]),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        for index in range(4)
    ]
    state_x = torch.cat([item[:, 0:1] for item in derivatives], dim=1)
    state_t = torch.cat([item[:, 1:2] for item in derivatives], dim=1)
    matrix_a_np, matrix_b_np = system_matrices(params)
    matrix_a = torch.as_tensor(matrix_a_np, device=zt.device, dtype=zt.dtype)
    matrix_b = torch.as_tensor(matrix_b_np, device=zt.device, dtype=zt.dtype)
    residual = state_t @ matrix_a.T + state_x @ matrix_b.T
    if normalized:
        scales = torch.as_tensor(
            state_scales(params), device=zt.device, dtype=zt.dtype
        )
        row_scale = (
            torch.abs(matrix_a) @ scales / params.t_final_s
            + torch.abs(matrix_b) @ scales / params.length_m
        )
        residual = residual / torch.clamp(row_scale, min=1.0e-30)
    return residual


def causal_pde_loss(
    residual: torch.Tensor,
    time_values: torch.Tensor,
    params: PhysicalParameters,
    bins: int,
    epsilon: float,
    minimum_weight: float = 0.0,
    time_horizon_s: float | None = None,
) -> tuple[torch.Tensor, list[float]]:
    point_loss = torch.mean(residual**2, dim=1)
    if bins <= 1:
        return torch.mean(point_loss), [1.0]
    horizon = params.t_final_s if time_horizon_s is None else float(time_horizon_s)
    if horizon <= 0.0:
        raise ValueError("causal time horizon must be positive")
    indices = torch.clamp(
        (bins * time_values.flatten() / horizon).long(), 0, bins - 1
    )
    losses: list[torch.Tensor] = []
    for index in range(bins):
        selected = point_loss[indices == index]
        losses.append(torch.mean(selected) if len(selected) else torch.mean(point_loss) * 0.0)
    weights: list[torch.Tensor] = []
    cumulative = torch.zeros((), device=residual.device, dtype=residual.dtype)
    for loss in losses:
        weights.append(
            torch.clamp(
                torch.exp(-epsilon * cumulative.detach()),
                min=float(minimum_weight),
            )
        )
        cumulative = cumulative + loss
    stacked_weights = torch.stack(weights)
    value = torch.sum(stacked_weights * torch.stack(losses)) / torch.sum(stacked_weights)
    return value, [float(item.detach().cpu()) for item in weights]


def loss_terms(
    model: FullDomainPINN,
    points: dict[str, torch.Tensor],
    params: PhysicalParameters,
    variant: dict[str, Any],
    training_progress: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    residual = pde_residuals(
        model,
        points["pde"],
        params,
        bool(variant["residual_normalization"]),
    )
    progress = float(np.clip(training_progress, 0.0, 1.0))
    epsilon_start = float(
        variant.get("causal_epsilon_start", variant["causal_epsilon"])
    )
    epsilon_end = float(
        variant.get("causal_epsilon_end", variant["causal_epsilon"])
    )
    causal_epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)
    minimum_causal_weight = float(variant.get("minimum_causal_weight", 0.0))
    pde_loss, causal_weights = causal_pde_loss(
        residual,
        points["pde"][:, 1],
        params,
        int(variant["causal_bins"]),
        causal_epsilon,
        minimum_causal_weight,
        time_horizon_s=float(torch.max(points["pde"][:, 1]).detach().cpu()),
    )
    normalized = bool(variant["residual_normalization"])
    scales = model.scales if normalized else torch.ones_like(model.scales)
    q0 = model.q0

    initial = model(points["initial"])
    initial_loss = torch.mean(((initial - q0) / scales) ** 2)
    left = model(points["left"])
    upstream_residual = torch.stack(
        [
            (left[:, 2] - q0[2]) / scales[2],
            left[:, 1] / scales[1],
        ],
        dim=1,
    )
    upstream_loss = torch.mean(upstream_residual**2)

    right_points = points["right"].detach().clone().requires_grad_(True)
    right = model(right_points)
    relative_target = valve_relative_velocity(params, right_points[:, 1:2])
    kinematic = (right[:, 0:1] - right[:, 1:2] - relative_target) / scales[0]
    kinematic_loss = torch.mean(kinematic**2)
    uz_t = torch.autograd.grad(
        right[:, 1:2],
        right_points,
        grad_outputs=torch.ones_like(right[:, 1:2]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]
    area_f, area_t = cross_section_areas(params)
    force = (
        params.valve_mass_kg * uz_t
        - area_f * right[:, 2:3]
        + area_t * right[:, 3:4]
    )
    force_scale = (
        max(
            area_f * float(model.scales[2].detach().cpu()),
            area_t * float(model.scales[3].detach().cpu()),
            1.0,
        )
        if normalized
        else 1.0
    )
    dynamic_loss = torch.mean((force / force_scale) ** 2)
    terms = {
        "pde": pde_loss,
        "initial": initial_loss,
        "upstream": upstream_loss,
        "valve_kinematic": kinematic_loss,
        "valve_dynamic": dynamic_loss,
    }
    diagnostics = {
        "pde_component_mse": [
            float(torch.mean(residual[:, index] ** 2).detach().cpu())
            for index in range(4)
        ],
        "causal_weights": causal_weights,
        "causal_epsilon": causal_epsilon,
        "minimum_causal_weight": minimum_causal_weight,
        "active_time_horizon_s": float(
            torch.max(points["pde"][:, 1]).detach().cpu()
        ),
    }
    return terms, diagnostics


class GradientNormBalancer:
    def __init__(self, names: Iterable[str]) -> None:
        self.names = tuple(names)
        self.weights = {name: 1.0 for name in self.names}
        self.beta = 0.9

    def update(
        self, losses: dict[str, torch.Tensor], model: nn.Module
    ) -> None:
        parameters = tuple(item for item in model.parameters() if item.requires_grad)
        norms: dict[str, float] = {}
        for name in self.names:
            if float(losses[name].detach().cpu()) <= 1.0e-30:
                norms[name] = 0.0
                continue
            gradients = torch.autograd.grad(
                losses[name], parameters, retain_graph=True, allow_unused=True
            )
            squared = sum(
                torch.sum(item.detach() ** 2)
                for item in gradients
                if item is not None
            )
            norms[name] = math.sqrt(max(float(squared.cpu()), 0.0))
        positive = [value for value in norms.values() if value > 1.0e-30]
        if not positive:
            return
        target = float(np.exp(np.mean(np.log(positive))))
        proposed = {
            name: (target / value if value > 1.0e-30 else self.weights[name])
            for name, value in norms.items()
        }
        normalizer = len(proposed) / max(sum(proposed.values()), 1.0e-30)
        for name in self.names:
            desired = float(np.clip(proposed[name] * normalizer, 0.05, 20.0))
            self.weights[name] = self.beta * self.weights[name] + (1.0 - self.beta) * desired
        normalizer = len(self.weights) / max(sum(self.weights.values()), 1.0e-30)
        for name in self.names:
            self.weights[name] = float(
                np.clip(self.weights[name] * normalizer, 0.05, 20.0)
            )

    def total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.weights[name] * losses[name] for name in self.names)


def adaptive_resample(
    model: FullDomainPINN,
    params: PhysicalParameters,
    train: TrainingConfig,
    variant: dict[str, Any],
    points: dict[str, torch.Tensor],
    rng: np.random.Generator,
) -> dict[str, Any]:
    pool_factor = int(
        variant.get("adaptive_pool_factor", train.adaptive_pool_factor)
    )
    pool_count = train.pde_points * pool_factor
    candidates_np = sample_pde_points(pool_count, params, variant, rng)
    candidates = torch.as_tensor(candidates_np, device=points["pde"].device)
    residual = pde_residuals(
        model,
        candidates,
        params,
        bool(variant["residual_normalization"]),
        create_graph=False,
    )
    score = torch.mean(residual**2, dim=1).detach().cpu().numpy()
    replacement_fraction = float(
        variant.get("adaptive_replacement_fraction", 1.0)
    )
    if not 0.0 < replacement_fraction <= 1.0:
        raise ValueError("adaptive_replacement_fraction must be in (0, 1]")
    replacement_count = min(
        train.pde_points,
        max(1, int(round(train.pde_points * replacement_fraction))),
    )
    time_bins = int(variant.get("adaptive_time_bins", 1))
    if time_bins <= 1:
        selected = np.argpartition(score, -replacement_count)[
            -replacement_count:
        ]
    else:
        candidate_times = candidates_np[:, 1]
        bin_indices = np.clip(
            (time_bins * candidate_times / params.t_final_s).astype(int),
            0,
            time_bins - 1,
        )
        quotas = np.full(time_bins, replacement_count // time_bins, dtype=int)
        quotas[: replacement_count % time_bins] += 1
        selected_parts: list[np.ndarray] = []
        used = np.zeros(pool_count, dtype=bool)
        for bin_index, quota in enumerate(quotas):
            available = np.flatnonzero(bin_indices == bin_index)
            take = min(int(quota), len(available))
            if take:
                chosen = available[np.argsort(score[available])[-take:]]
                selected_parts.append(chosen)
                used[chosen] = True
        selected = (
            np.concatenate(selected_parts)
            if selected_parts
            else np.empty(0, dtype=int)
        )
        remaining = replacement_count - len(selected)
        if remaining:
            available = np.flatnonzero(~used)
            fill = available[np.argsort(score[available])[-remaining:]]
            selected = np.concatenate([selected, fill])
    selected_tensor = torch.as_tensor(selected, device=candidates.device)
    adaptive_points = candidates[selected_tensor].detach()
    retained_count = train.pde_points - replacement_count
    if retained_count:
        if "pde_base" not in points:
            points["pde_base"] = points["pde"].detach().clone()
        base = points["pde_base"]
        retained_indices = torch.linspace(
            0,
            len(base) - 1,
            retained_count,
            device=base.device,
        ).long()
        points["pde"] = torch.cat(
            [base[retained_indices], adaptive_points], dim=0
        ).detach()
    else:
        points["pde"] = adaptive_points
    selected_bins = np.clip(
        (
            time_bins
            * candidates_np[selected, 1]
            / params.t_final_s
        ).astype(int),
        0,
        max(time_bins - 1, 0),
    )
    return {
        "pool_count": pool_count,
        "replacement_count": replacement_count,
        "retained_base_count": retained_count,
        "replacement_fraction": replacement_fraction,
        "time_bins": time_bins,
        "selected_per_time_bin": np.bincount(
            selected_bins, minlength=max(time_bins, 1)
        ).tolist(),
        "selected_score_min": float(np.min(score[selected])),
        "selected_score_max": float(np.max(score[selected])),
    }


def predict_grid(
    model: FullDomainPINN,
    params: PhysicalParameters,
    train: TrainingConfig,
) -> dict[str, np.ndarray]:
    x = np.linspace(0.0, params.length_m, train.plot_x)
    t = np.linspace(0.0, params.t_final_s, train.plot_t)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    device = next(model.parameters()).device
    inputs = torch.as_tensor(
        np.column_stack([xx.ravel(), tt.ravel()]), device=device
    )
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), 8192):
            chunks.append(model(inputs[start : start + 8192]).cpu().numpy())
    values = np.vstack(chunks).reshape(len(x), len(t), 4)
    result: dict[str, np.ndarray] = {"x": x, "t": t}
    for index, name in enumerate(STATE_NAMES):
        result[name] = values[:, :, index]
    return result


def run_signature(
    variant_name: str,
    variant: dict[str, Any],
    params: PhysicalParameters,
    train: TrainingConfig,
) -> dict[str, Any]:
    return {
        "model_revision": WP2_MODEL_REVISION,
        "variant": variant_name,
        "variant_spec": variant,
        "physical_parameters": asdict(params),
        "training_config": asdict(train),
    }


def load_completed_result(
    path: Path, signature: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("run_signature") != signature:
                return None
            result = {
                name: np.asarray(archive[name])
                for name in ("x", "t", *STATE_NAMES)
            }
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    cached_metadata = dict(metadata)
    cached_metadata["execution_status"] = "cache_hit"
    return result, cached_metadata


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def save_adam_checkpoint(
    path: Path,
    signature: dict[str, Any],
    epoch: int,
    model: FullDomainPINN,
    optimizer: torch.optim.Optimizer,
    balancer: GradientNormBalancer,
    points: dict[str, torch.Tensor],
    history: list[dict[str, Any]],
    rng: np.random.Generator,
    accumulated_training_seconds: float,
    device: torch.device,
) -> None:
    payload = {
        "run_signature": signature,
        "stage": "adam",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "balancer_weights": dict(balancer.weights),
        "points": {name: value.detach().cpu() for name, value in points.items()},
        "history": history,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        ),
        "accumulated_training_seconds": accumulated_training_seconds,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_adam_checkpoint(
    path: Path,
    signature: dict[str, Any],
    model: FullDomainPINN,
    optimizer: torch.optim.Optimizer,
    balancer: GradientNormBalancer,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[int, dict[str, torch.Tensor], list[dict[str, Any]], float] | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError):
        return None
    if payload.get("run_signature") != signature or payload.get("stage") != "adam":
        return None
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _optimizer_to(optimizer, device)
    balancer.weights = {
        str(name): float(value)
        for name, value in payload["balancer_weights"].items()
    }
    points = {
        str(name): value.to(device) for name, value in payload["points"].items()
    }
    history = list(payload["history"])
    rng.bit_generator.state = payload["numpy_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"])
    if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return (
        int(payload["epoch"]),
        points,
        history,
        float(payload.get("accumulated_training_seconds", 0.0)),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_curriculum_stage(epoch: int, epochs: int, windows: int) -> int:
    if epoch < 1 or epochs < 1 or windows < 1:
        raise ValueError("epoch, epochs and windows must be positive")
    # ceil(epoch * windows / epochs) gives every stage the same allocation
    # when epochs is divisible by windows and always reaches the full horizon.
    return min(windows, (epoch * windows + epochs - 1) // epochs)


def time_curriculum_horizon(
    params: PhysicalParameters, epoch: int, epochs: int, windows: int
) -> float:
    return (
        params.t_final_s
        * time_curriculum_stage(epoch, epochs, windows)
        / windows
    )


def train_variant(
    variant_name: str,
    variant: dict[str, Any],
    params: PhysicalParameters,
    train: TrainingConfig,
    output_dir: Path,
    force: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = run_signature(variant_name, variant, params, train)
    result_path = output_dir / "results.npz"
    checkpoint_path = output_dir / "checkpoint.pt"
    if not force:
        cached = load_completed_result(result_path, signature)
        if cached is not None:
            print(
                f"[{variant_name}/seed={train.seed}] cache hit: {result_path}"
            )
            return cached
    elif checkpoint_path.exists():
        checkpoint_path.unlink()

    device = select_device(train)
    model = FullDomainPINN(params, train, variant).to(device)
    curriculum_windows = int(variant.get("time_curriculum_windows", 1))
    if curriculum_windows < 1:
        raise ValueError("time_curriculum_windows must be positive")
    initial_horizon = time_curriculum_horizon(
        params, 1, train.epochs, curriculum_windows
    )
    points = make_training_points(
        params,
        train,
        variant,
        device,
        time_horizon_s=initial_horizon,
        rng_offset=0 if curriculum_windows == 1 else 1009,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train.learning_rate)
    balancer = GradientNormBalancer(
        ("pde", "initial", "upstream", "valve_kinematic", "valve_dynamic")
    )
    rng = np.random.default_rng(train.seed + 701)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    accumulated_training_seconds = 0.0
    execution_status = "new_training"
    if not force:
        restored = restore_adam_checkpoint(
            checkpoint_path,
            signature,
            model,
            optimizer,
            balancer,
            device,
            rng,
        )
        if restored is not None:
            start_epoch, points, history, accumulated_training_seconds = restored
            execution_status = "resumed_checkpoint"
            print(
                f"[{variant_name}/seed={train.seed}] resuming Adam from "
                f"epoch {start_epoch}/{train.epochs}"
            )
    current_curriculum_stage = time_curriculum_stage(
        max(start_epoch, 1), train.epochs, curriculum_windows
    )
    _synchronize(device)
    session_started = time.perf_counter()
    print(
        f"[{variant_name}/seed={train.seed}] full-domain PINN on {device}; "
        f"epochs={train.epochs}, labels=0, anchors=0"
    )
    for epoch in range(start_epoch + 1, train.epochs + 1):
        target_stage = time_curriculum_stage(
            epoch, train.epochs, curriculum_windows
        )
        if target_stage != current_curriculum_stage:
            current_curriculum_stage = target_stage
            active_horizon = (
                params.t_final_s * current_curriculum_stage / curriculum_windows
            )
            points = make_training_points(
                params,
                train,
                variant,
                device,
                time_horizon_s=active_horizon,
                rng_offset=1009 * current_curriculum_stage,
            )
            print(
                f"[{variant_name}/seed={train.seed}] expanding time curriculum "
                f"to stage {current_curriculum_stage}/{curriculum_windows}, "
                f"t<={active_horizon:.6g} s"
            )
        optimizer.zero_grad(set_to_none=True)
        losses, diagnostics = loss_terms(
            model,
            points,
            params,
            variant,
            training_progress=epoch / train.epochs,
        )
        if variant["gradient_balance"] and (
            epoch == 1 or epoch % train.gradient_balance_every == 0
        ):
            balancer.update(losses, model)
        total = balancer.total(losses)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        item: dict[str, Any] = {
            "epoch": epoch,
            "time_curriculum_stage": current_curriculum_stage,
            "time_curriculum_windows": curriculum_windows,
            "loss_total": float(total.detach().cpu()),
            **{
                f"loss_{name}": float(value.detach().cpu())
                for name, value in losses.items()
            },
            **{f"weight_{name}": balancer.weights[name] for name in losses},
            **diagnostics,
        }
        history.append(item)
        adaptive_every = int(
            variant.get(
                "adaptive_resample_every", train.adaptive_resample_every
            )
        )
        if variant["adaptive_resampling"] and epoch % adaptive_every == 0:
            item["adaptive_resample"] = adaptive_resample(
                model, params, train, variant, points, rng
            )
        if epoch == 1 or epoch == train.epochs or epoch % 100 == 0:
            print(
                f"[{variant_name}/seed={train.seed}] epoch={epoch:5d} "
                f"loss={item['loss_total']:.3e}"
            )
        if epoch % train.checkpoint_every == 0 or epoch == train.epochs:
            _synchronize(device)
            elapsed_to_checkpoint = (
                accumulated_training_seconds
                + time.perf_counter()
                - session_started
            )
            save_adam_checkpoint(
                checkpoint_path,
                signature,
                epoch,
                model,
                optimizer,
                balancer,
                points,
                history,
                rng,
                elapsed_to_checkpoint,
                device,
            )
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    if train.lbfgs_steps > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=train.lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad(set_to_none=True)
            losses, _ = loss_terms(
                model, points, params, variant, training_progress=1.0
            )
            value = balancer.total(losses)
            value.backward()
            return value

        optimizer_lbfgs.step(closure)
        final_losses, final_diagnostics = loss_terms(
            model, points, params, variant, training_progress=1.0
        )
        final_total = balancer.total(final_losses)
        history.append(
            {
                "epoch": train.epochs,
                "optimizer_stage": "lbfgs_final",
                "loss_total": float(final_total.detach().cpu()),
                **{
                    f"loss_{name}": float(value.detach().cpu())
                    for name, value in final_losses.items()
                },
                **{
                    f"weight_{name}": balancer.weights[name]
                    for name in final_losses
                },
                **final_diagnostics,
            }
        )

    _synchronize(device)
    elapsed = (
        accumulated_training_seconds + time.perf_counter() - session_started
    )
    _synchronize(device)
    prediction_started = time.perf_counter()
    result = predict_grid(model, params, train)
    _synchronize(device)
    prediction_seconds = time.perf_counter() - prediction_started
    finite = all(np.all(np.isfinite(result[name])) for name in STATE_NAMES)
    metadata = {
        "run_signature": signature,
        "model_revision": WP2_MODEL_REVISION,
        "variant": variant_name,
        "variant_spec": variant,
        "physical_parameters": asdict(params),
        "training_config": asdict(train),
        "uses_labels": False,
        "uses_anchor_points": False,
        "training_seconds": elapsed,
        "prediction_seconds": prediction_seconds,
        "trainable_parameters": sum(
            item.numel() for item in model.parameters() if item.requires_grad
        ),
        "device": str(device),
        "finite_output": finite,
        "final_history": history[-1],
        "execution_status": execution_status,
        "resumed_from_epoch": start_epoch,
    }
    temporary_result = output_dir / "results.tmp"
    with temporary_result.open("wb") as stream:
        np.savez_compressed(
            stream,
            **result,
            metadata=json.dumps(metadata, ensure_ascii=False),
        )
    temporary_result.replace(result_path)
    temporary_model = output_dir / "model.tmp"
    torch.save(
        {"state_dict": model.state_dict(), "metadata": metadata},
        temporary_model,
    )
    temporary_model.replace(output_dir / "model.pt")
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.semilogy(
        [item["epoch"] for item in history],
        [max(float(item["loss_total"]), 1.0e-30) for item in history],
        label="total",
    )
    for loss_name in ("pde", "initial", "upstream", "valve_kinematic", "valve_dynamic"):
        axis.semilogy(
            [item["epoch"] for item in history],
            [max(float(item[f"loss_{loss_name}"]), 1.0e-30) for item in history],
            label=loss_name,
            alpha=0.8,
        )
    axis.set(xlabel="epoch", ylabel="loss", title=f"{variant_name}, seed={train.seed}")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=170)
    plt.close(figure)
    return result, metadata


def reference_solution(
    params: PhysicalParameters,
    times: np.ndarray,
    n_cells: int,
    cfl: float,
) -> dict[str, np.ndarray]:
    from water16_reproduction.figure09.pinn_compare import (
        solve_matched_fssi_reference,
    )

    return solve_matched_fssi_reference(
        params, times, n_cells=n_cells, cfl=cfl
    )


def sample_reference(
    reference: dict[str, np.ndarray], position: float, state: str
) -> np.ndarray:
    return np.asarray(
        [
            np.interp(position, reference["z"], reference[state][:, index])
            for index in range(len(reference["t"]))
        ]
    )


def signal_metrics(
    reference: np.ndarray, prediction: np.ndarray, times: np.ndarray
) -> dict[str, float]:
    difference = prediction - reference
    scale = max(float(np.ptp(reference)), float(np.max(np.abs(reference))), 1.0e-12)
    centered_reference = reference - np.mean(reference)
    centered_prediction = prediction - np.mean(prediction)
    correlation = np.correlate(
        centered_prediction, centered_reference, mode="full"
    )
    lag_index = int(np.argmax(correlation)) - (len(reference) - 1)
    dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.0
    return {
        "relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), 1.0e-30)
        ),
        "nrmse_by_reference_scale": float(np.sqrt(np.mean(difference**2)) / scale),
        "peak_error": float(np.max(prediction) - np.max(reference)),
        "trough_error": float(np.min(prediction) - np.min(reference)),
        "cross_correlation_lag_s": lag_index * dt,
    }


def evaluate_result(
    result: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    params: PhysicalParameters,
    positions_over_l: Iterable[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ratio in positions_over_l:
        position = float(ratio) * params.length_m
        prediction_index = int(np.argmin(np.abs(result["x"] - position)))
        for state in STATE_NAMES:
            reference_signal = sample_reference(reference, position, state)
            prediction = np.interp(
                reference["t"], result["t"], result[state][prediction_index]
            )
            rows.append(
                {
                    "position_over_L": float(ratio),
                    "state": state,
                    **signal_metrics(reference_signal, prediction, reference["t"]),
                }
            )
    return rows


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = (
        "relative_l2",
        "nrmse_by_reference_scale",
        "peak_error",
        "trough_error",
        "cross_correlation_lag_s",
    )
    groups: dict[tuple[str, float, str], list[dict[str, float]]] = {}
    for record in records:
        for metric in record["metrics"]:
            key = (
                str(record["variant"]),
                float(metric["position_over_L"]),
                str(metric["state"]),
            )
            groups.setdefault(key, []).append(metric)
    rows: list[dict[str, Any]] = []
    for (variant, position, state), values in sorted(groups.items()):
        row: dict[str, Any] = {
            "variant": variant,
            "position_over_L": position,
            "state": state,
            "seed_count": len(values),
        }
        for name in metric_names:
            array = np.asarray([float(item[name]) for item in values])
            row[f"{name}_mean"] = float(np.mean(array))
            row[f"{name}_std"] = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
        rows.append(row)
    return rows


def write_metric_csvs(
    output_dir: Path,
    mode: str,
    records: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
) -> None:
    detailed: list[dict[str, Any]] = []
    for record in records:
        for metric in record["metrics"]:
            detailed.append(
                {
                    "variant": record["variant"],
                    "seed": record["seed"],
                    **metric,
                }
            )
    for suffix, rows in (("metrics", detailed), ("aggregate", aggregate)):
        path = output_dir / f"wp2_{mode}_{suffix}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)


def formal_acceptance(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    thresholds = config["evaluation"]["formal_acceptance"]
    required_variants = tuple(str(item) for item in thresholds["required_variants"])
    state_limits = thresholds["maximum_nrmse_by_state"]
    lag_limit = float(thresholds["maximum_absolute_cross_correlation_lag_s"])
    checks: list[dict[str, Any]] = []
    for record in records:
        for metric in record["metrics"]:
            nrmse_pass = float(metric["nrmse_by_reference_scale"]) <= float(
                state_limits[metric["state"]]
            )
            lag_pass = abs(float(metric["cross_correlation_lag_s"])) <= lag_limit
            checks.append(
                {
                    "variant": record["variant"],
                    "seed": record["seed"],
                    "position_over_L": metric["position_over_L"],
                    "state": metric["state"],
                    "nrmse_pass": nrmse_pass,
                    "lag_pass": lag_pass,
                    "used_for_primary_acceptance": record["variant"]
                    in required_variants,
                }
            )
    expected = {
        (variant, int(seed), float(position), state)
        for variant in required_variants
        for seed in config["formal_seeds"]
        for position in config["evaluation"]["positions_over_L"]
        for state in config["evaluation"]["states"]
    }
    present = {
        (
            str(item["variant"]),
            int(item["seed"]),
            float(item["position_over_L"]),
            str(item["state"]),
        )
        for item in checks
        if item["used_for_primary_acceptance"]
    }
    missing = sorted(expected - present)
    primary_checks = [
        item for item in checks if item["used_for_primary_acceptance"]
    ]
    return {
        "thresholds": thresholds,
        "required_variants": list(required_variants),
        "missing_required_curves": [
            {
                "variant": item[0],
                "seed": item[1],
                "position_over_L": item[2],
                "state": item[3],
            }
            for item in missing
        ],
        "all_curves_pass": not missing
        and bool(primary_checks)
        and all(
            item["nrmse_pass"] and item["lag_pass"]
            for item in primary_checks
        ),
        "checks": checks,
    }


def plot_comparison(
    path: Path,
    results: dict[str, dict[str, np.ndarray]],
    reference: dict[str, np.ndarray],
    params: PhysicalParameters,
    mode: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for index, (state, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
        axis = axes.flat[index]
        axis.plot(
            reference["t"],
            sample_reference(reference, params.length_m, state),
            color="black",
            linewidth=1.5,
            label="M0 MOC",
        )
        for variant, result in results.items():
            axis.plot(result["t"], result[state][-1], linewidth=1.0, label=variant)
        axis.set_title(state)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        f"WP2 {mode} outputs at valve (not an accuracy result)", y=0.99
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=6,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_audit(config: dict[str, Any], params: PhysicalParameters) -> dict[str, Any]:
    speeds = characteristic_basis(params)[0]
    return {
        "status": "pass",
        "experiment_id": config["experiment_id"],
        "protocol_revision": config["protocol_revision"],
        "pre_formal_amendments": config["pre_formal_amendments"],
        "variant_order": list(configured_variant_order(config)),
        "formal_seeds": config["formal_seeds"],
        "state_order": list(STATE_NAMES),
        "signed_characteristic_speeds_m_s": speeds.tolist(),
        "uses_training_labels": False,
        "uses_anchor_points": False,
        "reference_role": config["reference"]["role"],
        "variant_components": config["variants"],
        "fairness_rules": config["fairness_rules"],
        "scope": "protocol audit only; no training and no accuracy claim",
    }


def run_wp2(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    mode: str = "audit",
    selected_variants: Iterable[str] | None = None,
    selected_seeds: Iterable[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config = load_wp2_config(config_path)
    params = load_physical_case(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(config, params)
    (output_dir / "wp2_protocol_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if mode == "audit":
        return audit

    protocol_variants = configured_variant_order(config)
    variants = (
        protocol_variants if selected_variants is None else tuple(selected_variants)
    )
    invalid = sorted(set(variants) - set(protocol_variants))
    if invalid:
        raise ValueError(f"unknown WP2 variants: {invalid}")
    if not variants:
        raise ValueError("at least one WP2 variant is required")
    if mode not in {"smoke", "pilot", "train"}:
        raise ValueError(f"unsupported WP2 mode: {mode}")
    default_seeds = (
        config["smoke_training"]["seeds"]
        if mode == "smoke"
        else (
            config["pilot_training"]["seeds"]
            if mode == "pilot"
            else config["formal_seeds"]
        )
    )
    seeds = [int(value) for value in (selected_seeds or default_seeds)]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("WP2 seeds must be a non-empty unique list")
    if mode == "pilot" and len(seeds) != 1:
        raise ValueError("WP2 pilot mode accepts exactly one seed")
    run_params = (
        replace(params, t_final_s=float(config["smoke_training"]["t_final_s"]))
        if mode == "smoke"
        else params
    )
    all_results: dict[str, dict[str, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    reference: dict[str, np.ndarray] | None = None
    for seed in seeds:
        train = training_config(config, mode, int(seed))
        if reference is None:
            times = np.linspace(0.0, run_params.t_final_s, train.plot_t)
            reference_key = {
                "smoke": "smoke_cells",
                "pilot": "pilot_cells",
                "train": "formal_cells",
            }[mode]
            cells = int(config["reference"][reference_key])
            reference = reference_solution(
                run_params, times, cells, float(config["reference"]["cfl"])
            )
        for variant_name in variants:
            result, metadata = train_variant(
                variant_name,
                config["variants"][variant_name],
                run_params,
                train,
                output_dir / mode / variant_name / f"seed_{seed}",
                force=force,
            )
            if len(seeds) == 1:
                all_results[variant_name] = result
            records.append(
                {
                    "variant": variant_name,
                    "seed": int(seed),
                    "finite_output": metadata["finite_output"],
                    "training_seconds": metadata["training_seconds"],
                    "prediction_seconds": metadata["prediction_seconds"],
                    "epochs": train.epochs,
                    "device": metadata["device"],
                    "execution_status": metadata["execution_status"],
                    "resumed_from_epoch": metadata["resumed_from_epoch"],
                    "metrics": evaluate_result(
                        result,
                        reference,
                        run_params,
                        config["evaluation"]["positions_over_L"],
                    ),
                }
            )

    assert reference is not None
    if len(seeds) == 1:
        plot_comparison(
            output_dir / f"wp2_{mode}_comparison.png",
            all_results,
            reference,
            run_params,
            mode,
        )
    finite = all(bool(item["finite_output"]) for item in records)
    aggregate = aggregate_records(records)
    write_metric_csvs(output_dir, mode, records, aggregate)
    accuracy = formal_acceptance(config, records) if mode == "train" else None
    resource_estimate = None
    if mode == "pilot":
        formal_epochs = int(config["formal_training"]["epochs"])
        seconds_per_epoch = [
            float(item["training_seconds"]) / max(int(item["epochs"]), 1)
            for item in records
        ]
        representative = float(np.mean(seconds_per_epoch))
        formal_runs = len(protocol_variants) * len(config["formal_seeds"])
        resource_estimate = {
            "observed_seconds_per_adam_epoch": representative,
            "projected_adam_hours_per_formal_run": representative
            * formal_epochs
            / 3600.0,
            "projected_adam_hours_for_all_25_runs": representative
            * formal_epochs
            * formal_runs
            / 3600.0,
            "formal_run_count": formal_runs,
            "exclusions": (
                "rough estimate only; excludes 200-step L-BFGS, evaluation, I/O, "
                "and variant-dependent P4 adaptive-sampling overhead"
            ),
        }
    overall_pass = finite and (
        mode != "train" or (accuracy is not None and accuracy["all_curves_pass"])
    )
    execution_status_counts: dict[str, int] = {}
    for item in records:
        status = str(item["execution_status"])
        execution_status_counts[status] = execution_status_counts.get(status, 0) + 1
    report = {
        "status": "pass" if overall_pass else "review_required",
        "mode": mode,
        "experiment_id": config["experiment_id"],
        "variants": list(variants),
        "seeds": seeds,
        "reference_cells": int(
            config["reference"][
                {"smoke": "smoke_cells", "pilot": "pilot_cells", "train": "formal_cells"}[mode]
            ]
        ),
        "records": records,
        "aggregate_metrics": aggregate,
        "formal_accuracy_acceptance": accuracy,
        "resource_estimate": resource_estimate,
        "execution_status_counts": execution_status_counts,
        "acceptance": {
            "all_outputs_finite": finite,
            "method_interfaces": "pass" if finite else "review_required",
            "pinn_accuracy": (
                "not assessed in two-epoch smoke mode"
                if mode == "smoke"
                else (
                    "not assessed in resource pilot mode"
                    if mode == "pilot"
                    else (
                        "pass"
                        if accuracy is not None and accuracy["all_curves_pass"]
                        else "review_required"
                    )
                )
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "MOC is used only after training for evaluation.",
            "Smoke and pilot metrics are diagnostics and must not be reported as accuracy evidence.",
            "Pilot time projection excludes L-BFGS and is not a scheduling guarantee.",
            "The valve dynamic condition remains a soft residual in P2-P4 because it contains the learned time derivative.",
            "Matching completed groups are cached; matching Adam checkpoints resume at the next epoch.",
            "An interruption during L-BFGS restarts L-BFGS from the completed Adam checkpoint; Adam epochs are not repeated.",
        ],
    }
    (output_dir / f"wp2_{mode}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WP2 P0-P4 PINN ablation protocol")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=("audit", "smoke", "pilot", "train"),
        default="audit",
        help=(
            "pilot runs one short formal-size network; train explicitly starts "
            "the full five-seed experiment"
        ),
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="comma-separated subset; pilot defaults to P0, other modes to P0-P4",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="optional comma-separated seeds; pilot accepts exactly one",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore matching completed results and start selected groups from epoch zero",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_wp2_config(args.config.resolve())
    protocol_variants = configured_variant_order(protocol)
    default_variants = (
        (protocol_variants[0],) if args.mode == "pilot" else protocol_variants
    )
    selected_variants = (
        default_variants
        if args.variants is None
        else tuple(item.strip() for item in args.variants.split(",") if item.strip())
    )
    selected_seeds = (
        None
        if args.seeds is None
        else tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    )
    report = run_wp2(
        args.config.resolve(),
        args.output_dir.resolve(),
        mode=args.mode,
        selected_variants=selected_variants,
        selected_seeds=selected_seeds,
        force=args.force,
    )
    summary = {
        "status": report["status"],
        "mode": report.get("mode", "audit"),
        "experiment_id": report["experiment_id"],
        "variants": report.get("variants", report.get("variant_order")),
        "pinn_accuracy": report.get("acceptance", {}).get(
            "pinn_accuracy", "not assessed in audit mode"
        ),
        "resource_estimate": report.get("resource_estimate"),
        "execution_status_counts": report.get("execution_status_counts"),
        "output": str(args.output_dir.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
