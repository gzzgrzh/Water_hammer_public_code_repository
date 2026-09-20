"""Pure PINN reproduction entry point with adaptive loss balancing.

Dynamic cases use only governing-equation, initial-condition, and
boundary-condition losses. No numerical reference solution or labelled
anchor point is used during training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from water16_reproduction.common.cases import case_identifier
from water16_reproduction.common.paper import save_comparison
from water16_reproduction.common.physics import (
    PhysicalParameters,
    SolverSettings,
    cross_section_areas,
    fssi_coefficients,
    initial_pressure_pa,
    pipe_soil_lambdas,
    pressure_wave_speed,
    radial_velocity,
    solve_fssi,
    spectrum,
    with_soil_ratio,
)


SystemKind = Literal["fssi", "two_equation"]
PURE_PINN_REVISION = 1
DYNAMIC_FIGURES = (6, 9, 10, 13, 14)
STATIC_FIGURES = (5, 7, 8, 11, 12)
SUPPORTED_FIGURES = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
_SESSION_RESULTS: Dict[str, Dict[str, np.ndarray]] = {}


@dataclass
class PurePINNConfig:
    epochs: int = 5000
    lbfgs_steps: int = 200
    hidden_width: int = 128
    hidden_layers: int = 6
    pde_points: int = 5000
    initial_points: int = 500
    boundary_points: int = 500
    learning_rate: float = 1.0e-3
    seed: int = 17
    log_every: int = 100
    resample_every: int = 50
    adaptive_every: int = 25
    adaptive_beta: float = 0.9
    min_weight: float = 0.05
    max_weight: float = 20.0
    dtype: str = "float64"
    device: str = "auto"
    plot_z: int = 180
    plot_t: int = 320


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lbfgs-steps", type=int, default=200)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=6)
    parser.add_argument("--pde-points", type=int, default=5000)
    parser.add_argument("--initial-points", type=int, default=500)
    parser.add_argument("--boundary-points", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--resample-every", type=int, default=50)
    parser.add_argument("--adaptive-every", type=int, default=25)
    parser.add_argument("--adaptive-beta", type=float, default=0.9)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--quick", action="store_true", help="small code-only smoke run")
    parser.add_argument("--force", action="store_true", help="ignore matching pure-PINN cache")


def config_from_args(args: argparse.Namespace) -> PurePINNConfig:
    config = PurePINNConfig(
        epochs=args.epochs,
        lbfgs_steps=args.lbfgs_steps,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        pde_points=args.pde_points,
        initial_points=args.initial_points,
        boundary_points=args.boundary_points,
        learning_rate=args.learning_rate,
        resample_every=args.resample_every,
        adaptive_every=args.adaptive_every,
        adaptive_beta=args.adaptive_beta,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )
    if args.quick:
        config.epochs = min(config.epochs, 5)
        config.lbfgs_steps = 0
        config.hidden_width = min(config.hidden_width, 24)
        config.hidden_layers = min(config.hidden_layers, 2)
        config.pde_points = min(config.pde_points, 80)
        config.initial_points = min(config.initial_points, 30)
        config.boundary_points = min(config.boundary_points, 30)
        config.resample_every = 2
        config.adaptive_every = 1
        config.log_every = 1
        config.plot_z = 40
        config.plot_t = 80
    return config


def select_device(config: PurePINNConfig) -> torch.device:
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


class FourierMLP(nn.Module):
    def __init__(self, output_size: int, width: int, layers: int):
        super().__init__()
        frequencies = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
        self.register_buffer("frequencies", frequencies)
        input_size = 2 + 2 * 2 * len(frequencies)
        modules: list[nn.Module] = [nn.Linear(input_size, width), nn.Tanh()]
        for _ in range(max(layers - 1, 0)):
            modules.extend([nn.Linear(width, width), nn.Tanh()])
        modules.append(nn.Linear(width, output_size))
        self.network = nn.Sequential(*modules)

    def forward(self, normalized_zt: torch.Tensor) -> torch.Tensor:
        angles = math.pi * normalized_zt.unsqueeze(-1) * self.frequencies
        features = torch.cat(
            [
                normalized_zt,
                torch.sin(angles).flatten(1),
                torch.cos(angles).flatten(1),
            ],
            dim=1,
        )
        return self.network(features)


class PureCasePINN(nn.Module):
    def __init__(
        self,
        params: PhysicalParameters,
        config: PurePINNConfig,
        output_size: int,
    ):
        super().__init__()
        self.params = params
        self.mlp = FourierMLP(output_size, config.hidden_width, config.hidden_layers)

    def forward(self, zt: torch.Tensor) -> torch.Tensor:
        normalized = torch.cat(
            [
                2.0 * zt[:, 0:1] / self.params.length_m - 1.0,
                2.0 * zt[:, 1:2] / self.params.t_final_s - 1.0,
            ],
            dim=1,
        )
        return self.mlp(normalized)


def output_scales(params: PhysicalParameters, kind: SystemKind) -> np.ndarray:
    if kind == "two_equation":
        speed = math.sqrt(params.water_bulk_modulus_pa / params.water_density_kg_m3)
    else:
        speed = pressure_wave_speed(params)
    pressure = max(
        initial_pressure_pa(params),
        params.water_density_kg_m3 * speed * abs(params.initial_velocity_m_s),
        1.0,
    )
    velocity = max(abs(params.initial_velocity_m_s), 1.0e-3)
    if kind == "two_equation":
        return np.array([velocity, pressure])
    structure_speed = math.sqrt(params.pipe_E_pa / params.pipe_density_kg_m3)
    axial_velocity = max(
        pressure / (params.pipe_density_kg_m3 * structure_speed), 1.0e-6
    )
    area_f, area_t = cross_section_areas(params)
    stress = pressure * area_f / area_t
    return np.array([velocity, axial_velocity, pressure, stress])


def as_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=torch.get_default_dtype())


def physical_output(
    model: PureCasePINN, zt: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    return model(zt) * scales


def gradient(values: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        values,
        inputs,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True,
    )[0]


def sample_training_points(
    params: PhysicalParameters,
    config: PurePINNConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    pde = np.column_stack(
        [
            rng.uniform(0.0, params.length_m, config.pde_points),
            rng.uniform(0.0, params.t_final_s, config.pde_points),
        ]
    )
    initial = np.column_stack(
        [
            np.linspace(0.0, params.length_m, config.initial_points),
            np.zeros(config.initial_points),
        ]
    )
    first_time = params.t_final_s / max(config.boundary_points, 2)
    boundary_t = np.linspace(first_time, params.t_final_s, config.boundary_points)
    left = np.column_stack([np.zeros_like(boundary_t), boundary_t])
    right = np.column_stack(
        [np.full_like(boundary_t, params.length_m), boundary_t]
    )
    return {
        "pde": as_tensor(pde, device),
        "initial": as_tensor(initial, device),
        "left": as_tensor(left, device),
        "right": as_tensor(right, device),
    }


def pde_component_losses(
    model: PureCasePINN,
    points: torch.Tensor,
    params: PhysicalParameters,
    kind: SystemKind,
    scales: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    zt = points.detach().clone().requires_grad_(True)
    q = physical_output(model, zt, scales)
    derivatives = [
        gradient(q[:, index : index + 1], zt) for index in range(q.shape[1])
    ]
    length = params.length_m
    duration = params.t_final_s
    rho_f = params.water_density_kg_m3
    v_z = derivatives[0][:, 0:1]
    v_t = derivatives[0][:, 1:2]

    if kind == "two_equation":
        p_z = derivatives[1][:, 0:1]
        p_t = derivatives[1][:, 1:2]
        momentum_scale = scales[0] / duration + scales[1] / (rho_f * length)
        continuity_scale = scales[1] / (
            params.water_bulk_modulus_pa * duration
        ) + scales[0] / length
        momentum = (v_t + p_z / rho_f) / momentum_scale
        continuity = (p_t / params.water_bulk_modulus_pa + v_z) / continuity_scale
        return {
            "pde_fluid_momentum": torch.mean(momentum**2),
            "pde_fluid_continuity": torch.mean(continuity**2),
        }

    uz_z = derivatives[1][:, 0:1]
    uz_t = derivatives[1][:, 1:2]
    p_z = derivatives[2][:, 0:1]
    p_t = derivatives[2][:, 1:2]
    sigma_z_derivative = derivatives[3][:, 0:1]
    sigma_t = derivatives[3][:, 1:2]
    coeffs = fssi_coefficients(params)
    a = 1.0 / params.water_bulk_modulus_pa + 2.0 * coeffs["m"]
    c = 1.0 / params.pipe_E_pa - params.pipe_nu * coeffs["k"] / params.pipe_E_pa
    d = params.pipe_nu * coeffs["h"] / params.pipe_E_pa

    residuals = {
        "pde_fluid_momentum": (v_t + p_z / rho_f)
        / (scales[0] / duration + scales[2] / (rho_f * length)),
        "pde_pipe_momentum": (
            params.pipe_density_kg_m3 * uz_t - sigma_z_derivative
        )
        / (
            params.pipe_density_kg_m3 * scales[1] / duration
            + scales[3] / length
        ),
        "pde_fluid_continuity": (v_z + a * p_t + 2.0 * coeffs["n"] * sigma_t)
        / (
            scales[0] / length
            + abs(a) * scales[2] / duration
            + 2.0 * abs(coeffs["n"]) * scales[3] / duration
        ),
        "pde_pipe_constitutive": (uz_z - c * sigma_t + d * p_t)
        / (
            scales[1] / length
            + abs(c) * scales[3] / duration
            + abs(d) * scales[2] / duration
        ),
    }
    return {name: torch.mean(value**2) for name, value in residuals.items()}


def all_loss_components(
    model: PureCasePINN,
    data: Dict[str, torch.Tensor],
    params: PhysicalParameters,
    kind: SystemKind,
    scales: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    losses = pde_component_losses(model, data["pde"], params, kind, scales)

    initial_prediction = model(data["initial"])
    if kind == "fssi":
        initial_target = torch.tensor(
            [params.initial_velocity_m_s, 0.0, initial_pressure_pa(params), 0.0],
            device=scales.device,
        ) / scales
    else:
        initial_target = torch.tensor(
            [params.initial_velocity_m_s, initial_pressure_pa(params)],
            device=scales.device,
        ) / scales
    losses["initial"] = torch.mean((initial_prediction - initial_target) ** 2)

    left = physical_output(model, data["left"], scales)
    right_zt = data["right"].detach().clone().requires_grad_(kind == "fssi")
    right = physical_output(model, right_zt, scales)
    pressure_index = 2 if kind == "fssi" else 1
    boundary = torch.mean(
        ((left[:, pressure_index] - initial_pressure_pa(params)) / scales[pressure_index])
        ** 2
    )
    boundary = boundary + torch.mean((right[:, 0] / scales[0]) ** 2)
    if kind == "fssi":
        boundary = boundary + torch.mean((left[:, 1] / scales[1]) ** 2)
        uz_t = gradient(right[:, 1:2], right_zt)[:, 1:2]
        area_f, area_t = cross_section_areas(params)
        force = (
            params.valve_mass_kg * uz_t
            + area_f * right[:, 2:3]
            - area_t * right[:, 3:4]
        )
        force_scale = max(
            area_f * float(scales[2].detach().cpu()),
            area_t * float(scales[3].detach().cpu()),
            1.0,
        )
        boundary = boundary + torch.mean((force / force_scale) ** 2)
    losses["boundary"] = boundary
    return losses


class GradientNormBalancer:
    """Balance loss terms by equalising their parameter-gradient norms."""

    def __init__(self, names: Iterable[str], config: PurePINNConfig):
        self.names = tuple(names)
        self.beta = config.adaptive_beta
        self.minimum = config.min_weight
        self.maximum = config.max_weight
        self.weights = {name: 1.0 for name in self.names}

    def update(
        self, losses: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, float]:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        norms: Dict[str, float] = {}
        for name in self.names:
            grads = torch.autograd.grad(
                losses[name],
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            squared = torch.zeros((), device=losses[name].device)
            for grad_value in grads:
                if grad_value is not None:
                    squared = squared + torch.sum(grad_value.detach() ** 2)
            norms[name] = max(float(torch.sqrt(squared).cpu()), 1.0e-30)

        target = float(np.mean(list(norms.values())))
        proposed = {
            name: float(np.clip(target / norms[name], self.minimum, self.maximum))
            for name in self.names
        }
        normalizer = len(self.names) / max(sum(proposed.values()), 1.0e-30)
        proposed = {name: value * normalizer for name, value in proposed.items()}
        for name in self.names:
            self.weights[name] = (
                self.beta * self.weights[name] + (1.0 - self.beta) * proposed[name]
            )
        normalizer = len(self.names) / max(sum(self.weights.values()), 1.0e-30)
        self.weights = {
            name: float(
                np.clip(self.weights[name] * normalizer, self.minimum, self.maximum)
            )
            for name in self.names
        }
        return dict(self.weights)

    def total(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.weights[name] * losses[name] for name in self.names)


def pure_case_directory(
    root: Path, params: PhysicalParameters, kind: SystemKind
) -> Path:
    return root / case_identifier(params, kind)


def cache_matches(
    path: Path,
    params: PhysicalParameters,
    config: PurePINNConfig,
    kind: SystemKind,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        return (
            metadata.get("trainer") == "pure_pinn_adaptive_gradient_norm"
            and metadata.get("revision") == PURE_PINN_REVISION
            and metadata.get("system_kind") == kind
            and metadata.get("physical_parameters") == asdict(params)
            and metadata.get("training_config") == asdict(config)
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def load_result(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata"
        }


def predict_grid(
    model: PureCasePINN,
    params: PhysicalParameters,
    config: PurePINNConfig,
    scales: torch.Tensor,
    kind: SystemKind,
) -> Dict[str, np.ndarray]:
    z = np.linspace(0.0, params.length_m, config.plot_z)
    t = np.linspace(0.0, params.t_final_s, config.plot_t)
    zz, tt = np.meshgrid(z, t, indexing="ij")
    points = as_tensor(np.column_stack([zz.ravel(), tt.ravel()]), scales.device)
    batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points), 8192):
            batches.append(
                physical_output(model, points[start : start + 8192], scales)
                .cpu()
                .numpy()
            )
    values = np.vstack(batches)
    names = ("V", "uz", "P", "sigma_z") if kind == "fssi" else ("V", "P")
    result: Dict[str, np.ndarray] = {"z": z, "t": t}
    for index, name in enumerate(names):
        result[name] = values[:, index].reshape(len(z), len(t))
    return result


def save_history(history: list[Dict[str, float]], output_dir: Path) -> None:
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    epochs = [item["epoch"] for item in history]
    loss_names = sorted(
        key.removeprefix("loss_")
        for key in history[0]
        if key.startswith("loss_") and key != "loss_total"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].semilogy(
        epochs, [item["loss_total"] for item in history], label="total", linewidth=2
    )
    for name in loss_names:
        axes[0].semilogy(
            epochs, [item[f"loss_{name}"] for item in history], label=name
        )
        axes[1].plot(
            epochs, [item[f"weight_{name}"] for item in history], label=name
        )
    axes[0].set(xlabel="epoch", ylabel="loss", title="Pure PINN losses")
    axes[1].set(
        xlabel="epoch",
        ylabel="adaptive weight",
        title="Gradient-norm adaptive weights",
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    fig.savefig(output_dir / "training_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_case_summary(
    result: Dict[str, np.ndarray],
    params: PhysicalParameters,
    kind: SystemKind,
    output_path: Path,
) -> None:
    t = result["t"]
    middle = len(result["z"]) // 2
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axes[0, 0].plot(t, result["P"][-1], label="valve")
    axes[0, 0].plot(t, result["P"][middle], label="middle", alpha=0.75)
    axes[0, 0].set(title="Pressure", xlabel="time (s)", ylabel="P (Pa)")
    axes[0, 1].plot(t, result["V"][-1], label="valve")
    axes[0, 1].plot(t, result["V"][middle], label="middle", alpha=0.75)
    axes[0, 1].set(title="Fluid velocity", xlabel="time (s)", ylabel="V (m/s)")
    pressure_image = axes[0, 2].pcolormesh(
        t, result["z"], result["P"], shading="auto"
    )
    axes[0, 2].set(title="Pressure field", xlabel="time (s)", ylabel="z (m)")
    fig.colorbar(pressure_image, ax=axes[0, 2], label="P (Pa)")

    if kind == "fssi":
        axes[1, 0].plot(t, result["uz"][-1], label="u_z at valve")
        axes[1, 0].plot(
            t,
            result["sigma_z"][-1] / 1.0e6,
            label="sigma_z/1e6 at valve",
        )
        axes[1, 0].set(title="Structural response", xlabel="time (s)")
        axial_f, axial_a = spectrum(t, result["uz"][middle])
        radial = radial_velocity(
            params, t, result["P"][middle], result["sigma_z"][middle]
        )
        radial_f, radial_a = spectrum(t, radial)
        axes[1, 1].plot(axial_f, axial_a)
        axes[1, 1].set(
            xlim=(0, 100),
            title="Axial spectrum",
            xlabel="frequency (Hz)",
            ylabel="velocity (m/s)",
        )
        axes[1, 2].plot(radial_f, radial_a)
        axes[1, 2].set(
            xlim=(0, 100),
            title="Radial spectrum",
            xlabel="frequency (Hz)",
            ylabel="velocity (m/s)",
        )
    else:
        axes[1, 0].plot(t, result["P"][-1], label="valve pressure")
        axes[1, 0].set(title="Valve response", xlabel="time (s)")
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")

    for axis in axes.ravel():
        axis.grid(True, alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if labels:
            axis.legend()
    fig.suptitle(
        f"Pure PINN condition: {kind}, Es/E={params.soil_E_pa / params.pipe_E_pa:g}, "
        f"Mv={params.valve_mass_kg:g} kg, nu={params.pipe_nu:g}"
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_pure_case(
    name: str,
    params: PhysicalParameters,
    kind: SystemKind,
    config: PurePINNConfig,
    case_root: Path,
    force: bool,
) -> Dict[str, np.ndarray]:
    output_dir = pure_case_directory(case_root, params, kind)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_key = str(output_dir.resolve())
    if session_key in _SESSION_RESULTS:
        print(f"[{name}] reusing this pure-PINN condition from the current run")
        return _SESSION_RESULTS[session_key]
    result_path = output_dir / "results.npz"
    if result_path.exists() and not force and cache_matches(
        result_path, params, config, kind
    ):
        print(f"[{name}] using cached pure-PINN result: {result_path}")
        result = load_result(result_path)
        if not (output_dir / "case_summary.png").exists():
            save_case_summary(
                result, params, kind, output_dir / "case_summary.png"
            )
        _SESSION_RESULTS[session_key] = result
        return result
    if result_path.exists() and not force:
        print(f"[{name}] pure-PINN settings differ; retraining")

    device = select_device(config)
    scale_values = output_scales(params, kind)
    scales = as_tensor(scale_values, device)
    model = PureCasePINN(params, config, len(scale_values)).to(device)
    rng = np.random.default_rng(config.seed + 1009)
    data = sample_training_points(params, config, device, rng)
    initial_losses = all_loss_components(model, data, params, kind, scales)
    balancer = GradientNormBalancer(initial_losses.keys(), config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[Dict[str, float]] = []

    print(
        f"[{name}] training PURE {kind} PINN on {device}; "
        f"epochs={config.epochs}, anchors=0"
    )
    for epoch in range(1, config.epochs + 1):
        if epoch > 1 and epoch % max(config.resample_every, 1) == 0:
            data["pde"] = sample_training_points(
                params, config, device, rng
            )["pde"]
        optimizer.zero_grad(set_to_none=True)
        losses = all_loss_components(model, data, params, kind, scales)
        if epoch == 1 or epoch % max(config.adaptive_every, 1) == 0:
            balancer.update(losses, model)
        total = balancer.total(losses)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        item: Dict[str, float] = {
            "epoch": float(epoch),
            "loss_total": float(total.detach().cpu()),
        }
        for loss_name, loss_value in losses.items():
            item[f"loss_{loss_name}"] = float(loss_value.detach().cpu())
            item[f"weight_{loss_name}"] = balancer.weights[loss_name]
        history.append(item)

        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            weight_text = ", ".join(
                f"{loss_name}={balancer.weights[loss_name]:.2f}"
                for loss_name in balancer.names
            )
            print(
                f"[{name}] epoch {epoch:5d} total={item['loss_total']:.3e}; "
                f"weights: {weight_text}"
            )

    if config.lbfgs_steps > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=config.lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad(set_to_none=True)
            closure_losses = all_loss_components(
                model, data, params, kind, scales
            )
            value = balancer.total(closure_losses)
            value.backward()
            return value

        print(
            f"[{name}] L-BFGS steps={config.lbfgs_steps}; "
            "adaptive weights fixed at final Adam values"
        )
        optimizer_lbfgs.step(closure)

    result = predict_grid(model, params, config, scales, kind)
    metadata = {
        "trainer": "pure_pinn_adaptive_gradient_norm",
        "revision": PURE_PINN_REVISION,
        "case_name": name,
        "system_kind": kind,
        "physical_parameters": asdict(params),
        "training_config": asdict(config),
        "output_scales": scale_values.tolist(),
        "final_adaptive_weights": balancer.weights,
        "uses_anchor_points": False,
        "uses_labelled_data": False,
    }
    np.savez_compressed(
        result_path, **result, metadata=json.dumps(metadata, ensure_ascii=True)
    )
    torch.save(
        {"state_dict": model.state_dict(), "metadata": metadata},
        output_dir / "model.pt",
    )
    save_history(history, output_dir)
    save_case_summary(result, params, kind, output_dir / "case_summary.png")
    _SESSION_RESULTS[session_key] = result
    return result


def output_directory(root: Path, figure: int) -> Path:
    path = root / f"figure{figure:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def finish_comparison(
    figure: int, output_dir: Path, result_path: Path, label: str
) -> None:
    save_comparison(
        figure, result_path, output_dir / "comparison.png", label
    )
    print(f"[Figure {figure}] computed: {result_path}")
    print(f"[Figure {figure}] comparison: {output_dir / 'comparison.png'}")


def run_figure5(output_root: Path, quick: bool) -> None:
    output_dir = output_directory(output_root, 5)
    steps = np.array(
        [2e-3, 1e-3, 5e-4]
        if quick
        else [2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4]
    )
    params = PhysicalParameters()
    traces = []
    for dt in steps:
        solution = solve_fssi(
            params,
            SolverSettings(
                n_cells=120 if quick else 500,
                dt_s=float(dt),
                output_stride=1,
            ),
        )
        traces.append((solution["t"], solution["P"][:, -1]))
    reference_t, reference_p = traces[-1]
    errors = []
    for time, pressure in traces:
        comparison = np.interp(reference_t, time, pressure)
        errors.append(
            np.linalg.norm(comparison - reference_p)
            / max(np.linalg.norm(reference_p), 1.0)
        )
    result_path = output_dir / "computed.png"
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.loglog(steps, np.maximum(errors, 1.0e-14), "s-", color="black")
    axis.set(
        xlabel="time step, dt (s)",
        ylabel="relative L2 error to finest step",
        title="Figure 5 deterministic convergence (not a PINN case)",
    )
    axis.grid(True, which="both", alpha=0.25)
    fig.savefig(result_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    np.savez(
        output_dir / "convergence_data.npz",
        dt=steps,
        relative_l2=np.asarray(errors),
    )
    finish_comparison(
        5, output_dir, result_path, "Deterministic convergence metric"
    )


def run_figure6(
    config: PurePINNConfig, case_root: Path, output_root: Path, force: bool
) -> None:
    output_dir = output_directory(output_root, 6)
    params = with_soil_ratio(PhysicalParameters(), 1.0)
    fssi = train_pure_case(
        "figure06_fssi_EsE_1",
        params,
        "fssi",
        config,
        case_root,
        force,
    )
    classic = train_pure_case(
        "figure06_two_equation",
        params,
        "two_equation",
        config,
        case_root,
        force,
    )
    result_path = output_dir / "computed.png"
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(fssi["t"], fssi["P"][-1], label="pure four-equation PINN")
    axis.plot(
        classic["t"],
        classic["P"][-1],
        "--",
        label="pure two-equation PINN",
    )
    axis.set(
        xlabel="time (s)",
        ylabel="valve pressure (Pa)",
        title="Figure 6: pure PINN comparison",
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(result_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    finish_comparison(6, output_dir, result_path, "Pure PINNs, no anchors")


def run_figure7(output_root: Path) -> None:
    output_dir = output_directory(output_root, 7)
    eta = np.linspace(0.0, 2.5, 400)
    ratio = 1.0 - np.exp(-2.0 * eta**0.75)
    result_path = output_dir / "computed.png"
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(eta, ratio, color="black")
    axis.set(
        xlabel="eta = sigma/sigma0",
        ylabel="Pb/p_inner",
        title="Figure 7 algebraic curve (not a PINN case)",
    )
    axis.grid(True, alpha=0.25)
    fig.savefig(result_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    np.savez(output_dir / "curve_data.npz", eta=eta, ratio=ratio)
    finish_comparison(7, output_dir, result_path, "Algebraic curve; no network")


def run_figure8(output_root: Path) -> None:
    output_dir = output_directory(output_root, 8)
    ratios = np.logspace(-6, 1, 500)
    params = PhysicalParameters(coefficient_model="paper_reported")
    values = np.array(
        [
            pipe_soil_lambdas(with_soil_ratio(params, value))[0]
            for value in ratios
        ]
    )
    result_path = output_dir / "computed.png"
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(np.log10(ratios), values, color="black")
    axis.axhline(0.0, color="red", linewidth=0.8)
    axis.axhline(1.0, color="red", linewidth=0.8)
    axis.set(
        xlabel="log10(Es/E)",
        ylabel="Pb/p_inner",
        title="Figure 8 reported curve (not a PINN case)",
    )
    axis.grid(True, alpha=0.25)
    fig.savefig(result_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    np.savez(output_dir / "curve_data.npz", soil_ratio=ratios, ratio=values)
    finish_comparison(8, output_dir, result_path, "Reported curve; no network")


def run_figure9(
    config: PurePINNConfig, case_root: Path, output_root: Path, force: bool
) -> None:
    output_dir = output_directory(output_root, 9)
    exposed_params = with_soil_ratio(PhysicalParameters(), 0.0)
    restrained_params = with_soil_ratio(PhysicalParameters(), 0.1)
    exposed = train_pure_case(
        "figure09_exposed",
        exposed_params,
        "fssi",
        config,
        case_root,
        force,
    )
    restrained = train_pure_case(
        "figure09_with_PSC",
        restrained_params,
        "fssi",
        config,
        case_root,
        force,
    )
    result_path = output_dir / "computed.png"
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(
        exposed["t"],
        exposed["P"][-1],
        color="black",
        label="Es/E=0",
    )
    axis.plot(
        restrained["t"],
        restrained["P"][-1],
        color="red",
        label="Es/E=0.1",
    )
    axis.set(
        xlabel="time (s)",
        ylabel="valve pressure (Pa)",
        title="Figure 9: pure FSSI PINNs",
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(result_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    finish_comparison(9, output_dir, result_path, "Pure PINNs, no anchors")


def run_figure10(
    config: PurePINNConfig, case_root: Path, output_root: Path, force: bool
) -> None:
    output_dir = output_directory(output_root, 10)
    base = PhysicalParameters()
    mass_params = {
        12.0: replace(base, valve_mass_kg=12.0),
        120.0: replace(base, valve_mass_kg=120.0),
    }
    mass_cases = {
        value: train_pure_case(
            f"figure10_mass_{value:g}",
            params,
            "fssi",
            config,
            case_root,
            force,
        )
        for value, params in mass_params.items()
    }
    poisson_params = {
        0.0: replace(base, pipe_nu=0.0),
        0.3: replace(base, pipe_nu=0.3),
    }
    poisson_cases = {
        value: train_pure_case(
            f"figure10_nu_{value:g}",
            params,
            "fssi",
            config,
            case_root,
            force,
        )
        for value, params in poisson_params.items()
    }
    soil_params = {
        1.0e-3: with_soil_ratio(base, 1.0e-3),
        1.0e-1: with_soil_ratio(base, 1.0e-1),
    }
    soil_cases = {
        value: train_pure_case(
            f"figure10_EsE_{value:g}",
            params,
            "fssi",
            config,
            case_root,
            force,
        )
        for value, params in soil_params.items()
    }

    result_path = output_dir / "computed.png"
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 10.5), constrained_layout=True)
    for mass, result in mass_cases.items():
        axes[0].plot(result["t"], result["P"][-1], label=f"Mv={mass:g} kg")
    for nu, result in poisson_cases.items():
        axes[1].plot(result["t"], result["P"][-1], label=f"nu={nu:g}")
    for ratio, result in soil_cases.items():
        axes[2].plot(result["t"], result["P"][-1], label=f"Es/E={ratio:g}")
    titles = (
        "(a) junction coupling",
        "(b) Poisson coupling",
        "(c) pipe-soil coupling",
    )
    for axis, title in zip(axes, titles):
        axis.set(
            xlabel="time (s)",
            ylabel="valve pressure (Pa)",
            title=title,
        )
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.suptitle("Figure 10: six curves from five unique pure PINNs")
    fig.savefig(result_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    finish_comparison(
        10,
        output_dir,
        result_path,
        "Six curves from five pure PINN cases",
    )


def run_figure11(output_root: Path) -> None:
    output_dir = output_directory(output_root, 11)
    ratios = np.logspace(-4, 1, 600)
    params = PhysicalParameters(coefficient_model="paper_reported")
    coefficients = [
        fssi_coefficients(with_soil_ratio(params, value)) for value in ratios
    ]
    m = np.array([item["m"] for item in coefficients])
    n = np.array([item["n"] for item in coefficients])
    h = np.array([item["h"] for item in coefficients])
    k = np.array([item["k"] for item in coefficients])
    result_path = output_dir / "computed.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    axes[0].semilogx(ratios, 2.0 * m, label="2m")
    axes[0].semilogx(ratios, 2.0 * n, label="2n")
    axes[1].semilogx(ratios, k, label="k")
    axes[1].semilogx(ratios, h, label="h")
    for axis in axes:
        axis.set_xlabel("Es/E")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("coefficient (1/Pa)")
    axes[1].set_ylabel("dimensionless coefficient")
    fig.suptitle("Figure 11 coefficient sweep (not a PINN case)")
    fig.savefig(result_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    np.savez(
        output_dir / "coefficient_data.npz",
        soil_ratio=ratios,
        m=m,
        n=n,
        h=h,
        k=k,
    )
    finish_comparison(
        11, output_dir, result_path, "Coefficient sweep; no network"
    )


def run_figure12(output_root: Path) -> None:
    output_dir = output_directory(output_root, 12)
    ratios = np.logspace(-5, 1, 600)
    params = PhysicalParameters(coefficient_model="paper_reported")
    speeds = np.array(
        [pressure_wave_speed(with_soil_ratio(params, value)) for value in ratios]
    )
    result_path = output_dir / "computed.png"
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.semilogx(ratios, speeds, color="black")
    axis.set(
        xlabel="Es/E",
        ylabel="pressure wave speed (m/s)",
        title="Figure 12 eigenvalue sweep (not a PINN case)",
    )
    axis.grid(True, which="both", alpha=0.25)
    fig.savefig(result_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    np.savez(
        output_dir / "wave_speed_data.npz",
        soil_ratio=ratios,
        pressure_wave_speed=speeds,
    )
    finish_comparison(
        12, output_dir, result_path, "Eigenvalue sweep; no network"
    )


def soil_ratio_cases(
    config: PurePINNConfig,
    case_root: Path,
    force: bool,
) -> Dict[float, tuple[PhysicalParameters, Dict[str, np.ndarray]]]:
    cases = {}
    for ratio in (1.0e-3, 1.0e-2, 1.0e-1):
        params = with_soil_ratio(PhysicalParameters(), ratio)
        cases[ratio] = (
            params,
            train_pure_case(
                f"soil_ratio_{ratio:g}",
                params,
                "fssi",
                config,
                case_root,
                force,
            ),
        )
    return cases


def run_figure13(
    config: PurePINNConfig, case_root: Path, output_root: Path, force: bool
) -> None:
    output_dir = output_directory(output_root, 13)
    cases = soil_ratio_cases(config, case_root, force)
    result_path = output_dir / "computed.png"
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for ratio, (_, result) in cases.items():
        axis.plot(result["t"], result["P"][-1], label=f"Es/E={ratio:g}")
    axis.set(
        xlabel="time (s)",
        ylabel="valve pressure (Pa)",
        title="Figure 13: pure PINN predictions",
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(result_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    finish_comparison(13, output_dir, result_path, "Pure PINNs, no anchors")


def run_figure14(
    config: PurePINNConfig, case_root: Path, output_root: Path, force: bool
) -> None:
    output_dir = output_directory(output_root, 14)
    cases = soil_ratio_cases(config, case_root, force)
    result_path = output_dir / "computed.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ratio, (params, result) in cases.items():
        middle = len(result["z"]) // 2
        axial_f, axial_a = spectrum(result["t"], result["uz"][middle])
        radial = radial_velocity(
            params,
            result["t"],
            result["P"][middle],
            result["sigma_z"][middle],
        )
        radial_f, radial_a = spectrum(result["t"], radial)
        axes[0].plot(axial_f, axial_a, label=f"Es/E={ratio:g}")
        axes[1].plot(radial_f, radial_a, label=f"Es/E={ratio:g}")
    for axis, title in zip(
        axes, ("(a) axial vibration", "(b) radial vibration")
    ):
        axis.set(
            xlim=(0, 100),
            xlabel="frequency (Hz)",
            ylabel="vibration velocity (m/s)",
            title=title,
        )
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.suptitle("Figure 14: spectra from pure PINNs")
    fig.savefig(result_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    finish_comparison(14, output_dir, result_path, "Pure PINN spectra")


def parse_figures(value: str) -> tuple[int, ...]:
    normalized = value.strip().lower()
    if normalized == "all":
        return SUPPORTED_FIGURES
    if normalized == "dynamic":
        return DYNAMIC_FIGURES
    figures = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    unsupported = [figure for figure in figures if figure not in SUPPORTED_FIGURES]
    if unsupported:
        if 3 in unsupported:
            raise ValueError(
                "Figure 3 is not executable because the impact-test inputs are unpublished"
            )
        raise ValueError(f"Unsupported figures: {unsupported}")
    return figures


def main() -> None:
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce paper figures with pure PINNs for dynamic cases. "
            "Training uses PDE, IC, and BC losses only; anchor points are forbidden."
        )
    )
    parser.add_argument(
        "--figures",
        default="dynamic",
        help="dynamic, all, or comma-separated values such as 6,9,13,14",
    )
    parser.add_argument(
        "--case-root",
        default=str(package_root / "pure_pinn_cases"),
        help="shared pure-PINN model directory",
    )
    parser.add_argument(
        "--output-root",
        default=str(package_root / "pure_pinn_outputs"),
        help="per-figure pure-PINN comparison directory",
    )
    add_training_arguments(parser)
    args = parser.parse_args()

    figures = parse_figures(args.figures)
    config = config_from_args(args)
    case_root = Path(args.case_root)
    output_root = Path(args.output_root)
    case_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Pure PINN mode: labelled points=0, reference anchors=0")
    print(
        "Adaptive weighting: inverse gradient-norm balancing for every "
        f"{config.adaptive_every} Adam epochs"
    )
    runners = {
        5: lambda: run_figure5(output_root, args.quick),
        6: lambda: run_figure6(
            config, case_root, output_root, args.force
        ),
        7: lambda: run_figure7(output_root),
        8: lambda: run_figure8(output_root),
        9: lambda: run_figure9(
            config, case_root, output_root, args.force
        ),
        10: lambda: run_figure10(
            config, case_root, output_root, args.force
        ),
        11: lambda: run_figure11(output_root),
        12: lambda: run_figure12(output_root),
        13: lambda: run_figure13(
            config, case_root, output_root, args.force
        ),
        14: lambda: run_figure14(
            config, case_root, output_root, args.force
        ),
    }
    for figure in figures:
        print(f"\n=== Figure {figure} ===")
        runners[figure]()


if __name__ == "__main__":
    main()
