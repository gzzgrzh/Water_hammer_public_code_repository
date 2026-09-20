"""Reusable PyTorch PINN trainer for one explicitly defined paper case."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required: python -m pip install torch") from exc

from .physics import (
    PhysicalParameters,
    SolverSettings,
    cross_section_areas,
    fssi_coefficients,
    initial_pressure_pa,
    pressure_wave_speed,
    radial_velocity,
    solve_fssi,
    solve_two_equation,
    spectrum,
)


SystemKind = Literal["fssi", "two_equation"]
MODEL_REVISIONS: Dict[SystemKind, int] = {"fssi": 1, "two_equation": 2}


@dataclass
class PINNConfig:
    epochs: int = 4000
    lbfgs_steps: int = 100
    hidden_width: int = 128
    hidden_layers: int = 6
    pde_points: int = 5000
    initial_points: int = 500
    boundary_points: int = 500
    anchor_points: int = 1600
    learning_rate: float = 1.0e-3
    w_pde: float = 1.0
    w_initial: float = 20.0
    w_boundary: float = 20.0
    w_anchor: float = 5.0
    seed: int = 7
    log_every: int = 100
    dtype: str = "float64"
    device: str = "auto"
    plot_z: int = 180
    plot_t: int = 320


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick", action="store_true", help="small smoke-training run")
    parser.add_argument("--force", action="store_true", help="retrain even if results exist")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lbfgs-steps", type=int, default=100)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=6)
    parser.add_argument("--pde-points", type=int, default=5000)
    parser.add_argument("--initial-points", type=int, default=500)
    parser.add_argument("--boundary-points", type=int, default=500)
    parser.add_argument("--anchor-points", type=int, default=1600)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")


def config_from_args(args: argparse.Namespace) -> PINNConfig:
    config = PINNConfig(
        epochs=args.epochs,
        lbfgs_steps=args.lbfgs_steps,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        pde_points=args.pde_points,
        initial_points=args.initial_points,
        boundary_points=args.boundary_points,
        anchor_points=args.anchor_points,
        learning_rate=args.learning_rate,
        device=args.device,
        dtype=args.dtype,
    )
    if args.quick:
        config.epochs = min(config.epochs, 20)
        config.lbfgs_steps = 0
        config.hidden_width = min(config.hidden_width, 32)
        config.hidden_layers = min(config.hidden_layers, 3)
        config.pde_points = min(config.pde_points, 160)
        config.initial_points = min(config.initial_points, 50)
        config.boundary_points = min(config.boundary_points, 50)
        config.anchor_points = min(config.anchor_points, 100)
        config.plot_z = 50
        config.plot_t = 100
        config.log_every = 5
    return config


def select_device(config: PINNConfig) -> torch.device:
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
        frequencies = torch.tensor([1.0, 2.0, 4.0, 8.0])
        self.register_buffer("frequencies", frequencies)
        input_size = 2 + 2 * 2 * len(frequencies)
        modules: List[nn.Module] = [nn.Linear(input_size, width), nn.Tanh()]
        for _ in range(max(layers - 1, 0)):
            modules.extend([nn.Linear(width, width), nn.Tanh()])
        modules.append(nn.Linear(width, output_size))
        self.network = nn.Sequential(*modules)

    def forward(self, normalized_zt: torch.Tensor) -> torch.Tensor:
        angles = math.pi * normalized_zt.unsqueeze(-1) * self.frequencies
        features = torch.cat(
            [normalized_zt, torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)], dim=1
        )
        return self.network(features)


class CasePINN(nn.Module):
    def __init__(self, params: PhysicalParameters, config: PINNConfig, output_size: int):
        super().__init__()
        self.params = params
        self.mlp = FourierMLP(output_size, config.hidden_width, config.hidden_layers)

    def forward(self, zt: torch.Tensor) -> torch.Tensor:
        normalized = torch.cat(
            [2.0 * zt[:, 0:1] / self.params.length_m - 1.0,
             2.0 * zt[:, 1:2] / self.params.t_final_s - 1.0], dim=1
        )
        return self.mlp(normalized)


def output_scales(params: PhysicalParameters, kind: SystemKind) -> np.ndarray:
    speed = pressure_wave_speed(params) if kind == "fssi" else math.sqrt(
        params.water_bulk_modulus_pa / params.water_density_kg_m3
    )
    p_scale = max(initial_pressure_pa(params), params.water_density_kg_m3 * speed * abs(params.initial_velocity_m_s))
    if kind == "two_equation":
        return np.array([max(abs(params.initial_velocity_m_s), 1.0e-3), p_scale])
    structure_speed = math.sqrt(params.pipe_E_pa / params.pipe_density_kg_m3)
    uz_scale = max(p_scale / (params.pipe_density_kg_m3 * structure_speed), 1.0e-6)
    area_f, area_t = cross_section_areas(params)
    return np.array([max(abs(params.initial_velocity_m_s), 1.0e-3), uz_scale, p_scale, p_scale * area_f / area_t])


def tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, device=device)


def physical_output(model: CasePINN, zt: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return model(zt) * scales


def gradient(values: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        values, inputs, grad_outputs=torch.ones_like(values), create_graph=True, retain_graph=True
    )[0]


def interpolate_reference(
    baseline: Dict[str, np.ndarray], z: np.ndarray, t: np.ndarray, names: Tuple[str, ...]
) -> np.ndarray:
    zi = np.clip(np.searchsorted(baseline["z"], z), 0, len(baseline["z"]) - 1)
    ti = np.clip(np.searchsorted(baseline["t"], t), 0, len(baseline["t"]) - 1)
    return np.column_stack([baseline[name][ti, zi] for name in names])


def make_training_data(
    params: PhysicalParameters,
    config: PINNConfig,
    kind: SystemKind,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(config.seed + 101)
    pde = np.column_stack(
        [rng.uniform(0.0, params.length_m, config.pde_points),
         rng.uniform(0.0, params.t_final_s, config.pde_points)]
    )
    initial = np.column_stack(
        [np.linspace(0.0, params.length_m, config.initial_points), np.zeros(config.initial_points)]
    )
    first_time = params.t_final_s / max(config.boundary_points, 2)
    boundary_time = np.linspace(first_time, params.t_final_s, config.boundary_points)
    left = np.column_stack([np.zeros_like(boundary_time), boundary_time])
    right = np.column_stack([np.full_like(boundary_time, params.length_m), boundary_time])

    reference_settings = SolverSettings(
        n_cells=240 if config.epochs > 20 else 60,
        dt_s=5.0e-4 if config.epochs > 20 else 2.0e-3,
        output_stride=2,
        use_valve_mass_boundary=True,
    )
    baseline = solve_fssi(params, reference_settings) if kind == "fssi" else solve_two_equation(params, reference_settings)
    anchor_z = rng.uniform(0.0, params.length_m, config.anchor_points)
    anchor_t = rng.uniform(0.0, params.t_final_s, config.anchor_points)
    anchor_zt = np.column_stack([anchor_z, anchor_t])
    names = ("V", "uz", "P", "sigma_z") if kind == "fssi" else ("V", "P")
    anchor_target = interpolate_reference(baseline, anchor_z, anchor_t, names)
    return {
        "pde": tensor(pde, device),
        "initial": tensor(initial, device),
        "left": tensor(left, device),
        "right": tensor(right, device),
        "anchor": tensor(anchor_zt, device),
        "anchor_target": tensor(anchor_target, device),
    }, baseline


def normalized_pde_loss(
    model: CasePINN,
    zt_values: torch.Tensor,
    params: PhysicalParameters,
    kind: SystemKind,
    scales: torch.Tensor,
) -> torch.Tensor:
    zt = zt_values.detach().clone().requires_grad_(True)
    q = physical_output(model, zt, scales)
    derivatives = [gradient(q[:, i : i + 1], zt) for i in range(q.shape[1])]
    L = params.length_m
    T = params.t_final_s
    rho = params.water_density_kg_m3
    V = q[:, 0:1]
    V_z, V_t = derivatives[0][:, 0:1], derivatives[0][:, 1:2]

    if kind == "two_equation":
        P = q[:, 1:2]
        P_z, P_t = derivatives[1][:, 0:1], derivatives[1][:, 1:2]
        beta = 1.0 / params.water_bulk_modulus_pa
        r1_scale = scales[0] / T + scales[1] / (rho * L)
        r2_scale = beta * scales[1] / T + scales[0] / L
        r1 = (V_t + P_z / rho) / r1_scale
        r2 = (beta * P_t + V_z) / r2_scale
        return torch.mean(r1**2) + torch.mean(r2**2)

    uz, P, sigma = q[:, 1:2], q[:, 2:3], q[:, 3:4]
    uz_z, uz_t = derivatives[1][:, 0:1], derivatives[1][:, 1:2]
    P_z, P_t = derivatives[2][:, 0:1], derivatives[2][:, 1:2]
    sigma_z, sigma_t = derivatives[3][:, 0:1], derivatives[3][:, 1:2]
    coeffs = fssi_coefficients(params)
    a = 1.0 / params.water_bulk_modulus_pa + 2.0 * coeffs["m"]
    c = 1.0 / params.pipe_E_pa - params.pipe_nu * coeffs["k"] / params.pipe_E_pa
    d = params.pipe_nu * coeffs["h"] / params.pipe_E_pa
    residuals = [
        (V_t + P_z / rho) / (scales[0] / T + scales[2] / (rho * L)),
        (params.pipe_density_kg_m3 * uz_t - sigma_z)
        / (params.pipe_density_kg_m3 * scales[1] / T + scales[3] / L),
        (V_z + a * P_t + 2.0 * coeffs["n"] * sigma_t)
        / (scales[0] / L + abs(a) * scales[2] / T + 2.0 * abs(coeffs["n"]) * scales[3] / T),
        (uz_z - c * sigma_t + d * P_t)
        / (scales[1] / L + abs(c) * scales[3] / T + abs(d) * scales[2] / T),
    ]
    return sum(torch.mean(value**2) for value in residuals)


def loss_terms(
    model: CasePINN,
    data: Dict[str, torch.Tensor],
    params: PhysicalParameters,
    config: PINNConfig,
    kind: SystemKind,
    scales: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pde_loss = normalized_pde_loss(model, data["pde"], params, kind, scales)
    initial_prediction = model(data["initial"])
    if kind == "fssi":
        initial_target = torch.tensor(
            [params.initial_velocity_m_s, 0.0, initial_pressure_pa(params), 0.0],
            device=scales.device,
        ) / scales
    else:
        initial_target = torch.tensor(
            [params.initial_velocity_m_s, initial_pressure_pa(params)], device=scales.device
        ) / scales
    initial_loss = torch.mean((initial_prediction - initial_target) ** 2)

    left_prediction = physical_output(model, data["left"], scales)
    right_zt = data["right"].detach().clone().requires_grad_(True)
    right_prediction = physical_output(model, right_zt, scales)
    left_pressure_index = 2 if kind == "fssi" else 1
    boundary_loss = torch.mean(
        ((left_prediction[:, left_pressure_index] - initial_pressure_pa(params)) / scales[left_pressure_index]) ** 2
    ) + torch.mean((right_prediction[:, 0] / scales[0]) ** 2)
    if kind == "fssi":
        boundary_loss = boundary_loss + torch.mean((left_prediction[:, 1] / scales[1]) ** 2)
        uz_t = gradient(right_prediction[:, 1:2], right_zt)[:, 1:2]
        area_f, area_t = cross_section_areas(params)
        force = params.valve_mass_kg * uz_t + area_f * right_prediction[:, 2:3] - area_t * right_prediction[:, 3:4]
        pressure_scale = float(scales[2].detach().cpu())
        stress_scale = float(scales[3].detach().cpu())
        force_scale = max(area_f * pressure_scale, area_t * stress_scale, 1.0)
        boundary_loss = boundary_loss + torch.mean((force / force_scale) ** 2)

    anchor_prediction = model(data["anchor"])
    anchor_target = data["anchor_target"] / scales
    anchor_loss = torch.mean((anchor_prediction - anchor_target) ** 2)
    total = (
        config.w_pde * pde_loss
        + config.w_initial * initial_loss
        + config.w_boundary * boundary_loss
        + config.w_anchor * anchor_loss
    )
    parts = {
        "total": float(total.detach().cpu()),
        "pde": float(pde_loss.detach().cpu()),
        "initial": float(initial_loss.detach().cpu()),
        "boundary": float(boundary_loss.detach().cpu()),
        "anchor": float(anchor_loss.detach().cpu()),
    }
    return total, parts


def predict_grid(
    model: CasePINN,
    params: PhysicalParameters,
    config: PINNConfig,
    scales: torch.Tensor,
    kind: SystemKind,
) -> Dict[str, np.ndarray]:
    z = np.linspace(0.0, params.length_m, config.plot_z)
    t = np.linspace(0.0, params.t_final_s, config.plot_t)
    zz, tt = np.meshgrid(z, t, indexing="ij")
    inputs = torch.as_tensor(np.column_stack([zz.ravel(), tt.ravel()]), device=scales.device)
    batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), 8192):
            batches.append(physical_output(model, inputs[start : start + 8192], scales).cpu().numpy())
    values = np.vstack(batches)
    names = ("V", "uz", "P", "sigma_z") if kind == "fssi" else ("V", "P")
    result = {"z": z, "t": t}
    for index, name in enumerate(names):
        result[name] = values[:, index].reshape(len(z), len(t))
    return result


def load_result(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files if name != "metadata"}


def cache_matches(
    path: Path, params: PhysicalParameters, config: PINNConfig, kind: SystemKind
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        return (
            metadata.get("system_kind") == kind
            and metadata.get("model_revision", 1) == MODEL_REVISIONS[kind]
            and metadata.get("physical_parameters") == asdict(params)
            and metadata.get("training_config") == asdict(config)
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def save_case_summary(
    result: Dict[str, np.ndarray], params: PhysicalParameters, kind: SystemKind, output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    t = result["t"]
    middle = len(result["z"]) // 2
    axes[0, 0].plot(t, result["P"][-1], label="valve")
    axes[0, 0].plot(t, result["P"][middle], label="middle", alpha=0.75)
    axes[0, 0].set_title("Pressure")
    axes[0, 0].set_ylabel("P (Pa)")
    axes[0, 1].plot(t, result["V"][-1], label="valve")
    axes[0, 1].plot(t, result["V"][middle], label="middle", alpha=0.75)
    axes[0, 1].set_title("Fluid velocity")
    axes[0, 1].set_ylabel("V (m/s)")
    pressure_image = axes[0, 2].pcolormesh(t, result["z"], result["P"], shading="auto")
    axes[0, 2].set_title("Pressure field")
    axes[0, 2].set_ylabel("z (m)")
    fig.colorbar(pressure_image, ax=axes[0, 2], label="P (Pa)")
    if kind == "fssi":
        axes[1, 0].plot(t, result["uz"][-1], label="uz at valve")
        axes[1, 0].plot(t, result["sigma_z"][-1] / 1.0e6, label="sigma_z/1e6 at valve")
        axes[1, 0].set_title("Structural response")
        axes[1, 0].set_xlabel("time (s)")
        f_axial, a_axial = spectrum(t, result["uz"][middle])
        radial = radial_velocity(params, t, result["P"][middle], result["sigma_z"][middle])
        f_radial, a_radial = spectrum(t, radial)
        axes[1, 1].plot(f_axial, a_axial)
        axes[1, 1].set_xlim(0, 100)
        axes[1, 1].set_title("Axial spectrum")
        axes[1, 1].set_xlabel("frequency (Hz)")
        axes[1, 1].set_ylabel("velocity (m/s)")
        axes[1, 2].plot(f_radial, a_radial)
        axes[1, 2].set_xlim(0, 100)
        axes[1, 2].set_title("Radial spectrum")
        axes[1, 2].set_xlabel("frequency (Hz)")
        axes[1, 2].set_ylabel("velocity (m/s)")
    else:
        axes[1, 0].plot(t, result["P"][-1], label="valve pressure")
        axes[1, 0].set_title("Valve response")
        axes[1, 0].set_xlabel("time (s)")
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")
    for axis in axes[0, :]:
        axis.set_xlabel("time (s)")
    for axis in axes.ravel():
        axis.grid(True, alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if labels:
            axis.legend()
    fig.suptitle(
        f"One trained condition: {kind}, Es/E={params.soil_E_pa / params.pipe_E_pa:g}, "
        f"Mv={params.valve_mass_kg:g} kg, nu={params.pipe_nu:g}"
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def train_case(
    case_name: str,
    params: PhysicalParameters,
    output_dir: Path,
    config: PINNConfig,
    kind: SystemKind = "fssi",
    force: bool = False,
) -> Dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    if result_path.exists() and not force and cache_matches(result_path, params, config, kind):
        print(f"[{case_name}] using cached result: {result_path}")
        cached = load_result(result_path)
        summary_path = output_dir / "case_summary.png"
        if not summary_path.exists():
            save_case_summary(cached, params, kind, summary_path)
        return cached
    if result_path.exists() and not force:
        print(f"[{case_name}] cached settings differ; retraining this condition")

    device = select_device(config)
    scales_np = output_scales(params, kind)
    scales = torch.as_tensor(scales_np, device=device)
    model = CasePINN(params, config, len(scales_np)).to(device)
    data, _ = make_training_data(params, config, kind, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: List[Dict[str, float]] = []
    print(f"[{case_name}] training {kind} PINN on {device}; epochs={config.epochs}")
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = loss_terms(model, data, params, config, kind, scales)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        parts["epoch"] = float(epoch)
        history.append(parts)
        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            print(
                f"[{case_name}] epoch {epoch:5d} total={parts['total']:.3e} "
                f"pde={parts['pde']:.3e} bc={parts['boundary']:.3e} anchor={parts['anchor']:.3e}"
            )

    if config.lbfgs_steps > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=config.lbfgs_steps, history_size=50, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad(set_to_none=True)
            value, _ = loss_terms(model, data, params, config, kind, scales)
            value.backward()
            return value

        print(f"[{case_name}] LBFGS steps={config.lbfgs_steps}")
        optimizer_lbfgs.step(closure)

    result = predict_grid(model, params, config, scales, kind)
    metadata = {
        "case_name": case_name,
        "system_kind": kind,
        "model_revision": MODEL_REVISIONS[kind],
        "physical_parameters": asdict(params),
        "training_config": asdict(config),
        "output_scales": scales_np.tolist(),
        "final_loss": history[-1] if history else {},
    }
    np.savez_compressed(result_path, **result, metadata=json.dumps(metadata))
    torch.save(
        {"state_dict": model.state_dict(), "metadata": metadata}, output_dir / "model.pt"
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy([item["epoch"] for item in history], [item["total"] for item in history], label="total")
    ax.semilogy([item["epoch"] for item in history], [item["pde"] for item in history], label="PDE")
    ax.semilogy([item["epoch"] for item in history], [item["boundary"] for item in history], label="boundary")
    ax.semilogy([item["epoch"] for item in history], [item["anchor"] for item in history], label="anchor")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Training history: {case_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "training_history.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    save_case_summary(result, params, kind, output_dir / "case_summary.png")
    return result
