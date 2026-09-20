"""Characteristic time-marching pure PINN for the water-hammer cases.

The constant-coefficient hyperbolic PDE is embedded in the network through its
characteristics.  A small boundary network is trained independently in every
time slab.  The terminal state of one slab is the initial state of the next;
no measured values, FVM values, or paper-curve anchor points are used.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from water16_reproduction.common.cases import case_identifier
from water16_reproduction.common.physics import (
    PhysicalParameters,
    characteristic_basis,
    cross_section_areas,
    initial_pressure_pa,
)
from water16_reproduction.pure_pinn_reproduce import save_case_summary


SystemKind = Literal["fssi", "two_equation"]
WAVE_PINN_REVISION = 23
WAVE_PINN_CACHE_COMPATIBLE_REVISIONS = {23}
_SESSION_RESULTS: Dict[str, Dict[str, np.ndarray]] = {}


@dataclass
class WavePINNConfig:
    """Training controls.

    ``epochs`` and ``lbfgs_steps`` are totals distributed over all actual
    time slabs. ``causal_windows`` is a requested minimum; the trainer adds
    slabs when required by the fastest characteristic speed.
    """

    epochs: int = 6000
    lbfgs_steps: int = 200
    hidden_width: int = 64
    hidden_layers: int = 3
    pde_points: int = 6000
    boundary_points: int = 400
    learning_rate: float = 5.0e-4
    causal_windows: int = 8
    resample_every: int = 25
    adaptive_every: int = 10
    adaptive_beta: float = 0.9
    min_weight: float = 0.05
    max_weight: float = 20.0
    characteristic_cfl: float = 0.95
    max_boundary_residual: float = 0.10
    abort_boundary_residual: float = 0.25
    strict_boundary_validation: bool = False
    initial_stress_mode: str = "equilibrium"
    initial_velocity_m_s: float = 0.060
    assumed_inner_radius_m: float = 0.030
    valve_close_time_s: float = 0.03
    seed: int = 29
    log_every: int = 100
    dtype: str = "float64"
    device: str = "auto"
    plot_z: int = 180
    plot_t: int = 8001
    interface_points: int | None = None


def add_training_arguments(parser) -> None:
    parser.add_argument(
        "--epochs",
        type=int,
        default=6000,
        help="total Adam epochs distributed over all time slabs",
    )
    parser.add_argument(
        "--lbfgs-steps",
        type=int,
        default=200,
        help=(
            "requested total L-BFGS iterations; when enabled, every slab "
            "runs at least 12 iterations"
        ),
    )
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument(
        "--pde-points",
        type=int,
        default=6000,
        help="controls the carried interface-grid density (PDE is hard encoded)",
    )
    parser.add_argument(
        "--interface-points",
        type=int,
        default=None,
        help=(
            "explicit number of carried slab-interface points; by default it "
            "is derived from --pde-points and capped at 2000 for backward "
            "compatibility"
        ),
    )
    parser.add_argument("--boundary-points", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument(
        "--causal-windows",
        type=int,
        default=8,
        help="minimum number of time slabs; CFL-safe slabs are added automatically",
    )
    parser.add_argument("--resample-every", type=int, default=25)
    parser.add_argument("--adaptive-every", type=int, default=10)
    parser.add_argument("--adaptive-beta", type=float, default=0.9)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--characteristic-cfl", type=float, default=0.95)
    parser.add_argument(
        "--max-boundary-residual",
        type=float,
        default=0.10,
        help="largest normalized RMS valve-force residual accepted without warning",
    )
    parser.add_argument(
        "--abort-boundary-residual",
        type=float,
        default=0.25,
        help="normalized RMS valve-force residual that aborts a formal run",
    )
    parser.add_argument(
        "--allow-invalid-boundary",
        action="store_true",
        help="deprecated compatibility flag; non-strict continuation is now default",
    )
    parser.add_argument(
        "--strict-boundary-validation",
        action="store_true",
        help="abort the batch when an FSSI residual exceeds the abort threshold",
    )
    parser.add_argument(
        "--initial-stress-mode",
        choices=("equilibrium", "zero"),
        default="equilibrium",
        help="equilibrium enforces At*sigma0=Af*P0 before valve closure",
    )
    parser.add_argument(
        "--valve-close-time",
        type=float,
        default=0.03,
        help="smooth valve-closing duration in seconds; use 0 for instantaneous closure",
    )
    parser.add_argument(
        "--initial-velocity",
        type=float,
        default=0.060,
        help=(
            "assumed steady velocity in m/s; the paper does not publish the "
            "valve parameters needed to determine it"
        ),
    )
    parser.add_argument(
        "--inner-radius",
        type=float,
        default=0.030,
        help=(
            "assumed inner radius in m; the paper publishes only e/R=0.1"
        ),
    )
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--plot-time-points",
        type=int,
        default=8001,
        help="saved time samples; 8001 matches the paper's 1e-4 s FFT spacing",
    )
    parser.add_argument("--quick", action="store_true", help="small code-only smoke run")
    parser.add_argument("--force", action="store_true", help="ignore matching cache")


def config_from_args(args) -> WavePINNConfig:
    config = WavePINNConfig(
        epochs=args.epochs,
        lbfgs_steps=args.lbfgs_steps,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        pde_points=args.pde_points,
        boundary_points=args.boundary_points,
        learning_rate=args.learning_rate,
        causal_windows=args.causal_windows,
        resample_every=args.resample_every,
        adaptive_every=args.adaptive_every,
        adaptive_beta=args.adaptive_beta,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        characteristic_cfl=args.characteristic_cfl,
        max_boundary_residual=args.max_boundary_residual,
        abort_boundary_residual=args.abort_boundary_residual,
        strict_boundary_validation=(
            args.strict_boundary_validation and not args.allow_invalid_boundary
        ),
        initial_stress_mode=args.initial_stress_mode,
        initial_velocity_m_s=args.initial_velocity,
        assumed_inner_radius_m=args.inner_radius,
        valve_close_time_s=args.valve_close_time,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        plot_t=args.plot_time_points,
        interface_points=args.interface_points,
    )
    if config.epochs < 1:
        raise ValueError("--epochs must be positive")
    if config.causal_windows < 1:
        raise ValueError("--causal-windows must be at least 1")
    if not 0.0 < config.characteristic_cfl <= 1.0:
        raise ValueError("--characteristic-cfl must be in (0, 1]")
    if config.max_boundary_residual <= 0.0:
        raise ValueError("--max-boundary-residual must be positive")
    if config.abort_boundary_residual <= config.max_boundary_residual:
        raise ValueError(
            "--abort-boundary-residual must exceed --max-boundary-residual"
        )
    if config.boundary_points < 4:
        raise ValueError("--boundary-points must be at least 4")
    if config.valve_close_time_s < 0.0:
        raise ValueError("--valve-close-time cannot be negative")
    if config.initial_velocity_m_s <= 0.0:
        raise ValueError("--initial-velocity must be positive")
    if config.assumed_inner_radius_m <= 0.0:
        raise ValueError("--inner-radius must be positive")
    if config.plot_t < 100:
        raise ValueError("--plot-time-points must be at least 100")
    if config.interface_points is not None and config.interface_points < 16:
        raise ValueError("--interface-points must be at least 16")
    if args.quick:
        config.epochs = min(config.epochs, 8)
        config.lbfgs_steps = 0
        config.hidden_width = min(config.hidden_width, 24)
        config.hidden_layers = min(config.hidden_layers, 2)
        config.pde_points = min(config.pde_points, 80)
        config.boundary_points = min(config.boundary_points, 20)
        config.causal_windows = min(config.causal_windows, 2)
        config.resample_every = 2
        config.adaptive_every = 1
        config.log_every = 1
        config.plot_z = 40
        config.plot_t = 100
        if config.interface_points is not None:
            config.interface_points = min(config.interface_points, 80)
        config.strict_boundary_validation = False
    return config


def select_device(config: WavePINNConfig) -> torch.device:
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def characteristic_data(
    params: PhysicalParameters, kind: SystemKind
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if kind == "fssi":
        return characteristic_basis(params)

    rho = params.water_density_kg_m3
    bulk = params.water_bulk_modulus_pa
    speed = math.sqrt(bulk / rho)
    scales = np.array(
        [
            max(abs(params.initial_velocity_m_s), 1.0e-3),
            max(initial_pressure_pa(params), rho * speed * abs(params.initial_velocity_m_s)),
        ],
        dtype=float,
    )
    advection = np.array([[0.0, 1.0 / rho], [bulk, 0.0]], dtype=float)
    scaled = (advection * scales[np.newaxis, :]) / scales[:, np.newaxis]
    values, right = np.linalg.eig(scaled)
    order = np.argsort(np.real(values))
    values = np.real(values[order])
    right = np.real(right[:, order])
    return values, right, np.linalg.inv(right), scales


def signed_characteristic_speeds(
    params: PhysicalParameters, kind: SystemKind
) -> np.ndarray:
    return characteristic_data(params, kind)[0]


def output_scales(params: PhysicalParameters, kind: SystemKind) -> np.ndarray:
    return characteristic_data(params, kind)[3]


def actual_slab_count(
    params: PhysicalParameters,
    kind: SystemKind,
    config: WavePINNConfig,
) -> int:
    speeds = signed_characteristic_speeds(params, kind)
    required = math.ceil(
        params.t_final_s
        * float(np.max(np.abs(speeds)))
        / (config.characteristic_cfl * params.length_m)
    )
    return max(config.causal_windows, required, 1)


def as_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=torch.get_default_dtype())


class CharacteristicBoundaryNetwork(nn.Module):
    """Smooth one-dimensional network for incoming characteristic amplitudes."""

    def __init__(
        self,
        mode_count: int,
        t_start: float,
        t_end: float,
        config: WavePINNConfig,
    ):
        super().__init__()
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        layers: list[nn.Module] = []
        input_size = 1
        for _ in range(max(config.hidden_layers, 1)):
            layer = nn.Linear(input_size, config.hidden_width)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.extend([layer, nn.Tanh()])
            input_size = config.hidden_width
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(config.hidden_width, mode_count)
        self.register_buffer(
            "valve_base_time", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "valve_base_velocity", torch.empty(0), persistent=False
        )
        # Each slab starts from the propagated characteristic state.  A zero
        # correction avoids injecting an arbitrary boundary wave at the slab
        # interface before the boundary residual has been optimized.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def normalized_time(self, time: torch.Tensor) -> torch.Tensor:
        return ((time - self.t_start) / (self.t_end - self.t_start)).clamp(0.0, 1.0)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        tau = self.normalized_time(time)
        return self.output(self.hidden(2.0 * tau - 1.0))

def interpolate_profile(
    profile_z: torch.Tensor,
    profile_values: torch.Tensor,
    query_z: torch.Tensor,
) -> torch.Tensor:
    """Differentiable piecewise-linear interpolation in z."""

    flat = query_z.reshape(-1)
    clipped = flat.clamp(float(profile_z[0]), float(profile_z[-1]))
    indices = torch.searchsorted(profile_z, clipped, right=True) - 1
    indices = indices.clamp(0, len(profile_z) - 2)
    z0 = profile_z[indices]
    z1 = profile_z[indices + 1]
    fraction = (clipped - z0) / (z1 - z0)
    y0 = profile_values[indices]
    y1 = profile_values[indices + 1]
    result = y0 + fraction.unsqueeze(-1) * (y1 - y0)
    return result.reshape(*query_z.shape, profile_values.shape[-1])


def right_boundary_affine_force(
    time: torch.Tensor,
    t_start: float,
    profile_z: torch.Tensor,
    profile_w: torch.Tensor,
    speeds: torch.Tensor,
    physical_modes: torch.Tensor,
    params: PhysicalParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return b(t), s in Af*P-At*sigma = b(t) + s*uz at the valve."""

    elapsed = time - t_start
    columns = []
    for mode, speed_tensor in enumerate(speeds):
        if float(speed_tensor.detach().cpu()) > 0.0:
            foot = params.length_m - speed_tensor * elapsed
            value = interpolate_profile(
                profile_z, profile_w[:, mode : mode + 1], foot
            ).reshape(-1, 1)
        else:
            value = torch.zeros_like(time)
        columns.append(value)
    amplitudes = torch.cat(columns, dim=1)
    incoming = torch.nonzero(speeds < 0.0, as_tuple=False).flatten()
    conditions = torch.zeros(
        (2, len(speeds)), device=time.device, dtype=time.dtype
    )
    conditions[0, 0] = 1.0
    conditions[0, 1] = -1.0
    conditions[1, 1] = 1.0
    matrix = conditions @ physical_modes[:, incoming]
    physical_without_incoming = amplitudes @ physical_modes.T
    targets = torch.cat(
        [valve_velocity_target(params, time), torch.zeros_like(time)],
        dim=1,
    )
    solved = torch.linalg.solve(
        matrix,
        (targets - physical_without_incoming @ conditions.T).T,
    ).T
    amplitudes[:, incoming] = solved
    zero_axial_state = amplitudes @ physical_modes.T
    area_f, area_t = cross_section_areas(params)
    force_without_uz = (
        area_f * zero_axial_state[:, 2:3]
        - area_t * zero_axial_state[:, 3:4]
    )
    unit_axial_target = torch.tensor(
        [0.0, 1.0], device=time.device, dtype=time.dtype
    )
    incoming_response = torch.linalg.solve(matrix, unit_axial_target)
    physical_tangent = physical_modes[:, incoming] @ incoming_response
    force_sensitivity = (
        area_f * physical_tangent[2] - area_t * physical_tangent[3]
    )
    return force_without_uz, force_sensitivity


def prepare_valve_base_trajectory(
    model: CharacteristicBoundaryNetwork,
    profile_z: torch.Tensor,
    profile_w: torch.Tensor,
    speeds: torch.Tensor,
    right_vectors: torch.Tensor,
    scales: torch.Tensor,
    params: PhysicalParameters,
    minimum_points: int,
) -> None:
    """Integrate the assumed moving-valve end-mass balance."""

    physical_modes = scales[:, None] * right_vectors
    probe_time = torch.tensor(
        [[model.t_start]], device=profile_z.device, dtype=profile_z.dtype
    )
    _, sensitivity = right_boundary_affine_force(
        probe_time,
        model.t_start,
        profile_z,
        profile_w,
        speeds,
        physical_modes,
        params,
    )
    relaxation_rate = float(
        (-sensitivity / params.valve_mass_kg).detach().cpu()
    )
    duration = model.t_end - model.t_start
    stiffness_points = (
        math.ceil(6.0 * max(relaxation_rate, 0.0) * duration) + 1
    )
    point_count = max(minimum_points, stiffness_points, 64)
    point_count = min(point_count, 4096)
    time = torch.linspace(
        model.t_start,
        model.t_end,
        point_count,
        device=profile_z.device,
        dtype=profile_z.dtype,
    ).reshape(-1, 1)
    with torch.no_grad():
        force_without, sensitivity = right_boundary_affine_force(
            time,
            model.t_start,
            profile_z,
            profile_w,
            speeds,
            physical_modes,
            params,
        )
        rate = sensitivity / params.valve_mass_kg
        start_state = profile_w[-1:] @ physical_modes.T
        values = [start_state[:, 1:2]]
        for index in range(point_count - 1):
            delta_t = time[index + 1] - time[index]
            if abs(float(rate.detach().cpu())) < 1.0e-12:
                decay = torch.ones_like(rate)
                forcing_factor = delta_t
            else:
                decay = torch.exp(rate * delta_t)
                forcing_factor = torch.expm1(rate * delta_t) / rate
            average_force = 0.5 * (
                force_without[index] + force_without[index + 1]
            )
            next_value = (
                decay * values[-1]
                + forcing_factor * average_force / params.valve_mass_kg
            )
            values.append(next_value.reshape(1, 1))
        model.valve_base_time = time.flatten().detach()
        model.valve_base_velocity = torch.cat(values).flatten().detach()


def unprojected_incoming_amplitudes(
    model: CharacteristicBoundaryNetwork,
    time: torch.Tensor,
    profile_w: torch.Tensor,
    speeds: torch.Tensor,
    first_slab: bool,
) -> torch.Tensor:
    raw = model(time)
    tau = model.normalized_time(time)
    positive = (speeds > 0.0).reshape(1, -1)
    base = torch.where(positive, profile_w[0:1], profile_w[-1:])
    if first_slab:
        gate = torch.where(positive, tau, torch.ones_like(tau))
    else:
        gate = tau.expand(-1, len(speeds))
    return base + gate * raw


def valve_velocity_target(
    params: PhysicalParameters, time: torch.Tensor
) -> torch.Tensor:
    """Cosine valve-closing profile, with zero duration meaning an instant stop."""

    close_time = params.valve_close_time_s
    if close_time <= 0.0:
        return torch.zeros_like(time)
    phase = (time / close_time).clamp(0.0, 1.0)
    opening = 0.5 * (1.0 + torch.cos(math.pi * phase))
    return params.initial_velocity_m_s * opening


def projected_boundary_amplitudes(
    model: CharacteristicBoundaryNetwork,
    time: torch.Tensor,
    side: Literal["left", "right"],
    profile_z: torch.Tensor,
    profile_w: torch.Tensor,
    speeds: torch.Tensor,
    right_vectors: torch.Tensor,
    scales: torch.Tensor,
    params: PhysicalParameters,
    kind: SystemKind,
    first_slab: bool,
) -> torch.Tensor:
    """Return amplitudes after imposing the physical boundary conditions.

    Outgoing characteristics always come from the carried slab profile.  The
    classical branch hard-projects P=P0 or V=Vvalve.  The FSSI branch uses
    P=P0 and uz=0 upstream.  At the moving valve it prescribes the relative
    flow V-uz=Vrel together with a learned, unbounded uz; the latter is
    trained only through the valve force balance.
    """

    candidate = unprojected_incoming_amplitudes(
        model, time, profile_w, speeds, first_slab
    )
    elapsed = time - model.t_start
    columns = []
    for mode, speed_tensor in enumerate(speeds):
        speed = float(speed_tensor.detach().cpu())
        if side == "left" and speed < 0.0:
            foot = -speed_tensor * elapsed
            value = interpolate_profile(
                profile_z, profile_w[:, mode : mode + 1], foot
            ).reshape(-1, 1)
        elif side == "right" and speed > 0.0:
            foot = params.length_m - speed_tensor * elapsed
            value = interpolate_profile(
                profile_z, profile_w[:, mode : mode + 1], foot
            ).reshape(-1, 1)
        else:
            value = candidate[:, mode : mode + 1]
        columns.append(value)
    amplitudes = torch.cat(columns, dim=1)
    physical_modes = scales[:, None] * right_vectors

    if kind == "two_equation":
        if side == "left":
            incoming = torch.nonzero(speeds > 0.0, as_tuple=False).flatten()
            condition = torch.zeros(
                (1, len(scales)), device=time.device, dtype=time.dtype
            )
            condition[0, 1] = 1.0
            target = torch.full_like(time, initial_pressure_pa(params))
        else:
            incoming = torch.nonzero(speeds < 0.0, as_tuple=False).flatten()
            condition = torch.zeros(
                (1, len(scales)), device=time.device, dtype=time.dtype
            )
            condition[0, 0] = 1.0
            target = valve_velocity_target(params, time)
        matrix = condition @ physical_modes[:, incoming]
        physical_without_incoming = (
            amplitudes @ physical_modes.T
            - amplitudes[:, incoming] @ physical_modes[:, incoming].T
        )
        rhs = target - physical_without_incoming @ condition.T
        result = amplitudes.clone()
        result[:, incoming] = torch.linalg.solve(matrix, rhs.T).T
        return result

    if side == "left":
        incoming = torch.nonzero(speeds > 0.0, as_tuple=False).flatten()
        conditions = torch.zeros(
            (2, len(scales)), device=time.device, dtype=time.dtype
        )
        conditions[0, 2] = 1.0
        conditions[1, 1] = 1.0
        targets = torch.tensor(
            [initial_pressure_pa(params), 0.0],
            device=time.device,
            dtype=time.dtype,
        )
        matrix = conditions @ physical_modes[:, incoming]
        physical_without_incoming = (
            amplitudes @ physical_modes.T
            - amplitudes[:, incoming] @ physical_modes[:, incoming].T
        )
        rhs = targets.reshape(1, -1) - physical_without_incoming @ conditions.T
        solved = torch.linalg.solve(matrix, rhs.T).T
        result = amplitudes.clone()
        result[:, incoming] = solved
        return result

    incoming = torch.nonzero(speeds < 0.0, as_tuple=False).flatten()
    conditions = torch.zeros(
        (2, len(scales)), device=time.device, dtype=time.dtype
    )
    conditions[0, 0] = 1.0
    conditions[0, 1] = -1.0
    conditions[1, 1] = 1.0
    matrix = conditions @ physical_modes[:, incoming]
    physical_without_incoming = (
        amplitudes @ physical_modes.T
        - amplitudes[:, incoming] @ physical_modes[:, incoming].T
    )
    raw = model(time)
    start_physical = profile_w[-1:] @ physical_modes.T
    relative_target = valve_velocity_target(params, time)

    # With the valve-relative flow V-uz prescribed, the valve force is
    # affine in uz:
    # Af*P-At*sigma_z = force_without_uz + force_sensitivity*uz.
    # The right-end force balance is therefore a stiff, stable first-order
    # ODE. Embed its local equilibrium and homogeneous exponential so the
    # network learns only the smooth remainder instead of the fast boundary
    # layer at the start of every time slab.
    zero_axial_targets = torch.cat(
        [relative_target, torch.zeros_like(relative_target)], dim=1
    )
    zero_axial_solution = torch.linalg.solve(
        matrix,
        (
            zero_axial_targets
            - physical_without_incoming @ conditions.T
        ).T,
    ).T
    zero_axial_amplitudes = amplitudes.clone()
    zero_axial_amplitudes[:, incoming] = zero_axial_solution
    zero_axial_state = zero_axial_amplitudes @ physical_modes.T
    area_f, area_t = cross_section_areas(params)
    force_without_uz = (
        area_f * zero_axial_state[:, 2:3]
        - area_t * zero_axial_state[:, 3:4]
    )
    unit_axial_target = torch.tensor(
        [0.0, 1.0], device=time.device, dtype=time.dtype
    )
    incoming_response = torch.linalg.solve(matrix, unit_axial_target)
    physical_tangent = physical_modes[:, incoming] @ incoming_response
    force_sensitivity = (
        area_f * physical_tangent[2] - area_t * physical_tangent[3]
    )
    relaxation_rate = -force_sensitivity / params.valve_mass_kg
    if model.valve_base_time.numel() > 1:
        base_axial_velocity = interpolate_profile(
            model.valve_base_time,
            model.valve_base_velocity.reshape(-1, 1),
            time,
        ).reshape(-1, 1)
        elapsed = time - model.t_start
        if float(relaxation_rate.detach().cpu()) > 0.0:
            correction_gate = -torch.expm1(-relaxation_rate * elapsed)
        else:
            correction_gate = model.normalized_time(time)
        axial_velocity = (
            base_axial_velocity
            + correction_gate * scales[1] * raw[:, 0:1]
        )
    elif float(relaxation_rate.detach().cpu()) > 0.0:
        equilibrium = -force_without_uz / force_sensitivity
        start_force = (
            area_f * start_physical[:, 2:3]
            - area_t * start_physical[:, 3:4]
        )
        start_equilibrium = -(
            start_force - force_sensitivity * start_physical[:, 1:2]
        ) / force_sensitivity
        elapsed = time - model.t_start
        homogeneous = torch.exp(-relaxation_rate * elapsed)
        base_axial_velocity = equilibrium + homogeneous * (
            start_physical[:, 1:2] - start_equilibrium
        )
        correction_gate = -torch.expm1(-relaxation_rate * elapsed)
        axial_velocity = (
            base_axial_velocity
            + correction_gate * scales[1] * raw[:, 0:1]
        )
    else:
        tau = model.normalized_time(time)
        axial_velocity = (
            start_physical[:, 1:2] + tau * scales[1] * raw[:, 0:1]
        )
    targets = torch.cat(
        [relative_target, axial_velocity], dim=1
    )
    solved = torch.linalg.solve(
        matrix, (targets - physical_without_incoming @ conditions.T).T
    ).T
    result = amplitudes.clone()
    result[:, incoming] = solved
    return result


def characteristic_amplitudes(
    model: CharacteristicBoundaryNetwork,
    zt: torch.Tensor,
    profile_z: torch.Tensor,
    profile_w: torch.Tensor,
    speeds: torch.Tensor,
    right_vectors: torch.Tensor,
    scales: torch.Tensor,
    params: PhysicalParameters,
    kind: SystemKind,
    first_slab: bool,
) -> torch.Tensor:
    z = zt[:, 0:1]
    time = zt[:, 1:2]
    elapsed = time - model.t_start
    values = []
    for mode, speed_tensor in enumerate(speeds):
        speed = float(speed_tensor.detach().cpu())
        foot = z - speed_tensor * elapsed
        initial = interpolate_profile(
            profile_z, profile_w[:, mode : mode + 1], foot
        ).reshape(-1, 1)
        if speed > 0.0:
            boundary_time = time - z / speed_tensor
            incoming = projected_boundary_amplitudes(
                model,
                boundary_time.clamp(model.t_start, model.t_end),
                "left",
                profile_z,
                profile_w,
                speeds,
                right_vectors,
                scales,
                params,
                kind,
                first_slab,
            )[:, mode : mode + 1]
            value = torch.where(foot < 0.0, incoming, initial)
        else:
            boundary_time = time - (params.length_m - z) / (-speed_tensor)
            incoming = projected_boundary_amplitudes(
                model,
                boundary_time.clamp(model.t_start, model.t_end),
                "right",
                profile_z,
                profile_w,
                speeds,
                right_vectors,
                scales,
                params,
                kind,
                first_slab,
            )[:, mode : mode + 1]
            value = torch.where(foot > params.length_m, incoming, initial)
        values.append(value)
    return torch.cat(values, dim=1)


def physical_output(
    model: CharacteristicBoundaryNetwork,
    zt: torch.Tensor,
    profile_z: torch.Tensor,
    profile_w: torch.Tensor,
    speeds: torch.Tensor,
    right_vectors: torch.Tensor,
    scales: torch.Tensor,
    params: PhysicalParameters,
    first_slab: bool,
) -> torch.Tensor:
    amplitudes = characteristic_amplitudes(
        model,
        zt,
        profile_z,
        profile_w,
        speeds,
        right_vectors,
        scales,
        params,
        "fssi" if len(scales) == 4 else "two_equation",
        first_slab,
    )
    return (amplitudes @ right_vectors.T) * scales


def initial_physical_state(
    params: PhysicalParameters,
    kind: SystemKind,
    count: int,
    initial_stress_mode: str = "equilibrium",
) -> np.ndarray:
    if kind == "two_equation":
        row = [params.initial_velocity_m_s, initial_pressure_pa(params)]
    else:
        if initial_stress_mode == "equilibrium":
            area_f, area_t = cross_section_areas(params)
            initial_stress = area_f * initial_pressure_pa(params) / area_t
        elif initial_stress_mode == "zero":
            initial_stress = 0.0
        else:
            raise ValueError(
                "initial_stress_mode must be 'equilibrium' or 'zero'"
            )
        row = [
            params.initial_velocity_m_s,
            0.0,
            initial_pressure_pa(params),
            initial_stress,
        ]
    return np.repeat(np.asarray(row, dtype=float)[None, :], count, axis=0)


def physical_to_characteristic(
    physical: torch.Tensor,
    left_vectors: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    return (physical / scales) @ left_vectors.T


def sample_boundary_times(
    t_start: float,
    t_end: float,
    count: int,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    edges = np.linspace(0.0, 1.0, count + 1)
    fractions = edges[:-1] + rng.random(count) * np.diff(edges)
    epsilon = max(np.finfo(float).eps, 1.0e-7)
    fractions = np.clip(fractions, epsilon, 1.0)
    values = t_start + fractions * (t_end - t_start)
    return as_tensor(values[:, None], device)


def gradient(values: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        values,
        inputs,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True,
    )[0]


def boundary_losses(
    model: CharacteristicBoundaryNetwork,
    times: torch.Tensor,
    profile_z: torch.Tensor,
    profile_w: torch.Tensor,
    speeds: torch.Tensor,
    right_vectors: torch.Tensor,
    scales: torch.Tensor,
    params: PhysicalParameters,
    kind: SystemKind,
    first_slab: bool,
) -> Dict[str, torch.Tensor]:
    time_values = times.detach().clone().requires_grad_(kind == "fssi")
    left_points = torch.cat([torch.zeros_like(time_values), time_values], dim=1)
    right_points = torch.cat(
        [torch.full_like(time_values, params.length_m), time_values], dim=1
    )
    left = physical_output(
        model,
        left_points,
        profile_z,
        profile_w,
        speeds,
        right_vectors,
        scales,
        params,
        first_slab,
    )
    right = physical_output(
        model,
        right_points,
        profile_z,
        profile_w,
        speeds,
        right_vectors,
        scales,
        params,
        first_slab,
    )
    p_index = 1 if kind == "two_equation" else 2
    if kind == "two_equation":
        return {
            "bc_hard_projection": torch.mean(model(time_values) ** 2) * 0.0
            + torch.mean(
                (
                    (left[:, p_index] - initial_pressure_pa(params))
                    / scales[p_index]
                )
                ** 2
            )
            + torch.mean(
                (
                    (right[:, 0:1] - valve_velocity_target(params, time_values))
                    / scales[0]
                )
                ** 2
            )
        }

    uz_t = gradient(right[:, 1:2], time_values)
    area_f, area_t = cross_section_areas(params)
    # Energy-consistent moving end mass: pressure work accelerates the valve
    # while pipe-wall tension opposes that motion. This is an explicit
    # assumption because the buried-pipe valve law is incomplete in the paper.
    force = (
        params.valve_mass_kg * uz_t
        - area_f * right[:, 2:3]
        + area_t * right[:, 3:4]
    )
    force_scale = max(
        area_f * float(scales[2].detach().cpu()),
        area_t * float(scales[3].detach().cpu()),
        1.0,
    )
    normalized_force = force / force_scale
    force_chunks = torch.tensor_split(normalized_force, 4)
    losses = {
        f"bc_valve_force_t{index + 1}": torch.mean(chunk**2)
        for index, chunk in enumerate(force_chunks)
        if len(chunk)
    }
    start_time = torch.full_like(time_values[:1], model.t_start)
    start_point = torch.cat(
        [torch.full_like(start_time, params.length_m), start_time], dim=1
    )
    start_state = physical_output(
        model,
        start_point,
        profile_z,
        profile_w,
        speeds,
        right_vectors,
        scales,
        params,
        first_slab,
    )
    state_with_start = torch.cat([start_state, right], dim=0)
    time_with_start = torch.cat([start_time, time_values], dim=0)
    physical_modes = scales[:, None] * right_vectors
    incoming = torch.nonzero(speeds < 0.0, as_tuple=False).flatten()
    velocity_conditions = torch.zeros(
        (2, len(scales)), device=time_values.device, dtype=time_values.dtype
    )
    velocity_conditions[0, 0] = 1.0
    velocity_conditions[0, 1] = -1.0
    velocity_conditions[1, 1] = 1.0
    incoming_matrix = velocity_conditions @ physical_modes[:, incoming]
    unit_axial_target = torch.tensor(
        [0.0, 1.0], device=time_values.device, dtype=time_values.dtype
    )
    incoming_response = torch.linalg.solve(
        incoming_matrix, unit_axial_target
    )
    physical_tangent = physical_modes[:, incoming] @ incoming_response
    force_row = torch.tensor(
        [0.0, 0.0, area_f, -area_t],
        device=time_values.device,
        dtype=time_values.dtype,
    )
    force_sensitivity = force_row @ physical_tangent
    decay_rate = force_sensitivity / params.valve_mass_kg
    state_force = (
        area_f * state_with_start[:, 2:3]
        - area_t * state_with_start[:, 3:4]
    )
    independent_force = (
        state_force - force_sensitivity * state_with_start[:, 1:2]
    )
    delta_t = time_with_start[1:] - time_with_start[:-1]
    exponential = torch.exp(decay_rate * delta_t)
    if abs(float(decay_rate.detach().cpu())) < 1.0e-12:
        forcing_factor = delta_t
    else:
        forcing_factor = torch.expm1(decay_rate * delta_t) / decay_rate
    step_target = (
        exponential * state_with_start[:-1, 1:2]
        + forcing_factor
        * 0.5
        * (independent_force[:-1] + independent_force[1:])
        / params.valve_mass_kg
    )
    step_residual = (state_with_start[1:, 1:2] - step_target) / scales[1]
    step_chunks = torch.tensor_split(step_residual, 4)
    losses.update(
        {
            f"bc_valve_exponential_t{index + 1}": torch.mean(chunk**2)
            for index, chunk in enumerate(step_chunks)
            if len(chunk)
        }
    )
    return losses


class GradientNormBalancer:
    """Adaptive weights that equalise parameter-gradient norms."""

    def __init__(self, names: Iterable[str], config: WavePINNConfig):
        self.names = tuple(names)
        self.beta = config.adaptive_beta
        self.minimum = config.min_weight
        self.maximum = config.max_weight
        self.weights = {name: 1.0 for name in self.names}

    def update(
        self, losses: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, float]:
        parameters = [item for item in model.parameters() if item.requires_grad]
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
            norms[name] = max(float(torch.sqrt(squared).cpu()), 1.0e-20)
        target = math.exp(float(np.mean(np.log(list(norms.values())))))
        proposed = {
            name: float(np.clip(target / norms[name], self.minimum, self.maximum))
            for name in self.names
        }
        normalizer = len(self.names) / max(sum(proposed.values()), 1.0e-20)
        for name in self.names:
            target_weight = proposed[name] * normalizer
            self.weights[name] = (
                self.beta * self.weights[name]
                + (1.0 - self.beta) * target_weight
            )
        normalizer = len(self.names) / max(sum(self.weights.values()), 1.0e-20)
        self.weights = {
            name: float(
                np.clip(self.weights[name] * normalizer, self.minimum, self.maximum)
            )
            for name in self.names
        }
        return dict(self.weights)

    def total(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.weights[name] * losses[name] for name in self.names)


def wave_case_directory(
    root: Path, params: PhysicalParameters, kind: SystemKind
) -> Path:
    return root / case_identifier(params, kind)


def cache_metadata_matches(
    metadata: dict,
    params: PhysicalParameters,
    config: WavePINNConfig,
    kind: SystemKind,
) -> bool:
    validation_only_fields = {
        "max_boundary_residual",
        "abort_boundary_residual",
        "strict_boundary_validation",
    }
    cached_config = dict(metadata.get("training_config", {}))
    requested_config = asdict(config)
    # Revision 23 derived the interface density implicitly.  Treat a missing
    # field as the unchanged legacy default so completed formal runs remain
    # reusable after the refinement control was made explicit.
    cached_config.setdefault("interface_points", None)
    for field_name in validation_only_fields:
        cached_config.pop(field_name, None)
        requested_config.pop(field_name, None)
    return (
        metadata.get("trainer") == "characteristic_time_marching_pinn"
        and metadata.get("revision")
        in WAVE_PINN_CACHE_COMPATIBLE_REVISIONS
        and metadata.get("system_kind") == kind
        and metadata.get("physical_parameters") == asdict(params)
        and cached_config == requested_config
    )


def cache_matches(
    path: Path,
    params: PhysicalParameters,
    config: WavePINNConfig,
    kind: SystemKind,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        return cache_metadata_matches(metadata, params, config, kind)
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def load_result(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata"
        }


def diagnostic_report(
    result: Dict[str, np.ndarray],
    params: PhysicalParameters,
    kind: SystemKind,
    initial_stress_mode: str,
) -> Dict[str, float | str]:
    p0 = initial_pressure_pa(params)
    pressure = result["P"][-1]
    centered = pressure - np.mean(pressure)
    crossings = int(np.count_nonzero(centered[:-1] * centered[1:] < 0.0))
    if params.valve_close_time_s > 0.0:
        phase = np.clip(result["t"] / params.valve_close_time_s, 0.0, 1.0)
        valve_target = (
            params.initial_velocity_m_s
            * 0.5
            * (1.0 + np.cos(np.pi * phase))
        )
    else:
        valve_target = np.zeros_like(result["t"])
        valve_target[0] = params.initial_velocity_m_s
    closed = result["t"] >= params.valve_close_time_s
    valve_kinematic_velocity = result["V"][-1]
    if kind == "fssi":
        valve_kinematic_velocity = (
            valve_kinematic_velocity - result["uz"][-1]
        )
    report = {
        "initial_velocity_max_error": float(
            np.max(np.abs(result["V"][:, 0] - params.initial_velocity_m_s))
        ),
        "initial_pressure_max_error": float(
            np.max(np.abs(result["P"][:, 0] - p0))
        ),
        "upstream_pressure_max_error": float(
            np.max(np.abs(result["P"][0] - p0))
        ),
        "valve_velocity_max_abs": float(np.max(np.abs(result["V"][-1, 1:]))),
        "valve_kinematic_target_max_error": float(
            np.max(np.abs(valve_kinematic_velocity - valve_target))
        ),
        "closed_valve_relative_velocity_max_abs": float(
            np.max(np.abs(valve_kinematic_velocity[closed]))
            if np.any(closed)
            else np.nan
        ),
        "valve_pressure_standard_deviation_pa": float(np.std(pressure)),
        "valve_pressure_mean_crossings": float(crossings),
    }
    if kind == "fssi":
        valve_axial_max = float(np.max(np.abs(result["uz"][-1])))
        area_f, area_t = cross_section_areas(params)
        force = (
            params.valve_mass_kg
            * np.gradient(result["uz"][-1], result["t"], edge_order=2)
            - area_f * result["P"][-1]
            + area_t * result["sigma_z"][-1]
        )
        scales = output_scales(params, kind)
        force_scale = max(
            area_f * float(scales[2]),
            area_t * float(scales[3]),
            1.0,
        )
        expected_stress = initial_physical_state(
            params, kind, 1, initial_stress_mode
        )[0, 3]
        report["initial_axial_velocity_max_error"] = float(
            np.max(np.abs(result["uz"][:, 0]))
        )
        report["initial_stress_value_pa"] = float(expected_stress)
        report["initial_stress_max_error"] = float(
            np.max(np.abs(result["sigma_z"][:, 0] - expected_stress))
        )
        report["upstream_axial_velocity_max_error"] = float(
            np.max(np.abs(result["uz"][0]))
        )
        report["valve_axial_velocity_max_abs"] = valve_axial_max
        report["valve_force_balance_rms_n"] = float(
            np.sqrt(np.mean(force**2))
        )
        report["valve_force_balance_normalized_rms"] = float(
            np.sqrt(np.mean((force / force_scale) ** 2))
        )
    return report


def boundary_validation_status(
    normalized_residual: float,
    accepted_residual: float,
    abort_residual: float,
) -> str:
    """Classify an FSSI valve-force residual without hiding marginal cases."""

    if normalized_residual <= accepted_residual:
        return "pass"
    if normalized_residual <= abort_residual:
        return "warning"
    return "failed"


def annotate_boundary_validation(
    report: Dict[str, float | str],
    kind: SystemKind,
    config: WavePINNConfig,
) -> str:
    status = "pass"
    if kind == "fssi":
        status = boundary_validation_status(
            float(report["valve_force_balance_normalized_rms"]),
            config.max_boundary_residual,
            config.abort_boundary_residual,
        )
    report["boundary_validation_status"] = status
    report["boundary_validation_pass"] = float(status == "pass")
    report["boundary_validation_abort_threshold"] = (
        config.abort_boundary_residual
    )
    return status


def enforce_boundary_validation(
    name: str,
    report: Dict[str, float | str],
    kind: SystemKind,
    config: WavePINNConfig,
    output_dir: Path,
) -> None:
    if kind != "fssi":
        return
    status = str(report["boundary_validation_status"])
    residual = float(report["valve_force_balance_normalized_rms"])
    print(
        f"[{name}] valve-force normalized RMS={residual:.3e} "
        f"(accepted {config.max_boundary_residual:.3e}, "
        f"abort {config.abort_boundary_residual:.3e}, status {status})"
    )
    if status == "warning":
        print(
            f"[{name}] WARNING: the valve-force residual exceeds the "
            "acceptance target; comparison output will be generated and "
            "marked as marginal in diagnostics.json."
        )
    if status == "failed" and not config.strict_boundary_validation:
        print(
            f"[{name}] QUALITY FAILURE: the valve-force residual exceeds "
            "the abort threshold. The batch will continue, but this case "
            "must not be treated as a validated reproduction."
        )
    if config.strict_boundary_validation and status == "failed":
        raise RuntimeError(
            f"{name}: valve-force boundary residual {residual:.3e} exceeds "
            f"the abort threshold {config.abort_boundary_residual:.3e}; "
            f"diagnostic files were saved in {output_dir}"
        )


def save_training_history(
    history: list[Dict[str, float]], output_dir: Path
) -> None:
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    steps = [item["global_step"] for item in history]
    loss_names = sorted(
        key.removeprefix("loss_")
        for key in history[0]
        if key.startswith("loss_") and key != "loss_total"
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    axes[0].semilogy(
        steps, [item["loss_total"] for item in history], label="total", linewidth=2
    )
    for name in loss_names:
        axes[0].semilogy(
            steps,
            [max(item[f"loss_{name}"], 1.0e-30) for item in history],
            label=name,
        )
        axes[1].plot(
            steps,
            [item[f"weight_{name}"] for item in history],
            label=name,
        )
    axes[2].plot(
        steps,
        [item["slab_end_s"] for item in history],
        label="completed slab end",
    )
    axes[0].set(xlabel="global Adam step", ylabel="loss", title="Boundary PINN losses")
    axes[1].set(xlabel="global Adam step", ylabel="weight", title="Adaptive weights")
    axes[2].set(xlabel="global Adam step", ylabel="time (s)", title="True time marching")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    fig.savefig(output_dir / "training_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _allocate_steps(total: int, count: int, index: int, minimum: int = 0) -> int:
    if total <= 0:
        return minimum
    base, remainder = divmod(total, count)
    return max(minimum, base + (1 if index < remainder else 0))


def train_wave_case(
    name: str,
    params: PhysicalParameters,
    kind: SystemKind,
    config: WavePINNConfig,
    case_root: Path,
    force: bool,
) -> Dict[str, np.ndarray]:
    output_dir = wave_case_directory(case_root, params, kind)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_key = str(output_dir.resolve())
    if session_key in _SESSION_RESULTS:
        print(f"[{name}] reusing this condition from the current run")
        return _SESSION_RESULTS[session_key]
    result_path = output_dir / "results.npz"
    if result_path.exists() and not force and cache_matches(
        result_path, params, config, kind
    ):
        print(f"[{name}] using cached marching-PINN result: {result_path}")
        result = load_result(result_path)
        report = diagnostic_report(
            result, params, kind, config.initial_stress_mode
        )
        annotate_boundary_validation(report, kind, config)
        (output_dir / "diagnostics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        enforce_boundary_validation(name, report, kind, config, output_dir)
        _SESSION_RESULTS[session_key] = result
        return result
    if result_path.exists() and not force:
        print(f"[{name}] trainer revision/settings changed; retraining")

    device = select_device(config)
    speeds_np, right_np, left_np, scales_np = characteristic_data(params, kind)
    speeds = as_tensor(speeds_np, device)
    right_vectors = as_tensor(right_np, device)
    left_vectors = as_tensor(left_np, device)
    scales = as_tensor(scales_np, device)
    slab_count = actual_slab_count(params, kind, config)
    slab_edges = np.linspace(0.0, params.t_final_s, slab_count + 1)
    max_distance = float(np.max(np.abs(speeds_np))) * (slab_edges[1] - slab_edges[0])
    interface_count = (
        int(config.interface_points)
        if config.interface_points is not None
        else max(
            config.plot_z,
            min(2000, max(200, config.pde_points // 4)),
        )
    )
    if interface_count < config.plot_z:
        raise ValueError("interface_points cannot be smaller than plot_z")
    profile_z = as_tensor(
        np.linspace(0.0, params.length_m, interface_count), device
    )
    initial_q = as_tensor(
        initial_physical_state(
            params, kind, interface_count, config.initial_stress_mode
        ),
        device,
    )
    profile_w = physical_to_characteristic(initial_q, left_vectors, scales).detach()

    plot_z = np.linspace(0.0, params.length_m, config.plot_z)
    plot_t = np.linspace(0.0, params.t_final_s, config.plot_t)
    variable_names = ("V", "uz", "P", "sigma_z") if kind == "fssi" else ("V", "P")
    result_values = np.full(
        (config.plot_z, config.plot_t, len(variable_names)), np.nan, dtype=float
    )
    result_values[:, 0, :] = initial_physical_state(
        params, kind, config.plot_z, config.initial_stress_mode
    )
    history: list[Dict[str, float]] = []
    checkpoint_slabs = []
    global_step = 0
    rng = np.random.default_rng(config.seed + 2003)
    adam_allocations = [
        _allocate_steps(config.epochs, slab_count, index, minimum=1)
        for index in range(slab_count)
    ]
    lbfgs_allocations = [
        _allocate_steps(
            config.lbfgs_steps,
            slab_count,
            index,
            minimum=12 if config.lbfgs_steps > 0 else 0,
        )
        for index in range(slab_count)
    ]
    if kind == "two_equation":
        adam_allocations = [0] * slab_count
        lbfgs_allocations = [0] * slab_count

    print(
        f"[{name}] characteristic time-marching {kind} PINN on {device}; "
        f"labels=0, anchors=0, requested_slabs={config.causal_windows}, "
        f"actual_CFL_slabs={slab_count}"
    )
    print(
        f"[{name}] speeds="
        + ", ".join(f"{value:.2f}" for value in speeds_np)
        + f" m/s; max travel/slab={max_distance:.3f} m < L={params.length_m:g} m"
    )

    for slab in range(slab_count):
        t_start = float(slab_edges[slab])
        t_end = float(slab_edges[slab + 1])
        torch.manual_seed(config.seed + slab)
        model = CharacteristicBoundaryNetwork(
            len(speeds_np), t_start, t_end, config
        ).to(device)
        if kind == "fssi":
            prepare_valve_base_trajectory(
                model,
                profile_z,
                profile_w,
                speeds,
                right_vectors,
                scales,
                params,
                minimum_points=max(2 * config.boundary_points, 256),
            )
        adam_steps = adam_allocations[slab]
        lbfgs_steps = lbfgs_allocations[slab]
        times = sample_boundary_times(
            t_start, t_end, config.boundary_points, device, rng
        )
        initial_losses = boundary_losses(
            model,
            times,
            profile_z,
            profile_w,
            speeds,
            right_vectors,
            scales,
            params,
            kind,
            slab == 0,
        )
        balancer = GradientNormBalancer(initial_losses.keys(), config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        print(
            f"[{name}] slab {slab + 1:02d}/{slab_count}: "
            f"{t_start:.6f} <= t <= {t_end:.6f}, Adam={adam_steps}, "
            f"L-BFGS={lbfgs_steps}"
        )
        if kind == "two_equation":
            global_step += 1
            hard_loss = float(
                next(iter(initial_losses.values())).detach().cpu()
            )
            history.append(
                {
                    "global_step": float(global_step),
                    "slab": float(slab + 1),
                    "slab_start_s": t_start,
                    "slab_end_s": t_end,
                    "loss_total": hard_loss,
                    "optimizer_stage": "hard_boundary_projection",
                    "loss_bc_hard_projection": hard_loss,
                    "weight_bc_hard_projection": 1.0,
                }
            )
        for local_step in range(1, adam_steps + 1):
            if local_step > 1 and local_step % max(config.resample_every, 1) == 0:
                times = sample_boundary_times(
                    t_start, t_end, config.boundary_points, device, rng
                )
            optimizer.zero_grad(set_to_none=True)
            losses = boundary_losses(
                model,
                times,
                profile_z,
                profile_w,
                speeds,
                right_vectors,
                scales,
                params,
                kind,
                slab == 0,
            )
            if local_step == 1 or local_step % max(config.adaptive_every, 1) == 0:
                balancer.update(losses, model)
            total = balancer.total(losses)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            global_step += 1
            item: Dict[str, float] = {
                "global_step": float(global_step),
                "slab": float(slab + 1),
                "slab_start_s": t_start,
                "slab_end_s": t_end,
                "loss_total": float(total.detach().cpu()),
            }
            for loss_name, loss_value in losses.items():
                item[f"loss_{loss_name}"] = float(loss_value.detach().cpu())
                item[f"weight_{loss_name}"] = balancer.weights[loss_name]
            history.append(item)
            should_log = (
                local_step == 1
                or local_step == adam_steps
                or global_step % max(config.log_every, 1) == 0
            )
            if should_log:
                print(
                    f"[{name}] step={global_step:5d} slab={slab + 1:02d} "
                    f"local={local_step:4d}/{adam_steps} "
                    f"loss={item['loss_total']:.3e}"
                )

        if lbfgs_steps > 0:
            optimizer_lbfgs = torch.optim.LBFGS(
                model.parameters(),
                max_iter=lbfgs_steps,
                history_size=min(50, max(10, lbfgs_steps)),
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                optimizer_lbfgs.zero_grad(set_to_none=True)
                closure_losses = boundary_losses(
                    model,
                    times,
                    profile_z,
                    profile_w,
                    speeds,
                    right_vectors,
                    scales,
                    params,
                    kind,
                    slab == 0,
                )
                value = balancer.total(closure_losses)
                value.backward()
                return value

            optimizer_lbfgs.step(closure)
            final_losses = boundary_losses(
                model,
                times,
                profile_z,
                profile_w,
                speeds,
                right_vectors,
                scales,
                params,
                kind,
                slab == 0,
            )
            final_total = balancer.total(final_losses)
            lbfgs_item: Dict[str, float] = {
                "global_step": float(global_step) + 0.5,
                "slab": float(slab + 1),
                "slab_start_s": t_start,
                "slab_end_s": t_end,
                "loss_total": float(final_total.detach().cpu()),
                "optimizer_stage": "lbfgs",
            }
            for loss_name, loss_value in final_losses.items():
                lbfgs_item[f"loss_{loss_name}"] = float(
                    loss_value.detach().cpu()
                )
                lbfgs_item[f"weight_{loss_name}"] = balancer.weights[loss_name]
            history.append(lbfgs_item)
            print(
                f"[{name}] slab {slab + 1:02d} post-L-BFGS "
                f"loss={lbfgs_item['loss_total']:.3e}"
            )

        model.eval()
        time_mask = (plot_t > t_start) & (plot_t <= t_end + 1.0e-14)
        selected_times = plot_t[time_mask]
        if len(selected_times):
            zz, tt = np.meshgrid(plot_z, selected_times, indexing="ij")
            points = as_tensor(np.column_stack([zz.ravel(), tt.ravel()]), device)
            batches = []
            with torch.no_grad():
                for start in range(0, len(points), 8192):
                    batches.append(
                        physical_output(
                            model,
                            points[start : start + 8192],
                            profile_z,
                            profile_w,
                            speeds,
                            right_vectors,
                            scales,
                            params,
                            slab == 0,
                        )
                        .cpu()
                        .numpy()
                    )
            slab_values = np.vstack(batches).reshape(
                config.plot_z, len(selected_times), len(variable_names)
            )
            result_values[:, time_mask, :] = slab_values

        terminal_points = torch.column_stack(
            [
                profile_z,
                torch.full_like(profile_z, t_end),
            ]
        )
        with torch.no_grad():
            terminal_q = physical_output(
                model,
                terminal_points,
                profile_z,
                profile_w,
                speeds,
                right_vectors,
                scales,
                params,
                slab == 0,
            )
            next_profile_w = physical_to_characteristic(
                terminal_q, left_vectors, scales
            ).detach()
        checkpoint_slabs.append(
            {
                "t_start": t_start,
                "t_end": t_end,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "valve_base_time": model.valve_base_time.detach().cpu(),
                "valve_base_velocity": (
                    model.valve_base_velocity.detach().cpu()
                ),
                "initial_profile_w": profile_w.detach().cpu(),
            }
        )
        profile_w = next_profile_w

    if np.isnan(result_values).any():
        raise RuntimeError("Internal error: some plotting times were not assigned to a slab")
    result: Dict[str, np.ndarray] = {"z": plot_z, "t": plot_t}
    for index, variable_name in enumerate(variable_names):
        result[variable_name] = result_values[:, :, index]
    report = diagnostic_report(
        result, params, kind, config.initial_stress_mode
    )
    annotate_boundary_validation(report, kind, config)
    metadata = {
        "trainer": "characteristic_time_marching_pinn",
        "revision": WAVE_PINN_REVISION,
        "case_name": name,
        "system_kind": kind,
        "physical_parameters": asdict(params),
        "training_config": asdict(config),
        "requested_slabs": config.causal_windows,
        "actual_slabs": slab_count,
        "requested_total_lbfgs_steps": config.lbfgs_steps,
        "actual_total_lbfgs_steps": int(sum(lbfgs_allocations)),
        "characteristic_cfl": config.characteristic_cfl,
        "characteristic_speeds_m_s": speeds_np.tolist(),
        "output_scales": scales_np.tolist(),
        "interface_points": interface_count,
        "pde_enforcement": (
            "exact characteristic transport inside every slab; "
            "only physical boundary residuals are optimized"
        ),
        "interface_source": "previous slab terminal PINN prediction",
        "valve_model": (
            f"assumed cosine valve-relative-flow closure over "
            f"{params.valve_close_time_s:g} s "
            "plus M*uz_t-Af*P+At*sigma_z=0 right-end axial force balance "
            "as an explicit energy-consistent end-mass modelling assumption; "
            "paper Eq. (36) belongs to the separate impact experiment and is "
            "not used as the buried-pipe valve condition; "
            "unbounded neural axial-velocity correction and hard relative "
            "valve-flow projection V-uz=Vrel(t) "
            "with differential and exponential-step physics residuals; "
            "paper equations (37)-(39) are "
            "not used because the published K, n, opening history, and "
            "opening-to-flow relation are incomplete"
        ),
        "uses_anchor_points": False,
        "uses_labelled_data": False,
        "uses_reference_solver_during_training": False,
        "diagnostics": report,
    }
    np.savez_compressed(
        result_path, **result, metadata=json.dumps(metadata, ensure_ascii=True)
    )
    torch.save(
        {
            "metadata": metadata,
            "profile_z": profile_z.detach().cpu(),
            "slabs": checkpoint_slabs,
        },
        output_dir / "model.pt",
    )
    save_training_history(history, output_dir)
    save_case_summary(result, params, kind, output_dir / "case_summary.png")
    (output_dir / "diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"[{name}] completed: pressure std={report['valve_pressure_standard_deviation_pa']:.3e} Pa, "
        f"mean crossings={int(report['valve_pressure_mean_crossings'])}, "
        "closed-valve max|V-uz|="
        f"{report['closed_valve_relative_velocity_max_abs']:.3e} m/s"
    )
    enforce_boundary_validation(name, report, kind, config, output_dir)
    _SESSION_RESULTS[session_key] = result
    return result
