"""Governing equations and deterministic reference solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PhysicalParameters:
    length_m: float = 100.0
    inner_radius_m: float = 0.5
    wall_thickness_m: float = 0.05
    pipe_E_pa: float = 200.0e9
    pipe_nu: float = 0.30
    pipe_density_kg_m3: float = 7800.0
    water_density_kg_m3: float = 1000.0
    water_bulk_modulus_pa: float = 2.20e9
    soil_E_pa: float = 20.0e9
    gravity_m_s2: float = 9.80665
    head_difference_m: float = 4.0
    initial_velocity_m_s: float = 0.060
    valve_mass_kg: float = 100.0
    valve_close_time_s: float = 0.0
    t_final_s: float = 0.8
    coefficient_model: str = "printed_equations"
    # Optional one-way radial-compliance closure used only by the independently
    # registered Cao (2021) external-validation branch.  The default research
    # model leaves this unset and is therefore unchanged.
    radial_compliance_m_pa: float | None = None


@dataclass
class SolverSettings:
    n_cells: int = 500
    dt_s: float = 1.0e-4
    output_stride: int = 5
    use_valve_mass_boundary: bool = True
    max_steps: int = 200000


def soil_ratio(params: PhysicalParameters) -> float:
    return params.soil_E_pa / params.pipe_E_pa


def with_soil_ratio(params: PhysicalParameters, ratio: float) -> PhysicalParameters:
    return replace(params, soil_E_pa=ratio * params.pipe_E_pa)


def initial_pressure_pa(params: PhysicalParameters) -> float:
    return params.water_density_kg_m3 * params.gravity_m_s2 * params.head_difference_m


def cross_section_areas(params: PhysicalParameters) -> Tuple[float, float]:
    outer = params.inner_radius_m + params.wall_thickness_m
    fluid = math.pi * params.inner_radius_m**2
    pipe = math.pi * (outer**2 - params.inner_radius_m**2)
    return fluid, pipe


def pipe_soil_lambdas(params: PhysicalParameters) -> Tuple[float, float]:
    ratio = soil_ratio(params)
    if params.coefficient_model == "paper_reported":
        if ratio <= 0.0:
            return 0.0, 0.0
        value = 1.0 / (1.0 + math.exp(-2.4 * (math.log10(ratio) + 2.25)))
        return value, 0.0
    if params.coefficient_model != "printed_equations":
        raise ValueError("coefficient_model must be printed_equations or paper_reported")

    inner = params.inner_radius_m
    outer = inner + params.wall_thickness_m
    nu = params.pipe_nu
    cp = 2.0 / ((outer**2 / inner**2) - 1.0)
    cb = (1.0 + inner**2 / outer**2) / (1.0 - inner**2 / outer**2)
    denominator = 1.0 + ratio * (-1.0 + nu * cb)
    if abs(denominator) < 1.0e-12:
        raise ValueError("pipe-soil lambda denominator is nearly zero")
    return ratio * nu * cp / denominator, ratio * nu / denominator


def fssi_coefficients(params: PhysicalParameters) -> Dict[str, float]:
    if params.coefficient_model == "cao2021_radial_compliance":
        compliance = params.radial_compliance_m_pa
        if compliance is None or not math.isfinite(compliance) or compliance < 0.0:
            raise ValueError(
                "cao2021_radial_compliance requires a finite non-negative "
                "radial_compliance_m_pa"
            )
        # Cao's one-way benchmark has u_r = C_r P.  In the four-state
        # continuity equation u_r/R = m P + n sigma_z, hence m=C_r/R and
        # n=h=k=0 recover exactly that registered reduced closure while the
        # Feature-causal PINN still transports all four state variables.
        return {
            "lambda1": 0.0,
            "lambda2": 0.0,
            "m": compliance / params.inner_radius_m,
            "n": 0.0,
            "h": 0.0,
            "k": 0.0,
        }
    lambda1, lambda2 = pipe_soil_lambdas(params)
    if params.coefficient_model == "paper_reported":
        if soil_ratio(params) <= 0.0:
            log_ratio = -8.0
        else:
            log_ratio = math.log10(soil_ratio(params))

        def transition(center: float, slope: float) -> float:
            return 1.0 / (1.0 + math.exp(-slope * (log_ratio - center)))

        # Figure 11 contains four visibly different transition curves.  The
        # previous implementation incorrectly reused one logistic curve for
        # every coefficient and could not reproduce the 2m/2n crossover.
        transition_2m = transition(-1.92, 2.50)
        transition_2n = transition(-2.30, 2.50)
        transition_h = transition(-2.35, 2.55)
        transition_k = transition(-2.20, 2.75)
        return {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "m": 0.5
            * (
                (1.0 - transition_2m) * 2.35e-11
                + transition_2m * 5.36e-15
            ),
            "n": 0.5
            * (
                (1.0 - transition_2n) * 2.50e-11
                + transition_2n * 1.36e-13
            ),
            "h": -0.0938 * transition_h,
            "k": -0.0953 * transition_k,
        }

    rho_f = params.water_density_kg_m3
    rho_t = params.pipe_density_kg_m3
    denominator = (1.0 - lambda1) * rho_f + 2.0 * rho_t
    factor = 2.0 * rho_t / denominator
    radius_over_e = params.inner_radius_m / params.wall_thickness_m
    m = radius_over_e / params.pipe_E_pa * (1.0 - lambda1) * factor
    n = radius_over_e / params.pipe_E_pa * lambda2 * factor - params.pipe_nu / params.pipe_E_pa
    h = radius_over_e * (1.0 - lambda1) * factor
    k = radius_over_e * lambda2 * factor
    return {"lambda1": lambda1, "lambda2": lambda2, "m": m, "n": n, "h": h, "k": k}


def system_matrices(params: PhysicalParameters) -> Tuple[np.ndarray, np.ndarray]:
    coeffs = fssi_coefficients(params)
    rho_f = params.water_density_kg_m3
    rho_t = params.pipe_density_kg_m3
    E = params.pipe_E_pa
    nu = params.pipe_nu
    a = 1.0 / params.water_bulk_modulus_pa + 2.0 * coeffs["m"]
    c = 1.0 / E - nu * coeffs["k"] / E
    d = nu * coeffs["h"] / E
    matrix_a = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, rho_t, 0.0, 0.0],
         [0.0, 0.0, a, 2.0 * coeffs["n"]], [0.0, 0.0, d, -c]],
        dtype=float,
    )
    matrix_b = np.array(
        [[0.0, 0.0, 1.0 / rho_f, 0.0], [0.0, 0.0, 0.0, -1.0],
         [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=float,
    )
    return matrix_a, matrix_b


def characteristic_speeds(params: PhysicalParameters) -> np.ndarray:
    matrix_a, matrix_b = system_matrices(params)
    eigenvalues = np.linalg.eigvals(np.linalg.solve(matrix_a, matrix_b))
    real = np.real(eigenvalues[np.isclose(np.imag(eigenvalues), 0.0, atol=1.0e-8)])
    return np.sort(np.abs(real[np.abs(real) > 1.0e-9]))


def pressure_wave_speed(params: PhysicalParameters) -> float:
    return float(np.min(characteristic_speeds(params)))


def state_scales(params: PhysicalParameters) -> np.ndarray:
    pressure = max(
        initial_pressure_pa(params),
        params.water_density_kg_m3 * pressure_wave_speed(params) * abs(params.initial_velocity_m_s),
        1.0,
    )
    structure_speed = math.sqrt(params.pipe_E_pa / params.pipe_density_kg_m3)
    uz = max(pressure / (params.pipe_density_kg_m3 * structure_speed), 1.0e-6)
    area_f, area_t = cross_section_areas(params)
    sigma = pressure * area_f / area_t
    return np.array([max(abs(params.initial_velocity_m_s), 1.0e-3), uz, pressure, sigma])


def characteristic_basis(params: PhysicalParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix_a, matrix_b = system_matrices(params)
    advection = np.linalg.solve(matrix_a, matrix_b)
    scales = state_scales(params)
    scaled = (advection * scales[np.newaxis, :]) / scales[:, np.newaxis]
    values, right = np.linalg.eig(scaled)
    if np.max(np.abs(np.imag(values))) > 1.0e-7:
        raise ValueError("complex characteristic speeds")
    order = np.argsort(np.real(values))
    values = np.real(values[order])
    right = np.real(right[:, order])
    return values, right, np.linalg.inv(right), scales


def project_boundary_state(
    interior: np.ndarray,
    side: str,
    conditions: np.ndarray,
    targets: np.ndarray,
    speeds: np.ndarray,
    right: np.ndarray,
    left: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    amplitudes = left @ (interior / scales)
    incoming = np.flatnonzero(speeds > 0.0) if side == "left" else np.flatnonzero(speeds < 0.0)
    outgoing = np.flatnonzero(speeds <= 0.0) if side == "left" else np.flatnonzero(speeds >= 0.0)
    physical_right = scales[:, None] * right
    matrix = conditions @ physical_right[:, incoming]
    rhs = targets - conditions @ physical_right[:, outgoing] @ amplitudes[outgoing]
    amplitudes[incoming] = np.linalg.solve(matrix, rhs)
    return scales * (right @ amplitudes)


def solve_fssi(params: PhysicalParameters, settings: SolverSettings) -> Dict[str, np.ndarray]:
    speeds, right, left, scales = characteristic_basis(params)
    n = settings.n_cells
    dz = params.length_m / n
    steps = min(int(math.ceil(params.t_final_s / settings.dt_s)), settings.max_steps)
    dt = params.t_final_s / max(steps, 1)
    z = (np.arange(n) + 0.5) * dz
    state = np.zeros((n, 4), dtype=float)
    state[:, 0] = params.initial_velocity_m_s
    state[:, 2] = initial_pressure_pa(params)
    p0 = initial_pressure_pa(params)
    area_f, area_t = cross_section_areas(params)
    valve_uz_old = 0.0
    times: List[float] = []
    snapshots: List[np.ndarray] = []

    for step in range(steps + 1):
        time = step * dt
        if step % max(settings.output_stride, 1) == 0 or step == steps:
            times.append(time)
            snapshots.append(state.copy())
        if step == steps:
            break

        left_state = project_boundary_state(
            state[0], "left", np.array([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=float),
            np.array([p0, 0.0]), speeds, right, left, scales,
        )
        if settings.use_valve_mass_boundary:
            mass = params.valve_mass_kg
            conditions = np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, mass / dt, area_f, -area_t]], dtype=float
            )
            targets = np.array([0.0, mass * valve_uz_old / dt])
        else:
            conditions = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
            targets = np.array([0.0, 0.0])
        right_state = project_boundary_state(
            state[-1], "right", conditions, targets, speeds, right, left, scales
        )
        valve_uz_old = float(right_state[1])

        amplitudes = (state / scales) @ left.T
        left_amp = left @ (left_state / scales)
        right_amp = left @ (right_state / scales)
        updated = np.empty_like(amplitudes)
        for mode, speed in enumerate(speeds):
            departure = z - speed * dt
            updated[:, mode] = np.interp(
                departure, z, amplitudes[:, mode], left=left_amp[mode], right=right_amp[mode]
            )
        state = (updated @ right.T) * scales

    output = np.asarray(snapshots)
    return {
        "z": z,
        "t": np.asarray(times),
        "V": output[:, :, 0],
        "uz": output[:, :, 1],
        "P": output[:, :, 2],
        "sigma_z": output[:, :, 3],
    }


def solve_two_equation(params: PhysicalParameters, settings: SolverSettings) -> Dict[str, np.ndarray]:
    rho = params.water_density_kg_m3
    speed = math.sqrt(params.water_bulk_modulus_pa / rho)
    n = settings.n_cells
    dz = params.length_m / n
    steps = min(int(math.ceil(params.t_final_s / settings.dt_s)), settings.max_steps)
    dt = params.t_final_s / max(steps, 1)
    z = (np.arange(n) + 0.5) * dz
    state = np.zeros((n, 2), dtype=float)
    state[:, 0] = params.initial_velocity_m_s
    state[:, 1] = initial_pressure_pa(params)
    scales = np.array([max(abs(params.initial_velocity_m_s), 1e-3), rho * speed * abs(params.initial_velocity_m_s)])
    matrix = np.array([[0.0, 1.0 / rho], [rho * speed**2, 0.0]])
    scaled = (matrix * scales[None, :]) / scales[:, None]
    values, right = np.linalg.eig(scaled)
    order = np.argsort(np.real(values))
    values = np.real(values[order])
    right = np.real(right[:, order])
    left = np.linalg.inv(right)
    times: List[float] = []
    snapshots: List[np.ndarray] = []
    for step in range(steps + 1):
        time = step * dt
        if step % max(settings.output_stride, 1) == 0 or step == steps:
            times.append(time)
            snapshots.append(state.copy())
        if step == steps:
            break
        left_state = project_boundary_state(
            state[0], "left", np.array([[0.0, 1.0]]), np.array([initial_pressure_pa(params)]),
            values, right, left, scales,
        )
        right_state = project_boundary_state(
            state[-1], "right", np.array([[1.0, 0.0]]), np.array([0.0]),
            values, right, left, scales,
        )
        amplitudes = (state / scales) @ left.T
        updated = np.empty_like(amplitudes)
        for mode, mode_speed in enumerate(values):
            updated[:, mode] = np.interp(
                z - mode_speed * dt,
                z,
                amplitudes[:, mode],
                left=(left @ (left_state / scales))[mode],
                right=(left @ (right_state / scales))[mode],
            )
        state = (updated @ right.T) * scales
    output = np.asarray(snapshots)
    return {"z": z, "t": np.asarray(times), "V": output[:, :, 0], "P": output[:, :, 1]}


def radial_velocity(params: PhysicalParameters, t: np.ndarray, pressure: np.ndarray, stress: np.ndarray) -> np.ndarray:
    coeffs = fssi_coefficients(params)
    return params.inner_radius_m * (
        coeffs["m"] * np.gradient(pressure, t) + coeffs["n"] * np.gradient(stress, t)
    )


def spectrum(t: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centered = values - np.mean(values)
    window = np.hanning(len(centered))
    amplitude = 2.0 * np.abs(np.fft.rfft(centered * window)) / max(np.sum(window), 1.0)
    frequency = np.fft.rfftfreq(len(centered), float(np.mean(np.diff(t))))
    return frequency, amplitude
