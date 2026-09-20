"""WP1 verification: exact four-equation benchmark and MOC convergence.

The Tijsseling benchmark is implemented independently from the compacted-soil
coefficient branch.  The WP0 research baseline is checked in a second chain
with the existing same-equation MOC solver.  No PINN labels are generated or
consumed by this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from water16_reproduction.common.physics import PhysicalParameters
from water16_reproduction.research_baseline import (
    DEFAULT_CONFIG as WP0_CONFIG,
    DEFAULT_PROVENANCE as WP0_PROVENANCE,
    load_config as load_wp0_config,
    validate_config as validate_wp0_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT / "PINN_FSSI_research_plan"
DEFAULT_BENCHMARK_CONFIG = (
    RESEARCH_ROOT / "configs" / "tijsseling_problem_a.json"
)
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "outputs" / "wp1"
STATE_NAMES = ("V", "uz", "P", "sigma_z")
STATE_UNITS = ("m/s", "m/s", "Pa", "Pa")


def load_benchmark_config(path: Path = DEFAULT_BENCHMARK_CONFIG) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "case_id",
        "status",
        "source",
        "state_order",
        "physical_parameters",
        "initial_conditions",
        "boundary_conditions",
        "paper_reference",
        "verification",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"benchmark config missing keys: {missing}")
    if data["schema_version"] != 1:
        raise ValueError("benchmark schema_version must be 1")
    if data["state_order"] != list(STATE_NAMES):
        raise ValueError(f"benchmark state_order must be {list(STATE_NAMES)}")
    if data["boundary_conditions"]["downstream_variant"] not in {
        "free_massless_closed_valve",
        "fixed_closed_valve",
    }:
        raise ValueError("unsupported downstream boundary variant")
    values = data["physical_parameters"]
    for name, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"benchmark parameter {name} must be finite numeric")
        if name != "pipe_nu" and float(value) <= 0.0:
            raise ValueError(f"benchmark parameter {name} must be positive")
    if not 0.0 <= float(values["pipe_nu"]) < 0.5:
        raise ValueError("benchmark pipe_nu must satisfy 0 <= nu < 0.5")
    return data


def tijsseling_matrices(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return A and B for A*q_t+B*q_x=0 in [V, uz, P, sigma_z] order."""

    p = config["physical_parameters"]
    radius = float(p["inner_radius_m"])
    thickness = float(p["wall_thickness_m"])
    young = float(p["pipe_E_pa"])
    poisson = float(p["pipe_nu"])
    rho_s = float(p["pipe_density_kg_m3"])
    rho_f = float(p["water_density_kg_m3"])
    bulk = float(p["water_bulk_modulus_pa"])
    compliance = 1.0 / bulk + 2.0 * radius / (young * thickness)
    coupling = poisson * radius / (young * thickness)
    matrix_a = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, rho_s, 0.0, 0.0],
            [0.0, 0.0, compliance, -2.0 * poisson / young],
            [0.0, 0.0, coupling, -1.0 / young],
        ],
        dtype=float,
    )
    matrix_b = np.array(
        [
            [0.0, 0.0, 1.0 / rho_f, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return matrix_a, matrix_b


def benchmark_areas(config: dict[str, Any]) -> tuple[float, float]:
    p = config["physical_parameters"]
    radius = float(p["inner_radius_m"])
    outer = radius + float(p["wall_thickness_m"])
    return math.pi * radius**2, math.pi * (outer**2 - radius**2)


def benchmark_characteristics(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return signed speeds and a numerically scaled characteristic basis."""

    matrix_a, matrix_b = tijsseling_matrices(config)
    advection = np.linalg.solve(matrix_a, matrix_b)
    p = config["physical_parameters"]
    rho_f = float(p["water_density_kg_m3"])
    initial_velocity = float(p["initial_velocity_m_s"])
    unscaled_speeds = np.linalg.eigvals(advection)
    fluid_speed = float(np.min(np.abs(unscaled_speeds)))
    structural_speed = float(np.max(np.abs(unscaled_speeds)))
    pressure_scale = rho_f * fluid_speed * initial_velocity
    axial_velocity_scale = pressure_scale / (
        float(p["pipe_density_kg_m3"]) * structural_speed
    )
    area_f, area_s = benchmark_areas(config)
    stress_scale = pressure_scale * area_f / area_s
    scales = np.array(
        [initial_velocity, axial_velocity_scale, pressure_scale, stress_scale],
        dtype=float,
    )
    scaled_advection = (
        advection * scales[np.newaxis, :] / scales[:, np.newaxis]
    )
    values, right = np.linalg.eig(scaled_advection)
    if np.max(np.abs(np.imag(values))) > 1.0e-8:
        raise ValueError("Tijsseling benchmark produced complex wave speeds")
    order = np.argsort(np.real(values))
    values = np.real(values[order])
    right = np.real(right[:, order])
    left = np.linalg.inv(right)
    return values, right, left, scales


def benchmark_initial_state(config: dict[str, Any]) -> np.ndarray:
    initial = config["initial_conditions"]
    return np.array(
        [initial["V"], initial["uz"], initial["P"], initial["sigma_z"]],
        dtype=float,
    )


def benchmark_boundaries(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = np.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float
    )
    left_target = np.zeros(2, dtype=float)
    if (
        config["boundary_conditions"]["downstream_variant"]
        == "free_massless_closed_valve"
    ):
        area_f, area_s = benchmark_areas(config)
        right = np.array(
            [[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, area_f, -area_s]],
            dtype=float,
        )
    else:
        right = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float
        )
    return left, left_target, right, np.zeros(2, dtype=float)


def _project_boundary(
    candidate: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    conditions: np.ndarray,
    targets: np.ndarray,
    physical_modes: np.ndarray,
) -> np.ndarray:
    amplitudes = np.asarray(candidate, dtype=float).copy()
    matrix = conditions @ physical_modes[:, incoming]
    rhs = targets - conditions @ physical_modes[:, outgoing] @ amplitudes[outgoing]
    amplitudes[incoming] = np.linalg.solve(matrix, rhs)
    return amplitudes


class ExactCharacteristicRecursion:
    """Resolution-independent characteristic recursion for the linear benchmark."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.length = float(config["physical_parameters"]["length_m"])
        self.speeds, self.right, self.left, self.scales = (
            benchmark_characteristics(config)
        )
        self.physical_modes = self.scales[:, None] * self.right
        self.initial_state = benchmark_initial_state(config)
        self.initial_amplitudes = self.left @ (self.initial_state / self.scales)
        self.positive = np.flatnonzero(self.speeds > 0.0)
        self.negative = np.flatnonzero(self.speeds < 0.0)
        (
            self.left_conditions,
            self.left_targets,
            self.right_conditions,
            self.right_targets,
        ) = benchmark_boundaries(config)
        if len(self.positive) != 2 or len(self.negative) != 2:
            raise ValueError("benchmark must have two characteristic modes each way")

    @staticmethod
    def _time_key(time: float) -> float:
        return round(max(float(time), 0.0), 13)

    def _trace_mode(self, mode: int, position: float, time: float) -> float:
        if time < -1.0e-13:
            return float(self.initial_amplitudes[mode])
        speed = float(self.speeds[mode])
        departure = position - speed * time
        tolerance = 1.0e-12 * max(self.length, 1.0)
        if tolerance < departure < self.length - tolerance:
            return float(self.initial_amplitudes[mode])
        if speed > 0.0:
            hit_time = time - position / speed
            if hit_time < -1.0e-13:
                return float(self.initial_amplitudes[mode])
            return float(self._boundary_amplitudes("left", self._time_key(hit_time))[mode])
        hit_time = time - (self.length - position) / (-speed)
        if hit_time < -1.0e-13:
            return float(self.initial_amplitudes[mode])
        return float(self._boundary_amplitudes("right", self._time_key(hit_time))[mode])

    @lru_cache(maxsize=None)
    def _boundary_amplitudes(self, side: str, time: float) -> tuple[float, ...]:
        if side == "left":
            incoming, outgoing = self.positive, self.negative
            conditions, targets = self.left_conditions, self.left_targets
            position = 0.0
        elif side == "right":
            incoming, outgoing = self.negative, self.positive
            conditions, targets = self.right_conditions, self.right_targets
            position = self.length
        else:
            raise ValueError(f"unknown boundary side {side!r}")
        amplitudes = self.initial_amplitudes.copy()
        for mode in outgoing:
            amplitudes[mode] = self._trace_mode(int(mode), position, time)
        projected = _project_boundary(
            amplitudes,
            incoming,
            outgoing,
            conditions,
            targets,
            self.physical_modes,
        )
        return tuple(float(value) for value in projected)

    def state(self, position: float, time: float) -> np.ndarray:
        if not 0.0 <= position <= self.length:
            raise ValueError("position must lie inside the pipe")
        if time < 0.0:
            raise ValueError("time cannot be negative")
        tolerance = 1.0e-12 * max(self.length, 1.0)
        if position <= tolerance:
            amplitudes = np.asarray(
                self._boundary_amplitudes("left", self._time_key(time))
            )
        elif position >= self.length - tolerance:
            amplitudes = np.asarray(
                self._boundary_amplitudes("right", self._time_key(time))
            )
        elif time == 0.0:
            amplitudes = self.initial_amplitudes
        else:
            amplitudes = np.array(
                [
                    self._trace_mode(mode, float(position), float(time))
                    for mode in range(4)
                ]
            )
        return self.physical_modes @ amplitudes

    def evaluate(self, positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        output = np.empty((len(positions), len(times), 4), dtype=float)
        for ix, position in enumerate(positions):
            for it, time in enumerate(times):
                output[ix, it] = self.state(float(position), float(time))
        return output


def solve_tijsseling_moc(
    config: dict[str, Any], output_times: np.ndarray, n_cells: int, cfl: float
) -> dict[str, np.ndarray]:
    """Semi-Lagrangian characteristic MOC with exact boundary projection."""

    if n_cells < 4:
        raise ValueError("n_cells must be at least four")
    if not 0.0 < cfl <= 1.0:
        raise ValueError("cfl must lie in (0, 1]")
    times = np.asarray(output_times, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("output_times must be one-dimensional with at least two points")
    if not np.isclose(times[0], 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("output_times must start at zero and strictly increase")

    length = float(config["physical_parameters"]["length_m"])
    speeds, right, left, scales = benchmark_characteristics(config)
    physical_modes = scales[:, None] * right
    positive = np.flatnonzero(speeds > 0.0)
    negative = np.flatnonzero(speeds < 0.0)
    left_c, left_t, right_c, right_t = benchmark_boundaries(config)
    positions = np.linspace(0.0, length, n_cells + 1)
    dx = length / n_cells
    maximum_dt = cfl * dx / float(np.max(np.abs(speeds)))

    initial_state = benchmark_initial_state(config)
    state = np.repeat(initial_state[None, :], n_cells + 1, axis=0)
    amplitudes = (state / scales) @ left.T
    amplitudes[0] = _project_boundary(
        amplitudes[0], positive, negative, left_c, left_t, physical_modes
    )
    amplitudes[-1] = _project_boundary(
        amplitudes[-1], negative, positive, right_c, right_t, physical_modes
    )
    state = amplitudes @ physical_modes.T
    snapshots = [state.copy()]
    current_time = 0.0

    for target_time in times[1:]:
        while current_time < target_time - 1.0e-14:
            dt = min(maximum_dt, float(target_time - current_time))
            updated = np.empty_like(amplitudes)
            for mode, speed in enumerate(speeds):
                departure = positions - speed * dt
                updated[:, mode] = np.interp(
                    departure,
                    positions,
                    amplitudes[:, mode],
                    left=amplitudes[0, mode],
                    right=amplitudes[-1, mode],
                )
            updated[0] = _project_boundary(
                updated[0], positive, negative, left_c, left_t, physical_modes
            )
            updated[-1] = _project_boundary(
                updated[-1], negative, positive, right_c, right_t, physical_modes
            )
            amplitudes = updated
            state = amplitudes @ physical_modes.T
            current_time += dt
        snapshots.append(state.copy())
    values = np.stack(snapshots, axis=0)
    return {
        "x": positions,
        "t": times.copy(),
        "V": values[:, :, 0],
        "uz": values[:, :, 1],
        "P": values[:, :, 2],
        "sigma_z": values[:, :, 3],
    }


def _sample_moc(
    result: dict[str, np.ndarray], positions: np.ndarray
) -> np.ndarray:
    sampled = np.empty((len(positions), len(result["t"]), 4), dtype=float)
    for state_index, name in enumerate(STATE_NAMES):
        for time_index in range(len(result["t"])):
            sampled[:, time_index, state_index] = np.interp(
                positions, result["x"], result[name][time_index]
            )
    return sampled


def _signal_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    times: np.ndarray,
    scale: float,
) -> dict[str, float | None]:
    error = prediction - reference
    relative_l2 = float(
        np.linalg.norm(error)
        / max(np.linalg.norm(reference), math.sqrt(len(reference)) * scale * 1.0e-12)
    )
    nrmse = float(np.sqrt(np.mean(error**2)) / max(scale, 1.0e-30))
    centered_reference = reference - np.mean(reference)
    centered_prediction = prediction - np.mean(prediction)
    if np.linalg.norm(centered_reference) <= scale * 1.0e-12:
        phase_lag = None
    else:
        correlation = np.correlate(
            centered_prediction, centered_reference, mode="full"
        )
        lag = int(np.argmax(correlation) - (len(reference) - 1))
        phase_lag = float(lag * np.mean(np.diff(times)))
    return {
        "relative_l2": relative_l2,
        "nrmse_by_physical_scale": nrmse,
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "peak_error": float(np.max(prediction) - np.max(reference)),
        "trough_error": float(np.min(prediction) - np.min(reference)),
        "phase_lag_s": phase_lag,
    }


def _state_scales(config: dict[str, Any]) -> np.ndarray:
    return benchmark_characteristics(config)[3]


def _first_arrival(
    signal: np.ndarray, times: np.ndarray, initial: float, scale: float
) -> float | None:
    indices = np.flatnonzero(np.abs(signal - initial) > 1.0e-5 * scale)
    return None if len(indices) == 0 else float(times[indices[0]])


def _observed_orders(errors: list[float]) -> list[float | None]:
    orders: list[float | None] = [None]
    for coarse, fine in zip(errors[:-1], errors[1:]):
        if coarse > 0.0 and fine > 0.0:
            orders.append(float(math.log(coarse / fine, 2.0)))
        else:
            orders.append(None)
    return orders


def run_tijsseling_verification(
    config: dict[str, Any], output_dir: Path, quick: bool
) -> dict[str, Any]:
    verification = config["verification"]
    final_time = float(verification["exact_recursion_t_final_s"])
    point_count = int(verification["exact_output_points"])
    cells = [int(value) for value in verification["moc_cells"]]
    if quick:
        point_count = min(point_count, 81)
        cells = cells[:3]
    times = np.linspace(0.0, final_time, point_count)
    length = float(config["physical_parameters"]["length_m"])
    positions = length * np.asarray(
        verification["observation_positions_over_L"], dtype=float
    )
    exact_solver = ExactCharacteristicRecursion(config)
    exact = exact_solver.evaluate(positions, times)
    scales = _state_scales(config)

    rows: list[dict[str, Any]] = []
    moc_results: dict[int, dict[str, np.ndarray]] = {}
    for n_cells in cells:
        result = solve_tijsseling_moc(
            config, times, n_cells=n_cells, cfl=float(verification["moc_cfl"])
        )
        moc_results[n_cells] = result
        sampled = _sample_moc(result, positions)
        for position_index, position in enumerate(positions):
            for state_index, state_name in enumerate(STATE_NAMES):
                metrics = _signal_metrics(
                    exact[position_index, :, state_index],
                    sampled[position_index, :, state_index],
                    times,
                    max(
                        float(scales[state_index]),
                        float(
                            np.max(
                                np.abs(exact[position_index, :, state_index])
                            )
                        ),
                        float(np.ptp(exact[position_index, :, state_index])),
                        1.0e-30,
                    ),
                )
                rows.append(
                    {
                        "n_cells": n_cells,
                        "dx_m": length / n_cells,
                        "position_m": float(position),
                        "position_over_L": float(position / length),
                        "state": state_name,
                        **metrics,
                    }
                )

    for position in positions:
        for state_name in STATE_NAMES:
            selected = [
                row
                for row in rows
                if row["position_m"] == float(position)
                and row["state"] == state_name
            ]
            orders = _observed_orders(
                [float(row["nrmse_by_physical_scale"]) for row in selected]
            )
            for row, order in zip(selected, orders):
                row["observed_order_from_previous_grid"] = order

    _write_csv(output_dir / "tijsseling_moc_convergence.csv", rows)
    finest = cells[-1]
    finest_sampled = _sample_moc(moc_results[finest], positions)
    np.savez_compressed(
        output_dir / "tijsseling_exact_reference.npz",
        t=times,
        positions_m=positions,
        state_names=np.asarray(STATE_NAMES),
        exact=exact,
        finest_moc=finest_sampled,
        finest_moc_cells=np.asarray(finest),
    )
    _plot_exact_comparison(
        output_dir / "tijsseling_exact_vs_moc.png",
        times,
        exact,
        finest_sampled,
        positions,
        finest,
    )
    _plot_tijsseling_convergence(
        output_dir / "tijsseling_moc_convergence.png", rows, positions[-1]
    )

    speeds = exact_solver.speeds
    positive = np.sort(speeds[speeds > 0.0])
    paper = config["paper_reference"]
    speed_errors = {
        "fluid_m_s": float(
            positive[0] - float(paper["fluid_characteristic_speed_m_s"])
        ),
        "structural_m_s": float(
            positive[1] - float(paper["structural_characteristic_speed_m_s"])
        ),
    }
    left_c, left_t, right_c, right_t = benchmark_boundaries(config)
    left_states = exact_solver.evaluate(np.array([0.0]), times)[0]
    right_states = exact_solver.evaluate(np.array([length]), times)[0]
    area_f, area_s = benchmark_areas(config)
    boundary_scale = np.array(
        [scales[2], scales[1], scales[0], area_f * scales[2]], dtype=float
    )
    left_residual = left_states @ left_c.T - left_t
    right_residual = right_states @ right_c.T - right_t
    normalized_boundary_residual = max(
        float(np.max(np.abs(left_residual[:, 0])) / boundary_scale[0]),
        float(np.max(np.abs(left_residual[:, 1])) / boundary_scale[1]),
        float(np.max(np.abs(right_residual[:, 0])) / boundary_scale[2]),
        float(np.max(np.abs(right_residual[:, 1])) / boundary_scale[3]),
    )
    midpoint_exact = exact[0]
    midpoint_moc = finest_sampled[0]
    exact_arrival = _first_arrival(
        midpoint_exact[:, 2], times, midpoint_exact[0, 2], scales[2]
    )
    moc_arrival = _first_arrival(
        midpoint_moc[:, 2], times, midpoint_moc[0, 2], scales[2]
    )
    finest_rows = [row for row in rows if row["n_cells"] == finest]
    coarsest_rows = [row for row in rows if row["n_cells"] == cells[0]]
    improvement = {
        f"xL_{row['position_over_L']:g}_{row['state']}": (
            float(row["nrmse_by_physical_scale"])
            < float(
                next(
                    candidate["nrmse_by_physical_scale"]
                    for candidate in coarsest_rows
                    if candidate["position_m"] == row["position_m"]
                    and candidate["state"] == row["state"]
                )
            )
        )
        for row in finest_rows
    }
    return {
        "case_id": config["case_id"],
        "exact_method": (
            "resolution-independent recursive tracing of Riemann invariants "
            "with exact linear boundary projection"
        ),
        "verification_window_s": final_time,
        "moc_cells": cells,
        "computed_signed_characteristic_speeds_m_s": speeds.tolist(),
        "paper_positive_characteristic_speeds_m_s": [
            paper["fluid_characteristic_speed_m_s"],
            paper["structural_characteristic_speed_m_s"],
        ],
        "wave_speed_errors_m_s": speed_errors,
        "maximum_normalized_exact_boundary_residual": normalized_boundary_residual,
        "midpoint_pressure_first_arrival_s": {
            "exact": exact_arrival,
            "finest_moc": moc_arrival,
            "absolute_error": (
                None
                if exact_arrival is None or moc_arrival is None
                else abs(moc_arrival - exact_arrival)
            ),
        },
        "finest_grid_metrics": finest_rows,
        "finest_improves_over_coarsest": improvement,
        "all_monitored_signals_improve": all(improvement.values()),
        "reference_periods_s": {
            "four_L_over_fluid_speed": 4.0 * length / positive[0],
            "four_L_over_structural_speed": 4.0 * length / positive[1],
        },
    }


def _wp0_sample(
    result: dict[str, np.ndarray], position: float, state_name: str
) -> np.ndarray:
    index = int(np.argmin(np.abs(result["z"] - position)))
    return np.asarray(result[state_name][index], dtype=float)


def run_research_baseline_convergence(
    output_dir: Path, quick: bool
) -> dict[str, Any]:
    from water16_reproduction.figure09.pinn_compare import (
        solve_matched_fssi_reference,
    )

    wp0 = load_wp0_config(WP0_CONFIG)
    params = validate_wp0_config(wp0, WP0_PROVENANCE)
    final_time = min(0.20, params.t_final_s)
    output_points = 201 if not quick else 81
    times = np.linspace(0.0, final_time, output_points)
    cells = [80, 160, 320, int(wp0["reference_solver"]["n_cells"])]
    if quick:
        cells = cells[:3]
    results = {
        n_cells: solve_matched_fssi_reference(
            params,
            times,
            n_cells=n_cells,
            cfl=float(wp0["reference_solver"]["cfl"]),
        )
        for n_cells in cells
    }
    finest = cells[-1]
    reference = results[finest]
    positions = (0.5 * params.length_m, params.length_m)
    rows: list[dict[str, Any]] = []
    for n_cells in cells[:-1]:
        for position in positions:
            for state_name in STATE_NAMES:
                reference_signal = _wp0_sample(reference, position, state_name)
                signal = _wp0_sample(results[n_cells], position, state_name)
                scale = max(
                    float(np.ptp(reference_signal)),
                    float(np.max(np.abs(reference_signal))),
                    1.0e-12,
                )
                rows.append(
                    {
                        "n_cells": n_cells,
                        "reference_cells": finest,
                        "position_m": position,
                        "position_over_L": position / params.length_m,
                        "state": state_name,
                        **_signal_metrics(reference_signal, signal, times, scale),
                    }
                )
    _write_csv(output_dir / "research_baseline_moc_convergence.csv", rows)
    _plot_research_convergence(
        output_dir / "research_baseline_moc_convergence.png",
        times,
        results,
        params.length_m,
    )
    pressure_rows = [
        row
        for row in rows
        if row["position_over_L"] == 1.0 and row["state"] == "P"
    ]
    return {
        "case_id": wp0["case_id"],
        "config_sha256_source": "see wp0_baseline_audit.json",
        "verification_window_s": final_time,
        "moc_cells": cells,
        "reference_cells": finest,
        "reference_role": (
            "same-equation numerical reference for the current FSSI closure; "
            "not an exact solution or experimental validation"
        ),
        "valve_pressure_metrics_against_finest": pressure_rows,
        "coarser_pressure_error_decreases": all(
            float(fine["nrmse_by_physical_scale"])
            < float(coarse["nrmse_by_physical_scale"])
            for coarse, fine in zip(pressure_rows[:-1], pressure_rows[1:])
        )
        if len(pressure_rows) > 1
        else True,
    }


def run_pinn_smoke(output_dir: Path) -> dict[str, Any]:
    """Run a deliberately tiny integration smoke test, not an accuracy study."""

    from water16_reproduction.wave_marching_core import (
        WavePINNConfig,
        train_wave_case,
    )

    wp0 = load_wp0_config(WP0_CONFIG)
    params = validate_wp0_config(wp0, WP0_PROVENANCE)
    # Include the 0.03 s valve-closure instant so the trainer can evaluate its
    # downstream closed-valve diagnostic instead of reporting an empty mask.
    smoke_params = replace(params, t_final_s=min(0.035, params.t_final_s))
    config = WavePINNConfig(
        epochs=2,
        lbfgs_steps=0,
        hidden_width=12,
        hidden_layers=1,
        pde_points=24,
        boundary_points=12,
        learning_rate=5.0e-4,
        causal_windows=1,
        seed=int(wp0["reproducibility"]["seed"]),
        dtype="float64",
        device="cpu",
        plot_z=12,
        plot_t=21,
    )
    result = train_wave_case(
        "wp1_research_baseline_smoke",
        smoke_params,
        "fssi",
        config,
        output_dir / "pinn_smoke",
        force=True,
    )
    return {
        "status": "interface_pass",
        "epochs": config.epochs,
        "t_final_s": smoke_params.t_final_s,
        "output_shape": {
            name: list(np.asarray(result[name]).shape)
            for name in ("V", "uz", "P", "sigma_z")
        },
        "interpretation": (
            "code-path smoke test only; two epochs cannot establish PINN accuracy"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_exact_comparison(
    path: Path,
    times: np.ndarray,
    exact: np.ndarray,
    moc: np.ndarray,
    positions: np.ndarray,
    n_cells: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for state_index, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
        axis = axes.flat[state_index]
        for position_index, position in enumerate(positions):
            suffix = f"x={position:g} m"
            axis.plot(
                times,
                exact[position_index, :, state_index],
                linewidth=1.5,
                label=f"exact, {suffix}",
            )
            axis.plot(
                times,
                moc[position_index, :, state_index],
                "--",
                linewidth=1.0,
                label=f"MOC N={n_cells}, {suffix}",
            )
        axis.set_title(name)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Tijsseling Problem A: exact recursion vs MOC", y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_tijsseling_convergence(
    path: Path, rows: list[dict[str, Any]], valve_position: float
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for state_name in STATE_NAMES:
        selected = [
            row
            for row in rows
            if row["position_m"] == float(valve_position)
            and row["state"] == state_name
        ]
        axis.loglog(
            [row["dx_m"] for row in selected],
            [row["nrmse_by_physical_scale"] for row in selected],
            "o-",
            label=state_name,
        )
    axis.invert_xaxis()
    axis.set_xlabel("grid spacing dx (m)")
    axis.set_ylabel("NRMSE by physical scale")
    axis.set_title("Tijsseling Problem A MOC convergence at valve")
    axis.grid(which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_research_convergence(
    path: Path,
    times: np.ndarray,
    results: dict[int, dict[str, np.ndarray]],
    valve_position: float,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for state_index, (state_name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
        axis = axes.flat[state_index]
        for n_cells, result in results.items():
            axis.plot(
                times,
                _wp0_sample(result, valve_position, state_name),
                linewidth=1.0,
                label=f"N={n_cells}",
            )
        axis.set_title(state_name)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        "WP0 research baseline: same-equation MOC grid convergence", y=0.99
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(results),
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_wp1(
    benchmark_config_path: Path,
    output_dir: Path,
    quick: bool = False,
    skip_research_baseline: bool = False,
    pinn_smoke: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark_config(benchmark_config_path)
    tijsseling = run_tijsseling_verification(benchmark, output_dir, quick)
    tolerance = float(
        benchmark["verification"]["wave_speed_absolute_tolerance_m_s"]
    )
    speed_pass = all(
        abs(float(value)) <= tolerance
        for value in tijsseling["wave_speed_errors_m_s"].values()
    )
    exact_boundary_pass = (
        tijsseling["maximum_normalized_exact_boundary_residual"] <= 1.0e-10
    )
    research = (
        None
        if skip_research_baseline
        else run_research_baseline_convergence(output_dir, quick)
    )
    smoke = run_pinn_smoke(output_dir) if pinn_smoke else {"status": "not_run"}
    deterministic_pass = (
        speed_pass
        and exact_boundary_pass
        and tijsseling["all_monitored_signals_improve"]
        and (
            research is None
            or bool(research["coarser_pressure_error_decreases"])
        )
    )
    report = {
        "status": "pass" if deterministic_pass else "review_required",
        "scope": (
            "WP1 deterministic verification and optional PINN interface smoke; "
            "not full PINN training and not experimental validation"
        ),
        "tijsseling_problem_a": tijsseling,
        "wp0_research_baseline": research,
        "pinn_smoke": smoke,
        "acceptance": {
            "published_wave_speeds_within_tolerance": speed_pass,
            "exact_boundary_projection": exact_boundary_pass,
            "tijsseling_moc_refines_towards_exact": tijsseling[
                "all_monitored_signals_improve"
            ],
            "research_baseline_pressure_refines_towards_finest": (
                None
                if research is None
                else research["coarser_pressure_error_decreases"]
            ),
            "pinn_accuracy": (
                "not assessed; smoke test is code-path only"
                if pinn_smoke
                else "not run in WP1 deterministic stage"
            ),
        },
        "limitations": [
            "The exact recursion window is deliberately short because the number of boundary recursion branches grows rapidly with time.",
            "Tijsseling Problem A verifies the classical no-soil four-equation system; it does not validate the compacted-soil closure.",
            "The WP0 MOC chain is a same-equation numerical cross-check, not an independent experiment.",
        ],
    }
    (output_dir / "wp1_verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WP1 exact four-equation and MOC verification"
    )
    parser.add_argument("--benchmark-config", type=Path, default=DEFAULT_BENCHMARK_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-research-baseline", action="store_true")
    parser.add_argument(
        "--pinn-smoke",
        action="store_true",
        help="run a two-epoch code-path smoke test; this is not an accuracy run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_wp1(
        args.benchmark_config.resolve(),
        args.output_dir.resolve(),
        quick=args.quick,
        skip_research_baseline=args.skip_research_baseline,
        pinn_smoke=args.pinn_smoke,
    )
    summary = {
        "status": report["status"],
        "wave_speed_check": report["acceptance"][
            "published_wave_speeds_within_tolerance"
        ],
        "exact_boundary_check": report["acceptance"]["exact_boundary_projection"],
        "tijsseling_moc_convergence": report["acceptance"][
            "tijsseling_moc_refines_towards_exact"
        ],
        "research_baseline_convergence": report["acceptance"][
            "research_baseline_pressure_refines_towards_finest"
        ],
        "pinn_smoke": report["pinn_smoke"]["status"],
        "output": str((args.output_dir / "wp1_verification_report.json").resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
