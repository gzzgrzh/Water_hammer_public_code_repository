"""Figure 9 characteristic-causal PINN reproduction and paper comparison.

Both branches solve the complete four-equation FSSI system.  The Es/E=0
branch removes only pipe-soil coupling; Poisson and junction coupling remain.
No digitised paper values or deterministic-solver labels are used in training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from ..common.paper import extract_paper_figure, save_comparison
from ..common.physics import (
    PhysicalParameters,
    characteristic_basis,
    characteristic_speeds,
    cross_section_areas,
    fssi_coefficients,
    initial_pressure_pa,
    with_soil_ratio,
)
from ..wave_marching_core import (
    WAVE_PINN_REVISION,
    add_training_arguments,
    config_from_args,
    train_wave_case,
    wave_case_directory,
)


def figure09_parameters(
    inner_radius_m: float = 0.5,
    initial_velocity_m_s: float = 0.060,
    valve_close_time_s: float = 0.03,
) -> tuple[PhysicalParameters, PhysicalParameters]:
    """Return the two canonical Fig. 9 cases.

    The return values differ only in soil modulus.  Keeping this construction
    in one testable function prevents accidental changes to mass, Poisson
    ratio, valve law, geometry, or coefficient branch between the two PINNs.
    """

    base = replace(
        PhysicalParameters(),
        inner_radius_m=inner_radius_m,
        wall_thickness_m=0.1 * inner_radius_m,
        coefficient_model="printed_equations",
        initial_velocity_m_s=initial_velocity_m_s,
        valve_close_time_s=valve_close_time_s,
        valve_mass_kg=100.0,
        pipe_nu=0.30,
    )
    return with_soil_ratio(base, 0.0), with_soil_ratio(base, 0.1)


def _project_reference_boundary(
    transported_amplitudes: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    conditions: np.ndarray,
    targets: np.ndarray,
    physical_modes: np.ndarray,
) -> np.ndarray:
    """Project transported characteristic amplitudes onto two boundary laws."""

    amplitudes = transported_amplitudes.copy()
    matrix = conditions @ physical_modes[:, incoming]
    rhs = (
        targets
        - conditions
        @ physical_modes[:, outgoing]
        @ amplitudes[outgoing]
    )
    amplitudes[incoming] = np.linalg.solve(matrix, rhs)
    return amplitudes


def matched_valve_relative_velocity(
    params: PhysicalParameters, time: np.ndarray | float
) -> np.ndarray:
    """The same half-cosine relative-flow law hard-projected by the PINN."""

    values = np.asarray(time, dtype=float)
    if params.valve_close_time_s <= 0.0:
        return np.zeros_like(values)
    phase = np.clip(values / params.valve_close_time_s, 0.0, 1.0)
    return 0.5 * params.initial_velocity_m_s * (
        1.0 + np.cos(np.pi * phase)
    )


def solve_matched_fssi_reference(
    params: PhysicalParameters,
    output_times: np.ndarray,
    *,
    n_cells: int = 400,
    cfl: float = 0.90,
) -> dict[str, np.ndarray]:
    """Solve the current four-equation case by a label-free MOC reference.

    This deterministic reference deliberately uses exactly the same initial
    state and boundary equations as the Figure 9 PINN.  In particular, the
    right boundary is ``V-uz=Vrel(t)`` together with
    ``Mv*uz_t-Af*P+At*sigma_z=0``.  It is evaluated only after PINN training
    and is never supplied as a training label.
    """

    times = np.asarray(output_times, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("output_times must be a one-dimensional array")
    if not np.isclose(times[0], 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("output_times must start at zero and increase")
    if n_cells < 4:
        raise ValueError("n_cells must be at least four")
    if not 0.0 < cfl <= 1.0:
        raise ValueError("cfl must lie in (0, 1]")

    speeds, right, left, scales = characteristic_basis(params)
    physical_modes = scales[:, None] * right
    positive = np.flatnonzero(speeds > 0.0)
    negative = np.flatnonzero(speeds < 0.0)
    if len(positive) != 2 or len(negative) != 2:
        raise ValueError("the four-equation system must have two waves each way")

    z = np.linspace(0.0, params.length_m, n_cells + 1)
    dz = params.length_m / n_cells
    maximum_dt = cfl * dz / float(np.max(np.abs(speeds)))
    pressure0 = initial_pressure_pa(params)
    area_f, area_t = cross_section_areas(params)
    stress0 = area_f * pressure0 / area_t
    state = np.empty((n_cells + 1, 4), dtype=float)
    state[:] = (
        params.initial_velocity_m_s,
        0.0,
        pressure0,
        stress0,
    )
    amplitudes = (state / scales) @ left.T
    snapshots = [state.copy()]
    current_time = 0.0

    left_conditions = np.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=float,
    )
    left_targets = np.array([pressure0, 0.0], dtype=float)

    for target_time in times[1:]:
        while current_time < target_time - 1.0e-14:
            dt = min(maximum_dt, target_time - current_time)
            new_time = current_time + dt
            updated = np.empty_like(amplitudes)
            for mode, speed in enumerate(speeds):
                departure = z - speed * dt
                updated[:, mode] = np.interp(
                    departure,
                    z,
                    amplitudes[:, mode],
                    left=amplitudes[0, mode],
                    right=amplitudes[-1, mode],
                )

            left_amplitudes = _project_reference_boundary(
                updated[0],
                positive,
                negative,
                left_conditions,
                left_targets,
                physical_modes,
            )
            updated[0] = left_amplitudes

            relative_velocity = float(
                matched_valve_relative_velocity(params, new_time)
            )
            right_conditions = np.array(
                [
                    [1.0, -1.0, 0.0, 0.0],
                    [
                        0.0,
                        params.valve_mass_kg / dt,
                        -area_f,
                        area_t,
                    ],
                ],
                dtype=float,
            )
            right_targets = np.array(
                [
                    relative_velocity,
                    params.valve_mass_kg * state[-1, 1] / dt,
                ],
                dtype=float,
            )
            right_amplitudes = _project_reference_boundary(
                updated[-1],
                negative,
                positive,
                right_conditions,
                right_targets,
                physical_modes,
            )
            updated[-1] = right_amplitudes

            amplitudes = updated
            state = amplitudes @ physical_modes.T
            current_time = new_time
        snapshots.append(state.copy())

    values = np.stack(snapshots, axis=2)
    return {
        "z": z,
        "t": times.copy(),
        "V": values[:, 0, :],
        "uz": values[:, 1, :],
        "P": values[:, 2, :],
        "sigma_z": values[:, 3, :],
    }


def load_cached_wave_result(
    case_root: Path, params: PhysicalParameters
) -> dict[str, np.ndarray]:
    """Load a previously trained case without invoking the trainer."""

    result_path = wave_case_directory(
        case_root, params, "fssi"
    ) / "results.npz"
    if not result_path.exists():
        raise FileNotFoundError(
            f"cached PINN result not found: {result_path}; "
            "run once without --reference-only"
        )
    with np.load(result_path, allow_pickle=False) as archive:
        return {
            key: archive[key]
            for key in ("z", "t", "V", "uz", "P", "sigma_z")
        }


def digitize_paper_psc_image(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Digitise the red curve from a Figure 9 crop already held in memory."""

    image = image[..., :3]
    if image.dtype.kind in "ui":
        image = image.astype(float) / np.iinfo(image.dtype).max
    height, width = image.shape[:2]
    # Stable axes coordinates in common.paper.CROPS[9].
    x_left = int(round(0.205 * width))
    x_right = int(round(0.956 * width))
    y_top = int(round(0.116 * height))
    y_bottom = int(round(0.791 * height))
    red = (
        (image[..., 0] > 0.62)
        & (image[..., 1] < 0.68)
        & (image[..., 2] < 0.68)
        & ((image[..., 0] - image[..., 1]) > 0.18)
        & ((image[..., 0] - image[..., 2]) > 0.18)
    )
    pixels_x: list[int] = []
    pixels_y: list[float] = []
    for x in range(x_left, x_right + 1):
        candidates = np.flatnonzero(red[y_top : y_bottom + 1, x]) + y_top
        # Exclude the red legend sample at y~=126; the plotted red curve is
        # below y=200 throughout this crop.
        candidates = candidates[candidates >= int(round(0.194 * height))]
        if len(candidates):
            pixels_x.append(x)
            pixels_y.append(float(np.median(candidates)))
    if len(pixels_x) < 0.72 * (x_right - x_left):
        raise RuntimeError("Could not reliably digitise Figure 9's red curve")
    all_x = np.arange(x_left, x_right + 1)
    all_y = np.interp(all_x, np.asarray(pixels_x), np.asarray(pixels_y))
    time = 0.8 * (all_x - x_left) / (x_right - x_left)
    pressure = 1.2e5 - (all_y - y_top) / (y_bottom - y_top) * 1.6e5
    uncertainty_pa = 2.0 * 1.6e5 / (y_bottom - y_top)
    uncertainty_s = 2.0 * 0.8 / (x_right - x_left)
    return time, pressure, uncertainty_pa, uncertainty_s


def digitize_paper_psc_curve(
    paper_path: Path, csv_path: Path
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Digitise and save the paper's red ``With FSI`` curve."""

    time, pressure, uncertainty_pa, uncertainty_s = (
        digitize_paper_psc_image(mpimg.imread(paper_path))
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "time_s",
                "pressure_pa",
                "digitization_uncertainty_pa",
                "digitization_uncertainty_s",
            )
        )
        writer.writerows(
            zip(
                time,
                pressure,
                np.full_like(time, uncertainty_pa),
                np.full_like(time, uncertainty_s),
            )
        )
    return time, pressure, uncertainty_pa, uncertainty_s


def rmse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean((first - second) ** 2)))


def phase_statistics(
    time: np.ndarray, pressure: np.ndarray
) -> dict[str, dict[str, float]]:
    """Report the three intervals annotated in the paper's second surge."""

    windows = {
        "phase_1": (0.280, 0.315),
        "phase_2": (0.315, 0.365),
        "phase_3": (0.365, 0.400),
    }
    output: dict[str, dict[str, float]] = {}
    for name, (start, stop) in windows.items():
        mask = (time >= start) & (time <= stop)
        values = pressure[mask]
        output[name] = {
            "start_s": start,
            "stop_s": stop,
            "mean_pa": float(np.mean(values)),
            "minimum_pa": float(np.min(values)),
            "maximum_pa": float(np.max(values)),
        }
    return output


def build_metrics(
    exposed: dict[str, np.ndarray],
    restrained: dict[str, np.ndarray],
    exposed_reference: dict[str, np.ndarray],
    restrained_reference: dict[str, np.ndarray],
    paper_t: np.ndarray,
    paper_p: np.ndarray,
    uncertainty_pa: float,
    uncertainty_s: float,
    exposed_params: PhysicalParameters,
    restrained_params: PhysicalParameters,
    case_root: Path,
) -> dict[str, object]:
    time = restrained["t"]
    exposed_p = exposed["P"][-1]
    restrained_p = restrained["P"][-1]
    exposed_reference_p = exposed_reference["P"][-1]
    restrained_reference_p = restrained_reference["P"][-1]
    paper_on_model = np.interp(time, paper_t, paper_p)
    paper_span = float(np.ptp(paper_p))
    difference = exposed_p - restrained_p

    diagnostics = {}
    for name, params in (
        ("EsE_0", exposed_params),
        ("EsE_0p1", restrained_params),
    ):
        folder_name = (
            "fssi_EsE_0e00_Mv_100_nu_0p30"
            if name == "EsE_0"
            else "fssi_EsE_1e-1_Mv_100_nu_0p30"
        )
        path = case_root / folder_name / "diagnostics.json"
        diagnostics[name] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"status": "missing"}
        )
        diagnostics[name]["characteristic_speeds_m_s"] = (
            characteristic_speeds(params).tolist()
        )
        diagnostics[name]["coefficients"] = fssi_coefficients(params)

    return {
        "training_uses_paper_or_numerical_labels": False,
        "trainer_revision": WAVE_PINN_REVISION,
        "case_definition": {
            "EsE_0": (
                "complete four-equation model; PSC disabled only; "
                "Poisson and junction coupling retained"
            ),
            "EsE_0p1": (
                "complete four-equation model with Poisson, junction and PSC"
            ),
            "all_other_parameters_identical": True,
        },
        "paper_curve": {
            "digitized_curve": "red 'With FSI' curve",
            "digitization_uncertainty_pa": uncertainty_pa,
            "digitization_uncertainty_s": uncertainty_s,
            "pressure_range_pa": [
                float(np.min(paper_p)),
                float(np.max(paper_p)),
            ],
        },
        "restrained_EsE_0p1_vs_paper_red": {
            "rmse_pa": rmse(restrained_p, paper_on_model),
            "nrmse_by_paper_range": rmse(restrained_p, paper_on_model)
            / paper_span,
        },
        "PINN_vs_same_condition_MOC_reference": {
            "reference_is_used_for_training": False,
            "EsE_0": {
                "rmse_pa": rmse(exposed_p, exposed_reference_p),
                "nrmse_by_reference_range": rmse(
                    exposed_p, exposed_reference_p
                )
                / max(float(np.ptp(exposed_reference_p)), 1.0),
                "maximum_absolute_error_pa": float(
                    np.max(np.abs(exposed_p - exposed_reference_p))
                ),
            },
            "EsE_0p1_current_with_FSI": {
                "rmse_pa": rmse(restrained_p, restrained_reference_p),
                "nrmse_by_reference_range": rmse(
                    restrained_p, restrained_reference_p
                )
                / max(float(np.ptp(restrained_reference_p)), 1.0),
                "maximum_absolute_error_pa": float(
                    np.max(np.abs(restrained_p - restrained_reference_p))
                ),
            },
        },
        "computed_pressure_ranges_pa": {
            "EsE_0": [float(np.min(exposed_p)), float(np.max(exposed_p))],
            "EsE_0p1": [
                float(np.min(restrained_p)),
                float(np.max(restrained_p)),
            ],
        },
        "PSC_effect_computed": {
            "curve_difference_rmse_pa": float(
                np.sqrt(np.mean(difference**2))
            ),
            "curve_difference_max_abs_pa": float(
                np.max(np.abs(difference))
            ),
            "standard_deviation_EsE_0_pa": float(np.std(exposed_p)),
            "standard_deviation_EsE_0p1_pa": float(np.std(restrained_p)),
        },
        "second_surge_three_phase_statistics": {
            "EsE_0": phase_statistics(time, exposed_p),
            "EsE_0p1": phase_statistics(time, restrained_p),
            "paper_red": phase_statistics(paper_t, paper_p),
            "same_condition_MOC_EsE_0": phase_statistics(
                exposed_reference["t"], exposed_reference_p
            ),
            "same_condition_MOC_EsE_0p1": phase_statistics(
                restrained_reference["t"], restrained_reference_p
            ),
        },
        "physics_and_boundary_validation": diagnostics,
    }


def save_field_plot(
    output_path: Path,
    exposed: dict[str, np.ndarray],
    restrained: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.5), constrained_layout=True)
    fields = (
        ("P", "pressure (Pa)"),
        ("V", "fluid velocity (m/s)"),
        ("uz", "pipe axial velocity (m/s)"),
        ("sigma_z", "axial stress (Pa)"),
    )
    for row, (result, prefix) in enumerate(
        ((exposed, r"$E_s/E=0$"), (restrained, r"$E_s/E=0.1$"))
    ):
        for column, (key, label) in enumerate(fields):
            axis = axes[row, column]
            image = axis.pcolormesh(
                result["t"],
                result["z"],
                result[key],
                shading="auto",
                cmap="RdBu_r",
            )
            axis.set(
                xlabel="time (s)",
                ylabel="z (m)",
                title=f"{prefix}: {label}",
            )
            fig.colorbar(image, ax=axis)
    fig.suptitle("Figure 9 four-equation characteristic PINN fields")
    fig.savefig(output_path, dpi=175)
    plt.close(fig)


def save_result_plots(
    output_dir: Path,
    exposed: dict[str, np.ndarray],
    restrained: dict[str, np.ndarray],
    paper_t: np.ndarray,
    paper_p: np.ndarray,
    metrics: dict[str, object],
) -> None:
    result_path = output_dir / "computed.png"
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.0)},
        constrained_layout=True,
    )
    axes[0].plot(
        paper_t,
        paper_p,
        color="0.72",
        linewidth=5.0,
        label="paper: digitised red 'With FSI' curve",
    )
    axes[0].plot(
        exposed["t"],
        exposed["P"][-1],
        color="black",
        linewidth=1.5,
        label=r"PINN: $E_s/E=0$ (Poisson + junction)",
    )
    axes[0].plot(
        restrained["t"],
        restrained["P"][-1],
        color="red",
        linewidth=1.5,
        label=r"PINN: $E_s/E=0.1$ (+ PSC)",
    )
    for start, stop in ((0.280, 0.315), (0.315, 0.365), (0.365, 0.400)):
        axes[0].axvspan(start, stop, color="#4c78a8", alpha=0.055)
    axes[0].set(
        ylabel="valve pressure (Pa)",
        xlim=(0.0, 0.8),
        ylim=(-4.0e4, 1.2e5),
        title="Figure 9: four-equation PINNs and digitised paper curve",
    )
    axes[0].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8.5, loc="upper right")
    ranges = metrics["computed_pressure_ranges_pa"]
    axes[0].text(
        0.99,
        0.02,
        "computed ranges (Pa)\n"
        f"Es/E=0: [{ranges['EsE_0'][0]:.2e}, {ranges['EsE_0'][1]:.2e}]\n"
        f"Es/E=0.1: [{ranges['EsE_0p1'][0]:.2e}, {ranges['EsE_0p1'][1]:.2e}]",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.86},
    )
    difference = exposed["P"][-1] - restrained["P"][-1]
    axes[1].plot(exposed["t"], difference, color="#3569a8")
    axes[1].axhline(0.0, color="0.4", linewidth=0.7)
    axes[1].set(
        xlabel="time (s)",
        ylabel=r"$P_{E_s/E=0}-P_{E_s/E=0.1}$ (Pa)",
    )
    axes[1].grid(True, alpha=0.25)
    effect = metrics["PSC_effect_computed"]
    axes[1].text(
        0.01,
        0.06,
        f"difference RMS = {effect['curve_difference_rmse_pa']:.2e} Pa; "
        f"max = {effect['curve_difference_max_abs_pa']:.2e} Pa",
        transform=axes[1].transAxes,
        fontsize=8.5,
    )
    fig.savefig(result_path, dpi=190)
    plt.close(fig)

    save_field_plot(output_dir / "fssi_fields.png", exposed, restrained)
    save_comparison(
        9,
        result_path,
        output_dir / "comparison.png",
        "Characteristic-causal four-equation PINNs",
    )


def save_same_condition_comparison(
    output_dir: Path,
    exposed: dict[str, np.ndarray],
    restrained: dict[str, np.ndarray],
    exposed_reference: dict[str, np.ndarray],
    restrained_reference: dict[str, np.ndarray],
    metrics: dict[str, object],
) -> None:
    """Plot PINNs against independent MOC solutions of the same problem."""

    csv_path = output_dir / "same_condition_reference.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "time_s",
                "PINN_EsE_0_pressure_pa",
                "MOC_EsE_0_pressure_pa",
                "PINN_EsE_0p1_pressure_pa",
                "MOC_EsE_0p1_current_with_FSI_pressure_pa",
            )
        )
        writer.writerows(
            zip(
                exposed["t"],
                exposed["P"][-1],
                exposed_reference["P"][-1],
                restrained["P"][-1],
                restrained_reference["P"][-1],
            )
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.4, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.15)},
        constrained_layout=True,
    )
    axes[0].plot(
        exposed_reference["t"],
        exposed_reference["P"][-1],
        color="0.35",
        linewidth=2.0,
        label=r"same-condition MOC: $E_s/E=0$",
    )
    axes[0].plot(
        exposed["t"],
        exposed["P"][-1],
        color="black",
        linestyle="--",
        linewidth=1.25,
        label=r"PINN: $E_s/E=0$",
    )
    axes[0].plot(
        restrained_reference["t"],
        restrained_reference["P"][-1],
        color="#d62728",
        linewidth=2.0,
        label=r"same-condition MOC: $E_s/E=0.1$ (current With FSI)",
    )
    axes[0].plot(
        restrained["t"],
        restrained["P"][-1],
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.25,
        label=r"PINN: $E_s/E=0.1$",
    )
    axes[0].set(
        ylabel="valve pressure (Pa)",
        title=(
            "Figure 9 verification under identical current parameters, "
            "ICs and BCs"
        ),
    )
    axes[0].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8.2, loc="upper right")

    exposed_error = exposed["P"][-1] - exposed_reference["P"][-1]
    restrained_error = (
        restrained["P"][-1] - restrained_reference["P"][-1]
    )
    axes[1].plot(
        exposed["t"], exposed_error, color="black", label=r"$E_s/E=0$"
    )
    axes[1].plot(
        restrained["t"],
        restrained_error,
        color="#d62728",
        label=r"$E_s/E=0.1$",
    )
    axes[1].axhline(0.0, color="0.5", linewidth=0.7)
    axes[1].set(
        xlabel="time (s)",
        ylabel="PINN - MOC (Pa)",
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8.2)
    comparison = metrics["PINN_vs_same_condition_MOC_reference"]
    axes[1].text(
        0.99,
        0.05,
        "RMSE (Pa): "
        f"Es/E=0 {comparison['EsE_0']['rmse_pa']:.2e}; "
        "Es/E=0.1 "
        f"{comparison['EsE_0p1_current_with_FSI']['rmse_pa']:.2e}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
    )
    fig.savefig(
        output_dir / "same_condition_comparison.png", dpi=190
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve and compare both Figure 9 four-equation PINNs"
    )
    add_training_arguments(parser)
    parser.set_defaults(
        epochs=2400,
        lbfgs_steps=0,
        hidden_width=48,
        hidden_layers=3,
        pde_points=3600,
        boundary_points=240,
        plot_time_points=1601,
        inner_radius=0.5,
        valve_close_time=0.03,
        initial_velocity=0.060,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--reference-cells",
        type=int,
        default=400,
        help=(
            "spatial cells in the independent same-condition MOC reference"
        ),
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help=(
            "reuse cached PINN results and generate only MOC references, "
            "metrics and plots; never starts training"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    # Keep the two cases numerically identical except for Es/E.
    config.plot_z = 161 if not args.quick else config.plot_z
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_root = output_dir / "cases"

    # Values below follow the supplied Fig. 9 case specification.  In
    # particular, the printed-equation coefficient branch is held fixed.
    exposed_params, restrained_params = figure09_parameters(
        inner_radius_m=args.inner_radius,
        initial_velocity_m_s=args.initial_velocity,
        valve_close_time_s=args.valve_close_time,
    )
    base = restrained_params
    if args.reference_only:
        exposed = load_cached_wave_result(case_root, exposed_params)
        restrained = load_cached_wave_result(case_root, restrained_params)
    else:
        exposed = train_wave_case(
            "figure09_EsE_0_four_equation",
            exposed_params,
            "fssi",
            config,
            case_root,
            args.force,
        )
        restrained = train_wave_case(
            "figure09_EsE_0p1_four_equation",
            restrained_params,
            "fssi",
            config,
            case_root,
            args.force,
        )
    exposed_reference = solve_matched_fssi_reference(
        exposed_params,
        exposed["t"],
        n_cells=args.reference_cells,
    )
    restrained_reference = solve_matched_fssi_reference(
        restrained_params,
        restrained["t"],
        n_cells=args.reference_cells,
    )

    paper_path = extract_paper_figure(9, output_dir / "figure09_paper.png")
    paper_t, paper_p, uncertainty_pa, uncertainty_s = (
        digitize_paper_psc_curve(
            paper_path, output_dir / "paper_red_curve.csv"
        )
    )
    metrics = build_metrics(
        exposed,
        restrained,
        exposed_reference,
        restrained_reference,
        paper_t,
        paper_p,
        uncertainty_pa,
        uncertainty_s,
        exposed_params,
        restrained_params,
        case_root,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assumptions = {
        "parameters": asdict(base),
        "coefficient_branch": "printed_equations",
        "boundary_model": (
            "P(0,t)=P0, uz(0,t)=0; cosine relative-flow closure at z=L; "
            "M_v*uz_t-Af*P+At*sigma_z=0"
        ),
        "missing_publication_inputs": (
            "The paper does not publish the absolute pipe radius, initial "
            "steady flow, or K/n/opening-to-flow valve closure. Values follow "
            "the supplied case specification rather than curve calibration."
        ),
        "training_data": (
            "governing equations, IC and BC only; paper digitisation occurs "
            "after both cases have been solved; the same-condition MOC "
            "reference is also generated only after training"
        ),
        "same_condition_reference": (
            "independent characteristic MOC using the identical four-equation "
            "coefficients, initial equilibrium, reservoir boundary, cosine "
            "relative-flow closure and valve end-mass balance"
        ),
    }
    (output_dir / "assumptions.json").write_text(
        json.dumps(assumptions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_result_plots(
        output_dir, exposed, restrained, paper_t, paper_p, metrics
    )
    save_same_condition_comparison(
        output_dir,
        exposed,
        restrained,
        exposed_reference,
        restrained_reference,
        metrics,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
