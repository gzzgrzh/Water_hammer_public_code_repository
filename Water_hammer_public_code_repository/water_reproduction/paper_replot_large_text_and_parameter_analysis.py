"""Replot manuscript figures with readable text and extend engineering parameter analysis.

This script performs no training.  It reads completed CSV/JSON/NPZ evidence,
and only for the two public external datasets reloads locked checkpoints for
forward inference needed to redraw their time histories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from water16_reproduction import perugia_external_models as perugia
from water16_reproduction import revised_ablation_matrix as ablation
from water16_reproduction import revised_joint_boundary_moc as boundary
from water16_reproduction import revised_joint_threshold_scan as threshold
from water16_reproduction import revised_model_comparison as comparison
from water16_reproduction import xu2025_external_three_model as xu


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/paper_figures_large_text_v1"
SCAN_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/revised_joint_threshold_scan_v1"
BOUNDARY_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/revised_joint_boundary_moc_v1"
XU_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/xu2025_external_three_model_v2"
PERUGIA_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/perugia_external_three_model_v1"

COLORS = {
    "ann": "#777777",
    "coordinate_pinn": "#E07B39",
    "characteristic_pinn": "#2878B5",
}
MODEL_LABELS = {
    "ann": "ANN",
    "coordinate_pinn": "Standard PINN",
    "characteristic_pinn": "Proposed model",
}


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 13.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 11.0,
            "figure.titlesize": 15.0,
            "lines.linewidth": 1.8,
            "lines.markersize": 6.5,
            "savefig.dpi": 320,
        }
    )


def save(figure: plt.Figure, output: Path, stem: str) -> list[str]:
    names = []
    for suffix in ("png", "pdf"):
        target = output / f"{stem}.{suffix}"
        if target.exists():
            raise FileExistsError(target)
        figure.savefig(
            target,
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        names.append(target.name)
    plt.close(figure)
    return names


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows: list[dict[str, Any]] = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            if key.endswith("_safe") or key in {
                "registered_safe",
                "moc_safe",
                "classification_agreement",
                "false_safe",
            }:
                row[key] = str(value).strip().lower() == "true"
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replot_internal(output: Path) -> list[str]:
    source = ROOT / "PINN_FSSI_research_plan/outputs/revised_three_model_comparison_v1"
    rows = comparison.read_csv(source / "T05_three_model_case_metrics.csv")
    grouped = {
        model: [comparison.add_engineering_errors(row) for row in rows if row["model"] == model]
        for model in comparison.MODEL_ORDER
    }
    comparison.plot_error_boxplots(grouped, output)
    comparison.plot_peak_parity(grouped, output)
    comparison.plot_class_performance(grouped, output)
    comparison.plot_engineering_localization(grouped, output)
    comparison.plot_training_histories(
        {
            "ANN": comparison.DEFAULT_ANN / "history.json",
            "Coordinate PINN": comparison.DEFAULT_PINN / "history.json",
            "Characteristic model": ROOT
            / "PINN_FSSI_research_plan/outputs/parametric_fssi_hybrid_formal_v3_balanced_events/formal/history.json",
        },
        output,
    )
    return [f"F{number}" for number in range(11, 16)]


def replot_ablation(output: Path) -> list[str]:
    progress = json.loads(
        (ROOT / "PINN_FSSI_research_plan/outputs/revised_ablation_matrix_v1/progress.json").read_text(
            encoding="utf-8"
        )
    )
    rows = progress["completed"]
    ablation.plot_curve(rows, "labels_per_case", (96, 192, 384), output, "F16_non_oracle_label_efficiency")
    ablation.plot_curve(rows, "training_cases", (18, 36, 72), output, "F17_training_case_efficiency")
    ablation.plot_sampling(rows, output)
    return ["F16", "F17", "F18"]


def replot_threshold(output: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    rows = read_rows(SCAN_OUTPUT / "joint_threshold_scan_results.csv")
    config = json.loads(threshold.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    threshold.plot_utilization_maps(rows, config, output)
    threshold.plot_control_maps(rows, config, output)
    minimum = threshold.minimum_closure_rows(rows, config)
    threshold.plot_minimum_closure(minimum, config, output)
    threshold.plot_soil_closure_maps(rows, config, output)
    threshold.plot_threshold_sensitivity(rows, config, output)
    return rows, config, [f"F{number}" for number in range(28, 33)]


def plot_boundary_histories(rows: list[dict[str, Any]], output: Path) -> list[str]:
    critical = sorted(rows, key=lambda row: abs(float(row["moc_utilization"]) - 1.0))[:4]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True)
    for axis, row in zip(axes.ravel(), critical):
        case_dir = BOUNDARY_OUTPUT / "cases" / str(row["case_id"])
        with np.load(case_dir / "truth_evaluation_grid.npz") as archive:
            time_s = archive["t"]
            pressure = archive["P"][-1]
            stress = archive["sigma_z"][-1]
        metadata = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        pressure0 = float(metadata["initial_pressure_pa"])
        stress0 = float(metadata["initial_stress_pa"])
        axis.plot(time_s, (pressure - pressure0) * 1.0e-3, color="#4C78A8", lw=1.8)
        twin = axis.twinx()
        twin.plot(time_s, (stress - stress0) * 1.0e-6, color="#E15759", lw=1.6)
        axis.set_title(
            f"{str(row['case_id']).replace('joint_boundary_moc_', 'B')}: "
            f"MOC U={float(row['moc_utilization']):.3f}"
        )
        axis.set_ylabel(r"$\Delta P$ (kPa)", color="#4C78A8")
        twin.set_ylabel(r"$\Delta\sigma_z$ (MPa)", color="#E15759")
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    figure.tight_layout()
    return save(figure, output, "F36_boundary_critical_valve_histories")


def replot_boundary(output: Path) -> list[str]:
    rows = read_rows(BOUNDARY_OUTPUT / "T12_boundary_MOC_confirmation.csv")
    boundary.plot_parity(rows, output)
    boundary.plot_utilization(rows, output)
    boundary.plot_confusion(rows, output)
    plot_boundary_histories(rows, output)
    return ["F33", "F34", "F35", "F36"]


def replot_xu(output: Path, device: torch.device) -> list[str]:
    dtype = torch.float64
    primary_seed = xu.SEEDS[0]
    curves: dict[tuple[str, str], tuple[Any, np.ndarray, np.ndarray]] = {}
    for dataset in xu.DATASETS:
        field = xu.load_field(dataset, "release")
        for model_name in xu.MODELS:
            model_path = xu.run_directory(XU_OUTPUT, "formal", dataset, model_name, primary_seed) / "best_model.pt"
            model = xu.load_best_model(model_path, device, dtype)
            head, velocity = xu.predict_field(model, field, device, dtype)
            curves[(dataset, model_name)] = (field, head, velocity)
    for figure_number, dataset in ((38, "SFM"), (39, "TVB")):
        field = curves[(dataset, "ann")][0]
        figure, axes = plt.subplots(2, 2, figsize=(14.2, 8.4), sharex=True)
        for column, x_m in enumerate(field.x_m):
            axes[0, column].plot(field.time_s, field.head_m[:, column], color="black", lw=2.2, label="reference")
            axes[1, column].plot(field.time_s, field.velocity_m_s[:, column], color="black", lw=2.2, label="reference")
            for model_name in xu.MODELS:
                _, head, velocity = curves[(dataset, model_name)]
                label = MODEL_LABELS[model_name]
                axes[0, column].plot(field.time_s, head[:, column], color=COLORS[model_name], lw=1.7, label=label)
                axes[1, column].plot(field.time_s, velocity[:, column], color=COLORS[model_name], lw=1.7, label=label)
            axes[0, column].set_title(f"{dataset}, x={x_m:.0f} m")
            axes[1, column].set_xlabel("Time (s)")
            for axis in axes[:, column]:
                axis.grid(alpha=0.18)
        axes[0, 0].set_ylabel("Head (m)")
        axes[1, 0].set_ylabel("Velocity (m/s)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            frameon=False,
            ncol=4,
            fontsize=11,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        save(figure, output, f"F{figure_number}_xu2025_{dataset.lower()}_sealed_histories")

    rows = read_rows(XU_OUTPUT / "formal/sealed_release/T13_xu2025_sealed_test_metrics.csv")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), sharey=True)
    for axis, dataset in zip(axes, xu.DATASETS):
        positions = np.arange(len(xu.MODELS))
        head, velocity = [], []
        for model_name in xu.MODELS:
            selected = [row for row in rows if row["dataset"] == dataset and row["model"] == model_name]
            head.append(100.0 * np.mean([row["nrmse_dynamic_range"] for row in selected if row["variable"] == "head"]))
            velocity.append(100.0 * np.mean([row["nrmse_dynamic_range"] for row in selected if row["variable"] == "velocity"]))
        axis.bar(positions - 0.18, head, 0.36, label="head", color="#4C78A8")
        axis.bar(positions + 0.18, velocity, 0.36, label="velocity", color="#F58518")
        axis.set_xticks(positions, [MODEL_LABELS[name].replace(" ", "\n") for name in xu.MODELS])
        axis.set_title(dataset)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Test-location NRMSE (%)")
    axes[1].legend(frameon=False)
    figure.suptitle("Prediction errors for the Xu (2025) test data")
    figure.tight_layout()
    save(figure, output, "F40_xu2025_sealed_error_summary")
    return ["F38", "F39", "F40"]


def replot_perugia(output: Path, device: torch.device) -> list[str]:
    time_vector, test_signals = perugia.load_stage_data(perugia.DEFAULT_INPUT, "release")
    dtype = torch.float64
    primary_seed = perugia.SEEDS[0]
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    evaluation_time = None
    for model_name in perugia.MODEL_NAMES:
        checkpoint = PERUGIA_OUTPUT / "formal" / model_name / f"seed_{primary_seed}" / "best_model.pt"
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = perugia.NetworkSurrogate(model_name).to(device=device, dtype=dtype)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        _, references, predictions, evaluation_time = perugia.evaluate_sensors(
            model,
            time_vector,
            test_signals,
            perugia.SEALED_TEST_SENSORS,
            device,
            dtype,
        )
        for sensor in perugia.SEALED_TEST_SENSORS:
            curves[(model_name, sensor)] = (references[sensor], predictions[sensor])
    mask = (time_vector >= 0.0) & (time_vector <= 2.0)
    figure, axes = plt.subplots(len(perugia.SEALED_TEST_SENSORS), 1, figsize=(13.0, 9.2), sharex=True)
    for axis, sensor in zip(axes, perugia.SEALED_TEST_SENSORS):
        reference = curves[("ann", sensor)][0]
        axis.plot(time_vector[mask], reference, color="black", lw=1.8, label="experiment")
        for model_name in perugia.MODEL_NAMES:
            prediction = curves[(model_name, sensor)][1]
            axis.plot(time_vector[mask], prediction, lw=1.55, color=COLORS[model_name], label=model_name.replace("_", " "))
        axis.set_ylabel(f"{sensor}: $H-H_0$ (m)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=4, fontsize=11)
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    save(figure, output, "F26_perugia_sealed_test_histories")

    rows = read_rows(PERUGIA_OUTPUT / "formal/sealed_release/T09_perugia_sealed_test_metrics.csv")
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), gridspec_kw={"hspace": 0.35, "wspace": 0.30})
    positions = np.arange(len(perugia.MODEL_NAMES))
    values = [[100.0 * row["nrmse_dynamic_range"] for row in rows if row["model"] == model] for model in perugia.MODEL_NAMES]
    boxes = axes[0, 0].boxplot(values, positions=positions, widths=0.55, patch_artist=True, showmeans=True)
    for box, model in zip(boxes["boxes"], perugia.MODEL_NAMES):
        box.set_facecolor(COLORS[model])
        box.set_alpha(0.75)
    axes[0, 0].set_xticks(positions, [name.replace("_", "\n") for name in perugia.MODEL_NAMES])
    axes[0, 0].set(ylabel="Pressure NRMSE (%)", title="(a) Full-history error")
    axes[0, 0].grid(axis="y", alpha=0.2)
    sensor_x = np.arange(len(perugia.SEALED_TEST_SENSORS))
    width = 0.24
    for model_index, model in enumerate(perugia.MODEL_NAMES):
        peaks = [np.mean([100.0 * row["positive_peak_relative_error"] for row in rows if row["model"] == model and row["sensor"] == sensor]) for sensor in perugia.SEALED_TEST_SENSORS]
        arrivals = [np.nanmean([row["first_arrival_absolute_error_s"] for row in rows if row["model"] == model and row["sensor"] == sensor]) for sensor in perugia.SEALED_TEST_SENSORS]
        axes[0, 1].bar(sensor_x + (model_index - 1) * width, peaks, width, color=COLORS[model], label=model.replace("_", " "))
        axes[1, 0].bar(sensor_x + (model_index - 1) * width, arrivals, width, color=COLORS[model])
    axes[0, 1].set_xticks(sensor_x, perugia.SEALED_TEST_SENSORS)
    axes[0, 1].set(ylabel="Positive-peak error (%)", title="(b) Peak reconstruction")
    axes[0, 1].grid(axis="y", alpha=0.2)
    axes[0, 1].legend(frameon=False, fontsize=10)
    axes[1, 0].set_xticks(sensor_x, perugia.SEALED_TEST_SENSORS)
    axes[1, 0].set(ylabel="First-arrival error (s)", title="(c) Wave arrival")
    axes[1, 0].grid(axis="y", alpha=0.2)
    residuals = np.stack([curves[("characteristic_pinn", sensor)][1] - curves[("characteristic_pinn", sensor)][0] for sensor in perugia.SEALED_TEST_SENSORS])
    limit = max(float(np.quantile(np.abs(residuals), 0.99)), 1.0e-6)
    image = axes[1, 1].imshow(residuals, aspect="auto", origin="lower", extent=[float(evaluation_time[0]), float(evaluation_time[-1]), -0.5, 2.5], cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1, 1].set_yticks(np.arange(3), perugia.SEALED_TEST_SENSORS)
    axes[1, 1].set(xlabel="Time (s)", ylabel="Sealed sensor", title="(d) Characteristic-model residual")
    figure.colorbar(image, ax=axes[1, 1], pad=0.02, label="Prediction - experiment (m)")
    figure.tight_layout()
    save(figure, output, "F27_perugia_sealed_test_engineering_metrics")
    return ["F26", "F27"]


RESPONSE_DEFINITIONS = (
    ("maximum_pressure_increment_pa", 1.0e-3, r"Maximum pressure rise $\Delta P_{max}$ (kPa)"),
    ("minimum_absolute_pressure_pa", 1.0e-3, r"Minimum absolute pressure $P_{min}$ (kPa)"),
    ("maximum_absolute_axial_stress_increment_pa", 1.0e-6, r"Maximum axial stress $|\Delta\sigma_z|_{max}$ (MPa)"),
)


def raw_response_maps(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> list[str]:
    soils = config["scan"]["soil_levels_for_velocity_closure_maps"]
    figure, axes = plt.subplots(3, 3, figsize=(15.2, 12.0), sharex=True, sharey=True)
    for row_index, (metric, scale, label) in enumerate(RESPONSE_DEFINITIONS):
        grids = [threshold.grid_for(rows, "soil_to_pipe_modulus_ratio", float(soil), "closure_time_over_L_cf", "initial_velocity_m_s", metric) for soil in soils]
        low = min(float((values * scale).min()) for _, _, values in grids)
        high = max(float((values * scale).max()) for _, _, values in grids)
        levels = np.linspace(low, high, 20)
        for column, (soil, (closure, velocity, values)) in enumerate(zip(soils, grids)):
            contour = axes[row_index, column].contourf(closure, velocity, values * scale, levels=levels, cmap="viridis")
            if row_index == 0:
                axes[row_index, column].set_title(rf"$E_s/E={soil:g}$")
            if column == 0:
                axes[row_index, column].set_ylabel("Initial velocity $V_0$ (m/s)\n" + label)
            if row_index == 2:
                axes[row_index, column].set_xlabel(r"Closure time $t_c/(L/c_f)$")
        figure.colorbar(contour, ax=axes[row_index, :], pad=0.015, label=label)
    figure.suptitle("Raw engineering responses over initial velocity and closure time", y=0.995, fontweight="bold")
    return save(figure, output, "F41_raw_response_velocity_closure_maps")


def soil_response_maps(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> list[str]:
    velocities = config["scan"]["velocity_levels_for_soil_closure_maps_m_s"]
    figure, axes = plt.subplots(3, 3, figsize=(15.2, 12.0), sharex=True, sharey=True)
    for row_index, (metric, scale, label) in enumerate(RESPONSE_DEFINITIONS):
        grids = [threshold.grid_for(rows, "initial_velocity_m_s", float(velocity), "closure_time_over_L_cf", "soil_to_pipe_modulus_ratio", metric) for velocity in velocities]
        low = min(float((values * scale).min()) for _, _, values in grids)
        high = max(float((values * scale).max()) for _, _, values in grids)
        levels = np.linspace(low, high, 20)
        for column, (velocity, (closure, soil, values)) in enumerate(zip(velocities, grids)):
            contour = axes[row_index, column].contourf(closure, soil, values * scale, levels=levels, cmap="viridis")
            axes[row_index, column].set_yscale("log")
            if row_index == 0:
                axes[row_index, column].set_title(rf"$V_0={velocity:.2f}$ m/s")
            if column == 0:
                axes[row_index, column].set_ylabel(r"Soil restraint $E_s/E$" + "\n" + label)
            if row_index == 2:
                axes[row_index, column].set_xlabel(r"Closure time $t_c/(L/c_f)$")
        figure.colorbar(contour, ax=axes[row_index, :], pad=0.015, label=label)
    figure.suptitle("Raw engineering responses over soil restraint and closure time", y=0.995, fontweight="bold")
    return save(figure, output, "F42_raw_response_soil_closure_maps")


def nearest(value: float, available: list[float]) -> float:
    return min(available, key=lambda item: abs(item - value))


def parameter_effect_curves(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> list[str]:
    soils = [float(value) for value in config["scan"]["soil_levels_for_velocity_closure_maps"]]
    velocities = [float(value) for value in config["scan"]["velocity_levels_for_soil_closure_maps_m_s"]]
    available_velocity = sorted({float(row["initial_velocity_m_s"]) for row in rows if "velocity_closure" in str(row["scan_families"])})
    available_closure = sorted({float(row["closure_time_over_L_cf"]) for row in rows})
    middle_velocity = nearest(0.06, available_velocity)
    middle_closure = nearest(0.60, available_closure)
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.6))
    for column, (metric, scale, label) in enumerate(RESPONSE_DEFINITIONS):
        for soil in soils:
            selected = sorted(
                [row for row in rows if np.isclose(row["soil_to_pipe_modulus_ratio"], soil) and np.isclose(row["initial_velocity_m_s"], middle_velocity)],
                key=lambda row: row["closure_time_over_L_cf"],
            )
            axes[0, column].plot([row["closure_time_over_L_cf"] for row in selected], [row[metric] * scale for row in selected], label=rf"$E_s/E={soil:g}$")
        axes[0, column].set(xlabel=r"$t_c/(L/c_f)$", ylabel=label, title=rf"Closure effect at $V_0={middle_velocity:.3f}$ m/s")
        axes[0, column].grid(alpha=0.22)
        for velocity in velocities:
            selected = sorted(
                [row for row in rows if np.isclose(row["initial_velocity_m_s"], velocity) and np.isclose(row["closure_time_over_L_cf"], middle_closure)],
                key=lambda row: row["soil_to_pipe_modulus_ratio"],
            )
            axes[1, column].plot([row["soil_to_pipe_modulus_ratio"] for row in selected], [row[metric] * scale for row in selected], label=rf"$V_0={velocity:.2f}$ m/s")
        axes[1, column].set_xscale("log")
        axes[1, column].set(xlabel=r"Soil restraint $E_s/E$", ylabel=label, title=rf"Soil effect at $t_c/(L/c_f)={middle_closure:.2f}$")
        axes[1, column].grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    figure.suptitle("Single-factor engineering response curves", y=1.01, fontweight="bold")
    figure.tight_layout()
    return save(figure, output, "F43_single_factor_engineering_response_curves")


def critical_event_maps(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> list[str]:
    soil = float(config["scan"]["soil_levels_for_velocity_closure_maps"][1])
    fields = (
        ("maximum_pressure_location_over_L", "Max-pressure location x/L", "viridis"),
        ("minimum_pressure_location_over_L", "Min-pressure location x/L", "viridis"),
        ("stress_critical_location_over_L", "Max-stress location x/L", "viridis"),
        ("maximum_pressure_time_s", "Max-pressure time (s)", "plasma"),
        ("minimum_pressure_time_s", "Min-pressure time (s)", "plasma"),
        ("stress_critical_time_s", "Max-stress time (s)", "plasma"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.3), sharex=True, sharey=True)
    for axis, (field, label, cmap) in zip(axes.ravel(), fields):
        closure, velocity, values = threshold.grid_for(rows, "soil_to_pipe_modulus_ratio", soil, "closure_time_over_L_cf", "initial_velocity_m_s", field)
        image = axis.pcolormesh(closure, velocity, values, shading="auto", cmap=cmap)
        axis.set_title(label)
        axis.set_xlabel(r"$t_c/(L/c_f)$")
        axis.grid(alpha=0.12)
        figure.colorbar(image, ax=axis, pad=0.02)
    axes[0, 0].set_ylabel("Initial velocity $V_0$ (m/s)")
    axes[1, 0].set_ylabel("Initial velocity $V_0$ (m/s)")
    figure.suptitle(rf"Critical locations and event times at $E_s/E={soil:g}$", y=1.01, fontweight="bold")
    figure.tight_layout()
    return save(figure, output, "F44_critical_location_and_time_maps")


def parameter_summary(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> str:
    soils = [float(value) for value in config["scan"]["soil_levels_for_velocity_closure_maps"]]
    velocities = [float(value) for value in config["scan"]["velocity_levels_for_soil_closure_maps_m_s"]]
    closures = sorted({float(row["closure_time_over_L_cf"]) for row in rows})
    mid_closure = nearest(0.60, closures)
    velocity_values = sorted({float(row["initial_velocity_m_s"]) for row in rows if "velocity_closure" in str(row["scan_families"])})
    mid_velocity = nearest(0.06, velocity_values)
    summary: list[dict[str, Any]] = []

    def append_effect(effect: str, fixed: str, start: dict[str, Any], end: dict[str, Any]) -> None:
        row: dict[str, Any] = {"effect": effect, "fixed_condition": fixed}
        for metric, scale, _ in RESPONSE_DEFINITIONS:
            start_value = float(start[metric]) * scale
            end_value = float(end[metric]) * scale
            row[f"{metric}_start"] = start_value
            row[f"{metric}_end"] = end_value
            row[f"{metric}_change_percent"] = 100.0 * (end_value - start_value) / max(abs(start_value), 1.0e-12)
        summary.append(row)

    for soil in soils:
        selected = sorted([row for row in rows if np.isclose(row["soil_to_pipe_modulus_ratio"], soil) and np.isclose(row["initial_velocity_m_s"], mid_velocity)], key=lambda row: row["closure_time_over_L_cf"])
        append_effect("closure_0.1_to_1.2", f"Es/E={soil:g}; V0={mid_velocity:.3f}", selected[0], selected[-1])
    for velocity in velocities:
        selected = sorted([row for row in rows if np.isclose(row["initial_velocity_m_s"], velocity) and np.isclose(row["closure_time_over_L_cf"], mid_closure)], key=lambda row: row["soil_to_pipe_modulus_ratio"])
        append_effect("soil_min_to_max", f"V0={velocity:.2f}; tc={mid_closure:.2f}", selected[0], selected[-1])
    for soil in soils:
        selected = sorted([row for row in rows if np.isclose(row["soil_to_pipe_modulus_ratio"], soil) and np.isclose(row["closure_time_over_L_cf"], mid_closure)], key=lambda row: row["initial_velocity_m_s"])
        append_effect("velocity_0.03_to_0.09", f"Es/E={soil:g}; tc={mid_closure:.2f}", selected[0], selected[-1])
    path = output / "T14_parameter_effect_summary.csv"
    write_rows(path, summary)
    return path.name


def run(output: Path, device_name: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite paper figure output: {output}")
    output.mkdir(parents=True)
    paper_style()
    figures: list[str] = []
    figures.extend(replot_internal(output))
    figures.extend(replot_ablation(output))
    rows, config, names = replot_threshold(output)
    figures.extend(names)
    figures.extend(replot_boundary(output))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    figures.extend(replot_xu(output, device))
    figures.extend(replot_perugia(output, device))
    raw_response_maps(rows, config, output)
    soil_response_maps(rows, config, output)
    parameter_effect_curves(rows, config, output)
    critical_event_maps(rows, config, output)
    figures.extend(["F41", "F42", "F43", "F44"])
    table = parameter_summary(rows, config, output)
    report = {
        "status": "pass",
        "training_performed": False,
        "locked_checkpoint_inference_only": True,
        "source_scan_case_count": len(rows),
        "style": "large manuscript text",
        "figure_groups": figures,
        "parameter_analysis": [
            "raw velocity-closure response maps at three soil restraints",
            "raw soil-closure response maps at three initial velocities",
            "single-factor closure and soil response curves",
            "critical response location and event-time maps",
        ],
        "tables": [table],
    }
    (output / "large_text_and_parameter_analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def run_xu_only(output: Path, device_name: str) -> dict[str, Any]:
    """Redraw only the Xu SFM/TVB figures from completed checkpoints."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Xu figure output: {output}")
    output.mkdir(parents=True)
    paper_style()
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    figures = replot_xu(output, device)
    report = {
        "status": "pass",
        "training_performed": False,
        "locked_checkpoint_inference_only": True,
        "figure_groups": figures,
        "legend_position": "outside axes, upper center",
    }
    (output / "xu_figure_redraw_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--xu-only",
        action="store_true",
        help="redraw only F38-F40 from completed Xu checkpoints",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.xu_only:
        run_xu_only(args.output, args.device)
    else:
        run(args.output, args.device)
