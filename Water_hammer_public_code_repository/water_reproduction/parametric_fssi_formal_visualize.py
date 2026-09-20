"""Create MOC-referenced publication figures for a frozen formal FSSI checkpoint."""

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

from water16_reproduction.common.physics import cross_section_areas, initial_pressure_pa
from water16_reproduction.parametric_fssi_evaluate import load_checkpoint
from water16_reproduction.parametric_fssi_forward import PARAMETER_NAMES, materialize_cases
from water16_reproduction.parametric_fssi_hybrid import load_config as load_hybrid_config
from water16_reproduction.parametric_fssi_model import (
    ThreeParameterBoundaryModel,
    load_config as load_base_model_config,
    load_design,
    predict_field,
)
from water16_reproduction.parametric_fssi_reference import case_parameters, load_baseline
from water16_reproduction.wp1_verification import STATE_NAMES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYBRID_CONFIG = (
    ROOT / "PINN_FSSI_research_plan" / "configs" / "parametric_fssi_hybrid_formal_v1.json"
)
DEFAULT_CHECKPOINT = (
    ROOT / "PINN_FSSI_research_plan" / "outputs" / "parametric_fssi_hybrid_formal_v1"
    / "formal" / "checkpoint.pt"
)
DEFAULT_METRICS = (
    ROOT / "PINN_FSSI_research_plan" / "outputs" / "parametric_fssi_hybrid_formal_v1"
    / "validation_release" / "case_metrics.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "PINN_FSSI_research_plan" / "outputs" / "paper_figures_formal_validation_v1"
)

COLORS = {
    "train": "#2F6B9A",
    "validation": "#E28E2C",
    "test": "#3A923A",
    "V": "#4C78A8",
    "uz": "#F58518",
    "P": "#54A24B",
    "sigma_z": "#E45756",
    "reference": "#202020",
    "prediction": "#D1495B",
}
MARKERS = {"training": "o", "interpolation": "o", "boundary": "s", "combination": "^"}


def prepare_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = set(PARAMETER_NAMES) | {
        f"{state}_nrmse" for state in STATE_NAMES
    } | {
        "pressure_peak_relative_error", "stress_peak_relative_error",
        "reference_pressure_peak_increment_pa", "prediction_pressure_peak_increment_pa",
        "reference_stress_peak_increment_pa", "prediction_stress_peak_increment_pa",
        "reference_pressure_first_peak_time_s", "prediction_pressure_first_peak_time_s",
        "reference_stress_first_peak_time_s", "prediction_stress_first_peak_time_s",
        "reference_pressure_critical_location_over_L", "prediction_pressure_critical_location_over_L",
        "reference_stress_critical_location_over_L", "prediction_stress_critical_location_over_L",
    }
    for row in rows:
        for key in numeric & set(row):
            row[key] = float(row[key])
    return rows


def save(figure: plt.Figure, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing figure: {path}")
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def plot_formal_design(design: dict[str, Any], output: Path) -> dict[str, Any]:
    cases = materialize_cases(design, "formal")
    figure = plt.figure(figsize=(10.2, 7.4))
    grid = figure.add_gridspec(2, 2, wspace=0.28, hspace=0.30)
    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    axes = [figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])]
    for split in ("train", "validation", "test"):
        for test_class in ("training", "interpolation", "boundary", "combination"):
            subset = [case for case in cases if case["split"] == split and case["test_class"] == test_class]
            if not subset:
                continue
            x = np.log10([float(case[PARAMETER_NAMES[0]]) for case in subset])
            y = [float(case[PARAMETER_NAMES[1]]) for case in subset]
            z = [float(case[PARAMETER_NAMES[2]]) for case in subset]
            label = split if test_class in {"training", "interpolation"} else f"test-{test_class}"
            style = dict(s=32, marker=MARKERS[test_class], color=COLORS[split], edgecolor="white", linewidth=0.45)
            axis_3d.scatter(x, y, z, label=label, **style)
            axes[0].scatter(x, y, **style)
            axes[1].scatter(x, z, **style)
            axes[2].scatter(y, z, **style)
    axis_3d.set(xlabel=r"$\log_{10}(E_s/E)$", ylabel=r"$t_c/(L/c_f)$", zlabel=r"$V_0$ (m/s)", title="(a) Formal three-parameter design")
    handles, labels = axis_3d.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis_3d.legend(unique.values(), unique.keys(), frameon=False, loc="upper left")
    labels_2d = [
        (r"$\log_{10}(E_s/E)$", r"$t_c/(L/c_f)$", "(b) Soil restraint-closure time"),
        (r"$\log_{10}(E_s/E)$", r"$V_0$ (m/s)", "(c) Soil restraint-initial velocity"),
        (r"$t_c/(L/c_f)$", r"$V_0$ (m/s)", "(d) Closure time-initial velocity"),
    ]
    for axis, labels in zip(axes, labels_2d):
        axis.set(xlabel=labels[0], ylabel=labels[1], title=labels[2])
        axis.grid(alpha=0.22, linewidth=0.6)
    save(figure, output / "V03_formal_parameter_design.png")
    return {split: sum(case["split"] == split for case in cases) for split in ("train", "validation", "test")}


def plot_accuracy(
    rows: list[dict[str, Any]],
    acceptance: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"wspace": 0.28})
    state_fields = [("V_nrmse", "V"), ("uz_nrmse", r"$u_z$"), ("P_nrmse", "P"), ("sigma_z_nrmse", r"$\sigma_z$")]
    values = [[100.0 * row[field] for row in rows] for field, _ in state_fields]
    boxes = axes[0].boxplot(values, tick_labels=[label for _, label in state_fields], patch_artist=True, showfliers=False)
    for patch, key in zip(boxes["boxes"], STATE_NAMES):
        patch.set(facecolor=COLORS[key], alpha=0.58)
    for index, samples in enumerate(values, start=1):
        axes[0].scatter(index + np.linspace(-0.12, 0.12, len(samples)), samples, s=16, color="#303030", alpha=0.72)
    pressure_limit = 100.0 * float(acceptance["formal_pressure_nrmse"])
    stress_limit = 100.0 * float(acceptance["formal_axial_stress_nrmse"])
    pressure_peak_limit = 100.0 * float(acceptance["formal_pressure_peak_relative_error"])
    stress_peak_limit = 100.0 * float(acceptance["formal_stress_peak_relative_error"])
    axes[0].axhline(pressure_limit, color=COLORS["P"], linestyle="--", linewidth=1.0, label=f"Pressure limit ({pressure_limit:g}%)")
    axes[0].axhline(stress_limit, color=COLORS["sigma_z"], linestyle=":", linewidth=1.2, label=f"Stress limit ({stress_limit:g}%)")
    axes[0].set(ylabel="Full-field NRMSE (%)", title="(a) Four-state field accuracy")
    peaks = [
        [100.0 * row["pressure_peak_relative_error"] for row in rows],
        [100.0 * row["stress_peak_relative_error"] for row in rows],
    ]
    boxes = axes[1].boxplot(peaks, tick_labels=["Pressure peak", "Stress peak"], patch_artist=True, showfliers=False)
    for patch, key in zip(boxes["boxes"], ("P", "sigma_z")):
        patch.set(facecolor=COLORS[key], alpha=0.58)
    for index, samples in enumerate(peaks, start=1):
        axes[1].scatter(index + np.linspace(-0.12, 0.12, len(samples)), samples, s=16, color="#303030", alpha=0.72)
    axes[1].axhline(pressure_peak_limit, color=COLORS["P"], linestyle="--", linewidth=1.0, label=f"Pressure-peak limit ({pressure_peak_limit:g}%)")
    axes[1].axhline(stress_peak_limit, color=COLORS["sigma_z"], linestyle=":", linewidth=1.2, label=f"Stress-peak limit ({stress_peak_limit:g}%)")
    axes[1].set(ylabel="Relative error (%)", title="(b) Engineering peak accuracy")
    for axis in axes:
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axis.legend(frameon=False)
    save(figure, output / "V04_formal_accuracy_summary.png")
    return {
        "case_count": len(rows),
        "maximum_pressure_nrmse": max(row["P_nrmse"] for row in rows),
        "maximum_stress_nrmse": max(row["sigma_z_nrmse"] for row in rows),
    }


def plot_peak_parity(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), gridspec_kw={"wspace": 0.30})
    definitions = [
        ("reference_pressure_peak_increment_pa", "prediction_pressure_peak_increment_pa", 1e3, r"$\Delta P_{max}$ (kPa)", "P"),
        ("reference_stress_peak_increment_pa", "prediction_stress_peak_increment_pa", 1e6, r"$|\Delta\sigma_z|_{max}$ (MPa)", "sigma_z"),
    ]
    for axis, (reference_key, prediction_key, scale, label, color_key) in zip(axes, definitions):
        reference = np.asarray([row[reference_key] / scale for row in rows])
        prediction = np.asarray([row[prediction_key] / scale for row in rows])
        lower, upper = min(reference.min(), prediction.min()), max(reference.max(), prediction.max())
        padding = max(0.04 * (upper - lower), 1e-6)
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#333333", linewidth=1.0)
        for test_class in sorted({row["test_class"] for row in rows}):
            mask = np.asarray([row["test_class"] == test_class for row in rows])
            axis.scatter(reference[mask], prediction[mask], s=38, marker=MARKERS[test_class], color=COLORS[color_key], edgecolor="white", linewidth=0.5, label=test_class)
        axis.set(xlabel=f"MOC {label}", ylabel=f"Model {label}")
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].set_title("(a) Maximum pressure increment")
    axes[1].set_title("(b) Maximum axial-stress increment")
    axes[1].legend(frameon=False)
    save(figure, output / "V10_peak_parity.png")
    return {"case_count": len(rows)}


def plot_time_location_parity(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.0), gridspec_kw={"wspace": 0.30, "hspace": 0.30})
    definitions = [
        ("reference_pressure_first_peak_time_s", "prediction_pressure_first_peak_time_s", "Pressure first-peak time (s)"),
        ("reference_stress_first_peak_time_s", "prediction_stress_first_peak_time_s", "Stress first-peak time (s)"),
        ("reference_pressure_critical_location_over_L", "prediction_pressure_critical_location_over_L", r"Pressure critical location $x/L$"),
        ("reference_stress_critical_location_over_L", "prediction_stress_critical_location_over_L", r"Stress critical location $x/L$"),
    ]
    for axis, (reference_key, prediction_key, label) in zip(axes.ravel(), definitions):
        reference = np.asarray([row[reference_key] for row in rows])
        prediction = np.asarray([row[prediction_key] for row in rows])
        lower, upper = min(reference.min(), prediction.min()), max(reference.max(), prediction.max())
        padding = max(0.04 * (upper - lower), 1e-5)
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#333333", linewidth=1.0)
        axis.scatter(reference, prediction, s=32, color="#4C78A8", edgecolor="white", linewidth=0.5)
        axis.set(xlabel=f"MOC {label}", ylabel=f"Model {label}")
        axis.grid(alpha=0.22, linewidth=0.6)
    save(figure, output / "V11_peak_time_and_location_parity.png")
    return {"case_count": len(rows)}


def load_model(config_path: Path, checkpoint_path: Path, device_name: str):
    hybrid = load_hybrid_config(config_path)
    base = load_base_model_config(ROOT / hybrid["base_model_config"])
    design = load_design(base)
    baseline = load_baseline(design)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    model = ThreeParameterBoundaryModel(design, base, baseline).to(device)
    model.load_state_dict(load_checkpoint(checkpoint_path, device)["state_dict"])
    return hybrid, design, model, device


def case_field(model, hybrid: dict[str, Any], case: dict[str, Any], device: torch.device):
    path = ROOT / hybrid["reference_root"] / case["split"] / case["case_id"] / "truth_evaluation_grid.npz"
    with np.load(path, allow_pickle=False) as archive:
        x, t = archive["x"], archive["t"]
        reference = {state: archive[state] for state in STATE_NAMES}
    prediction = predict_field(model, case, x, t, device, int(hybrid["evaluation"]["batch_size"]))
    return x, t, reference, prediction


def plot_valve_history(model, hybrid: dict[str, Any], case: dict[str, Any], device: torch.device, output: Path, tag: str) -> dict[str, Any]:
    x, t, reference, prediction = case_field(model, hybrid, case, device)
    params, _ = case_parameters(model.baseline, case, model.t_final_s)
    pressure0 = initial_pressure_pa(params)
    area_f, area_t = cross_section_areas(params)
    stress0 = area_f * pressure0 / area_t
    transformations = {
        "V": (1.0, 0.0, r"$V$ (m/s)"), "uz": (1e3, 0.0, r"$u_z$ (mm/s)"),
        "P": (1e-3, pressure0, r"$\Delta P$ (kPa)"),
        "sigma_z": (1e-6, stress0, r"$\Delta\sigma_z$ (MPa)"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.8), sharex=True, gridspec_kw={"wspace": 0.28, "hspace": 0.24})
    for axis, state in zip(axes.ravel(), STATE_NAMES):
        scale, initial, ylabel = transformations[state]
        axis.plot(t, (reference[state][-1] - initial) * scale, color=COLORS["reference"], linewidth=1.2, label="MOC")
        axis.plot(t, (prediction[state][-1] - initial) * scale, color=COLORS["prediction"], linewidth=1.0, linestyle="--", label="Model")
        axis.axvline(params.valve_close_time_s, color="#888888", linewidth=0.8, linestyle=":")
        axis.set(ylabel=ylabel)
        axis.grid(alpha=0.20, linewidth=0.5)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(f"Valve-end four-state history: {case['case_id']}")
    save(figure, output / f"V06_{tag}_{case['case_id']}_valve_history.png")
    return {"case_id": case["case_id"], "x_points": len(x), "time_points": len(t)}


def plot_field_triptych(model, hybrid: dict[str, Any], case: dict[str, Any], device: torch.device, output: Path, state: str, tag: str) -> dict[str, Any]:
    x, t, reference, prediction = case_field(model, hybrid, case, device)
    params, _ = case_parameters(model.baseline, case, model.t_final_s)
    pressure0 = initial_pressure_pa(params)
    area_f, area_t = cross_section_areas(params)
    initial = pressure0 if state == "P" else area_f * pressure0 / area_t
    scale = 1e-3 if state == "P" else 1e-6
    unit = "kPa" if state == "P" else "MPa"
    reference_delta = (reference[state] - initial) * scale
    prediction_delta = (prediction[state] - initial) * scale
    error = prediction_delta - reference_delta
    common = max(float(np.max(np.abs(reference_delta))), float(np.max(np.abs(prediction_delta))), 1e-12)
    error_limit = max(float(np.max(np.abs(error))), 1e-12)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True, gridspec_kw={"wspace": 0.18})
    images = [
        axes[0].pcolormesh(t, x / params.length_m, reference_delta, shading="auto", cmap="RdBu_r", vmin=-common, vmax=common),
        axes[1].pcolormesh(t, x / params.length_m, prediction_delta, shading="auto", cmap="RdBu_r", vmin=-common, vmax=common),
        axes[2].pcolormesh(t, x / params.length_m, error, shading="auto", cmap="RdBu_r", vmin=-error_limit, vmax=error_limit),
    ]
    for axis, title in zip(axes, ("(a) MOC reference", "(b) Model prediction", "(c) Prediction - MOC")):
        axis.set(xlabel="Time (s)", title=title)
    axes[0].set_ylabel(r"Position $x/L$")
    figure.colorbar(images[0], ax=axes[:2], pad=0.02, label=f"Response increment ({unit})")
    figure.colorbar(images[2], ax=axes[2], pad=0.02, label=f"Prediction error ({unit})")
    figure.suptitle(f"{state} full-field comparison: {case['case_id']}", y=1.03)
    save(figure, output / f"V0{8 if state == 'P' else 9}_{tag}_{case['case_id']}_{state}_field.png")
    return {"case_id": case["case_id"], "maximum_absolute_error": error_limit, "unit": unit}


def select_cases(rows: list[dict[str, Any]], design: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    score = np.asarray([row["P_nrmse"] / 0.02 + row["sigma_z_nrmse"] / 0.03 for row in rows])
    typical_row = rows[int(np.argmin(np.abs(score - np.median(score))))]
    worst_row = rows[int(np.argmax(score))]
    cases = {case["case_id"]: case for case in materialize_cases(design, "formal")}
    return cases[typical_row["case_id"]], cases[worst_row["case_id"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-config", type=Path, default=DEFAULT_HYBRID_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite figure output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    prepare_style()
    rows = read_rows(args.metrics)
    hybrid, design, model, device = load_model(args.hybrid_config, args.checkpoint, args.device)
    typical, worst = select_cases(rows, design)
    figures = {
        "formal_design": plot_formal_design(design, args.output_dir),
        "accuracy": plot_accuracy(rows, hybrid["acceptance"], args.output_dir),
        "peak_parity": plot_peak_parity(rows, args.output_dir),
        "time_location_parity": plot_time_location_parity(rows, args.output_dir),
        "typical_history": plot_valve_history(model, hybrid, typical, device, args.output_dir, "typical"),
        "worst_history": plot_valve_history(model, hybrid, worst, device, args.output_dir, "worst"),
        "typical_pressure_field": plot_field_triptych(model, hybrid, typical, device, args.output_dir, "P", "typical"),
        "typical_stress_field": plot_field_triptych(model, hybrid, typical, device, args.output_dir, "sigma_z", "typical"),
        "worst_pressure_field": plot_field_triptych(model, hybrid, worst, device, args.output_dir, "P", "worst"),
        "worst_stress_field": plot_field_triptych(model, hybrid, worst, device, args.output_dir, "sigma_z", "worst"),
    }
    report = {"status": "pass", "device": str(device), "figures": figures}
    (args.output_dir / "figure_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
