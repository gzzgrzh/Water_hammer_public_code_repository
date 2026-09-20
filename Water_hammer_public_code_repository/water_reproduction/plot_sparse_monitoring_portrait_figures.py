"""Replot F47, F48, and F50 for portrait manuscript pages.

This script is inference-only.  It reads the completed three-model checkpoints
and sealed test outputs, then writes new portrait figures without modifying the
original results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from water16_reproduction.common.physics import cross_section_areas, initial_pressure_pa
from water16_reproduction.parametric_fssi_reference import case_parameters, load_baseline
from water16_reproduction import revised_sparse_monitoring_three_model as source_module


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "PINN_FSSI_research_plan/outputs/revised_sparse_monitoring_three_model_v1"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/revised_sparse_monitoring_portrait_v3"


def registered_configs(source: Path) -> dict[tuple[str, int, str], Path]:
    configs: dict[tuple[str, int, str], Path] = {}
    for condition in source_module.CONDITIONS:
        for seed in source_module.SEEDS:
            for model_name in source_module.MODELS:
                path = source / "registered_configs" / f"{model_name}__{condition}__seed{seed}.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                configs[(condition, seed, model_name)] = path
    return configs


def portrait_style() -> None:
    source_module.figure_style()
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "lines.linewidth": 1.8,
        }
    )


def save_new_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / f"{stem}.png", output / f"{stem}.pdf"]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"portrait output already exists for {stem}")
    figure.savefig(targets[0], dpi=320, bbox_inches="tight")
    figure.savefig(targets[1], bbox_inches="tight")
    plt.close(figure)


def plot_peak_parity(source: Path, output: Path) -> None:
    portrait_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 9.6))
    quantities = (
        (
            "reference_pressure_peak_increment_pa",
            "prediction_pressure_peak_increment_pa",
            "Pressure peak",
        ),
        (
            "reference_stress_peak_increment_pa",
            "prediction_stress_peak_increment_pa",
            "Axial-stress peak",
        ),
    )
    legend_handles = None
    legend_labels = None
    for row_index, condition in enumerate(source_module.CONDITIONS):
        for column_index, (reference_key, prediction_key, title) in enumerate(quantities):
            axis = axes[row_index, column_index]
            all_values: list[float] = []
            for model_name in source_module.MODELS:
                by_seed: list[dict[str, float]] = []
                case_order: list[str] = []
                references: dict[str, float] = {}
                for seed in source_module.SEEDS:
                    rows = source_module.read_case_rows(
                        source
                        / "test_release"
                        / condition
                        / f"seed_{seed}"
                        / model_name
                        / "test_case_metrics.csv"
                    )
                    if not case_order:
                        case_order = [row["case_id"] for row in rows]
                    by_seed.append({row["case_id"]: float(row[prediction_key]) / 1e6 for row in rows})
                    references = {row["case_id"]: float(row[reference_key]) / 1e6 for row in rows}
                actual = np.asarray([references[case] for case in case_order])
                predicted = np.asarray([[values[case] for case in case_order] for values in by_seed])
                mean = predicted.mean(axis=0)
                std = predicted.std(axis=0)
                axis.errorbar(
                    actual,
                    mean,
                    yerr=std,
                    fmt="o",
                    ms=4.5,
                    capsize=2.2,
                    color=source_module.MODEL_COLORS[model_name],
                    alpha=0.86,
                    label=source_module.MODEL_LABELS[model_name],
                )
                all_values.extend(actual.tolist() + mean.tolist())
            low, high = min(all_values), max(all_values)
            margin = 0.05 * max(high - low, abs(high), 1.0)
            axis.plot(
                [low - margin, high + margin],
                [low - margin, high + margin],
                "--",
                color="#555555",
                lw=1.2,
            )
            axis.set_xlim(low - margin, high + margin)
            axis.set_ylim(low - margin, high + margin)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.25)
            axis.set_xlabel("MOC reference (MPa)")
            axis.set_ylabel("Prediction (MPa)")
            condition_label = "Clean data" if condition == "clean" else "3% noisy data"
            axis.set_title(f"{condition_label}\n{title}", fontweight="bold")
            if legend_handles is None:
                legend_handles, legend_labels = axis.get_legend_handles_labels()
    fig.suptitle("Prediction of engineering peak responses", fontsize=14, fontweight="bold", y=0.985)
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.943), ncol=3)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.07, top=0.82, wspace=0.32, hspace=0.42)
    save_new_figure(fig, output, "F47_engineering_peak_prediction_parity_portrait")


def plot_histories(
    source: Path,
    output: Path,
    registered: dict[tuple[str, int, str], Path],
    models: dict[str, tuple[Any, Any]],
    design: dict[str, Any],
    device: torch.device,
) -> None:
    portrait_style()
    condition = "clean"
    cases = source_module.representative_cases(design)
    reference_config = source_module.read_json(registered[(condition, source_module.PRIMARY_SEED, "ann")])
    baseline = load_baseline(design)
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 10.7), sharex="col")
    legend_handles = None
    legend_labels = None
    with torch.inference_mode():
        for row, case in enumerate(cases):
            x, t, reference = source_module.load_reference(reference_config, case)
            params, _ = case_parameters(
                baseline,
                case,
                float(design["truth_solver"]["formal"]["t_final_s"]),
            )
            p0 = initial_pressure_pa(params)
            area_f, area_t = cross_section_areas(params)
            s0 = area_f * p0 / area_t
            stress_delta = np.abs(reference["sigma_z"] - s0)
            stress_location = int(np.unravel_index(int(np.argmax(stress_delta)), stress_delta.shape)[0])
            pressure_axis, stress_axis = axes[row]
            pressure_axis.plot(
                t,
                (reference["P"][-1] - p0) / 1e6,
                color=source_module.MODEL_COLORS["reference"],
                label="MOC reference",
            )
            stress_axis.plot(
                t,
                (reference["sigma_z"][stress_location] - s0) / 1e6,
                color=source_module.MODEL_COLORS["reference"],
                label="MOC reference",
            )
            for model_name, (model, predictor) in models.items():
                prediction = predictor(model, case, x, t, device, 8192)
                pressure_axis.plot(
                    t,
                    (prediction["P"][-1] - p0) / 1e6,
                    color=source_module.MODEL_COLORS[model_name],
                    label=source_module.MODEL_LABELS[model_name],
                    alpha=0.9,
                )
                stress_axis.plot(
                    t,
                    (prediction["sigma_z"][stress_location] - s0) / 1e6,
                    color=source_module.MODEL_COLORS[model_name],
                    label=source_module.MODEL_LABELS[model_name],
                    alpha=0.9,
                )
            class_label = {
                "interpolation": "Interpolation",
                "boundary": "Parameter-boundary",
                "combination": "Combined",
            }[case["test_class"]]
            pressure_axis.set_title(f"{class_label} case - valve pressure", fontweight="bold")
            stress_axis.set_title(
                f"{class_label} case - axial stress\nat x/L={x[stress_location] / params.length_m:.3f}",
                fontweight="bold",
            )
            pressure_axis.set_ylabel("Pressure increment (MPa)")
            stress_axis.set_ylabel("Axial-stress increment (MPa)")
            for axis in (pressure_axis, stress_axis):
                axis.grid(alpha=0.22)
            if legend_handles is None:
                legend_handles, legend_labels = pressure_axis.get_legend_handles_labels()
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Predicted response histories - clean sparse monitoring", fontsize=14, fontweight="bold", y=0.992)
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.947), ncol=4)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.06, top=0.865, wspace=0.30, hspace=0.43)
    save_new_figure(fig, output, "F48_predicted_histories_clean_portrait")


def plot_spatial_profiles(
    source: Path,
    output: Path,
    registered: dict[tuple[str, int, str], Path],
    models: dict[str, tuple[Any, Any]],
    design: dict[str, Any],
    device: torch.device,
) -> None:
    portrait_style()
    condition = "clean"
    case = source_module.representative_cases(design)[-1]
    config = source_module.read_json(registered[(condition, source_module.PRIMARY_SEED, "ann")])
    x, t, reference = source_module.load_reference(config, case)
    baseline = load_baseline(design)
    params, fluid_speed = case_parameters(
        baseline,
        case,
        float(design["truth_solver"]["formal"]["t_final_s"]),
    )
    p0 = initial_pressure_pa(params)
    area_f, area_t = cross_section_areas(params)
    s0 = area_f * p0 / area_t
    travel = params.length_m / fluid_speed
    target_times = [
        params.valve_close_time_s,
        params.valve_close_time_s + travel,
        params.valve_close_time_s + 2 * travel,
    ]
    indices = [int(np.argmin(np.abs(t - min(value, t[-1])))) for value in target_times]
    with torch.inference_mode():
        predictions = {
            name: predictor(model, case, x, t, device, 8192)
            for name, (model, predictor) in models.items()
        }
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 10.5), sharex="col")
    legend_handles = None
    legend_labels = None
    for row, index in enumerate(indices):
        pressure_axis, stress_axis = axes[row]
        pressure_axis.plot(
            x / params.length_m,
            (reference["P"][:, index] - p0) / 1e6,
            color=source_module.MODEL_COLORS["reference"],
            label="MOC reference",
        )
        stress_axis.plot(
            x / params.length_m,
            (reference["sigma_z"][:, index] - s0) / 1e6,
            color=source_module.MODEL_COLORS["reference"],
            label="MOC reference",
        )
        for model_name in source_module.MODELS:
            pressure_axis.plot(
                x / params.length_m,
                (predictions[model_name]["P"][:, index] - p0) / 1e6,
                color=source_module.MODEL_COLORS[model_name],
                label=source_module.MODEL_LABELS[model_name],
            )
            stress_axis.plot(
                x / params.length_m,
                (predictions[model_name]["sigma_z"][:, index] - s0) / 1e6,
                color=source_module.MODEL_COLORS[model_name],
                label=source_module.MODEL_LABELS[model_name],
            )
        pressure_axis.set_title(f"t={t[index]:.3f} s - pressure", fontweight="bold")
        stress_axis.set_title(f"t={t[index]:.3f} s - axial stress", fontweight="bold")
        pressure_axis.set_ylabel("Pressure increment (MPa)")
        stress_axis.set_ylabel("Axial-stress increment (MPa)")
        for axis in (pressure_axis, stress_axis):
            axis.grid(alpha=0.22)
        if legend_handles is None:
            legend_handles, legend_labels = pressure_axis.get_legend_handles_labels()
    axes[-1, 0].set_xlabel("Normalized pipe coordinate x/L")
    axes[-1, 1].set_xlabel("Normalized pipe coordinate x/L")
    fig.suptitle(
        "Full-field spatial profiles - representative combined-demand condition",
        fontsize=14,
        fontweight="bold",
        y=0.992,
    )
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.961), ncol=4)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.06, top=0.905, wspace=0.30, hspace=0.40)
    save_new_figure(fig, output, "F50_full_field_spatial_profiles_clean_portrait")


def write_progress(output: Path, completed: list[str], status: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "completed": completed, "training_started": False}
    (output / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def completed_steps(output: Path) -> list[str]:
    progress = output / "progress.json"
    if not progress.exists():
        return []
    payload = json.loads(progress.read_text(encoding="utf-8"))
    if payload.get("training_started") is not False:
        raise RuntimeError("invalid portrait-only progress file")
    return [str(step) for step in payload.get("completed", [])]


def require_step_files(output: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"checkpoint declares a missing figure: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if source == output:
        raise ValueError("source and portrait output directories must differ")
    if not (source / "three_model_sparse_monitoring_report.json").exists():
        raise FileNotFoundError("completed source report is missing")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    # The saved networks and the established prediction functions use double
    # precision.  Set it before reconstructing any model from its checkpoint.
    torch.set_default_dtype(torch.float64)
    registered = registered_configs(source)
    completed = completed_steps(output)
    valid_steps = {"F47", "F48", "F50"}
    if not set(completed).issubset(valid_steps):
        raise RuntimeError(f"unknown completed portrait steps: {completed}")
    write_progress(output, completed, "running")

    if "F47" in completed:
        require_step_files(output, "F47_engineering_peak_prediction_parity_portrait")
    else:
        plot_peak_parity(source, output)
        completed.append("F47")
        write_progress(output, completed, "running")

    models, design = source_module.primary_models(source, registered, "clean", device)
    if "F48" in completed:
        require_step_files(output, "F48_predicted_histories_clean_portrait")
    else:
        plot_histories(source, output, registered, models, design, device)
        completed.append("F48")
        write_progress(output, completed, "running")

    if "F50" in completed:
        require_step_files(output, "F50_full_field_spatial_profiles_clean_portrait")
    else:
        plot_spatial_profiles(source, output, registered, models, design, device)
        completed.append("F50")
    write_progress(output, completed, "complete")
    print((output / "progress.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
