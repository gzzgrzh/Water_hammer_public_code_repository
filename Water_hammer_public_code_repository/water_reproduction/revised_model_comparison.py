"""Create paper-ready ANN/PINN/characteristic comparison tables and figures."""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANN = ROOT / "PINN_FSSI_research_plan/outputs/revised_ann_physics_events_seed29041_v1/formal"
DEFAULT_PINN = ROOT / "PINN_FSSI_research_plan/outputs/revised_coordinate_pinn_physics_events_seed29041_v1/formal"
DEFAULT_CHARACTERISTIC = ROOT / "PINN_FSSI_research_plan/outputs/revised_characteristic_v3_physics_events_test_v1"
DEFAULT_CHARACTERISTIC_VALIDATION = ROOT / "PINN_FSSI_research_plan/outputs/parametric_fssi_hybrid_formal_v3_balanced_events/formal/held_out_case_metrics.csv"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/revised_three_model_comparison_v1"

MODEL_ORDER = ("ANN", "Coordinate PINN", "Characteristic model")
COLORS = {
    "ANN": "#888888",
    "Coordinate PINN": "#D9853B",
    "Characteristic model": "#2F6B9A",
}
DISPLAY_LABELS = {
    "ANN": "ANN",
    "Coordinate PINN": "Standard PINN",
    "Characteristic model": "Proposed model",
}
FIELD_METRICS = ("P_nrmse", "sigma_z_nrmse")
PEAK_METRICS = ("pressure_peak_relative_error", "stress_peak_relative_error")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = {
        "soil_to_pipe_modulus_ratio",
        "closure_time_over_L_cf",
        "initial_velocity_m_s",
        "V_nrmse",
        "uz_nrmse",
        "P_nrmse",
        "sigma_z_nrmse",
        "pressure_peak_relative_error",
        "stress_peak_relative_error",
        "reference_pressure_peak_increment_pa",
        "prediction_pressure_peak_increment_pa",
        "reference_stress_peak_increment_pa",
        "prediction_stress_peak_increment_pa",
        "reference_pressure_first_peak_time_s",
        "prediction_pressure_first_peak_time_s",
        "reference_stress_first_peak_time_s",
        "prediction_stress_first_peak_time_s",
        "reference_pressure_critical_location_over_L",
        "prediction_pressure_critical_location_over_L",
        "reference_stress_critical_location_over_L",
        "prediction_stress_critical_location_over_L",
    }
    for row in rows:
        for name in numeric & set(row):
            row[name] = float(row[name])
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def add_engineering_errors(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["pressure_first_peak_time_abs_error_ms"] = 1000.0 * abs(
        row["prediction_pressure_first_peak_time_s"]
        - row["reference_pressure_first_peak_time_s"]
    )
    result["stress_first_peak_time_abs_error_ms"] = 1000.0 * abs(
        row["prediction_stress_first_peak_time_s"]
        - row["reference_stress_first_peak_time_s"]
    )
    result["pressure_critical_location_abs_error_over_L"] = abs(
        row["prediction_pressure_critical_location_over_L"]
        - row["reference_pressure_critical_location_over_L"]
    )
    result["stress_critical_location_abs_error_over_L"] = abs(
        row["prediction_stress_critical_location_over_L"]
        - row["reference_stress_critical_location_over_L"]
    )
    return result


def summarize(rows_by_model: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    metrics = (
        "P_nrmse",
        "sigma_z_nrmse",
        "pressure_peak_relative_error",
        "stress_peak_relative_error",
        "pressure_first_peak_time_abs_error_ms",
        "stress_first_peak_time_abs_error_ms",
        "pressure_critical_location_abs_error_over_L",
        "stress_critical_location_abs_error_over_L",
    )
    summary = []
    for model in MODEL_ORDER:
        for split in ("validation", "test"):
            subset = [row for row in rows_by_model[model] if row["split"] == split]
            if not subset:
                continue
            output: dict[str, Any] = {
                "model": model,
                "split": split,
                "case_count": len(subset),
            }
            for metric in metrics:
                values = np.asarray([row[metric] for row in subset], dtype=float)
                output[f"{metric}_mean"] = float(np.mean(values))
                output[f"{metric}_median"] = float(np.median(values))
                output[f"{metric}_maximum"] = float(np.max(values))
            summary.append(output)
    return summary


def plot_error_boxplots(rows_by_model: dict[str, list[dict[str, Any]]], output: Path) -> None:
    metrics = (
        ("P_nrmse", "Pressure field NRMSE (%)"),
        ("sigma_z_nrmse", "Axial-stress field NRMSE (%)"),
        ("pressure_peak_relative_error", "Maximum-pressure error (%)"),
        ("stress_peak_relative_error", "Maximum-stress error (%)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        values = [
            100.0 * np.asarray(
                [row[metric] for row in rows_by_model[model] if row["split"] == "test"]
            )
            for model in MODEL_ORDER
        ]
        boxes = axis.boxplot(
            values,
            labels=[DISPLAY_LABELS[model] for model in MODEL_ORDER],
            patch_artist=True,
            showfliers=True,
        )
        for patch, model in zip(boxes["boxes"], MODEL_ORDER):
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.78)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=12)
    figure.suptitle("Prediction errors on 24 independent test cases", fontweight="bold")
    figure.tight_layout()
    save_figure(figure, output, "F11_three_model_error_boxplots")


def plot_peak_parity(rows_by_model: dict[str, list[dict[str, Any]]], output: Path) -> None:
    definitions = (
        (
            "reference_pressure_peak_increment_pa",
            "prediction_pressure_peak_increment_pa",
            "Maximum pressure increment (kPa)",
            1.0e-3,
        ),
        (
            "reference_stress_peak_increment_pa",
            "prediction_stress_peak_increment_pa",
            "Maximum axial-stress increment (MPa)",
            1.0e-6,
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.3, 4.8))
    for axis, (reference_name, prediction_name, label, scale) in zip(axes, definitions):
        all_values = []
        for model in MODEL_ORDER:
            rows = [row for row in rows_by_model[model] if row["split"] == "test"]
            reference = scale * np.asarray([row[reference_name] for row in rows])
            prediction = scale * np.asarray([row[prediction_name] for row in rows])
            all_values.extend(reference.tolist())
            all_values.extend(prediction.tolist())
            axis.scatter(
                reference,
                prediction,
                s=34,
                alpha=0.82,
                color=COLORS[model],
                label=DISPLAY_LABELS[model],
                edgecolor="white",
                linewidth=0.35,
            )
        lower, upper = min(all_values), max(all_values)
        pad = 0.05 * max(upper - lower, 1.0)
        axis.plot([lower - pad, upper + pad], [lower - pad, upper + pad], "k--", lw=1.1)
        axis.set_xlim(lower - pad, upper + pad)
        axis.set_ylim(lower - pad, upper + pad)
        axis.set_xlabel(f"MOC {label}")
        axis.set_ylabel(f"Predicted {label}")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=11)
    figure.suptitle("Predicted and MOC peak responses", fontweight="bold")
    figure.tight_layout()
    save_figure(figure, output, "F12_three_model_peak_parity")


def plot_class_performance(rows_by_model: dict[str, list[dict[str, Any]]], output: Path) -> None:
    classes = ("interpolation", "boundary", "combination")
    metrics = (("P_nrmse", "Pressure NRMSE (%)"), ("sigma_z_nrmse", "Stress NRMSE (%)"))
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    width = 0.23
    x = np.arange(len(classes))
    for axis, (metric, ylabel) in zip(axes, metrics):
        for index, model in enumerate(MODEL_ORDER):
            means = []
            maxima = []
            for case_class in classes:
                values = np.asarray(
                    [
                        100.0 * row[metric]
                        for row in rows_by_model[model]
                        if row["split"] == "test" and row["test_class"] == case_class
                    ]
                )
                means.append(float(np.mean(values)))
                maxima.append(float(np.max(values)))
            offset = (index - 1) * width
            axis.bar(
                x + offset,
                means,
                width,
                color=COLORS[model],
                label=DISPLAY_LABELS[model],
                alpha=0.82,
            )
            axis.scatter(x + offset, maxima, color=COLORS[model], marker="_", s=170, linewidth=2)
        axis.set_xticks(x, ("Interior", "Boundary", "Combination"))
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=11)
    figure.suptitle("Mean and maximum errors for the three test-case groups", fontweight="bold")
    figure.tight_layout()
    save_figure(figure, output, "F13_three_model_test_class_performance")


def plot_engineering_localization(rows_by_model: dict[str, list[dict[str, Any]]], output: Path) -> None:
    definitions = (
        ("pressure_first_peak_time_abs_error_ms", "Pressure first-peak time MAE (ms)"),
        ("stress_first_peak_time_abs_error_ms", "Stress first-peak time MAE (ms)"),
        ("pressure_critical_location_abs_error_over_L", "Pressure critical-location MAE (L)"),
        ("stress_critical_location_abs_error_over_L", "Stress critical-location MAE (L)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    for axis, (metric, ylabel) in zip(axes.ravel(), definitions):
        means = []
        maxima = []
        for model in MODEL_ORDER:
            values = np.asarray(
                [row[metric] for row in rows_by_model[model] if row["split"] == "test"]
            )
            means.append(float(np.mean(values)))
            maxima.append(float(np.max(values)))
        x = np.arange(len(MODEL_ORDER))
        axis.bar(x, means, color=[COLORS[model] for model in MODEL_ORDER], alpha=0.82)
        axis.scatter(x, maxima, color="#222222", marker="_", s=190, linewidth=2)
        axis.set_xticks(x, [DISPLAY_LABELS[model] for model in MODEL_ORDER], rotation=12)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("First-peak timing and critical-location errors (mean; maximum ticks)", fontweight="bold")
    figure.tight_layout()
    save_figure(figure, output, "F14_three_model_timing_location_errors")


def plot_training_histories(history_paths: dict[str, Path], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for model in MODEL_ORDER:
        history = json.loads(history_paths[model].read_text(encoding="utf-8"))
        epoch = np.asarray([float(row["epoch"]) for row in history])
        anchor = np.asarray([float(row["loss_training_anchor"]) for row in history])
        total = np.asarray([float(row["loss_total"]) for row in history])
        axes[0].plot(epoch, anchor, color=COLORS[model], label=DISPLAY_LABELS[model], lw=1.4)
        axes[1].plot(epoch, total, color=COLORS[model], label=DISPLAY_LABELS[model], lw=1.4)
    axes[0].set_ylabel("Sparse-label loss")
    axes[1].set_ylabel("Total training objective")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=11)
    figure.suptitle("Training histories using 384 samples per case", fontweight="bold")
    figure.tight_layout()
    save_figure(figure, output, "F15_three_model_training_histories")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite comparison output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    ann = [add_engineering_errors(row) for row in read_csv(args.ann / "held_out_case_metrics.csv")]
    coordinate = [
        add_engineering_errors(row)
        for row in read_csv(args.coordinate_pinn / "held_out_case_metrics.csv")
    ]
    characteristic_test = [
        add_engineering_errors(row) for row in read_csv(args.characteristic / "case_metrics.csv")
    ]
    characteristic_validation = [
        add_engineering_errors(row) for row in read_csv(args.characteristic_validation)
    ]
    rows_by_model = {
        "ANN": ann,
        "Coordinate PINN": coordinate,
        "Characteristic model": characteristic_validation + characteristic_test,
    }
    long_rows = []
    for model in MODEL_ORDER:
        for row in rows_by_model[model]:
            long_rows.append({"model": model, **row})
    write_csv(args.output_dir / "T05_three_model_case_metrics.csv", long_rows)
    summary = summarize(rows_by_model)
    write_csv(args.output_dir / "T06_three_model_summary.csv", summary)
    plot_error_boxplots(rows_by_model, args.output_dir)
    plot_peak_parity(rows_by_model, args.output_dir)
    plot_class_performance(rows_by_model, args.output_dir)
    plot_engineering_localization(rows_by_model, args.output_dir)
    plot_training_histories(
        {
            "ANN": args.ann / "history.json",
            "Coordinate PINN": args.coordinate_pinn / "history.json",
            "Characteristic model": args.characteristic_history,
        },
        args.output_dir,
    )
    test_summary = {row["model"]: row for row in summary if row["split"] == "test"}
    characteristic = test_summary["Characteristic model"]
    relative_reduction = {}
    for baseline in ("ANN", "Coordinate PINN"):
        relative_reduction[baseline] = {
            metric: 1.0 - characteristic[f"{metric}_mean"] / test_summary[baseline][f"{metric}_mean"]
            for metric in (*FIELD_METRICS, *PEAK_METRICS)
        }
    report = {
        "status": "pass",
        "comparison_scope": "same 72 training cases and 384 non-oracle physics-event state vectors per case",
        "test_case_count": 24,
        "models": list(MODEL_ORDER),
        "test_summary": test_summary,
        "relative_mean_error_reduction_of_characteristic_model": relative_reduction,
        "figures": sorted(path.name for path in args.output_dir.glob("F*.png")),
        "tables": sorted(path.name for path in args.output_dir.glob("T*.csv")),
        "caveat": "The characteristic checkpoint includes its registered staged optimization history; label and final-stage epoch budgets are aligned, but cumulative optimization cost is reported separately.",
    }
    (args.output_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", type=Path, default=DEFAULT_ANN)
    parser.add_argument("--coordinate-pinn", type=Path, default=DEFAULT_PINN)
    parser.add_argument("--characteristic", type=Path, default=DEFAULT_CHARACTERISTIC)
    parser.add_argument(
        "--characteristic-validation", type=Path, default=DEFAULT_CHARACTERISTIC_VALIDATION
    )
    parser.add_argument(
        "--characteristic-history",
        type=Path,
        default=ROOT / "PINN_FSSI_research_plan/outputs/parametric_fssi_hybrid_formal_v3_balanced_events/formal/history.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
