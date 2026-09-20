"""Pilot realistic sparse-monitoring supervision for the characteristic FSSI surrogate.

This resumable pilot changes only which state components are visible at the
registered 8 x 48 anchor locations and whether deterministic measurement noise
is added.  Validation and test fields are never used as labels.  The existing
36-case, four-state clean run is reused as the baseline; no baseline training
is repeated.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from water16_reproduction.parametric_fssi_hybrid import (
    load_config,
    run_training,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/revised_sparse_monitoring_pilot_v1"
SOURCE_CONFIG = (
    ROOT
    / "PINN_FSSI_research_plan/outputs/revised_ablation_matrix_v1/registered_configs"
    / "characteristic__labels_384__cases_36__sampling_event.json"
)
BASELINE_REPORT = (
    ROOT
    / "PINN_FSSI_research_plan/outputs/revised_ablation_matrix_v1/runs"
    / "labels_384__cases_36__sampling_event/characteristic/formal/model_report.json"
)

CONDITIONS = (
    ("pressure_stress_clean", ("P", "sigma_z"), 0.00),
    ("pressure_only_clean", ("P",), 0.00),
    ("pressure_stress_noise03", ("P", "sigma_z"), 0.03),
    ("pressure_only_noise03", ("P",), 0.03),
)

DISPLAY = {
    "four_state_clean": "4-state\nclean",
    "pressure_stress_clean": "P + stress\nclean",
    "pressure_only_clean": "P only\nclean",
    "pressure_stress_noise03": "P + stress\n3% noise",
    "pressure_only_noise03": "P only\n3% noise",
}

METRICS = (
    ("V_nrmse", "Fluid velocity NRMSE (%)"),
    ("uz_nrmse", "Pipe velocity NRMSE (%)"),
    ("P_nrmse", "Pressure NRMSE (%)"),
    ("sigma_z_nrmse", "Axial stress NRMSE (%)"),
    ("pressure_peak_relative_error", "Pressure-peak error (%)"),
    ("stress_peak_relative_error", "Stress-peak error (%)"),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        registered = read_json(path)
        if registered != payload:
            raise RuntimeError(f"registered configuration changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def make_config(
    source: dict[str, Any],
    slug: str,
    observed_states: tuple[str, ...],
    noise_fraction: float,
    path: Path,
) -> dict[str, Any]:
    config = deepcopy(source)
    config["model_id"] = f"revised_sparse_monitoring_{slug}_seed29041_v1"
    config["status"] = "registered_before_sparse_monitoring_pilot"
    config["registration_reason"] = (
        "Realistic monitoring pilot: fixed characteristic architecture, 36 training cases, "
        "8 x 48 event-aligned anchor locations, and deterministic component-wise visibility."
    )
    config["observation_policy"] = {
        "observed_states": list(observed_states),
        "noise_std_fraction_of_dynamic_scale": noise_fraction,
        "noise_seed": 29041,
        "noise_distribution": "independent Gaussian on observed components only",
        "unobserved_states_forbidden_from_anchor_loss": True,
    }
    vectors = int(config["anchor_policy"]["anchor_vectors_per_case"])
    cases = int(config["training"]["formal"]["case_limit"])
    config["anchor_policy"]["total_scalar_labels"] = vectors * cases * len(observed_states)
    config["monitoring_pilot_registration"] = {
        "condition": slug,
        "config_snapshot": relative(path),
        "baseline_retrained": False,
        "validation_and_test_labels_forbidden": True,
    }
    return config


def metric_row(slug: str, states: list[str], noise: float, report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    evaluation = report["evaluation"]
    if "validation" in evaluation:
        evaluation = evaluation["validation"]
    row: dict[str, Any] = {
        "condition": slug,
        "observed_states": "+".join(states),
        "noise_std_fraction": noise,
        "training_seconds": float(report.get("training_seconds", 0.0)),
        "report_path": relative(report_path),
        "validation_case_count": int(evaluation["case_count"]),
    }
    for metric, _ in METRICS:
        row[f"{metric}_mean"] = float(evaluation["mean_by_metric"][metric])
        row[f"{metric}_maximum"] = float(evaluation["maximum_by_metric"][metric])
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 12.5,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11,
        }
    )
    colors = ["#606060", "#2F6B9A", "#5B9BD5", "#D9853B", "#E9A46A"]
    x = np.arange(len(rows))
    labels = [DISPLAY[row["condition"]] for row in rows]
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), constrained_layout=True)
    for axis, (metric, title) in zip(axes.ravel(), METRICS):
        values = np.asarray([100.0 * float(row[f"{metric}_mean"]) for row in rows])
        bars = axis.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    figure.suptitle(
        "Validation accuracy under realistic state visibility and measurement noise",
        fontsize=15,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"F45_sparse_monitoring_observability_noise_pilot.{suffix}", dpi=320 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)


def plot_relative_heatmap(rows: list[dict[str, Any]], output: Path) -> None:
    baseline = rows[0]
    matrix = np.asarray(
        [
            [float(row[f"{metric}_mean"]) / max(float(baseline[f"{metric}_mean"]), 1.0e-12) for metric, _ in METRICS]
            for row in rows[1:]
        ]
    )
    figure, axis = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.8, vmax=max(3.0, float(np.max(matrix))))
    axis.set_xticks(np.arange(len(METRICS)), [title.replace(" (%)", "") for _, title in METRICS], rotation=22, ha="right")
    axis.set_yticks(np.arange(len(rows) - 1), [DISPLAY[row["condition"]].replace("\n", " ") for row in rows[1:]])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{matrix[i, j]:.2f}x", ha="center", va="center", fontsize=11, color="black")
    axis.set_title("Error ratio relative to four-state clean supervision", fontsize=14, fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Mean-error ratio")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"F46_sparse_monitoring_error_retention_heatmap.{suffix}", dpi=320 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    final_report = output / "sparse_monitoring_pilot_report.json"
    if final_report.exists():
        print(final_report.read_text(encoding="utf-8"), flush=True)
        return

    source = read_json(SOURCE_CONFIG)
    baseline_report = read_json(BASELINE_REPORT)
    rows = [
        metric_row(
            "four_state_clean",
            ["V", "uz", "P", "sigma_z"],
            0.0,
            baseline_report,
            BASELINE_REPORT,
        )
    ]
    progress_path = output / "progress.json"
    for index, (slug, states, noise) in enumerate(CONDITIONS, start=1):
        config_path = output / "registered_configs" / f"{slug}.json"
        config = make_config(source, slug, states, noise, config_path)
        stable_write_json(config_path, config)
        run_root = output / "runs" / slug
        print(f"[pilot] condition {index}/{len(CONDITIONS)}: {slug}", flush=True)
        report = run_training(load_config(config_path), "formal", run_root)
        report_path = run_root / "formal/model_report.json"
        rows.append(metric_row(slug, list(states), noise, report, report_path))
        progress_path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "completed_conditions": [row["condition"] for row in rows[1:]],
                    "next_condition_index": index + 1,
                    "total_conditions": len(CONDITIONS),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[pilot] checkpointed {slug}", flush=True)

    write_csv(output / "T15_sparse_monitoring_observability_noise_pilot.csv", rows)
    plot_metrics(rows, output)
    plot_relative_heatmap(rows, output)
    payload = {
        "status": "pass",
        "study_scope": "characteristic-model pilot on validation cases; no sealed test access",
        "baseline_retrained": False,
        "training_cases": 36,
        "anchor_locations_per_case": 384,
        "spatial_locations_per_case": 8,
        "time_locations_per_spatial_point": 48,
        "conditions": [row["condition"] for row in rows],
        "new_training_conditions": len(CONDITIONS),
        "validation_or_test_labels_used": False,
        "sealed_test_read": False,
        "table": "T15_sparse_monitoring_observability_noise_pilot.csv",
        "figures": [
            "F45_sparse_monitoring_observability_noise_pilot",
            "F46_sparse_monitoring_error_retention_heatmap",
        ],
    }
    final_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    progress_path.write_text(
        json.dumps({"status": "complete", "completed_conditions": list(payload["conditions"])[1:]}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
