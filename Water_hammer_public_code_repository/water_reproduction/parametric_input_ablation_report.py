"""Summarize parameter-input ablations and plot held-out validation traces."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from water16_reproduction.parametric_fssi_forward import materialize_cases
from water16_reproduction.parametric_fssi_model import (
    ThreeParameterBoundaryModel,
    load_config as load_base_model_config,
    load_design,
    predict_field,
)
from water16_reproduction.parametric_fssi_reference import load_baseline


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/parametric_fssi_input_ablation_report_v1"
REFERENCE_ROOT = ROOT / "PINN_FSSI_research_plan/outputs/parametric_fssi_reference_v1/formal"

RUNS = (
    (
        "Full conditioning",
        "PINN_FSSI_research_plan/configs/parametric_fssi_hybrid_formal_v3_balanced_events.json",
        "PINN_FSSI_research_plan/outputs/parametric_fssi_hybrid_formal_v3_balanced_events/formal",
    ),
    (
        r"Without raw $E_s/E$",
        "PINN_FSSI_research_plan/configs/parametric_fssi_input_ablation_no_soil_ratio_raw_v1.json",
        "PINN_FSSI_research_plan/outputs/parametric_fssi_input_ablation_no_soil_ratio_raw_v1/formal",
    ),
    (
        r"Without raw $t_c/(L/c_f)$",
        "PINN_FSSI_research_plan/configs/parametric_fssi_input_ablation_no_closure_ratio_raw_v1.json",
        "PINN_FSSI_research_plan/outputs/parametric_fssi_input_ablation_no_closure_ratio_raw_v1/formal",
    ),
    (
        r"Without raw $V_0$",
        "PINN_FSSI_research_plan/configs/parametric_fssi_input_ablation_no_initial_velocity_raw_v1.json",
        "PINN_FSSI_research_plan/outputs/parametric_fssi_input_ablation_no_initial_velocity_raw_v1/formal",
    ),
    (
        "Parameters in equations only",
        "PINN_FSSI_research_plan/configs/parametric_fssi_input_ablation_equations_only_v1.json",
        "PINN_FSSI_research_plan/outputs/parametric_fssi_input_ablation_equations_only_v1/formal",
    ),
)

COLORS = ("#2F5597", "#59A14F", "#F28E2B", "#AF7AA1", "#E15759")
MARKERS = ("o", "s", "^", "D", "X")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "held_out_case_metrics.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def summary_records() -> list[dict[str, Any]]:
    records = []
    for label, config_path, run_path in RUNS:
        report = read_json(ROOT / run_path / "model_report.json")
        mean = report["evaluation"]["mean_by_metric"]
        maximum = report["evaluation"]["maximum_by_metric"]
        records.append(
            {
                "label": label.replace("$", ""),
                "config": config_path,
                "run_dir": run_path,
                "training_minutes": float(report["training_seconds"]) / 60.0,
                "training_loss": float(report["final_training_loss"]["loss_total"]),
                "P_nrmse_mean_percent": 100.0 * float(mean["P_nrmse"]),
                "P_nrmse_max_percent": 100.0 * float(maximum["P_nrmse"]),
                "sigma_nrmse_mean_percent": 100.0 * float(mean["sigma_z_nrmse"]),
                "sigma_nrmse_max_percent": 100.0 * float(maximum["sigma_z_nrmse"]),
                "P_peak_mean_percent": 100.0 * float(mean["pressure_peak_relative_error"]),
                "sigma_peak_mean_percent": 100.0 * float(mean["stress_peak_relative_error"]),
            }
        )
    return records


def write_summary(records: list[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "input_ablation_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    full = records[0]
    report = {
        "status": "complete",
        "training_performed_by_this_report_script": False,
        "comparison_scope": "12 formal validation cases",
        "fixed_controls": [
            "72 training cases",
            "384 four-state samples per training case",
            "800 epochs",
            "same architecture, loss weights, seed and warm-start checkpoint",
            "engineering parameters remain active in the physical equations for every ablation",
        ],
        "records": records,
        "relative_increase_over_full": {
            row["label"]: {
                metric: row[metric] / full[metric] - 1.0
                for metric in (
                    "P_nrmse_mean_percent",
                    "sigma_nrmse_mean_percent",
                    "P_nrmse_max_percent",
                    "sigma_nrmse_max_percent",
                )
            }
            for row in records[1:]
        },
        "interpretation": [
            "The explicit soil-ratio coordinate gives a modest improvement because soil stiffness also enters the characteristic speeds and reconstruction.",
            "Removing the explicit closure-time coordinate substantially degrades pressure and stress prediction even though closure time remains in the valve law and closure-phase features.",
            "Removing the initial-velocity coordinate is especially damaging to axial-stress and stress-peak prediction.",
            "Using engineering parameters only in the equations is insufficient for one shared surrogate over the full parameter domain.",
        ],
    }
    (output_dir / "input_ablation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def plot_summary(records: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [row["label"] for row in records]
    labels = [
        "Full",
        r"No raw $E_s/E$",
        r"No raw $t_c/(L/c_f)$",
        r"No raw $V_0$",
        "Equations only",
    ]
    x = np.arange(len(records))
    width = 0.35
    figure, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), gridspec_kw={"wspace": 0.24})
    for axis, suffix, title in zip(
        axes,
        ("mean_percent", "max_percent"),
        ("Mean over 12 validation cases", "Maximum over 12 validation cases"),
    ):
        p_values = [row[f"P_nrmse_{suffix}"] for row in records]
        s_values = [row[f"sigma_nrmse_{suffix}"] for row in records]
        axis.bar(x - width / 2, p_values, width, color="#4C78A8", label="Pressure")
        axis.bar(x + width / 2, s_values, width, color="#E15759", label="Axial stress")
        axis.set_xticks(x, labels, rotation=17, ha="right")
        axis.set_ylabel("NRMSE (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        for index, value in enumerate(p_values):
            axis.text(index - width / 2, value + 0.16, f"{value:.2f}", ha="center", fontsize=9)
        for index, value in enumerate(s_values):
            axis.text(index + width / 2, value + 0.16, f"{value:.2f}", ha="center", fontsize=9)
    axes[0].legend(frameon=False)
    figure.savefig(output_dir / "I01_parameter_input_ablation_NRMSE.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_training(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    for (label, _, run_path), color in zip(RUNS, COLORS):
        history = read_json(ROOT / run_path / "history.json")
        epochs = [row["epoch"] for row in history]
        losses = [row["loss_total"] for row in history]
        axis.semilogy(epochs, losses, color=color, linewidth=1.8, label=label)
    axis.set(xlabel="Epoch", ylabel="Weighted training loss", title="Convergence under identical training settings")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=10)
    figure.savefig(output_dir / "I02_parameter_input_ablation_convergence.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_case_heatmap(output_dir: Path) -> None:
    case_ids = [row["case_id"] for row in load_rows(ROOT / RUNS[0][2])]
    matrices = []
    for metric in ("P_nrmse", "sigma_z_nrmse"):
        matrix = []
        for _, _, run_path in RUNS:
            rows = load_rows(ROOT / run_path)
            matrix.append([100.0 * float(row[metric]) for row in rows])
        matrices.append(np.asarray(matrix))
    figure, axes = plt.subplots(2, 1, figsize=(14.2, 6.8), gridspec_kw={"hspace": 0.42})
    short_cases = [case_id.rsplit("_", 1)[-1] for case_id in case_ids]
    model_labels = ["Full", r"No $E_s/E$", r"No $t_c$", r"No $V_0$", "Equations only"]
    for axis, matrix, title in zip(axes, matrices, ("Pressure NRMSE (%)", "Axial-stress NRMSE (%)")):
        image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0)
        axis.set_xticks(np.arange(len(short_cases)), short_cases)
        axis.set_yticks(np.arange(len(model_labels)), model_labels)
        axis.set_xlabel("Validation case number")
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(column_index, row_index, f"{value:.1f}", ha="center", va="center", fontsize=8, color="white" if value > 0.55 * matrix.max() else "black")
        figure.colorbar(image, ax=axis, pad=0.012, fraction=0.025)
    figure.savefig(output_dir / "I03_parameter_input_ablation_case_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def load_model(config_path: str, run_path: str, device: torch.device):
    hybrid_config = read_json(ROOT / config_path)
    base_config = copy.deepcopy(load_base_model_config(ROOT / hybrid_config["base_model_config"]))
    if "network_input_ablation" in hybrid_config:
        base_config["network"]["input_ablation"] = copy.deepcopy(hybrid_config["network_input_ablation"])
    design = load_design(base_config)
    model = ThreeParameterBoundaryModel(design, base_config, load_baseline(design)).to(device)
    checkpoint = torch.load(ROOT / run_path / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, design


def plot_trace(case_id: str, output_dir: Path, device: torch.device) -> None:
    models = []
    design = None
    for label, config_path, run_path in RUNS:
        model, design = load_model(config_path, run_path, device)
        models.append((label, model))
    assert design is not None
    cases = {case["case_id"]: case for case in materialize_cases(design, "formal")}
    case = cases[case_id]
    reference_path = REFERENCE_ROOT / "validation" / case_id / "truth_evaluation_grid.npz"
    with np.load(reference_path) as data:
        x = np.asarray(data["x"])
        t = np.asarray(data["t"])
        index = int(np.argmin(np.abs(x / x[-1] - 0.85625)))
        selected_x = np.asarray([x[index]])
        reference = {name: np.asarray(data[name])[index] for name in ("P", "sigma_z")}
    predictions = {
        label: predict_field(model, case, selected_x, t, device, 8192)
        for label, model in models
    }
    figure, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=True, gridspec_kw={"hspace": 0.14})
    scales = {"P": 1.0e6, "sigma_z": 1.0e6}
    ylabels = {"P": r"Pressure increment, $\Delta P$ (MPa)", "sigma_z": r"Axial-stress increment, $\Delta\sigma_z$ (MPa)"}
    for axis, state in zip(axes, ("P", "sigma_z")):
        axis.plot(t, (reference[state] - reference[state][0]) / scales[state], color="black", linewidth=2.5, label="MOC reference")
        for (label, _), color in zip(models, COLORS):
            values = predictions[label][state][0]
            linestyle = "-" if label == "Full conditioning" else "--"
            axis.plot(t, (values - values[0]) / scales[state], color=color, linewidth=1.65, linestyle=linestyle, label=label)
        axis.set_ylabel(ylabels[state])
        axis.grid(alpha=0.22)
    axes[0].set_title(
        f"Representative interpolation condition, x/L={x[index] / x[-1]:.3f}"
    )
    axes[1].set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False, fontsize=9.5)
    figure.subplots_adjust(top=0.84)
    figure.savefig(output_dir / f"I04_prediction_traces_{case_id}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def generate(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite input-ablation report: {output_dir}")
    output_dir.mkdir(parents=True)
    plt.rcParams.update({
        "font.size": 11.5,
        "axes.titlesize": 13.5,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
    })
    records = summary_records()
    write_summary(records, output_dir)
    plot_summary(records, output_dir)
    plot_training(output_dir)
    plot_case_heatmap(output_dir)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_trace("formal_validation_010", output_dir, device)
    plot_trace("formal_validation_012", output_dir, device)
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "device": str(device), "training_performed": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
