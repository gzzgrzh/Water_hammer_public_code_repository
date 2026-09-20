"""High-resolution MOC audit of preregistered joint-threshold boundary cases.

The characteristic PINN selects the cases before this script reads any new MOC
result.  Each completed case is an immutable checkpoint, so an interrupted run
can resume without repeating completed calculations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from water16_reproduction.common.physics import cross_section_areas, initial_pressure_pa
from water16_reproduction.parametric_fssi_forward import load_config
from water16_reproduction.parametric_fssi_reference import (
    case_parameters,
    downsample_truth,
    engineering_metrics,
    load_baseline,
)
from water16_reproduction.revised_joint_threshold_scan import METRICS, utilizations
from water16_reproduction.wp2_ablation import reference_solution


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / "PINN_FSSI_research_plan/configs/parametric_fssi_forward_v1.json"
DEFAULT_THRESHOLD_CONFIG = ROOT / "PINN_FSSI_research_plan/configs/revised_joint_threshold_assessment_v1.json"
DEFAULT_REGISTRATION = ROOT / "PINN_FSSI_research_plan/outputs/revised_joint_threshold_scan_v1/boundary_moc_case_registration.json"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/revised_joint_boundary_moc_v1"
PARAMETERS = ("soil_to_pipe_modulus_ratio", "closure_time_over_L_cf", "initial_velocity_m_s")
METRIC_LABELS = {
    "maximum_pressure_increment_pa": (1.0e-3, r"$\Delta P_{max}$ (kPa)"),
    "minimum_absolute_pressure_pa": (1.0e-3, r"$P_{abs,min}$ (kPa)"),
    "maximum_absolute_axial_stress_increment_pa": (1.0e-6, r"$|\Delta\sigma_z|_{max}$ (MPa)"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f"{path.name}.partial_{os.getpid()}_{time.time_ns()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pressure_policy(design: dict[str, Any]) -> dict[str, Any]:
    validity = design["pressure_validity"]
    return {
        "pressure_convention": validity["model_pressure_convention"],
        "atmospheric_pressure_pa": validity["atmospheric_pressure_pa"],
        "vapor_pressure_pa": validity["vapor_pressure_pa"],
        "minimum_cavitation_margin_pa": validity["minimum_cavitation_margin_pa"],
    }


def metric_relative_error(reference: float, prediction: float) -> float:
    return abs(float(prediction) - float(reference)) / max(abs(float(reference)), 1.0e-30)


def classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "true_safe": sum(bool(row["registered_safe"]) and bool(row["moc_safe"]) for row in rows),
        "false_safe": sum(bool(row["registered_safe"]) and not bool(row["moc_safe"]) for row in rows),
        "false_unsafe": sum(not bool(row["registered_safe"]) and bool(row["moc_safe"]) for row in rows),
        "true_unsafe": sum(not bool(row["registered_safe"]) and not bool(row["moc_safe"]) for row in rows),
    }


def save_case(
    case_dir: Path,
    case: dict[str, Any],
    truth: dict[str, np.ndarray],
    metadata: dict[str, Any],
    spatial_points: int,
) -> None:
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    if case_dir.exists():
        raise FileExistsError(f"refusing to overwrite completed or ambiguous case {case_dir}")
    staging = case_dir.parent / f".partial_{case_dir.name}_{os.getpid()}_{time.time_ns()}"
    staging.mkdir()
    saved = downsample_truth(truth, spatial_points, 2)
    np.savez_compressed(staging / "truth_evaluation_grid.npz", **saved)
    (staging / "result.json").write_text(json.dumps({"case": case, **metadata}, indent=2), encoding="utf-8")
    staging.replace(case_dir)


def completed_rows(output: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    cases_dir = output / "cases"
    if not cases_dir.exists():
        return rows
    for path in sorted(cases_dir.glob("*/result.json")):
        if path.parent.name.startswith(".partial_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["comparison_row"]
        rows[str(row["case_id"])] = row
    return rows


def plot_parity(rows: list[dict[str, Any]], output: Path) -> list[str]:
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), gridspec_kw={"wspace": 0.34})
    for axis, metric in zip(axes, METRICS):
        scale, label = METRIC_LABELS[metric]
        moc = np.asarray([float(row[f"moc_{metric}"]) * scale for row in rows])
        model = np.asarray([float(row[f"registered_{metric}"]) * scale for row in rows])
        lower, upper = min(moc.min(), model.min()), max(moc.max(), model.max())
        padding = 0.05 * max(upper - lower, 1.0e-12)
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="black", lw=0.9)
        colors = ["#59A14F" if bool(row["moc_safe"]) else "#E15759" for row in rows]
        axis.scatter(moc, model, c=colors, edgecolor="white", linewidth=0.5, s=38)
        axis.set(xlabel=f"MOC {label}", ylabel=f"PINN {label}")
        axis.grid(alpha=0.2)
    figure.savefig(output / "F33_boundary_MOC_PINN_parity.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output / "F33_boundary_MOC_PINN_parity.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return ["F33_boundary_MOC_PINN_parity.png", "F33_boundary_MOC_PINN_parity.pdf"]


def plot_utilization(rows: list[dict[str, Any]], output: Path) -> list[str]:
    labels = [str(row["case_id"]).replace("joint_boundary_moc_", "B") for row in rows]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(10.8, 4.4))
    axis.plot(x, [float(row["registered_utilization"]) for row in rows], "o-", color="#4C78A8", label="Proposed model")
    axis.plot(x, [float(row["moc_utilization"]) for row in rows], "s--", color="#E15759", label="Direct MOC")
    axis.axhline(1.0, color="black", ls=":", lw=1.0, label="Response limit")
    axis.set_xticks(x, labels, rotation=45)
    axis.set(xlabel="Boundary case", ylabel="Maximum response ratio")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(output / "F34_boundary_utilization_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output / "F34_boundary_utilization_comparison.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return ["F34_boundary_utilization_comparison.png", "F34_boundary_utilization_comparison.pdf"]


def plot_confusion(rows: list[dict[str, Any]], output: Path) -> list[str]:
    counts = classification_counts(rows)
    matrix = np.asarray([[counts["true_safe"], counts["false_safe"]], [counts["false_unsafe"], counts["true_unsafe"]]])
    figure, axis = plt.subplots(figsize=(4.8, 3.8))
    axis.imshow(matrix, cmap="Blues", vmin=0)
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(int(value)), ha="center", va="center", fontsize=13)
    axis.set_xticks([0, 1], ["MOC safe", "MOC unsafe"])
    axis.set_yticks([0, 1], ["Proposed model: safe", "Proposed model: unsafe"])
    axis.set_title("Classification near the response limits")
    # The four cells are labelled directly, so a colour bar is redundant and
    # previously crowded the title when the compact figure was placed in TeX.
    figure.tight_layout()
    figure.savefig(output / "F35_boundary_classification_matrix.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output / "F35_boundary_classification_matrix.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return ["F35_boundary_classification_matrix.png", "F35_boundary_classification_matrix.pdf"]


def plot_critical_histories(rows: list[dict[str, Any]], output: Path) -> list[str]:
    critical = sorted(rows, key=lambda row: abs(float(row["moc_utilization"]) - 1.0))[:4]
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), sharex=True)
    for axis, row in zip(axes.ravel(), critical):
        case_dir = output / "cases" / str(row["case_id"])
        with np.load(case_dir / "truth_evaluation_grid.npz") as archive:
            t = archive["t"]
            pressure = archive["P"][-1]
            stress = archive["sigma_z"][-1]
        metadata = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        pressure0 = float(metadata["initial_pressure_pa"])
        stress0 = float(metadata["initial_stress_pa"])
        axis.plot(t, (pressure - pressure0) * 1.0e-3, color="#4C78A8", lw=1.0, label=r"$\Delta P$ (kPa)")
        twin = axis.twinx()
        twin.plot(t, (stress - stress0) * 1.0e-6, color="#E15759", lw=0.9, label=r"$\Delta\sigma_z$ (MPa)")
        axis.set_title(f"{str(row['case_id']).replace('joint_boundary_moc_', 'B')}: MOC U={float(row['moc_utilization']):.3f}")
        axis.set_ylabel(r"$\Delta P$ (kPa)", color="#4C78A8")
        twin.set_ylabel(r"$\Delta\sigma_z$ (MPa)", color="#E15759")
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    figure.tight_layout()
    figure.savefig(output / "F36_boundary_critical_valve_histories.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output / "F36_boundary_critical_valve_histories.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return ["F36_boundary_critical_valve_histories.png", "F36_boundary_critical_valve_histories.pdf"]


def run(
    design_path: Path,
    threshold_config_path: Path,
    registration_path: Path,
    output: Path,
    case_limit: int | None = None,
    n_cells_override: int | None = None,
) -> dict[str, Any]:
    report_path = output / "boundary_moc_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    design = load_config(design_path)
    threshold_config = json.loads(threshold_config_path.read_text(encoding="utf-8"))
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("status") != "registered_before_new_MOC_runs":
        raise ValueError("boundary cases were not preregistered before MOC execution")
    if bool(threshold_config["boundary_moc_registration"]["selection_uses_moc_results"]):
        raise ValueError("MOC-informed case selection is prohibited")
    cases = list(registration["cases"])
    if case_limit is not None:
        cases = cases[:case_limit]
    n_cells = int(n_cells_override or threshold_config["boundary_moc_registration"]["formal_cells"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(exist_ok=True)
    manifest = {
        "status": "locked_before_first_new_MOC_result",
        "registration_sha256": sha256(registration_path),
        "threshold_config_sha256": sha256(threshold_config_path),
        "design_sha256": sha256(design_path),
        "case_count": len(cases),
        "n_cells": n_cells,
        "case_ids": [case["case_id"] for case in cases],
    }
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("existing boundary MOC manifest differs from requested run")
    else:
        write_json_atomic_new(manifest_path, manifest)

    baseline = load_baseline(design)
    solver = design["truth_solver"]["formal"]
    times = np.linspace(0.0, float(solver["t_final_s"]), int(solver["output_points"]))
    policy = pressure_policy(design)
    thresholds = threshold_config["threshold_scenarios"][threshold_config["primary_scenario"]]
    completed = completed_rows(output)
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        if case_id in completed:
            print(f"[{index}/{len(cases)}] {case_id} checkpoint exists", flush=True)
            continue
        case_started = time.perf_counter()
        params, fluid_speed = case_parameters(baseline, case, float(solver["t_final_s"]))
        truth = reference_solution(params, times, n_cells=n_cells, cfl=float(design["truth_solver"]["cfl"]))
        moc = engineering_metrics(truth, params, policy)
        registered = case["registered_model_prediction"]
        registered_utilizations, registered_safe, registered_control = utilizations(registered, thresholds)
        moc_utilizations, moc_safe, moc_control = utilizations(moc, thresholds)
        row: dict[str, Any] = {
            "case_id": case_id,
            "test_class": case["test_class"],
            **{name: float(case[name]) for name in PARAMETERS},
            "registered_safe": registered_safe,
            "moc_safe": moc_safe,
            "classification_agreement": registered_safe == moc_safe,
            "false_safe": registered_safe and not moc_safe,
            "registered_control": registered_control,
            "moc_control": moc_control,
            "registered_utilization": float(np.max(registered_utilizations)),
            "moc_utilization": float(np.max(moc_utilizations)),
            "runtime_s": time.perf_counter() - case_started,
        }
        for metric in METRICS:
            row[f"registered_{metric}"] = float(registered[metric])
            row[f"moc_{metric}"] = float(moc[metric])
            row[f"relative_error_{metric}"] = metric_relative_error(float(moc[metric]), float(registered[metric]))
        pressure0 = initial_pressure_pa(params)
        area_f, area_t = cross_section_areas(params)
        metadata = {
            "comparison_row": row,
            "n_cells": n_cells,
            "cfl": float(design["truth_solver"]["cfl"]),
            "fluid_wave_speed_m_s": fluid_speed,
            "closure_time_s": params.valve_close_time_s,
            "initial_pressure_pa": pressure0,
            "initial_stress_pa": area_f * pressure0 / area_t,
            "moc_pressure_validity_status": moc["pressure_validity_status"],
        }
        save_case(
            output / "cases" / case_id,
            case,
            truth,
            metadata,
            int(threshold_config["boundary_moc_registration"]["saved_spatial_points"]),
        )
        completed[case_id] = row
        print(f"[{index}/{len(cases)}] {case_id} completed in {row['runtime_s']:.2f} s", flush=True)

    rows = [completed[str(case["case_id"])] for case in cases]
    write_csv_new(output / "T12_boundary_MOC_confirmation.csv", rows)
    figures = plot_parity(rows, output) + plot_utilization(rows, output) + plot_confusion(rows, output) + plot_critical_histories(rows, output)
    counts = classification_counts(rows)
    report = {
        "status": "pass",
        "method": "new_high_resolution_same_equation_characteristic_MOC",
        "selection_was_locked_before_MOC": True,
        "case_count": len(rows),
        "n_cells": n_cells,
        "classification_counts": counts,
        "classification_agreement_rate": (counts["true_safe"] + counts["true_unsafe"]) / len(rows),
        "false_safe_rate_all_boundary_cases": counts["false_safe"] / len(rows),
        "maximum_relative_errors": {
            metric: max(float(row[f"relative_error_{metric}"]) for row in rows) for metric in METRICS
        },
        "mean_relative_errors": {
            metric: float(np.mean([float(row[f"relative_error_{metric}"]) for row in rows])) for metric in METRICS
        },
        "runtime_s_this_invocation": time.perf_counter() - started,
        "figures": figures,
        "table": "T12_boundary_MOC_confirmation.csv",
    }
    write_json_atomic_new(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--threshold-config", type=Path, default=DEFAULT_THRESHOLD_CONFIG)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--n-cells", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.design, args.threshold_config, args.registration, args.output_dir, args.case_limit, args.n_cells)


if __name__ == "__main__":
    main()
