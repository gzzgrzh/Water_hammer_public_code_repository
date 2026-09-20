"""Structural post-processing for the Engineering Structures manuscript route.

This script never trains a network. It reads the frozen characteristic-model
checkpoint and existing MOC results, recovers thick-cylinder stresses, and
writes checkpointed structural-response results and publication figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from water16_reproduction.parametric_fssi_formal_visualize import load_model
from water16_reproduction.parametric_fssi_model import predict_field
from water16_reproduction.parametric_fssi_reference import case_parameters


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "PINN_FSSI_research_plan/configs/parametric_fssi_hybrid_formal_v3_balanced_events.json"
MODEL_CHECKPOINT = ROOT / "PINN_FSSI_research_plan/outputs/parametric_fssi_hybrid_formal_v3_balanced_events/formal/checkpoint.pt"
SCAN_SOURCE = ROOT / "PINN_FSSI_research_plan/outputs/revised_joint_threshold_scan_v1/joint_threshold_scan_results.csv"
BOUNDARY_SOURCE = ROOT / "PINN_FSSI_research_plan/outputs/revised_joint_boundary_moc_v1"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/engineering_structures_postprocess_v1"
REPRESENTATIVE_CASE = "joint_boundary_moc_014"
REFERENCE_YIELD_STRENGTH_PA = 235.0e6
SCAN_SPATIAL_POINTS = 81
SCAN_TIME_POINTS = 401
PREDICTION_BATCH_SIZE = 65536


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def lame_wall_stresses(
    pressure_pa: np.ndarray,
    inner_radius_m: float,
    outer_radius_m: float,
    external_pressure_pa: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return tensile-positive radial and hoop stresses at both wall surfaces."""

    ri2 = inner_radius_m**2
    ro2 = outer_radius_m**2
    denominator = ro2 - ri2
    a = (pressure_pa * ri2 - external_pressure_pa * ro2) / denominator
    b = ri2 * ro2 * (pressure_pa - external_pressure_pa) / denominator
    return {
        "sigma_r_inner": a - b / ri2,
        "sigma_theta_inner": a + b / ri2,
        "sigma_r_outer": a - b / ro2,
        "sigma_theta_outer": a + b / ro2,
    }


def von_mises(
    sigma_r: np.ndarray, sigma_theta: np.ndarray, sigma_z: np.ndarray
) -> np.ndarray:
    return np.sqrt(
        0.5
        * (
            (sigma_theta - sigma_z) ** 2
            + (sigma_z - sigma_r) ** 2
            + (sigma_r - sigma_theta) ** 2
        )
    )


def structural_fields(
    pressure_pa: np.ndarray,
    axial_stress_pa: np.ndarray,
    inner_radius_m: float,
    wall_thickness_m: float,
    initial_pressure_pa: float,
    initial_axial_stress_pa: float,
) -> dict[str, np.ndarray]:
    outer_radius_m = inner_radius_m + wall_thickness_m
    wall = lame_wall_stresses(pressure_pa, inner_radius_m, outer_radius_m)
    vm_inner = von_mises(
        wall["sigma_r_inner"], wall["sigma_theta_inner"], axial_stress_pa
    )
    vm_outer = von_mises(
        wall["sigma_r_outer"], wall["sigma_theta_outer"], axial_stress_pa
    )
    initial_wall = lame_wall_stresses(
        np.asarray(initial_pressure_pa), inner_radius_m, outer_radius_m
    )
    initial_vm_inner = float(
        von_mises(
            initial_wall["sigma_r_inner"],
            initial_wall["sigma_theta_inner"],
            np.asarray(initial_axial_stress_pa),
        )
    )
    initial_vm_outer = float(
        von_mises(
            initial_wall["sigma_r_outer"],
            initial_wall["sigma_theta_outer"],
            np.asarray(initial_axial_stress_pa),
        )
    )
    vm_delta_inner = vm_inner - initial_vm_inner
    vm_delta_outer = vm_outer - initial_vm_outer
    controlling_inner = np.abs(vm_delta_inner) >= np.abs(vm_delta_outer)
    return {
        **wall,
        "sigma_vm_inner": vm_inner,
        "sigma_vm_outer": vm_outer,
        "delta_sigma_theta_inner": wall["sigma_theta_inner"]
        - float(initial_wall["sigma_theta_inner"]),
        "delta_sigma_vm_inner": vm_delta_inner,
        "delta_sigma_vm_outer": vm_delta_outer,
        "delta_sigma_vm_control": np.where(
            controlling_inner, vm_delta_inner, vm_delta_outer
        ),
        "sigma_vm_wall_max": np.maximum(vm_inner, vm_outer),
    }


def field_metrics(
    fields: dict[str, np.ndarray],
    x: np.ndarray,
    t: np.ndarray,
    yield_strength_pa: float,
) -> dict[str, Any]:
    hoop_delta = fields["delta_sigma_theta_inner"]
    vm_delta = fields["delta_sigma_vm_control"]
    vm_total = fields["sigma_vm_wall_max"]
    hoop_index = np.unravel_index(int(np.argmax(np.abs(hoop_delta))), hoop_delta.shape)
    vm_delta_index = np.unravel_index(int(np.argmax(np.abs(vm_delta))), vm_delta.shape)
    vm_total_index = np.unravel_index(int(np.argmax(vm_total)), vm_total.shape)
    return {
        "maximum_absolute_hoop_stress_increment_pa": float(
            np.abs(hoop_delta[hoop_index])
        ),
        "signed_hoop_stress_increment_at_absolute_max_pa": float(
            hoop_delta[hoop_index]
        ),
        "hoop_increment_location_over_L": float(x[hoop_index[0]] / x[-1]),
        "hoop_increment_time_s": float(t[hoop_index[1]]),
        "maximum_absolute_von_mises_increment_pa": float(
            np.abs(vm_delta[vm_delta_index])
        ),
        "signed_von_mises_increment_at_absolute_max_pa": float(
            vm_delta[vm_delta_index]
        ),
        "von_mises_increment_location_over_L": float(
            x[vm_delta_index[0]] / x[-1]
        ),
        "von_mises_increment_time_s": float(t[vm_delta_index[1]]),
        "maximum_total_von_mises_stress_pa": float(vm_total[vm_total_index]),
        "total_von_mises_location_over_L": float(x[vm_total_index[0]] / x[-1]),
        "total_von_mises_time_s": float(t[vm_total_index[1]]),
        "reference_yield_utilization": float(
            vm_total[vm_total_index] / yield_strength_pa
        ),
    }


def relative_error(prediction: float, reference: float) -> float:
    return abs(prediction - reference) / max(abs(reference), 1.0)


def case_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(row["case_id"]),
        "split": "engineering_structural_postprocess",
        "test_class": "engineering_structural_postprocess",
        "soil_to_pipe_modulus_ratio": float(row["soil_to_pipe_modulus_ratio"]),
        "closure_time_over_L_cf": float(row["closure_time_over_L_cf"]),
        "initial_velocity_m_s": float(row["initial_velocity_m_s"]),
    }


def load_frozen_model(device_name: str):
    hybrid, design, model, device = load_model(
        MODEL_CONFIG, MODEL_CHECKPOINT, device_name
    )
    model.eval()
    return hybrid, design, model, device


def process_boundary_cases(
    output: Path, device_name: str, yield_strength_pa: float
) -> dict[str, Any]:
    report_path = output / "boundary_structural_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    _, _, model, device = load_frozen_model(device_name)
    source_rows = read_csv(BOUNDARY_SOURCE / "T12_boundary_MOC_confirmation.csv")
    case_root = output / "boundary_cases"
    case_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, source_row in enumerate(source_rows, start=1):
        case = case_from_row(source_row)
        case_output = case_root / case["case_id"]
        metrics_path = case_output / "metrics.json"
        if metrics_path.exists():
            rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            print(f"[boundary {index}/{len(source_rows)}] reused {case['case_id']}", flush=True)
            continue
        case_output.mkdir(parents=True, exist_ok=True)
        source_dir = BOUNDARY_SOURCE / "cases" / case["case_id"]
        metadata = json.loads((source_dir / "result.json").read_text(encoding="utf-8"))
        with np.load(source_dir / "truth_evaluation_grid.npz", allow_pickle=False) as data:
            x = np.asarray(data["x"])
            t = np.asarray(data["t"])
            reference = {"P": np.asarray(data["P"]), "sigma_z": np.asarray(data["sigma_z"])}
        prediction = predict_field(
            model, case, x, t, device, PREDICTION_BATCH_SIZE
        )
        params, _ = case_parameters(model.baseline, case, model.t_final_s)
        p0 = float(metadata["initial_pressure_pa"])
        sz0 = float(metadata["initial_stress_pa"])
        reference_fields = structural_fields(
            reference["P"],
            reference["sigma_z"],
            params.inner_radius_m,
            params.wall_thickness_m,
            p0,
            sz0,
        )
        prediction_fields = structural_fields(
            prediction["P"],
            prediction["sigma_z"],
            params.inner_radius_m,
            params.wall_thickness_m,
            p0,
            sz0,
        )
        reference_metrics = field_metrics(
            reference_fields, x, t, yield_strength_pa
        )
        prediction_metrics = field_metrics(
            prediction_fields, x, t, yield_strength_pa
        )
        row: dict[str, Any] = {
            **case,
            "inner_radius_m": params.inner_radius_m,
            "wall_thickness_m": params.wall_thickness_m,
            "outer_radius_m": params.inner_radius_m + params.wall_thickness_m,
            "reference_yield_strength_pa": yield_strength_pa,
        }
        for key, value in reference_metrics.items():
            row[f"moc_{key}"] = value
        for key, value in prediction_metrics.items():
            row[f"model_{key}"] = value
        for metric in (
            "maximum_absolute_hoop_stress_increment_pa",
            "maximum_absolute_von_mises_increment_pa",
            "maximum_total_von_mises_stress_pa",
        ):
            row[f"relative_error_{metric}"] = relative_error(
                prediction_metrics[metric], reference_metrics[metric]
            )
        write_json_new(metrics_path, row)
        if case["case_id"] == REPRESENTATIVE_CASE:
            np.savez_compressed(
                case_output / "representative_structural_fields.npz",
                x=x,
                t=t,
                moc_delta_sigma_theta_inner=reference_fields[
                    "delta_sigma_theta_inner"
                ],
                model_delta_sigma_theta_inner=prediction_fields[
                    "delta_sigma_theta_inner"
                ],
                moc_delta_sigma_vm_control=reference_fields[
                    "delta_sigma_vm_control"
                ],
                model_delta_sigma_vm_control=prediction_fields[
                    "delta_sigma_vm_control"
                ],
            )
        rows.append(row)
        print(f"[boundary {index}/{len(source_rows)}] completed {case['case_id']}", flush=True)
    summary_path = output / "T18_boundary_structural_metrics.csv"
    write_csv_new(summary_path, rows)
    error_summary = {}
    for metric in (
        "maximum_absolute_hoop_stress_increment_pa",
        "maximum_absolute_von_mises_increment_pa",
        "maximum_total_von_mises_stress_pa",
    ):
        values = np.asarray([float(row[f"relative_error_{metric}"]) for row in rows])
        error_summary[metric] = {
            "mean_relative_error": float(values.mean()),
            "maximum_relative_error": float(values.max()),
        }
    report = {
        "status": "pass",
        "training_performed": False,
        "model_checkpoint": str(MODEL_CHECKPOINT.relative_to(ROOT)),
        "case_count": len(rows),
        "moc_cells": 5120,
        "spatial_points": 161,
        "time_points": 801,
        "stress_recovery": "thick-cylinder Lame solution at inner and outer walls",
        "external_radial_pressure_pa": 0.0,
        "axial_stress_source": "total axial stress from the four-equation MOC/model; initial pressure stress is not added again",
        "reference_yield_strength_pa": yield_strength_pa,
        "error_summary": error_summary,
        "maximum_moc_yield_utilization": float(
            max(float(row["moc_reference_yield_utilization"]) for row in rows)
        ),
        "runtime_s": time.perf_counter() - started,
        "table": summary_path.name,
    }
    write_json_new(report_path, report)
    return report


def scan_rows() -> list[dict[str, str]]:
    rows = read_csv(SCAN_SOURCE)
    return [row for row in rows if "velocity_closure" in row["scan_families"]]


def process_scan(
    output: Path,
    device_name: str,
    yield_strength_pa: float,
    chunk_size: int,
    max_cases: int | None,
) -> dict[str, Any]:
    report_path = output / "structural_scan_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    _, _, model, device = load_frozen_model(device_name)
    rows = scan_rows()
    if max_cases is not None:
        rows = rows[:max_cases]
    x = np.linspace(0.0, float(model.baseline.length_m), SCAN_SPATIAL_POINTS)
    t = np.linspace(0.0, float(model.t_final_s), SCAN_TIME_POINTS)
    chunks = output / "scan_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        path = chunks / f"structural_scan_{start + 1:05d}_{stop:05d}.csv"
        if path.exists():
            print(f"[scan] reused {path.name}", flush=True)
            continue
        chunk_rows: list[dict[str, Any]] = []
        for source_row in rows[start:stop]:
            case = case_from_row(source_row)
            prediction = predict_field(
                model, case, x, t, device, PREDICTION_BATCH_SIZE
            )
            params, _ = case_parameters(model.baseline, case, model.t_final_s)
            p0 = float(params.water_density_kg_m3 * params.gravity_m_s2 * params.head_difference_m)
            ri = params.inner_radius_m
            ro = ri + params.wall_thickness_m
            area_fluid = np.pi * ri**2
            area_wall = np.pi * (ro**2 - ri**2)
            sz0 = area_fluid * p0 / area_wall
            fields = structural_fields(
                prediction["P"],
                prediction["sigma_z"],
                ri,
                params.wall_thickness_m,
                p0,
                sz0,
            )
            metrics = field_metrics(fields, x, t, yield_strength_pa)
            chunk_rows.append(
                {
                    "case_id": case["case_id"],
                    "soil_to_pipe_modulus_ratio": case[
                        "soil_to_pipe_modulus_ratio"
                    ],
                    "closure_time_over_L_cf": case["closure_time_over_L_cf"],
                    "initial_velocity_m_s": case["initial_velocity_m_s"],
                    **metrics,
                }
            )
        write_csv_new(path, chunk_rows)
        progress = {
            "status": "running",
            "completed_cases": stop,
            "total_cases": len(rows),
            "last_chunk": path.name,
            "training_performed": False,
            "elapsed_s": time.perf_counter() - started,
        }
        (output / "scan_progress.json").write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )
        print(f"[scan] checkpoint {stop}/{len(rows)} {path.name}", flush=True)
    combined: list[dict[str, str]] = []
    for path in sorted(chunks.glob("structural_scan_*.csv")):
        combined.extend(read_csv(path))
    if len(combined) != len(rows):
        raise RuntimeError(
            f"structural scan incomplete: {len(combined)} rows for {len(rows)} cases"
        )
    table_path = output / "T19_structural_parameter_scan.csv"
    write_csv_new(table_path, combined)
    utilization = np.asarray(
        [float(row["reference_yield_utilization"]) for row in combined]
    )
    report = {
        "status": "pass",
        "training_performed": False,
        "source_scan_unique_cases": 13440,
        "structural_map_family": "velocity_closure at three soil-restraint levels",
        "case_count": len(combined),
        "spatial_points": SCAN_SPATIAL_POINTS,
        "time_points": SCAN_TIME_POINTS,
        "reference_yield_strength_pa": yield_strength_pa,
        "maximum_reference_yield_utilization": float(utilization.max()),
        "minimum_reference_yield_utilization": float(utilization.min()),
        "runtime_s_this_invocation": time.perf_counter() - started,
        "table": table_path.name,
    }
    write_json_new(report_path, report)
    return report


def timing_audit(output: Path, device_name: str, repeats: int) -> dict[str, Any]:
    report_path = output / "engineering_structures_timing_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    _, _, model, device = load_frozen_model(device_name)
    source_rows = read_csv(BOUNDARY_SOURCE / "T12_boundary_MOC_confirmation.csv")
    source_row = [row for row in source_rows if row["case_id"] == REPRESENTATIVE_CASE][0]
    case = case_from_row(source_row)
    x = np.linspace(0.0, float(model.baseline.length_m), 161)
    t = np.linspace(0.0, float(model.t_final_s), 801)
    for _ in range(3):
        predict_field(model, case, x, t, device, PREDICTION_BATCH_SIZE)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_times = []
    for _ in range(repeats):
        started = time.perf_counter()
        predict_field(model, case, x, t, device, PREDICTION_BATCH_SIZE)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_times.append(time.perf_counter() - started)
    moc_times = []
    reference_root = ROOT / "PINN_FSSI_research_plan/outputs/parametric_fssi_reference_v1/formal"
    for path in reference_root.glob("*/*/metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        moc_times.append(float(metadata["generation_runtime_s"]))
    if len(moc_times) != 108:
        raise RuntimeError(f"expected 108 MOC runtimes, found {len(moc_times)}")
    inference_median = float(np.median(inference_times))
    moc_median = float(np.median(moc_times))
    report = {
        "status": "pass",
        "training_performed": False,
        "device": str(device),
        "field_shape": [161, 801, 4],
        "inference_repeats": repeats,
        "inference_runtime_s": {
            "median": inference_median,
            "minimum": float(min(inference_times)),
            "maximum": float(max(inference_times)),
        },
        "moc_case_count": len(moc_times),
        "moc_runtime_s": {
            "median": moc_median,
            "minimum": float(min(moc_times)),
            "maximum": float(max(moc_times)),
        },
        "median_online_speedup": moc_median / inference_median,
        "timing_scope": "Frozen-checkpoint full-field inference versus recorded end-to-end 5120-cell MOC case generation on the same workstation.",
    }
    write_json_new(report_path, report)
    return report


def figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.0,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )


def save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_structural_fields(source_output: Path, figure_output: Path) -> None:
    source = (
        source_output
        / "boundary_cases"
        / REPRESENTATIVE_CASE
        / "representative_structural_fields.npz"
    )
    with np.load(source, allow_pickle=False) as data:
        x = np.asarray(data["x"])
        t = np.asarray(data["t"])
        hoop_moc = np.asarray(data["moc_delta_sigma_theta_inner"]) / 1.0e6
        hoop_model = np.asarray(data["model_delta_sigma_theta_inner"]) / 1.0e6
        vm_moc = np.asarray(data["moc_delta_sigma_vm_control"]) / 1.0e6
        vm_model = np.asarray(data["model_delta_sigma_vm_control"]) / 1.0e6
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 9.3))
    for axis, values, title in (
        (axes[0, 0], hoop_moc, "(a) MOC hoop-stress increment"),
        (axes[0, 1], vm_moc, "(b) MOC von Mises increment"),
    ):
        limit = float(np.max(np.abs(values)))
        image = axis.pcolormesh(
            x / x[-1], t, values.T, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        axis.set(xlabel="$x/L$", ylabel="Time (s)", title=title)
        colorbar = figure.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label("Stress increment (MPa)")
    axes[1, 0].plot(x / x[-1], np.max(np.abs(hoop_moc), axis=1), color="#1f77b4", label="MOC hoop")
    axes[1, 0].plot(x / x[-1], np.max(np.abs(hoop_model), axis=1), "--", color="#1f77b4", label="Model hoop")
    axes[1, 0].plot(x / x[-1], np.max(np.abs(vm_moc), axis=1), color="#d62728", label="MOC von Mises")
    axes[1, 0].plot(x / x[-1], np.max(np.abs(vm_model), axis=1), "--", color="#d62728", label="Model von Mises")
    axes[1, 0].set(xlabel="$x/L$", ylabel="Peak increment (MPa)", title="(c) Full-record spatial envelopes")
    axes[1, 0].grid(alpha=0.22)
    axes[1, 0].legend(frameon=False, ncol=2)
    critical_index = int(np.unravel_index(int(np.argmax(np.abs(vm_moc))), vm_moc.shape)[0])
    axes[1, 1].plot(t, hoop_moc[critical_index], color="#1f77b4", label="MOC hoop")
    axes[1, 1].plot(t, hoop_model[critical_index], "--", color="#1f77b4", label="Model hoop")
    axes[1, 1].plot(t, vm_moc[critical_index], color="#d62728", label="MOC von Mises")
    axes[1, 1].plot(t, vm_model[critical_index], "--", color="#d62728", label="Model von Mises")
    axes[1, 1].set(
        xlabel="Time (s)",
        ylabel="Stress increment (MPa)",
        title=f"(d) Histories at $x/L={x[critical_index] / x[-1]:.3f}$",
    )
    axes[1, 1].grid(alpha=0.22)
    axes[1, 1].legend(frameon=False, ncol=2)
    figure.suptitle(
        "Recovered structural response for a representative high-demand condition",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=2.0, w_pad=1.4)
    save_figure(figure, figure_output, "F53_structural_stress_fields_and_envelopes")


def plot_boundary_parity(source_output: Path, figure_output: Path) -> None:
    structural = read_csv(source_output / "T18_boundary_structural_metrics.csv")
    original = {
        row["case_id"]: row
        for row in read_csv(BOUNDARY_SOURCE / "T12_boundary_MOC_confirmation.csv")
    }
    definitions = (
        (
            "registered_maximum_pressure_increment_pa",
            "moc_maximum_pressure_increment_pa",
            "Pressure rise",
        ),
        (
            "registered_maximum_absolute_axial_stress_increment_pa",
            "moc_maximum_absolute_axial_stress_increment_pa",
            "Axial-stress increment",
        ),
        (
            "model_maximum_absolute_hoop_stress_increment_pa",
            "moc_maximum_absolute_hoop_stress_increment_pa",
            "Hoop-stress increment",
        ),
        (
            "model_maximum_absolute_von_mises_increment_pa",
            "moc_maximum_absolute_von_mises_increment_pa",
            "von Mises increment",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 8.6))
    for axis, (prediction_key, reference_key, title) in zip(axes.ravel(), definitions):
        if prediction_key.startswith("registered_"):
            predicted = np.asarray(
                [float(original[row["case_id"]][prediction_key]) / 1.0e6 for row in structural]
            )
            reference = np.asarray(
                [float(original[row["case_id"]][reference_key]) / 1.0e6 for row in structural]
            )
        else:
            predicted = np.asarray([float(row[prediction_key]) / 1.0e6 for row in structural])
            reference = np.asarray([float(row[reference_key]) / 1.0e6 for row in structural])
        low = float(min(reference.min(), predicted.min()))
        high = float(max(reference.max(), predicted.max()))
        margin = 0.08 * max(high - low, high, 0.1)
        axis.scatter(reference, predicted, s=34, color="#2f5597", alpha=0.85)
        axis.plot([low - margin, high + margin], [low - margin, high + margin], "--", color="#555555")
        errors = np.abs(predicted - reference) / np.maximum(np.abs(reference), 1.0e-12)
        axis.text(
            0.04,
            0.95,
            f"Mean error = {100.0 * errors.mean():.2f}%",
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
        axis.set(
            xlabel="5120-cell MOC (MPa)",
            ylabel="Model prediction (MPa)",
            title=title,
            xlim=(low - margin, high + margin),
            ylim=(low - margin, high + margin),
        )
        axis.grid(alpha=0.22)
        axis.set_aspect("equal", adjustable="box")
    figure.suptitle(
        "Hydraulic and structural peak reconstruction for 15 boundary cases",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=1.8, w_pad=1.5)
    save_figure(figure, figure_output, "F54_boundary_hydraulic_structural_parity")


def reshape_map(
    rows: list[dict[str, str]], soil: float, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if np.isclose(float(row["soil_to_pipe_modulus_ratio"]), soil)
    ]
    velocity = np.asarray(sorted({float(row["initial_velocity_m_s"]) for row in selected}))
    closure = np.asarray(sorted({float(row["closure_time_over_L_cf"]) for row in selected}))
    lookup = {
        (float(row["initial_velocity_m_s"]), float(row["closure_time_over_L_cf"])): float(row[metric])
        for row in selected
    }
    values = np.asarray([[lookup[(v, c)] for c in closure] for v in velocity])
    return velocity, closure, values


def plot_structural_maps(source_output: Path, figure_output: Path) -> None:
    rows = read_csv(source_output / "T19_structural_parameter_scan.csv")
    soils = (0.0003, 0.01, 0.1)
    hoop_maps = [reshape_map(rows, soil, "maximum_absolute_hoop_stress_increment_pa") for soil in soils]
    vm_maps = [reshape_map(rows, soil, "maximum_absolute_von_mises_increment_pa") for soil in soils]
    hoop_min = min(float(values.min()) for _, _, values in hoop_maps) / 1.0e6
    hoop_max = max(float(values.max()) for _, _, values in hoop_maps) / 1.0e6
    vm_min = min(float(values.min()) for _, _, values in vm_maps) / 1.0e6
    vm_max = max(float(values.max()) for _, _, values in vm_maps) / 1.0e6
    figure, axes = plt.subplots(3, 2, figsize=(8.2, 10.6), sharex=True, sharey=True)
    hoop_image = None
    vm_image = None
    for row_index, soil in enumerate(soils):
        for column_index, (maps, vmin, vmax, title) in enumerate(
            (
                (hoop_maps, hoop_min, hoop_max, "Hoop-stress increment"),
                (vm_maps, vm_min, vm_max, "von Mises increment"),
            )
        ):
            velocity, closure, values = maps[row_index]
            image = axes[row_index, column_index].pcolormesh(
                closure,
                velocity,
                values / 1.0e6,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            if column_index == 0:
                hoop_image = image
            else:
                vm_image = image
            axes[row_index, column_index].set_title(
                f"{title}\n$E_s/E={soil:g}$", fontweight="bold"
            )
            axes[row_index, column_index].set_ylabel("Initial velocity (m/s)")
            axes[row_index, column_index].set_xlabel("$t_c/(L/c_f)$")
    figure.subplots_adjust(left=0.10, right=0.88, bottom=0.07, top=0.89, hspace=0.38, wspace=0.22)
    cbar_hoop = figure.colorbar(hoop_image, ax=axes[:, 0], location="bottom", pad=0.08, fraction=0.03)
    cbar_hoop.set_label("Peak hoop-stress increment (MPa)")
    cbar_vm = figure.colorbar(vm_image, ax=axes[:, 1], location="bottom", pad=0.08, fraction=0.03)
    cbar_vm.set_label("Peak von Mises increment (MPa)")
    figure.suptitle(
        "Structural response over the parameter domain",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    save_figure(figure, figure_output, "F55_structural_response_parameter_maps")


def hydraulic_minimum_closure_rows() -> list[dict[str, Any]]:
    rows = scan_rows()
    output: list[dict[str, Any]] = []
    for soil in (0.0003, 0.01, 0.1):
        soil_rows = [
            row
            for row in rows
            if np.isclose(float(row["soil_to_pipe_modulus_ratio"]), soil)
        ]
        velocities = sorted({float(row["initial_velocity_m_s"]) for row in soil_rows})
        for velocity in velocities:
            selected = sorted(
                [
                    row
                    for row in soil_rows
                    if np.isclose(float(row["initial_velocity_m_s"]), velocity)
                ],
                key=lambda row: float(row["closure_time_over_L_cf"]),
            )
            safe = np.asarray(
                [
                    float(row["maximum_pressure_increment_pa"]) <= 130000.0
                    and float(row["minimum_absolute_pressure_pa"]) >= 120000.0
                    for row in selected
                ],
                dtype=bool,
            )
            minimum = float("nan")
            for index in range(len(selected)):
                if bool(np.all(safe[index:])):
                    minimum = float(selected[index]["closure_time_over_L_cf"])
                    break
            output.append(
                {
                    "soil_to_pipe_modulus_ratio": soil,
                    "initial_velocity_m_s": velocity,
                    "minimum_closure_time_over_L_cf": minimum,
                    "pressure_rise_limit_pa": 130000.0,
                    "minimum_absolute_pressure_limit_pa": 120000.0,
                }
            )
    return output


def plot_hydraulic_closure_screening(figure_output: Path) -> None:
    rows = hydraulic_minimum_closure_rows()
    write_csv_new(figure_output / "T21_hydraulic_minimum_closure.csv", rows)
    figure, axis = plt.subplots(figsize=(7.7, 5.6))
    colors = ("#4c78a8", "#f58518", "#54a24b")
    for soil, color in zip((0.0003, 0.01, 0.1), colors):
        selected = [
            row
            for row in rows
            if np.isclose(float(row["soil_to_pipe_modulus_ratio"]), soil)
        ]
        velocity = np.asarray([float(row["initial_velocity_m_s"]) for row in selected])
        closure = np.asarray(
            [float(row["minimum_closure_time_over_L_cf"]) for row in selected]
        )
        axis.plot(
            velocity,
            closure,
            marker="o",
            markersize=3.8,
            linewidth=1.8,
            color=color,
            label=fr"$E_s/E={soil:g}$",
        )
    axis.set(
        xlabel="Initial velocity (m/s)",
        ylabel="Minimum $t_c/(L/c_f)$",
        title="Minimum closure time from the hydraulic operating limits",
        xlim=(0.029, 0.091),
        ylim=(0.08, 1.23),
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    axis.text(
        0.03,
        0.97,
        "$\\Delta P_{\\max}\\leq130$ kPa and $P_{\\min}\\geq120$ kPa",
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    figure.tight_layout()
    save_figure(figure, figure_output, "F56_hydraulic_minimum_closure")


def parameter_effect_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    metrics = (
        "maximum_absolute_hoop_stress_increment_pa",
        "maximum_absolute_von_mises_increment_pa",
    )
    records: list[dict[str, Any]] = []
    soils = (0.0003, 0.01, 0.1)
    for metric in metrics:
        velocity_changes = []
        closure_changes = []
        for soil in soils:
            _, _, values = reshape_map(rows, soil, metric)
            velocity_changes.append(100.0 * (values[-1, 25] / values[0, 25] - 1.0))
            closure_changes.append(100.0 * (values[20, -1] / values[20, 0] - 1.0))
        soft = [
            row
            for row in rows
            if np.isclose(float(row["soil_to_pipe_modulus_ratio"]), soils[0])
            and np.isclose(float(row["initial_velocity_m_s"]), 0.06)
            and np.isclose(float(row["closure_time_over_L_cf"]), 0.60)
        ][0]
        stiff = [
            row
            for row in rows
            if np.isclose(float(row["soil_to_pipe_modulus_ratio"]), soils[-1])
            and np.isclose(float(row["initial_velocity_m_s"]), 0.06)
            and np.isclose(float(row["closure_time_over_L_cf"]), 0.60)
        ][0]
        soil_change = 100.0 * (float(stiff[metric]) / float(soft[metric]) - 1.0)
        records.append(
            {
                "metric": metric,
                "initial_velocity_0p03_to_0p09_percent_min": min(velocity_changes),
                "initial_velocity_0p03_to_0p09_percent_max": max(velocity_changes),
                "closure_ratio_0p10_to_1p20_percent_min": min(closure_changes),
                "closure_ratio_0p10_to_1p20_percent_max": max(closure_changes),
                "soil_ratio_0p0003_to_0p1_percent_at_nominal": soil_change,
            }
        )
    return records


def finalize(output: Path) -> dict[str, Any]:
    final_path = output / "engineering_structures_postprocess_report.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    boundary_report = json.loads(
        (output / "boundary_structural_report.json").read_text(encoding="utf-8")
    )
    scan_report = json.loads(
        (output / "structural_scan_report.json").read_text(encoding="utf-8")
    )
    scan = read_csv(output / "T19_structural_parameter_scan.csv")
    effects = parameter_effect_summary(scan)
    write_csv_new(output / "T20_structural_parameter_effects.csv", effects)
    figure_style()
    plot_structural_fields(output, output)
    plot_boundary_parity(output, output)
    plot_structural_maps(output, output)
    plot_hydraulic_closure_screening(output)
    report = {
        "status": "pass",
        "training_performed": False,
        "network_architecture_changed": False,
        "pipe_geometry_changed": False,
        "boundary_report": boundary_report,
        "scan_report": scan_report,
        "parameter_effects": effects,
        "figures": [
            "F53_structural_stress_fields_and_envelopes",
            "F54_boundary_hydraulic_structural_parity",
            "F55_structural_response_parameter_maps",
            "F56_hydraulic_minimum_closure",
        ],
        "scope_limit": "Nominal thick-cylinder stress recovery for the fixed pipe geometry; zero external radial pressure and no local stress concentration.",
    }
    write_json_new(final_path, report)
    return report


def replot(source_output: Path, figure_output: Path) -> dict[str, Any]:
    if figure_output.exists():
        raise FileExistsError(f"refusing to overwrite {figure_output}")
    figure_output.mkdir(parents=True)
    required = (
        source_output / "T18_boundary_structural_metrics.csv",
        source_output / "T19_structural_parameter_scan.csv",
        source_output
        / "boundary_cases"
        / REPRESENTATIVE_CASE
        / "representative_structural_fields.npz",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    figure_style()
    plot_structural_fields(source_output, figure_output)
    plot_boundary_parity(source_output, figure_output)
    plot_structural_maps(source_output, figure_output)
    plot_hydraulic_closure_screening(figure_output)
    report = {
        "status": "pass",
        "training_performed": False,
        "source_output": str(source_output),
        "figures": [
            "F53_structural_stress_fields_and_envelopes",
            "F54_boundary_hydraulic_structural_parity",
            "F55_structural_response_parameter_maps",
            "F56_hydraulic_minimum_closure",
        ],
    }
    write_json_new(figure_output / "structural_figure_replot_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("boundary", "scan", "timing", "finalize", "replot", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--yield-strength-pa", type=float, default=REFERENCE_YIELD_STRENGTH_PA)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timing-repeats", type=int, default=20)
    args = parser.parse_args()
    if args.stage != "replot":
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("boundary", "all"):
        process_boundary_cases(args.output_dir, args.device, args.yield_strength_pa)
    if args.stage in ("scan", "all"):
        process_scan(
            args.output_dir,
            args.device,
            args.yield_strength_pa,
            args.chunk_size,
            args.max_cases,
        )
    if args.stage == "timing":
        print(json.dumps(timing_audit(args.output_dir, args.device, args.timing_repeats), indent=2))
    if args.stage in ("finalize", "all"):
        print(json.dumps(finalize(args.output_dir), indent=2))
    if args.stage == "replot":
        print(json.dumps(replot(args.source_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
