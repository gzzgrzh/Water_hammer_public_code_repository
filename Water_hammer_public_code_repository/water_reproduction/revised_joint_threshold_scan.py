"""Checkpointed characteristic-model scan for joint response thresholds.

This script performs no training.  It evaluates the locked non-oracle
characteristic checkpoint on two registered families of engineering slices,
writes immutable case chunks, creates response/control maps, and registers new
near-boundary cases for later high-resolution MOC confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from water16_reproduction.parametric_fssi_formal_visualize import load_model
from water16_reproduction.parametric_fssi_model import predict_field
from water16_reproduction.parametric_fssi_reference import case_parameters, engineering_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "PINN_FSSI_research_plan/configs/revised_joint_threshold_assessment_v1.json"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/revised_joint_threshold_scan_v1"
PARAMETERS = ("soil_to_pipe_modulus_ratio", "closure_time_over_L_cf", "initial_velocity_m_s")
METRICS = (
    "maximum_pressure_increment_pa",
    "minimum_absolute_pressure_pa",
    "maximum_absolute_axial_stress_increment_pa",
)
CONTROL_NAMES = ("pressure", "axial_stress", "minimum_pressure", "joint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in PARAMETERS + METRICS + (
            "maximum_pressure_location_over_L",
            "minimum_pressure_location_over_L",
            "stress_critical_location_over_L",
        ):
            row[key] = float(row[key])
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def registered_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    scan = config["scan"]
    soil_values = np.geomspace(*config["parameter_domain"]["soil_to_pipe_modulus_ratio"], int(scan["soil_points"]))
    closure_values = np.linspace(*config["parameter_domain"]["closure_time_over_L_cf"], int(scan["closure_points"]))
    velocity_values = np.linspace(*config["parameter_domain"]["initial_velocity_m_s"], int(scan["velocity_points"]))
    cases: dict[tuple[float, float, float], dict[str, Any]] = {}

    def add(soil: float, closure: float, velocity: float, family: str) -> None:
        key = (float(f"{soil:.12g}"), float(f"{closure:.12g}"), float(f"{velocity:.12g}"))
        if key not in cases:
            cases[key] = {
                "case_id": "joint_scan_pending",
                "split": "engineering_scan",
                "test_class": family,
                "soil_to_pipe_modulus_ratio": key[0],
                "closure_time_over_L_cf": key[1],
                "initial_velocity_m_s": key[2],
                "scan_families": [family],
            }
        elif family not in cases[key]["scan_families"]:
            cases[key]["scan_families"].append(family)

    for soil in scan["soil_levels_for_velocity_closure_maps"]:
        for closure in closure_values:
            for velocity in velocity_values:
                add(float(soil), float(closure), float(velocity), "velocity_closure")
    for velocity in scan["velocity_levels_for_soil_closure_maps_m_s"]:
        for closure in closure_values:
            for soil in soil_values:
                add(float(soil), float(closure), float(velocity), "soil_closure")
    result = sorted(cases.values(), key=lambda row: (row[PARAMETERS[0]], row[PARAMETERS[1]], row[PARAMETERS[2]]))
    for index, case in enumerate(result, start=1):
        case["case_id"] = f"joint_scan_{index:05d}"
    return result


def load_completed_chunks(chunks_dir: Path) -> dict[str, dict[str, Any]]:
    completed = {}
    for path in sorted(chunks_dir.glob("chunk_*.csv")):
        for row in read_csv(path):
            if row["case_id"] in completed:
                raise RuntimeError(f"duplicate completed case {row['case_id']}")
            completed[row["case_id"]] = row
    return completed


def case_metrics(model: Any, hybrid: dict[str, Any], case: dict[str, Any], device: Any, config: dict[str, Any]) -> dict[str, Any]:
    scan = config["scan"]
    x = np.linspace(0.0, model.baseline.length_m, int(scan["spatial_points"]))
    t = np.linspace(0.0, model.t_final_s, int(scan["time_points"]))
    prediction = predict_field(model, case, x, t, device, int(scan["inference_batch_size"]))
    prediction["z"] = x
    prediction["t"] = t
    params, fluid_speed = case_parameters(model.baseline, case, model.t_final_s)
    registered_pressure = model.design["pressure_validity"]
    pressure_policy = {
        "pressure_convention": registered_pressure["model_pressure_convention"],
        "atmospheric_pressure_pa": registered_pressure["atmospheric_pressure_pa"],
        "vapor_pressure_pa": registered_pressure["vapor_pressure_pa"],
        "minimum_cavitation_margin_pa": registered_pressure["minimum_cavitation_margin_pa"],
    }
    row = engineering_metrics(prediction, params, pressure_policy)
    return {
        "case_id": case["case_id"],
        "scan_families": "+".join(case["scan_families"]),
        **{name: case[name] for name in PARAMETERS},
        "fluid_wave_speed_m_s": fluid_speed,
        **row,
    }


def utilizations(row: dict[str, Any], thresholds: dict[str, float]) -> tuple[np.ndarray, bool, str]:
    values = np.asarray(
        [
            float(row["maximum_pressure_increment_pa"]) / float(thresholds["maximum_pressure_increment_pa"]),
            float(row["maximum_absolute_axial_stress_increment_pa"]) / float(thresholds["maximum_absolute_axial_stress_increment_pa"]),
            float(thresholds["minimum_absolute_pressure_pa"]) / max(float(row["minimum_absolute_pressure_pa"]), 1.0e-30),
        ]
    )
    order = np.argsort(values)[::-1]
    control = "joint" if values[order[0]] - values[order[1]] <= 0.03 else CONTROL_NAMES[int(order[0])]
    return values, bool(np.max(values) <= 1.0), control


def enrich_thresholds(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    for row in rows:
        for scenario, thresholds in config["threshold_scenarios"].items():
            values, safe, control = utilizations(row, thresholds)
            row[f"{scenario}_pressure_utilization"] = float(values[0])
            row[f"{scenario}_stress_utilization"] = float(values[1])
            row[f"{scenario}_minimum_pressure_utilization"] = float(values[2])
            row[f"{scenario}_maximum_utilization"] = float(np.max(values))
            row[f"{scenario}_safe"] = safe
            row[f"{scenario}_control"] = control


def robust_minimum_closure(rows: list[dict[str, Any]], scenario: str) -> float:
    ordered = sorted(rows, key=lambda row: float(row["closure_time_over_L_cf"]))
    safe = np.asarray([bool(row[f"{scenario}_safe"]) for row in ordered])
    for index in range(len(ordered)):
        if bool(np.all(safe[index:])):
            return float(ordered[index]["closure_time_over_L_cf"])
    return float("nan")


def minimum_closure_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    scan = config["scan"]
    for soil in scan["soil_levels_for_velocity_closure_maps"]:
        for velocity in sorted({float(row["initial_velocity_m_s"]) for row in rows if "velocity_closure" in row["scan_families"]}):
            selected = [row for row in rows if np.isclose(float(row["soil_to_pipe_modulus_ratio"]), float(soil)) and np.isclose(float(row["initial_velocity_m_s"]), velocity)]
            if not selected:
                continue
            result = {"slice": "velocity", "soil_to_pipe_modulus_ratio": soil, "initial_velocity_m_s": velocity}
            for scenario in config["threshold_scenarios"]:
                result[f"{scenario}_minimum_closure_time_over_L_cf"] = robust_minimum_closure(selected, scenario)
            output.append(result)
    for velocity in scan["velocity_levels_for_soil_closure_maps_m_s"]:
        for soil in sorted({float(row["soil_to_pipe_modulus_ratio"]) for row in rows if "soil_closure" in row["scan_families"]}):
            selected = [row for row in rows if np.isclose(float(row["soil_to_pipe_modulus_ratio"]), soil) and np.isclose(float(row["initial_velocity_m_s"]), float(velocity))]
            if not selected:
                continue
            result = {"slice": "soil", "soil_to_pipe_modulus_ratio": soil, "initial_velocity_m_s": velocity}
            for scenario in config["threshold_scenarios"]:
                result[f"{scenario}_minimum_closure_time_over_L_cf"] = robust_minimum_closure(selected, scenario)
            output.append(result)
    return output


def grid_for(rows: list[dict[str, Any]], fixed_name: str, fixed_value: float, x_name: str, y_name: str, value_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if np.isclose(float(row[fixed_name]), fixed_value)]
    x_values = np.asarray(sorted({float(row[x_name]) for row in selected}))
    y_values = np.asarray(sorted({float(row[y_name]) for row in selected}))
    lookup = {(float(row[x_name]), float(row[y_name])): row for row in selected}
    values = np.empty((len(y_values), len(x_values)))
    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            values[y_index, x_index] = float(lookup[(x_value, y_value)][value_name])
    return x_values, y_values, values


def save_figure(figure: plt.Figure, output: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        figure.savefig(path, dpi=320 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(figure)
    return paths


def plot_utilization_maps(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> list[str]:
    scenario = config["primary_scenario"]
    levels = config["scan"]["soil_levels_for_velocity_closure_maps"]
    figure, axes = plt.subplots(1, len(levels), figsize=(13.0, 3.8), sharex=True, sharey=True)
    for axis, soil in zip(axes, levels):
        closure, velocity, values = grid_for(rows, "soil_to_pipe_modulus_ratio", float(soil), "closure_time_over_L_cf", "initial_velocity_m_s", f"{scenario}_maximum_utilization")
        contour = axis.contourf(closure, velocity, values, levels=np.linspace(0.45, max(1.6, float(np.max(values))), 20), cmap="viridis")
        axis.contour(closure, velocity, values, levels=[1.0], colors="white", linewidths=2.0)
        axis.set_title(rf"$E_s/E={soil:g}$")
        axis.set_xlabel(r"$t_c/(L/c_f)$")
        axis.grid(alpha=0.14)
    axes[0].set_ylabel(r"Initial velocity $V_0$ (m/s)")
    figure.colorbar(contour, ax=axes, pad=0.02, label="Maximum nominal utilization")
    figure.suptitle("Joint response-threshold utilization; white line denotes the nominal boundary", y=1.02)
    return save_figure(figure, output, "F28_nominal_velocity_closure_utilization")


def plot_control_maps(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> list[str]:
    scenario = config["primary_scenario"]
    levels = config["scan"]["soil_levels_for_velocity_closure_maps"]
    colors = ["#4C78A8", "#E15759", "#59A14F", "#B07AA1"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)
    figure, axes = plt.subplots(1, len(levels), figsize=(13.0, 3.8), sharex=True, sharey=True)
    for axis, soil in zip(axes, levels):
        selected = [row for row in rows if np.isclose(float(row["soil_to_pipe_modulus_ratio"]), float(soil))]
        for row in selected:
            row["_control_code"] = CONTROL_NAMES.index(str(row[f"{scenario}_control"]))
        closure, velocity, codes = grid_for(selected, "soil_to_pipe_modulus_ratio", float(soil), "closure_time_over_L_cf", "initial_velocity_m_s", "_control_code")
        _, _, utilization = grid_for(selected, "soil_to_pipe_modulus_ratio", float(soil), "closure_time_over_L_cf", "initial_velocity_m_s", f"{scenario}_maximum_utilization")
        axis.pcolormesh(closure, velocity, codes, cmap=cmap, norm=norm, shading="auto", alpha=0.88)
        axis.contour(closure, velocity, utilization, levels=[1.0], colors="black", linewidths=1.5)
        axis.set_title(rf"$E_s/E={soil:g}$")
        axis.set_xlabel(r"$t_c/(L/c_f)$")
    axes[0].set_ylabel(r"Initial velocity $V_0$ (m/s)")
    handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markersize=10, label=name.replace("_", " ")) for name, color in zip(CONTROL_NAMES, colors)]
    figure.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    figure.suptitle("Dominant nominal constraint; black line denotes the joint boundary", y=1.13)
    return save_figure(figure, output, "F29_nominal_velocity_closure_control")


def plot_minimum_closure(minimum_rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> list[str]:
    colors = {float(value): color for value, color in zip(config["scan"]["soil_levels_for_velocity_closure_maps"], ("#4C78A8", "#59A14F", "#E15759"))}
    styles = {"conservative": "--", "nominal": "-", "permissive": ":"}
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    rows = [row for row in minimum_rows if row["slice"] == "velocity"]
    for soil, color in colors.items():
        selected = sorted([row for row in rows if np.isclose(float(row["soil_to_pipe_modulus_ratio"]), soil)], key=lambda row: float(row["initial_velocity_m_s"]))
        for scenario, style in styles.items():
            axis.plot([row["initial_velocity_m_s"] for row in selected], [row[f"{scenario}_minimum_closure_time_over_L_cf"] for row in selected], color=color, ls=style, lw=1.7 if scenario == "nominal" else 1.1)
    soil_handles = [plt.Line2D([0], [0], color=color, lw=2, label=rf"$E_s/E={soil:g}$") for soil, color in colors.items()]
    scenario_handles = [plt.Line2D([0], [0], color="black", ls=style, label=scenario) for scenario, style in styles.items()]
    first = axis.legend(handles=soil_handles, loc="upper left", frameon=False)
    axis.add_artist(first)
    axis.legend(handles=scenario_handles, loc="lower right", frameon=False)
    axis.set(xlabel=r"Initial velocity $V_0$ (m/s)", ylabel=r"Robust minimum $t_c/(L/c_f)$")
    axis.grid(alpha=0.2)
    axis.set_title("Minimum closure time under three registered threshold scenarios")
    figure.tight_layout()
    return save_figure(figure, output, "F30_minimum_closure_vs_velocity")


def plot_soil_closure_maps(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> list[str]:
    scenario = config["primary_scenario"]
    levels = config["scan"]["velocity_levels_for_soil_closure_maps_m_s"]
    figure, axes = plt.subplots(1, len(levels), figsize=(13.0, 3.8), sharex=True, sharey=True)
    for axis, velocity in zip(axes, levels):
        closure, soil, values = grid_for(rows, "initial_velocity_m_s", float(velocity), "closure_time_over_L_cf", "soil_to_pipe_modulus_ratio", f"{scenario}_maximum_utilization")
        contour = axis.contourf(closure, soil, values, levels=np.linspace(0.45, max(1.6, float(np.max(values))), 20), cmap="viridis")
        axis.contour(closure, soil, values, levels=[1.0], colors="white", linewidths=2.0)
        axis.set_yscale("log")
        axis.set_title(rf"$V_0={velocity:.2f}$ m/s")
        axis.set_xlabel(r"$t_c/(L/c_f)$")
    axes[0].set_ylabel(r"Soil restraint $E_s/E$")
    figure.colorbar(contour, ax=axes, pad=0.02, label="Maximum nominal utilization")
    figure.suptitle("Soil-restraint effect on the joint response-threshold boundary", y=1.02)
    return save_figure(figure, output, "F31_nominal_soil_closure_utilization")


def plot_threshold_sensitivity(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> list[str]:
    scenarios = list(config["threshold_scenarios"])
    safe_fraction = [np.mean([bool(row[f"{scenario}_safe"]) for row in rows]) for scenario in scenarios]
    figure, (axis_a, axis_b) = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axis_a.bar(scenarios, np.asarray(safe_fraction) * 100.0, color=("#E15759", "#4C78A8", "#59A14F"))
    axis_a.set_ylabel("Safe fraction of registered scan (%)")
    axis_a.grid(axis="y", alpha=0.2)
    nominal = config["primary_scenario"]
    controls = [str(row[f"{nominal}_control"]) for row in rows if not bool(row[f"{nominal}_safe"])]
    counts = [controls.count(name) for name in CONTROL_NAMES]
    axis_b.bar([name.replace("_", "\n") for name in CONTROL_NAMES], counts, color=("#4C78A8", "#E15759", "#59A14F", "#B07AA1"))
    axis_b.set_ylabel("Nominal unsafe cases")
    axis_b.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return save_figure(figure, output, "F32_threshold_sensitivity_and_control_counts")


def maximin_select(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    coordinates = np.column_stack(
        [
            (np.log([float(row[PARAMETERS[0]]) for row in candidates]) - math.log(0.0003)) / (math.log(0.1) - math.log(0.0003)),
            (np.asarray([float(row[PARAMETERS[1]]) for row in candidates]) - 0.1) / 1.1,
            (np.asarray([float(row[PARAMETERS[2]]) for row in candidates]) - 0.03) / 0.06,
        ]
    )
    selected: list[int] = []
    categories = sorted({(str(row["nominal_control"]), bool(row["nominal_safe"])) for row in candidates})
    for category in categories:
        eligible = [index for index, row in enumerate(candidates) if (str(row["nominal_control"]), bool(row["nominal_safe"])) == category]
        if eligible and len(selected) < count:
            selected.append(min(eligible, key=lambda index: abs(float(candidates[index]["nominal_maximum_utilization"]) - 1.0)))
    while len(selected) < min(count, len(candidates)):
        remaining = [index for index in range(len(candidates)) if index not in selected]
        choice = max(remaining, key=lambda index: min(float(np.linalg.norm(coordinates[index] - coordinates[chosen])) for chosen in selected))
        selected.append(choice)
    return [candidates[index] for index in selected[:count]]


def register_boundary_cases(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    count = int(config["boundary_moc_registration"]["case_count"])
    ordered = sorted(rows, key=lambda row: abs(float(row["nominal_maximum_utilization"]) - 1.0))
    pool = ordered[: max(400, count * 20)]
    selected = maximin_select(pool, count)
    registration = []
    for index, row in enumerate(selected, start=1):
        registration.append(
            {
                "case_id": f"joint_boundary_moc_{index:03d}",
                "split": "engineering_boundary_audit",
                "test_class": "safe_side" if bool(row["nominal_safe"]) else "unsafe_side",
                **{name: float(row[name]) for name in PARAMETERS},
                "registered_model_prediction": {metric: float(row[metric]) for metric in METRICS},
                "registered_nominal_utilization": float(row["nominal_maximum_utilization"]),
                "registered_control": str(row["nominal_control"]),
                "moc_result_not_read_at_registration": True,
            }
        )
    payload = {
        "status": "registered_before_new_MOC_runs",
        "selection_rule": config["boundary_moc_registration"]["selection"],
        "case_count": len(registration),
        "cases": registration,
    }
    path = output / "boundary_moc_case_registration.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    flat = [{key: value for key, value in row.items() if key != "registered_model_prediction"} | row["registered_model_prediction"] for row in registration]
    write_csv(output / "T11_boundary_moc_case_registration.csv", flat)
    return registration


def finalize(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> dict[str, Any]:
    enrich_thresholds(rows, config)
    write_csv(output / "joint_threshold_scan_results.csv", rows)
    minimum_rows = minimum_closure_rows(rows, config)
    write_csv(output / "T10_minimum_closure_time.csv", minimum_rows)
    figures = []
    figures.extend(plot_utilization_maps(rows, config, output))
    figures.extend(plot_control_maps(rows, config, output))
    figures.extend(plot_minimum_closure(minimum_rows, config, output))
    figures.extend(plot_soil_closure_maps(rows, config, output))
    figures.extend(plot_threshold_sensitivity(rows, config, output))
    boundary = register_boundary_cases(rows, config, output)
    summary = {}
    for scenario in config["threshold_scenarios"]:
        summary[scenario] = {
            "safe_fraction": float(np.mean([bool(row[f"{scenario}_safe"]) for row in rows])),
            "maximum_utilization_range": [float(np.min([row[f"{scenario}_maximum_utilization"] for row in rows])), float(np.max([row[f"{scenario}_maximum_utilization"] for row in rows]))],
        }
    report = {
        "status": "pass",
        "case_count": len(rows),
        "terminology": config["terminology"],
        "primary_scenario": config["primary_scenario"],
        "scenario_summary": summary,
        "boundary_moc_cases_registered": len(boundary),
        "boundary_moc_completed": False,
        "figures": figures,
        "tables": ["T10_minimum_closure_time.csv", "T11_boundary_moc_case_registration.csv"],
        "claim_limits": config["claim_limits"],
    }
    (output / "joint_threshold_scan_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(config_path: Path, output: Path, device_name: str, max_cases: int | None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "registered_before_characteristic_model_scan":
        raise ValueError("threshold assessment must be registered before scanning")
    report_path = output / "joint_threshold_scan_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    cases = registered_cases(config)
    if max_cases is not None:
        cases = cases[:max_cases]
    manifest = {
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "case_count": len(cases),
        "cases": cases,
    }
    manifest_path = output / "registered_scan_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("registered scan manifest changed")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    completed = load_completed_chunks(chunks)
    hybrid, _design, model, device = load_model(ROOT / config["source_model"]["config"], ROOT / config["source_model"]["checkpoint"], device_name)
    pending_rows = []
    chunk_index = len(list(chunks.glob("chunk_*.csv"))) + 1
    started = time.perf_counter()
    interval = int(config["scan"]["case_checkpoint_interval"])
    for index, case in enumerate(cases, start=1):
        if case["case_id"] in completed:
            continue
        row = case_metrics(model, hybrid, case, device, config)
        pending_rows.append(row)
        if len(pending_rows) >= interval or index == len(cases):
            path = chunks / f"chunk_{chunk_index:05d}.csv"
            write_csv(path, pending_rows)
            for item in pending_rows:
                completed[item["case_id"]] = item
            print(f"[joint-scan] checkpoint={path.name} completed={len(completed)}/{len(cases)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
            pending_rows = []
            chunk_index += 1
    rows = [completed[case["case_id"]] for case in cases]
    if max_cases is not None:
        smoke = {"status": "pass", "mode": "partial_smoke", "case_count": len(rows), "completed_case_ids": [row["case_id"] for row in rows]}
        (output / "partial_smoke_report.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
        return smoke
    return finalize(rows, config, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.config, args.output_dir, args.device, args.max_cases)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
