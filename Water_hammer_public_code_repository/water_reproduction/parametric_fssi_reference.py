"""Generate independent MOC references for the frozen parametric FSSI study."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from water16_reproduction.common.physics import (
    PhysicalParameters,
    cross_section_areas,
    initial_pressure_pa,
    pressure_wave_speed,
    with_soil_ratio,
)
from water16_reproduction.parametric_fssi_forward import (
    DEFAULT_CONFIG,
    audit_config,
    load_config,
    materialize_cases,
)
from water16_reproduction.research_baseline import (
    DEFAULT_PROVENANCE,
    load_config as load_baseline_config,
    validate_config as validate_baseline_config,
)
from water16_reproduction.wp1_verification import STATE_NAMES
from water16_reproduction.wp2_ablation import reference_solution
from water16_reproduction.wp3_dataset import pressure_validity_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "PINN_FSSI_research_plan" / "outputs" / "parametric_fssi_reference_v1"
)


def load_baseline(config: dict[str, Any]) -> PhysicalParameters:
    path = ROOT / config["baseline_config"]
    baseline = validate_baseline_config(
        load_baseline_config(path), DEFAULT_PROVENANCE
    )
    overrides = dict(config.get("physical_parameter_overrides", {}))
    unknown = sorted(set(overrides) - set(asdict(baseline)))
    if unknown:
        raise ValueError(f"unknown physical parameter overrides: {unknown}")
    return replace(baseline, **overrides) if overrides else baseline


def case_parameters(
    baseline: PhysicalParameters, case: dict[str, Any], t_final_s: float
) -> tuple[PhysicalParameters, float]:
    params = with_soil_ratio(
        replace(
            baseline,
            initial_velocity_m_s=float(case["initial_velocity_m_s"]),
            t_final_s=float(t_final_s),
        ),
        float(case["soil_to_pipe_modulus_ratio"]),
    )
    fluid_speed = pressure_wave_speed(params)
    closure_time = (
        float(case["closure_time_over_L_cf"])
        * params.length_m
        / fluid_speed
    )
    return replace(params, valve_close_time_s=closure_time), fluid_speed


def first_local_peak_index(values: np.ndarray, start_index: int = 1) -> int:
    if len(values) < 3:
        return int(np.argmax(values))
    candidates = np.flatnonzero(
        (values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:])
    ) + 1
    candidates = candidates[candidates >= max(1, int(start_index))]
    return int(candidates[0]) if len(candidates) else int(np.argmax(values))


def engineering_metrics(
    truth: dict[str, np.ndarray],
    params: PhysicalParameters,
    pressure_policy: dict[str, Any],
) -> dict[str, Any]:
    pressure0 = initial_pressure_pa(params)
    area_f, area_t = cross_section_areas(params)
    stress0 = area_f * pressure0 / area_t
    pressure_delta = truth["P"] - pressure0
    stress_delta = truth["sigma_z"] - stress0

    pressure_max_index = np.unravel_index(
        int(np.argmax(pressure_delta)), pressure_delta.shape
    )
    pressure_min_index = np.unravel_index(
        int(np.argmin(truth["P"])), truth["P"].shape
    )
    stress_abs_index = np.unravel_index(
        int(np.argmax(np.abs(stress_delta))), stress_delta.shape
    )

    valve_pressure = truth["P"][-1]
    valve_stress_delta = np.abs(stress_delta[-1])
    start_index = int(np.searchsorted(truth["t"], params.valve_close_time_s))
    pressure_peak_index = first_local_peak_index(valve_pressure, start_index)
    stress_peak_index = first_local_peak_index(valve_stress_delta, start_index)
    validity = pressure_validity_metrics(truth["P"], pressure_policy)

    return {
        "maximum_pressure_increment_pa": float(pressure_delta[pressure_max_index]),
        "maximum_pressure_pa": float(truth["P"][pressure_max_index]),
        "maximum_pressure_location_over_L": float(
            truth["z"][pressure_max_index[0]] / params.length_m
        ),
        "maximum_pressure_time_s": float(truth["t"][pressure_max_index[1]]),
        "minimum_pressure_pa": float(truth["P"][pressure_min_index]),
        "minimum_absolute_pressure_pa": float(
            validity["minimum_absolute_pressure_pa"]
        ),
        "minimum_pressure_location_over_L": float(
            truth["z"][pressure_min_index[0]] / params.length_m
        ),
        "minimum_pressure_time_s": float(truth["t"][pressure_min_index[1]]),
        "maximum_absolute_axial_stress_increment_pa": float(
            np.abs(stress_delta[stress_abs_index])
        ),
        "signed_axial_stress_increment_at_absolute_max_pa": float(
            stress_delta[stress_abs_index]
        ),
        "stress_critical_location_over_L": float(
            truth["z"][stress_abs_index[0]] / params.length_m
        ),
        "stress_critical_time_s": float(truth["t"][stress_abs_index[1]]),
        "valve_pressure_first_peak_time_s": float(truth["t"][pressure_peak_index]),
        "valve_stress_first_peak_time_s": float(truth["t"][stress_peak_index]),
        "cavitation_margin_pa": float(validity["cavitation_margin_pa"]),
        "pressure_validity_status": validity["status"],
    }


def downsample_truth(
    truth: dict[str, np.ndarray], spatial_points: int, time_stride: int
) -> dict[str, np.ndarray]:
    target_x = np.linspace(float(truth["z"][0]), float(truth["z"][-1]), spatial_points)
    time_indices = np.arange(0, len(truth["t"]), time_stride, dtype=int)
    if time_indices[-1] != len(truth["t"]) - 1:
        time_indices = np.append(time_indices, len(truth["t"]) - 1)
    saved: dict[str, np.ndarray] = {
        "x": target_x,
        "t": truth["t"][time_indices],
        "state_names": np.asarray(STATE_NAMES),
    }
    for state in STATE_NAMES:
        values = truth[state][:, time_indices]
        saved[state] = np.stack(
            [np.interp(target_x, truth["z"], values[:, index]) for index in range(values.shape[1])],
            axis=1,
        )
    return saved


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def completed_case_row(
    run_dir: Path,
    case: dict[str, Any],
    expected_n_cells: int,
) -> dict[str, Any] | None:
    """Return a completed saved case without changing either saved artifact."""

    case_dir = run_dir / case["split"] / case["case_id"]
    field_path = case_dir / "truth_evaluation_grid.npz"
    metadata_path = case_dir / "metadata.json"
    if not case_dir.exists():
        return None
    if not field_path.exists() or not metadata_path.exists():
        raise RuntimeError(
            f"incomplete existing case directory requires manual audit: {case_dir}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["case"] != case:
        raise RuntimeError(f"saved case definition differs from frozen design: {case_dir}")
    if int(metadata["truth_solver"]["n_cells"]) != expected_n_cells:
        raise RuntimeError(f"saved grid resolution differs from current protocol: {case_dir}")
    metrics = metadata["engineering_metrics"]
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "test_class": case["test_class"],
        "soil_to_pipe_modulus_ratio": case["soil_to_pipe_modulus_ratio"],
        "closure_time_over_L_cf": case["closure_time_over_L_cf"],
        "initial_velocity_m_s": case["initial_velocity_m_s"],
        "fluid_characteristic_speed_m_s": metadata["derived"][
            "fluid_characteristic_speed_m_s"
        ],
        "closure_time_s": metadata["derived"]["closure_time_s"],
        **metrics,
        "runtime_s": metadata.get("generation_runtime_s"),
        "case_directory": (Path(case["split"]) / case["case_id"]).as_posix(),
    }


def generate_reference_dataset(
    config: dict[str, Any], mode: str, output_dir: Path
) -> dict[str, Any]:
    config_audit = audit_config(config)
    if config_audit["status"] != "pass":
        raise ValueError(f"frozen design audit failed: {config_audit['issues']}")
    stage = "pilot" if mode in {"smoke", "pilot"} else "formal"
    cases = materialize_cases(config, stage)
    solver = config["truth_solver"][mode]
    if mode == "smoke":
        cases = cases[: int(solver["case_limit"])]
    run_dir = output_dir / mode
    report_path = run_dir / "reference_dataset_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline(config)
    storage = config["truth_solver"]["storage"]
    pressure_policy = {
        "pressure_convention": config["pressure_validity"]["model_pressure_convention"],
        "atmospheric_pressure_pa": config["pressure_validity"]["atmospheric_pressure_pa"],
        "vapor_pressure_pa": config["pressure_validity"]["vapor_pressure_pa"],
        "minimum_cavitation_margin_pa": config["pressure_validity"]["minimum_cavitation_margin_pa"],
    }
    times = np.linspace(0.0, float(solver["t_final_s"]), int(solver["output_points"]))
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        completed = completed_case_row(run_dir, case, int(solver["n_cells"]))
        if completed is not None:
            rows.append(completed)
            print(
                f"[{index}/{len(cases)}] {case['case_id']} resumed from saved case",
                flush=True,
            )
            continue
        params, fluid_speed = case_parameters(baseline, case, float(solver["t_final_s"]))
        case_started = time.perf_counter()
        truth = reference_solution(
            params,
            times,
            n_cells=int(solver["n_cells"]),
            cfl=float(config["truth_solver"]["cfl"]),
        )
        metrics = engineering_metrics(truth, params, pressure_policy)
        if (
            config["pressure_validity"]["enforce_for_train_validation_test"]
            and metrics["pressure_validity_status"] != "pass"
        ):
            raise RuntimeError(
                f"{case['case_id']} failed cavitation margin: "
                f"{metrics['cavitation_margin_pa']:.6g} Pa"
            )
        saved = downsample_truth(
            truth,
            int(storage["saved_spatial_points"]),
            int(storage["saved_time_stride"]),
        )
        case_dir = run_dir / case["split"] / case["case_id"]
        case_dir.mkdir(parents=True)
        np.savez_compressed(case_dir / "truth_evaluation_grid.npz", **saved)
        metadata = {
            "experiment_id": config["experiment_id"],
            "case": case,
            "physical_parameters": asdict(params),
            "derived": {
                "fluid_characteristic_speed_m_s": fluid_speed,
                "closure_time_s": params.valve_close_time_s,
            },
            "truth_solver": {
                "method": config["truth_solver"]["method"],
                "uses_neural_model": False,
                "n_cells": int(solver["n_cells"]),
                "output_points_working": int(solver["output_points"]),
                "cfl": float(config["truth_solver"]["cfl"]),
            },
            "saved_field_shape": [
                int(storage["saved_spatial_points"]),
                int(len(saved["t"])),
            ],
            "engineering_metrics": metrics,
            "generation_runtime_s": time.perf_counter() - case_started,
        }
        (case_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "split": case["split"],
                "test_class": case["test_class"],
                "soil_to_pipe_modulus_ratio": case["soil_to_pipe_modulus_ratio"],
                "closure_time_over_L_cf": case["closure_time_over_L_cf"],
                "initial_velocity_m_s": case["initial_velocity_m_s"],
                "fluid_characteristic_speed_m_s": fluid_speed,
                "closure_time_s": params.valve_close_time_s,
                **metrics,
                "runtime_s": metadata["generation_runtime_s"],
                "case_directory": (Path(case["split"]) / case["case_id"]).as_posix(),
            }
        )
        print(
            f"[{index}/{len(cases)}] {case['case_id']} "
            f"margin={metrics['cavitation_margin_pa']:.3e} Pa",
            flush=True,
        )
        _write_manifest(run_dir / "manifest.csv", rows)
    _write_manifest(run_dir / "manifest.csv", rows)
    report = {
        "status": "pass",
        "experiment_id": config["experiment_id"],
        "mode": mode,
        "case_count": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in sorted({row["split"] for row in rows})
        },
        "minimum_cavitation_margin_pa": min(row["cavitation_margin_pa"] for row in rows),
        "truth_solver": {
            **solver,
            "cfl": config["truth_solver"]["cfl"],
            "uses_neural_model": False,
        },
        "runtime_s": time.perf_counter() - started,
        "manifest": f"{mode}/manifest.csv",
    }
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def audit_reference_dataset(
    config: dict[str, Any], mode: str, output_dir: Path
) -> dict[str, Any]:
    run_dir = output_dir / mode
    report_path = run_dir / "reference_dataset_report.json"
    manifest_path = run_dir / "manifest.csv"
    issues: list[str] = []
    if not report_path.exists() or not manifest_path.exists():
        issues.append("report or manifest is missing")
        rows: list[dict[str, str]] = []
    else:
        with manifest_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    for row in rows:
        case_dir = run_dir / row["case_directory"]
        field_path = case_dir / "truth_evaluation_grid.npz"
        metadata_path = case_dir / "metadata.json"
        if not field_path.exists() or not metadata_path.exists():
            issues.append(f"{row['case_id']}: missing field or metadata")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["truth_solver"].get("uses_neural_model", True):
            issues.append(f"{row['case_id']}: reference is not independent of neural model")
        with np.load(field_path, allow_pickle=False) as archive:
            expected = tuple(metadata["saved_field_shape"])
            for state in STATE_NAMES:
                if archive[state].shape != expected:
                    issues.append(f"{row['case_id']}:{state} shape mismatch")
                if not np.all(np.isfinite(archive[state])):
                    issues.append(f"{row['case_id']}:{state} has non-finite values")
    return {
        "status": "pass" if not issues else "failed",
        "mode": mode,
        "case_count": len(rows),
        "issues": issues,
    }


def normalized_field_rmse(reference: np.ndarray, comparison: np.ndarray) -> float:
    scale = max(float(np.ptp(reference)), float(np.max(np.abs(reference))), 1.0e-30)
    return float(np.sqrt(np.mean((comparison - reference) ** 2)) / scale)


def run_grid_convergence(
    config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    convergence = config["truth_solver"]["grid_convergence_check"]
    run_dir = output_dir / "grid_convergence"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {run_dir}")
    run_dir.mkdir(parents=True)
    pilot_by_id = {
        case["case_id"]: case for case in materialize_cases(config, "pilot")
    }
    cell_counts = [int(value) for value in convergence["cell_counts"]]
    if cell_counts != sorted(cell_counts) or len(cell_counts) < 3:
        raise ValueError("grid convergence requires at least three increasing grids")
    solver = config["truth_solver"]["pilot"]
    storage = config["truth_solver"]["storage"]
    times = np.linspace(0.0, float(solver["t_final_s"]), int(solver["output_points"]))
    baseline = load_baseline(config)
    case_reports: list[dict[str, Any]] = []
    for case_id in convergence["registered_cases"]:
        if case_id not in pilot_by_id:
            raise ValueError(f"unknown registered grid case: {case_id}")
        case = pilot_by_id[case_id]
        params, _ = case_parameters(baseline, case, float(solver["t_final_s"]))
        solutions: dict[int, dict[str, np.ndarray]] = {}
        runtimes: dict[int, float] = {}
        for n_cells in cell_counts:
            started = time.perf_counter()
            truth = reference_solution(
                params,
                times,
                n_cells=n_cells,
                cfl=float(config["truth_solver"]["cfl"]),
            )
            runtimes[n_cells] = time.perf_counter() - started
            solutions[n_cells] = downsample_truth(
                truth,
                int(storage["saved_spatial_points"]),
                int(storage["saved_time_stride"]),
            )
        comparisons: list[dict[str, Any]] = []
        for coarse, fine in zip(cell_counts[:-1], cell_counts[1:]):
            state_errors = {
                state: normalized_field_rmse(
                    solutions[fine][state], solutions[coarse][state]
                )
                for state in STATE_NAMES
            }
            comparisons.append(
                {
                    "coarse_cells": coarse,
                    "fine_cells": fine,
                    "state_nrmse": state_errors,
                    "maximum_four_state_nrmse": max(state_errors.values()),
                }
            )
        case_reports.append(
            {
                "case_id": case_id,
                "case": case,
                "runtime_s_by_cells": {str(key): value for key, value in runtimes.items()},
                "comparisons": comparisons,
            }
        )
        print(f"grid convergence completed: {case_id}", flush=True)

    last_comparisons = [case["comparisons"][-1] for case in case_reports]
    worst_pressure = max(item["state_nrmse"]["P"] for item in last_comparisons)
    worst_four_state = max(
        item["maximum_four_state_nrmse"] for item in last_comparisons
    )
    pressure_limit = float(convergence["maximum_pressure_nrmse_final_pair"])
    four_state_limit = float(convergence["maximum_four_state_nrmse_final_pair"])
    report = {
        "status": (
            "pass"
            if worst_pressure <= pressure_limit and worst_four_state <= four_state_limit
            else "failed"
        ),
        "registered_cases": list(convergence["registered_cases"]),
        "cell_counts": cell_counts,
        "case_reports": case_reports,
        "acceptance": {
            "final_grid_pair": cell_counts[-2:],
            "worst_pressure_nrmse_final_pair": worst_pressure,
            "pressure_limit": pressure_limit,
            "worst_four_state_nrmse_final_pair": worst_four_state,
            "four_state_limit": four_state_limit,
        },
    }
    (run_dir / "grid_convergence_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--grid-convergence", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.grid_convergence:
        report = run_grid_convergence(config, args.output_dir)
    elif args.audit:
        report = audit_reference_dataset(config, args.mode, args.output_dir)
    else:
        report = generate_reference_dataset(config, args.mode, args.output_dir)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
