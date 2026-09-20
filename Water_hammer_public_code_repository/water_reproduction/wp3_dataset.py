"""Generate the WP3 synthetic FSSI data library with an independent MOC.

The generator never calls a PINN.  It saves the full clean four-state field and
separate sparse observation files with controlled white noise, sensor offsets
and sampling-rate reduction.  Formal generation remains intentionally blocked
until WP2.5 freezes the converged MOC resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from water16_reproduction.common.physics import (
    PhysicalParameters,
    pressure_wave_speed,
    with_soil_ratio,
)
from water16_reproduction.wp1_verification import STATE_NAMES
from water16_reproduction.wp2_ablation import (
    load_physical_case,
    reference_solution,
    sample_reference,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "PINN_FSSI_research_plan" / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "wp3_synthetic_dataset_v1.json"
V2_CONFIG = CONFIG_DIR / "wp3_synthetic_dataset_v2.json"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan" / "outputs" / "wp3"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("WP3 config schema_version must be 1")
    cases = config.get("cases", [])
    identifiers = [case["case_id"] for case in cases]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("WP3 case_id values must be non-empty and unique")
    train = {case["case_id"] for case in cases if case["split"] == "train"}
    test = {case["case_id"] for case in cases if case["split"] != "train"}
    if not train or not test or train & test:
        raise ValueError("WP3 train and non-train cases must be distinct")
    for scenario in config["observation_scenarios"]:
        if int(scenario["sample_stride"]) < 1:
            raise ValueError("sample_stride must be positive")
        if float(scenario["white_noise_fraction"]) < 0.0:
            raise ValueError("white_noise_fraction cannot be negative")
    scale_definition = config.get(
        "noise_scale_definition", "max_peak_to_peak_or_absolute"
    )
    if scale_definition not in {
        "max_peak_to_peak_or_absolute",
        "peak_to_peak_with_absolute_fallback",
    }:
        raise ValueError(f"unsupported noise_scale_definition: {scale_definition}")
    pressure = config.get("pressure_validity")
    if pressure is not None:
        if pressure.get("pressure_convention") not in {"gauge", "absolute"}:
            raise ValueError("pressure_convention must be gauge or absolute")
        for name in (
            "atmospheric_pressure_pa",
            "vapor_pressure_pa",
            "minimum_cavitation_margin_pa",
        ):
            if float(pressure[name]) < 0.0:
                raise ValueError(f"pressure_validity.{name} cannot be negative")
    return config


def apply_physical_overrides(
    baseline: PhysicalParameters, config: dict[str, Any]
) -> PhysicalParameters:
    overrides = dict(config.get("physical_parameter_overrides", {}))
    unknown = sorted(set(overrides) - set(asdict(baseline)))
    if unknown:
        raise ValueError(f"unknown physical parameter overrides: {unknown}")
    return replace(baseline, **overrides) if overrides else baseline


def parameters_for_case(
    baseline: PhysicalParameters,
    case: dict[str, Any],
    t_final_s: float,
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


def clean_sensor_values(
    truth: dict[str, np.ndarray],
    params: PhysicalParameters,
    positions_over_l: list[float],
    states: list[str],
) -> np.ndarray:
    values = np.empty(
        (len(positions_over_l), len(states), len(truth["t"])), dtype=float
    )
    for ix, ratio in enumerate(positions_over_l):
        position = float(ratio) * params.length_m
        for iy, state in enumerate(states):
            values[ix, iy] = sample_reference(truth, position, state)
    return values


def noisy_observation(
    clean: np.ndarray,
    scenario: dict[str, Any],
    rng: np.random.Generator,
    scale_definition: str = "max_peak_to_peak_or_absolute",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peak_to_peak = np.ptp(clean, axis=-1)
    fallback = np.max(np.abs(clean), axis=-1)
    if scale_definition == "max_peak_to_peak_or_absolute":
        scale = np.maximum(peak_to_peak, np.maximum(fallback, 1.0e-30))
    elif scale_definition == "peak_to_peak_with_absolute_fallback":
        tolerance = np.finfo(float).eps * np.maximum(fallback, 1.0)
        scale = np.where(
            peak_to_peak > tolerance,
            peak_to_peak,
            np.maximum(fallback, 1.0e-30),
        )
    else:
        raise ValueError(f"unsupported noise scale definition: {scale_definition}")
    offsets = (
        rng.normal(size=clean.shape[:2])
        * float(scenario["zero_offset_fraction"])
        * scale
    )
    white = (
        rng.normal(size=clean.shape)
        * float(scenario["white_noise_fraction"])
        * scale[:, :, None]
    )
    noisy = clean + offsets[:, :, None] + white
    return noisy, offsets, scale


def pressure_validity_metrics(
    pressure: np.ndarray, policy: dict[str, Any] | None
) -> dict[str, Any]:
    minimum_model_pressure = float(np.min(pressure))
    if policy is None:
        return {
            "status": "not_assessed",
            "minimum_model_pressure_pa": minimum_model_pressure,
        }
    convention = str(policy["pressure_convention"])
    atmospheric = float(policy["atmospheric_pressure_pa"])
    vapor = float(policy["vapor_pressure_pa"])
    required_margin = float(policy["minimum_cavitation_margin_pa"])
    minimum_absolute = (
        minimum_model_pressure + atmospheric
        if convention == "gauge"
        else minimum_model_pressure
    )
    actual_margin = minimum_absolute - vapor
    accepted = actual_margin >= required_margin
    return {
        "status": "pass" if accepted else "failed",
        "pressure_convention": convention,
        "minimum_model_pressure_pa": minimum_model_pressure,
        "minimum_absolute_pressure_pa": minimum_absolute,
        "vapor_pressure_pa": vapor,
        "cavitation_margin_pa": actual_margin,
        "required_cavitation_margin_pa": required_margin,
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_case(
    config: dict[str, Any],
    solver: dict[str, Any],
    baseline: PhysicalParameters,
    case: dict[str, Any],
    case_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    params, fluid_speed = parameters_for_case(
        baseline, case, float(solver["t_final_s"])
    )
    times = np.linspace(
        0.0, params.t_final_s, int(solver["output_points"]), dtype=float
    )
    truth = reference_solution(
        params,
        times,
        n_cells=int(solver["n_cells"]),
        cfl=float(config["truth_solver"]["cfl"]),
    )
    pressure_validity = pressure_validity_metrics(
        truth["P"], config.get("pressure_validity")
    )
    if (
        config.get("pressure_validity", {}).get("enforce", False)
        and pressure_validity["status"] != "pass"
    ):
        raise RuntimeError(
            f"{case['case_id']} violates the cavitation-margin criterion: "
            f"{pressure_validity['cavitation_margin_pa']:.6g} Pa available, "
            f"{pressure_validity['required_cavitation_margin_pa']:.6g} Pa required"
        )
    case_dir = output_dir / case["split"] / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        case_dir / "truth.npz",
        x=truth["z"],
        t=truth["t"],
        state_names=np.asarray(STATE_NAMES),
        **{state: truth[state] for state in STATE_NAMES},
    )

    positions = [float(value) for value in config["sensors"]["positions_over_L"]]
    observed_states = list(config["sensors"]["observed_states"])
    clean = clean_sensor_values(truth, params, positions, observed_states)
    scenario_files: list[str] = []
    for scenario_index, scenario in enumerate(config["observation_scenarios"]):
        rng = np.random.default_rng(
            int(config["random_seed"]) + 100 * case_index + scenario_index
        )
        noisy, offsets, scales = noisy_observation(
            clean,
            scenario,
            rng,
            str(
                config.get(
                    "noise_scale_definition", "max_peak_to_peak_or_absolute"
                )
            ),
        )
        stride = int(scenario["sample_stride"])
        filename = f"observations_{scenario['name']}.npz"
        np.savez_compressed(
            case_dir / filename,
            t=truth["t"][::stride],
            sensor_positions_over_L=np.asarray(positions),
            observed_state_names=np.asarray(observed_states),
            clean=clean[:, :, ::stride],
            noisy=noisy[:, :, ::stride],
            sensor_offsets=offsets,
            signal_scales=scales,
            white_noise_fraction=np.asarray(
                float(scenario["white_noise_fraction"])
            ),
            zero_offset_fraction=np.asarray(
                float(scenario["zero_offset_fraction"])
            ),
            sample_stride=np.asarray(stride),
            noise_scale_definition=np.asarray(
                config.get(
                    "noise_scale_definition", "max_peak_to_peak_or_absolute"
                )
            ),
        )
        scenario_files.append(filename)

    metadata = {
        "dataset_id": config["dataset_id"],
        "case": case,
        "physical_parameters": asdict(params),
        "derived": {
            "fluid_characteristic_speed_m_s": fluid_speed,
            "closure_time_s": params.valve_close_time_s,
            "closure_time_over_L_cf": case["closure_time_over_L_cf"],
        },
        "truth_solver": {
            "method": config["truth_solver"]["method"],
            "n_cells": int(solver["n_cells"]),
            "cfl": float(config["truth_solver"]["cfl"]),
            "output_points": int(solver["output_points"]),
            "uses_pinn": False,
        },
        "pressure_validity": pressure_validity,
        "noise_scale_definition": config.get(
            "noise_scale_definition", "max_peak_to_peak_or_absolute"
        ),
        "truth_file": "truth.npz",
        "observation_files": scenario_files,
        "deferred_physics": config["deferred_physics"],
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "soil_to_pipe_modulus_ratio": case["soil_to_pipe_modulus_ratio"],
        "closure_time_over_L_cf": case["closure_time_over_L_cf"],
        "closure_time_s": params.valve_close_time_s,
        "initial_velocity_m_s": params.initial_velocity_m_s,
        "fluid_characteristic_speed_m_s": fluid_speed,
        "n_cells": int(solver["n_cells"]),
        "output_points": int(solver["output_points"]),
        "case_directory": (Path(case["split"]) / case["case_id"]).as_posix(),
        "minimum_absolute_pressure_pa": pressure_validity.get(
            "minimum_absolute_pressure_pa"
        ),
        "cavitation_margin_pa": pressure_validity.get("cavitation_margin_pa"),
        "pressure_validity_status": pressure_validity["status"],
    }


def generate_dataset(
    config: dict[str, Any], mode: str, output_dir: Path
) -> dict[str, Any]:
    solver = config["truth_solver"][mode]
    if solver["n_cells"] is None:
        raise RuntimeError(
            "formal MOC resolution is not frozen; complete WP2.5 and update "
            "wp3_synthetic_dataset_v1.json before formal generation"
        )
    baseline = apply_physical_overrides(load_physical_case(config), config)
    cases = list(config["cases"])
    if mode == "smoke":
        cases = cases[: int(solver["case_limit"])]
    run_dir = output_dir / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        generate_case(config, solver, baseline, case, index, run_dir)
        for index, case in enumerate(cases)
    ]
    _write_manifest(run_dir / "manifest.csv", rows)
    assessed_pressure = [
        row for row in rows if row["pressure_validity_status"] != "not_assessed"
    ]
    pressure_summary = (
        {
            "status": (
                "pass"
                if all(row["pressure_validity_status"] == "pass" for row in assessed_pressure)
                else "failed"
            ),
            "minimum_absolute_pressure_pa": min(
                float(row["minimum_absolute_pressure_pa"])
                for row in assessed_pressure
            ),
            "minimum_cavitation_margin_pa": min(
                float(row["cavitation_margin_pa"]) for row in assessed_pressure
            ),
            "policy": config["pressure_validity"],
        }
        if assessed_pressure
        else {"status": "not_assessed"}
    )
    report = {
        "status": "pass",
        "dataset_id": config["dataset_id"],
        "mode": mode,
        "case_count": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in sorted({row["split"] for row in rows})
        },
        "truth_solver": {
            **solver,
            "method": config["truth_solver"]["method"],
            "cfl": config["truth_solver"]["cfl"],
            "uses_pinn": False,
        },
        "observation_scenarios": config["observation_scenarios"],
        "formal_resolution_status": config["status"],
        "pressure_validity": pressure_summary,
        "noise_scale_definition": config.get(
            "noise_scale_definition", "max_peak_to_peak_or_absolute"
        ),
        "manifest": (Path(mode) / "manifest.csv").as_posix(),
    }
    (run_dir / "wp3_dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def audit_dataset(
    config: dict[str, Any], mode: str, output_dir: Path
) -> dict[str, Any]:
    solver = config["truth_solver"][mode]
    run_dir = output_dir / mode
    cases = list(config["cases"])
    if mode == "smoke":
        cases = cases[: int(solver["case_limit"])]
    issues: list[str] = []
    pressure_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    expected_truth_shape = (
        int(solver["n_cells"]) + 1,
        int(solver["output_points"]),
    )
    for case in cases:
        case_dir = run_dir / case["split"] / case["case_id"]
        truth_path = case_dir / "truth.npz"
        metadata_path = case_dir / "metadata.json"
        if not truth_path.exists() or not metadata_path.exists():
            issues.append(f"{case['case_id']}: missing truth.npz or metadata.json")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if bool(metadata.get("truth_solver", {}).get("uses_pinn", True)):
            issues.append(f"{case['case_id']}: truth is not marked independent of PINN")
        with np.load(truth_path) as truth:
            for state in STATE_NAMES:
                values = truth[state]
                if values.shape != expected_truth_shape:
                    issues.append(
                        f"{case['case_id']}:{state} has shape {values.shape}, "
                        f"expected {expected_truth_shape}"
                    )
                if not np.all(np.isfinite(values)):
                    issues.append(f"{case['case_id']}:{state} contains non-finite values")
            pressure_row = {
                "case_id": case["case_id"],
                **pressure_validity_metrics(
                    truth["P"], config.get("pressure_validity")
                ),
            }
            pressure_rows.append(pressure_row)
        for scenario in config["observation_scenarios"]:
            observation_path = case_dir / f"observations_{scenario['name']}.npz"
            if not observation_path.exists():
                issues.append(
                    f"{case['case_id']}: missing {observation_path.name}"
                )
                continue
            with np.load(observation_path) as observation:
                stride = int(scenario["sample_stride"])
                expected_samples = (
                    int(solver["output_points"]) - 1
                ) // stride + 1
                expected_shape = (
                    len(config["sensors"]["positions_over_L"]),
                    len(config["sensors"]["observed_states"]),
                    expected_samples,
                )
                if observation["clean"].shape != expected_shape:
                    issues.append(
                        f"{case['case_id']}:{observation_path.name} has shape "
                        f"{observation['clean'].shape}, expected {expected_shape}"
                    )
                if not np.all(np.isfinite(observation["noisy"])):
                    issues.append(
                        f"{case['case_id']}:{observation_path.name} contains "
                        "non-finite observations"
                    )
                normalized_white = (
                    observation["noisy"]
                    - observation["clean"]
                    - observation["sensor_offsets"][:, :, None]
                ) / observation["signal_scales"][:, :, None]
                noise_rows.append(
                    {
                        "case_id": case["case_id"],
                        "scenario": scenario["name"],
                        "configured_white_noise_fraction": float(
                            scenario["white_noise_fraction"]
                        ),
                        "realized_white_noise_rms_fraction": float(
                            np.sqrt(np.mean(normalized_white**2))
                        ),
                        "samples": expected_samples,
                    }
                )
    physical_pass = bool(pressure_rows) and all(
        row["status"] in {"pass", "not_assessed"} for row in pressure_rows
    )
    numerical_pass = not issues and len(pressure_rows) == len(cases)
    enforce_pressure = bool(
        config.get("pressure_validity", {}).get("enforce", False)
    )
    if numerical_pass and physical_pass:
        status = "pass"
    elif numerical_pass and not enforce_pressure:
        status = "warning"
    else:
        status = "failed"
    report = {
        "status": status,
        "dataset_id": config["dataset_id"],
        "mode": mode,
        "case_count_expected": len(cases),
        "case_count_audited": len(pressure_rows),
        "numerical_integrity": "pass" if numerical_pass else "failed",
        "pressure_validity": {
            "status": "pass" if physical_pass else "failed",
            "enforced": enforce_pressure,
            "minimum_absolute_pressure_pa": (
                min(
                    float(row["minimum_absolute_pressure_pa"])
                    for row in pressure_rows
                    if "minimum_absolute_pressure_pa" in row
                )
                if pressure_rows and config.get("pressure_validity")
                else None
            ),
            "minimum_cavitation_margin_pa": (
                min(
                    float(row["cavitation_margin_pa"])
                    for row in pressure_rows
                    if "cavitation_margin_pa" in row
                )
                if pressure_rows and config.get("pressure_validity")
                else None
            ),
            "cases": pressure_rows,
        },
        "noise_realizations": noise_rows,
        "issues": issues,
    }
    (run_dir / "wp3_dataset_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("smoke", "pilot", "formal"), default="smoke"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="audit an existing dataset without regenerating MOC truth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report = (
        audit_dataset(config, args.mode, args.output_dir)
        if args.audit_only
        else generate_dataset(config, args.mode, args.output_dir)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
