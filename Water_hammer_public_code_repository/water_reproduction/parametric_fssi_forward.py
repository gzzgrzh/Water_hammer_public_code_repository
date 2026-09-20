"""Audit and materialise the frozen three-parameter FSSI forward design.

This module does not train a neural model and does not generate MOC fields.  It
freezes the parameter coordinates and split membership used by the reference
data generator and by every later model comparison.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "PINN_FSSI_research_plan"
    / "configs"
    / "parametric_fssi_forward_v1.json"
)
PARAMETER_NAMES = (
    "soil_to_pipe_modulus_ratio",
    "closure_time_over_L_cf",
    "initial_velocity_m_s",
)
SPLITS = ("train", "validation", "test")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("parametric FSSI config schema_version must be 1")
    return config


def _physical_value(unit_value: float, definition: dict[str, Any]) -> float:
    lower = float(definition["lower"])
    upper = float(definition["upper"])
    if definition["coordinate"] == "linear":
        return lower + unit_value * (upper - lower)
    if definition["coordinate"] == "natural_log":
        return float(np.exp(np.log(lower) + unit_value * np.log(upper / lower)))
    raise ValueError(f"unsupported coordinate: {definition['coordinate']}")


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=float)
    for dimension in range(dimensions):
        result[:, dimension] = (rng.permutation(count) + rng.random(count)) / count
    return result


def _case_key(case: dict[str, Any]) -> tuple[float, ...]:
    return tuple(round(float(case[name]), 12) for name in PARAMETER_NAMES)


def materialize_cases(config: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "pilot":
        return [dict(case) for case in config["case_design"]["pilot"]["cases"]]
    if stage != "formal":
        raise ValueError("stage must be pilot or formal")

    design = config["case_design"]["formal"]
    fraction_lower, fraction_upper = map(float, design["interior_fraction_bounds"])
    domain = config["parameter_domain"]
    cases: list[dict[str, Any]] = []
    for generated in design["generated_splits"]:
        count = int(generated["count"])
        samples = _latin_hypercube(count, len(PARAMETER_NAMES), int(generated["seed"]))
        samples = fraction_lower + samples * (fraction_upper - fraction_lower)
        for index, row in enumerate(samples, start=1):
            case = {
                "case_id": f"formal_{generated['split']}_{index:03d}",
                "split": generated["split"],
                "test_class": generated["test_class"],
            }
            for parameter, unit_value in zip(PARAMETER_NAMES, row):
                case[parameter] = _physical_value(float(unit_value), domain[parameter])
            cases.append(case)
    cases.extend(dict(case) for case in design["registered_test_cases"])
    return cases


def audit_config(config: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if config.get("status") != "frozen_for_p1_2":
        issues.append("configuration is not frozen_for_p1_2")
    if tuple(config.get("model_scope", {}).get("state_order", ())) != (
        "V", "uz", "P", "sigma_z"
    ):
        issues.append("state_order must be [V, uz, P, sigma_z]")

    domain = config.get("parameter_domain", {})
    if set(domain) != set(PARAMETER_NAMES):
        issues.append("parameter domain must contain exactly the registered parameters")
    for name in PARAMETER_NAMES:
        definition = domain.get(name, {})
        if float(definition.get("lower", 0.0)) >= float(definition.get("upper", 0.0)):
            issues.append(f"invalid range for {name}")
        if definition.get("coordinate") not in {"linear", "natural_log"}:
            issues.append(f"invalid coordinate for {name}")

    stage_reports: dict[str, Any] = {}
    for stage in ("pilot", "formal"):
        cases = materialize_cases(config, stage)
        identifiers = [case["case_id"] for case in cases]
        keys = [_case_key(case) for case in cases]
        counts = Counter(case["split"] for case in cases)
        expected = config["case_design"][stage]["expected_counts"]
        if len(identifiers) != len(set(identifiers)):
            issues.append(f"{stage}: duplicate case_id")
        if len(keys) != len(set(keys)):
            issues.append(f"{stage}: duplicate parameter point across splits")
        if set(counts) != set(SPLITS):
            issues.append(f"{stage}: train/validation/test splits are required")
        for split in SPLITS:
            if counts[split] != int(expected[split]):
                issues.append(
                    f"{stage}: {split} count {counts[split]} != {expected[split]}"
                )
        for case in cases:
            for name in PARAMETER_NAMES:
                value = float(case[name])
                if not float(domain[name]["lower"]) <= value <= float(domain[name]["upper"]):
                    issues.append(f"{case['case_id']}: {name} outside registered domain")
        stage_reports[stage] = {
            "case_count": len(cases),
            "split_counts": {split: counts[split] for split in SPLITS},
            "test_class_counts": dict(Counter(case["test_class"] for case in cases)),
            "unique_parameter_points": len(set(keys)),
        }

    forbidden = set(config["training_data_policy"]["forbidden"])
    for required in {"validation cases", "test cases", "MOC full-field values"}:
        if required not in forbidden:
            issues.append(f"training data policy must forbid {required}")
    validation = config.get("validation_plan", {})
    if not validation.get("classic_equation_check"):
        issues.append("classic equation validation is missing")
    if not validation.get("independent_multicase_check"):
        issues.append("independent multi-case validation is missing")
    if len(validation.get("external_checks", [])) < 2:
        issues.append("at least two external feature/trend checks are required")
    if config["parameter_analysis"].get("start_condition") != (
        "classic and held-out numerical validation passed"
    ):
        issues.append("parameter analysis must be gated by validation")

    return {
        "status": "pass" if not issues else "failed",
        "experiment_id": config.get("experiment_id"),
        "parameter_names": list(PARAMETER_NAMES),
        "stages": stage_reports,
        "validation_layers": {
            "classic": validation.get("classic_equation_check"),
            "held_out": validation.get("independent_multicase_check"),
            "external_source_count": len(validation.get("external_checks", [])),
        },
        "parameter_analysis_gate": config["parameter_analysis"].get("start_condition"),
        "issues": issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("pilot", "formal"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.stage:
        result: Any = materialize_cases(config, args.stage)
    else:
        result = audit_config(config)
    print(json.dumps(result, indent=2))
    if isinstance(result, dict) and result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
