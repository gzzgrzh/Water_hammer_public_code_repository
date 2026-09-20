"""Validate the frozen WP0 research baseline without starting PINN training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from water16_reproduction.common.physics import (
    PhysicalParameters,
    characteristic_basis,
    cross_section_areas,
    fssi_coefficients,
    initial_pressure_pa,
    soil_ratio,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT / "PINN_FSSI_research_plan"
DEFAULT_CONFIG = RESEARCH_ROOT / "configs" / "research_baseline_v1.json"
DEFAULT_PROVENANCE = RESEARCH_ROOT / "parameter_provenance.csv"
DEFAULT_OUTPUT = RESEARCH_ROOT / "outputs" / "wp0_baseline_audit.json"

REQUIRED_TOP_LEVEL = {
    "schema_version",
    "case_id",
    "status",
    "purpose",
    "model_definition",
    "coordinate_convention",
    "physical_parameters",
    "initial_conditions",
    "boundary_conditions",
    "reference_solver",
    "reproducibility",
}

POSITIVE_PARAMETERS = {
    "length_m",
    "inner_radius_m",
    "wall_thickness_m",
    "pipe_E_pa",
    "pipe_density_kg_m3",
    "water_density_kg_m3",
    "water_bulk_modulus_pa",
    "soil_E_pa",
    "gravity_m_s2",
    "head_difference_m",
    "initial_velocity_m_s",
    "valve_mass_kg",
    "valve_close_time_s",
    "t_final_s",
}

CLAIM_PATTERNS = {
    "inner_radius_m": re.compile(
        r"(?:^|[,;\s])R\s*=\s*([0-9.eE+-]+)\s*m", re.IGNORECASE
    ),
    "initial_velocity_m_s": re.compile(
        r"V0\s*=\s*([0-9.eE+-]+)\s*m/s", re.IGNORECASE
    ),
    "valve_close_time_s": re.compile(
        r"tc\s*=\s*([0-9.eE+-]+)\s*s", re.IGNORECASE
    ),
}


class BaselineValidationError(ValueError):
    """Raised when the frozen baseline is internally inconsistent."""


def _canonical_hash(data: dict[str, Any]) -> str:
    encoded = json.dumps(
        data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise BaselineValidationError("baseline config must contain a JSON object")
    return data


def _compare_csv_value(parameter: str, csv_value: str, config_value: Any) -> bool:
    if isinstance(config_value, (int, float)) and not isinstance(config_value, bool):
        try:
            return math.isclose(
                float(csv_value), float(config_value), rel_tol=1.0e-12, abs_tol=0.0
            )
        except ValueError:
            return False
    return csv_value == str(config_value)


def validate_provenance(
    config: dict[str, Any], path: Path = DEFAULT_PROVENANCE
) -> tuple[list[str], list[dict[str, str]]]:
    errors: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    by_path: dict[str, dict[str, str]] = {}
    for row in rows:
        config_path = row.get("config_path", "")
        if not config_path:
            errors.append("provenance row has an empty config_path")
            continue
        if config_path in by_path:
            errors.append(f"duplicate provenance config_path: {config_path}")
        by_path[config_path] = row

    for name, value in config["physical_parameters"].items():
        config_path = f"physical_parameters.{name}"
        row = by_path.get(config_path)
        if row is None:
            errors.append(f"missing provenance row: {config_path}")
            continue
        if row.get("parameter") != name:
            errors.append(
                f"provenance parameter mismatch for {config_path}: "
                f"{row.get('parameter')!r}"
            )
        if not _compare_csv_value(name, row.get("value", ""), value):
            errors.append(
                f"provenance value mismatch for {config_path}: "
                f"csv={row.get('value')!r}, config={value!r}"
            )
        for required in (
            "unit",
            "source_class",
            "source_reference",
            "use_status",
            "uncertainty_or_limitation",
        ):
            if not row.get(required, "").strip():
                errors.append(f"{config_path} has empty provenance field {required}")
    return errors, rows


def validate_config(
    config: dict[str, Any], provenance_path: Path = DEFAULT_PROVENANCE
) -> PhysicalParameters:
    errors: list[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    extra = sorted(set(config) - REQUIRED_TOP_LEVEL)
    if missing:
        errors.append(f"missing top-level keys: {missing}")
    if extra:
        errors.append(f"unexpected top-level keys: {extra}")
    if errors:
        raise BaselineValidationError("; ".join(errors))

    if config["schema_version"] != 1:
        errors.append("schema_version must be 1")
    if config["status"] != "frozen":
        errors.append("baseline status must be 'frozen'")
    if config["model_definition"].get("system") != "four_equation_fssi":
        errors.append("model_definition.system must be four_equation_fssi")
    if config["model_definition"].get("state_order") != [
        "V",
        "uz",
        "P",
        "sigma_z",
    ]:
        errors.append("state_order must be [V, uz, P, sigma_z]")
    if config["model_definition"].get("friction_model") != "none":
        errors.append("WP0 baseline friction_model must be none")

    physical = config["physical_parameters"]
    expected_fields = {item.name for item in fields(PhysicalParameters)}
    # ``radial_compliance_m_pa`` was added later for the isolated Cao (2021)
    # comparison branch.  It is optional and defaults to ``None``; requiring it
    # in the already frozen WP0 JSON would retroactively change the baseline
    # artifact and its provenance hash.
    optional_extension_fields = {"radial_compliance_m_pa"}
    missing_physical = sorted(
        expected_fields - optional_extension_fields - set(physical)
    )
    extra_physical = sorted(set(physical) - expected_fields)
    if missing_physical:
        errors.append(f"missing physical parameters: {missing_physical}")
    if extra_physical:
        errors.append(f"unexpected physical parameters: {extra_physical}")
    if errors:
        raise BaselineValidationError("; ".join(errors))

    for name in POSITIVE_PARAMETERS:
        value = physical[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{name} must be numeric")
        elif not math.isfinite(float(value)) or float(value) <= 0.0:
            errors.append(f"{name} must be finite and positive")
    if not 0.0 <= float(physical["pipe_nu"]) < 0.5:
        errors.append("pipe_nu must satisfy 0 <= nu < 0.5")
    if float(physical["wall_thickness_m"]) >= float(physical["inner_radius_m"]):
        errors.append("wall_thickness_m must be smaller than inner_radius_m")
    model_branch = config["model_definition"].get("coefficient_model")
    if physical["coefficient_model"] != model_branch:
        errors.append(
            "physical_parameters.coefficient_model must match "
            "model_definition.coefficient_model"
        )
    if model_branch != "printed_equations":
        errors.append("WP0 baseline coefficient_model must be printed_equations")
    if config["initial_conditions"].get("initial_stress_mode") != "equilibrium":
        errors.append("WP0 baseline initial_stress_mode must be equilibrium")

    solver = config["reference_solver"]
    if int(solver.get("n_cells", 0)) < 2:
        errors.append("reference_solver.n_cells must be at least 2")
    if not 0.0 < float(solver.get("cfl", 0.0)) <= 1.0:
        errors.append("reference_solver.cfl must satisfy 0 < CFL <= 1")
    if int(solver.get("output_points", 0)) < 2:
        errors.append("reference_solver.output_points must be at least 2")

    reproducibility = config["reproducibility"]
    if reproducibility.get("dtype") != "float64":
        errors.append("WP0 baseline dtype must be float64")
    if int(reproducibility.get("minimum_independent_training_runs", 0)) < 5:
        errors.append("formal baseline requires at least five independent runs")

    provenance_errors, _ = validate_provenance(config, provenance_path)
    errors.extend(provenance_errors)
    if errors:
        raise BaselineValidationError("; ".join(errors))
    return PhysicalParameters(**physical)


def audit_assumptions_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"path": str(path), "status": "unreadable", "errors": [str(exc)]}

    parameters = data.get("parameters_common_to_all_cases", data.get("parameters"))
    if not isinstance(parameters, dict):
        return {"path": str(path), "status": "not_applicable", "errors": []}

    claim_text = str(data.get("missing_publication_inputs", ""))
    errors: list[str] = []
    claims: dict[str, float] = {}
    for parameter, pattern in CLAIM_PATTERNS.items():
        match = pattern.search(claim_text)
        if match is None:
            continue
        claimed = float(match.group(1))
        claims[parameter] = claimed
        actual = parameters.get(parameter)
        if isinstance(actual, (int, float)) and not math.isclose(
            claimed, float(actual), rel_tol=1.0e-12, abs_tol=1.0e-15
        ):
            errors.append(
                f"text claims {parameter}={claimed:g}, metadata stores {actual:g}"
            )
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "status": "contradiction" if errors else "pass",
        "claims_checked": claims,
        "errors": errors,
    }


def audit_legacy_assumptions() -> list[dict[str, Any]]:
    paths: list[Path] = []
    for figure in ("figure09", "figure10", "figure13"):
        paths.extend(
            sorted((PROJECT_ROOT / "water16_reproduction" / figure).glob(
                "outputs*/assumptions.json"
            ))
        )
    return [audit_assumptions_file(path) for path in paths]


def build_audit_report(
    config_path: Path = DEFAULT_CONFIG,
    provenance_path: Path = DEFAULT_PROVENANCE,
    include_legacy: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    params = validate_config(config, provenance_path)
    speeds, _, _, _ = characteristic_basis(params)
    area_f, area_t = cross_section_areas(params)
    coefficients = fssi_coefficients(params)
    fluid_speed = float(np.min(np.abs(speeds)))
    legacy = audit_legacy_assumptions() if include_legacy else []
    contradictions = [item for item in legacy if item["status"] == "contradiction"]
    if contradictions:
        details = "; ".join(
            f"{item['path']}: {', '.join(item['errors'])}" for item in contradictions
        )
        raise BaselineValidationError(f"legacy metadata contradictions: {details}")

    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = "not installed"

    return {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": config["case_id"],
        "config_path": str(config_path.resolve()),
        "config_sha256": _canonical_hash(config),
        "provenance_path": str(provenance_path.resolve()),
        "parameter_source_counts": _source_counts(provenance_path),
        "derived_quantities": {
            "soil_to_pipe_modulus_ratio": soil_ratio(params),
            "initial_pressure_pa": initial_pressure_pa(params),
            "fluid_area_m2": area_f,
            "pipe_area_m2": area_t,
            "signed_characteristic_speeds_m_s": speeds.tolist(),
            "fluid_characteristic_speed_m_s": fluid_speed,
            "four_L_over_cf_s": 4.0 * params.length_m / fluid_speed,
            "valve_closure_to_one_way_fluid_transit_ratio": (
                params.valve_close_time_s / (params.length_m / fluid_speed)
            ),
            "coefficients": coefficients,
        },
        "legacy_assumptions_audit": legacy,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch_version,
        },
        "scope": (
            "configuration/provenance/metadata audit only; no PINN training or "
            "claim of experimental validation"
        ),
    }


def _source_counts(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts: dict[str, int] = {}
    for row in rows:
        source_class = row["source_class"]
        counts[source_class] = counts.get(source_class, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen WP0 FSSI research baseline"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-legacy-audit",
        action="store_true",
        help="skip existing Figure 9/10/13 assumptions.json consistency checks",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="validate and print without writing JSON"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_audit_report(
        args.config.resolve(),
        args.provenance.resolve(),
        include_legacy=not args.skip_legacy_audit,
    )
    if not args.no_write:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    summary = {
        "status": report["status"],
        "case_id": report["case_id"],
        "config_sha256": report["config_sha256"],
        "soil_to_pipe_modulus_ratio": report["derived_quantities"][
            "soil_to_pipe_modulus_ratio"
        ],
        "characteristic_speeds_m_s": report["derived_quantities"][
            "signed_characteristic_speeds_m_s"
        ],
        "legacy_files_checked": len(report["legacy_assumptions_audit"]),
        "output": None if args.no_write else str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
