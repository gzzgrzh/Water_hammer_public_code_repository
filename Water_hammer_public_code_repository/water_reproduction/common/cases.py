"""Canonical case names and shared case-output locations."""

from pathlib import Path

from .physics import PhysicalParameters, soil_ratio


def case_identifier(params: PhysicalParameters, system_kind: str = "fssi") -> str:
    if system_kind == "two_equation":
        return "two_equation"
    ratio = f"{soil_ratio(params):.0e}".replace("+", "").replace("-0", "-")
    nu = f"{params.pipe_nu:.2f}".replace(".", "p")
    mass = f"{params.valve_mass_kg:g}".replace(".", "p")
    return f"fssi_EsE_{ratio}_Mv_{mass}_nu_{nu}"


def trained_case_dir(params: PhysicalParameters, system_kind: str = "fssi") -> Path:
    project_dir = Path(__file__).resolve().parents[1]
    return project_dir / "trained_cases" / case_identifier(params, system_kind)
