"""Release a frozen formal FSSI checkpoint from validation to one-time testing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from water16_reproduction.parametric_fssi_evaluate import evaluate_checkpoint
from water16_reproduction.parametric_fssi_hybrid import load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "PINN_FSSI_research_plan" / "configs" / "parametric_fssi_hybrid_formal_v1.json"
)
DEFAULT_CHECKPOINT = (
    ROOT / "PINN_FSSI_research_plan" / "outputs" / "parametric_fssi_hybrid_formal_v1"
    / "formal" / "checkpoint.pt"
)
DEFAULT_ROOT = (
    ROOT / "PINN_FSSI_research_plan" / "outputs" / "parametric_fssi_hybrid_formal_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acceptance(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    maximum = summary["maximum_by_metric"]
    limits = config["acceptance"]
    checks = {
        "pressure_field": maximum["P_nrmse"] <= float(limits["formal_pressure_nrmse"]),
        "stress_field": maximum["sigma_z_nrmse"] <= float(limits["formal_axial_stress_nrmse"]),
        "pressure_peak": maximum["pressure_peak_relative_error"] <= float(limits["formal_pressure_peak_relative_error"]),
        "stress_peak": maximum["stress_peak_relative_error"] <= float(limits["formal_stress_peak_relative_error"]),
    }
    return {"status": "pass" if all(checks.values()) else "failed", "checks": checks}


def validation_release(
    config_path: Path,
    checkpoint: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    output = output_root / "validation_release"
    report = evaluate_checkpoint(config_path, checkpoint, ["validation"], output, device)
    config = load_config(config_path)
    decision = acceptance(report["evaluation"], config)
    released = {
        **report,
        "status": decision["status"],
        "acceptance_checks": decision["checks"],
        "checkpoint_sha256": sha256(checkpoint),
        "test_labels_read": False,
        "release_decision": (
            "checkpoint frozen for one-time test evaluation"
            if decision["status"] == "pass"
            else "checkpoint rejected; test split remains sealed"
        ),
    }
    (output / "release_report.json").write_text(
        json.dumps(released, indent=2), encoding="utf-8"
    )
    return released


def test_release(
    config_path: Path,
    checkpoint: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    validation_report_path = output_root / "validation_release" / "release_report.json"
    if not validation_report_path.exists():
        raise RuntimeError("validation release report is missing; test split remains sealed")
    validation = json.loads(validation_report_path.read_text(encoding="utf-8"))
    if validation["status"] != "pass":
        raise RuntimeError("validation acceptance failed; test split remains sealed")
    current_hash = sha256(checkpoint)
    if validation["checkpoint_sha256"] != current_hash:
        raise RuntimeError("checkpoint changed after validation; test split remains sealed")
    output = output_root / "test_release"
    report = evaluate_checkpoint(config_path, checkpoint, ["test"], output, device)
    config = load_config(config_path)
    decision = acceptance(report["evaluation"], config)
    released = {
        **report,
        "status": decision["status"],
        "acceptance_checks": decision["checks"],
        "checkpoint_sha256": current_hash,
        "validation_release_report": str(validation_report_path.resolve()),
        "test_labels_read": True,
        "test_evaluation_policy": "single evaluation after checkpoint freeze",
    }
    (output / "release_report.json").write_text(
        json.dumps(released, indent=2), encoding="utf-8"
    )
    return released


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = (
        validation_release(args.config, args.checkpoint, args.output_root, args.device)
        if args.stage == "validation"
        else test_release(args.config, args.checkpoint, args.output_root, args.device)
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
