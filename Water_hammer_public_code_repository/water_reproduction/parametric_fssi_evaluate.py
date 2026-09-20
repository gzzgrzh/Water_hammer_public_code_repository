"""Evaluate a frozen parametric FSSI checkpoint on registered case splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from water16_reproduction.parametric_fssi_hybrid import (
    DEFAULT_CONFIG as DEFAULT_HYBRID_CONFIG,
    load_config as load_hybrid_config,
)
from water16_reproduction.parametric_fssi_model import (
    ThreeParameterBoundaryModel,
    evaluate_reference_cases,
    load_config as load_base_model_config,
    load_design,
)
from water16_reproduction.parametric_fssi_reference import load_baseline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "PINN_FSSI_research_plan"
    / "outputs"
    / "parametric_fssi_hybrid_v1"
    / "pilot"
    / "checkpoint.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "PINN_FSSI_research_plan"
    / "outputs"
    / "parametric_fssi_hybrid_v1"
    / "diagnostic"
)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def evaluate_checkpoint(
    hybrid_config_path: Path,
    checkpoint_path: Path,
    splits: list[str],
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output_dir}")
    hybrid = load_hybrid_config(hybrid_config_path)
    base = load_base_model_config(ROOT / hybrid["base_model_config"])
    design = load_design(base)
    baseline = load_baseline(design)
    torch.set_default_dtype(torch.float64)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = ThreeParameterBoundaryModel(design, base, baseline).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["state_dict"])
    evaluation_config = dict(base)
    evaluation_config["reference_root"] = hybrid["reference_root"]
    evaluation_config["evaluation"] = {
        "stage": hybrid.get("data_stage", hybrid["evaluation"].get("stage", "pilot")),
        "splits": splits,
        "batch_size": int(hybrid["evaluation"]["batch_size"]),
    }
    summary, rows = evaluate_reference_cases(
        model, design, evaluation_config, device
    )
    output_dir.mkdir(parents=True)
    with (output_dir / "case_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "pass",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "splits": splits,
        "device": str(device),
        "evaluation": summary,
        "interpretation": (
            "training-fit diagnostic only; not an independent validation result"
            if splits == ["train"]
            else "registered split evaluation"
        ),
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-config", type=Path, default=DEFAULT_HYBRID_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), default=["train"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_checkpoint(
        args.hybrid_config,
        args.checkpoint,
        list(args.splits),
        args.output_dir,
        args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
