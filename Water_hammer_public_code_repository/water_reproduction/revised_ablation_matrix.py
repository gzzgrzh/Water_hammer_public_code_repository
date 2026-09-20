"""Run resumable non-oracle label, train-case, and sampling ablations."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from water16_reproduction.parametric_fssi_ann import train as train_ann
from water16_reproduction.parametric_fssi_conventional_pinn import train as train_coordinate
from water16_reproduction.parametric_fssi_hybrid import (
    load_config as load_hybrid_config,
    run_training as train_characteristic,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "PINN_FSSI_research_plan/configs"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/revised_ablation_matrix_v1"
ANN_BASE = CONFIG_ROOT / "parametric_fssi_ann_baseline_v1.json"
COORDINATE_BASE = CONFIG_ROOT / "parametric_fssi_coordinate_pinn_physics_events_v1.json"
CHARACTERISTIC_BASE = CONFIG_ROOT / "parametric_fssi_hybrid_formal_v3_balanced_events.json"
PHYSICS_WARM_START = "PINN_FSSI_research_plan/outputs/parametric_fssi_model_v1/pilot/checkpoint.pt"
MODEL_ORDER = ("ANN", "Coordinate PINN", "Characteristic model")
COLORS = {"ANN": "#888888", "Coordinate PINN": "#D9853B", "Characteristic model": "#2F6B9A"}
DISPLAY_LABELS = {
    "ANN": "ANN",
    "Coordinate PINN": "Standard PINN",
    "Characteristic model": "Proposed model",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"registered config changed after creation: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def anchor_config(
    labels: int,
    cases: int,
    sampling: str,
    config_path: Path,
) -> dict[str, Any]:
    config = deepcopy(read_json(CHARACTERISTIC_BASE))
    times = labels // 8
    if labels not in (96, 192, 384) or 8 * times != labels:
        raise ValueError("registered label counts are 96, 192, and 384")
    policy = config["anchor_policy"]
    policy.update(
        {
            "case_selection": "nested_maximin_parameter_space_v1",
            "spatial_points_per_case": 8,
            "time_points_per_case": times,
            "anchor_vectors_per_case": labels,
            "total_anchor_vectors": labels * cases,
            "total_scalar_labels": 4 * labels * cases,
        }
    )
    for name in (
        "physics_time_points_per_case",
        "response_time_points_per_case",
        "candidate_physics_time_points_per_case",
    ):
        policy.pop(name, None)
    if sampling == "event":
        policy["selection"] = "balanced_dual_characteristic_event_aligned_nested_v1"
        policy["candidate_time_points_per_case"] = 48
        policy["time_positions"] = "nested subset of 48 dual-family physics-event times; no field values used"
    elif sampling == "uniform":
        policy["selection"] = "deterministic_uniform_indices_in_saved_evaluation_grid"
        policy.pop("candidate_time_points_per_case", None)
        policy["time_positions"] = "deterministic uniform indices over the saved time grid"
    elif sampling == "oracle":
        if labels != 384:
            raise ValueError("oracle sampling is registered only for the 384-vector comparison")
        policy["selection"] = "train_response_peak_enriched_nested_ablation_v4"
        policy["candidate_time_points_per_case"] = 48
        policy["candidate_physics_time_points_per_case"] = 32
        policy["time_positions"] = "32 physics-event plus 16 train-truth peak/gradient times; upper bound only"
    else:
        raise ValueError(f"unsupported sampling design: {sampling}")
    config.update(
        {
            "model_id": f"revised_characteristic_{labels}_labels_{cases}_cases_{sampling}",
            "status": "registered_revised_ablation_before_training",
            "registration_reason": "Nested revised-paper ablation with fixed model, optimizer, epoch count, and validation split.",
            "warm_start_checkpoint": PHYSICS_WARM_START,
        }
    )
    formal = config["training"]["formal"]
    formal.update(
        {
            "epochs": 600,
            "case_limit": cases,
            "learning_rate": 0.00008,
            "log_every": 50,
            "checkpoint_every": 50,
            "seed": 29041,
        }
    )
    config["evaluation"] = {"stage": "formal", "splits": ["validation"], "batch_size": 8192}
    config["ablation_registration"] = {
        "labels_per_case": labels,
        "training_cases": cases,
        "sampling": sampling,
        "config_snapshot": relative(config_path),
        "test_split_read": False,
    }
    return config


def model_config(
    model: str,
    labels: int,
    cases: int,
    sampling: str,
    anchor_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    if model == "ANN":
        config = deepcopy(read_json(ANN_BASE))
        config["model_id"] = f"revised_ann_{labels}_labels_{cases}_cases_{sampling}"
    elif model == "Coordinate PINN":
        config = deepcopy(read_json(COORDINATE_BASE))
        config["model_id"] = f"revised_coordinate_pinn_{labels}_labels_{cases}_cases_{sampling}"
    elif model == "Characteristic model":
        config = anchor_config(labels, cases, sampling, config_path)
        return config
    else:
        raise ValueError(model)
    config["status"] = "registered_revised_ablation_before_training"
    config["anchor_config"] = relative(anchor_path)
    formal = config["training"]["formal"]
    formal.update(
        {
            "epochs": 600,
            "case_limit": cases,
            "log_every": 50,
            "checkpoint_every": 50,
            "seed": 29041,
        }
    )
    config["evaluation"] = {
        "stage": "formal",
        "splits": ["validation"],
        "batch_size": 8192,
        "model_selection_uses_validation_only": True,
    }
    config["ablation_registration"] = {
        "labels_per_case": labels,
        "training_cases": cases,
        "sampling": sampling,
        "config_snapshot": relative(config_path),
        "test_split_read": False,
    }
    return config


def run_one(
    model: str,
    labels: int,
    cases: int,
    sampling: str,
    output_root: Path,
) -> dict[str, Any]:
    key = f"labels_{labels}__cases_{cases}__sampling_{sampling}"
    config_root = output_root / "registered_configs"
    anchor_path = config_root / f"anchor__{key}.json"
    anchor = anchor_config(labels, cases, sampling, anchor_path)
    stable_write_json(anchor_path, anchor)
    slug = {"ANN": "ann", "Coordinate PINN": "coordinate_pinn", "Characteristic model": "characteristic"}[model]
    config_path = config_root / f"{slug}__{key}.json"
    config = model_config(model, labels, cases, sampling, anchor_path, config_path)
    stable_write_json(config_path, config)
    run_root = output_root / "runs" / key / slug
    if model == "ANN":
        report = train_ann(config_path, run_root, "formal")
    elif model == "Coordinate PINN":
        report = train_coordinate(config_path, run_root, "formal")
    else:
        report = train_characteristic(load_hybrid_config(config_path), "formal", run_root)
    return {
        "key": key,
        "model": model,
        "labels_per_case": labels,
        "training_cases": cases,
        "sampling": sampling,
        "report_path": relative(run_root / "formal/model_report.json"),
        "status": report["status"],
        "validation": report["evaluation"].get("validation", report["evaluation"]),
        "training_seconds": report.get(
            "training_seconds_this_invocation", report.get("training_seconds", 0.0)
        ),
    }


def matrix_design(stage: str) -> list[tuple[str, int, int, str]]:
    designs: list[tuple[str, int, int, str]] = []
    if stage in ("labels", "all"):
        designs.extend(
            (model, labels, 72, "event")
            for labels in (96, 192, 384)
            for model in MODEL_ORDER
        )
    if stage in ("cases", "all"):
        designs.extend(
            (model, 384, cases, "event")
            for cases in (18, 36, 72)
            for model in MODEL_ORDER
        )
    if stage in ("sampling", "all"):
        designs.extend(
            ("Characteristic model", 384, 72, sampling)
            for sampling in ("uniform", "event", "oracle")
        )
    unique = []
    seen = set()
    for item in designs:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for row in rows:
        mean = row["validation"]["mean_by_metric"]
        maximum = row["validation"]["maximum_by_metric"]
        flat.append(
            {
                **{name: row[name] for name in ("key", "model", "labels_per_case", "training_cases", "sampling", "status", "training_seconds")},
                **{f"mean_{name}": value for name, value in mean.items()},
                **{f"maximum_{name}": value for name, value in maximum.items()},
            }
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def plot_curve(
    rows: list[dict[str, Any]],
    x_name: str,
    x_values: tuple[int, ...],
    output: Path,
    stem: str,
) -> None:
    metrics = (
        ("P_nrmse", "Pressure field NRMSE (%)"),
        ("sigma_z_nrmse", "Stress field NRMSE (%)"),
        ("pressure_peak_relative_error", "Maximum-pressure error (%)"),
        ("stress_peak_relative_error", "Maximum-stress error (%)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    for axis, (metric, ylabel) in zip(axes.ravel(), metrics):
        for model in MODEL_ORDER:
            subset = [
                row for row in rows
                if row["model"] == model
                and row["sampling"] == "event"
                and int(row[x_name]) in x_values
                and int(row["training_cases" if x_name == "labels_per_case" else "labels_per_case"])
                == (72 if x_name == "labels_per_case" else 384)
            ]
            subset.sort(key=lambda row: int(row[x_name]))
            if len(subset) != len(x_values):
                continue
            x = [int(row[x_name]) for row in subset]
            y = [100.0 * row["validation"]["mean_by_metric"][metric] for row in subset]
            ymax = [100.0 * row["validation"]["maximum_by_metric"][metric] for row in subset]
            axis.plot(x, y, "o-", color=COLORS[model], label=DISPLAY_LABELS[model], lw=1.7)
            axis.plot(x, ymax, "--", color=COLORS[model], alpha=0.45, lw=1.0)
        axis.set_xlabel("Samples per training case" if x_name == "labels_per_case" else "Training cases")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=11)
    figure.suptitle("Mean and maximum validation errors", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output / f"{stem}.png", dpi=320, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_sampling(rows: list[dict[str, Any]], output: Path) -> None:
    subset = [
        row for row in rows
        if row["model"] == "Characteristic model"
        and row["labels_per_case"] == 384
        and row["training_cases"] == 72
    ]
    order = ("uniform", "event", "oracle")
    metrics = (
        ("P_nrmse", "Pressure NRMSE (%)"),
        ("sigma_z_nrmse", "Stress NRMSE (%)"),
        ("pressure_peak_relative_error", "Pressure-peak error (%)"),
        ("stress_peak_relative_error", "Stress-peak error (%)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.8, 7.0))
    for axis, (metric, ylabel) in zip(axes.ravel(), metrics):
        values = []
        maxima = []
        for sampling in order:
            row = next(item for item in subset if item["sampling"] == sampling)
            values.append(100.0 * row["validation"]["mean_by_metric"][metric])
            maxima.append(100.0 * row["validation"]["maximum_by_metric"][metric])
        x = np.arange(3)
        axis.bar(x, values, color=("#999999", "#2F6B9A", "#6AAE75"), alpha=0.85)
        axis.scatter(x, maxima, marker="_", s=190, linewidth=2, color="#222222")
        axis.set_xticks(x, ("Uniform", "Wave-event based", "Full-field peak based"), rotation=10)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Effect of time-sample selection", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output / "F18_sampling_strategy_ablation.png", dpi=320, bbox_inches="tight")
    figure.savefig(output / "F18_sampling_strategy_ablation.pdf", bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    completed: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        completed = {row["key"] + "__" + row["model"]: row for row in read_json(progress_path)["completed"]}
    for model, labels, cases, sampling in matrix_design(args.stage):
        key = f"labels_{labels}__cases_{cases}__sampling_{sampling}__{model}"
        if key in completed:
            continue
        row = run_one(model, labels, cases, sampling, args.output_dir)
        completed[key] = row
        progress_path.write_text(
            json.dumps({"stage": args.stage, "completed": list(completed.values())}, indent=2),
            encoding="utf-8",
        )
        print(f"[matrix] checkpointed {key}", flush=True)
    rows = list(completed.values())
    write_rows(args.output_dir / "T07_revised_ablation_matrix.csv", rows)
    expected = matrix_design(args.stage)
    if args.stage in ("labels", "all"):
        plot_curve(rows, "labels_per_case", (96, 192, 384), args.output_dir, "F16_non_oracle_label_efficiency")
    if args.stage in ("cases", "all"):
        plot_curve(rows, "training_cases", (18, 36, 72), args.output_dir, "F17_training_case_efficiency")
    if args.stage in ("sampling", "all"):
        plot_sampling(rows, args.output_dir)
    report = {
        "status": "pass",
        "stage": args.stage,
        "registered_run_count": len(expected),
        "completed_run_count": len(rows),
        "epochs_per_run": 600,
        "seed": 29041,
        "test_split_read": False,
        "label_design": "nested non-oracle dual-family physics-event sampling",
        "case_design": "nested deterministic maximin subset of the 72 registered training cases",
        "progress_file": relative(progress_path),
        "figures": sorted(path.name for path in args.output_dir.glob("F*.png")),
    }
    (args.output_dir / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("labels", "cases", "sampling", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
