"""Formal three-model comparison for realistic sparse P + sigma monitoring.

The registered experiment holds the 36 training cases and 8 x 48 monitoring
locations fixed, and compares a data-only ANN, a conventional coordinate PINN,
and the characteristic physics-data model under clean and 3% noisy labels.
Training is resumable per model/seed.  The test split is evaluated only after
all validation runs exist.  In addition to error statistics, the script writes
engineering prediction plots for histories, spatial profiles, peaks, first-peak
times, and critical locations.
"""

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
import torch

from water16_reproduction.common.physics import cross_section_areas, initial_pressure_pa
from water16_reproduction.parametric_fssi_ann import ParametricCoordinateANN, train as train_ann
from water16_reproduction.parametric_fssi_conventional_pinn import (
    ParametricFullDomainPINN,
    evaluate as evaluate_coordinate,
    predict_field as predict_coordinate,
    train as train_coordinate,
)
from water16_reproduction.parametric_fssi_forward import materialize_cases
from water16_reproduction.parametric_fssi_hybrid import load_config as load_hybrid_config
from water16_reproduction.parametric_fssi_hybrid import run_training as train_characteristic
from water16_reproduction.parametric_fssi_model import (
    ThreeParameterBoundaryModel,
    evaluate_reference_cases,
    load_config as load_base_model_config,
    load_design,
    predict_field as predict_characteristic,
)
from water16_reproduction.parametric_fssi_reference import case_parameters, load_baseline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/revised_sparse_monitoring_three_model_v1"
ABLATION_CONFIG_ROOT = ROOT / "PINN_FSSI_research_plan/outputs/revised_ablation_matrix_v1/registered_configs"
PILOT_ROOT = ROOT / "PINN_FSSI_research_plan/outputs/revised_sparse_monitoring_pilot_v1"

SOURCE_CONFIGS = {
    "ann": ABLATION_CONFIG_ROOT / "ann__labels_384__cases_36__sampling_event.json",
    "coordinate_pinn": ABLATION_CONFIG_ROOT / "coordinate_pinn__labels_384__cases_36__sampling_event.json",
    "characteristic_pinn": ABLATION_CONFIG_ROOT / "characteristic__labels_384__cases_36__sampling_event.json",
}
MODELS = ("ann", "coordinate_pinn", "characteristic_pinn")
MODEL_LABELS = {
    "ann": "ANN",
    "coordinate_pinn": "Coordinate PINN",
    "characteristic_pinn": "Characteristic PINN (proposed)",
}
MODEL_COLORS = {
    "reference": "#151515",
    "ann": "#D9822B",
    "coordinate_pinn": "#4C78A8",
    "characteristic_pinn": "#2A9D6F",
}
CONDITIONS = {
    "clean": 0.0,
    "noise03": 0.03,
}
CONDITION_LABELS = {"clean": "Clean labels", "noise03": "3% noisy labels"}
SEEDS = (29041, 29053, 29059)
PRIMARY_SEED = 29041
METRICS = (
    "V_nrmse",
    "uz_nrmse",
    "P_nrmse",
    "sigma_z_nrmse",
    "pressure_peak_relative_error",
    "stress_peak_relative_error",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stable_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != payload:
            raise RuntimeError(f"registered configuration changed: {path}")
        return
    write_json(path, payload)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_anchor_config(condition: str, seed: int, path: Path) -> dict[str, Any]:
    config = deepcopy(read_json(SOURCE_CONFIGS["characteristic_pinn"]))
    noise = CONDITIONS[condition]
    config["model_id"] = f"sparse_monitoring_anchor_{condition}_seed{seed}_v1"
    config["status"] = "registered_before_formal_three_model_comparison"
    config["registration_reason"] = (
        "Fixed realistic monitoring: 36 cases, 8 spatial gauges, 48 event-aligned "
        "times, and pressure plus axial-stress supervision only."
    )
    config["observation_policy"] = {
        "observed_states": ["P", "sigma_z"],
        "noise_std_fraction_of_dynamic_scale": noise,
        "noise_seed": seed,
        "noise_distribution": "independent Gaussian on observed components only",
        "unobserved_states_forbidden_from_anchor_loss": True,
    }
    config["training"]["formal"]["seed"] = seed
    vectors = int(config["anchor_policy"]["anchor_vectors_per_case"])
    cases = int(config["training"]["formal"]["case_limit"])
    config["anchor_policy"]["states"] = ["P", "sigma_z"]
    config["anchor_policy"]["total_scalar_labels"] = vectors * cases * 2
    config["formal_comparison_registration"] = {
        "condition": condition,
        "seed": seed,
        "config_snapshot": relative(path),
        "validation_before_test": True,
        "test_labels_forbidden": True,
    }
    return config


def make_model_config(
    model_name: str, condition: str, seed: int, anchor_path: Path, path: Path
) -> dict[str, Any]:
    if model_name == "characteristic_pinn":
        config = deepcopy(read_json(anchor_path))
    else:
        config = deepcopy(read_json(SOURCE_CONFIGS[model_name]))
        config["anchor_config"] = relative(anchor_path)
        config["observation_policy"] = deepcopy(read_json(anchor_path)["observation_policy"])
        config["training"]["formal"]["seed"] = seed
    config["model_id"] = f"sparse_monitoring_{model_name}_{condition}_seed{seed}_v1"
    config["status"] = "registered_before_formal_three_model_comparison"
    config["evaluation"]["splits"] = ["validation"]
    config["formal_comparison_registration"] = {
        "model": model_name,
        "condition": condition,
        "seed": seed,
        "monitoring_vectors_per_case": 384,
        "observed_scalar_labels_per_case": 768,
        "training_cases": 36,
        "config_snapshot": relative(path),
        "test_split_read_during_training": False,
    }
    return config


def register_configs(output: Path) -> dict[tuple[str, int, str], Path]:
    registered: dict[tuple[str, int, str], Path] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            anchor_path = output / "registered_configs" / f"anchor__{condition}__seed{seed}.json"
            stable_write_json(anchor_path, make_anchor_config(condition, seed, anchor_path))
            for model_name in MODELS:
                path = output / "registered_configs" / f"{model_name}__{condition}__seed{seed}.json"
                stable_write_json(
                    path,
                    make_model_config(model_name, condition, seed, anchor_path, path),
                )
                registered[(condition, seed, model_name)] = path
    return registered


def pilot_characteristic_run(condition: str, seed: int) -> Path | None:
    if seed != PRIMARY_SEED:
        return None
    slug = "pressure_stress_clean" if condition == "clean" else "pressure_stress_noise03"
    path = PILOT_ROOT / "runs" / slug / "formal"
    return path if (path / "model_report.json").exists() else None


def run_dir(output: Path, condition: str, seed: int, model_name: str) -> Path:
    reused = pilot_characteristic_run(condition, seed) if model_name == "characteristic_pinn" else None
    if reused is not None:
        return reused
    return output / "runs" / condition / f"seed_{seed}" / model_name / "formal"


def train_all(
    output: Path, registered: dict[tuple[str, int, str], Path]
) -> None:
    completed: list[str] = []
    total = len(CONDITIONS) * len(SEEDS) * len(MODELS)
    for condition in CONDITIONS:
        for seed in SEEDS:
            for model_name in MODELS:
                task_id = f"{condition}/seed_{seed}/{model_name}"
                reused = pilot_characteristic_run(condition, seed) if model_name == "characteristic_pinn" else None
                if reused is not None:
                    print(f"[formal] reuse pilot checkpoint: {task_id}", flush=True)
                else:
                    root = output / "runs" / condition / f"seed_{seed}" / model_name
                    config_path = registered[(condition, seed, model_name)]
                    print(f"[formal] train {len(completed)+1}/{total}: {task_id}", flush=True)
                    if model_name == "ann":
                        train_ann(config_path, root, "formal")
                    elif model_name == "coordinate_pinn":
                        train_coordinate(config_path, root, "formal")
                    else:
                        train_characteristic(load_hybrid_config(config_path), "formal", root)
                completed.append(task_id)
                write_json(
                    output / "progress.json",
                    {
                        "status": "training_validation",
                        "completed": completed,
                        "completed_count": len(completed),
                        "total_count": total,
                        "next_task": None if len(completed) == total else "registered next matrix cell",
                    },
                )
                print(f"[formal] checkpointed {task_id}", flush=True)


def load_predictor(
    model_name: str, config_path: Path, model_path: Path, device: torch.device
) -> tuple[Any, dict[str, Any], dict[str, Any], Callable[..., dict[str, np.ndarray]]]:
    config = read_json(config_path)
    if model_name == "characteristic_pinn":
        base = load_base_model_config(ROOT / config["base_model_config"])
        design = load_design(base)
        baseline = load_baseline(design)
        model = ThreeParameterBoundaryModel(design, base, baseline).to(device)
        payload = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        predictor = predict_characteristic
    else:
        design = load_design(config)
        baseline = load_baseline(design)
        cls = ParametricCoordinateANN if model_name == "ann" else ParametricFullDomainPINN
        model = cls(design, baseline, config).to(device)
        payload = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(payload)
        predictor = predict_coordinate
    model.eval()
    return model, config, design, predictor


def test_evaluation(
    output: Path, registered: dict[tuple[str, int, str], Path], device: torch.device
) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    total = len(CONDITIONS) * len(SEEDS) * len(MODELS)
    done = 0
    for condition in CONDITIONS:
        for seed in SEEDS:
            for model_name in MODELS:
                release = output / "test_release" / condition / f"seed_{seed}" / model_name
                report_path = release / "test_report.json"
                metrics_path = release / "test_case_metrics.csv"
                if report_path.exists() and metrics_path.exists():
                    summary = read_json(report_path)["test"]
                else:
                    model, config, design, _ = load_predictor(
                        model_name,
                        registered[(condition, seed, model_name)],
                        run_dir(output, condition, seed, model_name) / "model.pt",
                        device,
                    )
                    eval_config = deepcopy(config)
                    eval_config["evaluation"]["splits"] = ["test"]
                    if model_name == "characteristic_pinn":
                        base = load_base_model_config(ROOT / config["base_model_config"])
                        base["reference_root"] = config["reference_root"]
                        base["evaluation"] = eval_config["evaluation"]
                        raw_summary, rows = evaluate_reference_cases(model, design, base, device)
                        summary = raw_summary
                    else:
                        raw_summary, rows = evaluate_coordinate(model, eval_config, device)
                        summary = raw_summary["test"]
                    release.mkdir(parents=True, exist_ok=True)
                    write_csv(metrics_path, rows)
                    write_json(
                        report_path,
                        {
                            "status": "complete",
                            "release_rule": "all 18 validation runs/checkpoints existed before first test access",
                            "condition": condition,
                            "seed": seed,
                            "model": model_name,
                            "test": summary,
                        },
                    )
                row: dict[str, Any] = {
                    "condition": condition,
                    "seed": seed,
                    "model": model_name,
                    "test_case_count": int(summary["case_count"]),
                }
                for metric in METRICS:
                    row[f"{metric}_mean"] = float(summary["mean_by_metric"][metric])
                    row[f"{metric}_maximum"] = float(summary["maximum_by_metric"][metric])
                aggregate.append(row)
                done += 1
                write_json(
                    output / "progress.json",
                    {
                        "status": "sealed_test_release",
                        "completed_test_evaluations": done,
                        "total_test_evaluations": total,
                    },
                )
    write_csv(output / "T16_three_model_sparse_monitoring_test_metrics.csv", aggregate)
    return aggregate


def figure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 13.5,
            "axes.titlesize": 14,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 10.5,
            "lines.linewidth": 2.0,
        }
    )


def save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(
            output / f"{stem}.{suffix}",
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def read_case_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_peak_parity(output: Path) -> None:
    figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 11.0), constrained_layout=True)
    quantities = (
        ("reference_pressure_peak_increment_pa", "prediction_pressure_peak_increment_pa", "Maximum pressure increment (MPa)"),
        ("reference_stress_peak_increment_pa", "prediction_stress_peak_increment_pa", "Maximum |axial stress increment| (MPa)"),
    )
    for row_index, condition in enumerate(CONDITIONS):
        for column_index, (reference_key, prediction_key, label) in enumerate(quantities):
            axis = axes[row_index, column_index]
            all_values: list[float] = []
            for model_name in MODELS:
                by_seed: list[dict[str, float]] = []
                case_order: list[str] = []
                for seed in SEEDS:
                    rows = read_case_rows(
                        output / "test_release" / condition / f"seed_{seed}" / model_name / "test_case_metrics.csv"
                    )
                    if not case_order:
                        case_order = [row["case_id"] for row in rows]
                    by_seed.append({row["case_id"]: float(row[prediction_key]) / 1e6 for row in rows})
                    references = {row["case_id"]: float(row[reference_key]) / 1e6 for row in rows}
                actual = np.asarray([references[case] for case in case_order])
                predicted = np.asarray([[values[case] for case in case_order] for values in by_seed])
                mean = predicted.mean(axis=0)
                std = predicted.std(axis=0)
                axis.errorbar(
                    actual,
                    mean,
                    yerr=std,
                    fmt="o",
                    ms=5.5,
                    capsize=2.5,
                    color=MODEL_COLORS[model_name],
                    alpha=0.85,
                    label=MODEL_LABELS[model_name],
                )
                all_values.extend(actual.tolist() + mean.tolist())
            low, high = min(all_values), max(all_values)
            margin = 0.05 * max(high - low, abs(high), 1.0)
            axis.plot([low - margin, high + margin], [low - margin, high + margin], "--", color="#555555", lw=1.4)
            axis.set_xlim(low - margin, high + margin)
            axis.set_ylim(low - margin, high + margin)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.25)
            axis.set_xlabel(f"MOC reference: {label}")
            axis.set_ylabel(f"Predicted: {label}")
            axis.set_title(f"{CONDITION_LABELS[condition]} — {label}", fontweight="bold")
    axes[0, 0].legend(loc="best")
    save_figure(fig, output, "F47_engineering_peak_prediction_parity")


def representative_cases(design: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [case for case in materialize_cases(design, "formal") if case["split"] == "test"]
    preferred = ("formal_test_001", "formal_test_boundary_002", "formal_test_combination_002")
    by_id = {case["case_id"]: case for case in cases}
    if all(case_id in by_id for case_id in preferred):
        return [by_id[case_id] for case_id in preferred]
    selected = []
    for test_class in ("interpolation", "boundary", "combination"):
        selected.append(sorted((case for case in cases if case["test_class"] == test_class), key=lambda c: c["case_id"])[0])
    return selected


def primary_models(
    output: Path,
    registered: dict[tuple[str, int, str], Path],
    condition: str,
    device: torch.device,
) -> tuple[dict[str, tuple[Any, Callable[..., dict[str, np.ndarray]]]], dict[str, Any]]:
    loaded = {}
    design: dict[str, Any] | None = None
    for model_name in MODELS:
        model, _, model_design, predictor = load_predictor(
            model_name,
            registered[(condition, PRIMARY_SEED, model_name)],
            run_dir(output, condition, PRIMARY_SEED, model_name) / "model.pt",
            device,
        )
        loaded[model_name] = (model, predictor)
        design = model_design
    assert design is not None
    return loaded, design


def load_reference(config: dict[str, Any], case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    path = ROOT / config["reference_root"] / case["split"] / case["case_id"] / "truth_evaluation_grid.npz"
    with np.load(path, allow_pickle=False) as archive:
        x, t = archive["x"], archive["t"]
        fields = {state: archive[state] for state in ("V", "uz", "P", "sigma_z")}
    return x, t, fields


def plot_histories(
    output: Path,
    registered: dict[tuple[str, int, str], Path],
    condition: str,
    device: torch.device,
) -> None:
    figure_style()
    models, design = primary_models(output, registered, condition, device)
    cases = representative_cases(design)
    reference_config = read_json(registered[(condition, PRIMARY_SEED, "ann")])
    baseline = load_baseline(design)
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 9.0), constrained_layout=True)
    for column, case in enumerate(cases):
        x, t, reference = load_reference(reference_config, case)
        params, _ = case_parameters(baseline, case, float(design["truth_solver"]["formal"]["t_final_s"]))
        p0 = initial_pressure_pa(params)
        area_f, area_t = cross_section_areas(params)
        s0 = area_f * p0 / area_t
        stress_delta = np.abs(reference["sigma_z"] - s0)
        stress_location = int(np.unravel_index(int(np.argmax(stress_delta)), stress_delta.shape)[0])
        axes[0, column].plot(t, (reference["P"][-1] - p0) / 1e6, color=MODEL_COLORS["reference"], label="MOC reference")
        axes[1, column].plot(t, (reference["sigma_z"][stress_location] - s0) / 1e6, color=MODEL_COLORS["reference"], label="MOC reference")
        for model_name, (model, predictor) in models.items():
            prediction = predictor(model, case, x, t, device, 8192)
            axes[0, column].plot(t, (prediction["P"][-1] - p0) / 1e6, color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name], alpha=0.9)
            axes[1, column].plot(t, (prediction["sigma_z"][stress_location] - s0) / 1e6, color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name], alpha=0.9)
        title = f"{case['test_class'].title()}: {case['case_id']}"
        axes[0, column].set_title(title, fontweight="bold")
        axes[0, column].set_ylabel("Valve pressure increment (MPa)")
        axes[1, column].set_ylabel(f"Axial stress increment (MPa)\nat x/L={x[stress_location]/params.length_m:.3f}")
        axes[1, column].set_xlabel("Time (s)")
        for axis in axes[:, column]:
            axis.grid(alpha=0.22)
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"Predicted response histories — {CONDITION_LABELS[condition]}", fontsize=16, fontweight="bold")
    number = "F48" if condition == "clean" else "F49"
    save_figure(fig, output, f"{number}_predicted_histories_{condition}")


def plot_spatial_profiles(
    output: Path,
    registered: dict[tuple[str, int, str], Path],
    device: torch.device,
) -> None:
    figure_style()
    condition = "clean"
    models, design = primary_models(output, registered, condition, device)
    case = representative_cases(design)[-1]
    config = read_json(registered[(condition, PRIMARY_SEED, "ann")])
    x, t, reference = load_reference(config, case)
    baseline = load_baseline(design)
    params, fluid_speed = case_parameters(baseline, case, float(design["truth_solver"]["formal"]["t_final_s"]))
    p0 = initial_pressure_pa(params)
    area_f, area_t = cross_section_areas(params)
    s0 = area_f * p0 / area_t
    travel = params.length_m / fluid_speed
    target_times = [params.valve_close_time_s, params.valve_close_time_s + travel, params.valve_close_time_s + 2 * travel]
    indices = [int(np.argmin(np.abs(t - min(value, t[-1])))) for value in target_times]
    predictions = {name: predictor(model, case, x, t, device, 8192) for name, (model, predictor) in models.items()}
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 8.8), constrained_layout=True)
    for column, index in enumerate(indices):
        axes[0, column].plot(x / params.length_m, (reference["P"][:, index] - p0) / 1e6, color=MODEL_COLORS["reference"], label="MOC reference")
        axes[1, column].plot(x / params.length_m, (reference["sigma_z"][:, index] - s0) / 1e6, color=MODEL_COLORS["reference"], label="MOC reference")
        for model_name in MODELS:
            axes[0, column].plot(x / params.length_m, (predictions[model_name]["P"][:, index] - p0) / 1e6, color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name])
            axes[1, column].plot(x / params.length_m, (predictions[model_name]["sigma_z"][:, index] - s0) / 1e6, color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name])
        axes[0, column].set_title(f"t={t[index]:.3f} s", fontweight="bold")
        axes[0, column].set_ylabel("Pressure increment (MPa)")
        axes[1, column].set_ylabel("Axial stress increment (MPa)")
        axes[1, column].set_xlabel("Normalized pipe coordinate x/L")
        for axis in axes[:, column]:
            axis.grid(alpha=0.22)
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"Full-field spatial profiles — {case['case_id']} (clean sparse monitoring)", fontsize=16, fontweight="bold")
    save_figure(fig, output, "F50_full_field_spatial_profiles_clean")


def plot_peak_time_location(output: Path) -> None:
    figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.5), constrained_layout=True)
    definitions = (
        ("reference_pressure_first_peak_time_s", "prediction_pressure_first_peak_time_s", "Pressure first-peak time (s)"),
        ("reference_stress_first_peak_time_s", "prediction_stress_first_peak_time_s", "Stress first-peak time (s)"),
        ("reference_pressure_critical_location_over_L", "prediction_pressure_critical_location_over_L", "Pressure critical location x/L"),
        ("reference_stress_critical_location_over_L", "prediction_stress_critical_location_over_L", "Stress critical location x/L"),
    )
    condition = "clean"
    for axis, (reference_key, prediction_key, label) in zip(axes.ravel(), definitions):
        values: list[float] = []
        for model_name in MODELS:
            rows = read_case_rows(output / "test_release" / condition / f"seed_{PRIMARY_SEED}" / model_name / "test_case_metrics.csv")
            actual = np.asarray([float(row[reference_key]) for row in rows])
            predicted = np.asarray([float(row[prediction_key]) for row in rows])
            axis.scatter(actual, predicted, s=38, color=MODEL_COLORS[model_name], alpha=0.82, label=MODEL_LABELS[model_name])
            values.extend(actual.tolist() + predicted.tolist())
        low, high = min(values), max(values)
        margin = 0.05 * max(high - low, 0.05)
        axis.plot([low - margin, high + margin], [low - margin, high + margin], "--", color="#555555", lw=1.4)
        axis.set_xlim(low - margin, high + margin)
        axis.set_ylim(low - margin, high + margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"MOC reference: {label}")
        axis.set_ylabel(f"Predicted: {label}")
        axis.set_title(label, fontweight="bold")
        axis.grid(alpha=0.22)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Prediction of response timing and critical locations (clean labels)", fontsize=16, fontweight="bold")
    save_figure(fig, output, "F51_peak_time_and_critical_location_prediction")


def plot_sorted_engineering_outputs(output: Path) -> None:
    figure_style()
    fig, axes = plt.subplots(2, 1, figsize=(17.0, 9.8), constrained_layout=True)
    condition = "clean"
    base_rows = read_case_rows(output / "test_release" / condition / f"seed_{PRIMARY_SEED}" / "characteristic_pinn" / "test_case_metrics.csv")
    definitions = (
        ("reference_pressure_peak_increment_pa", "prediction_pressure_peak_increment_pa", "Maximum pressure increment (MPa)"),
        ("reference_stress_peak_increment_pa", "prediction_stress_peak_increment_pa", "Maximum |axial stress increment| (MPa)"),
    )
    for axis, (reference_key, prediction_key, label) in zip(axes, definitions):
        ordered = sorted(base_rows, key=lambda row: float(row[reference_key]))
        case_ids = [row["case_id"] for row in ordered]
        reference = np.asarray([float(row[reference_key]) / 1e6 for row in ordered])
        axis.plot(np.arange(len(case_ids)), reference, "o-", color=MODEL_COLORS["reference"], label="MOC reference", markersize=4)
        for model_name in MODELS:
            rows = read_case_rows(output / "test_release" / condition / f"seed_{PRIMARY_SEED}" / model_name / "test_case_metrics.csv")
            by_id = {row["case_id"]: row for row in rows}
            prediction = np.asarray([float(by_id[case_id][prediction_key]) / 1e6 for case_id in case_ids])
            axis.plot(np.arange(len(case_ids)), prediction, "o-", color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name], markersize=3.5, alpha=0.9)
        axis.set_ylabel(label)
        axis.set_xticks(np.arange(len(case_ids)), [case_id.replace("formal_test_", "") for case_id in case_ids], rotation=55, ha="right")
        axis.grid(alpha=0.22)
        axis.set_xlabel("Held-out test case (sorted by MOC reference response)")
    axes[0].legend(ncol=4, loc="best")
    fig.suptitle("Case-by-case engineering response predictions (clean sparse monitoring)", fontsize=16, fontweight="bold")
    save_figure(fig, output, "F52_casewise_engineering_output_comparison")


def summarize_across_seeds(rows: list[dict[str, Any]], output: Path) -> None:
    summary_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for model_name in MODELS:
            subset = [row for row in rows if row["condition"] == condition and row["model"] == model_name]
            summary: dict[str, Any] = {"condition": condition, "model": model_name, "seed_count": len(subset)}
            for metric in METRICS:
                values = np.asarray([float(row[f"{metric}_mean"]) for row in subset])
                summary[f"{metric}_mean_across_seeds"] = float(values.mean())
                summary[f"{metric}_std_across_seeds"] = float(values.std())
            summary_rows.append(summary)
    write_csv(output / "T17_three_model_seed_summary.csv", summary_rows)


def generate_figures(
    output: Path,
    registered: dict[tuple[str, int, str], Path],
    device: torch.device,
) -> None:
    plot_peak_parity(output)
    plot_histories(output, registered, "clean", device)
    plot_histories(output, registered, "noise03", device)
    plot_spatial_profiles(output, registered, device)
    plot_peak_time_location(output)
    plot_sorted_engineering_outputs(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    final_report = output / "three_model_sparse_monitoring_report.json"
    if final_report.exists():
        print(final_report.read_text(encoding="utf-8"), flush=True)
        return

    registered = register_configs(output)
    train_all(output, registered)
    # Test release begins only after every matrix cell has a completed model report.
    missing = [
        str(run_dir(output, condition, seed, model_name) / "model_report.json")
        for condition in CONDITIONS
        for seed in SEEDS
        for model_name in MODELS
        if not (run_dir(output, condition, seed, model_name) / "model_report.json").exists()
    ]
    if missing:
        raise RuntimeError(f"test release blocked by {len(missing)} missing validation reports")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = test_evaluation(output, registered, device)
    summarize_across_seeds(rows, output)
    generate_figures(output, registered, device)
    payload = {
        "status": "pass",
        "scope": "formal ANN vs coordinate PINN vs characteristic PINN comparison under realistic sparse P+sigma monitoring",
        "training_cases": 36,
        "monitoring_vectors_per_case": 384,
        "spatial_gauges_per_case": 8,
        "time_samples_per_gauge": 48,
        "observed_states": ["P", "sigma_z"],
        "observed_scalar_labels_per_case": 768,
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "sealed_test_cases": 24,
        "test_release_after_validation_matrix_complete": True,
        "tables": [
            "T16_three_model_sparse_monitoring_test_metrics.csv",
            "T17_three_model_seed_summary.csv",
        ],
        "engineering_figures": [
            "F47_engineering_peak_prediction_parity",
            "F48_predicted_histories_clean",
            "F49_predicted_histories_noise03",
            "F50_full_field_spatial_profiles_clean",
            "F51_peak_time_and_critical_location_prediction",
            "F52_casewise_engineering_output_comparison",
        ],
    }
    write_json(final_report, payload)
    write_json(output / "progress.json", {"status": "complete", **payload})
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
