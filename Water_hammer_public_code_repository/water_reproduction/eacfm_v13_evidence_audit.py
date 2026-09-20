"""Post-process frozen water-hammer models for the EACFM manuscript.

This script performs no training.  It reads the archived sparse-monitoring
checkpoint, registered test metrics, MOC timing metadata, and deployment timing
record.  It then produces:

1. a four-equation interior residual audit on held-out parameter cases;
2. a compact first-peak timing comparison copied from the sealed test summary;
3. an offline-inclusive cost and break-even audit; and
4. an editable two-panel supplementary figure.

All outputs are written to a new directory.  Existing directories are never
overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SPARSE_CONFIG_REL = Path(
    "PINN_FSSI_research_plan/outputs/revised_sparse_monitoring_three_model_v1/"
    "registered_configs/characteristic_pinn__clean__seed29041.json"
)
SPARSE_CHECKPOINT_REL = Path(
    "PINN_FSSI_research_plan/outputs/revised_sparse_monitoring_pilot_v1/"
    "runs/pressure_stress_clean/formal/model.pt"
)
PHASE_SUMMARY_REL = Path(
    "PINN_FSSI_research_plan/outputs/revised_three_model_comparison_v1/"
    "T06_three_model_summary.csv"
)
DEPLOYMENT_MODEL_REPORT_REL = Path(
    "PINN_FSSI_research_plan/outputs/"
    "parametric_fssi_hybrid_formal_v3_balanced_events/formal/model_report.json"
)
TIMING_REPORT_REL = Path(
    "PINN_FSSI_research_plan/outputs/engineering_structures_timing_v1/"
    "engineering_structures_timing_report.json"
)
REFERENCE_ROOT_REL = Path(
    "PINN_FSSI_research_plan/outputs/parametric_fssi_reference_v1/formal"
)

EQUATION_LABELS = (
    "fluid continuity",
    "pipe axial equilibrium",
    "coupled pressure equation",
    "coupled axial-velocity equation",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows available for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require_assets(asset_root: Path) -> None:
    required = (
        SPARSE_CONFIG_REL,
        SPARSE_CHECKPOINT_REL,
        PHASE_SUMMARY_REL,
        DEPLOYMENT_MODEL_REPORT_REL,
        TIMING_REPORT_REL,
        REFERENCE_ROOT_REL,
    )
    missing = [str(path) for path in required if not (asset_root / path).exists()]
    if missing:
        raise FileNotFoundError("missing archived assets:\n" + "\n".join(missing))


def load_sparse_model(asset_root: Path, device: torch.device):
    if str(asset_root) not in sys.path:
        sys.path.insert(0, str(asset_root))
    torch.set_default_dtype(torch.float64)

    from water16_reproduction.parametric_fssi_model import (
        ThreeParameterBoundaryModel,
        load_config as load_base_model_config,
        load_design,
    )
    from water16_reproduction.parametric_fssi_reference import load_baseline

    config = read_json(asset_root / SPARSE_CONFIG_REL)
    base = load_base_model_config(asset_root / config["base_model_config"])
    design = load_design(base)
    baseline = load_baseline(design)
    model = ThreeParameterBoundaryModel(design, base, baseline).to(device)
    try:
        payload = torch.load(
            asset_root / SPARSE_CHECKPOINT_REL,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(asset_root / SPARSE_CHECKPOINT_REL, map_location=device)
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return model, config, base, design, baseline


def residual_audit(
    asset_root: Path,
    device: torch.device,
    output_dir: Path,
    spatial_points: int,
    time_points: int,
    max_cases: int | None,
) -> dict[str, Any]:
    from water16_reproduction.common.physics import state_scales, system_matrices
    from water16_reproduction.parametric_fssi_forward import materialize_cases
    from water16_reproduction.parametric_fssi_model import characteristic_state
    from water16_reproduction.parametric_fssi_reference import case_parameters

    model, config, _, design, baseline = load_sparse_model(asset_root, device)
    cases = [
        case
        for case in materialize_cases(design, "formal")
        if case["split"] == "test"
    ]
    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        raise RuntimeError("no held-out test cases were selected")

    all_values: list[np.ndarray] = []
    case_rows: list[dict[str, Any]] = []
    torch.set_default_dtype(torch.float64)

    for case_index, case in enumerate(cases, start=1):
        params, _ = case_parameters(baseline, case, model.t_final_s)
        x = np.linspace(0.015, 0.985, spatial_points) * params.length_m
        t = np.linspace(0.001, model.t_final_s - 0.001, time_points)
        xx, tt = np.meshgrid(x, t, indexing="ij")
        positions = torch.as_tensor(
            xx.reshape(-1, 1), device=device, dtype=torch.float64
        ).requires_grad_(True)
        times = torch.as_tensor(
            tt.reshape(-1, 1), device=device, dtype=torch.float64
        ).requires_grad_(True)

        state = characteristic_state(model, positions, times, case)
        q_x: list[torch.Tensor] = []
        q_t: list[torch.Tensor] = []
        for state_index in range(4):
            component = state[:, state_index : state_index + 1]
            ones = torch.ones_like(component)
            q_x.append(
                torch.autograd.grad(
                    component, positions, grad_outputs=ones, retain_graph=True
                )[0]
            )
            q_t.append(
                torch.autograd.grad(
                    component, times, grad_outputs=ones, retain_graph=True
                )[0]
            )

        matrix_a_np, matrix_b_np = system_matrices(params)
        matrix_a = torch.as_tensor(matrix_a_np, device=device, dtype=torch.float64)
        matrix_b = torch.as_tensor(matrix_b_np, device=device, dtype=torch.float64)
        residual = torch.cat(q_t, dim=1) @ matrix_a.T
        residual = residual + torch.cat(q_x, dim=1) @ matrix_b.T
        scales = torch.as_tensor(
            state_scales(params), device=device, dtype=torch.float64
        )
        row_scale = (
            torch.abs(matrix_a) @ scales / params.t_final_s
            + torch.abs(matrix_b) @ scales / params.length_m
        )
        normalized = (residual / row_scale).detach().cpu().numpy()
        absolute = np.abs(normalized)
        all_values.append(absolute)
        row: dict[str, Any] = {
            "case_id": case["case_id"],
            "test_class": case["test_class"],
            "sample_count": int(absolute.shape[0]),
        }
        for equation_index in range(4):
            values = absolute[:, equation_index]
            prefix = f"equation_{equation_index + 1}"
            row[f"{prefix}_rms"] = float(np.sqrt(np.mean(values**2)))
            row[f"{prefix}_p95_abs"] = float(np.quantile(values, 0.95))
            row[f"{prefix}_max_abs"] = float(np.max(values))
        case_rows.append(row)
        print(
            f"residual audit {case_index}/{len(cases)}: {case['case_id']}",
            flush=True,
        )

    stacked = np.vstack(all_values)
    summary_rows: list[dict[str, Any]] = []
    for equation_index, label in enumerate(EQUATION_LABELS):
        values = stacked[:, equation_index]
        summary_rows.append(
            {
                "equation": equation_index + 1,
                "label": label,
                "mean_abs_normalized_residual": float(np.mean(values)),
                "rms_normalized_residual": float(np.sqrt(np.mean(values**2))),
                "p95_abs_normalized_residual": float(np.quantile(values, 0.95)),
                "maximum_abs_normalized_residual": float(np.max(values)),
            }
        )

    write_csv(output_dir / "residual_case_metrics.csv", case_rows)
    write_csv(output_dir / "residual_summary.csv", summary_rows)
    return {
        "model": "sparse pressure-stress characteristic checkpoint",
        "checkpoint": str(SPARSE_CHECKPOINT_REL),
        "config": str(SPARSE_CONFIG_REL),
        "test_case_count": len(cases),
        "spatial_points_per_case": spatial_points,
        "time_points_per_case": time_points,
        "interior_points_per_case": spatial_points * time_points,
        "total_interior_points": int(stacked.shape[0]),
        "equations": summary_rows,
        "interpretation": (
            "Piecewise-smooth automatic-differentiation audit of A q_t + B q_x. "
            "Points exclude the two physical boundaries; residuals are normalized "
            "by equation-wise characteristic state and domain scales."
        ),
    }


def phase_summary(asset_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(asset_root / PHASE_SUMMARY_REL)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] != "test":
            continue
        selected.append(
            {
                "model": row["model"],
                "case_count": int(row["case_count"]),
                "pressure_first_peak_time_error_ms_mean": float(
                    row["pressure_first_peak_time_abs_error_ms_mean"]
                ),
                "pressure_first_peak_time_error_ms_median": float(
                    row["pressure_first_peak_time_abs_error_ms_median"]
                ),
                "pressure_first_peak_time_error_ms_maximum": float(
                    row["pressure_first_peak_time_abs_error_ms_maximum"]
                ),
                "stress_first_peak_time_error_ms_mean": float(
                    row["stress_first_peak_time_abs_error_ms_mean"]
                ),
                "stress_first_peak_time_error_ms_median": float(
                    row["stress_first_peak_time_abs_error_ms_median"]
                ),
                "stress_first_peak_time_error_ms_maximum": float(
                    row["stress_first_peak_time_abs_error_ms_maximum"]
                ),
            }
        )
    if len(selected) != 3:
        raise RuntimeError("expected three registered models in the sealed test summary")
    write_csv(output_dir / "first_peak_timing_summary.csv", selected)
    return selected


def cost_audit(asset_root: Path, output_dir: Path, scan_cases: int) -> dict[str, Any]:
    model_report = read_json(asset_root / DEPLOYMENT_MODEL_REPORT_REL)
    timing_report = read_json(asset_root / TIMING_REPORT_REL)
    metadata_paths = sorted(
        (asset_root / REFERENCE_ROOT_REL / "train").glob("*/metadata.json")
    )
    if len(metadata_paths) != 72:
        raise RuntimeError(f"expected 72 training-case metadata files, found {len(metadata_paths)}")
    generation_times = np.asarray(
        [float(read_json(path)["generation_runtime_s"]) for path in metadata_paths]
    )
    training_seconds = float(model_report["training_seconds"])
    inference_seconds = float(timing_report["inference_runtime_s"]["median"])
    moc_seconds = float(timing_report["moc_runtime_s"]["median"])
    data_generation_seconds = float(np.sum(generation_times))
    offline_seconds = data_generation_seconds + training_seconds
    marginal_saving = moc_seconds - inference_seconds
    if marginal_saving <= 0.0:
        raise RuntimeError("surrogate inference is not faster than the recorded MOC run")
    break_even_training_only = int(math.ceil(training_seconds / marginal_saving))
    break_even_all_offline = int(math.ceil(offline_seconds / marginal_saving))
    direct_scan_seconds = scan_cases * moc_seconds
    surrogate_scan_seconds = offline_seconds + scan_cases * inference_seconds

    report = {
        "training_case_count": len(metadata_paths),
        "training_data_generation_s": data_generation_seconds,
        "training_data_generation_median_case_s": float(np.median(generation_times)),
        "training_data_generation_minimum_case_s": float(np.min(generation_times)),
        "training_data_generation_maximum_case_s": float(np.max(generation_times)),
        "model_training_s": training_seconds,
        "offline_total_s": offline_seconds,
        "inference_median_s": inference_seconds,
        "moc_median_s": moc_seconds,
        "online_speedup": moc_seconds / inference_seconds,
        "break_even_queries_training_only": break_even_training_only,
        "break_even_queries_all_offline": break_even_all_offline,
        "screening_case_count": scan_cases,
        "direct_moc_screening_s": direct_scan_seconds,
        "surrogate_screening_with_offline_s": surrogate_scan_seconds,
        "screening_total_speedup_with_offline": direct_scan_seconds
        / surrogate_scan_seconds,
        "scope": (
            "Sequential runtime accounting using recorded 5120-cell MOC generation, "
            "the deployment checkpoint training time, and median frozen-model inference."
        ),
    }
    write_json(output_dir / "cost_audit.json", report)
    write_csv(
        output_dir / "cost_audit.csv",
        [{"metric": key, "value": value} for key, value in report.items()],
    )
    return report


def configure_figure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )


def plot_timing_and_cost(
    phase_rows: list[dict[str, Any]],
    cost: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    configure_figure_style()
    model_order = ("ANN", "Coordinate PINN", "Characteristic model")
    by_model = {row["model"]: row for row in phase_rows}
    if tuple(model for model in model_order if model in by_model) != model_order:
        raise RuntimeError("registered model labels do not match the plotting contract")

    colors = {
        "ANN": "#D9822B",
        "Coordinate PINN": "#4C78A8",
        "Characteristic model": "#2A9D6F",
    }
    fig, (ax_timing, ax_cost) = plt.subplots(1, 2, figsize=(7.1, 3.15))

    categories = ("Pressure", "Axial stress")
    x = np.arange(len(categories), dtype=float)
    width = 0.23
    for model_index, model in enumerate(model_order):
        row = by_model[model]
        means = np.asarray(
            [
                row["pressure_first_peak_time_error_ms_mean"],
                row["stress_first_peak_time_error_ms_mean"],
            ]
        )
        maxima = np.asarray(
            [
                row["pressure_first_peak_time_error_ms_maximum"],
                row["stress_first_peak_time_error_ms_maximum"],
            ]
        )
        if np.any(means <= 0) or np.any(maxima <= 0):
            raise ValueError("log-scale timing values must be strictly positive")
        offsets = x + (model_index - 1) * width
        ax_timing.bar(
            offsets,
            means,
            width=width,
            color=colors[model],
            edgecolor="none",
            linewidth=0,
            label=model,
            zorder=2,
        )
        ax_timing.scatter(
            offsets,
            maxima,
            s=13,
            marker="o",
            facecolor="white",
            edgecolor="#202020",
            linewidth=0.7,
            zorder=3,
        )
    ax_timing.set_yscale("log")
    ax_timing.set_xticks(x)
    ax_timing.set_xticklabels(categories)
    ax_timing.set_ylabel("First-peak time error (ms)")
    ax_timing.text(
        0.50,
        1.15,
        "Wave-timing accuracy on 24 test cases",
        transform=ax_timing.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    for legend_x, model in zip((0.09, 0.46, 0.85), model_order):
        ax_timing.text(
            legend_x,
            1.025,
            model,
            transform=ax_timing.transAxes,
            ha="center",
            va="bottom",
            color=colors[model],
            fontweight="bold",
            fontsize=7.6,
        )

    scan_cases = int(cost["screening_case_count"])
    queries = np.geomspace(1.0, float(scan_cases), 300)
    moc_hours = queries * float(cost["moc_median_s"]) / 3600.0
    model_hours = (
        float(cost["offline_total_s"])
        + queries * float(cost["inference_median_s"])
    ) / 3600.0
    ax_cost.plot(
        queries,
        moc_hours,
        color="#4D4D4D",
        linewidth=2.0,
        label="Direct 5120-cell MOC",
    )
    ax_cost.plot(
        queries,
        model_hours,
        color="#2A9D6F",
        linewidth=2.2,
        label="Characteristic model\n(offline stage included)",
    )
    break_even = int(cost["break_even_queries_all_offline"])
    break_even_hours = break_even * float(cost["moc_median_s"]) / 3600.0
    ax_cost.scatter(
        [break_even],
        [break_even_hours],
        s=26,
        color="#B64342",
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    ax_cost.annotate(
        f"Break-even: {break_even} cases",
        xy=(break_even, break_even_hours),
        xytext=(9, -18),
        textcoords="offset points",
        fontsize=7.2,
        color="#B64342",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
    )
    model_scan_hours = float(cost["surrogate_screening_with_offline_s"]) / 3600.0
    ax_cost.scatter(
        [scan_cases],
        [model_scan_hours],
        s=22,
        color="#2A9D6F",
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    ax_cost.annotate(
        f"{scan_cases:,}: {model_scan_hours:.2f} h",
        xy=(scan_cases, model_scan_hours),
        xytext=(-4, 12),
        textcoords="offset points",
        ha="right",
        fontsize=7.2,
        color="#247A59",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
    )
    ax_cost.set_xscale("log")
    ax_cost.set_yscale("log")
    ax_cost.set_xlabel("Number of queried parameter cases")
    ax_cost.set_ylabel("Cumulative runtime (h)")
    ax_cost.text(
        0.50,
        1.15,
        "Cost amortization with the offline stage",
        transform=ax_cost.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    ax_cost.text(
        0.03,
        1.06,
        "Direct 5120-cell MOC",
        transform=ax_cost.transAxes,
        ha="left",
        va="bottom",
        color="#4D4D4D",
        fontweight="bold",
        fontsize=7.6,
    )
    ax_cost.text(
        0.03,
        1.005,
        "Characteristic model + offline stage",
        transform=ax_cost.transAxes,
        ha="left",
        va="bottom",
        color="#2A9D6F",
        fontweight="bold",
        fontsize=7.6,
    )

    for label, axis in zip(("a", "b"), (ax_timing, ax_cost)):
        axis.text(
            -0.13,
            1.13,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.86, wspace=0.33)
    stem = output_dir / "F_eacfm_timing_cost_audit"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    qa = {
        "core_conclusion": (
            "The characteristic representation reduces first-peak timing errors "
            "and amortizes its complete offline cost in repeated-query deployment."
        ),
        "archetype": "quantitative two-panel comparison",
        "backend": "Python/matplotlib",
        "final_size_inches": [7.1, 3.15],
        "panel_a": {
            "role": "accuracy comparison",
            "sample": "24 sealed test cases",
            "center": "arithmetic mean absolute first-peak time error",
            "secondary_marker": "maximum absolute first-peak time error",
        },
        "panel_b": {
            "role": "offline-inclusive cost boundary",
            "timing_basis": "recorded medians; sequential execution",
        },
        "source_data": [
            "first_peak_timing_summary.csv",
            "cost_audit.csv",
        ],
    }
    write_json(output_dir / "figure_qa_notes.json", qa)
    return qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--spatial-points", type=int, default=33)
    parser.add_argument("--time-points", type=int, default=65)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--screening-cases", type=int, default=13440)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.resolve()
    output_dir = args.output_dir.resolve()
    require_assets(asset_root)
    if str(asset_root) not in sys.path:
        sys.path.insert(0, str(asset_root))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)

    if args.spatial_points < 5 or args.time_points < 5:
        raise ValueError("residual audit requires at least five points per axis")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    residual = residual_audit(
        asset_root,
        device,
        output_dir,
        int(args.spatial_points),
        int(args.time_points),
        args.max_cases,
    )
    phase = phase_summary(asset_root, output_dir)
    cost = cost_audit(asset_root, output_dir, int(args.screening_cases))
    figure = plot_timing_and_cost(phase, cost, output_dir)
    report = {
        "status": "complete",
        "training_performed": False,
        "device": str(device),
        "residual_audit": residual,
        "first_peak_timing": phase,
        "cost_audit": cost,
        "figure_contract": figure,
    }
    write_json(output_dir / "eacfm_v13_evidence_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
