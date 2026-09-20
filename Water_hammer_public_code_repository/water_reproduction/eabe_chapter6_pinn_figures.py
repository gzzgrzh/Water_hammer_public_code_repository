"""Create PINN-centred figures for Chapter 6 of the EABE manuscript.

The script performs no training and no new MOC simulation. It reads the frozen
four-state checkpoint, evaluates three previously registered boundary-audit
cases, and uses the existing 5120-cell MOC fields only as reference curves.
Historical inputs are read without modification, and all new files are written
to a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from water16_reproduction.parametric_fssi_formal_visualize import load_model
from water16_reproduction.parametric_fssi_model import predict_field


SELECTED_CASES = (
    "joint_boundary_moc_005",
    "joint_boundary_moc_014",
    "joint_boundary_moc_012",
)
REPRESENTATIVE_CASE = "joint_boundary_moc_014"
DISPLAY_TRAVEL_TIMES = 4.0
PREDICTION_BATCH_SIZE = 65536


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 400,
        }
    )


def save_bundle(figure: plt.Figure, output: Path, stem: str) -> list[str]:
    paths = {
        "pdf": output / f"{stem}.pdf",
        "svg": output / f"{stem}.svg",
        "png": output / f"{stem}.png",
        "tiff": output / f"{stem}.tiff",
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    figure.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    figure.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    figure.savefig(paths["png"], dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(paths["tiff"], dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [path.name for path in paths.values()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def critical_zone(x_over_l: np.ndarray, severity: np.ndarray) -> dict[str, float]:
    maximum = float(np.max(severity))
    mask = severity >= 0.95 * maximum
    indices = np.flatnonzero(mask)
    return {
        "critical_x_over_L": float(x_over_l[int(np.argmax(severity))]),
        "zone_95_min_x_over_L": float(x_over_l[indices[0]]),
        "zone_95_max_x_over_L": float(x_over_l[indices[-1]]),
        "zone_95_length_fraction": float(np.mean(mask)),
    }


def structural_response_figure(source_root: Path, output: Path) -> dict[str, Any]:
    source = (
        source_root
        / "PINN_FSSI_research_plan/outputs/engineering_structures_postprocess_v1"
        / "boundary_cases"
        / REPRESENTATIVE_CASE
        / "representative_structural_fields.npz"
    )
    with np.load(source, allow_pickle=False) as data:
        x = np.asarray(data["x"])
        t = np.asarray(data["t"])
        hoop_moc = np.asarray(data["moc_delta_sigma_theta_inner"]) / 1.0e6
        hoop_pinn = np.asarray(data["model_delta_sigma_theta_inner"]) / 1.0e6
        vm_moc = np.asarray(data["moc_delta_sigma_vm_control"]) / 1.0e6
        vm_pinn = np.asarray(data["model_delta_sigma_vm_control"]) / 1.0e6

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 8.0))
    for axis, values, title in (
        (axes[0, 0], hoop_pinn, "(a) Frozen-PINN hoop-stress increment"),
        (axes[0, 1], vm_pinn, "(b) Frozen-PINN von Mises increment"),
    ):
        limit = float(np.max(np.abs(values)))
        image = axis.pcolormesh(
            x / x[-1],
            t,
            values.T,
            shading="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            rasterized=True,
        )
        axis.set(xlabel=r"Normalized distance $x/L$", ylabel="Time (s)", title=title)
        colorbar = figure.colorbar(image, ax=axis, pad=0.025, aspect=28)
        colorbar.set_label("Stress increment (MPa)")

    x_over_l = x / x[-1]
    colors = {"hoop": "#2F6FA5", "vm": "#C43C39"}
    model_style = {"linestyle": "-", "linewidth": 2.2}
    moc_style = {"linestyle": "--", "linewidth": 1.5, "alpha": 0.82}
    axes[1, 0].plot(
        x_over_l,
        np.max(np.abs(hoop_pinn), axis=1),
        color=colors["hoop"],
        label="Frozen PINN, hoop",
        **model_style,
    )
    axes[1, 0].plot(
        x_over_l,
        np.max(np.abs(hoop_moc), axis=1),
        color=colors["hoop"],
        label="MOC (5120 cells), hoop",
        **moc_style,
    )
    axes[1, 0].plot(
        x_over_l,
        np.max(np.abs(vm_pinn), axis=1),
        color=colors["vm"],
        label="Frozen PINN, von Mises",
        **model_style,
    )
    axes[1, 0].plot(
        x_over_l,
        np.max(np.abs(vm_moc), axis=1),
        color=colors["vm"],
        label="MOC (5120 cells), von Mises",
        **moc_style,
    )
    axes[1, 0].set(
        xlabel=r"Normalized distance $x/L$",
        ylabel="Peak increment (MPa)",
        title="(c) Full-record spatial envelopes",
    )
    axes[1, 0].grid(alpha=0.22)

    critical_index = int(np.unravel_index(int(np.argmax(np.abs(vm_pinn))), vm_pinn.shape)[0])
    for values, color, label, style in (
        (hoop_pinn, colors["hoop"], "Frozen PINN, hoop", model_style),
        (hoop_moc, colors["hoop"], "MOC (5120 cells), hoop", moc_style),
        (vm_pinn, colors["vm"], "Frozen PINN, von Mises", model_style),
        (vm_moc, colors["vm"], "MOC (5120 cells), von Mises", moc_style),
    ):
        axes[1, 1].plot(t, values[critical_index], color=color, label=label, **style)
    axes[1, 1].set(
        xlabel="Time (s)",
        ylabel="Stress increment (MPa)",
        title=rf"(d) Histories at PINN-controlled $x/L={x_over_l[critical_index]:.3f}$",
    )
    axes[1, 1].grid(alpha=0.22)
    legend_handles, legend_labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
        handlelength=3.0,
    )
    figure.suptitle(
        "Frozen-PINN structural response and targeted MOC verification",
        fontsize=12.0,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.075, 1.0, 0.97), h_pad=2.0, w_pad=1.2)
    files = save_bundle(figure, output, "F57_PINN_structural_fields_and_verification")
    return {
        "source": str(source.relative_to(source_root)),
        "representative_case": REPRESENTATIVE_CASE,
        "pinn_controlled_x_over_L": float(x_over_l[critical_index]),
        "figure_files": files,
    }


def load_case_data(
    source_root: Path,
    model: Any,
    device: Any,
    boundary_source: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for case_id in SELECTED_CASES:
        case_dir = boundary_source / "cases" / case_id
        metadata = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        case = metadata["case"]
        with np.load(case_dir / "truth_evaluation_grid.npz", allow_pickle=False) as archive:
            x = np.asarray(archive["x"])
            time = np.asarray(archive["t"])
            reference = {
                "P": np.asarray(archive["P"]),
                "sigma_z": np.asarray(archive["sigma_z"]),
            }
        prediction = predict_field(model, case, x, time, device, PREDICTION_BATCH_SIZE)
        x_over_l = x / float(x[-1])
        p0 = float(metadata["initial_pressure_pa"])
        sz0 = float(metadata["initial_stress_pa"])
        label = (
            rf"$t_c/(L/c_f)={float(case['closure_time_over_L_cf']):.2f}$, "
            rf"$V_0={float(case['initial_velocity_m_s']):.4f}$ m/s"
        )
        item: dict[str, Any] = {
            "case_id": case_id,
            "case": case,
            "label": label,
            "x_over_l": x_over_l,
            "time_s": time,
            "fluid_wave_speed_m_s": float(metadata["fluid_wave_speed_m_s"]),
        }
        for method, fields in (("pinn", prediction), ("moc", reference)):
            pressure_increment = fields["P"] - p0
            stress_increment = fields["sigma_z"] - sz0
            item[f"{method}_pressure_increment_kpa"] = pressure_increment * 1.0e-3
            item[f"{method}_stress_increment_mpa"] = stress_increment * 1.0e-6
            item[f"{method}_pressure_rise_kpa"] = np.max(pressure_increment, axis=1) * 1.0e-3
            item[f"{method}_pressure_drop_kpa"] = -np.min(pressure_increment, axis=1) * 1.0e-3
            item[f"{method}_stress_envelope_mpa"] = np.max(np.abs(stress_increment), axis=1) * 1.0e-6

            for metric, values in (
                ("maximum_pressure_rise", item[f"{method}_pressure_rise_kpa"]),
                ("maximum_pressure_drop", item[f"{method}_pressure_drop_kpa"]),
                ("maximum_absolute_axial_stress", item[f"{method}_stress_envelope_mpa"]),
            ):
                summary_rows.append(
                    {
                        "case_id": case_id,
                        "method": "Frozen PINN" if method == "pinn" else "5120-cell MOC",
                        "soil_to_pipe_modulus_ratio": float(case["soil_to_pipe_modulus_ratio"]),
                        "closure_time_over_L_cf": float(case["closure_time_over_L_cf"]),
                        "initial_velocity_m_s": float(case["initial_velocity_m_s"]),
                        "metric": metric,
                        "global_value": float(np.max(values)),
                        **critical_zone(x_over_l, values),
                    }
                )
        cases.append(item)
    return cases, summary_rows


def aggregate_zones(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for metric in (
        "maximum_pressure_rise",
        "maximum_pressure_drop",
        "maximum_absolute_axial_stress",
    ):
        output[metric] = {}
        for method in ("Frozen PINN", "5120-cell MOC"):
            selected = [row for row in summary_rows if row["metric"] == metric and row["method"] == method]
            output[metric][method] = {
                "minimum_zone_start_x_over_L": float(min(row["zone_95_min_x_over_L"] for row in selected)),
                "maximum_zone_start_x_over_L": float(max(row["zone_95_min_x_over_L"] for row in selected)),
                "minimum_zone_end_x_over_L": float(min(row["zone_95_max_x_over_L"] for row in selected)),
                "maximum_zone_end_x_over_L": float(max(row["zone_95_max_x_over_L"] for row in selected)),
            }
    return output


def monitoring_figure(
    case_data: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    zones = aggregate_zones(summary_rows)
    colors = ("#2F6FA5", "#D17A32", "#4C9A62")
    figure = plt.figure(figsize=(7.2, 7.8))
    grid = figure.add_gridspec(
        2,
        6,
        height_ratios=(1.10, 0.92),
        left=0.08,
        right=0.96,
        bottom=0.13,
        top=0.90,
        hspace=0.38,
        wspace=1.34,
    )
    pressure_axis = figure.add_subplot(grid[0, 0:3])
    stress_axis = figure.add_subplot(grid[0, 3:6])
    envelope_axes = (
        figure.add_subplot(grid[1, 0:2]),
        figure.add_subplot(grid[1, 2:4]),
        figure.add_subplot(grid[1, 4:6]),
    )

    representative = next(item for item in case_data if item["case_id"] == REPRESENTATIVE_CASE)
    travel_time = 100.0 / representative["fluid_wave_speed_m_s"]
    normalized_time = representative["time_s"] / travel_time
    visible = normalized_time <= DISPLAY_TRAVEL_TIMES + 1.0e-12
    x_grid, t_grid = np.meshgrid(representative["x_over_l"], normalized_time[visible])
    for axis, values, title, label, cmap in (
        (
            pressure_axis,
            representative["pinn_pressure_increment_kpa"][:, visible].T,
            "(a) Frozen-PINN pressure-wave field",
            r"Pressure change $\Delta P$ (kPa)",
            "RdBu_r",
        ),
        (
            stress_axis,
            representative["pinn_stress_increment_mpa"][:, visible].T,
            "(b) Frozen-PINN axial-stress-wave field",
            r"Stress change $\Delta\sigma_z$ (MPa)",
            "PuOr_r",
        ),
    ):
        limit = float(np.max(np.abs(values)))
        mesh = axis.pcolormesh(
            x_grid,
            t_grid,
            values,
            shading="auto",
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            rasterized=True,
        )
        axis.set(
            xlabel=r"Normalized distance $x/L$",
            ylabel=(r"Normalized time $t/(L/c_f)$" if axis is pressure_axis else ""),
            title=title,
            xlim=(0.0, 1.0),
            ylim=(0.0, DISPLAY_TRAVEL_TIMES),
        )
        colorbar = figure.colorbar(mesh, ax=axis, pad=0.025, aspect=25, shrink=0.94)
        colorbar.set_label(label)

    definitions = (
        (
            "pressure_rise_kpa",
            "maximum_pressure_rise",
            "(c) Pressure-rise envelope",
            "Peak pressure rise (kPa)",
            "#F2C14E",
        ),
        (
            "pressure_drop_kpa",
            "maximum_pressure_drop",
            "(d) Pressure-drop envelope",
            "Peak pressure drop (kPa)",
            "#F2C14E",
        ),
        (
            "stress_envelope_mpa",
            "maximum_absolute_axial_stress",
            "(e) Axial-stress envelope",
            "Peak axial-stress change (MPa)",
            "#8E72B8",
        ),
    )
    for axis, (key, metric, title, ylabel, zone_color) in zip(envelope_axes, definitions):
        pinn_zone = zones[metric]["Frozen PINN"]
        zone_start = pinn_zone["minimum_zone_start_x_over_L"]
        zone_end = pinn_zone["maximum_zone_end_x_over_L"]
        axis.axvspan(zone_start, zone_end, color=zone_color, alpha=0.20, zorder=0)
        for color, item in zip(colors, case_data):
            axis.plot(
                item["x_over_l"],
                item[f"pinn_{key}"],
                color=color,
                linewidth=2.2,
                zorder=3,
            )
            axis.plot(
                item["x_over_l"],
                item[f"moc_{key}"],
                color=color,
                linestyle="--",
                linewidth=1.25,
                alpha=0.82,
                zorder=2,
            )
        axis.set(
            xlabel=r"Normalized distance $x/L$",
            ylabel=ylabel,
            title=title,
            xlim=(0.0, 1.0),
        )
        axis.grid(alpha=0.22)

    case_handles = [
        Line2D([0], [0], color=color, lw=2.2, label=item["label"])
        for color, item in zip(colors, case_data)
    ]
    method_handles = [
        Line2D([0], [0], color="#333333", lw=2.2, linestyle="-", label="Frozen PINN"),
        Line2D([0], [0], color="#333333", lw=1.4, linestyle="--", label="5120-cell MOC"),
        Line2D([0], [0], color="#B08A20", lw=7.0, alpha=0.25, label="PINN 95% priority-zone union"),
    ]
    figure.legend(
        handles=case_handles + method_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.01),
        ncol=3,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.5,
    )
    figure.suptitle(
        "PINN-derived monitoring zones verified by MOC",
        fontsize=12.0,
        fontweight="bold",
        y=0.995,
    )
    files = save_bundle(figure, output, "F58_PINN_MOC_monitoring_zones")
    return {"aggregate_zones": zones, "figure_files": files}


def run(source_root: Path, output: Path, device_name: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    configure_style()

    model_config = source_root / "PINN_FSSI_research_plan/configs/parametric_fssi_hybrid_formal_v3_balanced_events.json"
    checkpoint = source_root / "PINN_FSSI_research_plan/outputs/parametric_fssi_hybrid_formal_v3_balanced_events/formal/checkpoint.pt"
    boundary_source = source_root / "PINN_FSSI_research_plan/outputs/revised_joint_boundary_moc_v1"

    structural = structural_response_figure(source_root, output)
    _, _, model, device = load_model(model_config, checkpoint, device_name)
    case_data, summary_rows = load_case_data(source_root, model, device, boundary_source)
    monitoring = monitoring_figure(case_data, summary_rows, output)
    write_csv(output / "T22_PINN_MOC_monitoring_zones.csv", summary_rows)

    report = {
        "status": "pass",
        "training_performed": False,
        "new_moc_simulation_performed": False,
        "device": str(device),
        "model_config": str(model_config.relative_to(source_root)),
        "model_checkpoint": str(checkpoint.relative_to(source_root)),
        "selected_case_ids": list(SELECTED_CASES),
        "representative_case_id": REPRESENTATIVE_CASE,
        "critical_zone_definition": "spatial points with envelope at least 95% of the case maximum",
        "structural_response_figure": structural,
        "monitoring_figure": monitoring,
        "monitoring_zone_table": "T22_PINN_MOC_monitoring_zones.csv",
    }
    (output / "chapter6_pinn_figure_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.source_root, args.output_dir, args.device)
