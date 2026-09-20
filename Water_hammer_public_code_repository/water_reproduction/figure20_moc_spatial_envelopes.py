"""Create the MOC propagation-and-envelope composite for manuscript Figure 20.

The script performs no training and no new MOC simulation.  It reads three
completed 5120-cell MOC boundary-audit cases.  One intermediate-closure case
is used to explain wave propagation in time and space, while all three cases
are used to identify high-response pipe sections from full-record envelopes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "PINN_FSSI_research_plan/outputs/revised_joint_boundary_moc_v1"
DEFAULT_OUTPUT = ROOT / "PINN_FSSI_research_plan/outputs/figure20_moc_propagation_envelopes_v6"
DESIGN = ROOT / "PINN_FSSI_research_plan/configs/parametric_fssi_forward_v1.json"
SELECTED_CASES = (
    "joint_boundary_moc_005",
    "joint_boundary_moc_014",
    "joint_boundary_moc_012",
)
REPRESENTATIVE_CASE = "joint_boundary_moc_014"
DISPLAY_TRAVEL_TIMES = 4.0


def critical_zone(x_over_l: np.ndarray, severity: np.ndarray) -> dict[str, float]:
    maximum = float(np.max(severity))
    mask = severity >= 0.95 * maximum
    indices = np.flatnonzero(mask)
    critical_index = int(np.argmax(severity))
    return {
        "critical_x_over_L": float(x_over_l[critical_index]),
        "zone_95_min_x_over_L": float(x_over_l[indices[0]]),
        "zone_95_max_x_over_L": float(x_over_l[indices[-1]]),
        "zone_95_length_fraction": float(np.mean(mask)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    atmospheric = float(design["pressure_validity"]["atmospheric_pressure_pa"])
    case_data: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for case_id in SELECTED_CASES:
        case_dir = source / "cases" / case_id
        metadata = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        case = metadata["case"]
        with np.load(case_dir / "truth_evaluation_grid.npz") as archive:
            x = archive["x"].copy()
            time = archive["t"].copy()
            pressure = archive["P"].copy()
            stress = archive["sigma_z"].copy()

        x_over_l = x / float(x[-1])
        pressure0 = float(metadata["initial_pressure_pa"])
        stress0 = float(metadata["initial_stress_pa"])
        pressure_rise = np.max(pressure - pressure0, axis=1) * 1.0e-3
        minimum_absolute_pressure = (np.min(pressure, axis=1) + atmospheric) * 1.0e-3
        pressure_drop = (pressure0 - np.min(pressure, axis=1)) * 1.0e-3
        stress_envelope = np.max(np.abs(stress - stress0), axis=1) * 1.0e-6

        label = (
            rf"$t_c/(L/c_f)={float(case['closure_time_over_L_cf']):.2f}$, "
            rf"$V_0={float(case['initial_velocity_m_s']):.4f}$ m/s"
        )
        item = {
            "case_id": case_id,
            "case": case,
            "label": label,
            "x_over_l": x_over_l,
            "time_s": time,
            "pressure_increment_kpa": (pressure - pressure0) * 1.0e-3,
            "stress_increment_mpa": (stress - stress0) * 1.0e-6,
            "pressure_rise_kpa": pressure_rise,
            "minimum_absolute_pressure_kpa": minimum_absolute_pressure,
            "pressure_drop_kpa": pressure_drop,
            "stress_envelope_mpa": stress_envelope,
        }
        case_data.append(item)

        metric_definitions = (
            ("maximum_pressure_rise", pressure_rise, pressure_rise, "kPa", float(np.max(pressure_rise))),
            (
                "minimum_absolute_pressure",
                minimum_absolute_pressure,
                pressure_drop,
                "kPa",
                float(np.min(minimum_absolute_pressure)),
            ),
            ("maximum_absolute_axial_stress", stress_envelope, stress_envelope, "MPa", float(np.max(stress_envelope))),
        )
        for metric, plotted, severity, unit, global_value in metric_definitions:
            zone = critical_zone(x_over_l, severity)
            summary_rows.append(
                {
                    "case_id": case_id,
                    "soil_to_pipe_modulus_ratio": float(case["soil_to_pipe_modulus_ratio"]),
                    "closure_time_over_L_cf": float(case["closure_time_over_L_cf"]),
                    "initial_velocity_m_s": float(case["initial_velocity_m_s"]),
                    "metric": metric,
                    "global_value": global_value,
                    "unit": unit,
                    **zone,
                    "saved_spatial_points": int(len(plotted)),
                }
            )

    plt.rcParams.update(
        {
            "font.size": 16.5,
            "axes.titlesize": 18.0,
            "axes.labelsize": 17.0,
            "xtick.labelsize": 15.0,
            "ytick.labelsize": 15.0,
            "legend.fontsize": 13.0,
            "lines.linewidth": 2.5,
            "savefig.dpi": 320,
        }
    )
    colors = ("#2878B5", "#E07B39", "#4DAF4A")
    figure = plt.figure(figsize=(15.2, 13.6))
    grid = figure.add_gridspec(
        3,
        6,
        height_ratios=(1.14, 0.92, 0.40),
        left=0.075,
        right=0.945,
        bottom=0.065,
        top=0.965,
        hspace=0.42,
        wspace=1.20,
    )
    pressure_map_axis = figure.add_subplot(grid[0, 0:3])
    stress_map_axis = figure.add_subplot(grid[0, 3:6])
    envelope_axes = (
        figure.add_subplot(grid[1, 0:2]),
        figure.add_subplot(grid[1, 2:4]),
        figure.add_subplot(grid[1, 4:6]),
    )
    monitoring_axis = figure.add_subplot(grid[2, :])

    representative = next(item for item in case_data if item["case_id"] == REPRESENTATIVE_CASE)
    travel_time_s = float(representative["x_over_l"][-1] * x[-1]) / float(
        json.loads((source / "cases" / REPRESENTATIVE_CASE / "result.json").read_text(encoding="utf-8"))[
            "fluid_wave_speed_m_s"
        ]
    )
    normalized_time = representative["time_s"] / travel_time_s
    visible = normalized_time <= DISPLAY_TRAVEL_TIMES + 1.0e-12
    x_grid, time_grid = np.meshgrid(representative["x_over_l"], normalized_time[visible])

    map_definitions = (
        (
            pressure_map_axis,
            representative["pressure_increment_kpa"][:, visible].T,
            "(a) Pressure-wave propagation",
            r"Pressure change $\Delta P$ (kPa)",
            "RdBu_r",
        ),
        (
            stress_map_axis,
            representative["stress_increment_mpa"][:, visible].T,
            "(b) Axial-stress-wave propagation",
            r"Stress change $\Delta\sigma_z$ (MPa)",
            "PuOr_r",
        ),
    )
    for axis, field, title, colorbar_label, cmap in map_definitions:
        limit = float(np.max(np.abs(field)))
        mesh = axis.pcolormesh(
            x_grid,
            time_grid,
            field,
            shading="auto",
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel(r"Normalized distance $x/L$")
        axis.set_ylabel(r"Normalized time $t/(L/c_f)$")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, DISPLAY_TRAVEL_TIMES)
        colorbar = figure.colorbar(mesh, ax=axis, pad=0.025, aspect=25, shrink=0.93)
        colorbar.set_label(colorbar_label)

    definitions = (
        (
            "pressure_rise_kpa",
            "(c) Maximum pressure rise",
            r"$\max_t(\Delta P)$ (kPa)",
            (0.8875, 1.0),
            "Pressure-monitoring region",
        ),
        (
            "pressure_drop_kpa",
            "(d) Maximum pressure drop",
            r"$\max_t(-\Delta P)$ (kPa)",
            (0.90625, 1.0),
            "Low-pressure-monitoring region",
        ),
        (
            "stress_envelope_mpa",
            "(e) Maximum absolute\nstress change",
            r"$\max_t|\Delta\sigma_z|$ (MPa)",
            (0.0, 0.45625),
            "Strain-monitoring region",
        ),
    )
    for axis, (key, title, ylabel, zone, zone_label) in zip(envelope_axes, definitions):
        axis.axvspan(zone[0], zone[1], color="#F2C14E", alpha=0.22, label=zone_label)
        for color, item in zip(colors, case_data):
            values = item[key]
            x_over_l = item["x_over_l"]
            axis.plot(x_over_l, values, color=color, label=item["label"])
            index = int(np.argmax(values))
            axis.plot(
                x_over_l[index],
                values[index],
                marker="o",
                ms=7.0,
                markerfacecolor="white",
                markeredgewidth=1.8,
                color=color,
                linestyle="none",
            )
        axis.set_title(title)
        axis.set_xlabel(r"Normalized distance $x/L$")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.22)
    envelope_axes[0].legend(frameon=False, loc="upper left", fontsize=11.8)

    monitoring_axis.set_title("(f) Monitoring implication obtained from the full-record envelopes")
    monitoring_axis.set_xlim(0.0, 1.0)
    monitoring_axis.set_ylim(-0.55, 0.65)
    monitoring_axis.hlines(0.0, 0.0, 1.0, color="#424242", linewidth=8.0, zorder=1)
    monitoring_axis.axvspan(0.0, 0.45625, ymin=0.30, ymax=0.70, color="#7E57C2", alpha=0.28)
    monitoring_axis.axvspan(0.8875, 1.0, ymin=0.30, ymax=0.70, color="#E53935", alpha=0.28)
    monitoring_axis.scatter([0.08, 0.25, 0.43], [0.0, 0.0, 0.0], marker="D", s=78, color="#6A3D9A", zorder=3)
    monitoring_axis.scatter([0.91, 0.96, 1.0], [0.0, 0.0, 0.0], marker="o", s=82, color="#D32F2F", zorder=3)
    monitoring_axis.annotate(
        "Prioritize strain monitoring\n(upstream and first pipe section)",
        xy=(0.23, 0.05),
        xytext=(0.23, 0.42),
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "->", "color": "#6A3D9A", "lw": 1.6},
        color="#542788",
        fontweight="bold",
        fontsize=15.5,
    )
    monitoring_axis.annotate(
        "Prioritize pressure monitoring\n(downstream valve region)",
        xy=(0.95, 0.05),
        xytext=(0.79, 0.42),
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "->", "color": "#D32F2F", "lw": 1.6},
        color="#B2182B",
        fontweight="bold",
        fontsize=15.5,
    )
    monitoring_axis.text(0.0, -0.28, "Upstream", ha="left", va="center", fontweight="bold")
    monitoring_axis.text(1.0, -0.28, "Valve / downstream", ha="right", va="center", fontweight="bold")
    monitoring_axis.set_xlabel(r"Normalized distance $x/L$")
    monitoring_axis.set_yticks([])
    monitoring_axis.spines[["left", "right", "top"]].set_visible(False)

    figure_files = []
    for suffix in ("png", "pdf"):
        path = output / f"F44_MOC_propagation_envelopes_monitoring.{suffix}"
        figure.savefig(path, dpi=320 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        figure_files.append(path.name)
    plt.close(figure)

    table_name = "T15_MOC_spatial_critical_zones.csv"
    write_csv(output / table_name, summary_rows)
    report = {
        "status": "pass",
        "training_performed": False,
        "new_moc_simulation_performed": False,
        "source": str(source.relative_to(ROOT)),
        "selected_case_ids": list(SELECTED_CASES),
        "soil_to_pipe_modulus_ratio": 0.01,
        "representative_case_id": REPRESENTATIVE_CASE,
        "time_space_display_range": f"0 <= t/(L/c_f) <= {DISPLAY_TRAVEL_TIMES:g}",
        "envelope_time_range_s": [
            float(representative["time_s"][0]),
            float(representative["time_s"][-1]),
        ],
        "critical_zone_definition": "spatial points with response severity at least 95% of the case global maximum",
        "figures": figure_files,
        "table": table_name,
    }
    (output / "figure20_moc_spatial_envelopes_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.source_dir, args.output_dir)
