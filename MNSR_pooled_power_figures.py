"""Create pooled pin- and ring-power figures for thesis Section 5.3.3.

This is a post-processing-only workflow.  It reads the already validated
four-replicate pooled kappa-fission CSV files and never imports or runs OpenMC.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

_MATPLOTLIB_CONFIG = Path(__file__).resolve().parent / "build/.matplotlib"
_MATPLOTLIB_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIRMATION_ROOT = (
    PROJECT_ROOT / "build/optimization/real_ga/statistical_confirmation"
).resolve()
POSITION_TABLE_PATH = (PROJECT_ROOT / "build/position_table.csv").resolve()

REFERENCE_DUMMY_IDS = (285, 298, 311, 324, 337)
OPTIMIZED_DUMMY_IDS = (200, 213, 223, 239, 256)

REFERENCE_SOURCE = CONFIRMATION_ROOT / "reference/pooled_pin_power.csv"
OPTIMIZED_SOURCE = CONFIRMATION_ROOT / "2dee2f4df9bb/pooled_pin_power.csv"

EXPECTED_REFERENCE_FR = 1.1544141675305262
EXPECTED_REFERENCE_CV = 0.058960334949843894
EXPECTED_OPTIMIZED_FR = 1.1482078714312154
EXPECTED_OPTIMIZED_CV = 0.054351860073999655
FLOAT_TOLERANCE = 1.0e-12

REFERENCE_PIN_OUTPUT = CONFIRMATION_ROOT / "reference_pooled_pin_power.csv"
OPTIMIZED_PIN_OUTPUT = CONFIRMATION_ROOT / "optimized_pooled_pin_power.csv"
REFERENCE_RING_OUTPUT = CONFIRMATION_ROOT / "reference_pooled_ring_power.csv"
OPTIMIZED_RING_OUTPUT = CONFIRMATION_ROOT / "optimized_pooled_ring_power.csv"
RING_COMPARISON_OUTPUT = (
    CONFIRMATION_ROOT / "reference_vs_optimized_pooled_ring_comparison.csv"
)

FIGURE_5_7_PNG = CONFIRMATION_ROOT / "figure_5_7_pooled_pin_power_comparison.png"
FIGURE_5_7_PDF = CONFIRMATION_ROOT / "figure_5_7_pooled_pin_power_comparison.pdf"
FIGURE_5_7_SUMMARY = CONFIRMATION_ROOT / "figure_5_7_summary.json"
FIGURE_5_8_PNG = CONFIRMATION_ROOT / "figure_5_8_pooled_ring_power_comparison.png"
FIGURE_5_8_PDF = CONFIRMATION_ROOT / "figure_5_8_pooled_ring_power_comparison.pdf"
FIGURE_5_8_RANGE_PNG = (
    CONFIRMATION_ROOT / "figure_5_8_pooled_ring_power_with_range.png"
)
FIGURE_5_8_SUMMARY = CONFIRMATION_ROOT / "figure_5_8_summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_and_validate_pooled(
    source: Path,
    dummy_ids: tuple[int, ...],
    position_table: pd.DataFrame,
    expected_fr: float,
    expected_cv: float,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one pooled table and prove that its geometry/statistics are intact."""
    if not source.is_file():
        raise FileNotFoundError(f"找不到{label} pooled棒功率文件：{source}")

    table = pd.read_csv(source).sort_values("ordinary_position_id").reset_index(drop=True)
    if "ring" not in table.columns and "ring_index" in table.columns:
        table = table.rename(columns={"ring_index": "ring"})
    source_columns = {
        "ordinary_position_id",
        "ring",
        "local_index",
        "x_cm",
        "y_cm",
        "pooled_kappa_fission_mean",
        "pooled_kappa_fission_std",
        "relative_power",
        "relative_uncertainty",
    }
    missing_columns = sorted(source_columns.difference(table.columns))
    if missing_columns:
        raise ValueError(f"{label} pooled文件缺少列：{missing_columns}")
    if len(table) != 345:
        raise ValueError(f"{label}燃料棒数量应为345，实际为{len(table)}。")
    if table["ordinary_position_id"].duplicated().any():
        raise ValueError(f"{label}存在重复ordinary_position_id。")

    ordinary = position_table.loc[
        position_table["position_category"] == "ordinary"
    ].copy()
    ordinary["ordinary_position_id"] = ordinary["ordinary_position_id"].astype(int)
    allowed_ids = set(ordinary["ordinary_position_id"])
    dummy_set = set(dummy_ids)
    if len(allowed_ids) != 350 or len(dummy_set) != 5:
        raise ValueError("现有孔位映射必须包含350个普通孔位和5个唯一dummy。")
    expected_fuel_ids = allowed_ids.difference(dummy_set)
    actual_fuel_ids = set(table["ordinary_position_id"].astype(int))
    if actual_fuel_ids != expected_fuel_ids:
        missing = sorted(expected_fuel_ids.difference(actual_fuel_ids))
        unexpected = sorted(actual_fuel_ids.difference(expected_fuel_ids))
        raise ValueError(
            f"{label}燃料孔位与几何映射不一致；缺失={missing}，多余={unexpected}。"
        )
    if actual_fuel_ids.intersection(dummy_set):
        raise ValueError(f"{label} dummy被错误计入燃料棒统计。")

    mapping = ordinary.set_index("ordinary_position_id")
    mapped = mapping.loc[table["ordinary_position_id"].astype(int)]
    if "physical_position_id" not in table.columns:
        table["physical_position_id"] = mapped["physical_position_id"].to_numpy(int)
    integer_checks = {
        "physical_position_id": "physical_position_id",
        "ring": "ring_index",
        "local_index": "local_index",
    }
    for pooled_column, mapping_column in integer_checks.items():
        if not np.array_equal(
            table[pooled_column].to_numpy(int), mapped[mapping_column].to_numpy(int)
        ):
            raise ValueError(f"{label}的{pooled_column}与现有geometry mapping不一致。")
    for coordinate in ("x_cm", "y_cm"):
        if not np.allclose(
            table[coordinate].to_numpy(float),
            mapped[coordinate].to_numpy(float),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"{label}的{coordinate}与现有geometry mapping不一致。")

    numeric_columns = [
        "pooled_kappa_fission_mean",
        "pooled_kappa_fission_std",
        "relative_power",
        "relative_uncertainty",
    ]
    if not np.isfinite(table[numeric_columns].to_numpy(float)).all():
        raise ValueError(f"{label} pooled数据包含非有限数值。")
    if (table["pooled_kappa_fission_mean"] <= 0.0).any():
        raise ValueError(f"{label} pooled kappa-fission必须全部为正。")

    raw_mean = table["pooled_kappa_fission_mean"].to_numpy(float)
    raw_std = table["pooled_kappa_fission_std"].to_numpy(float)
    recomputed_relative_power = raw_mean / float(np.mean(raw_mean))
    recomputed_relative_uncertainty = raw_std / raw_mean
    if not np.allclose(
        recomputed_relative_power,
        table["relative_power"].to_numpy(float),
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise ValueError(f"{label} relative_power不是由pooled原始均值统一归一化得到。")
    if not np.allclose(
        recomputed_relative_uncertainty,
        table["relative_uncertainty"].to_numpy(float),
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise ValueError(f"{label} relative_uncertainty与pooled原始统计量不一致。")

    mean_relative_power = float(np.mean(recomputed_relative_power))
    fr = float(np.max(recomputed_relative_power))
    cv = float(np.std(recomputed_relative_power, ddof=0))
    if not np.isclose(mean_relative_power, 1.0, rtol=0.0, atol=FLOAT_TOLERANCE):
        raise ValueError(f"{label}平均相对功率不为1：{mean_relative_power}")
    if not np.isclose(fr, expected_fr, rtol=0.0, atol=FLOAT_TOLERANCE):
        raise ValueError(f"{label} Fr={fr}，与statistical confirmation摘要不一致。")
    if not np.isclose(cv, expected_cv, rtol=0.0, atol=FLOAT_TOLERANCE):
        raise ValueError(f"{label} CV={cv}，与statistical confirmation摘要不一致。")

    output = table.rename(
        columns={"pooled_kappa_fission_mean": "pooled_kappa_fission"}
    )[
        [
            "ordinary_position_id",
            "physical_position_id",
            "ring",
            "local_index",
            "x_cm",
            "y_cm",
            "pooled_kappa_fission",
            "pooled_kappa_fission_std",
            "relative_power",
            "relative_uncertainty",
        ]
    ].copy()
    max_row = output.loc[output["relative_power"].idxmax()]
    min_row = output.loc[output["relative_power"].idxmin()]
    summary = {
        "scheme": list(dummy_ids),
        "fuel_pin_count": int(len(output)),
        "unique_fuel_position_count": int(output["ordinary_position_id"].nunique()),
        "mean_relative_power": mean_relative_power,
        "Fr_pool": fr,
        "CV_pool": cv,
        "max_pin_id": int(max_row["ordinary_position_id"]),
        "max_relative_power": float(max_row["relative_power"]),
        "min_pin_id": int(min_row["ordinary_position_id"]),
        "min_relative_power": float(min_row["relative_power"]),
        "source_file": str(source),
        "source_sha256": _sha256(source),
        "pooling_basis": "four-replicate pooled raw kappa-fission",
    }
    return output, summary


def _ring_statistics(table: pd.DataFrame) -> pd.DataFrame:
    ring = (
        table.groupby("ring", sort=True)["relative_power"]
        .agg(
            fuel_pin_count="size",
            mean_relative_power="mean",
            min_relative_power="min",
            max_relative_power="max",
            std_relative_power=lambda values: float(np.std(values, ddof=0)),
        )
        .reset_index()
    )
    ring["ring"] = ring["ring"].astype(int)
    ring["fuel_pin_count"] = ring["fuel_pin_count"].astype(int)
    if list(ring["ring"]) != list(range(1, 11)):
        raise ValueError(f"圈层统计必须完整覆盖第1至10圈，实际为{list(ring['ring'])}。")
    if int(ring["fuel_pin_count"].sum()) != 345:
        raise ValueError("圈层fuel pin计数总和不是345。")
    return ring


def _configure_axis(axis: plt.Axes) -> None:
    axis.tick_params(axis="both", labelsize=11.5, width=1.2, length=5)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)


def _plot_figure_5_7(
    reference: pd.DataFrame,
    optimized: pd.DataFrame,
    position_table: pd.DataFrame,
    reference_summary: dict[str, Any],
    optimized_summary: dict[str, Any],
) -> tuple[float, float]:
    ordinary = position_table.loc[
        position_table["position_category"] == "ordinary"
    ].copy()
    ordinary["ordinary_position_id"] = ordinary["ordinary_position_id"].astype(int)
    tie_rods = position_table.loc[position_table["position_category"] == "tie_rod"]
    central = position_table.loc[position_table["position_category"] == "central"]
    if len(tie_rods) != 4 or len(central) != 1:
        raise ValueError("现有geometry mapping必须包含4根tie rods和1个central guide。")

    vmin = float(min(reference["relative_power"].min(), optimized["relative_power"].min()))
    vmax = float(max(reference["relative_power"].max(), optimized["relative_power"].max()))
    normalization = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colormap = "viridis"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 5.4),
        sharex=True,
        sharey=True,
        facecolor="white",
        constrained_layout=True,
    )
    scatter = None
    panels = (
        (
            axes[0],
            reference,
            REFERENCE_DUMMY_IDS,
            "(a) Reference arrangement",
            reference_summary["max_pin_id"],
        ),
        (
            axes[1],
            optimized,
            OPTIMIZED_DUMMY_IDS,
            "(b) Optimized arrangement",
            optimized_summary["max_pin_id"],
        ),
    )
    for axis, fuel, dummy_ids, panel_title, max_pin_id in panels:
        scatter = axis.scatter(
            fuel["x_cm"],
            fuel["y_cm"],
            c=fuel["relative_power"],
            s=35,
            marker="o",
            cmap=colormap,
            norm=normalization,
            linewidths=0.12,
            edgecolors=(0.15, 0.15, 0.15, 0.45),
            rasterized=False,
            zorder=2,
        )
        dummy = ordinary.loc[
            ordinary["ordinary_position_id"].isin(dummy_ids)
        ]
        axis.scatter(
            dummy["x_cm"],
            dummy["y_cm"],
            s=82,
            marker="D",
            facecolors="white",
            edgecolors="#7a3e9d",
            linewidths=1.8,
            zorder=5,
        )
        axis.scatter(
            tie_rods["x_cm"],
            tie_rods["y_cm"],
            s=72,
            marker="s",
            facecolors="white",
            edgecolors="#d55e00",
            linewidths=1.8,
            zorder=5,
        )
        axis.scatter(
            central["x_cm"],
            central["y_cm"],
            s=105,
            marker="X",
            c="#202020",
            linewidths=1.4,
            zorder=6,
        )
        max_pin = fuel.loc[
            fuel["ordinary_position_id"].astype(int) == int(max_pin_id)
        ].iloc[0]
        axis.scatter(
            [max_pin["x_cm"]],
            [max_pin["y_cm"]],
            s=145,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            zorder=7,
        )
        # The pooled maxima are close to the central guide.  Put the label
        # farther into the same quadrant so its text does not cover the guide.
        offset = np.array(
            [
                3.0 if float(max_pin["x_cm"]) >= 0.0 else -3.0,
                2.0 if float(max_pin["y_cm"]) >= 0.0 else -2.0,
            ]
        )
        axis.annotate(
            "Max pin",
            xy=(float(max_pin["x_cm"]), float(max_pin["y_cm"])),
            xytext=(float(max_pin["x_cm"] + offset[0]), float(max_pin["y_cm"] + offset[1])),
            fontsize=10.5,
            fontweight="semibold",
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.0},
            zorder=8,
        )
        axis.set_title(panel_title, fontsize=15.5, pad=9)
        axis.set_xlabel("x (cm)", fontsize=13.5, labelpad=6)
        axis.set_ylabel("y (cm)", fontsize=13.5, labelpad=6)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-12.0, 12.0)
        axis.set_ylim(-12.0, 12.0)
        axis.set_facecolor("white")
        _configure_axis(axis)

    if scatter is None:
        raise RuntimeError("图5-7燃料棒散点未生成。")
    colorbar = figure.colorbar(
        scatter,
        ax=axes,
        location="right",
        fraction=0.038,
        pad=0.025,
    )
    colorbar.set_label("Relative pin power", fontsize=13.0, labelpad=9)
    colorbar.ax.tick_params(labelsize=10.5, width=1.1, length=4)
    colorbar.outline.set_linewidth(1.0)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6.8,
               markerfacecolor="#6f8f78", markeredgecolor="#333333", label="Fuel pin"),
        Line2D([], [], marker="D", linestyle="none", markersize=7.5,
               markerfacecolor="white", markeredgecolor="#7a3e9d", markeredgewidth=1.6,
               label="Dummy elements"),
        Line2D([], [], marker="s", linestyle="none", markersize=7.5,
               markerfacecolor="white", markeredgecolor="#d55e00", markeredgewidth=1.6,
               label="Tie rods"),
        Line2D([], [], marker="X", linestyle="none", markersize=8.0,
               markerfacecolor="#202020", markeredgecolor="#202020",
               label="Control guide tube"),
        Line2D([], [], marker="o", linestyle="none", markersize=9.0,
               markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.5,
               label="Max pin"),
    ]
    legend = figure.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=5,
        frameon=True,
        fontsize=10.8,
        handletextpad=0.55,
        columnspacing=1.25,
        borderpad=0.55,
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_edgecolor("#666666")
    legend.get_frame().set_facecolor("white")

    figure.savefig(
        FIGURE_5_7_PNG,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    figure.savefig(
        FIGURE_5_7_PDF,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(figure)
    return vmin, vmax


def _ring_y_limits(reference_ring: pd.DataFrame, optimized_ring: pd.DataFrame) -> tuple[float, float]:
    values = np.r_[
        reference_ring["mean_relative_power"].to_numpy(float),
        optimized_ring["mean_relative_power"].to_numpy(float),
        1.0,
    ]
    data_range = float(np.max(values) - np.min(values))
    padding = max(0.035, data_range * 0.10)
    return float(np.min(values) - padding), float(np.max(values) + padding)


def _plot_ring_lines(
    reference_ring: pd.DataFrame,
    optimized_ring: pd.DataFrame,
    output_png: Path,
    output_pdf: Path | None,
    show_range: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.4), facecolor="white")
    reference_color = "#3b6ea8"
    optimized_color = "#c44e52"
    rings = reference_ring["ring"].to_numpy(int)
    if show_range:
        axis.fill_between(
            rings,
            reference_ring["min_relative_power"],
            reference_ring["max_relative_power"],
            color=reference_color,
            alpha=0.12,
            linewidth=0.0,
            label="Reference spatial pin range",
        )
        axis.fill_between(
            rings,
            optimized_ring["min_relative_power"],
            optimized_ring["max_relative_power"],
            color=optimized_color,
            alpha=0.12,
            linewidth=0.0,
            label="Optimized spatial pin range",
        )
    axis.plot(
        rings,
        reference_ring["mean_relative_power"],
        color=reference_color,
        marker="o",
        markersize=6.5,
        linewidth=2.0,
        label="Reference",
        zorder=3,
    )
    axis.plot(
        rings,
        optimized_ring["mean_relative_power"],
        color=optimized_color,
        marker="s",
        markersize=6.2,
        linewidth=2.0,
        label="Optimized",
        zorder=3,
    )
    axis.axhline(
        1.0,
        color="#333333",
        linestyle="--",
        linewidth=1.45,
        label="Core average = 1",
        zorder=2,
    )
    axis.set_xlabel("Ring number", fontsize=14.0, labelpad=7)
    axis.set_ylabel("Mean relative pin power", fontsize=14.0, labelpad=7)
    axis.set_xticks(range(1, 11))
    axis.set_xlim(0.7, 10.3)
    if show_range:
        all_bounds = np.r_[
            reference_ring[["min_relative_power", "max_relative_power"]].to_numpy(float).ravel(),
            optimized_ring[["min_relative_power", "max_relative_power"]].to_numpy(float).ravel(),
            1.0,
        ]
        padding = max(0.035, float(np.ptp(all_bounds)) * 0.06)
        axis.set_ylim(float(np.min(all_bounds) - padding), float(np.max(all_bounds) + padding))
    else:
        axis.set_ylim(*_ring_y_limits(reference_ring, optimized_ring))
    axis.grid(True, color="#777777", alpha=0.16, linewidth=0.7)
    axis.set_axisbelow(True)
    _configure_axis(axis)
    legend = axis.legend(
        loc="best",
        fontsize=11.0,
        frameon=True,
        borderpad=0.6,
        labelspacing=0.55,
    )
    legend.get_frame().set_linewidth(0.9)
    legend.get_frame().set_edgecolor("#777777")
    legend.get_frame().set_facecolor("white")
    figure.tight_layout()
    figure.savefig(
        output_png,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    if output_pdf is not None:
        figure.savefig(
            output_pdf,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
    plt.close(figure)


def _ring_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "ring": int(row["ring"]),
            "fuel_pin_count": int(row["fuel_pin_count"]),
            "mean_relative_power": float(row["mean_relative_power"]),
            "min_relative_power": float(row["min_relative_power"]),
            "max_relative_power": float(row["max_relative_power"]),
            "std_relative_power": float(row["std_relative_power"]),
        }
        for _, row in table.iterrows()
    ]


def build_pooled_power_figures() -> dict[str, Any]:
    """Validate pooled inputs, generate thesis tables/figures, and return summary."""
    CONFIRMATION_ROOT.mkdir(parents=True, exist_ok=True)
    if not POSITION_TABLE_PATH.is_file():
        raise FileNotFoundError(f"找不到现有geometry孔位映射：{POSITION_TABLE_PATH}")
    position_table = pd.read_csv(POSITION_TABLE_PATH)
    if len(position_table) != 355:
        raise ValueError(f"物理孔位应为355，实际为{len(position_table)}。")
    if position_table[["x_cm", "y_cm"]].duplicated().any():
        raise ValueError("现有geometry mapping包含重复圆柱中心坐标。")

    reference, reference_summary = _read_and_validate_pooled(
        REFERENCE_SOURCE,
        REFERENCE_DUMMY_IDS,
        position_table,
        EXPECTED_REFERENCE_FR,
        EXPECTED_REFERENCE_CV,
        "REFERENCE",
    )
    optimized, optimized_summary = _read_and_validate_pooled(
        OPTIMIZED_SOURCE,
        OPTIMIZED_DUMMY_IDS,
        position_table,
        EXPECTED_OPTIMIZED_FR,
        EXPECTED_OPTIMIZED_CV,
        "OPTIMIZED",
    )
    reference.to_csv(REFERENCE_PIN_OUTPUT, index=False, encoding="utf-8")
    optimized.to_csv(OPTIMIZED_PIN_OUTPUT, index=False, encoding="utf-8")

    reference_ring = _ring_statistics(reference)
    optimized_ring = _ring_statistics(optimized)
    reference_ring.to_csv(REFERENCE_RING_OUTPUT, index=False, encoding="utf-8")
    optimized_ring.to_csv(OPTIMIZED_RING_OUTPUT, index=False, encoding="utf-8")

    ring_comparison = reference_ring.merge(
        optimized_ring,
        on="ring",
        suffixes=("_reference", "_optimized"),
        validate="one_to_one",
    )
    comparison_output = pd.DataFrame(
        {
            "ring": ring_comparison["ring"].astype(int),
            "reference_mean": ring_comparison["mean_relative_power_reference"],
            "optimized_mean": ring_comparison["mean_relative_power_optimized"],
            "delta_mean": (
                ring_comparison["mean_relative_power_optimized"]
                - ring_comparison["mean_relative_power_reference"]
            ),
            "delta_percent": (
                (
                    ring_comparison["mean_relative_power_optimized"]
                    - ring_comparison["mean_relative_power_reference"]
                )
                / ring_comparison["mean_relative_power_reference"]
                * 100.0
            ),
            "reference_min": ring_comparison["min_relative_power_reference"],
            "reference_max": ring_comparison["max_relative_power_reference"],
            "optimized_min": ring_comparison["min_relative_power_optimized"],
            "optimized_max": ring_comparison["max_relative_power_optimized"],
            "fuel_pin_count_reference": ring_comparison["fuel_pin_count_reference"].astype(int),
            "fuel_pin_count_optimized": ring_comparison["fuel_pin_count_optimized"].astype(int),
        }
    )
    comparison_output.to_csv(RING_COMPARISON_OUTPUT, index=False, encoding="utf-8")

    vmin, vmax = _plot_figure_5_7(
        reference,
        optimized,
        position_table,
        reference_summary,
        optimized_summary,
    )
    figure_5_7_payload = {
        "data_basis": "four-replicate pooled raw kappa-fission",
        "pooling_formula": "P_i,pool=mean(P_i,r); sigma_i,pool=sqrt(sum(sigma_i,r^2))/4",
        "normalization": "relative_power_i=P_i,pool/mean(P_pool)",
        "reference": reference_summary,
        "optimized": optimized_summary,
        "shared_color_scale": True,
        "colormap": "viridis",
        "vmin": vmin,
        "vmax": vmax,
        "coordinate_limits_cm": {"x": [-12.0, 12.0], "y": [-12.0, 12.0]},
        "interpolation": False,
        "openmc_runs": 0,
        "ga_runs": 0,
    }
    _write_json(FIGURE_5_7_SUMMARY, figure_5_7_payload)

    _plot_ring_lines(
        reference_ring,
        optimized_ring,
        FIGURE_5_8_PNG,
        FIGURE_5_8_PDF,
        show_range=False,
    )
    _plot_ring_lines(
        reference_ring,
        optimized_ring,
        FIGURE_5_8_RANGE_PNG,
        None,
        show_range=True,
    )

    reference_max_ring = reference_ring.loc[
        reference_ring["mean_relative_power"].idxmax()
    ]
    reference_min_ring = reference_ring.loc[
        reference_ring["mean_relative_power"].idxmin()
    ]
    optimized_max_ring = optimized_ring.loc[
        optimized_ring["mean_relative_power"].idxmax()
    ]
    optimized_min_ring = optimized_ring.loc[
        optimized_ring["mean_relative_power"].idxmin()
    ]
    delta = comparison_output["delta_mean"].to_numpy(float)
    figure_5_8_payload = {
        "data_basis": "four-replicate pooled raw kappa-fission",
        "reference": {
            "rings": _ring_records(reference_ring),
            "maximum_mean_power_ring": int(reference_max_ring["ring"]),
            "maximum_mean_ring_power": float(reference_max_ring["mean_relative_power"]),
            "minimum_mean_power_ring": int(reference_min_ring["ring"]),
            "minimum_mean_ring_power": float(reference_min_ring["mean_relative_power"]),
        },
        "optimized": {
            "rings": _ring_records(optimized_ring),
            "maximum_mean_power_ring": int(optimized_max_ring["ring"]),
            "maximum_mean_ring_power": float(optimized_max_ring["mean_relative_power"]),
            "minimum_mean_power_ring": int(optimized_min_ring["ring"]),
            "minimum_mean_ring_power": float(optimized_min_ring["mean_relative_power"]),
        },
        "delta_ring_mean": [
            {
                "ring": int(ring),
                "optimized_minus_reference": float(value),
            }
            for ring, value in zip(comparison_output["ring"], delta)
        ],
        "rings_with_increased_mean_power": [
            int(ring)
            for ring, value in zip(comparison_output["ring"], delta)
            if value > 0.0
        ],
        "rings_with_decreased_mean_power": [
            int(ring)
            for ring, value in zip(comparison_output["ring"], delta)
            if value < 0.0
        ],
        "openmc_runs": 0,
        "ga_runs": 0,
    }
    _write_json(FIGURE_5_8_SUMMARY, figure_5_8_payload)

    return {
        "reference": reference_summary,
        "optimized": optimized_summary,
        "figure_5_7": figure_5_7_payload,
        "figure_5_8": figure_5_8_payload,
        "outputs": [
            str(path)
            for path in (
                REFERENCE_PIN_OUTPUT,
                OPTIMIZED_PIN_OUTPUT,
                REFERENCE_RING_OUTPUT,
                OPTIMIZED_RING_OUTPUT,
                RING_COMPARISON_OUTPUT,
                FIGURE_5_7_PNG,
                FIGURE_5_7_PDF,
                FIGURE_5_7_SUMMARY,
                FIGURE_5_8_PNG,
                FIGURE_5_8_PDF,
                FIGURE_5_8_RANGE_PNG,
                FIGURE_5_8_SUMMARY,
            )
        ],
        "openmc_runs": 0,
        "ga_runs": 0,
    }


if __name__ == "__main__":
    result = build_pooled_power_figures()
    print(json.dumps(result, ensure_ascii=False, indent=2))
