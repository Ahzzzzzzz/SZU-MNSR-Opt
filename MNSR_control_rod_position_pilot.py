"""MNSR-SZ central-control-rod axial validation and three-position pilot.

This workflow intentionally stops after one REFERENCE screening calculation at
each of 0.0, 11.25 and 24.0 cm.  It never launches the eight formal inserted
final replicates.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any
from unittest.mock import PropertyMock, patch

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import openmc
import pandas as pd

from MNSR_control_rod_flux_workflow import (
    _find_openmc_executable,
    _prepare_materials,
    load_model_context,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = (PROJECT_ROOT / "build/results/control_rod_analysis").resolve()
PILOT_ROOT = OUTPUT_ROOT / "axial_position_pilot"
REFERENCE = (285, 298, 311, 324, 337)
POSITIONS_CM = (0.0, 11.25, 24.0)
PARTICLES = 5000
BATCHES = 60
INACTIVE = 15
BASE_SEED = 20260819
MAX_NEW_PILOT_OPENMC_RUNS = 3
GEOMETRY_MODEL_VERSION = "iaea_reference_v1"
CONTROL_ROD_DEFINITION_VERSION = "mnsr_sz_independent_axial_surfaces_v1"
IAEA_LEU_TOTAL_ROD_WORTH_MK = 5.75
IAEA_URL = "https://www-pub.iaea.org/MTCD/Publications/PDF/TE1844-Web.pdf"

MODEL_DEFINITION_PATH = OUTPUT_ROOT / "control_rod_axial_model_definition.json"
GEOMETRY_VALIDATION_PATH = OUTPUT_ROOT / "control_rod_axial_geometry_validation.json"
PILOT_CSV_PATH = OUTPUT_ROOT / "control_rod_position_pilot.csv"
PILOT_SUMMARY_PATH = OUTPUT_ROOT / "control_rod_position_pilot_summary.json"
PLOT_PNG_PATH = OUTPUT_ROOT / "control_rod_axial_validation.png"
PLOT_PDF_PATH = OUTPUT_ROOT / "control_rod_axial_validation.pdf"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _stable_seed(position_cm: float) -> int:
    identity = {
        "base_seed": BASE_SEED,
        "scheme": list(REFERENCE),
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "control_rod_definition_version": CONTROL_ROD_DEFINITION_VERSION,
        "control_rod_state": "inserted",
        "control_rod_position_cm": position_cm,
        "profile": "screening_pilot",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 2) + 1


def write_model_definition(context: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "control_rod_axial_definition_version": CONTROL_ROD_DEFINITION_VERSION,
        "public_source_search_result": {
            "fully_inserted_absolute_z_position_found": False,
            "finding": (
                "IAEA-TECDOC-1844 gives the component dimensions, the core-centre "
                "origin, a 0-24 cm position curve and the 112.5 mm critical position, "
                "but does not state the absolute Cd bottom/top coordinates at position 0."
            ),
            "source": IAEA_URL,
            "relevant_pages": [99, 101, 104, 105, 109],
        },
        "control_rod_axial_model_assumption": context[
            "CONTROL_ROD_AXIAL_MODEL_ASSUMPTION"
        ],
        "assumption_status": "explicit_paper_modelling_assumption_not_as_built_claim",
        "coordinate_definition": {
            "core_centre_cm": [0.0, 0.0, 0.0],
            "position_0_meaning": "fully inserted reference used in this paper model",
            "positive_position_direction": "+z withdrawal",
            "moving_components": ["cadmium", "central_al_rod", "ss304_tube"],
            "fixed_component": "LT-21 guide tube",
        },
        "radial_dimensions": {
            "guide_tube": {
                "inner_diameter_mm": 9.0,
                "outer_diameter_mm": 12.0,
            },
            "cadmium": {
                "inner_diameter_mm": 1.9,
                "outer_diameter_mm": 3.9,
            },
            "central_al_rod": {"diameter_mm": 1.9},
            "ss304_tube": {
                "outer_diameter_mm": 5.0,
                "wall_thickness_mm": 0.5,
            },
        },
        "lengths_cm": {
            "guide_tube": context["GUIDE_TUBE_HEIGHT"],
            "cadmium": context["CD_ABSORBER_HEIGHT"],
            "central_al_rod": context["CONTROL_AL_ROD_HEIGHT"],
            "ss304_tube": context["CONTROL_SS_TUBE_HEIGHT"],
        },
        "position_0_axial_bounds_cm": {
            "guide_tube": {
                "z_bottom": context["Z_GUIDE_BOTTOM"],
                "z_top": context["Z_GUIDE_TOP"],
            },
            "cadmium": {"z_bottom": -13.3, "z_top": 13.3},
            "central_al_rod": {"z_bottom": -14.5, "z_top": 14.5},
            "ss304_tube": {"z_bottom": -22.5, "z_top": 22.5},
        },
        "pilot_positions_cm": list(POSITIONS_CM),
        "critical_position_context_cm": 11.25,
        "literature_leu_total_control_rod_worth_mk": IAEA_LEU_TOTAL_ROD_WORTH_MK,
        "literature_value_usage": "independent order-of-magnitude check only; no fitting",
        "materials_or_other_core_geometry_modified": False,
    }
    _write_json(MODEL_DEFINITION_PATH, payload)
    return payload


def _material_cell_map(geometry: openmc.Geometry) -> dict[str, str]:
    names = {
        "IAEA movable control rod inner aluminum",
        "IAEA movable cadmium absorber tube",
        "IAEA movable SS-304 control rod tube",
        "IAEA LT-21 control guide tube",
        "IAEA control rod corridor water complement",
    }
    return {
        cell.name: getattr(cell.fill, "name", "")
        for cell in geometry.get_all_cells().values()
        if cell.name in names
    }


def _to_uint8(image_array: np.ndarray) -> np.ndarray:
    array = np.asarray(image_array)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(np.rint(array * 255.0), 0, 255)
    return array.astype(np.uint8)


def _count_magenta(image_array: np.ndarray) -> int:
    rgb = _to_uint8(image_array)[..., :3]
    return int(np.sum(np.all(rgb == np.array([255, 0, 255]), axis=-1)))


def _openmc_plot_array(
    geometry: openmc.Geometry,
    colors: dict[openmc.Material, str],
    width: tuple[float, float],
    pixels: tuple[int, int],
) -> tuple[np.ndarray, tuple[float, float, float, float], int, str]:
    buffer = io.StringIO()
    with (
        contextlib.redirect_stdout(buffer),
        contextlib.redirect_stderr(buffer),
        patch.object(
            openmc.Model,
            "is_initialized",
            new_callable=PropertyMock,
            return_value=False,
        ),
    ):
        axis = geometry.plot(
            basis="xz",
            origin=(0.0, 0.0, 0.0),
            width=width,
            pixels=pixels,
            color_by="material",
            colors=colors,
            show_overlaps=True,
            overlap_color="magenta",
            legend=False,
            axis_units="cm",
            openmc_exec=_find_openmc_executable(),
        )
    image = axis.images[0]
    array = np.asarray(image.get_array()).copy()
    extent = tuple(float(value) for value in image.get_extent())
    overlap_pixels = _count_magenta(array)
    plt.close(axis.figure)
    return array, extent, overlap_pixels, buffer.getvalue()


def build_axial_validation_plot(
    context: dict[str, Any],
    geometry: openmc.Geometry,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    prepared_materials, aliases = _prepare_materials(context, geometry)
    material_by_name = {material.name: material for material in prepared_materials}
    colors = {
        material_by_name["LEU UO2 12.5wt%"]: "gold",
        material_by_name["Zircaloy-4"]: "silver",
        material_by_name["Al-303-1"]: "orange",
        material_by_name["LT-21"]: "dimgray",
        material_by_name["SS-304"]: "slategray",
        material_by_name["Light Water"]: "lightblue",
        material_by_name["Beryllium Reflector"]: "green",
        material_by_name["Natural Cadmium"]: "navy",
    }
    overview, overview_extent, overview_overlap, overview_log = _openmc_plot_array(
        geometry, colors, width=(46.0, 70.0), pixels=(1600, 2400)
    )
    detail, detail_extent, detail_overlap, detail_log = _openmc_plot_array(
        geometry, colors, width=(1.5, 52.0), pixels=(1200, 2600)
    )
    bounds = metadata["control_rod_axial_bounds"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17.0, 10.0),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    axes[0].imshow(overview, extent=overview_extent, origin="upper", interpolation="nearest")
    axes[1].imshow(detail, extent=detail_extent, origin="upper", interpolation="nearest")
    axes[0].set_title("(a) Reactor axial overview", fontsize=17)
    axes[1].set_title("(b) Central control-rod detail", fontsize=17)
    for axis in axes:
        axis.set_xlabel("x (cm)", fontsize=15)
        axis.set_ylabel("z (cm)", fontsize=15)
        axis.tick_params(labelsize=12.5, width=1.3, length=6)
        for spine in axis.spines.values():
            spine.set_linewidth(1.4)
    axes[0].set_aspect("equal")
    # The detail panel deliberately exaggerates the horizontal scale so the
    # 1.9-12 mm concentric components remain distinguishable in a paper figure.
    axes[1].set_aspect("auto")
    axes[1].set_xlim(-0.65, 0.65)
    axes[1].set_xticks((-0.6, -0.3, 0.0, 0.3, 0.6))
    axial_lines = [
        (-11.5, "Fuel active bottom", "#8c2d04", "--"),
        (11.5, "Fuel active top", "#8c2d04", "--"),
        (-12.9, "Guide bottom", "#333333", ":"),
        (12.9, "Guide top", "#333333", ":"),
        (bounds["cadmium"]["z_bottom_cm"], "Cd bottom", "#1d2a78", "-."),
        (bounds["cadmium"]["z_top_cm"], "Cd top", "#1d2a78", "-."),
        (bounds["central_al_rod"]["z_bottom_cm"], "Al bottom", "#e69f00", ":"),
        (bounds["central_al_rod"]["z_top_cm"], "Al top", "#e69f00", ":"),
        (bounds["ss304_tube"]["z_bottom_cm"], "SS bottom", "#6c6c7c", "--"),
        (bounds["ss304_tube"]["z_top_cm"], "SS top", "#6c6c7c", "--"),
    ]
    for z_value, _label, color, style in axial_lines:
        axes[1].axhline(z_value, color=color, linestyle=style, linewidth=0.9, alpha=0.85)
    axes[1].text(
        0.03,
        0.02,
        "position = 0 cm: Cd centred at core z = 0\nExplicit modelling assumption, not an as-built claim",
        transform=axes[1].transAxes,
        fontsize=11.5,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.4"},
    )
    legend_handles = [
        Patch(facecolor="#e6b800", label="12.5% LEU UO2 active fuel"),
        Patch(facecolor="#b9bcc1", label="Zircaloy-4 fuel cladding"),
        Patch(facecolor="#e69f00", label="Al-303-1 / central Al rod"),
        Patch(facecolor="#555555", label="LT-21 guide tube"),
        Patch(facecolor="#1d2a78", label="Natural Cd absorber"),
        Patch(facecolor="#7a7a8a", label="SS-304 outer tube"),
        Patch(facecolor="#9bd3e5", label="Light water"),
        Patch(facecolor="#4b9b50", label="Side and bottom Be"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=11.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(
        "MNSR-SZ control-rod axial geometry validation (fully inserted reference)",
        fontsize=19,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.90, wspace=0.24)
    fig.savefig(PLOT_PNG_PATH, dpi=400, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(PLOT_PDF_PATH, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return {
        "overview_overlap_pixels": overview_overlap,
        "detail_overlap_pixels": detail_overlap,
        "overlap_pixel_check_passed": overview_overlap == 0 and detail_overlap == 0,
        "overview_plot_log": overview_log,
        "detail_plot_log": detail_log,
        "thermal_scattering_aliases": aliases,
        "png_file": str(PLOT_PNG_PATH),
        "pdf_file": str(PLOT_PDF_PATH),
    }


def validate_geometry(context: dict[str, Any]) -> tuple[dict[str, Any], openmc.Geometry]:
    builder = context["build_geometry_for_scheme"]
    withdrawn_geometry, withdrawn_metadata = builder(
        REFERENCE,
        control_rod_state="withdrawn",
        control_rod_position_cm=None,
        model_variant="iaea_reference",
        create_plots=False,
    )
    geometries: dict[float, openmc.Geometry] = {}
    metadata_by_position: dict[float, dict[str, Any]] = {}
    for position in POSITIONS_CM:
        geometry, metadata = builder(
            REFERENCE,
            control_rod_state="inserted",
            control_rod_position_cm=position,
            model_variant="iaea_reference",
            create_plots=False,
        )
        geometries[position] = geometry
        metadata_by_position[position] = metadata

    position_zero = metadata_by_position[0.0]
    bounds = position_zero["control_rod_axial_bounds"]
    expected_counts = {
        "physical_positions": 355,
        "ordinary_positions": 350,
        "fuel": 345,
        "dummy": 5,
        "tie_rod": 4,
        "central": 1,
    }
    assignments = _material_cell_map(geometries[0.0])
    expected_assignments = {
        "IAEA movable control rod inner aluminum": "Al-303-1",
        "IAEA movable cadmium absorber tube": "Natural Cadmium",
        "IAEA movable SS-304 control rod tube": "SS-304",
        "IAEA LT-21 control guide tube": "LT-21",
        "IAEA control rod corridor water complement": "Light Water",
    }
    axial_lengths_correct = all([
        math.isclose(bounds["cadmium"]["length_cm"], 26.6),
        math.isclose(bounds["central_al_rod"]["length_cm"], 29.0),
        math.isclose(bounds["ss304_tube"]["length_cm"], 45.0),
        math.isclose(position_zero["guide_tube_bounds"]["height_cm"], 25.8),
    ])
    translation_checks = []
    for position in POSITIONS_CM:
        current = metadata_by_position[position]["control_rod_axial_bounds"]
        translation_checks.extend([
            math.isclose(current["cadmium"]["z_bottom_cm"], -13.3 + position),
            math.isclose(current["cadmium"]["z_top_cm"], 13.3 + position),
            math.isclose(current["central_al_rod"]["z_bottom_cm"], -14.5 + position),
            math.isclose(current["central_al_rod"]["z_top_cm"], 14.5 + position),
            math.isclose(current["ss304_tube"]["z_bottom_cm"], -22.5 + position),
            math.isclose(current["ss304_tube"]["z_top_cm"], 22.5 + position),
        ])
    withdrawn_fills = {
        cell.fill for cell in withdrawn_geometry.get_all_cells().values()
    }
    inserted_fills = {
        cell.fill for cell in geometries[0.0].get_all_cells().values()
    }
    plot_check = build_axial_validation_plot(
        context, geometries[0.0], metadata_by_position[0.0]
    )
    checks = {
        "component_counts_unchanged_all_positions": all(
            item["counts"] == expected_counts for item in metadata_by_position.values()
        ),
        "dummy_positions_unchanged_all_positions": all(
            tuple(item["dummy_position_ids"]) == REFERENCE
            for item in metadata_by_position.values()
        ),
        "independent_axial_bounds_present": bounds is not None,
        "axial_lengths_correct": axial_lengths_correct,
        "position_zero_cd_symmetric_about_core_centre": (
            math.isclose(bounds["cadmium"]["z_bottom_cm"], -13.3)
            and math.isclose(bounds["cadmium"]["z_top_cm"], 13.3)
        ),
        "moving_components_translate_together": all(translation_checks),
        "guide_remains_fixed": all(
            item["guide_tube_bounds"] == position_zero["guide_tube_bounds"]
            for item in metadata_by_position.values()
        ),
        "material_assignments_correct": assignments == expected_assignments,
        "withdrawn_active_geometry_has_no_cadmium": (
            context["material_map"]["cadmium"] not in withdrawn_fills
        ),
        "inserted_active_geometry_has_cadmium": (
            context["material_map"]["cadmium"] in inserted_fills
        ),
        "strict_cd_outer_diameter_3p9_mm": math.isclose(
            20.0 * context["R_CADMIUM_OUTER"], 3.9
        ),
        "external_boundaries_unchanged": (
            withdrawn_metadata["vacuum_boundary_names"]
            == position_zero["vacuum_boundary_names"]
        ),
        "xz_overlap_pixel_check_passed": plot_check["overlap_pixel_check_passed"],
    }
    status = "geometry_ready_for_pilot" if all(checks.values()) else "geometry_failed"
    payload = {
        "status": status,
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "control_rod_axial_definition_version": CONTROL_ROD_DEFINITION_VERSION,
        "reference_dummy_ids": list(REFERENCE),
        "control_rod_axial_model_assumption": context[
            "CONTROL_ROD_AXIAL_MODEL_ASSUMPTION"
        ],
        "position_zero_bounds": bounds,
        "guide_tube_bounds": position_zero["guide_tube_bounds"],
        "position_bounds": {
            str(position): metadata_by_position[position]["control_rod_axial_bounds"]
            for position in POSITIONS_CM
        },
        "material_assignments": assignments,
        "checks": checks,
        "plot_overlap_check": plot_check,
        "openmc_geometry_debug_transport_check": "pending_position_0_pilot",
        "formal_final_runs_authorized": False,
        "pilot_runs_authorized": status == "geometry_ready_for_pilot",
    }
    _write_json(GEOMETRY_VALIDATION_PATH, payload)
    if status != "geometry_ready_for_pilot":
        raise RuntimeError("控制棒轴向几何检查未通过，禁止启动pilot。")
    return payload, geometries[0.0]


def _create_settings(seed: int) -> openmc.Settings:
    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = PARTICLES
    settings.batches = BATCHES
    settings.inactive = INACTIVE
    settings.seed = seed
    settings.temperature = {
        "method": "nearest",
        "tolerance": 10.0,
        "default": 293.15,
    }
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(
            (-10.9, -10.9, -11.5),
            (10.9, 10.9, 11.5),
        ),
        constraints={"fissionable": True},
    )
    entropy_mesh = openmc.RegularMesh(name="Pilot Shannon entropy mesh")
    entropy_mesh.lower_left = (-11.5, -11.5, -11.5)
    entropy_mesh.upper_right = (11.5, 11.5, 11.5)
    entropy_mesh.dimension = (10, 10, 1)
    settings.entropy_mesh = entropy_mesh
    settings.output = {"tallies": False}
    return settings


def _entropy_summary(entropy: np.ndarray) -> dict[str, Any]:
    tail = entropy[-10:]
    slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0])
    value_range = float(np.ptp(tail))
    return {
        "tail_count": 10,
        "first": float(tail[0]),
        "last": float(tail[-1]),
        "range": value_range,
        "linear_slope_per_batch": slope,
        "heuristically_stable": abs(slope) < 0.01 and value_range < 0.10,
    }


def _collect_warnings(text: str) -> list[str]:
    return sorted({
        line.strip()
        for line in text.splitlines()
        if "warning" in line.lower()
    })


def run_position(
    context: dict[str, Any],
    position_cm: float,
    run_index: int,
) -> dict[str, Any]:
    run_directory = PILOT_ROOT / f"position_{str(position_cm).replace('.', 'p')}_cm"
    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(f"pilot目录非空，拒绝覆盖：{run_directory}")
    run_directory.mkdir(parents=True, exist_ok=True)
    seed = _stable_seed(position_cm)
    geometry, metadata = context["build_geometry_for_scheme"](
        REFERENCE,
        control_rod_state="inserted",
        control_rod_position_cm=position_cm,
        model_variant="iaea_reference",
        create_plots=False,
    )
    materials, aliases = _prepare_materials(context, geometry)
    settings = _create_settings(seed)
    model = openmc.Model(
        materials=materials,
        geometry=geometry,
        settings=settings,
        tallies=openmc.Tallies(),
    )
    identity = {
        "scheme": list(REFERENCE),
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "control_rod_axial_definition_version": CONTROL_ROD_DEFINITION_VERSION,
        "control_rod_state": "inserted",
        "control_rod_position_cm": position_cm,
        "profile": "screening_pilot",
        "particles": PARTICLES,
        "batches": BATCHES,
        "inactive": INACTIVE,
        "seed": seed,
        "run_index": run_index,
    }
    _write_json(run_directory / "run_manifest.json", {"identity": identity, "status": "prepared"})
    model.export_to_xml(directory=run_directory)
    buffer = io.StringIO()
    started = time.perf_counter()
    try:
        with (
            contextlib.redirect_stdout(buffer),
            contextlib.redirect_stderr(buffer),
            patch.object(
                openmc.Model,
                "is_initialized",
                new_callable=PropertyMock,
                return_value=False,
            ),
        ):
            statepoint_path = model.run(
                cwd=run_directory,
                openmc_exec=_find_openmc_executable(),
                geometry_debug=position_cm == 0.0,
                export_model_xml=False,
            )
        output_text = buffer.getvalue()
        (run_directory / "openmc_output.txt").write_text(output_text, encoding="utf-8")
        statepoint_path = Path(statepoint_path)
        if not statepoint_path.is_absolute():
            statepoint_path = run_directory / statepoint_path
        statepoint_path = statepoint_path.resolve()
        if not statepoint_path.is_file():
            raise FileNotFoundError(f"缺少pilot statepoint：{statepoint_path}")
        with openmc.StatePoint(statepoint_path) as statepoint:
            keff = statepoint.keff
            entropy = np.asarray(statepoint.entropy, dtype=float).reshape(-1)
            generations = np.asarray(statepoint.k_generation, dtype=float).reshape(-1)
        if entropy.size != BATCHES or generations.size != BATCHES:
            raise ValueError("pilot逐批keff或Shannon熵数量不是60。")
        keff_mean = float(keff.nominal_value)
        keff_std = float(keff.std_dev)
        rho = (keff_mean - 1.0) / keff_mean
        rho_std = keff_std / keff_mean**2
        phase = np.where(np.arange(1, BATCHES + 1) <= INACTIVE, "inactive", "active")
        pd.DataFrame({
            "batch": np.arange(1, BATCHES + 1),
            "phase": phase,
            "k_generation": generations,
            "shannon_entropy": entropy,
        }).to_csv(run_directory / "history.csv", index=False, encoding="utf-8")
        overlap_summary_match = re.search(
            r"CELL OVERLAP CHECK SUMMARY", output_text, flags=re.IGNORECASE
        )
        overlap_error_match = re.search(
            r"overlapping cells detected|overlap check found|cell overlaps detected",
            output_text,
            flags=re.IGNORECASE,
        )
        result = {
            "identity": identity,
            "status": "completed",
            "keff_mean": keff_mean,
            "keff_std": keff_std,
            "reactivity": rho,
            "reactivity_std": rho_std,
            "reactivity_mk": rho * 1000.0,
            "reactivity_std_mk": rho_std * 1000.0,
            "entropy": _entropy_summary(entropy),
            "geometry_debug_enabled": position_cm == 0.0,
            "geometry_debug_no_overlap_message": bool(overlap_summary_match),
            "geometry_debug_overlap_error_message": bool(overlap_error_match),
            "warnings": _collect_warnings(output_text),
            "control_rod_axial_bounds": metadata["control_rod_axial_bounds"],
            "thermal_scattering_aliases": aliases,
            "statepoint_file": str(statepoint_path),
            "openmc_output_file": str(run_directory / "openmc_output.txt"),
            "runtime_seconds": float(time.perf_counter() - started),
        }
        _write_json(run_directory / "result.json", result)
        _write_json(
            run_directory / "run_manifest.json",
            {"identity": identity, "status": "completed", "result_file": str(run_directory / "result.json")},
        )
        return result
    except Exception as error:
        (run_directory / "openmc_output.txt").write_text(buffer.getvalue(), encoding="utf-8")
        failure = {
            "identity": identity,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "runtime_seconds": float(time.perf_counter() - started),
        }
        _write_json(run_directory / "error.json", failure)
        _write_json(run_directory / "run_manifest.json", failure)
        raise


def summarize_pilot(
    results: list[dict[str, Any]],
    geometry_validation: dict[str, Any],
) -> dict[str, Any]:
    if len(results) != 3:
        raise ValueError("pilot必须恰好包含3个位置。")
    rows = []
    rho0 = results[0]["reactivity"]
    rho0_std = results[0]["reactivity_std"]
    for result in results:
        delta_rho = result["reactivity"] - rho0
        delta_std = math.sqrt(result["reactivity_std"] ** 2 + rho0_std**2)
        rows.append({
            "control_rod_position_cm": result["identity"]["control_rod_position_cm"],
            "seed": result["identity"]["seed"],
            "keff_mean": result["keff_mean"],
            "keff_std": result["keff_std"],
            "reactivity": result["reactivity"],
            "reactivity_std": result["reactivity_std"],
            "reactivity_mk": result["reactivity_mk"],
            "reactivity_std_mk": result["reactivity_std_mk"],
            "withdrawal_worth_from_position_0_mk": delta_rho * 1000.0,
            "withdrawal_worth_std_mk": delta_std * 1000.0,
            "entropy_stable": result["entropy"]["heuristically_stable"],
            "geometry_debug_enabled": result["geometry_debug_enabled"],
            "geometry_debug_no_overlap_message": result["geometry_debug_no_overlap_message"],
            "runtime_seconds": result["runtime_seconds"],
            "status": result["status"],
        })
    table = pd.DataFrame(rows).sort_values("control_rod_position_cm").reset_index(drop=True)
    table.to_csv(PILOT_CSV_PATH, index=False, encoding="utf-8")
    keff_values = table["keff_mean"].to_numpy(float)
    rho_values = table["reactivity"].to_numpy(float)
    total_worth_mk = float(table.iloc[-1]["withdrawal_worth_from_position_0_mk"])
    total_worth_std_mk = float(table.iloc[-1]["withdrawal_worth_std_mk"])
    mid_worth_mk = float(table.iloc[1]["withdrawal_worth_from_position_0_mk"])
    mid_worth_std_mk = float(table.iloc[1]["withdrawal_worth_std_mk"])
    monotonic_keff = bool(np.all(np.diff(keff_values) > 0.0))
    monotonic_rho = bool(np.all(np.diff(rho_values) > 0.0))
    mid_curve_fraction = mid_worth_mk / total_worth_mk if total_worth_mk > 0.0 else math.nan
    middle_position_is_curve_middle = 0.25 <= mid_curve_fraction <= 0.75
    first_step_direction_resolved = mid_worth_mk > mid_worth_std_mk
    same_order = 0.5 <= total_worth_mk / IAEA_LEU_TOTAL_ROD_WORTH_MK <= 2.0
    geometry_debug_passed = bool(
        results[0]["geometry_debug_enabled"]
        and results[0]["geometry_debug_no_overlap_message"]
        and not results[0]["geometry_debug_overlap_error_message"]
    )
    checks = {
        "exactly_three_pilot_runs": len(results) == MAX_NEW_PILOT_OPENMC_RUNS,
        "all_runs_completed": all(result["status"] == "completed" for result in results),
        "all_keff_finite": bool(np.all(np.isfinite(keff_values))),
        "all_entropy_heuristically_stable": all(
            result["entropy"]["heuristically_stable"] for result in results
        ),
        "keff_increases_with_withdrawal": monotonic_keff,
        "reactivity_increases_with_withdrawal": monotonic_rho,
        "middle_position_lies_between_endpoints": 0.0 < mid_curve_fraction < 1.0,
        "middle_position_is_curve_middle_by_point_estimate": middle_position_is_curve_middle,
        "first_withdrawal_step_direction_resolved_above_1sigma": first_step_direction_resolved,
        "position_0_geometry_debug_no_overlaps": geometry_debug_passed,
        "xz_overlap_pixel_check_passed": geometry_validation["checks"][
            "xz_overlap_pixel_check_passed"
        ],
        "total_worth_same_order_as_5p75_mk": same_order,
    }
    ready = all(checks.values())
    all_warnings = sorted({warning for result in results for warning in result["warnings"]})
    payload = {
        "status": "ready_for_formal_inserted" if ready else "not_ready_for_formal_inserted",
        "formal_inserted_final_runs_started": 0,
        "pilot_openmc_runs": len(results),
        "profile": "screening_pilot",
        "particles": PARTICLES,
        "batches": BATCHES,
        "inactive": INACTIVE,
        "reference_dummy_ids": list(REFERENCE),
        "positions_cm": list(POSITIONS_CM),
        "checks": checks,
        "position_results": rows,
        "total_withdrawal_worth_0_to_24_mk": total_worth_mk,
        "total_withdrawal_worth_std_mk": total_worth_std_mk,
        "middle_position_worth_mk": mid_worth_mk,
        "middle_position_worth_std_mk": mid_worth_std_mk,
        "middle_position_fraction_of_total": mid_curve_fraction,
        "literature_leu_total_control_rod_worth_mk": IAEA_LEU_TOTAL_ROD_WORTH_MK,
        "pilot_to_literature_worth_ratio": total_worth_mk / IAEA_LEU_TOTAL_ROD_WORTH_MK,
        "literature_comparison": (
            "same_order_of_magnitude" if same_order else "different_order_requires_model_discussion"
        ),
        "interpretation_limits": [
            "Screening statistics are only a small-scale physical-direction check.",
            "The 5.75 mk IAEA value was not used to tune geometry or materials.",
            "The position-0 absolute z placement is an explicit symmetric modelling assumption.",
        ],
        "openmc_warnings": all_warnings,
        "runtime_seconds_total": float(sum(result["runtime_seconds"] for result in results)),
        "geometry_validation_file": str(GEOMETRY_VALIDATION_PATH),
        "model_definition_file": str(MODEL_DEFINITION_PATH),
        "pilot_csv_file": str(PILOT_CSV_PATH),
        "plot_png_file": str(PLOT_PNG_PATH),
        "plot_pdf_file": str(PLOT_PDF_PATH),
        "ga_runs": 0,
        "formal_final_runs": 0,
        "materials_modified": False,
        "dummy_optimization_modified": False,
        "other_core_geometry_modified": False,
    }
    _write_json(PILOT_SUMMARY_PATH, payload)
    geometry_validation["openmc_geometry_debug_transport_check"] = {
        "enabled_for_position_cm": 0.0,
        "passed": geometry_debug_passed,
        "log_file": results[0]["openmc_output_file"],
    }
    geometry_validation["pilot_result_status"] = payload["status"]
    geometry_validation["formal_final_runs_authorized"] = ready
    _write_json(GEOMETRY_VALIDATION_PATH, geometry_validation)
    return payload


def load_existing_pilot_results() -> list[dict[str, Any]]:
    """Reload the three completed runs and refresh log-derived overlap flags."""
    results = []
    for position in POSITIONS_CM:
        run_directory = PILOT_ROOT / f"position_{str(position).replace('.', 'p')}_cm"
        result_path = run_directory / "result.json"
        output_path = run_directory / "openmc_output.txt"
        if not result_path.is_file() or not output_path.is_file():
            raise FileNotFoundError(f"缺少已完成pilot结果：{run_directory}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            raise RuntimeError(f"已有pilot并非completed：{result_path}")
        output_text = output_path.read_text(encoding="utf-8")
        if position == 0.0:
            result["geometry_debug_no_overlap_message"] = bool(re.search(
                r"CELL OVERLAP CHECK SUMMARY", output_text, flags=re.IGNORECASE
            ))
            result["geometry_debug_overlap_error_message"] = bool(re.search(
                r"overlapping cells detected|overlap check found|cell overlaps detected",
                output_text,
                flags=re.IGNORECASE,
            ))
            _write_json(result_path, result)
        results.append(result)
    return results


def main(run_pilot: bool, summarize_existing: bool = False) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    context = load_model_context()
    model_definition = write_model_definition(context)
    geometry_validation, _geometry = validate_geometry(context)
    if summarize_existing:
        return summarize_pilot(load_existing_pilot_results(), geometry_validation)
    if not run_pilot:
        return {
            "status": "geometry_ready_for_pilot",
            "model_definition": model_definition,
            "geometry_validation": geometry_validation,
            "pilot_openmc_runs": 0,
            "formal_final_runs": 0,
        }
    existing_files = [path for path in PILOT_ROOT.glob("position_*_cm/*") if path.is_file()]
    if existing_files:
        raise FileExistsError("pilot输出目录已有文件，拒绝覆盖或读取为本轮新计算。")
    results = [
        run_position(context, position, index)
        for index, position in enumerate(POSITIONS_CM)
    ]
    if len(results) > MAX_NEW_PILOT_OPENMC_RUNS:
        raise RuntimeError("pilot运行超过3次硬上限。")
    return summarize_pilot(results, geometry_validation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-pilot",
        action="store_true",
        help="Run exactly three REFERENCE screening pilot positions.",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Refresh summaries from the existing three pilot runs without transport.",
    )
    arguments = parser.parse_args()
    if arguments.run_pilot and arguments.summarize_existing:
        parser.error("--run-pilot和--summarize-existing不能同时使用。")
    print(json.dumps(
        main(arguments.run_pilot, arguments.summarize_existing),
        ensure_ascii=False,
        indent=2,
    ))
