"""Control-rod validation and pooled withdrawn-flux comparison workflow.

The inserted transport branch is deliberately blocked when the current
geometry does not provide a paper-consistent, explicit 26.6 cm Cd axial
position.  The independent withdrawn flux branch can still run eight final
replicates.  No genetic-algorithm or energy-spectrum code is present here.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, cast
import warnings
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent
_MPL_CONFIG = PROJECT_ROOT / "build/.matplotlib"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
import numpy as np
import openmc
import pandas as pd


REFERENCE = (285, 298, 311, 324, 337)
OPTIMIZED = (200, 213, 223, 239, 256)
SCHEMES = {"REFERENCE": REFERENCE, "OPTIMIZED": OPTIMIZED}

GEOMETRY_MODEL_VERSION = "iaea_reference_v1"
MODEL_VARIANT = "iaea_reference"
TOP_BE_THICKNESS_CM = 0.0
TEMPERATURE_K = 293.15

PARTICLES = 20_000
BATCHES = 150
INACTIVE = 50
ACTIVE_BATCHES = BATCHES - INACTIVE
FLUX_BASE_SEED = 20260818
MAX_NEW_INSERTED_OPENMC_RUNS = 8
MAX_NEW_FLUX_OPENMC_RUNS = 8
MAX_TOTAL_NEW_OPENMC_RUNS = 16

FLUX_PROFILE = "withdrawn_flux_final"
FLUX_TALLY_VERSION = "core_flux_mesh_xy_100x100x1_v1"
FLUX_LOWER_LEFT = (-12.0, -12.0, -11.5)
FLUX_UPPER_RIGHT = (12.0, 12.0, 11.5)
FLUX_DIMENSION = (100, 100, 1)
CORE_RADIUS_CM = 11.5

CONTROL_ROOT = (PROJECT_ROOT / "build/results/control_rod_analysis").resolve()
FLUX_ROOT = (PROJECT_ROOT / "build/results/flux_comparison").resolve()
FLUX_FIGURE_ROOT = FLUX_ROOT / "figures"
CONFIRMATION_ROOT = (
    PROJECT_ROOT / "build/optimization/real_ga/statistical_confirmation"
).resolve()
for _directory in (CONTROL_ROOT, FLUX_ROOT, FLUX_FIGURE_ROOT):
    _directory.mkdir(parents=True, exist_ok=True)

RUNTIME_COUNTER = {
    "new_inserted_openmc_runs": 0,
    "new_flux_openmc_runs": 0,
    "total_new_openmc_runs": 0,
    "failures": 0,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _execute_notebook_cells(
    path: Path,
    cell_indices: list[int],
    namespace: dict[str, Any],
) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        cell = notebook["cells"][cell_index]
        if cell.get("cell_type") != "code":
            raise TypeError(f"{path.name}单元{cell_index}不是代码单元。")
        source = "".join(cell.get("source", []))
        exec(
            compile(source, f"{path.name}:cell_{cell_index}", "exec"),
            namespace,
        )


def load_model_context() -> dict[str, Any]:
    """Load only material/geometry definition cells; skip plots and exports."""
    namespace: dict[str, Any] = {
        "__name__": "mnsr_model_context",
        "__file__": str(PROJECT_ROOT / "MNSR_geometry_development.ipynb"),
    }
    material_cells = [1, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    geometry_cells = [2, 5, 6, 8, 10, 12, 14, 16, 18, 30, 52]
    _execute_notebook_cells(
        PROJECT_ROOT / "MNSR_materials_development.ipynb",
        material_cells,
        namespace,
    )
    _execute_notebook_cells(
        PROJECT_ROOT / "MNSR_geometry_development.ipynb",
        geometry_cells,
        namespace,
    )
    required = {
        "materials",
        "material_map",
        "ordinary_positions_df",
        "build_geometry_for_scheme",
    }
    missing = sorted(required.difference(namespace))
    if missing:
        raise RuntimeError(f"模型上下文缺少对象：{missing}")
    return namespace


def validate_control_rod_geometry(context: dict[str, Any]) -> dict[str, Any]:
    """Record the actual inserted geometry and enforce the axial stop rule."""
    builder = context["build_geometry_for_scheme"]
    material_map = context["material_map"]
    withdrawn_geometry, withdrawn_metadata = builder(
        REFERENCE,
        control_rod_state="withdrawn",
        export_directory=None,
        create_plots=False,
        model_variant=MODEL_VARIANT,
        top_be_thickness_cm=TOP_BE_THICKNESS_CM,
    )
    inserted_geometry, inserted_metadata = builder(
        REFERENCE,
        control_rod_state="inserted",
        export_directory=None,
        create_plots=False,
        model_variant=MODEL_VARIANT,
        top_be_thickness_cm=TOP_BE_THICKNESS_CM,
    )
    optimized_inserted_geometry, optimized_inserted_metadata = builder(
        OPTIMIZED,
        control_rod_state="inserted",
        export_directory=None,
        create_plots=False,
        model_variant=MODEL_VARIANT,
        top_be_thickness_cm=TOP_BE_THICKNESS_CM,
    )

    guide_bottom = float(context["Z_GUIDE_BOTTOM"])
    guide_top = float(context["Z_GUIDE_TOP"])
    actual_control_length = guide_top - guide_bottom
    radial = {
        "guide_tube": {
            "inner_radius_cm": float(context["R_GUIDE_INNER"]),
            "outer_radius_cm": float(context["R_GUIDE_OUTER"]),
            "inner_diameter_mm": float(2.0 * context["R_GUIDE_INNER"] * 10.0),
            "outer_diameter_mm": float(2.0 * context["R_GUIDE_OUTER"] * 10.0),
        },
        "central_al_rod": {
            "outer_radius_cm": float(context["R_CONTROL_INNER_AL"]),
            "outer_diameter_mm": float(2.0 * context["R_CONTROL_INNER_AL"] * 10.0),
        },
        "cadmium_absorber": {
            "inner_radius_cm": float(context["R_CONTROL_INNER_AL"]),
            "outer_radius_cm": float(context["R_CADMIUM_OUTER"]),
            "inner_diameter_mm": float(2.0 * context["R_CONTROL_INNER_AL"] * 10.0),
            "outer_diameter_mm": float(2.0 * context["R_CADMIUM_OUTER"] * 10.0),
        },
        "ss304_outer_tube": {
            "inner_radius_cm": float(context["R_SS304_INNER"]),
            "outer_radius_cm": float(context["R_SS304_OUTER"]),
            "outer_diameter_mm": float(2.0 * context["R_SS304_OUTER"] * 10.0),
            "wall_thickness_mm": float(
                (context["R_SS304_OUTER"] - context["R_SS304_INNER"]) * 10.0
            ),
        },
    }
    axial = {
        "guide_tube": {
            "z_bottom_cm": guide_bottom,
            "z_top_cm": guide_top,
            "length_cm": actual_control_length,
            "paper_length_cm": 25.8,
        },
        "cadmium_absorber_actual": {
            "z_bottom_cm": guide_bottom,
            "z_top_cm": guide_top,
            "length_cm": actual_control_length,
            "paper_length_cm": 26.6,
            "independent_axial_surfaces_present": False,
        },
        "central_al_rod_actual": {
            "z_bottom_cm": guide_bottom,
            "z_top_cm": guide_top,
            "length_cm": actual_control_length,
        },
        "ss304_outer_tube_actual": {
            "z_bottom_cm": guide_bottom,
            "z_top_cm": guide_top,
            "length_cm": actual_control_length,
        },
        "fully_inserted_reference_position_explicitly_defined": False,
    }

    withdrawn_cells = list(withdrawn_geometry.get_all_cells().values())
    inserted_cells = list(inserted_geometry.get_all_cells().values())
    inserted_control_names = {
        cell.name: getattr(cell.fill, "name", None)
        for cell in inserted_cells
        if cell.name.startswith("IAEA control")
        or "cadmium" in cell.name.lower()
        or "SS-304 control" in cell.name
        or "LT-21 control guide" in cell.name
    }
    withdrawn_has_cd = any(cell.fill is material_map["cadmium"] for cell in withdrawn_cells)
    inserted_has_cd = any(cell.fill is material_map["cadmium"] for cell in inserted_cells)
    inserted_has_ss = any(
        cell.fill is material_map["control_rod_clad"] for cell in inserted_cells
    )
    inserted_has_withdrawn_water = any(
        cell.name == "IAEA withdrawn guide internal water" for cell in inserted_cells
    )
    expected_counts = {
        "physical_positions": 355,
        "ordinary_positions": 350,
        "fuel": 345,
        "dummy": 5,
        "tie_rod": 4,
        "central": 1,
    }
    counts_ok = (
        withdrawn_metadata["counts"] == expected_counts
        and inserted_metadata["counts"] == expected_counts
        and optimized_inserted_metadata["counts"] == expected_counts
    )
    boundaries_unchanged = (
        withdrawn_metadata["vacuum_boundary_names"]
        == inserted_metadata["vacuum_boundary_names"]
        == optimized_inserted_metadata["vacuum_boundary_names"]
        and np.allclose(
            withdrawn_geometry.bounding_box.lower_left,
            inserted_geometry.bounding_box.lower_left,
        )
        and np.allclose(
            withdrawn_geometry.bounding_box.upper_right,
            inserted_geometry.bounding_box.upper_right,
        )
    )
    dummy_positions_unchanged = (
        tuple(withdrawn_metadata["dummy_position_ids"])
        == tuple(inserted_metadata["dummy_position_ids"])
        == REFERENCE
        and tuple(optimized_inserted_metadata["dummy_position_ids"]) == OPTIMIZED
    )
    central_radial_partition_check = bool(
        radial["central_al_rod"]["outer_radius_cm"]
        < radial["cadmium_absorber"]["outer_radius_cm"]
        < radial["ss304_outer_tube"]["inner_radius_cm"]
        < radial["ss304_outer_tube"]["outer_radius_cm"]
        < radial["guide_tube"]["inner_radius_cm"]
        < radial["guide_tube"]["outer_radius_cm"]
        and not inserted_has_withdrawn_water
    )
    failure_reasons = []
    if not math.isclose(actual_control_length, 26.6, abs_tol=1.0e-12):
        failure_reasons.append(
            "当前Cd absorber与guide_axial_region共用-12.9至+12.9 cm，实际长度25.8 cm，"
            "未实现论文给出的26.6 cm Cd吸收段。"
        )
    failure_reasons.append(
        "当前inserted实现没有独立Cd/Al/SS轴向平面或明确的fully inserted轴向定位；"
        "无法在不猜测插入深度的前提下确认论文3.3工况。"
    )
    status = "failed" if failure_reasons else "passed"
    payload = {
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "model_variant": MODEL_VARIANT,
        "temperature_k": TEMPERATURE_K,
        "reference_dummy_ids": list(REFERENCE),
        "optimized_dummy_ids": list(OPTIMIZED),
        "radial_dimensions": radial,
        "axial_dimensions": axial,
        "materials": {
            "central_al_rod": material_map["control_rod_inner"].name,
            "cadmium_absorber": material_map["cadmium"].name,
            "ss304_outer_tube": material_map["control_rod_clad"].name,
            "guide_tube": material_map["guide_tube"].name,
            "water_gaps": material_map["water"].name,
        },
        "checks": {
            "strict_3p9_mm_cd_outer_diameter": math.isclose(
                radial["cadmium_absorber"]["outer_diameter_mm"], 3.9
            ),
            "guide_dimensions_match_9_12_258_mm": bool(
                math.isclose(radial["guide_tube"]["inner_diameter_mm"], 9.0)
                and math.isclose(radial["guide_tube"]["outer_diameter_mm"], 12.0)
                and math.isclose(actual_control_length, 25.8)
            ),
            "ss_dimensions_match_5_mm_od_0p5_mm_wall": bool(
                math.isclose(radial["ss304_outer_tube"]["outer_diameter_mm"], 5.0)
                and math.isclose(radial["ss304_outer_tube"]["wall_thickness_mm"], 0.5)
            ),
            "withdrawn_cd_absent": not withdrawn_has_cd,
            "inserted_cd_present": inserted_has_cd,
            "inserted_ss304_present": inserted_has_ss,
            "inserted_withdrawn_internal_water_absent": not inserted_has_withdrawn_water,
            "central_radial_partition_analytic_check": central_radial_partition_check,
            "component_counts_unchanged": counts_ok,
            "dummy_positions_unchanged": dummy_positions_unchanged,
            "external_boundaries_unchanged": boundaries_unchanged,
            "openmc_geometry_overlap_plot_check": (
                "not_run_due_to_prior_axial_definition_failure"
            ),
        },
        "active_inserted_control_cells": inserted_control_names,
        "failure_reasons": failure_reasons,
        "inserted_geometry_status": status,
        "formal_inserted_transport_authorized": status == "passed",
        "inserted_openmc_runs": 0,
    }
    _write_json(CONTROL_ROOT / "control_rod_geometry_validation.json", payload)
    return payload


def _settings_xml_values(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    return {
        "particles": int(root.findtext("particles", "-1")),
        "batches": int(root.findtext("batches", "-1")),
        "inactive": int(root.findtext("inactive", "-1")),
        "seed": int(root.findtext("seed", "-1")),
    }


def audit_withdrawn_confirmation() -> dict[str, Any]:
    """Audit and combine existing R0-R3 withdrawn keff/reactivity results."""
    table_path = CONFIRMATION_ROOT / "confirmation_replicates.csv"
    if not table_path.is_file():
        raise FileNotFoundError(f"找不到statistical confirmation结果：{table_path}")
    table = pd.read_csv(table_path)
    summaries: dict[str, Any] = {}
    audit_rows = []
    for label, scheme_label in (("REFERENCE", "REFERENCE"), ("OPTIMIZED", "C1")):
        rows = table.loc[table["scheme_label"] == scheme_label].sort_values("replicate")
        if list(rows["replicate"].astype(int)) != [0, 1, 2, 3]:
            raise RuntimeError(f"{label} withdrawn R0-R3不完整。")
        for _, row in rows.iterrows():
            run_directory = Path(row["run_directory"]).resolve()
            settings = _settings_xml_values(run_directory / "settings.xml")
            if (settings["particles"], settings["batches"], settings["inactive"]) != (
                PARTICLES,
                BATCHES,
                INACTIVE,
            ):
                raise RuntimeError(f"{label} R{int(row['replicate'])}统计配置不一致。")
            statepoint = Path(row["statepoint_file"]).resolve()
            if not statepoint.is_file():
                raise FileNotFoundError(f"缺少withdrawn statepoint：{statepoint}")
            if int(row["replicate"]) > 0:
                manifest = json.loads(
                    (run_directory / "run_manifest.json").read_text(encoding="utf-8")
                )["identity"]
                if (
                    manifest["geometry_model_version"] != GEOMETRY_MODEL_VERSION
                    or manifest["model_variant"] != MODEL_VARIANT
                    or manifest["control_rod_state"] != "withdrawn"
                ):
                    raise RuntimeError(f"{label} R{int(row['replicate'])}缓存身份不一致。")
            audit_rows.append(
                {
                    "scheme_label": label,
                    "scheme": json.dumps(list(SCHEMES[label])),
                    "replicate": int(row["replicate"]),
                    "seed": settings["seed"],
                    "keff": float(row["keff"]),
                    "keff_std": float(row["keff_std"]),
                    "rho": float(row["rho"]),
                    "rho_std": float(row["rho_std"]),
                    "rho_mk": float(row["rho_mk"]),
                    "run_directory": str(run_directory),
                    "statepoint_file": str(statepoint),
                    "status": "validated_existing_withdrawn",
                }
            )
        keff = rows["keff"].to_numpy(float)
        keff_std = rows["keff_std"].to_numpy(float)
        rho = rows["rho"].to_numpy(float)
        rho_std = rows["rho_std"].to_numpy(float)
        summaries[label] = {
            "scheme": list(SCHEMES[label]),
            "replicate_count": 4,
            "keff_combined": float(np.mean(keff)),
            "keff_uncertainty": float(np.sqrt(np.sum(keff_std**2)) / 4.0),
            "keff_replicate_mean": float(np.mean(keff)),
            "keff_replicate_sample_std": float(np.std(keff, ddof=1)),
            "keff_min": float(np.min(keff)),
            "keff_max": float(np.max(keff)),
            "rho_combined": float(np.mean(rho)),
            "rho_uncertainty": float(np.sqrt(np.sum(rho_std**2)) / 4.0),
            "rho_mk_combined": float(np.mean(rho) * 1000.0),
            "rho_mk_uncertainty": float(np.sqrt(np.sum(rho_std**2)) / 4.0 * 1000.0),
            "aggregation": "equal-weight mean of four final replicates; independent uncertainty sqrt(sum sigma_r^2)/4",
            "control_rod_state": "withdrawn",
            "geometry_model_version": GEOMETRY_MODEL_VERSION,
        }
    audit_path = CONTROL_ROOT / "withdrawn_confirmation_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False, encoding="utf-8")
    return {"schemes": summaries, "audit_file": str(audit_path)}


def write_blocked_control_outputs(
    geometry_validation: dict[str, Any],
    withdrawn_audit: dict[str, Any],
) -> dict[str, Any]:
    warning_payload = {
        "status": "not_run_geometry_validation_failed",
        "inserted_openmc_runs": 0,
        "warnings": [],
        "note": "未启动inserted输运，因此没有可报告的Cd核数据运行警告。",
    }
    warning_path = CONTROL_ROOT / "control_rod_nuclear_data_warnings.json"
    _write_json(warning_path, warning_payload)
    summary = {
        "task": "thesis_5_4_control_rod_analysis",
        "analysis_status": "blocked_geometry_validation_failed",
        "geometry_validation_status": geometry_validation["inserted_geometry_status"],
        "geometry_validation_file": str(
            CONTROL_ROOT / "control_rod_geometry_validation.json"
        ),
        "nuclear_data_warning_file": str(warning_path),
        "withdrawn_confirmation": withdrawn_audit,
        "inserted_openmc_runs": 0,
        "max_new_inserted_openmc_runs": MAX_NEW_INSERTED_OPENMC_RUNS,
        "control_rod_comparison_file": None,
        "table_5_5_file": None,
        "not_generated_reason": (
            "当前inserted模型没有论文一致的26.6 cm Cd独立轴向定义；"
            "按正式停止条件不得生成或伪造CRW/SDM数值表。"
        ),
        "physics_model_modified": False,
        "ga_runs": 0,
        "energy_spectrum_runs": 0,
    }
    summary_path = CONTROL_ROOT / "control_rod_analysis_summary.json"
    _write_json(summary_path, summary)
    return summary


def _find_openmc_executable() -> Path:
    candidates = [
        Path(sys.executable).with_name("openmc"),
        Path(sys.prefix) / "envs/openmc-env/bin/openmc",
    ]
    executable = next((item.resolve() for item in candidates if item.is_file()), None)
    if executable is None:
        raise FileNotFoundError("找不到OpenMC可执行程序。")
    return executable


def _prepare_materials(
    context: dict[str, Any],
    geometry: openmc.Geometry,
) -> tuple[openmc.Materials, dict[str, str]]:
    cross_sections = Path(openmc.config["cross_sections"]).expanduser().resolve()
    if not cross_sections.is_file():
        raise FileNotFoundError(f"截面库不存在：{cross_sections}")
    root = ET.parse(cross_sections).getroot()
    thermal_names = {
        library.get("materials")
        for library in root.findall("library")
        if library.get("type") == "thermal"
    }
    materials = context["materials"]
    material_map = context["material_map"]
    original = material_map["beryllium"]
    be_cells = [
        cell for cell in geometry.get_all_cells().values() if cell.fill is original
    ]
    if len(be_cells) != 2:
        raise RuntimeError(f"无顶部铍时应有2个Be cell，实际{len(be_cells)}。")
    if "c_Be_in_Be" in thermal_names:
        prepared = openmc.Materials(list(materials))
        prepared.cross_sections = str(cross_sections)
        return prepared, {}
    if "c_Be" not in thermal_names:
        raise FileNotFoundError("截面库不含c_Be_in_Be或兼容c_Be。")
    replacement = openmc.Material(name=original.name)
    for nuclide in original.nuclides:
        replacement.add_nuclide(nuclide.name, nuclide.percent, nuclide.percent_type)
    replacement.set_density(original.density_units, original.density)
    if original.temperature is not None:
        replacement.temperature = cast(float, original.temperature)
    replacement.add_s_alpha_beta("c_Be")
    for cell in be_cells:
        cell.fill = replacement
    prepared = openmc.Materials(
        [replacement if material is original else material for material in materials]
    )
    prepared.cross_sections = str(cross_sections)
    return prepared, {"c_Be_in_Be": "c_Be"}


def get_flux_seed(scheme: tuple[int, ...], replicate: int) -> int:
    payload = {
        "base_seed": FLUX_BASE_SEED,
        "scheme": list(scheme),
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "control_rod_state": "withdrawn",
        "profile": FLUX_PROFILE,
        "replicate": int(replicate),
        "tally_configuration": FLUX_TALLY_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return int(digest, 16) % 2_147_483_646 + 1


def _create_flux_settings(seed: int) -> openmc.Settings:
    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = PARTICLES
    settings.batches = BATCHES
    settings.inactive = INACTIVE
    settings.seed = int(seed)
    source_box = openmc.stats.Box(
        (-11.25, -11.25, -11.499999),
        (11.25, 11.25, 11.499999),
    )
    try:
        settings.source = openmc.IndependentSource(
            space=source_box,
            constraints={"fissionable": True},
        )
    except TypeError:
        settings.source = openmc.IndependentSource(space=source_box)
        settings.source.constraints = {"fissionable": True}
    settings.temperature = {"method": "nearest", "tolerance": 100.0}
    entropy_mesh = openmc.RegularMesh(name="Flux comparison entropy mesh")
    entropy_mesh.lower_left = (-11.5, -11.5, -11.5)
    entropy_mesh.upper_right = (11.5, 11.5, 11.5)
    entropy_mesh.dimension = (10, 10, 10)
    settings.entropy_mesh = entropy_mesh
    return settings


def _create_flux_tally() -> tuple[openmc.Tallies, openmc.RegularMesh]:
    mesh = openmc.RegularMesh(name="Core XY flux mesh")
    mesh.lower_left = FLUX_LOWER_LEFT
    mesh.upper_right = FLUX_UPPER_RIGHT
    mesh.dimension = FLUX_DIMENSION
    tally = openmc.Tally(name="core_flux_mesh_xy")
    tally.filters = [openmc.MeshFilter(mesh)]
    tally.scores = ["flux"]
    return openmc.Tallies([tally]), mesh


def _entropy_summary(values: np.ndarray) -> dict[str, Any]:
    active = np.asarray(values, dtype=float)[INACTIVE:]
    tail = active[-20:]
    slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0])
    return {
        "tail_batch_count": int(tail.size),
        "first_value": float(tail[0]),
        "last_value": float(tail[-1]),
        "absolute_first_to_last_change": float(abs(tail[-1] - tail[0])),
        "range": float(np.ptp(tail)),
        "linear_slope_per_batch": slope,
        "heuristically_stable": bool(np.ptp(tail) <= 0.10 and abs(slope) <= 0.005),
    }


def _flux_identity(
    label: str,
    scheme: tuple[int, ...],
    replicate: int,
    seed: int,
) -> dict[str, Any]:
    cross_sections = Path(openmc.config["cross_sections"]).expanduser().resolve()
    return {
        "scheme_label": label,
        "scheme": list(scheme),
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "model_variant": MODEL_VARIANT,
        "control_rod_state": "withdrawn",
        "top_be_thickness_cm": TOP_BE_THICKNESS_CM,
        "profile": FLUX_PROFILE,
        "replicate": int(replicate),
        "particles": PARTICLES,
        "batches": BATCHES,
        "inactive": INACTIVE,
        "seed": int(seed),
        "tally_configuration": FLUX_TALLY_VERSION,
        "mesh_lower_left": list(FLUX_LOWER_LEFT),
        "mesh_upper_right": list(FLUX_UPPER_RIGHT),
        "mesh_dimension": list(FLUX_DIMENSION),
        "cross_sections": str(cross_sections),
        "cross_sections_sha256": _sha256_file(cross_sections),
        "materials_notebook_sha256": _sha256_file(
            PROJECT_ROOT / "MNSR_materials_development.ipynb"
        ),
        "geometry_notebook_sha256": _sha256_file(
            PROJECT_ROOT / "MNSR_geometry_development.ipynb"
        ),
        "openmc_version": openmc.__version__,
    }


@dataclass
class FluxReplicate:
    result: dict[str, Any]
    table: pd.DataFrame


def run_flux_replicate(
    context: dict[str, Any],
    label: str,
    scheme: tuple[int, ...],
    replicate: int,
) -> FluxReplicate:
    if RUNTIME_COUNTER["new_flux_openmc_runs"] >= MAX_NEW_FLUX_OPENMC_RUNS:
        raise RuntimeError("新增flux OpenMC运行将超过8次硬上限。")
    if RUNTIME_COUNTER["total_new_openmc_runs"] >= MAX_TOTAL_NEW_OPENMC_RUNS:
        raise RuntimeError("本轮OpenMC运行将超过16次总硬上限。")
    seed = get_flux_seed(scheme, replicate)
    run_directory = FLUX_ROOT / label.lower() / f"replicate_{replicate}"
    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(f"flux新算例目录非空，拒绝覆盖或读取旧缓存：{run_directory}")
    run_directory.mkdir(parents=True, exist_ok=True)
    identity = _flux_identity(label, scheme, replicate, seed)
    manifest_path = run_directory / "run_manifest.json"
    _write_json(manifest_path, {"identity": identity, "status": "prepared"})
    output_path = run_directory / "openmc_output.txt"
    output_buffer = io.StringIO()
    start = time.perf_counter()
    try:
        geometry, metadata = context["build_geometry_for_scheme"](
            scheme,
            control_rod_state="withdrawn",
            export_directory=None,
            create_plots=False,
            model_variant=MODEL_VARIANT,
            top_be_thickness_cm=TOP_BE_THICKNESS_CM,
        )
        if (
            metadata["geometry_model_version"] != GEOMETRY_MODEL_VERSION
            or metadata["control_rod_state"] != "withdrawn"
            or metadata["counts"]["fuel"] != 345
            or metadata["counts"]["dummy"] != 5
            or metadata["counts"]["tie_rod"] != 4
        ):
            raise RuntimeError(f"{label} R{replicate}几何身份或构件计数错误。")
        if any(
            cell.fill is context["material_map"]["cadmium"]
            for cell in geometry.get_all_cells().values()
        ):
            raise RuntimeError("withdrawn flux模型中意外出现Cd。")
        materials, aliases = _prepare_materials(context, geometry)
        settings = _create_flux_settings(seed)
        tallies, requested_mesh = _create_flux_tally()
        if len(tallies) != 1 or tallies[0].name != "core_flux_mesh_xy":
            raise RuntimeError("flux算例必须且只能包含core_flux_mesh_xy tally。")
        model = openmc.Model(
            materials=materials,
            geometry=geometry,
            settings=settings,
            tallies=tallies,
        )
        model.export_to_xml(directory=run_directory)
        RUNTIME_COUNTER["new_flux_openmc_runs"] += 1
        RUNTIME_COUNTER["total_new_openmc_runs"] += 1
        from unittest.mock import PropertyMock, patch

        with (
            contextlib.redirect_stdout(output_buffer),
            contextlib.redirect_stderr(output_buffer),
            patch.object(
                openmc.Model,
                "is_initialized",
                new_callable=PropertyMock,
                return_value=False,
            ),
        ):
            returned = model.run(
                cwd=run_directory,
                openmc_exec=_find_openmc_executable(),
                export_model_xml=False,
            )
        output_path.write_text(output_buffer.getvalue(), encoding="utf-8")
        statepoint_path = (
            Path(returned) if returned is not None else run_directory / "statepoint.150.h5"
        )
        if not statepoint_path.is_absolute():
            statepoint_path = run_directory / statepoint_path
        statepoint_path = statepoint_path.resolve()
        if not statepoint_path.is_file():
            raise FileNotFoundError(f"缺少flux statepoint：{statepoint_path}")
        with openmc.StatePoint(statepoint_path) as statepoint:
            keff = statepoint.keff
            tally = statepoint.get_tally(name="core_flux_mesh_xy")
            mean = np.asarray(
                tally.get_reshaped_data(value="mean", expand_dims=True), dtype=float
            )[..., 0, 0]
            std = np.asarray(
                tally.get_reshaped_data(value="std_dev", expand_dims=True), dtype=float
            )[..., 0, 0]
            mesh = tally.filters[0].mesh
            centroids = np.asarray(mesh.centroids, dtype=float)
            generation = np.asarray(statepoint.k_generation, dtype=float).reshape(-1)
            entropy = np.asarray(statepoint.entropy, dtype=float).reshape(-1)
        if tuple(mean.shape) != FLUX_DIMENSION or tuple(std.shape) != FLUX_DIMENSION:
            raise ValueError(f"flux mesh tally形状错误：{mean.shape}, {std.shape}")
        if centroids.shape != (*FLUX_DIMENSION, 3):
            raise ValueError(f"flux mesh中心坐标形状错误：{centroids.shape}")
        if generation.size != BATCHES or entropy.size != BATCHES:
            raise ValueError("flux逐批keff或Shannon熵不是150项。")
        if not all(
            np.all(np.isfinite(array)) for array in (mean, std, generation, entropy)
        ):
            raise ValueError("flux statepoint包含非有限数值。")
        if np.any(mean <= 0.0):
            raise ValueError("flux mesh包含非正均值。")
        relative_uncertainty = std / mean
        ix, iy, _iz = np.indices(FLUX_DIMENSION)
        table = pd.DataFrame(
            {
                "ix": ix.reshape(-1),
                "iy": iy.reshape(-1),
                "x_center_cm": centroids[..., 0].reshape(-1),
                "y_center_cm": centroids[..., 1].reshape(-1),
                "raw_flux": mean.reshape(-1),
                "raw_flux_std": std.reshape(-1),
                "relative_uncertainty": relative_uncertainty.reshape(-1),
            }
        )
        mesh_path = run_directory / "flux_mesh.csv"
        table.to_csv(mesh_path, index=False, encoding="utf-8")
        phase = np.where(np.arange(1, BATCHES + 1) <= INACTIVE, "inactive", "active")
        generation_path = run_directory / "generation_history.csv"
        entropy_path = run_directory / "entropy_history.csv"
        pd.DataFrame(
            {"batch": np.arange(1, BATCHES + 1), "phase": phase, "k_generation": generation}
        ).to_csv(generation_path, index=False, encoding="utf-8")
        pd.DataFrame(
            {"batch": np.arange(1, BATCHES + 1), "phase": phase, "shannon_entropy": entropy}
        ).to_csv(entropy_path, index=False, encoding="utf-8")
        result = {
            "identity": identity,
            "status": "completed",
            "keff": float(keff.nominal_value),
            "keff_std": float(keff.std_dev),
            "entropy_stability": _entropy_summary(entropy),
            "mean_core_relative_uncertainty": None,
            "calculation_seconds": float(time.perf_counter() - start),
            "run_directory": str(run_directory),
            "statepoint_file": str(statepoint_path),
            "flux_mesh_file": str(mesh_path),
            "generation_history_file": str(generation_path),
            "entropy_history_file": str(entropy_path),
            "openmc_output_file": str(output_path),
            "thermal_scattering_aliases": aliases,
            "requested_mesh_id": int(requested_mesh.id),
        }
        core_mask = np.hypot(table["x_center_cm"], table["y_center_cm"]) <= CORE_RADIUS_CM
        result["mean_core_relative_uncertainty"] = float(
            table.loc[core_mask, "relative_uncertainty"].mean()
        )
        result_path = run_directory / "result.json"
        _write_json(result_path, result)
        _write_json(
            manifest_path,
            {"identity": identity, "status": "completed", "result_file": str(result_path)},
        )
        return FluxReplicate(result=result, table=table)
    except Exception as error:
        RUNTIME_COUNTER["failures"] += 1
        output_path.write_text(output_buffer.getvalue(), encoding="utf-8")
        failure = {
            "identity": identity,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "calculation_seconds": float(time.perf_counter() - start),
        }
        _write_json(run_directory / "error.json", failure)
        _write_json(manifest_path, failure)
        raise RuntimeError(f"{label} flux R{replicate}失败：{error}") from error


def _pool_flux(
    label: str,
    replicates: list[FluxReplicate],
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
    if len(replicates) != 4:
        raise ValueError(f"{label} flux必须恰好4个replicate。")
    tables = [item.table.sort_values(["ix", "iy"]).reset_index(drop=True) for item in replicates]
    coordinates = tables[0][["ix", "iy", "x_center_cm", "y_center_cm"]]
    for table in tables[1:]:
        if not np.array_equal(table[["ix", "iy"]].to_numpy(int), coordinates[["ix", "iy"]].to_numpy(int)):
            raise RuntimeError(f"{label}不同replicate的mesh索引不一致。")
        if not np.allclose(
            table[["x_center_cm", "y_center_cm"]],
            coordinates[["x_center_cm", "y_center_cm"]],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"{label}不同replicate的mesh中心不一致。")
    means = np.stack([table["raw_flux"].to_numpy(float) for table in tables])
    stds = np.stack([table["raw_flux_std"].to_numpy(float) for table in tables])
    pooled_mean = np.mean(means, axis=0)
    pooled_std = np.sqrt(np.sum(stds**2, axis=0)) / 4.0
    relative_uncertainty = pooled_std / pooled_mean
    output = coordinates.copy()
    output["raw_flux"] = pooled_mean
    output["raw_flux_std"] = pooled_std
    output["relative_uncertainty"] = relative_uncertainty
    radius = np.hypot(output["x_center_cm"], output["y_center_cm"])
    core_mask = radius <= CORE_RADIUS_CM
    core_mean = float(output.loc[core_mask, "raw_flux"].mean())
    output["core_region"] = core_mask
    output["normalized_flux"] = output["raw_flux"] / core_mean
    normalized = output["normalized_flux"].to_numpy(float).reshape(FLUX_DIMENSION)
    normalized_std = (output["raw_flux_std"] / core_mean).to_numpy(float).reshape(FLUX_DIMENSION)
    single_uncertainty = np.mean(
        np.stack([table["relative_uncertainty"].to_numpy(float) for table in tables]),
        axis=0,
    )
    reduction_ratio = float(
        np.mean(relative_uncertainty[core_mask]) / np.mean(single_uncertainty[core_mask])
    )
    if not 0.40 <= reduction_ratio <= 0.60:
        raise RuntimeError(f"{label} pooled flux误差下降比例异常：{reduction_ratio}")
    output_path = FLUX_ROOT / f"{label.lower()}_pooled_flux_mesh.csv"
    output.to_csv(output_path, index=False, encoding="utf-8")
    summary = {
        "scheme": list(SCHEMES[label]),
        "replicate_count": 4,
        "core_mean_raw_flux": core_mean,
        "pooled_mean_core_relative_uncertainty": float(
            np.mean(relative_uncertainty[core_mask])
        ),
        "pooled_max_core_relative_uncertainty": float(
            np.max(relative_uncertainty[core_mask])
        ),
        "mean_single_replicate_core_relative_uncertainty": float(
            np.mean(single_uncertainty[core_mask])
        ),
        "uncertainty_reduction_ratio": reduction_ratio,
        "core_normalized_mean": float(output.loc[core_mask, "normalized_flux"].mean()),
        "pooled_flux_file": str(output_path),
    }
    return output, summary, normalized, normalized_std


def _point_summary(
    table: pd.DataFrame,
    values: np.ndarray,
    core_mask: np.ndarray,
    mode: str,
) -> dict[str, Any]:
    valid = np.flatnonzero(core_mask)
    if mode == "max":
        index = int(valid[np.argmax(values[valid])])
    else:
        index = int(valid[np.argmin(values[valid])])
    row = table.iloc[index]
    return {
        "value": float(values[index]),
        "ix": int(row["ix"]),
        "iy": int(row["iy"]),
        "x_cm": float(row["x_center_cm"]),
        "y_cm": float(row["y_center_cm"]),
    }


def _format_axis(axis: plt.Axes) -> None:
    axis.set_xlabel("x (cm)", fontsize=12.5)
    axis.set_ylabel("y (cm)", fontsize=12.5)
    axis.set_xlim(-12.0, 12.0)
    axis.set_ylim(-12.0, 12.0)
    axis.set_aspect("equal")
    axis.tick_params(labelsize=10.5, width=1.1, length=4.5)
    axis.add_patch(Circle((0.0, 0.0), CORE_RADIUS_CM, fill=False, ec="#333333", lw=0.8, alpha=0.65))
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)


def _save_flux_figures(
    reference_array: np.ndarray,
    optimized_array: np.ndarray,
    core_mask_3d: np.ndarray,
    reference_raw: np.ndarray,
    optimized_raw: np.ndarray,
) -> dict[str, Any]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    reference_2d = reference_array[:, :, 0]
    optimized_2d = optimized_array[:, :, 0]
    core_mask = core_mask_3d[:, :, 0]
    reference_masked = np.ma.masked_where(~core_mask, reference_2d)
    optimized_masked = np.ma.masked_where(~core_mask, optimized_2d)
    difference = optimized_2d - reference_2d
    difference_masked = np.ma.masked_where(~core_mask, difference)
    common_vmin = float(min(reference_2d[core_mask].min(), optimized_2d[core_mask].min()))
    common_vmax = float(max(reference_2d[core_mask].max(), optimized_2d[core_mask].max()))
    diff_limit = float(np.max(np.abs(difference[core_mask])))
    extent = (-12.0, 12.0, -12.0, 12.0)

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), facecolor="white", constrained_layout=True)
    common_norm = Normalize(vmin=common_vmin, vmax=common_vmax)
    first_image = axes[0].imshow(
        reference_masked.T, origin="lower", extent=extent, cmap="viridis", norm=common_norm
    )
    axes[1].imshow(
        optimized_masked.T, origin="lower", extent=extent, cmap="viridis", norm=common_norm
    )
    diff_image = axes[2].imshow(
        difference_masked.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-diff_limit, vcenter=0.0, vmax=diff_limit),
    )
    titles = (
        "(a) Reference arrangement",
        "(b) Optimized arrangement",
        "(c) Optimized - Reference",
    )
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontsize=13.5, pad=8)
        _format_axis(axis)
    common_colorbar = figure.colorbar(first_image, ax=axes[:2], fraction=0.035, pad=0.025)
    common_colorbar.set_label("Normalized neutron flux", fontsize=11.5)
    common_colorbar.ax.tick_params(labelsize=9.5)
    diff_colorbar = figure.colorbar(diff_image, ax=axes[2], fraction=0.046, pad=0.025)
    diff_colorbar.set_label(r"$\Delta\phi^*$", fontsize=11.5)
    diff_colorbar.ax.tick_params(labelsize=9.5)
    normalized_png = FLUX_FIGURE_ROOT / "figure_5_9_normalized_flux_comparison.png"
    normalized_pdf = FLUX_FIGURE_ROOT / "figure_5_9_normalized_flux_comparison.pdf"
    figure.savefig(normalized_png, dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    figure.savefig(normalized_pdf, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(figure)

    raw_ref = reference_raw[:, :, 0]
    raw_opt = optimized_raw[:, :, 0]
    raw_norm = Normalize(vmin=float(min(raw_ref.min(), raw_opt.min())), vmax=float(max(raw_ref.max(), raw_opt.max())))
    raw_figure, raw_axes = plt.subplots(1, 2, figsize=(10.8, 5.2), facecolor="white", constrained_layout=True)
    raw_image = raw_axes[0].imshow(raw_ref.T, origin="lower", extent=extent, cmap="viridis", norm=raw_norm)
    raw_axes[1].imshow(raw_opt.T, origin="lower", extent=extent, cmap="viridis", norm=raw_norm)
    for axis, title in zip(raw_axes, ("(a) Reference arrangement", "(b) Optimized arrangement")):
        axis.set_title(title, fontsize=13.5, pad=8)
        _format_axis(axis)
    raw_colorbar = raw_figure.colorbar(raw_image, ax=raw_axes, fraction=0.038, pad=0.025)
    raw_colorbar.set_label("Flux tally per source particle", fontsize=11.5)
    raw_colorbar.ax.tick_params(labelsize=9.5)
    raw_png = FLUX_FIGURE_ROOT / "figure_5_9_raw_flux_comparison.png"
    raw_pdf = FLUX_FIGURE_ROOT / "figure_5_9_raw_flux_comparison.pdf"
    raw_figure.savefig(raw_png, dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    raw_figure.savefig(raw_pdf, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(raw_figure)
    return {
        "normalized_png": str(normalized_png),
        "normalized_pdf": str(normalized_pdf),
        "raw_png": str(raw_png),
        "raw_pdf": str(raw_pdf),
        "normalized_common_vmin": common_vmin,
        "normalized_common_vmax": common_vmax,
        "difference_symmetric_limit": diff_limit,
    }


def _radial_flux_profile(
    reference: pd.DataFrame,
    optimized: pd.DataFrame,
) -> pd.DataFrame:
    radius = np.hypot(reference["x_center_cm"], reference["y_center_cm"])
    rows = []
    pitch = 1.095
    for ring in range(1, 11):
        nominal = ring * pitch
        inner = nominal - 0.5 * pitch
        outer = 0.5 * (nominal + (ring + 1) * pitch) if ring < 10 else CORE_RADIUS_CM
        mask = (radius >= inner) & (radius < outer if ring < 10 else radius <= outer)
        rows.append(
            {
                "ring": ring,
                "nominal_radius_cm": nominal,
                "radial_inner_cm": inner,
                "radial_outer_cm": outer,
                "mesh_cell_count": int(mask.sum()),
                "reference_mean_normalized_flux": float(reference.loc[mask, "normalized_flux"].mean()),
                "optimized_mean_normalized_flux": float(optimized.loc[mask, "normalized_flux"].mean()),
                "delta_mean_normalized_flux": float(
                    optimized.loc[mask, "normalized_flux"].mean()
                    - reference.loc[mask, "normalized_flux"].mean()
                ),
            }
        )
    profile = pd.DataFrame(rows)
    path = FLUX_ROOT / "reference_vs_optimized_radial_flux_profile.csv"
    profile.to_csv(path, index=False, encoding="utf-8")
    return profile


def execute_flux_comparison(context: dict[str, Any]) -> dict[str, Any]:
    """Run exactly eight new withdrawn final flux cases and pool mesh tallies."""
    seeds = [
        {
            "scheme_label": label,
            "scheme": list(scheme),
            "replicate": replicate,
            "seed": get_flux_seed(scheme, replicate),
        }
        for label, scheme in SCHEMES.items()
        for replicate in range(4)
    ]
    if len({row["seed"] for row in seeds}) != 8:
        raise RuntimeError("8个flux随机种子不是全局唯一。")
    _write_json(
        FLUX_ROOT / "flux_run_plan.json",
        {
            "runs": seeds,
            "new_flux_run_hard_limit": MAX_NEW_FLUX_OPENMC_RUNS,
            "total_new_run_hard_limit": MAX_TOTAL_NEW_OPENMC_RUNS,
            "inserted_runs_planned": 0,
            "flux_runs_planned": 8,
            "control_rod_state": "withdrawn",
            "mesh": {
                "lower_left": list(FLUX_LOWER_LEFT),
                "upper_right": list(FLUX_UPPER_RIGHT),
                "dimension": list(FLUX_DIMENSION),
            },
        },
    )
    start = time.perf_counter()
    all_replicates: dict[str, list[FluxReplicate]] = {label: [] for label in SCHEMES}
    for label, scheme in SCHEMES.items():
        for replicate in range(4):
            all_replicates[label].append(
                run_flux_replicate(context, label, scheme, replicate)
            )
    if RUNTIME_COUNTER["new_flux_openmc_runs"] != 8:
        raise RuntimeError(
            f"新增flux OpenMC运行应严格为8，实际{RUNTIME_COUNTER['new_flux_openmc_runs']}。"
        )
    if RUNTIME_COUNTER["new_inserted_openmc_runs"] != 0:
        raise RuntimeError("几何验证失败后不应启动inserted OpenMC。")
    reference, reference_summary, reference_norm, reference_norm_std = _pool_flux(
        "REFERENCE", all_replicates["REFERENCE"]
    )
    optimized, optimized_summary, optimized_norm, optimized_norm_std = _pool_flux(
        "OPTIMIZED", all_replicates["OPTIMIZED"]
    )
    core_mask_flat = reference["core_region"].to_numpy(bool)
    core_mask = core_mask_flat.reshape(FLUX_DIMENSION)
    ref_values = reference["normalized_flux"].to_numpy(float)
    opt_values = optimized["normalized_flux"].to_numpy(float)
    delta = opt_values - ref_values
    delta_std = np.sqrt(
        reference_norm_std.reshape(-1) ** 2 + optimized_norm_std.reshape(-1) ** 2
    )
    rms_difference = float(np.sqrt(np.mean(delta[core_mask_flat] ** 2)))
    rms_statistical_uncertainty = float(
        np.sqrt(np.mean(delta_std[core_mask_flat] ** 2))
    )
    status = (
        "resolved_spatial_redistribution"
        if rms_difference > 2.0 * rms_statistical_uncertainty
        else "weak_relative_to_statistics"
    )
    comparison = {
        "max_positive_delta_phi_star": _point_summary(
            reference, delta, core_mask_flat, "max"
        ),
        "min_negative_delta_phi_star": _point_summary(
            reference, delta, core_mask_flat, "min"
        ),
        "rms_normalized_flux_difference": rms_difference,
        "mean_absolute_normalized_flux_difference": float(
            np.mean(np.abs(delta[core_mask_flat]))
        ),
        "rms_combined_statistical_uncertainty": rms_statistical_uncertainty,
        "difference_to_uncertainty_ratio": float(
            rms_difference / rms_statistical_uncertainty
        ),
        "core_cells_with_flux_increase": int(np.sum(delta[core_mask_flat] > 0.0)),
        "core_cells_with_flux_decrease": int(np.sum(delta[core_mask_flat] < 0.0)),
        "core_cells_equal_within_float": int(np.sum(delta[core_mask_flat] == 0.0)),
        "flux_difference_status": status,
    }
    reference_summary.update(
        {
            "normalized_max": _point_summary(reference, ref_values, core_mask_flat, "max"),
            "normalized_min": _point_summary(reference, ref_values, core_mask_flat, "min"),
        }
    )
    optimized_summary.update(
        {
            "normalized_max": _point_summary(optimized, opt_values, core_mask_flat, "max"),
            "normalized_min": _point_summary(optimized, opt_values, core_mask_flat, "min"),
        }
    )
    figures = _save_flux_figures(
        reference_norm,
        optimized_norm,
        core_mask,
        reference["raw_flux"].to_numpy(float).reshape(FLUX_DIMENSION),
        optimized["raw_flux"].to_numpy(float).reshape(FLUX_DIMENSION),
    )
    profile = _radial_flux_profile(reference, optimized)
    replicate_rows = [item.result for values in all_replicates.values() for item in values]
    pd.DataFrame(
        [
            {
                "scheme_label": item["identity"]["scheme_label"],
                "scheme": json.dumps(item["identity"]["scheme"]),
                "replicate": item["identity"]["replicate"],
                "seed": item["identity"]["seed"],
                "keff": item["keff"],
                "keff_std": item["keff_std"],
                "entropy_stable": item["entropy_stability"]["heuristically_stable"],
                "mean_core_relative_uncertainty": item["mean_core_relative_uncertainty"],
                "calculation_seconds": item["calculation_seconds"],
                "run_directory": item["run_directory"],
                "statepoint_file": item["statepoint_file"],
                "status": item["status"],
            }
            for item in replicate_rows
        ]
    ).to_csv(FLUX_ROOT / "flux_replicates.csv", index=False, encoding="utf-8")
    summary = {
        "task": "thesis_5_3_3_figure_5_9_flux_comparison",
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "model_variant": MODEL_VARIANT,
        "control_rod_state": "withdrawn",
        "reference_scheme": list(REFERENCE),
        "optimized_scheme": list(OPTIMIZED),
        "particles": PARTICLES,
        "batches": BATCHES,
        "inactive": INACTIVE,
        "active_batches": ACTIVE_BATCHES,
        "new_flux_openmc_runs": RUNTIME_COUNTER["new_flux_openmc_runs"],
        "new_inserted_openmc_runs": RUNTIME_COUNTER["new_inserted_openmc_runs"],
        "total_new_openmc_runs": RUNTIME_COUNTER["total_new_openmc_runs"],
        "failures": RUNTIME_COUNTER["failures"],
        "total_wall_time_seconds": float(time.perf_counter() - start),
        "mesh": {
            "lower_left_cm": list(FLUX_LOWER_LEFT),
            "upper_right_cm": list(FLUX_UPPER_RIGHT),
            "dimension": list(FLUX_DIMENSION),
            "interpretation": "active-height XY projection / axial integral mesh tally",
        },
        "normalization": "each scheme divided by its own core-region mean over cell centers r<=11.5 cm",
        "raw_flux_units": "OpenMC eigenvalue flux tally per source particle; not normalized to 30 kW",
        "pooling_formula": "phi_j,pool=mean(phi_j,r); sigma_j,pool=sqrt(sum sigma_j,r^2)/4",
        "reference": reference_summary,
        "optimized": optimized_summary,
        "comparison": comparison,
        "figures": figures,
        "radial_flux_profile_file": str(
            FLUX_ROOT / "reference_vs_optimized_radial_flux_profile.csv"
        ),
        "flux_replicates_file": str(FLUX_ROOT / "flux_replicates.csv"),
        "energy_spectrum_runs": 0,
        "ga_runs": 0,
        "physics_model_modified": False,
        "status": "completed",
    }
    _write_json(FLUX_ROOT / "figure_5_9_flux_summary.json", summary)
    _write_json(FLUX_ROOT / "flux_comparison_summary.json", summary)
    return summary


def execute_workflow(run_flux: bool) -> dict[str, Any]:
    context = load_model_context()
    geometry_validation = validate_control_rod_geometry(context)
    withdrawn_audit = audit_withdrawn_confirmation()
    control_summary = write_blocked_control_outputs(
        geometry_validation,
        withdrawn_audit,
    )
    if geometry_validation["inserted_geometry_status"] != "failed":
        raise RuntimeError(
            "本工作流针对当前已知轴向缺陷应得到failed；若几何已修改，必须重新审查后另行实现inserted运行。"
        )
    flux_summary = None
    if run_flux:
        flux_summary = execute_flux_comparison(context)
    result = {
        "control_rod_analysis": control_summary,
        "flux_comparison": flux_summary,
        "runtime_counter": dict(RUNTIME_COUNTER),
    }
    _write_json(PROJECT_ROOT / "build/results/control_rod_flux_workflow_summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-flux",
        action="store_true",
        help="Run exactly eight new withdrawn final flux calculations.",
    )
    arguments = parser.parse_args()
    workflow_result = execute_workflow(run_flux=arguments.run_flux)
    print(json.dumps(_jsonable(workflow_result), ensure_ascii=False, indent=2))
