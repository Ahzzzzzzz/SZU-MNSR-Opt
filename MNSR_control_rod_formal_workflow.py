"""Formal MNSR-SZ central-control-rod worth workflow for thesis Section 5.4.

The workflow compares the same movable control-rod component at 0 and 24 cm.
It runs exactly four independent final replicates for each of two layouts and
two positions (16 transport runs at most).  It never uses the historical
water-only ``control_rod_state='withdrawn'`` result as ``rho_out``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import PropertyMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "build/.matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import openmc
import pandas as pd

from MNSR_control_rod_flux_workflow import (
    _find_openmc_executable,
    _prepare_materials,
    load_model_context,
)


REFERENCE = (285, 298, 311, 324, 337)
OPTIMIZED = (200, 213, 223, 239, 256)
SCHEMES = {"REFERENCE": REFERENCE, "OPTIMIZED": OPTIMIZED}
POSITIONS_CM = (0.0, 24.0)
REPLICATES = (1, 2, 3, 4)

GEOMETRY_MODEL_VERSION = "iaea_reference_v1"
MODEL_VARIANT = "iaea_reference"
CONTROL_ROD_AXIAL_DEFINITION_VERSION = "mnsr_sz_independent_axial_surfaces_v1"
CALCULATION_PROFILE = "control_rod_final"
TEMPERATURE_K = 293.15
PARTICLES = 20_000
BATCHES = 150
INACTIVE = 50
ACTIVE_BATCHES = BATCHES - INACTIVE
BASE_SEED = 20260820
MAX_NEW_CONTROL_ROD_FINAL_RUNS = 16
EXPECTED_STATEPOINT_NAME = "statepoint.150.h5"
IAEA_LEU_CONTROL_ROD_WORTH_MK = 5.75

OUTPUT_ROOT = (PROJECT_ROOT / "build/results/control_rod_analysis").resolve()
RUN_ROOT = OUTPUT_ROOT / "formal_final"
PREFLIGHT_PATH = OUTPUT_ROOT / "control_rod_formal_preflight.json"
REPLICATES_PATH = OUTPUT_ROOT / "control_rod_final_replicates.csv"
POSITION_SUMMARY_PATH = OUTPUT_ROOT / "control_rod_position_summary.csv"
COMPARISON_PATH = OUTPUT_ROOT / "control_rod_comparison.csv"
TABLE_5_5_PATH = OUTPUT_ROOT / "table_5_5_control_rod_candidate.csv"
FINAL_SUMMARY_PATH = OUTPUT_ROOT / "control_rod_final_summary.json"
WARNINGS_PATH = OUTPUT_ROOT / "control_rod_final_nuclear_data_warnings.json"

EXPECTED_COUNTS = {
    "physical_positions": 355,
    "ordinary_positions": 350,
    "fuel": 345,
    "dummy": 5,
    "tie_rod": 4,
    "central": 1,
}
EXPECTED_BOUNDS = {
    0.0: {
        "cadmium": (-13.3, 13.3, 26.6),
        "central_al_rod": (-14.5, 14.5, 29.0),
        "ss304_tube": (-22.5, 22.5, 45.0),
    },
    24.0: {
        "cadmium": (10.7, 37.3, 26.6),
        "central_al_rod": (9.5, 38.5, 29.0),
        "ss304_tube": (1.5, 46.5, 45.0),
    },
}
EXPECTED_GUIDE_BOUNDS = (-12.9, 12.9, 25.8)

SOURCE_FILES_TO_PROTECT = (
    PROJECT_ROOT / "MNSR_materials_development.ipynb",
    PROJECT_ROOT / "MNSR_geometry_development.ipynb",
    PROJECT_ROOT / "MNSR_optimization_development.ipynb",
)


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


def _protected_source_hashes() -> dict[str, str]:
    return {str(path): _sha256_file(path) for path in SOURCE_FILES_TO_PROTECT}


def stable_seed(
    scheme_label: str,
    scheme: tuple[int, ...],
    position_cm: float,
    replicate: int,
) -> int:
    """Return a stable OpenMC seed derived from the complete case identity."""
    payload = {
        "base_seed": BASE_SEED,
        "scheme_label": scheme_label,
        "scheme": list(scheme),
        "position_cm": float(position_cm),
        "replicate": int(replicate),
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "control_rod_axial_definition_version": (
            CONTROL_ROD_AXIAL_DEFINITION_VERSION
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_646 + 1


def _case_identity(
    scheme_label: str,
    scheme: tuple[int, ...],
    position_cm: float,
    replicate: int,
) -> dict[str, Any]:
    return {
        "scheme_label": scheme_label,
        "dummy_position_ids": list(scheme),
        "control_rod_state": "inserted",
        "control_rod_position_cm": float(position_cm),
        "replicate": int(replicate),
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "model_variant": MODEL_VARIANT,
        "control_rod_axial_definition_version": (
            CONTROL_ROD_AXIAL_DEFINITION_VERSION
        ),
        "calculation_profile": CALCULATION_PROFILE,
        "particles": PARTICLES,
        "batches": BATCHES,
        "inactive": INACTIVE,
        "temperature_k": TEMPERATURE_K,
        "base_seed": BASE_SEED,
        "random_seed": stable_seed(
            scheme_label, scheme, position_cm, replicate
        ),
        "tally_count": 0,
        "tally_names": [],
        "expected_statepoint": EXPECTED_STATEPOINT_NAME,
        "uses_water_only_withdrawn_cache": False,
    }


def _case_directory(
    scheme_label: str, position_cm: float, replicate: int
) -> Path:
    position_label = str(position_cm).replace(".", "p")
    return (
        RUN_ROOT
        / scheme_label.lower()
        / f"position_{position_label}_cm"
        / f"replicate_{replicate}"
    )


def _close(value: float, expected: float) -> bool:
    return math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1.0e-10)


def _material_assignments(geometry: openmc.Geometry) -> dict[str, str]:
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


def _count_overlap_pixels(
    geometry: openmc.Geometry,
    basis: str,
    width: tuple[float, float],
    pixels: tuple[int, int],
) -> int:
    material_colors = {
        material: color
        for material, color in zip(
            geometry.get_all_materials().values(),
            (
                "gold",
                "silver",
                "orange",
                "dimgray",
                "slategray",
                "lightblue",
                "green",
                "navy",
            ),
            strict=False,
        )
    }
    plot = openmc.Plot()
    plot.basis = basis
    plot.origin = (0.0, 0.0, 0.0)
    plot.width = width
    plot.pixels = pixels
    plot.color_by = "material"
    plot.colors = material_colors
    plot.show_overlaps = True
    plot.overlap_color = "magenta"
    with TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        materials = openmc.Materials(list(geometry.get_all_materials().values()))
        materials.cross_sections = str(Path(openmc.config["cross_sections"]).resolve())
        model = openmc.Model(
            materials=materials,
            geometry=geometry,
            plots=openmc.Plots([plot]),
        )
        model.export_to_xml(directory=directory)
        openmc.plot_geometry(
            output=True,
            cwd=directory,
            openmc_exec=_find_openmc_executable(),
        )
        image_path = directory / f"plot_{plot.id}.png"
        if not image_path.is_file():
            image_path = image_path.with_suffix(".ppm")
        if not image_path.is_file():
            raise FileNotFoundError(f"OpenMC未生成{basis.upper()}重叠检查图。")
        array = np.asarray(mpimg.imread(image_path))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(np.rint(array * 255.0), 0, 255)
    rgb = array.astype(np.uint8)[..., :3]
    count = int(np.sum(np.all(rgb == np.array([255, 0, 255]), axis=-1)))
    return count


def run_preflight(context: dict[str, Any]) -> dict[str, Any]:
    """Rebuild and validate all four formal geometry combinations."""
    builder = context["build_geometry_for_scheme"]
    records: list[dict[str, Any]] = []
    boundary_names: list[tuple[str, ...]] = []
    bounding_boxes: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    expected_assignments = {
        "IAEA movable control rod inner aluminum": "Al-303-1",
        "IAEA movable cadmium absorber tube": "Natural Cadmium",
        "IAEA movable SS-304 control rod tube": "SS-304",
        "IAEA LT-21 control guide tube": "LT-21",
        "IAEA control rod corridor water complement": "Light Water",
    }

    for scheme_label, scheme in SCHEMES.items():
        for position_cm in POSITIONS_CM:
            geometry, metadata = builder(
                scheme,
                control_rod_state="inserted",
                control_rod_position_cm=position_cm,
                model_variant=MODEL_VARIANT,
                top_be_thickness_cm=0.0,
                create_plots=False,
            )
            materials, aliases = _prepare_materials(context, geometry)
            bounds = metadata["control_rod_axial_bounds"]
            expected = EXPECTED_BOUNDS[position_cm]
            axial_bounds_ok = all(
                _close(bounds[name]["z_bottom_cm"], values[0])
                and _close(bounds[name]["z_top_cm"], values[1])
                and _close(bounds[name]["length_cm"], values[2])
                for name, values in expected.items()
            )
            guide = metadata["guide_tube_bounds"]
            guide_ok = (
                _close(guide["z_bottom_cm"], EXPECTED_GUIDE_BOUNDS[0])
                and _close(guide["z_top_cm"], EXPECTED_GUIDE_BOUNDS[1])
                and _close(guide["height_cm"], EXPECTED_GUIDE_BOUNDS[2])
            )
            xy_overlap = _count_overlap_pixels(
                geometry, "xy", (24.0, 24.0), (900, 900)
            )
            xz_overlap = _count_overlap_pixels(
                geometry, "xz", (4.0, 70.0), (700, 1400)
            )
            temperatures_ok = all(
                material.temperature is not None
                and _close(float(material.temperature), TEMPERATURE_K)
                for material in materials
            )
            assignments = _material_assignments(geometry)
            checks = {
                "component_counts": metadata["counts"] == EXPECTED_COUNTS,
                "dummy_ids": tuple(metadata["dummy_position_ids"]) == scheme,
                "fuel_count": len(metadata["fuel_position_ids"]) == 345,
                "control_rod_state": metadata["control_rod_state"] == "inserted",
                "control_rod_position": _close(
                    metadata["control_rod_position_cm"], position_cm
                ),
                "axial_definition_version": (
                    metadata["control_rod_axial_definition_version"]
                    == CONTROL_ROD_AXIAL_DEFINITION_VERSION
                ),
                "exact_axial_bounds": axial_bounds_ok,
                "fixed_guide_bounds": guide_ok,
                "material_assignments": assignments == expected_assignments,
                "temperature_293p15_k": temperatures_ok,
                "no_top_beryllium": _close(
                    metadata.get("top_be_thickness_cm", 0.0), 0.0
                ),
                "xy_overlap_pixels_zero": xy_overlap == 0,
                "xz_overlap_pixels_zero": xz_overlap == 0,
            }
            boundary_names.append(tuple(metadata["vacuum_boundary_names"]))
            bounding_boxes.append(
                (
                    tuple(float(value) for value in geometry.bounding_box.lower_left),
                    tuple(float(value) for value in geometry.bounding_box.upper_right),
                )
            )
            records.append(
                {
                    "scheme_label": scheme_label,
                    "dummy_position_ids": list(scheme),
                    "position_cm": position_cm,
                    "control_rod_axial_bounds": bounds,
                    "guide_tube_bounds": guide,
                    "material_assignments": assignments,
                    "thermal_scattering_aliases": aliases,
                    "xy_overlap_pixels": xy_overlap,
                    "xz_overlap_pixels": xz_overlap,
                    "vacuum_boundary_names": metadata["vacuum_boundary_names"],
                    "bounding_box_cm": {
                        "lower_left": bounding_boxes[-1][0],
                        "upper_right": bounding_boxes[-1][1],
                    },
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )

    seeds = [
        stable_seed(label, scheme, position, replicate)
        for label, scheme in SCHEMES.items()
        for position in POSITIONS_CM
        for replicate in REPLICATES
    ]
    global_checks = {
        "four_geometry_combinations_checked": len(records) == 4,
        "all_case_geometry_checks_passed": all(item["passed"] for item in records),
        "strict_cd_outer_diameter_3p9_mm": _close(
            20.0 * context["R_CADMIUM_OUTER"], 3.9
        ),
        "all_external_boundary_names_identical": len(set(boundary_names)) == 1,
        "all_external_bounding_boxes_identical": len(set(bounding_boxes)) == 1,
        "exactly_16_unique_sha256_seeds": len(seeds) == 16
        and len(set(seeds)) == 16,
        "openmc_executable_exists": _find_openmc_executable().is_file(),
        "cross_sections_file_exists": Path(
            openmc.config["cross_sections"]
        ).expanduser().resolve().is_file(),
        "no_transport_tallies_requested": True,
        "water_only_withdrawn_result_excluded": True,
        "optional_11p25_final_not_scheduled": True,
    }
    payload = {
        "status": "passed" if all(global_checks.values()) else "failed",
        "formal_run_authorization_basis": (
            "The user explicitly removed the low-statistics 11.25 cm midpoint "
            "criterion as a hard stop; endpoint pilot direction and geometry checks passed."
        ),
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "model_variant": MODEL_VARIANT,
        "control_rod_axial_definition_version": (
            CONTROL_ROD_AXIAL_DEFINITION_VERSION
        ),
        "position_definition": {
            "position_0_cm": "fully inserted reference in this paper model",
            "position_24_cm": "withdrawn reference in the same movable-component model",
            "positive_direction": "+z",
            "as_built_claim": False,
        },
        "calculation_profile": CALCULATION_PROFILE,
        "particles": PARTICLES,
        "batches": BATCHES,
        "inactive": INACTIVE,
        "temperature_k": TEMPERATURE_K,
        "planned_transport_runs": 16,
        "maximum_new_transport_runs": MAX_NEW_CONTROL_ROD_FINAL_RUNS,
        "seed_matrix": seeds,
        "geometry_records": records,
        "global_checks": global_checks,
        "openmc_executable": str(_find_openmc_executable()),
        "cross_sections": str(Path(openmc.config["cross_sections"]).resolve()),
        "old_withdrawn_cache_read": False,
        "old_pilot_transport_result_used_as_formal_result": False,
    }
    _write_json(PREFLIGHT_PATH, payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"正式控制棒计算预检失败：{PREFLIGHT_PATH}")
    return payload


def _create_settings(seed: int) -> openmc.Settings:
    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = PARTICLES
    settings.batches = BATCHES
    settings.inactive = INACTIVE
    settings.seed = int(seed)
    settings.temperature = {
        "method": "nearest",
        "tolerance": 10.0,
        "default": TEMPERATURE_K,
    }
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(
            (-11.25, -11.25, -11.499999),
            (11.25, 11.25, 11.499999),
        ),
        constraints={"fissionable": True},
    )
    entropy_mesh = openmc.RegularMesh(name="Control rod final entropy mesh")
    entropy_mesh.lower_left = (-11.5, -11.5, -11.5)
    entropy_mesh.upper_right = (11.5, 11.5, 11.5)
    entropy_mesh.dimension = (10, 10, 1)
    settings.entropy_mesh = entropy_mesh
    settings.statepoint = {"batches": [BATCHES]}
    settings.output = {"tallies": False}
    return settings


def _entropy_summary(entropy: np.ndarray) -> dict[str, Any]:
    active = np.asarray(entropy, dtype=float)[INACTIVE:]
    tail = active[-20:]
    slope = float(np.polyfit(np.arange(tail.size, dtype=float), tail, 1)[0])
    value_range = float(np.ptp(tail))
    return {
        "active_batch_count": int(active.size),
        "tail_batch_count": int(tail.size),
        "first_value": float(tail[0]),
        "last_value": float(tail[-1]),
        "absolute_first_to_last_change": float(abs(tail[-1] - tail[0])),
        "range": value_range,
        "linear_slope_per_batch": slope,
        "heuristically_stable": bool(
            value_range <= 0.10 and abs(slope) <= 0.005
        ),
    }


def _collect_warnings(text: str) -> list[str]:
    return sorted(
        {
            line.strip()
            for line in text.splitlines()
            if "warning" in line.lower()
        }
    )


def _validate_cached_result(
    path: Path, expected_identity: dict[str, Any]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("identity") != expected_identity:
        raise RuntimeError(f"正式算例缓存身份不匹配，拒绝读取或覆盖：{path}")
    required_files = (
        result["statepoint_file"],
        result["openmc_output_file"],
        result["generation_history_file"],
        result["entropy_history_file"],
    )
    if not all(Path(item).is_file() for item in required_files):
        raise FileNotFoundError(f"正式算例缓存文件不完整：{path.parent}")
    generation = pd.read_csv(result["generation_history_file"])
    entropy = pd.read_csv(result["entropy_history_file"])
    if len(generation) != BATCHES or len(entropy) != BATCHES:
        raise ValueError(f"正式算例缓存历史长度错误：{path.parent}")
    return result


def run_case(
    context: dict[str, Any],
    scheme_label: str,
    scheme: tuple[int, ...],
    position_cm: float,
    replicate: int,
    runtime_counter: dict[str, int],
) -> dict[str, Any]:
    identity = _case_identity(scheme_label, scheme, position_cm, replicate)
    run_directory = _case_directory(scheme_label, position_cm, replicate)
    result_path = run_directory / "result.json"
    cached = _validate_cached_result(result_path, identity)
    if cached is not None:
        print(f"[缓存] {scheme_label} pos={position_cm:g} cm R{replicate}", flush=True)
        return cached
    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(f"正式算例目录非空且无有效缓存，拒绝覆盖：{run_directory}")
    if runtime_counter["new_runs"] >= MAX_NEW_CONTROL_ROD_FINAL_RUNS:
        raise RuntimeError("已达到16次新OpenMC正式运行硬上限。")
    run_directory.mkdir(parents=True, exist_ok=True)

    geometry, metadata = context["build_geometry_for_scheme"](
        scheme,
        control_rod_state="inserted",
        control_rod_position_cm=position_cm,
        model_variant=MODEL_VARIANT,
        top_be_thickness_cm=0.0,
        create_plots=False,
    )
    if metadata["counts"] != EXPECTED_COUNTS:
        raise RuntimeError("运行前几何构件计数发生变化。")
    if tuple(metadata["dummy_position_ids"]) != scheme:
        raise RuntimeError("运行前假元件编号发生变化。")
    materials, aliases = _prepare_materials(context, geometry)
    settings = _create_settings(identity["random_seed"])
    tallies = openmc.Tallies()
    if len(tallies) != 0:
        raise AssertionError("控制棒final不得创建输运tally。")
    model = openmc.Model(
        materials=materials,
        geometry=geometry,
        settings=settings,
        tallies=tallies,
    )
    _write_json(
        run_directory / "run_manifest.json",
        {"identity": identity, "status": "prepared"},
    )
    model.export_to_xml(directory=run_directory)
    runtime_counter["new_runs"] += 1
    print(
        f"[运行 {runtime_counter['new_runs']:02d}/16] {scheme_label} "
        f"pos={position_cm:g} cm R{replicate} seed={identity['random_seed']}",
        flush=True,
    )
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
                export_model_xml=False,
            )
        output_text = buffer.getvalue()
        output_path = run_directory / "openmc_output.txt"
        output_path.write_text(output_text, encoding="utf-8")
        statepoint_path = Path(statepoint_path)
        if not statepoint_path.is_absolute():
            statepoint_path = run_directory / statepoint_path
        statepoint_path = statepoint_path.resolve()
        if statepoint_path.name != EXPECTED_STATEPOINT_NAME or not statepoint_path.is_file():
            raise FileNotFoundError(
                f"未生成预期正式statepoint：{run_directory / EXPECTED_STATEPOINT_NAME}"
            )
        with openmc.StatePoint(statepoint_path) as statepoint:
            keff = statepoint.keff
            generations = np.asarray(statepoint.k_generation, dtype=float).reshape(-1)
            entropy = np.asarray(statepoint.entropy, dtype=float).reshape(-1)
        if generations.size != BATCHES or entropy.size != BATCHES:
            raise ValueError("逐批keff或Shannon熵长度不是150。")
        if not np.all(np.isfinite(generations)) or not np.all(np.isfinite(entropy)):
            raise ValueError("逐批keff或Shannon熵包含非有限值。")
        keff_mean = float(keff.nominal_value)
        keff_std = float(keff.std_dev)
        if not math.isfinite(keff_mean) or not math.isfinite(keff_std) or keff_std <= 0.0:
            raise ValueError("正式keff或其标准差无效。")
        rho = (keff_mean - 1.0) / keff_mean
        rho_std = keff_std / keff_mean**2
        batches = np.arange(1, BATCHES + 1)
        phases = np.where(batches <= INACTIVE, "inactive", "active")
        generation_path = run_directory / "generation_history.csv"
        entropy_path = run_directory / "entropy_history.csv"
        pd.DataFrame(
            {"batch": batches, "phase": phases, "generation_keff": generations}
        ).to_csv(generation_path, index=False, encoding="utf-8")
        pd.DataFrame(
            {"batch": batches, "phase": phases, "shannon_entropy": entropy}
        ).to_csv(entropy_path, index=False, encoding="utf-8")
        result = {
            "identity": identity,
            "status": "completed",
            "keff_mean": keff_mean,
            "keff_std": keff_std,
            "reactivity": rho,
            "reactivity_std": rho_std,
            "reactivity_mk": rho * 1000.0,
            "reactivity_std_mk": rho_std * 1000.0,
            "reactivity_pcm": rho * 100_000.0,
            "reactivity_std_pcm": rho_std * 100_000.0,
            "entropy": _entropy_summary(entropy),
            "warnings": _collect_warnings(output_text),
            "cd106_negative_probability_warning": bool(
                re.search(r"Cd106.*negative|negative.*Cd106", output_text, re.IGNORECASE)
            ),
            "thermal_scattering_aliases": aliases,
            "control_rod_axial_bounds": metadata["control_rod_axial_bounds"],
            "guide_tube_bounds": metadata["guide_tube_bounds"],
            "statepoint_file": str(statepoint_path),
            "openmc_output_file": str(output_path.resolve()),
            "generation_history_file": str(generation_path.resolve()),
            "entropy_history_file": str(entropy_path.resolve()),
            "runtime_seconds": float(time.perf_counter() - started),
        }
        _write_json(result_path, result)
        _write_json(
            run_directory / "run_manifest.json",
            {
                "identity": identity,
                "status": "completed",
                "result_file": str(result_path.resolve()),
            },
        )
        print(
            f"[完成] {scheme_label} pos={position_cm:g} cm R{replicate}: "
            f"keff={keff_mean:.7f} ± {keff_std:.7f}, "
            f"{result['runtime_seconds']:.1f} s",
            flush=True,
        )
        return result
    except Exception as error:
        output_path = run_directory / "openmc_output.txt"
        output_path.write_text(buffer.getvalue(), encoding="utf-8")
        failure = {
            "identity": identity,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "runtime_seconds": float(time.perf_counter() - started),
        }
        _write_json(run_directory / "error.json", failure)
        _write_json(run_directory / "run_manifest.json", failure)
        raise RuntimeError(
            f"{scheme_label} position={position_cm:g} cm R{replicate}失败：{error}"
        ) from error


def _replicate_consistency(
    keff_values: np.ndarray, keff_stds: np.ndarray
) -> dict[str, Any]:
    weights = 1.0 / np.square(keff_stds)
    weighted_mean = float(np.sum(weights * keff_values) / np.sum(weights))
    chi_square = float(np.sum(np.square(keff_values - weighted_mean) / np.square(keff_stds)))
    threshold_95_df3 = 7.814727903
    return {
        "diagnostic": "inverse-variance chi-square consistency diagnostic only",
        "degrees_of_freedom": 3,
        "chi_square": chi_square,
        "chi_square_95_percent_threshold": threshold_95_df3,
        "status": "stable" if chi_square <= threshold_95_df3 else "needs_review",
    }


def aggregate_position(
    scheme_label: str,
    scheme: tuple[int, ...],
    position_cm: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(results) != 4:
        raise ValueError("每个scheme × position必须恰好有4个replicate。")
    keff = np.asarray([item["keff_mean"] for item in results], dtype=float)
    keff_std = np.asarray([item["keff_std"] for item in results], dtype=float)
    rho = np.asarray([item["reactivity"] for item in results], dtype=float)
    rho_std = np.asarray([item["reactivity_std"] for item in results], dtype=float)
    keff_uncertainty = float(np.sqrt(np.sum(np.square(keff_std))) / 4.0)
    rho_uncertainty = float(np.sqrt(np.sum(np.square(rho_std))) / 4.0)
    consistency = _replicate_consistency(keff, keff_std)
    entropy_all_stable = all(item["entropy"]["heuristically_stable"] for item in results)
    return {
        "scheme_label": scheme_label,
        "dummy_position_ids": list(scheme),
        "control_rod_position_cm": position_cm,
        "replicate_count": 4,
        "keff_combined": float(np.mean(keff)),
        "keff_uncertainty": keff_uncertainty,
        "keff_replicate_mean": float(np.mean(keff)),
        "keff_replicate_sample_std": float(np.std(keff, ddof=1)),
        "keff_min": float(np.min(keff)),
        "keff_max": float(np.max(keff)),
        "rho_combined": float(np.mean(rho)),
        "rho_uncertainty": rho_uncertainty,
        "rho_mk_combined": float(np.mean(rho) * 1000.0),
        "rho_mk_uncertainty": rho_uncertainty * 1000.0,
        "rho_pcm_combined": float(np.mean(rho) * 100_000.0),
        "rho_pcm_uncertainty": rho_uncertainty * 100_000.0,
        "entropy_status": "stable" if entropy_all_stable else "needs_review",
        "all_entropy_tails_stable": entropy_all_stable,
        "replicate_stability": consistency,
        "replicate_stability_status": consistency["status"],
        "aggregation_method": (
            "equal-weight mean of four final replicates; independent uncertainty "
            "sqrt(sum sigma_r^2)/4"
        ),
    }


def _position_summary_frame(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in summaries:
        row = dict(item)
        row["dummy_position_ids"] = json.dumps(item["dummy_position_ids"])
        row["replicate_stability_chi_square"] = item["replicate_stability"]["chi_square"]
        row.pop("replicate_stability")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_results(
    results: list[dict[str, Any]],
    preflight: dict[str, Any],
    source_hashes_before: dict[str, str],
    runtime_counter: dict[str, int],
    workflow_started: float,
) -> dict[str, Any]:
    if len(results) != 16 or not all(item["status"] == "completed" for item in results):
        raise RuntimeError("正式控制棒结果不是16个完整成功算例。")
    replicate_rows = []
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for result in results:
        identity = result["identity"]
        key = (identity["scheme_label"], float(identity["control_rod_position_cm"]))
        grouped.setdefault(key, []).append(result)
        replicate_rows.append(
            {
                "scheme_label": identity["scheme_label"],
                "dummy_position_ids": json.dumps(identity["dummy_position_ids"]),
                "control_rod_position_cm": identity["control_rod_position_cm"],
                "replicate": identity["replicate"],
                "random_seed": identity["random_seed"],
                "keff_mean": result["keff_mean"],
                "keff_std": result["keff_std"],
                "rho": result["reactivity"],
                "rho_std": result["reactivity_std"],
                "rho_mk": result["reactivity_mk"],
                "rho_std_mk": result["reactivity_std_mk"],
                "entropy_stable": result["entropy"]["heuristically_stable"],
                "entropy_tail_range": result["entropy"]["range"],
                "entropy_tail_slope_per_batch": result["entropy"]["linear_slope_per_batch"],
                "cd106_warning": result["cd106_negative_probability_warning"],
                "runtime_seconds": result["runtime_seconds"],
                "run_directory": str(Path(result["statepoint_file"]).parent),
                "statepoint_file": result["statepoint_file"],
                "openmc_output_file": result["openmc_output_file"],
                "status": result["status"],
            }
        )
    replicate_frame = pd.DataFrame(replicate_rows).sort_values(
        ["scheme_label", "control_rod_position_cm", "replicate"]
    )
    replicate_frame.to_csv(REPLICATES_PATH, index=False, encoding="utf-8")

    position_summaries = []
    by_label_position: dict[tuple[str, float], dict[str, Any]] = {}
    for scheme_label, scheme in SCHEMES.items():
        for position_cm in POSITIONS_CM:
            values = sorted(
                grouped[(scheme_label, position_cm)],
                key=lambda item: item["identity"]["replicate"],
            )
            summary = aggregate_position(scheme_label, scheme, position_cm, values)
            position_summaries.append(summary)
            by_label_position[(scheme_label, position_cm)] = summary
    _position_summary_frame(position_summaries).to_csv(
        POSITION_SUMMARY_PATH, index=False, encoding="utf-8"
    )

    scheme_metrics: dict[str, dict[str, Any]] = {}
    for scheme_label in SCHEMES:
        inserted = by_label_position[(scheme_label, 0.0)]
        withdrawn = by_label_position[(scheme_label, 24.0)]
        crw = withdrawn["rho_combined"] - inserted["rho_combined"]
        crw_std = math.sqrt(
            withdrawn["rho_uncertainty"] ** 2 + inserted["rho_uncertainty"] ** 2
        )
        sdm = -inserted["rho_combined"]
        sdm_std = inserted["rho_uncertainty"]
        shutdown_status = (
            "positive_shutdown_margin"
            if inserted["rho_combined"] < 0.0 and inserted["keff_combined"] < 1.0
            else "absolute_shutdown_not_demonstrated"
        )
        literature_ratio = crw * 1000.0 / IAEA_LEU_CONTROL_ROD_WORTH_MK
        scheme_metrics[scheme_label] = {
            "dummy_position_ids": list(SCHEMES[scheme_label]),
            "position_0": inserted,
            "position_24": withdrawn,
            "control_rod_worth": crw,
            "control_rod_worth_std": crw_std,
            "control_rod_worth_mk": crw * 1000.0,
            "control_rod_worth_std_mk": crw_std * 1000.0,
            "control_rod_worth_pcm": crw * 100_000.0,
            "control_rod_worth_std_pcm": crw_std * 100_000.0,
            "shutdown_margin": sdm,
            "shutdown_margin_std": sdm_std,
            "shutdown_margin_mk": sdm * 1000.0,
            "shutdown_margin_std_mk": sdm_std * 1000.0,
            "shutdown_margin_pcm": sdm * 100_000.0,
            "shutdown_margin_std_pcm": sdm_std * 100_000.0,
            "shutdown_status": shutdown_status,
            "relative_control_rod_worth_reportable": bool(
                crw > 0.0
                and inserted["replicate_stability_status"] == "stable"
                and withdrawn["replicate_stability_status"] == "stable"
            ),
            "iaea_leu_worth_mk": IAEA_LEU_CONTROL_ROD_WORTH_MK,
            "worth_to_iaea_ratio": literature_ratio,
            "literature_order_of_magnitude_status": (
                "same_order_of_magnitude"
                if 0.5 <= literature_ratio <= 2.0
                else "different_order_requires_model_discussion"
            ),
        }

    ref = scheme_metrics["REFERENCE"]
    opt = scheme_metrics["OPTIMIZED"]
    delta_crw = opt["control_rod_worth"] - ref["control_rod_worth"]
    delta_crw_std = math.sqrt(
        opt["control_rod_worth_std"] ** 2 + ref["control_rod_worth_std"] ** 2
    )
    delta_sdm = opt["shutdown_margin"] - ref["shutdown_margin"]
    delta_sdm_std = math.sqrt(
        opt["shutdown_margin_std"] ** 2 + ref["shutdown_margin_std"] ** 2
    )
    control_worth_change_status = (
        "not_resolved" if abs(delta_crw) <= delta_crw_std else "resolved"
    )
    shutdown_margin_change_status = (
        "not_resolved" if abs(delta_sdm) <= delta_sdm_std else "resolved"
    )
    comparison = {
        "reference_control_rod_worth_mk": ref["control_rod_worth_mk"],
        "reference_control_rod_worth_std_mk": ref["control_rod_worth_std_mk"],
        "optimized_control_rod_worth_mk": opt["control_rod_worth_mk"],
        "optimized_control_rod_worth_std_mk": opt["control_rod_worth_std_mk"],
        "delta_CRW": delta_crw,
        "delta_CRW_std": delta_crw_std,
        "delta_CRW_mk": delta_crw * 1000.0,
        "delta_CRW_std_mk": delta_crw_std * 1000.0,
        "control_worth_change_status": control_worth_change_status,
        "reference_shutdown_margin_mk": ref["shutdown_margin_mk"],
        "reference_shutdown_margin_std_mk": ref["shutdown_margin_std_mk"],
        "optimized_shutdown_margin_mk": opt["shutdown_margin_mk"],
        "optimized_shutdown_margin_std_mk": opt["shutdown_margin_std_mk"],
        "delta_SDM": delta_sdm,
        "delta_SDM_std": delta_sdm_std,
        "delta_SDM_mk": delta_sdm * 1000.0,
        "delta_SDM_std_mk": delta_sdm_std * 1000.0,
        "shutdown_margin_change_status": shutdown_margin_change_status,
    }
    pd.DataFrame([comparison]).to_csv(COMPARISON_PATH, index=False, encoding="utf-8")

    table_rows = [
        (
            "k_eff, position 24 cm",
            "dimensionless",
            ref["position_24"]["keff_combined"],
            ref["position_24"]["keff_uncertainty"],
            opt["position_24"]["keff_combined"],
            opt["position_24"]["keff_uncertainty"],
        ),
        (
            "rho_out",
            "mk",
            ref["position_24"]["rho_mk_combined"],
            ref["position_24"]["rho_mk_uncertainty"],
            opt["position_24"]["rho_mk_combined"],
            opt["position_24"]["rho_mk_uncertainty"],
        ),
        (
            "k_eff, position 0 cm",
            "dimensionless",
            ref["position_0"]["keff_combined"],
            ref["position_0"]["keff_uncertainty"],
            opt["position_0"]["keff_combined"],
            opt["position_0"]["keff_uncertainty"],
        ),
        (
            "rho_in",
            "mk",
            ref["position_0"]["rho_mk_combined"],
            ref["position_0"]["rho_mk_uncertainty"],
            opt["position_0"]["rho_mk_combined"],
            opt["position_0"]["rho_mk_uncertainty"],
        ),
        (
            "Control rod worth",
            "mk",
            ref["control_rod_worth_mk"],
            ref["control_rod_worth_std_mk"],
            opt["control_rod_worth_mk"],
            opt["control_rod_worth_std_mk"],
        ),
        (
            "Shutdown margin",
            "mk",
            ref["shutdown_margin_mk"],
            ref["shutdown_margin_std_mk"],
            opt["shutdown_margin_mk"],
            opt["shutdown_margin_std_mk"],
        ),
    ]
    pd.DataFrame(
        table_rows,
        columns=[
            "quantity",
            "unit",
            "Reference arrangement",
            "Reference uncertainty",
            "Optimized arrangement",
            "Optimized uncertainty",
        ],
    ).to_csv(TABLE_5_5_PATH, index=False, encoding="utf-8")

    warnings = sorted({warning for item in results for warning in item["warnings"]})
    cd_warning_count = sum(item["cd106_negative_probability_warning"] for item in results)
    _write_json(
        WARNINGS_PATH,
        {
            "status": "recorded_nonfatal_openmc_nuclear_data_warning",
            "formal_run_count": 16,
            "runs_with_cd106_negative_probability_warning": cd_warning_count,
            "all_unique_warnings": warnings,
            "interpretation": (
                "The Cd106 negative probability-table warning is reported, not "
                "silently removed and not used to tune geometry or materials."
            ),
        },
    )

    source_hashes_after = _protected_source_hashes()
    sources_unchanged = source_hashes_after == source_hashes_before
    all_replicates_stable = all(
        item["replicate_stability_status"] == "stable" for item in position_summaries
    )
    all_entropy_stable = all(item["all_entropy_tails_stable"] for item in position_summaries)
    summary = {
        "task": "thesis_5_4_formal_control_rod_worth_and_shutdown_margin",
        "status": "completed" if sources_unchanged else "completed_but_source_hash_changed",
        "geometry_model_version": GEOMETRY_MODEL_VERSION,
        "model_variant": MODEL_VARIANT,
        "control_rod_axial_definition_version": (
            CONTROL_ROD_AXIAL_DEFINITION_VERSION
        ),
        "calculation_profile": CALCULATION_PROFILE,
        "particles": PARTICLES,
        "batches": BATCHES,
        "inactive": INACTIVE,
        "active_batches": ACTIVE_BATCHES,
        "temperature_k": TEMPERATURE_K,
        "base_seed": BASE_SEED,
        "schemes": {label: list(value) for label, value in SCHEMES.items()},
        "positions_cm": list(POSITIONS_CM),
        "replicates_per_scheme_position": 4,
        "planned_formal_openmc_runs": 16,
        "new_formal_openmc_runs_this_invocation": runtime_counter["new_runs"],
        "completed_formal_results": len(results),
        "all_16_completed": len(results) == 16,
        "position_11p25_final_runs": 0,
        "same_movable_component_used_at_both_positions": True,
        "water_only_withdrawn_result_used": False,
        "scheme_results": scheme_metrics,
        "comparison": comparison,
        "all_replicate_groups_statistically_stable": all_replicates_stable,
        "all_shannon_entropy_tails_stable": all_entropy_stable,
        "cd106_warning": {
            "runs_with_warning": cd_warning_count,
            "formal_run_count": 16,
            "warning_file": str(WARNINGS_PATH),
        },
        "literature_context": {
            "iaea_mnsr_sz_leu_calculated_total_worth_mk": (
                IAEA_LEU_CONTROL_ROD_WORTH_MK
            ),
            "usage": "order-of-magnitude comparison only; no fitting",
            "historical_heu_calculated_mk": 8.0,
            "historical_heu_experimental_mk": 7.1,
        },
        "absolute_keff_bias_note": (
            "Known positive absolute-keff bias is not shifted or tuned. Relative "
            "control-rod worth and absolute shutdown margin are evaluated separately."
        ),
        "preflight_file": str(PREFLIGHT_PATH),
        "replicates_file": str(REPLICATES_PATH),
        "position_summary_file": str(POSITION_SUMMARY_PATH),
        "comparison_file": str(COMPARISON_PATH),
        "table_5_5_candidate_file": str(TABLE_5_5_PATH),
        "warnings_file": str(WARNINGS_PATH),
        "formal_run_root": str(RUN_ROOT),
        "runtime_seconds_total_this_workflow": float(time.perf_counter() - workflow_started),
        "sum_openmc_runtime_seconds": float(
            sum(item["runtime_seconds"] for item in results)
        ),
        "protected_source_hashes_before": source_hashes_before,
        "protected_source_hashes_after": source_hashes_after,
        "protected_sources_unchanged": sources_unchanged,
        "ga_results_modified": False,
        "dummy_layouts_modified": False,
        "materials_modified": False,
        "other_core_geometry_modified": False,
        "preflight_status": preflight["status"],
    }
    _write_json(FINAL_SUMMARY_PATH, summary)
    return summary


def _existing_valid_case_count() -> int:
    count = 0
    for scheme_label, scheme in SCHEMES.items():
        for position_cm in POSITIONS_CM:
            for replicate in REPLICATES:
                identity = _case_identity(scheme_label, scheme, position_cm, replicate)
                result_path = _case_directory(
                    scheme_label, position_cm, replicate
                ) / "result.json"
                if _validate_cached_result(result_path, identity) is not None:
                    count += 1
    return count


def print_run_confirmation(existing_count: int) -> None:
    seed_rows = []
    for scheme_label, scheme in SCHEMES.items():
        for position_cm in POSITIONS_CM:
            for replicate in REPLICATES:
                seed_rows.append(
                    (
                        scheme_label,
                        position_cm,
                        replicate,
                        stable_seed(scheme_label, scheme, position_cm, replicate),
                    )
                )
    print("\n=== 正式控制棒计算运行前确认 ===", flush=True)
    print(f"运行目录: {RUN_ROOT}", flush=True)
    print(f"geometry_model_version: {GEOMETRY_MODEL_VERSION}", flush=True)
    print(f"model_variant: {MODEL_VARIANT}", flush=True)
    print(f"axial definition: {CONTROL_ROD_AXIAL_DEFINITION_VERSION}", flush=True)
    print(f"profile: {CALCULATION_PROFILE}", flush=True)
    print(
        f"particles={PARTICLES}, batches={BATCHES}, inactive={INACTIVE}, "
        f"temperature={TEMPERATURE_K} K",
        flush=True,
    )
    print(f"REFERENCE dummy IDs: {REFERENCE}", flush=True)
    print(f"OPTIMIZED dummy IDs: {OPTIMIZED}", flush=True)
    print("control rod positions: inserted component at 0 and 24 cm", flush=True)
    print("tally数量和名称: 0, []", flush=True)
    print(f"预计statepoint: {EXPECTED_STATEPOINT_NAME}", flush=True)
    print("旧water-only withdrawn缓存: 未读取", flush=True)
    print(f"有效本轮缓存: {existing_count}; 待新增运行: {16 - existing_count}", flush=True)
    print("SHA-256独立种子矩阵:", flush=True)
    for row in seed_rows:
        print(f"  {row[0]} pos={row[1]:g} R{row[2]}: {row[3]}", flush=True)
    print("=== 确认结束，开始单次16算例矩阵 ===\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只做四种正式几何组合的预检，不启动输运。",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="通过预检后执行或严格复用16个正式算例。",
    )
    args = parser.parse_args()
    if args.preflight_only == args.run:
        parser.error("必须且只能指定--preflight-only或--run。")

    workflow_started = time.perf_counter()
    source_hashes_before = _protected_source_hashes()
    context = load_model_context()
    preflight = run_preflight(context)
    print(f"正式预检：{preflight['status']}；详情：{PREFLIGHT_PATH}", flush=True)
    if args.preflight_only:
        print("预检模式未启动OpenMC输运。", flush=True)
        return

    existing_count = _existing_valid_case_count()
    new_needed = 16 - existing_count
    if not 0 <= new_needed <= MAX_NEW_CONTROL_ROD_FINAL_RUNS:
        raise RuntimeError("正式计算数量不满足0到16的硬限制。")
    print_run_confirmation(existing_count)
    runtime_counter = {"new_runs": 0}
    results = []
    for scheme_label, scheme in SCHEMES.items():
        for position_cm in POSITIONS_CM:
            for replicate in REPLICATES:
                results.append(
                    run_case(
                        context,
                        scheme_label,
                        scheme,
                        position_cm,
                        replicate,
                        runtime_counter,
                    )
                )
    if runtime_counter["new_runs"] != new_needed:
        raise RuntimeError("实际新增运行数与运行前缓存审计不一致。")
    summary = summarize_results(
        results,
        preflight,
        source_hashes_before,
        runtime_counter,
        workflow_started,
    )
    print("\n正式控制棒计算与聚合完成。", flush=True)
    print(f"实际新增OpenMC运行数: {runtime_counter['new_runs']}", flush=True)
    print(f"全部16个结果完成: {summary['all_16_completed']}", flush=True)
    print(f"汇总文件: {FINAL_SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
