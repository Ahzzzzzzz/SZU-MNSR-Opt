"""Six-scheme independent final-replicate statistical confirmation workflow.

Run from ``MNSR_optimization_development.ipynb`` with ``%run -i``.  This module
does not call the genetic algorithm and does not create flux, spectrum, or
inserted-control-rod models.
"""

from __future__ import annotations

from collections import Counter
import contextlib
import hashlib
import io
import json
import math
from numbers import Real
from pathlib import Path
import time
from typing import cast
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import openmc
import pandas as pd


CONFIRMATION_ROOT = (REAL_GA_ROOT / "statistical_confirmation").resolve()
CONFIRMATION_ROOT.mkdir(parents=True, exist_ok=True)
CONFIRMATION_BASE_SEED = 20260817
MAX_NEW_CONFIRMATION_OPENMC_RUNS = 18
CONFIRMATION_PROFILE = "final_confirmation"
CONFIRMATION_SCHEMES = {
    "REFERENCE": (285, 298, 311, 324, 337),
    "C1": (200, 213, 223, 239, 256),
    "C2": (102, 200, 213, 237, 256),
    "C3": (102, 200, 213, 223, 256),
    "C4": (102, 153, 200, 237, 281),
    "C5": (102, 178, 200, 223, 256),
}
ORIGINAL_BEST_SCHEME = CONFIRMATION_SCHEMES["C1"]
CONFIRMATION_COUNTER = {"new_openmc_runs": 0, "failures": 0}


def _confirmation_json(path, payload):
    Path(path).write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_confirmation_seed(normalized_dummy_ids, replicate_index):
    """Return a deterministic positive OpenMC seed for replicate 1, 2, or 3."""
    scheme = validate_individual(normalized_dummy_ids)
    if isinstance(replicate_index, bool) or int(replicate_index) not in {1, 2, 3}:
        raise ValueError("replicate_index只允许1、2或3。")
    payload = {
        "base_seed": CONFIRMATION_BASE_SEED,
        "geometry_model_version": REAL_GA_GEOMETRY_VERSION,
        "profile": CONFIRMATION_PROFILE,
        "replicate_index": int(replicate_index),
        "normalized_dummy_ids": list(scheme),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return int(digest, 16) % 2_147_483_646 + 1


def _settings_values(settings_path):
    root = ET.parse(settings_path).getroot()
    return {
        "particles": int(root.findtext("particles")),
        "batches": int(root.findtext("batches")),
        "inactive": int(root.findtext("inactive")),
        "seed": int(root.findtext("seed")),
    }


def _read_statepoint_power(statepoint_path):
    with openmc.StatePoint(statepoint_path) as statepoint:
        tally = statepoint.get_tally(name="fuel_pin_kappa_fission")
        raw_mean = np.asarray(tally.mean, dtype=float).reshape(-1)
        raw_std = np.asarray(tally.std_dev, dtype=float).reshape(-1)
        keff = statepoint.keff
        keff_mean = float(keff.nominal_value)
        keff_std = float(keff.std_dev)
        generation_values = np.asarray(statepoint.k_generation, dtype=float).reshape(-1)
        try:
            entropy_values = np.asarray(statepoint.entropy, dtype=float).reshape(-1)
        except (AttributeError, KeyError):
            entropy_values = None
    if entropy_values is None:
        with h5py.File(statepoint_path, "r") as statepoint_h5:
            entropy_values = np.asarray(statepoint_h5["entropy"][...], dtype=float).reshape(-1)
    if raw_mean.size != 345 or raw_std.size != 345:
        raise ValueError(f"statepoint燃料棒Tally不是345项：{statepoint_path}")
    return raw_mean, raw_std, keff_mean, keff_std, generation_values, entropy_values


def _standard_replicate_result(
    label,
    scheme,
    replicate,
    seed,
    run_directory,
    statepoint_path,
    pin_path,
    calculation_seconds,
    cache_hit=False,
):
    pin = pd.read_csv(pin_path).sort_values("ordinary_position_id").reset_index(drop=True)
    raw_mean, raw_std, keff, keff_std, generation_values, entropy_values = _read_statepoint_power(statepoint_path)
    if len(pin) != 345:
        raise ValueError(f"{label} R{replicate}棒功率CSV不是345行。")
    if not np.allclose(raw_mean, pin["kappa_fission_mean"].to_numpy(float), rtol=1.0e-12, atol=0.0):
        raise ValueError(f"{label} R{replicate} statepoint与pin_power原始均值不一致。")
    if not np.allclose(raw_std, pin["kappa_fission_std"].to_numpy(float), rtol=1.0e-12, atol=0.0):
        raise ValueError(f"{label} R{replicate} statepoint与pin_power原始标准差不一致。")
    relative_power = raw_mean / float(np.mean(raw_mean))
    relative_uncertainty = raw_std / raw_mean
    max_index = int(np.argmax(relative_power))
    min_index = int(np.argmin(relative_power))
    rho = (keff - 1.0) / keff
    rho_std = keff_std / keff**2
    entropy_summary = _entropy_tail_summary(entropy_values, 50)
    return {
        "scheme_label": label,
        "scheme": list(scheme),
        "scheme_id": scheme_key(scheme),
        "replicate": int(replicate),
        "openmc_seed": int(seed),
        "keff": keff,
        "keff_std": keff_std,
        "rho": float(rho),
        "rho_std": float(rho_std),
        "rho_mk": float(rho * 1000.0),
        "rho_std_mk": float(rho_std * 1000.0),
        "Fr": float(np.max(relative_power)),
        "CV": float(np.std(relative_power, ddof=0)),
        "maximum_relative_pin_power": float(relative_power[max_index]),
        "maximum_power_position_id": int(pin.iloc[max_index]["ordinary_position_id"]),
        "minimum_relative_pin_power": float(relative_power[min_index]),
        "minimum_power_position_id": int(pin.iloc[min_index]["ordinary_position_id"]),
        "mean_pin_relative_uncertainty": float(np.mean(relative_uncertainty)),
        "median_pin_relative_uncertainty": float(np.median(relative_uncertainty)),
        "p95_pin_relative_uncertainty": float(np.percentile(relative_uncertainty, 95)),
        "max_pin_relative_uncertainty": float(np.max(relative_uncertainty)),
        "calculation_seconds": float(calculation_seconds),
        "entropy_stability_status": "stable" if entropy_summary["heuristically_stable"] else "needs_review",
        "entropy_tail_summary": entropy_summary,
        "run_directory": str(Path(run_directory).resolve()),
        "statepoint_file": str(Path(statepoint_path).resolve()),
        "pin_power_file": str(Path(pin_path).resolve()),
        "cache_hit": bool(cache_hit),
        "status": "completed",
    }, pin


def _existing_final_sources():
    table_path = REAL_GA_ROOT / "formal_primary" / "top10_candidates_final.csv"
    if not table_path.is_file():
        raise FileNotFoundError(f"找不到已有final排名：{table_path}")
    table = pd.read_csv(table_path)
    sources = {}
    for label, scheme in CONFIRMATION_SCHEMES.items():
        scheme_text = json.dumps(list(scheme))
        row = table.loc[table["dummy_ids"] == scheme_text]
        if len(row) != 1:
            raise RuntimeError(f"已有final排名中无法唯一定位{label}：{scheme}")
        row = row.iloc[0]
        run_directory = Path(row["run_directory"]).resolve()
        settings = _settings_values(run_directory / "settings.xml")
        if settings["particles"] != 20000 or settings["batches"] != 150 or settings["inactive"] != 50:
            raise RuntimeError(f"{label}已有final配置不一致：{settings}")
        statepoint_path = run_directory / "statepoint.150.h5"
        pin_path = Path(row["pin_power_file"]).resolve()
        required = (statepoint_path, pin_path, run_directory / "settings.xml", run_directory / "openmc_output.txt")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{label}已有final文件缺失：{missing}")
        result, pin = _standard_replicate_result(
            label,
            scheme,
            0,
            settings["seed"],
            run_directory,
            statepoint_path,
            pin_path,
            0.0,
            cache_hit=True,
        )
        result["existing_screening_rank"] = int(row["screening_rank"])
        result["existing_final_rank"] = int(row["final_rank"])
        sources[label] = {"result": result, "pin": pin}
    return sources


def _confirmation_identity(label, scheme, replicate, seed, existing_seed):
    try:
        cross_sections = Path(openmc.config["cross_sections"]).expanduser().resolve()
    except KeyError as error:
        raise FileNotFoundError("尚未配置OpenMC截面库。") from error
    if not cross_sections.is_file():
        raise FileNotFoundError(f"OpenMC截面库不存在：{cross_sections}")
    return {
        "scheme_label": label,
        "normalized_dummy_ids": list(scheme),
        "scheme_id": scheme_key(scheme),
        "replicate": int(replicate),
        "profile": CONFIRMATION_PROFILE,
        "geometry_model_version": REAL_GA_GEOMETRY_VERSION,
        "model_variant": REAL_GA_MODEL_VARIANT,
        "control_rod_state": REAL_GA_CONTROL_ROD_STATE,
        "particles": 20000,
        "batches": 150,
        "inactive": 50,
        "openmc_seed": int(seed),
        "existing_replicate_0_seed": int(existing_seed),
        "tally_version": REAL_GA_TALLY_VERSION,
        "material_model_version": REAL_GA_MATERIAL_VERSION,
        "materials_notebook_sha256": _sha256_file(materials_notebook_path),
        "geometry_notebook_sha256": _sha256_file(geometry_notebook_path),
        "cross_sections": str(cross_sections),
        "cross_sections_sha256": _sha256_file(cross_sections),
        "openmc_version": openmc.__version__,
        "seed_method": "SHA-256(base seed, geometry version, profile, replicate index, normalized IDs)",
    }


def evaluate_confirmation_replicate(label, scheme, replicate, existing_seed, allow_openmc=False):
    """Run or safely reuse one independent final_confirmation replicate."""
    scheme = validate_individual(scheme)
    if replicate not in {1, 2, 3}:
        raise ValueError("新增replicate只允许1、2或3。")
    seed = get_confirmation_seed(scheme, replicate)
    if seed == int(existing_seed):
        raise RuntimeError(f"{label} R{replicate}种子与已有R0重复。")
    scheme_directory = "reference" if label == "REFERENCE" else scheme_key(scheme)
    run_directory = CONFIRMATION_ROOT / scheme_directory / f"replicate_{replicate}"
    result_path = run_directory / "result.json"
    manifest_path = run_directory / "run_manifest.json"
    identity = _confirmation_identity(label, scheme, replicate, seed, existing_seed)
    if result_path.is_file() and manifest_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached.get("status") == "completed" and manifest.get("identity") == identity:
            cached["cache_hit"] = True
            cached["calculation_seconds"] = 0.0
            pin = pd.read_csv(cached["pin_power_file"])
            return cached, pin
        raise RuntimeError(f"{run_directory}存在不匹配或不完整的旧确认结果。")
    if not allow_openmc:
        raise PermissionError("统计确认OpenMC安全开关未开启。")
    if CONFIRMATION_COUNTER["new_openmc_runs"] >= MAX_NEW_CONFIRMATION_OPENMC_RUNS:
        raise RealGASafetyLimitError("新增统计确认OpenMC调用将超过18次硬上限。")
    run_directory.mkdir(parents=True, exist_ok=True)
    _confirmation_json(manifest_path, {"identity": identity, "status": "prepared"})
    start = time.perf_counter()
    output_buffer = io.StringIO()
    output_path = run_directory / "openmc_output.txt"
    try:
        geometry, metadata = build_geometry_for_scheme(
            scheme,
            control_rod_state="withdrawn",
            export_directory=None,
            create_plots=False,
            model_variant="iaea_reference",
            top_be_thickness_cm=0.0,
        )
        if metadata.get("geometry_model_version") != "iaea_reference_v1":
            raise RuntimeError(f"{label}确认几何版本错误。")
        expected_counts = {"fuel": 345, "dummy": 5, "tie_rod": 4, "central": 1}
        if any(int(metadata["counts"][key]) != value for key, value in expected_counts.items()):
            raise RuntimeError(f"{label}确认几何构件数量错误：{metadata['counts']}")
        materials_for_run, aliases = _prepare_real_ga_materials(
            geometry, Path(identity["cross_sections"]), 2
        )
        materials_for_run.cross_sections = identity["cross_sections"]
        settings = create_settings("final", seed)
        entropy_mesh = openmc.RegularMesh(name="Confirmation entropy mesh")
        entropy_mesh.lower_left = (-11.5, -11.5, -11.5)
        entropy_mesh.upper_right = (11.5, 11.5, 11.5)
        entropy_mesh.dimension = (10, 10, 10)
        settings.entropy_mesh = entropy_mesh
        tallies, fuel_ids = create_power_tally(metadata["fuel_meat_cells"])
        if len(tallies) != 1 or tallies[0].name != "fuel_pin_kappa_fission":
            raise RuntimeError("统计确认只能包含345棒kappa-fission Tally。")
        model = openmc.Model(
            materials=materials_for_run,
            geometry=geometry,
            settings=settings,
            tallies=tallies,
        )
        model.export_to_xml(directory=run_directory)
        CONFIRMATION_COUNTER["new_openmc_runs"] += 1
        from unittest.mock import PropertyMock, patch
        try:
            with (
                contextlib.redirect_stdout(output_buffer),
                contextlib.redirect_stderr(output_buffer),
                patch.object(openmc.Model, "is_initialized", new_callable=PropertyMock, return_value=False),
            ):
                returned = model.run(
                    cwd=run_directory,
                    openmc_exec=_find_real_openmc_executable(),
                    export_model_xml=False,
                )
        finally:
            output_path.write_text(output_buffer.getvalue(), encoding="utf-8")
        statepoint_path = Path(returned) if returned is not None else run_directory / "statepoint.150.h5"
        if not statepoint_path.is_absolute():
            statepoint_path = run_directory / statepoint_path
        raw_mean, raw_std, _, _, generation_values, entropy_values = _read_statepoint_power(statepoint_path)
        if generation_values.size != 150 or entropy_values.size != 150:
            raise ValueError("统计确认逐批keff或Shannon熵不是150项。")
        if np.any(raw_mean <= 0.0) or not all(np.all(np.isfinite(values)) for values in (raw_mean, raw_std, generation_values, entropy_values)):
            raise ValueError("统计确认statepoint包含非正或非有限数据。")
        relative_power = raw_mean / float(np.mean(raw_mean))
        relative_uncertainty = raw_std / raw_mean
        positions = ordinary_positions_df[[
            "ordinary_position_id", "physical_position_id", "ring_index",
            "local_index", "x_cm", "y_cm",
        ]]
        pin_path = run_directory / "pin_power.csv"
        pd.DataFrame({
            "ordinary_position_id": fuel_ids,
            "kappa_fission_mean": raw_mean,
            "kappa_fission_std": raw_std,
            "relative_power": relative_power,
            "relative_uncertainty": relative_uncertainty,
        }).merge(positions, on="ordinary_position_id", how="left", validate="one_to_one").to_csv(
            pin_path, index=False, encoding="utf-8"
        )
        phase = np.where(np.arange(1, 151) <= 50, "inactive", "active")
        generation_path = run_directory / "generation_history.csv"
        entropy_path = run_directory / "entropy_history.csv"
        pd.DataFrame({"batch": np.arange(1, 151), "phase": phase, "k_generation": generation_values}).to_csv(
            generation_path, index=False, encoding="utf-8"
        )
        pd.DataFrame({"batch": np.arange(1, 151), "phase": phase, "shannon_entropy": entropy_values}).to_csv(
            entropy_path, index=False, encoding="utf-8"
        )
        result, pin = _standard_replicate_result(
            label,
            scheme,
            replicate,
            seed,
            run_directory,
            statepoint_path,
            pin_path,
            time.perf_counter() - start,
            cache_hit=False,
        )
        result.update({
            "generation_history_file": str(generation_path.resolve()),
            "entropy_history_file": str(entropy_path.resolve()),
            "openmc_output_file": str(output_path.resolve()),
            "run_manifest_file": str(manifest_path.resolve()),
            "thermal_scattering_aliases": aliases,
        })
        _confirmation_json(result_path, result)
        _confirmation_json(manifest_path, {"identity": identity, "status": "completed", "result_file": str(result_path.resolve())})
        return result, pin
    except Exception as error:
        CONFIRMATION_COUNTER["failures"] += 1
        output_path.write_text(output_buffer.getvalue(), encoding="utf-8")
        failure = {
            "identity": identity,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "calculation_seconds": float(time.perf_counter() - start),
        }
        _confirmation_json(run_directory / "error.json", failure)
        _confirmation_json(manifest_path, failure)
        raise RuntimeError(f"{label} R{replicate}统计确认失败：{error}") from error


def _pool_scheme_pin_tallies(label, replicate_pins):
    ordered = []
    expected_ids = None
    for replicate in range(4):
        pin = replicate_pins[replicate].sort_values("ordinary_position_id").reset_index(drop=True)
        ids = tuple(int(value) for value in pin["ordinary_position_id"])
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise RuntimeError(f"{label}不同replicate的345根燃料棒ID顺序不一致。")
        ordered.append(pin)
    raw_means = np.vstack([pin["kappa_fission_mean"].to_numpy(float) for pin in ordered])
    raw_stds = np.vstack([pin["kappa_fission_std"].to_numpy(float) for pin in ordered])
    pooled_raw_mean = np.mean(raw_means, axis=0)
    pooled_raw_std = np.sqrt(np.sum(raw_stds**2, axis=0)) / 4.0
    relative_power = pooled_raw_mean / float(np.mean(pooled_raw_mean))
    relative_uncertainty = pooled_raw_std / pooled_raw_mean
    mapping_columns = [
        column for column in (
            "ordinary_position_id", "physical_position_id", "ring", "ring_index",
            "local_index", "x_cm", "y_cm",
        ) if column in ordered[0].columns
    ]
    pooled = ordered[0][mapping_columns].copy()
    pooled["pooled_kappa_fission_mean"] = pooled_raw_mean
    pooled["pooled_kappa_fission_std"] = pooled_raw_std
    pooled["relative_power"] = relative_power
    pooled["relative_uncertainty"] = relative_uncertainty
    scheme_dir = CONFIRMATION_ROOT / ("reference" if label == "REFERENCE" else scheme_key(CONFIRMATION_SCHEMES[label]))
    scheme_dir.mkdir(parents=True, exist_ok=True)
    pooled_path = scheme_dir / "pooled_pin_power.csv"
    pooled.to_csv(pooled_path, index=False, encoding="utf-8")
    max_index = int(np.argmax(relative_power))
    min_index = int(np.argmin(relative_power))
    replicate_mean_uncertainty = np.mean([
        float(np.mean(pin["kappa_fission_std"].to_numpy(float) / pin["kappa_fission_mean"].to_numpy(float)))
        for pin in ordered
    ])
    return {
        "Fr_pooled": float(relative_power[max_index]),
        "CV_pooled": float(np.std(relative_power, ddof=0)),
        "maximum_power_position_pooled": int(pooled.iloc[max_index]["ordinary_position_id"]),
        "minimum_power_position_pooled": int(pooled.iloc[min_index]["ordinary_position_id"]),
        "pooled_peak_pin_relative_uncertainty": float(relative_uncertainty[max_index]),
        "pooled_mean_pin_uncertainty": float(np.mean(relative_uncertainty)),
        "pooled_median_pin_uncertainty": float(np.median(relative_uncertainty)),
        "pooled_p95_pin_uncertainty": float(np.percentile(relative_uncertainty, 95)),
        "pooled_max_pin_uncertainty": float(np.max(relative_uncertainty)),
        "mean_single_replicate_pin_uncertainty": float(replicate_mean_uncertainty),
        "uncertainty_reduction_ratio": float(np.mean(relative_uncertainty) / replicate_mean_uncertainty),
        "pooled_pin_power_file": str(pooled_path.resolve()),
    }


def _confirmation_analysis(all_results, all_pins, total_wall_time):
    replicate_table = pd.DataFrame(all_results).sort_values(["scheme_label", "replicate"])
    replicate_table["scheme"] = replicate_table["scheme"].apply(json.dumps)
    replicate_path = CONFIRMATION_ROOT / "confirmation_replicates.csv"
    replicate_table.to_csv(replicate_path, index=False, encoding="utf-8")

    t_critical_df3_975 = 3.182446305
    scheme_rows = []
    for label, scheme in CONFIRMATION_SCHEMES.items():
        rows = replicate_table.loc[replicate_table["scheme_label"] == label].sort_values("replicate")
        fr = rows["Fr"].to_numpy(float)
        cv = rows["CV"].to_numpy(float)
        keff = rows["keff"].to_numpy(float)
        rho_mk = rows["rho_mk"].to_numpy(float)
        pooled = _pool_scheme_pin_tallies(
            label, {replicate: all_pins[(label, replicate)] for replicate in range(4)}
        )
        max_positions = [int(value) for value in rows["maximum_power_position_id"]]
        max_counts = Counter(max_positions)
        max_mode, max_count = max_counts.most_common(1)[0]
        fr_std = float(np.std(fr, ddof=1))
        fr_ci_half = t_critical_df3_975 * fr_std / math.sqrt(4.0)
        median = float(np.median(fr))
        mad = float(np.median(np.abs(fr - median)))
        outlier_threshold = max(5.0 * mad, 0.05)
        replicate_outlier_flag = bool(np.any(np.abs(fr - median) > outlier_threshold))
        row = {
            "scheme_label": label,
            "scheme": json.dumps(list(scheme)),
            "scheme_id": scheme_key(scheme),
            "existing_screening_rank": int(rows.iloc[0]["existing_screening_rank"]),
            "existing_final_rank": int(rows.iloc[0]["existing_final_rank"]),
            **{f"Fr_R{index}": float(fr[index]) for index in range(4)},
            "Fr_mean": float(np.mean(fr)),
            "Fr_std": fr_std,
            "Fr_median": median,
            "Fr_min": float(np.min(fr)),
            "Fr_max": float(np.max(fr)),
            "Fr_t95_ci_lower": float(np.mean(fr) - fr_ci_half),
            "Fr_t95_ci_upper": float(np.mean(fr) + fr_ci_half),
            "CV_mean": float(np.mean(cv)),
            "CV_std": float(np.std(cv, ddof=1)),
            "keff_mean": float(np.mean(keff)),
            "keff_std_between_replicates": float(np.std(keff, ddof=1)),
            "rho_mk_mean": float(np.mean(rho_mk)),
            "rho_mk_std_between_replicates": float(np.std(rho_mk, ddof=1)),
            "max_pin_mode": int(max_mode),
            "max_pin_stability_fraction": float(max_count / 4.0),
            "max_pin_positions": json.dumps(max_positions),
            "replicate_outlier_flag": replicate_outlier_flag,
            **pooled,
        }
        scheme_rows.append(row)
    scheme_summary = pd.DataFrame(scheme_rows)
    reference = scheme_summary.loc[scheme_summary["scheme_label"] == "REFERENCE"].iloc[0]
    for index, row in scheme_summary.iterrows():
        scheme_summary.loc[index, "delta_Fr_pooled"] = float(row["Fr_pooled"] - reference["Fr_pooled"])
        scheme_summary.loc[index, "Fr_improvement_pooled"] = float(reference["Fr_pooled"] - row["Fr_pooled"])
        scheme_summary.loc[index, "Fr_improvement_percent_pooled"] = float(
            (reference["Fr_pooled"] - row["Fr_pooled"]) / reference["Fr_pooled"] * 100.0
        )
        scheme_summary.loc[index, "CV_improvement_percent_pooled"] = float(
            (reference["CV_pooled"] - row["CV_pooled"]) / reference["CV_pooled"] * 100.0
        )
        scheme_summary.loc[index, "delta_keff_pooled"] = float(row["keff_mean"] - reference["keff_mean"])
        scheme_summary.loc[index, "delta_rho_mk_pooled"] = float(row["rho_mk_mean"] - reference["rho_mk_mean"])
        if row["scheme_label"] == "REFERENCE":
            lower_count = 0
        else:
            candidate_fr = replicate_table.loc[replicate_table["scheme_label"] == row["scheme_label"]].sort_values("replicate")["Fr"].to_numpy(float)
            reference_fr = replicate_table.loc[replicate_table["scheme_label"] == "REFERENCE"].sort_values("replicate")["Fr"].to_numpy(float)
            lower_count = int(np.sum(candidate_fr < reference_fr))
        scheme_summary.loc[index, "replicates_below_reference_count"] = lower_count
    scheme_summary = scheme_summary.sort_values("Fr_pooled").reset_index(drop=True)
    scheme_summary.insert(0, "pooled_rank", np.arange(1, len(scheme_summary) + 1))
    scheme_summary.to_csv(
        CONFIRMATION_ROOT / "confirmation_scheme_summary.csv", index=False, encoding="utf-8"
    )

    rank_rows = []
    for _, row in scheme_summary.iterrows():
        record = {
            "scheme_label": row["scheme_label"],
            "scheme": row["scheme"],
            "original_screening_rank": int(row["existing_screening_rank"]),
            "original_final_rank": int(row["existing_final_rank"]),
            "pooled_rank": int(row["pooled_rank"]),
        }
        for replicate in range(4):
            replicate_subset = replicate_table.loc[replicate_table["replicate"] == replicate].copy()
            replicate_subset["replicate_rank"] = replicate_subset["Fr"].rank(method="min", ascending=True).astype(int)
            record[f"replicate_{replicate}_rank"] = int(
                replicate_subset.loc[replicate_subset["scheme_label"] == row["scheme_label"], "replicate_rank"].iloc[0]
            )
        record["max_pin_mode"] = int(row["max_pin_mode"])
        record["max_pin_stability_fraction"] = float(row["max_pin_stability_fraction"])
        rank_rows.append(record)
    rank_table = pd.DataFrame(rank_rows).sort_values("pooled_rank")
    rank_table.to_csv(
        CONFIRMATION_ROOT / "confirmation_rank_stability.csv", index=False, encoding="utf-8"
    )

    best = scheme_summary.iloc[0]
    reference = scheme_summary.loc[scheme_summary["scheme_label"] == "REFERENCE"].iloc[0]
    original_best = scheme_summary.loc[scheme_summary["scheme_label"] == "C1"].iloc[0]
    if best["Fr_pooled"] >= reference["Fr_pooled"]:
        confirmation_status = "no_confirmed_Fr_improvement"
    else:
        improvement = float(reference["Fr_pooled"] - best["Fr_pooled"])
        combined_peak_standard_error = math.sqrt(
            (float(reference["Fr_pooled"]) * float(reference["pooled_peak_pin_relative_uncertainty"])) ** 2
            + (float(best["Fr_pooled"]) * float(best["pooled_peak_pin_relative_uncertainty"])) ** 2
        )
        basic_conditions = (
            float(best["CV_pooled"]) < float(reference["CV_pooled"])
            and int(best["replicates_below_reference_count"]) >= 3
            and not bool(best["replicate_outlier_flag"])
        )
        confirmation_status = (
            "confirmed_advantage"
            if basic_conditions and improvement > 1.96 * combined_peak_standard_error
            else "advantage_but_statistically_weak"
        )
    recommended_for_thesis = confirmation_status != "no_confirmed_Fr_improvement"
    rank_changes = {
        row["scheme_label"]: {
            "original_final_rank": int(row["existing_final_rank"]),
            "pooled_rank": int(row["pooled_rank"]),
        }
        for _, row in scheme_summary.iterrows()
    }
    summary = {
        "schemes_checked": [list(scheme) for scheme in CONFIRMATION_SCHEMES.values()],
        "scheme_labels": list(CONFIRMATION_SCHEMES),
        "replicates_per_scheme": 4,
        "existing_final_replicates_reused": 6,
        "new_openmc_runs": int(CONFIRMATION_COUNTER["new_openmc_runs"]),
        "failures": int(CONFIRMATION_COUNTER["failures"]),
        "total_wall_time": float(total_wall_time),
        "reference_pooled_Fr": float(reference["Fr_pooled"]),
        "reference_pooled_CV": float(reference["CV_pooled"]),
        "original_best_scheme": list(ORIGINAL_BEST_SCHEME),
        "original_best_pooled_Fr": float(original_best["Fr_pooled"]),
        "original_best_pooled_CV": float(original_best["CV_pooled"]),
        "best_confirmed_scheme": json.loads(best["scheme"]),
        "best_confirmed_scheme_label": best["scheme_label"],
        "best_confirmed_pooled_Fr": float(best["Fr_pooled"]),
        "best_confirmed_pooled_CV": float(best["CV_pooled"]),
        "Fr_improvement_percent_pooled": float(best["Fr_improvement_percent_pooled"]),
        "CV_improvement_percent_pooled": float(best["CV_improvement_percent_pooled"]),
        "delta_keff": float(best["delta_keff_pooled"]),
        "delta_rho_mk": float(best["delta_rho_mk_pooled"]),
        "original_best_remained_first": bool(best["scheme_label"] == "C1"),
        "rank_changes": rank_changes,
        "reference_pooled_mean_pin_uncertainty": float(reference["pooled_mean_pin_uncertainty"]),
        "best_pooled_mean_pin_uncertainty": float(best["pooled_mean_pin_uncertainty"]),
        "reference_uncertainty_reduction_ratio": float(reference["uncertainty_reduction_ratio"]),
        "best_uncertainty_reduction_ratio": float(best["uncertainty_reduction_ratio"]),
        "confirmation_status": confirmation_status,
        "recommended_for_thesis": bool(recommended_for_thesis),
        "thesis_wording": (
            "在当前统计精度下表现出较优的功率平坦化特征"
            if confirmation_status == "advantage_but_statistically_weak"
            else None
        ),
        "final_scheme_lock_recommendation": bool(recommended_for_thesis),
        "pooling_formula": "P_i,pooled=mean(P_i,r); sigma_i,pooled=sqrt(sum_r sigma_i,r^2)/4",
        "t_interval_note": "基于4次独立重复和df=3 t分布的辅助区间，不作为严格工程置信区间。",
        "replicate_rank_note": "不同方案的replicate seed彼此独立，按replicate编号比较仅用于描述性稳定性检查，不视为配对样本。",
        "ga_runs": 0,
        "control_rod_inserted_runs": 0,
        "flux_mesh_runs": 0,
        "energy_spectrum_runs": 0,
        "physics_model_modified": False,
    }
    _confirmation_json(CONFIRMATION_ROOT / "statistical_confirmation_summary.json", summary)
    flat = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        for key, value in summary.items()
    }
    pd.DataFrame([flat]).to_csv(
        CONFIRMATION_ROOT / "statistical_confirmation_summary.csv", index=False, encoding="utf-8"
    )
    _confirmation_json(CONFIRMATION_ROOT / "pooling_method.json", {
        "raw_tally_source": "fuel_pin_kappa_fission from four statepoint files",
        "equal_history_weighting": True,
        "mean_formula": "mean of four raw kappa-fission means for each fuel pin",
        "standard_error_formula": "sqrt(sum of four raw variance estimates) / 4",
        "normalization": "divide pooled raw pin means by their 345-pin arithmetic mean",
    })
    return {
        "replicates": replicate_table,
        "scheme_summary": scheme_summary,
        "rank_stability": rank_table,
        "summary": summary,
    }


def execute_statistical_confirmation():
    """Reuse six R0 finals, run exactly 18 new replicates, and pool raw tallies."""
    if not RUN_STATISTICAL_CONFIRMATION:
        raise PermissionError("RUN_STATISTICAL_CONFIRMATION=False，未授权统计确认。")
    if RUN_FORMAL_REAL_GA or RUN_FINALIST_REVIEW:
        raise RuntimeError("统计确认阶段禁止同时开启GA或finalist工作流。")
    print("STATISTICAL CONFIRMATION RUN CONFIRMATION")
    print("schemes =", CONFIRMATION_SCHEMES)
    print("existing final reused as replicate 0 = 6")
    print("new replicates per scheme = 3")
    print("new OpenMC run hard limit = 18")
    print("particles / batches / inactive = 20000 / 150 / 50")
    print("geometry / model / rod state = iaea_reference_v1 / iaea_reference / withdrawn")
    print("GA = OFF; inserted rod = OFF; flux = OFF; spectrum = OFF")
    existing = _existing_final_sources()
    seed_records = []
    for label, scheme in CONFIRMATION_SCHEMES.items():
        r0_seed = int(existing[label]["result"]["openmc_seed"])
        seeds = [get_confirmation_seed(scheme, replicate) for replicate in (1, 2, 3)]
        if len(set(seeds)) != 3 or r0_seed in seeds:
            raise RuntimeError(f"{label}确认种子不独立：R0={r0_seed}, R1-R3={seeds}")
        for replicate, seed in enumerate(seeds, start=1):
            seed_records.append({"scheme_label": label, "scheme": json.dumps(list(scheme)), "replicate": replicate, "openmc_seed": seed, "replicate_0_seed": r0_seed})
    if len({record["openmc_seed"] for record in seed_records}) != 18:
        raise RuntimeError("18个新增确认seed不是全局唯一。")
    pd.DataFrame(seed_records).to_csv(
        CONFIRMATION_ROOT / "confirmation_seeds.csv", index=False, encoding="utf-8"
    )
    start = time.perf_counter()
    all_results = []
    all_pins = {}
    for label, scheme in CONFIRMATION_SCHEMES.items():
        r0_result = dict(existing[label]["result"])
        r0_result["existing_screening_rank"] = int(existing[label]["result"]["existing_screening_rank"])
        r0_result["existing_final_rank"] = int(existing[label]["result"]["existing_final_rank"])
        all_results.append(r0_result)
        all_pins[(label, 0)] = existing[label]["pin"]
        for replicate in (1, 2, 3):
            result, pin = evaluate_confirmation_replicate(
                label,
                scheme,
                replicate,
                existing[label]["result"]["openmc_seed"],
                allow_openmc=True,
            )
            result["existing_screening_rank"] = int(existing[label]["result"]["existing_screening_rank"])
            result["existing_final_rank"] = int(existing[label]["result"]["existing_final_rank"])
            all_results.append(result)
            all_pins[(label, replicate)] = pin
    if CONFIRMATION_COUNTER["new_openmc_runs"] != 18:
        raise RuntimeError(f"新增确认OpenMC运行应严格为18，实际{CONFIRMATION_COUNTER['new_openmc_runs']}。")
    if CONFIRMATION_COUNTER["failures"] != 0:
        raise RuntimeError("统计确认存在失败算例，停止pooled判定。")
    analysis = _confirmation_analysis(all_results, all_pins, time.perf_counter() - start)
    print("统计确认完成：", CONFIRMATION_ROOT)
    return analysis
