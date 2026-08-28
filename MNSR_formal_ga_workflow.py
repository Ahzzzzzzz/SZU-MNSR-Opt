"""MNSR正式GA screening与前10候选final复核工作流。

本文件由MNSR_optimization_development.ipynb使用 ``%run -i`` 调用，复用Notebook中
已经通过pilot验证的遗传算子、几何构建器和OpenMC评价器。默认安全开关位于Notebook中。
"""

from __future__ import annotations

from functools import cmp_to_key
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORMAL_GA_ROOT = (REAL_GA_ROOT / "formal_primary").resolve()
FORMAL_CHECKPOINT_ROOT = (REAL_GA_CHECKPOINT_ROOT / "formal_primary").resolve()
FINALISTS_ROOT = (REAL_GA_ROOT / "finalists").resolve()
for _directory in (FORMAL_GA_ROOT, FORMAL_CHECKPOINT_ROOT, FINALISTS_ROOT):
    _directory.mkdir(parents=True, exist_ok=True)

FORMAL_CONFIGURATION = {
    "population_size": 20,
    "generations": 15,
    "elite_size": 2,
    "crossover_probability": 0.8,
    "mutation_probability": 0.2,
    "tournament_size": 3,
    "ga_seed": 20260728,
    "patience": None,
    "screening_profile": "screening",
    "particles": 5000,
    "batches": 60,
    "inactive": 15,
    "rho_min": None,
    "rho_max": None,
    "max_new_screening_openmc_cases": 280,
}
FINAL_CONFIGURATION = {
    "profile": "final",
    "particles": 20000,
    "batches": 150,
    "inactive": 50,
    "max_new_final_openmc_cases": 10,
}


def _json_string(values):
    return json.dumps([int(value) for value in values], ensure_ascii=False)


def _scheme_from_value(value):
    if isinstance(value, str):
        value = json.loads(value)
    value = [int(item) for item in value]
    return validate_individual(value)


def _write_stage_json(path, payload):
    Path(path).write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _print_formal_confirmation():
    print("FORMAL REAL GA CONFIRMATION")
    print("geometry_model_version =", REAL_GA_GEOMETRY_VERSION)
    print("model_variant =", REAL_GA_MODEL_VARIANT)
    print("control_rod_state =", REAL_GA_CONTROL_ROD_STATE)
    print("population_size = 20")
    print("generations = 15")
    print("elite_size = 2")
    print("crossover_probability = 0.8")
    print("mutation_probability = 0.2")
    print("tournament_size = 3")
    print("ga_seed = 20260728")
    print("screening particles = 5000")
    print("screening batches = 60")
    print("screening inactive = 15")
    print("rho_min = None")
    print("rho_max = None")
    print("fitness = Fr")
    print("pilot history reused = False")
    print("pilot physical cache reuse allowed = True")
    print("MAX_NEW_SCREENING_OPENMC_CASES = 280")
    print("mock evaluator = OFF")
    print("real OpenMC evaluator = ON")


def _pilot_cache_keys():
    path = REAL_GA_PILOT_ROOT / "pilot_unique_evaluations.csv"
    if not path.is_file():
        return set()
    table = pd.read_csv(path)
    return set(table["cache_key"].dropna().astype(str)) if "cache_key" in table else set()


def _formal_history_table(run):
    rows = []
    for item in run["generation_history"]:
        rows.append({
            "generation": int(item["generation"]),
            "population_size": int(item["population_size"]),
            "generation_best_fitness": float(item["best_fitness"]),
            "generation_mean_fitness": float(item["mean_fitness"]),
            "generation_median_fitness": float(item["median_fitness"]),
            "generation_worst_fitness": float(item["worst_fitness"]),
            "global_best_fitness": float(item["global_best_fitness"]),
            "generation_best_scheme": _json_string(item["best_scheme"]),
            "global_best_scheme": _json_string(item["global_best_scheme"]),
            "generation_best_Fr": item.get("best_Fr"),
            "generation_best_CV": item.get("best_CV"),
            "generation_best_keff": item.get("best_keff"),
            "generation_best_keff_std": item.get("best_keff_std"),
            "generation_best_rho_mk": item.get("best_rho_mk"),
            "new_openmc_runs": int(item["new_openmc_runs"]),
            "cache_hits": int(item["cache_hits"]),
            "failures": int(item["failures"]),
            "generation_wall_time": float(item["generation_wall_time"]),
        })
    return pd.DataFrame(rows)


def _formal_all_individuals_table(run):
    cache = run["evaluation_cache"]
    rows = []
    for item in run["all_individuals"]:
        scheme = _scheme_from_value(item["dummy_ids"])
        result = cache[",".join(map(str, scheme))]
        rows.append({
            "generation": int(item["generation"]),
            "individual_index": int(item["individual_index"]),
            "scheme_id": item["scheme_id"],
            "dummy_ids": _json_string(scheme),
            "fitness": float(item["fitness"]),
            "Fr": item.get("Fr"),
            "CV": item.get("CV"),
            "keff": item.get("keff"),
            "keff_std": item.get("keff_std"),
            "rho": result.get("reactivity", result.get("rho")),
            "rho_mk": item.get("rho_mk"),
            "max_power_position_id": result.get("max_power_position_id"),
            "mean_pin_relative_uncertainty": result.get("mean_pin_relative_uncertainty"),
            "cache_hit": bool(item.get("cache_hit", False)),
            "status": item.get("status"),
            "calculation_seconds": float(item.get("calculation_seconds", 0.0)),
        })
    return pd.DataFrame(rows)


def _formal_unique_table(run):
    rows = []
    for memory_key, result in run["evaluation_cache"].items():
        scheme = _scheme_from_value(memory_key.split(","))
        row = {
            "scheme_id": result.get("scheme_id", scheme_key(scheme)),
            "dummy_ids": _json_string(scheme),
            "cache_key": result.get("cache_key"),
            "profile": result.get("profile"),
            "openmc_seed": result.get("openmc_seed"),
            "Fr": result.get("radial_peaking_factor"),
            "CV": result.get("power_coefficient_of_variation"),
            "keff": result.get("keff"),
            "keff_std": result.get("keff_std"),
            "rho": result.get("reactivity", result.get("rho")),
            "rho_mk": result.get("rho_mk"),
            "max_power_position_id": result.get("max_power_position_id"),
            "mean_pin_relative_uncertainty": result.get("mean_pin_relative_uncertainty"),
            "median_pin_relative_uncertainty": result.get("median_pin_relative_uncertainty"),
            "p95_pin_relative_uncertainty": result.get("p95_pin_relative_uncertainty"),
            "max_pin_relative_uncertainty": result.get("max_pin_relative_uncertainty"),
            "entropy_tail_summary": json.dumps(result.get("entropy_tail_summary"), ensure_ascii=False),
            "calculation_seconds": result.get("calculation_seconds"),
            "cache_hit": bool(result.get("cache_hit", False)),
            "cache_source": result.get("cache_source"),
            "run_directory": result.get("run_directory"),
            "status": result.get("status"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _save_convergence_plot(history):
    figure, axis = plt.subplots(figsize=(10.5, 7.0), facecolor="white")
    generation = history["generation"].to_numpy(int)
    axis.plot(generation, history["generation_best_fitness"], marker="o", lw=2.0, label="Generation best")
    axis.plot(generation, history["generation_mean_fitness"], marker="s", lw=2.0, label="Generation mean")
    axis.plot(generation, history["global_best_fitness"], marker="^", lw=2.0, label="Global best")
    axis.set_xlabel("Generation", fontsize=15)
    axis.set_ylabel("Fitness / radial peaking factor", fontsize=15)
    axis.tick_params(labelsize=12, width=1.3, length=6)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=12, frameon=True)
    for spine in axis.spines.values():
        spine.set_linewidth(1.3)
    figure.tight_layout()
    figure.savefig(FORMAL_GA_ROOT / "ga_convergence_real.png", dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(FORMAL_GA_ROOT / "ga_convergence_real.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _rank_screening_candidates(unique_table):
    successful = unique_table.loc[unique_table["status"] == "completed"].copy()
    numeric = ["Fr", "CV", "keff", "keff_std", "rho", "rho_mk"]
    for column in numeric:
        successful[column] = pd.to_numeric(successful[column], errors="coerce")
    if successful.empty or not np.all(np.isfinite(successful["Fr"])):
        raise RuntimeError("正式GA没有完整、有限的screening Fr结果。")
    successful = successful.sort_values(["Fr", "CV", "scheme_id"], kind="mergesort").reset_index(drop=True)
    successful.insert(0, "rank", np.arange(1, len(successful) + 1))
    reference_row = successful.loc[successful["dummy_ids"] == _json_string(REFERENCE_DUMMY_IDS)]
    if len(reference_row) != 1:
        raise RuntimeError("唯一screening结果中缺少参考排布。")
    reference = reference_row.iloc[0]
    successful["delta_keff_vs_reference_screening"] = successful["keff"] - float(reference["keff"])
    successful["delta_rho_vs_reference_screening"] = successful["rho_mk"] - float(reference["rho_mk"])
    top20_columns = [
        "rank", "scheme_id", "dummy_ids", "Fr", "CV", "keff", "keff_std", "rho_mk",
        "delta_keff_vs_reference_screening", "delta_rho_vs_reference_screening",
        "max_power_position_id", "mean_pin_relative_uncertainty",
    ]
    successful.head(20)[top20_columns].to_csv(
        FORMAL_GA_ROOT / "top20_candidates_screening.csv", index=False, encoding="utf-8"
    )
    return successful, int(reference["rank"])


def run_formal_screening_ga(resume=True):
    """从独立generation 0运行或恢复正式20×15真实screening GA。"""
    _print_formal_confirmation()
    if tuple(REFERENCE_DUMMY_IDS) != (285, 298, 311, 324, 337):
        raise RuntimeError("参考方案已发生变化，拒绝正式运行。")
    if RUN_PROFILES["screening"] != {"particles": 5000, "batches": 60, "inactive": 15}:
        raise RuntimeError("screening配置与正式方案不一致。")
    checkpoint_path = FORMAL_CHECKPOINT_ROOT / "checkpoint.json"
    actual_resume = bool(resume and checkpoint_path.is_file())
    if actual_resume:
        print("发现配置匹配的formal_primary checkpoint，将从下一代恢复：", checkpoint_path)
    else:
        print("未读取pilot轨迹；正式GA从全新generation 0开始。")
    runtime_counter = {"new_screening_openmc_runs": 0, "new_final_openmc_runs": 0}

    def evaluator(individual):
        return evaluate_scheme_real(
            individual,
            profile_name="screening",
            force_rerun=False,
            run_openmc=True,
            allow_real_openmc=True,
            runtime_counter=runtime_counter,
            max_new_screening_cases=280,
            max_new_final_cases=10,
        )

    start = time.perf_counter()
    run = run_genetic_algorithm(
        evaluator,
        population_size=20,
        generations=15,
        elite_size=2,
        crossover_probability=0.8,
        mutation_probability=0.2,
        tournament_size=3,
        ga_seed=20260728,
        constraints={"rho_min": None, "rho_max": None, "penalty_weight": 10000.0, "failure_penalty": 1.0e6},
        checkpoint_directory=FORMAL_CHECKPOINT_ROOT,
        resume=actual_resume,
        evaluator_name="formal_real_openmc_screening",
        runtime_counter=runtime_counter,
    )
    run["stage_wall_time"] = float(time.perf_counter() - start)
    history = _formal_history_table(run)
    all_individuals = _formal_all_individuals_table(run)
    unique = _formal_unique_table(run)
    if len(history) != 15 or list(history["generation"]) != list(range(15)):
        raise RuntimeError("正式GA没有完整结束generation 0至14。")
    if len(all_individuals) != 300:
        raise RuntimeError(f"正式GA个体记录应为300行，实际{len(all_individuals)}行。")
    if int(run["new_openmc_runs_total"]) > 280:
        raise RuntimeError("正式GA新screening调用超过280次硬上限。")
    if not np.allclose(all_individuals.loc[all_individuals["status"] == "completed", "fitness"], all_individuals.loc[all_individuals["status"] == "completed", "Fr"], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("成功方案出现fitness不等于Fr。")
    history.to_csv(FORMAL_GA_ROOT / "formal_generation_history.csv", index=False, encoding="utf-8")
    all_individuals.to_csv(FORMAL_GA_ROOT / "formal_all_individuals.csv", index=False, encoding="utf-8")
    unique.to_csv(FORMAL_GA_ROOT / "formal_unique_evaluations.csv", index=False, encoding="utf-8")
    successful, reference_rank = _rank_screening_candidates(unique)
    _save_convergence_plot(history)
    pilot_keys = _pilot_cache_keys()
    reused_pilot = int(sum(key in pilot_keys and ids != _json_string(REFERENCE_DUMMY_IDS) for key, ids in zip(unique["cache_key"].astype(str), unique["dummy_ids"])))
    manifest = {
        "configuration": FORMAL_CONFIGURATION,
        "fresh_generation_zero": not actual_resume,
        "pilot_history_reused": False,
        "pilot_physical_cache_reuse_allowed": True,
        "actual_completed_generations": len(history),
        "total_population_evaluations": len(all_individuals),
        "unique_screening_schemes": len(unique),
        "reused_pilot_physics_caches": reused_pilot,
        "new_screening_openmc_runs": int(run["new_openmc_runs_total"]),
        "cache_hits": int(all_individuals["cache_hit"].sum()),
        "failures": int((all_individuals["status"] != "completed").sum()),
        "screening_wall_time": float(history["generation_wall_time"].sum()),
        "reference_screening_rank": reference_rank,
        "global_best_screening_Fr": float(successful.iloc[0]["Fr"]),
        "global_best_screening_scheme": json.loads(successful.iloc[0]["dummy_ids"]),
        "checkpoint_file": str(checkpoint_path),
    }
    _write_stage_json(FORMAL_GA_ROOT / "formal_screening_manifest.json", manifest)
    return {
        "run": run,
        "history": history,
        "all_individuals": all_individuals,
        "unique": unique,
        "ranked_screening": successful,
        "manifest": manifest,
    }


def _reference_final_result(screening_rank):
    directory = Path("build/results/reference/iaea_reference_v1/final").resolve()
    summary_path = directory / "reference_final_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"缺少参考方案final汇总：{summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pin_path = Path(summary["pin_power_file"])
    ring_path = Path(summary["ring_power_file"])
    if not all(path.is_file() for path in (pin_path, ring_path, directory / "statepoint.150.h5")):
        raise FileNotFoundError("参考方案final原始文件不完整。")
    pin = pd.read_csv(pin_path)
    uncertainty = summary["pin_power_relative_uncertainty"]
    return {
        "screening_rank": int(screening_rank),
        "scheme_id": scheme_key(REFERENCE_DUMMY_IDS),
        "dummy_ids": _json_string(REFERENCE_DUMMY_IDS),
        "is_reference": True,
        "cache_hit": True,
        "profile": "final",
        "particles": 20000,
        "batches": 150,
        "inactive": 50,
        "keff": float(summary["keff_mean"]),
        "keff_std": float(summary["keff_std"]),
        "rho": float(summary["reactivity"]),
        "rho_std": float(summary["reactivity_std"]),
        "rho_mk": float(summary["reactivity_mk"]),
        "rho_std_mk": float(summary["reactivity_std_mk"]),
        "Fr": float(summary["radial_peaking_factor"]),
        "CV": float(summary["power_coefficient_of_variation"]),
        "maximum_relative_pin_power": float(pin["relative_power"].max()),
        "max_power_position_id": int(summary["max_power_position_id"]),
        "minimum_relative_pin_power": float(pin["relative_power"].min()),
        "mean_pin_relative_uncertainty": float(uncertainty["mean"]),
        "median_pin_relative_uncertainty": float(uncertainty["median"]),
        "p95_pin_relative_uncertainty": float(uncertainty["p95"]),
        "max_pin_relative_uncertainty": float(uncertainty["maximum"]),
        "pin_power_file": str(pin_path),
        "ring_power_file": str(ring_path),
        "run_directory": str(directory),
        "calculation_seconds": 0.0,
        "status": "completed",
    }


def _augment_candidate_final(result, screening_rank):
    if result.get("status") != "completed":
        return {"screening_rank": int(screening_rank), **result}
    pin_path = Path(result["pin_power_file"])
    pin = pd.read_csv(pin_path)
    ring_column = "ring_index" if "ring_index" in pin.columns else "ring"
    ring = pin.groupby(ring_column, as_index=False).agg(
        fuel_count=("relative_power", "size"),
        mean_relative_power=("relative_power", "mean"),
        max_relative_power=("relative_power", "max"),
        min_relative_power=("relative_power", "min"),
        standard_deviation=("relative_power", lambda values: float(np.std(values, ddof=0))),
    ).rename(columns={ring_column: "ring"})
    ring_path = Path(result["run_directory"]) / "ring_power.csv"
    ring.to_csv(ring_path, index=False, encoding="utf-8")
    keff, keff_std = float(result["keff"]), float(result["keff_std"])
    rho_std = keff_std / keff**2
    augmented = {
        **result,
        "screening_rank": int(screening_rank),
        "dummy_ids": _json_string(result["normalized_dummy_ids"]),
        "is_reference": False,
        "rho": float(result["reactivity"]),
        "rho_std": float(rho_std),
        "rho_std_mk": float(rho_std * 1000.0),
        "Fr": float(result["radial_peaking_factor"]),
        "CV": float(result["power_coefficient_of_variation"]),
        "maximum_relative_pin_power": float(pin["relative_power"].max()),
        "minimum_relative_pin_power": float(pin["relative_power"].min()),
        "ring_power_file": str(ring_path.resolve()),
    }
    result_path = Path(result["run_directory"]) / "result.json"
    _write_stage_json(result_path, augmented)
    return augmented


def _final_compare(left, right):
    delta_fr = float(left["Fr"]) - float(right["Fr"])
    if abs(delta_fr) >= 1.0e-4:
        return -1 if delta_fr < 0.0 else 1
    delta_cv = float(left["CV"]) - float(right["CV"])
    if not math.isclose(delta_cv, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        return -1 if delta_cv < 0.0 else 1
    left_rho = abs(float(left["delta_rho_mk"]))
    right_rho = abs(float(right["delta_rho_mk"]))
    return -1 if left_rho < right_rho else (1 if left_rho > right_rho else 0)


def run_finalist_review(ranked_screening):
    """复核screening前10；参考方案只读取已有final，不重新计算。"""
    top10 = ranked_screening.head(10).copy()
    reference_screening = ranked_screening.loc[
        ranked_screening["dummy_ids"] == _json_string(REFERENCE_DUMMY_IDS)
    ].iloc[0]
    reference_final = _reference_final_result(int(reference_screening["rank"]))
    runtime_counter = {"new_screening_openmc_runs": 0, "new_final_openmc_runs": 0}
    reviewed = []
    start = time.perf_counter()
    for _, screening_row in top10.iterrows():
        scheme = _scheme_from_value(screening_row["dummy_ids"])
        if scheme == tuple(REFERENCE_DUMMY_IDS):
            reviewed.append(dict(reference_final))
            continue
        result = evaluate_scheme_real(
            scheme,
            profile_name="final",
            force_rerun=False,
            run_openmc=True,
            allow_real_openmc=True,
            runtime_counter=runtime_counter,
            max_new_screening_cases=280,
            max_new_final_cases=10,
        )
        reviewed.append(_augment_candidate_final(result, int(screening_row["rank"])))
        if result.get("status") != "completed":
            raise RuntimeError(f"finalist {scheme} 计算失败：{result.get('error_message')}")
    if not any(item["is_reference"] for item in reviewed):
        reviewed.append(dict(reference_final))
    final_wall_time = float(time.perf_counter() - start)
    for item in reviewed:
        item["delta_keff"] = float(item["keff"]) - float(reference_final["keff"])
        item["delta_rho_mk"] = float(item["rho_mk"]) - float(reference_final["rho_mk"])
        item["Fr_improvement"] = float(reference_final["Fr"]) - float(item["Fr"])
        item["Fr_improvement_percent"] = item["Fr_improvement"] / float(reference_final["Fr"]) * 100.0
        item["CV_improvement"] = float(reference_final["CV"]) - float(item["CV"])
        item["CV_improvement_percent"] = item["CV_improvement"] / float(reference_final["CV"]) * 100.0
        screening_row = ranked_screening.loc[ranked_screening["scheme_id"] == item["scheme_id"]]
        item["screening_Fr"] = float(screening_row.iloc[0]["Fr"]) if len(screening_row) else np.nan
    ranked_final = sorted(reviewed, key=cmp_to_key(_final_compare))
    for index, item in enumerate(ranked_final, start=1):
        item["final_rank"] = index
    final_table = pd.DataFrame(ranked_final)
    desired = [
        "screening_rank", "final_rank", "scheme_id", "dummy_ids", "is_reference", "keff", "keff_std",
        "rho", "rho_std", "rho_mk", "rho_std_mk", "Fr", "CV", "maximum_relative_pin_power",
        "max_power_position_id", "minimum_relative_pin_power", "mean_pin_relative_uncertainty",
        "median_pin_relative_uncertainty", "p95_pin_relative_uncertainty", "max_pin_relative_uncertainty",
        "delta_keff", "delta_rho_mk", "Fr_improvement", "Fr_improvement_percent",
        "CV_improvement", "CV_improvement_percent", "screening_Fr", "cache_hit", "run_directory",
        "pin_power_file", "ring_power_file", "status", "calculation_seconds",
    ]
    final_table[[column for column in desired if column in final_table.columns]].to_csv(
        FORMAL_GA_ROOT / "top10_candidates_final.csv", index=False, encoding="utf-8"
    )
    comparison = final_table[["screening_rank", "final_rank", "dummy_ids", "screening_Fr", "Fr"]].copy()
    comparison.columns = ["screening_rank", "final_rank", "scheme", "screening_Fr", "final_Fr"]
    comparison["rank_change"] = comparison["final_rank"] - comparison["screening_rank"]
    comparison.to_csv(FORMAL_GA_ROOT / "screening_vs_final_ranking.csv", index=False, encoding="utf-8")
    best = ranked_final[0]
    improvement = float(reference_final["Fr"]) - float(best["Fr"])
    optimization_status = "verified_improvement" if improvement > 0.0 and not best["is_reference"] else "no_verified_improvement"
    combined_typical_uncertainty = math.sqrt(
        float(reference_final["mean_pin_relative_uncertainty"]) ** 2
        + float(best["mean_pin_relative_uncertainty"]) ** 2
    )
    relative_improvement = max(0.0, improvement / float(reference_final["Fr"]))
    optimization_significance = (
        "supported"
        if optimization_status == "verified_improvement" and relative_improvement > 2.0 * combined_typical_uncertainty
        else "needs_additional_check"
    )
    return {
        "ranked_final": ranked_final,
        "final_table": final_table,
        "reference_final": reference_final,
        "best": best,
        "final_openmc_runs": int(runtime_counter["new_final_openmc_runs"]),
        "final_wall_time": final_wall_time,
        "optimization_status": optimization_status,
        "optimization_significance": optimization_significance,
        "screening_final_rank_changed": bool(np.any(comparison["rank_change"] != 0)),
        "screening_first_final_rank": int(comparison.loc[comparison["screening_rank"] == 1, "final_rank"].iloc[0]),
    }


def _best_scheme_spatial_data(best_scheme):
    scheme = _scheme_from_value(best_scheme)
    rows = ordinary_positions_df.set_index("ordinary_position_id").loc[list(scheme)].reset_index().copy()
    rows["radius_cm"] = np.hypot(rows["x_cm"], rows["y_cm"])
    rows["theta_deg"] = np.mod(np.degrees(np.arctan2(rows["y_cm"], rows["x_cm"])), 360.0)
    output = rows[[
        "ordinary_position_id", "physical_position_id", "ring_index", "local_index",
        "radius_cm", "theta_deg", "x_cm", "y_cm",
    ]].rename(columns={"ring_index": "ring"})
    output.to_csv(FORMAL_GA_ROOT / "best_scheme_positions.csv", index=False, encoding="utf-8")
    xy = output[["x_cm", "y_cm"]].to_numpy(float)
    pairwise = [
        {"id_a": int(output.iloc[i]["ordinary_position_id"]), "id_b": int(output.iloc[j]["ordinary_position_id"]), "distance_cm": float(np.linalg.norm(xy[i] - xy[j]))}
        for i in range(5) for j in range(i + 1, 5)
    ]
    angles = np.sort(output["theta_deg"].to_numpy(float))
    gaps = np.diff(np.r_[angles, angles[0] + 360.0])
    metrics = {
        "rings": sorted(set(int(value) for value in output["ring"])),
        "mean_radius_cm": float(output["radius_cm"].mean()),
        "minimum_radius_cm": float(output["radius_cm"].min()),
        "maximum_radius_cm": float(output["radius_cm"].max()),
        "pairwise_distances": pairwise,
        "nearest_dummy_spacing_cm": float(min(item["distance_cm"] for item in pairwise)),
        "azimuthal_gaps_deg": [float(value) for value in gaps],
    }
    _write_stage_json(FORMAL_GA_ROOT / "best_scheme_spatial_metrics.json", metrics)
    return output, metrics


def _save_comparison_figures(screening, reference_final, best):
    best_scheme = _scheme_from_value(best["dummy_ids"])
    reference_scheme = tuple(REFERENCE_DUMMY_IDS)
    positions = pd.read_csv("build/position_table.csv")
    ordinary = positions.loc[positions["position_category"] == "ordinary"].copy()
    tie_rods = positions.loc[positions["position_category"] == "tie_rod"]
    center = positions.loc[positions["position_category"] == "central"]

    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True, facecolor="white")
    for axis, scheme, label in zip(axes, (reference_scheme, best_scheme), ("Reference", "Optimized")):
        dummy_mask = ordinary["ordinary_position_id"].astype(int).isin(scheme)
        axis.scatter(ordinary.loc[~dummy_mask, "x_cm"], ordinary.loc[~dummy_mask, "y_cm"], s=12, c="#d95f02", label="Fuel positions")
        axis.scatter(ordinary.loc[dummy_mask, "x_cm"], ordinary.loc[dummy_mask, "y_cm"], s=75, c="#1b9e77", marker="s", label="Dummy elements")
        axis.scatter(tie_rods["x_cm"], tie_rods["y_cm"], s=90, c="#7570b3", marker="D", label="Tie rods")
        axis.scatter(center["x_cm"], center["y_cm"], s=110, facecolors="none", edgecolors="black", linewidths=1.8, label="Control guide tube")
        axis.set_title(label, fontsize=17)
        axis.set_xlabel("x (cm)", fontsize=14)
        axis.set_ylabel("y (cm)", fontsize=14)
        axis.set_aspect("equal")
        axis.set_xlim(-12, 12)
        axis.set_ylim(-12, 12)
        axis.grid(alpha=0.15)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11)
    figure.tight_layout()
    figure.savefig(FORMAL_GA_ROOT / "reference_vs_optimized_layout.png", dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(FORMAL_GA_ROOT / "reference_vs_optimized_layout.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    reference_pin = pd.read_csv(reference_final["pin_power_file"])
    optimized_pin = pd.read_csv(best["pin_power_file"])
    value_min = min(reference_pin["relative_power"].min(), optimized_pin["relative_power"].min())
    value_max = max(reference_pin["relative_power"].max(), optimized_pin["relative_power"].max())
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True, facecolor="white")
    scatter = None
    for axis, table, label in zip(axes, (reference_pin, optimized_pin), ("Reference", "Optimized")):
        scatter = axis.scatter(table["x_cm"], table["y_cm"], c=table["relative_power"], s=34, cmap="viridis", vmin=value_min, vmax=value_max)
        axis.set_title(label, fontsize=17)
        axis.set_xlabel("x (cm)", fontsize=14)
        axis.set_ylabel("y (cm)", fontsize=14)
        axis.set_aspect("equal")
        axis.grid(alpha=0.12)
    figure.colorbar(scatter, ax=axes, label="Relative pin power", fraction=0.035, pad=0.04)
    figure.savefig(FORMAL_GA_ROOT / "reference_vs_optimized_pin_power.png", dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(FORMAL_GA_ROOT / "reference_vs_optimized_pin_power.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    reference_ring = pd.read_csv(reference_final["ring_power_file"])
    optimized_ring = pd.read_csv(best["ring_power_file"])
    figure, axis = plt.subplots(figsize=(10.5, 7.0), facecolor="white")
    axis.plot(reference_ring["ring"], reference_ring["mean_relative_power"], marker="o", lw=2.0, label="Reference")
    axis.plot(optimized_ring["ring"], optimized_ring["mean_relative_power"], marker="s", lw=2.0, label="Optimized")
    axis.axhline(1.0, color="black", ls="--", lw=1.5, label="Core average = 1")
    axis.set_xlabel("Ring", fontsize=15)
    axis.set_ylabel("Mean relative pin power", fontsize=15)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=12)
    figure.tight_layout()
    figure.savefig(FORMAL_GA_ROOT / "reference_vs_optimized_ring_power.png", dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(FORMAL_GA_ROOT / "reference_vs_optimized_ring_power.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    reference_screening = screening.loc[screening["dummy_ids"] == _json_string(reference_scheme)].iloc[0]
    best_screening = screening.iloc[0]
    best_verified_screening = screening.loc[screening["scheme_id"] == best["scheme_id"]].iloc[0]
    figure, axis = plt.subplots(figsize=(10.5, 7.0), facecolor="white")
    axis.scatter(screening["delta_rho_vs_reference_screening"], screening["Fr"], s=24, alpha=0.6, color="#4c78a8", label="Unique screening schemes")
    axis.scatter([0.0], [reference_screening["Fr"]], s=110, marker="*", color="black", label="Reference", zorder=4)
    axis.scatter([best_screening["delta_rho_vs_reference_screening"]], [best_screening["Fr"]], s=85, marker="D", color="#e45756", label="Best screening", zorder=4)
    axis.scatter([best_verified_screening["delta_rho_vs_reference_screening"]], [best_verified_screening["Fr"]], s=95, marker="s", facecolors="none", edgecolors="#54a24b", linewidths=2.0, label="Best verified", zorder=4)
    axis.set_xlabel(r"$\Delta\rho$ relative to reference (mk)", fontsize=15)
    axis.set_ylabel("Screening radial peaking factor", fontsize=15)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=11)
    figure.tight_layout()
    figure.savefig(FORMAL_GA_ROOT / "fr_vs_delta_rho_screening.png", dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(FORMAL_GA_ROOT / "fr_vs_delta_rho_screening.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _save_formal_summary(screening_stage, final_stage):
    history = screening_stage["history"]
    screening = screening_stage["ranked_screening"]
    manifest = screening_stage["manifest"]
    reference_final = final_stage["reference_final"]
    best = final_stage["best"]
    spatial_metrics = None
    figure_files = []
    if final_stage["optimization_status"] == "verified_improvement":
        _, spatial_metrics = _best_scheme_spatial_data(best["dummy_ids"])
        _save_comparison_figures(screening, reference_final, best)
        figure_files = [
            str(FORMAL_GA_ROOT / name)
            for stem in (
                "reference_vs_optimized_layout", "reference_vs_optimized_pin_power",
                "reference_vs_optimized_ring_power", "fr_vs_delta_rho_screening",
            )
            for name in (f"{stem}.png", f"{stem}.pdf")
        ]
    summary = {
        "population": 20,
        "generations": 15,
        "elite": 2,
        "crossover": 0.8,
        "mutation": 0.2,
        "tournament": 3,
        "ga_seed": 20260728,
        "total_population_evaluations": int(manifest["total_population_evaluations"]),
        "unique_screening_schemes": int(manifest["unique_screening_schemes"]),
        "reused_pilot_physics_caches": int(manifest["reused_pilot_physics_caches"]),
        "new_screening_openmc_runs": int(manifest["new_screening_openmc_runs"]),
        "cache_hits": int(manifest["cache_hits"]),
        "failures": int(manifest["failures"]),
        "screening_wall_time": float(manifest["screening_wall_time"]),
        "final_openmc_runs": int(final_stage["final_openmc_runs"]),
        "final_wall_time": float(final_stage["final_wall_time"]),
        "reference_screening_rank": int(manifest["reference_screening_rank"]),
        "initial_best_Fr": float(history.iloc[0]["generation_best_Fr"]),
        "final_generation_best_Fr": float(history.iloc[-1]["generation_best_Fr"]),
        "global_best_screening_Fr": float(manifest["global_best_screening_Fr"]),
        "global_best_screening_scheme": manifest["global_best_screening_scheme"],
        "best_verified_scheme": json.loads(best["dummy_ids"]),
        "best_verified_Fr": float(best["Fr"]),
        "best_verified_CV": float(best["CV"]),
        "best_verified_keff": float(best["keff"]),
        "best_verified_keff_std": float(best["keff_std"]),
        "best_verified_rho_mk": float(best["rho_mk"]),
        "best_verified_rho_std_mk": float(best["rho_std_mk"]),
        "reference_final_Fr": float(reference_final["Fr"]),
        "reference_final_CV": float(reference_final["CV"]),
        "reference_final_keff": float(reference_final["keff"]),
        "reference_final_rho_mk": float(reference_final["rho_mk"]),
        "Fr_improvement_percent": float(best["Fr_improvement_percent"]),
        "CV_improvement_percent": float(best["CV_improvement_percent"]),
        "delta_keff": float(best["delta_keff"]),
        "delta_rho_mk": float(best["delta_rho_mk"]),
        "screening_final_rank_changed": bool(final_stage["screening_final_rank_changed"]),
        "screening_first_final_rank": int(final_stage["screening_first_final_rank"]),
        "optimization_status": final_stage["optimization_status"],
        "optimization_significance": final_stage["optimization_significance"],
        "recommend_second_independent_ga_seed": final_stage["optimization_status"] == "no_verified_improvement",
        "best_scheme_spatial_metrics": spatial_metrics,
        "paper_candidate_figures": figure_files,
        "control_rod_inserted_runs": 0,
        "physics_model_modified": False,
    }
    _write_stage_json(FORMAL_GA_ROOT / "formal_ga_summary.json", summary)
    flat = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in summary.items()}
    pd.DataFrame([flat]).to_csv(FORMAL_GA_ROOT / "formal_ga_summary.csv", index=False, encoding="utf-8")
    return summary


def execute_formal_workflow(run_finalists=True, resume=True):
    """执行正式GA，并在明确授权时复核screening前10候选。"""
    if not RUN_FORMAL_REAL_GA:
        raise PermissionError("RUN_FORMAL_REAL_GA=False，未授权正式GA。")
    screening_stage = run_formal_screening_ga(resume=resume)
    if not run_finalists:
        print("正式screening GA已完成；RUN_FINALIST_REVIEW=False，停止在候选排序。")
        return {"screening": screening_stage, "final": None, "summary": None}
    if not RUN_FINALIST_REVIEW:
        raise PermissionError("RUN_FINALIST_REVIEW=False，未授权finalist复核。")
    final_stage = run_finalist_review(screening_stage["ranked_screening"])
    summary = _save_formal_summary(screening_stage, final_stage)
    print("正式GA与finalist复核完成：", FORMAL_GA_ROOT)
    return {"screening": screening_stage, "final": final_stage, "summary": summary}
