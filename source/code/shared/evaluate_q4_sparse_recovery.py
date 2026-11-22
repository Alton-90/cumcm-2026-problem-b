"""Frozen, paired offline Q4 comparison: v2 recovery versus v3 sparse fallback."""

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from offline_arena import OfflineArena, Source
from search_policy_recovery import run_search as run_v2
from search_policy_recovery_sparse import (AXIAL_STEP_M, LATERAL_VALUES_M, VERSION,
                                           _clear_points, run_search as run_v3)
from simulator_client import SimulatorClient

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "Q4" / "experiments" / "round3"
ROUND2_CASES = ROOT / "results" / "Q4" / "experiments" / "round2" / "cases.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _case_from_record(record):
    return {**{key: value for key, value in record.items() if key != "sources"},
            "sources": [Source(**source) for source in record["sources"]]}


def _new_holdout_cases():
    """The round-2 declared-case structure, independently regenerated for seeds 51000..51023."""
    cases = []
    for index, seed in enumerate(range(51000, 51024)):
        rng = random.Random(seed)
        count = (10, 13, 16)[index % 3]
        channels = rng.sample(range(1, 21), count)
        direction_count = (0, 1, count - 1, count, count // 2, count)[index % 6]
        sources = []
        for j, channel in enumerate(channels):
            angle = rng.random() * 2.0 * math.pi
            radius_from_origin = 1800.0 * math.sqrt(rng.random())
            x, y = radius_from_origin * math.cos(angle), radius_from_origin * math.sin(angle)
            family = index // 6
            if family == 1:
                angle = 2.0 * math.pi * j / count
                x, y = 1800.0 * math.cos(angle), 1800.0 * math.sin(angle)
            elif family == 2:
                grid = [(xx, yy) for xx in range(-1800, 1801, 600) for yy in range(-1800, 1801, 600)
                        if xx * xx + yy * yy <= 1800 ** 2]
                x, y = grid[j % len(grid)]
            elif family == 3 and j < 3:
                x, y = ((0.0, 0.0), (5.0, 0.0), (20.0, 0.0))[j]
            reception = 1000.0 if index % 3 == 0 else 1500.0 if index % 3 == 1 else rng.uniform(1000, 1500)
            heading = (math.degrees(angle) if family == 1 else rng.uniform(0, 360)) if j < direction_count else None
            sources.append(Source(channel, float(x), float(y), reception, heading))
        cases.append({"case_id": f"holdout_q4_{seed}", "seed": seed, "split": "holdout",
                      "sources": sources, "error_mode": ("fixed_hash", "positive_edge", "negative_edge")[index % 3]})
    return cases


def _frozen_cases():
    development = [_case_from_record({**record, "split": "development"})
                   for record in json.loads(ROUND2_CASES.read_text(encoding="utf-8"))]
    return development + _new_holdout_cases()


def _serializable(cases):
    return [{**{key: value for key, value in case.items() if key != "sources"},
             "sources": [asdict(source) for source in case["sources"]]} for case in cases]


def _geometry_check():
    points = _clear_points((0.0, 0.0), 0.0)
    expected = [(float(t), q) for q in LATERAL_VALUES_M for t in range(0, 1501, 30)]
    samples_t = list(range(0, 1501, 15))
    samples_q = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
    worst = max(min(math.dist((t, q), p) for p in points) for t in samples_t for q in samples_q)
    theoretical = math.sqrt(15.0 ** 2 + 10.0 ** 2)
    return {"point_count": len(points), "first": list(points[0]), "last": list(points[-1]),
            "expected_point_count": 204, "endpoint_grid_matches": set(points) == set(expected),
            "sample_worst_nearest_distance_m": worst, "theoretical_bound_m": theoretical,
            "bound_below_20m": theoretical < 20.0}


def _run_policy(name, policy, cases):
    rows = []
    for case in cases:
        arena = OfflineArena(case["sources"], case["seed"], case["error_mode"])
        client = SimulatorClient("synthetic-team", transport=arena, log_path=False)
        started = time.perf_counter()
        result, error = {}, None
        try:
            client.enter()
            result = policy(client, question=4, method="main")
        except Exception as caught:
            error = repr(caught)
        finally:
            if client.entered and not client.ended:
                client.exit()
        elapsed = time.perf_counter() - started
        cleared = len(arena.cleared)
        calculated = (arena.distance_m / 5.0 + 5.0 * arena.measure_count + arena.switch_count
                      + 3.0 * arena.clear_count + 2.0 * cleared)
        violations = []
        if error or not result.get("completed"): violations.append("execution_not_completed")
        if cleared != len(case["sources"]): violations.append("sources_remain")
        if set(result.get("cleared_channels", [])) != arena.cleared: violations.append("clear_evidence_mismatch")
        excluded = set(result.get("excluded_channels", []))
        if excluded & set(arena.sources): violations.append("false_exclusion")
        if excluded | arena.cleared != set(range(1, 21)): violations.append("channel_accounting")
        if abs(calculated - arena.virtual_time_s) > 1e-6 or abs(client.virtual_time_s - arena.virtual_time_s) > 1e-6:
            violations.append("time_accounting")
        if result.get("extra_bearing_count", 0) > 8 * len(case["sources"]): violations.append("extra_measure_budget")
        rows.append({"case_id": case["case_id"], "split": case["split"], "source_count": len(case["sources"]),
                     "cleared_count": cleared, "virtual_time_s": arena.virtual_time_s,
                     "mean_time_per_source_s": None if not cleared else arena.virtual_time_s / cleared,
                     "clear_attempts": arena.clear_count, "fallback_count": result.get("fallback_count", 0),
                     "fallback_clear_count": result.get("fallback_clear_count", 0), "measurements": arena.measure_count,
                     "actual_execution_s": elapsed, "policy": result, "error": error, "violations": violations})
    _save(OUT / "metrics" / f"{name}.json", {"scope": "SYNTHETIC_NOT_OFFICIAL", "cases": rows})
    return rows


def _summary(rows):
    return {"cases": len(rows), "mean_time_per_source_s": statistics.mean(row["mean_time_per_source_s"] for row in rows),
            "clear_attempts": sum(row["clear_attempts"] for row in rows),
            "fallback_count": sum(row["fallback_count"] for row in rows),
            "fallback_clear_count": sum(row["fallback_clear_count"] for row in rows),
            "virtual_time_s": sum(row["virtual_time_s"] for row in rows)}


def evaluate():
    cases = _frozen_cases()
    frozen = _serializable(cases)
    case_path = OUT / "cases.json"
    criteria = {"both_methods_all_cases_completed_and_accounted": True,
                "candidate_all_mean_time_per_source_not_increase": True,
                "candidate_holdout_mean_time_per_source_not_increase": True,
                "candidate_each_case_time_ratio_at_most": 1.10,
                "development_cases": 36, "new_holdout_cases": 24,
                "sparse_grid": {"axial_step_m": 30, "lateral_values_m": list(LATERAL_VALUES_M), "point_count": 204}}
    manifest_path = OUT / "preflight.json"
    if case_path.exists():
        assert json.loads(case_path.read_text(encoding="utf-8")) == frozen, "Round-3 cases are frozen."
    else:
        _save(case_path, frozen)
    manifest = {"scope": "SYNTHETIC_NOT_OFFICIAL", "criteria": criteria,
                "inputs": [_ref(case_path), _ref(ROUND2_CASES), _ref(ROOT / "code" / "shared" / "search_policy_recovery.py"),
                           _ref(ROOT / "code" / "shared" / "search_policy_recovery_sparse.py"),
                           _ref(ROOT / "code" / "shared" / "offline_arena.py"), _ref(Path(__file__).resolve())]}
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest, "Round-3 preflight inputs changed."
    else:
        _save(manifest_path, manifest)
    geometry = _geometry_check()
    assert geometry["point_count"] == 204 and geometry["endpoint_grid_matches"] and geometry["bound_below_20m"]
    v2, v3 = _run_policy("recovery_v2", run_v2, cases), _run_policy("recovery_v3_sparse30", run_v3, cases)
    comparisons = []
    for old, new in zip(v2, v3):
        assert old["case_id"] == new["case_id"]
        comparisons.append({"case_id": old["case_id"], "split": old["split"],
                            "time_ratio_v3_to_v2": new["virtual_time_s"] / old["virtual_time_s"],
                            "v2_time_s": old["virtual_time_s"], "v3_time_s": new["virtual_time_s"]})
    summaries = {}
    for split in ("all", "development", "holdout"):
        old = v2 if split == "all" else [row for row in v2 if row["split"] == split]
        new = v3 if split == "all" else [row for row in v3 if row["split"] == split]
        summaries[split] = {"v2": _summary(old), "v3": _summary(new),
                            "mean_time_ratio_v3_to_v2": _summary(new)["mean_time_per_source_s"] / _summary(old)["mean_time_per_source_s"]}
    constraints_ok = all(not row["violations"] for rows in (v2, v3) for row in rows)
    per_case_ok = all(row["time_ratio_v3_to_v2"] <= 1.10 for row in comparisons)
    means_ok = (summaries["all"]["v3"]["mean_time_per_source_s"] <= summaries["all"]["v2"]["mean_time_per_source_s"]
                and summaries["holdout"]["v3"]["mean_time_per_source_s"] <= summaries["holdout"]["v2"]["mean_time_per_source_s"])
    accepted = constraints_ok and per_case_ok and means_ok and geometry["bound_below_20m"]
    comparison = {"scope": "SYNTHETIC_NOT_OFFICIAL", "candidate": VERSION, "accepted_by_predeclared_numeric_criteria": accepted,
                  "all_cases_constraint_pass": constraints_ok, "means_not_increase": means_ok,
                  "max_case_time_ratio": max(row["time_ratio_v3_to_v2"] for row in comparisons),
                  "regressions_over_10_percent": [row for row in comparisons if row["time_ratio_v3_to_v2"] > 1.10],
                  "geometry": geometry, "summaries": summaries, "case_comparisons": comparisons,
                  "inputs": manifest["inputs"], "criteria": criteria,
                  "limitations": ["Synthetic arena and holdout are not official simulator evidence.",
                                  "Source truth is held only by OfflineArena; policies receive interface responses.",
                                  "No formal test or network interface was called."]}
    _save(OUT / "comparison.json", comparison)
    brief = ("# Q4 round-3 sparse fallback comparison\n\n"
             f"Scope: synthetic offline only. Candidate: `{VERSION}`.\n\n"
             f"Acceptance: `{accepted}`. Max v3/v2 case ratio: `{comparison['max_case_time_ratio']:.6f}`.\n\n"
             f"All mean ratio: `{summaries['all']['mean_time_ratio_v3_to_v2']:.6f}`; "
             f"holdout mean ratio: `{summaries['holdout']['mean_time_ratio_v3_to_v2']:.6f}`.\n\n"
             f"Clear attempts v2/v3: `{summaries['all']['v2']['clear_attempts']}` / `{summaries['all']['v3']['clear_attempts']}`. "
             f"Fallbacks v2/v3: `{summaries['all']['v2']['fallback_count']}` / `{summaries['all']['v3']['fallback_count']}`.\n\n"
             "The 204-point sparse grid has continuous nearest-point bound `sqrt(15^2+10^2)` m, below 20 m.\n")
    (OUT / "brief.md").write_text(brief, encoding="utf-8")
    print(json.dumps({"accepted": accepted, "max_ratio": comparison["max_case_time_ratio"], "summaries": summaries}, ensure_ascii=False))
    return accepted


if __name__ == "__main__":
    raise SystemExit(0 if evaluate() else 1)
