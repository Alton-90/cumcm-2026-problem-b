"""Frozen, paired Q4 synthetic evaluation: compact versus polygon fallback.

No network endpoints are used.  OfflineArena retains all sources privately and
each policy receives only a normal SimulatorClient.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict
from pathlib import Path

from offline_arena import OfflineArena, Source
from simulator_client import SimulatorClient
import search_policy_compact
import search_policy_polygon


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "polygon"
COMPACT_CASES = ROOT / "results" / "compact_grid" / "cases.json"


def _save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _new_holdouts():
    """Predeclared independent Q4 holdouts, fixed seeds 72000 through 72023."""
    result = []
    for index in range(24):
        seed = 72000 + index
        rng = random.Random(seed)
        count = (10, 13, 16)[index % 3]
        sources = []
        for source_index, channel in enumerate(rng.sample(range(1, 21), count)):
            angle = rng.uniform(0, 2 * math.pi)
            radius_from_origin = 1800 * math.sqrt(rng.random())
            x, y = radius_from_origin * math.cos(angle), radius_from_origin * math.sin(angle)
            if index // 6 == 1:
                angle = 2 * math.pi * source_index / count
                x, y = 1800 * math.cos(angle), 1800 * math.sin(angle)
            if index // 6 == 2:
                grid = [(gx, gy) for gx in range(-1400, 1401, 700)
                        for gy in range(-1400, 1401, 700) if gx * gx + gy * gy <= 1800 ** 2]
                x, y = grid[source_index % len(grid)]
            if index // 6 == 3 and source_index < 3:
                x, y = [(0, 0), (5, 0), (20, 0)][source_index]
            directional_count = (0, 1, count - 1, count, count // 2, count)[index % 6]
            heading = (math.degrees(angle) if index // 6 == 1 else rng.uniform(0, 360)) if source_index < directional_count else None
            reception_radius = 1000 if index % 3 == 0 else 1500 if index % 3 == 1 else rng.uniform(1000, 1500)
            sources.append(Source(channel, float(x), float(y), reception_radius, heading))
        result.append({"question": 4, "case_id": f"polygon_q4_{seed}", "split": "holdout", "seed": seed,
                       "error_mode": ["fixed_hash", "positive_edge", "negative_edge"][index % 3], "sources": sources})
    return result


def declared_cases():
    source_cases = json.loads(COMPACT_CASES.read_text(encoding="utf-8"))
    development = []
    for item in source_cases:
        if item["question"] != 4:
            continue
        development.append({"question": 4, "case_id": item["case_id"], "split": "development",
                            "origin_split": item["split"], "seed": item["seed"],
                            "error_mode": item["error_mode"],
                            "sources": [Source(**source) for source in item["sources"]]})
    if len(development) != 36:
        raise ValueError("expected all 36 Q4 compact cases as development")
    return development + _new_holdouts()


def _serializable_cases(cases):
    return [{**{key: value for key, value in item.items() if key != "sources"},
             "sources": [asdict(source) for source in item["sources"]]} for item in cases]


def _freeze_cases(cases):
    path = OUT / "cases.json"
    payload = _serializable_cases(cases)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen polygon cases differ from declared cases")
    else:
        _save(path, payload)
    return path


def _freeze_hashes(cases_path):
    inputs = [
        "code/shared/search_policy.py", "code/shared/search_policy_recovery.py",
        "code/shared/search_policy_compact.py", "code/shared/search_policy_polygon.py",
        "code/shared/offline_arena.py", "code/shared/evaluate_polygon.py",
        "results/compact_grid/cases.json", str(cases_path.relative_to(ROOT)),
    ]
    payload = [{"path": item, "sha256": _sha256(ROOT / item)} for item in inputs]
    path = OUT / "inputs_before_run.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen polygon input hashes differ from current inputs")
    else:
        _save(path, payload)
    return payload


def _run(policy, case):
    arena = OfflineArena(case["sources"], case["seed"], case["error_mode"])
    client = SimulatorClient("synthetic", transport=arena, log_path=False)
    client.enter()
    state = policy(client, question=4, method="main")
    client.exit()
    cleared = len(arena.cleared)
    accounted = (arena.distance_m / 5 + 5 * arena.measure_count + arena.switch_count +
                 3 * arena.clear_count + 2 * cleared)
    accounted_channels = (set(state["cleared_channels"]) | set(state["excluded_channels"]) |
                          set(state["unresolved_channels"]))
    passed = (state.get("completed") and cleared == len(case["sources"]) and
              set(state["cleared_channels"]) == arena.cleared and
              not (set(state["excluded_channels"]) & set(arena.sources)) and
              not state["unresolved_channels"] and accounted_channels == set(range(1, 21)) and
              len(state["cleared_channels"]) + len(state["excluded_channels"]) == 20 and
              abs(accounted - arena.virtual_time_s) < 1e-6)
    return {
        "all_checks_pass": bool(passed), "cleared": cleared, "n": len(case["sources"]),
        "time_s": arena.virtual_time_s, "mean_s": arena.virtual_time_s / cleared if cleared else None,
        "movement_m": arena.distance_m, "measures": arena.measure_count, "clears": arena.clear_count,
        "fallback_attempts": state.get("fallback_clear_count", 0),
        "polygon_fallback_attempts": state.get("polygon_fallback_clear_count", 0),
        "legacy_fallback_attempts": state.get("legacy_fallback_clear_count", 0),
    }


def _summary(rows, split):
    selected = rows if split == "all" else [row for row in rows if row["split"] == split]
    compact_mean = statistics.mean(row["compact"]["mean_s"] for row in selected)
    polygon_mean = statistics.mean(row["polygon"]["mean_s"] for row in selected)
    return {
        "cases": len(selected), "compact_mean_s": compact_mean, "polygon_mean_s": polygon_mean,
        "improvement": 1 - polygon_mean / compact_mean,
        "max_ratio": max(row["ratio"] for row in selected),
        "compact_fallback_attempts": sum(row["compact"]["fallback_attempts"] for row in selected),
        "polygon_fallback_attempts": sum(row["polygon"]["fallback_attempts"] for row in selected),
        "polygon_legacy_fallback_attempts": sum(row["polygon"]["legacy_fallback_attempts"] for row in selected),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = declared_cases()
    cases_path = _freeze_cases(cases)
    hashes = _freeze_hashes(cases_path)
    rows = []
    for case in cases:
        compact = _run(search_policy_compact.run_search, case)
        polygon = _run(search_policy_polygon.run_search, case)
        rows.append({"case_id": case["case_id"], "split": case["split"], "seed": case["seed"],
                     "ratio": polygon["time_s"] / compact["time_s"], "compact": compact, "polygon": polygon})
    summaries = {split: _summary(rows, split) for split in ("all", "development", "holdout")}
    all_checks_pass = all(row[method]["all_checks_pass"] for row in rows for method in ("compact", "polygon"))
    accepted = (all_checks_pass and all(summaries[split]["improvement"] >= 0.03 for split in ("all", "holdout")) and
                summaries["all"]["max_ratio"] <= 1.10)
    comparison = {
        "scope": "SYNTHETIC_ONLY", "artifact_type": "polygon_fallback_paired_comparison",
        "case_counts": {"development": 36, "holdout": 24, "all": 60},
        "criteria": {"all_sources_cleared_and_20_channels_accounted": True,
                     "mean_per_source_improvement_minimum": 0.03, "max_case_ratio_maximum": 1.10},
        "all_checks_pass": all_checks_pass, "accepted": accepted, "summaries": summaries,
        "rows": rows, "inputs": hashes,
        "polygon_cover_bound_m": search_policy_polygon.CELL_COVER_RADIUS_M,
    }
    _save(OUT / "comparison.json", comparison)
    print(json.dumps({"accepted": accepted, "all_checks_pass": all_checks_pass,
                      "summaries": summaries, "polygon_cover_bound_m": search_policy_polygon.CELL_COVER_RADIUS_M},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
