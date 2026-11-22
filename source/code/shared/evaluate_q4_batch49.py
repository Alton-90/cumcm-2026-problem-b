"""Offline paired Q4 comparison for compact, polygon, and batch49 candidates."""

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
import search_policy_compact as compact
import search_policy_polygon as polygon
import search_policy_q4_batch as batch49


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q4_batch49"
COMPACT_CASES = ROOT / "results" / "compact_grid" / "cases.json"
POLICIES = {"compact": compact.run_search, "polygon": polygon.run_search, "batch49": batch49.run_search}


def _save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _holdouts():
    result = []
    for index, seed in enumerate(range(82000, 82024)):
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
                x, y = ((0, 0), (5, 0), (20, 0))[source_index]
            directional_count = (0, 1, count - 1, count, count // 2, count)[index % 6]
            heading = (math.degrees(angle) if index // 6 == 1 else rng.uniform(0, 360)) if source_index < directional_count else None
            reception = 1000 if index % 3 == 0 else 1500 if index % 3 == 1 else rng.uniform(1000, 1500)
            sources.append(Source(channel, float(x), float(y), reception, heading))
        result.append({"question": 4, "case_id": f"batch49_q4_{seed}", "split": "holdout", "seed": seed,
                       "error_mode": ("fixed_hash", "positive_edge", "negative_edge")[index % 3], "sources": sources})
    return result


def _cases():
    base = json.loads(COMPACT_CASES.read_text(encoding="utf-8"))
    development = [{"question": 4, "case_id": item["case_id"], "split": "development", "seed": item["seed"],
                    "error_mode": item["error_mode"], "sources": [Source(**source) for source in item["sources"]]}
                   for item in base if item["question"] == 4]
    if len(development) != 36:
        raise ValueError("expected 36 fixed Q4 compact cases")
    return development + _holdouts()


def _serialized(cases):
    return [{**{key: value for key, value in case.items() if key != "sources"},
             "sources": [asdict(source) for source in case["sources"]]} for case in cases]


def _freeze_cases(cases):
    target = OUT / "cases.json"
    payload = _serialized(cases)
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen batch49 cases differ from declared cases")
    else:
        _save(target, payload)
    return target


def _freeze_hashes(cases_path):
    files = [
        "code/shared/search_policy.py", "code/shared/search_policy_recovery.py",
        "code/shared/search_policy_compact.py", "code/shared/search_policy_polygon.py",
        "code/shared/search_policy_q4_batch.py", "code/shared/offline_arena.py",
        "code/shared/evaluate_q4_batch49.py", "results/compact_grid/cases.json",
        str(cases_path.relative_to(ROOT)),
    ]
    payload = [{"path": file, "sha256": _sha256(ROOT / file)} for file in files]
    target = OUT / "inputs_before_run.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen batch49 input hashes differ from current files")
    else:
        _save(target, payload)
    return payload


def _run(policy, case):
    arena = OfflineArena(case["sources"], case["seed"], case["error_mode"])
    client = SimulatorClient("synthetic", transport=arena, log_path=False)
    client.enter()
    state = policy(client, question=4, method="main")
    client.exit()
    cleared = len(arena.cleared)
    reconstructed = (arena.distance_m / 5 + 5 * arena.measure_count + arena.switch_count +
                     3 * arena.clear_count + 2 * cleared)
    groups = [set(state["cleared_channels"]), set(state["excluded_channels"]), set(state["unresolved_channels"])]
    union = set().union(*groups)
    channels_accounted = union == set(range(1, 21)) and sum(len(group) for group in groups) == len(union)
    full_clear = (cleared == len(case["sources"]) and set(state["cleared_channels"]) == arena.cleared and
                  not (set(state["excluded_channels"]) & set(arena.sources)))
    clock_ok = abs(reconstructed - arena.virtual_time_s) < 1e-6
    return {
        "all_checks_pass": bool(state.get("completed") and full_clear and channels_accounted and clock_ok),
        "full_source_clear": full_clear, "channels_accounted": channels_accounted, "clock_ok": clock_ok,
        "time_s": arena.virtual_time_s, "mean_s": arena.virtual_time_s / cleared if cleared else None,
        "clock_residual_s": arena.virtual_time_s - reconstructed, "cleared": cleared, "n": len(case["sources"]),
        "movement_m": arena.distance_m, "measures": arena.measure_count, "clears": arena.clear_count,
        "fallback_attempts": state.get("fallback_clear_count", 0),
    }


def _summary(rows, split, method):
    selected = rows if split == "all" else [row for row in rows if row["split"] == split]
    return {"cases": len(selected), "mean_per_source_s": statistics.mean(row[method]["mean_s"] for row in selected),
            "total_fallback_attempts": sum(row[method]["fallback_attempts"] for row in selected)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    cases_path = _freeze_cases(cases)
    hashes = _freeze_hashes(cases_path)
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "split": case["split"], "seed": case["seed"]}
        for name, policy in POLICIES.items():
            row[name] = _run(policy, case)
        row["batch49_to_polygon_ratio"] = row["batch49"]["time_s"] / row["polygon"]["time_s"]
        row["batch49_to_compact_ratio"] = row["batch49"]["time_s"] / row["compact"]["time_s"]
        rows.append(row)
    summaries = {split: {method: _summary(rows, split, method) for method in POLICIES}
                 for split in ("all", "development", "holdout")}
    comparison = {
        "scope": "SYNTHETIC_ONLY", "artifact_type": "q4_batch49_paired_comparison",
        "case_counts": {"development": 36, "holdout": 24, "all": 60},
        "criteria": {"full_source_clear": True, "all_20_channels_accounted": True, "reconstructed_virtual_clock": True},
        "all_checks_pass": all(row[method]["all_checks_pass"] for row in rows for method in POLICIES),
        "summaries": summaries,
        "batch49_ratios": {"max_to_polygon": max(row["batch49_to_polygon_ratio"] for row in rows),
                           "max_to_compact": max(row["batch49_to_compact_ratio"] for row in rows)},
        "rows": rows, "inputs": hashes,
    }
    _save(OUT / "comparison.json", comparison)
    print(json.dumps({"all_checks_pass": comparison["all_checks_pass"], "summaries": summaries,
                      "batch49_ratios": comparison["batch49_ratios"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
