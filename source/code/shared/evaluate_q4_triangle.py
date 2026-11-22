"""Offline same-case comparison of Q4 triangle coverage against prior candidates."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from offline_arena import OfflineArena, Source
from simulator_client import SimulatorClient
import search_policy_compact as compact
import search_policy_polygon as polygon
import search_policy_q4_batch as batch49
import search_policy_q4_triangle as triangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q4_triangle"
SOURCE_CASES = ROOT / "results" / "q4_batch49" / "cases.json"
POLICIES = {"compact": compact.run_search, "polygon": polygon.run_search,
            "batch49": batch49.run_search, "triangle980": triangle.run_search}


def _save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_cases():
    payload = json.loads(SOURCE_CASES.read_text(encoding="utf-8"))
    target = OUT / "cases.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ValueError("triangle frozen cases differ from batch49 source cases")
    else:
        _save(target, payload)
    return payload, target


def _freeze_hashes(cases_path):
    files = [
        "code/shared/search_policy.py", "code/shared/search_policy_recovery.py",
        "code/shared/search_policy_compact.py", "code/shared/search_policy_polygon.py",
        "code/shared/search_policy_q4_batch.py", "code/shared/search_policy_q4_triangle.py",
        "code/shared/offline_arena.py", "code/shared/evaluate_q4_triangle.py",
        "results/q4_batch49/cases.json", str(cases_path.relative_to(ROOT)),
    ]
    hashes = [{"path": file, "sha256": _sha256(ROOT / file)} for file in files]
    target = OUT / "inputs_before_run.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != hashes:
            raise ValueError("triangle frozen input hashes differ from current files")
    else:
        _save(target, hashes)
    return hashes


def _run(policy, case):
    sources = [Source(**source) for source in case["sources"]]
    arena = OfflineArena(sources, case["seed"], case["error_mode"])
    client = SimulatorClient("synthetic", transport=arena, log_path=False)
    client.enter()
    state = policy(client, question=4, method="main")
    client.exit()
    cleared = len(arena.cleared)
    reconstructed = arena.distance_m / 5 + 5 * arena.measure_count + arena.switch_count + 3 * arena.clear_count + 2 * cleared
    groups = [set(state["cleared_channels"]), set(state["excluded_channels"]), set(state["unresolved_channels"])]
    union = set().union(*groups)
    channels_accounted = union == set(range(1, 21)) and sum(len(group) for group in groups) == len(union)
    full_clear = cleared == len(sources) and set(state["cleared_channels"]) == arena.cleared and not (set(state["excluded_channels"]) & set(arena.sources))
    clock_ok = abs(reconstructed - arena.virtual_time_s) < 1e-6
    return {"all_checks_pass": bool(state.get("completed") and full_clear and channels_accounted and clock_ok),
            "full_source_clear": full_clear, "channels_accounted": channels_accounted, "clock_ok": clock_ok,
            "time_s": arena.virtual_time_s, "mean_s": arena.virtual_time_s / cleared if cleared else None,
            "clock_residual_s": arena.virtual_time_s - reconstructed, "cleared": cleared, "n": len(sources),
            "movement_m": arena.distance_m, "measures": arena.measure_count, "clears": arena.clear_count,
            "fallback_attempts": state.get("fallback_clear_count", 0), "coverage_points_total": state["coverage_points_total"]}


def _summary(rows, split, method):
    selected = rows if split == "all" else [row for row in rows if row["split"] == split]
    return {"cases": len(selected), "mean_per_source_s": statistics.mean(row[method]["mean_s"] for row in selected),
            "total_fallback_attempts": sum(row[method]["fallback_attempts"] for row in selected)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases, cases_path = _freeze_cases()
    hashes = _freeze_hashes(cases_path)
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "split": case["split"], "seed": case["seed"]}
        for name, policy in POLICIES.items():
            row[name] = _run(policy, case)
        row["triangle980_to_batch49_ratio"] = row["triangle980"]["time_s"] / row["batch49"]["time_s"]
        rows.append(row)
    summaries = {split: {method: _summary(rows, split, method) for method in POLICIES}
                 for split in ("all", "development", "holdout")}
    coverage_points = triangle.triangle_coverage_points()
    comparison = {
        "scope": "SYNTHETIC_ONLY", "artifact_type": "q4_triangle980_same_case_comparison",
        "same_cases_as": "results/q4_batch49/cases.json", "case_counts": {"development": 36, "holdout": 24, "all": 60},
        "all_checks_pass": all(row[method]["all_checks_pass"] for row in rows for method in POLICIES),
        "criteria": {"full_source_clear": True, "all_20_channels_accounted": True, "reconstructed_virtual_clock": True},
        "triangle_coverage": {
            "point_count": len(coverage_points), "triangle_side_m": triangle.TRIANGLE_SIDE_M,
            "source_disk_radius_m": triangle.SOURCE_RADIUS_M, "minimum_receive_radius_m": triangle.MIN_RECEIVE_RADIUS_M,
            "directional_vertex_distance_bound_m": triangle.TRIANGLE_SIDE_M,
            "guarantee": "Every source belongs to a retained closed-disk-intersecting triangle. Any closed directional half-plane through it contains a triangle vertex at distance <=980m<1000m; all incident triangles are retained on grid edges and vertices."
        },
        "summaries": summaries,
        "triangle980_ratios": {"max_to_batch49": max(row["triangle980_to_batch49_ratio"] for row in rows)},
        "rows": rows, "inputs": hashes,
    }
    _save(OUT / "comparison.json", comparison)
    print(json.dumps({"all_checks_pass": comparison["all_checks_pass"], "triangle_coverage": comparison["triangle_coverage"],
                      "summaries": summaries, "triangle980_ratios": comparison["triangle980_ratios"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
