"""Frozen 36-case paired development screen: Q3 joint-route versus cover-then-route."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from offline_arena import OfflineArena, Source
from simulator_client import SimulatorClient
import search_policy_joint_route_q3 as joint_route
import search_policy_q3_cover_then_route as cover_then_route


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q3_cover_then_route" / "development"
SOURCE_CASES = ROOT / "results" / "feasible_q3" / "cases.json"
POLICIES = {"joint_route": joint_route.run_search, "cover_then_route": cover_then_route.run_search}


def _save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_cases():
    cases = json.loads(SOURCE_CASES.read_text(encoding="utf-8"))[:36]
    if len(cases) != 36:
        raise ValueError("expected at least 36 frozen Q3 development cases")
    target = OUT / "cases.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != cases:
        raise ValueError("cover-then-route cases differ from frozen Q3 development source")
    if not target.exists():
        _save(target, cases)
    return cases, target


def _freeze_hashes(cases_path):
    files = [
        "code/shared/search_policy.py", "code/shared/search_policy_compact.py", "code/shared/search_policy_feasible_q3.py",
        "code/shared/search_policy_joint_route_q3.py", "code/shared/search_policy_q3_cover_then_route.py",
        "code/shared/offline_arena.py", "code/shared/evaluate_q3_cover_then_route_development.py",
        "results/feasible_q3/cases.json", str(cases_path.relative_to(ROOT)),
    ]
    hashes = [{"path": path, "sha256": _sha256(ROOT / path)} for path in files]
    target = OUT / "inputs_before_run.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != hashes:
        raise ValueError("frozen Q3 cover-then-route development inputs differ from current files")
    if not target.exists():
        _save(target, hashes)
    return hashes


def _run(policy, case):
    sources = [Source(**source) for source in case["sources"]]
    arena = OfflineArena(sources, case["seed"], case["error_mode"])
    client = SimulatorClient("synthetic", transport=arena, log_path=False)
    client.enter()
    state = policy(client, question=3, method="main")
    client.exit()
    cleared = len(arena.cleared)
    movement_s, measure_s, switch_s = arena.distance_m / 5, 5 * arena.measure_count, arena.switch_count
    clears_s = 3 * arena.clear_count + 2 * cleared
    reconstructed = movement_s + measure_s + switch_s + clears_s
    groups = [set(state["cleared_channels"]), set(state["excluded_channels"]), set(state["unresolved_channels"])]
    union = set().union(*groups)
    accounted = union == set(range(1, 21)) and sum(len(group) for group in groups) == len(union)
    full_clear = cleared == len(sources) and set(state["cleared_channels"]) == arena.cleared and not (set(state["excluded_channels"]) & set(arena.sources))
    return {"all_checks_pass": bool(state.get("completed") and full_clear and accounted and abs(reconstructed - arena.virtual_time_s) < 1e-6),
            "full_source_clear": full_clear, "channels_accounted": accounted, "time_s": arena.virtual_time_s,
            "mean_s": arena.virtual_time_s / cleared if cleared else None, "movement_s": movement_s,
            "measure_s": measure_s, "switch_s": switch_s, "clears_s": clears_s,
            "clock_residual_s": arena.virtual_time_s - reconstructed, "fallback_attempts": state.get("fallback_clear_count", 0),
            "maximum_count_stop_used": state.get("maximum_count_stop_used", False),
            "coverage_points_visited": len(state.get("coverage_points_visited", []))}


def _summary(rows, method):
    return {"cases": len(rows), "mean_per_source_s": statistics.mean(row[method]["mean_s"] for row in rows),
            "mean_movement_s": statistics.mean(row[method]["movement_s"] for row in rows),
            "mean_measure_s": statistics.mean(row[method]["measure_s"] for row in rows),
            "mean_clears_s": statistics.mean(row[method]["clears_s"] for row in rows),
            "mean_switch_s": statistics.mean(row[method]["switch_s"] for row in rows),
            "total_fallback_attempts": sum(row[method]["fallback_attempts"] for row in rows),
            "maximum_count_stops": sum(row[method]["maximum_count_stop_used"] for row in rows)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases, cases_path = _freeze_cases()
    hashes = _freeze_hashes(cases_path)
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "seed": case["seed"]}
        for name, policy in POLICIES.items():
            row[name] = _run(policy, case)
        row["cover_then_route_to_joint_ratio"] = row["cover_then_route"]["time_s"] / row["joint_route"]["time_s"]
        rows.append(row)
    summaries = {name: _summary(rows, name) for name in POLICIES}
    all_checks = all(row[name]["all_checks_pass"] for row in rows for name in POLICIES)
    improvement = 1 - summaries["cover_then_route"]["mean_per_source_s"] / summaries["joint_route"]["mean_per_source_s"]
    comparison = {"scope": "SYNTHETIC_DEVELOPMENT_ONLY", "artifact_type": "q3_cover_then_route_development_comparison",
                  "case_count": len(rows), "all_checks_pass": all_checks, "accepted": bool(all_checks and improvement > 0),
                  "selection_rule": "advance only when all paired cases pass full-clear, 20-channel and reconstructed-clock checks and cover-then-route has strictly lower mean_per_source_s.",
                  "improvement_vs_joint_route": improvement, "summaries": summaries, "rows": rows, "inputs": hashes}
    _save(OUT / "comparison.json", comparison)
    print(json.dumps({"all_checks_pass": all_checks, "accepted": comparison["accepted"], "improvement": improvement, "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
