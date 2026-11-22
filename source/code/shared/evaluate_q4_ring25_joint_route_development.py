"""Frozen 36-case development comparison: ring25 versus ring25 joint-route."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from offline_arena import OfflineArena, Source
from simulator_client import SimulatorClient
import search_policy_q4_ring25 as ring25
import search_policy_q4_ring25_joint_route as joint_route


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q4_ring25_joint_route" / "development_v3"
SOURCE_CASES = ROOT / "results" / "q4_reuse" / "development" / "cases.json"
POLICIES = {"ring25": ring25.run_search, "ring25_joint_route": joint_route.run_search}


def _save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_cases():
    cases = json.loads(SOURCE_CASES.read_text(encoding="utf-8"))
    if len(cases) != 36:
        raise ValueError("expected the frozen 36 Q4 reuse development cases")
    target = OUT / "cases.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != cases:
        raise ValueError("joint-route development cases differ from frozen ring25 cases")
    if not target.exists():
        _save(target, cases)
    return cases, target


def _freeze_hashes(cases_path):
    files = [
        "code/shared/search_policy.py", "code/shared/search_policy_polygon.py",
        "code/shared/search_policy_q4_batch.py", "code/shared/search_policy_q4_reuse.py",
        "code/shared/search_policy_q4_ring25.py", "code/shared/search_policy_joint_route_q3.py",
        "code/shared/search_policy_q4_ring25_joint_route.py", "code/shared/offline_arena.py",
        "code/shared/evaluate_q4_ring25_joint_route_development.py",
        "results/q4_reuse/development/cases.json", str(cases_path.relative_to(ROOT)),
    ]
    hashes = [{"path": path, "sha256": _sha256(ROOT / path)} for path in files]
    target = OUT / "inputs_before_run.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != hashes:
        raise ValueError("frozen joint-route development input hashes differ from current files")
    if not target.exists():
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
    movement_s, measure_s, switch_s = arena.distance_m / 5, 5 * arena.measure_count, arena.switch_count
    clears_s = 3 * arena.clear_count + 2 * cleared
    reconstructed = movement_s + measure_s + switch_s + clears_s
    groups = [set(state["cleared_channels"]), set(state["excluded_channels"]), set(state["unresolved_channels"])]
    union = set().union(*groups)
    accounted = union == set(range(1, 21)) and sum(len(group) for group in groups) == len(union)
    full_clear = cleared == len(sources) and set(state["cleared_channels"]) == arena.cleared and not (set(state["excluded_channels"]) & set(arena.sources))
    return {
        "all_checks_pass": bool(state.get("completed") and full_clear and accounted and abs(reconstructed - arena.virtual_time_s) < 1e-6),
        "full_source_clear": full_clear, "channels_accounted": accounted, "time_s": arena.virtual_time_s,
        "mean_s": arena.virtual_time_s / cleared if cleared else None, "movement_s": movement_s,
        "measure_s": measure_s, "switch_s": switch_s, "clears_s": clears_s,
        "clock_residual_s": arena.virtual_time_s - reconstructed,
        "fallback_attempts": state.get("fallback_clear_count", 0),
        "maximum_count_stop_used": state.get("maximum_count_stop_used", False),
    }


def _summary(rows, method):
    return {
        "cases": len(rows), "mean_per_source_s": statistics.mean(row[method]["mean_s"] for row in rows),
        "mean_movement_s": statistics.mean(row[method]["movement_s"] for row in rows),
        "mean_measure_s": statistics.mean(row[method]["measure_s"] for row in rows),
        "mean_clears_s": statistics.mean(row[method]["clears_s"] for row in rows),
        "mean_switch_s": statistics.mean(row[method]["switch_s"] for row in rows),
        "total_fallback_attempts": sum(row[method]["fallback_attempts"] for row in rows),
        "maximum_count_stops": sum(row[method]["maximum_count_stop_used"] for row in rows),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases, cases_path = _freeze_cases()
    hashes = _freeze_hashes(cases_path)
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "seed": case["seed"]}
        for name, policy in POLICIES.items():
            row[name] = _run(policy, case)
        row["joint_route_to_ring25_ratio"] = row["ring25_joint_route"]["time_s"] / row["ring25"]["time_s"]
        rows.append(row)
    summaries = {name: _summary(rows, name) for name in POLICIES}
    all_checks = all(row[name]["all_checks_pass"] for row in rows for name in POLICIES)
    improvement = 1 - summaries["ring25_joint_route"]["mean_per_source_s"] / summaries["ring25"]["mean_per_source_s"]
    accepted = bool(all_checks and improvement > 0)
    comparison = {
        "scope": "SYNTHETIC_DEVELOPMENT_ONLY", "artifact_type": "q4_ring25_joint_route_development_v3_comparison",
        "case_count": len(rows), "all_checks_pass": all_checks, "accepted": accepted,
        "selection_rule": "advance only when every paired case passes full-clear, 20-channel, and reconstructed-clock checks and joint-route mean_per_source_s is strictly lower than ring25.",
        "improvement_vs_ring25": improvement, "coverage_certificate": ring25.coverage_certificate(),
        "route_dependency": joint_route.ROUTE_DEPENDENCY_SHA256, "summaries": summaries, "rows": rows, "inputs": hashes,
    }
    _save(OUT / "comparison.json", comparison)
    print(json.dumps({"all_checks_pass": all_checks, "accepted": accepted, "improvement": improvement, "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
