"""Final independent paired Q3 evaluation: feasible_q3 versus joint_route."""

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
import search_policy_feasible_q3 as feasible
import search_policy_joint_route_q3 as joint


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "joint_route_q3" / "final_holdout"
POLICIES = {"feasible_q3": feasible.run_search, "joint_route": joint.run_search}


def _save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases():
    cases = []
    for index, seed in enumerate(range(94000, 94060)):
        rng = random.Random(seed)
        count = (10, 13, 16)[index % 3]
        sources = []
        for source_index, channel in enumerate(rng.sample(range(1, 21), count)):
            angle = rng.uniform(0, 2 * math.pi)
            layout = index % 6
            if layout == 0:
                distance = 1800.0
                angle = 2 * math.pi * source_index / count
            elif layout == 1:
                distance = 1000.0
            elif layout == 2:
                distance = 1800.0 * math.sqrt(rng.random())
            elif layout == 3:
                distance = (0.0, 5.0, 20.0, 999.999, 1000.0, 1800.0)[source_index % 6]
                angle = math.radians(30 + 60 * (source_index % 6))
            elif layout == 4:
                distance = 1800.0
                angle = math.radians(30 + 60 * (source_index % 6))
            else:
                distance = 1800.0 * math.sqrt(rng.random())
            reception = (1000.0, 1250.0, 1500.0)[index % 3]
            sources.append(Source(channel, distance * math.cos(angle), distance * math.sin(angle), reception, None))
        cases.append({"question": 3, "case_id": f"joint_route_q3_{seed}", "split": "final_independent_holdout",
                      "seed": seed, "error_mode": ("fixed_hash", "positive_edge", "negative_edge")[index % 3], "sources": sources})
    return cases


def _freeze_cases(cases):
    payload = [{**{key: value for key, value in case.items() if key != "sources"},
                "sources": [asdict(source) for source in case["sources"]]} for case in cases]
    target = OUT / "cases.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen joint-route final cases differ from declared seeds")
    else:
        _save(target, payload)
    return target


def _freeze_hashes(cases_path):
    files = ["code/shared/search_policy.py", "code/shared/search_policy_compact.py",
             "code/shared/search_policy_feasible_q3.py", "code/shared/search_policy_joint_route_q3.py",
             "code/shared/offline_arena.py", "code/shared/evaluate_joint_route_q3_final.py",
             str(cases_path.relative_to(ROOT))]
    hashes = [{"path": file, "sha256": _sha256(ROOT / file)} for file in files]
    target = OUT / "inputs_before_run.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != hashes:
            raise ValueError("frozen joint-route final input hashes differ from current files")
    else:
        _save(target, hashes)
    return hashes


def _run(policy, case):
    arena = OfflineArena(case["sources"], case["seed"], case["error_mode"])
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
    full_clear = cleared == len(case["sources"]) and set(state["cleared_channels"]) == arena.cleared and not (set(state["excluded_channels"]) & set(arena.sources))
    return {"all_checks_pass": bool(state.get("completed") and full_clear and accounted and abs(reconstructed - arena.virtual_time_s) < 1e-6),
            "full_source_clear": full_clear, "channels_accounted": accounted, "time_s": arena.virtual_time_s,
            "mean_s": arena.virtual_time_s / cleared if cleared else None, "movement_s": movement_s,
            "measure_s": measure_s, "switch_s": switch_s, "clears_s": clears_s,
            "clock_residual_s": arena.virtual_time_s - reconstructed, "fallback_attempts": state.get("fallback_clear_count", 0)}


def _summary(rows, method):
    return {"cases": len(rows), "mean_per_source_s": statistics.mean(row[method]["mean_s"] for row in rows),
            "mean_movement_s": statistics.mean(row[method]["movement_s"] for row in rows),
            "mean_measure_s": statistics.mean(row[method]["measure_s"] for row in rows),
            "mean_clears_s": statistics.mean(row[method]["clears_s"] for row in rows),
            "mean_switch_s": statistics.mean(row[method]["switch_s"] for row in rows),
            "total_fallback_attempts": sum(row[method]["fallback_attempts"] for row in rows)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    cases_path = _freeze_cases(cases)
    hashes = _freeze_hashes(cases_path)
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "seed": case["seed"]}
        for name, policy in POLICIES.items():
            row[name] = _run(policy, case)
        row["joint_to_feasible_ratio"] = row["joint_route"]["time_s"] / row["feasible_q3"]["time_s"]
        rows.append(row)
    summaries = {name: _summary(rows, name) for name in POLICIES}
    all_checks = all(row[name]["all_checks_pass"] for row in rows for name in POLICIES)
    accepted = all_checks and summaries["joint_route"]["mean_per_source_s"] < summaries["feasible_q3"]["mean_per_source_s"]
    comparison = {"scope": "SYNTHETIC_FINAL_INDEPENDENT_HOLDOUT", "artifact_type": "joint_route_q3_final_comparison",
                  "case_count": len(rows), "seed_range": [94000, 94059], "all_checks_pass": all_checks, "accepted": accepted,
                  "acceptance_rule": "accepted requires all cases to clear every source with all 20 channels accounted and joint_route mean_per_source_s to be strictly lower than feasible_q3.",
                  "review": {"coverage": "origin plus six 1150m ring points, unchanged from feasible_q3", "truth_access": "route destinations derive only from fixed scan points and Tracked public-response feasible clouds", "exception_exit": "outer exception marks stopped and unresolved channels"},
                  "summaries": summaries, "rows": rows, "inputs": hashes}
    _save(OUT / "comparison.json", comparison)
    print(json.dumps({"all_checks_pass": all_checks, "accepted": accepted, "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
