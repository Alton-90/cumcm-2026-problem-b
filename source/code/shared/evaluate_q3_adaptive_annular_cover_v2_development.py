"""Paired 36-case offline development screen for adaptive Q3 coverage v2."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from offline_arena import OfflineArena, Source
from simulator_client import SimulatorClient
import search_policy_q3_combined as baseline
import search_policy_q3_adaptive_annular_cover_v2 as candidate

R = Path(__file__).resolve().parents[2]
O = R / "results" / "q3_adaptive_annular_cover" / "development_v2"
SRC = R / "results" / "feasible_q3" / "cases.json"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage_contract(state):
    expected_unseen = set(range(1, 21))
    event_contract = True
    current_threshold = True
    necessary_scan_gain = True
    scanned_points = set()
    duplicate_scan_points = 0
    for event in state.get("coverage_events", []):
        scanned = set(event.get("scanned_unseen_channels", []))
        newly_seen = set(event.get("newly_seen_channels", []))
        event_contract &= scanned == expected_unseen and newly_seen <= scanned
        expected_unseen -= newly_seen
        point = tuple(round(float(value), 6) for value in event.get("point", []))
        if point in scanned_points:
            duplicate_scan_points += 1
        scanned_points.add(point)
        if event.get("kind") == "current":
            current_threshold &= event.get("new_arc_deg", 0) + 1e-8 >= candidate.MIN_NEW_ARC_DEG
        if event.get("kind") == "adaptive":
            necessary_scan_gain &= event.get("new_arc_deg", 0) > 1e-8
    intervals = state.get("annulus_intervals_deg", [])
    disjoint = all(
        0 <= start < end <= 360
        and (index == 0 or intervals[index - 1][1] <= start + 1e-8)
        for index, (start, end) in enumerate(intervals)
    )
    covered = sum(end - start for start, end in intervals)
    continuous = bool(
        state.get("inner_disk_covered")
        and state.get("annulus_coverage_complete")
        and covered >= 360 - 1e-7
    )
    return {
        "event_contract_pass": bool(event_contract),
        "current_threshold_pass": bool(current_threshold),
        "necessary_scan_gain_pass": bool(necessary_scan_gain),
        "duplicate_scan_points": duplicate_scan_points,
        "intervals_disjoint": bool(disjoint),
        "continuous_coverage_complete": continuous,
        "coverage_or_maximum_stop": bool(state.get("maximum_count_stop_used") or continuous),
        "annulus_covered_degrees": covered,
    }


def run(policy, case, diagnose=False):
    sources = [Source(**item) for item in case["sources"]]
    arena = OfflineArena(sources, case["seed"], case["error_mode"])
    client = SimulatorClient("synthetic", transport=arena, log_path=False)
    client.enter()
    state = policy(client, question=3, method="main")
    client.exit()
    count = len(sources)
    movement_s = arena.distance_m / 5
    measure_s = 5 * arena.measure_count
    switch_s = arena.switch_count
    clears_s = 3 * arena.clear_count + 2 * len(arena.cleared)
    groups = [set(state[name]) for name in ("cleared_channels", "excluded_channels", "unresolved_channels")]
    union = set().union(*groups)
    accounted = union == set(range(1, 21)) and sum(map(len, groups)) == len(union)
    false_excluded = bool(set(state["excluded_channels"]) & set(arena.sources))
    full_clear = len(arena.cleared) == count and set(state["cleared_channels"]) == arena.cleared and not false_excluded
    cost_ok = abs(movement_s + measure_s + switch_s + clears_s - arena.virtual_time_s) < 1e-6
    coverage = coverage_contract(state) if diagnose else {}
    diagnostic_ok = not diagnose or (
        coverage["event_contract_pass"]
        and coverage["current_threshold_pass"]
        and coverage["necessary_scan_gain_pass"]
        and coverage["duplicate_scan_points"] == 0
        and coverage["intervals_disjoint"]
        and coverage["coverage_or_maximum_stop"]
    )
    return {
        "all_checks_pass": bool(state.get("completed") and full_clear and accounted and cost_ok and diagnostic_ok),
        "full_source_clear": full_clear,
        "channels_accounted": accounted,
        "false_excluded_source": false_excluded,
        "time_s": arena.virtual_time_s,
        "mean_s": arena.virtual_time_s / count,
        "movement_s": movement_s,
        "measure_s": measure_s,
        "switch_s": switch_s,
        "clears_s": clears_s,
        "fallback_attempts": state.get("fallback_clear_count", 0),
        "adaptive_scan_count": state.get("adaptive_scan_count", 0),
        "opportunistic_current_scan_count": state.get("opportunistic_current_scan_count", 0),
        "fallback_coverage_used": state.get("fallback_coverage_used", False),
        "fallback_coverage_scan_count": state.get("fallback_coverage_scan_count", 0),
        "replan_bearing_count": state.get("replan_bearing_count", 0),
        "maximum_count_stop_used": state.get("maximum_count_stop_used", False),
        **coverage,
    }


def summarize(rows, name):
    return {
        key: statistics.mean(item[name][key] for item in rows)
        for key in ("mean_s", "movement_s", "measure_s", "switch_s", "clears_s")
    }


def main():
    O.mkdir(parents=True, exist_ok=True)
    cases = json.loads(SRC.read_text(encoding="utf-8"))[:36]
    copied_cases = O / "cases.json"
    if copied_cases.exists() and json.loads(copied_cases.read_text(encoding="utf-8")) != cases:
        raise ValueError("case mismatch")
    if not copied_cases.exists():
        save(copied_cases, cases)
    files = [
        "code/shared/search_policy.py",
        "code/shared/search_policy_compact.py",
        "code/shared/search_policy_feasible_q3.py",
        "code/shared/search_policy_joint_route_q3.py",
        "code/shared/search_policy_q3_combined.py",
        "code/shared/search_policy_q3_adaptive_annular_cover.py",
        "code/shared/search_policy_q3_adaptive_annular_cover_v2.py",
        "code/shared/route_multistart.py",
        "code/shared/offline_arena.py",
        "code/shared/simulator_client.py",
        "code/shared/evaluate_q3_adaptive_annular_cover_v2_development.py",
        "results/feasible_q3/cases.json",
        str(copied_cases.relative_to(R)),
    ]
    inputs = [{"path": item, "sha256": sha(R / item)} for item in files]
    save(O / "inputs_before_run.json", inputs)
    rows = []
    for case in cases:
        rows.append(
            {
                "case_id": case["case_id"],
                "seed": case["seed"],
                "combined": run(baseline.run_search, case),
                "adaptive_annular_v2": run(candidate.run_search, case, True),
            }
        )
    summaries = {name: summarize(rows, name) for name in ("combined", "adaptive_annular_v2")}
    checks = all(item[name]["all_checks_pass"] for item in rows for name in summaries)
    improvement = 1 - summaries["adaptive_annular_v2"]["mean_s"] / summaries["combined"]["mean_s"]
    output = {
        "scope": "SYNTHETIC_DEVELOPMENT_ONLY",
        "case_count": len(cases),
        "all_checks_pass": checks,
        "accepted": bool(checks and improvement > 0),
        "improvement_vs_combined": improvement,
        "continuous_coverage_complete_cases": sum(
            item["adaptive_annular_v2"]["continuous_coverage_complete"] for item in rows
        ),
        "maximum_count_stop_cases": sum(
            item["adaptive_annular_v2"]["maximum_count_stop_used"] for item in rows
        ),
        "fallback_coverage_cases": sum(
            item["adaptive_annular_v2"]["fallback_coverage_used"] for item in rows
        ),
        "summaries": summaries,
        "rows": rows,
        "inputs": inputs,
    }
    save(O / "comparison.json", output)
    print(
        json.dumps(
            {
                "all_checks_pass": checks,
                "accepted": output["accepted"],
                "improvement": improvement,
                "continuous_coverage_complete_cases": output["continuous_coverage_complete_cases"],
                "maximum_count_stop_cases": output["maximum_count_stop_cases"],
                "fallback_coverage_cases": output["fallback_coverage_cases"],
                "summaries": summaries,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
