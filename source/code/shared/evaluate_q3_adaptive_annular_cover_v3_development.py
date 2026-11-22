"""Final 36-case development screen for adaptive annular Q3 coverage v3."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import evaluate_q3_adaptive_annular_cover_v2_development as common
import search_policy_q3_combined as baseline
import search_policy_q3_adaptive_annular_cover_v3 as candidate

R = Path(__file__).resolve().parents[2]
O = R / "results" / "q3_adaptive_annular_cover" / "development_v3"
SRC = R / "results" / "feasible_q3" / "cases.json"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage_contract(state):
    expected_unseen = set(range(1, 21))
    seen = set()
    event_contract = True
    current_threshold = True
    necessary_scan_gain = True
    radius_contract = True
    known_measure_contract = True
    known_measure_count = 0
    scanned_points = set()
    duplicate_scan_points = 0
    for event in state.get("coverage_events", []):
        scanned = set(event.get("scanned_unseen_channels", []))
        newly_seen = set(event.get("newly_seen_channels", []))
        measured_known = set(event.get("measured_known_channels", []))
        event_contract &= scanned == expected_unseen and newly_seen <= scanned
        known_measure_contract &= measured_known <= seen and not (measured_known & scanned)
        known_measure_count += len(measured_known)
        expected_unseen -= newly_seen
        seen |= newly_seen
        point = tuple(round(float(value), 6) for value in event.get("point", []))
        if point in scanned_points:
            duplicate_scan_points += 1
        scanned_points.add(point)
        if event.get("kind") == "current":
            current_threshold &= event.get("new_arc_deg", 0) + 1e-8 >= candidate.MIN_NEW_ARC_DEG
        if event.get("kind") == "adaptive":
            necessary_scan_gain &= event.get("new_arc_deg", 0) > 1e-8
            half_width = event.get("planned_piece_half_width_deg")
            bounds = event.get("planned_radius_interval_m")
            planned_radius = event.get("planned_radius_m")
            if not isinstance(half_width, (int, float)) or not isinstance(bounds, list) or len(bounds) != 2:
                radius_contract = False
            else:
                recomputed = candidate._radius_interval(half_width)
                actual_radius = math.hypot(*event["point"])
                radius_contract &= bool(
                    recomputed
                    and abs(bounds[0] - recomputed[0]) <= 1e-7
                    and abs(bounds[1] - recomputed[1]) <= 1e-7
                    and bounds[0] - 1e-7 <= actual_radius <= bounds[1] + 1e-7
                    and abs(actual_radius - planned_radius) <= 1e-7
                    and event.get("annulus_half_width_deg", 0) + 1e-7 >= half_width
                )
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
        "adaptive_radius_contract_pass": bool(radius_contract),
        "known_measure_contract_pass": bool(
            known_measure_contract and known_measure_count == state.get("coverage_known_measure_count", 0)
        ),
        "coverage_known_measure_count": known_measure_count,
        "duplicate_scan_points": duplicate_scan_points,
        "intervals_disjoint": bool(disjoint),
        "continuous_coverage_complete": continuous,
        "coverage_or_maximum_stop": bool(state.get("maximum_count_stop_used") or continuous),
        "annulus_covered_degrees": covered,
    }


def diagnostic_pass(result):
    return all(
        result[key]
        for key in (
            "event_contract_pass",
            "current_threshold_pass",
            "necessary_scan_gain_pass",
            "adaptive_radius_contract_pass",
            "known_measure_contract_pass",
            "intervals_disjoint",
            "coverage_or_maximum_stop",
        )
    ) and result["duplicate_scan_points"] == 0


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
        "code/shared/search_policy_q3_adaptive_annular_cover_v3.py",
        "code/shared/route_multistart.py",
        "code/shared/offline_arena.py",
        "code/shared/simulator_client.py",
        "code/shared/evaluate_q3_adaptive_annular_cover_v2_development.py",
        "code/shared/evaluate_q3_adaptive_annular_cover_v3_development.py",
        "results/feasible_q3/cases.json",
        str(copied_cases.relative_to(R)),
    ]
    inputs = [{"path": item, "sha256": sha(R / item)} for item in files]
    save(O / "inputs_before_run.json", inputs)
    original_contract = common.coverage_contract
    common.coverage_contract = coverage_contract
    try:
        rows = []
        for case in cases:
            base_result = common.run(baseline.run_search, case)
            candidate_result = common.run(candidate.run_search, case, True)
            candidate_result["all_checks_pass"] &= diagnostic_pass(candidate_result)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "seed": case["seed"],
                    "combined": base_result,
                    "adaptive_annular_v3": candidate_result,
                }
            )
    finally:
        common.coverage_contract = original_contract
    summaries = {
        name: {
            key: statistics.mean(item[name][key] for item in rows)
            for key in ("mean_s", "movement_s", "measure_s", "switch_s", "clears_s")
        }
        for name in ("combined", "adaptive_annular_v3")
    }
    checks = all(item[name]["all_checks_pass"] for item in rows for name in summaries)
    improvement = 1 - summaries["adaptive_annular_v3"]["mean_s"] / summaries["combined"]["mean_s"]
    output = {
        "scope": "SYNTHETIC_DEVELOPMENT_ONLY",
        "case_count": len(cases),
        "all_checks_pass": checks,
        "accepted": bool(checks and improvement > 0),
        "improvement_vs_combined": improvement,
        "continuous_coverage_complete_cases": sum(
            item["adaptive_annular_v3"]["continuous_coverage_complete"] for item in rows
        ),
        "maximum_count_stop_cases": sum(
            item["adaptive_annular_v3"]["maximum_count_stop_used"] for item in rows
        ),
        "fallback_coverage_cases": sum(
            item["adaptive_annular_v3"]["fallback_coverage_used"] for item in rows
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
