"""One-shot paired offline evaluation of Q4 edge-waypoint v1."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

from evaluate_q4_shadow_recovery_development import TrackedArena
from offline_arena import Source
from search_policy import _clear_points
from simulator_client import SimulatorClient
import search_policy_q4_edge_waypoint_v1 as candidate
import search_policy_q4_ring22_joint_route as baseline


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/Q4/experiments/edge_waypoint_v1"
STATES = OUT / "states"
CASES = OUT / "cases.json"
GENERATION = OUT / "case_generation.json"
MANIFEST = OUT / "inputs_before_run.json"
COMPARISON = OUT / "metrics/comparison.json"
DECISION = OUT / "decision.json"
RUN_SUMMARY = OUT / "run_summary.json"
MOVEMENT_RELATIVE_MIN = 0.05
TIME_RELATIVE_MIN = 0.02
WORST_TIME_RATIO_MAX = 1.10
FLOAT_TOLERANCE = 1e-6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def public_fingerprint(actions):
    raw = json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def decompose(actions):
    position, channel = (0.0, 0.0), 1
    result = {
        "movement_m": 0.0, "movement_s": 0.0, "measure_s": 0.0,
        "switch_s": 0.0, "successful_clear_s": 0.0, "failed_clear_s": 0.0,
    }
    public_cleared = set()
    post_clear_measure_count = 0
    for action in actions:
        point = tuple(action["point"])
        distance = math.dist(position, point)
        result["movement_m"] += distance
        result["movement_s"] += distance / 5.0
        position = point
        if action["path"] == "/measure":
            result["measure_s"] += 5.0
            result["switch_s"] += float(action["channel"] != channel)
            channel = action["channel"]
            post_clear_measure_count += int(action["channel"] in public_cleared)
        elif action["response"].get("clear_result") == "success":
            result["successful_clear_s"] += 5.0
            public_cleared.add(action["channel"])
        else:
            result["failed_clear_s"] += 3.0
    result["total_s"] = sum(value for key, value in result.items() if key.endswith("_s"))
    result["post_clear_measure_count"] = post_clear_measure_count
    return result


def candidate_state_checks(state):
    events = state.get("edge_waypoint_events", [])
    opportunity_checks = state.get("edge_waypoint_opportunity_checks", [])
    localizers = state.get("localization_budget_events", [])
    used = state.get("edge_waypoint_used_channels", [])
    checks = {
        "event_counter_matches": len(events) == state.get("edge_waypoint_measure_count")
        == state.get("edge_waypoint_certified_count"),
        "opportunity_counter_matches": len(opportunity_checks)
        == state.get("edge_waypoint_opportunity_check_count"),
        "used_channels_unique_and_match": len(used) == len(set(used)) == len(events),
        "every_selected_candidate_certified": all(
            event["selected"]["certificate"]["certified"] for event in events
        ),
        "every_waypoint_on_segment": all(
            event["split_certificate"]["on_segment"] for event in events
        ),
        "every_compatibility_use_ordering_only": all(
            event["selected"].get("compatibility_used_only_for_ordering") is True
            for event in events
        ),
        "no_signal_retains_track": all(
            event.get("track_retained_on_no_signal") is True for event in events
            if event["response"].get("measure_result") == "no_signal"
        ),
        "reorders_have_strict_guard": all(
            event["continuation"].get("strictly_lower") is True
            and event["continuation"].get("first_node_differs") is True
            for event in events
            if event["continuation"].get("continuation_mode") == "guarded_replan"
        ),
        "localizer_budget_at_most_8": all(
            0 <= item["local_extra_bearing_delta"] <= item["budget_limit"] == 8
            for item in localizers
        ),
        "legacy_fallback_has_304_points": len(_clear_points((0.0, 0.0), 0.0)) == 304,
    }
    return {"all_checks_pass": all(checks.values()), "checks": checks}


def run(policy, case, is_candidate):
    sources = [Source(**item) for item in case["sources"]]
    arena = TrackedArena(sources, case["seed"], case["error_mode"])
    client = SimulatorClient("offline", transport=arena, log_path=False)
    client.enter()
    started = time.perf_counter()
    state = policy(client, question=4, method="main")
    cpu_s = time.perf_counter() - started
    client.exit()
    actions = arena.action_log
    groups = [set(state[key]) for key in (
        "cleared_channels", "excluded_channels", "unresolved_channels"
    )]
    source_channels = set(arena.sources)
    decomposition = decompose(actions)
    full_clear = arena.cleared == source_channels == groups[0]
    accounted = set.union(*groups) == set(range(1, 21)) and sum(map(len, groups)) == 20
    false_excluded = bool(source_channels & groups[1])
    clock_residual = decomposition["total_s"] - arena.virtual_time_s
    clock_ok = abs(clock_residual) <= FLOAT_TOLERANCE and abs(
        float(state["virtual_time_s"]) - arena.virtual_time_s
    ) <= FLOAT_TOLERANCE
    measures = sum(action["path"] == "/measure" for action in actions)
    clears = sum(action["path"] == "/clear" for action in actions)
    counters_ok = (
        state["measure_count"] == measures == arena.measure_count
        and state["clear_count"] == clears == arena.clear_count
    )
    coverage_ok = (
        state.get("coverage_points_total") == 22
        and state.get("coverage_geometry") == baseline.coverage_certificate()
    )
    domain_ok = all(math.hypot(source.x, source.y) <= 1800.0 + 1e-7 for source in sources)
    structural = candidate_state_checks(state) if is_candidate else {
        "all_checks_pass": True, "checks": {"not_applicable_to_frozen_baseline": True}
    }
    checks = {
        "completed": bool(state.get("completed")),
        "not_stopped": not state.get("stopped"),
        "full_source_clear": full_clear,
        "channels_accounted": accounted,
        "no_true_source_false_excluded": not false_excluded,
        "clock_accounting": clock_ok,
        "action_counters": counters_ok,
        "ring22_certificate": coverage_ok,
        "source_domain_valid": domain_ok,
        "zero_post_clear_measurements": decomposition["post_clear_measure_count"] == 0,
        "candidate_structural_checks": structural["all_checks_pass"],
    }
    source_count = len(sources)
    metrics = {
        "all_checks_pass": all(checks.values()), "checks": checks,
        "candidate_structural_checks": structural,
        "source_count": source_count, "time_s": arena.virtual_time_s,
        "movement_m": decomposition["movement_m"],
        "mean_movement_m_per_source": decomposition["movement_m"] / source_count,
        "mean_total_s_per_source": arena.virtual_time_s / source_count,
        "movement_s": decomposition["movement_s"],
        "measure_s": decomposition["measure_s"],
        "switch_s": decomposition["switch_s"],
        "successful_clear_s": decomposition["successful_clear_s"],
        "failed_clear_s": decomposition["failed_clear_s"],
        "clock_residual_s": clock_residual,
        "measure_count": arena.measure_count, "clear_count": arena.clear_count,
        "policy_cpu_s": cpu_s,
        "waypoint_event_count": len(state.get("edge_waypoint_events", [])),
        "waypoint_no_signal_count": state.get("edge_waypoint_no_signal_count", 0),
        "waypoint_reorder_count": state.get("edge_waypoint_reorder_count", 0),
        "waypoint_forced_continue_count": state.get("edge_waypoint_forced_continue_count", 0),
        "public_action_fingerprint": public_fingerprint(actions),
    }
    return metrics, state, actions


def summarize(rows, name):
    values = [row[name] for row in rows]
    total_sources = sum(item["source_count"] for item in values)
    total_time = sum(item["time_s"] for item in values)
    total_movement = sum(item["movement_m"] for item in values)
    summary = {
        "case_count": len(values),
        "all_checks_pass": all(item["all_checks_pass"] for item in values),
        "total_sources": total_sources,
        "mean_case_movement_m_per_source": statistics.mean(
            item["mean_movement_m_per_source"] for item in values
        ),
        "pooled_movement_m_per_source": total_movement / total_sources,
        "mean_case_total_s_per_source": statistics.mean(
            item["mean_total_s_per_source"] for item in values
        ),
        "pooled_total_s_per_source": total_time / total_sources,
        "max_policy_cpu_s": max(item["policy_cpu_s"] for item in values),
        "total_waypoint_events": sum(item["waypoint_event_count"] for item in values),
        "total_waypoint_no_signal": sum(item["waypoint_no_signal_count"] for item in values),
        "total_waypoint_reorders": sum(item["waypoint_reorder_count"] for item in values),
        "total_waypoint_forced_continues": sum(
            item["waypoint_forced_continue_count"] for item in values
        ),
    }
    for component in (
        "movement_s", "measure_s", "switch_s", "successful_clear_s", "failed_clear_s"
    ):
        summary[f"mean_{component}_per_source"] = statistics.mean(
            item[component] / item["source_count"] for item in values
        )
        summary[f"pooled_{component}_per_source"] = sum(
            item[component] for item in values
        ) / total_sources
    return summary


def input_manifest():
    paths = [
        CASES, GENERATION,
        ROOT / "planning/q4_edge_waypoint_v1_experiment.md",
        ROOT / "planning/q4_edge_waypoint_v1_predeclaration.json",
        ROOT / "planning/q4_edge_waypoint_v1_candidate_freeze.json",
        ROOT / "code/Q4/q4_code_plan_edge_waypoint_v1.md",
        ROOT / "code/shared/generate_q4_edge_waypoint_v1_cases.py",
        ROOT / "code/shared/audit_q4_edge_waypoint_v1_offline.py",
        Path(__file__), Path(candidate.__file__), Path(baseline.__file__),
        ROOT / "code/shared/q4_edge_waypoint_geometry.py",
        ROOT / "code/shared/q4_joint_compatibility.py",
        ROOT / "code/shared/search_policy.py",
        ROOT / "code/shared/search_policy_polygon.py",
        ROOT / "code/shared/search_policy_q4_batch.py",
        ROOT / "code/shared/search_policy_q4_reuse.py",
        ROOT / "code/shared/search_policy_q4_ring22.py",
        ROOT / "code/shared/search_policy_joint_route_q3.py",
        ROOT / "code/shared/evaluate_q4_shadow_recovery_development.py",
        ROOT / "code/shared/offline_arena.py",
        ROOT / "code/shared/simulator_client.py",
        ROOT / "results/optimization_200_400/ring22_continuous_certificate.json",
    ]
    return [{"path": relative(path), "sha256": sha(path)} for path in paths]


def make_run_summary(rows, summaries, comparison_path_hash, inputs, accepted):
    outputs = [{"path": relative(COMPARISON), "sha256": comparison_path_hash}]
    methods = []
    for method_id, role, module, label in (
        ("q4_ring22_joint_route_v2", "usable_baseline", baseline, "baseline"),
        ("q4_edge_waypoint_v1", "main", candidate, "candidate"),
    ):
        methods.append({
            "method_id": method_id, "role": role, "status": "success",
            "language": "python",
            "script": {"path": relative(Path(module.__file__)), "sha256": sha(Path(module.__file__))},
            "outputs": outputs,
            "metrics_summary": summaries[label],
            "constraint_violations": [] if summaries[label]["all_checks_pass"] else ["hard_check_failure"],
            "execution_time_seconds": sum(row[label]["policy_cpu_s"] for row in rows),
        })
    return {
        "schema_version": 2, "artifact_type": "run-summary", "question_id": "Q4",
        "round": "edge_waypoint_v1", "approved_decision_id": "q4-method-20260910",
        "methods": methods,
        "comparison": {
            "population_id": sha(CASES),
            "metric_contract": (
                "same fresh 42 cases; all-clear first; primary mean movement m/source; "
                "co-primary total virtual s/source with action decomposition"
            ),
        },
        "selected_output": {
            "method_id": "q4_edge_waypoint_v1" if accepted else "q4_ring22_joint_route_v2",
            "parameters": {
                "minimum_scan_edge_m": 1000.0, "waypoint_fractions": [0.5, 0.25, 0.75],
                "minimum_range_m": 1000.0, "localizer_extra_budget": 8,
                "ring22_points": 22, "legacy_fallback_points": 304,
            },
            "composition": "screen-only structural winner; not an approved method replacement",
            "evaluator": "synthetic_black_box_offline_arena",
            "resolution": {"fresh_seed_start": 65000, "fresh_seed_end": 65041},
        },
        "output_obligations": [
            {"id": "fresh_42_paired_offline_screen", "satisfied": True, "evidence": outputs},
            {"id": "official_practice_evaluation", "satisfied": False, "evidence": []},
            {"id": "formal_test", "satisfied": False, "evidence": []},
        ],
        "random_seed": 65000,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "scope": "fresh synthetic paired offline only"},
        "inputs": inputs,
        "contract_evidence": {
            "population_reconciliation": {"generated": 42, "used_per_method": 42, "excluded": 0},
            "input_population_contract": {
                "status": "APPLICABLE", "audited": 42, "parsed": 42,
                "excluded": 0, "eligible": 42,
                "used_by_method": {"q4_ring22_joint_route_v2": 42, "q4_edge_waypoint_v1": 42},
            },
            "continuous_coverage_reference": "results/optimization_200_400/ring22_continuous_certificate.json",
            "error_field": "fixed by exact coordinate/channel/seed in offline diagnostic",
        },
        "extensions": {
            "candidate_unapproved": True, "automatic_screen_accepted": accepted,
            "not_official": True, "formal_tests_used": 0, "live_api_calls": 0,
            "warnings": [
                "OfflineArena is diagnostic and not an official simulator replica.",
                "A passing screen would still require a separate human method decision."
            ],
        },
    }


def main():
    if COMPARISON.exists() or DECISION.exists() or RUN_SUMMARY.exists():
        raise FileExistsError("one-shot output already exists; evaluation will not rerun")
    if not CASES.is_file() or not GENERATION.is_file():
        raise FileNotFoundError("frozen fresh cases must be generated first")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if len(cases) != 42 or [case["seed"] for case in cases] != list(range(65000, 65042)):
        raise ValueError("fresh case population does not match the predeclaration")
    inputs = input_manifest()
    save(MANIFEST, inputs)
    if STATES.exists() and any(STATES.iterdir()):
        raise FileExistsError("partial state outputs already exist")
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "seed": case["seed"],
               "error_mode": case["error_mode"], "source_count": len(case["sources"])}
        for name, policy, is_candidate in (
            ("baseline", baseline.run_search, False),
            ("candidate", candidate.run_search, True),
        ):
            metrics, state, actions = run(policy, case, is_candidate)
            row[name] = metrics
            state_path = STATES / f"{case['case_id']}__{name}.json"
            save(state_path, {"state": state, "actions": actions})
            row[f"{name}_state"] = {"path": relative(state_path), "sha256": sha(state_path)}
        row["candidate_to_baseline_time_ratio"] = (
            row["candidate"]["time_s"] / row["baseline"]["time_s"]
        )
        row["candidate_to_baseline_movement_ratio"] = (
            row["candidate"]["movement_m"] / row["baseline"]["movement_m"]
        )
        rows.append(row)
        print(json.dumps({
            "case_id": row["case_id"],
            "hard": row["baseline"]["all_checks_pass"] and row["candidate"]["all_checks_pass"],
            "waypoints": row["candidate"]["waypoint_event_count"],
            "movement_ratio": row["candidate_to_baseline_movement_ratio"],
            "time_ratio": row["candidate_to_baseline_time_ratio"],
        }, ensure_ascii=False), flush=True)
    summaries = {name: summarize(rows, name) for name in ("baseline", "candidate")}
    baseline_move = summaries["baseline"]["mean_case_movement_m_per_source"]
    candidate_move = summaries["candidate"]["mean_case_movement_m_per_source"]
    movement_reduction = (baseline_move - candidate_move) / baseline_move
    baseline_time = summaries["baseline"]["mean_case_total_s_per_source"]
    candidate_time = summaries["candidate"]["mean_case_total_s_per_source"]
    time_reduction = (baseline_time - candidate_time) / baseline_time
    worst_time = max(rows, key=lambda row: row["candidate_to_baseline_time_ratio"])
    hard = all(summary["all_checks_pass"] for summary in summaries.values())
    gate_checks = {
        "all_42_paired_hard_checks_pass": hard,
        "mean_movement_reduction_at_least_5_percent": movement_reduction >= MOVEMENT_RELATIVE_MIN,
        "mean_total_time_reduction_at_least_2_percent": time_reduction >= TIME_RELATIVE_MIN,
        "worst_single_case_total_time_ratio_at_most_1_10":
            worst_time["candidate_to_baseline_time_ratio"] <= WORST_TIME_RATIO_MAX,
    }
    accepted = all(gate_checks.values())
    comparison = {
        "schema_version": 1,
        "artifact_type": "q4_edge_waypoint_v1_paired_offline_result",
        "scope": "FRESH_SYNTHETIC_PAIRED_OFFLINE_ONLY_NOT_PRACTICE_OR_FORMAL_TEST",
        "accepted": accepted, "decision": "SCREEN_ELIGIBLE" if accepted else "REJECT_AND_STOP",
        "all_hard_checks_pass": hard,
        "predeclared_gate": {
            "movement_relative_reduction_min": MOVEMENT_RELATIVE_MIN,
            "total_time_relative_reduction_min": TIME_RELATIVE_MIN,
            "worst_time_ratio_max": WORST_TIME_RATIO_MAX,
        },
        "gate_checks": gate_checks,
        "mean_movement_relative_reduction": movement_reduction,
        "mean_total_time_relative_reduction": time_reduction,
        "worst_time_case": {
            "case_id": worst_time["case_id"],
            "ratio": worst_time["candidate_to_baseline_time_ratio"],
        },
        "summaries": summaries, "rows": rows, "inputs": inputs,
    }
    save(COMPARISON, comparison)
    save(DECISION, {
        "schema_version": 1, "experiment_id": "q4_edge_waypoint_v1",
        "accepted": accepted, "decision": comparison["decision"],
        "next_step": "stop_and_preserve_complete_evidence_without_tuning_or_additional_cases",
        "paper_or_frozen_policy_change_authorized": False,
        "practice_or_formal_test_authorized": False,
    })
    run_summary = make_run_summary(rows, summaries, sha(COMPARISON), inputs, accepted)
    save(RUN_SUMMARY, run_summary)
    print(json.dumps({
        "accepted": accepted, "all_hard_checks_pass": hard,
        "movement_reduction": movement_reduction, "time_reduction": time_reduction,
        "worst_time_ratio": worst_time["candidate_to_baseline_time_ratio"],
        "waypoint_events": summaries["candidate"]["total_waypoint_events"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
