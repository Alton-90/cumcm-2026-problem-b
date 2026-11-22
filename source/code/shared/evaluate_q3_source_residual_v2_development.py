"""One-shot post-v1 exploratory screen for Q3 source-residual ordering.

This script uses only the already seen 36-case synthetic development set.  It
does not import the strict-84 wrapper, read a strict-set file, or contact a live
simulator.  The paired baseline is the unchanged cost-rollout-v2 controller so
the causal contrast is limited to source-prefix ordering.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import statistics
from time import perf_counter

import q3_source_residual_v2 as residual
import search_policy_q3_cost_rollout_v2 as baseline
import search_policy_q3_source_residual_v2 as candidate
from evaluate_q3_robust_parallax_development import run, save, sha256, same_point


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q3_source_residual_v2" / "development"
CASE_FILE = ROOT / "results" / "feasible_q3" / "cases.json"
CASE_COUNT = 36
RELATIVE_THRESHOLD = 0.03
ABSOLUTE_THRESHOLD_S_PER_SOURCE = 5.0
WORST_CASE_RATIO_LIMIT = 1.10


def static_policy_audit():
    files = [
        ROOT / "code" / "shared" / "q3_source_residual_v2.py",
        ROOT / "code" / "shared" / "search_policy_q3_source_residual_v2.py",
    ]
    forbidden_imports = {
        "offline_arena",
        "simulator_client",
        "urllib",
        "requests",
        "search_policy_q3_cost_rollout_strict84_selected",
    }
    forbidden_attributes = {"sources", "transport", "source_x", "source_y", "source_radius"}
    errors = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    if item.name.split(".")[0] in forbidden_imports:
                        errors.append(f"{path.name}: forbidden import {item.name}")
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
                if module in forbidden_imports:
                    errors.append(f"{path.name}: forbidden import {node.module}")
            elif isinstance(node, ast.Attribute) and node.attr.lower() in forbidden_attributes:
                errors.append(f"{path.name}: forbidden attribute {node.attr}")
            elif isinstance(node, ast.Name) and node.id in {
                "OfflineArena",
                "Source",
                "LocalHttpTransport",
            }:
                errors.append(f"{path.name}: forbidden name {node.id}")
    return {
        "all_checks_pass": not errors,
        "errors": errors,
        "files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in files],
        "strict84_wrapper_imported": False,
        "strict_confirmation_inputs_opened": 0,
        "live_or_gui_calls": 0,
    }


def _score_close(first, second, tolerance=1e-8):
    if first is None or second is None:
        return first is second
    return abs(float(first) - float(second)) <= tolerance


def audit_events(state, actions):
    errors = []
    events = state.get("source_residual_events", [])
    replayed = 0
    overrides = 0
    scan_locks = 0
    source_decisions = 0
    maximum_score_error = 0.0
    for index, event in enumerate(events):
        prefix = f"event {index}"
        snapshot = event.get("snapshot", {})
        decision = event.get("decision", {})
        route = snapshot.get("proxy_route", [])
        if not route:
            errors.append(prefix + ": empty proxy route")
            continue
        replay = residual.select_from_snapshot(snapshot)
        replayed += 1
        for key in ("reason", "baseline_key", "selected_key", "selected_rank", "override"):
            if replay.get(key) != decision.get(key):
                errors.append(prefix + f": replay mismatch for {key}")
        for key in ("predicted_gain_s", "best_available_gain_s"):
            if not _score_close(replay.get(key), decision.get(key)):
                errors.append(prefix + f": replay mismatch for {key}")
        if len(replay["records"]) != len(decision.get("records", [])):
            errors.append(prefix + ": replay record count mismatch")
        else:
            for expected, observed in zip(replay["records"], decision["records"]):
                if expected["key"] != observed.get("key") or not _score_close(
                    expected["score_s"], observed.get("score_s")
                ):
                    errors.append(prefix + ": replay score mismatch")
                    break
                if expected["score_s"] is not None:
                    maximum_score_error = max(
                        maximum_score_error,
                        abs(expected["score_s"] - observed["score_s"]),
                    )

        baseline_key = f"{route[0]['kind']}:{route[0]['identifier']}"
        if decision.get("baseline_key") != baseline_key:
            errors.append(prefix + ": baseline key is not frozen route rank zero")
        selected_rank = decision.get("selected_rank")
        if not isinstance(selected_rank, int) or not 0 <= selected_rank < len(route):
            errors.append(prefix + ": selected rank is invalid")
            continue

        leading_sources = 0
        for task in route:
            if task["kind"] != "source" or leading_sources >= residual.SOURCE_PREFIX_LIMIT:
                break
            leading_sources += 1
        if snapshot.get("consecutive_source_count") != leading_sources:
            errors.append(prefix + ": source-prefix count differs from frozen route")

        if route[0]["kind"] == "scan":
            scan_locks += 1
            if (
                selected_rank != 0
                or decision.get("override")
                or route[selected_rank]["kind"] != "scan"
                or decision.get("records")
            ):
                errors.append(prefix + ": frozen first scan was not completely retained")
        else:
            source_decisions += 1
            if not 0 <= selected_rank < leading_sources or route[selected_rank]["kind"] != "source":
                errors.append(prefix + ": source decision left consecutive source prefix")
            if decision.get("override"):
                overrides += 1
                if (
                    selected_rank == 0
                    or decision.get("predicted_gain_s", -math.inf)
                    + residual.TIE_TOLERANCE_S
                    < residual.MIN_RESIDUAL_SAVING_S
                ):
                    errors.append(prefix + ": source override violates fixed margin")
            elif selected_rank != 0:
                errors.append(prefix + ": non-override changed frozen first source")

            for record in decision.get("records", []):
                components = record["components"]
                rank = record["rank"]
                remaining = [task for task_index, task in enumerate(route) if task_index != rank]
                expected_successor = None if not remaining else (
                    f"{remaining[0]['kind']}:{remaining[0]['identifier']}"
                )
                if components.get("successor_key") != expected_successor:
                    errors.append(prefix + ": residual successor is not frozen remainder head")
                if record["score_s"] is not None:
                    recomputed_score = (
                        components["entry_move_s"]
                        + components["predicted_exit_to_frozen_successor_s"]
                    )
                    if not _score_close(recomputed_score, record["score_s"]):
                        errors.append(prefix + ": residual score contains non-transfer cost")
                    omitted = set(components.get("omitted_order_independent_service_costs", []))
                    if omitted != {
                        "measure",
                        "channel_switch",
                        "clear",
                        "within_source_localization",
                        "finite_fallback",
                    }:
                        errors.append(prefix + ": intrinsic service cancellation is incomplete")

        action_index = event.get("action_count_before")
        if not isinstance(action_index, int) or not 0 <= action_index < len(actions):
            errors.append(prefix + ": no auditable first public action")
            continue
        action = actions[action_index]
        selected_task = route[selected_rank]
        if selected_task["kind"] == "scan":
            if action["path"] != "/measure" or not same_point(
                action["point"], selected_task["point"]
            ):
                errors.append(prefix + ": locked scan differs from first public action")
        else:
            selected_record = decision["records"][selected_rank]
            components = selected_record["components"]
            mode = components.get("mode")
            expected_path = (
                "/clear" if mode in ("candidate_clear", "polygon_clear") else "/measure"
            )
            expected_point = components.get("execution_entry_point")
            if (
                action["path"] != expected_path
                or action["channel"] != selected_task["identifier"]
                or expected_point is None
                or not same_point(action["point"], expected_point)
            ):
                errors.append(prefix + ": selected source entry differs from first public action")

    if overrides != state.get("source_residual_override_count"):
        errors.append("source residual override counter mismatch")
    if scan_locks != state.get("source_residual_scan_lock_count"):
        errors.append("scan lock counter mismatch")
    return {
        "all_checks_pass": not errors,
        "errors": errors,
        "event_count": len(events),
        "replayed_event_count": replayed,
        "override_count": overrides,
        "scan_lock_count": scan_locks,
        "source_decision_count": source_decisions,
        "category_flip_count": 0 if not errors else None,
        "maximum_replay_score_error_s": maximum_score_error,
    }


def summarize(rows, name):
    values = [row[name] for row in rows]
    output = {
        "case_count": len(values),
        "all_checks_pass": all(value["all_checks_pass"] for value in values),
        "mean_s_per_source": statistics.mean(value["mean_s"] for value in values),
        "total_policy_wall_s": sum(value["policy_wall_s"] for value in values),
        "max_policy_wall_s": max(value["policy_wall_s"] for value in values),
        "fallback_clear_count": sum(value["fallback_count"] for value in values),
        "cost_rollout_scoring_cpu_s": sum(
            value.get("cost_rollout_scoring_cpu_s", 0.0) for value in values
        ),
        "source_residual_scoring_cpu_s": sum(
            value.get("source_residual_scoring_cpu_s", 0.0) for value in values
        ),
        "source_residual_event_count": sum(
            value.get("source_residual_audit", {}).get("event_count", 0) for value in values
        ),
        "source_residual_override_count": sum(
            value.get("source_residual_audit", {}).get("override_count", 0) for value in values
        ),
        "scan_lock_count": sum(
            value.get("source_residual_audit", {}).get("scan_lock_count", 0) for value in values
        ),
    }
    for key in ("movement_s", "measure_s", "switch_s", "successful_clear_s", "failed_clear_s"):
        output[key + "_per_source"] = statistics.mean(
            value[key] / value["source_count"] for value in values
        )
    return output


def input_manifest(cases_copy):
    paths = [
        "code/shared/search_policy.py",
        "code/shared/search_policy_compact.py",
        "code/shared/search_policy_feasible_q3.py",
        "code/shared/search_policy_q3_combined.py",
        "code/shared/route_multistart.py",
        "code/shared/q3_cost_rollout_v2.py",
        "code/shared/search_policy_q3_cost_rollout_v2.py",
        "code/shared/q3_source_residual_v2.py",
        "code/shared/search_policy_q3_source_residual_v2.py",
        "code/shared/evaluate_q3_source_residual_v2_development.py",
        "code/shared/evaluate_q3_robust_parallax_development.py",
        "code/shared/offline_arena.py",
        "code/shared/simulator_client.py",
        "planning/q3_source_residual_v2_experiment.md",
        "results/feasible_q3/cases.json",
        str(cases_copy.relative_to(ROOT)).replace("\\", "/"),
    ]
    return [{"path": path, "sha256": sha256(ROOT / path)} for path in paths]


def main():
    if OUT.exists():
        raise RuntimeError("source-residual-v2 development output exists; refusing overwrite")
    OUT.mkdir(parents=True)
    cases = json.loads(CASE_FILE.read_text(encoding="utf-8"))[:CASE_COUNT]
    if len(cases) != CASE_COUNT or any(
        case.get("question") != 3 or case.get("split") != "development" for case in cases
    ):
        raise ValueError("expected the same frozen first 36 Q3 development cases")
    cases_copy = OUT / "cases.json"
    save(cases_copy, cases)
    inputs = input_manifest(cases_copy)
    save(OUT / "inputs_before_run.json", inputs)
    static_audit = static_policy_audit()
    scorer_tests = residual.self_test()
    save(OUT / "preflight.json", {"static_policy_audit": static_audit, "scorer_self_test": scorer_tests})
    if not static_audit["all_checks_pass"] or not scorer_tests["all_checks_pass"]:
        raise RuntimeError("preflight failed before the one-shot paired run")

    rows = []
    suite_started = perf_counter()
    for index, case in enumerate(cases, start=1):
        row = {
            "case_id": case["case_id"],
            "seed": case["seed"],
            "error_mode": case["error_mode"],
        }
        for name, policy in (("baseline", baseline.run_search), ("candidate", candidate.run_search)):
            metrics, state, actions = run(policy, case)
            metrics["cost_rollout_scoring_cpu_s"] = state.get(
                "cost_rollout_scoring_cpu_s", 0.0
            )
            if name == "candidate":
                event_audit = audit_events(state, actions)
                metrics["source_residual_audit"] = event_audit
                metrics["source_residual_scoring_cpu_s"] = state.get(
                    "source_residual_scoring_cpu_s", 0.0
                )
                metrics["all_checks_pass"] = bool(
                    metrics["all_checks_pass"] and event_audit["all_checks_pass"]
                )
            save(
                OUT / "states" / f"{case['case_id']}__{name}.json",
                {"case_id": case["case_id"], "metrics": metrics, "state": state, "actions": actions},
            )
            row[name] = metrics
        row["candidate_to_baseline_ratio"] = (
            row["candidate"]["time_s"] / row["baseline"]["time_s"]
        )
        row["candidate_minus_baseline_s_per_source"] = (
            row["candidate"]["mean_s"] - row["baseline"]["mean_s"]
        )
        rows.append(row)
        save(OUT / "progress.json", {"completed_pairs": len(rows), "rows": rows})
        print(
            f"[{index:02d}/{CASE_COUNT}] {case['case_id']} "
            f"base={row['baseline']['mean_s']:.6f} "
            f"candidate={row['candidate']['mean_s']:.6f} "
            f"ratio={row['candidate_to_baseline_ratio']:.6f} "
            f"overrides={row['candidate']['source_residual_audit']['override_count']} "
            f"checks={row['candidate']['all_checks_pass']}",
            flush=True,
        )

    summaries = {name: summarize(rows, name) for name in ("baseline", "candidate")}
    saving = summaries["baseline"]["mean_s_per_source"] - summaries["candidate"]["mean_s_per_source"]
    relative = saving / summaries["baseline"]["mean_s_per_source"]
    worst = max(rows, key=lambda item: item["candidate_to_baseline_ratio"])
    unchanged = all(sha256(ROOT / item["path"]) == item["sha256"] for item in inputs)
    after_inputs = [
        {"path": item["path"], "sha256": sha256(ROOT / item["path"])} for item in inputs
    ]
    save(OUT / "inputs_after_run.json", after_inputs)
    all_hard_checks = bool(
        unchanged
        and len(rows) == CASE_COUNT
        and summaries["baseline"]["all_checks_pass"]
        and summaries["candidate"]["all_checks_pass"]
        and static_audit["all_checks_pass"]
        and scorer_tests["all_checks_pass"]
    )
    performance_gate = relative >= RELATIVE_THRESHOLD or saving >= ABSOLUTE_THRESHOLD_S_PER_SOURCE
    worst_case_gate = worst["candidate_to_baseline_ratio"] <= WORST_CASE_RATIO_LIMIT
    development_gate = bool(all_hard_checks and performance_gate and worst_case_gate)
    result = {
        "schema_version": 2,
        "artifact_type": "q3_source_residual_v2_paired_development_screen",
        "scope": "POST_V1_EXPLORATORY_FIXED_36_DEVELOPMENT_ONLY",
        "independent_validation": False,
        "baseline": "q3_cost_rollout_v2",
        "candidate": candidate.VERSION,
        "primary_causal_contrast": "source_prefix_residual_transfer_ordering_only",
        "frozen_complete_fallback": "q3_combined_v1",
        "case_count": len(rows),
        "all_hard_checks_pass": all_hard_checks,
        "inputs_unchanged": unchanged,
        "development_gate_pass": development_gate,
        "method_accepted": False,
        "performance_gate_pass": performance_gate,
        "worst_case_gate_pass": worst_case_gate,
        "predeclared_gate": {
            "minimum_relative_improvement_or": RELATIVE_THRESHOLD,
            "minimum_absolute_saving_s_per_source_or": ABSOLUTE_THRESHOLD_S_PER_SOURCE,
            "maximum_case_total_time_ratio": WORST_CASE_RATIO_LIMIT,
            "all_36_pairs_full_clear_and_event_replay_required": True,
        },
        "absolute_saving_s_per_source": saving,
        "relative_improvement": relative,
        "worst_case": {
            "case_id": worst["case_id"],
            "candidate_to_baseline_ratio": worst["candidate_to_baseline_ratio"],
            "candidate_minus_baseline_s_per_source": worst[
                "candidate_minus_baseline_s_per_source"
            ],
        },
        "parameters": {
            "consecutive_source_prefix_limit": residual.SOURCE_PREFIX_LIMIT,
            "minimum_residual_saving_s": residual.MIN_RESIDUAL_SAVING_S,
            "tie_tolerance_s": residual.TIE_TOLERANCE_S,
            "intrinsic_source_service_costs_cancelled": True,
            "scan_source_category_flips_allowed": False,
        },
        "summaries": summaries,
        "rows": rows,
        "static_policy_audit": static_audit,
        "scorer_self_test": scorer_tests,
        "inputs": inputs,
        "suite_wall_s": perf_counter() - suite_started,
        "population": {
            "audited": len(rows),
            "eligible": CASE_COUNT,
            "excluded": 0,
            "baseline_used": len(rows),
            "candidate_used": len(rows),
        },
        "online_practice_runs": 0,
        "formal_runs": 0,
        "strict_84_confirmation_runs": 0,
        "strict84_selected_wrapper_part_of_experiment": False,
        "next_step": (
            "await_independent_review_then_build_separate_new_holdout"
            if development_gate
            else "stop_without_tuning_rerun_holdout_or_simulator"
        ),
    }
    save(OUT / "comparison.json", result)
    save(
        OUT / "run_summary.json",
        {
            "schema_version": 2,
            "artifact_type": "run-summary",
            "question": "Q3",
            "run_scope": result["scope"],
            "method": candidate.VERSION,
            "baseline": result["baseline"],
            "status": "completed",
            "independent_validation": False,
            "all_checks_pass": all_hard_checks,
            "development_gate_pass": development_gate,
            "mean_s_per_source": summaries["candidate"]["mean_s_per_source"],
            "baseline_mean_s_per_source": summaries["baseline"]["mean_s_per_source"],
            "absolute_saving_s_per_source": saving,
            "relative_improvement": relative,
            "worst_case_ratio": worst["candidate_to_baseline_ratio"],
            "comparison_artifact": "results/q3_source_residual_v2/development/comparison.json",
        },
    )
    save(
        OUT / "decision.json",
        {
            "development_gate_pass": development_gate,
            "method_accepted": False,
            "reason": "development_gate_met" if development_gate else "development_gate_not_met",
            "next_step": result["next_step"],
        },
    )
    print(
        json.dumps(
            {
                "all_hard_checks_pass": all_hard_checks,
                "development_gate_pass": development_gate,
                "absolute_saving_s_per_source": saving,
                "relative_improvement": relative,
                "worst_case": result["worst_case"],
                "summaries": summaries,
                "suite_wall_s": result["suite_wall_s"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
