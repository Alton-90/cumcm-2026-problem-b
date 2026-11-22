"""One-shot paired 36-case development screen for Q3 service-aware routing.

The official simulator, GUI, localhost, and the independent 84-case strict set
are outside this script's scope.  Truth is owned by the existing offline
evaluator and is used only for post-run completion checks.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
from time import perf_counter

import q3_service_aware_rollout as service
import search_policy_q3_combined as baseline
import search_policy_q3_service_aware as candidate
from evaluate_q3_robust_parallax_development import run, save, sha256, same_point


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q3_service_aware" / "development"
CASE_FILE = ROOT / "results" / "feasible_q3" / "cases.json"
CASE_COUNT = 36
RELATIVE_THRESHOLD = 0.03
ABSOLUTE_THRESHOLD_S_PER_SOURCE = 5.0
WORST_CASE_RATIO_LIMIT = 1.10


def static_policy_audit():
    files = [
        ROOT / "code" / "shared" / "q3_service_aware_rollout.py",
        ROOT / "code" / "shared" / "search_policy_q3_service_aware.py",
    ]
    forbidden_imports = {"offline_arena", "simulator_client", "urllib", "requests"}
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
            elif isinstance(node, ast.Name) and node.id in {"OfflineArena", "Source", "LocalHttpTransport"}:
                errors.append(f"{path.name}: forbidden name {node.id}")
    return {
        "all_checks_pass": not errors,
        "errors": errors,
        "files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in files],
        "strict_confirmation_inputs_opened": 0,
        "live_or_gui_calls": 0,
    }


def _score_close(first, second, tolerance=1e-8):
    if first is None or second is None:
        return first is second
    return abs(float(first) - float(second)) <= tolerance


def audit_service_events(state, actions):
    errors = []
    events = state.get("service_route_events", [])
    replayed_count = 0
    override_count = 0
    maximum_replay_score_error = 0.0
    maximum_predicted_gain = 0.0
    for index, event in enumerate(events):
        prefix = f"event {index}"
        snapshot = event.get("snapshot", {})
        decision = event.get("decision", {})
        route = snapshot.get("proxy_route", [])
        if not route or not 1 <= snapshot.get("prefix_tasks", 0) <= service.PREFIX_TASKS:
            errors.append(prefix + ": invalid frozen-route prefix")
            continue
        replay = service.select_from_snapshot(snapshot)
        replayed_count += 1
        if (
            replay["baseline_key"] != decision.get("baseline_key")
            or replay["selected_key"] != decision.get("selected_key")
            or replay["selected_rank"] != decision.get("selected_rank")
            or replay["override"] != decision.get("override")
            or not _score_close(replay["predicted_gain_s"], decision.get("predicted_gain_s"))
            or not _score_close(replay["best_available_gain_s"], decision.get("best_available_gain_s"))
        ):
            errors.append(prefix + ": selection is not replayable from saved public snapshot")
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
                    maximum_replay_score_error = max(
                        maximum_replay_score_error,
                        abs(expected["score_s"] - observed["score_s"]),
                    )
        if decision.get("baseline_key") != f"{route[0]['kind']}:{route[0]['identifier']}":
            errors.append(prefix + ": baseline task is not frozen route rank zero")
        selected_rank = decision.get("selected_rank")
        if not isinstance(selected_rank, int) or not 0 <= selected_rank < snapshot["prefix_tasks"]:
            errors.append(prefix + ": selected task lies outside fixed prefix")
            continue
        selected_record = decision["records"][selected_rank]
        if decision.get("override"):
            override_count += 1
            gain = decision.get("predicted_gain_s")
            maximum_predicted_gain = max(maximum_predicted_gain, float(gain))
            if selected_rank == 0 or gain + service.TIE_TOLERANCE_S < service.OVERRIDE_MARGIN_S:
                errors.append(prefix + ": override violates fixed five-second margin")
        elif selected_rank != 0:
            errors.append(prefix + ": non-override did not retain frozen first task")

        for task in route[: snapshot["prefix_tasks"]]:
            if task["kind"] == "scan":
                planned = task["public"]["planned_measure_channels"]
                derived = [
                    item["channel"]
                    for item in task["public"]["channel_decisions"]
                    if item["decision"] == "measure"
                ]
                if planned != derived or planned != sorted(set(planned)):
                    errors.append(prefix + ": scan plan differs from saved public decisions")

        action_index = event.get("action_count_before")
        if not isinstance(action_index, int) or not 0 <= action_index < len(actions):
            errors.append(prefix + ": selected task has no auditable first action")
            continue
        action = actions[action_index]
        task = route[selected_rank]
        if task["kind"] == "scan":
            planned = task["public"]["planned_measure_channels"]
            if planned and (
                action["path"] != "/measure"
                or action["channel"] != planned[0]
                or not same_point(action["point"], task["point"])
            ):
                errors.append(prefix + ": selected scan differs from first public action")
        else:
            public = task["public"]
            mode = public["mode"]
            expected_path = "/clear" if mode in ("candidate_clear", "polygon_clear") else "/measure"
            if mode == "current_pose_measure":
                expected_point = snapshot["position"]
            elif mode in ("candidate_clear", "polygon_clear"):
                expected_point = public["center"]
            else:
                expected_point = selected_record["components"].get("selected_entry_point")
            if (
                action["path"] != expected_path
                or action["channel"] != task["identifier"]
                or expected_point is None
                or not same_point(action["point"], expected_point)
            ):
                errors.append(prefix + ": selected source macro differs from first public action")

    if override_count != state.get("service_route_override_count"):
        errors.append("service-route override counter mismatch")
    return {
        "all_checks_pass": not errors,
        "errors": errors,
        "event_count": len(events),
        "replayed_event_count": replayed_count,
        "override_count": override_count,
        "maximum_replay_score_error_s": maximum_replay_score_error,
        "maximum_predicted_gain_s": maximum_predicted_gain,
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
        "service_route_event_count": sum(
            value.get("service_route_audit", {}).get("event_count", 0) for value in values
        ),
        "service_route_override_count": sum(
            value.get("service_route_audit", {}).get("override_count", 0) for value in values
        ),
        "service_route_scoring_cpu_s": sum(
            value.get("service_route_scoring_cpu_s", 0.0) for value in values
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
        "code/shared/q3_service_aware_rollout.py",
        "code/shared/search_policy_q3_service_aware.py",
        "code/shared/evaluate_q3_service_aware_development.py",
        "code/shared/evaluate_q3_robust_parallax_development.py",
        "code/shared/offline_arena.py",
        "code/shared/simulator_client.py",
        "planning/q3_service_aware_rollout_experiment.md",
        "results/feasible_q3/cases.json",
        str(cases_copy.relative_to(ROOT)).replace("\\", "/"),
    ]
    return [{"path": path, "sha256": sha256(ROOT / path)} for path in paths]


def main():
    if OUT.exists():
        raise RuntimeError("service-aware development output already exists; refusing overwrite")
    OUT.mkdir(parents=True)
    cases = json.loads(CASE_FILE.read_text(encoding="utf-8"))[:CASE_COUNT]
    if len(cases) != CASE_COUNT or any(
        case.get("question") != 3 or case.get("split") != "development" for case in cases
    ):
        raise ValueError("expected the frozen first 36 Q3 development cases")
    cases_copy = OUT / "cases.json"
    save(cases_copy, cases)
    inputs = input_manifest(cases_copy)
    save(OUT / "inputs_before_run.json", inputs)
    static_audit = static_policy_audit()
    scorer_tests = service.self_test()
    save(OUT / "preflight.json", {"static_policy_audit": static_audit, "scorer_self_test": scorer_tests})
    if not static_audit["all_checks_pass"] or not scorer_tests["all_checks_pass"]:
        raise RuntimeError("preflight failed before paired development run")

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
            if name == "candidate":
                event_audit = audit_service_events(state, actions)
                metrics["service_route_audit"] = event_audit
                metrics["service_route_scoring_cpu_s"] = state.get(
                    "service_route_scoring_cpu_s", 0.0
                )
                metrics["all_checks_pass"] = bool(
                    metrics["all_checks_pass"] and event_audit["all_checks_pass"]
                )
            save(
                OUT / "states" / f"{case['case_id']}__{name}.json",
                {
                    "case_id": case["case_id"],
                    "metrics": metrics,
                    "state": state,
                    "actions": actions,
                },
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
            f"overrides={row['candidate']['service_route_audit']['override_count']} "
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
    population_ok = len(rows) == CASE_COUNT
    all_hard_checks = bool(
        unchanged
        and population_ok
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
        "artifact_type": "q3_service_aware_paired_development_screen",
        "scope": "SYNTHETIC_FIXED_36_DEVELOPMENT_ONLY",
        "baseline": "q3_combined_v1",
        "candidate": candidate.VERSION,
        "case_count": len(rows),
        "all_hard_checks_pass": all_hard_checks,
        "inputs_unchanged": unchanged,
        "development_gate_pass": development_gate,
        "method_accepted": False,
        "predeclared_gate": {
            "minimum_relative_improvement_or": RELATIVE_THRESHOLD,
            "minimum_absolute_saving_s_per_source_or": ABSOLUTE_THRESHOLD_S_PER_SOURCE,
            "maximum_case_total_time_ratio": WORST_CASE_RATIO_LIMIT,
            "all_36_pairs_and_full_clear_required": True,
        },
        "performance_gate_pass": performance_gate,
        "worst_case_gate_pass": worst_case_gate,
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
            "route_prefix_tasks": service.PREFIX_TASKS,
            "override_margin_s": service.OVERRIDE_MARGIN_S,
            "exit_allowance_s": service.EXIT_ALLOWANCE_S,
            "maximum_public_hypotheses": 9,
            "local_rollout_angle_biases_deg": [-1.0, 0.0, 1.0],
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
        "next_step": (
            "independent_review_before_separate_new_holdout"
            if development_gate
            else "stop_without_tuning_or_holdout_or_simulator_run"
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
            "all_checks_pass": all_hard_checks,
            "development_gate_pass": development_gate,
            "mean_s_per_source": summaries["candidate"]["mean_s_per_source"],
            "baseline_mean_s_per_source": summaries["baseline"]["mean_s_per_source"],
            "absolute_saving_s_per_source": saving,
            "relative_improvement": relative,
            "worst_case_ratio": worst["candidate_to_baseline_ratio"],
            "comparison_artifact": "results/q3_service_aware/development/comparison.json",
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
