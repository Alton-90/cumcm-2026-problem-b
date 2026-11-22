"""Frozen 36-case paired offline screen for q4_service_aware_route_v1."""
from __future__ import annotations

import json
import math
from pathlib import Path

import search_policy_q4_ring22_joint_route as baseline
import search_policy_q4_service_aware_route as candidate
from evaluate_q4_domain_prior_development import run, save, sha, summarize


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q4_service_aware_route/development"
CASES = ROOT / "results/q4_reuse/development/cases.json"
RELATIVE_MIN = 0.05
ABSOLUTE_MIN_S_PER_SOURCE = 10.0
WORST_RATIO_MAX = 1.10
FLOAT_TOL = 1e-7


def _same_point(first, second):
    return math.dist(tuple(first), tuple(second)) <= FLOAT_TOL


def _choice_order(choice):
    return (choice["total_s"], choice["movement_s"], tuple(choice["first"]))


def verify_service_audit(state, actions):
    errors = []
    changed = 0
    decisions = state.get("service_route_decisions", [])
    for index, event in enumerate(decisions):
        nodes = event.get("nodes", [])
        profiles = event.get("profiles", [])
        choices = event.get("choices", [])
        node_keys = {(item["kind"], item["ident"]) for item in nodes}
        profile_keys = {tuple(item["key"]) for item in profiles}
        choice_keys = {tuple(item["first"]) for item in choices}
        if not nodes or profile_keys != node_keys or choice_keys != node_keys:
            errors.append(f"decision {index}: nodes/profiles/choices do not reconcile")
            continue
        if not all(math.isfinite(float(item[field])) for item in choices for field in
                   ("total_s", "movement_s", "scan_action_s", "source_action_s")):
            errors.append(f"decision {index}: non-finite proxy value")
            continue

        minimum = min(choices, key=_choice_order)
        best_first = tuple(event["best_first"])
        anchor_first = tuple(event["anchor_first"])
        selected_first = tuple(event["selected_first"])
        if tuple(minimum["first"]) != best_first:
            errors.append(f"decision {index}: best first node is not the minimum proxy")
        anchor_total = float(event["anchor_proxy"]["total_s"])
        best_total = float(event["best_proxy"]["total_s"])
        saving = anchor_total - best_total
        if abs(saving - float(event["proxy_saving_s"])) > FLOAT_TOL:
            errors.append(f"decision {index}: proxy saving mismatch")
        expected_change = best_first != anchor_first and saving + FLOAT_TOL >= float(event["minimum_proxy_saving_s"])
        if bool(event["changed_first"]) != expected_change:
            errors.append(f"decision {index}: threshold decision mismatch")
        expected_selected = best_first if expected_change else anchor_first
        if selected_first != expected_selected:
            errors.append(f"decision {index}: selected first node mismatch")
        changed += expected_change

        action_count = int(event["measure_count_before"]) + int(event["clear_count_before"])
        expected_position = (0.0, 0.0) if action_count == 0 else tuple(actions[action_count - 1]["point"])
        if action_count > len(actions) or not _same_point(event["position"], expected_position):
            errors.append(f"decision {index}: public position/action index mismatch")

    if changed != int(state.get("service_route_changed_first_count", -1)):
        errors.append("changed-first counter mismatch")

    skips = state.get("duplicate_supplement_skips", [])
    for index, event in enumerate(skips):
        action_count = int(event["measure_count_before"]) + int(event["clear_count_before"])
        if action_count > len(actions):
            errors.append(f"duplicate skip {index}: action index beyond history")
            continue
        prior = actions[:action_count]
        previous_measure = next((action for action in reversed(prior)
                                 if action["path"] == "/measure" and action["channel"] == event["channel"]), None)
        if previous_measure is None or not _same_point(previous_measure["point"], event["position"]):
            errors.append(f"duplicate skip {index}: no same-channel same-point public measurement")
        expected_position = (0.0, 0.0) if not prior else tuple(prior[-1]["point"])
        if not _same_point(event["position"], expected_position):
            errors.append(f"duplicate skip {index}: not at current public position")
        if math.dist(tuple(event["position"]), tuple(event["initial_point"])) < 50.0 - FLOAT_TOL:
            errors.append(f"duplicate skip {index}: original supplement would already be suppressed")

    if len(skips) != int(state.get("duplicate_supplement_skip_count", -1)):
        errors.append("duplicate-supplement counter mismatch")
    if state.get("compatibility_used_for_decision") is not False:
        errors.append("compatibility helper must not certify or remove candidates in v1")
    return {
        "all_checks_pass": not errors,
        "errors": errors,
        "decision_count": len(decisions),
        "changed_first_count": changed,
        "duplicate_supplement_skip_count": len(skips),
    }


def static_scope_checks():
    text = Path(candidate.__file__).read_text(encoding="utf-8")
    checks = [
        {
            "name": "uses_frozen_localizer",
            "pass": "from search_policy_q4_batch import _TrackedClient, _localize, _state" in text,
        },
        {
            "name": "uses_ring22_certificate",
            "pass": "from search_policy_q4_ring22 import coverage_certificate, ring22_points" in text,
        },
        {
            "name": "does_not_import_offline_truth",
            "pass": "offline_arena" not in text and "from offline_arena import Source" not in text,
        },
        {
            "name": "does_not_use_compatibility_as_certificate",
            "pass": "q4_joint_compatibility" not in text,
        },
        {
            "name": "version_is_independent",
            "pass": candidate.VERSION == "q4_service_aware_route_v1",
        },
    ]
    return {"all_checks_pass": all(item["pass"] for item in checks), "checks": checks}


def main():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if len(cases) != 36:
        raise ValueError("the predeclared paired development population must contain 36 cases")

    relative_inputs = [
        "planning/q4_service_aware_route_experiment.md",
        "results/q4_reuse/development/cases.json",
        "results/optimization_200_400/ring22_continuous_certificate.json",
        "code/shared/search_policy.py",
        "code/shared/search_policy_polygon.py",
        "code/shared/search_policy_joint_route_q3.py",
        "code/shared/search_policy_q4_batch.py",
        "code/shared/search_policy_q4_reuse.py",
        "code/shared/search_policy_q4_ring22.py",
        "code/shared/search_policy_q4_ring22_joint_route.py",
        "code/shared/search_policy_q4_service_aware_route.py",
        "code/shared/evaluate_q4_shadow_recovery_development.py",
        "code/shared/evaluate_q4_domain_prior_development.py",
        "code/shared/evaluate_q4_service_aware_route_development.py",
        "code/shared/offline_arena.py",
        "code/shared/simulator_client.py",
        "code/shared/shadow_cone_geometry.py",
    ]
    inputs = [{"path": item, "sha256": sha(ROOT / item)} for item in relative_inputs]
    manifest = OUT / "inputs_before_run.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != inputs:
        raise ValueError("frozen inputs differ; create a new experiment identity instead of overwriting")
    save(manifest, inputs)
    save(OUT / "cases.json", cases)
    scope_checks = static_scope_checks()
    save(OUT / "static_scope_checks.json", scope_checks)

    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "seed": case["seed"], "error_mode": case["error_mode"]}
        for name, policy in (("baseline", baseline.run_search), ("candidate", candidate.run_search)):
            metrics, state, actions = run(policy, case, domain=False)
            if name == "candidate":
                audit = verify_service_audit(state, actions)
                metrics["service_audit"] = audit
                metrics["all_checks_pass"] = metrics["all_checks_pass"] and audit["all_checks_pass"]
                metrics["service_route_changed_first_count"] = state["service_route_changed_first_count"]
                metrics["duplicate_supplement_skip_count"] = state["duplicate_supplement_skip_count"]
            row[name] = metrics
            save(OUT / "states" / f"{case['case_id']}__{name}.json", {"state": state, "actions": actions})
        row["candidate_to_baseline_ratio"] = row["candidate"]["time_s"] / row["baseline"]["time_s"]
        row["paired_delta_s_per_source"] = row["candidate"]["mean_s"] - row["baseline"]["mean_s"]
        rows.append(row)
        print(json.dumps({
            "case_id": row["case_id"],
            "baseline": row["baseline"]["mean_s"],
            "candidate": row["candidate"]["mean_s"],
            "checks": row["baseline"]["all_checks_pass"] and row["candidate"]["all_checks_pass"],
        }, ensure_ascii=False), flush=True)

    summaries = {name: summarize(rows, name) for name in ("baseline", "candidate")}
    saving = summaries["baseline"]["mean_s_per_source"] - summaries["candidate"]["mean_s_per_source"]
    relative = saving / summaries["baseline"]["mean_s_per_source"]
    worst = max(rows, key=lambda item: item["candidate_to_baseline_ratio"])
    best = min(rows, key=lambda item: item["candidate_to_baseline_ratio"])
    hard_checks = scope_checks["all_checks_pass"] and all(summary["all_checks_pass"] for summary in summaries.values())
    accepted = (
        hard_checks
        and saving >= ABSOLUTE_MIN_S_PER_SOURCE
        and relative >= RELATIVE_MIN
        and worst["candidate_to_baseline_ratio"] <= WORST_RATIO_MAX
    )
    result = {
        "scope": "PAIRED_OFFLINE_DEVELOPMENT_ONLY_NOT_PRACTICE_OR_FORMAL_TEST",
        "artifact_type": "q4_service_aware_route_development_comparison",
        "case_count": len(cases),
        "all_hard_checks_pass": hard_checks,
        "accepted": bool(accepted),
        "predeclared_gate": {
            "relative_min": RELATIVE_MIN,
            "absolute_min_s_per_source": ABSOLUTE_MIN_S_PER_SOURCE,
            "worst_ratio_max": WORST_RATIO_MAX,
        },
        "absolute_saving_s_per_source": saving,
        "relative_improvement": relative,
        "worst_case": {"case_id": worst["case_id"], "ratio": worst["candidate_to_baseline_ratio"]},
        "best_case": {"case_id": best["case_id"], "ratio": best["candidate_to_baseline_ratio"]},
        "candidate_route_decision_count": sum(row["candidate"]["service_audit"]["decision_count"] for row in rows),
        "candidate_changed_first_count": sum(row["candidate"]["service_route_changed_first_count"] for row in rows),
        "candidate_duplicate_supplement_skip_count": sum(row["candidate"]["duplicate_supplement_skip_count"] for row in rows),
        "summaries": summaries,
        "rows": rows,
        "static_scope_checks": scope_checks,
        "inputs": inputs,
    }
    save(OUT / "comparison.json", result)
    save(OUT / "decision.json", {
        "accepted": bool(accepted),
        "human_verdict_recorded": False,
        "next_step": "independent_review_before_any_fresh_validation" if accepted
                     else "retain_negative_result_and_stop_without_tuning_holdout_or_practice",
    })
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("rows", "inputs", "static_scope_checks")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

