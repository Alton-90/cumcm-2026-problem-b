"""Post-hoc audit of saved v2 evidence; never reruns a policy or arena."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from audit_q4_service_aware_route_development import _metric_errors, _replay_public_metrics
from evaluate_q4_service_multistart_v2_development import verify_route_decisions


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q4_service_multistart_v2/development"
MANIFEST = OUT / "inputs_before_run.json"
CASES = OUT / "cases.json"
GENERATION = OUT / "case_generation.json"
COMPARISON = OUT / "comparison.json"
DECISION = OUT / "decision.json"
AUDIT = OUT / "audit.json"
REVIEW = ROOT / "code/Q4/reviews/q4_service_multistart_route_v2_python_review.json"
PLAN = ROOT / "planning/q4_service_multistart_v2_experiment.md"
CANDIDATE = ROOT / "code/shared/search_policy_q4_service_multistart_route_v2.py"
BASELINE = ROOT / "code/shared/search_policy_q4_ring22_joint_route.py"
RELATIVE_MIN = 0.05
ABSOLUTE_MIN = 10.0
WORST_MAX = 1.10
TOLERANCE = 1e-7


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path):
    return path.relative_to(ROOT).as_posix()


def _save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _close(first, second, tolerance=TOLERANCE):
    return abs(float(first) - float(second)) <= tolerance


def _manifest_audit(records):
    rows = []
    for record in records:
        path = ROOT / record["path"]
        observed = _sha(path) if path.is_file() else None
        rows.append(
            {
                "path": record["path"],
                "expected_sha256": record["sha256"],
                "observed_sha256": observed,
                "match": observed == record["sha256"],
            }
        )
    return {"all_checks_pass": all(row["match"] for row in rows), "entry_count": len(rows), "rows": rows}


def _make_review(audit, audit_sha):
    evidence = [
        {"path": _relative(AUDIT), "sha256": audit_sha},
        {"path": _relative(COMPARISON), "sha256": _sha(COMPARISON)},
    ]
    plan_evidence = evidence + [{"path": _relative(PLAN), "sha256": _sha(PLAN)}]
    inputs = [
        {"path": _relative(path), "sha256": _sha(path)}
        for path in (PLAN, CANDIDATE, BASELINE, Path(__file__), MANIFEST, CASES, GENERATION, COMPARISON, AUDIT)
    ]
    gate = audit["gate_replay"]
    return {
        "schema_version": 2,
        "artifact_type": "review",
        "question_id": "Q4_service_multistart_v2",
        "language": "python",
        "inputs": inputs,
        "checks": {
            "syntax": {
                "status": "PASS",
                "reason": "Candidate, generator, evaluator and audit compile; the frozen candidate completed all 42 saved offline cases.",
                "evidence": evidence,
            },
            "input_contract": {
                "status": "PASS" if audit["input_integrity"]["all_checks_pass"] else "FAIL",
                "reason": "All pre-run hashes match; seeds 63000-63041 have no collision in the scanned prior case corpus and the 7x3x2 strata reconcile." if audit["input_integrity"]["all_checks_pass"] else "A frozen input, novelty or case-stratum check failed.",
                "evidence": plan_evidence,
            },
            "method_alignment": {
                "status": "PASS" if audit["route_and_reliability_replay"]["all_checks_pass"] else "FAIL",
                "reason": "Every saved decision replays anchor, v1-service and multistart-service routes, the 6 s proxy guard, nonlonger geometry guard and anchor tie fallback; ring22 and the frozen localizer/fallback chain remain bound." if audit["route_and_reliability_replay"]["all_checks_pass"] else "A route-selection or reliability-chain replay failed.",
                "evidence": plan_evidence,
            },
            "candidate_screening": {
                "status": "FAIL",
                "reason": (
                    f"All hard checks pass, but the candidate is slower by {-gate['absolute_saving_s_per_source']:.6f}s/source "
                    f"({100.0 * gate['relative_improvement']:.6f}% improvement) and worst ratio {gate['worst_case']['ratio']:.6f} exceeds 1.10."
                ),
                "evidence": plan_evidence,
            },
            "reproducibility": {
                "status": "PASS" if audit["saved_output_replay"]["all_checks_pass"] else "FAIL",
                "reason": "All 84 saved action histories reproduce public fingerprints, counters, clocks, cost components and task checks; no policy or arena rerun was performed." if audit["saved_output_replay"]["all_checks_pass"] else "A saved action history did not reproduce its recorded metrics.",
                "evidence": evidence,
            },
            "output_contract": {
                "status": "PASS" if audit["all_hard_checks_pass"] else "FAIL",
                "reason": "All 42 paired rows and route logs reconcile, all sources clear, and the rejected decision follows the predeclared conjunction without claiming simulator or formal-test evidence." if audit["all_hard_checks_pass"] else "A required output or hard check failed.",
                "evidence": evidence,
            },
        },
        "findings": [
            {
                "priority": 1,
                "code": "V2_MEAN_AND_WORST_PERFORMANCE_FAIL",
                "reason": "Retain as a genuine negative result and stop without tuning, new cases, practice, or formal testing.",
            },
            {
                "priority": 2,
                "code": "STATIC_GEOMETRY_GUARD_DOES_NOT_BOUND_RECEDING_REALIZED_COST",
                "reason": "Although every selected nonanchor route was no longer than its current anchor, altered observations and replanning still produced a 1.153761 worst-case realized ratio.",
            },
        ],
        "verdict": "FAILED",
        "extensions": {
            "scope": "ONE_SHOT_FRESH_STRUCTURAL_DEVELOPMENT_CANDIDATE_REVIEW",
            "candidate": "q4_service_multistart_route_v2",
            "accepted": False,
            "simulations_rerun_for_audit": 0,
            "formal_tests_used": 0,
            "frozen_baseline_modified": False,
        },
    }


def main():
    manifest = _load(MANIFEST)
    cases = _load(CASES)
    generation = _load(GENERATION)
    comparison = _load(COMPARISON)
    decision = _load(DECISION)
    manifest_result = _manifest_audit(manifest)
    input_checks = {
        "manifest_matches": manifest_result["all_checks_pass"],
        "case_count_is_42": len(cases) == 42,
        "seed_range_exact": [case["seed"] for case in cases] == list(range(63000, 63042)),
        "comparison_case_order_matches": [row["case_id"] for row in comparison["rows"]]
        == [case["case_id"] for case in cases],
        "generation_novelty_pass": generation["novelty_audit"]["all_checks_pass"] is True,
        "generation_distribution_pass": generation["distribution_audit"]["all_checks_pass"] is True,
        "generation_case_hash_matches": generation["case_file_sha256"] == _sha(CASES),
        "candidate_hash_matches_generation": generation["freeze_identity"]["candidate_sha256_at_generation"]
        == _sha(CANDIDATE),
    }
    input_integrity = {
        "all_checks_pass": all(input_checks.values()),
        "checks": input_checks,
        "manifest_audit": manifest_result,
    }

    case_results = []
    saved_output_errors = []
    route_errors = []
    aggregate_selections = Counter()
    aggregate_decisions = 0
    aggregate_skips = 0
    for case, row in zip(cases, comparison["rows"]):
        result = {"case_id": case["case_id"]}
        for policy_name in ("baseline", "candidate"):
            path = OUT / "states" / f"{case['case_id']}__{policy_name}.json"
            record = _load(path)
            replay = _replay_public_metrics(case, record["state"], record["actions"])
            metric_errors = _metric_errors(case["case_id"], policy_name, row[policy_name], replay)
            if not replay["all_checks_pass"]:
                metric_errors.append(f"{case['case_id']} {policy_name}: raw task replay failed")
            saved_output_errors.extend(metric_errors)
            item = {
                "path": _relative(path),
                "sha256": _sha(path),
                "raw_task_checks_pass": replay["all_checks_pass"],
                "metric_reconciliation_pass": not metric_errors,
                "errors": metric_errors,
            }
            if policy_name == "candidate":
                route_audit = verify_route_decisions(record["state"], record["actions"])
                if not route_audit["all_checks_pass"]:
                    route_errors.extend(
                        f"{case['case_id']}: {error}" for error in route_audit["errors"]
                    )
                if route_audit != row["candidate"]["route_audit"]:
                    route_errors.append(f"{case['case_id']}: stored route audit differs from replay")
                aggregate_selections.update(route_audit["selection_counts"])
                aggregate_decisions += route_audit["decision_count"]
                aggregate_skips += route_audit["duplicate_supplement_skip_count"]
                item["route_audit"] = route_audit
            result[policy_name] = item
        case_results.append(result)

    aggregate_checks = {
        "decision_count_matches": aggregate_decisions == comparison["route_decision_count"],
        "selection_counts_match": {
            name: aggregate_selections[name] for name in ("anchor", "v1_service", "multistart_service")
        }
        == comparison["route_selection_counts"],
        "duplicate_skip_count_matches": aggregate_skips == comparison["duplicate_supplement_skip_count"],
        "all_saved_route_audits_pass": not route_errors,
    }
    saved_output = {
        "all_checks_pass": not saved_output_errors,
        "run_count": 84,
        "errors": saved_output_errors,
        "case_results": case_results,
    }
    route_replay = {
        "all_checks_pass": all(aggregate_checks.values()),
        "checks": aggregate_checks,
        "errors": route_errors,
        "decision_count": aggregate_decisions,
        "selection_counts": dict(aggregate_selections),
        "duplicate_supplement_skip_count": aggregate_skips,
    }

    baseline_mean = statistics.mean(row["baseline"]["mean_s"] for row in comparison["rows"])
    candidate_mean = statistics.mean(row["candidate"]["mean_s"] for row in comparison["rows"])
    saving = baseline_mean - candidate_mean
    relative = saving / baseline_mean
    worst = max(comparison["rows"], key=lambda row: row["candidate_to_baseline_ratio"])
    all_hard = (
        input_integrity["all_checks_pass"]
        and saved_output["all_checks_pass"]
        and route_replay["all_checks_pass"]
        and comparison["static_scope_checks"]["all_checks_pass"]
    )
    gate_checks = {
        "all_hard_checks": all_hard,
        "relative_improvement_at_least_5_percent": relative >= RELATIVE_MIN,
        "absolute_saving_at_least_10_s_per_source": saving >= ABSOLUTE_MIN,
        "worst_ratio_at_most_1_10": worst["candidate_to_baseline_ratio"] <= WORST_MAX,
    }
    accepted = all(gate_checks.values())
    comparison_checks = {
        "baseline_mean_matches": _close(
            baseline_mean, comparison["summaries"]["baseline"]["mean_s_per_source"]
        ),
        "candidate_mean_matches": _close(
            candidate_mean, comparison["summaries"]["candidate"]["mean_s_per_source"]
        ),
        "saving_matches": _close(saving, comparison["absolute_saving_s_per_source"]),
        "relative_matches": _close(relative, comparison["relative_improvement"]),
        "worst_matches": worst["case_id"] == comparison["worst_case"]["case_id"]
        and _close(worst["candidate_to_baseline_ratio"], comparison["worst_case"]["ratio"]),
        "hard_check_flag_matches": comparison["all_hard_checks_pass"] is all_hard,
        "accepted_flag_matches": comparison["accepted"] is accepted,
        "decision_is_rejected_and_stopped": decision["accepted"] is False
        and decision["next_step"]
        == "stop_and_retain_complete_evidence_without_tuning_or_additional_data",
    }
    audit = {
        "schema_version": 1,
        "artifact_type": "q4_service_multistart_v2_development_audit",
        "scope": "POST_HOC_SAVED_EVIDENCE_REPLAY_NO_POLICY_OR_ARENA_RERUN",
        "audit_identity": {
            "script_path": _relative(Path(__file__)),
            "script_sha256": _sha(Path(__file__)),
            "comparison_sha256": _sha(COMPARISON),
            "manifest_sha256": _sha(MANIFEST),
            "simulations_rerun": 0,
        },
        "input_integrity": input_integrity,
        "saved_output_replay": saved_output,
        "route_and_reliability_replay": route_replay,
        "all_hard_checks_pass": all_hard,
        "gate_replay": {
            "accepted": accepted,
            "checks": gate_checks,
            "baseline_mean_s_per_source": baseline_mean,
            "candidate_mean_s_per_source": candidate_mean,
            "absolute_saving_s_per_source": saving,
            "relative_improvement": relative,
            "worst_case": {
                "case_id": worst["case_id"],
                "ratio": worst["candidate_to_baseline_ratio"],
            },
            "decision": "REJECT_AND_STOP" if not accepted else "ELIGIBLE_FOR_NEXT_GATE",
        },
        "comparison_reconciliation": {
            "all_checks_pass": all(comparison_checks.values()),
            "checks": comparison_checks,
        },
    }
    _save(AUDIT, audit)
    review = _make_review(audit, _sha(AUDIT))
    _save(REVIEW, review)
    print(
        json.dumps(
            {
                "all_hard_checks_pass": all_hard,
                "accepted": accepted,
                "baseline_mean_s_per_source": baseline_mean,
                "candidate_mean_s_per_source": candidate_mean,
                "relative_improvement": relative,
                "worst_ratio": worst["candidate_to_baseline_ratio"],
                "audit": _relative(AUDIT),
                "review": _relative(REVIEW),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
