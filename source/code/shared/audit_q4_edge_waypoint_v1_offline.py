"""Saved-evidence counterfactual audit for Q4 edge-waypoint v1; no reruns."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import search_policy_polygon as polygon_policy
from evaluate_q4_edge_waypoint_v1_offline import decompose
from q4_edge_waypoint_geometry import (
    ROUTE_BOUND_MARGIN_M,
    node_key,
    route_connector_upper_bound,
    route_keys,
    split_edge_certificate,
    visibility_certificate,
)
from q4_joint_compatibility import check_fixed_position
from search_policy import _add_bearing, _bearing_triangle, _clear_points
from search_policy_joint_route_q3 import _route


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/Q4/experiments/edge_waypoint_v1"
CASES = OUT / "cases.json"
GENERATION = OUT / "case_generation.json"
MANIFEST = OUT / "inputs_before_run.json"
COMPARISON = OUT / "metrics/comparison.json"
DECISION = OUT / "decision.json"
AUDIT = OUT / "audit/counterfactual_audit.json"
ROBUSTNESS = OUT / "metrics/robustness_summary.json"
RETROSPECTIVE = OUT / "retrospective.json"
REVIEW = ROOT / "code/Q4/reviews/q4_edge_waypoint_v1_python_review.json"
FREEZE = ROOT / "planning/q4_edge_waypoint_v1_candidate_freeze.json"
CANDIDATE = ROOT / "code/shared/search_policy_q4_edge_waypoint_v1.py"
GEOMETRY = ROOT / "code/shared/q4_edge_waypoint_geometry.py"
FLOAT_TOLERANCE = 1e-6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def close(first, second, tolerance=FLOAT_TOLERANCE):
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=tolerance)


def same_point(first, second, tolerance=FLOAT_TOLERANCE):
    return math.dist(tuple(first), tuple(second)) <= tolerance


def normalize_history(actions, channel):
    history, measured, positives = [], [], []
    for action in actions:
        if action["path"] != "/measure" or int(action["channel"]) != int(channel):
            continue
        point = [float(value) for value in action["point"]]
        measured.append(point)
        response = action["response"]
        kind = response.get("measure_result")
        if kind == "direction" and polygon_policy._valid_svd(response.get("svd_deg")):
            history.append({
                "point": point, "result": "direction",
                "bearing_deg": float(response["svd_deg"]),
            })
            positives.append(point)
        elif kind in ("near", "no_signal"):
            history.append({"point": point, "result": kind})
            if kind == "near":
                positives.append(point)
    return history, measured, positives


def reconstruct_polygon(actions, channel):
    track = None
    for action in actions:
        if action["path"] != "/measure" or int(action["channel"]) != int(channel):
            continue
        response = action["response"]
        if response.get("measure_result") != "direction" or not polygon_policy._valid_svd(
            response.get("svd_deg")
        ):
            continue
        point = tuple(action["point"])
        bearing = response["svd_deg"]
        if track is None:
            track = (point, bearing, _bearing_triangle(point, bearing))
        else:
            clipped = _add_bearing(track[2], point, bearing)
            if clipped:
                track = (track[0], track[1], clipped)
    return None if track is None else track[2]


def polygons_close(first, second, tolerance=FLOAT_TOLERANCE):
    if first is None or second is None or len(first) != len(second):
        return first is None and second is None
    return all(same_point(a, b, tolerance) for a, b in zip(first, second))


def static_policy_audit():
    source = CANDIDATE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    names = set()
    attributes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)
    banned_imports = sorted(set(imports) & {
        "offline_arena", "evaluate_q4_shadow_recovery_development",
        "evaluate_q4_edge_waypoint_v1_offline",
    })
    banned_names = sorted(names & {"sources", "source_truth", "arena", "error_mode", "seed"})
    banned_attributes = sorted(attributes & {
        "sources", "heading", "error_mode", "seed", "cleared"
    })
    source_checks = {
        "no_truth_or_evaluator_import": not banned_imports,
        "no_truth_names": not banned_names,
        "no_truth_attributes": not banned_attributes,
        "frozen_localizer_imported_not_reimplemented":
            "from search_policy_q4_batch import _TrackedClient, _localize, _state" in source,
        "ring22_imported":
            "from search_policy_q4_ring22 import coverage_certificate, ring22_points" in source,
        "legacy_304_points_present": len(_clear_points((0.0, 0.0), 0.0)) == 304,
        "no_local_http_transport_reference": "LocalHttpTransport" not in source,
    }
    return {
        "all_checks_pass": all(source_checks.values()), "checks": source_checks,
        "banned_imports": banned_imports, "banned_names": banned_names,
        "banned_attributes": banned_attributes,
    }


def route_replay(event):
    continuation = event["continuation"]
    mode = continuation.get("continuation_mode")
    if mode == "maximum_count_stop":
        return {"all_checks_pass": continuation.get("reason") == "maximum_count_stop_after_waypoint",
                "checks": {"terminal_stop_recorded": True}}
    nodes = [
        (item["kind"], int(item["ident"]), tuple(item["destination"]))
        for item in continuation["nodes"]
    ]
    position = tuple(event["selected"]["waypoint"])
    original = event["original_scan_node"]
    original_key = ("scan", int(original["ident"]))
    target = next((node for node in nodes if node_key(node) == original_key), None)
    errors = []
    if target is None:
        errors.append("original scan missing")
        return {"all_checks_pass": False, "errors": errors}
    remaining = [node for node in nodes if node_key(node) != original_key]
    continue_route = [target] + _route(target[2], remaining)
    replan_route = _route(position, nodes)
    radii = {}
    for key, value in continuation["radius_by_node"].items():
        kind, ident = key.split(":")
        radii[(kind, int(ident))] = float(value)
    continue_bound = route_connector_upper_bound(position, continue_route, radii)
    replan_bound = route_connector_upper_bound(position, replan_route, radii)
    first_differs = node_key(replan_route[0]) != original_key
    strictly_lower = replan_bound < continue_bound - ROUTE_BOUND_MARGIN_M
    checks = {
        "continue_route_replays": route_keys(continue_route) == continuation["continue_route"],
        "replan_route_replays": route_keys(replan_route) == continuation["replan_route"],
        "continue_bound_replays": close(continue_bound, continuation["continue_bound_m"]),
        "replan_bound_replays": close(replan_bound, continuation["replan_bound_m"]),
        "first_differs_replays": first_differs is continuation["first_node_differs"],
        "strictly_lower_replays": strictly_lower is continuation["strictly_lower"],
        "allow_reorder_replays": (first_differs and strictly_lower)
        is continuation["allow_reorder"],
        "mode_matches_guard": (
            mode == "guarded_replan" if first_differs and strictly_lower
            else mode == "immediate_original_scan"
        ),
    }
    return {"all_checks_pass": all(checks.values()), "checks": checks,
            "continue_bound_m": continue_bound, "replan_bound_m": replan_bound}


def event_audit(event, actions):
    errors = []
    before_count = int(event["public_action_count_before"])
    if not 0 <= before_count < len(actions):
        return {"all_checks_pass": False, "errors": ["bad public prefix index"]}
    prefix = actions[:before_count]
    action = actions[before_count]
    selected = event["selected"]
    channel = int(selected["channel"])
    waypoint = selected["waypoint"]
    start = (0.0, 0.0) if not prefix else tuple(prefix[-1]["point"])
    history, measured, positives = normalize_history(prefix, channel)
    reconstructed_polygon = reconstruct_polygon(prefix, channel)
    certificate = visibility_certificate(
        waypoint, selected["polygon"], positives, measured
    )
    center_result = check_fixed_position(
        selected["track_center"], history, source_type="either"
    ).to_dict()
    split = split_edge_certificate(
        start, waypoint, event["original_scan_node"]["destination"]
    )
    snapshot = event["selection_snapshot"]
    eligible = snapshot["eligible_candidates"]
    checks = {
        "waypoint_action_is_next_public_action": action["path"] == "/measure"
        and int(action["channel"]) == channel and same_point(action["point"], waypoint),
        "response_matches_public_action": action["response"] == event["response"],
        "edge_start_matches_public_prefix": same_point(snapshot["edge_start"], start),
        "edge_end_matches_original_scan": same_point(
            snapshot["edge_end"], event["original_scan_node"]["destination"]
        ),
        "only_fixed_fraction": selected["fraction"] in (0.5, 0.25, 0.75),
        "history_replays_from_public_prefix": history == selected["public_history"],
        "measured_points_replay": measured == selected["measured_points"],
        "positive_points_replay_including_near": positives == selected["positive_points"],
        "track_polygon_replays_from_public_bearings": polygons_close(
            reconstructed_polygon, selected["polygon"]
        ),
        "visibility_certificate_replays": certificate["certified"]
        and certificate["checks"] == selected["certificate"]["checks"]
        and close(certificate["maximum_polygon_vertex_distance_m"],
                  selected["certificate"]["maximum_polygon_vertex_distance_m"]),
        "compatibility_center_status_replays": center_result["status"]
        == selected["compatibility_center_diagnostic"]["status"],
        "compatibility_declared_ordering_only":
            selected.get("compatibility_used_only_for_ordering") is True,
        "selected_is_first_sorted_eligible": bool(eligible) and eligible[0] == selected
        and all(tuple(eligible[index]["score"]) <= tuple(eligible[index + 1]["score"])
                for index in range(len(eligible) - 1)),
        "split_certificate_replays": split["on_segment"]
        and close(split["split_residual_m"], event["split_certificate"]["split_residual_m"]),
        "edge_is_long": snapshot["edge_length_m"] >= snapshot["minimum_edge_length_m"],
        "history_separation_strict":
            certificate["minimum_same_channel_history_distance_m"] > 1.0,
    }
    if action["response"].get("measure_result") == "no_signal":
        checks["no_signal_conservative"] = (
            event["response_disposition"]
            == "no_signal_cached_only_retain_track_and_fallback"
            and event["track_transition"]["track_equal_before_after"] is True
            and event["track_retained_on_no_signal"] is True
        )
    route = route_replay(event)
    checks["route_guard_replays"] = route["all_checks_pass"]
    continuation_mode = event["continuation"].get("continuation_mode")
    if continuation_mode == "immediate_original_scan":
        destination = event["original_scan_node"]["destination"]
        next_different = next(
            (candidate_action for candidate_action in actions[before_count + 1:]
             if not same_point(candidate_action["point"], waypoint)), None
        )
        checks["continued_endpoint_is_first_nonzero_move"] = (
            next_different is not None and same_point(next_different["point"], destination)
        )
        checks["continued_split_equals_original_edge"] = split["on_segment"]
    return {"all_checks_pass": all(checks.values()) and not errors,
            "checks": checks, "errors": errors, "route_replay": route,
            "counterfactual": {
                "direct_edge_m": split["original_edge_m"],
                "split_edge_m": split["split_sum_m"],
                "movement_delta_before_any_reorder_m": split["split_residual_m"],
                "extra_stopped_measure_time_s": 5.0,
            }}


def replay_metrics(saved, state, actions, case):
    decomposition = decompose(actions)
    source_count = len(case["sources"])
    cleared = set(state["cleared_channels"])
    truth = {int(source["channel"]) for source in case["sources"]}
    groups = [set(state[key]) for key in (
        "cleared_channels", "excluded_channels", "unresolved_channels"
    )]
    checks = {
        "action_fingerprint": hashlib.sha256(
            json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest() == saved["public_action_fingerprint"],
        "movement_m": close(decomposition["movement_m"], saved["movement_m"]),
        "time_s": close(decomposition["total_s"], saved["time_s"]),
        "movement_per_source": close(
            decomposition["movement_m"] / source_count,
            saved["mean_movement_m_per_source"]
        ),
        "time_per_source": close(
            decomposition["total_s"] / source_count,
            saved["mean_total_s_per_source"]
        ),
        "all_components": all(close(decomposition[key], saved[key]) for key in (
            "movement_s", "measure_s", "switch_s", "successful_clear_s", "failed_clear_s"
        )),
        "all_truth_sources_cleared": cleared == truth,
        "channels_partition": set.union(*groups) == set(range(1, 21))
        and sum(map(len, groups)) == 20,
        "completed_and_not_stopped": state.get("completed") is True
        and state.get("stopped") is False,
        "ring22_count": state.get("coverage_points_total") == 22,
        "zero_post_clear_measurements": decomposition["post_clear_measure_count"] == 0,
    }
    return {"all_checks_pass": all(checks.values()), "checks": checks}


def review_payload(audit, accepted):
    audit_hash = sha(AUDIT)
    comparison_hash = sha(COMPARISON)
    evidence = [
        {"path": relative(AUDIT), "sha256": audit_hash},
        {"path": relative(COMPARISON), "sha256": comparison_hash},
    ]
    screening_reason = (
        "The one-shot candidate passed every predeclared performance gate. It remains "
        "an unapproved offline candidate requiring a separate human decision."
        if accepted else
        "The one-shot candidate failed at least one predeclared performance gate and is "
        "rejected without tuning, replacement seeds, added cases, practice, or formal testing."
    )
    checks = {
        "syntax": {"status": "PASS", "reason": "All frozen modules compile and all saved runs completed.",
                   "evidence": evidence},
        "input_contract": {"status": "PASS" if audit["input_integrity"]["all_checks_pass"] else "FAIL",
                           "reason": "Pre-run hashes and fresh-case generation evidence were replayed.",
                           "evidence": evidence},
        "method_alignment": {"status": "PASS" if audit["structural_audit"]["all_checks_pass"] else "FAIL",
                             "reason": "Visibility, route, budget, ring22, and fallback invariants were replayed.",
                             "evidence": evidence},
        "candidate_screening": {"status": "PASS" if accepted else "FAIL",
                                "reason": screening_reason, "evidence": evidence},
        "reproducibility": {"status": "PASS" if audit["saved_evidence_replay"]["all_checks_pass"] else "FAIL",
                            "reason": "All 84 saved ledgers were replayed without a policy or arena rerun.",
                            "evidence": evidence},
        "output_contract": {"status": "PASS" if audit["all_hard_checks_pass"] else "FAIL",
                            "reason": "Paired metrics, truth-clear checks, and action-cost accounting reconcile.",
                            "evidence": evidence},
    }
    total_events = audit["event_summary"]["total_events"]
    findings = []
    if total_events == 0:
        findings.append({
            "priority": 1, "code": "NO_NONTRIVIAL_CERTIFIED_EDGE_OPPORTUNITIES",
            "reason": "The conservative continuous-state certificate produced zero events; do not relax it post hoc."
        })
    if not accepted:
        findings.append({
            "priority": 1, "code": "PREDECLARED_PERFORMANCE_GATE_FAILED",
            "reason": "Preserve the negative result and retain the frozen comparator."
        })
    return {
        "schema_version": 2, "artifact_type": "review",
        "question_id": "Q4_edge_waypoint_v1", "language": "python",
        "inputs": [
            {"path": relative(CANDIDATE), "sha256": sha(CANDIDATE)},
            {"path": relative(GEOMETRY), "sha256": sha(GEOMETRY)},
            {"path": relative(COMPARISON), "sha256": comparison_hash},
            {"path": relative(AUDIT), "sha256": audit_hash},
        ],
        "checks": checks, "findings": findings,
        "verdict": "PASSED" if all(item["status"] == "PASS" for item in checks.values()) else "FAILED",
        "extensions": {
            "scope": "ONE_SHOT_FRESH_STRUCTURAL_OFFLINE_CANDIDATE_REVIEW",
            "accepted": accepted, "simulations_rerun_for_audit": 0,
            "formal_tests_used": 0, "live_api_calls": 0,
            "frozen_baseline_modified": False,
        },
    }


def main():
    required = [CASES, GENERATION, MANIFEST, COMPARISON, DECISION, FREEZE]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("paired saved evidence is incomplete")
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    case_by_id = {case["case_id"]: case for case in cases}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    input_checks = []
    for item in manifest:
        path = ROOT / item["path"]
        input_checks.append({
            "path": item["path"], "exists": path.is_file(),
            "expected_sha256": item["sha256"],
            "actual_sha256": sha(path) if path.is_file() else None,
            "matches": path.is_file() and sha(path) == item["sha256"],
        })
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    freeze_map = {item["path"]: item["sha256"] for item in freeze["frozen_candidate_files"]}
    freeze_checks = {
        "candidate_hash": sha(CANDIDATE) == freeze_map[relative(CANDIDATE)],
        "geometry_hash": sha(GEOMETRY) == freeze_map[relative(GEOMETRY)],
        "baseline_hash": next(
            item["sha256"] for item in manifest
            if item["path"] == "code/shared/search_policy_q4_ring22_joint_route.py"
        ) == "001f740403631fd3aa21a53cd9b61af34558b656bb1fe492e270d680f063cbf3",
    }
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    novelty_checks = {
        "case_count_42": len(cases) == 42,
        "seeds_exact": [case["seed"] for case in cases] == list(range(65000, 65042)),
        "case_hash": sha(CASES) == generation["case_file_sha256"],
        "novelty": generation["novelty_audit"]["all_checks_pass"],
        "distribution": generation["distribution_audit"]["all_checks_pass"],
    }
    input_integrity = {
        "all_checks_pass": all(item["matches"] for item in input_checks)
        and all(freeze_checks.values()) and all(novelty_checks.values()),
        "manifest_checks": input_checks, "candidate_freeze_checks": freeze_checks,
        "fresh_case_checks": novelty_checks,
    }
    static = static_policy_audit()
    row_audits, event_audits = [], []
    for row in comparison["rows"]:
        case = case_by_id[row["case_id"]]
        pair = {"case_id": row["case_id"], "methods": {}}
        for name in ("baseline", "candidate"):
            state_ref = row[f"{name}_state"]
            state_path = ROOT / state_ref["path"]
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            replay = replay_metrics(row[name], payload["state"], payload["actions"], case)
            pair["methods"][name] = {
                "state_hash_matches": sha(state_path) == state_ref["sha256"],
                "metric_replay": replay,
            }
            if name == "candidate":
                for event in payload["state"].get("edge_waypoint_events", []):
                    audited = event_audit(event, payload["actions"])
                    event_audits.append({"case_id": row["case_id"],
                                         "event_index": event["event_index"], **audited})
                budgets = payload["state"].get("localization_budget_events", [])
                pair["candidate_budget_checks"] = {
                    "all_at_most_8": all(
                        0 <= item["local_extra_bearing_delta"] <= 8 for item in budgets
                    ),
                    "event_count": len(budgets),
                }
        pair["all_checks_pass"] = all(
            item["state_hash_matches"] and item["metric_replay"]["all_checks_pass"]
            for item in pair["methods"].values()
        ) and pair["candidate_budget_checks"]["all_at_most_8"]
        row_audits.append(pair)
    event_all = all(item["all_checks_pass"] for item in event_audits)
    structural = {
        "all_checks_pass": static["all_checks_pass"] and event_all
        and all(item["candidate_budget_checks"]["all_at_most_8"] for item in row_audits),
        "static_policy_audit": static,
        "all_event_replays_pass": event_all,
        "event_audits": event_audits,
        "localizer_budget_all_pass": all(
            item["candidate_budget_checks"]["all_at_most_8"] for item in row_audits
        ),
    }
    saved = {
        "all_checks_pass": all(item["all_checks_pass"] for item in row_audits),
        "paired_case_count": len(row_audits), "row_audits": row_audits,
    }
    hard = input_integrity["all_checks_pass"] and structural["all_checks_pass"] \
        and saved["all_checks_pass"] and comparison["all_hard_checks_pass"]
    audit = {
        "schema_version": 1,
        "artifact_type": "q4_edge_waypoint_v1_counterfactual_audit",
        "scope": "POST_HOC_SAVED_EVIDENCE_REPLAY_NO_POLICY_OR_ARENA_RERUN",
        "audit_identity": {
            "script_path": relative(Path(__file__)), "script_sha256": sha(Path(__file__)),
            "comparison_sha256": sha(COMPARISON), "manifest_sha256": sha(MANIFEST),
            "simulations_rerun": 0, "live_api_calls": 0, "formal_tests": 0,
        },
        "input_integrity": input_integrity,
        "saved_evidence_replay": saved,
        "structural_audit": structural,
        "event_summary": {
            "total_events": len(event_audits),
            "all_event_replays_pass": event_all,
            "zero_event_result_allowed_by_predeclaration": len(event_audits) == 0,
            "no_certificate_relaxation": True,
        },
        "all_hard_checks_pass": hard,
        "gate_reconciliation": {
            "accepted": comparison["accepted"], "checks": comparison["gate_checks"],
            "decision": comparison["decision"],
        },
    }
    save(AUDIT, audit)
    robustness = {
        "schema_version": 1, "artifact_type": "candidate-robustness-summary",
        "question_id": "Q4", "experiment_id": "q4_edge_waypoint_v1",
        "scope": "fresh synthetic paired offline only",
        "risk_checks": {
            "three_error_modes_balanced": generation["distribution_audit"]["global_checks"][
                "each_error_mode_has_fourteen_cases"
            ],
            "source_counts_10_through_16_balanced": generation["distribution_audit"][
                "global_checks"
            ]["each_source_count_has_six_cases"],
            "boundary_radius_cases_included": generation["distribution_audit"][
                "global_checks"
            ]["all_case_level_checks_pass"],
            "continuous_visibility_certificate_replayed": event_all,
            "zero_event_feasibility_is_valid_negative": len(event_audits) == 0,
            "worst_time_ratio_within_1_10": comparison["gate_checks"][
                "worst_single_case_total_time_ratio_at_most_1_10"
            ],
        },
        "all_hard_checks_pass": hard,
        "performance_gate_accepted": comparison["accepted"],
        "limitations": [
            "Synthetic OfflineArena is not the official simulator.",
            "A zero-event result tests feasibility of this sufficient certificate, not every possible safe certificate.",
        ],
    }
    save(ROBUSTNESS, robustness)
    retrospective = {
        "schema_version": 1, "experiment_id": "q4_edge_waypoint_v1",
        "decision": comparison["decision"], "accepted": comparison["accepted"],
        "all_hard_checks_pass": hard,
        "total_certified_waypoint_events": len(event_audits),
        "mean_movement_relative_reduction": comparison["mean_movement_relative_reduction"],
        "mean_total_time_relative_reduction": comparison["mean_total_time_relative_reduction"],
        "worst_time_ratio": comparison["worst_time_case"]["ratio"],
        "interpretation": (
            "The sufficient continuous-state certificate found no nontrivial opportunity in this population; "
            "retain this as a feasibility-screen negative and do not relax post hoc."
            if not event_audits else
            "Certified opportunities occurred; the predeclared conjunction alone determines rejection or screen eligibility."
        ),
        "stop_rule_applied": True,
        "next_step": "retain frozen comparator; no tuning, new seeds, GUI/live API, practice, or formal test",
    }
    save(RETROSPECTIVE, retrospective)
    review = review_payload(audit, comparison["accepted"] and hard)
    save(REVIEW, review)
    print(json.dumps({
        "all_hard_checks_pass": hard, "accepted": comparison["accepted"] and hard,
        "events": len(event_audits),
        "movement_reduction": comparison["mean_movement_relative_reduction"],
        "time_reduction": comparison["mean_total_time_relative_reduction"],
        "worst_time_ratio": comparison["worst_time_case"]["ratio"],
        "audit": relative(AUDIT), "review": relative(REVIEW),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
