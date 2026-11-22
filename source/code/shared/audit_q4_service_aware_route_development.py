"""Read-only v2 audit of the frozen Q4 service-aware development outputs.

This script does not run either policy.  It independently replays the logged
proxy arithmetic and public action accounting, preserves the original v1
comparison, and writes a versioned audit plus a candidate-specific review.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from search_policy_joint_route_q3 import _route as anchor_route
from search_policy_q4_ring22 import coverage_certificate


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q4_service_aware_route/development"
MANIFEST = OUT / "inputs_before_run.json"
COMPARISON = OUT / "comparison.json"
SAVED_CASES = OUT / "cases.json"
SOURCE_CASES = ROOT / "results/q4_reuse/development/cases.json"
STATIC_SCOPE = OUT / "static_scope_checks.json"
CANDIDATE = ROOT / "code/shared/search_policy_q4_service_aware_route.py"
BASELINE = ROOT / "code/shared/search_policy_q4_ring22_joint_route.py"
PLAN = ROOT / "planning/q4_service_aware_route_experiment.md"
AUDIT = OUT / "audit_v2.json"
REVIEW = ROOT / "code/Q4/reviews/q4_service_aware_route_v1_python_review.json"

AUDIT_VERSION = "q4_service_aware_route_development_audit_v2"
FLOAT_TOL = 1e-7
DIRECT_RADIUS_M = 19.9
MAX_LOCAL_MEASUREMENTS = 8
ROBOT_SPEED_MPS = 5.0
MEASURE_WITH_SWITCH_PROXY_S = 6.0
SUCCESSFUL_CLEAR_PROXY_S = 5.0
REUSE_MIN_BASELINE_M = 50.0
MIN_PROXY_SAVING_S = 6.0
RELATIVE_MIN = 0.05
ABSOLUTE_MIN_S_PER_SOURCE = 10.0
WORST_RATIO_MAX = 1.10


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path):
    return path.relative_to(ROOT).as_posix()


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _close(first, second, tolerance=FLOAT_TOL):
    return abs(float(first) - float(second)) <= tolerance


def _same_point(first, second):
    return math.dist(tuple(map(float, first)), tuple(map(float, second))) <= FLOAT_TOL


def _node_key(node):
    return (str(node[0]), int(node[1]))


def _proxy_fields(proxy):
    return {
        "total_s": float(proxy["total_s"]),
        "movement_s": float(proxy["movement_s"]),
        "scan_action_s": float(proxy["scan_action_s"]),
        "source_action_s": float(proxy["source_action_s"]),
        "duplicate_supplement_skips": int(proxy["duplicate_supplement_skips"]),
    }


def _proxy_errors(label, proxy):
    errors = []
    try:
        values = _proxy_fields(proxy)
    except (KeyError, TypeError, ValueError) as error:
        return [f"{label}: invalid proxy record: {error}"]
    for field in ("total_s", "movement_s", "scan_action_s", "source_action_s"):
        if not math.isfinite(values[field]) or values[field] < -FLOAT_TOL:
            errors.append(f"{label}: {field} is non-finite or negative")
    if values["duplicate_supplement_skips"] < 0:
        errors.append(f"{label}: negative duplicate count")
    parts = values["movement_s"] + values["scan_action_s"] + values["source_action_s"]
    if not _close(values["total_s"], parts):
        errors.append(f"{label}: proxy total does not equal its components")
    return errors


def _proxy_matches(observed, expected):
    try:
        first = _proxy_fields(observed)
        second = _proxy_fields(expected)
    except (KeyError, TypeError, ValueError):
        return False
    numeric = ("total_s", "movement_s", "scan_action_s", "source_action_s")
    return all(_close(first[field], second[field]) for field in numeric) and (
        first["duplicate_supplement_skips"] == second["duplicate_supplement_skips"]
    )


def _independent_proxy(position, route, profiles, scannable_channels, last_measure_points):
    """Recompute the declared service proxy without importing candidate code."""
    here = tuple(map(float, position))
    active_scan_channels = {int(channel) for channel in scannable_channels}
    last_measure = {
        int(channel): tuple(map(float, point)) for channel, point in last_measure_points.items()
    }
    movement_s = 0.0
    scan_action_s = 0.0
    source_action_s = 0.0
    duplicate_supplement_skips = 0
    for node in route:
        profile = profiles[_node_key(node)]
        entry = tuple(map(float, profile["entry"]))
        movement_s += math.dist(here, entry) / ROBOT_SPEED_MPS
        if profile["kind"] == "scan":
            scan_action_s += MEASURE_WITH_SWITCH_PROXY_S * len(active_scan_channels)
            for channel in active_scan_channels:
                last_measure[channel] = entry
            here = tuple(map(float, profile["exit"]))
            continue

        channel = int(profile["channel"])
        if not bool(profile["direct"]):
            initial_point = tuple(map(float, profile["initial_point"]))
            if math.dist(here, initial_point) >= REUSE_MIN_BASELINE_M:
                if channel in last_measure and _same_point(last_measure[channel], here):
                    duplicate_supplement_skips += 1
                else:
                    source_action_s += MEASURE_WITH_SWITCH_PROXY_S
                    last_measure[channel] = here
            movement_s += float(profile["internal_distance_m"]) / ROBOT_SPEED_MPS
        source_action_s += float(profile["local_action_s"])
        active_scan_channels.discard(channel)
        here = tuple(map(float, profile["exit"]))
    return {
        "total_s": movement_s + scan_action_s + source_action_s,
        "movement_s": movement_s,
        "scan_action_s": scan_action_s,
        "source_action_s": source_action_s,
        "duplicate_supplement_skips": duplicate_supplement_skips,
    }


def _profile_errors(event_index, nodes, profiles):
    errors = []
    node_map = {_node_key(node): node for node in nodes}
    for key, profile in profiles.items():
        label = f"decision {event_index} profile {key}"
        node = node_map[key]
        if profile.get("kind") != key[0]:
            errors.append(f"{label}: kind mismatch")
            continue
        try:
            entry = tuple(map(float, profile["entry"]))
            exit_point = tuple(map(float, profile["exit"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}: invalid entry or exit")
            continue
        if not _same_point(entry, node[2]) or not _same_point(exit_point, node[2]):
            errors.append(f"{label}: entry/exit do not match the public node destination")
        if key[0] == "scan":
            continue
        try:
            radius = profile.get("radius_m")
            direct = bool(profile["direct"])
            levels = int(profile["local_measurement_levels"])
            internal = float(profile["internal_distance_m"])
            local_action = float(profile["local_action_s"])
            initial = tuple(map(float, profile["initial_point"]))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{label}: invalid source service fields: {error}")
            continue
        if int(profile.get("channel", -1)) != key[1]:
            errors.append(f"{label}: channel mismatch")
        if radius is None:
            expected_direct = False
            expected_levels = MAX_LOCAL_MEASUREMENTS
            expected_internal = 0.0
        else:
            radius = float(radius)
            expected_direct = radius <= DIRECT_RADIUS_M
            ratio = max(radius, DIRECT_RADIUS_M) / DIRECT_RADIUS_M
            expected_levels = 0 if expected_direct else min(
                MAX_LOCAL_MEASUREMENTS, max(1, math.ceil(math.log2(ratio)))
            )
            expected_internal = 0.0 if expected_direct else math.dist(initial, entry)
        expected_action = SUCCESSFUL_CLEAR_PROXY_S if expected_direct else (
            MEASURE_WITH_SWITCH_PROXY_S * expected_levels + SUCCESSFUL_CLEAR_PROXY_S
        )
        if direct != expected_direct:
            errors.append(f"{label}: direct flag mismatch")
        if levels != expected_levels or not 0 <= levels <= MAX_LOCAL_MEASUREMENTS:
            errors.append(f"{label}: local measurement level mismatch")
        if not _close(internal, expected_internal) or not _close(local_action, expected_action):
            errors.append(f"{label}: service cost mismatch")
    return errors


def _audit_decision(event_index, event, actions):
    errors = []
    try:
        nodes = [
            (item["kind"], int(item["ident"]), tuple(map(float, item["destination"])))
            for item in event["nodes"]
        ]
        profile_items = event["profiles"]
        choices = event["choices"]
        node_keys = [_node_key(node) for node in nodes]
        profile_keys = [tuple(item["key"]) for item in profile_items]
        choice_keys = [tuple(item["first"]) for item in choices]
    except (KeyError, TypeError, ValueError) as error:
        return [f"decision {event_index}: malformed decision record: {error}"], False
    expected_keys = sorted(node_keys)
    if not nodes or len(set(node_keys)) != len(node_keys):
        errors.append(f"decision {event_index}: empty or duplicate nodes")
    if sorted(profile_keys) != expected_keys or len(set(profile_keys)) != len(profile_keys):
        errors.append(f"decision {event_index}: nodes and profiles do not reconcile")
    if choice_keys != expected_keys or len(set(choice_keys)) != len(choice_keys):
        errors.append(f"decision {event_index}: choices are not one sorted record per node")
    if errors:
        return errors, False

    profiles = {tuple(item["key"]): item for item in profile_items}
    errors.extend(_profile_errors(event_index, nodes, profiles))
    errors.extend(_proxy_errors(f"decision {event_index} anchor", event.get("anchor_proxy", {})))
    errors.extend(_proxy_errors(f"decision {event_index} best", event.get("best_proxy", {})))
    for choice_index, choice in enumerate(choices):
        errors.extend(_proxy_errors(f"decision {event_index} choice {choice_index}", choice))

    action_count = int(event.get("measure_count_before", -1)) + int(event.get("clear_count_before", -1))
    if action_count < 0 or action_count > len(actions):
        errors.append(f"decision {event_index}: public action index out of range")
        prior = []
    else:
        prior = actions[:action_count]
        measure_count = sum(action.get("path") == "/measure" for action in prior)
        clear_count = sum(action.get("path") == "/clear" for action in prior)
        if measure_count != int(event["measure_count_before"]) or clear_count != int(event["clear_count_before"]):
            errors.append(f"decision {event_index}: public action counters do not match the prefix")
        expected_position = (0.0, 0.0) if not prior else tuple(prior[-1]["point"])
        if not _same_point(event["position"], expected_position):
            errors.append(f"decision {event_index}: public position/action index mismatch")

    actual_last_measure = {}
    for action in prior:
        if action.get("path") == "/measure":
            actual_last_measure[int(action["channel"])] = tuple(action["point"])
    logged_last_measure = {
        int(channel): tuple(point) for channel, point in event.get("last_measure_points", {}).items()
    }
    if set(actual_last_measure) != set(logged_last_measure) or any(
        not _same_point(actual_last_measure[channel], logged_last_measure[channel])
        for channel in actual_last_measure.keys() & logged_last_measure.keys()
    ):
        errors.append(f"decision {event_index}: last public measurement map mismatch")
    successful_before = {
        int(action["channel"])
        for action in prior
        if action.get("path") == "/clear" and action.get("response", {}).get("clear_result") == "success"
    }
    scannable = [int(channel) for channel in event.get("scannable_channels", [])]
    if len(scannable) != len(set(scannable)) or not set(scannable) <= set(range(1, 21)):
        errors.append(f"decision {event_index}: invalid scannable channel set")
    if successful_before & set(scannable):
        errors.append(f"decision {event_index}: cleared channel retained in scan proxy")

    try:
        anchor = anchor_route(tuple(event["position"]), nodes)
    except Exception as error:  # static, deterministic route replay only
        errors.append(f"decision {event_index}: frozen route replay failed: {type(error).__name__}: {error}")
        return errors, False
    anchor_first = tuple(event.get("anchor_first", ()))
    if not anchor or _node_key(anchor[0]) != anchor_first:
        errors.append(f"decision {event_index}: frozen anchor first node mismatch")
    recomputed_anchor = _independent_proxy(
        event["position"], anchor, profiles, scannable, logged_last_measure
    )
    if not _proxy_matches(event.get("anchor_proxy", {}), recomputed_anchor):
        errors.append(f"decision {event_index}: frozen anchor proxy mismatch")

    recomputed_choices = []
    for first in sorted(nodes, key=_node_key):
        remaining = list(nodes)
        remaining.remove(first)
        suffix = anchor_route(profiles[_node_key(first)]["exit"], remaining)
        route = [first] + suffix
        proxy = _independent_proxy(event["position"], route, profiles, scannable, logged_last_measure)
        recomputed_choices.append({"first": _node_key(first), **proxy})
    for choice_index, (logged, recomputed) in enumerate(zip(choices, recomputed_choices)):
        if tuple(logged["first"]) != tuple(recomputed["first"]) or not _proxy_matches(logged, recomputed):
            errors.append(f"decision {event_index}: choice {choice_index} proxy mismatch")

    pool = [{"origin": "full_anchor", "first": anchor_first, **recomputed_anchor}]
    pool.extend({"origin": "first_enumeration", **item} for item in recomputed_choices)
    expected_best = min(
        pool,
        key=lambda item: (float(item["total_s"]), float(item["movement_s"]), tuple(item["first"])),
    )
    best_first = tuple(event.get("best_first", ()))
    if best_first != tuple(expected_best["first"]):
        errors.append(f"decision {event_index}: best first node mismatch")
    if not _proxy_matches(event.get("best_proxy", {}), expected_best):
        errors.append(f"decision {event_index}: best proxy mismatch")
    saving = recomputed_anchor["total_s"] - expected_best["total_s"]
    if not _close(event.get("proxy_saving_s", math.nan), saving):
        errors.append(f"decision {event_index}: proxy saving mismatch")
    threshold = float(event.get("minimum_proxy_saving_s", math.nan))
    if not _close(threshold, MIN_PROXY_SAVING_S):
        errors.append(f"decision {event_index}: undeclared change threshold")
    expected_changed = best_first != anchor_first and saving >= threshold
    if bool(event.get("changed_first")) != expected_changed:
        errors.append(f"decision {event_index}: threshold decision mismatch")
    expected_selected = best_first if expected_changed else anchor_first
    if tuple(event.get("selected_first", ())) != expected_selected:
        errors.append(f"decision {event_index}: selected first node mismatch")
    return errors, expected_changed


def _audit_duplicate_skips(state, actions):
    errors = []
    skips = state.get("duplicate_supplement_skips", [])
    for index, event in enumerate(skips):
        action_count = int(event.get("measure_count_before", -1)) + int(event.get("clear_count_before", -1))
        if action_count < 0 or action_count > len(actions):
            errors.append(f"duplicate skip {index}: public action index out of range")
            continue
        prior = actions[:action_count]
        measure_count = sum(action.get("path") == "/measure" for action in prior)
        clear_count = sum(action.get("path") == "/clear" for action in prior)
        if measure_count != int(event["measure_count_before"]) or clear_count != int(event["clear_count_before"]):
            errors.append(f"duplicate skip {index}: public action counters do not match the prefix")
        previous = next(
            (
                action
                for action in reversed(prior)
                if action.get("path") == "/measure" and int(action["channel"]) == int(event["channel"])
            ),
            None,
        )
        if previous is None or not _same_point(previous["point"], event["position"]):
            errors.append(f"duplicate skip {index}: no same-channel same-point public measurement")
        expected_position = (0.0, 0.0) if not prior else tuple(prior[-1]["point"])
        if not _same_point(event["position"], expected_position):
            errors.append(f"duplicate skip {index}: event is not at the current public position")
        if not _same_point(event["position"], event["last_measure_point"]):
            errors.append(f"duplicate skip {index}: last-measure point mismatch")
        if math.dist(tuple(event["position"]), tuple(event["initial_point"])) < REUSE_MIN_BASELINE_M - FLOAT_TOL:
            errors.append(f"duplicate skip {index}: frozen supplement would already be suppressed")
    if len(skips) != int(state.get("duplicate_supplement_skip_count", -1)):
        errors.append("duplicate-supplement counter mismatch")
    return errors


def _replay_public_metrics(case, state, actions):
    source_channels = {int(source["channel"]) for source in case["sources"]}
    groups = [set(map(int, state[name])) for name in ("cleared_channels", "excluded_channels", "unresolved_channels")]
    successful_channels = {
        int(action["channel"])
        for action in actions
        if action.get("path") == "/clear" and action.get("response", {}).get("clear_result") == "success"
    }
    full_clear = successful_channels == source_channels == groups[0]
    channels_accounted = set.union(*groups) == set(range(1, 21)) and sum(map(len, groups)) == 20
    false_excluded = bool(source_channels & groups[1])
    position = (0.0, 0.0)
    current_channel = 1
    public_cleared = set()
    post_clear_measure_count = 0
    costs = {
        "movement_s": 0.0,
        "measure_s": 0.0,
        "switch_s": 0.0,
        "successful_clear_s": 0.0,
        "failed_clear_s": 0.0,
    }
    for action in actions:
        point = tuple(action["point"])
        costs["movement_s"] += math.dist(position, point) / ROBOT_SPEED_MPS
        position = point
        if action["path"] == "/measure":
            costs["measure_s"] += 5.0
            costs["switch_s"] += int(int(action["channel"]) != current_channel)
            current_channel = int(action["channel"])
            post_clear_measure_count += int(action["channel"]) in public_cleared
        elif action.get("response", {}).get("clear_result") == "success":
            costs["successful_clear_s"] += 5.0
            public_cleared.add(int(action["channel"]))
        else:
            costs["failed_clear_s"] += 3.0
    virtual_time_s = sum(costs.values())
    measures = sum(action["path"] == "/measure" for action in actions)
    clears = sum(action["path"] == "/clear" for action in actions)
    expected_geometry = coverage_certificate()
    checks = {
        "completed": state.get("completed") is True and state.get("stopped") is False,
        "full_source_clear": full_clear,
        "channels_accounted": channels_accounted,
        "no_false_excluded_source": not false_excluded,
        "clock_accounting": _close(state.get("virtual_time_s", math.nan), virtual_time_s, 1e-6),
        "action_counters": int(state.get("measure_count", -1)) == measures
        and int(state.get("clear_count", -1)) == clears,
        "coverage_certificate": int(state.get("coverage_points_total", -1)) == 22
        and state.get("coverage_geometry") == expected_geometry,
        "source_domain": all(math.hypot(float(source["x"]), float(source["y"])) <= 1800.0 + FLOAT_TOL for source in case["sources"]),
        "no_post_clear_measure": post_clear_measure_count == 0,
    }
    fingerprint = hashlib.sha256(
        json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "full_source_clear": full_clear,
        "channels_accounted": channels_accounted,
        "false_excluded_source": false_excluded,
        "post_clear_measure_count": post_clear_measure_count,
        "source_count": len(source_channels),
        "time_s": virtual_time_s,
        "mean_s": virtual_time_s / len(source_channels),
        "measure_count": measures,
        "clear_attempts": clears,
        "public_action_fingerprint": fingerprint,
        **costs,
    }


def _metric_errors(case_id, policy_name, row_metrics, replayed):
    errors = []
    exact_fields = (
        "full_source_clear",
        "channels_accounted",
        "false_excluded_source",
        "post_clear_measure_count",
        "source_count",
        "measure_count",
        "clear_attempts",
        "public_action_fingerprint",
    )
    numeric_fields = (
        "time_s",
        "mean_s",
        "movement_s",
        "measure_s",
        "switch_s",
        "successful_clear_s",
        "failed_clear_s",
    )
    for field in exact_fields:
        if row_metrics.get(field) != replayed[field]:
            errors.append(f"{case_id} {policy_name}: metric {field} mismatch")
    for field in numeric_fields:
        if not _close(row_metrics.get(field, math.nan), replayed[field], 1e-6):
            errors.append(f"{case_id} {policy_name}: metric {field} mismatch")
    return errors


def _manifest_audit(records):
    mismatches = []
    for record in records:
        path = ROOT / record["path"]
        observed = _sha(path) if path.is_file() else None
        if observed != record["sha256"]:
            mismatches.append(
                {"path": record["path"], "expected_sha256": record["sha256"], "observed_sha256": observed}
            )
    return {
        "path": _relative(MANIFEST),
        "sha256": _sha(MANIFEST),
        "entry_count": len(records),
        "all_entries_match": not mismatches,
        "mismatches": mismatches,
    }


def _static_source_checks(candidate_text, original_scope):
    try:
        compile(candidate_text, str(CANDIDATE), "exec")
        compile(Path(__file__).read_text(encoding="utf-8"), str(Path(__file__)), "exec")
        syntax_pass = True
    except SyntaxError:
        syntax_pass = False
    checks = {
        "syntax_compiles": syntax_pass,
        "original_scope_checks_pass": original_scope.get("all_checks_pass") is True,
        "uses_frozen_localizer_and_state": "from search_policy_q4_batch import _TrackedClient, _localize, _state" in candidate_text,
        "uses_frozen_route": "from search_policy_joint_route_q3 import _route as _anchor_route" in candidate_text,
        "uses_ring22_points_and_certificate": "from search_policy_q4_ring22 import coverage_certificate, ring22_points" in candidate_text,
        "preserves_eight_level_proxy_cap": "MAX_LOCAL_MEASUREMENTS = 8" in candidate_text,
        "preserves_full_localizer_call": "_localize(tracked, track[0], track[1], channel, state, track[2])" in candidate_text,
        "does_not_import_offline_truth": "offline_arena" not in candidate_text,
        "does_not_import_compatibility": "q4_joint_compatibility" not in candidate_text,
        "records_compatibility_as_unused": "compatibility_used_for_decision=False" in candidate_text,
        "frozen_baseline_not_edited": _sha(BASELINE)
        == next(record["sha256"] for record in _load(MANIFEST) if record["path"] == _relative(BASELINE)),
    }
    return {"all_checks_pass": all(checks.values()), "checks": checks}


def _review(audit, audit_sha):
    evidence = [
        {"path": _relative(AUDIT), "sha256": audit_sha},
        {"path": _relative(COMPARISON), "sha256": _sha(COMPARISON)},
    ]
    plan_evidence = evidence + [{"path": _relative(PLAN), "sha256": _sha(PLAN)}]
    inputs = [
        {"path": _relative(path), "sha256": _sha(path)}
        for path in (PLAN, CANDIDATE, BASELINE, Path(__file__), MANIFEST, SOURCE_CASES, COMPARISON, AUDIT)
    ]
    gate = audit["corrected_gate"]
    return {
        "schema_version": 2,
        "artifact_type": "review",
        "question_id": "Q4_service_aware_route_v1",
        "language": "python",
        "inputs": inputs,
        "checks": {
            "syntax": {
                "status": "PASS" if audit["static_source_checks"]["checks"]["syntax_compiles"] else "FAIL",
                "reason": "Candidate and audit sources compile; the candidate completed all 36 saved offline development cases." if audit["static_source_checks"]["checks"]["syntax_compiles"] else "A reviewed Python source did not compile.",
                "evidence": evidence,
            },
            "input_contract": {
                "status": "PASS" if audit["input_integrity"]["all_checks_pass"] else "FAIL",
                "reason": "All 17 pre-run hashes match, and the same frozen 36 cases are paired for baseline and candidate." if audit["input_integrity"]["all_checks_pass"] else "A frozen input identity or paired-population check failed.",
                "evidence": evidence,
            },
            "method_alignment": {
                "status": "PASS" if audit["static_source_checks"]["all_checks_pass"] else "FAIL",
                "reason": "The development candidate stays inside fixed_cover_bearing: frozen ring22 coverage, 19.9 m certificate, eight-measure localizer and complete fallback remain; compatibility and hidden truth are not used." if audit["static_source_checks"]["all_checks_pass"] else "A declared reliability or information-boundary check failed.",
                "evidence": plan_evidence,
            },
            "candidate_screening": {
                "status": "FAIL",
                "reason": (
                    f"All hard checks pass and mean saving is {gate['absolute_saving_s_per_source']:.6f}s/source, "
                    f"but relative improvement {100.0 * gate['relative_improvement']:.6f}% is below the predeclared 5% threshold; candidate remains rejected."
                ),
                "evidence": plan_evidence,
            },
            "reproducibility": {
                "status": "PASS" if audit["output_replay"]["all_checks_pass"] else "FAIL",
                "reason": "Seeds, error modes and input hashes are frozen; all 72 saved public action histories independently reproduce their fingerprints, counters, clocks and cost components. No simulation rerun was performed." if audit["output_replay"]["all_checks_pass"] else "A saved action history did not reproduce its recorded metrics.",
                "evidence": evidence,
            },
            "output_contract": {
                "status": "PASS" if audit["corrected_all_hard_checks_pass"] else "FAIL",
                "reason": "The saved 36-pair evidence reconciles and all sources clear. The v1 evaluator's three proxy-minimum flags were false positives caused by omitting the full frozen anchor from its audit pool; the versioned correction does not alter the original outputs." if audit["corrected_all_hard_checks_pass"] else "A corrected hard check remains unresolved.",
                "evidence": evidence,
            },
        },
        "findings": [
            {
                "priority": 1,
                "code": "DEVELOPMENT_GAIN_BELOW_PREDECLARED_RELATIVE_THRESHOLD",
                "reason": "Retain the negative result; do not tune on these cases or proceed to holdout, practice, or formal testing.",
            },
            {
                "priority": 2,
                "code": "V1_EVALUATOR_OMITTED_FULL_ANCHOR_FROM_AUDIT_POOL",
                "reason": "Three audit-only flags are corrected in audit_v2.json; comparison.json and decision.json remain immutable evidence.",
            },
        ],
        "verdict": "FAILED",
        "extensions": {
            "scope": "DEVELOPMENT_CANDIDATE_REVIEW",
            "candidate": "q4_service_aware_route_v1",
            "accepted": False,
            "formal_tests_used": 0,
            "simulations_rerun_for_v2_audit": 0,
            "frozen_baseline_modified": False,
            "original_comparison_modified": False,
        },
    }


def main():
    manifest_records = _load(MANIFEST)
    comparison = _load(COMPARISON)
    source_cases = _load(SOURCE_CASES)
    saved_cases = _load(SAVED_CASES)
    original_scope = _load(STATIC_SCOPE)
    case_ids = [case["case_id"] for case in source_cases]
    row_ids = [row["case_id"] for row in comparison["rows"]]
    population_checks = {
        "source_case_count_is_36": len(source_cases) == 36,
        "saved_case_copy_matches_source": saved_cases == source_cases,
        "case_ids_are_unique": len(case_ids) == len(set(case_ids)),
        "comparison_rows_match_case_order": row_ids == case_ids,
        "case_seed_and_error_mode_match": all(
            row["seed"] == case["seed"] and row["error_mode"] == case["error_mode"]
            for row, case in zip(comparison["rows"], source_cases)
        ),
    }
    manifest_audit = _manifest_audit(manifest_records)
    input_integrity = {
        "all_checks_pass": manifest_audit["all_entries_match"] and all(population_checks.values()),
        "manifest": manifest_audit,
        "population_checks": population_checks,
    }

    candidate_text = CANDIDATE.read_text(encoding="utf-8")
    static_checks = _static_source_checks(candidate_text, original_scope)
    case_results = []
    all_output_errors = []
    corrected_decisions = 0
    corrected_changed = 0
    corrected_skips = 0
    raw_task_pass = True
    original_false_positive_cases = []
    for case, row in zip(source_cases, comparison["rows"]):
        case_result = {
            "case_id": case["case_id"],
            "seed": case["seed"],
            "error_mode": case["error_mode"],
        }
        for policy_name in ("baseline", "candidate"):
            state_path = OUT / "states" / f"{case['case_id']}__{policy_name}.json"
            record = _load(state_path)
            state, actions = record["state"], record["actions"]
            replayed = _replay_public_metrics(case, state, actions)
            metric_errors = _metric_errors(case["case_id"], policy_name, row[policy_name], replayed)
            all_output_errors.extend(metric_errors)
            raw_task_pass = raw_task_pass and replayed["all_checks_pass"] and not metric_errors
            entry = {
                "state_actions_path": _relative(state_path),
                "state_actions_sha256": _sha(state_path),
                "raw_task_checks_pass": replayed["all_checks_pass"],
                "metric_reconciliation_pass": not metric_errors,
                "errors": metric_errors,
            }
            if policy_name == "candidate":
                decision_errors = []
                changed = 0
                decisions = state.get("service_route_decisions", [])
                for index, event in enumerate(decisions):
                    errors, event_changed = _audit_decision(index, event, actions)
                    decision_errors.extend(errors)
                    changed += int(event_changed)
                if changed != int(state.get("service_route_changed_first_count", -1)):
                    decision_errors.append("changed-first counter mismatch")
                skip_errors = _audit_duplicate_skips(state, actions)
                if state.get("compatibility_used_for_decision") is not False:
                    decision_errors.append("compatibility must remain unused in v1")
                if state.get("version") != "q4_service_aware_route_v1":
                    decision_errors.append("candidate version mismatch")
                decision_errors.extend(skip_errors)
                corrected_decisions += len(decisions)
                corrected_changed += changed
                corrected_skips += len(state.get("duplicate_supplement_skips", []))
                old_audit = row["candidate"].get("service_audit", {})
                corrected_pass = not decision_errors
                if not old_audit.get("all_checks_pass", False) and corrected_pass:
                    original_false_positive_cases.append(
                        {
                            "case_id": case["case_id"],
                            "old_errors": old_audit.get("errors", []),
                            "corrected_errors": [],
                        }
                    )
                entry["service_audit_v2"] = {
                    "all_checks_pass": corrected_pass,
                    "decision_count": len(decisions),
                    "changed_first_count": changed,
                    "duplicate_supplement_skip_count": len(state.get("duplicate_supplement_skips", [])),
                    "errors": decision_errors,
                }
                all_output_errors.extend(f"{case['case_id']} candidate: {error}" for error in decision_errors)
            case_result[policy_name] = entry
        case_results.append(case_result)

    output_replay_pass = raw_task_pass and not all_output_errors
    baseline_mean = statistics.mean(row["baseline"]["mean_s"] for row in comparison["rows"])
    candidate_mean = statistics.mean(row["candidate"]["mean_s"] for row in comparison["rows"])
    saving = baseline_mean - candidate_mean
    relative = saving / baseline_mean
    worst = max(comparison["rows"], key=lambda row: row["candidate_to_baseline_ratio"])
    gate_checks = {
        "all_hard_checks": input_integrity["all_checks_pass"] and static_checks["all_checks_pass"] and output_replay_pass,
        "absolute_saving_at_least_10_s_per_source": saving >= ABSOLUTE_MIN_S_PER_SOURCE,
        "relative_improvement_at_least_5_percent": relative >= RELATIVE_MIN,
        "worst_case_ratio_at_most_1_10": worst["candidate_to_baseline_ratio"] <= WORST_RATIO_MAX,
    }
    accepted = all(gate_checks.values())
    audit = {
        "schema_version": 1,
        "artifact_type": "q4_service_aware_route_development_audit",
        "audit_version": AUDIT_VERSION,
        "scope": "POST_HOC_READ_ONLY_REAUDIT_OF_FROZEN_36_CASE_OUTPUTS_NO_SIMULATION_RERUN",
        "audit_identity": {
            "script_path": _relative(Path(__file__)),
            "script_sha256": _sha(Path(__file__)),
            "original_manifest_path": _relative(MANIFEST),
            "original_manifest_sha256": _sha(MANIFEST),
            "original_comparison_path": _relative(COMPARISON),
            "original_comparison_sha256": _sha(COMPARISON),
            "simulations_rerun": 0,
        },
        "input_integrity": input_integrity,
        "static_source_checks": static_checks,
        "output_replay": {
            "all_checks_pass": output_replay_pass,
            "raw_task_checks_all_72_runs_pass": raw_task_pass,
            "metric_reconciliation_all_72_runs_pass": not any(
                result[policy]["errors"]
                for result in case_results
                for policy in ("baseline", "candidate")
            ),
            "errors": all_output_errors,
            "case_results": case_results,
        },
        "service_audit_v2": {
            "all_checks_pass": not any(
                result["candidate"]["service_audit_v2"]["errors"] for result in case_results
            ),
            "decision_count": corrected_decisions,
            "changed_first_count": corrected_changed,
            "duplicate_supplement_skip_count": corrected_skips,
            "compatibility_used_for_decision": False,
            "original_false_positive_cases": original_false_positive_cases,
            "correction": "The v1 audit minimized only enumerated first-plus-recomputed-suffix choices. The policy also compares the complete frozen anchor route, which v2 includes in the candidate pool.",
        },
        "corrected_all_hard_checks_pass": gate_checks["all_hard_checks"],
        "corrected_gate": {
            "accepted": accepted,
            "checks": gate_checks,
            "baseline_mean_s_per_source": baseline_mean,
            "candidate_mean_s_per_source": candidate_mean,
            "absolute_saving_s_per_source": saving,
            "relative_improvement": relative,
            "worst_case": {
                "case_id": worst["case_id"],
                "candidate_to_baseline_ratio": worst["candidate_to_baseline_ratio"],
            },
            "decision": "REJECT_AND_RETAIN_NEGATIVE_RESULT" if not accepted else "ELIGIBLE_FOR_INDEPENDENT_REVIEW",
            "next_step": "stop_without_tuning_holdout_practice_or_formal_test" if not accepted else "independent_review",
        },
        "original_artifacts_unchanged": {
            "comparison_json": True,
            "decision_json": True,
            "frozen_baseline_policy": True,
        },
    }
    _save(AUDIT, audit)
    review = _review(audit, _sha(AUDIT))
    _save(REVIEW, review)
    print(
        json.dumps(
            {
                "audit": _relative(AUDIT),
                "review": _relative(REVIEW),
                "corrected_all_hard_checks_pass": audit["corrected_all_hard_checks_pass"],
                "accepted": accepted,
                "absolute_saving_s_per_source": saving,
                "relative_improvement": relative,
                "worst_ratio": worst["candidate_to_baseline_ratio"],
                "original_false_positive_cases": [item["case_id"] for item in original_false_positive_cases],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
