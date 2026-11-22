"""One-shot paired offline evaluation of q4_service_multistart_route_v2."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import search_policy_q4_ring22_joint_route as baseline
import search_policy_q4_service_multistart_route_v2 as candidate
from evaluate_q4_domain_prior_development import run, save, sha, summarize
from route_multistart import route as multistart_route
from search_policy_joint_route_q3 import _route as anchor_route
from search_policy_q4_service_aware_route import _proxy_cost


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q4_service_multistart_v2/development"
CASES = OUT / "cases.json"
GENERATION = OUT / "case_generation.json"
COMPARISON = OUT / "comparison.json"
DECISION = OUT / "decision.json"
PRECOMMIT = OUT / "run_precommit.json"
RELATIVE_MIN = 0.05
ABSOLUTE_MIN_S_PER_SOURCE = 10.0
WORST_RATIO_MAX = 1.10
MIN_PROXY_SAVING_S = 6.0
GEOMETRY_TOLERANCE_M = 1e-7
PROXY_TIE_TOLERANCE_S = 1e-7
FLOAT_TOL = 1e-7


def _node_key(node):
    return (str(node[0]), int(node[1]))


def _route_keys(route):
    return [_node_key(node) for node in route]


def _route_length(position, route):
    points = [tuple(position)] + [tuple(node[2]) for node in route]
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


def _same_point(first, second):
    return math.dist(tuple(first), tuple(second)) <= FLOAT_TOL


def _close(first, second, tolerance=FLOAT_TOL):
    return abs(float(first) - float(second)) <= tolerance


def _proxy_match(first, second):
    fields = ("total_s", "movement_s", "scan_action_s", "source_action_s")
    return all(_close(first[field], second[field]) for field in fields) and int(
        first["duplicate_supplement_skips"]
    ) == int(second["duplicate_supplement_skips"])


def _construct_v1(position, nodes, profiles, scannable, last_measure, anchor, anchor_proxy):
    best_route = anchor
    best_proxy = anchor_proxy
    choices = []
    for first in sorted(nodes, key=_node_key):
        remaining = list(nodes)
        remaining.remove(first)
        route = [first] + anchor_route(profiles[_node_key(first)]["exit"], remaining)
        proxy = _proxy_cost(position, route, profiles, scannable, last_measure)
        choices.append({"first": _node_key(first), **proxy})
        order = (proxy["total_s"], proxy["movement_s"], _node_key(first))
        best_order = (best_proxy["total_s"], best_proxy["movement_s"], _node_key(best_route[0]))
        if order < best_order:
            best_route, best_proxy = route, proxy
    return best_route, best_proxy, choices


def _duplicate_skip_errors(state, actions):
    errors = []
    skips = state.get("duplicate_supplement_skips", [])
    for index, event in enumerate(skips):
        action_count = int(event["measure_count_before"]) + int(event["clear_count_before"])
        if not 0 <= action_count <= len(actions):
            errors.append(f"duplicate skip {index}: action index out of range")
            continue
        prior = actions[:action_count]
        previous = next(
            (
                action
                for action in reversed(prior)
                if action["path"] == "/measure" and int(action["channel"]) == int(event["channel"])
            ),
            None,
        )
        if previous is None or not _same_point(previous["point"], event["position"]):
            errors.append(f"duplicate skip {index}: no same-channel same-point public measure")
        expected_position = (0.0, 0.0) if not prior else prior[-1]["point"]
        if not _same_point(expected_position, event["position"]):
            errors.append(f"duplicate skip {index}: not at current public position")
        if math.dist(tuple(event["position"]), tuple(event["initial_point"])) < 50.0 - FLOAT_TOL:
            errors.append(f"duplicate skip {index}: frozen supplement was already suppressed")
    if len(skips) != int(state.get("duplicate_supplement_skip_count", -1)):
        errors.append("duplicate skip counter mismatch")
    return errors


def verify_route_decisions(state, actions):
    errors = []
    selections = Counter()
    decisions = state.get("joint_route_decisions", [])
    for index, event in enumerate(decisions):
        try:
            nodes = [
                (item["kind"], int(item["ident"]), tuple(item["destination"]))
                for item in event["nodes"]
            ]
            node_keys = [_node_key(node) for node in nodes]
            profiles = {tuple(item["key"]): item for item in event["profiles"]}
            scannable = [int(channel) for channel in event["scannable_channels"]]
            last_measure = {
                int(channel): tuple(point) for channel, point in event["last_measure_points"].items()
            }
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"decision {index}: malformed event: {error}")
            continue
        if not nodes or len(set(node_keys)) != len(node_keys):
            errors.append(f"decision {index}: empty or duplicate nodes")
            continue
        if set(profiles) != set(node_keys) or len(profiles) != len(nodes):
            errors.append(f"decision {index}: nodes/profiles do not reconcile")
            continue
        if len(scannable) != len(set(scannable)) or not set(scannable) <= set(range(1, 21)):
            errors.append(f"decision {index}: invalid scannable channels")

        action_count = int(event["measure_count_before"]) + int(event["clear_count_before"])
        if not 0 <= action_count <= len(actions):
            errors.append(f"decision {index}: action index out of range")
            prior = []
        else:
            prior = actions[:action_count]
            if sum(action["path"] == "/measure" for action in prior) != int(event["measure_count_before"]):
                errors.append(f"decision {index}: measure prefix count mismatch")
            if sum(action["path"] == "/clear" for action in prior) != int(event["clear_count_before"]):
                errors.append(f"decision {index}: clear prefix count mismatch")
            expected_position = (0.0, 0.0) if not prior else prior[-1]["point"]
            if not _same_point(event["position"], expected_position):
                errors.append(f"decision {index}: position differs from public history")
        public_last = {}
        for action in prior:
            if action["path"] == "/measure":
                public_last[int(action["channel"])] = tuple(action["point"])
        if set(public_last) != set(last_measure) or any(
            not _same_point(public_last[channel], last_measure[channel]) for channel in public_last
        ):
            errors.append(f"decision {index}: last-measure map differs from public history")

        anchor = anchor_route(event["position"], nodes)
        anchor_proxy = _proxy_cost(event["position"], anchor, profiles, scannable, last_measure)
        v1_route, v1_proxy, v1_choices = _construct_v1(
            event["position"], nodes, profiles, scannable, last_measure, anchor, anchor_proxy
        )
        multi_route = multistart_route(event["position"], nodes)
        multi_proxy = _proxy_cost(
            event["position"], multi_route, profiles, scannable, last_measure
        )
        expected = {
            "anchor": (anchor, anchor_proxy),
            "v1_service": (v1_route, v1_proxy),
            "multistart_service": (multi_route, multi_proxy),
        }
        anchor_length = _route_length(event["position"], anchor)
        expected_eligible = []
        for name, (route, proxy) in expected.items():
            record = event["routes"].get(name, {})
            length = _route_length(event["position"], route)
            saving = anchor_proxy["total_s"] - proxy["total_s"]
            proxy_guard = saving >= MIN_PROXY_SAVING_S
            geometry_guard = length <= anchor_length + GEOMETRY_TOLERANCE_M
            eligible = name != "anchor" and proxy_guard and geometry_guard
            if _route_keys(route) != [tuple(key) for key in record.get("route", [])]:
                errors.append(f"decision {index}: {name} route replay mismatch")
            if not _proxy_match(proxy, record.get("proxy", {})):
                errors.append(f"decision {index}: {name} proxy replay mismatch")
            if not _close(length, record.get("geometry_length_m", math.nan)):
                errors.append(f"decision {index}: {name} geometry length mismatch")
            if not _close(saving, record.get("proxy_saving_vs_anchor_s", math.nan)):
                errors.append(f"decision {index}: {name} proxy saving mismatch")
            if record.get("proxy_guard_pass") is not proxy_guard:
                errors.append(f"decision {index}: {name} proxy guard mismatch")
            if record.get("geometry_guard_pass") is not geometry_guard:
                errors.append(f"decision {index}: {name} geometry guard mismatch")
            if record.get("eligible") is not eligible:
                errors.append(f"decision {index}: {name} eligibility mismatch")
            if eligible:
                expected_eligible.append(name)

        logged_choices = event.get("v1_first_choices", [])
        if len(logged_choices) != len(v1_choices):
            errors.append(f"decision {index}: v1 first-choice count mismatch")
        else:
            for choice_index, (logged, replayed) in enumerate(zip(logged_choices, v1_choices)):
                if tuple(logged.get("first", ())) != tuple(replayed["first"]) or not _proxy_match(logged, replayed):
                    errors.append(f"decision {index}: v1 choice {choice_index} mismatch")

        if event.get("eligible_names") != expected_eligible:
            errors.append(f"decision {index}: eligible route list mismatch")
        expected_selected = "anchor"
        expected_reason = "no_nonanchor_route_passed_both_guards"
        expected_ties = []
        if expected_eligible:
            minimum = min(expected[name][1]["total_s"] for name in expected_eligible)
            expected_ties = [
                name
                for name in expected_eligible
                if abs(expected[name][1]["total_s"] - minimum) <= PROXY_TIE_TOLERANCE_S
            ]
            if len(expected_ties) == 1:
                expected_selected = expected_ties[0]
                expected_reason = "unique_lowest_eligible_public_service_proxy"
            else:
                expected_reason = "eligible_proxy_tie_falls_back_to_anchor"
        if event.get("tie_names") != expected_ties:
            errors.append(f"decision {index}: tie list mismatch")
        if event.get("selected_strategy") != expected_selected:
            errors.append(f"decision {index}: selected strategy mismatch")
        if event.get("selection_reason") != expected_reason:
            errors.append(f"decision {index}: selection reason mismatch")
        selected_route = expected[expected_selected][0]
        if [tuple(key) for key in event.get("selected_route", [])] != _route_keys(selected_route):
            errors.append(f"decision {index}: selected route mismatch")
        if tuple(event.get("selected_first", ())) != _node_key(selected_route[0]):
            errors.append(f"decision {index}: selected first node mismatch")
        if not _close(event.get("minimum_proxy_saving_s", math.nan), MIN_PROXY_SAVING_S):
            errors.append(f"decision {index}: proxy threshold changed")
        if not _close(event.get("geometry_tolerance_m", math.nan), GEOMETRY_TOLERANCE_M):
            errors.append(f"decision {index}: geometry tolerance changed")
        if not _close(event.get("proxy_tie_tolerance_s", math.nan), PROXY_TIE_TOLERANCE_S):
            errors.append(f"decision {index}: tie tolerance changed")
        selections[expected_selected] += 1

    expected_counts = {name: selections[name] for name in ("anchor", "v1_service", "multistart_service")}
    if state.get("joint_route_selection_counts") != expected_counts:
        errors.append("aggregate route-selection counters mismatch")
    errors.extend(_duplicate_skip_errors(state, actions))
    if state.get("compatibility_used_for_decision") is not False:
        errors.append("compatibility module must not participate in v2")
    return {
        "all_checks_pass": not errors,
        "errors": errors,
        "decision_count": len(decisions),
        "selection_counts": expected_counts,
        "duplicate_supplement_skip_count": len(state.get("duplicate_supplement_skips", [])),
    }


def static_scope_checks(generation):
    text = Path(candidate.__file__).read_text(encoding="utf-8")
    checks = [
        {"name": "candidate_version_frozen", "pass": candidate.VERSION == "q4_service_multistart_route_v2"},
        {"name": "uses_frozen_anchor", "pass": "from search_policy_joint_route_q3 import _route as _anchor_route" in text},
        {"name": "uses_existing_multistart", "pass": "from route_multistart import route as _multistart_route" in text},
        {"name": "inherits_v1_public_service_proxy", "pass": "from search_policy_q4_service_aware_route import (" in text and "_proxy_cost" in text},
        {"name": "uses_ring22_certificate", "pass": "from search_policy_q4_ring22 import coverage_certificate, ring22_points" in text},
        {"name": "preserves_frozen_localizer", "pass": "from search_policy_q4_batch import _localize, _state" in text},
        {"name": "does_not_import_offline_truth", "pass": "offline_arena" not in text},
        {"name": "does_not_use_compatibility", "pass": "q4_joint_compatibility" not in text},
        {"name": "fresh_case_novelty_pass", "pass": generation["novelty_audit"]["all_checks_pass"] is True},
        {"name": "fresh_case_distribution_pass", "pass": generation["distribution_audit"]["all_checks_pass"] is True},
        {"name": "candidate_precedes_case_generation", "pass": generation["freeze_identity"]["candidate_sha256_at_generation"] == sha(Path(candidate.__file__))},
        {"name": "case_hash_matches_generation", "pass": generation["case_file_sha256"] == sha(CASES)},
    ]
    return {"all_checks_pass": all(item["pass"] for item in checks), "checks": checks}


def main():
    if COMPARISON.exists() or DECISION.exists():
        raise ValueError("v2 one-shot comparison already exists; do not rerun or overwrite")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    if len(cases) != 42 or [case["seed"] for case in cases] != list(range(63000, 63042)):
        raise ValueError("fresh42 case identity does not match the predeclared rule")
    if not generation["novelty_audit"]["all_checks_pass"] or not generation["distribution_audit"]["all_checks_pass"]:
        raise ValueError("case generation audit failed")

    relative_inputs = [
        "planning/q4_service_multistart_v2_experiment.md",
        "code/shared/generate_q4_service_multistart_v2_cases.py",
        "code/shared/search_policy_q4_service_multistart_route_v2.py",
        "code/shared/evaluate_q4_service_multistart_v2_development.py",
        "code/shared/search_policy_q4_service_aware_route.py",
        "code/shared/search_policy_q4_ring22_joint_route.py",
        "code/shared/route_multistart.py",
        "code/shared/search_policy.py",
        "code/shared/search_policy_polygon.py",
        "code/shared/search_policy_joint_route_q3.py",
        "code/shared/search_policy_q4_batch.py",
        "code/shared/search_policy_q4_reuse.py",
        "code/shared/search_policy_q4_ring22.py",
        "code/shared/evaluate_q4_domain_prior_development.py",
        "code/shared/evaluate_q4_shadow_recovery_development.py",
        "code/shared/offline_arena.py",
        "code/shared/simulator_client.py",
        "code/shared/shadow_cone_geometry.py",
        "results/optimization_200_400/ring22_continuous_certificate.json",
        "results/q4_service_multistart_v2/development/cases.json",
        "results/q4_service_multistart_v2/development/case_generation.json",
    ]
    inputs = [{"path": path, "sha256": sha(ROOT / path)} for path in relative_inputs]
    manifest = OUT / "inputs_before_run.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != inputs:
        raise ValueError("v2 frozen inputs changed; do not overwrite this experiment identity")
    save(manifest, inputs)
    scope_checks = static_scope_checks(generation)
    save(OUT / "static_scope_checks.json", scope_checks)
    precommit = {
        "scope": "ONE_SHOT_PAIRED_OFFLINE_STRUCTURAL_DEVELOPMENT_ONLY",
        "candidate": candidate.VERSION,
        "baseline": baseline.VERSION,
        "case_count": 42,
        "seed_range": [63000, 63041],
        "predeclared_gate": {
            "relative_min": RELATIVE_MIN,
            "absolute_min_s_per_source": ABSOLUTE_MIN_S_PER_SOURCE,
            "worst_ratio_max": WORST_RATIO_MAX,
        },
        "inputs": inputs,
    }
    if PRECOMMIT.exists() and json.loads(PRECOMMIT.read_text(encoding="utf-8")) != precommit:
        raise ValueError("run precommit identity changed")
    save(PRECOMMIT, precommit)

    rows = []
    for case in cases:
        row = {
            "case_id": case["case_id"],
            "seed": case["seed"],
            "source_count_stratum": case["source_count_stratum"],
            "error_mode": case["error_mode"],
            "replicate": case["replicate"],
        }
        for name, policy in (("baseline", baseline.run_search), ("candidate", candidate.run_search)):
            metrics, state, actions = run(policy, case, domain=False)
            if name == "candidate":
                route_audit = verify_route_decisions(state, actions)
                metrics["route_audit"] = route_audit
                metrics["all_checks_pass"] = metrics["all_checks_pass"] and route_audit["all_checks_pass"]
                metrics["joint_route_selection_counts"] = state["joint_route_selection_counts"]
                metrics["duplicate_supplement_skip_count"] = state["duplicate_supplement_skip_count"]
            row[name] = metrics
            save(OUT / "states" / f"{case['case_id']}__{name}.json", {"state": state, "actions": actions})
        row["candidate_to_baseline_ratio"] = row["candidate"]["time_s"] / row["baseline"]["time_s"]
        row["paired_delta_s_per_source"] = row["candidate"]["mean_s"] - row["baseline"]["mean_s"]
        rows.append(row)
        print(
            json.dumps(
                {
                    "case_id": row["case_id"],
                    "baseline": row["baseline"]["mean_s"],
                    "candidate": row["candidate"]["mean_s"],
                    "checks": row["baseline"]["all_checks_pass"] and row["candidate"]["all_checks_pass"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summaries = {name: summarize(rows, name) for name in ("baseline", "candidate")}
    saving = summaries["baseline"]["mean_s_per_source"] - summaries["candidate"]["mean_s_per_source"]
    relative = saving / summaries["baseline"]["mean_s_per_source"]
    worst = max(rows, key=lambda row: row["candidate_to_baseline_ratio"])
    best = min(rows, key=lambda row: row["candidate_to_baseline_ratio"])
    hard_checks = scope_checks["all_checks_pass"] and all(
        summary["all_checks_pass"] for summary in summaries.values()
    )
    accepted = (
        hard_checks
        and relative >= RELATIVE_MIN
        and saving >= ABSOLUTE_MIN_S_PER_SOURCE
        and worst["candidate_to_baseline_ratio"] <= WORST_RATIO_MAX
    )
    selection_counts = Counter()
    for row in rows:
        selection_counts.update(row["candidate"]["joint_route_selection_counts"])
    result = {
        "scope": "ONE_SHOT_PAIRED_OFFLINE_STRUCTURAL_DEVELOPMENT_ONLY_NOT_V1_HOLDOUT",
        "artifact_type": "q4_service_multistart_v2_development_comparison",
        "case_count": len(cases),
        "seed_range": [63000, 63041],
        "all_hard_checks_pass": hard_checks,
        "accepted": bool(accepted),
        "predeclared_gate": precommit["predeclared_gate"],
        "absolute_saving_s_per_source": saving,
        "relative_improvement": relative,
        "worst_case": {"case_id": worst["case_id"], "ratio": worst["candidate_to_baseline_ratio"]},
        "best_case": {"case_id": best["case_id"], "ratio": best["candidate_to_baseline_ratio"]},
        "route_decision_count": sum(row["candidate"]["route_audit"]["decision_count"] for row in rows),
        "route_selection_counts": {
            name: selection_counts[name] for name in ("anchor", "v1_service", "multistart_service")
        },
        "duplicate_supplement_skip_count": sum(
            row["candidate"]["duplicate_supplement_skip_count"] for row in rows
        ),
        "summaries": summaries,
        "rows": rows,
        "case_generation": generation,
        "static_scope_checks": scope_checks,
        "inputs": inputs,
    }
    save(COMPARISON, result)
    save(
        DECISION,
        {
            "accepted": bool(accepted),
            "human_verdict_recorded": False,
            "next_step": "stop_and_retain_complete_evidence_without_tuning_or_additional_data",
        },
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("rows", "inputs", "case_generation", "static_scope_checks")},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
