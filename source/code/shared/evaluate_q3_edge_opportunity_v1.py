"""One-shot paired OfflineArena evaluation for Q3 edge-opportunity v1."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
from time import perf_counter
from datetime import datetime, timezone

from offline_arena import OfflineArena, Source
from simulator_client import SimulatorClient
import q3_edge_opportunity_v1 as scorer
import search_policy_q3_combined as baseline
import search_policy_q3_edge_opportunity_v1 as candidate


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q3_edge_opportunity_v1" / "development"
CASES = OUT / "cases.json"
FREEZE = OUT / "freeze_manifest.json"
COLLISION = OUT / "collision_audit.json"
CASE_COUNT = 42
BASELINE_SHA256 = "b70898f273ac0261e46fb2812d0380a218e0343364c915820c9ac04375ca41ab"
MIN_DISTANCE_REDUCTION = 0.05
MIN_TIME_REDUCTION = 0.02
MAX_WORST_TIME_RATIO = 1.10


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)+"\n",
        encoding="utf-8",
    )


class ActionArena(OfflineArena):
    """Evaluator-side recorder; never passed to a policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_log = []

    def post(self, path, payload):
        before_position = self.position
        before_channel = self.channel
        before_time = self.virtual_time_s
        result = super().post(path, payload)
        if (
            path in ("/measure", "/clear")
            and result.status == 200
            and result.body.get("accepted") is True
        ):
            position = payload["position"]
            point = (float(position["x"]), float(position["y"]))
            distance = math.dist(before_position, point)
            switched = path == "/measure" and payload["channel"] != before_channel
            self.action_log.append(
                {
                    "action_index": len(self.action_log),
                    "path": path,
                    "channel": payload["channel"],
                    "before_point": list(before_position),
                    "point": list(point),
                    "distance_m": distance,
                    "movement_s": distance/5.0,
                    "switched": bool(switched),
                    "action_time_increment_s": self.virtual_time_s-before_time,
                    "response": dict(result.body),
                }
            )
        return result


def same_point(first, second, tolerance=1e-7):
    return math.dist(first, second) <= tolerance


def public_only_state(state):
    forbidden = {
        "truth",
        "true_source",
        "source_x",
        "source_y",
        "source_radius",
        "heading",
        "arena",
        "transport",
    }

    def visit(value):
        if isinstance(value, dict):
            return all(
                str(key).lower() not in forbidden and visit(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return all(visit(item) for item in value)
        return True

    return visit(state)


def static_policy_audit():
    policy_files = [
        ROOT / "code/shared/q3_edge_opportunity_v1.py",
        ROOT / "code/shared/search_policy_q3_edge_opportunity_v1.py",
    ]
    forbidden_imports = {
        "offline_arena",
        "simulator_client",
        "urllib",
        "requests",
        "search_policy_q3_cost_rollout_strict84_selected",
    }
    forbidden_attributes = {
        "sources",
        "transport",
        "source_x",
        "source_y",
        "source_radius",
    }
    errors = []
    for path in policy_files:
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
    baseline_path = ROOT / "code/shared/search_policy_q3_combined.py"
    checks = {
        "no_forbidden_policy_import_or_attribute": not errors,
        "frozen_baseline_hash_exact": sha256(baseline_path) == BASELINE_SHA256,
        "seven_point_expression_retained": "range(6)" in (
            ROOT / "code/shared/search_policy_q3_edge_opportunity_v1.py"
        ).read_text(encoding="utf-8"),
        "strict84_wrapper_not_imported": all(
            "strict84" not in path.read_text(encoding="utf-8")
            for path in policy_files
        ),
    }
    return {
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "errors": errors,
        "policy_files": [
            str(path.relative_to(ROOT)).replace("\\", "/") for path in policy_files
        ],
        "strict84_cases_executed": 0,
        "live_or_gui_calls": 0,
    }


def verify_freeze():
    manifest = json.loads(FREEZE.read_text(encoding="utf-8"))
    errors = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        observed = sha256(path) if path.is_file() else None
        if observed != item["sha256"]:
            errors.append(
                {"path": item["path"], "expected": item["sha256"], "observed": observed}
            )
    return manifest, errors


def run_arm(policy, case):
    evaluator_sources = [Source(**row) for row in case["sources"]]
    arena = ActionArena(evaluator_sources, case["seed"], case["error_mode"])
    client = SimulatorClient("offline-edge-opportunity", transport=arena, log_path=False)
    client.enter()
    started = perf_counter()
    state = policy(client, question=3, method="main")
    policy_wall_s = perf_counter()-started
    client.exit()
    movement_s = arena.distance_m/5.0
    measure_s = 5.0*arena.measure_count
    switch_s = float(arena.switch_count)
    successful_clear_s = 5.0*len(arena.cleared)
    failed_clear_s = 3.0*(arena.clear_count-len(arena.cleared))
    reconstructed = movement_s+measure_s+switch_s+successful_clear_s+failed_clear_s
    groups = [
        set(state[name])
        for name in ("cleared_channels", "excluded_channels", "unresolved_channels")
    ]
    accounted = (
        set().union(*groups) == set(range(1, 21))
        and sum(map(len, groups)) == 20
    )
    false_excluded = bool(set(state["excluded_channels"]) & set(arena.sources))
    full_clear = (
        len(arena.cleared) == len(evaluator_sources)
        and set(state["cleared_channels"]) == arena.cleared
        and not false_excluded
    )
    clock_ok = (
        abs(reconstructed-arena.virtual_time_s) <= 1e-6
        and abs(state.get("virtual_time_s", math.inf)-arena.virtual_time_s) <= 1e-6
    )
    counters_ok = (
        state.get("measure_count") == arena.measure_count
        and state.get("clear_count") == arena.clear_count
    )
    base_checks = {
        "state_completed": bool(state.get("completed")),
        "state_not_stopped": not state.get("stopped"),
        "full_source_clear": full_clear,
        "all_channels_accounted": accounted,
        "clock_decomposition": clock_ok,
        "action_counters": counters_ok,
        "seven_coverage_points": state.get("coverage_points_total") == 7,
        "public_only_strategy_state": public_only_state(state),
    }
    metrics = {
        "all_checks_pass": all(base_checks.values()),
        "base_checks": base_checks,
        "source_count": len(evaluator_sources),
        "distance_m": arena.distance_m,
        "distance_m_per_source": arena.distance_m/len(evaluator_sources),
        "time_s": arena.virtual_time_s,
        "time_s_per_source": arena.virtual_time_s/len(evaluator_sources),
        "movement_s": movement_s,
        "measure_s": measure_s,
        "switch_s": switch_s,
        "successful_clear_s": successful_clear_s,
        "failed_clear_s": failed_clear_s,
        "measure_count": arena.measure_count,
        "switch_count": arena.switch_count,
        "clear_attempt_count": arena.clear_count,
        "successful_clear_count": len(arena.cleared),
        "failed_clear_count": arena.clear_count-len(arena.cleared),
        "clock_residual_s": arena.virtual_time_s-reconstructed,
        "policy_wall_s": policy_wall_s,
        "action_fingerprint": hashlib.sha256(
            json.dumps(arena.action_log, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return metrics, state, arena.action_log


def _selected_record(decision):
    for record in decision.get("candidate_records", []):
        if record.get("fraction") == decision.get("selected_fraction"):
            return record
    return None


def audit_opportunity_events(state, actions):
    errors = []
    inserted = 0
    hits = 0
    reorders = 0
    selected_edges = set()
    maximum_collinearity_residual = 0.0
    maximum_feasible_distance = 0.0
    events = state.get("edge_opportunity_events", [])
    for index, event in enumerate(events):
        prefix = f"event {index}"
        decision = event.get("decision", {})
        if decision.get("channel") is None or decision.get("public_cloud_count", 0) < 1:
            if decision.get("selected"):
                errors.append(prefix+": selected without tracked public feasible cloud")
        if not event.get("inserted"):
            if decision.get("selected"):
                errors.append(prefix+": selected decision has no inserted action")
            continue
        inserted += 1
        edge_key = tuple(event.get("edge_key", []))
        if edge_key in selected_edges:
            errors.append(prefix+": same geometric source edge probed more than once")
        selected_edges.add(edge_key)
        record = _selected_record(decision)
        if record is None or not record.get("eligible"):
            errors.append(prefix+": inserted action lacks eligible selected record")
            continue
        if not all(record.get("checks", {}).values()):
            errors.append(prefix+": not all fixed waypoint checks passed")
        residual = abs(record["collinearity_residual_m"])
        maximum_collinearity_residual = max(maximum_collinearity_residual, residual)
        maximum_feasible_distance = max(
            maximum_feasible_distance,
            record["maximum_public_feasible_distance_m"],
        )
        if residual > scorer.COLLINEARITY_TOLERANCE_M:
            errors.append(prefix+": triangle equality failed")
        if record["maximum_public_feasible_distance_m"] > scorer.VISIBILITY_LIMIT_M:
            errors.append(prefix+": public feasible-set visibility certificate failed")
        if record["predicted_net_m"] <= 0.0:
            errors.append(prefix+": non-positive 25/30m break-even")
        if record["predicted_worst_radius_shrink_m"] < scorer.MIN_WORST_CASE_RADIUS_SHRINK_M:
            errors.append(prefix+": insufficient fixed public scenario shrink")
        certificate = event.get("baseline_next_real_action_certificate", {})
        if (
            not certificate.get("certified")
            or certificate.get("path") != "/measure"
            or certificate.get("channel") != decision.get("channel")
            or not same_point(certificate.get("point"), decision.get("original_endpoint"))
            or certificate.get("estimate_radius_m", -math.inf) <= 60.0
        ):
            errors.append(prefix+": baseline next-real-action certificate failed")
        action_index = event.get("action_count_before")
        if not isinstance(action_index, int) or not 0 <= action_index < len(actions):
            errors.append(prefix+": inserted action index is invalid")
            continue
        action = actions[action_index]
        if (
            action["path"] != "/measure"
            or action["channel"] != decision["channel"]
            or not same_point(action["point"], decision["selected_point"])
            or action["response"].get("measure_result") != event.get("measure_result")
        ):
            errors.append(prefix+": saved event does not match public action replay")
        if action["response"].get("measure_result") == "no_signal":
            errors.append(prefix+": guaranteed-visible measurement returned no_signal")
        else:
            hits += 1
        reorder = event.get("reorder_decision")
        if event.get("reorder_applied"):
            reorders += 1
            if not state.get("edge_opportunity_allow_reorder") or not reorder or not reorder.get("reorder"):
                errors.append(prefix+": reorder applied without strict certificate")
        if reorder:
            arithmetic = (
                reorder["forced_upper_bound_m"]-reorder["free_upper_bound_m"]
            )
            if abs(arithmetic-reorder["strict_bound_decrease_m"]) > 1e-7:
                errors.append(prefix+": upper-bound arithmetic mismatch")
            if reorder.get("reorder") and (
                not reorder.get("different_first_task")
                or not reorder.get("strict_upper_bound_decrease")
                or arithmetic <= scorer.REORDER_BOUND_EPSILON_M
            ):
                errors.append(prefix+": reorder flag violates strict upper-bound rule")
        remaining_actual = sum(
            later["distance_m"] for later in actions[action_index+1:]
        )
        event["actual_distance_after_opportunity_action_m"] = remaining_actual
    if inserted != state.get("edge_opportunity_inserted_count"):
        errors.append("inserted opportunity counter mismatch")
    if hits != state.get("edge_opportunity_hit_count"):
        errors.append("opportunity hit counter mismatch")
    if reorders != state.get("edge_opportunity_reorder_count"):
        errors.append("opportunity reorder counter mismatch")
    if state.get("edge_opportunity_guarantee_failure_count"):
        errors.append("one or more guaranteed-visible probes failed")
    if not state.get("edge_opportunity_allow_reorder") and reorders:
        errors.append("cache-only arm applied a reorder")
    return {
        "all_checks_pass": not errors,
        "errors": errors,
        "opportunity_considered_count": len(events),
        "inserted_count": inserted,
        "hit_count": hits,
        "reorder_count": reorders,
        "maximum_collinearity_residual_m": maximum_collinearity_residual,
        "maximum_public_feasible_distance_m": maximum_feasible_distance,
    }


def first_divergence_audit(baseline_actions, arm_state, arm_actions):
    inserted_events = [
        event for event in arm_state.get("edge_opportunity_events", [])
        if event.get("inserted")
    ]
    if not inserted_events:
        return {
            "applicable": False,
            "all_checks_pass": True,
            "reason": "no_inserted_opportunity",
        }
    event = inserted_events[0]
    index = event["action_count_before"]

    def action_equal(first, second):
        return (
            first["path"] == second["path"]
            and first["channel"] == second["channel"]
            and same_point(first["point"], second["point"])
            and first["response"].get("measure_result")
            == second["response"].get("measure_result")
            and first["response"].get("clear_result")
            == second["response"].get("clear_result")
            and first["response"].get("svd_deg")
            == second["response"].get("svd_deg")
        )

    prefix_ok = (
        index <= len(baseline_actions)
        and index <= len(arm_actions)
        and all(
            action_equal(baseline_actions[item], arm_actions[item])
            for item in range(index)
        )
    )
    certificate = event["baseline_next_real_action_certificate"]
    baseline_next = baseline_actions[index] if index < len(baseline_actions) else None
    candidate_next = arm_actions[index] if index < len(arm_actions) else None
    baseline_next_ok = bool(
        baseline_next
        and baseline_next["path"] == certificate["path"]
        and baseline_next["channel"] == certificate["channel"]
        and same_point(baseline_next["point"], certificate["point"])
    )
    candidate_waypoint_ok = bool(
        candidate_next
        and candidate_next["path"] == "/measure"
        and candidate_next["channel"] == event["decision"]["channel"]
        and same_point(candidate_next["point"], event["decision"]["selected_point"])
    )
    record = _selected_record(event["decision"])
    triangle_ok = bool(
        record
        and abs(record["collinearity_residual_m"])
        <= scorer.COLLINEARITY_TOLERANCE_M
    )
    return {
        "applicable": True,
        "action_index": index,
        "identical_public_action_prefix": prefix_ok,
        "baseline_next_action_matches_certified_endpoint": baseline_next_ok,
        "candidate_next_action_is_collinear_waypoint": candidate_waypoint_ok,
        "zero_increment_triangle_certificate": triangle_ok,
        "baseline_next_action": baseline_next,
        "candidate_next_action": candidate_next,
        "all_checks_pass": prefix_ok and baseline_next_ok and candidate_waypoint_ok and triangle_ok,
    }


def summarize(rows, arm):
    values = [row[arm] for row in rows]
    summary = {
        "case_count": len(values),
        "all_checks_pass": all(item["all_checks_pass"] for item in values),
        "mean_distance_m_per_source": statistics.mean(
            item["distance_m_per_source"] for item in values
        ),
        "mean_time_s_per_source": statistics.mean(
            item["time_s_per_source"] for item in values
        ),
        "total_distance_m": sum(item["distance_m"] for item in values),
        "total_time_s": sum(item["time_s"] for item in values),
        "total_policy_wall_s": sum(item["policy_wall_s"] for item in values),
        "max_policy_wall_s": max(item["policy_wall_s"] for item in values),
    }
    for name in (
        "movement_s",
        "measure_s",
        "switch_s",
        "successful_clear_s",
        "failed_clear_s",
    ):
        summary[name+"_per_source"] = statistics.mean(
            item[name]/item["source_count"] for item in values
        )
    if arm != "baseline":
        summary.update(
            total_opportunities=sum(
                item["event_audit"]["opportunity_considered_count"] for item in values
            ),
            total_inserted=sum(item["event_audit"]["inserted_count"] for item in values),
            total_hits=sum(item["event_audit"]["hit_count"] for item in values),
            total_reorders=sum(item["event_audit"]["reorder_count"] for item in values),
        )
    return summary


def main():
    comparison_path = OUT / "comparison.json"
    if comparison_path.exists():
        raise RuntimeError("one-shot result exists; refusing rerun or overwrite")
    if not CASES.is_file() or not FREEZE.is_file() or not COLLISION.is_file():
        raise RuntimeError("freeze, collision audit, and generated cases are required")
    collision = json.loads(COLLISION.read_text(encoding="utf-8"))
    if not collision.get("all_checks_pass"):
        raise RuntimeError("collision audit did not pass")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    expected_seeds = list(range(64000, 64042))
    if (
        len(cases) != CASE_COUNT
        or [case.get("seed") for case in cases] != expected_seeds
        or len({case.get("case_id") for case in cases}) != CASE_COUNT
    ):
        raise RuntimeError("fixed 42-case contract mismatch")
    freeze_manifest, freeze_errors = verify_freeze()
    static_audit = static_policy_audit()
    scorer_test = scorer.self_test()
    preflight = {
        "freeze_errors": freeze_errors,
        "freeze_verified": not freeze_errors,
        "static_policy_audit": static_audit,
        "scorer_self_test": scorer_test,
        "case_contract_pass": True,
        "collision_audit_pass": True,
        "all_checks_pass": not freeze_errors
        and static_audit["all_checks_pass"]
        and scorer_test["all_checks_pass"],
    }
    save(OUT / "preflight.json", preflight)
    if not preflight["all_checks_pass"]:
        raise RuntimeError("preflight failed; fixed cases were not executed")
    input_paths = [
        item["path"] for item in freeze_manifest["files"]
    ]+[
        "results/q3_edge_opportunity_v1/development/cases.json",
        "results/q3_edge_opportunity_v1/development/collision_audit.json",
        "results/q3_edge_opportunity_v1/development/freeze_manifest.json",
    ]
    inputs_before = [
        {"path": path, "sha256": sha256(ROOT / path)} for path in input_paths
    ]
    save(OUT / "inputs_before_run.json", inputs_before)

    rows = []
    suite_started = perf_counter()
    arms = (
        ("baseline", baseline.run_search),
        ("cache_only", candidate.run_search_cache_only),
        ("candidate", candidate.run_search),
    )
    for case_index, case in enumerate(cases, start=1):
        row = {
            "case_id": case["case_id"],
            "seed": case["seed"],
            "error_mode": case["error_mode"],
            "source_count": len(case["sources"]),
        }
        payloads = {}
        for arm, policy in arms:
            metrics, state, actions = run_arm(policy, case)
            payloads[arm] = {"metrics": metrics, "state": state, "actions": actions}
            row[arm] = metrics
        for arm in ("cache_only", "candidate"):
            payload = payloads[arm]
            state = payload["state"]
            case_saving = row["baseline"]["distance_m"]-row[arm]["distance_m"]
            for event in state.get("edge_opportunity_events", []):
                event["paired_case_actual_distance_saved_vs_baseline_m"] = case_saving
                event["actual_saving_scope"] = "whole_paired_case_not_event_attributed"
                event["event_level_causal_attribution_claimed"] = False
            event_audit = audit_opportunity_events(state, payload["actions"])
            divergence = first_divergence_audit(
                payloads["baseline"]["actions"], state, payload["actions"]
            )
            payload["metrics"]["event_audit"] = event_audit
            payload["metrics"]["first_divergence_audit"] = divergence
            payload["metrics"]["all_checks_pass"] = bool(
                payload["metrics"]["all_checks_pass"]
                and event_audit["all_checks_pass"]
                and divergence["all_checks_pass"]
            )
        row["candidate_to_baseline_time_ratio"] = (
            row["candidate"]["time_s"]/row["baseline"]["time_s"]
        )
        row["candidate_distance_saved_m"] = (
            row["baseline"]["distance_m"]-row["candidate"]["distance_m"]
        )
        row["cache_only_distance_saved_m"] = (
            row["baseline"]["distance_m"]-row["cache_only"]["distance_m"]
        )
        row["reorder_incremental_distance_saved_m"] = (
            row["cache_only"]["distance_m"]-row["candidate"]["distance_m"]
        )
        for arm, payload in payloads.items():
            save(
                OUT / "states" / f"{case['case_id']}__{arm}.json",
                {"case_id": case["case_id"], **payload},
            )
        rows.append(row)
        save(
            OUT / "progress.json",
            {"completed_triplets": len(rows), "planned_triplets": CASE_COUNT, "rows": rows},
        )
        print(
            f"[{case_index:02d}/{CASE_COUNT}] {case['case_id']} "
            f"m/src {row['baseline']['distance_m_per_source']:.3f} -> "
            f"{row['candidate']['distance_m_per_source']:.3f}; "
            f"s/src {row['baseline']['time_s_per_source']:.3f} -> "
            f"{row['candidate']['time_s_per_source']:.3f}; "
            f"events={row['candidate']['event_audit']['inserted_count']} "
            f"checks={row['candidate']['all_checks_pass']}",
            flush=True,
        )

    summaries = {arm: summarize(rows, arm) for arm, _policy in arms}
    baseline_distance = summaries["baseline"]["mean_distance_m_per_source"]
    candidate_distance = summaries["candidate"]["mean_distance_m_per_source"]
    baseline_time = summaries["baseline"]["mean_time_s_per_source"]
    candidate_time = summaries["candidate"]["mean_time_s_per_source"]
    distance_reduction = (baseline_distance-candidate_distance)/baseline_distance
    time_reduction = (baseline_time-candidate_time)/baseline_time
    worst = max(rows, key=lambda row: row["candidate_to_baseline_time_ratio"])
    inputs_after = [
        {"path": item["path"], "sha256": sha256(ROOT / item["path"])}
        for item in inputs_before
    ]
    save(OUT / "inputs_after_run.json", inputs_after)
    inputs_unchanged = inputs_before == inputs_after
    all_hard = bool(
        inputs_unchanged
        and not freeze_errors
        and static_audit["all_checks_pass"]
        and scorer_test["all_checks_pass"]
        and len(rows) == CASE_COUNT
        and all(summaries[arm]["all_checks_pass"] for arm, _policy in arms)
    )
    gates = {
        "all_42_main_pairs_full_clear_and_hard_audits": all_hard,
        "mean_distance_per_source_reduction_at_least_5pct": distance_reduction
        >= MIN_DISTANCE_REDUCTION,
        "mean_time_per_source_reduction_at_least_2pct": time_reduction
        >= MIN_TIME_REDUCTION,
        "worst_case_total_time_ratio_at_most_1_10": worst[
            "candidate_to_baseline_time_ratio"
        ]
        <= MAX_WORST_TIME_RATIO,
    }
    gate_pass = all(gates.values())
    result = {
        "schema_version": 2,
        "artifact_type": "q3_edge_opportunity_v1_one_shot_paired_development",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "FRESH_FIXED_42_SYNTHETIC_DEVELOPMENT_ONLY",
        "independent_validation": False,
        "baseline": baseline.VERSION,
        "candidate": candidate.VERSION,
        "counterfactual_arm": candidate.VERSION+"_cache_only",
        "case_count": len(rows),
        "all_hard_checks_pass": all_hard,
        "inputs_unchanged": inputs_unchanged,
        "development_gate_pass": gate_pass,
        "method_accepted": False,
        "decision": (
            "development_gate_passed_await_new_independent_holdout"
            if gate_pass
            else "rejected_stop_without_tuning_seed_change_or_extra_trials"
        ),
        "gates": gates,
        "mean_distance_per_source_reduction_fraction": distance_reduction,
        "mean_time_per_source_reduction_fraction": time_reduction,
        "worst_case": {
            "case_id": worst["case_id"],
            "ratio": worst["candidate_to_baseline_time_ratio"],
            "candidate_distance_saved_m": worst["candidate_distance_saved_m"],
        },
        "summaries": summaries,
        "counterfactual_decomposition": {
            "cache_measurement_effect_mean_m_per_source": summaries["baseline"][
                "mean_distance_m_per_source"
            ]-summaries["cache_only"]["mean_distance_m_per_source"],
            "reorder_incremental_effect_mean_m_per_source": summaries["cache_only"][
                "mean_distance_m_per_source"
            ]-summaries["candidate"]["mean_distance_m_per_source"],
            "event_attribution_claimed": False,
        },
        "rows": rows,
        "static_policy_audit": static_audit,
        "scorer_self_test": scorer_test,
        "freeze_manifest_sha256": sha256(FREEZE),
        "cases_sha256": sha256(CASES),
        "collision_audit_sha256": sha256(COLLISION),
        "suite_wall_s": perf_counter()-suite_started,
        "online_practice_runs": 0,
        "formal_runs": 0,
        "strict84_confirmation_runs": 0,
    }
    save(comparison_path, result)
    save(
        OUT / "counterfactual_audit.json",
        {
            "schema_version": 1,
            "artifact_type": "q3_edge_opportunity_counterfactual_audit",
            "all_checks_pass": all(
                row[arm].get("first_divergence_audit", {}).get("all_checks_pass", True)
                for row in rows for arm in ("cache_only", "candidate")
            ),
            "contrast": "baseline_vs_cache_only_then_cache_only_vs_full_candidate",
            "decomposition": result["counterfactual_decomposition"],
            "case_level_only": True,
            "event_level_causal_attribution_claimed": False,
            "rows": [
                {
                    "case_id": row["case_id"],
                    "baseline_minus_cache_only_distance_m": row["cache_only_distance_saved_m"],
                    "cache_only_minus_candidate_distance_m": row[
                        "reorder_incremental_distance_saved_m"
                    ],
                    "cache_only_first_divergence": row["cache_only"][
                        "first_divergence_audit"
                    ],
                    "candidate_first_divergence": row["candidate"][
                        "first_divergence_audit"
                    ],
                }
                for row in rows
            ],
        },
    )
    save(
        OUT / "decision.json",
        {
            "development_gate_pass": gate_pass,
            "method_accepted": False,
            "reason": result["decision"],
            "next_step": (
                "new_independent_holdout_only"
                if gate_pass
                else "stop_no_tuning_no_more_cases_no_live_run"
            ),
        },
    )
    save(
        OUT / "run_summary.json",
        {
            "schema_version": 2,
            "artifact_type": "run-summary",
            "question": "Q3",
            "run_scope": result["scope"],
            "method": candidate.VERSION,
            "baseline": baseline.VERSION,
            "status": "completed",
            "independent_validation": False,
            "all_checks_pass": all_hard,
            "development_gate_pass": gate_pass,
            "mean_distance_m_per_source": candidate_distance,
            "baseline_mean_distance_m_per_source": baseline_distance,
            "distance_reduction_fraction": distance_reduction,
            "mean_time_s_per_source": candidate_time,
            "baseline_mean_time_s_per_source": baseline_time,
            "time_reduction_fraction": time_reduction,
            "worst_case_ratio": worst["candidate_to_baseline_time_ratio"],
            "comparison_artifact": "results/q3_edge_opportunity_v1/development/comparison.json",
        },
    )
    print(
        json.dumps(
            {
                "development_gate_pass": gate_pass,
                "all_hard_checks_pass": all_hard,
                "distance_reduction_fraction": distance_reduction,
                "time_reduction_fraction": time_reduction,
                "worst_case": result["worst_case"],
                "summaries": summaries,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
