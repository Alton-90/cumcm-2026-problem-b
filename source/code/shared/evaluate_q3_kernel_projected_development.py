"""Frozen 36-case paired development screen for Q3 reception-kernel projection."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
import statistics

import numpy as np

from offline_arena import OfflineArena, Source
from simulator_client import SimulatorClient
import search_policy_q3_combined as baseline
import search_policy_q3_kernel_projected as candidate
from q3_reception_kernel import (
    FEASIBILITY_TOLERANCE_M,
    KERNEL_RADIUS_M,
    cloud_hash,
    convex_hull,
    maximum_distance,
    projected_kernel_target,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q3_kernel_projected" / "development"
STATES = OUT / "states"
SOURCE_CASES = ROOT / "results" / "feasible_q3" / "cases.json"
CASE_COUNT = 36
IMPROVEMENT_THRESHOLD = 0.03
WORST_RATIO_LIMIT = 1.10
TRUTH_ENVELOPE_M = 15.0


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrackedArena(OfflineArena):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_log = []
        self.measure_ordinal = 0

    def post(self, path, payload):
        before = self.position
        result = super().post(path, payload)
        if path in ("/measure", "/clear") and result.status == 200 and result.body.get("accepted") is True:
            position = payload["position"]
            point = (float(position["x"]), float(position["y"]))
            record = {
                "path": path,
                "channel": payload["channel"],
                "point": [point[0], point[1]],
                "movement_s": math.dist(before, point) / 5,
                "response": dict(result.body),
            }
            if path == "/measure":
                record["measure_ordinal"] = self.measure_ordinal
                self.measure_ordinal += 1
            self.action_log.append(record)
        return result


def initial_grid():
    axis = np.arange(-1800.0, 1800.1, 20.0)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    return points[np.linalg.norm(points, axis=1) <= 1815]


def update_cloud(cloud, observation):
    point = np.asarray(observation["point"], dtype=float)
    delta = cloud - point
    distance = np.linalg.norm(delta, axis=1)
    kind = observation["measure_result"]
    if kind == "no_signal":
        return cloud[distance >= 985]
    if kind == "near":
        return cloud[distance <= 20]
    if kind == "direction":
        svd = observation.get("svd_deg")
        if isinstance(svd, bool) or not isinstance(svd, (int, float)) or not math.isfinite(svd):
            return cloud
        angle = math.radians(svd)
        unit = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-unit[1], unit[0]])
        keep = (
            (distance <= 1515)
            & (delta @ unit >= -15)
            & (np.abs(delta @ normal) <= distance * math.sin(math.radians(1.005)) + 15)
        )
        return cloud[keep]
    return cloud


def same_point(first, second, tolerance=1e-7):
    return math.dist(first, second) <= tolerance


def audit_public_observations(state, sources, action_log):
    observations = state.get("public_observations", [])
    measure_actions = [action for action in action_log if action["path"] == "/measure"]
    errors = []
    if len(observations) != len(measure_actions):
        errors.append("public observation count differs from accepted measure count")
    needed = {
        event.get("observation_index")
        for event in state.get("kernel_projection_events", [])
        if isinstance(event.get("observation_index"), int)
    }
    snapshots = {}
    grid = initial_grid()
    clouds = {}
    source_by_channel = {source.channel: source for source in sources}
    truth_checks = 0
    minimum_truth_distance = math.inf
    for index, observation in enumerate(observations):
        if index < len(measure_actions):
            action = measure_actions[index]
            if (
                observation["channel"] != action["channel"]
                or not same_point(observation["point"], action["point"])
                or observation["measure_result"] != action["response"].get("measure_result")
                or observation.get("svd_deg") != action["response"].get("svd_deg")
            ):
                errors.append(f"observation {index}: public log differs from transport ledger")
        channel = observation["channel"]
        before = clouds.get(channel, grid)
        if index in needed:
            snapshots[index] = before.copy()
        after = update_cloud(before, observation)
        clouds[channel] = after
        if observation.get("cloud_count_after") != len(after):
            errors.append(f"observation {index}: cloud count mismatch")
        if observation.get("cloud_hash_after") != cloud_hash(after):
            errors.append(f"observation {index}: cloud hash mismatch")
        source = source_by_channel.get(channel)
        if source is not None:
            truth_checks += 1
            if len(after) == 0:
                errors.append(f"observation {index}: true channel cloud became empty")
            else:
                distance = float(
                    np.linalg.norm(after - np.array([source.x, source.y]), axis=1).min()
                )
                minimum_truth_distance = min(minimum_truth_distance, distance)
                if distance > TRUTH_ENVELOPE_M + 1e-9:
                    errors.append(
                        f"observation {index}: no grid candidate within 15m of true source"
                    )
    return {
        "all_checks_pass": not errors,
        "errors": errors,
        "observation_count": len(observations),
        "truth_survival_checks": truth_checks,
        "minimum_truth_grid_distance_m": None
        if minimum_truth_distance == math.inf
        else minimum_truth_distance,
        "snapshots": snapshots,
    }


def audit_projection_events(state, observation_audit):
    observations = state.get("public_observations", [])
    snapshots = observation_audit.pop("snapshots")
    errors = []
    per_channel_last = {}
    used_kernel = 0
    stationary = 0
    maximum_residual = -math.inf
    for index, event in enumerate(state.get("kernel_projection_events", [])):
        prefix = f"projection event {index}"
        observation_index = event.get("observation_index")
        if observation_index not in snapshots or not 0 <= observation_index < len(observations):
            errors.append(f"{prefix}: missing pre-action cloud snapshot")
            continue
        observation = observations[observation_index]
        channel = event.get("channel")
        if (
            observation["channel"] != channel
            or not same_point(observation["point"], event.get("target", (math.inf, math.inf)))
            or observation["measure_result"] != event.get("measure_result")
        ):
            errors.append(f"{prefix}: action does not match public observation")
        number = event.get("replan_measure_number")
        if not isinstance(number, int) or not 1 <= number <= candidate.MAX_REPLAN_BEARINGS:
            errors.append(f"{prefix}: replan measure number outside frozen budget")
        if number != per_channel_last.get(channel, 0) + 1:
            errors.append(f"{prefix}: per-channel replan measure sequence is not contiguous")
        per_channel_last[channel] = number
        cloud = snapshots[observation_index]
        if event.get("cloud_hash_before") != cloud_hash(cloud):
            errors.append(f"{prefix}: pre-action cloud hash mismatch")
        logged_hull = np.asarray(event.get("convex_hull_vertices", []), dtype=float)
        actual_hull = convex_hull(cloud)
        if logged_hull.shape != actual_hull.shape or not np.allclose(
            logged_hull, actual_hull, atol=0, rtol=0
        ):
            errors.append(f"{prefix}: convex-hull constraints mismatch")
        if event.get("used_kernel"):
            used_kernel += 1
            target = np.asarray(event["target"], dtype=float)
            maximum = maximum_distance(target, cloud)
            residual = maximum - KERNEL_RADIUS_M
            maximum_residual = max(maximum_residual, residual)
            if maximum > KERNEL_RADIUS_M + FEASIBILITY_TOLERANCE_M:
                errors.append(f"{prefix}: kernel constraint residual exceeds tolerance")
            if abs(maximum - event.get("max_cloud_distance_m", math.inf)) > 1e-8:
                errors.append(f"{prefix}: logged max-cloud distance mismatch")
            if abs(residual - event.get("kernel_residual_m", math.inf)) > 1e-8:
                errors.append(f"{prefix}: logged kernel residual mismatch")
            if event.get("useful") is not True:
                errors.append(f"{prefix}: kernel action failed ValueTracked.useful")
            baseline_m = event.get("baseline_m")
            if baseline_m is not None and baseline_m < candidate.MINIMUM_BASELINE_M - 1e-9:
                errors.append(f"{prefix}: kernel action violates 50m baseline")
            if event.get("measure_result") == "no_signal":
                errors.append(f"{prefix}: guaranteed-reception action returned no_signal")
            if observation.get("action_role") != "kernel_projection":
                errors.append(f"{prefix}: kernel action role mismatch")
            stationary += bool(event.get("stationary_projection"))
        elif observation.get("action_role") != "mean_anchor_replan":
            errors.append(f"{prefix}: mean fallback action role mismatch")
    if used_kernel != state.get("kernel_projection_count"):
        errors.append("kernel projection counter mismatch")
    if stationary != state.get("stationary_projection_count"):
        errors.append("stationary projection counter mismatch")
    if state.get("kernel_no_signal_count") != 0:
        errors.append("at least one kernel action returned no_signal")
    return {
        "all_checks_pass": not errors,
        "errors": errors,
        "event_count": len(state.get("kernel_projection_events", [])),
        "used_kernel_count": used_kernel,
        "stationary_projection_count": stationary,
        "maximum_kernel_residual_m": None
        if maximum_residual == -math.inf
        else maximum_residual,
        "per_channel_replan_measure_count": {
            str(channel): count for channel, count in sorted(per_channel_last.items())
        },
    }


def contains_forbidden_truth_key(value):
    forbidden = {"truth", "source_x", "source_y", "source_radius", "heading", "true_source"}
    if isinstance(value, dict):
        return any(
            str(key).lower() in forbidden or contains_forbidden_truth_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_forbidden_truth_key(item) for item in value)
    return False


def coverage_structure_pass(state):
    expected = [(0.0, 0.0)] + [
        (1124 * math.cos(index * math.pi / 3), 1124 * math.sin(index * math.pi / 3))
        for index in range(6)
    ]
    anchors = state.get("coverage_anchor_points")
    if anchors is not None and (
        len(anchors) != len(expected)
        or any(not same_point(observed, wanted) for observed, wanted in zip(anchors, expected))
    ):
        return False
    if state.get("coverage_points_total") != 7:
        return False
    for point in state.get("coverage_scan_points_visited", []):
        radius = math.hypot(*point)
        if radius <= 1e-7:
            continue
        if not 1124 - 1e-6 <= radius <= 1730 + 1e-6:
            return False
        angle = math.atan2(point[1], point[0]) % (2 * math.pi)
        if min(abs(angle - index * math.pi / 3) for index in range(7)) > 1e-7:
            return False
    return True


def action_cost_details(arena, state=None):
    actions = arena.action_log
    measure_actions = [action for action in actions if action["path"] == "/measure"]
    if state is not None and len(state.get("public_observations", [])) == len(measure_actions):
        roles = {
            index: observation["action_role"]
            for index, observation in enumerate(state["public_observations"])
        }
        coverage_ordinals = {
            action["measure_ordinal"]
            for action in measure_actions
            if roles[action["measure_ordinal"]] == "coverage_scan"
        }
    else:
        coverage_ordinals = set()
        index = 0
        while index < len(actions):
            if actions[index]["path"] != "/measure":
                index += 1
                continue
            end = index + 1
            while (
                end < len(actions)
                and actions[end]["path"] == "/measure"
                and same_point(actions[end]["point"], actions[index]["point"])
            ):
                end += 1
            if end - index >= 2:
                coverage_ordinals.update(
                    actions[position]["measure_ordinal"] for position in range(index, end)
                )
            index = end
    localization_movement = sum(
        action["movement_s"]
        for action in measure_actions
        if action["measure_ordinal"] not in coverage_ordinals
    )
    direct_clear_movement = sum(
        action["movement_s"] for action in actions if action["path"] == "/clear"
    )
    successful_clears = sum(
        action["path"] == "/clear" and action["response"].get("clear_result") == "success"
        for action in actions
    )
    failed_clears = sum(
        action["path"] == "/clear"
        and action["response"].get("clear_result") != "success"
        for action in actions
    )
    return {
        "localization_measurement_movement_s": localization_movement,
        "direct_clear_movement_s": direct_clear_movement,
        "successful_clear_count": successful_clears,
        "failed_clear_count": failed_clears,
    }


def run(policy, case, diagnose=False):
    sources = [Source(**item) for item in case["sources"]]
    arena = TrackedArena(sources, case["seed"], case["error_mode"])
    client = SimulatorClient("synthetic", transport=arena, log_path=False)
    client.enter()
    started = perf_counter()
    state = policy(client, question=3, method="main")
    wall_s = perf_counter() - started
    client.exit()
    movement_s = arena.distance_m / 5
    measure_s = 5 * arena.measure_count
    switch_s = arena.switch_count
    clears_s = 3 * arena.clear_count + 2 * len(arena.cleared)
    reconstructed = movement_s + measure_s + switch_s + clears_s
    groups = [set(state[name]) for name in ("cleared_channels", "excluded_channels", "unresolved_channels")]
    union = set().union(*groups)
    accounted = union == set(range(1, 21)) and sum(map(len, groups)) == len(union)
    false_excluded = bool(set(state["excluded_channels"]) & set(arena.sources))
    full_clear = (
        len(arena.cleared) == len(sources)
        and set(state["cleared_channels"]) == arena.cleared
        and not false_excluded
    )
    clock_ok = (
        abs(reconstructed - arena.virtual_time_s) < 1e-6
        and abs(state.get("virtual_time_s", math.inf) - arena.virtual_time_s) < 1e-6
    )
    counters_ok = state.get("measure_count") == arena.measure_count and state.get("clear_count") == arena.clear_count
    coverage_ok = coverage_structure_pass(state)
    if diagnose:
        observation_audit = audit_public_observations(state, sources, arena.action_log)
        projection_audit = audit_projection_events(state, observation_audit)
        public_only_log = not contains_forbidden_truth_key(state)
    else:
        observation_audit = {"all_checks_pass": True, "errors": [], "observation_count": 0, "truth_survival_checks": 0, "minimum_truth_grid_distance_m": None}
        projection_audit = {"all_checks_pass": True, "errors": [], "event_count": 0, "used_kernel_count": 0, "stationary_projection_count": 0, "maximum_kernel_residual_m": None, "per_channel_replan_measure_count": {}}
        public_only_log = True
    all_checks = bool(
        state.get("completed")
        and not state.get("stopped")
        and full_clear
        and accounted
        and clock_ok
        and counters_ok
        and coverage_ok
        and observation_audit["all_checks_pass"]
        and projection_audit["all_checks_pass"]
        and public_only_log
    )
    costs = action_cost_details(arena, state if diagnose else None)
    metrics = {
        "all_checks_pass": all_checks,
        "completed": bool(state.get("completed")),
        "stopped": bool(state.get("stopped")),
        "full_source_clear": full_clear,
        "channels_accounted": accounted,
        "false_excluded_source": false_excluded,
        "clock_accounting_pass": clock_ok,
        "action_counter_pass": counters_ok,
        "coverage_structure_pass": coverage_ok,
        "public_only_strategy_log_pass": public_only_log,
        "clock_residual_s": arena.virtual_time_s - reconstructed,
        "time_s": arena.virtual_time_s,
        "mean_s": arena.virtual_time_s / len(sources),
        "source_count": len(sources),
        "movement_s": movement_s,
        "measure_s": measure_s,
        "switch_s": switch_s,
        "clears_s": clears_s,
        "measure_count": arena.measure_count,
        "clear_attempt_count": arena.clear_count,
        "fallback_attempts": state.get("fallback_clear_count", 0),
        "stationary_projection_count": state.get("stationary_projection_count", 0),
        "mean_fallback_count": state.get("mean_fallback_count", 0),
        "kernel_projection_count": state.get("kernel_projection_count", 0),
        "per_channel_localization_actions": state.get("per_channel_replan_measure_count", {}),
        "policy_wall_s": wall_s,
        "observation_audit": observation_audit,
        "projection_audit": projection_audit,
        **costs,
    }
    return metrics, state


def geometry_self_tests():
    tests = []

    def add(name, passed, details):
        tests.append({"name": name, "passed": bool(passed), "details": details})

    cloud = np.array([[-200.0, -200.0], [200.0, -200.0], [200.0, 200.0], [-200.0, 200.0]])
    segment = projected_kernel_target(cloud, (-1500.0, 0.0), (1500.0, 0.0))
    add(
        "route_segment_intersects_kernel",
        segment["status"] == "ok"
        and segment["insertion_cost_m"] <= 1e-9
        and segment["max_cloud_distance_m"] <= KERNEL_RADIUS_M + FEASIBILITY_TOLERANCE_M,
        segment,
    )
    empty = projected_kernel_target(np.array([[-1000.0, 0.0], [1000.0, 0.0]]), (0.0, 0.0), (1.0, 0.0))
    add(
        "empty_kernel_falls_back",
        empty["status"] == "empty_or_unresolved_kernel" and empty["point"] is None,
        empty,
    )
    interior_cloud = np.array([[0.0, 0.0], [100.0, 0.0], [40.0, 20.0]])
    hull = convex_hull(interior_cloud)
    test_point = (400.0, 300.0)
    add(
        "convex_hull_constraints_cover_cloud",
        abs(maximum_distance(test_point, hull) - maximum_distance(test_point, interior_cloud)) <= 1e-12,
        {
            "hull": hull.tolist(),
            "max_hull_distance_m": maximum_distance(test_point, hull),
            "max_cloud_distance_m": maximum_distance(test_point, interior_cloud),
        },
    )
    return {"all_checks_pass": all(test["passed"] for test in tests), "tests": tests}


def summarize(rows, method):
    values = [row[method] for row in rows]
    return {
        "case_count": len(values),
        "all_checks_pass": all(value["all_checks_pass"] for value in values),
        "mean_per_source_s": statistics.mean(value["mean_s"] for value in values),
        "mean_movement_s": statistics.mean(value["movement_s"] for value in values),
        "mean_measure_s": statistics.mean(value["measure_s"] for value in values),
        "mean_switch_s": statistics.mean(value["switch_s"] for value in values),
        "mean_clears_s": statistics.mean(value["clears_s"] for value in values),
        "mean_localization_measurement_movement_s": statistics.mean(
            value["localization_measurement_movement_s"] for value in values
        ),
        "mean_direct_clear_movement_s": statistics.mean(
            value["direct_clear_movement_s"] for value in values
        ),
        "total_successful_clears": sum(value["successful_clear_count"] for value in values),
        "total_failed_clears": sum(value["failed_clear_count"] for value in values),
        "total_fallback_attempts": sum(value["fallback_attempts"] for value in values),
        "total_stationary_projections": sum(value["stationary_projection_count"] for value in values),
        "total_mean_fallbacks": sum(value["mean_fallback_count"] for value in values),
        "total_kernel_projections": sum(value["kernel_projection_count"] for value in values),
        "total_policy_wall_s": sum(value["policy_wall_s"] for value in values),
        "max_policy_wall_s": max(value["policy_wall_s"] for value in values),
    }


def freeze_inputs(cases):
    if len(cases) != CASE_COUNT or any(
        case.get("question") != 3 or case.get("split") != "development" for case in cases
    ):
        raise ValueError("expected exactly the frozen 36 Q3 development cases")
    copied = OUT / "cases.json"
    if copied.exists() and json.loads(copied.read_text(encoding="utf-8")) != cases:
        raise ValueError("existing case copy differs from frozen Q3 development cases")
    if not copied.exists():
        save(copied, cases)
    files = [
        "code/shared/search_policy.py",
        "code/shared/search_policy_compact.py",
        "code/shared/search_policy_feasible_q3.py",
        "code/shared/search_policy_q3_combined.py",
        "code/shared/route_multistart.py",
        "code/shared/q3_reception_kernel.py",
        "code/shared/search_policy_q3_kernel_projected.py",
        "code/shared/offline_arena.py",
        "code/shared/simulator_client.py",
        "code/shared/evaluate_q3_kernel_projected_development.py",
        "results/feasible_q3/cases.json",
        str(copied.relative_to(ROOT)).replace("\\", "/"),
    ]
    inputs = [{"path": path, "sha256": sha256(ROOT / path)} for path in files]
    save(OUT / "inputs_before_run.json", inputs)
    return inputs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    STATES.mkdir(parents=True, exist_ok=True)
    self_tests = geometry_self_tests()
    save(OUT / "geometry_self_test.json", self_tests)
    if not self_tests["all_checks_pass"]:
        raise RuntimeError("kernel geometry self-tests failed; paired suite was not started")
    cases = json.loads(SOURCE_CASES.read_text(encoding="utf-8"))[:CASE_COUNT]
    inputs = freeze_inputs(cases)
    rows = []
    suite_started = perf_counter()
    for index, case in enumerate(cases, start=1):
        baseline_metrics, baseline_state = run(baseline.run_search, case)
        save(STATES / f"{case['case_id']}__baseline.json", {"case_id": case["case_id"], "metrics": baseline_metrics, "state": baseline_state})
        candidate_metrics, candidate_state = run(candidate.run_search, case, diagnose=True)
        save(STATES / f"{case['case_id']}__candidate.json", {"case_id": case["case_id"], "metrics": candidate_metrics, "state": candidate_state})
        ratio = candidate_metrics["time_s"] / baseline_metrics["time_s"]
        row = {
            "case_id": case["case_id"],
            "seed": case["seed"],
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "candidate_to_baseline_ratio": ratio,
            "candidate_minus_baseline_s_per_source": candidate_metrics["mean_s"] - baseline_metrics["mean_s"],
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(cases)}] {case['case_id']} base={baseline_metrics['mean_s']:.6f} "
            f"candidate={candidate_metrics['mean_s']:.6f} ratio={ratio:.6f} "
            f"checks={candidate_metrics['all_checks_pass']} "
            f"kernel={candidate_metrics['kernel_projection_count']} mean_fallback={candidate_metrics['mean_fallback_count']}",
            flush=True,
        )
    summaries = {name: summarize(rows, name) for name in ("baseline", "candidate")}
    all_checks = self_tests["all_checks_pass"] and all(
        row[name]["all_checks_pass"] for row in rows for name in ("baseline", "candidate")
    )
    relative_improvement = 1 - summaries["candidate"]["mean_per_source_s"] / summaries["baseline"]["mean_per_source_s"]
    worst = max(rows, key=lambda row: row["candidate_to_baseline_ratio"])
    accepted = bool(
        all_checks
        and relative_improvement >= IMPROVEMENT_THRESHOLD
        and worst["candidate_to_baseline_ratio"] <= WORST_RATIO_LIMIT
    )
    output = {
        "scope": "SYNTHETIC_FIXED_36_DEVELOPMENT_ONLY",
        "artifact_type": "q3_guaranteed_reception_kernel_projected_screen",
        "case_count": len(rows),
        "baseline": "search_policy_q3_combined.run_search",
        "candidate": "search_policy_q3_kernel_projected.run_search",
        "all_hard_checks_pass": all_checks,
        "accepted": accepted,
        "stop_after_development": not accepted,
        "acceptance_rule": {
            "all_correctness_checks_pass": True,
            "minimum_mean_relative_improvement": IMPROVEMENT_THRESHOLD,
            "maximum_any_case_time_ratio": WORST_RATIO_LIMIT,
        },
        "relative_improvement": relative_improvement,
        "worst_case": {
            "case_id": worst["case_id"],
            "candidate_to_baseline_ratio": worst["candidate_to_baseline_ratio"],
            "candidate_minus_baseline_s_per_source": worst["candidate_minus_baseline_s_per_source"],
        },
        "suite_wall_s": perf_counter() - suite_started,
        "geometry_self_tests": self_tests,
        "summaries": summaries,
        "rows": rows,
        "inputs": inputs,
    }
    save(OUT / "comparison.json", output)
    save(
        OUT / "decision.json",
        {
            "accepted": accepted,
            "next_step": "await_independent_review_before_any_holdout"
            if accepted
            else "stop_without_tuning_or_60_case_or_simulator_run",
            "reason": "development_threshold_met" if accepted else "development_acceptance_rule_not_met",
        },
    )
    print(
        json.dumps(
            {
                "all_hard_checks_pass": all_checks,
                "accepted": accepted,
                "relative_improvement": relative_improvement,
                "worst_case": output["worst_case"],
                "summaries": summaries,
                "suite_wall_s": output["suite_wall_s"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
