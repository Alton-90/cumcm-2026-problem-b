"""Paired 36-case development screen for conservative Q4 shadow recovery."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
from time import perf_counter
import statistics

from offline_arena import OfflineArena, Source
from search_policy import _add_bearing, _bearing_triangle, _box_circle
import search_policy_polygon as polygon_policy
import search_policy_q4_ring22_joint_route as baseline
import search_policy_q4_shadow_recovery as candidate
from shadow_cone_geometry import (
    _binary_float_interval,
    _hull,
    _point,
    _select_pair,
    contains_point,
    polygon_area,
    shadow_outer_polygon,
)
from simulator_client import SimulatorClient


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q4_shadow_recovery" / "development"
STATES = OUT / "states"
SOURCE_CASES = ROOT / "results" / "q4_reuse" / "development" / "cases.json"
REQUIRED_CASES = 36
RELATIVE_THRESHOLD = 0.05
ABSOLUTE_THRESHOLD_S_PER_SOURCE = 10.0
POINT_TOLERANCE_M = 1e-6


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_point(first, second, tolerance=POINT_TOLERANCE_M):
    return math.dist(first, second) <= tolerance


def append_unique(mapping, channel, point):
    point = (float(point[0]), float(point[1]))
    values = mapping.setdefault(channel, [])
    if not any(same_point(point, old) for old in values):
        values.append(point)


class TrackedArena(OfflineArena):
    """Truth-owning evaluator transport with an independent accepted-action ledger."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_log = []

    def post(self, path, payload):
        result = super().post(path, payload)
        if path in ("/measure", "/clear") and result.status == 200 and result.body.get("accepted") is True:
            position = payload["position"]
            self.action_log.append(
                {
                    "path": path,
                    "channel": payload["channel"],
                    "point": [float(position["x"]), float(position["y"])],
                    "response": dict(result.body),
                }
            )
        return result


def reconstruct_public_history(action_log):
    positives, negatives, measured = {}, {}, {}
    cleared = set()
    post_clear_measure_count = 0
    for action in action_log:
        channel = action["channel"]
        if action["path"] == "/clear":
            if action["response"].get("clear_result") == "success":
                cleared.add(channel)
            continue
        if channel in cleared:
            post_clear_measure_count += 1
            continue
        point = action["point"]
        append_unique(measured, channel, point)
        kind = action["response"].get("measure_result")
        if kind == "near" or (
            kind == "direction" and polygon_policy._valid_svd(action["response"].get("svd_deg"))
        ):
            append_unique(positives, channel, point)
        elif kind == "no_signal":
            append_unique(negatives, channel, point)
    return positives, negatives, measured, post_clear_measure_count


def normalized_history(mapping):
    return {
        str(channel): [[point[0], point[1]] for point in points]
        for channel, points in sorted(mapping.items())
    }


def compare_histories(state_history, positives, negatives, measured):
    expected_channels = sorted(set(positives) | set(negatives) | set(measured))
    expected = {
        str(channel): {
            "positive": normalized_history({channel: positives.get(channel, [])}).get(str(channel), []),
            "negative": normalized_history({channel: negatives.get(channel, [])}).get(str(channel), []),
            "measured": normalized_history({channel: measured.get(channel, [])}).get(str(channel), []),
        }
        for channel in expected_channels
    }
    return state_history == expected


def pair_matches(observed, expected):
    if observed is None or expected is None:
        return observed is None and expected is None
    return all(any(same_point(point, other) for other in expected) for point in observed) and all(
        any(same_point(point, other) for other in observed) for point in expected
    )


def validate_shadow_events(state, sources, action_log):
    source_by_channel = {source.channel: source for source in sources}
    positives, negatives, measured, post_clear_measure_count = reconstruct_public_history(action_log)
    history_capture_pass = compare_histories(
        state.get("shadow_public_history", {}), positives, negatives, measured
    )
    required = {
        "channel",
        "localization_id",
        "before_polygon",
        "after_polygon",
        "positive_pair",
        "negative",
        "old_binary_next",
        "new_center",
        "extra_used",
        "actual_exit_recovery",
        "actual_next_kind",
    }
    errors = []
    last_extra = {}
    applied = 0
    max_event_cpu = 0.0
    for index, event in enumerate(state.get("shadow_events", [])):
        prefix = f"event {index}"
        missing = sorted(required - set(event))
        if missing:
            errors.append(f"{prefix}: missing fields {missing}")
            continue
        channel = event["channel"]
        source = source_by_channel.get(channel)
        if source is None:
            errors.append(f"{prefix}: event on absent channel {channel}")
            continue
        before = event["before_polygon"]
        after = event["after_polygon"]
        candidate_polygon = event.get("candidate_polygon", after)
        truth = (source.x, source.y)
        if not contains_point(before, truth, tolerance=1e-5):
            errors.append(f"{prefix}: truth absent before pruning")
        if not contains_point(after, truth, tolerance=1e-5):
            errors.append(f"{prefix}: truth absent after actual transition")
        if event.get("geometry_status") == "updated" and not contains_point(
            candidate_polygon, truth, tolerance=1e-5
        ):
            errors.append(f"{prefix}: truth absent from updated candidate")
        area_before = polygon_area(before)
        area_after = polygon_area(after)
        if area_after > area_before + max(1e-6, area_before * 1e-10):
            errors.append(f"{prefix}: actual polygon expanded")
        negative = tuple(event["negative"])
        if not any(same_point(negative, point) for point in negatives.get(channel, [])):
            errors.append(f"{prefix}: negative is not in independently reconstructed history")
        snapshot = event.get("positive_snapshot", [])
        status, selection = _select_pair(snapshot, negative)
        expected_pair = None if selection is None else [selection[1], selection[2]]
        if event.get("positive_pair") is not None and not pair_matches(
            event["positive_pair"], expected_pair
        ):
            errors.append(f"{prefix}: selected pair is not the current max-angle pair")
        geometry_status = event.get("geometry_status")
        if geometry_status not in (None, status, "updated", "no_strict_shrink"):
            errors.append(f"{prefix}: geometry status disagrees with pair selection")
        key = (channel, event["localization_id"])
        extra_used = event["extra_used"]
        if not isinstance(extra_used, int) or not 1 <= extra_used <= candidate.MAX_EXTRA_BEARINGS:
            errors.append(f"{prefix}: extra_used outside 1..{candidate.MAX_EXTRA_BEARINGS}")
        if key in last_extra and extra_used < last_extra[key]:
            errors.append(f"{prefix}: extra budget reset within localization")
        last_extra[key] = extra_used
        max_event_cpu = max(max_event_cpu, float(event.get("geometry_cpu_s", 0.0)))
        if event["actual_exit_recovery"]:
            applied += 1
            if event.get("geometry_status") != "updated":
                errors.append(f"{prefix}: recovery exit without updated geometry")
            if not area_after < area_before - max(1e-6, area_before * 1e-10):
                errors.append(f"{prefix}: recovery exit without strict shrink")
            radius = event.get("candidate_radius")
            if radius is None:
                errors.append(f"{prefix}: recovery exit lacks candidate radius")
            elif radius <= candidate.DIRECT_RADIUS:
                if event["actual_next_kind"] != "direct_clear":
                    errors.append(f"{prefix}: clearable polygon not routed to direct clear")
            else:
                if event["actual_next_kind"] != "new_measure":
                    errors.append(f"{prefix}: non-clearable polygon not routed to a new measure")
                if event.get("new_center_previously_measured") is not False:
                    errors.append(f"{prefix}: next measured center was already measured")
                if same_point(event["new_center"], event["negative"]):
                    errors.append(f"{prefix}: next measured center repeats current negative")
                if same_point(event["new_center"], event["old_binary_next"]):
                    errors.append(f"{prefix}: next measured center equals old binary action")
        elif event["actual_next_kind"] != "old_binary":
            errors.append(f"{prefix}: unapplied event changed next action kind")
    if applied != state.get("shadow_applied_count") or applied != state.get("shadow_exit_recovery_count"):
        errors.append("applied-event counters disagree")
    if not history_capture_pass:
        errors.append("public history wrapper differs from independent action ledger")
    return {
        "all_event_checks_pass": not errors,
        "errors": errors,
        "event_count": len(state.get("shadow_events", [])),
        "applied_event_count": applied,
        "history_capture_pass": history_capture_pass,
        "post_clear_measure_count": post_clear_measure_count,
        "max_geometry_event_cpu_s": max_event_cpu,
    }


def run(policy, case, diagnose_shadow=False):
    sources = [Source(**item) for item in case["sources"]]
    arena = TrackedArena(sources, case["seed"], case["error_mode"])
    client = SimulatorClient("offline", transport=arena, log_path=False)
    client.enter()
    started = perf_counter()
    state = policy(client, question=4, method="main")
    wall_s = perf_counter() - started
    client.exit()
    cleared_count = len(arena.cleared)
    movement_s = arena.distance_m / 5
    measure_s = 5 * arena.measure_count
    switch_s = arena.switch_count
    clears_s = 3 * arena.clear_count + 2 * cleared_count
    reconstructed = movement_s + measure_s + switch_s + clears_s
    groups = [set(state[name]) for name in ("cleared_channels", "excluded_channels", "unresolved_channels")]
    union = set().union(*groups)
    channels_accounted = union == set(range(1, 21)) and sum(map(len, groups)) == len(union)
    false_excluded = bool(set(state["excluded_channels"]) & set(arena.sources))
    full_clear = (
        cleared_count == len(sources)
        and set(state["cleared_channels"]) == arena.cleared
        and not false_excluded
    )
    coverage_ok = (
        state.get("coverage_points_total") == 22
        and state.get("coverage_geometry") == baseline.coverage_certificate()
    )
    counter_ok = (
        state.get("measure_count") == arena.measure_count
        and state.get("clear_count") == arena.clear_count
    )
    clock_ok = (
        abs(reconstructed - arena.virtual_time_s) < 1e-6
        and abs(float(state.get("virtual_time_s", math.inf)) - arena.virtual_time_s) < 1e-6
    )
    event_checks = (
        validate_shadow_events(state, sources, arena.action_log)
        if diagnose_shadow
        else {
            "all_event_checks_pass": True,
            "errors": [],
            "event_count": 0,
            "applied_event_count": 0,
            "history_capture_pass": True,
            "post_clear_measure_count": 0,
            "max_geometry_event_cpu_s": 0.0,
        }
    )
    all_checks = bool(
        state.get("completed")
        and not state.get("stopped")
        and full_clear
        and channels_accounted
        and coverage_ok
        and counter_ok
        and clock_ok
        and event_checks["all_event_checks_pass"]
    )
    metrics = {
        "all_checks_pass": all_checks,
        "completed": bool(state.get("completed")),
        "stopped": bool(state.get("stopped")),
        "full_source_clear": full_clear,
        "channels_accounted": channels_accounted,
        "false_excluded_source": false_excluded,
        "coverage_certificate_pass": coverage_ok,
        "action_counter_pass": counter_ok,
        "clock_accounting_pass": clock_ok,
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
        "policy_wall_cpu_s": wall_s,
        "shadow_geometry_cpu_s": state.get("shadow_geometry_cpu_s", 0.0),
        "shadow_attempt_count": state.get("shadow_attempt_count", 0),
        "shadow_applied_count": state.get("shadow_applied_count", 0),
        "shadow_cpu_fallback_count": state.get("shadow_cpu_fallback_count", 0),
        "event_checks": event_checks,
    }
    return metrics, state


def geometry_self_tests():
    square = [(-1500.0, -1500.0), (1500.0, -1500.0), (1500.0, 1500.0), (-1500.0, 1500.0)]
    tests = []

    def record(name, passed, details):
        tests.append({"name": name, "passed": bool(passed), "details": details})

    directional = shadow_outer_polygon(square, [(100.0, -600.0), (100.0, 600.0)], (-100.0, 0.0))
    record(
        "directional_halfplane_no_signal",
        directional["status"] == "updated"
        and contains_point(directional["polygon"], (0.0, 0.0))
        and polygon_area(directional["polygon"]) < polygon_area(square),
        directional,
    )

    distance_positives = [(500.0, 300.0), (300.0, 500.0)]
    distance_negative = (1001.0, 1001.0)
    distance_loss = shadow_outer_polygon(square, distance_positives, distance_negative)
    distance_preconditions = all(math.dist(point, (0.0, 0.0)) <= 1000 for point in distance_positives) and math.dist(
        distance_negative, (0.0, 0.0)
    ) > 1000
    record(
        "radius_1000_distance_loss",
        distance_preconditions
        and distance_loss["status"] == "updated"
        and contains_point(distance_loss["polygon"], (0.0, 0.0)),
        {"preconditions": distance_preconditions, "result": distance_loss},
    )

    single = shadow_outer_polygon(square, [(0.0, 0.0)], (2.0, 0.0))
    parallel = shadow_outer_polygon(square, [(0.0, 0.0), (1.0, 0.0)], (2.0, 0.0))
    record(
        "single_and_parallel_no_trigger",
        single["status"] == "insufficient_positive"
        and parallel["status"] == "degenerate_positive_pair"
        and single["polygon"] == square
        and parallel["polygon"] == square,
        {"single": single, "parallel": parallel},
    )

    conflict = shadow_outer_polygon(square, [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)], (1.0, 0.5))
    line_conflict = shadow_outer_polygon(square, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], (1.5, 0.0))
    same_conflict = shadow_outer_polygon(square, [(0.0, 0.0), (2.0, 0.0)], (0.0, 0.0))
    record(
        "negative_in_positive_hull_conflicts",
        all(
            item["status"] == "negative_inside_positive_hull" and item["polygon"] == square
            for item in (conflict, line_conflict, same_conflict)
        ),
        {"full_hull": conflict, "line_hull": line_conflict, "same_point": same_conflict},
    )

    rationals = [Fraction(1, 10), Fraction(-1, 3), Fraction(2**53 + 1, 2**20)]
    interval_checks = []
    for value in rationals:
        low, high = _binary_float_interval(value)
        passed = Fraction.from_float(low) <= value <= Fraction.from_float(high)
        interval_checks.append(
            {"value": str(value), "low": low, "high": high, "contains": passed}
        )
    exact_polygon = [
        (Fraction(1, 10), Fraction(1, 7)),
        (Fraction(10, 3), Fraction(1, 11)),
        (Fraction(7, 2), Fraction(13, 5)),
    ]
    corner_points = []
    for x, y in exact_polygon:
        low_x, high_x = _binary_float_interval(x)
        low_y, high_y = _binary_float_interval(y)
        corner_points.extend(
            (Fraction.from_float(cx), Fraction.from_float(cy))
            for cx, cy in itertools.product((low_x, high_x), (low_y, high_y))
        )
    float_hull_exact = _hull(corner_points)
    exact_vertices_contained = all(
        all(
            (
                (float_hull_exact[(i + 1) % len(float_hull_exact)][0] - float_hull_exact[i][0])
                * (point[1] - float_hull_exact[i][1])
                - (float_hull_exact[(i + 1) % len(float_hull_exact)][1] - float_hull_exact[i][1])
                * (point[0] - float_hull_exact[i][0])
            )
            >= 0
            for i in range(len(float_hull_exact))
        )
        for point in exact_polygon
    )
    record(
        "fraction_to_binary_float_outer_envelope",
        all(item["contains"] for item in interval_checks) and exact_vertices_contained,
        {"intervals": interval_checks, "exact_vertices_contained": exact_vertices_contained},
    )

    return {"all_checks_pass": all(item["passed"] for item in tests), "tests": tests}


def summarize(rows, method):
    values = [row[method] for row in rows]
    return {
        "case_count": len(values),
        "mean_per_source_s": statistics.mean(item["mean_s"] for item in values),
        "mean_total_time_s": statistics.mean(item["time_s"] for item in values),
        "mean_movement_s": statistics.mean(item["movement_s"] for item in values),
        "mean_measure_s": statistics.mean(item["measure_s"] for item in values),
        "mean_switch_s": statistics.mean(item["switch_s"] for item in values),
        "mean_clears_s": statistics.mean(item["clears_s"] for item in values),
        "total_fallback_attempts": sum(item["fallback_attempts"] for item in values),
        "total_policy_wall_cpu_s": sum(item["policy_wall_cpu_s"] for item in values),
        "max_policy_wall_cpu_s": max(item["policy_wall_cpu_s"] for item in values),
        "all_checks_pass": all(item["all_checks_pass"] for item in values),
    }


def freeze_inputs(cases):
    if len(cases) != REQUIRED_CASES or any(case.get("split") != "development" for case in cases):
        raise ValueError("expected exactly 36 frozen development cases")
    copied = OUT / "cases.json"
    if copied.exists() and json.loads(copied.read_text(encoding="utf-8")) != cases:
        raise ValueError("existing shadow-recovery case copy differs from frozen source")
    if not copied.exists():
        save(copied, cases)
    files = [
        "code/shared/search_policy.py",
        "code/shared/search_policy_polygon.py",
        "code/shared/search_policy_q4_batch.py",
        "code/shared/search_policy_q4_reuse.py",
        "code/shared/search_policy_q4_ring22.py",
        "code/shared/search_policy_q4_ring22_joint_route.py",
        "code/shared/search_policy_joint_route_q3.py",
        "code/shared/shadow_cone_geometry.py",
        "code/shared/search_policy_q4_shadow_recovery.py",
        "code/shared/offline_arena.py",
        "code/shared/simulator_client.py",
        "code/shared/evaluate_q4_shadow_recovery_development.py",
        "results/optimization_200_400/ring22_continuous_certificate.json",
        "results/q4_reuse/development/cases.json",
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
        raise RuntimeError("geometry self-tests failed; 36-case run was not started")

    cases = json.loads(SOURCE_CASES.read_text(encoding="utf-8"))
    inputs = freeze_inputs(cases)
    rows = []
    suite_started = perf_counter()
    for index, case in enumerate(cases, start=1):
        base_metrics, base_state = run(baseline.run_search, case)
        save(
            STATES / f"{case['case_id']}__baseline.json",
            {"case_id": case["case_id"], "metrics": base_metrics, "state": base_state},
        )
        candidate_metrics, candidate_state = run(candidate.run_search, case, diagnose_shadow=True)
        save(
            STATES / f"{case['case_id']}__candidate.json",
            {"case_id": case["case_id"], "metrics": candidate_metrics, "state": candidate_state},
        )
        delta = candidate_metrics["mean_s"] - base_metrics["mean_s"]
        row = {
            "case_id": case["case_id"],
            "seed": case["seed"],
            "baseline": base_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_baseline_s_per_source": delta,
            "candidate_to_baseline_ratio": candidate_metrics["mean_s"] / base_metrics["mean_s"],
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(cases)}] {case['case_id']} "
            f"base={base_metrics['mean_s']:.6f} cand={candidate_metrics['mean_s']:.6f} "
            f"delta={delta:+.6f} checks={candidate_metrics['all_checks_pass']} "
            f"events={candidate_metrics['event_checks']['event_count']} "
            f"applied={candidate_metrics['event_checks']['applied_event_count']}",
            flush=True,
        )

    summaries = {name: summarize(rows, name) for name in ("baseline", "candidate")}
    checks = self_tests["all_checks_pass"] and all(
        row[name]["all_checks_pass"] for row in rows for name in ("baseline", "candidate")
    )
    baseline_mean = summaries["baseline"]["mean_per_source_s"]
    candidate_mean = summaries["candidate"]["mean_per_source_s"]
    absolute_saving = baseline_mean - candidate_mean
    relative_improvement = 1 - candidate_mean / baseline_mean
    worst = max(rows, key=lambda row: row["candidate_minus_baseline_s_per_source"])
    total_geometry_cpu = sum(row["candidate"]["shadow_geometry_cpu_s"] for row in rows)
    total_events = sum(row["candidate"]["event_checks"]["event_count"] for row in rows)
    total_applied = sum(row["candidate"]["event_checks"]["applied_event_count"] for row in rows)
    max_event_cpu = max(
        row["candidate"]["event_checks"]["max_geometry_event_cpu_s"] for row in rows
    )
    accepted = bool(
        checks
        and relative_improvement >= RELATIVE_THRESHOLD
        and absolute_saving >= ABSOLUTE_THRESHOLD_S_PER_SOURCE
    )
    output = {
        "scope": "SYNTHETIC_FIXED_36_DEVELOPMENT_ONLY",
        "artifact_type": "q4_shadow_recovery_paired_development_screen",
        "case_count": len(rows),
        "baseline": "search_policy_q4_ring22_joint_route.run_search",
        "candidate": "search_policy_q4_shadow_recovery.run_search",
        "all_hard_checks_pass": checks,
        "accepted": accepted,
        "stop_after_development": not accepted,
        "acceptance_rule": {
            "hard_checks": "zero baseline or candidate errors on all 36 paired cases",
            "minimum_relative_improvement": RELATIVE_THRESHOLD,
            "minimum_absolute_saving_s_per_source": ABSOLUTE_THRESHOLD_S_PER_SOURCE,
            "interpretation": "development screen only; not a prediction of official-simulator performance",
        },
        "relative_improvement": relative_improvement,
        "absolute_saving_s_per_source": absolute_saving,
        "worst_case_degradation": {
            "case_id": worst["case_id"],
            "candidate_minus_baseline_s_per_source": worst[
                "candidate_minus_baseline_s_per_source"
            ],
            "candidate_to_baseline_ratio": worst["candidate_to_baseline_ratio"],
        },
        "cpu": {
            "suite_wall_s": perf_counter() - suite_started,
            "total_shadow_geometry_cpu_s": total_geometry_cpu,
            "mean_shadow_geometry_cpu_s_per_event": total_geometry_cpu / total_events
            if total_events
            else 0.0,
            "max_shadow_geometry_event_cpu_s": max_event_cpu,
            "event_cpu_budget_s": candidate.GEOMETRY_EVENT_CPU_BUDGET_S,
            "channel_cpu_budget_s": candidate.GEOMETRY_CHANNEL_CPU_BUDGET_S,
        },
        "shadow_event_totals": {"attempted": total_events, "applied": total_applied},
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
            "next_step": "eligible_for_new_independently_randomized_60_case_holdout"
            if accepted
            else "stop_without_tuning_or_60_case_or_practice_run",
            "reason": "threshold_met" if accepted else "development_acceptance_rule_not_met",
        },
    )
    print(
        json.dumps(
            {
                "all_hard_checks_pass": checks,
                "accepted": accepted,
                "relative_improvement": relative_improvement,
                "absolute_saving_s_per_source": absolute_saving,
                "worst_case_degradation": output["worst_case_degradation"],
                "shadow_event_totals": output["shadow_event_totals"],
                "cpu": output["cpu"],
                "summaries": summaries,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
