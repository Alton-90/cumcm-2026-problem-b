"""Paired 36-case Q3 development screen for robust local parallax selection."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from time import perf_counter

from offline_arena import OfflineArena, Source
from simulator_client import SimulatorClient
import search_policy as old
import search_policy_q3_combined as baseline
import search_policy_q3_robust_parallax as candidate
from q3_robust_parallax_geometry import (
    ADOPTION_RATIO,
    BASE_HALF_ANGLE_DEG,
    BIN_HALF_WIDTH_DEG,
    BOUND_TIE_TOLERANCE_M,
    MOVEMENT_SLACK_M,
    NEAR_DIAMETER_BOUND_M,
    OFFSETS_M,
    RECEPTION_LIMIT_M,
    ROBUST_HALF_ANGLE_DEG,
    choose_measurement_point,
    clip_angle_bin,
    maximum_vertex_distance,
    polygon_diameter,
    robust_diameter_bound,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q3_robust_parallax" / "development"
STATES = OUT / "states"
SOURCE_CASES = ROOT / "results" / "feasible_q3" / "cases.json"
CASE_COUNT = 36
RELATIVE_THRESHOLD = 0.03
ABSOLUTE_THRESHOLD_S_PER_SOURCE = 5.0


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrackedArena(OfflineArena):
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


def same_point(first, second, tolerance=1e-7):
    return math.dist(first, second) <= tolerance


def circular_distance_deg(first, second):
    return abs((float(first) - float(second) + 180) % 360 - 180)


def contains_point(polygon, point, tolerance=1e-6):
    if not polygon:
        return False
    if len(polygon) == 1:
        return same_point(polygon[0], point, tolerance)
    area2 = sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[index][1] * polygon[(index + 1) % len(polygon)][0]
        for index in range(len(polygon))
    )
    orientation = 1 if area2 >= 0 else -1
    return all(
        orientation
        * (
            (polygon[(index + 1) % len(polygon)][0] - polygon[index][0])
            * (point[1] - polygon[index][1])
            - (polygon[(index + 1) % len(polygon)][1] - polygon[index][1])
            * (point[0] - polygon[index][0])
        )
        >= -tolerance
        for index in range(len(polygon))
    )


def geometry_self_tests():
    tests = []

    def add(name, passed, details):
        tests.append({"name": name, "passed": bool(passed), "details": details})

    endpoint_angles = [0.0, 0.5, 359.5, 359.99, 0.01, 180.5, 179.5]
    endpoint_coverage = {
        str(angle): min(circular_distance_deg(angle, center) for center in range(360))
        for angle in endpoint_angles
    }
    add(
        "integer_bins_cover_endpoints_and_wrap",
        all(distance <= BIN_HALF_WIDTH_DEG + 1e-12 for distance in endpoint_coverage.values()),
        endpoint_coverage,
    )

    square = [(-10.0, -10.0), (20.0, -10.0), (20.0, 10.0), (-10.0, 10.0)]
    wrap_polygon = clip_angle_bin(square, (0.0, 0.0), 0)
    inside_points = [
        (10 * math.cos(math.radians(angle)), 10 * math.sin(math.radians(angle)))
        for angle in (-1.5, 0.0, 1.5, 358.5)
    ]
    add(
        "wedge_clip_crosses_zero_360",
        all(contains_point(wrap_polygon, point) for point in inside_points),
        {"clipped_polygon": wrap_polygon, "inside_points": inside_points},
    )

    degenerate = [(1.0, 2.0), (1.0, 2.0)]
    degenerate_score = robust_diameter_bound(degenerate, (0.0, 0.0))
    add(
        "near_and_degenerate_polygon",
        polygon_diameter(degenerate) == 0
        and degenerate_score["upper_bound_m"] == NEAR_DIAMETER_BOUND_M,
        degenerate_score,
    )

    sample_polygon = [(0.0, -100.0), (900.0, -20.0), (900.0, 20.0), (0.0, 100.0)]
    selection = choose_measurement_point(sample_polygon, (450.0, 0.0), (0.0, 0.0), 0.0)
    candidate_checks = [
        record["maximum_vertex_distance_m"] <= RECEPTION_LIMIT_M + 1e-9
        for record in selection["candidates"]
        if record["offset_m"] != 0
        and record["movement_cap_pass"]
        and record["guaranteed_reception_pass"]
    ]
    add(
        "candidate_reception_vertex_certificate",
        bool(candidate_checks) and all(candidate_checks),
        selection,
    )
    return {"all_checks_pass": all(test["passed"] for test in tests), "tests": tests}


def audit_events(state, sources, actions):
    measurements = [action for action in actions if action["path"] == "/measure"]
    source_map = {source.channel: source for source in sources}
    errors = []
    selected = 0
    original = 0
    maximum_receive_distance = 0.0
    maximum_recomputed_bound_error = 0.0
    for index, event in enumerate(state.get("robust_parallax_events", [])):
        prefix = f"event {index}"
        measure_index = event.get("measure_index", 0) - 1
        if not 0 <= measure_index < len(measurements):
            errors.append(f"{prefix}: invalid measurement index")
            continue
        action = measurements[measure_index]
        if (
            action["channel"] != event["channel"]
            or not same_point(action["point"], event["selected_point"])
            or action["response"].get("measure_result") != event["measure_result"]
            or action["response"].get("svd_deg") != event["svd_deg"]
        ):
            errors.append(f"{prefix}: event differs from public action")
        if not 0 <= event.get("extra_used_before", -1) < old.MAX_EXTRA_BEARINGS:
            errors.append(f"{prefix}: local 8-measure budget index invalid")
        polygon = event["before_polygon"]
        source = source_map[event["channel"]]
        truth = (source.x, source.y)
        if not contains_point(polygon, truth, tolerance=1e-5):
            errors.append(f"{prefix}: true source absent before selection")
        recomputed = choose_measurement_point(
            polygon,
            tuple(event["original_center"]),
            tuple(event["current_position"]),
            event["initial_svd_deg"],
        )
        if (
            recomputed["used_offset"] != event["used_offset"]
            or not same_point(recomputed["selected_point"], event["selected_point"])
            or abs(recomputed["original_upper_bound_m"] - event["original_upper_bound_m"]) > 1e-8
        ):
            errors.append(f"{prefix}: selection is not reproducible from saved public polygon")
        bound_error = abs(
            recomputed["original_upper_bound_m"] - event["original_upper_bound_m"]
        )
        maximum_recomputed_bound_error = max(maximum_recomputed_bound_error, bound_error)
        if event["used_offset"]:
            selected += 1
            maximum = maximum_vertex_distance(event["selected_point"], polygon)
            maximum_receive_distance = max(maximum_receive_distance, maximum)
            if maximum > RECEPTION_LIMIT_M + 1e-8:
                errors.append(f"{prefix}: selected offset violates guaranteed reception")
            original_move = math.dist(event["current_position"], event["original_center"])
            selected_move = math.dist(event["current_position"], event["selected_point"])
            if selected_move > original_move + MOVEMENT_SLACK_M + 1e-8:
                errors.append(f"{prefix}: selected offset violates movement cap")
            if event["best_offset_upper_bound_m"] > ADOPTION_RATIO * event[
                "original_upper_bound_m"
            ] + 1e-8:
                errors.append(f"{prefix}: selected offset violates 0.8 adoption rule")
            if event["measure_result"] == "no_signal":
                errors.append(f"{prefix}: guaranteed-reception offset returned no_signal")
        else:
            original += 1
        if event["measure_result"] == "direction":
            returned = event["svd_deg"]
            bin_center = min(range(360), key=lambda center: circular_distance_deg(returned, center))
            if circular_distance_deg(returned, bin_center) > BIN_HALF_WIDTH_DEG + 1e-12:
                errors.append(f"{prefix}: returned direction not covered by integer bin")
            outer = clip_angle_bin(polygon, event["selected_point"], bin_center)
            if not contains_point(outer, truth, tolerance=1e-5):
                errors.append(f"{prefix}: actual direction truth not covered by robust bin wedge")
            actual_after = old._add_bearing(
                polygon, event["selected_point"], event["svd_deg"]
            )
            if not actual_after or not contains_point(actual_after, truth, tolerance=1e-5):
                errors.append(f"{prefix}: true source absent after actual bearing update")
        elif event["measure_result"] == "near":
            if math.dist(event["selected_point"], truth) > 5 + 1e-8:
                errors.append(f"{prefix}: near response inconsistent with truth")
    if selected != state.get("robust_parallax_selected_count"):
        errors.append("selected-event counter mismatch")
    if original != state.get("robust_parallax_original_center_count"):
        errors.append("original-center counter mismatch")
    if state.get("robust_parallax_guarantee_failure_count") != 0:
        errors.append("guaranteed-reception failure counter is nonzero")
    return {
        "all_checks_pass": not errors,
        "errors": errors,
        "event_count": len(state.get("robust_parallax_events", [])),
        "selected_offset_count": selected,
        "original_center_count": original,
        "maximum_selected_receive_distance_m": maximum_receive_distance,
        "maximum_recomputed_bound_error_m": maximum_recomputed_bound_error,
    }


def public_only_state(state):
    forbidden = {"truth", "true_source", "source_x", "source_y", "source_radius", "heading"}

    def visit(value):
        if isinstance(value, dict):
            return all(str(key).lower() not in forbidden and visit(item) for key, item in value.items())
        if isinstance(value, list):
            return all(visit(item) for item in value)
        return True

    return visit(state)


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
    successful_clear_s = 5 * len(arena.cleared)
    failed_clear_s = 3 * (arena.clear_count - len(arena.cleared))
    reconstructed = movement_s + measure_s + switch_s + successful_clear_s + failed_clear_s
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
    expected_anchors = [(0.0, 0.0)] + [
        (1124 * math.cos(index * math.pi / 3), 1124 * math.sin(index * math.pi / 3))
        for index in range(6)
    ]
    anchors = state.get("coverage_anchor_points")
    coverage_ok = state.get("coverage_points_total") == 7 and (
        anchors is None
        or (
            len(anchors) == len(expected_anchors)
            and all(same_point(point, expected) for point, expected in zip(anchors, expected_anchors))
        )
    )
    event_audit = audit_events(state, sources, arena.action_log) if diagnose else {
        "all_checks_pass": True,
        "errors": [],
        "event_count": 0,
        "selected_offset_count": 0,
        "original_center_count": 0,
        "maximum_selected_receive_distance_m": 0.0,
        "maximum_recomputed_bound_error_m": 0.0,
    }
    public_only = public_only_state(state)
    all_checks = bool(
        state.get("completed")
        and not state.get("stopped")
        and full_clear
        and accounted
        and clock_ok
        and counters_ok
        and coverage_ok
        and public_only
        and event_audit["all_checks_pass"]
    )
    fingerprint = hashlib.sha256(
        json.dumps(arena.action_log, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
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
        "public_only_strategy_state_pass": public_only,
        "clock_residual_s": arena.virtual_time_s - reconstructed,
        "source_count": len(sources),
        "time_s": arena.virtual_time_s,
        "mean_s": arena.virtual_time_s / len(sources),
        "movement_s": movement_s,
        "measure_s": measure_s,
        "switch_s": switch_s,
        "successful_clear_s": successful_clear_s,
        "failed_clear_s": failed_clear_s,
        "measure_count": arena.measure_count,
        "clear_attempt_count": arena.clear_count,
        "fallback_count": state.get("fallback_clear_count", 0),
        "selected_offset_count": state.get("robust_parallax_selected_count", 0),
        "original_center_count": state.get("robust_parallax_original_center_count", 0),
        "scoring_cpu_s": state.get("robust_parallax_scoring_cpu_s", 0.0),
        "policy_wall_s": wall_s,
        "event_audit": event_audit,
        "public_action_fingerprint": fingerprint,
    }
    return metrics, state, arena.action_log


def summarize(rows, method):
    values = [row[method] for row in rows]
    output = {
        "case_count": len(values),
        "all_checks_pass": all(value["all_checks_pass"] for value in values),
        "mean_s_per_source": statistics.mean(value["mean_s"] for value in values),
        "total_selected_offsets": sum(value["selected_offset_count"] for value in values),
        "total_original_centers": sum(value["original_center_count"] for value in values),
        "total_fallbacks": sum(value["fallback_count"] for value in values),
        "total_scoring_cpu_s": sum(value["scoring_cpu_s"] for value in values),
        "total_policy_wall_s": sum(value["policy_wall_s"] for value in values),
        "max_policy_wall_s": max(value["policy_wall_s"] for value in values),
    }
    for key in ("movement_s", "measure_s", "switch_s", "successful_clear_s", "failed_clear_s"):
        output[f"mean_{key}_per_source"] = statistics.mean(
            value[key] / value["source_count"] for value in values
        )
    return output


def freeze_inputs(cases):
    if len(cases) != CASE_COUNT or any(
        case.get("question") != 3 or case.get("split") != "development" for case in cases
    ):
        raise ValueError("expected the same frozen 36 Q3 development cases")
    copied = OUT / "cases.json"
    if copied.exists() and json.loads(copied.read_text(encoding="utf-8")) != cases:
        raise ValueError("existing robust-parallax case copy differs from frozen source")
    if not copied.exists():
        save(copied, cases)
    files = [
        "code/shared/search_policy.py",
        "code/shared/search_policy_compact.py",
        "code/shared/search_policy_feasible_q3.py",
        "code/shared/search_policy_q3_combined.py",
        "code/shared/route_multistart.py",
        "code/shared/q3_robust_parallax_geometry.py",
        "code/shared/search_policy_q3_robust_parallax.py",
        "code/shared/evaluate_q3_robust_parallax_development.py",
        "code/shared/offline_arena.py",
        "code/shared/simulator_client.py",
        "planning/q3_robust_parallax_experiment.md",
        "references/user_q2_20260911/思路初稿_第二检测点选择.md",
        "references/user_q2_20260911/B题第二问初稿_鲁棒选点.pdf",
        "references/user_q2_20260911/source_manifest.json",
        "results/feasible_q3/cases.json",
        str(copied.relative_to(ROOT)).replace("\\", "/"),
    ]
    inputs = [{"path": path, "sha256": sha256(ROOT / path)} for path in files]
    save(OUT / "inputs_before_run.json", inputs)
    return inputs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    STATES.mkdir(parents=True, exist_ok=True)
    tests = geometry_self_tests()
    save(OUT / "geometry_self_test.json", tests)
    if not tests["all_checks_pass"]:
        raise RuntimeError("robust-parallax geometry self-tests failed before paired run")
    cases = json.loads(SOURCE_CASES.read_text(encoding="utf-8"))[:CASE_COUNT]
    inputs = freeze_inputs(cases)
    rows = []
    suite_started = perf_counter()
    for index, case in enumerate(cases, start=1):
        baseline_metrics, baseline_state, baseline_actions = run(baseline.run_search, case)
        save(
            STATES / f"{case['case_id']}__baseline.json",
            {"case_id": case["case_id"], "metrics": baseline_metrics, "state": baseline_state, "actions": baseline_actions},
        )
        candidate_metrics, candidate_state, candidate_actions = run(candidate.run_search, case, True)
        save(
            STATES / f"{case['case_id']}__candidate.json",
            {"case_id": case["case_id"], "metrics": candidate_metrics, "state": candidate_state, "actions": candidate_actions},
        )
        row = {
            "case_id": case["case_id"],
            "seed": case["seed"],
            "error_mode": case["error_mode"],
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "candidate_to_baseline_ratio": candidate_metrics["time_s"] / baseline_metrics["time_s"],
            "candidate_minus_baseline_s_per_source": candidate_metrics["mean_s"] - baseline_metrics["mean_s"],
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(cases)}] {case['case_id']} base={baseline_metrics['mean_s']:.6f} "
            f"candidate={candidate_metrics['mean_s']:.6f} ratio={row['candidate_to_baseline_ratio']:.6f} "
            f"checks={candidate_metrics['all_checks_pass']} selected={candidate_metrics['selected_offset_count']}",
            flush=True,
        )
    summaries = {name: summarize(rows, name) for name in ("baseline", "candidate")}
    all_checks = tests["all_checks_pass"] and all(
        row[name]["all_checks_pass"] for row in rows for name in ("baseline", "candidate")
    )
    saving = summaries["baseline"]["mean_s_per_source"] - summaries["candidate"]["mean_s_per_source"]
    relative = saving / summaries["baseline"]["mean_s_per_source"]
    worst = max(rows, key=lambda row: row["candidate_to_baseline_ratio"])
    accepted = bool(
        all_checks
        and relative >= RELATIVE_THRESHOLD
        and saving >= ABSOLUTE_THRESHOLD_S_PER_SOURCE
    )
    output = {
        "schema_version": 2,
        "artifact_type": "q3_robust_parallax_paired_development_screen",
        "scope": "SYNTHETIC_FIXED_36_DEVELOPMENT_ONLY",
        "case_count": len(rows),
        "baseline": "search_policy_q3_combined.run_search",
        "candidate": "search_policy_q3_robust_parallax.run_search",
        "all_hard_checks_pass": all_checks,
        "accepted": accepted,
        "stop_after_development": not accepted,
        "predeclared_gate": {
            "minimum_relative_improvement": RELATIVE_THRESHOLD,
            "minimum_absolute_saving_s_per_source": ABSOLUTE_THRESHOLD_S_PER_SOURCE,
        },
        "absolute_saving_s_per_source": saving,
        "relative_improvement": relative,
        "worst_case": {
            "case_id": worst["case_id"],
            "candidate_to_baseline_ratio": worst["candidate_to_baseline_ratio"],
            "candidate_minus_baseline_s_per_source": worst["candidate_minus_baseline_s_per_source"],
        },
        "suite_wall_s": perf_counter() - suite_started,
        "parameters": {
            "offsets_m": list(OFFSETS_M),
            "reception_limit_m": RECEPTION_LIMIT_M,
            "movement_slack_m": MOVEMENT_SLACK_M,
            "base_half_angle_deg": BASE_HALF_ANGLE_DEG,
            "bin_half_width_deg": BIN_HALF_WIDTH_DEG,
            "robust_half_angle_deg": ROBUST_HALF_ANGLE_DEG,
            "near_diameter_bound_m": NEAR_DIAMETER_BOUND_M,
            "adoption_ratio": ADOPTION_RATIO,
            "bound_tie_tolerance_m": BOUND_TIE_TOLERANCE_M,
            "angle_bins": 360,
        },
        "geometry_self_tests": tests,
        "summaries": summaries,
        "rows": rows,
        "inputs": inputs,
    }
    save(OUT / "comparison.json", output)
    save(
        OUT / "decision.json",
        {
            "accepted": accepted,
            "next_step": "await_total_control_independent_review"
            if accepted
            else "stop_without_tuning_or_60_case_or_simulator_run",
            "reason": "development_gate_met" if accepted else "development_gate_not_met",
        },
    )
    print(
        json.dumps(
            {
                "all_hard_checks_pass": all_checks,
                "accepted": accepted,
                "absolute_saving_s_per_source": saving,
                "relative_improvement": relative,
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
