"""One-step service-aware task scoring for the experimental Q3 open route.

The scorer consumes only public policy state.  It treats a discovered source as
an open-route macro action: move to its entry, localize/clear, then leave from a
public source-position hypothesis toward the next task.  The actual controller
retains the frozen coverage and localization fallbacks.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

import q3_cost_rollout_v2 as local_rollout
import search_policy as old


PREFIX_TASKS = 4
OVERRIDE_MARGIN_S = 5.0
EXIT_ALLOWANCE_S = 4.0
TIE_TOLERANCE_S = 1e-9


def _point(value):
    return [float(value[0]), float(value[1])]


def _key(kind, identifier):
    return f"{kind}:{int(identifier)}"


def _nearest_continuation_s(exit_point, tasks):
    if not tasks:
        return 0.0
    return min(math.dist(exit_point, task["point"]) for task in tasks) / 5.0


def _useful_without_side_effect(client, channel, point):
    """Reproduce ValueTracked.useful without changing its skip counter."""
    if client.last_positions.get(channel) == tuple(point):
        return False, "same_point"
    estimate = client.estimate(channel)
    if estimate and estimate[1] <= 60.0:
        return False, "estimate_radius_at_most_60"
    cloud = client.clouds.get(channel)
    if cloud is None or len(cloud) == 0:
        return True, "no_public_cloud"
    delta = cloud - np.asarray(point, dtype=float)
    distance = np.linalg.norm(delta, axis=1)
    if float(distance.min()) > 1515.0:
        return False, "all_public_candidates_out_of_range"
    if float(distance.max()) < 985.0:
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        center = np.arctan2(delta[:, 1].mean(), delta[:, 0].mean())
        deviations = (angles - center + np.pi) % (2 * np.pi) - np.pi
        if float(deviations.max() - deviations.min()) < math.radians(2.0):
            return False, "public_bearing_already_determined"
    return True, "informative"


def _planned_scan_channels(client, tracks, cleared, point):
    planned = []
    reasons = []
    for channel in range(1, 21):
        if channel in cleared:
            reasons.append({"channel": channel, "decision": "skip_cleared"})
            continue
        if channel in tracks:
            useful, reason = _useful_without_side_effect(client, channel, point)
            if not useful:
                reasons.append({"channel": channel, "decision": "skip", "reason": reason})
                continue
            box = old._box_circle(tracks[channel][2])
            if box and box[1] <= old.LOCATE_CLEAR_RADIUS_M:
                reasons.append({"channel": channel, "decision": "skip_localized_polygon"})
                continue
        planned.append(channel)
        reasons.append({"channel": channel, "decision": "measure"})
    return planned, reasons


def _public_history(history):
    return [
        {
            "action": str(event["action"]),
            "point": _point(event["point"]),
            "response": dict(event["response"]),
        }
        for event in history
    ]


def make_snapshot(client, proxy_route, tracks, cleared_channels):
    """Freeze the compact public inputs needed to replay one route decision."""
    route = [
        {"kind": kind, "identifier": int(identifier), "point": _point(point)}
        for kind, identifier, point in proxy_route
    ]
    prefix_count = min(PREFIX_TASKS, len(route))
    for task in route[:prefix_count]:
        if task["kind"] == "scan":
            planned, reasons = _planned_scan_channels(
                client, tracks, cleared_channels, tuple(task["point"])
            )
            task["public"] = {
                "planned_measure_channels": planned,
                "channel_decisions": reasons,
            }
            continue

        channel = task["identifier"]
        initial_point, initial_response, polygon = tracks[channel]
        polygon = [_point(vertex) for vertex in polygon]
        useful, useful_reason = _useful_without_side_effect(client, channel, client.position)
        source = {
            "channel": channel,
            "initial_point": _point(initial_point),
            "initial_svd_deg": float(initial_response["svd_deg"]),
            "polygon": polygon,
            "current_pose_useful": bool(useful),
            "current_pose_useful_reason": useful_reason,
        }
        if math.dist(client.position, initial_point) > 1.0 and useful:
            source["mode"] = "current_pose_measure"
        else:
            polygon_box = old._box_circle(polygon)
            estimate = client.estimate(channel)
            if estimate is not None and estimate[1] <= 60.0:
                source.update(
                    mode="candidate_clear",
                    center=_point(estimate[0]),
                    uncertainty_m=float(estimate[1]),
                )
            elif polygon_box is not None and polygon_box[1] <= old.LOCATE_CLEAR_RADIUS_M:
                source.update(
                    mode="polygon_clear",
                    center=_point(polygon_box[0]),
                    uncertainty_m=float(polygon_box[1]),
                )
            else:
                center_radius = estimate or polygon_box
                history = _public_history(client.public_history.get(channel, []))
                cloud = client.clouds.get(channel)
                axis = local_rollout.longest_axis(polygon)
                hypotheses = local_rollout.compatible_hypotheses(
                    cloud, polygon, history, axis
                )
                source.update(
                    mode="rollout",
                    center=None if center_radius is None else _point(center_radius[0]),
                    uncertainty_m=None if center_radius is None else float(center_radius[1]),
                    history=history,
                    hypotheses=hypotheses,
                    public_cloud=[] if cloud is None else [_point(item) for item in cloud],
                )
        task["public"] = source
    return {
        "position": _point(client.position),
        "current_channel": int(client.c.channel),
        "prefix_tasks": prefix_count,
        "proxy_route": route,
    }


def _switch_cost(start_channel, channels):
    cost = 0.0
    current = start_channel
    for channel in channels:
        if channel != current:
            cost += 1.0
        current = channel
    return cost


def _score_scan(snapshot, task, remaining):
    position = snapshot["position"]
    point = task["point"]
    channels = task["public"]["planned_measure_channels"]
    travel = math.dist(position, point) / 5.0
    measures = 5.0 * len(channels)
    switches = _switch_cost(snapshot["current_channel"], channels)
    continuation = _nearest_continuation_s(point, remaining)
    score = travel + measures + switches + continuation
    return score, {
        "entry_move_s": travel,
        "planned_measure_s": measures,
        "planned_switch_s": switches,
        "predicted_local_service_s": 0.0,
        "predicted_exit_to_next_s": continuation,
        "exit_allowance_s": 0.0,
    }


def _score_source(snapshot, task, remaining):
    position = snapshot["position"]
    channel = task["identifier"]
    public = task["public"]
    mode = public["mode"]
    switch = 0.0 if channel == snapshot["current_channel"] else 1.0
    if mode == "current_pose_measure":
        continuation = _nearest_continuation_s(position, remaining)
        score = 5.0 + switch + continuation
        return score, {
            "mode": mode,
            "entry_move_s": 0.0,
            "planned_measure_s": 5.0,
            "planned_switch_s": switch,
            "predicted_local_service_s": 0.0,
            "predicted_exit_to_next_s": continuation,
            "exit_allowance_s": 0.0,
        }
    if mode in ("candidate_clear", "polygon_clear"):
        center = public["center"]
        travel = math.dist(position, center) / 5.0
        continuation = _nearest_continuation_s(center, remaining)
        score = travel + 5.0 + continuation + EXIT_ALLOWANCE_S
        return score, {
            "mode": mode,
            "entry_move_s": travel,
            "planned_measure_s": 0.0,
            "planned_switch_s": 0.0,
            "predicted_local_service_s": 5.0,
            "predicted_exit_to_next_s": continuation,
            "exit_allowance_s": EXIT_ALLOWANCE_S,
        }
    if mode != "rollout" or public.get("center") is None or not public.get("hypotheses"):
        return None, {"mode": mode, "unavailable_reason": "no_public_rollout_hypotheses"}

    cloud = public["public_cloud"]
    proposal = local_rollout.choose_point(
        public["polygon"],
        public["center"],
        position,
        cloud,
        public["history"],
        public["initial_point"],
        public["initial_svd_deg"],
        0,
        None,
    )
    first_target = proposal["selected_point"]
    chosen_record = next(
        (
            record
            for record in proposal["candidates"]
            if math.dist(record["point"], first_target) <= 1e-8
        ),
        None,
    )
    if chosen_record is None or chosen_record.get("score_s") is None:
        return None, {
            "mode": mode,
            "selected_entry_point": _point(first_target),
            "unavailable_reason": "entry_rollout_has_no_finite_completion",
        }
    local_maxima = [float(value) for value in chosen_record["scenario_maxima_s"]]
    if len(local_maxima) != len(public["hypotheses"]):
        return None, {
            "mode": mode,
            "selected_entry_point": _point(first_target),
            "unavailable_reason": "entry_rollout_hypothesis_count_mismatch",
        }
    per_position = []
    continuations = []
    for hypothesis, local_maximum in zip(public["hypotheses"], local_maxima):
        continuation = _nearest_continuation_s(hypothesis["point"], remaining)
        scenario = local_maximum + continuation + EXIT_ALLOWANCE_S + switch
        if not math.isfinite(scenario):
            return None, {
                "mode": mode,
                "selected_entry_point": _point(first_target),
                "unavailable_reason": "non_finite_local_completion",
            }
        continuations.append(float(continuation))
        per_position.append(float(scenario))
    score = float(np.mean(per_position))
    return score, {
        "mode": mode,
        "selected_entry_point": _point(first_target),
        "entry_selection_used_alternative": bool(proposal["used_alternative"]),
        "entry_original_score_s": proposal["original_score_s"],
        "entry_selected_score_s": proposal["selected_score_s"],
        "hypothesis_count": len(public["hypotheses"]),
        "scenario_local_maxima_s": local_maxima,
        "scenario_exit_to_next_s": continuations,
        "scenario_total_s": per_position,
        "planned_switch_s": switch,
        "exit_allowance_s": EXIT_ALLOWANCE_S,
        "predicted_local_service_s": float(np.mean(local_maxima)),
        "predicted_exit_to_next_s": float(np.mean(continuations)),
    }


def select_from_snapshot(snapshot: dict[str, Any]):
    """Return the deterministic task choice and replayable score records."""
    route = snapshot["proxy_route"]
    prefix_count = min(int(snapshot["prefix_tasks"]), len(route), PREFIX_TASKS)
    records = []
    for rank, task in enumerate(route[:prefix_count]):
        remaining = [other for index, other in enumerate(route) if index != rank]
        if task["kind"] == "scan":
            score, components = _score_scan(snapshot, task, remaining)
        elif task["kind"] == "source":
            score, components = _score_source(snapshot, task, remaining)
        else:
            score, components = None, {"unavailable_reason": "unknown_task_kind"}
        records.append(
            {
                "rank": rank,
                "key": _key(task["kind"], task["identifier"]),
                "kind": task["kind"],
                "identifier": task["identifier"],
                "point": list(task["point"]),
                "score_s": score,
                "components": components,
            }
        )

    baseline = records[0]
    scored = [record for record in records if record["score_s"] is not None]
    best = min(scored, key=lambda record: (record["score_s"], record["rank"])) if scored else baseline
    gain = None
    override = False
    if baseline["score_s"] is not None and best["score_s"] is not None:
        gain = float(baseline["score_s"] - best["score_s"])
        override = bool(
            best["rank"] != 0
            and gain + TIE_TOLERANCE_S >= OVERRIDE_MARGIN_S
        )
    selected = best if override else baseline
    return {
        "baseline_key": baseline["key"],
        "selected_key": selected["key"],
        "selected_rank": selected["rank"],
        "override": override,
        "predicted_gain_s": gain if override else 0.0,
        "best_available_gain_s": gain,
        "records": records,
    }


def choose_task(client, proxy_route, tracks, cleared_channels):
    snapshot = make_snapshot(client, proxy_route, tracks, cleared_channels)
    decision = select_from_snapshot(snapshot)
    selected = proxy_route[decision["selected_rank"]]
    return selected, {"snapshot": snapshot, "decision": decision}


def self_test():
    """Pure arithmetic checks; does not instantiate a simulator transport."""
    base = {
        "position": [0.0, 0.0],
        "current_channel": 1,
        "prefix_tasks": 2,
        "proxy_route": [
            {
                "kind": "scan",
                "identifier": 0,
                "point": [100.0, 0.0],
                "public": {"planned_measure_channels": [], "channel_decisions": []},
            },
            {
                "kind": "scan",
                "identifier": 1,
                "point": [10.0, 0.0],
                "public": {"planned_measure_channels": [], "channel_decisions": []},
            },
        ],
    }
    override = select_from_snapshot(base)
    close = dict(base)
    close["proxy_route"] = [dict(task) for task in base["proxy_route"]]
    close["proxy_route"][0] = dict(close["proxy_route"][0], point=[20.0, 0.0])
    retained = select_from_snapshot(close)
    tests = [
        {
            "name": "large_predicted_gain_overrides_within_prefix",
            "passed": override["override"] and override["selected_key"] == "scan:1",
            "details": override,
        },
        {
            "name": "sub_margin_gain_retains_frozen_first_task",
            "passed": not retained["override"] and retained["selected_key"] == "scan:0",
            "details": retained,
        },
    ]
    return {"all_checks_pass": all(test["passed"] for test in tests), "tests": tests}
