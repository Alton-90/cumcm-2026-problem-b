"""Public-state residual transfer scorer for the exploratory Q3 v2 route.

Only consecutive source tasks at the head of the frozen proxy route may trade
places.  Order-independent source service costs are deliberately absent: the
score contains only motion to the actual public entry and predicted motion from
the service exit to the first task left in frozen order.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

import q3_cost_rollout_v2 as local_rollout
import search_policy as old


SOURCE_PREFIX_LIMIT = 4
MIN_RESIDUAL_SAVING_S = 5.0
TIE_TOLERANCE_S = 1e-9


def _point(value):
    return [float(value[0]), float(value[1])]


def _key(task):
    return f"{task['kind']}:{int(task['identifier'])}"


def _useful_without_side_effect(client, channel, point):
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


def _history_copy(history):
    return [
        {
            "action": str(event["action"]),
            "point": _point(event["point"]),
            "response": dict(event["response"]),
        }
        for event in history
    ]


def _source_public(client, channel, tracks):
    initial_point, initial_response, polygon = tracks[channel]
    polygon = [_point(vertex) for vertex in polygon]
    useful, useful_reason = _useful_without_side_effect(
        client, channel, client.position
    )
    public = {
        "channel": int(channel),
        "initial_point": _point(initial_point),
        "initial_svd_deg": float(initial_response["svd_deg"]),
        "polygon": polygon,
        "current_pose_useful": bool(useful),
        "current_pose_useful_reason": useful_reason,
    }
    if math.dist(client.position, initial_point) > 1.0 and useful:
        public.update(
            mode="current_pose_measure",
            deterministic_entry=_point(client.position),
            deterministic_exit=_point(client.position),
        )
        return public

    estimate = client.estimate(channel)
    polygon_box = old._box_circle(polygon)
    if estimate is not None:
        center, uncertainty = estimate
        if uncertainty <= 60.0:
            public.update(
                mode="candidate_clear",
                deterministic_entry=_point(center),
                deterministic_exit=_point(center),
                uncertainty_m=float(uncertainty),
            )
            return public
        # This precedence exactly matches the executing localizer: once a
        # public grid estimate exists, its radius replaces the polygon radius.
        rollout_center = center
        rollout_uncertainty = uncertainty
    else:
        if polygon_box is None:
            public.update(mode="unavailable", unavailable_reason="empty_polygon")
            return public
        rollout_center, rollout_uncertainty = polygon_box
        if rollout_uncertainty <= old.LOCATE_CLEAR_RADIUS_M:
            public.update(
                mode="polygon_clear",
                deterministic_entry=_point(rollout_center),
                deterministic_exit=_point(rollout_center),
                uncertainty_m=float(rollout_uncertainty),
            )
            return public

    cloud = client.clouds.get(channel)
    public.update(
        mode="rollout",
        center=_point(rollout_center),
        uncertainty_m=float(rollout_uncertainty),
        history=_history_copy(client.public_history.get(channel, [])),
        public_cloud=[] if cloud is None else [_point(item) for item in cloud],
    )
    return public


def make_snapshot(client, proxy_route, tracks):
    route = [
        {"kind": kind, "identifier": int(identifier), "point": _point(point)}
        for kind, identifier, point in proxy_route
    ]
    snapshot = {
        "position": _point(client.position),
        "current_channel": int(client.c.channel),
        "proxy_route": route,
        "source_prefix_limit": SOURCE_PREFIX_LIMIT,
    }
    if not route or route[0]["kind"] == "scan":
        snapshot["consecutive_source_count"] = 0
        return snapshot
    count = 0
    for task in route:
        if task["kind"] != "source" or count >= SOURCE_PREFIX_LIMIT:
            break
        task["public"] = _source_public(client, task["identifier"], tracks)
        count += 1
    snapshot["consecutive_source_count"] = count
    return snapshot


def _score_source(snapshot, rank):
    route = snapshot["proxy_route"]
    task = route[rank]
    public = task["public"]
    remaining = [item for index, item in enumerate(route) if index != rank]
    successor = remaining[0] if remaining else None
    mode = public["mode"]

    if mode in ("current_pose_measure", "candidate_clear", "polygon_clear"):
        entry = public["deterministic_entry"]
        exits = [public["deterministic_exit"]]
        hypotheses = 1
    elif mode == "rollout":
        proposal = local_rollout.choose_point(
            public["polygon"],
            public["center"],
            snapshot["position"],
            public["public_cloud"],
            public["history"],
            public["initial_point"],
            public["initial_svd_deg"],
            0,
            None,
        )
        entry = proposal["selected_point"]
        exits = [item["point"] for item in proposal["hypotheses"]]
        hypotheses = len(exits)
    else:
        return {
            "score_s": None,
            "mode": mode,
            "execution_entry_point": None,
            "unavailable_reason": public.get("unavailable_reason", "unavailable_mode"),
        }

    entry_move = math.dist(snapshot["position"], entry) / 5.0
    if successor is None:
        exit_transfer = 0.0
    elif exits:
        exit_transfer = float(
            np.mean([math.dist(exit_point, successor["point"]) / 5.0 for exit_point in exits])
        )
    else:
        return {
            "score_s": None,
            "mode": mode,
            "execution_entry_point": _point(entry),
            "successor_key": _key(successor),
            "unavailable_reason": "no_public_exit_hypotheses",
        }
    score = entry_move + exit_transfer
    return {
        "score_s": float(score),
        "mode": mode,
        "execution_entry_point": _point(entry),
        "successor_key": None if successor is None else _key(successor),
        "successor_point": None if successor is None else list(successor["point"]),
        "entry_move_s": float(entry_move),
        "predicted_exit_to_frozen_successor_s": float(exit_transfer),
        "exit_hypothesis_count": hypotheses,
        "omitted_order_independent_service_costs": [
            "measure",
            "channel_switch",
            "clear",
            "within_source_localization",
            "finite_fallback",
        ],
    }


def select_from_snapshot(snapshot: dict[str, Any]):
    route = snapshot["proxy_route"]
    if not route:
        raise ValueError("proxy route must not be empty")
    baseline_key = _key(route[0])
    source_count = int(snapshot["consecutive_source_count"])
    if route[0]["kind"] == "scan":
        return {
            "reason": "scan_first_locked",
            "baseline_key": baseline_key,
            "selected_key": baseline_key,
            "selected_rank": 0,
            "override": False,
            "predicted_gain_s": 0.0,
            "best_available_gain_s": None,
            "records": [],
        }
    if source_count < 1 or source_count > SOURCE_PREFIX_LIMIT:
        raise ValueError("invalid consecutive source prefix")

    records = []
    for rank in range(source_count):
        task = route[rank]
        components = _score_source(snapshot, rank)
        records.append(
            {
                "rank": rank,
                "key": _key(task),
                "kind": "source",
                "identifier": task["identifier"],
                "score_s": components.pop("score_s"),
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
            and gain + TIE_TOLERANCE_S >= MIN_RESIDUAL_SAVING_S
        )
    selected = best if override else baseline
    return {
        "reason": "source_prefix_scored" if source_count >= 2 else "single_source_retained",
        "baseline_key": baseline_key,
        "selected_key": selected["key"],
        "selected_rank": selected["rank"],
        "override": override,
        "predicted_gain_s": gain if override else 0.0,
        "best_available_gain_s": gain,
        "records": records,
    }


def choose_task(client, proxy_route, tracks):
    snapshot = make_snapshot(client, proxy_route, tracks)
    decision = select_from_snapshot(snapshot)
    return proxy_route[decision["selected_rank"]], {
        "snapshot": snapshot,
        "decision": decision,
    }


def self_test():
    def source(identifier, point):
        return {
            "kind": "source",
            "identifier": identifier,
            "point": list(point),
            "public": {
                "mode": "candidate_clear",
                "deterministic_entry": list(point),
                "deterministic_exit": list(point),
            },
        }

    locked = {
        "position": [0.0, 0.0],
        "current_channel": 1,
        "source_prefix_limit": SOURCE_PREFIX_LIMIT,
        "consecutive_source_count": 0,
        "proxy_route": [
            {"kind": "scan", "identifier": 0, "point": [5.0, 0.0]},
            {"kind": "source", "identifier": 2, "point": [1.0, 0.0]},
        ],
    }
    reordered = {
        "position": [0.0, 0.0],
        "current_channel": 1,
        "source_prefix_limit": SOURCE_PREFIX_LIMIT,
        "consecutive_source_count": 2,
        "proxy_route": [source(2, (100.0, 0.0)), source(3, (10.0, 0.0)),
                        {"kind": "scan", "identifier": 0, "point": [200.0, 0.0]}],
    }
    retained = {
        **reordered,
        "proxy_route": [source(2, (20.0, 0.0)), source(3, (10.0, 0.0)),
                        {"kind": "scan", "identifier": 0, "point": [200.0, 0.0]}],
    }
    lock_decision = select_from_snapshot(locked)
    reorder_decision = select_from_snapshot(reordered)
    retain_decision = select_from_snapshot(retained)
    component_keys = set(reorder_decision["records"][0]["components"])
    tests = [
        {
            "name": "scan_first_is_unconditionally_locked",
            "passed": not lock_decision["override"] and lock_decision["selected_rank"] == 0,
            "details": lock_decision,
        },
        {
            "name": "large_source_residual_gain_reorders_only_sources",
            "passed": reorder_decision["override"] and reorder_decision["selected_key"] == "source:3",
            "details": reorder_decision,
        },
        {
            "name": "sub_margin_source_gain_retains_frozen_first",
            "passed": not retain_decision["override"] and retain_decision["selected_key"] == "source:2",
            "details": retain_decision,
        },
        {
            "name": "score_contains_only_entry_and_exit_transfer",
            "passed": {
                "entry_move_s",
                "predicted_exit_to_frozen_successor_s",
            }.issubset(component_keys)
            and "predicted_local_service_s" not in component_keys
            and "planned_measure_s" not in component_keys,
            "details": sorted(component_keys),
        },
    ]
    return {"all_checks_pass": all(item["passed"] for item in tests), "tests": tests}
