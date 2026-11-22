"""Frozen Q4 ring22 route plus certified stopped measures on scan edges."""
from __future__ import annotations

import math

import search_policy_polygon as polygon_policy
from q4_edge_waypoint_geometry import (
    ROUTE_BOUND_MARGIN_M,
    node_key,
    route_connector_upper_bound,
    route_keys,
    select_waypoint,
    split_edge_certificate,
)
from search_policy import _add_bearing, _bearing_triangle, _box_circle, _resolve_clear
from search_policy_joint_route_q3 import _route
from search_policy_q4_batch import _TrackedClient, _localize, _state
from search_policy_q4_reuse import _supplement_from_current_pose
from search_policy_q4_ring22 import coverage_certificate, ring22_points


VERSION = "q4_edge_waypoint_v1"
DIRECT_RADIUS = 19.9
MAX_SOURCES = 16


def box(track):
    return _box_circle(track[2])


def stop(state):
    if len(state["cleared_channels"]) != MAX_SOURCES:
        return False
    state["excluded_channels"].update(set(range(1, 21)) - state["cleared_channels"])
    state["maximum_count_certificate"] = {
        "maximum_sources": 16,
        "cleared_distinct_channels": 16,
        "rule": "remaining channels excluded only after 16 successful distinct clears",
    }
    state["maximum_count_stop_used"] = True
    return True


def direct(client, channel, track, state):
    public_box = box(track)
    if public_box is None or public_box[1] > DIRECT_RADIUS:
        return False
    state["direct_clear_skip_measure_count"] += 1
    return _resolve_clear(
        client, public_box[0], channel, state, "19.9m public-polygon certificate"
    )


class PublicLedgerClient(_TrackedClient):
    """Record accepted public action responses without evaluator-side fields."""

    def __init__(self, client):
        super().__init__(client)
        self.histories = {channel: [] for channel in range(1, 21)}
        self.measured_points = {channel: [] for channel in range(1, 21)}
        self.action_count = 0

    def measure(self, x, y, channel):
        response = super().measure(x, y, channel)
        self.action_count += 1
        point = (float(x), float(y))
        self.measured_points[channel].append(point)
        kind = response.get("measure_result")
        if kind == "direction" and polygon_policy._valid_svd(response.get("svd_deg")):
            self.histories[channel].append({
                "point": list(point), "result": "direction",
                "bearing_deg": float(response["svd_deg"]),
            })
        elif kind in ("near", "no_signal"):
            self.histories[channel].append({"point": list(point), "result": kind})
        return response

    def clear(self, x, y, channel):
        response = super().clear(x, y, channel)
        self.action_count += 1
        return response

    def positive_points(self, channel):
        return [
            tuple(item["point"]) for item in self.histories[channel]
            if item["result"] in ("direction", "near")
        ]


def _nodes(points, remaining, tracks, cleared, attempted, position):
    result = [("scan", index, points[index]) for index in remaining]
    for channel, track in tracks.items():
        if channel in cleared or channel in attempted:
            continue
        public_box = box(track)
        if public_box:
            result.append(("source", channel, public_box[0]))
        elif math.dist(position, track[0]) >= 50.0:
            result.append(("source", channel, position))
    return result


def _radii(nodes, tracks):
    values = {}
    for node in nodes:
        if node[0] == "scan":
            values[node_key(node)] = 0.0
        else:
            public_box = box(tracks[node[1]])
            values[node_key(node)] = 1500.0 if public_box is None else public_box[1]
    return values


def _records(client, tracks, state, attempted, used_channels):
    records = []
    for channel, track in tracks.items():
        if channel in state["cleared_channels"] or channel in attempted or channel in used_channels:
            continue
        public_box = box(track)
        if public_box is None or public_box[1] <= DIRECT_RADIUS:
            continue
        records.append({
            "channel": channel,
            "center": public_box[0],
            "radius": public_box[1],
            "polygon": track[2],
            "positive_points": client.positive_points(channel),
            "measured_points": client.measured_points[channel],
            "history": client.histories[channel],
        })
    return records


def _continuation_decision(position, original_node, nodes, tracks):
    original_key = node_key(original_node)
    current_original = next((node for node in nodes if node_key(node) == original_key), None)
    if current_original is None:
        return {
            "allow_reorder": False,
            "reason": "original_scan_node_missing",
            "continue_route": [], "replan_route": [],
            "continue_bound_m": None, "replan_bound_m": None,
        }
    remaining = [node for node in nodes if node_key(node) != original_key]
    continue_route = [current_original] + _route(current_original[2], remaining)
    replan_route = _route(position, nodes)
    radii = _radii(nodes, tracks)
    continue_bound = route_connector_upper_bound(position, continue_route, radii)
    replan_bound = route_connector_upper_bound(position, replan_route, radii)
    first_differs = bool(replan_route and node_key(replan_route[0]) != original_key)
    strictly_lower = replan_bound < continue_bound - ROUTE_BOUND_MARGIN_M
    return {
        "allow_reorder": bool(first_differs and strictly_lower),
        "reason": "strict_bound_drop" if first_differs and strictly_lower
                  else "cache_update_and_continue_original_scan",
        "continue_route": route_keys(continue_route),
        "replan_route": route_keys(replan_route),
        "continue_bound_m": continue_bound,
        "replan_bound_m": replan_bound,
        "strict_margin_m": ROUTE_BOUND_MARGIN_M,
        "first_node_differs": first_differs,
        "strictly_lower": strictly_lower,
        "radius_by_node": {f"{key[0]}:{key[1]}": value for key, value in radii.items()},
        "nodes": [
            {"kind": node[0], "ident": node[1], "destination": list(node[2])}
            for node in nodes
        ],
    }


def _apply_waypoint_measure(client, selection, tracks, attempted, state):
    channel = int(selection["channel"])
    waypoint = tuple(selection["waypoint"])
    track_before = tracks[channel]
    response = client.measure(*waypoint, channel)
    state["measure_count"] += 1
    state["extra_bearing_count"] += 1
    state["edge_waypoint_measure_count"] += 1
    kind = response.get("measure_result")
    disposition = "retain_track"
    if kind == "near":
        state["edge_waypoint_near_count"] += 1
        if _resolve_clear(client, waypoint, channel, state, "near certified edge waypoint"):
            disposition = "cleared_near"
        else:
            disposition = "near_clear_failed_retain_track"
    elif kind == "direction" and polygon_policy._valid_svd(response.get("svd_deg")):
        state["edge_waypoint_direction_count"] += 1
        old_point, old_bearing, polygon = track_before
        clipped = _add_bearing(polygon, waypoint, response["svd_deg"])
        if clipped:
            tracks[channel] = (old_point, old_bearing, clipped)
            attempted.discard(channel)
            state["polygon_track_update_count"] += 1
            state["edge_waypoint_polygon_update_count"] += 1
            disposition = "public_direction_intersection_applied"
        else:
            disposition = "empty_intersection_ignored_retain_track"
    elif kind == "no_signal":
        state["edge_waypoint_no_signal_count"] += 1
        disposition = "no_signal_cached_only_retain_track_and_fallback"
    else:
        state["edge_waypoint_invalid_count"] += 1
        state["warnings"].append(
            f"channel {channel}: invalid certified edge-waypoint response retained"
        )
        disposition = "invalid_response_retain_track_and_fallback"
    track_after = tracks.get(channel)
    return response, disposition, {
        "track_equal_before_after": track_after == track_before,
        "polygon_before": [list(point) for point in track_before[2]],
        "polygon_after": None if track_after is None
        else [list(point) for point in track_after[2]],
    }


def run_search(client, question=4, method="main"):
    if question != 4 or method != "main":
        raise ValueError("Q4 edge-waypoint v1 supports only question=4, method='main'")
    points = ring22_points()
    state = _state(points)
    state.update({
        "coverage_geometry": coverage_certificate(),
        "route_strategy": "frozen_receding_open_2opt_plus_certified_scan_edge_waypoint",
        "polygon_track_update_count": 0,
        "reuse_measure_count": 0,
        "reuse_polygon_update_count": 0,
        "direct_clear_skip_measure_count": 0,
        "maximum_count_stop_used": False,
        "maximum_count_certificate": None,
        "edge_waypoint_opportunity_check_count": 0,
        "edge_waypoint_considered_candidate_count": 0,
        "edge_waypoint_eligible_candidate_count": 0,
        "edge_waypoint_measure_count": 0,
        "edge_waypoint_certified_count": 0,
        "edge_waypoint_direction_count": 0,
        "edge_waypoint_near_count": 0,
        "edge_waypoint_no_signal_count": 0,
        "edge_waypoint_invalid_count": 0,
        "edge_waypoint_polygon_update_count": 0,
        "edge_waypoint_forced_continue_count": 0,
        "edge_waypoint_reorder_count": 0,
        "edge_waypoint_events": [],
        "edge_waypoint_opportunity_checks": [],
        "localization_budget_events": [],
    })
    tracked = PublicLedgerClient(client)
    seen, tracks, attempted, used_waypoint_channels = set(), {}, set(), set()
    remaining = list(range(len(points)))
    try:
        terminate = False
        while remaining or any(
            channel not in state["cleared_channels"] and channel not in attempted
            for channel in tracks
        ):
            if stop(state):
                break
            nodes = _nodes(
                points, remaining, tracks, state["cleared_channels"], attempted,
                tracked.position
            )
            if not nodes:
                break
            kind, ident, destination = _route(tracked.position, nodes)[0]
            if kind == "scan":
                edge_start = tracked.position
                selection_snapshot = select_waypoint(
                    edge_start, destination,
                    _records(tracked, tracks, state, attempted, used_waypoint_channels),
                )
                state["edge_waypoint_opportunity_check_count"] += 1
                state["edge_waypoint_considered_candidate_count"] += selection_snapshot[
                    "considered_candidate_count"
                ]
                state["edge_waypoint_eligible_candidate_count"] += selection_snapshot[
                    "eligible_candidate_count"
                ]
                state["edge_waypoint_opportunity_checks"].append({
                    key: value for key, value in selection_snapshot.items()
                    if key != "eligible_candidates"
                })
                selection = selection_snapshot["selected"]
                if selection is not None:
                    channel = int(selection["channel"])
                    used_waypoint_channels.add(channel)
                    state["edge_waypoint_certified_count"] += 1
                    history_before_count = len(tracked.histories[channel])
                    action_count_before = tracked.action_count
                    response, disposition, track_transition = _apply_waypoint_measure(
                        tracked, selection, tracks, attempted, state
                    )
                    event = {
                        "event_index": len(state["edge_waypoint_events"]),
                        "public_action_count_before": action_count_before,
                        "public_history_count_before": history_before_count,
                        "original_scan_node": {
                            "kind": "scan", "ident": ident,
                            "destination": list(destination),
                        },
                        "selection_snapshot": selection_snapshot,
                        "selected": selection,
                        "split_certificate": split_edge_certificate(
                            edge_start, selection["waypoint"], destination
                        ),
                        "response": dict(response),
                        "response_disposition": disposition,
                        "track_transition": track_transition,
                        "track_retained_on_no_signal": response.get("measure_result") != "no_signal"
                        or track_transition["track_equal_before_after"],
                    }
                    if stop(state):
                        event["continuation"] = {
                            "allow_reorder": False,
                            "reason": "maximum_count_stop_after_waypoint",
                            "continuation_mode": "maximum_count_stop",
                        }
                        state["edge_waypoint_events"].append(event)
                        terminate = True
                        break
                    nodes_after = _nodes(
                        points, remaining, tracks, state["cleared_channels"], attempted,
                        tracked.position
                    )
                    continuation = _continuation_decision(
                        tracked.position, ("scan", ident, destination), nodes_after, tracks
                    )
                    if continuation["allow_reorder"]:
                        continuation["continuation_mode"] = "guarded_replan"
                        state["edge_waypoint_reorder_count"] += 1
                    else:
                        continuation["continuation_mode"] = "immediate_original_scan"
                        state["edge_waypoint_forced_continue_count"] += 1
                    event["continuation"] = continuation
                    state["edge_waypoint_events"].append(event)
                    if continuation["allow_reorder"]:
                        continue

                remaining.remove(ident)
                for channel in range(1, 21):
                    if stop(state):
                        break
                    if channel in state["cleared_channels"]:
                        continue
                    old_track = tracks.get(channel)
                    public_box = box(old_track) if old_track else None
                    if public_box and public_box[1] <= DIRECT_RADIUS:
                        continue
                    response = tracked.measure(*destination, channel)
                    state["measure_count"] += 1
                    state["coverage_measurements_by_channel"][channel] += 1
                    result = response.get("measure_result")
                    if result == "no_signal":
                        continue
                    seen.add(channel)
                    if result == "near":
                        _resolve_clear(
                            tracked, destination, channel, state, "near ring22 joint coverage"
                        )
                    elif result == "direction" and polygon_policy._valid_svd(
                        response.get("svd_deg")
                    ):
                        if old_track is None:
                            tracks[channel] = (
                                destination, response["svd_deg"],
                                _bearing_triangle(destination, response["svd_deg"]),
                            )
                        else:
                            old_point, old_bearing, polygon = old_track
                            clipped = _add_bearing(
                                polygon, destination, response["svd_deg"]
                            )
                            tracks[channel] = (
                                (old_point, old_bearing, clipped) if clipped else old_track
                            )
                        attempted.discard(channel)
                        state["polygon_track_update_count"] += 1
                    else:
                        state["warnings"].append(
                            f"channel {channel}: invalid public response"
                        )
                        state["unresolved_channels"].add(channel)
                state["batch_count"] += 1
            else:
                channel = ident
                track = tracks[channel]
                attempted.add(channel)
                if direct(tracked, channel, track, state):
                    continue
                track, cleared = _supplement_from_current_pose(
                    tracked, channel, track, state
                )
                if cleared:
                    continue
                tracks[channel] = track
                if direct(tracked, channel, track, state):
                    continue
                before = state["extra_bearing_count"]
                before_actions = tracked.action_count
                _localize(
                    tracked, track[0], track[1], channel, state, track[2]
                )
                state["localization_budget_events"].append({
                    "channel": channel,
                    "public_action_count_before": before_actions,
                    "extra_bearing_count_before": before,
                    "extra_bearing_count_after": state["extra_bearing_count"],
                    "local_extra_bearing_delta": state["extra_bearing_count"] - before,
                    "budget_limit": 8,
                })
        if terminate:
            pass
        if not state["maximum_count_stop_used"]:
            for channel in range(1, 21):
                if channel not in state["cleared_channels"]:
                    (state["unresolved_channels"] if channel in seen
                     else state["excluded_channels"]).add(channel)
    except Exception as error:
        state["stopped"] = True
        state["warnings"].append(
            f"protocol/transport stop: {type(error).__name__}: {error}"
        )
        state["unresolved_channels"].update(
            set(range(1, 21)) - state["cleared_channels"] - state["excluded_channels"]
        )
    state["edge_waypoint_used_channels"] = sorted(used_waypoint_channels)
    state["public_observation_history"] = {
        str(channel): list(history)
        for channel, history in tracked.histories.items() if history
    }
    for key in ("cleared_channels", "excluded_channels", "unresolved_channels"):
        state[key] = sorted(state[key])
    state.update(
        completed=(not state["stopped"] and not state["unresolved_channels"]
                   and len(state["cleared_channels"]) + len(state["excluded_channels"]) == 20),
        question=4,
        method="main",
        version=VERSION,
        virtual_time_s=tracked.virtual_time_s,
    )
    return state


__all__ = ["VERSION", "run_search"]
