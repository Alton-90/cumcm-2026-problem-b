"""Versioned Q4 service-aware outer ordering over the frozen ring22 chain.

Only the next outer task and a redundant current-pose first measurement may
change.  Detection coverage, the public bearing polygon, the eight-measurement
local budget, binary recovery, polygon fallback, and legacy 304-point fallback
remain the frozen implementations.
"""
from __future__ import annotations

import math

import search_policy_polygon as polygon_policy
from search_policy import _add_bearing, _bearing_triangle, _box_circle, _resolve_clear
from search_policy_joint_route_q3 import _route as _anchor_route
from search_policy_q4_batch import _TrackedClient, _localize, _state
from search_policy_q4_reuse import REUSE_MIN_BASELINE_M, _supplement_from_current_pose
from search_policy_q4_ring22 import coverage_certificate, ring22_points


VERSION = "q4_service_aware_route_v1"
DIRECT_RADIUS_M = 19.9
MAX_SOURCES = 16
ROBOT_SPEED_MPS = 5.0
MEASURE_WITH_SWITCH_PROXY_S = 6.0
SUCCESSFUL_CLEAR_PROXY_S = 5.0
MAX_LOCAL_MEASUREMENTS = 8
MIN_PROXY_SAVING_S = 6.0
POINT_TOLERANCE_M = 1e-7


def _point(value):
    return (float(value[0]), float(value[1]))


def _same_point(first, second):
    return first is not None and second is not None and math.dist(_point(first), _point(second)) <= POINT_TOLERANCE_M


def _node_key(node):
    return (str(node[0]), int(node[1]))


def _box(track):
    return _box_circle(track[2])


class ServiceTrackedClient(_TrackedClient):
    """Track only normal public measurement locations for duplicate avoidance."""

    def __init__(self, client):
        super().__init__(client)
        self.last_measure_point = {}

    def measure(self, x, y, channel):
        response = super().measure(x, y, channel)
        self.last_measure_point[int(channel)] = (float(x), float(y))
        return response


def _stop(state):
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


def _direct(client, channel, track, state):
    bound = _box(track)
    if bound is None or bound[1] > DIRECT_RADIUS_M:
        return False
    state["direct_clear_skip_measure_count"] += 1
    return _resolve_clear(client, bound[0], channel, state, "19.9m public-polygon certificate")


def _local_measurement_levels(radius_m):
    ratio = max(float(radius_m), DIRECT_RADIUS_M) / DIRECT_RADIUS_M
    return min(MAX_LOCAL_MEASUREMENTS, max(1, math.ceil(math.log2(ratio))))


def _service_profiles(nodes, tracks):
    profiles = {}
    for node in nodes:
        kind, ident, destination = node
        key = _node_key(node)
        if kind == "scan":
            profiles[key] = {
                "kind": "scan",
                "entry": _point(destination),
                "exit": _point(destination),
            }
            continue

        track = tracks[int(ident)]
        bound = _box(track)
        if bound is None:
            profiles[key] = {
                "kind": "source",
                "channel": int(ident),
                "entry": _point(destination),
                "exit": _point(destination),
                "initial_point": _point(track[0]),
                "radius_m": None,
                "direct": False,
                "local_measurement_levels": MAX_LOCAL_MEASUREMENTS,
                "internal_distance_m": 0.0,
                "local_action_s": MEASURE_WITH_SWITCH_PROXY_S * MAX_LOCAL_MEASUREMENTS + SUCCESSFUL_CLEAR_PROXY_S,
            }
            continue

        center, radius = _point(bound[0]), float(bound[1])
        direct = radius <= DIRECT_RADIUS_M
        levels = 0 if direct else _local_measurement_levels(radius)
        profiles[key] = {
            "kind": "source",
            "channel": int(ident),
            "entry": center,
            "exit": center,
            "initial_point": _point(track[0]),
            "radius_m": radius,
            "direct": direct,
            "local_measurement_levels": levels,
            "internal_distance_m": 0.0 if direct else math.dist(_point(track[0]), center),
            "local_action_s": SUCCESSFUL_CLEAR_PROXY_S if direct else MEASURE_WITH_SWITCH_PROXY_S * levels + SUCCESSFUL_CLEAR_PROXY_S,
        }
    return profiles


def _scannable_channels(state, tracks):
    channels = set(range(1, 21)) - set(state["cleared_channels"])
    for channel, track in tracks.items():
        bound = _box(track)
        if bound is not None and bound[1] <= DIRECT_RADIUS_M:
            channels.discard(channel)
    return channels


def _proxy_cost(position, route, profiles, scannable_channels, last_measure_points):
    here = _point(position)
    active_scan_channels = set(scannable_channels)
    last_measure = {int(channel): _point(point) for channel, point in last_measure_points.items()}
    movement_s = 0.0
    scan_action_s = 0.0
    source_action_s = 0.0
    duplicate_supplement_skips = 0

    for node in route:
        profile = profiles[_node_key(node)]
        entry = _point(profile["entry"])
        movement_s += math.dist(here, entry) / ROBOT_SPEED_MPS

        if profile["kind"] == "scan":
            scan_action_s += MEASURE_WITH_SWITCH_PROXY_S * len(active_scan_channels)
            for channel in active_scan_channels:
                last_measure[channel] = entry
            here = _point(profile["exit"])
            continue

        channel = int(profile["channel"])
        if not profile["direct"]:
            initial_point = _point(profile["initial_point"])
            if math.dist(here, initial_point) >= REUSE_MIN_BASELINE_M:
                if _same_point(last_measure.get(channel), here):
                    duplicate_supplement_skips += 1
                else:
                    source_action_s += MEASURE_WITH_SWITCH_PROXY_S
                    last_measure[channel] = here
            movement_s += float(profile["internal_distance_m"]) / ROBOT_SPEED_MPS
        source_action_s += float(profile["local_action_s"])
        active_scan_channels.discard(channel)
        here = _point(profile["exit"])

    total_s = movement_s + scan_action_s + source_action_s
    return {
        "total_s": total_s,
        "movement_s": movement_s,
        "scan_action_s": scan_action_s,
        "source_action_s": source_action_s,
        "duplicate_supplement_skips": duplicate_supplement_skips,
    }


def _plan(position, nodes, tracks, state, client):
    profiles = _service_profiles(nodes, tracks)
    scannable = _scannable_channels(state, tracks)
    anchor = _anchor_route(position, nodes)
    anchor_proxy = _proxy_cost(position, anchor, profiles, scannable, client.last_measure_point)

    choices = []
    best_route = anchor
    best_proxy = anchor_proxy
    for first in sorted(nodes, key=_node_key):
        remaining = list(nodes)
        remaining.remove(first)
        exit_point = profiles[_node_key(first)]["exit"]
        route = [first] + _anchor_route(exit_point, remaining)
        proxy = _proxy_cost(position, route, profiles, scannable, client.last_measure_point)
        choices.append({"first": _node_key(first), **proxy})
        candidate_order = (proxy["total_s"], proxy["movement_s"], _node_key(first))
        best_order = (best_proxy["total_s"], best_proxy["movement_s"], _node_key(best_route[0]))
        if candidate_order < best_order:
            best_route, best_proxy = route, proxy

    proxy_saving = anchor_proxy["total_s"] - best_proxy["total_s"]
    changed = _node_key(best_route[0]) != _node_key(anchor[0]) and proxy_saving >= MIN_PROXY_SAVING_S
    selected = best_route if changed else anchor
    if changed:
        state["service_route_changed_first_count"] += 1

    state["service_route_decisions"].append({
        "position": _point(position),
        "nodes": [{"kind": node[0], "ident": int(node[1]), "destination": _point(node[2])} for node in nodes],
        "profiles": [{"key": key, **profile} for key, profile in sorted(profiles.items())],
        "scannable_channels": sorted(scannable),
        "last_measure_points": {str(channel): point for channel, point in sorted(client.last_measure_point.items())},
        "anchor_first": _node_key(anchor[0]),
        "best_first": _node_key(best_route[0]),
        "selected_first": _node_key(selected[0]),
        "anchor_proxy": anchor_proxy,
        "best_proxy": best_proxy,
        "proxy_saving_s": proxy_saving,
        "minimum_proxy_saving_s": MIN_PROXY_SAVING_S,
        "changed_first": changed,
        "choices": choices,
        "measure_count_before": state["measure_count"],
        "clear_count_before": state["clear_count"],
    })
    return selected


def _supplement_without_duplicate(client, channel, track, state):
    initial_point = _point(track[0])
    if math.dist(client.position, initial_point) < REUSE_MIN_BASELINE_M:
        return track, False
    last_point = client.last_measure_point.get(channel)
    if _same_point(last_point, client.position):
        state["duplicate_supplement_skip_count"] += 1
        state["duplicate_supplement_skips"].append({
            "channel": int(channel),
            "position": _point(client.position),
            "last_measure_point": _point(last_point),
            "initial_point": initial_point,
            "measure_count_before": state["measure_count"],
            "clear_count_before": state["clear_count"],
        })
        return track, False
    return _supplement_from_current_pose(client, channel, track, state)


def run_search(client, question=4, method="main"):
    if question != 4 or method != "main":
        raise ValueError("Q4 service-aware route supports only question=4, method='main'")

    points = ring22_points()
    state = _state(points)
    state.update(
        coverage_geometry=coverage_certificate(),
        route_strategy="service_aware_first_task_with_frozen_route_suffix",
        service_proxy_version="entry_service_exit_v1",
        service_proxy_min_saving_s=MIN_PROXY_SAVING_S,
        service_route_decisions=[],
        service_route_changed_first_count=0,
        duplicate_supplement_skip_count=0,
        duplicate_supplement_skips=[],
        compatibility_used_for_decision=False,
        polygon_track_update_count=0,
        reuse_measure_count=0,
        reuse_polygon_update_count=0,
        direct_clear_skip_measure_count=0,
        maximum_count_stop_used=False,
        maximum_count_certificate=None,
    )
    tracked = ServiceTrackedClient(client)
    seen = set()
    tracks = {}
    attempted = set()
    remaining = list(range(len(points)))

    try:
        while remaining or any(channel not in state["cleared_channels"] and channel not in attempted for channel in tracks):
            if _stop(state):
                break
            nodes = [("scan", index, points[index]) for index in remaining]
            for channel, track in tracks.items():
                if channel in state["cleared_channels"] or channel in attempted:
                    continue
                bound = _box(track)
                if bound is not None:
                    nodes.append(("source", channel, bound[0]))
                elif math.dist(tracked.position, track[0]) >= REUSE_MIN_BASELINE_M:
                    nodes.append(("source", channel, tracked.position))
            if not nodes:
                break

            kind, ident, destination = _plan(tracked.position, nodes, tracks, state, tracked)[0]
            if kind == "scan":
                remaining.remove(ident)
                for channel in range(1, 21):
                    if _stop(state):
                        break
                    if channel in state["cleared_channels"]:
                        continue
                    old_track = tracks.get(channel)
                    bound = _box(old_track) if old_track else None
                    if bound is not None and bound[1] <= DIRECT_RADIUS_M:
                        continue
                    response = tracked.measure(*destination, channel)
                    state["measure_count"] += 1
                    state["coverage_measurements_by_channel"][channel] += 1
                    result = response.get("measure_result")
                    if result == "no_signal":
                        continue
                    seen.add(channel)
                    if result == "near":
                        _resolve_clear(tracked, destination, channel, state, "near ring22 joint coverage")
                    elif result == "direction" and polygon_policy._valid_svd(response.get("svd_deg")):
                        if old_track is None:
                            tracks[channel] = (destination, response["svd_deg"], _bearing_triangle(destination, response["svd_deg"]))
                        else:
                            initial_point, initial_svd, polygon = old_track
                            clipped = _add_bearing(polygon, destination, response["svd_deg"])
                            tracks[channel] = (initial_point, initial_svd, clipped) if clipped else old_track
                        attempted.discard(channel)
                        state["polygon_track_update_count"] += 1
                    else:
                        state["warnings"].append(f"channel {channel}: invalid public response")
                        state["unresolved_channels"].add(channel)
                state["batch_count"] += 1
                continue

            channel = ident
            track = tracks[channel]
            attempted.add(channel)
            if _direct(tracked, channel, track, state):
                continue
            track, cleared = _supplement_without_duplicate(tracked, channel, track, state)
            if cleared:
                continue
            tracks[channel] = track
            if _direct(tracked, channel, track, state):
                continue
            _localize(tracked, track[0], track[1], channel, state, track[2])

        if not state["maximum_count_stop_used"]:
            for channel in range(1, 21):
                if channel not in state["cleared_channels"]:
                    target = state["unresolved_channels"] if channel in seen else state["excluded_channels"]
                    target.add(channel)
    except Exception as error:
        state["stopped"] = True
        state["warnings"].append(f"protocol/transport stop: {type(error).__name__}: {error}")
        state["unresolved_channels"].update(
            set(range(1, 21)) - state["cleared_channels"] - state["excluded_channels"]
        )

    for key in ("cleared_channels", "excluded_channels", "unresolved_channels"):
        state[key] = sorted(state[key])
    state.update(
        completed=(
            not state["stopped"]
            and not state["unresolved_channels"]
            and len(state["cleared_channels"]) + len(state["excluded_channels"]) == 20
        ),
        question=4,
        method="main",
        version=VERSION,
        virtual_time_s=tracked.virtual_time_s,
    )
    return state


__all__ = ["VERSION", "run_search"]

