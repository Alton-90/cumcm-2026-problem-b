"""Independent Q4 triangle-cover variants that reuse the robot's current pose.

Variant A takes a supplementary bearing at the robot's current position before
localizing a pending channel, provided it is at least 50 m from that channel's
initial observation. Variant B first makes one bounded 60 m-radius clear
attempt, then follows A on failure. All source information remains private to
the simulator; the policy sees only public responses.
"""

from __future__ import annotations

import math

import search_policy_polygon as polygon_policy
from search_policy import LOCATE_CLEAR_RADIUS_M, _add_bearing, _bearing_triangle, _box_circle, _resolve_clear
from search_policy_q4_batch import _TrackedClient, _localize, _state
from search_policy_q4_triangle import triangle_coverage_points


VERSION = "q4_reuse_v1"
REUSE_MIN_BASELINE_M = 50.0
EARLY_CLEAR_RADIUS_M = 60.0


def _box(track):
    return _box_circle(track[2])


def _supplement_from_current_pose(client, channel, track, state):
    """Merge one new public bearing without moving back to the old ray center."""
    initial_point, initial_svd, polygon = track
    if math.dist(client.position, initial_point) < REUSE_MIN_BASELINE_M:
        return track, False
    response = client.measure(client.position[0], client.position[1], channel)
    state["measure_count"] += 1
    state["extra_bearing_count"] += 1
    state["reuse_measure_count"] += 1
    kind = response.get("measure_result")
    if kind == "near":
        if _resolve_clear(client, client.position, channel, state, "current-pose near reuse"):
            return None, True
        return track, False
    if kind == "direction":
        svd = response.get("svd_deg")
        if polygon_policy._valid_svd(svd):
            clipped = _add_bearing(polygon, client.position, svd)
            if clipped:
                state["reuse_polygon_update_count"] += 1
                return (initial_point, initial_svd, clipped), False
            state["warnings"].append(f"channel {channel}: ignored empty current-pose intersection")
            return track, False
        state["warnings"].append(f"channel {channel}: invalid current-pose direction")
        return track, False
    if kind != "no_signal":
        state["warnings"].append(f"channel {channel}: unknown current-pose result {kind!r}")
    return track, False


def _try_early_clear(client, channel, track, state):
    box = _box(track)
    if box is None or box[1] > EARLY_CLEAR_RADIUS_M:
        return False
    state["early_clear_attempt_count"] += 1
    if _resolve_clear(client, box[0], channel, state, "60m polygon candidate check"):
        state["early_clear_success_count"] += 1
        return True
    return False


def run_search(client, question=4, method="main", variant="A"):
    if question != 4 or method != "main" or variant not in ("A", "B"):
        raise ValueError("reuse policy supports question=4, method='main', variant='A'|'B'")
    points = triangle_coverage_points()
    state = _state(points)
    state.update({"reuse_variant": variant, "reuse_measure_count": 0, "reuse_polygon_update_count": 0,
                  "early_clear_attempt_count": 0, "early_clear_success_count": 0,
                  "coverage_geometry": {"triangle_side_m": 980.0, "source_radius_m": 1800.0,
                                        "minimum_receive_radius_m": 1000.0}})
    tracked = _TrackedClient(client)
    seen = set()
    remaining = list(points)
    try:
        while remaining:
            scan_point = min(remaining, key=lambda point: (math.dist(tracked.position, point), point[1], point[0]))
            remaining.remove(scan_point)
            pending = {}
            for channel in range(1, 21):
                if channel in state["cleared_channels"]:
                    continue
                response = tracked.measure(scan_point[0], scan_point[1], channel)
                state["measure_count"] += 1
                state["coverage_measurements_by_channel"][channel] += 1
                kind = response.get("measure_result")
                if kind == "no_signal":
                    continue
                seen.add(channel)
                if kind == "near":
                    _resolve_clear(tracked, scan_point, channel, state, "near triangle coverage")
                elif kind == "direction":
                    svd = response.get("svd_deg")
                    if not polygon_policy._valid_svd(svd):
                        state["warnings"].append(f"channel {channel}: invalid coverage direction")
                        state["unresolved_channels"].add(channel)
                    else:
                        pending[channel] = (scan_point, svd, _bearing_triangle(scan_point, svd))
                else:
                    state["warnings"].append(f"channel {channel}: unknown measure result {kind!r}")
                    state["unresolved_channels"].add(channel)
            state["batch_count"] += 1
            while pending:
                def distance_to_center(item):
                    box = _box(item[1])
                    return math.inf if box is None else math.dist(tracked.position, box[0])
                channel, track = min(pending.items(), key=distance_to_center)
                del pending[channel]
                if variant == "B" and _try_early_clear(tracked, channel, track, state):
                    continue
                track, cleared_from_reuse = _supplement_from_current_pose(tracked, channel, track, state)
                if cleared_from_reuse:
                    continue
                _localize(tracked, track[0], track[1], channel, state, track[2])
        for channel in range(1, 21):
            if channel in state["cleared_channels"]:
                continue
            if channel in seen:
                state["unresolved_channels"].add(channel)
            else:
                state["excluded_channels"].add(channel)
    except Exception as error:
        state["stopped"] = True
        state["warnings"].append(f"protocol/transport stop: {type(error).__name__}: {error}")
        state["unresolved_channels"].update(set(range(1, 21)) - state["cleared_channels"] - state["excluded_channels"])
    for key in ("cleared_channels", "excluded_channels", "unresolved_channels"):
        state[key] = sorted(state[key])
    state["completed"] = (not state["stopped"] and not state["unresolved_channels"] and
                          len(state["cleared_channels"]) + len(state["excluded_channels"]) == 20)
    state.update(question=4, method="main", version=VERSION, virtual_time_s=tracked.virtual_time_s)
    return state
