"""Q4 ring25 cover with public-polygon source nodes and rolling open 2-opt routing.

The route contains only certified coverage points and centers of feasible
polygons built from public measurements.  It never reads arena/source truth.
"""

from __future__ import annotations

import hashlib
import math

import search_policy_polygon as polygon_policy
from search_policy import _add_bearing, _bearing_triangle, _box_circle, _resolve_clear
from search_policy_joint_route_q3 import _route
from search_policy_q4_batch import _TrackedClient, _localize, _state
from search_policy_q4_reuse import _supplement_from_current_pose
from search_policy_q4_ring25 import coverage_certificate, ring25_points


VERSION = "q4_ring25_joint_route_v3"
DIRECT_CLEAR_CERTIFICATE_RADIUS_M = 19.9
MAXIMUM_SOURCE_COUNT = 16
ROUTE_DEPENDENCY_SHA256 = hashlib.sha256(
    (__import__("pathlib").Path(__file__).with_name("search_policy_joint_route_q3.py")).read_bytes()
).hexdigest()


def _box(track):
    return _box_circle(track[2])


def _reached_source_limit(state):
    """Stop only after a valid certificate that all 16 possible sources cleared."""
    if len(state["cleared_channels"]) != MAXIMUM_SOURCE_COUNT:
        return False
    state["excluded_channels"].update(set(range(1, 21)) - state["cleared_channels"])
    state["maximum_count_certificate"] = {
        "maximum_sources": MAXIMUM_SOURCE_COUNT,
        "cleared_distinct_channels": len(state["cleared_channels"]),
        "rule": "The question bounds the number of sources by 16; remaining channels are excluded only after 16 successful distinct clears.",
    }
    state["maximum_count_stop_used"] = True
    return True


def _direct_clear_if_certified(client, channel, track, state):
    box = _box(track)
    if box is None or box[1] > DIRECT_CLEAR_CERTIFICATE_RADIUS_M:
        return False
    state["direct_clear_skip_measure_count"] += 1
    return _resolve_clear(client, box[0], channel, state, "19.9m public-polygon certificate")


def run_search(client, question=4, method="main"):
    if question != 4 or method != "main":
        raise ValueError("q4 ring25 joint-route policy supports only question=4, method='main'")
    points = ring25_points()
    state = _state(points)
    state.update({
        "coverage_geometry": coverage_certificate(),
        "route_strategy": "receding_open_2opt_over_unvisited_scans_and_public_polygon_centers",
        "route_dependency": {"path": "code/shared/search_policy_joint_route_q3.py", "sha256": ROUTE_DEPENDENCY_SHA256},
        "polygon_track_update_count": 0,
        "reuse_measure_count": 0,
        "reuse_polygon_update_count": 0,
        "direct_clear_skip_measure_count": 0,
        "maximum_count_stop_used": False,
        "maximum_count_certificate": None,
    })
    tracked = _TrackedClient(client)
    seen, tracks, attempted = set(), {}, set()
    remaining = list(range(len(points)))
    try:
        while remaining or any(channel not in state["cleared_channels"] and channel not in attempted for channel in tracks):
            if _reached_source_limit(state):
                break
            nodes = [("scan", index, points[index]) for index in remaining]
            for channel, track in tracks.items():
                if channel in state["cleared_channels"] or channel in attempted:
                    continue
                box = _box(track)
                if box is not None:
                    nodes.append(("source", channel, box[0]))
                elif math.dist(tracked.position, track[0]) >= 50.0:
                    # A single public ray has no bounded center yet.  Once the
                    # robot has naturally left its first observation, take the
                    # reuse-A current-pose bearing before any fallback.
                    nodes.append(("source", channel, tracked.position))
            if not nodes:
                break
            kind, identifier, destination = _route(tracked.position, nodes)[0]
            if kind == "scan":
                remaining.remove(identifier)
                for channel in range(1, 21):
                    if _reached_source_limit(state):
                        break
                    if channel in state["cleared_channels"]:
                        continue
                    existing = tracks.get(channel)
                    if existing is not None:
                        box = _box(existing)
                        if box is not None and box[1] <= DIRECT_CLEAR_CERTIFICATE_RADIUS_M:
                            continue
                    response = tracked.measure(destination[0], destination[1], channel)
                    state["measure_count"] += 1
                    state["coverage_measurements_by_channel"][channel] += 1
                    result = response.get("measure_result")
                    if result == "no_signal":
                        continue
                    seen.add(channel)
                    if result == "near":
                        _resolve_clear(tracked, destination, channel, state, "near ring25 joint-route coverage")
                    elif result == "direction":
                        svd = response.get("svd_deg")
                        if not polygon_policy._valid_svd(svd):
                            state["warnings"].append(f"channel {channel}: invalid coverage direction")
                            state["unresolved_channels"].add(channel)
                        elif existing is None:
                            tracks[channel] = (destination, svd, _bearing_triangle(destination, svd))
                            attempted.discard(channel)
                            state["polygon_track_update_count"] += 1
                        else:
                            merged = _add_bearing(existing[2], destination, svd)
                            if merged:
                                tracks[channel] = (existing[0], existing[1], merged)
                                attempted.discard(channel)
                                state["polygon_track_update_count"] += 1
                    else:
                        state["warnings"].append(f"channel {channel}: unknown coverage result {result!r}")
                        state["unresolved_channels"].add(channel)
                state["batch_count"] += 1
            else:
                channel = identifier
                track = tracks[channel]
                attempted.add(channel)
                if _direct_clear_if_certified(tracked, channel, track, state):
                    continue
                track, cleared_from_reuse = _supplement_from_current_pose(tracked, channel, track, state)
                if cleared_from_reuse:
                    continue
                tracks[channel] = track
                if _direct_clear_if_certified(tracked, channel, track, state):
                    continue
                _localize(tracked, track[0], track[1], channel, state, track[2])
        if not state["maximum_count_stop_used"]:
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
