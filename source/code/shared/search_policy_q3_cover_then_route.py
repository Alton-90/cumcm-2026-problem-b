"""Q3: complete the certified cover first, then jointly route public source estimates.

Unlike joint_route_q3, this policy does not divert from the seven-point cover
for individual localization.  All public bearings are accumulated at coverage
points, then the resulting source estimates are visited by one open 2-opt
route.  No arena/source truth is accessed.
"""

from __future__ import annotations

import math

import search_policy as old
from search_policy_compact import _state
from search_policy_feasible_q3 import Tracked, localize


VERSION = "q3_cover_then_route_v1"
MAXIMUM_SOURCE_COUNT = 16


def _valid_svd(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value < 360


def _route(position, nodes):
    """Deterministic nearest-insertion seed with bounded open-path 2-opt."""
    pending, route, here = list(nodes), [], position
    while pending:
        item = min(pending, key=lambda item: (math.dist(here, item[2]), item[0], item[1]))
        pending.remove(item)
        route.append(item)
        here = item[2]
    for _ in range(10):
        gain, swap = 0.0, None
        for i in range(len(route) - 1):
            before = position if i == 0 else route[i - 1][2]
            for j in range(i + 1, len(route)):
                old_cost = math.dist(before, route[i][2])
                new_cost = math.dist(before, route[j][2])
                if j + 1 < len(route):
                    old_cost += math.dist(route[j][2], route[j + 1][2])
                    new_cost += math.dist(route[i][2], route[j + 1][2])
                if old_cost - new_cost > gain + 1e-7:
                    gain, swap = old_cost - new_cost, (i, j)
        if swap is None:
            break
        i, j = swap
        route[i:j + 1] = reversed(route[i:j + 1])
    return route


def _source_limit_reached(state):
    if len(state["cleared_channels"]) != MAXIMUM_SOURCE_COUNT:
        return False
    state["excluded_channels"].update(set(range(1, 21)) - state["cleared_channels"])
    state["maximum_count_certificate"] = {
        "maximum_sources": MAXIMUM_SOURCE_COUNT,
        "cleared_distinct_channels": len(state["cleared_channels"]),
        "rule": "Only 16 successful distinct clears establish that no remaining channel can contain a source.",
    }
    state["maximum_count_stop_used"] = True
    return True


def _coverage_points():
    return [(0.0, 0.0)] + [(1150.0 * math.cos(k * math.pi / 3), 1150.0 * math.sin(k * math.pi / 3)) for k in range(6)]


def run_search(client, question=3, method="main"):
    if question != 3 or method != "main":
        raise ValueError("q3 cover-then-route supports only question=3, method='main'")
    points = _coverage_points()
    state = _state(3, points)
    state.update({
        "route_strategy": "complete_certified_cover_then_public_estimate_open_2opt",
        "coverage_certificate": "origin plus all six 1150m hexagon vertices are measured for every uncleared channel before unknown-channel exclusion",
        "coverage_points_visited": [],
        "polygon_track_update_count": 0,
        "maximum_count_stop_used": False,
        "maximum_count_certificate": None,
    })
    tracked = Tracked(client)
    tracks, seen = {}, set()
    try:
        remaining = list(enumerate(points))
        while remaining:
            if _source_limit_reached(state):
                break
            nodes = [("scan", index, point) for index, point in remaining]
            _, index, point = _route(tracked.position, nodes)[0]
            remaining = [item for item in remaining if item[0] != index]
            state["coverage_points_visited"].append(index)
            for channel in range(1, 21):
                if _source_limit_reached(state):
                    break
                if channel in state["cleared_channels"]:
                    continue
                response = tracked.measure(point[0], point[1], channel)
                state["measure_count"] += 1
                state["coverage_measurements_by_channel"][channel] += 1
                result = response.get("measure_result")
                if result == "no_signal":
                    continue
                seen.add(channel)
                if result == "near":
                    old._resolve_clear(tracked, point, channel, state, "near certified coverage")
                elif result == "direction" and _valid_svd(response.get("svd_deg")):
                    if channel not in tracks:
                        tracks[channel] = (point, response, old._bearing_triangle(point, response["svd_deg"]))
                    else:
                        initial, first_response, polygon = tracks[channel]
                        merged = old._add_bearing(polygon, point, response["svd_deg"])
                        if merged:
                            tracks[channel] = (initial, first_response, merged)
                    state["polygon_track_update_count"] += 1
                else:
                    state["warnings"].append(f"channel {channel}: invalid public direction response")
                    state["unresolved_channels"].add(channel)
        if not state["maximum_count_stop_used"]:
            # Every fixed cover point has now been visited; only this fact, not
            # a finite grid surrogate, certifies exclusion of never-seen channels.
            state["excluded_channels"].update(set(range(1, 21)) - seen)
            pending = set(tracks) - state["cleared_channels"]
            while pending:
                if _source_limit_reached(state):
                    break
                nodes = []
                for channel in pending:
                    estimate = tracked.estimate(channel) or old._box_circle(tracks[channel][2])
                    if estimate is not None:
                        nodes.append(("source", channel, estimate[0]))
                if not nodes:
                    break
                _, channel, destination = _route(tracked.position, nodes)[0]
                pending.remove(channel)
                initial, first_response, polygon = tracks[channel]
                if math.dist(tracked.position, initial) > 1.0:
                    response = tracked.measure(tracked.position[0], tracked.position[1], channel)
                    state["measure_count"] += 1
                    state["extra_bearing_count"] += 1
                    if response.get("measure_result") == "near" and old._resolve_clear(tracked, tracked.position, channel, state, "near route reuse"):
                        continue
                    if response.get("measure_result") == "direction" and _valid_svd(response.get("svd_deg")):
                        merged = old._add_bearing(polygon, tracked.position, response["svd_deg"])
                        if merged:
                            polygon = merged
                            tracks[channel] = (initial, first_response, polygon)
                localize(tracked, initial, first_response, channel, state, polygon)
            state["unresolved_channels"].update(seen - state["cleared_channels"])
    except Exception as error:
        state["stopped"] = True
        state["warnings"].append(f"protocol/transport stop: {type(error).__name__}: {error}")
        state["unresolved_channels"].update(set(range(1, 21)) - state["cleared_channels"] - state["excluded_channels"])
    for key in ("cleared_channels", "excluded_channels", "unresolved_channels"):
        state[key] = sorted(state[key])
    state["completed"] = (not state["stopped"] and not state["unresolved_channels"] and
                          len(state["cleared_channels"]) + len(state["excluded_channels"]) == 20)
    state.update(question=3, method="main", version=VERSION, virtual_time_s=tracked.virtual_time_s)
    return state
