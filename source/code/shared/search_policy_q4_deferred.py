"""Q4 deferred feasible-polygon localization over the certified triangle cover.

Variant A retains all directional feasible polygons until coverage completes.
Variant B additionally clears already-clearable polygons only when their center
is within 400 m of the robot after a completed scan batch.  Both variants use
only public measurement and clear responses.
"""

from __future__ import annotations

import math

import search_policy_polygon as polygon_policy
from search_policy import LOCATE_CLEAR_RADIUS_M, _add_bearing, _bearing_triangle, _box_circle, _resolve_clear
from search_policy_q4_batch import _TrackedClient, _localize, _state
from search_policy_q4_triangle import _nearest_route, triangle_coverage_points


VERSION = "q4_deferred_v1"
IMMEDIATE_CLEAR_DISTANCE_M = 400.0


def _valid_polygon_center(track):
    box = _box_circle(track[2])
    return None if box is None else box


def _clear_if_certified(client, channel, track, state, reason):
    box = _valid_polygon_center(track)
    if box is None:
        state["warnings"].append(f"channel {channel}: empty deferred polygon")
        return False
    center, radius = box
    if radius > LOCATE_CLEAR_RADIUS_M:
        return False
    return _resolve_clear(client, center, channel, state, reason)


def run_search(client, question=4, method="main", variant="A"):
    if question != 4 or method != "main" or variant not in ("A", "B"):
        raise ValueError("deferred policy supports question=4, method='main', variant='A'|'B'")
    points = triangle_coverage_points()
    state = _state(points)
    state.update({"deferred_variant": variant, "deferred_track_updates": 0,
                  "deferred_immediate_clear_count": 0, "deferred_final_clear_count": 0,
                  "deferred_polygon_empty_count": 0,
                  "coverage_geometry": {"triangle_side_m": 980.0, "source_radius_m": 1800.0,
                                        "minimum_receive_radius_m": 1000.0}})
    tracked = _TrackedClient(client)
    tracks = {}
    seen = set()
    try:
        for scan_point in _nearest_route(points):
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
                    tracks.pop(channel, None)
                elif kind == "direction":
                    svd = response.get("svd_deg")
                    if not polygon_policy._valid_svd(svd):
                        state["warnings"].append(f"channel {channel}: invalid coverage direction")
                        state["unresolved_channels"].add(channel)
                        continue
                    if channel not in tracks:
                        tracks[channel] = (scan_point, svd, _bearing_triangle(scan_point, svd))
                    else:
                        initial_point, initial_svd, feasible_polygon = tracks[channel]
                        clipped = _add_bearing(feasible_polygon, scan_point, svd)
                        if clipped:
                            tracks[channel] = (initial_point, initial_svd, clipped)
                            state["deferred_track_updates"] += 1
                        else:
                            state["deferred_polygon_empty_count"] += 1
                            state["warnings"].append(f"channel {channel}: discarded empty deferred intersection")
                else:
                    state["warnings"].append(f"channel {channel}: unknown measure result {kind!r}")
                    state["unresolved_channels"].add(channel)
            state["batch_count"] += 1
            if variant == "B":
                candidates = []
                for channel, track in tracks.items():
                    box = _valid_polygon_center(track)
                    if box is not None and box[1] <= LOCATE_CLEAR_RADIUS_M and math.dist(tracked.position, box[0]) <= IMMEDIATE_CLEAR_DISTANCE_M:
                        candidates.append((math.dist(tracked.position, box[0]), channel, track))
                for _, channel, track in sorted(candidates):
                    if channel not in tracks or channel in state["cleared_channels"]:
                        continue
                    if _clear_if_certified(tracked, channel, track, state, "nearby deferred polygon"):
                        state["deferred_immediate_clear_count"] += 1
                        tracks.pop(channel, None)

        while tracks:
            def distance_to_center(item):
                box = _valid_polygon_center(item[1])
                return math.inf if box is None else math.dist(tracked.position, box[0])
            channel, track = min(tracks.items(), key=distance_to_center)
            del tracks[channel]
            initial_point, initial_svd, feasible_polygon = track
            if _clear_if_certified(tracked, channel, track, state, "deferred polygon completion"):
                state["deferred_final_clear_count"] += 1
                continue
            _localize(tracked, initial_point, initial_svd, channel, state, feasible_polygon)

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
