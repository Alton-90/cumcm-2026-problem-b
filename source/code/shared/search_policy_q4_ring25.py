"""Q4 reuse-A localizer over a certified 25-point concentric-ring cover.

The source disk is contained in the outer 16-gon.  Its triangulation uses
only edges shorter than the 1000 m minimum reception radius, so every source
has directional and omnidirectional coverage through public measurements.
"""

from __future__ import annotations

import math

import search_policy_polygon as polygon_policy
from search_policy import _bearing_triangle, _box_circle, _resolve_clear
from search_policy_q4_batch import _TrackedClient, _localize, _state
from search_policy_q4_reuse import _supplement_from_current_pose


VERSION = "q4_ring25_v1"
INNER_RADIUS_M = 995.0
OUTER_RADIUS_M = 1840.0
MIN_RECEIVE_RADIUS_M = 1000.0


def ring25_points():
    origin = (0.0, 0.0)
    inner = [(INNER_RADIUS_M * math.cos(k * math.pi / 4), INNER_RADIUS_M * math.sin(k * math.pi / 4))
             for k in range(8)]
    outer = [(OUTER_RADIUS_M * math.cos(k * math.pi / 8), OUTER_RADIUS_M * math.sin(k * math.pi / 8))
             for k in range(16)]
    return [origin] + inner + outer


def coverage_certificate():
    points = ring25_points()
    inner, outer = points[1:9], points[9:]
    triangles = []
    for k in range(8):
        next_inner = (k + 1) % 8
        middle_outer, next_outer = (2 * k + 1) % 16, (2 * k + 2) % 16
        triangles.extend([
            (points[0], inner[k], inner[next_inner]),
            (inner[k], outer[2 * k], outer[middle_outer]),
            (inner[k], outer[middle_outer], inner[next_inner]),
            (inner[next_inner], outer[middle_outer], outer[next_outer]),
        ])
    lengths = [math.dist(a, b) for triangle in triangles for a, b in
               ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))]
    return {"point_count": len(points), "triangle_count": len(triangles), "max_triangle_edge_m": max(lengths),
            "outer_16gon_inradius_m": OUTER_RADIUS_M * math.cos(math.pi / 16),
            "minimum_receive_radius_m": MIN_RECEIVE_RADIUS_M}


def run_search(client, question=4, method="main"):
    if question != 4 or method != "main":
        raise ValueError("ring25 policy supports only question=4, method='main'")
    points = ring25_points()
    state = _state(points)
    state.update({"reuse_variant": "A", "reuse_measure_count": 0, "reuse_polygon_update_count": 0,
                  "early_clear_attempt_count": 0, "early_clear_success_count": 0,
                  "coverage_geometry": coverage_certificate()})
    tracked = _TrackedClient(client)
    seen, remaining = set(), list(points)
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
                    _resolve_clear(tracked, scan_point, channel, state, "near ring25 coverage")
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
                    box = _box_circle(item[1][2])
                    return math.inf if box is None else math.dist(tracked.position, box[0])
                channel, track = min(pending.items(), key=distance_to_center)
                del pending[channel]
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
