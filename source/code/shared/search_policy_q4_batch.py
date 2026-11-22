"""Independent Q4 compact-49 candidate with batch-then-nearest localization.

At every compact coverage point, all currently uncleared channels are measured
before any directional localization.  Pending channels are then localized in
nearest-current-polygon-center order.  The policy only receives client
responses and does not import arena or source data.
"""

from __future__ import annotations

import math

import search_policy_polygon as polygon_policy
from search_policy_compact import coverage_points
from search_policy import (LOCATE_CLEAR_RADIUS_M, MAX_EXTRA_BEARINGS, _add_bearing,
                           _bearing_triangle, _box_circle, _resolve_clear)


VERSION = "q4_batch49_v1"


class _TrackedClient:
    def __init__(self, client):
        self._client = client
        self.position = (0.0, 0.0)

    @property
    def virtual_time_s(self):
        return self._client.virtual_time_s

    def measure(self, x, y, channel):
        response = self._client.measure(x, y, channel)
        self.position = (x, y)
        return response

    def clear(self, x, y, channel):
        response = self._client.clear(x, y, channel)
        self.position = (x, y)
        return response


def _localize(client, initial_point, initial_svd, channel, state, feasible_polygon):
    """Recovery localizer copied from polygon policy, retaining its current polygon."""
    polygon = feasible_polygon
    last_visible = initial_point
    previous_candidate = None
    extra_used = 0
    while True:
        box = _box_circle(polygon)
        if box is None:
            state["warnings"].append(f"channel {channel}: empty bearing polygon")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        candidate, radius = box
        if radius <= LOCATE_CLEAR_RADIUS_M:
            if _resolve_clear(client, candidate, channel, state, "bearing localization"):
                return True
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        if extra_used >= MAX_EXTRA_BEARINGS:
            state["warnings"].append(f"channel {channel}: extra bearing limit reached")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        if previous_candidate and math.dist(candidate, previous_candidate) <= 1e-6:
            state["warnings"].append(f"channel {channel}: repeated localization center")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        previous_candidate = candidate
        response = client.measure(candidate[0], candidate[1], channel)
        state["measure_count"] += 1
        state["extra_bearing_count"] += 1
        extra_used += 1
        kind = response.get("measure_result")
        if kind == "near":
            if _resolve_clear(client, candidate, channel, state, "near follow-up"):
                return True
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        if kind == "direction":
            svd = response.get("svd_deg")
            if not polygon_policy._valid_svd(svd):
                state["warnings"].append(f"channel {channel}: follow-up direction lacks valid svd_deg")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            polygon = _add_bearing(polygon, candidate, svd)
            last_visible = candidate
            continue
        if kind != "no_signal":
            state["warnings"].append(f"channel {channel}: unknown follow-up result {kind!r}")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)

        state["recovery_count"] += 1
        invisible = candidate
        while True:
            box = _box_circle(polygon)
            if box is None:
                state["warnings"].append(f"channel {channel}: empty recovery polygon")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            recovery_center, recovery_radius = box
            if recovery_radius <= LOCATE_CLEAR_RADIUS_M:
                if _resolve_clear(client, recovery_center, channel, state, "recovery localization"):
                    return True
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            if extra_used >= MAX_EXTRA_BEARINGS:
                state["warnings"].append(f"channel {channel}: recovery budget exhausted")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            midpoint = ((last_visible[0] + invisible[0]) / 2.0, (last_visible[1] + invisible[1]) / 2.0)
            if math.dist(midpoint, last_visible) <= 1e-6 or math.dist(midpoint, invisible) <= 1e-6:
                state["warnings"].append(f"channel {channel}: recovery midpoint repeated")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            response = client.measure(midpoint[0], midpoint[1], channel)
            state["measure_count"] += 1
            state["extra_bearing_count"] += 1
            state["recovery_measure_count"] += 1
            extra_used += 1
            kind = response.get("measure_result")
            if kind == "near":
                if _resolve_clear(client, midpoint, channel, state, "near recovery"):
                    return True
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            if kind == "direction":
                svd = response.get("svd_deg")
                if not polygon_policy._valid_svd(svd):
                    state["warnings"].append(f"channel {channel}: recovery direction lacks valid svd_deg")
                    return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
                polygon = _add_bearing(polygon, midpoint, svd)
                last_visible = midpoint
                state["recovery_positive_count"] += 1
                continue
            if kind == "no_signal":
                invisible = midpoint
                continue
            state["warnings"].append(f"channel {channel}: unknown recovery result {kind!r}")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)


def _state(points):
    return {
        "cleared_channels": set(), "excluded_channels": set(), "unresolved_channels": set(),
        "measure_count": 0, "clear_count": 0, "extra_bearing_count": 0,
        "fallback_count": 0, "fallback_clear_count": 0, "warnings": [], "stopped": False,
        "coverage_points_total": len(points),
        "coverage_measurements_by_channel": {channel: 0 for channel in range(1, 21)},
        "recovery_count": 0, "recovery_measure_count": 0, "recovery_positive_count": 0,
        "polygon_fallback_count": 0, "polygon_fallback_point_count": 0,
        "polygon_fallback_clear_count": 0, "polygon_fallback_exhausted_count": 0,
        "polygon_fallback_invalid_count": 0, "legacy_fallback_count": 0,
        "legacy_fallback_clear_count": 0, "batch_count": 0,
    }


def run_search(client, question=4, method="main"):
    if question != 4 or method != "main":
        raise ValueError("q4_batch policy supports only question=4, method='main'")
    points = coverage_points(4)
    state = _state(points)
    tracked = _TrackedClient(client)
    seen = set()
    try:
        for scan_point in points:
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
                    _resolve_clear(tracked, scan_point, channel, state, "near batch coverage")
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
                channel, (initial_point, initial_svd, feasible_polygon) = min(pending.items(), key=distance_to_center)
                del pending[channel]
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
