"""Independent compact-grid candidate with polygon-bounded Q4 fallback scans.

Q3 delegates unchanged to ``search_policy_compact``.  Q4 retains the recovery
localizer but searches the current feasible polygon before the legacy 304-point
scan.  Policies receive only the public client actions; no arena/source state
is imported or inspected.
"""

from __future__ import annotations

import math

from search_policy import (LOCATE_CLEAR_RADIUS_M, MAX_EXTRA_BEARINGS, _add_bearing,
                           _bearing_triangle, _box_circle, _clear_points, _resolve_clear)
from search_policy_compact import coverage_points, run_search as compact_run_search


VERSION = "polygon_v1"
CELL_MAX_SIDE_M = 24.0
CELL_COVER_RADIUS_M = 12.0 * math.sqrt(2.0)
POLYGON_TOL = 1e-8


def _valid_svd(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value) and 0.0 <= value < 360.0)


def _polygon_is_valid(polygon):
    if not isinstance(polygon, list) or len(polygon) < 3:
        return False
    if not all(isinstance(point, tuple) and len(point) == 2 and
               all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
               for point in polygon):
        return False
    twice_area = sum(point[0] * polygon[(index + 1) % len(polygon)][1] -
                     point[1] * polygon[(index + 1) % len(polygon)][0]
                     for index, point in enumerate(polygon))
    return abs(twice_area) > POLYGON_TOL


def _axis_centers(lower, upper):
    """Cell centers with side length <=24 m; a zero-width axis has one center."""
    width = upper - lower
    if width < -POLYGON_TOL:
        raise ValueError("invalid rectangle bounds")
    if width <= POLYGON_TOL:
        return [lower]
    count = math.ceil(width / CELL_MAX_SIDE_M)
    return [lower + (index + 0.5) * width / count for index in range(count)]


def polygon_fallback_points(polygon, initial_point, initial_svd):
    """Cover a feasible polygon's initial-bearing bounding rectangle in snake order.

    Every rectangle point lies within ``CELL_COVER_RADIUS_M`` (<20m) of a
    returned center.  ``None`` means the polygon/basis is invalid.
    """
    if not _polygon_is_valid(polygon) or not _valid_svd(initial_svd):
        return None
    theta = math.radians(float(initial_svd))
    axis, lateral = (math.cos(theta), math.sin(theta)), (-math.sin(theta), math.cos(theta))
    projected = [((point[0] - initial_point[0]) * axis[0] + (point[1] - initial_point[1]) * axis[1],
                  (point[0] - initial_point[0]) * lateral[0] + (point[1] - initial_point[1]) * lateral[1])
                 for point in polygon]
    us = _axis_centers(min(point[0] for point in projected), max(point[0] for point in projected))
    vs = _axis_centers(min(point[1] for point in projected), max(point[1] for point in projected))
    result = []
    for row, v in enumerate(vs):
        row_us = us if row % 2 == 0 else list(reversed(us))
        for u in row_us:
            result.append((initial_point[0] + u * axis[0] + v * lateral[0],
                           initial_point[1] + u * axis[1] + v * lateral[1]))
    return result


def _fallback_scan(client, initial_point, initial_svd, channel, state, polygon):
    """Try the current polygon cover, then legacy 304 points only after exhaustion."""
    state["fallback_count"] += 1
    points = polygon_fallback_points(polygon, initial_point, initial_svd)
    if points is not None:
        state["polygon_fallback_count"] += 1
        state["polygon_fallback_point_count"] += len(points)
        for candidate in points:
            state["fallback_clear_count"] += 1
            state["polygon_fallback_clear_count"] += 1
            if _resolve_clear(client, candidate, channel, state, "fallback scan"):
                return True
        state["polygon_fallback_exhausted_count"] += 1
    else:
        state["polygon_fallback_invalid_count"] += 1

    if not _valid_svd(initial_svd):
        state["warnings"].append(f"channel {channel}: no valid bearing basis for legacy fallback")
        return False
    state["legacy_fallback_count"] += 1
    for candidate in _clear_points(initial_point, initial_svd):
        state["fallback_clear_count"] += 1
        state["legacy_fallback_clear_count"] += 1
        if _resolve_clear(client, candidate, channel, state, "fallback scan"):
            return True
    state["warnings"].append(f"channel {channel}: polygon and all 304 legacy fallback points exhausted")
    return False


def _measure(client, point, channel, state, *, recovery=False):
    response = client.measure(point[0], point[1], channel)
    state["measure_count"] += 1
    state["extra_bearing_count"] += 1
    if recovery:
        state["recovery_measure_count"] += 1
    return response


def _handle_direction(client, point, response, channel, state):
    initial_svd = response.get("svd_deg")
    if not _valid_svd(initial_svd):
        state["warnings"].append(f"channel {channel}: direction response lacks valid svd_deg")
        return False
    polygon = _bearing_triangle(point, initial_svd)
    last_visible = point
    previous_candidate = None
    extra_used = 0
    while True:
        box = _box_circle(polygon)
        if box is None:
            state["warnings"].append(f"channel {channel}: empty bearing polygon")
            return _fallback_scan(client, point, initial_svd, channel, state, polygon)
        candidate, radius = box
        if radius <= LOCATE_CLEAR_RADIUS_M:
            if _resolve_clear(client, candidate, channel, state, "bearing localization"):
                return True
            return _fallback_scan(client, point, initial_svd, channel, state, polygon)
        if extra_used >= MAX_EXTRA_BEARINGS:
            state["warnings"].append(f"channel {channel}: extra bearing limit reached")
            return _fallback_scan(client, point, initial_svd, channel, state, polygon)
        if previous_candidate and math.dist(candidate, previous_candidate) <= 1e-6:
            state["warnings"].append(f"channel {channel}: repeated localization center")
            return _fallback_scan(client, point, initial_svd, channel, state, polygon)
        previous_candidate = candidate
        follow_up = _measure(client, candidate, channel, state)
        extra_used += 1
        kind = follow_up.get("measure_result")
        if kind == "near":
            if _resolve_clear(client, candidate, channel, state, "near follow-up"):
                return True
            return _fallback_scan(client, point, initial_svd, channel, state, polygon)
        if kind == "direction":
            svd = follow_up.get("svd_deg")
            if not _valid_svd(svd):
                state["warnings"].append(f"channel {channel}: follow-up direction lacks valid svd_deg")
                return _fallback_scan(client, point, initial_svd, channel, state, polygon)
            polygon = _add_bearing(polygon, candidate, svd)
            last_visible = candidate
            continue
        if kind != "no_signal":
            state["warnings"].append(f"channel {channel}: unknown follow-up result {kind!r}")
            return _fallback_scan(client, point, initial_svd, channel, state, polygon)

        state["recovery_count"] += 1
        invisible = candidate
        while True:
            box = _box_circle(polygon)
            if box is None:
                state["warnings"].append(f"channel {channel}: empty recovery polygon")
                return _fallback_scan(client, point, initial_svd, channel, state, polygon)
            recovery_center, recovery_radius = box
            if recovery_radius <= LOCATE_CLEAR_RADIUS_M:
                if _resolve_clear(client, recovery_center, channel, state, "recovery localization"):
                    return True
                return _fallback_scan(client, point, initial_svd, channel, state, polygon)
            if extra_used >= MAX_EXTRA_BEARINGS:
                state["warnings"].append(f"channel {channel}: recovery budget exhausted")
                return _fallback_scan(client, point, initial_svd, channel, state, polygon)
            midpoint = ((last_visible[0] + invisible[0]) / 2.0, (last_visible[1] + invisible[1]) / 2.0)
            if math.dist(midpoint, last_visible) <= 1e-6 or math.dist(midpoint, invisible) <= 1e-6:
                state["warnings"].append(f"channel {channel}: recovery midpoint repeated")
                return _fallback_scan(client, point, initial_svd, channel, state, polygon)
            binary_response = _measure(client, midpoint, channel, state, recovery=True)
            extra_used += 1
            binary_kind = binary_response.get("measure_result")
            if binary_kind == "near":
                if _resolve_clear(client, midpoint, channel, state, "near recovery"):
                    return True
                return _fallback_scan(client, point, initial_svd, channel, state, polygon)
            if binary_kind == "direction":
                svd = binary_response.get("svd_deg")
                if not _valid_svd(svd):
                    state["warnings"].append(f"channel {channel}: recovery direction lacks valid svd_deg")
                    return _fallback_scan(client, point, initial_svd, channel, state, polygon)
                polygon = _add_bearing(polygon, midpoint, svd)
                last_visible = midpoint
                state["recovery_positive_count"] += 1
                continue
            if binary_kind == "no_signal":
                invisible = midpoint
                continue
            state["warnings"].append(f"channel {channel}: unknown recovery result {binary_kind!r}")
            return _fallback_scan(client, point, initial_svd, channel, state, polygon)


def run_search(client, question=4, method="main"):
    """Run unchanged compact Q3 or polygon-fallback compact Q4."""
    if question == 3:
        return compact_run_search(client, question=3, method=method)
    if question != 4 or method != "main":
        raise ValueError("polygon policy supports question=3|4 and method='main'")
    points = coverage_points(4)
    state = {"cleared_channels": set(), "excluded_channels": set(), "unresolved_channels": set(),
             "measure_count": 0, "clear_count": 0, "extra_bearing_count": 0,
             "fallback_count": 0, "fallback_clear_count": 0, "warnings": [], "stopped": False,
             "coverage_points_total": len(points),
             "coverage_measurements_by_channel": {channel: 0 for channel in range(1, 21)},
             "recovery_count": 0, "recovery_measure_count": 0, "recovery_positive_count": 0,
             "polygon_fallback_count": 0, "polygon_fallback_point_count": 0,
             "polygon_fallback_clear_count": 0, "polygon_fallback_exhausted_count": 0,
             "polygon_fallback_invalid_count": 0, "legacy_fallback_count": 0,
             "legacy_fallback_clear_count": 0}
    seen = set()
    try:
        for scan_point in points:
            for channel in range(1, 21):
                if channel in state["cleared_channels"]:
                    continue
                response = client.measure(scan_point[0], scan_point[1], channel)
                state["measure_count"] += 1
                state["coverage_measurements_by_channel"][channel] += 1
                kind = response.get("measure_result")
                if kind == "no_signal":
                    continue
                seen.add(channel)
                if kind == "near":
                    _resolve_clear(client, scan_point, channel, state, "near coverage point")
                elif kind == "direction":
                    _handle_direction(client, scan_point, response, channel, state)
                else:
                    state["warnings"].append(f"channel {channel}: unknown measure result {kind!r}")
                    state["unresolved_channels"].add(channel)
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
    state["cleared_channels"] = sorted(state["cleared_channels"])
    state["excluded_channels"] = sorted(state["excluded_channels"])
    state["unresolved_channels"] = sorted(state["unresolved_channels"])
    state["completed"] = (not state["stopped"] and not state["unresolved_channels"] and
                          len(state["cleared_channels"]) + len(state["excluded_channels"]) == 20)
    state["question"], state["method"], state["version"] = 4, "main", VERSION
    state["virtual_time_s"] = getattr(client, "virtual_time_s", None)
    return state
