"""Q4 round-2 recovery policy; v1 search_policy remains untouched."""

import math

from search_policy import (LOCATE_CLEAR_RADIUS_M, MAX_EXTRA_BEARINGS, _add_bearing,
                           _bearing_triangle, _box_circle, _clear_points,
                           _coverage_points, _resolve_clear)


def _valid_svd(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0.0 <= value < 360.0)


def _fallback_scan(client, initial_point, initial_svd, channel, state):
    state["fallback_count"] += 1
    for point in _clear_points(initial_point, initial_svd):
        state["fallback_clear_count"] += 1
        if _resolve_clear(client, point, channel, state, "fallback scan"):
            return True
    state["warnings"].append(f"channel {channel}: all 304 fallback clear points exhausted")
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
            return _fallback_scan(client, point, initial_svd, channel, state)
        candidate, radius = box
        if radius <= LOCATE_CLEAR_RADIUS_M:
            if _resolve_clear(client, candidate, channel, state, "bearing localization"):
                return True
            return _fallback_scan(client, point, initial_svd, channel, state)
        if extra_used >= MAX_EXTRA_BEARINGS:
            state["warnings"].append(f"channel {channel}: extra bearing limit reached")
            return _fallback_scan(client, point, initial_svd, channel, state)
        if previous_candidate and math.dist(candidate, previous_candidate) <= 1e-6:
            state["warnings"].append(f"channel {channel}: repeated localization center")
            return _fallback_scan(client, point, initial_svd, channel, state)
        previous_candidate = candidate
        follow_up = _measure(client, candidate, channel, state)
        extra_used += 1
        kind = follow_up.get("measure_result")
        if kind == "near":
            if _resolve_clear(client, candidate, channel, state, "near follow-up"):
                return True
            return _fallback_scan(client, point, initial_svd, channel, state)
        if kind == "direction":
            svd = follow_up.get("svd_deg")
            if not _valid_svd(svd):
                state["warnings"].append(f"channel {channel}: follow-up direction lacks valid svd_deg")
                return _fallback_scan(client, point, initial_svd, channel, state)
            polygon = _add_bearing(polygon, candidate, svd)
            last_visible = candidate
            continue
        if kind != "no_signal":
            state["warnings"].append(f"channel {channel}: unknown follow-up result {kind!r}")
            return _fallback_scan(client, point, initial_svd, channel, state)

        # A no_signal only brackets a recovery segment; it never clips polygon.
        state["recovery_count"] += 1
        invisible = candidate
        while True:
            # Retain (last_visible, invisible) while checking whether positive
            # bearings have already made the feasible polygon clearable.
            box = _box_circle(polygon)
            if box is None:
                state["warnings"].append(f"channel {channel}: empty recovery polygon")
                return _fallback_scan(client, point, initial_svd, channel, state)
            recovery_center, recovery_radius = box
            if recovery_radius <= LOCATE_CLEAR_RADIUS_M:
                if _resolve_clear(client, recovery_center, channel, state, "recovery localization"):
                    return True
                return _fallback_scan(client, point, initial_svd, channel, state)
            if extra_used >= MAX_EXTRA_BEARINGS:
                state["warnings"].append(f"channel {channel}: recovery budget exhausted")
                return _fallback_scan(client, point, initial_svd, channel, state)
            midpoint = ((last_visible[0] + invisible[0]) / 2.0,
                        (last_visible[1] + invisible[1]) / 2.0)
            if math.dist(midpoint, last_visible) <= 1e-6 or math.dist(midpoint, invisible) <= 1e-6:
                state["warnings"].append(f"channel {channel}: recovery midpoint repeated")
                return _fallback_scan(client, point, initial_svd, channel, state)
            binary_response = _measure(client, midpoint, channel, state, recovery=True)
            extra_used += 1
            binary_kind = binary_response.get("measure_result")
            if binary_kind == "near":
                if _resolve_clear(client, midpoint, channel, state, "near recovery"):
                    return True
                return _fallback_scan(client, point, initial_svd, channel, state)
            if binary_kind == "direction":
                svd = binary_response.get("svd_deg")
                if not _valid_svd(svd):
                    state["warnings"].append(f"channel {channel}: recovery direction lacks valid svd_deg")
                    return _fallback_scan(client, point, initial_svd, channel, state)
                polygon = _add_bearing(polygon, midpoint, svd)
                last_visible = midpoint
                state["recovery_positive_count"] += 1
                continue
            if binary_kind == "no_signal":
                invisible = midpoint
                continue
            state["warnings"].append(f"channel {channel}: unknown recovery result {binary_kind!r}")
            return _fallback_scan(client, point, initial_svd, channel, state)


def run_search(client, question=4, method="main"):
    """Run Q4 recovery policy. The caller retains /enter and /exit ownership."""
    if question != 4 or method != "main":
        raise ValueError("recovery policy supports only question=4, method='main'")
    coverage_points = _coverage_points(4)
    state = {"cleared_channels": set(), "excluded_channels": set(), "unresolved_channels": set(),
             "measure_count": 0, "clear_count": 0, "extra_bearing_count": 0,
             "fallback_count": 0, "fallback_clear_count": 0, "warnings": [], "stopped": False,
             "coverage_points_total": len(coverage_points),
             "coverage_measurements_by_channel": {channel: 0 for channel in range(1, 21)},
             "recovery_count": 0, "recovery_measure_count": 0, "recovery_positive_count": 0}
    seen = set()
    try:
        for scan_point in coverage_points:
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
    state["question"], state["method"] = 4, "main"
    state["virtual_time_s"] = getattr(client, "virtual_time_s", None)
    return state
