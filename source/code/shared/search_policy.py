"""Approved Q3/Q4 fixed-cover policy; it never calls /enter or /exit itself."""

import math


EPSILON_DEG = 1.005
LOCATE_CLEAR_RADIUS_M = 19.9
MAX_EXTRA_BEARINGS = 8
POLY_TOL = 1e-8


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _point_on_edge(a, b, va, vb):
    denominator = va - vb
    if abs(denominator) < POLY_TOL:
        return a
    fraction = va / denominator
    return (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))


def _clip_halfplane(polygon, origin, ray, keep_nonnegative):
    """Clip a convex polygon to cross(ray, point-origin)>=0 or <=0."""
    if not polygon:
        return []
    output = []
    previous = polygon[-1]
    previous_value = _cross(ray, _sub(previous, origin)) * keep_nonnegative
    previous_inside = previous_value >= -POLY_TOL
    for current in polygon:
        current_value = _cross(ray, _sub(current, origin)) * keep_nonnegative
        current_inside = current_value >= -POLY_TOL
        if current_inside != previous_inside:
            output.append(_point_on_edge(previous, current, previous_value, current_value))
        if current_inside:
            output.append(current)
        previous, previous_value, previous_inside = current, current_value, current_inside
    return output


def _bearing_triangle(point, svd_deg):
    epsilon = math.radians(EPSILON_DEG)
    theta = math.radians(float(svd_deg))
    axis = (math.cos(theta), math.sin(theta))
    lateral = (-axis[1], axis[0])
    end = (point[0] + 1500.0 * axis[0], point[1] + 1500.0 * axis[1])
    half_width = 1500.0 * math.tan(epsilon)
    return [point, (end[0] + half_width * lateral[0], end[1] + half_width * lateral[1]),
            (end[0] - half_width * lateral[0], end[1] - half_width * lateral[1])]


def _add_bearing(polygon, point, svd_deg):
    epsilon = math.radians(EPSILON_DEG)
    theta = math.radians(float(svd_deg))
    lower = (math.cos(theta - epsilon), math.sin(theta - epsilon))
    upper = (math.cos(theta + epsilon), math.sin(theta + epsilon))
    return _clip_halfplane(_clip_halfplane(polygon, point, lower, 1.0), point, upper, -1.0)


def _box_circle(polygon):
    if not polygon:
        return None
    min_x, max_x = min(p[0] for p in polygon), max(p[0] for p in polygon)
    min_y, max_y = min(p[1] for p in polygon), max(p[1] for p in polygon)
    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    radius = max(math.hypot(p[0] - center[0], p[1] - center[1]) for p in polygon)
    return center, radius


def _coverage_points(question):
    step = 1200 if question == 3 else 600
    points = []
    for row, y in enumerate(range(-2400, 2401, step)):
        xs = list(range(-2400, 2401, step))
        if row % 2:
            xs.reverse()
        points.extend((float(x), float(y)) for x in xs)
    origin = (0.0, 0.0)
    return [origin] + [p for p in points if p != origin]


def _clear_points(initial_point, svd_deg):
    """304 points in the original observation coordinate system, snake ordered."""
    theta = math.radians(float(svd_deg))
    axis, lateral = (math.cos(theta), math.sin(theta)), (-math.sin(theta), math.cos(theta))
    points = []
    for row, q in enumerate((-30.0, -10.0, 10.0, 30.0)):
        values = range(0, 1501, 20)
        if row % 2:
            values = reversed(list(values))
        for t in values:
            points.append((initial_point[0] + t * axis[0] + q * lateral[0],
                           initial_point[1] + t * axis[1] + q * lateral[1]))
    return points


def _resolve_clear(client, point, channel, state, reason):
    response = client.clear(point[0], point[1], channel)
    state["clear_count"] += 1
    if response.get("clear_result") == "success":
        state["cleared_channels"].add(channel)
        state["unresolved_channels"].discard(channel)
        return True
    if reason != "fallback scan":
        state["warnings"].append(f"channel {channel}: clear failed after {reason}")
    return False


def _fallback_scan(client, initial_point, initial_svd, channel, state):
    state["fallback_count"] += 1
    for point in _clear_points(initial_point, initial_svd):
        state["fallback_clear_count"] += 1
        if _resolve_clear(client, point, channel, state, "fallback scan"):
            return True
    state["warnings"].append(f"channel {channel}: all 304 fallback clear points exhausted")
    return False


def _handle_direction(client, point, response, channel, state, method):
    svd = response.get("svd_deg")
    if isinstance(svd, bool) or not isinstance(svd, (int, float)) or not math.isfinite(svd) or not 0 <= svd < 360:
        state["warnings"].append(f"channel {channel}: direction response lacks numeric svd_deg")
        return False
    if method == "baseline":
        return _fallback_scan(client, point, svd, channel, state)
    polygon = _bearing_triangle(point, svd)
    previous_center = None
    for extra_used in range(MAX_EXTRA_BEARINGS + 1):
        box = _box_circle(polygon)
        if box is None:
            state["warnings"].append(f"channel {channel}: empty bearing polygon")
            return _fallback_scan(client, point, svd, channel, state)
        center, radius = box
        if radius <= LOCATE_CLEAR_RADIUS_M:
            if _resolve_clear(client, center, channel, state, "bearing localization"):
                return True
            return _fallback_scan(client, point, svd, channel, state)
        if extra_used >= MAX_EXTRA_BEARINGS:
            state["warnings"].append(f"channel {channel}: extra bearing limit reached")
            return _fallback_scan(client, point, svd, channel, state)
        if previous_center and math.hypot(center[0] - previous_center[0], center[1] - previous_center[1]) <= 1e-6:
            state["warnings"].append(f"channel {channel}: repeated localization center")
            return _fallback_scan(client, point, svd, channel, state)
        previous_center = center
        follow_up = client.measure(center[0], center[1], channel)
        state["measure_count"] += 1
        state["extra_bearing_count"] += 1
        kind = follow_up.get("measure_result")
        if kind == "near":
            if _resolve_clear(client, center, channel, state, "near follow-up"):
                return True
            return _fallback_scan(client, point, svd, channel, state)
        if kind == "direction":
            next_svd = follow_up.get("svd_deg")
            if isinstance(next_svd, bool) or not isinstance(next_svd, (int, float)) or not math.isfinite(next_svd) or not 0 <= next_svd < 360:
                state["warnings"].append(f"channel {channel}: follow-up direction lacks svd_deg")
                return _fallback_scan(client, point, svd, channel, state)
            polygon = _add_bearing(polygon, center, next_svd)
            continue
        state["warnings"].append(f"channel {channel}: follow-up no_signal/unknown; retaining fallback")
        return _fallback_scan(client, point, svd, channel, state)
    raise AssertionError("unreachable extra-bearing loop")


def run_search(client, question=3, method="main"):
    """Run approved policy using only public client calls; outer code owns enter/exit."""
    if question not in (3, 4):
        raise ValueError("question must be 3 or 4")
    if method not in ("main", "baseline"):
        raise ValueError("method must be 'main' or 'baseline'")
    state = {"cleared_channels": set(), "excluded_channels": set(), "unresolved_channels": set(),
             "measure_count": 0, "clear_count": 0, "extra_bearing_count": 0,
             "fallback_count": 0, "fallback_clear_count": 0, "warnings": [], "stopped": False,
             "coverage_points_total": len(_coverage_points(question)),
             "coverage_measurements_by_channel": {channel: 0 for channel in range(1, 21)}}
    seen = set()
    try:
        for scan_point in _coverage_points(question):
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
                    _handle_direction(client, scan_point, response, channel, state, method)
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
    state["question"], state["method"] = question, method
    state["virtual_time_s"] = getattr(client, "virtual_time_s", None)
    return state
