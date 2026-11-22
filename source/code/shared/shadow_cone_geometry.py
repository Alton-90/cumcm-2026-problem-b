"""Exact-rational conservative shadow-cone pruning for Q4 localization."""
from __future__ import annotations

from fractions import Fraction
import itertools
import math
from time import perf_counter

DELTA_M = Fraction(1, 10_000)
PAIR_REL_TOL = 1e-12


def _f(value):
    return value if isinstance(value, Fraction) else Fraction.from_float(float(value))


def _point(point):
    return _f(point[0]), _f(point[1])


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def _area2(polygon):
    return sum(_cross(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon)))


def _hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def build(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2 and _cross(_sub(result[-1], result[-2]), _sub(point, result[-1])) <= 0:
                result.pop()
            result.append(point)
        return result

    lower = build(points)
    upper = build(reversed(points))
    return lower[:-1] + upper[:-1]


def _clip(polygon, coefficients):
    if not polygon:
        return []
    a, b, c = coefficients

    def value(point):
        return a * point[0] + b * point[1] + c

    output = []
    previous = polygon[-1]
    previous_value = value(previous)
    previous_inside = previous_value <= 0
    for current in polygon:
        current_value = value(current)
        current_inside = current_value <= 0
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if denominator:
                t = previous_value / denominator
                output.append(
                    (
                        previous[0] + t * (current[0] - previous[0]),
                        previous[1] + t * (current[1] - previous[1]),
                    )
                )
        if current_inside:
            output.append(current)
        previous, previous_value, previous_inside = current, current_value, current_inside
    return output


def _contains_convex(polygon, point):
    if len(polygon) < 3:
        return False
    orientation = 1 if _area2(polygon) > 0 else -1
    return all(
        orientation * _cross(_sub(polygon[(i + 1) % len(polygon)], polygon[i]), _sub(point, polygon[i])) >= 0
        for i in range(len(polygon))
    )


def _on_segment(start, end, point):
    if _cross(_sub(end, start), _sub(point, start)) != 0:
        return False
    return (
        min(start[0], end[0]) <= point[0] <= max(start[0], end[0])
        and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])
    )


def _select_pair(positives, negative):
    unique = list(dict.fromkeys(_point(point) for point in positives))
    b = _point(negative)
    if b in unique:
        return "negative_inside_positive_hull", None
    if len(unique) < 2:
        return "insufficient_positive", None
    positive_hull = _hull(unique)
    if (
        _contains_convex(positive_hull, b)
        or (len(positive_hull) == 2 and _on_segment(positive_hull[0], positive_hull[1], b))
    ):
        return "negative_inside_positive_hull", None
    best = None
    for first, second in itertools.combinations(unique, 2):
        u, v = _sub(b, first), _sub(b, second)
        uf, vf = tuple(map(float, u)), tuple(map(float, v))
        nu, nv = math.hypot(*uf), math.hypot(*vf)
        if nu == 0 or nv == 0:
            continue
        cross_value = _cross(u, v)
        if abs(float(cross_value)) <= PAIR_REL_TOL * nu * nv:
            continue
        if cross_value < 0:
            first, second, u, v = second, first, v, u
            cross_value = -cross_value
        angle = math.atan2(float(cross_value), float(u[0] * v[0] + u[1] * v[1]))
        if angle <= 1e-9 or math.pi - angle <= 1e-9:
            continue
        candidate = (angle, first, second, u, v)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return ("ok", best) if best is not None else ("degenerate_positive_pair", None)


def _binary_float_interval(value):
    """Smallest adjacent-binary-float interval known to contain an exact rational."""
    rounded = float(value)
    rounded_exact = Fraction.from_float(rounded)
    low = math.nextafter(rounded, -math.inf) if rounded_exact > value else rounded
    high = math.nextafter(rounded, math.inf) if rounded_exact < value else rounded
    return low, high


def _outward_float_polygon(polygon):
    """Convert an exact convex polygon to a provable binary-float convex outer hull."""
    corners = []
    for x, y in polygon:
        low_x, high_x = _binary_float_interval(x)
        low_y, high_y = _binary_float_interval(y)
        for corner_x, corner_y in itertools.product((low_x, high_x), (low_y, high_y)):
            corners.append((Fraction.from_float(corner_x), Fraction.from_float(corner_y)))
    return [tuple(map(float, point)) for point in _hull(corners)]


def polygon_area(polygon):
    if not polygon:
        return 0.0
    return abs(
        sum(
            polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
            - polygon[i][1] * polygon[(i + 1) % len(polygon)][0]
            for i in range(len(polygon))
        )
    ) / 2


def contains_point(polygon, point, tolerance=1e-7):
    if len(polygon) < 3:
        return False
    orientation = 1 if sum(
        polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
        - polygon[i][1] * polygon[(i + 1) % len(polygon)][0]
        for i in range(len(polygon))
    ) > 0 else -1
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        cross_value = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
        if orientation * cross_value < -tolerance:
            return False
    return True


def shadow_outer_polygon(polygon, positives, negative, delta=DELTA_M):
    """Return a conservative convex outer polygon after one negative response."""
    started = perf_counter()
    if not isinstance(delta, Fraction):
        delta = _f(delta)
    source = [_point(point) for point in polygon]
    if len(source) < 3 or _area2(source) == 0:
        return {"status": "invalid_input_polygon", "polygon": polygon, "cpu_s": perf_counter() - started}
    status, selection = _select_pair(positives, negative)
    if selection is None:
        return {"status": status, "polygon": polygon, "cpu_s": perf_counter() - started}
    angle, first, second, u, v = selection
    b = _point(negative)
    margin_u = delta * (abs(u[0]) + abs(u[1]))
    margin_v = delta * (abs(v[0]) + abs(v[1]))
    first_halfplane = (-u[1], u[0], u[1] * b[0] - u[0] * b[1] - margin_u)
    second_halfplane = (v[1], -v[0], -b[0] * v[1] + b[1] * v[0] - margin_v)
    left = _clip(source, first_halfplane)
    right = _clip(source, second_halfplane)
    exact_hull = _hull(left + right)
    if len(exact_hull) < 3 or _area2(exact_hull) == 0:
        return {
            "status": "empty_or_degenerate_result",
            "polygon": polygon,
            "pair": [tuple(map(float, first)), tuple(map(float, second))],
            "cpu_s": perf_counter() - started,
        }
    output = _outward_float_polygon(exact_hull)
    before_area, after_area = polygon_area(polygon), polygon_area(output)
    strict = after_area < before_area - max(1e-6, before_area * 1e-10)
    return {
        "status": "updated" if strict else "no_strict_shrink",
        "polygon": output if strict else polygon,
        "pair": [tuple(map(float, first)), tuple(map(float, second))],
        "angle_deg": math.degrees(angle),
        "area_before": before_area,
        "area_after": after_area,
        "strict_shrink": strict,
        "delta_m": float(delta),
        "cpu_s": perf_counter() - started,
    }
