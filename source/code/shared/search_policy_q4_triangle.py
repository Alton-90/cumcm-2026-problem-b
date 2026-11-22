"""Q4 batch-localization candidate with a certified 980 m triangular coverage set.

The coverage set is the union of all vertices of equilateral triangles that
intersect the closed 1800 m source disk.  It uses no hidden source information:
the disk radius and minimum reception radius are fixed task constraints.
"""

from __future__ import annotations

import math

import search_policy_polygon as polygon_policy
from search_policy import _bearing_triangle, _box_circle, _resolve_clear
from search_policy_q4_batch import _TrackedClient, _localize, _state


VERSION = "q4_triangle980_v1"
TRIANGLE_SIDE_M = 980.0
SOURCE_RADIUS_M = 1800.0
MIN_RECEIVE_RADIUS_M = 1000.0
GRID_HEIGHT_M = TRIANGLE_SIDE_M * math.sqrt(3.0) / 2.0
GRID_INDEX_LIMIT = 10


def _vertex(i, j):
    return (TRIANGLE_SIDE_M * (i + 0.5 * j), GRID_HEIGHT_M * j)


def _distance_origin_to_segment(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.hypot(a[0], a[1])
    fraction = max(0.0, min(1.0, -(a[0] * dx + a[1] * dy) / denominator))
    return math.hypot(a[0] + fraction * dx, a[1] + fraction * dy)


def _origin_in_triangle(triangle):
    signs = []
    for index, point in enumerate(triangle):
        other = triangle[(index + 1) % 3]
        signs.append(point[0] * other[1] - point[1] * other[0])
    return min(signs) >= -1e-9 or max(signs) <= 1e-9


def _triangle_intersects_closed_source_disk(triangle):
    if _origin_in_triangle(triangle):
        return True
    return min(_distance_origin_to_segment(triangle[index], triangle[(index + 1) % 3])
               for index in range(3)) <= SOURCE_RADIUS_M + 1e-9


def triangle_coverage_points():
    """Return all retained vertices in deterministic row-snake order, origin first.

    Any triangle that contains a source point in the closed source disk itself
    intersects the disk and is retained.  All incident triangles are retained
    for sources on grid edges/vertices because intersection is inclusive.
    """
    vertices = set()
    for i in range(-GRID_INDEX_LIMIT, GRID_INDEX_LIMIT + 1):
        for j in range(-GRID_INDEX_LIMIT, GRID_INDEX_LIMIT + 1):
            a, b, c, d = _vertex(i, j), _vertex(i + 1, j), _vertex(i, j + 1), _vertex(i + 1, j + 1)
            for triangle in ((a, b, c), (b, d, c)):
                if _triangle_intersects_closed_source_disk(triangle):
                    vertices.update(triangle)
    rows = {}
    for point in vertices:
        row = round(point[1] / GRID_HEIGHT_M)
        rows.setdefault(row, []).append(point)
    ordered = []
    for row in sorted(rows):
        row_points = sorted(rows[row])
        if (row - min(rows)) % 2:
            row_points.reverse()
        ordered.extend(row_points)
    origin = (0.0, 0.0)
    return [origin] + [point for point in ordered if point != origin]


def _nearest_route(points):
    """Deterministic nearest-neighbour traversal; coverage is order independent."""
    remaining = list(points)
    current = (0.0, 0.0)
    route = []
    while remaining:
        point = min(remaining, key=lambda candidate: (math.dist(current, candidate), candidate[1], candidate[0]))
        remaining.remove(point)
        route.append(point)
        current = point
    return route


def run_search(client, question=4, method="main"):
    if question != 4 or method != "main":
        raise ValueError("q4_triangle policy supports only question=4, method='main'")
    points = triangle_coverage_points()
    state = _state(points)
    state["coverage_geometry"] = {"triangle_side_m": TRIANGLE_SIDE_M, "source_radius_m": SOURCE_RADIUS_M,
                                  "minimum_receive_radius_m": MIN_RECEIVE_RADIUS_M,
                                  "directional_vertex_distance_bound_m": TRIANGLE_SIDE_M}
    tracked = _TrackedClient(client)
    seen = set()
    try:
        for scan_point in _nearest_route(points):
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
