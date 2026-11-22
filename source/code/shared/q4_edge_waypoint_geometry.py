"""Public-only geometry for the frozen Q4 edge-waypoint v1 experiment.

The visibility certificate is sufficient, not necessary.  It combines the
continuous positive-half-plane semantics of ``q4_joint_compatibility`` with
the public 1000 m lower reception-range bound.  Compatibility at the public
polygon center is exposed only as a deterministic ordering diagnostic.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from q4_joint_compatibility import MIN_RANGE_M, check_fixed_position


MIN_SCAN_EDGE_M = 1000.0
WAYPOINT_FRACTIONS = (0.50, 0.25, 0.75)
HISTORY_SEPARATION_M = 1.0
HISTORY_SEPARATION_TOLERANCE_M = 1e-9
HULL_CROSS_MARGIN_M2 = 1e-6
RANGE_MARGIN_M = 1e-6
ROUTE_BOUND_MARGIN_M = 1e-7


def _point(value: Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("point must contain two coordinates")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError("point coordinates must be finite")
    return result


def _cross(origin, first, second) -> float:
    return ((first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0]))


def convex_hull(points: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    """Return unique extreme vertices counter-clockwise without collinear middles."""

    values = sorted({_point(point) for point in points})
    if len(values) <= 1:
        return values
    lower: list[tuple[float, float]] = []
    for point in values:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(values):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def visibility_certificate(
        waypoint: Sequence[float], polygon: Iterable[Sequence[float]],
        positive_points: Iterable[Sequence[float]],
        measured_points: Iterable[Sequence[float]]) -> dict:
    """Certify visibility for every source state represented by the public track."""

    waypoint = _point(waypoint)
    polygon = [_point(point) for point in polygon]
    positives = [_point(point) for point in positive_points]
    measured = [_point(point) for point in measured_points]
    hull = convex_hull(positives)
    crosses = [
        _cross(hull[index], hull[(index + 1) % len(hull)], waypoint)
        for index in range(len(hull))
    ] if len(hull) >= 3 else []
    minimum_cross = min(crosses) if crosses else None
    strict_hull = bool(
        len(hull) >= 3
        and minimum_cross is not None
        and minimum_cross >= HULL_CROSS_MARGIN_M2
    )
    max_polygon_distance = max(
        (math.dist(waypoint, vertex) for vertex in polygon), default=math.inf
    )
    range_ok = max_polygon_distance <= MIN_RANGE_M - RANGE_MARGIN_M
    minimum_history_distance = min(
        (math.dist(waypoint, old) for old in measured), default=math.inf
    )
    separated = minimum_history_distance > (
        HISTORY_SEPARATION_M + HISTORY_SEPARATION_TOLERANCE_M
    )
    checks = {
        "nonempty_public_polygon": bool(polygon),
        "three_noncollinear_positive_hull_vertices": len(hull) >= 3,
        "strict_positive_hull_interior": strict_hull,
        "all_polygon_vertices_within_minimum_range": range_ok,
        "same_channel_history_distance_strictly_gt_1m": separated,
    }
    return {
        "certified": all(checks.values()),
        "checks": checks,
        "waypoint": list(waypoint),
        "positive_hull": [list(point) for point in hull],
        "positive_point_count": len(positives),
        "minimum_hull_cross_m2": minimum_cross,
        "hull_cross_margin_m2": HULL_CROSS_MARGIN_M2,
        "maximum_polygon_vertex_distance_m": max_polygon_distance,
        "minimum_range_m": MIN_RANGE_M,
        "range_margin_m": MIN_RANGE_M - max_polygon_distance,
        "minimum_same_channel_history_distance_m": minimum_history_distance,
        "history_separation_m": HISTORY_SEPARATION_M,
        "proof": (
            "positive_receiver_convex_hull_is_inside_every_compatible_directional_"
            "heading_halfplane_and_every_feasible_source_is_within_minimum_range"
        ),
    }


def _compatibility(center, history) -> dict:
    try:
        result = check_fixed_position(center, history, source_type="either")
        return result.to_dict()
    except (TypeError, ValueError) as error:
        return {
            "status": "uncertain",
            "keep_candidate": True,
            "source_type": "either",
            "uncertainty_reasons": [f"public compatibility diagnostic error: {error}"],
        }


def select_waypoint(start: Sequence[float], end: Sequence[float], records: Iterable[Mapping]) -> dict:
    """Select one certified waypoint deterministically from public records only."""

    start, end = _point(start), _point(end)
    edge_length = math.dist(start, end)
    records = list(records)
    eligible = []
    considered = 0
    if edge_length >= MIN_SCAN_EDGE_M:
        for record in sorted(records, key=lambda item: int(item["channel"])):
            center = _point(record["center"])
            compatibility = _compatibility(center, record["history"])
            status_rank = {"compatible": 0, "uncertain": 1, "incompatible": 2}.get(
                compatibility.get("status"), 1
            )
            for fraction_rank, fraction in enumerate(WAYPOINT_FRACTIONS):
                considered += 1
                waypoint = (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
                certificate = visibility_certificate(
                    waypoint, record["polygon"], record["positive_points"],
                    record["measured_points"]
                )
                if not certificate["certified"]:
                    continue
                score = (
                    status_rank,
                    -float(record["radius"]),
                    -float(certificate["minimum_same_channel_history_distance_m"]),
                    -float(certificate["range_margin_m"]),
                    fraction_rank,
                    int(record["channel"]),
                )
                eligible.append({
                    "channel": int(record["channel"]),
                    "fraction": fraction,
                    "fraction_rank": fraction_rank,
                    "waypoint": list(waypoint),
                    "track_center": list(center),
                    "track_radius_m": float(record["radius"]),
                    "polygon": [list(_point(point)) for point in record["polygon"]],
                    "positive_points": [list(_point(point)) for point in record["positive_points"]],
                    "measured_points": [list(_point(point)) for point in record["measured_points"]],
                    "public_history": [dict(item) for item in record["history"]],
                    "compatibility_center_diagnostic": compatibility,
                    "compatibility_used_only_for_ordering": True,
                    "certificate": certificate,
                    "score": list(score),
                })
    eligible.sort(key=lambda item: tuple(item["score"]))
    return {
        "edge_start": list(start),
        "edge_end": list(end),
        "edge_length_m": edge_length,
        "minimum_edge_length_m": MIN_SCAN_EDGE_M,
        "record_count": len(records),
        "fraction_count_per_record": len(WAYPOINT_FRACTIONS),
        "considered_candidate_count": considered,
        "eligible_candidate_count": len(eligible),
        "eligible_candidates": eligible,
        "selected": None if not eligible else eligible[0],
    }


def node_key(node) -> tuple[str, int]:
    return str(node[0]), int(node[1])


def route_keys(route) -> list[list[object]]:
    return [[str(node[0]), int(node[1])] for node in route]


def route_connector_upper_bound(position, route, radius_by_key: Mapping) -> float:
    """Worst-case connector bound for exact scan points and source regions."""

    previous_center = _point(position)
    previous_radius = 0.0
    total = 0.0
    for node in route:
        center = _point(node[2])
        radius = max(0.0, float(radius_by_key.get(node_key(node), 0.0)))
        total += math.dist(previous_center, center) + previous_radius + radius
        previous_center, previous_radius = center, radius
    return total


def split_edge_certificate(start, waypoint, end) -> dict:
    start, waypoint, end = _point(start), _point(waypoint), _point(end)
    original = math.dist(start, end)
    first = math.dist(start, waypoint)
    second = math.dist(waypoint, end)
    cross = _cross(start, end, waypoint)
    residual = first + second - original
    return {
        "original_edge_m": original,
        "first_segment_m": first,
        "second_segment_m": second,
        "split_sum_m": first + second,
        "split_residual_m": residual,
        "cross_product_m2": cross,
        "on_segment": abs(cross) <= 1e-7 and abs(residual) <= 1e-7,
    }


__all__ = [
    "HISTORY_SEPARATION_M", "HULL_CROSS_MARGIN_M2", "MIN_SCAN_EDGE_M",
    "RANGE_MARGIN_M", "ROUTE_BOUND_MARGIN_M", "WAYPOINT_FRACTIONS",
    "convex_hull", "node_key", "route_connector_upper_bound", "route_keys",
    "select_waypoint", "split_edge_certificate", "visibility_certificate",
]
