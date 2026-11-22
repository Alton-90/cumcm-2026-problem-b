"""Conservative Q3 local parallax scoring over continuous bearing polygons."""
from __future__ import annotations

import itertools
import math

from search_policy import _clip_halfplane


OFFSETS_M = (25.0, 50.0, 100.0, 200.0)
RECEPTION_LIMIT_M = 999.9
MOVEMENT_SLACK_M = 50.0
BASE_HALF_ANGLE_DEG = 1.005
BIN_HALF_WIDTH_DEG = 0.5
ROBUST_HALF_ANGLE_DEG = BASE_HALF_ANGLE_DEG + BIN_HALF_WIDTH_DEG
NEAR_DIAMETER_BOUND_M = 10.0
ADOPTION_RATIO = 0.8
BOUND_TIE_TOLERANCE_M = 1e-6


def polygon_diameter(polygon):
    if len(polygon) < 2:
        return 0.0
    return max(math.dist(first, second) for first, second in itertools.combinations(polygon, 2))


def clip_angle_bin(polygon, point, center_deg, half_angle_deg=ROBUST_HALF_ANGLE_DEG):
    theta = math.radians(float(center_deg))
    half = math.radians(float(half_angle_deg))
    lower = (math.cos(theta - half), math.sin(theta - half))
    upper = (math.cos(theta + half), math.sin(theta + half))
    return _clip_halfplane(
        _clip_halfplane(polygon, point, lower, 1.0), point, upper, -1.0
    )


def robust_diameter_bound(polygon, point):
    """Upper-bound every possible 0.01-degree reading via 360 adjacent integer bins."""
    maximum = NEAR_DIAMETER_BOUND_M
    nonempty_bins = 0
    maximizing_bin = None
    for center_deg in range(360):
        clipped = clip_angle_bin(polygon, point, center_deg)
        if not clipped:
            continue
        nonempty_bins += 1
        diameter = polygon_diameter(clipped)
        if diameter > maximum:
            maximum = diameter
            maximizing_bin = center_deg
    return {
        "upper_bound_m": maximum,
        "nonempty_bin_count": nonempty_bins,
        "maximizing_bin_deg": maximizing_bin,
        "near_diameter_bound_m": NEAR_DIAMETER_BOUND_M,
        "robust_half_angle_deg": ROBUST_HALF_ANGLE_DEG,
        "integer_bin_count": 360,
    }


def maximum_vertex_distance(point, polygon):
    if not polygon:
        return math.inf
    return max(math.dist(point, vertex) for vertex in polygon)


def choose_measurement_point(polygon, original_center, current_position, initial_svd_deg):
    axis_angle = math.radians(float(initial_svd_deg))
    normal = (-math.sin(axis_angle), math.cos(axis_angle))
    original_move = math.dist(current_position, original_center)
    move_cap = original_move + MOVEMENT_SLACK_M
    original_score = robust_diameter_bound(polygon, original_center)
    records = [
        {
            "label": "original_center",
            "point": [float(original_center[0]), float(original_center[1])],
            "offset_m": 0.0,
            "movement_m": original_move,
            "maximum_vertex_distance_m": maximum_vertex_distance(original_center, polygon),
            "movement_cap_pass": True,
            "guaranteed_reception_pass": maximum_vertex_distance(original_center, polygon)
            <= RECEPTION_LIMIT_M + 1e-9,
            **original_score,
        }
    ]
    feasible = []
    for offset in OFFSETS_M:
        for sign in (-1.0, 1.0):
            point = (
                original_center[0] + sign * offset * normal[0],
                original_center[1] + sign * offset * normal[1],
            )
            movement = math.dist(current_position, point)
            maximum = maximum_vertex_distance(point, polygon)
            movement_ok = movement <= move_cap + 1e-9
            reception_ok = maximum <= RECEPTION_LIMIT_M + 1e-9
            record = {
                "label": f"normal_{'plus' if sign > 0 else 'minus'}_{int(offset)}m",
                "point": [float(point[0]), float(point[1])],
                "offset_m": sign * offset,
                "movement_m": movement,
                "maximum_vertex_distance_m": maximum,
                "movement_cap_pass": movement_ok,
                "guaranteed_reception_pass": reception_ok,
                "upper_bound_m": None,
                "nonempty_bin_count": None,
                "maximizing_bin_deg": None,
            }
            if movement_ok and reception_ok:
                score = robust_diameter_bound(polygon, point)
                record.update(score)
                feasible.append(record)
            records.append(record)

    best = None
    for record in feasible:
        if best is None:
            best = record
            continue
        bound_difference = record["upper_bound_m"] - best["upper_bound_m"]
        if bound_difference < -BOUND_TIE_TOLERANCE_M or (
            abs(bound_difference) <= BOUND_TIE_TOLERANCE_M
            and record["movement_m"] < best["movement_m"] - 1e-9
        ):
            best = record
    threshold = ADOPTION_RATIO * original_score["upper_bound_m"]
    selected = best is not None and best["upper_bound_m"] <= threshold + 1e-9
    point = tuple(best["point"]) if selected else tuple(original_center)
    return {
        "selected_point": point,
        "used_offset": selected,
        "selection_reason": "robust_bound_ratio_pass" if selected else "retain_original_center",
        "normal": [normal[0], normal[1]],
        "move_cap_m": move_cap,
        "original_upper_bound_m": original_score["upper_bound_m"],
        "adoption_threshold_m": threshold,
        "best_offset_label": None if best is None else best["label"],
        "best_offset_upper_bound_m": None if best is None else best["upper_bound_m"],
        "selected_maximum_vertex_distance_m": maximum_vertex_distance(point, polygon),
        "candidates": records,
    }
