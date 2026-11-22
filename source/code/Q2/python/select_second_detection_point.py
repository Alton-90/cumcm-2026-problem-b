"""Generic robust selector for the second bearing-measurement point in Q2.

The program uses only information available after the first measurement.  It
constructs a conservative outer polygon for the possible source set, an inner
polygon for the guaranteed-reception region, and a deterministic finite-grid
approximation to the movement--worst-posterior-diameter frontier.

The angular and spatial discretisations are numerical certificates/bounds, not
probability assumptions and not a claim of continuous global optimality.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


Point = tuple[float, float]
EPS = 1.0e-9


def clip_halfplane(
    polygon: Sequence[Point], a: float, b: float, c: float
) -> list[Point]:
    """Clip a convex polygon by ``a*x+b*y <= c``."""
    if not polygon:
        return []
    result: list[Point] = []
    for p, q in zip(polygon, polygon[1:] + polygon[:1]):
        fp = a * p[0] + b * p[1] - c
        fq = a * q[0] + b * q[1] - c
        p_in = fp <= EPS
        q_in = fq <= EPS
        if p_in:
            result.append(p)
        if p_in != q_in:
            t = fp / (fp - fq)
            result.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return deduplicate_adjacent(result)


def deduplicate_adjacent(polygon: Sequence[Point], tolerance: float = 1.0e-7) -> list[Point]:
    result: list[Point] = []
    for point in polygon:
        if not result or math.dist(point, result[-1]) > tolerance:
            result.append(point)
    if len(result) > 1 and math.dist(result[0], result[-1]) <= tolerance:
        result.pop()
    return result


def clip_wedge(
    polygon: Sequence[Point], apex: Point, bearing_deg: float, half_angle_deg: float
) -> list[Point]:
    lo = math.radians(bearing_deg - half_angle_deg)
    hi = math.radians(bearing_deg + half_angle_deg)
    polygon = clip_halfplane(
        polygon,
        math.sin(lo),
        -math.cos(lo),
        math.sin(lo) * apex[0] - math.cos(lo) * apex[1],
    )
    return clip_halfplane(
        polygon,
        -math.sin(hi),
        math.cos(hi),
        -math.sin(hi) * apex[0] + math.cos(hi) * apex[1],
    )


def clip_outer_disk(
    polygon: Sequence[Point], center: Point, radius_m: float, sides: int
) -> list[Point]:
    """Conservatively outer-approximate a disk by tangent half-planes."""
    result = list(polygon)
    for index in range(sides):
        angle = 2.0 * math.pi * index / sides
        nx, ny = math.cos(angle), math.sin(angle)
        result = clip_halfplane(
            result,
            nx,
            ny,
            radius_m + nx * center[0] + ny * center[1],
        )
        if not result:
            break
    return result


def initial_outer_polygon(
    first_point: Point,
    bearing_deg: float,
    *,
    half_angle_deg: float = 1.0,
    source_domain_radius_m: float = 1800.0,
    maximum_reception_radius_m: float = 1500.0,
    circle_sides: int = 360,
) -> list[Point]:
    """Outer polygon for W1 intersect target disk intersect B(p1, 1500)."""
    extent = source_domain_radius_m + maximum_reception_radius_m + 100.0
    polygon: list[Point] = [
        (-extent, -extent),
        (extent, -extent),
        (extent, extent),
        (-extent, extent),
    ]
    polygon = clip_outer_disk(
        polygon, (0.0, 0.0), source_domain_radius_m, circle_sides
    )
    polygon = clip_outer_disk(
        polygon, first_point, maximum_reception_radius_m, circle_sides
    )
    return clip_wedge(polygon, first_point, bearing_deg, half_angle_deg)


def diameter(polygon: Sequence[Point]) -> float:
    return max(
        (math.dist(p, q) for p, q in itertools.combinations(polygon, 2)),
        default=0.0,
    )


def maximum_vertex_distance(point: Point, polygon: Sequence[Point]) -> float:
    return max((math.dist(point, vertex) for vertex in polygon), default=math.inf)


def complete_sector_anchor(
    first_point: Point,
    bearing_deg: float,
    maximum_reception_radius_m: float = 1500.0,
    half_angle_deg: float = 1.0,
) -> Point:
    theta = math.radians(bearing_deg)
    offset = maximum_reception_radius_m / (2.0 * math.cos(math.radians(half_angle_deg)))
    return (
        first_point[0] + offset * math.cos(theta),
        first_point[1] + offset * math.sin(theta),
    )


def guaranteed_reception_boundary(
    source_outer_polygon: Sequence[Point],
    interior_point: Point,
    *,
    guaranteed_radius_m: float = 1000.0,
    safety_margin_m: float = 0.1,
    directions: int = 180,
    direction_offset_deg: float = 0.0,
) -> list[Point]:
    """Return an inner polygon of the intersection of vertex-centred disks.

    Every radial endpoint is on the exact intersection boundary for the outer
    source polygon.  Chords between endpoints remain in that convex
    intersection, so the returned polygon is a continuous safety certificate.
    """
    radius = guaranteed_radius_m - safety_margin_m
    if maximum_vertex_distance(interior_point, source_outer_polygon) > radius + EPS:
        raise ValueError("the supplied interior point is not reception-safe")
    boundary: list[Point] = []
    for index in range(directions):
        angle = math.radians(direction_offset_deg) + 2.0 * math.pi * index / directions
        direction = (math.cos(angle), math.sin(angle))
        upper = math.inf
        for vertex in source_outer_polygon:
            wx = interior_point[0] - vertex[0]
            wy = interior_point[1] - vertex[1]
            projection = wx * direction[0] + wy * direction[1]
            discriminant = projection * projection + radius * radius - wx * wx - wy * wy
            if discriminant < -1.0e-7:
                raise ValueError("empty guaranteed-reception region")
            upper = min(upper, -projection + math.sqrt(max(0.0, discriminant)))
        boundary.append(
            (
                interior_point[0] + upper * direction[0],
                interior_point[1] + upper * direction[1],
            )
        )
    return boundary


def point_in_convex_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    sign = 0
    for p, q in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (q[0] - p[0]) * (point[1] - p[1]) - (q[1] - p[1]) * (point[0] - p[0])
        if abs(cross) <= 1.0e-7:
            continue
        current = 1 if cross > 0.0 else -1
        if sign and current != sign:
            return False
        sign = current
    return True


def segment_circle_intersections(
    p: Point, q: Point, center: Point, radius_m: float
) -> list[Point]:
    dx, dy = q[0] - p[0], q[1] - p[1]
    fx, fy = p[0] - center[0], p[1] - center[1]
    aa = dx * dx + dy * dy
    if aa <= EPS:
        return []
    bb = 2.0 * (fx * dx + fy * dy)
    cc = fx * fx + fy * fy - radius_m * radius_m
    discriminant = bb * bb - 4.0 * aa * cc
    if discriminant < -1.0e-7:
        return []
    root = math.sqrt(max(0.0, discriminant))
    values = [(-bb - root) / (2.0 * aa), (-bb + root) / (2.0 * aa)]
    result: list[Point] = []
    for value in values:
        if -EPS <= value <= 1.0 + EPS:
            point = (p[0] + value * dx, p[1] + value * dy)
            if not result or math.dist(point, result[-1]) > 1.0e-7:
                result.append(point)
    return result


def correlated_reception_certificate(
    point: Point,
    source_outer_polygon: Sequence[Point],
    first_point: Point,
    *,
    guaranteed_radius_m: float = 1000.0,
    safety_margin_m: float = 0.1,
) -> dict:
    """Certify reception using the information carried by the first detection.

    If a compatible source is at distance r from the first point, its reception
    radius is at least max(1000, r).  The near part is checked by a farthest-point
    condition.  For r >= 1000, squaring the margin condition reduces it to the
    minimum of z.h-eta*|h| on the polygon outside the 1000 m circle.  Its minima
    occur at polygon vertices, edge/circle intersections, or the antipodal
    stationary point on a retained circle arc.
    """
    circle_points: list[Point] = []
    for p, q in zip(
        source_outer_polygon,
        source_outer_polygon[1:] + source_outer_polygon[:1],
    ):
        circle_points.extend(
            segment_circle_intersections(p, q, first_point, guaranteed_radius_m)
        )

    near_points = [
        vertex
        for vertex in source_outer_polygon
        if math.dist(vertex, first_point) <= guaranteed_radius_m + EPS
    ] + circle_points
    far_points = [
        vertex
        for vertex in source_outer_polygon
        if math.dist(vertex, first_point) >= guaranteed_radius_m - EPS
    ] + circle_points

    zx, zy = point[0] - first_point[0], point[1] - first_point[1]
    z_norm = math.hypot(zx, zy)
    if z_norm > EPS:
        antipode = (
            first_point[0] - guaranteed_radius_m * zx / z_norm,
            first_point[1] - guaranteed_radius_m * zy / z_norm,
        )
        if point_in_convex_polygon(antipode, source_outer_polygon):
            near_points.append(antipode)
            far_points.append(antipode)

    near_max_distance = max(
        (math.dist(point, source) for source in near_points), default=0.0
    )
    near_slack_m = guaranteed_radius_m - near_max_distance

    far_slacks: list[float] = []
    for source in far_points:
        hx, hy = source[0] - first_point[0], source[1] - first_point[1]
        h_norm = math.hypot(hx, hy)
        far_slacks.append(
            zx * hx
            + zy * hy
            - 0.5 * z_norm * z_norm
            - safety_margin_m * h_norm
            + 0.5 * safety_margin_m * safety_margin_m
        )
    far_squared_slack_m2 = min(far_slacks) if far_slacks else None
    feasible = (
        near_slack_m >= safety_margin_m - 1.0e-7
        and (far_squared_slack_m2 is None or far_squared_slack_m2 >= -1.0e-5)
    )
    return {
        "feasible": feasible,
        "certified_distance_margin_m": safety_margin_m,
        "near_distance_slack_m": near_slack_m,
        "far_squared_margin_slack_m2": far_squared_slack_m2,
        "near_extreme_points_checked": len(near_points),
        "far_extreme_points_checked": len(far_points),
    }


def correlated_reception_boundary(
    source_outer_polygon: Sequence[Point],
    first_point: Point,
    interior_point: Point,
    *,
    safety_margin_m: float = 0.1,
    directions: int = 180,
    direction_offset_deg: float = 0.0,
) -> list[Point]:
    if not correlated_reception_certificate(
        interior_point,
        source_outer_polygon,
        first_point,
        safety_margin_m=safety_margin_m,
    )["feasible"]:
        raise ValueError("the supplied interior point is not reception-safe")
    boundary: list[Point] = []
    for index in range(directions):
        angle = math.radians(direction_offset_deg) + 2.0 * math.pi * index / directions
        direction = (math.cos(angle), math.sin(angle))
        lo, hi = 0.0, 4000.0
        for _ in range(55):
            mid = 0.5 * (lo + hi)
            candidate = (
                interior_point[0] + mid * direction[0],
                interior_point[1] + mid * direction[1],
            )
            if correlated_reception_certificate(
                candidate,
                source_outer_polygon,
                first_point,
                safety_margin_m=safety_margin_m,
            )["feasible"]:
                lo = mid
            else:
                hi = mid
        boundary.append(
            (
                interior_point[0] + lo * direction[0],
                interior_point[1] + lo * direction[1],
            )
        )
    return boundary


def posterior_diameter_upper(
    source_outer_polygon: Sequence[Point],
    second_point: Point,
    *,
    half_angle_deg: float = 1.0,
    bearing_bin_deg: float = 1.0,
) -> float:
    """Conservative upper bound over every possible second bearing response."""
    expanded = half_angle_deg + bearing_bin_deg / 2.0
    count = int(round(360.0 / bearing_bin_deg))
    maximum = 10.0  # the near response gives a set of diameter at most 10 m
    for index in range(count):
        posterior = clip_wedge(
            source_outer_polygon,
            second_point,
            index * bearing_bin_deg,
            expanded,
        )
        if posterior:
            maximum = max(maximum, diameter(posterior))
    return maximum


def local_coordinates(point: Point, first_point: Point, bearing_deg: float) -> Point:
    angle = math.radians(bearing_deg)
    dx, dy = point[0] - first_point[0], point[1] - first_point[1]
    return (
        dx * math.cos(angle) + dy * math.sin(angle),
        -dx * math.sin(angle) + dy * math.cos(angle),
    )


def world_coordinates(point: Point, first_point: Point, bearing_deg: float) -> Point:
    angle = math.radians(bearing_deg)
    return (
        first_point[0] + point[0] * math.cos(angle) - point[1] * math.sin(angle),
        first_point[1] + point[0] * math.sin(angle) + point[1] * math.cos(angle),
    )


def candidate_record(
    point: Point,
    first_point: Point,
    bearing_deg: float,
    source_outer_polygon: Sequence[Point],
    bearing_bin_deg: float,
    reception_mode: str,
) -> dict:
    upper = posterior_diameter_upper(
        source_outer_polygon, point, bearing_bin_deg=bearing_bin_deg
    )
    local = local_coordinates(point, first_point, bearing_deg)
    record = {
        "world_m": [point[0], point[1]],
        "local_m": [local[0], local[1]],
        "movement_m": math.dist(point, first_point),
        "reception_margin_m": 1000.0 - maximum_vertex_distance(point, source_outer_polygon),
        "posterior_diameter_upper_m": upper,
    }
    if reception_mode == "correlated_initial_reception":
        certificate = correlated_reception_certificate(
            point, source_outer_polygon, first_point
        )
        record["correlated_reception_certificate"] = certificate
        record["reception_margin_m"] = (
            min(
                certificate["near_distance_slack_m"],
                certificate["certified_distance_margin_m"],
            )
            if certificate["far_extreme_points_checked"]
            else certificate["near_distance_slack_m"]
        )
    return record


def sample_local_grid(
    polygon: Sequence[Point],
    first_point: Point,
    bearing_deg: float,
    budget_m: float | None,
    step_m: float,
    bounds: tuple[float, float, float, float] | None = None,
) -> list[Point]:
    if bounds is None:
        local_polygon = [local_coordinates(p, first_point, bearing_deg) for p in polygon]
        xs = [p[0] for p in local_polygon]
        ys = [p[1] for p in local_polygon]
        bounds = (min(xs), max(xs), min(ys), max(ys))
    xmin, xmax, ymin, ymax = (round(value, 8) for value in bounds)
    points: list[Point] = []
    nx = int(math.ceil((xmax - xmin) / step_m))
    ny = int(math.ceil((ymax - ymin) / step_m))
    for ix in range(nx + 1):
        x = xmin + min(ix * step_m, xmax - xmin)
        for iy in range(ny + 1):
            y = ymin + min(iy * step_m, ymax - ymin)
            point = world_coordinates((x, y), first_point, bearing_deg)
            if budget_m is not None and math.dist(point, first_point) > budget_m + EPS:
                continue
            if point_in_convex_polygon(point, polygon):
                points.append(point)
    return points


def select_second_point(
    first_point: Point,
    bearing_deg: float,
    *,
    budget_m: float | None = 1000.0,
    near_optimal_ratio: float = 0.05,
    circle_sides: int = 360,
    safe_boundary_directions: int = 180,
    coarse_cells: int = 18,
    final_bearing_bin_deg: float = 0.125,
    reception_mode: str = "correlated_initial_reception",
) -> dict:
    if budget_m is not None and budget_m <= 0.0:
        raise ValueError("movement budget must be positive")
    if near_optimal_ratio < 0.0:
        raise ValueError("near-optimal tolerance must be nonnegative")
    if circle_sides < 36 or safe_boundary_directions < 36 or coarse_cells < 4:
        raise ValueError("polygon and grid resolutions are too small for a safe run")
    bins = round(360.0 / final_bearing_bin_deg) if final_bearing_bin_deg > 0.0 else 0
    if bins <= 0 or abs(bins * final_bearing_bin_deg - 360.0) > 1.0e-8:
        raise ValueError("bearing-bin width must be positive and divide 360 degrees")
    if reception_mode not in {"conservative_1000", "correlated_initial_reception"}:
        raise ValueError("unknown reception mode")

    source_polygon = initial_outer_polygon(
        first_point,
        bearing_deg,
        circle_sides=circle_sides,
    )
    if len(source_polygon) < 3:
        raise ValueError("the first observation is inconsistent with the target domain")

    anchor = complete_sector_anchor(first_point, bearing_deg)
    if reception_mode == "conservative_1000":
        safe_polygon = guaranteed_reception_boundary(
            source_polygon,
            anchor,
            directions=safe_boundary_directions,
            direction_offset_deg=bearing_deg,
        )
    else:
        safe_polygon = correlated_reception_boundary(
            source_polygon,
            first_point,
            anchor,
            directions=safe_boundary_directions,
            direction_offset_deg=bearing_deg,
        )
    safe_polygon_local = [
        local_coordinates(point, first_point, bearing_deg) for point in safe_polygon
    ]
    xs = [p[0] for p in safe_polygon_local]
    ys = [p[1] for p in safe_polygon_local]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    coarse_step = max(5.0, span / coarse_cells)

    coarse_points = sample_local_grid(
        safe_polygon, first_point, bearing_deg, budget_m, coarse_step
    )
    mandatory = [anchor]
    mandatory.extend(safe_polygon[:: max(1, len(safe_polygon) // 24)])
    if budget_m is not None:
        budget_directions = max(720, safe_boundary_directions)
        for index in range(budget_directions):
            angle = 2.0 * math.pi * index / budget_directions
            mandatory.append(
                world_coordinates(
                    (budget_m * math.cos(angle), budget_m * math.sin(angle)),
                    first_point,
                    bearing_deg,
                )
            )
    for point in mandatory:
        if (budget_m is None or math.dist(point, first_point) <= budget_m + EPS) and point_in_convex_polygon(point, safe_polygon):
            coarse_points.append(point)
    if not coarse_points:
        raise ValueError("the movement budget does not reach the guaranteed-reception region")

    def evaluate(points: Iterable[Point], angular_step: float) -> list[dict]:
        unique = {(round(p[0], 7), round(p[1], 7)): p for p in points}
        return [
            candidate_record(
                p,
                first_point,
                bearing_deg,
                source_polygon,
                angular_step,
                reception_mode,
            )
            for p in unique.values()
        ]

    def selection_key(row: dict) -> tuple[float, float, float, float]:
        # Quantisation prevents harmless floating-point rotation noise from
        # selecting a different member of an otherwise equivalent tie.
        return (
            round(row["posterior_diameter_upper_m"], 6),
            round(row["movement_m"], 6),
            round(row["local_m"][0], 6),
            round(row["local_m"][1], 6),
        )

    coarse = evaluate(coarse_points, final_bearing_bin_deg)
    coarse.sort(key=selection_key)

    # Retain every evaluated feasible incumbent. A local grid need not contain
    # its seed (notably a point on the budget circle), so replacing the pool
    # can return a worse point than one that has already been certified.
    refinement: list[dict] = coarse
    evaluated = {
        tuple(round(value, 7) for value in row["world_m"]): row
        for row in coarse
    }
    current_step = coarse_step
    for angular_step, seed_count in (
        (final_bearing_bin_deg, 6),
        (final_bearing_bin_deg, 6),
    ):
        current_step /= 4.0
        points: list[Point] = []
        for seed in refinement[:seed_count]:
            sx, sy = seed["local_m"]
            bounds = (
                sx - 2.5 * current_step,
                sx + 2.5 * current_step,
                sy - 2.5 * current_step,
                sy + 2.5 * current_step,
            )
            points.extend(
                sample_local_grid(
                    safe_polygon,
                    first_point,
                    bearing_deg,
                    budget_m,
                    current_step,
                    bounds,
                )
            )
        for row in evaluate(points, angular_step):
            evaluated[tuple(round(value, 7) for value in row["world_m"])] = row
        refinement = list(evaluated.values())
        refinement.sort(key=selection_key)

    best = refinement[0]
    threshold = best["posterior_diameter_upper_m"] * (1.0 + near_optimal_ratio)
    near = [
        row
        for row in refinement
        if row["posterior_diameter_upper_m"] <= threshold + EPS
    ]
    near.sort(key=lambda row: (row["movement_m"], row["posterior_diameter_upper_m"]))

    # The frontier is over the evaluated finite set, not the continuous domain.
    pareto: list[dict] = []
    best_upper = math.inf
    for row in sorted(
        refinement,
        key=lambda row: (row["movement_m"], row["posterior_diameter_upper_m"]),
    ):
        if row["posterior_diameter_upper_m"] < best_upper:
            pareto.append(row)
            best_upper = row["posterior_diameter_upper_m"]

    return {
        "schema_version": 1,
        "artifact_type": "q2_generic_second_point_selection",
        "input": {
            "first_point_m": list(first_point),
            "first_bearing_deg": bearing_deg % 360.0,
            "movement_budget_m": budget_m,
            "reception_mode": reception_mode,
        },
        "constants": {
            "bearing_error_half_angle_deg": 1.0,
            "source_domain_radius_m": 1800.0,
            "maximum_reception_radius_m": 1500.0,
            "guaranteed_reception_radius_m": 1000.0,
            "near_response_radius_m": 5.0,
            "continuous_reception_safety_margin_m": 0.1,
        },
        "approximations": {
            "first_source_set": f"outer polygon using {circle_sides} tangent half-planes per disk",
            "safe_candidate_region": f"inner polygon with {safe_boundary_directions} radial directions",
            "final_second_bearing_bin_deg": final_bearing_bin_deg,
            "near_optimal_relative_tolerance": near_optimal_ratio,
            "continuous_global_optimality_claimed": False,
        },
        "first_source_outer_polygon_world_m": [list(p) for p in source_polygon],
        "guaranteed_reception_inner_polygon_world_m": [list(p) for p in safe_polygon],
        "complete_sector_safe_anchor_world_m": list(anchor),
        "recommended_point": best,
        "near_optimal_candidate_points": near,
        "finite_candidate_pareto_frontier": pareto,
        "evaluated_unique_candidate_count": len(evaluated),
        "certificate": {
            "safe_polygon_max_endpoint_distance_m": (
                max(maximum_vertex_distance(point, source_polygon) for point in safe_polygon)
                if reception_mode == "conservative_1000"
                else None
            ),
            "safe_polygon_is_continuously_safe": True,
            "reason": "the safe set is convex and each returned edge is a chord between certified boundary points",
            "posterior_metric_is_conservative_upper_bound": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-x", type=float, required=True)
    parser.add_argument("--first-y", type=float, required=True)
    parser.add_argument("--bearing-deg", type=float, required=True)
    parser.add_argument("--budget-m", type=float, default=1000.0)
    parser.add_argument("--near-optimal-ratio", type=float, default=0.05)
    parser.add_argument("--circle-sides", type=int, default=360)
    parser.add_argument("--safe-directions", type=int, default=180)
    parser.add_argument("--coarse-cells", type=int, default=18)
    parser.add_argument("--bearing-bin-deg", type=float, default=0.125)
    parser.add_argument(
        "--reception-mode",
        choices=("conservative_1000", "correlated_initial_reception"),
        default="correlated_initial_reception",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = select_second_point(
        (args.first_x, args.first_y),
        args.bearing_deg,
        budget_m=args.budget_m,
        near_optimal_ratio=args.near_optimal_ratio,
        circle_sides=args.circle_sides,
        safe_boundary_directions=args.safe_directions,
        coarse_cells=args.coarse_cells,
        final_bearing_bin_deg=args.bearing_bin_deg,
        reception_mode=args.reception_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["recommended_point"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
