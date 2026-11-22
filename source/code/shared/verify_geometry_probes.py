"""Bounded numerical diagnostics for existing Q1/Q2/Q3-Q4 derivations.

This is not a simulator client and not a search strategy implementation.
It uses only the Python standard library and reports finite numerical probes;
the accompanying Markdown derivations remain the analytical authority.
"""

import json
import math
import random
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "diagnostics" / "geometry_probes.json"
TOL = 1e-8


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def mul(k, a):
    return (k * a[0], k * a[1])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def norm(a):
    return math.hypot(a[0], a[1])


def unit(angle):
    return (math.cos(angle), math.sin(angle))


def line_intersection(p, d, q, e):
    den = cross(d, e)
    if abs(den) < TOL:
        return None
    return add(p, mul(cross(sub(q, p), e) / den, d))


def dedupe(points):
    out = []
    for p in points:
        if not any(norm(sub(p, q)) < 1e-7 for q in out):
            out.append(p)
    return out


def angle_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def q1_probe():
    side = 36.0
    vertices = [(0.0, 0.0), (36.0, 0.0), (18.0, 18.0 * math.sqrt(3.0))]
    eps = math.radians(1.0)
    constraints, lines, sensors = [], [], []
    # On CCW edge E->F, lower ray is the edge; upper ray turns 2 degrees inward.
    for e, f in zip(vertices, vertices[1:] + vertices[:1]):
        u = mul(1.0 / side, sub(f, e))
        sensor = sub(e, mul(1000.0, u))
        theta = math.atan2(u[1], u[0]) + eps
        um, up = unit(theta - eps), unit(theta + eps)
        sensors.append(sensor)
        lines.extend([(sensor, um), (sensor, up)])
        constraints.append((sensor, um, 1))   # cross(um, z-s) >= 0
        constraints.append((sensor, up, -1))  # cross(up, z-s) <= 0

    candidates = []
    for (p, d), (q, e) in combinations(lines, 2):
        z = line_intersection(p, d, q, e)
        if z is not None and all(sign * cross(ray, sub(z, s)) >= -TOL for s, ray, sign in constraints):
            candidates.append(z)
    found = dedupe(candidates)
    found.sort()
    expected = sorted(vertices)
    vertex_match = max(min(norm(sub(v, w)) for w in found) for v in expected)
    diameter = max(norm(sub(a, b)) for a, b in combinations(found, 2))
    g = (18.0, 6.0 * math.sqrt(3.0))
    cover_radius = max(norm(sub(v, g)) for v in vertices)
    source_distances = [norm(sub(g, s)) for s in sensors]
    errors = []
    for sensor, read in zip(sensors, [math.atan2(sub(vertices[(i + 1) % 3], vertices[i])[1], sub(vertices[(i + 1) % 3], vertices[i])[0]) + eps for i in range(3)]):
        actual = math.atan2(g[1] - sensor[1], g[0] - sensor[0])
        errors.append(angle_diff_deg(math.degrees(read), math.degrees(actual)))
    return {
        "epsilon_deg": 1.0,
        "halfplanes": 6,
        "feasible_pair_intersections": [[round(x, 9), round(y, 9)] for x, y in found],
        "expected_vertices": [[round(x, 9), round(y, 9)] for x, y in expected],
        "max_vertex_match_error_m": vertex_match,
        "diameter_m": diameter,
        "expected_diameter_m": 36.0,
        "cover_radius_m": cover_radius,
        "expected_cover_radius_m": 12.0 * math.sqrt(3.0),
        "coverage_radius_exceeds_20m": cover_radius > 20.0,
        "source_at_centroid": [g[0], g[1]],
        "source_sensor_distances_m": source_distances,
        "all_source_distances_in_5_to_1100m": all(5.0 < d <= 1100.0 for d in source_distances),
        "reading_errors_deg": errors,
        "max_reading_error_deg": max(errors),
        "all_reading_errors_at_most_1deg": max(errors) <= 1.0 + TOL,
        "pass": vertex_match < 1e-6 and abs(diameter - 36.0) < 1e-6 and abs(cover_radius - 12.0 * math.sqrt(3.0)) < 1e-6,
    }


def q2_probe(epsilon_deg):
    eps = math.radians(epsilon_deg)
    c = (750.0, 0.0)
    m = 750.0 * math.sqrt(5.0 - 4.0 * math.cos(eps))
    r = 1000.0 - m
    # Finite diagnostic only: samples F0 and candidate circle boundaries.
    source_samples = []
    for ti in range(13):
        t = 1500.0 * ti / 12.0
        for di in range(25):
            delta = -eps + 2.0 * eps * di / 24.0
            source_samples.append((t * math.cos(delta), t * math.sin(delta)))
    point_samples = [c]
    for i in range(72):
        phi = 2.0 * math.pi * i / 72.0
        point_samples.append(add(c, mul(r, unit(phi))))
    worst = max(norm(sub(p, g)) for p in point_samples for g in source_samples)
    return {
        "epsilon_deg": epsilon_deg,
        "M_m": m,
        "candidate_center_m": list(c),
        "candidate_radius_m": r,
        "sample_source_count": len(source_samples),
        "sample_candidate_point_count": len(point_samples),
        "sample_max_distance_m": worst,
        "sample_contains_F0_at_1000m": worst <= 1000.0 + 1e-8,
        "note": "Sampling is a diagnostic only; the analytic triangle-inequality proof is in q2_second_point_notes.md.",
    }


def coverage_probe():
    grid = [(float(x), float(y)) for x in range(-2400, 2401, 600) for y in range(-2400, 2401, 600)]
    cases = []
    # Circle boundary, source-grid nodes inside target circle, and seeded random sources.
    for i in range(360):
        a = 2.0 * math.pi * i / 360.0
        cases.append((1800.0 * math.cos(a), 1800.0 * math.sin(a), "circle"))
    for x in range(-1800, 1801, 600):
        for y in range(-1800, 1801, 600):
            if x * x + y * y <= 1800.0 ** 2 + TOL:
                cases.append((float(x), float(y), "source_grid"))
    rng = random.Random(20260910)
    for _ in range(400):
        a = rng.random() * 2.0 * math.pi
        radius = 1800.0 * math.sqrt(rng.random())
        cases.append((radius * math.cos(a), radius * math.sin(a), "random"))
    # Q3 has no directional-coverage requirement: the coarser 25-point lattice
    # is sufficient because every target-disk point is within 600*sqrt(2) m.
    q3_grid = [(float(x), float(y)) for x in range(-2400, 2401, 1200) for y in range(-2400, 2401, 1200)]
    q3_worst_nearest = max(min(norm(sub((x, y), p)) for p in q3_grid) for x, y, _ in cases)
    direction_angles = [2.0 * math.pi * i / 24.0 for i in range(24)]
    failures, worst_best_distance, checked = [], 0.0, 0
    for x, y, tag in cases:
        g = (x, y)
        for a in direction_angles:
            n = unit(a)
            eligible = [p for p in grid if norm(sub(p, g)) <= 1000.0 + TOL and dot(n, sub(p, g)) > TOL]
            checked += 1
            if not eligible:
                failures.append({"source": [x, y], "tag": tag, "direction_deg": math.degrees(a)})
            else:
                worst_best_distance = max(worst_best_distance, min(norm(sub(p, g)) for p in eligible))
    qmax = 1500.0 * math.sin(math.radians(1.005))
    clear_grid = [(float(t), float(q)) for t in range(0, 1501, 20) for q in (-30.0, -10.0, 10.0, 30.0)]
    # Boundary and midpoint probes include the analytic worst locations q=-20,0,20 and t=10 mod 20.
    probe_ts = list(range(0, 1501, 10))
    probe_qs = [-qmax, -20.0, -10.0, 0.0, 10.0, 20.0, qmax]
    worst_clear = max(min(norm(sub((t, q), p)) for p in clear_grid) for t in probe_ts for q in probe_qs)
    movement = (51400.0 + 16.0 * 7591.0) / 5.0
    switching = 9720.0
    clearing = 16.0 * (303.0 * 3.0 + 5.0)
    total = movement + switching + clearing
    return {
        "q3_coarse_scan_grid_points": len(q3_grid),
        "q3_coarse_grid_sample_worst_nearest_distance_m": q3_worst_nearest,
        "q3_coarse_grid_theoretical_bound_m": 600.0 * math.sqrt(2.0),
        "q3_coarse_grid_bound_below_1000m": 600.0 * math.sqrt(2.0) < 1000.0,
        "scan_grid_points": len(grid),
        "coverage_cases": {"circle": 360, "source_grid": sum(1 for _, _, t in cases if t == "source_grid"), "random": 400},
        "emission_directions_per_case": len(direction_angles),
        "checked_source_direction_pairs": checked,
        "coverage_failures": failures[:5],
        "coverage_failure_count": len(failures),
        "worst_nearest_eligible_detection_distance_m": worst_best_distance,
        "clear_rectangle_half_width_m": qmax,
        "clear_grid_points": len(clear_grid),
        "clear_grid_sample_worst_nearest_distance_m": worst_clear,
        "clear_grid_theoretical_bound_m": math.sqrt(200.0),
        "clear_grid_bound_below_20m": math.sqrt(200.0) < 20.0,
        "virtual_time_upper_bound_s": total,
        "virtual_time_upper_bound_hours": total / 3600.0,
        "below_17_hours": total < 17.0 * 3600.0,
    }


def main():
    report = {
        "purpose": "Bounded numerical diagnostics; not an official simulator run or full strategy.",
        "q1_triangle_halfplane_probe": q1_probe(),
        "q2_candidate_circle_probe": [q2_probe(1.0), q2_probe(1.005)],
        "q3_q4_coverage_and_bound_probe": coverage_probe(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "q1_pass": report["q1_triangle_halfplane_probe"]["pass"],
        "q2_sample_checks": [x["sample_contains_F0_at_1000m"] for x in report["q2_candidate_circle_probe"]],
        "coverage_failure_count": report["q3_q4_coverage_and_bound_probe"]["coverage_failure_count"],
        "hours": report["q3_q4_coverage_and_bound_probe"]["virtual_time_upper_bound_hours"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
