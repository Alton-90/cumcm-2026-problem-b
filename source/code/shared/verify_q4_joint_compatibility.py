"""Reproducible boundary and model checks for q4_joint_compatibility.

No simulator, hidden truth API, search policy, or online endpoint is used.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

from q4_joint_compatibility import (
    PublicObservation,
    check_fixed_position,
    circular_difference_deg,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q4_joint_compatibility/verification.json"
SEED = 20260911


def direction(point, candidate=(0.0, 0.0), bearing=None):
    if bearing is None:
        bearing = math.degrees(math.atan2(candidate[1] - point[1],
                                          candidate[0] - point[0])) % 360.0
    return PublicObservation(tuple(point), "direction", bearing)


def polar(radius, degrees):
    angle = math.radians(degrees)
    return (radius * math.cos(angle), radius * math.sin(angle))


named_checks = []


def record(name, condition, result=None, note=None):
    assert condition, name
    item = {"name": name, "pass": True}
    if result is not None:
        item.update(status=result.status,
                    nominal_compatible=result.nominal_compatible,
                    keep_candidate=result.keep_candidate)
    if note:
        item["note"] = note
    named_checks.append(item)


# Empty and ordinary interiors.
r = check_fixed_position((0, 0), [])
record("empty_history_all_headings", r.status == "compatible" and
       len(r.orientation_intervals_deg) == 1, r)

r = check_fixed_position((0, 0), [direction((-1000, 0))])
record("single_positive_closed_halfcircle", r.nominal_compatible and
       r.status == "compatible", r)

# 0/360 bearing seam.
point = tuple(-value for value in polar(1000, 0.2))
r = check_fixed_position((0, 0), [direction(point, bearing=359.8)])
record("bearing_wrap_across_zero", r.nominal_compatible and
       circular_difference_deg(0.2, 359.8) < 1.005, r)
r = check_fixed_position((0, 0), [direction(point, bearing=358.0)])
record("bearing_wrap_outside_tolerance", r.status == "incompatible" and
       not r.keep_candidate, r)
r = check_fixed_position((0, 0), [direction((-1000, 0), bearing=1.005)])
record("bearing_error_exactly_1_005deg_closed_boundary",
       r.nominal_compatible and r.status == "uncertain" and r.keep_candidate, r)

# 5 m near/direction split is closed for near and strict for direction.
r = check_fixed_position((0, 0), [direction((5, 0))])
record("direction_exactly_5m_nominally_impossible", not r.nominal_compatible and
       r.status == "uncertain" and r.keep_candidate, r)
r = check_fixed_position((0, 0), [direction((4, 0))])
record("direction_safely_inside_5m_impossible", r.status == "incompatible", r)
r = check_fixed_position((0, 0), [PublicObservation((5, 0), "near")])
record("near_exactly_5m_closed_boundary", r.nominal_compatible and
       r.status == "uncertain" and r.keep_candidate, r)
r = check_fixed_position((0, 0), [PublicObservation((0, 0), "near")])
record("near_zero_vector_is_visible_for_every_heading",
       r.nominal_compatible and r.status == "compatible", r)
r = check_fixed_position((0, 0), [PublicObservation((6, 0), "near")])
record("near_beyond_5m_impossible", r.status == "incompatible", r)
r = check_fixed_position((100, 0), [PublicObservation((0, 0), "near")])
record("near_counterexample_to_unaugmented_derivation", r.status == "incompatible", r,
       "With P+ and P- defined as direction/no_signal only, this false candidate would pass.")

# Failed clear: success is closed at 20 m, so a failure implies strict >20 m.
r = check_fixed_position((0, 0), [], [(20, 0)])
record("failed_clear_exactly_20m_nominally_impossible", not r.nominal_compatible and
       r.status == "uncertain" and r.keep_candidate, r)
r = check_fixed_position((0, 0), [], [(19, 0)])
record("failed_clear_inside_20m_impossible", r.status == "incompatible", r)
r = check_fixed_position((0, 0), [], [(21, 0)])
record("failed_clear_outside_20m_allowed", r.status == "compatible", r)

# Closed source-area and range boundaries.
r = check_fixed_position((1800, 0), [])
record("source_region_1800m_closed_boundary", r.nominal_compatible and
       r.status == "uncertain", r)
r = check_fixed_position((1800.001, 0), [])
record("outside_source_region_impossible", r.status == "incompatible", r)
r = check_fixed_position((0, 0), [direction((1500, 0))])
record("range_1500m_closed_boundary", r.nominal_compatible and
       math.isclose(r.range_star_m, 1500.0) and r.status == "uncertain", r)
r = check_fixed_position((0, 0), [direction((1500.001, 0))])
record("positive_beyond_1500m_impossible", r.status == "incompatible", r)

# R* boundary: equality is in range and therefore a no_signal must be strict shadow.
two_opposite = [direction((1000, 0)), direction((-1000, 0))]
r = check_fixed_position((0, 0), two_opposite)
degenerate = [item for item in r.orientation_intervals_deg if item["degenerate"]]
record("closed_positive_constraints_leave_two_degenerate_headings",
       r.nominal_compatible and len(degenerate) == 2 and r.status == "uncertain", r)
r = check_fixed_position((0, 0), two_opposite +
                         [PublicObservation((0, 1000), "no_signal")])
record("open_negative_removes_only_one_degenerate_heading",
       r.nominal_compatible and len(r.orientation_intervals_deg) == 1 and
       r.orientation_intervals_deg[0]["degenerate"], r)
r = check_fixed_position((0, 0), two_opposite +
                         [PublicObservation((1000, 0), "no_signal")])
record("negative_on_heading_boundary_is_visible_not_shadow",
       not r.nominal_compatible and r.status == "uncertain" and r.keep_candidate, r)
r = check_fixed_position((0, 0), two_opposite +
                         [PublicObservation((1000.001, 0), "no_signal")])
record("negative_just_outside_Rstar_is_range_explained", r.nominal_compatible, r)
r = check_fixed_position((0, 0), [PublicObservation((0, 0), "no_signal")])
record("forced_negative_zero_vector_impossible", r.status == "incompatible" and
       not r.keep_candidate, r)

# A genuinely separated empty heading intersection, not a sampling artefact.
three = [direction(polar(900, angle)) for angle in (0, 120, 240)]
r = check_fixed_position((0, 0), three)
record("three_positive_halfplanes_have_robust_empty_intersection",
       r.status == "incompatible" and not r.nominal_compatible, r)
r_either = check_fixed_position((0, 0), three, source_type="either")
record("unknown_type_keeps_omnidirectional_union",
       r_either.status == "compatible" and r_either.nominal_compatible and
       not r_either.directional_nominal_compatible and
       r_either.omnidirectional_nominal_compatible, r_either)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def oracle_heading_exists(positive, negative):
    """Independent exact-finite-cell oracle; no regular heading grid."""
    if any(v == (0.0, 0.0) for v in negative):
        return False
    vectors = [v for v in (*positive, *negative) if v != (0.0, 0.0)]
    if not vectors:
        return True
    rays = []
    angles = []
    for x, y in vectors:
        for ray in ((-y, x), (y, -x)):
            rays.append(ray)
            angles.append(math.atan2(ray[1], ray[0]) % (2 * math.pi))
    angles = sorted(set(angles))
    candidates = list(rays)
    for index, first in enumerate(angles):
        second = angles[(index + 1) % len(angles)]
        if index + 1 == len(angles):
            second += 2 * math.pi
        middle = (first + second) / 2
        candidates.append((math.cos(middle), math.sin(middle)))
    return any(all(dot(h, v) >= -1e-12 for v in positive) and
               all(dot(h, v) < -1e-12 for v in negative)
               for h in candidates)


def direct_original_exists(positive, negative):
    """Search all radial change points/interiors in the unreduced model."""
    distances = [math.hypot(*v) for v in (*positive, *negative)]
    boundaries = sorted(set([1000.0, 1500.0] +
                            [d for d in distances if 1000.0 <= d <= 1500.0]))
    radii = boundaries + [(a + b) / 2 for a, b in zip(boundaries, boundaries[1:])]
    for radius in radii:
        if not all(math.hypot(*v) <= radius for v in positive):
            continue
        forced = [v for v in negative if math.hypot(*v) <= radius]
        if oracle_heading_exists(positive, forced):
            return True
    return False


def as_history(positive, negative):
    values = [direction(v) for v in positive]
    values.extend(PublicObservation(tuple(v), "no_signal") for v in negative)
    return values


# Independent unreduced-radius + exact finite critical-heading comparison.
rng = random.Random(SEED)
original_comparisons = 0
for _ in range(4000):
    positive = [polar(rng.uniform(50, 1550), rng.uniform(0, 360))
                for _ in range(rng.randrange(0, 5))]
    negative = [polar(rng.uniform(0.1, 1700), rng.uniform(0, 360))
                for _ in range(rng.randrange(0, 6))]
    expected = direct_original_exists(positive, negative)
    actual = check_fixed_position((0, 0), as_history(positive, negative),
                                  source_type="directional").nominal_compatible
    assert actual == expected, (positive, negative, expected, actual)
    original_comparisons += 1


def public_response(source, radius, heading_deg, point, error):
    vector = (point[0] - source[0], point[1] - source[1])
    distance = math.hypot(*vector)
    visible = distance <= radius
    if visible and heading_deg is not None:
        h = polar(1.0, heading_deg)
        visible = dot(h, vector) >= 0.0
    if not visible:
        return PublicObservation(point, "no_signal")
    if distance <= 5.0:
        return PublicObservation(point, "near")
    true_bearing = math.degrees(math.atan2(source[1] - point[1],
                                           source[0] - point[0])) % 360.0
    reported = round((true_bearing + error) % 360.0, 2) % 360.0
    return PublicObservation(point, "direction", reported)


# Histories generated from the original response model must never delete truth.
truth_histories = 0
truth_receipts = 0
truth_status_counts = {"compatible": 0, "uncertain": 0, "incompatible": 0}
for source_type in ("directional", "omnidirectional"):
    for case_index in range(750):
        source = polar(rng.uniform(0, 1600), rng.uniform(0, 360))
        radius = rng.uniform(1050, 1450)
        heading = rng.uniform(0, 360) if source_type == "directional" else None
        points = []
        if case_index % 10 == 0:
            near_angle = heading if heading is not None else rng.uniform(0, 360)
            offset = polar(3.0, near_angle)
            points.append((source[0] + offset[0], source[1] + offset[1]))
        for _ in range(12):
            offset = polar(rng.uniform(20, 1750), rng.uniform(0, 360))
            points.append((source[0] + offset[0], source[1] + offset[1]))
        history = [public_response(source, radius, heading, point,
                                   rng.uniform(-0.999, 0.999)) for point in points]
        far_clear_offset = polar(rng.uniform(25, 200), rng.uniform(0, 360))
        failed = [(source[0] + far_clear_offset[0],
                   source[1] + far_clear_offset[1])]
        result = check_fixed_position(source, history, failed,
                                      source_type=source_type)
        assert result.keep_candidate and result.nominal_compatible, result
        truth_status_counts[result.status] += 1
        truth_histories += 1
        truth_receipts += len(history)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


payload = {
    "artifact_type": "q4_fixed_position_joint_compatibility_verification",
    "all_checks_pass": True,
    "theorem_verdict": (
        "Correct for a fixed candidate and directional source after scalar public-response "
        "constraints are enforced; near receipts must join the positive set, and unknown "
        "source type requires the union with the omnidirectional case."
    ),
    "range_elimination": {
        "chosen_range_m": "R*=max(1000,max distance to every direction/near receipt)",
        "negative_at_equal_distance": "inside the closed range and therefore must be strict shadow",
        "positive_halfplane": "closed (dot >= 0)",
        "forced_negative_halfplane": "open (dot < 0)",
    },
    "near_counterexample": {
        "history": [{"point": [0, 0], "measure_result": "near"}],
        "false_candidate": [100, 0],
        "why_original_shorthand_fails": (
            "If P+ means direction only and P- means no_signal, both are empty and the shorthand "
            "accepts; the original response model rejects because near requires distance <=5 m."
        ),
        "repair": "Treat near as positive visibility and separately require its closed <=5 m distance.",
    },
    "continuous_heading_method": (
        "intersection of split circular arcs with explicit open/closed endpoints; "
        "independent oracle enumerates every constraint boundary ray and every open cell"
    ),
    "named_boundary_checks": named_checks,
    "named_boundary_check_count": len(named_checks),
    "unreduced_original_model_comparisons": original_comparisons,
    "truth_history_checks": truth_histories,
    "truth_receipt_checks": truth_receipts,
    "truth_status_counts": truth_status_counts,
    "random_seed": SEED,
    "integration": {
        "target": "code/shared/search_policy_q4_ring22_joint_route.py public-history layer",
        "unknown_type_argument": "source_type='either'",
        "deletion_rule": "delete only status='incompatible'; retain uncertain",
        "preserve": ["22-point continuous coverage", "fallback localization/clear chain"],
    },
    "scope": {
        "public_inputs_only": True,
        "online_simulator_access": False,
        "search_policy_run": False,
        "performance_batch_run": False,
        "continuous_source_area_exclusion_certificate": False,
    },
    "files": {
        "checker": "code/shared/q4_joint_compatibility.py",
        "checker_sha256": sha256(ROOT / "code/shared/q4_joint_compatibility.py"),
        "verifier": "code/shared/verify_q4_joint_compatibility.py",
        "verifier_sha256": sha256(Path(__file__).resolve()),
        "plan": "planning/q4_joint_compatibility_plan.md",
    },
    "rerun": "D:/python/python.exe -X utf8 code/shared/verify_q4_joint_compatibility.py",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: payload[key] for key in
                  ("all_checks_pass", "named_boundary_check_count",
                   "unreduced_original_model_comparisons", "truth_history_checks",
                   "truth_receipt_checks", "truth_status_counts")}, ensure_ascii=False))
