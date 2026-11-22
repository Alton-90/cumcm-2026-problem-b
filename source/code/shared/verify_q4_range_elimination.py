"""Finite boundary checks for a proved fixed-position parameter elimination.

This is not a Q4 search policy or a continuous area-coverage certificate.
"""
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]


def visible(relative_point, radius, heading):
    return math.hypot(*relative_point) <= radius and dot(heading, relative_point) >= 0


def original(positive, negative, radius, heading):
    return 1000 <= radius <= 1500 and all(visible(p, radius, heading) for p in positive) and all(
        not visible(p, radius, heading) for p in negative)


def eliminated(positive, negative, heading):
    lower = max(1000.0, *(math.hypot(*p) for p in positive))
    return lower <= 1500 and all(dot(heading, p) >= 0 for p in positive) and all(
        dot(heading, n) < 0 for n in negative if math.hypot(*n) <= lower)


cases = [
    ('inside_range_shadow_is_possible', [(1200, 0)], [(0, 500)], (1, -1), True),
    ('closed_halfplane_boundary_is_visible', [(1200, 0)], [(0, 500)], (1, 0), False),
    ('range_boundary_is_visible', [(1200, 0)], [(0, 1200)], (1, 1), False),
    ('just_outside_minimum_range_is_allowed', [(1200, 0)], [(0, 1201)], (1, 1), True),
    ('negative_along_positive_ray_is_impossible', [(1200, 0)], [(800, 0)], (1, 0), False),
    ('positive_observations_can_require_range_above_1000', [(1300, 0)], [], (1, 0), True),
    ('positive_beyond_1500_is_impossible', [(1501, 0)], [], (1, 0), False),
]
checks = []
for name, positive, negative, heading, expected in cases:
    value = eliminated(positive, negative, heading)
    assert value == expected, name
    lower = max(1000.0, *(math.hypot(*p) for p in positive))
    assert original(positive, negative, lower, heading) == expected, name
    checks.append({'name': name, 'pass': True})

# For each tested heading compare the reduced predicate with the unreduced
# model at all its finitely many radial change points and interval interiors.
points = [(500, 0), (0, 500), (-500, 0), (1200, 0), (0, 1200), (0, 1201), (-1400, 0)]
headings = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if (x, y) != (0, 0)]
count = 0
for positive_point, negatives in itertools.product(points, itertools.combinations(points, 2)):
    positive = [positive_point]
    lower = max(1000.0, math.hypot(*positive_point))
    boundaries = sorted(set([1000.0, 1500.0, lower]+[
        math.hypot(*n) for n in negatives if 1000 <= math.hypot(*n) <= 1500]))
    radii = boundaries+[(a+b)/2 for a, b in zip(boundaries, boundaries[1:])]
    for heading in headings:
        direct = any(original(positive, negatives, radius, heading) for radius in radii)
        assert direct == eliminated(positive, negatives, heading)
        count += 1

out = ROOT/'results/literature_transfer/q4_range_elimination_checks.json'
out.parent.mkdir(parents=True, exist_ok=True)
payload = {'all_checks_pass': True, 'named_boundary_cases': checks,
           'finite_fixed_heading_comparisons': count,
           'scope': 'fixed_position_parameter_elimination_sanity_check',
           'continuous_source_coverage_certificate': False, 'search_policy_run': False,
           'proof_location': 'planning/literature/related_paper_analysis.md section 4'}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(payload, ensure_ascii=False))
