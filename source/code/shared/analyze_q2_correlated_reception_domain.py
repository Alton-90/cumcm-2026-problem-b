"""Exploratory Q2 diagnostic for the reception-radius correlation.

This script does not define an approved method, a usable baseline, or a
canonical experiment run.  It asks one narrower question: after a successful
first reception, does the implied lower bound on the unknown reception radius
certifiably enlarge the safe second-point domain, and does that enlargement
improve the same conservative posterior-diameter score used by the existing
Q2 paper diagnostic?

The continuous certificate is derived for the complete 1500 m, +/-1 degree
sector example.  Search is finite and coarse-to-fine; no continuous global
optimality is claimed.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from verify_paper_q12_geometry import farthest_sector, sector_outer, upper


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "paper_q12_geometry_v2"
OUT_PATH = OUT_DIR / "q2_correlated_reception_domain_diagnostic.json"
STAGE1_PATH = OUT_DIR / "q2_correlated_reception_domain_diagnostic_stage1.json"

HALF_ANGLE_DEG = 1.0
SECTOR_RADIUS_M = 1500.0
GUARANTEED_RADIUS_M = 1000.0
SAFETY_MARGIN_M = 0.1
BUDGETS_M = (550.0, 600.0, 700.0, 800.0, 900.0, 1000.0)


@dataclass(frozen=True)
class SearchLevel:
    name: str
    circle_sides: int
    bearing_bin_deg: float
    step_m: float
    half_width_m: float | None


SEARCH_LEVELS = (
    SearchLevel("global_25m", 360, 0.5, 25.0, None),
    SearchLevel("local_5m", 720, 0.25, 5.0, 50.0),
    SearchLevel("local_2p5m", 1440, 0.125, 2.5, 15.0),
)

VERIFY_LEVELS = (
    (360, 0.5),
    (720, 0.25),
    (1440, 0.125),
    (2880, 0.0625),
    (5760, 0.03125),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frange(start: float, stop: float, step: float):
    count = int(math.floor((stop - start) / step + 1e-9))
    for index in range(count + 1):
        yield round(start + index * step, 10)


def opposite_endpoint_distance(a_m: float, b_m: float, radius_m: float) -> float:
    delta = math.radians(HALF_ANGLE_DEG)
    return math.hypot(
        a_m - radius_m * math.cos(delta),
        abs(b_m) + radius_m * math.sin(delta),
    )


def constant_1000_slack_m(a_m: float, b_m: float) -> float:
    """Exact minimum reception slack over the complete 1500 m sector."""
    farthest = farthest_sector(a_m, abs(b_m), HALF_ANGLE_DEG)
    return GUARANTEED_RADIUS_M - farthest


def correlated_slack_m(a_m: float, b_m: float) -> float:
    """Exact minimum slack under R(g) >= max(1000, ||g-p1||).

    For 0 <= r <= 1000, distance is convex in r, so the maximum is at
    r=0 or r=1000.  For r >= 1000, r-distance is nondecreasing in r, so
    the smallest slack is again at r=1000.  The worst angle is the sector
    endpoint opposite the sign of b.
    """
    return min(
        GUARANTEED_RADIUS_M - math.hypot(a_m, b_m),
        GUARANTEED_RADIUS_M - opposite_endpoint_distance(a_m, b_m, 1000.0),
    )


def safe(domain: str, a_m: float, b_m: float) -> bool:
    if a_m < 0.0 or b_m < 0.0:
        return False
    if domain == "constant_1000":
        return constant_1000_slack_m(a_m, b_m) >= SAFETY_MARGIN_M - 1e-9
    if domain == "first_reception_correlated":
        delta = math.radians(HALF_ANGLE_DEG)
        min_projection_m = a_m * math.cos(delta) - b_m * math.sin(delta)
        return (
            correlated_slack_m(a_m, b_m) >= SAFETY_MARGIN_M - 1e-9
            and min_projection_m >= SAFETY_MARGIN_M - 1e-9
        )
    raise ValueError(f"unknown domain: {domain}")


def b_limit(domain: str, a_m: float) -> float:
    """Upper half-plane cross-section of the certified symmetric domain."""
    certified_radius = GUARANTEED_RADIUS_M - SAFETY_MARGIN_M
    if not 0.0 <= a_m <= certified_radius:
        return 0.0
    radial_sq = certified_radius**2 - a_m**2
    if radial_sq < 0.0:
        return 0.0
    if domain == "constant_1000":
        source_radius = SECTOR_RADIUS_M
    elif domain == "first_reception_correlated":
        source_radius = GUARANTEED_RADIUS_M
    else:
        raise ValueError(f"unknown domain: {domain}")
    delta = math.radians(HALF_ANGLE_DEG)
    axial = a_m - source_radius * math.cos(delta)
    endpoint_sq = certified_radius**2 - axial**2
    if endpoint_sq <= 0.0:
        return 0.0
    endpoint_limit = math.sqrt(endpoint_sq) - source_radius * math.sin(delta)
    limits = [math.sqrt(radial_sq), endpoint_limit]
    if domain == "first_reception_correlated":
        # The r>=1000 margin proof also requires the minimum forward
        # projection to be at least eta.
        projection_limit = (
            a_m * math.cos(delta) - SAFETY_MARGIN_M
        ) / math.sin(delta)
        limits.append(projection_limit)
    return max(0.0, min(limits))


def certified_area_m2(domain: str, step_m: float) -> float:
    """Deterministic midpoint quadrature over symmetric cross-sections."""
    stop = GUARANTEED_RADIUS_M - SAFETY_MARGIN_M
    count = int(math.ceil(stop / step_m))
    half_area = 0.0
    for index in range(count):
        left = index * step_m
        right = min(stop, (index + 1) * step_m)
        midpoint = 0.5 * (left + right)
        half_area += b_limit(domain, midpoint) * (right - left)
    return 2.0 * half_area


def candidate_record(a_m: float, b_m: float, score_m: float) -> dict:
    return {
        "a_m": a_m,
        "b_abs_m": b_m,
        "movement_m": math.hypot(a_m, b_m),
        "posterior_diameter_upper_m": score_m,
        "constant_1000_slack_m": constant_1000_slack_m(a_m, b_m),
        "correlated_slack_m": correlated_slack_m(a_m, b_m),
    }


def search_domain(domain: str, polygons: dict[int, list[tuple[float, float]]]):
    score_cache: dict[tuple[int, float, float, float], float] = {}

    def score(level: SearchLevel, a_m: float, b_m: float) -> float:
        key = (level.circle_sides, level.bearing_bin_deg, a_m, b_m)
        if key not in score_cache:
            score_cache[key] = upper(
                polygons[level.circle_sides],
                (a_m, b_m),
                HALF_ANGLE_DEG,
                level.bearing_bin_deg,
            )
        return score_cache[key]

    traces: dict[float, list[dict]] = {budget: [] for budget in BUDGETS_M}
    incumbents: dict[float, tuple[float, float]] = {}
    finalist_pools: dict[float, set[tuple[float, float]]] = {
        budget: set() for budget in BUDGETS_M
    }

    for level_index, level in enumerate(SEARCH_LEVELS):
        per_budget_points: dict[float, set[tuple[float, float]]] = {
            budget: set() for budget in BUDGETS_M
        }
        if level_index == 0:
            points = {
                (a_m, b_m)
                for a_m in frange(0.0, 1000.0, level.step_m)
                for b_m in frange(0.0, 1000.0, level.step_m)
                if safe(domain, a_m, b_m)
            }
            for budget in BUDGETS_M:
                per_budget_points[budget] = {
                    point for point in points if math.hypot(*point) <= budget + 1e-9
                }
        else:
            assert level.half_width_m is not None
            for budget in BUDGETS_M:
                center_a, center_b = incumbents[budget]
                start_a = max(0.0, center_a - level.half_width_m)
                stop_a = min(1000.0, center_a + level.half_width_m)
                start_b = max(0.0, center_b - level.half_width_m)
                stop_b = min(1000.0, center_b + level.half_width_m)
                points = {
                    (a_m, b_m)
                    for a_m in frange(start_a, stop_a, level.step_m)
                    for b_m in frange(start_b, stop_b, level.step_m)
                    if safe(domain, a_m, b_m)
                    and math.hypot(a_m, b_m) <= budget + 1e-9
                }
                points.add(incumbents[budget])
                per_budget_points[budget] = points

        for budget in BUDGETS_M:
            points = per_budget_points[budget]
            if not points:
                raise RuntimeError(f"no safe {domain} candidate for budget {budget}")
            best_score, best_move, best_a, best_b = min(
                (
                    score(level, a_m, b_m),
                    math.hypot(a_m, b_m),
                    a_m,
                    b_m,
                )
                for a_m, b_m in points
            )
            incumbents[budget] = (best_a, best_b)
            finalist_pools[budget].add((best_a, best_b))
            row = candidate_record(best_a, best_b, best_score)
            row.update(
                {
                    "level": level.name,
                    "circle_sides": level.circle_sides,
                    "bearing_bin_deg": level.bearing_bin_deg,
                    "spatial_step_m": level.step_m,
                    "evaluated_candidates": len(points),
                }
            )
            traces[budget].append(row)

    return traces, finalist_pools


def final_select(
    domain: str,
    budget_m: float,
    pool: set[tuple[float, float]],
    verify_poly,
):
    # Re-rank the winner from each search level under the finest evaluator.
    # This tests resolution stability without spending most of the run on
    # candidates that were already dominated within their own search level.
    candidates = [
        point
        for point in pool
        if safe(domain, *point) and math.hypot(*point) <= budget_m + 1e-9
    ]
    best_score, best_move, best_a, best_b = min(
        (
            upper(verify_poly, point, HALF_ANGLE_DEG, VERIFY_LEVELS[-1][1]),
            math.hypot(*point),
            point[0],
            point[1],
        )
        for point in candidates
    )
    row = candidate_record(best_a, best_b, best_score)
    row["evaluated_finalist_count"] = len(candidates)
    return row


def evaluator_convergence(a_m: float, b_m: float, polygons: dict):
    rows = []
    for circle_sides, bearing_bin_deg in VERIFY_LEVELS:
        value = upper(
            polygons[circle_sides],
            (a_m, b_m),
            HALF_ANGLE_DEG,
            bearing_bin_deg,
        )
        rows.append(
            {
                "circle_sides": circle_sides,
                "bearing_bin_deg": bearing_bin_deg,
                "posterior_diameter_upper_m": value,
            }
        )
    return rows


def dense_safety_sanity(a_m: float, b_m: float, correlated: bool) -> dict:
    """Finite sanity check only; the analytic certificate carries the proof."""
    min_slack = math.inf
    argmin = None
    for radial_step in range(0, 601):
        radius_m = 1500.0 * radial_step / 600.0
        guaranteed = (
            max(GUARANTEED_RADIUS_M, radius_m)
            if correlated
            else GUARANTEED_RADIUS_M
        )
        for angle_step in range(0, 81):
            angle_deg = -HALF_ANGLE_DEG + 2.0 * HALF_ANGLE_DEG * angle_step / 80.0
            angle = math.radians(angle_deg)
            source = (radius_m * math.cos(angle), radius_m * math.sin(angle))
            slack = guaranteed - math.dist((a_m, b_m), source)
            if slack < min_slack:
                min_slack = slack
                argmin = {"radius_m": radius_m, "angle_deg": angle_deg}
    return {
        "sampled_min_slack_m": min_slack,
        "sampled_argmin": argmin,
        "radial_step_m": 2.5,
        "angular_step_deg": 0.025,
        "proof_status": "sanity_check_only_analytic_certificate_is_authoritative",
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    polygons = {
        sides: sector_outer(HALF_ANGLE_DEG, sides)
        for sides, _ in VERIFY_LEVELS
    }

    domains = ("constant_1000", "first_reception_correlated")
    search_traces = {}
    pools = {}
    for domain in domains:
        search_traces[domain], pools[domain] = search_domain(domain, polygons)

    # The correlated set contains the constant-radius set.  Include every
    # constant-domain candidate in the correlated final re-ranking pool so a
    # reported exploratory improvement cannot be an artifact of excluding the
    # existing candidate.
    for budget in BUDGETS_M:
        pools["first_reception_correlated"][budget].update(
            pools["constant_1000"][budget]
        )

    selected = {domain: {} for domain in domains}
    convergence = {domain: {} for domain in domains}
    safety_checks = {domain: {} for domain in domains}
    for domain in domains:
        for budget in BUDGETS_M:
            row = final_select(
                domain,
                budget,
                pools[domain][budget],
                polygons[VERIFY_LEVELS[-1][0]],
            )
            selected[domain][str(int(budget))] = row
            convergence[domain][str(int(budget))] = evaluator_convergence(
                row["a_m"], row["b_abs_m"], polygons
            )
            safety_checks[domain][str(int(budget))] = dense_safety_sanity(
                row["a_m"],
                row["b_abs_m"],
                correlated=(domain == "first_reception_correlated"),
            )

    area_rows = []
    for step_m in (2.0, 1.0, 0.5, 0.25):
        constant_area = certified_area_m2("constant_1000", step_m)
        correlated_area = certified_area_m2("first_reception_correlated", step_m)
        area_rows.append(
            {
                "quadrature_step_m": step_m,
                "constant_1000_area_m2": constant_area,
                "first_reception_correlated_area_m2": correlated_area,
                "area_ratio": correlated_area / constant_area,
            }
        )

    comparisons = []
    improvements_over_2pct = 0
    nonworse_count = 0
    max_followup_change_pct = 0.0
    for budget in BUDGETS_M:
        key = str(int(budget))
        old = selected["constant_1000"][key]
        new = selected["first_reception_correlated"][key]
        improvement_pct = 100.0 * (
            old["posterior_diameter_upper_m"]
            - new["posterior_diameter_upper_m"]
        ) / old["posterior_diameter_upper_m"]
        if improvement_pct >= 2.0:
            improvements_over_2pct += 1
        if improvement_pct >= -1e-9:
            nonworse_count += 1
        verify_values = [
            row["posterior_diameter_upper_m"]
            for row in convergence["first_reception_correlated"][key]
        ]
        followup_change_pct = (
            100.0 * abs(verify_values[-1] - verify_values[-2]) / verify_values[-1]
        )
        max_followup_change_pct = max(
            max_followup_change_pct, followup_change_pct
        )
        comparisons.append(
            {
                "budget_m": budget,
                "constant_1000_upper_m": old["posterior_diameter_upper_m"],
                "correlated_upper_m": new["posterior_diameter_upper_m"],
                "upper_improvement_pct": improvement_pct,
                "constant_1000_movement_m": old["movement_m"],
                "correlated_movement_m": new["movement_m"],
                "correlated_last_refinement_change_pct": followup_change_pct,
            }
        )

    followup_screening_rule = {
        "declared_before_interpretation": True,
        "exploratory_only": True,
        "reason": "the initial screen missed only its 1 percent resolution-stability threshold; this follow-up adds one predeclared finer level and does not alter the saved initial result",
        "requirements": {
            "finest_area_ratio_at_least": 1.10,
            "nonworse_budget_count": len(BUDGETS_M),
            "budgets_improved_at_least_2pct": 3,
            "max_2880_to_5760_evaluator_change_pct": 0.75,
            "all_selected_analytic_slack_at_least_m": SAFETY_MARGIN_M,
        },
        "consequence_if_met": "worthy_of_human_method_selection_review_not_automatic_paper_upgrade",
        "consequence_if_not_met": "retain_constant_1000_paper_strategy_and_record_negative_result",
    }
    all_safe = all(
        selected[domain][str(int(budget))][
            "correlated_slack_m" if domain == "first_reception_correlated" else "constant_1000_slack_m"
        ]
        >= SAFETY_MARGIN_M - 1e-9
        for domain in domains
        for budget in BUDGETS_M
    )
    followup_screen_pass = (
        area_rows[-1]["area_ratio"] >= 1.10
        and nonworse_count == len(BUDGETS_M)
        and improvements_over_2pct >= 3
        and max_followup_change_pct <= 0.75
        and all_safe
    )
    prior_stage1 = None
    if STAGE1_PATH.exists():
        prior_data = json.loads(STAGE1_PATH.read_text(encoding="utf-8"))
        prior_stage1 = {
            "path": "results/paper_q12_geometry_v2/q2_correlated_reception_domain_diagnostic_stage1.json",
            "sha256": sha256(STAGE1_PATH),
            "screening_result": prior_data.get("screening_result"),
        }

    data = {
        "schema_version": 1,
        "artifact_type": "q2_correlated_reception_exploratory_diagnostic",
        "status": "EXPLORATORY_NOT_APPROVED_METHOD_OR_BASELINE",
        "scope": "complete 1500 m sector example after one successful omnidirectional reception",
        "constants": {
            "half_angle_deg": HALF_ANGLE_DEG,
            "sector_radius_m": SECTOR_RADIUS_M,
            "guaranteed_minimum_reception_radius_m": GUARANTEED_RADIUS_M,
            "continuous_safety_margin_m": SAFETY_MARGIN_M,
            "budgets_m": list(BUDGETS_M),
        },
        "continuous_certificate": {
            "unknown_radius_lower_bound": "R(g) >= max(1000, ||g-p1||)",
            "candidate_coordinates": "x=(a,b), a>=0, b>=0; mirror gives the other side",
            "constant_1000_check": "max(||x||, ||x-1500*u_opposite||) <= 999.9",
            "correlated_check": "max(||x||, ||x-1000*u_opposite||) <= 999.9",
            "reason": [
                "for r<=1000, squared distance is convex in r and is maximized at r=0 or r=1000",
                "for r>=1000, reception slack r-||x-r*u|| is nondecreasing in r, so r=1000 is worst",
                "over the bearing sector, the endpoint opposite the lateral displacement is worst",
            ],
            "finite_sampling_used_as_proof": False,
        },
        "followup_screening_rule": followup_screening_rule,
        "preserved_initial_screen": prior_stage1,
        "certified_domain_area_convergence": area_rows,
        "search_levels": [level.__dict__ for level in SEARCH_LEVELS],
        "search_traces": {
            domain: {
                str(int(budget)): rows
                for budget, rows in search_traces[domain].items()
            }
            for domain in domains
        },
        "finest_selected_candidates": selected,
        "evaluator_convergence": convergence,
        "dense_safety_sanity_checks": safety_checks,
        "paired_budget_comparison": comparisons,
        "screening_result": {
            "status": "PASS" if followup_screen_pass else "FAIL",
            "stage": "predeclared_finer_resolution_followup",
            "finest_area_ratio": area_rows[-1]["area_ratio"],
            "nonworse_budget_count": nonworse_count,
            "budgets_improved_at_least_2pct": improvements_over_2pct,
            "max_2880_to_5760_evaluator_change_pct": max_followup_change_pct,
            "all_selected_candidates_meet_analytic_margin": all_safe,
            "interpretation": followup_screening_rule[
                "consequence_if_met"
                if followup_screen_pass
                else "consequence_if_not_met"
            ],
        },
        "limitations": [
            "finite candidate search; no continuous global optimality claim",
            "posterior diameter is a conservative angular-bin and outer-polygon upper bound",
            "complete-sector example only; target-disk truncation and arbitrary first-point geometry are not yet swept",
            "no simulator, live API, practice test, or formal test was used",
            "screening success cannot create user approval, a method card, a baseline, or a canonical run summary",
        ],
        "inputs": [
            {
                "path": "code/shared/analyze_q2_correlated_reception_domain.py",
                "sha256": sha256(
                    ROOT
                    / "code"
                    / "shared"
                    / "analyze_q2_correlated_reception_domain.py"
                ),
            },
            {
                "path": "code/shared/verify_paper_q12_geometry.py",
                "sha256": sha256(ROOT / "code" / "shared" / "verify_paper_q12_geometry.py"),
            },
            {
                "path": "results/paper_q12_geometry/verification.json",
                "sha256": sha256(ROOT / "results" / "paper_q12_geometry" / "verification.json"),
            },
            {
                "path": "methods/Q2/q2_second_point_notes_v2.md",
                "sha256": sha256(ROOT / "methods" / "Q2" / "q2_second_point_notes_v2.md"),
            },
            *(
                [
                    {
                        "path": "results/paper_q12_geometry_v2/q2_correlated_reception_domain_diagnostic_stage1.json",
                        "sha256": sha256(STAGE1_PATH),
                    }
                ]
                if STAGE1_PATH.exists()
                else []
            ),
        ],
    }
    OUT_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUT_PATH),
                "screening_result": data["screening_result"],
                "paired_budget_comparison": comparisons,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
