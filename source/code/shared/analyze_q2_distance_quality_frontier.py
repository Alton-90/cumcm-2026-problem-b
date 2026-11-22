"""Deterministic Q2 distance--quality diagnostic for the complete-sector example.

This is a paper diagnostic, not a simulator policy and not a global optimizer.
It uses an analytic continuous reception certificate and a conservative angular
binning upper bound for the posterior diameter.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from verify_paper_q12_geometry import farthest_sector, sector_outer, upper


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "paper_q12_geometry_v2"
TABLE_PATH = ROOT / "paper" / "latex" / "full_q2_pareto_table_v3.tex"

HALF_ANGLE_DEG = 1.0
SECTOR_RADIUS_M = 1500.0
RECEPTION_RADIUS_M = 1000.0
SAFETY_MARGIN_M = 0.1
BUDGETS_M = (550.0, 600.0, 700.0, 800.0, 900.0, 1000.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frange(start: float, stop: float, step: float):
    count = int(round((stop - start) / step))
    for index in range(count + 1):
        yield start + index * step


def feasible(a_m: float, b_m: float, budget_m: float) -> bool:
    movement_m = math.hypot(a_m, b_m)
    max_distance_m = farthest_sector(a_m, b_m, HALF_ANGLE_DEG)
    delta = math.radians(HALF_ANGLE_DEG)
    return (
        movement_m <= budget_m + 1e-9
        and a_m * math.cos(delta) >= b_m * math.sin(delta) - 1e-9
        and max_distance_m <= RECEPTION_RADIUS_M - SAFETY_MARGIN_M + 1e-9
    )


def score(poly, angular_step_deg: float, a_m: float, b_m: float):
    return upper(poly, (a_m, b_m), HALF_ANGLE_DEG, angular_step_deg)


def best_on_grid(
    budget_m: float,
    poly,
    angular_step_deg: float,
    a_values,
    b_values,
):
    a_values = tuple(a_values)
    b_values = tuple(b_values)
    candidates = []
    for a_m in a_values:
        for b_m in b_values:
            if not feasible(a_m, b_m, budget_m):
                continue
            movement_m = math.hypot(a_m, b_m)
            max_distance_m = farthest_sector(a_m, b_m, HALF_ANGLE_DEG)
            upper_m = score(poly, angular_step_deg, a_m, b_m)
            candidates.append((upper_m, movement_m, a_m, b_m, max_distance_m))
    if not candidates:
        raise RuntimeError(f"no safe candidate for budget {budget_m}")
    return min(candidates)


def optimize_budget(budget_m: float, polygons):
    coarse = best_on_grid(
        budget_m,
        polygons[0],
        0.5,
        frange(450.0, 1000.0, 25.0),
        frange(0.0, 1000.0, 25.0),
    )
    _, _, coarse_a, coarse_b, _ = coarse
    medium = best_on_grid(
        budget_m,
        polygons[1],
        0.25,
        frange(coarse_a - 50.0, coarse_a + 50.0, 5.0),
        frange(max(0.0, coarse_b - 50.0), coarse_b + 50.0, 5.0),
    )
    _, _, medium_a, medium_b, _ = medium
    fine = best_on_grid(
        budget_m,
        polygons[2],
        0.125,
        frange(medium_a - 20.0, medium_a + 20.0, 2.5),
        frange(max(0.0, medium_b - 20.0), medium_b + 20.0, 2.5),
    )
    upper_m, movement_m, a_m, b_m, max_distance_m = fine
    return {
        "budget_m": budget_m,
        "a_m": a_m,
        "b_abs_m": b_m,
        "symmetric_points_local_m": [[a_m, b_m], [a_m, -b_m]],
        "movement_m": movement_m,
        "continuous_sector_max_distance_m": max_distance_m,
        "reception_margin_m": RECEPTION_RADIUS_M - max_distance_m,
        "posterior_diameter_upper_m": upper_m,
        "selection_scope": "best point found by the declared nested finite grids",
    }


def write_table(rows):
    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{完整扇形例的行走预算--定位质量候选前沿}\label{tab:q2pareto}",
        r"\begin{tabular}{rrrrr}\toprule",
        r"预算（m）&局部坐标$(a,|b|)$（m）&实际移动（m）&接收余量（m）&直径上界（m）\\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['budget_m']:.0f}&({row['a_m']:.1f},{row['b_abs_m']:.1f})"
            f"&{row['movement_m']:.2f}&{row['reception_margin_m']:.2f}"
            f"&{row['posterior_diameter_upper_m']:.2f}\\\\"
        )
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    polygons = (
        sector_outer(HALF_ANGLE_DEG, 360),
        sector_outer(HALF_ANGLE_DEG, 720),
        sector_outer(HALF_ANGLE_DEG, 1440),
    )
    rows = [optimize_budget(budget, polygons) for budget in BUDGETS_M]

    delta = math.radians(HALF_ANGLE_DEG)
    axis_a = SECTOR_RADIUS_M / (2.0 * math.cos(delta))
    b_max = min(
        math.sqrt(RECEPTION_RADIUS_M**2 - axis_a**2),
        math.sqrt(
            RECEPTION_RADIUS_M**2
            - (axis_a - SECTOR_RADIUS_M * math.cos(delta)) ** 2
        )
        - SECTOR_RADIUS_M * math.sin(delta),
    )
    legacy_b = b_max - 0.01
    axis = {
        "a_m": axis_a,
        "b_abs_m": 0.0,
        "movement_m": axis_a,
        "continuous_sector_max_distance_m": farthest_sector(
            axis_a, 0.0, HALF_ANGLE_DEG
        ),
        "posterior_diameter_upper_m": score(
            polygons[2], 0.125, axis_a, 0.0
        ),
    }
    extreme = {
        "a_m": axis_a,
        "b_abs_m": legacy_b,
        "movement_m": math.hypot(axis_a, legacy_b),
        "continuous_sector_max_distance_m": farthest_sector(
            axis_a, legacy_b, HALF_ANGLE_DEG
        ),
        "posterior_diameter_upper_m": score(
            polygons[2], 0.125, axis_a, legacy_b
        ),
    }
    data = {
        "schema_version": 1,
        "artifact_type": "q2_distance_quality_diagnostic",
        "scope": "complete 1500 m sector example; paper diagnostic only",
        "constants": {
            "half_angle_deg": HALF_ANGLE_DEG,
            "sector_radius_m": SECTOR_RADIUS_M,
            "guaranteed_reception_radius_m": RECEPTION_RADIUS_M,
            "continuous_reception_safety_margin_m": SAFETY_MARGIN_M,
        },
        "continuous_reception_certificate": {
            "statement": "for x=(a,b), the sector maximum is the maximum of the distance to the apex and to the opposite outer-arc endpoint",
            "checked_condition": "maximum distance <= 999.9 m",
            "finite_sampling_used_for_reception": False,
        },
        "posterior_bound": {
            "outer_circle_sides": 1440,
            "second_bearing_bin_width_deg": 0.125,
            "expanded_half_angle_deg": 1.0625,
            "meaning": "conservative posterior-diameter upper bound for each saved candidate",
        },
        "search": {
            "levels": [
                {"a_step_m": 25.0, "b_step_m": 25.0, "circle_sides": 360, "bearing_bin_deg": 0.5},
                {"local_half_width_m": 50.0, "step_m": 5.0, "circle_sides": 720, "bearing_bin_deg": 0.25},
                {"local_half_width_m": 20.0, "step_m": 2.5, "circle_sides": 1440, "bearing_bin_deg": 0.125},
            ],
            "global_continuous_optimality_claimed": False,
        },
        "axis_reference": axis,
        "previous_extreme_lateral_reference": extreme,
        "budget_frontier_candidates": rows,
        "interpretation": {
            "candidate_frontier_not_single_mandatory_point": True,
            "route_planner_may_choose_budget": True,
            "no_probability_model_assumed": True,
            "not_a_simulator_score": True,
        },
        "inputs": [
            {
                "path": "code/shared/verify_paper_q12_geometry.py",
                "sha256": sha256(ROOT / "code" / "shared" / "verify_paper_q12_geometry.py"),
            },
            {
                "path": "results/paper_q12_geometry/verification.json",
                "sha256": sha256(ROOT / "results" / "paper_q12_geometry" / "verification.json"),
            },
        ],
    }
    out_path = OUT_DIR / "q2_distance_quality_frontier.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_table(rows)
    print(json.dumps({"output": str(out_path), "table": str(TABLE_PATH), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
