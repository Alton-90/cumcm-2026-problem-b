"""Exploratory robustness study for the Q2 correlated-reception enhancement.

This is deliberately not a formal R6 robustness artifact: Q2 does not yet have
an approved method card, baseline decision, or canonical run summary.  The
script records predeclared numerical checks and keeps the comparison evidence
separate from submission-gate status.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from select_second_detection_point import (
    initial_outer_polygon,
    posterior_diameter_upper,
    select_second_point,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "Q2" / "diagnostics" / "correlated_reception_round1"

PREDECLARED_RULES = {
    "dense_reception_margin_m": {
        "threshold": 0.09,
        "direction": "at_least",
        "reason": "the analytic construction reserves 0.1 m; the dense audit may lose at most 0.01 m to numerical evaluation",
    },
    "last_angular_refinement_change_m": {
        "threshold": 3.0,
        "direction": "at_most",
        "reason": "a final 0.0625 degree check should change the 0.125 degree bound by no more than 3 m",
    },
    "spatial_resolution_objective_range_m": {
        "threshold": 1.5,
        "direction": "at_most",
        "reason": "the selected objective should be stable across 12, 18, and 30 coarse cells",
    },
    "rotation_local_coordinate_error_m": {
        "threshold": 1.0e-6,
        "direction": "at_most",
        "reason": "a 90 degree rotation about the target-domain centre is an exact symmetry",
    },
}


def dense_minimum_reception_slack(
    point: tuple[float, float],
    first_point: tuple[float, float],
    bearing_deg: float,
    *,
    radial_steps: int = 400,
    angular_steps: int = 240,
) -> float:
    minimum = math.inf
    for angular_index in range(angular_steps + 1):
        offset_deg = -1.0 + 2.0 * angular_index / angular_steps
        angle = math.radians(bearing_deg + offset_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        for radial_index in range(radial_steps + 1):
            radius = 1500.0 * radial_index / radial_steps
            source = (
                first_point[0] + radius * ux,
                first_point[1] + radius * uy,
            )
            if math.hypot(*source) > 1800.0 + 1.0e-9:
                continue
            effective_radius = max(1000.0, radius)
            minimum = min(minimum, effective_radius - math.dist(point, source))
    return minimum


def compact(result: dict) -> dict:
    return {
        "input": result["input"],
        "recommended_point": result["recommended_point"],
        "safe_polygon_vertices": len(result["guaranteed_reception_inner_polygon_world_m"]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    conservative_600 = select_second_point(
        (0.0, 0.0),
        0.0,
        budget_m=600.0,
        reception_mode="conservative_1000",
    )
    correlated_600 = select_second_point(
        (0.0, 0.0),
        0.0,
        budget_m=600.0,
        reception_mode="correlated_initial_reception",
    )
    correlated_600_rotated = select_second_point(
        (0.0, 0.0),
        90.0,
        budget_m=600.0,
        reception_mode="correlated_initial_reception",
    )

    fixed_point = tuple(correlated_600["recommended_point"]["world_m"])
    angular_rows = []
    fine_source_polygon = initial_outer_polygon(
        (0.0, 0.0), 0.0, circle_sides=1440
    )
    for width in (1.0, 0.5, 0.25, 0.125, 0.0625):
        angular_rows.append(
            {
                "bearing_bin_deg": width,
                "posterior_diameter_upper_m": posterior_diameter_upper(
                    fine_source_polygon, fixed_point, bearing_bin_deg=width
                ),
            }
        )

    circle_rows = []
    for sides in (90, 180, 360, 720, 1440):
        polygon = initial_outer_polygon((0.0, 0.0), 0.0, circle_sides=sides)
        circle_rows.append(
            {
                "circle_sides": sides,
                "posterior_diameter_upper_m": posterior_diameter_upper(
                    polygon, fixed_point, bearing_bin_deg=0.125
                ),
            }
        )

    spatial_rows = []
    for cells in (12, 18, 30):
        result = select_second_point(
            (0.0, 0.0),
            0.0,
            budget_m=600.0,
            coarse_cells=cells,
            reception_mode="correlated_initial_reception",
        )
        point = result["recommended_point"]
        spatial_rows.append(
            {
                "coarse_cells": cells,
                "local_m": point["local_m"],
                "movement_m": point["movement_m"],
                "posterior_diameter_upper_m": point[
                    "posterior_diameter_upper_m"
                ],
            }
        )

    budget_rows = []
    for budget in (300.0, 400.0, 500.0, 600.0, 800.0, 1000.0):
        result = select_second_point(
            (0.0, 0.0),
            0.0,
            budget_m=budget,
            reception_mode="correlated_initial_reception",
        )
        budget_rows.append(
            {
                "budget_m": budget,
                **result["recommended_point"],
            }
        )

    safety_points = [fixed_point]
    correlated_boundary = correlated_600[
        "guaranteed_reception_inner_polygon_world_m"
    ]
    safety_points.extend(
        tuple(point) for point in correlated_boundary[:: max(1, len(correlated_boundary) // 24)]
    )
    dense_slacks = [
        dense_minimum_reception_slack(point, (0.0, 0.0), 0.0)
        for point in safety_points
    ]

    east = correlated_600["recommended_point"]
    north = correlated_600_rotated["recommended_point"]
    rotation_error = max(
        abs(east["local_m"][0] - north["local_m"][0]),
        abs(east["local_m"][1] - north["local_m"][1]),
        abs(
            east["posterior_diameter_upper_m"]
            - north["posterior_diameter_upper_m"]
        ),
    )
    observed = {
        "dense_reception_margin_m": min(dense_slacks),
        "last_angular_refinement_change_m": abs(
            angular_rows[-1]["posterior_diameter_upper_m"]
            - angular_rows[-2]["posterior_diameter_upper_m"]
        ),
        "spatial_resolution_objective_range_m": (
            max(row["posterior_diameter_upper_m"] for row in spatial_rows)
            - min(row["posterior_diameter_upper_m"] for row in spatial_rows)
        ),
        "rotation_local_coordinate_error_m": rotation_error,
    }
    checks = {
        "dense_reception_margin": observed["dense_reception_margin_m"]
        >= PREDECLARED_RULES["dense_reception_margin_m"]["threshold"],
        "angular_refinement": observed["last_angular_refinement_change_m"]
        <= PREDECLARED_RULES["last_angular_refinement_change_m"]["threshold"],
        "spatial_resolution": observed["spatial_resolution_objective_range_m"]
        <= PREDECLARED_RULES["spatial_resolution_objective_range_m"]["threshold"],
        "rotation_equivariance": observed["rotation_local_coordinate_error_m"]
        <= PREDECLARED_RULES["rotation_local_coordinate_error_m"]["threshold"],
    }
    if not all(checks.values()):
        raise AssertionError({name: value for name, value in checks.items() if not value})

    conservative_value = conservative_600["recommended_point"][
        "posterior_diameter_upper_m"
    ]
    correlated_value = correlated_600["recommended_point"][
        "posterior_diameter_upper_m"
    ]
    summary = {
        "schema_version": 1,
        "artifact_type": "q2_exploratory_robustness_diagnostic",
        "formal_r6_robustness_claimed": False,
        "predeclared_rules": PREDECLARED_RULES,
        "observed": observed,
        "checks": checks,
        "all_predeclared_checks_pass": True,
        "canonical_comparison": {
            "conservative_1000": compact(conservative_600),
            "correlated_initial_reception": compact(correlated_600),
            "upper_bound_reduction_m": conservative_value - correlated_value,
            "upper_bound_reduction_percent": 100.0
            * (conservative_value - correlated_value)
            / conservative_value,
            "interpretation": "exploratory comparison; no pass threshold was chosen",
        },
        "angular_resolution": angular_rows,
        "circle_resolution": circle_rows,
        "spatial_resolution": spatial_rows,
        "correlated_budget_frontier": budget_rows,
        "dense_safety_audit": {
            "points_checked": len(safety_points),
            "radial_steps": 400,
            "angular_steps": 240,
            "minimum_sampled_reception_slack_m": min(dense_slacks),
        },
        "limitations": [
            "The dense audit supplements but does not replace the analytic reception certificate.",
            "The posterior objective remains a conservative finite angular-bin upper bound.",
            "No continuous global optimum is claimed.",
            "Q2 lacks the approved method card and baseline contract required for a formal R6 robustness verdict.",
        ],
    }
    output = OUT / "q2_correlated_reception_diagnostic.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
