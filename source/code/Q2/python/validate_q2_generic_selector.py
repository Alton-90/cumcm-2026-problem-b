"""Deterministic regression checks for the generic Q2 selector."""
from __future__ import annotations

import json
import math
from pathlib import Path

from select_second_detection_point import select_second_point


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "Q2" / "generic_selector"


def close(a: float, b: float, tolerance: float = 1.0e-6) -> bool:
    return abs(a - b) <= tolerance


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    origin_east = select_second_point((0.0, 0.0), 0.0, budget_m=1000.0)
    origin_north = select_second_point((0.0, 0.0), 90.0, budget_m=1000.0)
    boundary_outward = select_second_point((1500.0, 0.0), 0.0, budget_m=1000.0)
    budget_600 = select_second_point((0.0, 0.0), 0.0, budget_m=600.0)
    conservative_600 = select_second_point(
        (0.0, 0.0),
        0.0,
        budget_m=600.0,
        reception_mode="conservative_1000",
    )

    east = origin_east["recommended_point"]
    north = origin_north["recommended_point"]
    boundary = boundary_outward["recommended_point"]
    short = budget_600["recommended_point"]

    checks = {
        "correlated_mode_is_default": (
            origin_east["input"]["reception_mode"]
            == "correlated_initial_reception"
        ),
        "continuous_reception_certificate_present": (
            origin_east["certificate"]["safe_polygon_is_continuously_safe"]
        ),
        "conservative_fallback_certificate": (
            conservative_600["certificate"]["safe_polygon_max_endpoint_distance_m"]
            <= 999.900001
        ),
        "rotation_local_x": close(east["local_m"][0], north["local_m"][0]),
        "rotation_local_y": close(east["local_m"][1], north["local_m"][1]),
        "rotation_quality": close(
            east["posterior_diameter_upper_m"],
            north["posterior_diameter_upper_m"],
        ),
        "rotation_world_point": (
            close(north["world_m"][0], -east["world_m"][1])
            and close(north["world_m"][1], east["world_m"][0])
        ),
        "boundary_truncation_improves_quality": (
            boundary["posterior_diameter_upper_m"]
            < 0.25 * east["posterior_diameter_upper_m"]
        ),
        "boundary_truncation_shortens_move": boundary["movement_m"] < 400.0,
        "budget_600_respected": short["movement_m"] <= 600.000001,
        "correlated_mode_improves_budget_600_bound": (
            short["posterior_diameter_upper_m"]
            < conservative_600["recommended_point"]["posterior_diameter_upper_m"]
        ),
        "all_recommended_points_safe": all(
            row["reception_margin_m"] >= -1.0e-6
            for row in (east, north, boundary, short)
        ),
    }
    if not all(checks.values()):
        raise AssertionError({name: passed for name, passed in checks.items() if not passed})

    for name, data in (
        ("origin_east", origin_east),
        ("origin_north", origin_north),
        ("boundary_outward", boundary_outward),
        ("origin_east_budget600", budget_600),
        ("origin_east_budget600_conservative", conservative_600),
    ):
        (OUT / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    summary = {
        "schema_version": 1,
        "artifact_type": "q2_generic_selector_validation",
        "checks": checks,
        "all_checks_pass": True,
        "representative_results": {
            "origin_east_budget_1000": east,
            "origin_east_budget_600": short,
            "origin_east_budget_600_conservative": conservative_600[
                "recommended_point"
            ],
            "boundary_outward_budget_1000": boundary,
        },
        "interpretation": {
            "boundary_case_first_source_set_is_truncated_by_target_disk": True,
            "results_are_deterministic_finite_grid_upper_bounds": True,
            "continuous_global_optimality_claimed": False,
            "simulator_score": False,
        },
    }
    (OUT / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
