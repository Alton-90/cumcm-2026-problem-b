"""Offline regression for incumbent retention and the finite Q2 Pareto set.

This diagnoses the existing selector; it is not a formal test or R6 approval.
Outputs go to a new directory, preserving the earlier numerical evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import select_second_detection_point as selector


ROOT = Path(__file__).resolve().parents[3]


def point_key(row: dict) -> tuple:
    return tuple(round(value, 7) for value in row["world_m"])


def check_case(name: str, first: tuple, bearing: float, budget: float) -> dict:
    seen = []
    original = selector.candidate_record

    def record(*args, **kwargs):
        row = original(*args, **kwargs)
        seen.append(row)
        return row

    with patch.object(selector, "candidate_record", record):
        result = selector.select_second_point(
            first, bearing, budget_m=budget, circle_sides=72,
            safe_boundary_directions=36, coarse_cells=4,
            final_bearing_bin_deg=2.0,
        )
    unique = {point_key(row): row for row in seen}
    rows = list(unique.values())
    selected = result["recommended_point"]
    best_upper = min(row["posterior_diameter_upper_m"] for row in rows)
    assert selected["posterior_diameter_upper_m"] <= best_upper + 1.0e-6
    assert len(rows) == result["evaluated_unique_candidate_count"]

    # Independent exhaustive dominance check, not the production sort/scan.
    expected = {
        point_key(row) for row in rows
        if not any(
            other["movement_m"] <= row["movement_m"]
            and other["posterior_diameter_upper_m"] <= row["posterior_diameter_upper_m"]
            and (
                other["movement_m"] < row["movement_m"]
                or other["posterior_diameter_upper_m"] < row["posterior_diameter_upper_m"]
            )
            for other in rows
        )
    }
    actual_rows = result["finite_candidate_pareto_frontier"]
    # Equal objective pairs may share one representative on the frontier.
    objective = lambda row: (row["movement_m"], row["posterior_diameter_upper_m"])
    assert {objective(unique[key]) for key in expected} == {objective(row) for row in actual_rows}
    threshold = selected["posterior_diameter_upper_m"] * 1.05
    assert {point_key(row) for row in result["near_optimal_candidate_points"]} == {
        point_key(row) for row in rows
        if row["posterior_diameter_upper_m"] <= threshold + selector.EPS
    }
    assert all(row["movement_m"] <= budget + 1.0e-7 for row in actual_rows)
    assert all(row["correlated_reception_certificate"]["feasible"] for row in actual_rows)

    # A dense radial/angle diagnostic is separate from the analytic certificate.
    # Keep only sources inside the target disk, as required by this case.
    minimum_sample_slack = math.inf
    for row in actual_rows:
        for radial_index in range(1, 31):
            radius = radial_index * 50.0
            for angular_index in range(21):
                angle = math.radians(bearing - 1.0 + angular_index * 0.1)
                source = (first[0] + radius * math.cos(angle), first[1] + radius * math.sin(angle))
                if math.hypot(*source) <= 1800.0 + 1.0e-9:
                    minimum_sample_slack = min(
                        minimum_sample_slack,
                        max(1000.0, radius) - math.dist(source, row["world_m"]),
                    )
    assert minimum_sample_slack >= 0.09999
    return {
        "case": name, "first_point_m": first, "bearing_deg": bearing,
        "budget_m": budget, "evaluation_count": len(seen),
        "unique_count": len(rows), "pareto_count": len(actual_rows),
        "selected": selected, "best_evaluated_upper_m": best_upper,
        "minimum_sample_reception_slack_m": minimum_sample_slack,
        "checks": {
            "incumbent_not_lost": True, "pareto_matches_exhaustive_dominance": True,
            "near_set_uses_all_evaluated_points": True,
            "frontier_is_budget_feasible_and_certified": True,
            "dense_reception_sanity_check": True,
        },
    }


def main() -> None:
    rows = [
        check_case("origin_east_regression", (0.0, 0.0), 0.0, 600.0),
        check_case("origin_north_rotation", (0.0, 0.0), 90.0, 600.0),
        check_case("boundary_outward", (1500.0, 0.0), 0.0, 600.0),
    ]
    assert abs(rows[0]["best_evaluated_upper_m"] - rows[1]["best_evaluated_upper_m"]) < 1.0e-6
    script = Path(selector.__file__)
    output = {
        "artifact_type": "q2_candidate_retention_diagnostic",
        "status": "PASS", "evidence_scope": "offline regression only; not an R6 gate",
        "selector_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resolution": {"circle_sides": 72, "safe_directions": 36, "coarse_cells": 4, "bearing_bin_deg": 2.0},
        "pre_fix_reproduction": {
            "source_sha256": "ef8ef642f280b49ad813050b66da62c51c500d04aaee0428cd72debacffe6e2d",
            "case": "origin_east_regression", "evaluation_count": 565,
            "returned_upper_m": 267.2366307039837,
            "already_evaluated_feasible_upper_m": 255.13443064279105,
            "already_evaluated_point_m": [424.2640687119284, -424.2640687119286],
            "provenance": "read-only instrumented execution before patch; historical implementation remains in Git",
        },
        "cases": rows,
        "limits": [
            "Coarse resolution is chosen to expose incumbent loss, not to publish localization performance.",
            "Pareto optimality applies only to the evaluated finite candidates.",
            "Sampling checks reception numerically and does not replace the continuous certificate.",
            "No simulator, live API, formal test, or Q3/Q4 locked strategy was executed or changed.",
        ],
    }
    destination = ROOT / "results/Q2/diagnostics/candidate_retention_20260913/validation_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
