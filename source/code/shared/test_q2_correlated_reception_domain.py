"""Regression tests for the exploratory Q2 correlated-radius diagnostic."""
from __future__ import annotations

import json
import math
import unittest

import analyze_q2_correlated_reception_domain as diagnostic


class CorrelatedReceptionCertificateTests(unittest.TestCase):
    def test_constant_domain_is_subset_on_dense_candidate_grid(self):
        for a_m in range(0, 1001, 25):
            for b_m in range(0, 1001, 25):
                if diagnostic.safe("constant_1000", a_m, b_m):
                    self.assertTrue(
                        diagnostic.safe("first_reception_correlated", a_m, b_m),
                        (a_m, b_m),
                    )

    def test_correlated_slack_is_no_larger_than_dense_sample_minimum(self):
        candidates = (
            (285.0, 470.0),
            (360.0, 480.0),
            (510.0, 477.5),
            (592.5, 537.5),
            (720.0, 540.0),
            (835.0, 550.0),
        )
        for a_m, b_m in candidates:
            analytic = diagnostic.correlated_slack_m(a_m, b_m)
            sampled = math.inf
            for radial_step in range(0, 301):
                radius_m = 5.0 * radial_step
                reception_radius_m = max(1000.0, radius_m)
                for angle_step in range(0, 41):
                    angle_deg = -1.0 + 0.05 * angle_step
                    angle = math.radians(angle_deg)
                    source = (
                        radius_m * math.cos(angle),
                        radius_m * math.sin(angle),
                    )
                    sampled = min(
                        sampled,
                        reception_radius_m - math.dist((a_m, b_m), source),
                    )
            self.assertGreaterEqual(sampled + 1e-9, analytic)
            self.assertLess(sampled - analytic, 0.05)

    def test_saved_result_remains_exploratory_and_safe(self):
        data = json.loads(diagnostic.OUT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            data["status"], "EXPLORATORY_NOT_APPROVED_METHOD_OR_BASELINE"
        )
        self.assertEqual(
            data["preserved_initial_screen"]["screening_result"]["status"],
            "FAIL",
        )
        self.assertEqual(data["screening_result"]["status"], "PASS")
        for row in data["finest_selected_candidates"][
            "first_reception_correlated"
        ].values():
            self.assertGreaterEqual(
                row["correlated_slack_m"] + 1e-9,
                diagnostic.SAFETY_MARGIN_M,
            )


if __name__ == "__main__":
    unittest.main()
