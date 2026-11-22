"""Targeted offline tests for Q4 recovery policy; no arena or simulator access."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "shared"))
from search_policy_recovery import run_search


class OpaqueClient:
    virtual_time_s = 0
    def __getattr__(self, name):
        if name in {"sources", "heading", "radius", "seed", "truth"}:
            raise AssertionError(f"policy must not access hidden field {name}")
        raise AttributeError(name)


class ShadowThenPositiveClient(OpaqueClient):
    def __init__(self): self.channel_one_points = []; self.channel_one_calls = 0
    def measure(self, x, y, channel):
        if channel != 1: return {"measure_result": "no_signal"}
        self.channel_one_calls += 1
        self.channel_one_points.append((x, y))
        if self.channel_one_calls == 1: return {"measure_result": "direction", "svd_deg": 0.0}
        if self.channel_one_calls == 2: return {"measure_result": "no_signal"}
        if self.channel_one_calls == 3: return {"measure_result": "direction", "svd_deg": 0.0}
        if self.channel_one_calls == 4: return {"measure_result": "near"}
        return {"measure_result": "no_signal"}
    def clear(self, x, y, channel): return {"clear_result": "success"}


class AllNegativeClient(OpaqueClient):
    def __init__(self): self.channel_one_calls = 0
    def measure(self, x, y, channel):
        if channel == 1:
            self.channel_one_calls += 1
            if self.channel_one_calls == 1: return {"measure_result": "direction", "svd_deg": 0.0}
        return {"measure_result": "no_signal"}
    def clear(self, x, y, channel): return {"clear_result": "no_target_in_range"}


class FailingClient(OpaqueClient):
    def measure(self, x, y, channel): raise ConnectionError("synthetic transport failure")
    def clear(self, x, y, channel): raise AssertionError("clear must not run")


class RecoveryTests(unittest.TestCase):
    def test_shadow_binary_positive_then_localization_continues(self):
        client = ShadowThenPositiveClient()
        result = run_search(client)
        self.assertTrue(result["completed"])
        self.assertEqual(result["cleared_channels"], [1])
        self.assertEqual(result["recovery_count"], 1)
        self.assertEqual(result["recovery_measure_count"], 2)
        self.assertEqual(result["recovery_positive_count"], 1)
        visible, invisible, positive_midpoint, next_midpoint = client.channel_one_points[0:4]
        self.assertAlmostEqual(positive_midpoint[0], (visible[0] + invisible[0]) / 2.0)
        self.assertAlmostEqual(positive_midpoint[1], (visible[1] + invisible[1]) / 2.0)
        self.assertAlmostEqual(next_midpoint[0], (positive_midpoint[0] + invisible[0]) / 2.0)
        self.assertAlmostEqual(next_midpoint[1], (positive_midpoint[1] + invisible[1]) / 2.0)
        self.assertEqual(result["recovery_measure_count"], 2)

    def test_all_negative_recovery_is_bounded_and_falls_back(self):
        result = run_search(AllNegativeClient())
        self.assertFalse(result["completed"])
        self.assertIn(1, result["unresolved_channels"])
        self.assertEqual(result["fallback_count"], 1)
        self.assertEqual(result["fallback_clear_count"], 304)
        self.assertEqual(result["extra_bearing_count"], 8)

    def test_hidden_truth_is_not_required(self):
        result = run_search(ShadowThenPositiveClient())
        self.assertTrue(result["completed"])

    def test_transport_exception_never_reports_complete(self):
        result = run_search(FailingClient())
        self.assertTrue(result["stopped"])
        self.assertFalse(result["completed"])
        self.assertEqual(result["unresolved_channels"], list(range(1, 21)))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    output = ROOT / "results" / "Q4" / "experiments" / "round2" / "recovery_unit_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"tests_run": result.testsRun, "failures": len(result.failures),
                                  "errors": len(result.errors), "successful": result.wasSuccessful(),
                                  "scope": "offline fake-client unit checks only; no real simulator or full comparison"},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
