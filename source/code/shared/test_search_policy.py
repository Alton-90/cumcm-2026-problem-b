"""Offline edge-path tests for approved search policy; no simulator access."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "shared"))
from search_policy import run_search


class HalfScanFailureClient:
    virtual_time_s = 0
    def __init__(self): self.calls = 0
    def measure(self, x, y, channel):
        self.calls += 1
        if self.calls == 3: raise ConnectionError("synthetic half-scan failure")
        return {"measure_result": "no_signal"}
    def clear(self, x, y, channel): raise AssertionError("clear not expected")


class DirectionClearFailureClient:
    virtual_time_s = 0
    def measure(self, x, y, channel):
        return {"measure_result": "direction", "svd_deg": 0.0} if channel == 1 else {"measure_result": "no_signal"}
    def clear(self, x, y, channel): return {"clear_result": "no_target_in_range"}


class NearSuccessClient:
    virtual_time_s = 0
    def __init__(self): self.first = True
    def measure(self, x, y, channel):
        if self.first and channel == 1:
            self.first = False
            return {"measure_result": "near"}
        return {"measure_result": "no_signal"}
    def clear(self, x, y, channel): return {"clear_result": "success"}


class SearchPolicyTests(unittest.TestCase):
    def test_half_scan_failure_cannot_exclude_unscanned_channels(self):
        result = run_search(HalfScanFailureClient(), question=3)
        self.assertTrue(result["stopped"])
        self.assertFalse(result["completed"])
        self.assertEqual(result["excluded_channels"], [])
        self.assertEqual(result["unresolved_channels"], list(range(1, 21)))

    def test_direction_and_all_clear_misses_never_complete(self):
        result = run_search(DirectionClearFailureClient(), question=3, method="baseline")
        self.assertFalse(result["completed"])
        self.assertIn(1, result["unresolved_channels"])
        self.assertEqual(result["cleared_channels"], [])

    def test_near_success_still_scans_other_channels_to_exclusion(self):
        result = run_search(NearSuccessClient(), question=3)
        self.assertTrue(result["completed"])
        self.assertEqual(result["cleared_channels"], [1])
        self.assertEqual(result["excluded_channels"], list(range(2, 21)))
        self.assertEqual(result["coverage_measurements_by_channel"][2], 25)


if __name__ == "__main__":
    unittest.main(verbosity=1)
