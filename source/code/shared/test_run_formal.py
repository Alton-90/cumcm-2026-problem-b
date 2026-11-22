"""Offline mock checks for formal launcher authorization and routing."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "shared"))
from run_formal import main


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.ended = self.uncertain = False
        self.virtual_time_s = 0.0
        self.log_path = None
        self.entered = self.exited = False
    def enter(self): self.entered = True; return {"real_timestamp_ms": 1000, "virtual_time_s": 0}
    def exit(self): self.ended = self.exited = True; return {"real_timestamp_ms": 2000, "virtual_time_s": 0}
    def force_exit(self): return self.exit()
    def retry_pending(self): raise AssertionError("no retry expected")
    def remaining_real_time_s(self): return 1200.0


class FormalRunnerTests(unittest.TestCase):
    def test_default_and_unconfirmed_never_construct_client(self):
        constructed = []
        factory = lambda *a, **k: constructed.append(True)
        self.assertEqual(main([], client_factory=factory), 0)
        self.assertEqual(main(["--question", "3", "--robot-id", "team"], client_factory=factory), 0)
        self.assertEqual(constructed, [])

    def test_q3_and_q4_route_to_fixed_policies(self):
        calls, summaries = [], []
        def policy(adapter, *, question, method):
            calls.append((question, method)); return {"completed": True, "cleared_channels": [1, 2]}
        def writer(question, summary): summaries.append((question, summary)); return Path("mock-summary.json")
        validator = lambda: ({"scope": "SYNTHETIC_NOT_OFFICIAL"}, [{"path": "mock", "sha256": "x"}])
        for question in (3, 4):
            code = main(["--formal-confirmed", "--question", str(question), "--robot-id", "team"],
                        client_factory=lambda *a, **k: FakeClient(), q3_policy=policy, q4_policy=policy,
                        summary_writer=writer, q4_validator=validator)
            self.assertEqual(code, 0)
        self.assertEqual(calls, [(3, "main"), (4, "main")])
        self.assertEqual(summaries[0][1]["method"], "main")
        self.assertEqual(summaries[1][1]["method"], "recovery_v2")
        self.assertEqual(summaries[0][1]["average_virtual_time_per_cleared_source_s"], 0.0)
        self.assertIn("practice_runner", summaries[1][1]["code_hashes"])
        self.assertIn("search_policy", summaries[1][1]["code_hashes"])
        self.assertIn("q4_recovery_validator_entry", summaries[1][1]["code_hashes"])

    def test_rejected_q4_fingerprint_never_constructs_client(self):
        constructed = []
        def rejected(): raise RuntimeError("changed fingerprint")
        result = main(["--formal-confirmed", "--question", "4", "--robot-id", "team"],
                      client_factory=lambda *a, **k: constructed.append(True),
                      q4_validator=rejected)
        self.assertEqual(result, 2)
        self.assertEqual(constructed, [])

    def test_exit_failure_does_not_report_completed_strategy(self):
        captured = []
        class ExitFailure(FakeClient):
            def exit(self): raise RuntimeError("exit rejected")
        result = main(["--formal-confirmed", "--question", "3", "--robot-id", "team"],
                      client_factory=lambda *a, **k: ExitFailure(),
                      q3_policy=lambda *a, **k: {"completed": True, "cleared_channels": [1]},
                      summary_writer=lambda q, s: captured.append(s) or Path("mock-summary.json"))
        self.assertEqual(result, 1)
        self.assertEqual(captured[0]["status"], "failed_or_unresolved")
        self.assertIn("exit rejected", captured[0]["exit_error"])

    def test_policy_exception_is_nonzero_and_never_official_success(self):
        captured = []
        def failing(*args, **kwargs): raise RuntimeError("synthetic failure")
        code = main(["--formal-confirmed", "--question", "3", "--robot-id", "team"],
                    client_factory=lambda *a, **k: FakeClient(), q3_policy=failing,
                    summary_writer=lambda q, s: captured.append(s) or Path("mock-summary.json"))
        self.assertEqual(code, 1)
        self.assertFalse(captured[0]["official_result"])
        self.assertEqual(captured[0]["upload_status"], "not_verified")
        self.assertEqual(captured[0]["status"], "failed_or_unresolved")
        self.assertIsNone(captured[0]["average_virtual_time_per_cleared_source_s"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FormalRunnerTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
