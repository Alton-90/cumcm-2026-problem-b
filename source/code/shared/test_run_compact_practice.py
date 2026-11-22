"""Offline mocks for the compact practice entrypoint; no simulator traffic."""

import unittest

import run_compact_practice as runner


class _NoClient:
    def __init__(self, *args, **kwargs):
        raise AssertionError("client must not be constructed")


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.ended = False
        self.uncertain = False
        self.virtual_time_s = 20.0
        self.log_path = None

    def enter(self):
        return {"real_timestamp_ms": 1000}

    def exit(self):
        self.ended = True
        return {"real_timestamp_ms": 2000}


def _evidence():
    return {"comparison_sha256": "0" * 64, "validated_inputs": [], "summaries": {}}


def _completed_policy(calls):
    def policy(client, *, question, method):
        calls.append((question, method, type(client).__name__))
        return {"completed": True, "cleared_channels": [1, 2]}
    return policy


class CompactPracticeEntrypointTests(unittest.TestCase):
    def test_default_is_help_only_and_constructs_no_client(self):
        self.assertEqual(runner.main([], client_factory=_NoClient), 0)

    def test_pre_client_rejection_uses_no_client_and_local_fingerprint(self):
        def reject():
            raise ValueError("tampered")
        self.assertEqual(runner.main(["--practice-confirmed", "--question", "3", "--robot-id", "r"],
                                     client_factory=_NoClient, comparison_validator=reject), 2)

    def test_q3_and_q4_route_to_compact_policy_with_main_method(self):
        for question in (3, 4):
            calls = []
            self.assertEqual(runner.main(["--practice-confirmed", "--question", str(question), "--robot-id", "r"],
                                         client_factory=_FakeClient, compact_policy=_completed_policy(calls),
                                         comparison_validator=_evidence,
                                         summary_writer=lambda q, payload: runner.ROOT / "offline-summary.json"), 0)
            self.assertEqual(calls, [(question, "main", "RetryingClient")])


if __name__ == "__main__":
    unittest.main()
