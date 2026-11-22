"""Offline protocol tests for simulator_client; no socket or simulator access."""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "shared"))
from simulator_client import (ActionRejected, HttpResult, HttpStatusFailure, LocalHttpTransport,
                              SimulatorClient, SimulatorClientError, StateUncertain, TransportFailure)
from practice_runner import RetryingClient


def accepted(virtual_time=0, **more):
    return HttpResult(200, {"accepted": True, "real_timestamp_ms": 1,
                            "virtual_time_s": virtual_time, **more})


class FakeTransport:
    def __init__(self, events):
        self.events = list(events)
        self.calls = []

    def post(self, path, payload):
        self.calls.append((path, dict(payload)))
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


class ClientTests(unittest.TestCase):
    def test_runner_retry_adapter_keeps_request_identity(self):
        t = FakeTransport([accepted(0, remaining_real_duration_s=120),
                           TransportFailure('timeout'), accepted(5)])
        c = SimulatorClient('team', transport=t, log_path=False)
        c.enter()
        RetryingClient(c).measure(1, 2, 3)
        self.assertEqual(t.calls[1], t.calls[2])

    def test_retry_adapter_preserves_exit_budget(self):
        t = FakeTransport([accepted(0, remaining_real_duration_s=120), TransportFailure('timeout')])
        c = SimulatorClient('team', transport=t, log_path=False)
        c.enter()
        with patch.object(c, 'remaining_real_time_s', side_effect=[100.0, 0.0]):
            with self.assertRaises(SimulatorClientError):
                RetryingClient(c).measure(1, 2, 3)
        self.assertEqual(len(t.calls), 2)
        self.assertTrue(c.uncertain)

    def test_rejected_action_does_not_change_state(self):
        t = FakeTransport([accepted(0, remaining_real_duration_s=120), accepted(7),
                           HttpResult(200, {"accepted": False, "virtual_time_s": 0})])
        c = SimulatorClient("team", transport=t, log_path=False)
        c.enter()
        c.measure(0, 0, 1)
        with self.assertRaises(ActionRejected):
            c.measure(3, 4, 2)
        self.assertEqual(c.position, (0.0, 0.0))
        self.assertEqual(c.channel, 1)
        self.assertEqual(c.virtual_time_s, 7)

    def test_delayed_enter_retry_does_not_restart_clock(self):
        t = FakeTransport([TransportFailure('lost enter response'),
                           accepted(0, remaining_real_duration_s=1200)])
        c = SimulatorClient('team', transport=t, log_path=False)
        with patch('simulator_client.time.monotonic', side_effect=[100.0, 501.0]):
            with self.assertRaises(TransportFailure):
                c.enter()
            c.retry_pending()
            self.assertEqual(c.remaining_real_time_s(), 799.0)

    def test_invalid_response_blocks_new_actions(self):
        transport = LocalHttpTransport()
        transport._opener = MagicMock()
        response = transport._opener.open.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = b'[]'
        c = SimulatorClient('team', transport=transport, log_path=False)
        with self.assertRaises(TransportFailure):
            c.enter()
        self.assertTrue(c.uncertain)
        with self.assertRaises(StateUncertain):
            c.measure(0, 0, 1)

    def test_clear_keeps_measurement_channel(self):
        t = FakeTransport([accepted(0, remaining_real_duration_s=120), accepted(5), accepted(9)])
        c = SimulatorClient("team", transport=t, log_path=False)
        c.enter(); c.measure(10, 0, 3); c.clear(20, 0, 9)
        self.assertEqual(c.position, (20, 0))
        self.assertEqual(c.channel, 3)
        self.assertEqual(c.virtual_time_s, 9)

    def test_transport_retry_reuses_identical_id_and_payload(self):
        t = FakeTransport([accepted(0, remaining_real_duration_s=120), TransportFailure("timeout"), accepted(5)])
        c = SimulatorClient("team", transport=t, log_path=False)
        c.enter()
        with self.assertRaises(TransportFailure):
            c.measure(10, 20, 2)
        first = t.calls[-1][1]
        c.retry_pending()
        second = t.calls[-1][1]
        self.assertEqual(first, second)
        self.assertEqual(c.channel, 2)
        self.assertFalse(c.uncertain)

    def test_uncertain_failure_blocks_new_action_but_force_exit_exists(self):
        t = FakeTransport([accepted(0, remaining_real_duration_s=120), TransportFailure("disconnect"), accepted(0, exit_reason="user_exit")])
        c = SimulatorClient("team", transport=t, log_path=False)
        c.enter()
        with self.assertRaises(TransportFailure):
            c.measure(1, 2, 1)
        with self.assertRaises(StateUncertain):
            c.clear(1, 2, 1)
        c.force_exit()
        self.assertTrue(c.ended)

    def test_http_error_is_distinct_and_does_not_change_state(self):
        t = FakeTransport([accepted(0, remaining_real_duration_s=120), HttpStatusFailure(400, {"accepted": False})])
        c = SimulatorClient("team", transport=t, log_path=False)
        c.enter()
        with self.assertRaises(HttpStatusFailure):
            c.measure(1, 2, 1)
        self.assertFalse(c.uncertain)
        self.assertEqual(c.position, (0.0, 0.0))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClientTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    output = ROOT / "results" / "diagnostics" / "client_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"tests_run": result.testsRun, "failures": len(result.failures),
                                  "errors": len(result.errors), "successful": result.wasSuccessful(),
                                  "scope": "offline fake transport only; no real simulator request"}, indent=2) + "\n",
                      encoding="utf-8")
    raise SystemExit(not result.wasSuccessful())
