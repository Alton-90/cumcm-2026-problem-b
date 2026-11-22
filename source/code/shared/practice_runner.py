"""Shared explicit practice runner for Q3/Q4; never starts a test by default."""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "shared"))
from search_policy import run_search
from simulator_client import SimulatorClient, SimulatorClientError, TransportFailure


def _call_with_pending_retries(client, action, *args, max_pending_retries=2):
    """At most two retries, always through the client's unchanged pending payload."""
    try:
        return getattr(client, action)(*args)
    except TransportFailure:
        for _ in range(max_pending_retries):
            remaining = client.remaining_real_time_s()
            request_budget = getattr(client.transport, 'timeout_s', 0.0)
            if action in ('measure', 'clear') and remaining is not None and remaining <= client.exit_reserve_s + request_budget:
                raise SimulatorClientError('Insufficient budget for pending-action retry; exit required')
            # retry_pending reuses the identical request_id and JSON body.
            try:
                return client.retry_pending()
            except TransportFailure:
                continue
        raise


class RetryingClient:
    """Adapter exposing only policy actions while retaining client state ownership."""

    def __init__(self, client, max_pending_retries=2):
        self._client = client
        self._max_pending_retries = max_pending_retries

    @property
    def virtual_time_s(self):
        return self._client.virtual_time_s

    def _call(self, action, *args):
        return _call_with_pending_retries(self._client, action, *args,
                                          max_pending_retries=self._max_pending_retries)

    def measure(self, x, y, channel):
        return self._call("measure", x, y, channel)

    def clear(self, x, y, channel):
        return self._call("clear", x, y, channel)


def _write_summary(question, payload):
    folder = ROOT / "results" / f"Q{question}" / "practice" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_practice(question, argv=None):
    parser = argparse.ArgumentParser(description=f"Explicit practice-only Q{question} runner; no flags only show help.")
    parser.add_argument("--practice-confirmed", action="store_true")
    parser.add_argument("--robot-id")
    parser.add_argument("--port", type=int, default=2026)
    parser.add_argument("--method", choices=("main", "baseline"), default="main")
    args = parser.parse_args(argv)
    if not args.practice_confirmed or not args.robot_id:
        parser.print_help()
        return 0

    client = SimulatorClient(args.robot_id, port=args.port)
    policy = None
    error = None
    exit_error = None
    entered = False
    started_at = None
    entered_response = None
    exit_response = None
    try:
        started_at = time.monotonic()
        entered_response = _call_with_pending_retries(client, "enter")
        entered = True
        policy = run_search(RetryingClient(client), question=question, method=args.method)
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        if entered and not client.ended:
            try:
                if client.uncertain:
                    exit_response = _call_with_pending_retries(client, "force_exit")
                else:
                    exit_response = _call_with_pending_retries(client, "exit")
            except Exception as caught:
                exit_error = f"{type(caught).__name__}: {caught}"
    elapsed = None if started_at is None else time.monotonic() - started_at
    cleared = None if policy is None else len(policy["cleared_channels"])
    virtual_time = client.virtual_time_s
    completed = bool(policy and policy.get("completed") and error is None and exit_error is None)
    summary = {
        "artifact_type": "practice_run_summary",
        "question": question,
        "method": args.method,
        "official_result": False,
        "status": "completed" if completed else "failed_or_unresolved",
        "local_elapsed_s_from_enter_request": elapsed,
        "simulator_elapsed_s": ((exit_response['real_timestamp_ms'] - entered_response['real_timestamp_ms']) / 1000
                                 if entered_response and exit_response and 'real_timestamp_ms' in entered_response
                                 and 'real_timestamp_ms' in exit_response else None),
        "virtual_time_s": virtual_time,
        "cleared_channel_count": cleared,
        "average_virtual_time_per_cleared_source_s": None if not cleared or virtual_time is None else virtual_time / cleared,
        "true_source_count": None,
        "clear_ratio": None,
        "policy": policy,
        "client_log_path": None if client.log_path is None else str(client.log_path),
        "error": error,
        "exit_error": exit_error,
    }
    summary_path = _write_summary(question, summary)
    print(json.dumps({"status": summary["status"], "cleared": cleared, "virtual_time_s": virtual_time,
                      "summary": str(summary_path), "log": summary["client_log_path"]}, ensure_ascii=False))
    return 0 if completed else 1
