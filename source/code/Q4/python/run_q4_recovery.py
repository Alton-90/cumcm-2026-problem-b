"""Independent practice-only entrypoint for the evaluated Q4 recovery candidate."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "shared"))
from practice_runner import RetryingClient, _call_with_pending_retries, _write_summary
from search_policy_recovery import run_search
from simulator_client import SimulatorClient

COMPARISON_PATH = ROOT / "results" / "Q4" / "experiments" / "round2" / "comparison.json"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_comparison(path=COMPARISON_PATH):
    """Return verified comparison evidence, or fail before any client is constructed."""
    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    if evidence.get("accepted_by_predeclared_numeric_criteria") is not True:
        raise RuntimeError("recovery comparison is not accepted by its predeclared numeric criteria")
    verified = []
    for item in evidence.get("inputs", []):
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("comparison input record is malformed")
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents or not target.is_file():
            raise RuntimeError(f"comparison input is unavailable: {relative}")
        actual = _sha256(target)
        if actual != expected:
            raise RuntimeError(f"comparison input hash mismatch: {relative}")
        verified.append({"path": relative, "sha256": actual})
    if not verified:
        raise RuntimeError("comparison has no verifiable input records")
    return evidence, verified


def _fingerprints():
    files = {
        "new_entrypoint": Path(__file__).resolve(),
        "recovery_policy_helper": ROOT / "code" / "shared" / "search_policy_recovery.py",
        "v1_policy_helper": ROOT / "code" / "shared" / "search_policy.py",
        "simulator_client": ROOT / "code" / "shared" / "simulator_client.py",
    }
    return {name: _sha256(path) for name, path in files.items()}


def main(argv=None, *, client_factory=SimulatorClient, policy=run_search,
         comparison_path=COMPARISON_PATH, summary_writer=_write_summary):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice-confirmed", action="store_true")
    parser.add_argument("--robot-id")
    parser.add_argument("--port", type=int, default=2026)
    args = parser.parse_args(argv)
    if not args.practice_confirmed or not args.robot_id:
        parser.print_help()
        return 0
    try:
        evidence, verified_inputs = _validate_comparison(comparison_path)
    except Exception as error:
        print(f"Recovery candidate refused before network/client setup: {error}", file=sys.stderr)
        return 2

    client = client_factory(args.robot_id, port=args.port)
    policy_state = None
    error_text = None
    exit_error = None
    entered = False
    enter_response = None
    exit_response = None
    local_started = time.monotonic()
    try:
        enter_response = _call_with_pending_retries(client, "enter")
        entered = True
        policy_state = policy(RetryingClient(client), question=4, method="main")
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
    finally:
        if entered and not client.ended:
            try:
                action = "force_exit" if client.uncertain else "exit"
                exit_response = _call_with_pending_retries(client, action)
            except Exception as error:
                exit_error = f"{type(error).__name__}: {error}"
    local_elapsed = time.monotonic() - local_started
    cleared = None if policy_state is None else len(policy_state.get("cleared_channels", []))
    virtual_time = client.virtual_time_s
    completed = bool(policy_state and policy_state.get("completed") and not error_text and not exit_error)
    interface_elapsed = None
    if enter_response and exit_response and all("real_timestamp_ms" in r for r in (enter_response, exit_response)):
        interface_elapsed = (exit_response["real_timestamp_ms"] - enter_response["real_timestamp_ms"]) / 1000.0
    summary = {
        "artifact_type": "practice_run_summary",
        "question": 4,
        "method": "recovery_v2",
        "official_result": False,
        "candidate_only": True,
        "status": "completed" if completed else "failed_or_unresolved",
        "local_elapsed_s_from_enter_request": local_elapsed,
        "simulator_elapsed_s_from_interface_timestamps": interface_elapsed,
        "virtual_time_s": virtual_time,
        "cleared_channel_count": cleared,
        "average_virtual_time_per_cleared_source_s": None if not cleared or virtual_time is None else virtual_time / cleared,
        "true_source_count": None,
        "clear_ratio": None,
        "policy": policy_state,
        "client_log_path": None if client.log_path is None else str(client.log_path),
        "comparison_path": str(comparison_path),
        "comparison_scope": evidence.get("scope"),
        "verified_comparison_inputs": verified_inputs,
        "strategy_fingerprints": _fingerprints(),
        "error": error_text,
        "exit_error": exit_error,
    }
    summary_path = summary_writer(4, summary)
    print(json.dumps({"status": summary["status"], "cleared": cleared, "virtual_time_s": virtual_time,
                      "summary": str(summary_path), "log": summary["client_log_path"]}, ensure_ascii=False))
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
