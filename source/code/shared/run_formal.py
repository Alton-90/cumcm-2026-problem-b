"""Explicit formal-session launcher. It records a session, never an official score."""

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "Q4" / "python"))
from practice_runner import RetryingClient, _call_with_pending_retries
from search_policy import run_search as run_q3_search
from search_policy_recovery import run_search as run_q4_recovery
from simulator_client import SimulatorClient
from run_q4_recovery import _validate_comparison


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprints(question):
    files = {"formal_entrypoint": Path(__file__).resolve(),
             "practice_runner": ROOT / "code" / "shared" / "practice_runner.py",
             "search_policy": ROOT / "code" / "shared" / "search_policy.py",
             "recovery_policy": ROOT / "code" / "shared" / "search_policy_recovery.py",
             "q4_recovery_validator_entry": ROOT / "code" / "Q4" / "python" / "run_q4_recovery.py",
             "simulator_client": ROOT / "code" / "shared" / "simulator_client.py"}
    return {name: _sha256(path) for name, path in files.items()}


def _write_formal_summary(question, payload):
    folder = ROOT / "results" / f"Q{question}" / "formal" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None, *, client_factory=SimulatorClient, q3_policy=run_q3_search,
         q4_policy=run_q4_recovery, summary_writer=_write_formal_summary,
         q4_validator=_validate_comparison):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-confirmed", action="store_true")
    parser.add_argument("--question", type=int, choices=(3, 4))
    parser.add_argument("--robot-id")
    parser.add_argument("--port", type=int, default=2026)
    args = parser.parse_args(argv)
    if not args.formal_confirmed or args.question is None or not args.robot_id:
        parser.print_help()
        return 0
    comparison_evidence = comparison_inputs = None
    if args.question == 4:
        try:
            comparison_evidence, comparison_inputs = q4_validator()
        except Exception as error:
            print(f"Q4 formal launcher refused before client setup: {error}", file=sys.stderr)
            return 2

    client = client_factory(args.robot_id, port=args.port)
    policy_state = None
    error_text = exit_error = None
    entered = False
    enter_response = exit_response = None
    local_started = time.monotonic()
    try:
        enter_response = _call_with_pending_retries(client, "enter")
        entered = True
        adapter = RetryingClient(client)
        if args.question == 3:
            policy_state = q3_policy(adapter, question=3, method="main")
            method = "main"
        else:
            policy_state = q4_policy(adapter, question=4, method="main")
            method = "recovery_v2"
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        method = "main" if args.question == 3 else "recovery_v2"
    finally:
        if entered and not client.ended:
            try:
                exit_response = _call_with_pending_retries(client, "force_exit" if client.uncertain else "exit")
            except Exception as error:
                exit_error = f"{type(error).__name__}: {error}"
    local_elapsed = time.monotonic() - local_started
    interface_elapsed = None
    if enter_response and exit_response and all("real_timestamp_ms" in item for item in (enter_response, exit_response)):
        interface_elapsed = (exit_response["real_timestamp_ms"] - enter_response["real_timestamp_ms"]) / 1000.0
    cleared = None if policy_state is None else len(policy_state.get("cleared_channels", []))
    completed = bool(policy_state and policy_state.get("completed") and not error_text and not exit_error)
    summary = {
        "artifact_type": "formal_session_summary",
        "formal_session_recorded": entered,
        "official_result": False,
        "upload_status": "not_verified",
        "question": args.question,
        "method": method,
        "status": "completed_strategy_not_officially_verified" if completed else "failed_or_unresolved",
        "local_elapsed_s_from_enter_request": local_elapsed,
        "simulator_elapsed_s_from_interface_timestamps": interface_elapsed,
        "virtual_time_s": client.virtual_time_s,
        "cleared_channel_count": cleared,
        "average_virtual_time_per_cleared_source_s": (
            None if not cleared or client.virtual_time_s is None else client.virtual_time_s / cleared),
        "true_source_count": None,
        "clear_ratio": None,
        "policy": policy_state,
        "client_log_path": None if client.log_path is None else str(client.log_path),
        "code_hashes": _fingerprints(args.question),
        "q4_comparison_scope": None if comparison_evidence is None else comparison_evidence.get("scope"),
        "q4_verified_comparison_inputs": comparison_inputs,
        "error": error_text,
        "exit_error": exit_error,
    }
    summary_path = summary_writer(args.question, summary)
    print(json.dumps({"status": summary["status"], "summary": str(summary_path),
                      "log": summary["client_log_path"], "upload_status": "not_verified"}, ensure_ascii=False))
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
