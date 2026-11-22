"""Explicit practice-only launcher for frozen Q3 joint_route evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "shared"))

from practice_runner import RetryingClient, _call_with_pending_retries
from search_policy_joint_route_q3 import VERSION, run_search
from simulator_client import SimulatorClient


FINAL = ROOT / "results" / "joint_route_q3" / "final_holdout"
COMPARISON = FINAL / "comparison.json"
INPUTS = FINAL / "inputs_before_run.json"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable evidence: {path.relative_to(ROOT)}") from error


def _validate_final_evidence():
    comparison = _load(COMPARISON)
    if comparison.get("scope") != "SYNTHETIC_FINAL_INDEPENDENT_HOLDOUT":
        raise ValueError("joint-route evidence has the wrong scope")
    if comparison.get("accepted") is not True or comparison.get("all_checks_pass") is not True or comparison.get("case_count") != 60:
        raise ValueError("joint-route evidence is not an accepted 60-case result")
    inputs = comparison.get("inputs")
    if not isinstance(inputs, list) or not inputs or inputs != _load(INPUTS):
        raise ValueError("joint-route frozen input manifest mismatch")
    verified = []
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("invalid frozen input entry")
        relative, expected = item["path"], item["sha256"]
        target = (ROOT / relative).resolve()
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("invalid frozen input path or hash")
        if ROOT not in target.parents or not target.is_file() or _sha256(target) != expected:
            raise ValueError(f"frozen input hash mismatch: {relative}")
        verified.append({"path": relative, "sha256": expected})
    return {"comparison_sha256": _sha256(COMPARISON), "validated_inputs": verified,
            "summaries": comparison.get("summaries")}


def _write_summary(payload):
    folder = ROOT / "results" / "Q3" / "practice" / "joint_route_q3" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None, *, client_factory=SimulatorClient, policy=run_search, validator=_validate_final_evidence, summary_writer=_write_summary):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice-confirmed", action="store_true")
    parser.add_argument("--question", type=int, choices=(3,))
    parser.add_argument("--robot-id")
    parser.add_argument("--port", type=int, default=2026)
    args = parser.parse_args(argv)
    if not args.practice_confirmed or args.question is None or not args.robot_id:
        parser.print_help()
        return 0
    try:
        evidence = validator()
    except Exception as error:
        fingerprint = hashlib.sha256(f"{type(error).__name__}:{error}".encode("utf-8", "replace")).hexdigest()[:16]
        print(json.dumps({"status": "refused_pre_client", "practice_only": True, "error_fingerprint": fingerprint}), file=sys.stderr)
        return 2
    client = client_factory(args.robot_id, port=args.port)
    state = None
    error_text = exit_error = None
    entered = False
    enter_response = exit_response = None
    started = time.monotonic()
    try:
        enter_response = _call_with_pending_retries(client, "enter")
        entered = True
        state = policy(RetryingClient(client), question=3, method="main")
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
    finally:
        if entered and not client.ended:
            try:
                exit_response = _call_with_pending_retries(client, "force_exit" if client.uncertain else "exit")
            except Exception as error:
                exit_error = f"{type(error).__name__}: {error}"
    cleared = None if state is None else len(state.get("cleared_channels", []))
    completed = bool(state and state.get("completed") and not error_text and not exit_error)
    interface_elapsed = None
    if enter_response and exit_response and all("real_timestamp_ms" in response for response in (enter_response, exit_response)):
        interface_elapsed = (exit_response["real_timestamp_ms"] - enter_response["real_timestamp_ms"]) / 1000.0
    summary = {"artifact_type": "joint_route_q3_practice_summary", "practice_confirmed": True, "practice_only": True,
               "official_result": False, "question": 3, "method": "joint_route_q3_main", "policy_version": VERSION,
               "policy_sha256": _sha256(ROOT / "code" / "shared" / "search_policy_joint_route_q3.py"),
               "status": "completed" if completed else "failed_or_unresolved", "local_elapsed_s_from_enter_request": time.monotonic() - started,
               "simulator_elapsed_s_from_interface_timestamps": interface_elapsed, "virtual_time_s": client.virtual_time_s,
               "cleared_channel_count": cleared,
               "average_virtual_time_per_cleared_source_s": None if not cleared or client.virtual_time_s is None else client.virtual_time_s / cleared,
               "true_source_count": None, "clear_ratio": None, "policy": state,
               "client_log_path": None if client.log_path is None else str(client.log_path),
               "validated_final_holdout": evidence, "error": error_text, "exit_error": exit_error}
    summary_path = summary_writer(summary)
    print(json.dumps({"status": summary["status"], "practice_only": True, "summary": str(summary_path), "log": summary["client_log_path"]}, ensure_ascii=False))
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
