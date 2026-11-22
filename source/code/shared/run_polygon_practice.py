"""Explicit practice-only launcher for the independently evaluated compact policy.

No command starts a session unless all three explicit arguments are supplied.
Before a client is constructed, the frozen synthetic comparison must be
accepted and every recorded input digest must match the current workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "shared"))

from practice_runner import RetryingClient, _call_with_pending_retries
from search_policy_polygon import VERSION, run_search
from simulator_client import SimulatorClient


COMPARISON_PATH = ROOT / "results" / "polygon" / "comparison.json"
INPUTS_PATH = ROOT / "results" / "polygon" / "inputs_before_run.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error_fingerprint(error: Exception) -> str:
    """Return a local-only error identifier; it performs no client or network work."""
    payload = f"{type(error).__name__}:{error}".encode("utf-8", "replace")
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON evidence {path.relative_to(ROOT)}") from error


def _validated_comparison() -> dict[str, Any]:
    """Validate synthetic evidence locally, before any client construction."""
    comparison = _load_json(COMPARISON_PATH)
    if comparison.get("scope") != "SYNTHETIC_ONLY":
        raise ValueError("comparison scope is not SYNTHETIC_ONLY")
    if comparison.get("accepted") is not True or comparison.get("all_checks_pass") is not True:
        raise ValueError("compact comparison is not accepted with all checks passing")
    inputs = comparison.get("inputs")
    frozen_inputs = _load_json(INPUTS_PATH)
    if not isinstance(inputs, list) or not inputs or inputs != frozen_inputs:
        raise ValueError("comparison inputs are absent or differ from inputs_before_run")
    verified: list[dict[str, str]] = []
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("comparison input entry has an invalid schema")
        relative = item["path"]
        expected = item["sha256"]
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("comparison input path or digest is invalid")
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents or not target.is_file():
            raise ValueError(f"comparison input is missing or outside workspace: {relative}")
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(f"comparison input digest mismatch: {relative}")
        verified.append({"path": relative, "sha256": actual})
    return {"comparison_sha256": _sha256(COMPARISON_PATH), "validated_inputs": verified,
            "summaries": comparison.get("summaries")}


def _write_practice_summary(question: int, payload: dict[str, Any]) -> Path:
    folder = (ROOT / "results" / f"Q{question}" / "practice" / "polygon" /
              (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None, *, client_factory=SimulatorClient, compact_policy=run_search,
         comparison_validator=_validated_comparison, summary_writer=_write_practice_summary) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice-confirmed", action="store_true")
    parser.add_argument("--question", type=int, choices=(4,))
    parser.add_argument("--robot-id")
    parser.add_argument("--port", type=int, default=2026)
    args = parser.parse_args(argv)
    if not args.practice_confirmed or args.question is None or not args.robot_id:
        parser.print_help()
        return 0

    try:
        evidence = comparison_validator()
    except Exception as error:
        print(json.dumps({"status": "refused_pre_client", "practice_only": True,
                          "error_fingerprint": _error_fingerprint(error)}, ensure_ascii=False), file=sys.stderr)
        return 2

    client = client_factory(args.robot_id, port=args.port)
    policy_state = None
    error_text = exit_error = None
    entered = False
    enter_response = exit_response = None
    started_at = time.monotonic()
    try:
        enter_response = _call_with_pending_retries(client, "enter")
        entered = True
        policy_state = compact_policy(RetryingClient(client), question=args.question, method="main")
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
    finally:
        if entered and not client.ended:
            try:
                exit_response = _call_with_pending_retries(client, "force_exit" if client.uncertain else "exit")
            except Exception as error:
                exit_error = f"{type(error).__name__}: {error}"

    local_elapsed = time.monotonic() - started_at
    interface_elapsed = None
    if enter_response and exit_response and all("real_timestamp_ms" in item for item in (enter_response, exit_response)):
        interface_elapsed = (exit_response["real_timestamp_ms"] - enter_response["real_timestamp_ms"]) / 1000.0
    cleared = None if policy_state is None else len(policy_state.get("cleared_channels", []))
    completed = bool(policy_state and policy_state.get("completed") and not error_text and not exit_error)
    policy_path = ROOT / "code" / "shared" / "search_policy_polygon.py"
    summary = {
        "artifact_type": "compact_practice_run_summary",
        "practice_confirmed": True,
        "practice_only": True,
        "official_result": False,
        "question": args.question,
        "method": "polygon_main",
        "policy_version": VERSION,
        "policy_sha256": _sha256(policy_path),
        "status": "completed" if completed else "failed_or_unresolved",
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
        "verified_compact_comparison": evidence,
        "error": error_text,
        "exit_error": exit_error,
    }
    summary_path = summary_writer(args.question, summary)
    print(json.dumps({"status": summary["status"], "practice_only": True, "summary": str(summary_path),
                      "log": summary["client_log_path"]}, ensure_ascii=False))
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
