"""Post-hoc accounting for the frozen offline/practice simulator transcripts.

This script never calls the simulator.  It reads accepted log entries only and
decomposes their virtual-time increments using the simulator's published cost
rules: movement at 5 m/s, measurement 5 s, channel switch 1 s, and clear
actions of 5 s (success) or 3 s (no target).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


LOGS = {
    "Q3_verified_practice": ["20260910-205427-2ef140f2.jsonl"],
    "Q4_recovery_v2_practice": [
        "20260910-212655-cc8703c7.jsonl",
        "20260910-213433-9a8238ab.jsonl",
        "20260910-214802-f6fa4cee.jsonl",
    ],
}


def _accepted_actions(path: Path) -> list[dict[str, Any]]:
    seen: set[str] = set()
    actions: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        response = item.get("response", {})
        request_id = item.get("request", {}).get("request_id")
        if not response.get("accepted") or not request_id or request_id in seen:
            continue
        seen.add(request_id)
        actions.append(item)
    return actions


def _diagnose_log(path: Path) -> dict[str, Any]:
    actions = _accepted_actions(path)
    if not actions or actions[0]["path"] != "/enter" or actions[-1]["path"] != "/exit":
        raise ValueError(f"Expected one accepted enter/exit transcript: {path.name}")

    current_position = (0.0, 0.0)
    current_channel = 1
    buckets = Counter()
    result_counts = Counter()
    success_indices: list[int] = []

    for index, action in enumerate(actions[1:], start=1):
        route = action["path"]
        request = action["request"]
        response = action["response"]
        if route not in {"/measure", "/clear"}:
            continue

        position = request["position"]
        target = (float(position["x"]), float(position["y"]))
        buckets["movement_s"] += math.dist(current_position, target) / 5.0
        current_position = target
        if route == "/measure":
            buckets["measurement_s"] += 5.0
            channel = int(request["channel"])
            if channel != current_channel:
                buckets["channel_switch_s"] += 1.0
            current_channel = channel
            result_counts[f"measure:{response.get('measure_result', 'unknown')}"] += 1
        else:
            clear_result = response.get("clear_result", "unknown")
            buckets["clear_s"] += 5.0 if clear_result == "success" else 3.0
            result_counts[f"clear:{clear_result}"] += 1
            if clear_result == "success":
                success_indices.append(index)

    exit_time = float(actions[-1]["response"]["virtual_time_s"])
    accounted_time = sum(buckets.values())
    last_success_index = success_indices[-1] if success_indices else None
    if last_success_index is None:
        tail_time = exit_time
        tail_actions = actions[1:-1]
    else:
        last_success_time = float(actions[last_success_index]["response"]["virtual_time_s"])
        tail_time = exit_time - last_success_time
        tail_actions = actions[last_success_index + 1 : -1]

    return {
        "log_file": path.name,
        "accepted_action_count_including_enter_exit": len(actions),
        "virtual_time_s": exit_time,
        "time_components_s": {key: round(value, 6) for key, value in buckets.items()},
        "accounted_virtual_time_s": round(accounted_time, 6),
        "accounting_residual_s": round(exit_time - accounted_time, 9),
        "event_counts": dict(sorted(result_counts.items())),
        "successful_clear_count": len(success_indices),
        "tail_after_last_successful_clear": {
            "virtual_time_s": round(tail_time, 6),
            "measure_count": sum(a["path"] == "/measure" for a in tail_actions),
            "clear_count": sum(a["path"] == "/clear" for a in tail_actions),
            "action_count_excluding_exit": len(tail_actions),
        },
    }


def _aggregate(logs: list[dict[str, Any]]) -> dict[str, Any]:
    total_components = Counter()
    total_events = Counter()
    tail = Counter()
    for log in logs:
        total_components.update(log["time_components_s"])
        total_events.update(log["event_counts"])
        tail.update(log["tail_after_last_successful_clear"])
    return {
        "run_count": len(logs),
        "mean_virtual_time_s": round(sum(x["virtual_time_s"] for x in logs) / len(logs), 6),
        "mean_time_components_s": {
            key: round(value / len(logs), 6) for key, value in sorted(total_components.items())
        },
        "total_event_counts": dict(sorted(total_events.items())),
        "mean_tail_after_last_successful_clear": {
            key: round(value / len(logs), 6) for key, value in sorted(tail.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--brief-out", type=Path, required=True)
    args = parser.parse_args()

    groups = {
        group: [_diagnose_log(args.log_dir / filename) for filename in filenames]
        for group, filenames in LOGS.items()
    }
    report = {
        "schema_version": 1,
        "artifact_type": "post_hoc_practice_log_diagnosis",
        "scope": "Offline/practice transcripts only; no simulator calls were made.",
        "interpretation_guardrail": (
            "All counts and time splits are retrospective transcript accounting. "
            "They are not online inputs, do not use N or hidden source truths, and do not alter a policy."
        ),
        "selection_note": (
            "The three Q4 logs were selected because their success counts and total virtual times "
            "match the recorded recovery_v2 three-run practice summary."
        ),
        "groups": {group: {"logs": logs, "aggregate": _aggregate(logs)} for group, logs in groups.items()},
        "recommendations": [
            "Prioritize shrinking redundant coverage and recovery travel: movement is the dominant measured component.",
            "Use a bounded terminal-stop or unresolved-region rule only if it preserves the existing status semantics; the measured tail quantifies the potential target.",
            "Do not infer source counts or locations from these post-hoc summaries; any replacement policy must use only response-local observations.",
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Offline/practice algorithm diagnosis",
        "",
        "This is post-hoc accounting of already saved practice logs. It did not call the simulator, is not an online policy input, and contains no inferred N or source truth.",
        "",
    ]
    for group, payload in report["groups"].items():
        aggregate = payload["aggregate"]
        components = aggregate["mean_time_components_s"]
        tail = aggregate["mean_tail_after_last_successful_clear"]
        lines.extend([
            f"## {group}",
            "",
            f"Runs: {aggregate['run_count']}; mean virtual time: {aggregate['mean_virtual_time_s']:.3f} s.",
            "",
            f"Mean decomposition — movement {components.get('movement_s', 0):.3f} s; measurement {components.get('measurement_s', 0):.3f} s; channel switching {components.get('channel_switch_s', 0):.3f} s; clear actions {components.get('clear_s', 0):.3f} s.",
            "",
            f"Mean tail after the final successful clear — {tail.get('virtual_time_s', 0):.3f} s across {tail.get('measure_count', 0):.1f} measures and {tail.get('clear_count', 0):.1f} clear attempts.",
            "",
        ])
    lines.extend([
        "## Bounded optimization implication",
        "",
        "The directly observed bottleneck is travel, followed by repeated measurement/clear activity in the final tail. A safe next comparison is therefore a smaller, geometry-bounded coverage grid with the current localize/clear semantics preserved; terminal-stop changes need a separate semantic check.",
        "",
    ])
    args.brief_out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
