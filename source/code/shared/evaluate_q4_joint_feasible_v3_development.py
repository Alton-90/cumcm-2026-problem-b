"""Offline-only development comparison for Q4 joint-feasible v3."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from offline_arena import OfflineArena, Source
from simulator_client import SimulatorClient
import search_policy_q4_ring22_joint_route as base
import search_policy_q4_joint_feasible_v2 as model
import search_policy_q4_joint_feasible_v3 as cand

R = Path(__file__).resolve().parents[2]
O = R / "results" / "q4_joint_feasible" / "development_v3"
SRC = R / "results" / "q4_reuse" / "development" / "cases.json"


class TrackedArena(OfflineArena):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_points = []

    def post(self, path, payload):
        if path == "/clear":
            position = payload["position"]
            self.clear_points.append(
                (payload["channel"], round(float(position["x"]), 6), round(float(position["y"]), 6))
            )
        return super().post(path, payload)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(policy, case, diagnose=False):
    sources = [Source(**item) for item in case["sources"]]
    arena = TrackedArena(sources, case["seed"], case["error_mode"])
    client = SimulatorClient("offline", transport=arena, log_path=False)
    client.enter()
    state = policy(client, question=4, method="main")
    client.exit()
    cleared_count = len(arena.cleared)
    movement_s = arena.distance_m / 5
    measure_s = 5 * arena.measure_count
    switch_s = arena.switch_count
    clears_s = 3 * arena.clear_count + 2 * cleared_count
    groups = [set(state[name]) for name in ("cleared_channels", "excluded_channels", "unresolved_channels")]
    union = set().union(*groups)
    accounted = union == set(range(1, 21)) and sum(map(len, groups)) == len(union)
    false_excluded = bool(set(state["excluded_channels"]) & set(arena.sources))
    full_clear = (
        cleared_count == len(sources)
        and set(state["cleared_channels"]) == arena.cleared
        and not false_excluded
    )
    survives = True
    checked = 0
    empty = []
    if diagnose:
        for source in sources:
            history = state.get("feasible_histories", {}).get(str(source.channel), [])
            if not any(item[1] == "direction" for item in history):
                continue
            feasible = model.FeasibleClient(None)
            feasible.history[source.channel] = history
            feasible._initialize(source.channel)
            feasible._rebuild(source.channel)
            candidates = feasible.models[source.channel]["q"]
            checked += 1
            if not len(candidates):
                empty.append(source.channel)
                survives = False
            elif min(
                math.dist((float(point[0]), float(point[1])), (source.x, source.y))
                for point in candidates
            ) > 15 + 1e-9:
                survives = False
    repeated_clear_points = len(arena.clear_points) - len(set(arena.clear_points))
    cost_ok = abs(movement_s + measure_s + switch_s + clears_s - arena.virtual_time_s) < 1e-6
    return {
        "all_checks_pass": bool(
            state.get("completed")
            and full_clear
            and accounted
            and survives
            and cost_ok
            and (not diagnose or repeated_clear_points == 0)
        ),
        "full_source_clear": full_clear,
        "channels_accounted": accounted,
        "false_excluded_source": false_excluded,
        "truth_neighborhood_survival": survives,
        "diagnosed_true_models": checked,
        "empty_model_channels": empty,
        "repeated_same_channel_point_clears": repeated_clear_points,
        "time_s": arena.virtual_time_s,
        "mean_s": arena.virtual_time_s / cleared_count,
        "movement_s": movement_s,
        "measure_s": measure_s,
        "switch_s": switch_s,
        "clears_s": clears_s,
        "fallback_attempts": state.get("fallback_clear_count", 0),
        "feasible_clear_attempts": state.get("feasible_clear_attempt_count", 0),
        "feasible_failed_clears": state.get("feasible_failed_clear_count", 0),
        "current_pose_measures": state.get("feasible_current_pose_measure_count", 0),
        "no_progress_stops": state.get("feasible_no_progress_count", 0),
        "q4_recoveries": state.get("feasible_q4_recovery_count", 0),
    }


def summarize(rows, name):
    return {
        key: statistics.mean(item[name][key] for item in rows)
        for key in ("mean_s", "movement_s", "measure_s", "switch_s", "clears_s")
    }


def main():
    O.mkdir(parents=True, exist_ok=True)
    cases = json.loads(SRC.read_text(encoding="utf-8"))
    copied_cases = O / "cases.json"
    if not copied_cases.exists():
        save(copied_cases, cases)
    files = [
        "code/shared/search_policy.py",
        "code/shared/search_policy_polygon.py",
        "code/shared/search_policy_q4_batch.py",
        "code/shared/search_policy_joint_route_q3.py",
        "code/shared/search_policy_q4_ring22.py",
        "code/shared/search_policy_q4_ring22_joint_route.py",
        "code/shared/search_policy_q4_joint_feasible.py",
        "code/shared/search_policy_q4_joint_feasible_v2.py",
        "code/shared/search_policy_q4_joint_feasible_v3.py",
        "code/shared/evaluate_q4_joint_feasible_v3_development.py",
        "results/optimization_200_400/ring22_continuous_certificate.json",
        "results/q4_reuse/development/cases.json",
        str(copied_cases.relative_to(R)),
    ]
    inputs = [{"path": item, "sha256": sha(R / item)} for item in files]
    save(O / "inputs_before_run.json", inputs)
    rows = []
    for case in cases:
        rows.append(
            {
                "case_id": case["case_id"],
                "seed": case["seed"],
                "ring22_joint": run(base.run_search, case),
                "joint_feasible_v3": run(cand.run_search, case, True),
            }
        )
    summaries = {name: summarize(rows, name) for name in ("ring22_joint", "joint_feasible_v3")}
    checks = all(item[name]["all_checks_pass"] for item in rows for name in summaries)
    improvement = 1 - summaries["joint_feasible_v3"]["mean_s"] / summaries["ring22_joint"]["mean_s"]
    output = {
        "scope": "SYNTHETIC_DEVELOPMENT_ONLY",
        "case_count": len(cases),
        "all_checks_pass": checks,
        "accepted": bool(checks and improvement >= 0.03),
        "acceptance_threshold": 0.03,
        "improvement_vs_ring22_joint": improvement,
        "all_truth_neighborhood_survival": all(
            item["joint_feasible_v3"]["truth_neighborhood_survival"] for item in rows
        ),
        "summaries": summaries,
        "rows": rows,
        "inputs": inputs,
    }
    save(O / "comparison.json", output)
    print(
        json.dumps(
            {
                "all_checks_pass": checks,
                "accepted": output["accepted"],
                "improvement": improvement,
                "survival": output["all_truth_neighborhood_survival"],
                "summaries": summaries,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
