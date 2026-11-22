"""Final independent offline holdout for the development-selected Q4 reuse_A."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict
from pathlib import Path

from offline_arena import OfflineArena, Source
from simulator_client import SimulatorClient
import search_policy_q4_reuse as reuse


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q4_reuse" / "final_holdout"
DEVELOPMENT_COMPARISON = ROOT / "results" / "q4_reuse" / "development" / "comparison.json"


def _save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases():
    cases = []
    for index, seed in enumerate(range(93000, 93060)):
        rng = random.Random(seed)
        count = (10, 13, 16)[index % 3]
        sources = []
        for source_index, channel in enumerate(rng.sample(range(1, 21), count)):
            angle = rng.uniform(0, 2 * math.pi)
            distance = 1800 * math.sqrt(rng.random())
            x, y = distance * math.cos(angle), distance * math.sin(angle)
            layout = index % 6
            if layout == 1:
                angle = 2 * math.pi * source_index / count
                x, y = 1800 * math.cos(angle), 1800 * math.sin(angle)
            elif layout == 2:
                grid = [(gx, gy) for gx in range(-1400, 1401, 700)
                        for gy in range(-1400, 1401, 700) if gx * gx + gy * gy <= 1800 ** 2]
                x, y = grid[source_index % len(grid)]
            elif layout == 3 and source_index < 3:
                x, y = ((0, 0), (5, 0), (20, 0))[source_index]
            elif layout == 4:
                angle = math.radians(30 + 60 * (source_index % 6))
                x, y = 1800 * math.cos(angle), 1800 * math.sin(angle)
            directional_count = (0, 1, count - 1, count, count // 2, count)[index % 6]
            heading = (math.degrees(angle) if layout in (1, 4) else rng.uniform(0, 360)) if source_index < directional_count else None
            reception = 1000 if index % 3 == 0 else 1500 if index % 3 == 1 else rng.uniform(1000, 1500)
            sources.append(Source(channel, float(x), float(y), reception, heading))
        cases.append({"question": 4, "case_id": f"reuse_final_q4_{seed}", "split": "final_independent_holdout",
                      "seed": seed, "error_mode": ("fixed_hash", "positive_edge", "negative_edge")[index % 3], "sources": sources})
    return cases


def _freeze_cases(cases):
    payload = [{**{key: value for key, value in case.items() if key != "sources"},
                "sources": [asdict(source) for source in case["sources"]]} for case in cases]
    target = OUT / "cases.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen final reuse cases differ from declared seeds")
    else:
        _save(target, payload)
    return target


def _freeze_hashes(cases_path):
    files = ["code/shared/search_policy.py", "code/shared/search_policy_polygon.py",
             "code/shared/search_policy_q4_batch.py", "code/shared/search_policy_q4_triangle.py",
             "code/shared/search_policy_q4_reuse.py", "code/shared/offline_arena.py",
             "code/shared/evaluate_q4_reuse_final.py", "results/q4_reuse/development/comparison.json",
             str(cases_path.relative_to(ROOT))]
    hashes = [{"path": file, "sha256": _sha256(ROOT / file)} for file in files]
    target = OUT / "inputs_before_run.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != hashes:
            raise ValueError("frozen final reuse hashes differ from current files")
    else:
        _save(target, hashes)
    return hashes


def _run(case):
    arena = OfflineArena(case["sources"], case["seed"], case["error_mode"])
    client = SimulatorClient("synthetic", transport=arena, log_path=False)
    client.enter()
    state = reuse.run_search(client, question=4, method="main", variant="A")
    client.exit()
    cleared = len(arena.cleared)
    movement_s, measure_s, switch_s = arena.distance_m / 5, 5 * arena.measure_count, arena.switch_count
    clears_s = 3 * arena.clear_count + 2 * cleared
    reconstructed = movement_s + measure_s + switch_s + clears_s
    groups = [set(state["cleared_channels"]), set(state["excluded_channels"]), set(state["unresolved_channels"])]
    union = set().union(*groups)
    accounted = union == set(range(1, 21)) and sum(len(group) for group in groups) == len(union)
    full_clear = cleared == len(case["sources"]) and set(state["cleared_channels"]) == arena.cleared and not (set(state["excluded_channels"]) & set(arena.sources))
    return {"all_checks_pass": bool(state.get("completed") and full_clear and accounted and abs(reconstructed - arena.virtual_time_s) < 1e-6),
            "full_source_clear": full_clear, "channels_accounted": accounted, "time_s": arena.virtual_time_s,
            "mean_s": arena.virtual_time_s / cleared if cleared else None, "movement_s": movement_s,
            "measure_s": measure_s, "switch_s": switch_s, "clears_s": clears_s,
            "clock_residual_s": arena.virtual_time_s - reconstructed, "fallback_attempts": state.get("fallback_clear_count", 0),
            "reuse_measure_count": state.get("reuse_measure_count", 0)}


def main():
    development = json.loads(DEVELOPMENT_COMPARISON.read_text(encoding="utf-8"))
    if development.get("selected_for_final_holdout") != "reuse_A":
        raise ValueError("reuse_A was not the frozen development-selected candidate")
    OUT.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    cases_path = _freeze_cases(cases)
    hashes = _freeze_hashes(cases_path)
    rows = [{"case_id": case["case_id"], "seed": case["seed"], "result": _run(case)} for case in cases]
    summary = {"cases": len(rows), "mean_per_source_s": statistics.mean(row["result"]["mean_s"] for row in rows),
               "mean_movement_s": statistics.mean(row["result"]["movement_s"] for row in rows),
               "mean_measure_s": statistics.mean(row["result"]["measure_s"] for row in rows),
               "mean_clears_s": statistics.mean(row["result"]["clears_s"] for row in rows),
               "mean_switch_s": statistics.mean(row["result"]["switch_s"] for row in rows),
               "total_fallback_attempts": sum(row["result"]["fallback_attempts"] for row in rows)}
    comparison = {"scope": "SYNTHETIC_FINAL_INDEPENDENT_HOLDOUT", "artifact_type": "q4_reuse_final_holdout",
                  "selection_provenance": "Only reuse_A was run because it was selected by the frozen development comparison; no baseline or rejected variant was evaluated on seeds 93000-93059.",
                  "case_count": len(rows), "seed_range": [93000, 93059],
                  "all_checks_pass": all(row["result"]["all_checks_pass"] for row in rows), "summary": summary,
                  "rows": rows, "inputs": hashes}
    _save(OUT / "comparison.json", comparison)
    print(json.dumps({"all_checks_pass": comparison["all_checks_pass"], "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
