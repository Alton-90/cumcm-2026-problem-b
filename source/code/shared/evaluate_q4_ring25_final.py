"""Final independent paired Q4 comparison: reuse_A versus accepted ring25."""

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
import search_policy_q4_ring25 as ring25


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q4_ring25" / "final_holdout"
DEVELOPMENT = ROOT / "results" / "q4_ring25" / "development" / "comparison.json"
POLICIES = {"reuse_A": reuse.run_search, "ring25": ring25.run_search}


def _save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases():
    cases = []
    for index, seed in enumerate(range(95000, 95060)):
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
                grid = [(gx, gy) for gx in range(-1400, 1401, 700) for gy in range(-1400, 1401, 700)
                        if gx * gx + gy * gy <= 1800 ** 2]
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
        cases.append({"question": 4, "case_id": f"ring25_q4_{seed}", "split": "final_independent_holdout",
                      "seed": seed, "error_mode": ("fixed_hash", "positive_edge", "negative_edge")[index % 3], "sources": sources})
    return cases


def _freeze_cases(cases):
    payload = [{**{key: value for key, value in case.items() if key != "sources"},
                "sources": [asdict(source) for source in case["sources"]]} for case in cases]
    target = OUT / "cases.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen ring25 final cases differ from declared seeds")
    else:
        _save(target, payload)
    return target


def _freeze_hashes(cases_path):
    files = ["code/shared/search_policy.py", "code/shared/search_policy_polygon.py",
             "code/shared/search_policy_q4_batch.py", "code/shared/search_policy_q4_triangle.py",
             "code/shared/search_policy_q4_reuse.py", "code/shared/search_policy_q4_ring25.py",
             "code/shared/offline_arena.py", "code/shared/evaluate_q4_ring25_final.py",
             "results/q4_ring25/development/comparison.json", str(cases_path.relative_to(ROOT))]
    hashes = [{"path": path, "sha256": _sha256(ROOT / path)} for path in files]
    target = OUT / "inputs_before_run.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != hashes:
            if (OUT / "comparison.json").exists():
                raise ValueError("frozen ring25 final hashes differ from current files")
            _save(target, hashes)
    else:
        _save(target, hashes)
    return hashes


def _run(policy, case):
    sources = [source if isinstance(source, Source) else Source(**source) for source in case["sources"]]
    arena = OfflineArena(sources, case["seed"], case["error_mode"])
    client = SimulatorClient("synthetic", transport=arena, log_path=False)
    client.enter()
    state = policy(client, question=4, method="main")
    client.exit()
    cleared = len(arena.cleared)
    movement_s, measure_s, switch_s = arena.distance_m / 5, 5 * arena.measure_count, arena.switch_count
    clears_s = 3 * arena.clear_count + 2 * cleared
    reconstructed = movement_s + measure_s + switch_s + clears_s
    groups = [set(state["cleared_channels"]), set(state["excluded_channels"]), set(state["unresolved_channels"])]
    union = set().union(*groups)
    accounted = union == set(range(1, 21)) and sum(len(group) for group in groups) == len(union)
    full_clear = cleared == len(sources) and set(state["cleared_channels"]) == arena.cleared and not (set(state["excluded_channels"]) & set(arena.sources))
    return {"all_checks_pass": bool(state.get("completed") and full_clear and accounted and abs(reconstructed - arena.virtual_time_s) < 1e-6),
            "full_source_clear": full_clear, "channels_accounted": accounted, "time_s": arena.virtual_time_s,
            "mean_s": arena.virtual_time_s / cleared if cleared else None, "movement_s": movement_s,
            "measure_s": measure_s, "switch_s": switch_s, "clears_s": clears_s,
            "clock_residual_s": arena.virtual_time_s - reconstructed, "fallback_attempts": state.get("fallback_clear_count", 0)}


def _summary(rows, method):
    return {"cases": len(rows), "mean_per_source_s": statistics.mean(row[method]["mean_s"] for row in rows),
            "mean_movement_s": statistics.mean(row[method]["movement_s"] for row in rows),
            "mean_measure_s": statistics.mean(row[method]["measure_s"] for row in rows),
            "mean_clears_s": statistics.mean(row[method]["clears_s"] for row in rows),
            "mean_switch_s": statistics.mean(row[method]["switch_s"] for row in rows),
            "total_fallback_attempts": sum(row[method]["fallback_attempts"] for row in rows)}


def main():
    development = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    if development.get("accepted") is not True:
        raise ValueError("ring25 did not pass frozen development selection")
    OUT.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    cases_path = _freeze_cases(cases)
    hashes = _freeze_hashes(cases_path)
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "seed": case["seed"]}
        for name, policy in POLICIES.items():
            row[name] = _run(policy, case)
        row["ring25_to_reuse_A_ratio"] = row["ring25"]["time_s"] / row["reuse_A"]["time_s"]
        rows.append(row)
    summaries = {name: _summary(rows, name) for name in POLICIES}
    all_checks = all(row[name]["all_checks_pass"] for row in rows for name in POLICIES)
    improvement = 1 - summaries["ring25"]["mean_per_source_s"] / summaries["reuse_A"]["mean_per_source_s"]
    comparison = {"scope": "SYNTHETIC_FINAL_INDEPENDENT_HOLDOUT", "artifact_type": "q4_ring25_final_comparison",
                  "case_count": len(rows), "seed_range": [95000, 95059], "all_checks_pass": all_checks,
                  "accepted": bool(all_checks and improvement > 0),
                  "acceptance_rule": "accepted requires all paired cases to pass full-clear, 20-channel, and reconstructed-clock checks, plus a strictly lower ring25 mean_per_source_s.",
                  "improvement_vs_reuse_A": improvement, "coverage_certificate": ring25.coverage_certificate(),
                  "summaries": summaries, "rows": rows, "inputs": hashes}
    _save(OUT / "comparison.json", comparison)
    print(json.dumps({"all_checks_pass": all_checks, "accepted": comparison["accepted"], "improvement": improvement,
                      "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
