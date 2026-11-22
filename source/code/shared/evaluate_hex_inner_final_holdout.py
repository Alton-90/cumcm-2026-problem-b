"""Final independent Q3 holdout: compact, hex_online, and frozen hex_inner.

This evaluator is offline-only.  OfflineArena retains source data privately;
each candidate sees only the normal SimulatorClient surface.
"""

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
import search_policy_compact as compact
import search_policy_hex_online as hex_online
import search_policy_hex_inner as hex_inner


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "hex_inner_q3" / "final_holdout"
METHODS = {"compact": compact.run_search, "hex_online": hex_online.run_search, "hex_inner": hex_inner.run_search}


def _save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_layout(seed, index):
    """Six predeclared stress layouts covering boundary, bisector, and R=1000."""
    rng = random.Random(seed)
    count = (10, 13, 16)[index % 3]
    channels = rng.sample(range(1, 21), count)
    pattern = index % 6
    sources = []
    for source_index, channel in enumerate(channels):
        if pattern == 0:  # exact outer circle and 30-degree Voronoi boundaries
            angle = math.radians(30 + 60 * (source_index % 6))
            radius = 1800.0
        elif pattern == 1:  # origin's limiting reception boundary
            angle = rng.uniform(0, 2 * math.pi)
            radius = 1000.0
        elif pattern == 2:  # outer circle, each six-sector angular midpoint
            angle = math.radians(30 + 60 * (source_index % 6))
            radius = 1800.0
        elif pattern == 3:  # mixed exact edge radii
            angle = math.radians(30 + 60 * (source_index % 6))
            radius = (0.0, 5.0, 20.0, 999.999, 1000.0, 1800.0)[source_index % 6]
        elif pattern == 4:  # random disk at the limiting reception radius
            angle = rng.uniform(0, 2 * math.pi)
            radius = 1800.0 * math.sqrt(rng.random())
        else:  # mixed interior and circle edge
            angle = rng.uniform(0, 2 * math.pi)
            radius = 1800.0 if source_index % 4 == 0 else 1800.0 * math.sqrt(rng.random())
        sources.append(Source(channel, radius * math.cos(angle), radius * math.sin(angle), 1000.0, None))
    return sources, ("outer_angle30", "origin_r1000", "outer_angle30_repeat", "mixed_edges", "random_r1000", "mixed_disk")[pattern]


def declared_cases():
    cases = []
    for index, seed in enumerate(range(81000, 81060)):
        sources, layout = _source_layout(seed, index)
        cases.append({"question": 3, "case_id": f"final_q3_{seed}", "split": "final_independent_holdout",
                      "seed": seed, "layout": layout,
                      "error_mode": ("fixed_hash", "positive_edge", "negative_edge")[index % 3], "sources": sources})
    return cases


def _serialized(cases):
    return [{**{key: value for key, value in case.items() if key != "sources"},
             "sources": [asdict(source) for source in case["sources"]]} for case in cases]


def _freeze_cases(cases):
    path = OUT / "cases.json"
    payload = _serialized(cases)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen final holdout cases differ from declared seeds/layouts")
    else:
        _save(path, payload)
    return path


def _freeze_hashes(cases_path):
    paths = [
        "code/shared/search_policy.py", "code/shared/search_policy_compact.py",
        "code/shared/search_policy_hex_online.py", "code/shared/search_policy_hex_inner.py",
        "code/shared/offline_arena.py", "code/shared/evaluate_hex_inner_final_holdout.py",
        str(cases_path.relative_to(ROOT)),
    ]
    payload = [{"path": path, "sha256": _sha256(ROOT / path)} for path in paths]
    target = OUT / "inputs_before_run.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen final holdout hashes differ from current inputs")
    else:
        _save(target, payload)
    return payload


def _run(policy, case):
    arena = OfflineArena(case["sources"], case["seed"], case["error_mode"])
    client = SimulatorClient("synthetic", transport=arena, log_path=False)
    client.enter()
    state = policy(client, question=3, method="main")
    client.exit()
    cleared = len(arena.cleared)
    reconstructed = (arena.distance_m / 5 + 5 * arena.measure_count + arena.switch_count +
                     3 * arena.clear_count + 2 * cleared)
    groups = [set(state["cleared_channels"]), set(state["excluded_channels"]), set(state["unresolved_channels"])]
    partition = set().union(*groups)
    accounted = (partition == set(range(1, 21)) and sum(len(group) for group in groups) == len(partition))
    full_clear = (cleared == len(case["sources"]) and set(state["cleared_channels"]) == arena.cleared and
                  not (set(state["excluded_channels"]) & set(arena.sources)))
    time_ok = abs(reconstructed - arena.virtual_time_s) < 1e-6
    passed = bool(state.get("completed") and full_clear and accounted and time_ok)
    return {
        "all_checks_pass": passed, "full_source_clear": full_clear, "channels_accounted": accounted,
        "virtual_clock_reconstructed": reconstructed, "virtual_time_s": arena.virtual_time_s,
        "clock_residual_s": arena.virtual_time_s - reconstructed, "cleared": cleared, "n": len(case["sources"]),
        "mean_s": arena.virtual_time_s / cleared if cleared else None, "movement_m": arena.distance_m,
        "measures": arena.measure_count, "switches": arena.switch_count, "clears": arena.clear_count,
        "fallback_attempts": state.get("fallback_clear_count", 0),
    }


def _summary(rows, method):
    values = [row[method]["mean_s"] for row in rows]
    return {"cases": len(rows), "mean_per_source_s": statistics.mean(values), "max_per_source_s": max(values),
            "total_fallback_attempts": sum(row[method]["fallback_attempts"] for row in rows)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = declared_cases()
    cases_path = _freeze_cases(cases)
    hashes = _freeze_hashes(cases_path)
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "seed": case["seed"], "layout": case["layout"],
               "error_mode": case["error_mode"]}
        for name, policy in METHODS.items():
            row[name] = _run(policy, case)
        row["hex_online_to_compact_ratio"] = row["hex_online"]["virtual_time_s"] / row["compact"]["virtual_time_s"]
        row["hex_inner_to_compact_ratio"] = row["hex_inner"]["virtual_time_s"] / row["compact"]["virtual_time_s"]
        rows.append(row)
    summaries = {name: _summary(rows, name) for name in METHODS}
    comparison = {
        "scope": "SYNTHETIC_FINAL_INDEPENDENT_HOLDOUT",
        "selection_provenance": "Seeds 81000-81059 were generated only for this final comparison. Earlier per-candidate holdouts were used for screening and are not reported here as final evidence.",
        "all_checks_pass": all(row[name]["all_checks_pass"] for row in rows for name in METHODS),
        "criteria": {"full_source_clear": True, "all_20_channels_accounted": True, "reconstructed_virtual_clock": True},
        "case_count": len(rows), "seed_range": [81000, 81059], "summaries": summaries, "rows": rows,
        "inputs": hashes,
        "hex_inner_geometry": {"ring_radius_m": 1150.0, "minimum_reception_radius_m": 1000.0,
                               "outer_angle30_worst_distance_m": math.sqrt(1800 ** 2 + 1150 ** 2 - 2 * 1800 * 1150 * math.cos(math.pi / 6))},
    }
    _save(OUT / "comparison.json", comparison)
    print(json.dumps({"all_checks_pass": comparison["all_checks_pass"], "case_count": len(rows),
                      "summaries": summaries, "hex_inner_geometry": comparison["hex_inner_geometry"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
