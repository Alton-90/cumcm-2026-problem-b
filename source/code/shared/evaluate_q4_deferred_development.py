"""Frozen-development comparison: triangle980 versus deferred A and B."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from offline_arena import OfflineArena, Source
from simulator_client import SimulatorClient
import search_policy_q4_deferred as deferred
import search_policy_q4_triangle as triangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q4_deferred" / "development"
SOURCE_CASES = ROOT / "results" / "q4_triangle" / "cases.json"
METHODS = {"triangle980": (triangle.run_search, None), "deferred_A": (deferred.run_search, "A"),
           "deferred_B": (deferred.run_search, "B")}


def _save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_cases():
    payload = [case for case in json.loads(SOURCE_CASES.read_text(encoding="utf-8")) if case["split"] == "development"]
    if len(payload) != 36:
        raise ValueError("expected 36 fixed Q4 development cases")
    target = OUT / "cases.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen deferred development cases differ from source")
    else:
        _save(target, payload)
    return payload, target


def _freeze_hashes(cases_path):
    files = [
        "code/shared/search_policy.py", "code/shared/search_policy_compact.py",
        "code/shared/search_policy_polygon.py", "code/shared/search_policy_q4_batch.py",
        "code/shared/search_policy_q4_triangle.py", "code/shared/search_policy_q4_deferred.py",
        "code/shared/offline_arena.py", "code/shared/evaluate_q4_deferred_development.py",
        "results/q4_triangle/cases.json", str(cases_path.relative_to(ROOT)),
    ]
    payload = [{"path": file, "sha256": _sha256(ROOT / file)} for file in files]
    target = OUT / "inputs_before_run.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen deferred development hashes differ from current files")
    else:
        _save(target, payload)
    return payload


def _run(policy, variant, case):
    sources = [Source(**source) for source in case["sources"]]
    arena = OfflineArena(sources, case["seed"], case["error_mode"])
    client = SimulatorClient("synthetic", transport=arena, log_path=False)
    client.enter()
    kwargs = {"question": 4, "method": "main"}
    if variant is not None:
        kwargs["variant"] = variant
    state = policy(client, **kwargs)
    client.exit()
    cleared = len(arena.cleared)
    movement_s = arena.distance_m / 5
    measure_s = 5 * arena.measure_count
    switch_s = arena.switch_count
    clears_s = 3 * arena.clear_count + 2 * cleared
    reconstructed = movement_s + measure_s + switch_s + clears_s
    groups = [set(state["cleared_channels"]), set(state["excluded_channels"]), set(state["unresolved_channels"])]
    union = set().union(*groups)
    accounted = union == set(range(1, 21)) and sum(len(group) for group in groups) == len(union)
    full_clear = cleared == len(sources) and set(state["cleared_channels"]) == arena.cleared and not (set(state["excluded_channels"]) & set(arena.sources))
    return {"all_checks_pass": bool(state.get("completed") and full_clear and accounted and abs(reconstructed - arena.virtual_time_s) < 1e-6),
            "full_source_clear": full_clear, "channels_accounted": accounted, "time_s": arena.virtual_time_s,
            "mean_s": arena.virtual_time_s / cleared if cleared else None,
            "movement_s": movement_s, "measure_s": measure_s, "switch_s": switch_s, "clears_s": clears_s,
            "clock_residual_s": arena.virtual_time_s - reconstructed, "cleared": cleared, "n": len(sources),
            "fallback_attempts": state.get("fallback_clear_count", 0),
            "deferred_immediate_clear_count": state.get("deferred_immediate_clear_count", 0),
            "deferred_final_clear_count": state.get("deferred_final_clear_count", 0)}


def _summary(rows, method):
    return {"cases": len(rows), "mean_per_source_s": statistics.mean(row[method]["mean_s"] for row in rows),
            "mean_movement_s": statistics.mean(row[method]["movement_s"] for row in rows),
            "mean_measure_s": statistics.mean(row[method]["measure_s"] for row in rows),
            "mean_clears_s": statistics.mean(row[method]["clears_s"] for row in rows),
            "mean_switch_s": statistics.mean(row[method]["switch_s"] for row in rows),
            "total_fallback_attempts": sum(row[method]["fallback_attempts"] for row in rows)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases, cases_path = _freeze_cases()
    hashes = _freeze_hashes(cases_path)
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "seed": case["seed"]}
        for name, (policy, variant) in METHODS.items():
            row[name] = _run(policy, variant, case)
        rows.append(row)
    summaries = {name: _summary(rows, name) for name in METHODS}
    eligible = [name for name in ("deferred_A", "deferred_B") if all(row[name]["all_checks_pass"] for row in rows)]
    selected = min(eligible, key=lambda name: summaries[name]["mean_per_source_s"]) if eligible else None
    comparison = {"scope": "SYNTHETIC_DEVELOPMENT_ONLY", "artifact_type": "q4_deferred_development_comparison",
                  "case_count": len(rows), "all_checks_pass": all(row[name]["all_checks_pass"] for row in rows for name in METHODS),
                  "selection_rule": "Choose the all-checks-pass deferred variant with lower development mean_per_source_s; only it may run the new final holdout.",
                  "selected_for_final_holdout": selected, "summaries": summaries, "rows": rows, "inputs": hashes}
    _save(OUT / "comparison.json", comparison)
    print(json.dumps({"all_checks_pass": comparison["all_checks_pass"], "selected_for_final_holdout": selected,
                      "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
