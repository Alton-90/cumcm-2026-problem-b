"""Generate the frozen fresh 42-case Q4 edge-waypoint v1 set once."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/Q4/experiments/edge_waypoint_v1"
TARGET = OUT / "cases.json"
METADATA = OUT / "case_generation.json"
PLAN = ROOT / "planning/q4_edge_waypoint_v1_experiment.md"
CODE_PLAN = ROOT / "code/Q4/q4_code_plan_edge_waypoint_v1.md"
PREDECLARATION = ROOT / "planning/q4_edge_waypoint_v1_predeclaration.json"
CANDIDATE_FREEZE = ROOT / "planning/q4_edge_waypoint_v1_candidate_freeze.json"
GEOMETRY = ROOT / "code/shared/q4_edge_waypoint_geometry.py"
CANDIDATE = ROOT / "code/shared/search_policy_q4_edge_waypoint_v1.py"
SEED_START = 65000
SEED_END = 65041
MODES = ("fixed_hash", "positive_edge", "negative_edge")
RECEPTION_RADII = (1000.0, 1250.0, 1500.0)
RULE_VERSION = "q4_edge_waypoint_v1_fresh42_rule_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def case_records(payload):
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return [item for item in payload["cases"] if isinstance(item, dict)]
    return []


def prior_case_corpus():
    files, seeds, case_ids = [], set(), set()
    for path in sorted(ROOT.rglob("cases.json")):
        if path.resolve() == TARGET.resolve():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = case_records(payload)
        for record in records:
            if isinstance(record.get("seed"), int):
                seeds.add(record["seed"])
            if isinstance(record.get("case_id"), str):
                case_ids.add(record["case_id"])
        files.append({
            "path": relative(path), "sha256": sha(path),
            "case_record_count": len(records),
        })
    return files, seeds, case_ids


def make_cases():
    cases = []
    for source_count in range(10, 17):
        for mode_index, error_mode in enumerate(MODES):
            for replicate in range(2):
                index = (source_count - 10) * 6 + mode_index * 2 + replicate
                seed = SEED_START + index
                rng = random.Random(seed)
                channels = rng.sample(range(1, 21), source_count)
                phase = rng.uniform(0.0, 2.0 * math.pi)
                slot = 2.0 * math.pi / source_count
                sources = []
                for source_index, channel in enumerate(channels):
                    band_index = (source_index + mode_index + replicate) % 3
                    if band_index == 0:
                        radial_distance = 150.0 + 450.0 * math.sqrt(rng.random())
                        band_name = "inner"
                    elif band_index == 1:
                        radial_distance = 750.0 + 650.0 * rng.random()
                        band_name = "middle"
                    else:
                        radial_distance = 1800.0 if (source_index + seed) % 2 == 0 else 1799.0
                        band_name = "boundary"
                    angle = (
                        phase + slot * source_index
                        + rng.uniform(-0.25 * slot, 0.25 * slot)
                    ) % (2.0 * math.pi)
                    x = radial_distance * math.cos(angle)
                    y = radial_distance * math.sin(angle)
                    reception_radius = RECEPTION_RADII[
                        (source_index + mode_index + 2 * replicate) % 3
                    ]
                    directional = (source_index + mode_index + replicate) % 2 == 0
                    if not directional:
                        heading = None
                    elif band_name == "boundary":
                        heading = math.degrees(angle) % 360.0
                    else:
                        heading = rng.uniform(0.0, 360.0)
                    sources.append({
                        "channel": int(channel), "x": float(x), "y": float(y),
                        "radius": reception_radius,
                        "heading": None if heading is None else float(heading),
                    })
                cases.append({
                    "question": 4,
                    "case_id": f"edge_waypoint_v1_q4_{seed}",
                    "split": "fresh_structural_offline_edge_waypoint_v1",
                    "seed": seed,
                    "source_count_stratum": source_count,
                    "error_mode": error_mode,
                    "replicate": replicate,
                    "sources": sources,
                })
    return cases


def band(radius):
    if 150.0 - 1e-9 <= radius <= 600.0 + 1e-9:
        return "inner"
    if 750.0 - 1e-9 <= radius <= 1400.0 + 1e-9:
        return "middle"
    if 1799.0 - 1e-9 <= radius <= 1800.0 + 1e-9:
        return "boundary"
    return "invalid"


def distribution_audit(cases):
    source_counts = Counter(case["source_count_stratum"] for case in cases)
    error_modes = Counter(case["error_mode"] for case in cases)
    cells = Counter(
        (case["source_count_stratum"], case["error_mode"]) for case in cases
    )
    case_checks = []
    for case in cases:
        sources = case["sources"]
        bands = Counter(band(math.hypot(source["x"], source["y"])) for source in sources)
        radii = Counter(source["radius"] for source in sources)
        types = Counter(
            "directional" if source["heading"] is not None else "omnidirectional"
            for source in sources
        )
        checks = {
            "declared_count_matches": len(sources) == case["source_count_stratum"],
            "channels_unique": len({source["channel"] for source in sources}) == len(sources),
            "channels_in_range": all(1 <= source["channel"] <= 20 for source in sources),
            "positions_in_declared_bands": "invalid" not in bands,
            "position_bands_balanced": max(bands.values()) - min(bands.values()) <= 1,
            "reception_radii_valid": set(radii) <= set(RECEPTION_RADII),
            "reception_radii_balanced": max(radii.values()) - min(radii.values()) <= 1,
            "directionality_balanced": max(types.values()) - min(types.values()) <= 1,
        }
        case_checks.append({
            "case_id": case["case_id"], "all_checks_pass": all(checks.values()),
            "checks": checks, "position_band_counts": dict(sorted(bands.items())),
            "reception_radius_counts": {str(k): v for k, v in sorted(radii.items())},
            "directionality_counts": dict(sorted(types.items())),
        })
    global_checks = {
        "case_count_is_42": len(cases) == 42,
        "seed_range_exact": [case["seed"] for case in cases]
        == list(range(SEED_START, SEED_END + 1)),
        "case_ids_unique": len({case["case_id"] for case in cases}) == len(cases),
        "each_source_count_has_six_cases": source_counts
        == Counter({count: 6 for count in range(10, 17)}),
        "each_error_mode_has_fourteen_cases": error_modes
        == Counter({mode: 14 for mode in MODES}),
        "each_cell_has_two_cases": cells
        == Counter({(count, mode): 2 for count in range(10, 17) for mode in MODES}),
        "all_case_level_checks_pass": all(item["all_checks_pass"] for item in case_checks),
    }
    return {
        "all_checks_pass": all(global_checks.values()),
        "global_checks": global_checks,
        "source_count_case_counts": {str(k): v for k, v in sorted(source_counts.items())},
        "error_mode_case_counts": dict(sorted(error_modes.items())),
        "case_checks": case_checks,
    }


def main():
    required = [PLAN, CODE_PLAN, PREDECLARATION, CANDIDATE_FREEZE, GEOMETRY, CANDIDATE]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("predeclaration, code plan, and frozen candidate must exist")
    freeze = json.loads(CANDIDATE_FREEZE.read_text(encoding="utf-8"))
    expected = {item["path"]: item["sha256"] for item in freeze["frozen_candidate_files"]}
    for path in (GEOMETRY, CANDIDATE):
        if sha(path) != expected[relative(path)]:
            raise ValueError("candidate changed after prospective freeze")
    cases = make_cases()
    distribution = distribution_audit(cases)
    if not distribution["all_checks_pass"]:
        raise ValueError("predeclared distribution failed its own audit")
    if TARGET.exists() or METADATA.exists():
        if not TARGET.exists() or not METADATA.exists():
            raise ValueError("partial frozen case artifacts exist")
        if json.loads(TARGET.read_text(encoding="utf-8")) != cases:
            raise ValueError("frozen cases differ from generator")
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        if metadata["case_file_sha256"] != sha(TARGET):
            raise ValueError("frozen case hash mismatch")
        print(json.dumps({"status": "already_frozen", "sha256": sha(TARGET)}))
        return
    prior_files, prior_seeds, prior_ids = prior_case_corpus()
    collisions = sorted(set(range(SEED_START, SEED_END + 1)) & prior_seeds)
    id_collisions = sorted({case["case_id"] for case in cases} & prior_ids)
    if collisions or id_collisions:
        raise ValueError(f"fresh identity collision: seeds={collisions}, ids={id_collisions}")
    save(TARGET, cases)
    metadata = {
        "schema_version": 1,
        "artifact_type": "q4_edge_waypoint_v1_case_generation",
        "rule_version": RULE_VERSION,
        "scope": "FRESH_SYNTHETIC_STRUCTURAL_OFFLINE_ONLY",
        "freeze_identity": {
            "generator_path": relative(Path(__file__)),
            "generator_sha256": sha(Path(__file__)),
            "plan_path": relative(PLAN), "plan_sha256": sha(PLAN),
            "code_plan_path": relative(CODE_PLAN), "code_plan_sha256": sha(CODE_PLAN),
            "predeclaration_path": relative(PREDECLARATION),
            "predeclaration_sha256": sha(PREDECLARATION),
            "candidate_freeze_path": relative(CANDIDATE_FREEZE),
            "candidate_freeze_sha256": sha(CANDIDATE_FREEZE),
            "geometry_sha256_at_generation": sha(GEOMETRY),
            "candidate_sha256_at_generation": sha(CANDIDATE),
        },
        "seed_range": [SEED_START, SEED_END],
        "case_count": len(cases),
        "case_file_path": relative(TARGET),
        "case_file_sha256": sha(TARGET),
        "novelty_audit": {
            "all_checks_pass": not collisions and not id_collisions,
            "seed_collisions": collisions, "case_id_collisions": id_collisions,
            "prior_case_file_count": len(prior_files),
            "prior_case_files": prior_files,
        },
        "distribution_audit": distribution,
    }
    save(METADATA, metadata)
    print(json.dumps({
        "status": "generated_and_frozen", "case_count": len(cases),
        "seed_range": [SEED_START, SEED_END], "case_sha256": sha(TARGET),
        "prior_case_files_scanned": len(prior_files),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
