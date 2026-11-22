"""Generate the predeclared Q3 edge-opportunity cases after code freeze.

Historical JSON is scanned only as text for case_id/seed metadata.  No policy,
arena, strict confirmation case, GUI, or live endpoint is executed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q3_edge_opportunity_v1" / "development"
CASE_FILE = OUT / "cases.json"
COLLISION_FILE = OUT / "collision_audit.json"
FREEZE_FILE = OUT / "freeze_manifest.json"
FIRST_SEED = 64000
LAST_SEED = 64041
CASE_ID_PREFIX = "q3_edge_opportunity_v1_"
CASE_ID_PATTERN = re.compile(r'"case_id"\s*:\s*"([^"]+)"')
SEED_PATTERN = re.compile(r'"seed"\s*:\s*(-?\d+)')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)+"\n",
        encoding="utf-8",
    )


def verify_freeze_manifest():
    if not FREEZE_FILE.exists():
        raise RuntimeError("freeze_manifest.json must exist before collision scan or generation")
    manifest = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    errors = []
    for item in manifest.get("files", []):
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            errors.append(item["path"])
    if errors:
        raise RuntimeError("post-freeze input drift: "+", ".join(errors))
    return manifest


def historical_metadata():
    identifiers = set()
    seeds = set()
    scanned = []
    results_root = ROOT / "results"
    for path in sorted(results_root.rglob("*.json")):
        if OUT in path.parents or "case" not in path.name.lower():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        file_ids = CASE_ID_PATTERN.findall(text)
        file_seeds = [int(value) for value in SEED_PATTERN.findall(text)]
        if not file_ids and not file_seeds:
            continue
        identifiers.update(file_ids)
        seeds.update(file_seeds)
        scanned.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "case_id_occurrences": len(file_ids),
                "seed_occurrences": len(file_seeds),
            }
        )
    return identifiers, seeds, scanned


def build_cases():
    cases = []
    modes = ("fixed_hash", "positive_edge", "negative_edge")
    for index, seed in enumerate(range(FIRST_SEED, LAST_SEED+1)):
        rng = random.Random(seed)
        count = 10+(index % 7)
        channels = rng.sample(range(1, 21), count)
        source_rows = []
        for channel in channels:
            angle = rng.uniform(0.0, 2.0*math.pi)
            if index % 7 == 0:
                radial_distance = 1800.0
            elif index % 7 == 1:
                radial_distance = 1000.0
            else:
                radial_distance = 1800.0*math.sqrt(rng.random())
            if index % 5 == 0:
                reception = 1000.0
            elif index % 5 == 1:
                reception = 1500.0
            else:
                reception = rng.uniform(1000.0, 1500.0)
            source_rows.append(
                {
                    "channel": channel,
                    "x": radial_distance*math.cos(angle),
                    "y": radial_distance*math.sin(angle),
                    "radius": reception,
                    "heading": None,
                }
            )
        cases.append(
            {
                "case_id": CASE_ID_PREFIX+str(seed),
                "question": 3,
                "split": "fresh_fixed_development",
                "seed": seed,
                "error_mode": modes[index % len(modes)],
                "sources": source_rows,
            }
        )
    return cases


def main():
    if CASE_FILE.exists() or COLLISION_FILE.exists():
        raise RuntimeError("collision/case artifact already exists; refusing rerun or overwrite")
    manifest = verify_freeze_manifest()
    identifiers, historical_seeds, scanned = historical_metadata()
    target_seeds = list(range(FIRST_SEED, LAST_SEED+1))
    target_ids = [CASE_ID_PREFIX+str(seed) for seed in target_seeds]
    id_collisions = sorted(set(target_ids) & identifiers)
    seed_collisions = sorted(set(target_seeds) & historical_seeds)
    audit = {
        "schema_version": 1,
        "artifact_type": "q3_edge_opportunity_case_collision_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_only_scan": True,
        "historical_cases_executed": 0,
        "strict84_cases_executed": 0,
        "live_or_gui_calls": 0,
        "target_seed_range": [FIRST_SEED, LAST_SEED],
        "target_case_ids": target_ids,
        "scanned_files": scanned,
        "unique_historical_case_id_count": len(identifiers),
        "unique_historical_seed_count": len(historical_seeds),
        "case_id_collisions": id_collisions,
        "seed_collisions": seed_collisions,
        "all_checks_pass": not id_collisions and not seed_collisions,
        "freeze_manifest_sha256": sha256(FREEZE_FILE),
        "generator_sha256": sha256(Path(__file__)),
        "freeze_verified": bool(manifest),
    }
    save(COLLISION_FILE, audit)
    if not audit["all_checks_pass"]:
        raise RuntimeError("predeclared case id/seed collision; aborting without replacement")
    cases = build_cases()
    if (
        len(cases) != 42
        or [case["seed"] for case in cases] != target_seeds
        or [case["case_id"] for case in cases] != target_ids
    ):
        raise AssertionError("predeclared case construction invariant failed")
    save(CASE_FILE, cases)
    print(
        json.dumps(
            {
                "collision_audit": str(COLLISION_FILE),
                "cases": str(CASE_FILE),
                "case_count": len(cases),
                "cases_sha256": sha256(CASE_FILE),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
