"""Strict bound-cache confirmation for the frozen Q3 optimizer benchmark.

This entrypoint does not rerun the nine 48-proposal training studies.  It
fail-closes on non-finite or inconsistent cache data, audits the complete
bound-cache corpus before and after confirmation, evaluates the nine frozen
training winners on the 84 frozen selection cases, and applies the declared
conservative tie rule.  Legacy unbound caches are never read by confirmation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import statistics
import tempfile

import benchmark_q3_optimizers as b


OUT = b.OUT
BOUND = OUT / "bound_cache"
PLAN = b.load(OUT / "plan.json")
TIE_TOLERANCE = 1e-9


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def rel(path):
    return Path(path).relative_to(b.ROOT).as_posix()


def file_ref(path):
    path = Path(path)
    return {"path": rel(path), "sha256": b.core.original.sha(path)}


def require_bool(value, label):
    if type(value) is not bool:
        raise ValueError(label + " must be a boolean")
    return value


def require_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(label + " must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def require_count(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(label + " must be a non-negative integer")
    return int(value)


def reject_nonfinite_tree(value, label="value"):
    """Reject every non-finite numeric leaf, including unreported intermediates."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, Real):
        require_finite(value, label)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            reject_nonfinite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_nonfinite_tree(item, f"{label}[{index}]")


def frozen_cases(label):
    path = OUT / f"{label}_cases.json"
    cases = b.load(path)
    expected_sha = PLAN[f"{label}_sha256"]
    if b.core.original.sha(path) != expected_sha:
        raise ValueError(f"Frozen {label} case file hash changed")
    reject_nonfinite_tree(cases, label + "_cases")
    identifiers = [case.get("case_id") for case in cases]
    if len(identifiers) != len(set(identifiers)) or any(not value for value in identifiers):
        raise ValueError(f"Frozen {label} case identifiers are missing or duplicated")
    return cases


def binding(cases, label, parameters=None):
    expected_cases = frozen_cases(label)
    if cases != expected_cases:
        raise ValueError("Case contents or ordering differ from frozen " + label + " dataset")
    if parameters is None:
        normalized = None
    else:
        if len(parameters) != 3:
            raise ValueError("Parameter vector must contain exactly three values")
        normalized = [require_finite(value, f"parameters[{index}]") for index, value in enumerate(parameters)]
    return {
        "plan_sha256": b.core.original.sha(OUT / "plan.json"),
        "case_file_sha256": b.core.original.sha(OUT / f"{label}_cases.json"),
        "ordered_case_ids": [case["case_id"] for case in cases],
        "dependency_manifest_sha256": digest(PLAN["inputs"]),
        "parameters": normalized,
        "versions": PLAN["versions"],
    }


def validate_baseline(rows, cases):
    reject_nonfinite_tree(rows, "baseline")
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError("Baseline count differs from case count")
    for index, (row, case) in enumerate(zip(rows, cases, strict=True)):
        source_count = require_count(row.get("source_count"), f"baseline[{index}].source_count")
        if source_count != len(case["sources"]):
            raise ValueError("Baseline source count differs from frozen case")
        if not require_bool(row.get("all_checks_pass"), f"baseline[{index}].all_checks_pass"):
            raise ValueError("Baseline hard checks failed")
        time_s = require_finite(row.get("time_s"), f"baseline[{index}].time_s")
        mean_s = require_finite(row.get("mean_s"), f"baseline[{index}].mean_s")
        if time_s <= 0:
            raise ValueError("Baseline time must be positive")
        if abs(mean_s - time_s / source_count) > 1e-9:
            raise ValueError("Baseline time denominator invalid")


def validate_row(row, parameters, cases, base):
    reject_nonfinite_tree(row, "candidate")
    validate_baseline(base, cases)
    expected_parameters = binding(cases, "training" if cases == frozen_cases("training") else "validation", parameters)["parameters"]
    recorded_parameters = row.get("parameters")
    if not isinstance(recorded_parameters, list) or len(recorded_parameters) != 3:
        raise ValueError("Cached parameter vector has invalid shape")
    recorded_parameters = [
        require_finite(value, f"candidate.parameters[{index}]")
        for index, value in enumerate(recorded_parameters)
    ]
    if recorded_parameters != expected_parameters:
        raise ValueError("Cached parameters mismatch")
    recorded_cases = row.get("cases")
    if not isinstance(recorded_cases, list) or len(recorded_cases) != len(cases):
        raise ValueError("Cached candidate case count mismatch")

    deltas = []
    ratios = []
    means = []
    for index, (recorded, case, reference) in enumerate(zip(recorded_cases, cases, base, strict=True)):
        if recorded.get("case_id") != case["case_id"]:
            raise ValueError("Candidate case identity/order mismatch")
        metrics = recorded.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Candidate metrics missing")
        source_count = require_count(metrics.get("source_count"), f"candidate.cases[{index}].source_count")
        if source_count != len(case["sources"]):
            raise ValueError("Candidate source count mismatch")
        candidate_time = require_finite(metrics.get("time_s"), f"candidate.cases[{index}].time_s")
        candidate_mean = require_finite(metrics.get("mean_s"), f"candidate.cases[{index}].mean_s")
        reference_time = require_finite(reference.get("time_s"), f"baseline[{index}].time_s")
        reference_mean = require_finite(reference.get("mean_s"), f"baseline[{index}].mean_s")
        if candidate_time <= 0 or reference_time <= 0:
            raise ValueError("Candidate and baseline times must be positive")
        if abs(candidate_mean - candidate_time / source_count) > 1e-9:
            raise ValueError("Candidate time denominator invalid")

        delta = require_finite(recorded.get("delta_s_per_source"), f"candidate.cases[{index}].delta")
        ratio = require_finite(recorded.get("time_ratio"), f"candidate.cases[{index}].ratio")
        expected_delta = candidate_mean - reference_mean
        expected_ratio = candidate_time / reference_time
        if not math.isfinite(expected_delta) or not math.isfinite(expected_ratio):
            raise ValueError("Recomputed candidate comparison is non-finite")
        if abs(delta - expected_delta) > 1e-8:
            raise ValueError("Cached paired delta mismatch")
        if abs(ratio - expected_ratio) > 1e-9:
            raise ValueError("Cached ratio mismatch")

        errors = recorded.get("errors")
        if not isinstance(errors, list):
            raise ValueError("Candidate error list missing")
        metrics_pass = require_bool(metrics.get("all_checks_pass"), f"candidate.cases[{index}].metrics_pass")
        hard_pass = require_bool(recorded.get("hard_checks_pass"), f"candidate.cases[{index}].hard_pass")
        if hard_pass != (metrics_pass and not errors):
            raise ValueError("Hard check aggregation mismatch")
        action_sha = recorded.get("action_sha256")
        if not isinstance(action_sha, str) or len(action_sha) != 64:
            raise ValueError("Candidate action digest missing or malformed")
        deltas.append(delta)
        ratios.append(ratio)
        means.append(candidate_mean)

    expected_delta = statistics.mean(deltas)
    expected_mean = statistics.mean(means)
    expected_worst = max(ratios)
    stored_delta = require_finite(row.get("paired_mean_delta_s_per_source"), "candidate.paired_mean_delta")
    stored_mean = require_finite(row.get("mean_s_per_source"), "candidate.mean_s_per_source")
    stored_worst = require_finite(row.get("worst_case_ratio"), "candidate.worst_case_ratio")
    if abs(stored_delta - expected_delta) > 1e-8:
        raise ValueError("Cached objective mismatch")
    if abs(stored_mean - expected_mean) > 1e-8:
        raise ValueError("Cached mean_s_per_source mismatch")
    if abs(stored_worst - expected_worst) > 1e-9:
        raise ValueError("Cached worst_case_ratio mismatch")
    expected_feasible = all(item["hard_checks_pass"] for item in recorded_cases)
    if require_bool(row.get("all_checks_pass"), "candidate.all_checks_pass") != expected_feasible:
        raise ValueError("Cached feasibility mismatch")
    signature = hashlib.sha256(
        "|".join(item["action_sha256"] for item in recorded_cases).encode("utf-8")
    ).hexdigest()
    if row.get("behavior_signature") != signature:
        raise ValueError("Behavior signature mismatch")


def load_envelope(path, expected_binding, validator):
    cache = b.load(path)
    if cache.get("binding") != expected_binding:
        raise ValueError("Bound-cache identity changed: " + rel(path))
    data = cache.get("data")
    if cache.get("data_sha256") != digest(data):
        raise ValueError("Bound-cache data digest changed: " + rel(path))
    validator(data)
    return data


def save_envelope(path, expected_binding, data, provenance, validator):
    """Validate before saving; this has no legacy-cache dependency."""
    validator(data)
    b.save(
        path,
        {
            "binding": expected_binding,
            "data": data,
            "data_sha256": digest(data),
            "provenance": provenance,
        },
    )


def get_baseline(cases, label):
    expected = binding(cases, label)
    path = BOUND / f"{label}_baseline.json"
    validator = lambda rows: validate_baseline(rows, cases)
    if path.exists():
        return load_envelope(path, expected, validator), True, 0
    rows = []
    for case in cases:
        metrics, _, _ = b.run(b.core.baseline.run_search, case)
        rows.append(metrics)
    save_envelope(
        path,
        expected,
        rows,
        "Recalculated all baseline cases under strict-v2 binding; no unbound baseline was read.",
        validator,
    )
    return rows, False, len(cases)


def get_evaluation(parameters, cases, base, label):
    expected = binding(cases, label, parameters)
    path = BOUND / label / (b.key(parameters) + ".json")
    validator = lambda row: validate_row(row, parameters, cases, base)
    if path.exists():
        return load_envelope(path, expected, validator), True, 0, path
    row = b.core.evaluate(parameters, cases, base)
    row["behavior_signature"] = hashlib.sha256(
        "|".join(item["action_sha256"] for item in row["cases"]).encode("utf-8")
    ).hexdigest()
    row["created_at"] = b.core.original.now()
    save_envelope(
        path,
        expected,
        row,
        {"kind": "New strict-v2 evaluation", "legacy_cache_read": False},
        validator,
    )
    return row, False, len(cases), path


def study_results():
    expected = {(method, seed) for method in b.METHODS for seed in b.SEEDS}
    results = []
    seen = set()
    for method in b.METHODS:
        for seed in b.SEEDS:
            path = OUT / "studies" / f"{method}_{seed}" / "result.json"
            if not path.exists():
                raise ValueError("Missing completed study: " + rel(path))
            study = b.load(path)
            reject_nonfinite_tree(study, "study")
            identity = (study.get("method"), study.get("seed"))
            if identity != (method, seed) or identity in seen:
                raise ValueError("Study identity mismatch or duplicate")
            seen.add(identity)
            history = study.get("history")
            if study.get("proposals") != 48 or not isinstance(history, list) or len(history) != 48:
                raise ValueError("Completed study budget mismatch")
            for proposal, item in enumerate(history, 1):
                if item.get("proposal") != proposal:
                    raise ValueError("Study proposal ordering mismatch")
                require_finite(item.get("objective"), "study objective")
                parameters = item.get("parameters")
                if not isinstance(parameters, list) or len(parameters) != 3:
                    raise ValueError("Study parameter vector malformed")
                for index, value in enumerate(parameters):
                    require_finite(value, f"study parameters[{index}]")
                cache_path = BOUND / "training" / (b.key(parameters) + ".json")
                if not cache_path.exists():
                    raise ValueError("Missing strict training cache for completed study")
                cases = frozen_cases("training")
                base, _, _ = get_baseline(cases, "training")
                row = load_envelope(
                    cache_path,
                    binding(cases, "training", parameters),
                    lambda value, p=parameters: validate_row(value, p, cases, base),
                )
                expected_objective = b.objective(row)
                if not math.isfinite(expected_objective) or abs(item["objective"] - expected_objective) > 1e-8:
                    raise ValueError("Study history score differs from strict cache")
            best = min(history, key=lambda item: (item["objective"], item["proposal"]))
            if study.get("best") != best:
                raise ValueError("Study best row mismatch")
            results.append((path, study))
    if seen != expected:
        raise ValueError("Nine-study matrix is incomplete")
    return results


def cache_manifest_and_validate():
    entries = []
    counts = {}
    for label in ("training", "validation"):
        cases = frozen_cases(label)
        baseline_path = BOUND / f"{label}_baseline.json"
        if baseline_path.exists():
            load_envelope(
                baseline_path,
                binding(cases, label),
                lambda rows: validate_baseline(rows, cases),
            )
            entries.append(file_ref(baseline_path))
        candidate_paths = sorted((BOUND / label).glob("*.json"))
        base = None
        if candidate_paths:
            if not baseline_path.exists():
                raise ValueError("Candidate bound caches exist without their bound baseline")
            base, _, _ = get_baseline(cases, label)
        for path in candidate_paths:
            envelope = b.load(path)
            parameters = envelope.get("binding", {}).get("parameters")
            if not isinstance(parameters, list) or b.key(parameters) != path.stem:
                raise ValueError("Bound-cache filename and parameter key differ")
            load_envelope(
                path,
                binding(cases, label, parameters),
                lambda row, p=parameters: validate_row(row, p, cases, base),
            )
            entries.append(file_ref(path))
        counts[label + "_candidate_envelopes"] = len(candidate_paths)
        counts[label + "_baseline_envelopes"] = int(baseline_path.exists())
    return entries, counts


def run_selftests():
    cases = frozen_cases("training")
    base, _, _ = get_baseline(cases, "training")
    first_path = next(iter(sorted((BOUND / "training").glob("*.json"))), None)
    if first_path is None:
        raise ValueError("Self-tests need one already-bound training cache")
    envelope = b.load(first_path)
    parameters = envelope["binding"]["parameters"]
    valid = copy.deepcopy(envelope["data"])
    validate_row(valid, parameters, cases, base)

    controls = []
    for name, mutate in (
        ("NaN objective rejected", lambda row: row.__setitem__("paired_mean_delta_s_per_source", float("nan"))),
        ("infinite ratio rejected", lambda row: row["cases"][0].__setitem__("time_ratio", float("inf"))),
        ("forged mean aggregate rejected", lambda row: row.__setitem__("mean_s_per_source", -999.0)),
        ("forged worst ratio rejected", lambda row: row.__setitem__("worst_case_ratio", -999.0)),
    ):
        broken = copy.deepcopy(valid)
        mutate(broken)
        try:
            validate_row(broken, parameters, cases, base)
        except ValueError:
            controls.append(name)
        else:
            raise AssertionError(name + " negative control unexpectedly passed")

    tie_records = [
        {"method": "tpe", "seed": 2, "parameters": [100.0, 0.0, 1.25], "safe": True, "validation_delta": -3.0, "worst_case_ratio": 1.1},
        {"method": "gp_ei", "seed": 1, "parameters": [75.0, 0.0, 1.25], "safe": True, "validation_delta": -3.0 + 5e-10, "worst_case_ratio": 1.1},
        {"method": "differential_evolution", "seed": 3, "parameters": [75.0, 2.5, 1.0], "safe": True, "validation_delta": -3.0, "worst_case_ratio": 1.1},
    ]
    selected, evidence = conservative_select(tie_records)
    if selected["parameters"] != [75.0, 0.0, 1.25] or len(evidence["tied_records"]) != 3:
        raise AssertionError("Conservative tie rule self-test failed")
    controls.append("1e-9 conservative tie rule applied")

    with tempfile.TemporaryDirectory(dir=b.ROOT / "tmp") as temporary:
        path = Path(temporary) / "empty_bound_cache" / "candidate.json"
        expected = binding(cases, "training", parameters)
        save_envelope(
            path,
            expected,
            valid,
            {"kind": "In-memory empty-cache self-test", "legacy_cache_read": False},
            lambda row: validate_row(row, parameters, cases, base),
        )
        load_envelope(path, expected, lambda row: validate_row(row, parameters, cases, base))
    controls.append("empty bound cache initialized without legacy files")
    return controls


def audit(phase, run_id, selftest_controls):
    studies = study_results()
    manifest, counts = cache_manifest_and_validate()
    wrapper_path = Path(__file__)
    inputs = [
        file_ref(OUT / "plan.json"),
        file_ref(OUT / "training_cases.json"),
        file_ref(OUT / "validation_cases.json"),
        file_ref(wrapper_path),
        file_ref(OUT / "cache_binding_addendum_v2.json"),
    ]
    audit_data = {
        "scope": "STRICT_V2_COMPLETE_BOUND_CACHE_AUDIT",
        "phase": phase,
        "run_id": run_id,
        "passed": True,
        "at": b.core.original.now(),
        "inputs": inputs,
        "nine_completed_studies": [file_ref(path) for path, _ in studies],
        "cache_counts": counts,
        "cache_manifest": manifest,
        "cache_manifest_sha256": digest(manifest),
        "negative_controls": selftest_controls,
        "limits": [
            "This is deterministic consistency validation, not proof that synthetic cases match the GUI distribution.",
            "The 84 cases are selection validation and are not the final independent 20 GUI cases.",
        ],
    }
    path = OUT / "binding_audits_v2" / f"{run_id}_{phase}.json"
    b.save(path, audit_data)
    return path, audit_data


def tie_key(record):
    parameters = record["parameters"]
    return (
        require_finite(parameters[0], "MOVE_SLACK"),
        require_finite(parameters[1], "gain"),
        abs(require_finite(parameters[2], "offset") - 1.0),
        require_finite(parameters[2], "offset"),
        b.METHODS.index(record["method"]),
        record["seed"],
    )


def conservative_select(records):
    feasible = [record for record in records if require_bool(record.get("safe"), "record.safe")]
    if not feasible:
        raise ValueError("No feasible validation record is available for selection")
    for record in feasible:
        require_finite(record.get("validation_delta"), "record.validation_delta")
        require_finite(record.get("worst_case_ratio"), "record.worst_case_ratio")
    best_value = min(record["validation_delta"] for record in feasible)
    tied = [record for record in feasible if record["validation_delta"] - best_value <= TIE_TOLERANCE]
    ordered = sorted(tied, key=tie_key)
    selected = ordered[0]
    evidence = {
        "feasible_records": len(feasible),
        "best_validation_delta_s_per_source": best_value,
        "tolerance_s_per_source": TIE_TOLERANCE,
        "rule": [
            "retain feasible records within 1e-9 s/source of the minimum validation objective",
            "prefer smaller MOVE_SLACK",
            "then prefer smaller gain threshold",
            "then prefer offset scale closer to 1",
        ],
        "deterministic_representation_only_fallback": [
            "smaller numeric offset if distance to 1 remains tied",
            "frozen method order",
            "seed",
        ],
        "tied_records": [
            {
                "method": record["method"],
                "seed": record["seed"],
                "parameters": record["parameters"],
                "validation_delta": record["validation_delta"],
                "tie_key": list(tie_key(record)),
            }
            for record in ordered
        ],
        "selected": {
            "method": selected["method"],
            "seed": selected["seed"],
            "parameters": selected["parameters"],
            "validation_delta": selected["validation_delta"],
        },
        "unique_statistical_optimum_claimed": False,
    }
    return selected, evidence


def create_addendum():
    addendum = {
        "scope": "CACHE_BINDING_CORRECTION_AND_CONFIRMATION_ONLY",
        "original_plan_sha256": b.core.original.sha(OUT / "plan.json"),
        "strict_wrapper_sha256": b.core.original.sha(Path(__file__)),
        "no_training_rerun": True,
        "no_objective_sampler_parameter_grid_or_case_change": True,
        "legacy_cache_read_by_confirmation": False,
        "finite_rule": "Every numeric leaf and every scalar used by validation, optimization, ranking, or reporting must be a finite real value; baseline and candidate aggregates are recomputed.",
        "tie_rule": "For equal feasible validation objective within 1e-9 s/source, prefer smaller MOVE_SLACK, then smaller gain threshold, then offset scale closest to 1.",
        "representation_fallback": "If the declared rule still ties, smaller numeric offset, frozen method order, and seed select one deterministic representative without claiming a unique statistical optimum.",
    }
    path = OUT / "cache_binding_addendum_v2.json"
    if path.exists():
        if b.load(path) != addendum:
            raise ValueError("Strict-v2 cache binding addendum differs from current code or plan")
    else:
        b.save(path, addendum)
    return path


def confirm():
    b.validate()
    create_addendum()
    controls = run_selftests()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    pre_path, pre = audit("pre", run_id, controls)

    validation_cases = frozen_cases("validation")
    validation_base, baseline_hit, baseline_runs = get_baseline(validation_cases, "validation")
    records = []
    cache_requests = 0
    cache_hits = 0
    candidate_runs = 0
    cache_paths = {}
    first_record_by_key = {}
    for study_path, study in study_results():
        parameters = study["best"]["parameters"]
        row, hit, physical_runs, cache_path = get_evaluation(
            parameters, validation_cases, validation_base, "validation"
        )
        cache_requests += 1
        cache_hits += int(hit)
        candidate_runs += physical_runs
        configuration_key = b.key(parameters)
        duplicate_of = first_record_by_key.get(configuration_key)
        record_id = f"{study['method']}:{study['seed']}"
        first_record_by_key.setdefault(configuration_key, record_id)
        cache_paths[configuration_key] = cache_path
        record = {
            "record_id": record_id,
            "method": study["method"],
            "seed": study["seed"],
            "study_result": file_ref(study_path),
            "parameters": row["parameters"],
            "configuration_key": configuration_key,
            "configuration_duplicate_of": duplicate_of,
            "training_delta": study["best"]["objective"],
            "validation_delta": row["paired_mean_delta_s_per_source"],
            "validation_mean": row["mean_s_per_source"],
            "safe": row["all_checks_pass"],
            "worst_case_ratio": row["worst_case_ratio"],
            "validation_behavior_signature": row["behavior_signature"],
            "validation_cache_hit_for_this_request": hit,
            "validation_cache": file_ref(cache_path),
        }
        records.append(record)
        print(json.dumps({"confirmation": record}, ensure_ascii=False), flush=True)

    selected, tie_evidence = conservative_select(records)
    algorithms = {}
    for method in b.METHODS:
        part = [record for record in records if record["method"] == method]
        values = [record["validation_delta"] for record in part]
        algorithms[method] = {
            "seeds": len(part),
            "all_safe": all(record["safe"] for record in part),
            "mean_validation_delta": statistics.mean(values),
            "seed_sd": statistics.stdev(values),
            "worst_seed_delta": max(values),
            "unique_parameter_configurations": len({record["configuration_key"] for record in part}),
            "unique_validation_behaviors": len({record["validation_behavior_signature"] for record in part}),
        }

    behavior_groups = []
    signatures = sorted({record["validation_behavior_signature"] for record in records})
    for signature in signatures:
        members = [record for record in records if record["validation_behavior_signature"] == signature]
        representative, evidence = conservative_select(members)
        behavior_groups.append(
            {
                "validation_behavior_signature": signature,
                "member_record_ids": [record["record_id"] for record in members],
                "unique_parameters": sorted({tuple(record["parameters"]) for record in members}),
                "representative_record_id": representative["record_id"],
                "selection_evidence": evidence,
            }
        )
    for group in behavior_groups:
        group["unique_parameters"] = [list(values) for values in group["unique_parameters"]]

    post_path, post = audit("post", run_id, controls)
    bound_inputs = [
        file_ref(OUT / "plan.json"),
        file_ref(Path(__file__)),
        file_ref(OUT / "cache_binding_addendum_v2.json"),
        file_ref(OUT / "training_cases.json"),
        file_ref(OUT / "validation_cases.json"),
        file_ref(BOUND / "validation_baseline.json"),
        file_ref(pre_path),
        file_ref(post_path),
    ]
    bound_inputs.extend(file_ref(path) for _, path in sorted(cache_paths.items()))
    bound_inputs.extend(record["study_result"] for record in records)

    comparison = {
        "scope": "OFFLINE_OPTIMIZER_BENCHMARK_AND_PARAMETER_SELECTION_STRICT_V2",
        "created_at": b.core.original.now(),
        "run_id": run_id,
        "inputs": bound_inputs,
        "binding_digest_sha256": digest(bound_inputs),
        "algorithms": algorithms,
        "winner_records": records,
        "behavior_groups": behavior_groups,
        "selection": tie_evidence,
        "selected_record_id": selected["record_id"],
        "selected_parameters": selected["parameters"],
        "selection_validation_baseline_mean_s_per_source": statistics.mean(
            require_finite(row["mean_s"], "validation baseline mean") for row in validation_base
        ),
        "execution_accounting": {
            "training_proposals_reused_not_rerun": 432,
            "winner_records_requested": len(records),
            "unique_winner_parameter_configurations": len(cache_paths),
            "validation_candidate_cache_requests": cache_requests,
            "validation_candidate_cache_hits_this_action": cache_hits,
            "validation_candidate_cache_misses_this_action": cache_requests - cache_hits,
            "new_candidate_policy_simulations_this_action": candidate_runs,
            "validation_baseline_cache_hit_this_action": baseline_hit,
            "new_baseline_policy_simulations_this_action": baseline_runs,
            "total_new_policy_simulations_this_action": candidate_runs + baseline_runs,
            "selection_evidence_candidate_case_rows": len(cache_paths) * len(validation_cases),
            "selection_evidence_baseline_case_rows": len(validation_cases),
            "note": "One policy simulation means one policy-case execution. Duplicate winner parameters reuse one bound 84-case cache while still remaining nine algorithm-seed records.",
        },
        "audit_freshness": {
            "pre_audit": file_ref(pre_path),
            "post_audit": file_ref(post_path),
            "post_cache_manifest_sha256": post["cache_manifest_sha256"],
            "post_cache_counts": post["cache_counts"],
        },
        "claims": {
            "gui_claims": False,
            "final_20_gui_validation_completed": False,
            "selection_validation_only": True,
            "unique_statistical_optimum_claimed": False,
            "paper_frozen_numbers_changed": False,
        },
    }
    reject_nonfinite_tree(comparison, "comparison")
    path = OUT / "comparison_strict_v2.json"
    if path.exists():
        prior = OUT / "comparison_history_v2" / f"comparison_before_{run_id}.json"
        b.save(prior, b.load(path))
    b.save(path, comparison)
    print(
        json.dumps(
            {
                "comparison": rel(path),
                "selected_record": comparison["selected_record_id"],
                "selected_parameters": comparison["selected_parameters"],
                "new_policy_simulations": comparison["execution_accounting"]["total_new_policy_simulations_this_action"],
                "candidate_cache_hits": cache_hits,
                "post_audit": rel(post_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return comparison


def check_only():
    b.validate()
    create_addendum()
    controls = run_selftests()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path, data = audit("check", run_id, controls)
    print(
        json.dumps(
            {
                "audit": rel(path),
                "cache_counts": data["cache_counts"],
                "negative_controls": controls,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "confirm"])
    args = parser.parse_args()
    if args.action == "check":
        check_only()
    else:
        confirm()
