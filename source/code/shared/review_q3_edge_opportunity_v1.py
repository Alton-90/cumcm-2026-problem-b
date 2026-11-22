"""Evidence-bound Python review for the frozen Q3 edge-opportunity experiment."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q3_edge_opportunity_v1/development"
REVIEW = ROOT / "code/Q3/reviews/q3_edge_opportunity_v1_python_review.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(path):
    return {"path": path, "sha256": sha256(ROOT / path)}


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8"
    )


def main():
    comparison_path = OUT / "comparison.json"
    counterfactual_path = OUT / "counterfactual_audit.json"
    if not comparison_path.is_file() or not counterfactual_path.is_file():
        raise RuntimeError("completed comparison and counterfactual audit are required")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    counterfactual = json.loads(counterfactual_path.read_text(encoding="utf-8"))
    source_paths = [
        "code/shared/q3_edge_opportunity_v1.py",
        "code/shared/search_policy_q3_edge_opportunity_v1.py",
        "code/shared/generate_q3_edge_opportunity_v1_cases.py",
        "code/shared/evaluate_q3_edge_opportunity_v1.py",
        "code/shared/review_q3_edge_opportunity_v1.py",
    ]
    syntax_ok = True
    syntax_errors = []
    for relative in source_paths:
        try:
            ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        except SyntaxError as error:
            syntax_ok = False
            syntax_errors.append(f"{relative}: {error}")
    summaries = comparison["summaries"]
    candidate_events = summaries["candidate"]
    static_audit = comparison["static_policy_audit"]
    hard_ok = bool(comparison["all_hard_checks_pass"])
    gate_pass = bool(comparison["development_gate_pass"])
    action_semantics_ok = bool(
        hard_ok
        and counterfactual.get("all_checks_pass")
        and candidate_events.get("total_inserted", 0)
        == candidate_events.get("total_hits", -1)
    )
    checks = {
        "syntax": {
            "status": "PASS" if syntax_ok else "FAIL",
            "reason": (
                "All five frozen Python experiment files parse successfully."
                if syntax_ok else "; ".join(syntax_errors)
            ),
            "evidence": [evidence(path) for path in source_paths],
        },
        "input_contract": {
            "status": "PASS" if hard_ok else "FAIL",
            "reason": (
                "The freeze manifest stayed byte-stable; the metadata-only collision scan passed; exactly 42 fixed seeds were evaluated in three paired arms with no exclusions."
                if hard_ok
                else "One or more frozen-input, pairing, completion, or accounting checks failed; see comparison.json."
            ),
            "evidence": [
                evidence("planning/q3_edge_opportunity_v1_experiment.md"),
                evidence("results/q3_edge_opportunity_v1/development/freeze_manifest.json"),
                evidence("results/q3_edge_opportunity_v1/development/collision_audit.json"),
                evidence("results/q3_edge_opportunity_v1/development/cases.json"),
            ],
        },
        "method_alignment": {
            "status": "PASS" if static_audit["all_checks_pass"] else "FAIL",
            "reason": "The candidate retains the seven-point combined route, enters the probe only after the current-pose source branch has failed, and keeps the unchanged localizer/fallback. It remains an exploratory candidate, never the frozen main policy.",
            "evidence": [
                evidence("methods/Q3/q3_method_card.json"),
                evidence("code/Q3/q3_edge_opportunity_v1_code_plan.md"),
                evidence("code/shared/search_policy_q3_combined.py"),
                evidence("code/shared/search_policy_q3_edge_opportunity_v1.py"),
            ],
        },
        "action_semantics": {
            "status": "PASS" if action_semantics_ok else "FAIL",
            "reason": (
                f"All {candidate_events.get('total_inserted', 0)} inserted measures hit; event replay checked the true baseline next action, public feasible-set <=999.9m certificate, triangle equality, 25/30m break-even, and strict reorder bound."
                if action_semantics_ok
                else "At least one opportunity-action, guarantee, first-divergence, or reorder-bound audit failed."
            ),
            "evidence": [
                evidence("results/q3_edge_opportunity_v1/development/comparison.json"),
                evidence("results/q3_edge_opportunity_v1/development/counterfactual_audit.json"),
            ],
        },
        "output_contract": {
            "status": "PASS" if hard_ok else "FAIL",
            "reason": "Distance per source is primary; seconds per source and movement/measure/switch/successful-clear/failed-clear components reconcile. All predeclared AND gates and the non-promotion decision are explicit.",
            "evidence": [
                evidence("results/q3_edge_opportunity_v1/development/comparison.json"),
                evidence("results/q3_edge_opportunity_v1/development/run_summary.json"),
                evidence("results/q3_edge_opportunity_v1/development/decision.json"),
            ],
        },
        "information_boundary": {
            "status": "PASS" if static_audit["all_checks_pass"] else "FAIL",
            "reason": "Policy/scorer ASTs import neither OfflineArena nor SimulatorClient/transport, access no source truth, and report zero strict84, GUI, formal, or live executions. Evaluator-owned source truth is used only to score full clear.",
            "evidence": [
                evidence("code/shared/q3_edge_opportunity_v1.py"),
                evidence("code/shared/search_policy_q3_edge_opportunity_v1.py"),
                evidence("results/q3_edge_opportunity_v1/development/comparison.json"),
            ],
        },
        "counterfactual_scope": {
            "status": "PASS" if counterfactual.get("all_checks_pass") else "FAIL",
            "reason": "The cache-only arm isolates measurement/cache effects from the strict-bound reorder increment. Saved actual meter differences are labelled whole-case contrasts; no per-event causal attribution is claimed.",
            "evidence": [
                evidence("results/q3_edge_opportunity_v1/development/counterfactual_audit.json")
            ],
        },
    }
    review_pass = all(item["status"] == "PASS" for item in checks.values())
    findings = []
    if not gate_pass:
        findings.append(
            {
                "id": "Q3-EDGE1-PERF-001",
                "severity": "BLOCKING_FOR_PROMOTION",
                "status": "OPEN_BY_DESIGN",
                "summary": (
                    "The one-shot candidate failed at least one predeclared performance or hard-audit gate: "
                    + json.dumps(comparison["gates"], ensure_ascii=False)
                ),
                "disposition": "Reject and stop without tuning, seed changes, extra cases, strict84 reuse, GUI, or simulator execution; retain q3_combined_v1.",
            }
        )
    else:
        findings.append(
            {
                "id": "Q3-EDGE1-SCOPE-001",
                "severity": "LIMITATION",
                "status": "RECORDED",
                "summary": "The prospective 42-case screen is synthetic development evidence, not independent confirmation.",
                "disposition": "Keep the frozen main unchanged and validate only on a separately generated untouched holdout after human approval.",
            }
        )
    review = {
        "schema_version": 2,
        "artifact_type": "review",
        "question_id": "Q3.edge_opportunity_v1",
        "language": "python",
        "checks": checks,
        "findings": findings,
        "verdict": "PASSED" if review_pass else "FAILED",
        "inputs": [
            evidence("code/Q3/q3_code_plan.md"),
            evidence("methods/Q3/q3_method_card.json"),
            evidence("methods/Q3/probes/risk_probe_summary.json"),
            evidence("planning/data/data_profile.json"),
            evidence("planning/q3_edge_opportunity_v1_experiment.md"),
            evidence("planning/q3_edge_opportunity_v1_predeclaration.json"),
            *[evidence(path) for path in source_paths],
            evidence("results/q3_edge_opportunity_v1/development/comparison.json"),
            evidence("results/q3_edge_opportunity_v1/development/counterfactual_audit.json"),
            evidence("results/q3_edge_opportunity_v1/development/run_summary.json"),
        ],
        "extensions": {
            "review_scope": "one-shot offline implementation and evidence integrity; not method promotion",
            "development_gate_pass": gate_pass,
            "independent_validation": False,
            "simulator_or_gui_runs": 0,
            "strict84_runs": 0,
            "canonical_existing_q3_review_replaced": False,
        },
    }
    save(REVIEW, review)
    save(OUT / "review.json", review)
    postmortem = [
        "# Q3 edge-opportunity v1 — one-shot postmortem",
        "",
        f"Decision: **{comparison['decision']}**.",
        "",
        f"- Hard audits: `{comparison['all_hard_checks_pass']}`.",
        f"- Mean distance/source reduction: `{comparison['mean_distance_per_source_reduction_fraction']:.6%}` (gate >= 5%).",
        f"- Mean time/source reduction: `{comparison['mean_time_per_source_reduction_fraction']:.6%}` (gate >= 2%).",
        f"- Worst single-case time ratio: `{comparison['worst_case']['ratio']:.6f}` (gate <= 1.10).",
        f"- Candidate opportunities/insertions/hits/reorders: `{candidate_events.get('total_opportunities', 0)}` / `{candidate_events.get('total_inserted', 0)}` / `{candidate_events.get('total_hits', 0)}` / `{candidate_events.get('total_reorders', 0)}`.",
        "",
        "The cache-only and full arms preserve whole-case counterfactual differences. They do not identify an individual probe's causal contribution. The frozen combined controller remains the complete fallback and was not edited.",
        "",
        (
            "Because every preregistered gate passed, the only permitted next step is a genuinely untouched independent holdout after a new decision."
            if gate_pass
            else "Because at least one preregistered gate failed, this branch stops here: no constant changes, seed replacement, appended trial, strict84 reuse, GUI, live API, or formal test."
        ),
        "",
    ]
    (OUT / "postmortem.md").write_text("\n".join(postmortem), encoding="utf-8")
    print(json.dumps({"review": str(REVIEW), "verdict": review["verdict"], "development_gate_pass": gate_pass}, ensure_ascii=False))


if __name__ == "__main__":
    main()
