"""Run the declared synthetic cases; never contacts the official simulator."""
import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from offline_arena import OfflineArena, make_cases
from simulator_client import SimulatorClient

ROOT = Path(__file__).resolve().parents[2]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=list), encoding='utf-8')


def reference(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_question(question):
    cases = make_cases(question)
    dataset = [dict(case_id=c['case_id'], seed=c['seed'], error_mode=c['error_mode'],
                    sources=[asdict(s) for s in c['sources']]) for c in cases]
    dataset_path = ROOT / 'results' / 'diagnostics' / f'synthetic_cases_q{question}.json'
    save(dataset_path, dict(label='SYNTHETIC_NOT_OFFICIAL', cases=dataset))
    folder = ROOT / 'results' / f'Q{question}' / 'experiments' / 'round1'
    metrics, method_records = {}, []
    for method in ('baseline', 'main'):
        entry = ROOT / 'code' / f'Q{question}' / 'python' / f'q{question}_{method}.py'
        spec = importlib.util.spec_from_file_location(f'entry_q{question}_{method}', entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = []
        for case in cases:
            arena = OfflineArena(case['sources'], case['seed'], case['error_mode'])
            client = SimulatorClient('synthetic-team', transport=arena, log_path=False)
            started = time.perf_counter()
            policy_result, error = {}, None
            try:
                client.enter()
                policy_result = module.run(client)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            finally:
                if client.entered and not client.ended:
                    client.exit()
            elapsed = time.perf_counter()-started
            count = len(arena.cleared)
            expected_time = arena.distance_m/5 + 5*arena.measure_count + arena.switch_count + 3*arena.clear_count + 2*count
            violations = []
            if count != len(case['sources']):
                violations.append('sources_remain')
            if error:
                violations.append('execution_exception')
            if not policy_result.get('completed', False):
                violations.append('policy_not_completed')
            if policy_result.get('completed') and count != len(case['sources']):
                violations.append('false_completion')
            if abs(expected_time-arena.virtual_time_s) > 1e-6:
                violations.append('virtual_time_accounting_mismatch')
            if abs(client.virtual_time_s-arena.virtual_time_s) > 1e-6:
                violations.append('client_clock_mismatch')
            if arena.virtual_time_s > 360000:
                violations.append('virtual_time_limit')
            if elapsed > 1200:
                violations.append('offline_real_time_limit')
            reported_cleared = set(policy_result.get('cleared_channels', []))
            if reported_cleared != arena.cleared:
                violations.append('reported_clear_set_mismatch')
            excluded = set(policy_result.get('excluded_channels', []))
            if excluded & set(arena.sources):
                violations.append('existing_source_marked_absent')
            if policy_result.get('completed') and (reported_cleared | excluded != set(range(1,21))):
                violations.append('incomplete_channel_accounting')
            row = dict(case_id=case['case_id'], seed=case['seed'], error_mode=case['error_mode'],
                       source_count=len(case['sources']), cleared_count=count,
                       clearance_ratio=count/len(case['sources']), virtual_time_s=arena.virtual_time_s,
                       mean_time_per_cleared_s=arena.virtual_time_s/count if count else None,
                       real_execution_s=elapsed, measurement_actions=arena.measure_count,
                       clear_actions=arena.clear_count, movement_m=arena.distance_m,
                       switch_count=arena.switch_count, independently_accounted_time_s=expected_time,
                       violations=violations, error=error, policy_result=policy_result)
            rows.append(row)
        summary = dict(cases=len(rows), completed_cases=sum(not row['violations'] for row in rows),
                       source_count=sum(row['source_count'] for row in rows),
                       cleared_count=sum(row['cleared_count'] for row in rows),
                       mean_case_average_time_s=statistics.mean(row['mean_time_per_cleared_s'] for row in rows
                                                               if row['mean_time_per_cleared_s'] is not None),
                       pooled_time_per_cleared_s=(sum(row['virtual_time_s'] for row in rows)
                                                 / sum(row['cleared_count'] for row in rows))
                                                 if sum(row['cleared_count'] for row in rows) else None,
                       max_case_execution_s=max(row['real_execution_s'] for row in rows),
                       max_case_virtual_time_s=max(row['virtual_time_s'] for row in rows),
                       failures=[dict(case_id=row['case_id'], violations=row['violations'])
                                 for row in rows if row['violations']])
        metrics[method] = summary
        summary['output_fingerprint'] = hashlib.sha256(json.dumps(dict(
            method=method, question=question, entry=reference(entry),
            policy=reference(ROOT/'code/shared/search_policy.py'), dataset=reference(dataset_path),
            cases=[{k:v for k,v in row.items() if k != 'real_execution_s'} for row in rows]
        ), sort_keys=True, default=list).encode()).hexdigest()
        path = folder / 'metrics' / (method+'.json')
        save(path, dict(label='SYNTHETIC_NOT_OFFICIAL', method=method, summary=summary, cases=rows))
        method_records.append(dict(method_id='fixed_cover_bearing' if method=='main' else 'fixed_cover_sweep',
                                   role='main' if method=='main' else 'usable_baseline',
                                   status='success' if not summary['failures'] else 'failed', language='python',
                                   script=reference(entry),
                                   outputs=[reference(path)], metrics_summary=summary,
                                   constraint_violations=summary['failures'],
                                   execution_time_seconds=sum(row['real_execution_s'] for row in rows)))
        print(f'Q{question} {method}: {summary}', flush=True)
    comparison_path = folder / 'metrics' / 'offline_metrics.json'
    save(comparison_path, dict(label='SYNTHETIC_NOT_OFFICIAL', methods=metrics,
                              reduction_in_mean_case_average=(1-metrics['main']['mean_case_average_time_s']
                                                               /metrics['baseline']['mean_case_average_time_s'])))
    inputs=[reference(ROOT/p) for p in ['code/shared/search_policy.py','code/shared/offline_arena.py',
                                      'code/shared/simulator_client.py','code/shared/run_offline_validation.py',
                                      'planning/q3_q4_implementation_contract.md',
                                      'planning/parse/problem_parse.json','planning/data/data_profile.json',
                                      f'methods/Q{question}/q{question}_method_card.json',
                                      f'methods/Q{question}/probes/risk_probe_summary.json',
                                      f'methods/Q{question}/q{question}_decisions.jsonl']]
    inputs.append(reference(dataset_path))
    selected=dict(method_id='fixed_cover_bearing', parameters={'max_extra_measurements':8,
                  'error_half_angle_deg':1.005, 'clear_enclosure_threshold_m':19.9,
                  'question':question}, composition='fixed coverage + bearing shrink + bounded sweep',
                  evaluator='synthetic_black_box_v1', resolution={'scan_spacing_m':1200 if question==3 else 600,
                                                               'clear_lattice_spacing_m':20})
    fingerprint=hashlib.sha256(json.dumps(dict(inputs=inputs,selected=selected),sort_keys=True).encode()).hexdigest()
    summary=dict(schema_version=2,artifact_type='run-summary',question_id=f'Q{question}',round='round1',
                 approved_decision_id=f'q{question}-method-20260910',methods=method_records,
                 comparison=dict(population_id=reference(dataset_path)['sha256'],
                                 metric_contract='Same cases: full clearance first, then mean virtual time per cleared source; actual local runtime separately.'),
                 selected_output=selected,
                 output_obligations=[dict(id='synthetic_full_clearance',satisfied=all(not m['failures'] for m in metrics.values()),
                                          evidence=[reference(comparison_path)]),
                                     dict(id='official_practice_evaluation',satisfied=False,evidence=[]),
                                     dict(id='three_official_results_and_original_logs',satisfied=False,evidence=[])],
                 random_seed=question*10000,environment=dict(python=platform.python_version(),
                                                           platform=platform.platform(),scope='synthetic offline only'),
                 inputs=inputs,contract_evidence=dict(population_reconciliation={'generated':12,'used_per_method':12,'excluded':0},
                                                      input_population_contract={'status':'APPLICABLE','audited':12,'parsed':12,'excluded':0,
                                                         'eligible':12,'used_by_method':{'fixed_cover_bearing':12,'fixed_cover_sweep':12}},
                                                      continuous_coverage_reference='planning/search_coverage_notes.md',
                                                      error_field='fixed by exact coordinate, channel and seed; official mechanism unknown'),
                 extensions=dict(implementation_fingerprint=fingerprint,not_official=True,
                                 warnings=['Synthetic wall time excludes real HTTP/network overhead.',
                                           'Not final frozen results; official obligations remain outstanding.']))
    save(folder/'run_summary.json',summary)
    return all(not m['failures'] for m in metrics.values())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', type=int, choices=(3,4))
    args=parser.parse_args()
    outcomes=[run_question(q) for q in ([args.question] if args.question else [3,4])]
    raise SystemExit(0 if all(outcomes) else 1)
