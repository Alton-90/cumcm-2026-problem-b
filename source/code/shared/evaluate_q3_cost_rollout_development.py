"""Fixed paired development screen. Truth is used only after each policy run."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from time import perf_counter

import search_policy as old
import search_policy_q3_combined as baseline
import search_policy_q3_cost_rollout as candidate
from q3_cost_rollout import choose_point, compatible_hypotheses, longest_axis
from evaluate_q3_robust_parallax_development import run, save, sha256, contains_point, same_point

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/q3_cost_rollout/development'
CASE_FILE = ROOT/'results/feasible_q3/cases.json'


def audit(case, state, actions):
    sources = {s['channel']: (s['x'], s['y']) for s in case['sources']}
    measures = [a for a in actions if a['path'] == '/measure']
    errors = []
    selected = 0
    no_hypotheses = 0
    for index, event in enumerate(state.get('cost_rollout_events', [])):
        prefix = f'event {index}'
        if not 0 <= event['measure_index']-1 < len(measures):
            errors.append(prefix+': measurement index')
            continue
        action = measures[event['measure_index']-1]
        point = event['selected_point']
        poly = event['before_polygon']
        location = sources[event['channel']]
        if action['channel'] != event['channel'] or not same_point(point, action['point']) or action['response'] != event['response']:
            errors.append(prefix+': public action mismatch')
        if not contains_point(poly, location, tolerance=1e-5):
            errors.append(prefix+': source absent from prior polygon')
        if not 0 <= event['extra_used_before'] < 8:
            errors.append(prefix+': local budget violation')
        if action['response'].get('measure_result') == 'direction':
            after = old._add_bearing(poly, point, action['response']['svd_deg'])
            if not contains_point(after, location, tolerance=1e-5):
                errors.append(prefix+': source absent after measurement')
        if event['no_compatible_grid_hypotheses']:
            no_hypotheses += 1
        if not event['used_alternative']:
            if not same_point(point, event['original_center']):
                errors.append(prefix+': baseline anchor not retained')
            continue
        selected += 1
        if max(math.dist(point, p) for p in poly) > 999.9+1e-8:
            errors.append(prefix+': guaranteed reception violated')
        if action['response'].get('measure_result') == 'no_signal':
            errors.append(prefix+': guaranteed point lost reception')
        if math.dist(point, event['current_position']) > math.dist(event['original_center'], event['current_position'])+100+1e-8:
            errors.append(prefix+': movement cap violated')
        if event['original_score_s']-event['selected_score_s'] < 5-1e-8:
            errors.append(prefix+': predicted gain gate violated')
        # Replay just the decision using only its saved public information.
        replay = choose_point(poly, event['original_center'], event['current_position'],
                              event['public_cloud'], event['public_history'], event['initial_point'],
                              event['initial_svd_deg'], event['extra_used_before'])
        if not replay['used_alternative'] or not same_point(replay['selected_point'], point) or abs(replay['selected_score_s']-event['selected_score_s']) > 1e-8:
            errors.append(prefix+': public-state decision is not reproducible')
    return {'pass': not errors, 'errors': errors, 'selected_count': selected,
            'event_count': len(state.get('cost_rollout_events', [])), 'no_hypothesis_count': no_hypotheses}


def summarize(rows, name):
    values = [row[name] for row in rows]
    out = {'case_count': len(values), 'all_checks_pass': all(v['all_checks_pass'] for v in values),
           'mean_s_per_source': statistics.mean(v['mean_s'] for v in values),
           'total_policy_wall_s': sum(v['policy_wall_s'] for v in values),
           'max_policy_wall_s': max(v['policy_wall_s'] for v in values),
           'selected_count': sum(v.get('rollout_audit', {}).get('selected_count', 0) for v in values),
           'no_hypothesis_count': sum(v.get('rollout_audit', {}).get('no_hypothesis_count', 0) for v in values),
           'fallback_clear_count': sum(v['fallback_count'] for v in values)}
    for key in ('movement_s', 'measure_s', 'switch_s', 'successful_clear_s', 'failed_clear_s'):
        out[key+'_per_source'] = statistics.mean(v[key]/v['source_count'] for v in values)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'comparison.json').exists():
        raise RuntimeError('An existing completed experiment will not be overwritten')
    cases = json.loads(CASE_FILE.read_text(encoding='utf-8'))[:36]
    assert len(cases) == 36 and all(c['question'] == 3 and c['split'] == 'development' for c in cases)
    save(OUT/'cases.json', cases)
    paths = ['code/shared/'+name for name in ('search_policy.py', 'search_policy_compact.py',
             'search_policy_feasible_q3.py', 'search_policy_q3_combined.py', 'route_multistart.py',
             'q3_cost_rollout.py', 'search_policy_q3_cost_rollout.py', 'build_q3_cost_rollout_policy.py',
             'evaluate_q3_cost_rollout_development.py', 'evaluate_q3_robust_parallax_development.py',
             'offline_arena.py', 'simulator_client.py')]
    paths += ['planning/q3_cost_rollout_experiment.md', 'results/feasible_q3/cases.json',
              'planning/literature/6pro_20260911/B题_文献研读成果_交总控.md']
    inputs = [{'path': p, 'sha256': sha256(ROOT/p)} for p in paths]
    save(OUT/'inputs_before_run.json', inputs)
    rows = []
    started = perf_counter()
    for index, case in enumerate(cases):
        row = {'case_id': case['case_id'], 'seed': case['seed'], 'error_mode': case['error_mode']}
        for name, policy in [('baseline', baseline.run_search), ('candidate', candidate.run_search)]:
            metrics, state, actions = run(policy, case)
            if name == 'candidate':
                metrics['rollout_audit'] = audit(case, state, actions)
                metrics['all_checks_pass'] &= metrics['rollout_audit']['pass']
                metrics['rollout_scoring_cpu_s'] = state.get('cost_rollout_scoring_cpu_s', 0.0)
            save(OUT/'states'/f"{case['case_id']}__{name}.json",
                 {'case_id': case['case_id'], 'metrics': metrics, 'state': state, 'actions': actions})
            row[name] = metrics
        row['ratio'] = row['candidate']['time_s']/row['baseline']['time_s']
        row['delta_s_per_source'] = row['candidate']['mean_s']-row['baseline']['mean_s']
        rows.append(row)
        save(OUT/'progress.json', {'completed_pairs': len(rows), 'rows': rows})
        print(f"[{index+1:02}/36] {case['case_id']} base={row['baseline']['mean_s']:.5f} "
              f"candidate={row['candidate']['mean_s']:.5f} ratio={row['ratio']:.5f} "
              f"selected={row['candidate']['rollout_audit']['selected_count']} checks={row['candidate']['all_checks_pass']}", flush=True)
    summaries = {name: summarize(rows, name) for name in ('baseline', 'candidate')}
    saving = summaries['baseline']['mean_s_per_source']-summaries['candidate']['mean_s_per_source']
    relative = saving/summaries['baseline']['mean_s_per_source']
    worst = max(rows, key=lambda row: row['ratio'])
    unchanged = all(sha256(ROOT/item['path']) == item['sha256'] for item in inputs)
    checks = unchanged and all(v['all_checks_pass'] for v in summaries.values())
    gate = checks and saving >= 5.0 and relative >= 0.03 and worst['ratio'] <= 1.10
    result = {'scope': 'SYNTHETIC_FIXED_36_DEVELOPMENT_ONLY', 'candidate': candidate.VERSION,
              'case_count': len(rows), 'all_hard_checks_pass': checks, 'inputs_unchanged': unchanged,
              'development_gate_pass': gate, 'method_accepted': False,
              'predeclared_gate': {'relative_saving': 0.03, 'absolute_saving_s_per_source': 5.0, 'max_case_ratio': 1.10},
              'absolute_saving_s_per_source': saving, 'relative_improvement': relative,
              'worst_case': {'case_id': worst['case_id'], 'ratio': worst['ratio'], 'delta_s_per_source': worst['delta_s_per_source']},
              'summaries': summaries, 'rows': rows, 'inputs': inputs, 'suite_wall_s': perf_counter()-started,
              'next_step': 'independent_review_before_new_holdout' if gate else 'stop_without_tuning_or_simulator_run',
              'online_practice_runs': 0, 'formal_runs': 0,
              'population': {'audited': 36, 'eligible': 36, 'excluded': 0, 'baseline_used': 36, 'candidate_used': 36}}
    save(OUT/'comparison.json', result)
    print(json.dumps({k: result[k] for k in ('all_hard_checks_pass', 'development_gate_pass',
          'absolute_saving_s_per_source', 'relative_improvement', 'worst_case', 'summaries')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
