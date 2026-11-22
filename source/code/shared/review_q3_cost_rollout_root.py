"""Reproduce extreme cases and report comparison-population structure."""
import json
import math
from pathlib import Path
import statistics
import search_policy_q3_combined as baseline
import search_policy_q3_cost_rollout as candidate
from evaluate_q3_robust_parallax_development import run, save, sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/q3_cost_rollout/development'
comparison = json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
cases = json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
mapping = {c['case_id']: c for c in cases}
best = min(comparison['rows'], key=lambda r: r['ratio'])
worst = max(comparison['rows'], key=lambda r: r['ratio'])
errors = []
reruns = []
for row in [best, worst]:
    for name, policy in [('baseline', baseline.run_search), ('candidate', candidate.run_search)]:
        metrics, state, actions = run(policy, mapping[row['case_id']])
        original = row[name]
        keys = ('mean_s', 'time_s', 'movement_s', 'measure_s', 'switch_s', 'successful_clear_s',
                'failed_clear_s', 'measure_count', 'clear_attempt_count', 'public_action_fingerprint',
                'completed', 'full_source_clear', 'channels_accounted', 'false_excluded_source')
        matches = all(metrics[key] == original[key] for key in keys)
        if not matches:
            errors.append(f"{row['case_id']} {name}: rerun mismatch")
        reruns.append({'case_id': row['case_id'], 'method': name,
                       'action_and_cost_metrics_match': matches, 'hard_checks_pass': metrics['all_checks_pass']})
        save(OUT/'root_reruns'/f"{row['case_id']}__{name}.json", {'metrics': metrics, 'state': state, 'actions': actions})

def key(items):
    return json.dumps(items, sort_keys=True, separators=(',', ':'))

geometry = {key(sorted((s['x'], s['y']) for s in case['sources'])) for case in cases}
physical = {key(sorted(case['sources'], key=lambda s: s['channel'])) for case in cases}
ratios = [row['ratio'] for row in comparison['rows']]
before = json.loads((OUT/'inputs_before_run.json').read_text(encoding='utf-8'))
unchanged = all(sha256(ROOT/item['path']) == item['sha256'] for item in before)
if not unchanged:
    errors.append('frozen input changed')
report = {'root_review_pass': not errors, 'errors': errors, 'extreme_case_reruns': reruns,
          'all_inputs_unchanged': unchanged, 'win_tie_loss': {'wins': sum(x < 1-1e-10 for x in ratios),
              'ties': sum(abs(x-1) <= 1e-10 for x in ratios), 'losses': sum(x > 1+1e-10 for x in ratios)},
          'population': {'paired_development_records': len(cases), 'unique_position_layouts': len(geometry),
                         'unique_physical_layouts_including_channels_and_ranges': len(physical),
                         'independent_holdout': False, 'all_source_instances': sum(len(c['sources']) for c in cases)},
          'case_ratio_median': statistics.median(ratios),
          'best_case': {k: best[k] for k in ('case_id', 'ratio', 'delta_s_per_source')},
          'worst_case': {k: worst[k] for k in ('case_id', 'ratio', 'delta_s_per_source')},
          'performance_gate_pass': comparison['development_gate_pass'], 'method_accepted': False,
          'note': 'Correctness and replay checks do not establish generalization or approve the failed performance gate.'}
save(OUT/'root_review.json', report)
print(json.dumps(report, ensure_ascii=False, indent=2))
