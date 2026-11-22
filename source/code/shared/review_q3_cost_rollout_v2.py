"""Regression checks for the state-alignment repair, not a performance screen."""
import ast
import json
import math
from pathlib import Path

import q3_cost_rollout as v1
import q3_cost_rollout_v2 as v2
import search_policy_q3_cost_rollout_v2 as policy_v2
from evaluate_q3_robust_parallax_development import run, save, sha256

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'results/q3_cost_rollout/development'
OUT = ROOT/'results/q3_cost_rollout/repair_v2'
fixture = json.loads((OLD/'repeated_center_fixture.json').read_text(encoding='utf-8'))
state = fixture['state']
arguments = (state['first_target'], state['position'], state['polygon'], state['public_cloud'],
             state['public_history'], state['initial_point'], state['initial_svd_deg'], state['extra_used'],
             state['hypothesis'], state['bias_deg'])
old_cost = v1.predicted_completion(*arguments)
new_cost = v2.predicted_completion(*arguments, previous_center=state['previous_center'])
# Independently sum the straight segments and clear fees for the immediate
# fallback branch. This uses no candidate predictor or fallback point helper.
theta = math.radians(state['initial_svd_deg'])
axis = (math.cos(theta), math.sin(theta))
normal = (-axis[1], axis[0])
position = tuple(state['position'])
movement = 0.0
clears = 0
reference_cost = None
for row, lateral in enumerate((-30.0, -10.0, 10.0, 30.0)):
    distances = list(range(0, 1501, 20))
    if row % 2:
        distances.reverse()
    for distance in distances:
        point = (state['initial_point'][0]+distance*axis[0]+lateral*normal[0],
                 state['initial_point'][1]+distance*axis[1]+lateral*normal[1])
        movement += math.dist(position, point)
        position = point
        clears += 1
        if math.dist(position, state['hypothesis']['point']) <= 20.0:
            reference_cost = movement/5.0+3.0*clears+2.0
            break
    if reference_cost is not None:
        break
assert reference_cost is not None
assert abs(old_cost-947.1199520383616) < 1e-8
assert abs(new_cost-reference_cost) < 1e-8
assert abs(reference_cost-939.1199520383616) < 1e-8 and clears == 113
decision = v2.choose_point(state['polygon'], state['original_center'], state['position'],
    state['public_cloud'], state['public_history'], state['initial_point'], state['initial_svd_deg'],
    state['extra_used'], state['previous_center'])
assert abs(decision['original_score_s']-reference_cost) < 1e-8

source_v1 = (ROOT/'code/shared/q3_cost_rollout.py').read_text(encoding='utf-8')
source_v2 = (ROOT/'code/shared/q3_cost_rollout_v2.py').read_text(encoding='utf-8')
assert all(getattr(v1, name) == getattr(v2, name) for name in
           ('OFFSETS', 'BACKOFFS', 'MOVE_SLACK', 'RECEPTION_LIMIT', 'MIN_PREDICTED_SAVING', 'MAX_HYPOTHESES', 'BIASES'))
for text in (source_v1, source_v2, (ROOT/'code/shared/search_policy_q3_cost_rollout_v2.py').read_text(encoding='utf-8')):
    ast.parse(text)

near = v2.predicted_completion((780.0, 0.0), (0.0, 0.0), state['polygon'], state['public_cloud'],
    [], (0.0, 0.0), 0.0, 7, state['hypothesis'], 1.0, previous_center=(750.0, 0.0))
assert abs(near-(780/5+5+5)) < 1e-8

cases = json.loads((OLD/'cases.json').read_text(encoding='utf-8'))
rows = []
for case_id in ('new_q3_61021', 'synthetic_q3_01'):
    case = next(c for c in cases if c['case_id'] == case_id)
    original = json.loads((OLD/'states'/f'{case_id}__candidate.json').read_text(encoding='utf-8'))
    metrics, run_state, actions = run(policy_v2.run_search, case)
    keys = ('time_s', 'mean_s', 'movement_s', 'measure_s', 'switch_s', 'failed_clear_s',
            'successful_clear_s', 'measure_count', 'clear_attempt_count', 'public_action_fingerprint')
    matches = all(metrics[key] == original['metrics'][key] for key in keys)
    assert matches and metrics['all_checks_pass'], case_id
    rows.append({'case_id': case_id, 'full_clear': metrics['full_source_clear'],
                 'actions_and_costs_match_v1': matches})
    save(OUT/'states'/f'{case_id}.json', {'metrics': metrics, 'state': run_state, 'actions': actions})

unchanged_v1 = all(sha256(ROOT/item['path']) == item['sha256'] for item in
                   json.loads((OLD/'inputs_before_run.json').read_text(encoding='utf-8')))
assert unchanged_v1
paths = ['code/shared/q3_cost_rollout_v2.py', 'code/shared/search_policy_q3_cost_rollout_v2.py',
         'code/shared/build_q3_cost_rollout_v2.py', 'code/shared/review_q3_cost_rollout_v2.py',
         'planning/q3_cost_rollout_v2_fix.md', 'results/q3_cost_rollout/development/repeated_center_fixture.json',
         'results/q3_cost_rollout/development/independent_review.json']
result = {'repair_regression_pass': True, 'scope': 'Q3CR-001_FIX_AND_TWO_EXISTING_CASES_ONLY',
          'fixed_finding': 'Q3CR-001', 'repeated_center': {'v1_s': old_cost, 'v2_s': new_cost,
                'independent_immediate_fallback_s': reference_cost, 'v2_score_for_original_center_s': decision['original_score_s'],
                'unexpected_extra_8s_removed': abs(old_cost-new_cost-8.0) < 1e-8},
          'parameters_unchanged': True, 'v1_inputs_and_evidence_unchanged': unchanged_v1,
          'last_allowed_measurement_cost_s': near, 'existing_extreme_case_regressions': rows,
          'fixture_reachability_note': 'Fixture obeys the public geometric response model but includes off-grid cloud coordinates; reachability from the exact current 20m initialization grid has not been proved. The repair aligns the interfaces independently of that reachability claim.',
          'new_full_development_screen_run': False, 'independent_holdout_run': False,
          'online_or_formal_runs': 0, 'method_accepted': False,
          'inputs': [{'path': p, 'sha256': sha256(ROOT/p)} for p in paths]}
save(OUT/'review.json', result)
print(json.dumps(result, ensure_ascii=False, indent=2))
