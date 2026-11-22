"""Small semantic checks before the fixed development comparison."""
import ast
import json
import math
from pathlib import Path
import numpy as np
import search_policy as old
from q3_cost_rollout import predicted_completion, compatible_hypotheses, longest_axis, choose_point

ROOT = Path(__file__).resolve().parents[2]
SHARED = Path(__file__).resolve().parent
for name in ('q3_cost_rollout.py', 'search_policy_q3_cost_rollout.py', 'evaluate_q3_cost_rollout_development.py'):
    ast.parse((SHARED/name).read_text(encoding='utf-8'))
baseline = (SHARED/'search_policy_q3_combined.py').read_text(encoding='utf-8')
candidate = (SHARED/'search_policy_q3_cost_rollout.py').read_text(encoding='utf-8')
base_control = baseline[baseline.index('def run_search('):]
cand_control = candidate[candidate.index('def run_search('):].replace('client=PublicTracked(client)', 'client=ValueTracked(client)')
assert base_control == cand_control
polygon = old._bearing_triangle((0.0, 0.0), 0.0)
cloud = np.array([(float(x), 0.0) for x in range(500, 1501, 20)])
history = [{'action': 'measure', 'point': [0.0, 0.0], 'response': {'measure_result': 'direction', 'svd_deg': 0.0}}]
hypothesis = {'point': [1000.0, 0.0], 'range_m': 1000.0}
near = predicted_completion((1000.0, 0.0), (0.0, 0.0), polygon, cloud, history,
                            (0.0, 0.0), 0.0, 7, hypothesis, 1.0)
assert abs(near-210.0) <= 1e-9
hypos = compatible_hypotheses(cloud, polygon, history, longest_axis(polygon))
assert len(hypos) == 9 and all(item['range_m'] >= math.dist(item['point'], (0, 0)) for item in hypos)
contradictory = history+[{'action': 'measure', 'point': [0.0, 0.0], 'response': {'measure_result': 'no_signal'}}]
assert not compatible_hypotheses(cloud, polygon, contradictory, longest_axis(polygon))
empty = choose_point(polygon, (750.0, 0.0), (0.0, 0.0), np.empty((0, 2)), history, (0, 0), 0, 0)
assert empty['selected_point'] == [750.0, 0.0] and not empty['used_alternative']
decision = choose_point(polygon, (750.0, 0.0), (0.0, 0.0), cloud, history, (0, 0), 0, 0)
assert decision == choose_point(polygon, (750.0, 0.0), (0.0, 0.0), cloud, history, (0, 0), 0, 0)
for record in decision['candidates'][1:]:
    if record['eligible']:
        assert record['max_vertex_distance_m'] <= 999.9 and record['movement_m'] <= 850.0 and record['fresh_point']
out = ROOT/'results/q3_cost_rollout/development/semantic_checks.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'all_checks_pass': True, 'checks': ['syntax', 'frozen_global_controller_identical',
    'near_at_last_allowed_measurement_cost_210s', 'range_consistent_with_positive_history',
    'inconsistent_history_rejected', 'empty_cloud_keeps_baseline', 'scoring_determinism',
    'candidate_reception_and_travel_filters'], 'example_decision': decision}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'checks': 'PASS', 'example_selected': decision['used_alternative'], 'output': str(out)}, ensure_ascii=False))
