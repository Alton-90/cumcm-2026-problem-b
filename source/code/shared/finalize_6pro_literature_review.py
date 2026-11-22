"""Bind the checked literature report and register a rejected development screen."""
import hashlib
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]

def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))

def digest(relative):
    return hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()

def write(relative, value):
    (ROOT/relative).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

analysis_path = 'planning/literature/related_paper_analysis.md'
review_path = 'results/q3_cost_rollout/development/independent_review.json'
assert (ROOT/review_path).is_file(), 'Independent review must exist before final binding'
comparison = read('results/q3_cost_rollout/development/comparison.json')
root_review = read('results/q3_cost_rollout/development/root_review.json')
assert not comparison['development_gate_pass'] and not comparison['method_accepted']
assert root_review['root_review_pass']
for item in read('results/q3_cost_rollout/development/inputs_before_run.json'):
    assert digest(item['path']) == item['sha256'], 'Experiment input changed: '+item['path']
manifest = read('planning/literature/6pro_20260911/source_manifest.json')
paths = [item['workspace_path'] for item in manifest]
paths += ['planning/literature/6pro_20260911/source_manifest.json',
          'planning/literature/6pro_20260911/independent_transfer_review.json',
          'planning/q3_cost_rollout_experiment.md',
          'results/q3_cost_rollout/development/comparison.json',
          'results/q3_cost_rollout/development/root_review.json',
          'results/q3_cost_rollout/development/semantic_checks.json', review_path,
          'results/q3_cost_rollout/development/repeated_center_fixture.json',
          'results/q3_cost_rollout/repair_v2/review.json',
          'planning/q3_cost_rollout_v2_fix.md',
          'code/shared/q3_cost_rollout_v2.py', 'code/shared/search_policy_q3_cost_rollout_v2.py',
          'results/q3_robust_parallax/development/comparison.json',
          'results/five_run_batches/q3_combined_v1.json',
          'results/five_run_batches/q4_joint_ring22_v2.json',
          'results/literature_transfer/q4_range_elimination_checks.json',
          'code/shared/verify_q4_range_elimination.py',
          'code/shared/q3_cost_rollout.py', 'code/shared/search_policy_q3_cost_rollout.py']
binding = {'schema_version': 2, 'artifact_type': 'report-binding', 'question_id': None,
           'report_kind': 'literature', 'report': {'path': analysis_path, 'sha256': digest(analysis_path)},
           'inputs': [{'path': p, 'sha256': digest(p)} for p in paths],
           'extensions': {'scope': 'user_supplied_literature_and_fixed_development_screen',
                          'method_accepted': False, 'paper_frozen_numbers_changed': False,
                          'new_Q4_range_elimination': 'local_derivation_only_not_a_benchmark_result'}}
write('planning/literature/related_paper_analysis.json', binding)
status_path = 'results/optimization_200_400/status.json'
status = read(status_path)
status['latest_literature_review'] = {'report': analysis_path,
    'binding': 'planning/literature/related_paper_analysis.json',
    'source_manifest': 'planning/literature/6pro_20260911/source_manifest.json',
    'Q3_cost_rollout': {'evidence': 'results/q3_cost_rollout/development/comparison.json',
       'independent_review': review_path, 'development_records': 36, 'all_source_instances_cleared': 468,
       'baseline_s_per_source': comparison['summaries']['baseline']['mean_s_per_source'],
       'candidate_s_per_source': comparison['summaries']['candidate']['mean_s_per_source'],
       'relative_improvement': comparison['relative_improvement'],
       'absolute_saving_s_per_source': comparison['absolute_saving_s_per_source'],
       'development_gate_pass': False, 'accepted': False,
       'reason': 'relative saving below predeclared 3 percent gate',
       'v1_independent_implementation_review': 'latent_state_alignment_gap_not_triggered_in_saved_runs',
       'v2_repair': {'evidence': 'results/q3_cost_rollout/repair_v2/review.json',
                     'regression_checked': True, 'new_full_performance_screen': False}},
    'Q4_range_elimination': {'state': 'derived_and_boundary_checked_not_a_search_policy',
        'evidence': 'results/literature_transfer/q4_range_elimination_checks.json'},
    'practice_runs_started': 0, 'formal_runs_started': 0}
status['parallel_Q4_joint_compatibility_task'] = {'thread_id': '<archived-conversation>',
    'scope': 'continuous_heading_feasibility_helper_and_boundary_checks', 'status_at_registration': 'running',
    'expected_outputs': ['code/shared/q4_joint_compatibility.py', 'results/q4_joint_compatibility/']}
status['updated_at'] = datetime.now().astimezone().isoformat()
write(status_path, status)
print(json.dumps({'report': analysis_path, 'binding': 'planning/literature/related_paper_analysis.json',
                  'registered_candidate_accepted': False, 'frozen_practice_numbers': status['current_practice_batch_means_s_per_source']}, ensure_ascii=False))
