"""Produce a minimal state-alignment repair without changing v1 evidence."""
from pathlib import Path

HERE = Path(__file__).resolve().parent

def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

helper = (HERE/'q3_cost_rollout.py').read_text(encoding='utf-8')
helper = once(helper, 'initial_svd, extra_used, hypothesis, bias):',
              'initial_svd, extra_used, hypothesis, bias, previous_center=None):')
helper = once(helper, '    previous_target = None\n', '    previous_target = previous_center\n')
helper = once(helper, 'def choose_point(polygon, center, position, cloud, history, initial_point, initial_svd, extra_used):',
              'def choose_point(polygon, center, position, cloud, history, initial_point, initial_svd, extra_used, previous_center=None):')
helper = once(helper, 'initial_point, initial_svd, extra_used, item, bias) for bias in BIASES)',
              'initial_point, initial_svd, extra_used, item, bias, previous_center) for bias in BIASES)')
(HERE/'q3_cost_rollout_v2.py').write_text(helper, encoding='utf-8')

policy = (HERE/'search_policy_q3_cost_rollout.py').read_text(encoding='utf-8')
policy = once(policy, 'from q3_cost_rollout import choose_point', 'from q3_cost_rollout_v2 import choose_point')
policy = once(policy, "VERSION = 'q3_cost_rollout_v1'", "VERSION = 'q3_cost_rollout_v2'")
policy = once(policy, 'proposal = choose_point(polygon, center, client.position, cloud, history, point, svd, extra_used)',
              'proposal = choose_point(polygon, center, client.position, cloud, history, point, svd, extra_used, previous_center)')
policy = once(policy, 'extra_used_before=extra_used, before_polygon=',
              'previous_center=None if previous_center is None else list(previous_center),\n                     extra_used_before=extra_used, before_polygon=')
(HERE/'search_policy_q3_cost_rollout_v2.py').write_text(policy, encoding='utf-8')
print('Created v2 repair; v1 files and parameters preserved.')
