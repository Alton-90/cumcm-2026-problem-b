"""Create an isolated policy by retaining the frozen global controller verbatim."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
original = (HERE/'search_policy_q3_combined.py').read_text(encoding='utf-8')
local_source = (HERE/'search_policy_feasible_q3.py').read_text(encoding='utf-8')
localize = local_source[local_source.index('def localize('):local_source.index('\ndef run_search(')]
insertion = '''        scoring_start = perf_counter()
        cloud = client.clouds.get(channel)
        history = client.public_history.get(channel, [])
        proposal = choose_point(polygon, center, client.position, cloud, history, point, svd, extra_used)
        state['cost_rollout_scoring_cpu_s'] = state.get('cost_rollout_scoring_cpu_s', 0.0) + perf_counter()-scoring_start
        event = dict(proposal, channel=channel, initial_point=list(point), initial_svd_deg=svd,
                     extra_used_before=extra_used, before_polygon=[list(p) for p in polygon],
                     current_position=list(client.position), original_center=list(center),
                     public_cloud=cloud.tolist() if proposal['used_alternative'] else None,
                     public_history=list(history) if proposal['used_alternative'] else None,
                     measure_index=state['measure_count']+1)
        center = tuple(proposal['selected_point'])
'''
needle = '        if previous_center and math.hypot'
assert localize.count(needle) == 1
localize = localize.replace(needle, insertion+needle)
localize = localize.replace('        previous_center = center\n', "        state.setdefault('cost_rollout_events', []).append(event)\n        previous_center = center\n")
needle = '        kind = follow_up.get("measure_result")'
assert localize.count(needle) == 1
localize = localize.replace(needle, "        event['response'] = dict(follow_up)\n"+needle)
header = '''"""Experimental Q3 cost rollout; frozen coverage/routing controller is preserved."""
import math
from time import perf_counter
from search_policy_compact import _state
import search_policy as old
import search_policy_q3_combined as combined
from q3_cost_rollout import choose_point

VERSION = 'q3_cost_rollout_v1'
_route = combined._route
_valid = combined._valid


class PublicTracked(combined.ValueTracked):
    def __init__(self, client):
        super().__init__(client)
        self.public_history = {}

    def measure(self, x, y, channel):
        response = super().measure(x, y, channel)
        self.public_history.setdefault(channel, []).append(
            {'action': 'measure', 'point': [x, y], 'response': dict(response)})
        return response

    def clear(self, x, y, channel):
        response = super().clear(x, y, channel)
        self.public_history.setdefault(channel, []).append(
            {'action': 'clear', 'point': [x, y], 'response': dict(response)})
        return response


'''
controller = original[original.index('def run_search('):].replace('client=ValueTracked(client)', 'client=PublicTracked(client)')
(HERE/'search_policy_q3_cost_rollout.py').write_text(header+localize+'\n'+controller, encoding='utf-8')
print('Isolated candidate created; original policy files were only read.')
