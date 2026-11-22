"""Experimental Q3 cost rollout; frozen coverage/routing controller is preserved."""
import math
from time import perf_counter
from search_policy_compact import _state
import search_policy as old
import search_policy_q3_combined as combined
from q3_cost_rollout_v2 import choose_point

VERSION = 'q3_cost_rollout_v2'
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


def localize(client,point,response,channel,state,polygon):
    svd = response.get("svd_deg")
    if isinstance(svd, bool) or not isinstance(svd, (int, float)) or not math.isfinite(svd) or not 0 <= svd < 360:
        state["warnings"].append(f"channel {channel}: direction response lacks numeric svd_deg")
        return False
    previous_center = None
    for extra_used in range(old.MAX_EXTRA_BEARINGS + 1):
        box = old._box_circle(polygon)
        if box is None:
            state["warnings"].append(f"channel {channel}: empty bearing polygon")
            return old._fallback_scan(client, point, svd, channel, state)
        center, radius = box
        estimate=client.estimate(channel)
        if estimate is not None:
            center,radius=estimate
            if radius<=60:
                if old._resolve_clear(client,center,channel,state,'candidate check'):return True
        if radius <= old.LOCATE_CLEAR_RADIUS_M:
            if old._resolve_clear(client, center, channel, state, "bearing localization"):
                return True
            return old._fallback_scan(client, point, svd, channel, state)
        if extra_used >= old.MAX_EXTRA_BEARINGS:
            state["warnings"].append(f"channel {channel}: extra bearing limit reached")
            return old._fallback_scan(client, point, svd, channel, state)
        scoring_start = perf_counter()
        cloud = client.clouds.get(channel)
        history = client.public_history.get(channel, [])
        proposal = choose_point(polygon, center, client.position, cloud, history, point, svd, extra_used, previous_center)
        state['cost_rollout_scoring_cpu_s'] = state.get('cost_rollout_scoring_cpu_s', 0.0) + perf_counter()-scoring_start
        event = dict(proposal, channel=channel, initial_point=list(point), initial_svd_deg=svd,
                     previous_center=None if previous_center is None else list(previous_center),
                     extra_used_before=extra_used, before_polygon=[list(p) for p in polygon],
                     current_position=list(client.position), original_center=list(center),
                     public_cloud=cloud.tolist() if proposal['used_alternative'] else None,
                     public_history=list(history) if proposal['used_alternative'] else None,
                     measure_index=state['measure_count']+1)
        center = tuple(proposal['selected_point'])
        if previous_center and math.hypot(center[0] - previous_center[0], center[1] - previous_center[1]) <= 1e-6:
            state["warnings"].append(f"channel {channel}: repeated localization center")
            return old._fallback_scan(client, point, svd, channel, state)
        state.setdefault('cost_rollout_events', []).append(event)
        previous_center = center
        follow_up = client.measure(center[0], center[1], channel)
        state["measure_count"] += 1
        state["extra_bearing_count"] += 1
        event['response'] = dict(follow_up)
        kind = follow_up.get("measure_result")
        if kind == "near":
            if old._resolve_clear(client, center, channel, state, "near follow-up"):
                return True
            return old._fallback_scan(client, point, svd, channel, state)
        if kind == "direction":
            next_svd = follow_up.get("svd_deg")
            if isinstance(next_svd, bool) or not isinstance(next_svd, (int, float)) or not math.isfinite(next_svd) or not 0 <= next_svd < 360:
                state["warnings"].append(f"channel {channel}: follow-up direction lacks svd_deg")
                return old._fallback_scan(client, point, svd, channel, state)
            polygon = old._add_bearing(polygon, center, next_svd)
            continue
        state["warnings"].append(f"channel {channel}: follow-up no_signal/unknown; retaining fallback")
        return old._fallback_scan(client, point, svd, channel, state)
    raise AssertionError("unreachable extra-bearing loop")


def run_search(client,question=3,method='main'):
 if question!=3 or method!='main':raise ValueError('Q3 main only')
 client=PublicTracked(client)
 points=[(0.,0.)]+[(1124*math.cos(k*math.pi/3),1124*math.sin(k*math.pi/3)) for k in range(6)]
 state=_state(3,points);tracks={};seen=set();remaining=list(range(len(points)));attempted=set()
 try:
  while remaining or any(c not in state['cleared_channels'] and c not in attempted for c in tracks):
   if len(state['cleared_channels'])==16:break
   nodes=[('scan',i,points[i]) for i in remaining]
   for ch,(p,a,poly) in tracks.items():
    if ch in state['cleared_channels'] or ch in attempted:continue
    box=client.estimate(ch) or old._box_circle(poly)
    if box:nodes.append(('source',ch,box[0]))
   if not nodes:break
   kind,index,destination=_route(client.position,nodes)[0]
   if kind=='scan':
    remaining.remove(index);point=destination
    for ch in range(1,21):
     if ch in state['cleared_channels']:continue
     if ch in tracks:
      if not client.useful(ch,point):continue
      box=old._box_circle(tracks[ch][2])
      if box and box[1]<=old.LOCATE_CLEAR_RADIUS_M:continue
     a=client.measure(*point,ch);state['measure_count']+=1;state['coverage_measurements_by_channel'][ch]+=1
     result=a.get('measure_result')
     if result=='no_signal':continue
     seen.add(ch)
     if result=='near':old._resolve_clear(client,point,ch,state,'near joint-route scan')
     elif result=='direction' and _valid(a.get('svd_deg')):
      if ch not in tracks:tracks[ch]=(point,a,old._bearing_triangle(point,a['svd_deg']))
      else:
       p,b,poly=tracks[ch];new=old._add_bearing(poly,point,a['svd_deg'])
       if new:tracks[ch]=(p,b,new)
     else:state['warnings'].append(f'channel {ch}: invalid public response')
   else:
    ch=index;p,a,poly=tracks[ch];attempted.add(ch)
    if math.dist(client.position,p)>1 and client.useful(ch,client.position):
     here=client.position;extra=client.measure(*here,ch);state['measure_count']+=1;state['extra_bearing_count']+=1
     if extra.get('measure_result')=='near' and old._resolve_clear(client,here,ch,state,'near current pose'):continue
     if extra.get('measure_result')=='direction' and _valid(extra.get('svd_deg')):
      new=old._add_bearing(poly,here,extra['svd_deg'])
      if new:poly=new
     tracks[ch]=(p,a,poly);attempted.discard(ch);continue
    localize(client,p,a,ch,state,poly)
  state['excluded_channels']=set(range(1,21))-seen
  state['unresolved_channels']=seen-state['cleared_channels']
 except Exception as e:
  state['stopped']=True;state['warnings'].append(str(e));state['unresolved_channels']=set(range(1,21))-state['cleared_channels']-state['excluded_channels']
 for key in ('cleared_channels','excluded_channels','unresolved_channels'):state[key]=sorted(state[key])
 state['skipped_known_measures']=client.skipped_known_measures
 state.update(completed=not state['stopped'] and not state['unresolved_channels'] and len(state['cleared_channels'])+len(state['excluded_channels'])==20,question=3,method='main',version=VERSION,virtual_time_s=client.virtual_time_s)
 return state
