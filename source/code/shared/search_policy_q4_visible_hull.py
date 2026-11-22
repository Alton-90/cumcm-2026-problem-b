"""Q4 safe visible-hull replacement of selected binary recovery measurements."""
from __future__ import annotations
import math
import search_policy_polygon as polygon_policy
from search_policy import _add_bearing,_bearing_triangle,_box_circle,_resolve_clear
from search_policy_joint_route_q3 import _route
from search_policy_q4_batch import _TrackedClient,_state
from search_policy import MAX_EXTRA_BEARINGS,LOCATE_CLEAR_RADIUS_M
from visible_hull_recovery import proposal
from search_policy_q4_reuse import _supplement_from_current_pose
from search_policy_q4_ring22 import coverage_certificate,ring22_points
VERSION='q4_visible_hull_v1';DIRECT_RADIUS=19.9;MAX_SOURCES=16

class HistoryClient(_TrackedClient):
 def __init__(self,client):
  super().__init__(client);self.positive={};self.measured={}
 def measure(self,x,y,ch):
  r=super().measure(x,y,ch);point=(x,y)
  self.measured.setdefault(ch,[])
  if point not in self.measured[ch]:self.measured[ch].append(point)
  if r.get('measure_result') in ('direction','near'):
   self.positive.setdefault(ch,[])
   if point not in self.positive[ch]:self.positive[ch].append(point)
  return r

def recovery_next(client,channel,polygon,last_visible,invisible,extra_used,state):
 old=((last_visible[0]+invisible[0])/2,(last_visible[1]+invisible[1])/2)
 p=proposal(client.positive.get(channel,[]),client.measured.get(channel,[]),polygon,client.position,old)
 if p is None:return old
 p.update(channel=channel,extra_used=extra_used,measure_index=state['measure_count']+1)
 state['visible_hull_events'].append(p)
 return p['point']

def _localize(client, initial_point, initial_svd, channel, state, feasible_polygon):
    """Recovery localizer copied from polygon policy, retaining its current polygon."""
    polygon = feasible_polygon
    last_visible = initial_point
    previous_candidate = None
    extra_used = 0
    while True:
        box = _box_circle(polygon)
        if box is None:
            state["warnings"].append(f"channel {channel}: empty bearing polygon")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        candidate, radius = box
        if radius <= LOCATE_CLEAR_RADIUS_M:
            if _resolve_clear(client, candidate, channel, state, "bearing localization"):
                return True
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        if extra_used >= MAX_EXTRA_BEARINGS:
            state["warnings"].append(f"channel {channel}: extra bearing limit reached")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        if previous_candidate and math.dist(candidate, previous_candidate) <= 1e-6:
            state["warnings"].append(f"channel {channel}: repeated localization center")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        previous_candidate = candidate
        response = client.measure(candidate[0], candidate[1], channel)
        state["measure_count"] += 1
        state["extra_bearing_count"] += 1
        extra_used += 1
        kind = response.get("measure_result")
        if kind == "near":
            if _resolve_clear(client, candidate, channel, state, "near follow-up"):
                return True
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
        if kind == "direction":
            svd = response.get("svd_deg")
            if not polygon_policy._valid_svd(svd):
                state["warnings"].append(f"channel {channel}: follow-up direction lacks valid svd_deg")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            polygon = _add_bearing(polygon, candidate, svd)
            last_visible = candidate
            continue
        if kind != "no_signal":
            state["warnings"].append(f"channel {channel}: unknown follow-up result {kind!r}")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)

        state["recovery_count"] += 1
        invisible = candidate
        while True:
            box = _box_circle(polygon)
            if box is None:
                state["warnings"].append(f"channel {channel}: empty recovery polygon")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            recovery_center, recovery_radius = box
            if recovery_radius <= LOCATE_CLEAR_RADIUS_M:
                if _resolve_clear(client, recovery_center, channel, state, "recovery localization"):
                    return True
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            if extra_used >= MAX_EXTRA_BEARINGS:
                state["warnings"].append(f"channel {channel}: recovery budget exhausted")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            midpoint = recovery_next(client, channel, polygon, last_visible, invisible, extra_used, state)
            if math.dist(midpoint, last_visible) <= 1e-6 or math.dist(midpoint, invisible) <= 1e-6:
                state["warnings"].append(f"channel {channel}: recovery midpoint repeated")
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            response = client.measure(midpoint[0], midpoint[1], channel)
            state["measure_count"] += 1
            state["extra_bearing_count"] += 1
            state["recovery_measure_count"] += 1
            extra_used += 1
            kind = response.get("measure_result")
            if kind == "near":
                if _resolve_clear(client, midpoint, channel, state, "near recovery"):
                    return True
                return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
            if kind == "direction":
                svd = response.get("svd_deg")
                if not polygon_policy._valid_svd(svd):
                    state["warnings"].append(f"channel {channel}: recovery direction lacks valid svd_deg")
                    return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)
                polygon = _add_bearing(polygon, midpoint, svd)
                last_visible = midpoint
                state["recovery_positive_count"] += 1
                continue
            if kind == "no_signal":
                invisible = midpoint
                continue
            state["warnings"].append(f"channel {channel}: unknown recovery result {kind!r}")
            return polygon_policy._fallback_scan(client, initial_point, initial_svd, channel, state, polygon)


def box(track):return _box_circle(track[2])
def stop(state):
 if len(state['cleared_channels'])!=MAX_SOURCES:return False
 state['excluded_channels'].update(set(range(1,21))-state['cleared_channels']);state['maximum_count_certificate']={'maximum_sources':16,'cleared_distinct_channels':16,'rule':'remaining channels excluded only after 16 successful distinct clears'};state['maximum_count_stop_used']=True;return True
def direct(client,ch,track,state):
 b=box(track)
 if b is None or b[1]>DIRECT_RADIUS:return False
 state['direct_clear_skip_measure_count']+=1;return _resolve_clear(client,b[0],ch,state,'19.9m public-polygon certificate')
def run_search(client,question=4,method='main'):
 if question!=4 or method!='main':raise ValueError('Q4 ring22 joint route only')
 pts=ring22_points();s=_state(pts);s['visible_hull_events']=[];s.update(coverage_geometry=coverage_certificate(),route_strategy='receding_open_2opt_over_scans_and_public_polygon_centers',polygon_track_update_count=0,reuse_measure_count=0,reuse_polygon_update_count=0,direct_clear_skip_measure_count=0,maximum_count_stop_used=False,maximum_count_certificate=None)
 c=HistoryClient(client);seen=set();tracks={};attempted=set();remaining=list(range(len(pts)))
 try:
  while remaining or any(ch not in s['cleared_channels'] and ch not in attempted for ch in tracks):
   if stop(s):break
   nodes=[('scan',i,pts[i]) for i in remaining]
   for ch,tr in tracks.items():
    if ch in s['cleared_channels'] or ch in attempted:continue
    b=box(tr)
    if b:nodes.append(('source',ch,b[0]))
    elif math.dist(c.position,tr[0])>=50:nodes.append(('source',ch,c.position))
   if not nodes:break
   kind,ident,dest=_route(c.position,nodes)[0]
   if kind=='scan':
    remaining.remove(ident)
    for ch in range(1,21):
     if stop(s):break
     if ch in s['cleared_channels']:continue
     oldtr=tracks.get(ch);b=box(oldtr) if oldtr else None
     if b and b[1]<=DIRECT_RADIUS:continue
     r=c.measure(*dest,ch);s['measure_count']+=1;s['coverage_measurements_by_channel'][ch]+=1;k=r.get('measure_result')
     if k=='no_signal':continue
     seen.add(ch)
     if k=='near':_resolve_clear(c,dest,ch,s,'near ring22 joint coverage')
     elif k=='direction' and polygon_policy._valid_svd(r.get('svd_deg')):
      if oldtr is None:tracks[ch]=(dest,r['svd_deg'],_bearing_triangle(dest,r['svd_deg']))
      else:
       p,a,poly=oldtr;m=_add_bearing(poly,dest,r['svd_deg']);tracks[ch]=(p,a,m) if m else oldtr
      attempted.discard(ch);s['polygon_track_update_count']+=1
     else:s['warnings'].append(f'channel {ch}: invalid public response');s['unresolved_channels'].add(ch)
    s['batch_count']+=1
   else:
    ch=ident;tr=tracks[ch];attempted.add(ch)
    if direct(c,ch,tr,s):continue
    tr,cleared=_supplement_from_current_pose(c,ch,tr,s)
    if cleared:continue
    tracks[ch]=tr
    if direct(c,ch,tr,s):continue
    _localize(c,tr[0],tr[1],ch,s,tr[2])
  if not s['maximum_count_stop_used']:
   for ch in range(1,21):
    if ch not in s['cleared_channels']:(s['unresolved_channels'] if ch in seen else s['excluded_channels']).add(ch)
 except Exception as e:
  s['stopped']=True;s['warnings'].append(f'protocol/transport stop: {type(e).__name__}: {e}');s['unresolved_channels'].update(set(range(1,21))-s['cleared_channels']-s['excluded_channels'])
 for k in ('cleared_channels','excluded_channels','unresolved_channels'):s[k]=sorted(s[k])
 s.update(completed=not s['stopped'] and not s['unresolved_channels'] and len(s['cleared_channels'])+len(s['excluded_channels'])==20,question=4,method='main',version=VERSION,virtual_time_s=c.virtual_time_s);return s
