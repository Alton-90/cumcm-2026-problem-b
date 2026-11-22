"""Delay public polygon localizations above 150m radius until more coverage bearings arrive; finish all known sources after the cover."""
from __future__ import annotations
import math
import search_policy_polygon as polygon_policy
from search_policy import _add_bearing,_bearing_triangle,_box_circle,_resolve_clear
from search_policy_joint_route_q3 import _route
from search_policy_q4_batch import _TrackedClient,_localize,_state
from search_policy_q4_reuse import _supplement_from_current_pose
from search_policy_q4_ring22 import coverage_certificate,ring22_points
VERSION='q4_delayed_localize_v1';DIRECT_RADIUS=19.9;MAX_SOURCES=16
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
 pts=ring22_points();s=_state(pts);s.update(coverage_geometry=coverage_certificate(),route_strategy='receding_open_2opt_over_scans_and_public_polygon_centers',polygon_track_update_count=0,reuse_measure_count=0,reuse_polygon_update_count=0,direct_clear_skip_measure_count=0,maximum_count_stop_used=False,maximum_count_certificate=None)
 c=_TrackedClient(client);seen=set();tracks={};attempted=set();remaining=list(range(len(pts)))
 try:
  while remaining or any(ch not in s['cleared_channels'] and ch not in attempted for ch in tracks):
   if stop(s):break
   nodes=[('scan',i,pts[i]) for i in remaining]
   for ch,tr in tracks.items():
    if ch in s['cleared_channels'] or ch in attempted:continue
    b=box(tr)
    if b and (b[1]<=150.0 or not remaining):nodes.append(('source',ch,b[0]))
    elif b is None and math.dist(c.position,tr[0])>=50:nodes.append(('source',ch,c.position))
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
