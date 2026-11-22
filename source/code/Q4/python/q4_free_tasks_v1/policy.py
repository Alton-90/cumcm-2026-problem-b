"""Certified ring22 cover with Q4 joint-route v3 public-polygon planning."""
from __future__ import annotations
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'code/Q4/python'),str(ROOT/'code/shared')]
from pruning import Tasks
import math
import search_policy_polygon as polygon_policy
from search_policy import _add_bearing,_bearing_triangle,_box_circle,_resolve_clear
from search_policy_joint_route_q3 import _route
from search_policy_q4_batch import _TrackedClient,_state
from localizer_free import _localize
from search_policy_q4_reuse import _supplement_from_current_pose
from search_policy_q4_ring22 import coverage_certificate,ring22_points
VERSION='q4_free_tasks_v1';DIRECT_RADIUS=19.9;MAX_SOURCES=16
def box(track):return _box_circle(track[2])
def stop(state):
 if len(state['cleared_channels'])!=MAX_SOURCES:return False
 state['excluded_channels'].update(set(range(1,21))-state['cleared_channels']);state['maximum_count_certificate']={'maximum_sources':16,'cleared_distinct_channels':16,'rule':'remaining channels excluded only after 16 successful distinct clears'};state['maximum_count_stop_used']=True;return True
def direct(client,ch,track,state):
 b=box(track)
 if b is None or b[1]>DIRECT_RADIUS:return False
 state['direct_clear_skip_measure_count']+=1;return _resolve_clear(client,b[0],ch,state,'19.9m public-polygon certificate')
def run_search(client,question=4,method='main',variant='relaxed',settings=(3.,6.,0.),step_fraction=.25):
 if question!=4 or method!='main':raise ValueError('Q4 ring22 joint route only')
 pts=ring22_points();s=_state(pts);s.update(coverage_geometry=coverage_certificate(),route_strategy='receding_open_2opt_over_scans_and_public_polygon_centers',polygon_track_update_count=0,reuse_measure_count=0,reuse_polygon_update_count=0,direct_clear_skip_measure_count=0,maximum_count_stop_used=False,maximum_count_certificate=None)
 tasks=Tasks(pts);s['free_tasks']=tasks.stats
 c=Tracked(client,variant);c.settings=settings;c.step_fraction=step_fraction;seen=set();tracks={};attempted=set();remaining=list(range(len(pts)))
 try:
  while remaining or any(ch not in s['cleared_channels'] and ch not in attempted for ch in tracks):
   if stop(s):break
   tasks.prune(remaining,c,tracks,seen,s['cleared_channels'])
   nodes=[('scan',i,pts[i]) for i in remaining]
   for ch,tr in tracks.items():
    if ch in s['cleared_channels'] or ch in attempted:continue
    b=box(tr)
    if b:nodes.append(('source',ch,b[0]))
    elif math.dist(c.position,tr[0])>=50:nodes.append(('source',ch,c.position))
   if not nodes:break
   kind,ident,dest=_route(c.position,nodes)[0]
   if kind=='scan':
    # Keep this stop in obligations until its sequential channel batch finishes.
    for ch in range(1,21):
     if stop(s):break
     if ch in s['cleared_channels']:continue
     oldtr=tracks.get(ch);b=box(oldtr) if oldtr else None
     if b and b[1]<=DIRECT_RADIUS:continue
     if tasks.try_delete(ident,ch,remaining,c,tracks,seen):
      s['free_tasks']['omitted_scan_measurements']+=1;continue
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
    remaining.remove(ident)
    s['batch_count']+=1
   else:
    ch=ident;tr=tracks[ch];attempted.add(ch)
    if direct(c,ch,tr,s):continue
    tr,cleared=_supplement_from_current_pose(c,ch,tr,s)
    if cleared:continue
    tracks[ch]=tr
    if direct(c,ch,tr,s):continue
    _localize(c,tr[0],tr[1],ch,s,tr[2])
    if c.info_enabled and ch in c.local_polygons:tracks[ch]=(tr[0],tr[1],c.local_polygons[ch])
  if not s['maximum_count_stop_used']:
   for ch in range(1,21):
    if ch not in s['cleared_channels']:(s['unresolved_channels'] if ch in seen else s['excluded_channels']).add(ch)
 except Exception as e:
  s['stopped']=True;s['warnings'].append(f'protocol/transport stop: {type(e).__name__}: {e}');s['unresolved_channels'].update(set(range(1,21))-s['cleared_channels']-s['excluded_channels'])
 for k in ('cleared_channels','excluded_channels','unresolved_channels'):s[k]=sorted(s[k])
 s.update(completed=not s['stopped'] and not s['unresolved_channels'] and len(s['cleared_channels'])+len(s['excluded_channels'])==20,question=4,method='main',version=VERSION,virtual_time_s=c.virtual_time_s);return s

class Tracked(_TrackedClient):
 def __init__(self,client,variant):
  super().__init__(client)
  if variant not in ('control','information','early','combined','relaxed'):raise ValueError(variant)
  self.info_enabled=variant in ('information','combined','relaxed');self.early_enabled=variant in ('early','combined','relaxed')
  self.relaxed_enabled=variant=='relaxed'
  self.latest_visible={};self.local_polygons={}
  self.negatives={ch:[] for ch in range(1,21)};self.receipt_count=0
 def measure(self,x,y,ch):
  response=super().measure(x,y,ch)
  self.receipt_count+=1
  if response.get('measure_result')=='no_signal':self.negatives[ch].append((x,y))
  if response.get('measure_result')=='near' or (response.get('measure_result')=='direction' and polygon_policy._valid_svd(response.get('svd_deg'))):self.latest_visible[ch]=(x,y)
  return response
