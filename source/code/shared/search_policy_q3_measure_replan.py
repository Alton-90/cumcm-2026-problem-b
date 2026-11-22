"""Q3 joint routing with one current-pose bearing before each source replan."""
from __future__ import annotations
import math
import search_policy as old
from search_policy_compact import _state
from search_policy_feasible_q3 import Tracked,localize
VERSION='q3_measure_replan_v1';MAX_SOURCES=16
def valid(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0<=x<360
def route(pos,nodes):
 p=list(nodes);r=[];here=pos
 while p:
  x=min(p,key=lambda x:(math.dist(here,x[2]),x[0],x[1]));p.remove(x);r.append(x);here=x[2]
 for _ in range(10):
  gain=0;swap=None
  for i in range(len(r)-1):
   before=pos if i==0 else r[i-1][2]
   for j in range(i+1,len(r)):
    oldc=math.dist(before,r[i][2]);newc=math.dist(before,r[j][2])
    if j+1<len(r):oldc+=math.dist(r[j][2],r[j+1][2]);newc+=math.dist(r[i][2],r[j+1][2])
    if oldc-newc>gain+1e-7:gain,swap=oldc-newc,(i,j)
  if swap is None:break
  i,j=swap;r[i:j+1]=reversed(r[i:j+1])
 return r
def reached(state):
 if len(state['cleared_channels'])!=MAX_SOURCES:return False
 state['excluded_channels'].update(set(range(1,21))-state['cleared_channels']);state['maximum_count_certificate']={'maximum_sources':16,'cleared_distinct_channels':16};state['maximum_count_stop_used']=True;return True
def run_search(client,question=3,method='main'):
 if question!=3 or method!='main':raise ValueError('Q3 main only')
 c=Tracked(client);pts=[(0.,0.)]+[(1150*math.cos(k*math.pi/3),1150*math.sin(k*math.pi/3)) for k in range(6)];s=_state(3,pts);tracks={};seen=set();remaining=list(range(7));attempted=set();supplemented=set();s.update(route_strategy='one_current_pose_bearing_then_global_replan',replan_bearing_count=0,maximum_count_stop_used=False,maximum_count_certificate=None)
 try:
  while remaining or any(ch not in s['cleared_channels'] and ch not in attempted for ch in tracks):
   if reached(s):break
   nodes=[('scan',i,pts[i]) for i in remaining]
   for ch,(p,a,poly) in tracks.items():
    if ch in s['cleared_channels'] or ch in attempted:continue
    box=c.estimate(ch) or old._box_circle(poly)
    if box:nodes.append(('source',ch,box[0]))
   if not nodes:break
   kind,idx,dest=route(c.position,nodes)[0]
   if kind=='scan':
    remaining.remove(idx)
    for ch in range(1,21):
     if reached(s):break
     if ch in s['cleared_channels']:continue
     box=old._box_circle(tracks[ch][2]) if ch in tracks else None
     if box and box[1]<=old.LOCATE_CLEAR_RADIUS_M:continue
     a=c.measure(*dest,ch);s['measure_count']+=1;s['coverage_measurements_by_channel'][ch]+=1;k=a.get('measure_result')
     if k=='no_signal':continue
     seen.add(ch)
     if k=='near':old._resolve_clear(c,dest,ch,s,'near replan scan')
     elif k=='direction' and valid(a.get('svd_deg')):
      if ch not in tracks:tracks[ch]=(dest,a,old._bearing_triangle(dest,a['svd_deg']))
      else:
       p,b,poly=tracks[ch];m=old._add_bearing(poly,dest,a['svd_deg']);tracks[ch]=(p,b,m) if m else tracks[ch]
      attempted.discard(ch)
     else:s['warnings'].append(f'channel {ch}: invalid public response')
   else:
    ch=idx;p,a,poly=tracks[ch];key=(ch,round(c.position[0],6),round(c.position[1],6))
    if key not in supplemented and math.dist(c.position,p)>1:
     supplemented.add(key);extra=c.measure(*c.position,ch);s['measure_count']+=1;s['extra_bearing_count']+=1;s['replan_bearing_count']+=1
     if extra.get('measure_result')=='near' and old._resolve_clear(c,c.position,ch,s,'near replan reuse'):continue
     if extra.get('measure_result')=='direction' and valid(extra.get('svd_deg')):
      m=old._add_bearing(poly,c.position,extra['svd_deg']);tracks[ch]=(p,a,m) if m else tracks[ch]
     attempted.discard(ch);continue
    attempted.add(ch);localize(c,p,a,ch,s,tracks[ch][2])
  if not s['maximum_count_stop_used']:
   s['excluded_channels']=set(range(1,21))-seen;s['unresolved_channels']=seen-s['cleared_channels']
 except Exception as e:
  s['stopped']=True;s['warnings'].append(str(e));s['unresolved_channels']=set(range(1,21))-s['cleared_channels']-s['excluded_channels']
 for k in ('cleared_channels','excluded_channels','unresolved_channels'):s[k]=sorted(s[k])
 s.update(completed=not s['stopped'] and not s['unresolved_channels'] and len(s['cleared_channels'])+len(s['excluded_channels'])==20,question=3,method='main',version=VERSION,virtual_time_s=c.virtual_time_s);return s
