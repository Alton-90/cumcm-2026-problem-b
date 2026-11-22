"""Q4 joint feasible-domain localizer; public observations only."""
from __future__ import annotations
import math
import numpy as np
import search_policy_polygon as polygon_policy
from search_policy import _add_bearing,_bearing_triangle,_box_circle,_resolve_clear
from search_policy_joint_route_q3 import _route
from search_policy_q4_batch import _state
from search_policy_q4_ring22 import coverage_certificate,ring22_points
from search_policy_q4_joint_feasible import FeasibleClient as V1Client,HEADINGS,angle_delta,valid
VERSION='q4_joint_feasible_v2';MAX_STEPS=8
class FeasibleClient(V1Client):
 def _rebuild(self,ch):
  q=self.grid.copy();hm=np.ones((len(q),72),dtype=bool);om=np.ones(len(q),dtype=bool);rmin=np.full(len(q),1000.);pos=[];neg=[]
  for p,k,svd in self.history[ch]:
   source_delta=q-np.asarray(p);d=np.linalg.norm(source_delta,axis=1)
   if k=='direction' and valid(svd):
    t=math.radians(svd);u=np.array([math.cos(t),math.sin(t)]);v=np.array([-u[1],u[0]]);keep=(d<=1515)&(source_delta@u>=-15)&(np.abs(source_delta@v)<=d*math.sin(math.radians(1.005))+15);q,hm,om,rmin,d=q[keep],hm[keep],om[keep],rmin[keep],d[keep];pos.append(np.asarray(p));rmin=np.maximum(rmin,d-15)
   elif k=='no_signal':neg.append(np.asarray(p))
  for p in pos:
   vec=p-q;d=np.linalg.norm(vec,axis=1);bearing=(np.degrees(np.arctan2(vec[:,1],vec[:,0]))+360)%360;eps=np.where(d>15,np.degrees(np.arcsin(np.minimum(1,15/np.maximum(d,1e-9))))+2.5,90.);hm&=angle_delta(HEADINGS[None,:],bearing[:,None])<=np.minimum(180,90+eps)[:,None]
  for p in neg:
   vec=p-q;d=np.linalg.norm(vec,axis=1);forced=d+15<=rmin;om[forced]=False;bearing=(np.degrees(np.arctan2(vec[:,1],vec[:,0]))+360)%360;eps=np.where(d>15,np.degrees(np.arcsin(np.minimum(1,15/np.maximum(d,1e-9))))+2.5,90.);hm[forced]&=angle_delta(HEADINGS[None,:],bearing[forced,None])>=np.maximum(0,90-eps[forced])[:,None]
  alive=om|hm.any(axis=1);self.models[ch]={'q':q[alive],'heading':hm[alive],'omni':om[alive],'rmin':rmin[alive]}
def feasible_localize(c,initial,svd,ch,s,poly):
 failed=set()
 for _ in range(MAX_STEPS):
  est=c.estimate(ch)
  if est is None:break
  center,radius=est;key=(round(center[0],6),round(center[1],6))
  if radius<=60 and key not in failed:
   s['feasible_clear_attempt_count']+=1
   if _resolve_clear(c,center,ch,s,'feasible <=60m candidate'):return True
   failed.add(key);s['feasible_failed_clear_count']+=1
  r=c.measure(*center,ch);s['measure_count']+=1;s['extra_bearing_count']+=1;s['feasible_measure_count']+=1;k=r.get('measure_result')
  if k=='near' and key not in failed:
   s['feasible_clear_attempt_count']+=1
   if _resolve_clear(c,center,ch,s,'near feasible candidate'):return True
   failed.add(key);s['feasible_failed_clear_count']+=1
  elif k=='direction' and valid(r.get('svd_deg')):
   m=_add_bearing(poly,center,r['svd_deg']);poly=m if m else poly
  elif k not in ('no_signal','near'):break
 s['feasible_fallback_count']+=1
 return polygon_policy._fallback_scan(c,initial,svd,ch,s,poly)
def stop(s):
 if len(s['cleared_channels'])!=16:return False
 s['excluded_channels'].update(set(range(1,21))-s['cleared_channels']);s['maximum_count_certificate']={'maximum_sources':16,'cleared_distinct_channels':16};s['maximum_count_stop_used']=True;return True
def run_search(client,question=4,method='main'):
 if question!=4 or method!='main':raise ValueError('Q4 main only')
 pts=ring22_points();s=_state(pts);s.update(coverage_geometry=coverage_certificate(),feasible_model='corrected 20m grid + omni/72 heading bins',feasible_measure_count=0,feasible_clear_attempt_count=0,feasible_failed_clear_count=0,feasible_fallback_count=0,maximum_count_stop_used=False,maximum_count_certificate=None);c=FeasibleClient(client);seen=set();tracks={};attempted=set();remaining=list(range(len(pts)))
 try:
  while remaining or any(ch not in s['cleared_channels'] and ch not in attempted for ch in tracks):
   if stop(s):break
   nodes=[('scan',i,pts[i]) for i in remaining]
   for ch,tr in tracks.items():
    if ch in s['cleared_channels'] or ch in attempted:continue
    est=c.estimate(ch) or _box_circle(tr[2]);nodes.append(('source',ch,est[0] if est else tr[0]))
   if not nodes:break
   kind,idx,dest=_route(c.position,nodes)[0]
   if kind=='scan':
    remaining.remove(idx)
    for ch in range(1,21):
     if stop(s):break
     if ch in s['cleared_channels']:continue
     r=c.measure(*dest,ch);s['measure_count']+=1;s['coverage_measurements_by_channel'][ch]+=1;k=r.get('measure_result')
     if k=='no_signal':continue
     seen.add(ch)
     if k=='near':_resolve_clear(c,dest,ch,s,'near feasible coverage')
     elif k=='direction' and valid(r.get('svd_deg')):
      if ch not in tracks:tracks[ch]=(dest,r['svd_deg'],_bearing_triangle(dest,r['svd_deg']))
      else:
       p,a,poly=tracks[ch];m=_add_bearing(poly,dest,r['svd_deg']);tracks[ch]=(p,a,m) if m else tracks[ch]
      attempted.discard(ch)
     else:s['unresolved_channels'].add(ch)
   else:
    ch=idx;attempted.add(ch);p,a,poly=tracks[ch];feasible_localize(c,p,a,ch,s,poly)
  if not s['maximum_count_stop_used']:
   for ch in range(1,21):
    if ch not in s['cleared_channels']:(s['unresolved_channels'] if ch in seen else s['excluded_channels']).add(ch)
 except Exception as e:s['stopped']=True;s['warnings'].append(str(e));s['unresolved_channels'].update(set(range(1,21))-s['cleared_channels']-s['excluded_channels'])
 s['feasible_histories']={str(ch):hist for ch,hist in c.history.items()}
 s['feasible_nonempty_channels']=sorted(ch for ch,m in c.models.items() if len(m['q']))
 for k in ('cleared_channels','excluded_channels','unresolved_channels'):s[k]=sorted(s[k])
 s.update(completed=not s['stopped'] and not s['unresolved_channels'] and len(s['cleared_channels'])+len(s['excluded_channels'])==20,question=4,method='main',version=VERSION,virtual_time_s=c.virtual_time_s);return s
