"""Q4 ring22 joint route with public position/heading feasibility clouds."""
from __future__ import annotations
import math
import numpy as np
import search_policy_polygon as polygon_policy
from search_policy import _add_bearing,_bearing_triangle,_box_circle,_resolve_clear
from search_policy_joint_route_q3 import _route
from search_policy_q4_batch import _localize,_state
from search_policy_q4_reuse import _supplement_from_current_pose
from search_policy_q4_ring22 import coverage_certificate,ring22_points
VERSION='q4_joint_feasible_v1';MAX_SOURCES=16;DIRECT=19.9
HEADINGS=np.arange(0.,360.,5.)
def valid(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0<=x<360
def angle_delta(a,b):return np.abs((a-b+180)%360-180)
class FeasibleClient:
 def __init__(self,client):
  self.client=client;self.position=(0.,0.);axis=np.arange(-1800.,1800.1,20.);xx,yy=np.meshgrid(axis,axis);p=np.column_stack((xx.ravel(),yy.ravel()));self.grid=p[np.linalg.norm(p,axis=1)<=1815];self.models={};self.history={}
 @property
 def virtual_time_s(self):return self.client.virtual_time_s
 def clear(self,x,y,ch):
  r=self.client.clear(x,y,ch);self.position=(x,y);return r
 def measure(self,x,y,ch):
  r=self.client.measure(x,y,ch);self.position=(x,y);self.history.setdefault(ch,[]).append(((float(x),float(y)),r.get('measure_result'),r.get('svd_deg')))
  if r.get('measure_result')=='direction' and valid(r.get('svd_deg')):
   if ch not in self.models:self._initialize(ch)
   self._rebuild(ch)
  elif ch in self.models:self._rebuild(ch)
  return r
 def _initialize(self,ch):
  self.models[ch]={'q':self.grid.copy(),'heading':np.ones((len(self.grid),72),dtype=bool),'omni':np.ones(len(self.grid),dtype=bool),'rmin':np.full(len(self.grid),1000.)}
 def _rebuild(self,ch):
  q=self.grid.copy();hm=np.ones((len(q),72),dtype=bool);om=np.ones(len(q),dtype=bool);rmin=np.full(len(q),1000.);positives=[];negatives=[]
  for p,k,svd in self.history[ch]:
   delta=np.asarray(p)-q;d=np.linalg.norm(delta,axis=1)
   if k=='direction' and valid(svd):
    t=math.radians(svd);u=np.array([math.cos(t),math.sin(t)]);v=np.array([-u[1],u[0]]);keep=(d<=1515)&(delta@u>=-15)&(np.abs(delta@v)<=d*math.sin(math.radians(1.005))+15);q,hm,om,rmin,d=q[keep],hm[keep],om[keep],rmin[keep],d[keep];positives.append(np.asarray(p));rmin=np.maximum(rmin,d-15)
   elif k=='no_signal':negatives.append(np.asarray(p))
  for p in positives:
   vec=p-q;d=np.linalg.norm(vec,axis=1);bearing=(np.degrees(np.arctan2(vec[:,1],vec[:,0]))+360)%360;eps=np.where(d>15,np.degrees(np.arcsin(np.minimum(1,15/np.maximum(d,1e-9))))+2.5,90.);hm&=angle_delta(HEADINGS[None,:],bearing[:,None])<=np.minimum(180,90+eps)[:,None]
  for p in negatives:
   vec=p-q;d=np.linalg.norm(vec,axis=1);forced=d+15<=rmin;om[forced]=False;bearing=(np.degrees(np.arctan2(vec[:,1],vec[:,0]))+360)%360;eps=np.where(d>15,np.degrees(np.arcsin(np.minimum(1,15/np.maximum(d,1e-9))))+2.5,90.);hm[forced]&=angle_delta(HEADINGS[None,:],bearing[forced,None])>=np.maximum(0,90-eps[forced])[:,None]
  alive=om|hm.any(axis=1);self.models[ch]={'q':q[alive],'heading':hm[alive],'omni':om[alive],'rmin':rmin[alive]}
 def estimate(self,ch):
  m=self.models.get(ch)
  if not m or not len(m['q']):return None
  q=m['q'];weights=m['omni'].astype(float)+m['heading'].sum(axis=1)/72;center=np.average(q,axis=0,weights=np.maximum(weights,1/72));return (float(center[0]),float(center[1])),float(np.linalg.norm(q-center,axis=1).max()+15)
def stop(s):
 if len(s['cleared_channels'])!=16:return False
 s['excluded_channels'].update(set(range(1,21))-s['cleared_channels']);s['maximum_count_certificate']={'maximum_sources':16,'cleared_distinct_channels':16};s['maximum_count_stop_used']=True;return True
def run_search(client,question=4,method='main'):
 if question!=4 or method!='main':raise ValueError('Q4 main only')
 pts=ring22_points();s=_state(pts);s.update(coverage_geometry=coverage_certificate(),feasible_model='20m position grid + omni/72 heading bins',maximum_count_stop_used=False,maximum_count_certificate=None,reuse_measure_count=0,reuse_polygon_update_count=0);c=FeasibleClient(client);seen=set();tracks={};attempted=set();remaining=list(range(len(pts)))
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
    ch=idx;tr=tracks[ch];attempted.add(ch);est=c.estimate(ch)
    if est and est[1]<=DIRECT and _resolve_clear(c,est[0],ch,s,'feasible 19.9m'):continue
    tr,done=_supplement_from_current_pose(c,ch,tr,s);tracks[ch]=tr if tr else tracks[ch]
    if not done:_localize(c,tr[0],tr[1],ch,s,tr[2])
  if not s['maximum_count_stop_used']:
   for ch in range(1,21):
    if ch not in s['cleared_channels']:(s['unresolved_channels'] if ch in seen else s['excluded_channels']).add(ch)
 except Exception as e:s['stopped']=True;s['warnings'].append(str(e));s['unresolved_channels'].update(set(range(1,21))-s['cleared_channels']-s['excluded_channels'])
 for k in ('cleared_channels','excluded_channels','unresolved_channels'):s[k]=sorted(s[k])
 s.update(completed=not s['stopped'] and not s['unresolved_channels'] and len(s['cleared_channels'])+len(s['excluded_channels'])==20,question=4,method='main',version=VERSION,virtual_time_s=c.virtual_time_s);return s
