"""Receding open-route planning over certified scan points and observed sources.

The seven-point Q3 cover and exact fallback stay intact. Source destinations
are estimates based only on accumulated public bearings and no-signal data.
"""
import math
import numpy as np
import search_policy as old
from search_policy_compact import _state
from search_policy_feasible_q3 import Tracked,localize
VERSION='q3_selective_cover_v1'

def _valid(s):
 return isinstance(s,(int,float)) and not isinstance(s,bool) and math.isfinite(s) and 0<=s<360

def _route(position,nodes):
 """Nearest insertion seed with deterministic open-path 2-opt improvement."""
 pending=list(nodes);route=[];here=position
 while pending:
  item=min(pending,key=lambda item:(math.dist(here,item[2]),item[0],item[1]))
  pending.remove(item);route.append(item);here=item[2]
 for _ in range(10):
  best=0.;swap=None
  for i in range(len(route)-1):
   before=position if i==0 else route[i-1][2]
   for j in range(i+1,len(route)):
    oldcost=math.dist(before,route[i][2]);newcost=math.dist(before,route[j][2])
    if j+1<len(route):
     oldcost+=math.dist(route[j][2],route[j+1][2]);newcost+=math.dist(route[i][2],route[j+1][2])
    if oldcost-newcost>best+1e-7:best=oldcost-newcost;swap=(i,j)
  if swap is None:break
  i,j=swap;route[i:j+1]=reversed(route[i:j+1])
 return route

def run_search(client,question=3,method='main'):
 if question!=3 or method!='main':raise ValueError('Q3 main only')
 client=Tracked(client)
 points=[(0.,0.)]+[(1150*math.cos(k*math.pi/3),1150*math.sin(k*math.pi/3)) for k in range(6)]
 state=_state(3,points);tracks={};seen=set();remaining=list(range(len(points)));attempted=set()
 state.update(opportunistic_scans=0,grid_absence_certificate=False,maximum_count_certificate=False)
 try:
  while remaining or any(c not in state['cleared_channels'] and c not in attempted for c in tracks):
   unseen=set(range(1,21))-seen
   if len(seen)>=16:
    remaining=[];state['maximum_count_certificate']=True
   elif unseen and all(c in client.clouds and len(client.clouds[c])==0 for c in unseen):
    remaining=[];state['grid_absence_certificate']=True
   nodes=[('scan',i,points[i]) for i in remaining]
   for ch,(p,a,poly) in tracks.items():
    if ch in state['cleared_channels'] or ch in attempted:continue
    box=client.estimate(ch) or old._box_circle(poly)
    if box:nodes.append(('source',ch,box[0]))
   if not nodes:break
   kind,index,destination=_route(client.position,nodes)[0]
   if remaining and unseen and all(c in client.clouds for c in unseen):
    cloud=client.clouds[min(unseen)]
    # Never-seen channels share all sweep points. This is a conservative
    # spatial cover score, not a count or a true-source coordinate query.
    gain=lambda p:int(np.sum(np.linalg.norm(cloud-np.array(p),axis=1)<985))
    cost=6*len(unseen)
    here_gain=gain(client.position)
    best=max(gain(points[i])/(6*(20-len(state['cleared_channels']))+math.dist(client.position,points[i])/5) for i in remaining)
    if here_gain>=200 and here_gain/cost>best*1.1:
     kind,index,destination='scan',-1,client.position
   if kind=='scan':
    if index>=0:remaining.remove(index)
    else:state['opportunistic_scans']+=1
    point=destination
    for ch in range(1,21):
     if ch in state['cleared_channels']:continue
     if index==-1 and ch in seen:continue
     if ch in tracks:
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
    if math.dist(client.position,p)>1:
     here=client.position;extra=client.measure(*here,ch);state['measure_count']+=1;state['extra_bearing_count']+=1
     if extra.get('measure_result')=='near' and old._resolve_clear(client,here,ch,state,'near current pose'):continue
     if extra.get('measure_result')=='direction' and _valid(extra.get('svd_deg')):
      new=old._add_bearing(poly,here,extra['svd_deg'])
      if new:poly=new
    localize(client,p,a,ch,state,poly)
  state['excluded_channels']=set(range(1,21))-seen
  state['unresolved_channels']=seen-state['cleared_channels']
 except Exception as e:
  state['stopped']=True;state['warnings'].append(str(e));state['unresolved_channels']=set(range(1,21))-state['cleared_channels']-state['excluded_channels']
 for key in ('cleared_channels','excluded_channels','unresolved_channels'):state[key]=sorted(state[key])
 state.update(completed=not state['stopped'] and not state['unresolved_channels'] and len(state['cleared_channels'])+len(state['excluded_channels'])==20,question=3,method='main',version=VERSION,virtual_time_s=client.virtual_time_s)
 return state
