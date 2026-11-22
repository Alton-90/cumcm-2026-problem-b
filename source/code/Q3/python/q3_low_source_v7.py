"""Sparse-observation Q3 route control with a rotated seven-region cover.
No true source count is accepted by this policy. All tunable priorities are proxies;
continuous coverage and the original public-measurement localizer stay mandatory.
"""
from __future__ import annotations
import math
from threading import RLock
import search_policy as old
import q3_cost_rollout_v2 as rollout
import search_policy_q3_combined as combined
from search_policy_compact import _state
from search_policy_q3_cost_rollout_v2 import PublicTracked,localize
from route_multistart import route as _fixed_route
from search_policy_joint_route_q3 import _route as cheap_route
VERSION='q3_low_source_v7'
_valid=combined._valid
DEFAULTS={'orientation_bins':1,'radius_cap_m':1730.,'scan_latency_weight':0.,'sparse_cutoff':12,
          'route_detour_cap_m':0.,'move_slack_m':175.,'min_saving_s':0.,'offset_scale':.5}
DOMAINS={'orientation_bins':[1,4,8,12],'radius_cap_m':[1240.,1400.,1560.,1730.],
         'scan_latency_weight':[0.,.5,1.,2.,4.],'sparse_cutoff':[10,12,14,16],
         'route_detour_cap_m':[0.,100.,250.,500.],'move_slack_m':[50.,100.,175.,250.,350.],
         'min_saving_s':[0.,2.,5.,10.,15.],'offset_scale':[.25,.5,.75,1.,1.5]}
_LOCK=RLock()
def parameters(values=None):
 p=dict(DEFAULTS)
 if values is not None:
  if set(values)-set(p):raise ValueError('Unknown parameter keys')
  p.update(values)
 for key,value in p.items():
  if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value not in DOMAINS[key]:raise ValueError('Invalid parameter '+key)
 return p

def path_length(position,nodes):
 return sum(math.dist(a,b) for a,b in zip([position]+[n[2] for n in nodes],[n[2] for n in nodes]))

def radial_route(position,nodes,cap):
 """Optimize radial scan positions inside independently certified sectors."""
 route=_fixed_route(position,nodes)
 for repeat in range(3):
  for i,(kind,identifier,p) in enumerate(route):
   if kind!='scan' or identifier==0:continue
   a=math.atan2(p[1],p[0]);u=(math.cos(a),math.sin(a))
   before=position if i==0 else route[i-1][2]
   after=None if i+1==len(route) else route[i+1][2]
   def cost(d):
    q=(d*u[0],d*u[1])
    return math.dist(before,q)+(0.0 if after is None else math.dist(q,after))
   lo,hi=1124.,cap;g=(5**.5-1)/2
   x=hi-g*(hi-lo);y=lo+g*(hi-lo);fx,fy=cost(x),cost(y)
   for _ in range(30):
    if fx<fy:hi,y,fy=y,x,fx;x=hi-g*(hi-lo);fx=cost(x)
    else:lo,x,fx=x,y,fy;y=lo+g*(hi-lo);fy=cost(y)
   d=min((1124.,cap,(lo+hi)/2,math.hypot(*p)),key=cost)
   route[i]=(kind,identifier,(d*u[0],d*u[1]))
  route=_fixed_route(position,route)
 return route


class SparsePlanner:
 def __init__(self,p):self.p=p;self.rotation_deg=None;self.events=[]
 def rotate_nodes(self,nodes,angle,radial=1124.):
  result=[]
  for kind,identifier,point in nodes:
   if kind=='scan' and identifier>0:
    a=math.radians(angle)+(identifier-1)*math.pi/3
    point=(radial*math.cos(a),radial*math.sin(a))
   result.append((kind,identifier,point))
  return result
 def objective(self,position,nodes,discovered):
  weight=self.p['scan_latency_weight']*max(0.,(self.p['sparse_cutoff']+1-discovered)/(self.p['sparse_cutoff']+1))
  distance=0.;here=position;arrivals=[]
  for kind,identifier,point in nodes:
   distance+=math.dist(here,point);here=point
   if kind=='scan':arrivals.append(distance)
  return distance+weight*(sum(arrivals)/len(arrivals) if arrivals else 0.)
 def route(self,position,nodes,discovered):
  # Choose a frame once, after the complete origin scan; each subsequent six-anchor
  # destination uses the same frame. This preserves a global continuous disk cover.
  if self.rotation_deg is None and not any(k=='scan' and i==0 for k,i,_ in nodes):
   bins=self.p['orientation_bins'];options=[]
   for j in range(bins):
    angle=60*j/bins
    trial=self.rotate_nodes(nodes,angle,(1124.+self.p['radius_cap_m'])/2)
    path=cheap_route(position,trial)
    options.append((self.objective(position,path,discovered),angle))
   self.rotation_deg=min(options)[1]
   self.events.append({'event':'coverage_frame_selected','rotation_deg':self.rotation_deg,'discovered_channels':discovered,
      'orientation_bins':bins,'proxy_scores':[[angle,score] for score,angle in options]})
  mapped=self.rotate_nodes(nodes,self.rotation_deg or 0.)
  route=radial_route(position,mapped,self.p['radius_cap_m'])
  if self.p['scan_latency_weight']==0 or self.p['route_detour_cap_m']==0:return route
  reference=path_length(position,route);best=route;score=self.objective(position,route,discovered)
  # Bounded relocation; no claim that the proxy is the complete POMDP objective.
  for _ in range(2):
   variants=[list(reversed(best)),[x for x in best if x[0]=='scan']+[x for x in best if x[0]!='scan']]
   for i,node in enumerate(best):
    if node[0]=='scan' and i>0:variants.append([node]+best[:i]+best[i+1:])
   changed=False
   for trial in variants:
    length=path_length(position,trial)
    if length>reference+self.p['route_detour_cap_m']+1e-7:continue
    value=self.objective(position,trial,discovered)
    if value<score-1e-7:best=trial;score=value;changed=True
   if not changed:break
  if best!=route:self.events.append({'event':'scan_priority_reorder','discovered_channels':discovered,
     'length_before_m':reference,'length_after_m':path_length(position,best),
     'objective_before':self.objective(position,route,discovered),'objective_after':score,
     'first_task_before':list(route[0][:2]),'first_task_after':list(best[0][:2])})
  return best


def run_search(client,question=3,method='main',parameters_override=None):
 p=parameters(parameters_override)
 keys=('MOVE_SLACK','MIN_PREDICTED_SAVING','OFFSETS','BACKOFFS','MAX_HYPOTHESES','RECEPTION_LIMIT','BIASES')
 values=(p['move_slack_m'],p['min_saving_s'],tuple(x*p['offset_scale'] for x in (.1,.25,.5)),(.25,.5),9,999.9,(-1.,0.,1.))
 with _LOCK:
  before=tuple(getattr(rollout,k) for k in keys)
  try:
   for k,v in zip(keys,values):setattr(rollout,k,v)
   return _run(client,p,question,method)
  finally:
   for k,v in zip(keys,before):setattr(rollout,k,v)

def _run(client,p,question=3,method='main'):
 if question!=3 or method!='main':raise ValueError('Q3 main only')
 client=PublicTracked(client);planner=SparsePlanner(p)
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
   kind,index,destination=planner.route(client.position,nodes,len(seen))[0]
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
 state['low_source_parameters']=planner.p.copy();state['coverage_rotation_deg']=planner.rotation_deg or 0.;state['route_diagnostics']=planner.events
 return state
