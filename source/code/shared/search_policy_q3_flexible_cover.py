"""Receding open-route planning over certified scan points and observed sources.

The center covers r <= 1000. Six 60 degree sectors cover 1000 <= r <= 1800; each scan point may slide radially from 1124 to 1730 m. Squared distance is convex in both radii and maximal at angular endpoint 30 degrees; all four endpoint distances are below 1000 m. Source destinations
are estimates based only on accumulated public bearings and no-signal data.
"""
import math
import search_policy as old
from search_policy_compact import _state
from search_policy_feasible_q3 import Tracked,localize
VERSION='q3_flexible_cover_v1'

def _valid(s):
 return isinstance(s,(int,float)) and not isinstance(s,bool) and math.isfinite(s) and 0<=s<360

def _fixed_route(position,nodes):
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

def _route(position,nodes):
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
   lo,hi=1124.,1730.;g=(5**.5-1)/2
   x=hi-g*(hi-lo);y=lo+g*(hi-lo);fx,fy=cost(x),cost(y)
   for _ in range(30):
    if fx<fy:hi,y,fy=y,x,fx;x=hi-g*(hi-lo);fx=cost(x)
    else:lo,x,fx=x,y,fy;y=lo+g*(hi-lo);fy=cost(y)
   d=min((1124.,1730.,(lo+hi)/2,math.hypot(*p)),key=cost)
   route[i]=(kind,identifier,(d*u[0],d*u[1]))
  route=_fixed_route(position,route)
 return route

def run_search(client,question=3,method='main'):
 if question!=3 or method!='main':raise ValueError('Q3 main only')
 client=Tracked(client)
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
