"""Q3 opportunistic scans with a conservative disk-union coverage certificate."""
import math
import search_policy as old
from search_policy_feasible_q3 import Tracked,localize
from search_policy_compact import _state
VERSION='adaptive_feasible_v1'
RADIUS=999.0

def covered_domain(positions):
 """Prove coverage via complete quadtree cells. Uncertain means False."""
 stack=[(-1800.,1800.,-1800.,1800.,0)]
 while stack:
  xl,xh,yl,yh,d=stack.pop()
  nx=max(xl,min(0.,xh)); ny=max(yl,min(0.,yh))
  if nx*nx+ny*ny>1800**2: continue
  if any(max((x-p[0])**2+(y-p[1])**2 for x in (xl,xh) for y in (yl,yh))<=RADIUS**2 for p in positions): continue
  if d>=8: return False
  xm=(xl+xh)/2;ym=(yl+yh)/2
  stack.extend([(xl,xm,yl,ym,d+1),(xm,xh,yl,ym,d+1),(xl,xm,ym,yh,d+1),(xm,xh,ym,yh,d+1)])
 return True

def valid(a):
 v=a.get('svd_deg');return not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<360

def run_search(client,question=3,method='main'):
 if question!=3 or method!='main':raise ValueError('Q3 only')
 c=Tracked(client);points=[(0.,0.)]+[(1150*math.cos(k*math.pi/3),1150*math.sin(k*math.pi/3)) for k in range(6)]
 state=_state(3,points);tracks={};seen=set();sweeps=[];remaining=points[1:];cert=False
 def sweep():
  here=c.position
  # All as-yet unseen channels share these coverage disks.
  if any(math.dist(here,p)<1 for p in sweeps):return
  for ch in range(1,21):
   if ch in seen:continue
   a=c.measure(*here,ch);state['measure_count']+=1;state['coverage_measurements_by_channel'][ch]+=1
   kind=a.get('measure_result')
   if kind=='no_signal':continue
   seen.add(ch)
   if kind=='near':old._resolve_clear(c,here,ch,state,'near sweep')
   elif kind=='direction' and valid(a):
    if ch not in tracks:tracks[ch]=(here,a,old._bearing_triangle(here,a['svd_deg']))
    else:
     p,b,poly=tracks[ch];new=old._add_bearing(poly,here,a['svd_deg'])
     if new:tracks[ch]=(p,b,new)
   else:state['warnings'].append(f'invalid observation channel {ch}')
  sweeps.append(here)
 try:
  sweep()
  while True:
   pending=[ch for ch in tracks if ch not in state['cleared_channels']]
   if pending:
    ch=min(pending,key=lambda k:math.dist(c.position,(c.estimate(k) or old._box_circle(tracks[k][2]))[0]));p,a,poly=tracks.pop(ch)
    localize(c,p,a,ch,state,poly)
    if ch not in state['cleared_channels']:
     state['unresolved_channels'].add(ch)
    sweep()
    continue
   cert=covered_domain(sweeps)
   if cert or not remaining:break
   point=min(remaining,key=lambda p:math.dist(c.position,p));remaining.remove(point)
   # Move by a normal measurement, then sweep all unseen channels at this point.
   unknown=[ch for ch in range(1,21) if ch not in seen]
   if not unknown:break
   ch=unknown[0];a=c.measure(*point,ch);state['measure_count']+=1
   # Repeat in sweep intentionally uses the same fixed point error; no independence assumed.
   sweep()
  state['excluded_channels']=set(range(1,21))-seen if cert or not remaining else set()
  state['unresolved_channels']=seen-state['cleared_channels']
 except Exception as e:
  state['stopped']=True;state['warnings'].append(str(e));state['unresolved_channels']=set(range(1,21))-state['cleared_channels']-state['excluded_channels']
 for key in ('cleared_channels','excluded_channels','unresolved_channels'):state[key]=sorted(state[key])
 state.update(completed=not state['stopped'] and not state['unresolved_channels'] and len(state['cleared_channels'])+len(state['excluded_channels'])==20,question=3,method='main',version=VERSION,virtual_time_s=c.virtual_time_s,coverage_certificate=cert,opportunistic_sweep_count=len(sweeps),remaining_hex_points=len(remaining))
 return state
