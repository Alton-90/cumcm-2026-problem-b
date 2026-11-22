"""Q3 batched coverage and retained bearing intersections; public observations only."""
import math
import numpy as np
import search_policy as old
from search_policy_compact import coverage_points, _state
VERSION='shared_baseline_v1'
class Tracked:
 def __init__(self,c):
  self.c=c; self.position=(0.,0.); axis=np.arange(-1800.,1800.1,20.); xx,yy=np.meshgrid(axis,axis);p=np.column_stack((xx.ravel(),yy.ravel()));self.grid=p[np.linalg.norm(p,axis=1)<=1815];self.clouds={}
 def estimate(self,ch):
  p=self.clouds.get(ch)
  if p is None or not len(p):return None
  center=p.mean(axis=0);radius=np.linalg.norm(p-center,axis=1).max()+15;return tuple(center),float(radius)
 def measure(self,x,y,ch):
  a=self.c.measure(x,y,ch); self.position=(x,y)
  p=self.clouds.get(ch,self.grid);delta=p-np.array([x,y]);d=np.linalg.norm(delta,axis=1);kind=a.get('measure_result')
  if kind=='no_signal':p=p[d>=985]
  elif kind=='near':p=p[d<=20]
  elif kind=='direction':
   svd=a.get('svd_deg')
   if isinstance(svd,(int,float)) and not isinstance(svd,bool) and math.isfinite(svd) and 0<=svd<360:
    t=math.radians(svd);u=np.array([math.cos(t),math.sin(t)]);v=np.array([-u[1],u[0]]);p=p[(d<=1515)&(delta@u>=-15)&(np.abs(delta@v)<=d*math.sin(math.radians(1.005))+15)]
  self.clouds[ch]=p
  return a
 def clear(self,x,y,ch):
  a=self.c.clear(x,y,ch); self.position=(x,y); return a
 @property
 def virtual_time_s(self): return self.c.virtual_time_s

def localize(client,point,response,channel,state,polygon):
    svd = response.get("svd_deg")
    if isinstance(svd, bool) or not isinstance(svd, (int, float)) or not math.isfinite(svd) or not 0 <= svd < 360:
        state["warnings"].append(f"channel {channel}: direction response lacks numeric svd_deg")
        return False
    previous_center = None
    for extra_used in range(old.MAX_EXTRA_BEARINGS + 1):
        box = old._box_circle(polygon)
        if box is None:
            state["warnings"].append(f"channel {channel}: empty bearing polygon")
            return old._fallback_scan(client, point, svd, channel, state)
        center, radius = box
        estimate=client.estimate(channel)
        if estimate is not None:
            center,radius=estimate
            if radius<=60:
                if old._resolve_clear(client,center,channel,state,'candidate check'):return True
        if radius <= old.LOCATE_CLEAR_RADIUS_M:
            if old._resolve_clear(client, center, channel, state, "bearing localization"):
                return True
            return old._fallback_scan(client, point, svd, channel, state)
        if extra_used >= old.MAX_EXTRA_BEARINGS:
            state["warnings"].append(f"channel {channel}: extra bearing limit reached")
            return old._fallback_scan(client, point, svd, channel, state)
        if previous_center and math.hypot(center[0] - previous_center[0], center[1] - previous_center[1]) <= 1e-6:
            state["warnings"].append(f"channel {channel}: repeated localization center")
            return old._fallback_scan(client, point, svd, channel, state)
        previous_center = center
        follow_up = client.measure(center[0], center[1], channel)
        state["measure_count"] += 1
        state["extra_bearing_count"] += 1
        kind = follow_up.get("measure_result")
        if kind == "near":
            if old._resolve_clear(client, center, channel, state, "near follow-up"):
                return True
            return old._fallback_scan(client, point, svd, channel, state)
        if kind == "direction":
            next_svd = follow_up.get("svd_deg")
            if isinstance(next_svd, bool) or not isinstance(next_svd, (int, float)) or not math.isfinite(next_svd) or not 0 <= next_svd < 360:
                state["warnings"].append(f"channel {channel}: follow-up direction lacks svd_deg")
                return old._fallback_scan(client, point, svd, channel, state)
            polygon = old._add_bearing(polygon, center, next_svd)
            continue
        state["warnings"].append(f"channel {channel}: follow-up no_signal/unknown; retaining fallback")
        return old._fallback_scan(client, point, svd, channel, state)
    raise AssertionError("unreachable extra-bearing loop")


def run_search(client,question=3,method='main'):
 if question!=3 or method!='main': raise ValueError('Q3 only')
 client=Tracked(client); points=[(0.,0.)]+[(1150*math.cos(k*math.pi/3),1150*math.sin(k*math.pi/3)) for k in range(6)]; state=_state(3,points); tracks={}; seen=set()
 try:
  remaining=list(points)
  while remaining:
   point=min(remaining,key=lambda p:math.dist(client.position,p)); remaining.remove(point)
   for ch in range(1,21):
    if ch in state['cleared_channels']: continue
    if ch in tracks and old._box_circle(tracks[ch][2])[1]<=old.LOCATE_CLEAR_RADIUS_M: continue
    a=client.measure(*point,ch); state['measure_count']+=1; state['coverage_measurements_by_channel'][ch]+=1
    kind=a.get('measure_result')
    if kind=='no_signal': continue
    seen.add(ch)
    if kind=='near': old._resolve_clear(client,point,ch,state,'near batch')
    elif kind=='direction':
     svd=a.get('svd_deg')
     if isinstance(svd,bool) or not isinstance(svd,(int,float)) or not math.isfinite(svd) or not 0<=svd<360:
      state['warnings'].append(f'channel {ch}: invalid direction'); state['unresolved_channels'].add(ch); continue
     if ch not in tracks: tracks[ch]=(point,a,old._bearing_triangle(point,a['svd_deg']))
     else:
      p,b,poly=tracks[ch]; clipped=old._add_bearing(poly,point,a['svd_deg'])
      if clipped: tracks[ch]=(p,b,clipped)
    else: state['unresolved_channels'].add(ch)
   pending=[ch for ch in tracks if ch not in state['cleared_channels']]
   if len(pending)>=2:
    angles=[math.radians(tracks[ch][1]['svd_deg']) for ch in pending]
    direction=max([k*math.pi/4 for k in range(8)],key=lambda t:sum(math.sin(t-a)**2 for a in angles))
    baseline=(point[0]+200*math.cos(direction),point[1]+200*math.sin(direction))
    for ch in pending:
     a=client.measure(*baseline,ch);state['measure_count']+=1;state['extra_bearing_count']+=1
     if a.get('measure_result')=='near':
      old._resolve_clear(client,baseline,ch,state,'shared baseline near')
     elif a.get('measure_result')=='direction':
      svd=a.get('svd_deg')
      if isinstance(svd,(int,float)) and not isinstance(svd,bool) and math.isfinite(svd) and 0<=svd<360:
       p,b,poly=tracks[ch];poly2=old._add_bearing(poly,baseline,svd)
       if poly2:tracks[ch]=(p,b,poly2)
    pending=[ch for ch in pending if ch not in state['cleared_channels']]
   while pending:
    ch=min(pending,key=lambda c:math.dist(client.position,(client.estimate(c) or old._box_circle(tracks[c][2]))[0])); pending.remove(ch)
    p,a,poly=tracks[ch]
    if math.dist(client.position,p)>1:
     here=client.position; extra=client.measure(*here,ch); state['measure_count']+=1; state['extra_bearing_count']+=1
     if extra.get('measure_result')=='near':
      if old._resolve_clear(client,here,ch,state,'near reuse'): continue
     if extra.get('measure_result')=='direction':
      svd=extra.get('svd_deg')
      if isinstance(svd,(int,float)) and not isinstance(svd,bool) and math.isfinite(svd) and 0<=svd<360:
       clipped=old._add_bearing(poly,here,svd)
       if clipped: poly=clipped
    localize(client,p,a,ch,state,poly)
  state['excluded_channels']=set(range(1,21))-seen
  state['unresolved_channels']=seen-state['cleared_channels']
 except Exception as e:
  state['stopped']=True; state['warnings'].append(str(e)); state['unresolved_channels']=set(range(1,21))-state['cleared_channels']-state['excluded_channels']
 for key in ('cleared_channels','excluded_channels','unresolved_channels'): state[key]=sorted(state[key])
 state.update(completed=not state['stopped'] and not state['unresolved_channels'] and len(state['cleared_channels'])+len(state['excluded_channels'])==20,question=3,method='main',version=VERSION,virtual_time_s=client.virtual_time_s)
 return state
