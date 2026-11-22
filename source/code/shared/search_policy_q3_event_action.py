"""Q3 joint route that yields to global replanning after each localization action."""
import math
import search_policy as old
from search_policy_compact import _state
from search_policy_feasible_q3 import Tracked,localize
from search_policy_q3_measure_replan import route,valid,reached
VERSION='q3_event_action_v2'
def run_search(client,question=3,method='main'):
 if question!=3 or method!='main':raise ValueError('Q3 main only')
 c=Tracked(client);pts=[(0.,0.)]+[(1150*math.cos(k*math.pi/3),1150*math.sin(k*math.pi/3)) for k in range(6)];s=_state(3,pts);tracks={};seen=set();remaining=list(range(7));progress={};finalized=set();supplemented=set();failed_clear_points=set();s.update(route_strategy='one_localization_action_then_global_replan',event_measure_count=0,event_clear_count=0,current_pose_supplement_count=0,failed_candidate_clear_count=0,finalizer_count=0,maximum_count_stop_used=False,maximum_count_certificate=None)
 try:
  while remaining or any(ch not in s['cleared_channels'] and ch not in finalized for ch in tracks):
   if reached(s):break
   nodes=[('scan',i,pts[i]) for i in remaining]
   for ch,tr in tracks.items():
    if ch in s['cleared_channels'] or ch in finalized:continue
    est=c.estimate(ch) or old._box_circle(tr[2]);nodes.append(('source',ch,est[0] if est else tr[0]))
   if not nodes:break
   kind,idx,dest=route(c.position,nodes)[0]
   if kind=='scan':
    remaining.remove(idx)
    for ch in range(1,21):
     if reached(s):break
     if ch in s['cleared_channels']:continue
     a=c.measure(*dest,ch);s['measure_count']+=1;s['coverage_measurements_by_channel'][ch]+=1;k=a.get('measure_result')
     if k=='no_signal':continue
     seen.add(ch)
     if k=='near':old._resolve_clear(c,dest,ch,s,'near event scan')
     elif k=='direction' and valid(a.get('svd_deg')):
      if ch not in tracks:tracks[ch]=(dest,a,old._bearing_triangle(dest,a['svd_deg']))
      else:
       p,b,poly=tracks[ch];m=old._add_bearing(poly,dest,a['svd_deg']);tracks[ch]=(p,b,m) if m else tracks[ch]
     else:s['warnings'].append(f'channel {ch}: invalid response')
   else:
    ch=idx;p,a,poly=tracks[ch];steps=progress.get(ch,0);pose_key=(ch,round(c.position[0],6),round(c.position[1],6))
    if pose_key not in supplemented and math.dist(c.position,p)>1:
     supplemented.add(pose_key);r=c.measure(*c.position,ch);s['measure_count']+=1;s['extra_bearing_count']+=1;s['current_pose_supplement_count']+=1
     if r.get('measure_result')=='near' and old._resolve_clear(c,c.position,ch,s,'near current-pose event'):continue
     if r.get('measure_result')=='direction' and valid(r.get('svd_deg')):
      m=old._add_bearing(poly,c.position,r['svd_deg']);tracks[ch]=(p,a,m) if m else tracks[ch]
     continue
    poly=tracks[ch][2];est=c.estimate(ch) or old._box_circle(poly);clear_key=None if est is None else (ch,round(est[0][0],6),round(est[0][1],6))
    if est and est[1]<=60 and steps<99 and clear_key not in failed_clear_points:
     s['event_clear_count']+=1
     if old._resolve_clear(c,est[0],ch,s,'event candidate'):continue
     failed_clear_points.add(clear_key);s['failed_candidate_clear_count']+=1;progress[ch]=99;continue
    if est and steps<4:
     center=est[0];r=c.measure(*center,ch);s['measure_count']+=1;s['extra_bearing_count']+=1;s['event_measure_count']+=1;progress[ch]=steps+1
     if r.get('measure_result')=='near':
      s['event_clear_count']+=1;old._resolve_clear(c,center,ch,s,'near event action')
     elif r.get('measure_result')=='direction' and valid(r.get('svd_deg')):
      m=old._add_bearing(poly,center,r['svd_deg']);tracks[ch]=(p,a,m) if m else tracks[ch]
     else:progress[ch]=99
     continue
    s['finalizer_count']+=1;localize(c,p,a,ch,s,poly);progress[ch]=99;finalized.add(ch)
  if not s['maximum_count_stop_used']:
   s['excluded_channels']=set(range(1,21))-seen;s['unresolved_channels']=seen-s['cleared_channels']
 except Exception as e:s['stopped']=True;s['warnings'].append(str(e));s['unresolved_channels']=set(range(1,21))-s['cleared_channels']-s['excluded_channels']
 for k in ('cleared_channels','excluded_channels','unresolved_channels'):s[k]=sorted(s[k])
 s.update(completed=not s['stopped'] and not s['unresolved_channels'] and len(s['cleared_channels'])+len(s['excluded_channels'])==20,question=3,method='main',version=VERSION,virtual_time_s=c.virtual_time_s);return s
