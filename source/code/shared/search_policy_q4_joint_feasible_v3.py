"""Q4 feasible-domain localization followed by the reliable Q4 recovery chain."""
from __future__ import annotations
import math
import search_policy_q4_joint_feasible_v2 as v2
from search_policy import _add_bearing,_resolve_clear
from search_policy_q4_batch import _localize
VERSION='q4_joint_feasible_v3'
def feasible_localize(c,initial,svd,ch,s,poly):
 measured=set();failed=set();previous=None
 if math.dist(c.position,initial)>=50:
  key=(round(c.position[0],6),round(c.position[1],6));measured.add(key);r=c.measure(*c.position,ch);s['measure_count']+=1;s['extra_bearing_count']+=1;s['feasible_current_pose_measure_count']=s.get('feasible_current_pose_measure_count',0)+1
  if r.get('measure_result')=='near' and _resolve_clear(c,c.position,ch,s,'near feasible current pose'):return True
  if r.get('measure_result')=='direction' and v2.valid(r.get('svd_deg')):
   merged=_add_bearing(poly,c.position,r['svd_deg']);poly=merged if merged else poly
 for _ in range(v2.MAX_STEPS):
  est=c.estimate(ch)
  if est is None:break
  center,radius=est;key=(round(center[0],6),round(center[1],6));signature=(key,round(radius,6))
  if signature==previous or key in measured:
   s['feasible_no_progress_count']=s.get('feasible_no_progress_count',0)+1;break
  previous=signature
  if radius<=60 and key not in failed:
   s['feasible_clear_attempt_count']+=1
   if _resolve_clear(c,center,ch,s,'feasible <=60m candidate'):return True
   failed.add(key);s['feasible_failed_clear_count']+=1
  measured.add(key);r=c.measure(*center,ch);s['measure_count']+=1;s['extra_bearing_count']+=1;s['feasible_measure_count']+=1;k=r.get('measure_result')
  if k=='near' and key not in failed:
   s['feasible_clear_attempt_count']+=1
   if _resolve_clear(c,center,ch,s,'near feasible candidate'):return True
   failed.add(key);s['feasible_failed_clear_count']+=1
  elif k=='direction' and v2.valid(r.get('svd_deg')):
   merged=_add_bearing(poly,center,r['svd_deg']);poly=merged if merged else poly
  elif k not in ('no_signal','near'):break
 s['feasible_q4_recovery_count']=s.get('feasible_q4_recovery_count',0)+1
 return _localize(c,initial,svd,ch,s,poly)
def run_search(client,question=4,method='main'):
 old=v2.feasible_localize;v2.feasible_localize=feasible_localize
 try:s=v2.run_search(client,question=question,method=method)
 finally:v2.feasible_localize=old
 s['version']=VERSION
 for key in ('feasible_current_pose_measure_count','feasible_no_progress_count','feasible_q4_recovery_count'):s.setdefault(key,0)
 return s
